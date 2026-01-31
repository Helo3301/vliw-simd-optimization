"""
BF200 Exp17: ALU-based tree indices + 14b init

Based on Theory 214 (Level-3 Tree Fusion).
Key optimization: defer idx computation from R1/R2 to R3.
Instead of incrementally building idx at each round, compute addr directly
from bit0/bit1/bit2/bit3 at R3 using FMA chain:
  s1 = 2*bit0 + bit1
  s2 = 2*s1 + bit2
  addr = FMA(s2, 2, bit3) + (fp + 15)

Saves 2 VALU/desk/block * 4 desks * 4 groups * 2 tiles * 2 blocks = 128 VALU
Expected: 7907 - 128 = 7779 VALU, floor 1297 (was 1318)
"""

import random
import argparse
import sys
import os
from collections import defaultdict

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


def _vec_range(base: int, length: int = VLEN) -> range:
    return range(base, base + length)


def _slot_rw(engine: str, slot: tuple) -> tuple[list[int], list[int]]:
    reads: list[int] = []
    writes: list[int] = []

    if engine == "alu":
        _op, dest, a1, a2 = slot
        reads = [a1, a2]
        writes = [dest]
    elif engine == "valu":
        op = slot[0]
        if op == "vbroadcast":
            dest, src = slot[1], slot[2]
            reads = [src]
            writes = list(_vec_range(dest))
        elif op == "multiply_add":
            dest, a, b, c = slot[1], slot[2], slot[3], slot[4]
            reads = list(_vec_range(a)) + list(_vec_range(b)) + list(_vec_range(c))
            writes = list(_vec_range(dest))
        else:
            _op, dest, a1, a2 = slot
            reads = list(_vec_range(a1)) + list(_vec_range(a2))
            writes = list(_vec_range(dest))
    elif engine == "load":
        op = slot[0]
        if op == "load":
            dest, addr = slot[1], slot[2]
            reads = [addr]
            writes = [dest]
        elif op == "vload":
            dest, addr = slot[1], slot[2]
            reads = [addr]
            writes = list(_vec_range(dest))
        elif op == "const":
            dest = slot[1]
            writes = [dest]
        elif op == "load_offset":
            dest, addr, _lane = slot[1], slot[2], slot[3]
            reads = [addr]
            writes = [dest]
        else:
            raise NotImplementedError(f"Unknown load op {slot}")
    elif engine == "store":
        op = slot[0]
        if op == "store":
            addr, src = slot[1], slot[2]
            reads = [addr, src]
        elif op == "vstore":
            addr, src = slot[1], slot[2]
            reads = [addr] + list(_vec_range(src))
        else:
            raise NotImplementedError(f"Unknown store op {slot}")
    elif engine == "flow":
        op = slot[0]
        if op == "select":
            dest, cond, a, b = slot[1], slot[2], slot[3], slot[4]
            reads = [cond, a, b]
            writes = [dest]
        elif op == "add_imm":
            dest, a = slot[1], slot[2]
            reads = [a]
            writes = [dest]
        elif op == "vselect":
            dest, cond, a, b = slot[1], slot[2], slot[3], slot[4]
            reads = list(_vec_range(cond)) + list(_vec_range(a)) + list(_vec_range(b))
            writes = list(_vec_range(dest))
        elif op in ("halt", "pause", "trace_write", "jump", "jump_indirect", "cond_jump", "cond_jump_rel", "coreid"):
            pass
        else:
            raise NotImplementedError(f"Unknown flow op {slot}")

    return reads, writes


def _schedule_slots_greedy(slots):
    """Original greedy scheduler."""
    cycles = []
    usage = []
    ready_time = defaultdict(int)
    last_write = defaultdict(lambda: -1)
    last_read = defaultdict(lambda: -1)

    def ensure_cycle(cycle):
        while len(cycles) <= cycle:
            cycles.append({})
            usage.append(defaultdict(int))

    def find_cycle(engine, earliest):
        cycle = earliest
        limit = SLOT_LIMITS[engine]
        while True:
            ensure_cycle(cycle)
            if usage[cycle][engine] < limit:
                return cycle
            cycle += 1

    for engine, slot in slots:
        reads, writes = _slot_rw(engine, slot)
        earliest = 0
        for addr in reads:
            earliest = max(earliest, ready_time[addr])
        for addr in writes:
            earliest = max(earliest, last_write[addr] + 1, last_read[addr])

        cycle = find_cycle(engine, earliest)
        ensure_cycle(cycle)
        cycles[cycle].setdefault(engine, []).append(slot)
        usage[cycle][engine] += 1

        for addr in reads:
            if last_read[addr] < cycle:
                last_read[addr] = cycle
        for addr in writes:
            last_write[addr] = cycle
            ready_time[addr] = cycle + 1

    return [c for c in cycles if c]


def _schedule_slots(slots: list[tuple[str, tuple]]) -> list[dict[str, list[tuple]]]:
    """Priority-based topological reordering + greedy scheduling."""
    import heapq

    n = len(slots)
    if n == 0:
        return []

    # Build dependency graph
    rw = []
    for engine, slot in slots:
        reads, writes = _slot_rw(engine, slot)
        rw.append((list(reads), list(writes)))

    write_map = {}
    read_map = defaultdict(list)
    preds = [set() for _ in range(n)]
    succs = [set() for _ in range(n)]

    for i in range(n):
        reads, writes = rw[i]
        for addr in reads:
            if addr in write_map:
                preds[i].add(write_map[addr])
                succs[write_map[addr]].add(i)
        for addr in writes:
            if addr in write_map:
                w = write_map[addr]
                preds[i].add(w)
                succs[w].add(i)
            for r in read_map.get(addr, []):
                if r != i:
                    preds[i].add(r)
                    succs[r].add(i)
        for addr in reads:
            read_map[addr].append(i)
        for addr in writes:
            write_map[addr] = i
            read_map[addr] = []

    # Compute downstream VALU count: max VALU ops on any path from this op to a sink
    downstream_valu = [0] * n
    computed = [False] * n

    def compute_dv(start):
        stack = [(start, False)]
        while stack:
            v, processed = stack.pop()
            if computed[v]:
                continue
            if processed:
                for s in succs[v]:
                    downstream_valu[v] = max(downstream_valu[v],
                        downstream_valu[s] + (1 if slots[s][0] == 'valu' else 0))
                computed[v] = True
                continue
            stack.append((v, True))
            for s in succs[v]:
                if not computed[s]:
                    stack.append((s, False))

    for i in range(n - 1, -1, -1):
        if not computed[i]:
            compute_dv(i)

    best_result = _schedule_slots_greedy(slots)
    best_cycles = len(best_result)
    best_ordered = None

    # Compute downstream flow count
    downstream_flow = [0] * n
    computed_df = [False] * n
    def compute_df(start):
        stack = [(start, False)]
        while stack:
            v, processed = stack.pop()
            if computed_df[v]:
                continue
            if processed:
                for s in succs[v]:
                    downstream_flow[v] = max(downstream_flow[v],
                        downstream_flow[s] + (1 if slots[s][0] == 'flow' else 0))
                computed_df[v] = True
                continue
            stack.append((v, True))
            for s in succs[v]:
                if not computed_df[s]:
                    stack.append((s, False))
    for i in range(n - 1, -1, -1):
        if not computed_df[i]:
            compute_df(i)

    # Try multiple priority strategies
    priority_fns = [
        # VALU first
        lambda i: (0 if slots[i][0] == 'valu' else 1, i),
        # Non-VALU with high downstream VALU first (dv>25), then VALU
        lambda i: ((0, -downstream_valu[i]) if slots[i][0] != 'valu' and downstream_valu[i] > 25
                   else (1 if slots[i][0] == 'valu' else 2,), i),
        # ALU first, then dv>25
        lambda i: ((0,) if slots[i][0] == 'alu'
                   else (1, -downstream_valu[i]) if slots[i][0] != 'valu' and downstream_valu[i] > 25
                   else (2 if slots[i][0] == 'valu' else 3,), i),
        # dv>28 (wider threshold)
        lambda i: ((0, -downstream_valu[i]) if slots[i][0] != 'valu' and downstream_valu[i] > 28
                   else (1 if slots[i][0] == 'valu' else 2,), i),
    ]

    # Shot 81: Collect ALL priority orderings (not just best) for SA starting points
    all_orderings = []
    for priority_fn in priority_fns:
        in_deg = [len(preds[i]) for i in range(n)]
        ready = []
        for i in range(n):
            if in_deg[i] == 0:
                heapq.heappush(ready, (priority_fn(i), i))

        ordered_idx = []
        while ready:
            _, op_idx = heapq.heappop(ready)
            ordered_idx.append(op_idx)
            for s in succs[op_idx]:
                in_deg[s] -= 1
                if in_deg[s] == 0:
                    heapq.heappush(ready, (priority_fn(s), s))

        ordered = [slots[i] for i in ordered_idx]
        result = _schedule_slots_greedy(ordered)
        greedy_cycles = len(result)
        all_orderings.append((greedy_cycles, ordered_idx[:], result))
        if greedy_cycles < best_cycles:
            best_cycles = greedy_cycles
            best_result = result
            best_ordered = ordered_idx[:]

    # Shot 85: Multi-phase SA with wide blocks and chain refinement
    if n > 10:
        import random as _rng
        _rng_state = _rng.getstate()

        def _run_sa(_start_order, _start_cycles, _seed, _n_iters=10000,
                    _temp=5.0, _cool=0.9995, _max_block=16):
            _rng.seed(_seed)
            _cur_order = _start_order[:]
            _cur_cycles = _start_cycles
            _best_c = _start_cycles
            _best_o = _start_order[:]
            _best_r = None
            _temperature = _temp
            for _ in range(_n_iters):
                _bs = _rng.randint(2, _max_block)
                _ps = _rng.randrange(max(1, n - _bs))
                _pe = min(_ps + _bs, n)
                _block = _cur_order[_ps:_pe]
                _rev = _block[::-1]
                _pm = {}
                for _p in range(len(_cur_order)):
                    _pm[_cur_order[_p]] = _p
                for _ii, _oi in enumerate(_rev):
                    _pm[_oi] = _ps + _ii
                _valid = True
                for _ii, _oi in enumerate(_rev):
                    _np = _ps + _ii
                    for _pr in preds[_oi]:
                        if _pm[_pr] >= _np:
                            _valid = False
                            break
                    if not _valid:
                        break
                    for _sc in succs[_oi]:
                        if _pm[_sc] <= _np:
                            _valid = False
                            break
                    if not _valid:
                        break
                if not _valid:
                    continue
                _old = _cur_order[_ps:_pe]
                _cur_order[_ps:_pe] = _old[::-1]
                _nr = _schedule_slots_greedy([slots[i] for i in _cur_order])
                _nc = len(_nr)
                _delta = _nc - _cur_cycles
                if _delta < 0 or _rng.random() < 2.718 ** (-_delta / max(_temperature, 0.01)):
                    _cur_cycles = _nc
                    if _cur_cycles < _best_c:
                        _best_c = _cur_cycles
                        _best_o = _cur_order[:]
                        _best_r = _nr
                else:
                    _cur_order[_ps:_pe] = _old
                _temperature *= _cool
            return _best_c, _best_o, _best_r

        # Phase 1: SA from best priority orderings (skip fn0=VALU-first, gives 1504)
        for _greedy_c, _start_order, _start_result in all_orderings[1:]:
            for _sa_seed in [24, 42]:
                _bc, _bo, _br = _run_sa(_start_order, _greedy_c, _sa_seed,
                                        _n_iters=5000, _max_block=16)
                if _bc < best_cycles:
                    best_cycles = _bc
                    best_ordered = _bo
                    best_result = _br if _br is not None else _schedule_slots_greedy([slots[i] for i in _bo])

        # Phase 2: Chain SA - refine from best with diverse seeds
        if best_ordered is not None:
            for _sa_seed in [1, 7, 13, 17, 31, 53, 97, 127]:
                _bc, _bo, _br = _run_sa(best_ordered, best_cycles, _sa_seed,
                                        _n_iters=5000, _temp=5.0, _cool=0.9990, _max_block=16)
                if _bc < best_cycles:
                    best_cycles = _bc
                    best_ordered = _bo
                    best_result = _br if _br is not None else _schedule_slots_greedy([slots[i] for i in _bo])

        _rng.setstate(_rng_state)

    return best_result


class KernelBuilder:
    """A1: R10 Branch Skip - based on B4-2 with R10 optimization"""
    def __init__(self):
        self.slots: list[tuple[str, tuple]] = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self.vconst_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def emit(self, engine: str, slot: tuple):
        self.slots.append((engine, slot))

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE
        return addr

    def alloc_vec(self, name=None):
        return self.alloc_scratch(name, VLEN)

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name or f"c_{val}")
            self.emit("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def scratch_vconst(self, val, name=None):
        if val not in self.vconst_map:
            scalar = self.scratch_const(val)
            addr = self.alloc_vec(name or f"v_{val}")
            self.emit("valu", ("vbroadcast", addr, scalar))
            self.vconst_map[val] = addr
        return self.vconst_map[val]

    def build_kernel(self, forest_height: int, n_nodes: int, batch_size: int, rounds: int):
        tmp_scalar = self.alloc_scratch("tmp_scalar")
        tmp_addr = self.alloc_scratch("tmp_addr")

        # Shot 63: n_nodes removed (unused - was only for v_n_nodes broadcast)
        fast_init_vars = [("forest_values_p", 4), ("inp_indices_p", 5), ("inp_values_p", 6)]
        for var_name, _ in fast_init_vars:
            self.alloc_scratch(var_name)
        for var_name, idx in fast_init_vars:
            self.emit("load", ("const", tmp_scalar, idx))
            self.emit("load", ("load", self.scratch[var_name], tmp_scalar))

        # Shot 63: v_zero and v_n_nodes removed (unused)
        v_one = self.scratch_vconst(1, "v_one")
        v_two = self.scratch_vconst(2, "v_two")

        v_forest_p = self.alloc_vec("v_forest_p")
        self.emit("valu", ("vbroadcast", v_forest_p, self.scratch["forest_values_p"]))

        # THEORY 1: Precompute v_1_minus_fp = 1 - forest_p for addr-tracking branch
        v_1_minus_fp = self.alloc_vec("v_1_minus_fp")
        self.emit("valu", ("-", v_1_minus_fp, v_one, v_forest_p))

        # Shot 63: v_fp_plus_1 removed (unused - emit_branch_idx_to_addr not called)

        # Theory 222: Precompute v_fp_plus_15 for deferred addr computation
        v_fp_plus_15 = self.alloc_vec("v_fp_plus_15")
        self.emit("valu", ("+", v_fp_plus_15, v_forest_p, self.scratch_vconst(15, "v_fifteen")))

        FMA_MULTIPLIERS = {0: 4097, 2: 33, 4: 9}
        v_hash_consts = []
        v_hash_shifts = []
        v_fma_mult = {}

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            v_const = self.scratch_vconst(val1, f"v_hash_const_{hi}")
            v_hash_consts.append(v_const)
            if hi in FMA_MULTIPLIERS:
                v_fma_mult[hi] = self.scratch_vconst(FMA_MULTIPLIERS[hi], f"v_fma_mult_{hi}")
                v_hash_shifts.append(None)
            else:
                v_shift = self.scratch_vconst(val3, f"v_hash_shift_{hi}")
                v_hash_shifts.append(v_shift)

        NUM_PRELOADED = 15  # Theory 214: levels 0-3 = 1+2+4+8 = 15 nodes
        v_tree = []
        # Pre-allocate const addresses for tree indices
        # Indices 0,1,2 use existing cached consts. Indices 3-14 computed via ALU post-pause.
        tree_idx_consts = []
        tree_idx_deferred = []  # (target_addr, will be computed post-pause)
        for i in range(NUM_PRELOADED):
            v_node = self.alloc_vec(f"v_tree_{i}")
            v_tree.append(v_node)
            if i <= 2:
                # These are already cached as const_map entries
                tree_idx_consts.append(self.scratch_const(i))
            else:
                # Allocate scratch but don't load const yet
                addr = self.alloc_scratch(f"tree_idx_{i}")
                tree_idx_consts.append(addr)
                tree_idx_deferred.append((addr, i))

        # All diff vectors removed: R1/R12 use vselect, R2/R13 use 3 vselect
        # (saves 3 VALU SUB ops and 24 scratch slots)

        NUM_DESKS = 16
        desks = []
        for d in range(NUM_DESKS):
            desk = {
                'idx': self.alloc_vec(f"v_idx_{d}"),
                'val': self.alloc_vec(f"v_val_{d}"),
                'node_val': self.alloc_vec(f"v_node_{d}"),
                'addr': self.alloc_vec(f"v_addr_{d}"),
                'tmp1': self.alloc_vec(f"v_tmp1_{d}"),
                'tmp2': self.alloc_vec(f"v_tmp2_{d}"),
                'bit0': self.alloc_vec(f"v_bit0_{d}"),
                'bit1': self.alloc_vec(f"v_bit1_{d}"),  # Theory 214: save bit1 for level-3 vselect
            }
            desks.append(desk)

        offset_regs = [self.alloc_scratch(f"off_{d}") for d in range(NUM_DESKS)]
        addr_tmp = [self.alloc_scratch(f"addr_tmp_{i}") for i in range(32)]

        print(f"Scratch usage: {self.scratch_ptr} / {SCRATCH_SIZE}")

        self.emit("flow", ("pause",))

        # Exp14e+ALU: Compute tree indices 3-14 via ALU (frees 12 load slots pre-pause)
        # Using existing const(1) and const(2) scratch addresses
        c1 = self.const_map[1]  # scratch addr holding 1
        c2 = self.const_map[2]  # scratch addr holding 2
        # Compute const(3) = 1+2, const(4) = 2+2
        self.emit("alu", ("+", tree_idx_consts[3], c1, c2))        # 3 = 1+2
        self.emit("alu", ("+", tree_idx_consts[4], c2, c2))        # 4 = 2+2
        # Compute const(5..7) from const(4)
        self.emit("alu", ("+", tree_idx_consts[5], tree_idx_consts[4], c1))  # 5 = 4+1
        self.emit("alu", ("+", tree_idx_consts[6], tree_idx_consts[4], c2))  # 6 = 4+2
        self.emit("alu", ("+", tree_idx_consts[7], tree_idx_consts[4], tree_idx_consts[3]))  # 7 = 4+3
        # Compute const(8) = 4+4
        self.emit("alu", ("+", tree_idx_consts[8], tree_idx_consts[4], tree_idx_consts[4]))  # 8 = 4+4
        # Compute const(9..11) from const(8)
        self.emit("alu", ("+", tree_idx_consts[9], tree_idx_consts[8], c1))   # 9 = 8+1
        self.emit("alu", ("+", tree_idx_consts[10], tree_idx_consts[8], c2))  # 10 = 8+2
        self.emit("alu", ("+", tree_idx_consts[11], tree_idx_consts[8], tree_idx_consts[3]))  # 11 = 8+3
        # Compute const(12) = 8+4
        self.emit("alu", ("+", tree_idx_consts[12], tree_idx_consts[8], tree_idx_consts[4]))  # 12 = 8+4
        # Compute const(13,14) from const(12)
        self.emit("alu", ("+", tree_idx_consts[13], tree_idx_consts[12], c1))  # 13 = 12+1
        self.emit("alu", ("+", tree_idx_consts[14], tree_idx_consts[12], c2))  # 14 = 12+2

        # Exp14b: Load tree[0] first, rest later
        self.emit("alu", ("+", tmp_addr, self.scratch["forest_values_p"], tree_idx_consts[0]))
        self.emit("load", ("load", tmp_scalar, tmp_addr))
        self.emit("valu", ("vbroadcast", v_tree[0], tmp_scalar))

        # Exp67: Precompute C5 ^ tree[0] for folding R11 XOR into R10 hash
        v_c5_xor_t0 = self.alloc_vec("v_c5_xor_t0")
        self.emit("valu", ("^", v_c5_xor_t0, v_hash_consts[5], v_tree[0]))

        def emit_hash_interleaved(group_desks):
            # Interleave desk order (even first, then odd) + per-desk hash (all stages per desk)
            gd = [group_desks[0], group_desks[1], group_desks[2], group_desks[3]]
            for d in gd:
                desk = desks[d]
                self.emit("valu", ("multiply_add", desk['val'], desk['val'], v_fma_mult[0], v_hash_consts[0]))
                self.emit("valu", ("^", desk['tmp1'], desk['val'], v_hash_consts[1]))
                self.emit("valu", (">>", desk['tmp2'], desk['val'], v_hash_shifts[1]))
                self.emit("valu", ("^", desk['val'], desk['tmp1'], desk['tmp2']))
                self.emit("valu", ("multiply_add", desk['val'], desk['val'], v_fma_mult[2], v_hash_consts[2]))
                self.emit("valu", ("+", desk['tmp1'], desk['val'], v_hash_consts[3]))
                self.emit("valu", ("<<", desk['tmp2'], desk['val'], v_hash_shifts[3]))
                self.emit("valu", ("^", desk['val'], desk['tmp1'], desk['tmp2']))
                self.emit("valu", ("multiply_add", desk['val'], desk['val'], v_fma_mult[4], v_hash_consts[4]))
                self.emit("valu", ("^", desk['tmp1'], desk['val'], v_hash_consts[5]))
                self.emit("valu", (">>", desk['tmp2'], desk['val'], v_hash_shifts[5]))
                self.emit("valu", ("^", desk['val'], desk['tmp1'], desk['tmp2']))

        def emit_branch(desk_idx):
            d = desks[desk_idx]
            self.emit("valu", ("&", d['tmp1'], d['val'], v_one))
            self.emit("valu", ("multiply_add", d['idx'], d['idx'], v_two, v_one))
            self.emit("valu", ("+", d['idx'], d['idx'], d['tmp1']))

        def emit_branch_addr_tracking(desk_idx):
            """Branch that updates addr instead of idx. addr_new = 2*addr + (1-fp) + bit"""
            d = desks[desk_idx]
            self.emit("valu", ("&", d['tmp1'], d['val'], v_one))
            self.emit("valu", ("multiply_add", d['addr'], d['addr'], v_two, v_1_minus_fp))
            self.emit("valu", ("+", d['addr'], d['addr'], d['tmp1']))

        def emit_xor_with_node(desk_idx, node_vec):
            d = desks[desk_idx]
            self.emit("valu", ("^", d['val'], d['val'], node_vec))

        # Theory 222: Fused rounds 0+1+2+3 with deferred idx computation
        def emit_rounds_0_1_2_3_fused(group_desks):
            # === Round 0 ===
            for d in group_desks:
                emit_xor_with_node(d, v_tree[0])
            emit_hash_interleaved(group_desks)
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("&", desk['bit0'], desk['val'], v_one))

            # === Round 1 === (vselect for node selection)
            for d in group_desks:
                desk = desks[d]
                self.emit("flow", ("vselect", desk['node_val'], desk['bit0'], v_tree[2], v_tree[1]))
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("^", desk['val'], desk['val'], desk['node_val']))
            emit_hash_interleaved(group_desks)
            for d in group_desks:
                desk = desks[d]
                # Theory 222: only extract bit1, defer idx computation to R3
                self.emit("valu", ("&", desk['bit1'], desk['val'], v_one))

            # === Round 2 === (3 vselect cascade using bit1 register)
            for d in group_desks:
                desk = desks[d]
                self.emit("flow", ("vselect", desk['tmp2'], desk['bit1'], v_tree[4], v_tree[3]))
                self.emit("flow", ("vselect", desk['node_val'], desk['bit1'], v_tree[6], v_tree[5]))
                self.emit("flow", ("vselect", desk['node_val'], desk['bit0'], desk['node_val'], desk['tmp2']))
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("^", desk['val'], desk['val'], desk['node_val']))
            emit_hash_interleaved(group_desks)
            # Theory 222: extract bit2 into idx (safe from hash clobbering)
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("&", desk['idx'], desk['val'], v_one))  # bit2 in idx (safe)

            # === Round 3 === (7-vselect cascade for level-3 node)
            # Select tree[7 + 4*bit0 + 2*bit1 + bit2] from tree[7..14]
            # bit2 is now in idx (not tmp1!)
            for d in group_desks:
                desk = desks[d]
                # bit0=0 side:
                self.emit("flow", ("vselect", desk['tmp2'],     desk['idx'], v_tree[8],  v_tree[7]))
                self.emit("flow", ("vselect", desk['node_val'], desk['idx'], v_tree[10], v_tree[9]))
                self.emit("flow", ("vselect", desk['tmp2'],     desk['bit1'], desk['node_val'], desk['tmp2']))
                # bit0=1 side:
                self.emit("flow", ("vselect", desk['node_val'], desk['idx'], v_tree[12], v_tree[11]))
                self.emit("flow", ("vselect", desk['addr'],     desk['idx'], v_tree[14], v_tree[13]))
                self.emit("flow", ("vselect", desk['node_val'], desk['bit1'], desk['addr'], desk['node_val']))
                # Final bit0 select:
                self.emit("flow", ("vselect", desk['node_val'], desk['bit0'], desk['node_val'], desk['tmp2']))
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("^", desk['val'], desk['val'], desk['node_val']))
            emit_hash_interleaved(group_desks)
            # Theory 222: Deferred addr computation from bit0/bit1/bit2/bit3
            # addr = fp + 15 + 8*bit0 + 4*bit1 + 2*bit2 + bit3
            # bit2 is in idx (preserved through hash since hash uses val/tmp1/tmp2)
            # Computed as: s = FMA(bit0, 2, bit1) -> FMA(s, 2, bit2) -> FMA(s, 2, bit3) -> ADD(s, fp+15)
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("&", desk['tmp1'], desk['val'], v_one))  # bit3 -> tmp1
                self.emit("valu", ("multiply_add", desk['addr'], desk['bit0'], v_two, desk['bit1']))  # s = 2*bit0 + bit1
                self.emit("valu", ("multiply_add", desk['addr'], desk['addr'], v_two, desk['idx']))   # s = 2*s + bit2 (idx=bit2)
                self.emit("valu", ("multiply_add", desk['addr'], desk['addr'], v_two, desk['tmp1']))  # s = 2*s + bit3
                self.emit("valu", ("+", desk['addr'], desk['addr'], v_fp_plus_15))  # addr = s + fp + 15

        # Theory 222: Fused rounds 11+12+13+14 with deferred idx computation
        def emit_rounds_11_12_13_14_fused(group_desks):
            # === Round 11 === (XOR with tree[0] folded into R10's hash)
            # No XOR needed - folded into R10's hash stage 5 via v_c5_xor_t0
            emit_hash_interleaved(group_desks)
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("&", desk['bit0'], desk['val'], v_one))

            # === Round 12 === (vselect for node selection)
            for d in group_desks:
                desk = desks[d]
                self.emit("flow", ("vselect", desk['node_val'], desk['bit0'], v_tree[2], v_tree[1]))
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("^", desk['val'], desk['val'], desk['node_val']))
            emit_hash_interleaved(group_desks)
            for d in group_desks:
                desk = desks[d]
                # Theory 222: only extract bit1, defer idx computation to R14
                self.emit("valu", ("&", desk['bit1'], desk['val'], v_one))

            # === Round 13 === (3 vselect cascade using bit1 register)
            for d in group_desks:
                desk = desks[d]
                self.emit("flow", ("vselect", desk['tmp2'], desk['bit1'], v_tree[4], v_tree[3]))
                self.emit("flow", ("vselect", desk['node_val'], desk['bit1'], v_tree[6], v_tree[5]))
                self.emit("flow", ("vselect", desk['node_val'], desk['bit0'], desk['node_val'], desk['tmp2']))
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("^", desk['val'], desk['val'], desk['node_val']))
            emit_hash_interleaved(group_desks)
            # Theory 222: extract bit2 into idx (safe from hash clobbering)
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("&", desk['idx'], desk['val'], v_one))  # bit2 in idx (safe)

            # === Round 14 === (7-vselect cascade for level-3 node)
            # bit2 is now in idx (not tmp1!)
            for d in group_desks:
                desk = desks[d]
                # bit0=0 side:
                self.emit("flow", ("vselect", desk['tmp2'],     desk['idx'], v_tree[8],  v_tree[7]))
                self.emit("flow", ("vselect", desk['node_val'], desk['idx'], v_tree[10], v_tree[9]))
                self.emit("flow", ("vselect", desk['tmp2'],     desk['bit1'], desk['node_val'], desk['tmp2']))
                # bit0=1 side:
                self.emit("flow", ("vselect", desk['node_val'], desk['idx'], v_tree[12], v_tree[11]))
                self.emit("flow", ("vselect", desk['addr'],     desk['idx'], v_tree[14], v_tree[13]))
                self.emit("flow", ("vselect", desk['node_val'], desk['bit1'], desk['addr'], desk['node_val']))
                # Final:
                self.emit("flow", ("vselect", desk['node_val'], desk['bit0'], desk['node_val'], desk['tmp2']))
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("^", desk['val'], desk['val'], desk['node_val']))
            emit_hash_interleaved(group_desks)
            # Theory 222: Deferred addr computation from bit0/bit1/bit2/bit3
            # bit2 is in idx (preserved through hash)
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("&", desk['tmp1'], desk['val'], v_one))  # bit3 -> tmp1
                self.emit("valu", ("multiply_add", desk['addr'], desk['bit0'], v_two, desk['bit1']))  # s = 2*bit0 + bit1
                self.emit("valu", ("multiply_add", desk['addr'], desk['addr'], v_two, desk['idx']))   # s = 2*s + bit2 (idx=bit2)
                self.emit("valu", ("multiply_add", desk['addr'], desk['addr'], v_two, desk['tmp1']))  # s = 2*s + bit3
                self.emit("valu", ("+", desk['addr'], desk['addr'], v_fp_plus_15))  # addr = s + fp + 15

        def emit_gather_round_interleaved(group_desks):
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("+", desk['addr'], v_forest_p, desk['idx']))
            for d in group_desks:
                desk = desks[d]
                for lane in range(VLEN):
                    self.emit("load", ("load", desk['node_val'] + lane, desk['addr'] + lane))
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("^", desk['val'], desk['val'], desk['node_val']))
            emit_hash_interleaved(group_desks)
            for d in group_desks:
                emit_branch(d)

        def emit_gather_round_addr_tracking(group_desks):
            """Gather round using addr-tracking: addr is already the gather address"""
            # No addr computation needed - addr is already ready from previous branch!
            # Gather
            for d in group_desks:
                desk = desks[d]
                for lane in range(VLEN):
                    self.emit("load", ("load", desk['node_val'] + lane, desk['addr'] + lane))
            # XOR
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("^", desk['val'], desk['val'], desk['node_val']))
            # Hash
            emit_hash_interleaved(group_desks)
            # Branch (updates addr, not idx)
            for d in group_desks:
                emit_branch_addr_tracking(d)

        # *** R10 with branch skip + addr-tracking + R11 XOR fold ***
        def emit_hash_r10_folded(group_desks):
            """R10 hash: fold R11's XOR(tree[0]) into stage 5 by using C5^tree[0]"""
            gd = [group_desks[0], group_desks[3], group_desks[2], group_desks[1]]
            for d in gd:
                desk = desks[d]
                self.emit("valu", ("multiply_add", desk['val'], desk['val'], v_fma_mult[0], v_hash_consts[0]))
                self.emit("valu", ("^", desk['tmp1'], desk['val'], v_hash_consts[1]))
                self.emit("valu", (">>", desk['tmp2'], desk['val'], v_hash_shifts[1]))
                self.emit("valu", ("^", desk['val'], desk['tmp1'], desk['tmp2']))
                self.emit("valu", ("multiply_add", desk['val'], desk['val'], v_fma_mult[2], v_hash_consts[2]))
                self.emit("valu", ("+", desk['tmp1'], desk['val'], v_hash_consts[3]))
                self.emit("valu", ("<<", desk['tmp2'], desk['val'], v_hash_shifts[3]))
                self.emit("valu", ("^", desk['val'], desk['tmp1'], desk['tmp2']))
                self.emit("valu", ("multiply_add", desk['val'], desk['val'], v_fma_mult[4], v_hash_consts[4]))
                self.emit("valu", ("^", desk['tmp1'], desk['val'], v_c5_xor_t0))
                self.emit("valu", (">>", desk['tmp2'], desk['val'], v_hash_shifts[5]))
                self.emit("valu", ("^", desk['val'], desk['tmp1'], desk['tmp2']))

        def emit_round_10_optimized(group_desks):
            """Round 10: addr ready from R9. Skip branch. Fold R11 XOR into hash stage 5."""
            for d in group_desks:
                desk = desks[d]
                for lane in range(VLEN):
                    self.emit("load", ("load", desk['node_val'] + lane, desk['addr'] + lane))
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("^", desk['val'], desk['val'], desk['node_val']))
            emit_hash_r10_folded(group_desks)

        def emit_round_15_final_interleaved(group_desks):
            """R15: addr already ready from R14 addr-tracking. No branch needed."""
            # No addr computation needed - addr is ready from R14's addr-tracking branch!
            for d in group_desks:
                desk = desks[d]
                for lane in range(VLEN):
                    self.emit("load", ("load", desk['node_val'] + lane, desk['addr'] + lane))
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("^", desk['val'], desk['val'], desk['node_val']))
            emit_hash_interleaved(group_desks)

        def emit_tile_interleaved(tile_idx):
            tile_offset = tile_idx * NUM_DESKS * VLEN

            for d in range(NUM_DESKS):
                self.emit("load", ("const", offset_regs[d], tile_offset + d * VLEN))

            for d in range(NUM_DESKS):
                self.emit("alu", ("+", addr_tmp[d*2], self.scratch["inp_indices_p"], offset_regs[d]))
                self.emit("alu", ("+", addr_tmp[d*2+1], self.scratch["inp_values_p"], offset_regs[d]))

            if tile_idx == 0:
                # Exp14b: Load G0 desks first, then tree[1..14] interleaved with G1-G3 desks
                for d in range(4):
                    self.emit("load", ("vload", desks[d]['idx'], addr_tmp[d*2]))
                    self.emit("load", ("vload", desks[d]['val'], addr_tmp[d*2+1]))
                # Now G0 can start R0 (tree[0] already loaded)
                # Load tree[1..14] and remaining desks
                for i in range(1, NUM_PRELOADED):
                    self.emit("alu", ("+", tmp_addr, self.scratch["forest_values_p"], tree_idx_consts[i]))
                    self.emit("load", ("load", tmp_scalar, tmp_addr))
                    self.emit("valu", ("vbroadcast", v_tree[i], tmp_scalar))
                for d in range(4, NUM_DESKS):
                    self.emit("load", ("vload", desks[d]['idx'], addr_tmp[d*2]))
                    self.emit("load", ("vload", desks[d]['val'], addr_tmp[d*2+1]))
            else:
                for d in range(NUM_DESKS):
                    self.emit("load", ("vload", desks[d]['idx'], addr_tmp[d*2]))
                    self.emit("load", ("vload", desks[d]['val'], addr_tmp[d*2+1]))

            GROUP_SIZE = 4
            num_full_groups = NUM_DESKS // GROUP_SIZE

            all_groups = []
            for g in range(num_full_groups):
                all_groups.append(list(range(g * GROUP_SIZE, (g + 1) * GROUP_SIZE)))

            # Shot 66: Interleave groups at round-block level for better scheduling
            for group_desks in all_groups:
                emit_rounds_0_1_2_3_fused(group_desks)
            for group_desks in all_groups:
                for _rnd in range(4, 10):
                    emit_gather_round_addr_tracking(group_desks)
            for group_desks in all_groups:
                emit_round_10_optimized(group_desks)
            for group_desks in all_groups:
                emit_rounds_11_12_13_14_fused(group_desks)
            for group_desks in all_groups:
                emit_round_15_final_interleaved(group_desks)

            for d in range(NUM_DESKS):
                self.emit("store", ("vstore", addr_tmp[d*2], desks[d]['idx']))
                self.emit("store", ("vstore", addr_tmp[d*2+1], desks[d]['val']))

        emit_tile_interleaved(0)
        emit_tile_interleaved(1)

        phases = []
        current_phase = []
        for engine, slot in self.slots:
            if engine == "flow" and slot == ("pause",):
                phases.append(current_phase)
                current_phase = []
            else:
                current_phase.append((engine, slot))
        phases.append(current_phase)

        # Shot 57: Merge init+main phases for unified scheduling,
        # then re-inject pause at cycle 1 (init doesn't modify output).
        if len(phases) >= 2 and phases[0] and phases[1]:
            merged = phases[0] + phases[1]
            merged_instrs = _schedule_slots(merged)
            self.instrs = merged_instrs[:1]
            self.instrs.append({"flow": [("pause",)]})
            self.instrs.extend(merged_instrs[1:])
            self.instrs.append({"flow": [("pause",)]})
        else:
            self.instrs = []
            for i, phase in enumerate(phases):
                if phase:
                    phase_instrs = _schedule_slots(phase)
                    self.instrs.extend(phase_instrs)
                if i < len(phases) - 1:
                    self.instrs.append({"flow": [("pause",)]})
            self.instrs.append({"flow": [("pause",)]})

        # Count VALU ops for reporting
        valu_count = sum(1 for e, s in self.slots if e == "valu")
        print(f"Total slots: {len(self.slots)}, VALU ops: {valu_count}, Cycles: {len(self.instrs)}")


BASELINE = 147734


def do_kernel_test(forest_height: int, rounds: int, batch_size: int, seed: int = 123,
                   trace: bool = False, prints: bool = False, check: bool = False):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)

    value_trace = {}
    machine = Machine(mem, kb.instrs, kb.debug_info(), n_cores=N_CORES, value_trace=value_trace, trace=trace)
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if check:
            assert (machine.mem[inp_values_p : inp_values_p + len(inp.values)]
                    == ref_mem[inp_values_p : inp_values_p + len(inp.values)]), f"Incorrect result on round {i}"

    print("CYCLES: ", machine.cycle)
    print(f"Speedup over baseline: {BASELINE / machine.cycle:.1f}x")
    return machine.cycle


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()

    if args.check:
        cycles = do_kernel_test(10, 16, 256, check=True)
        print(f"Correctness check {'PASSED' if cycles else 'FAILED'}! Cycles: {cycles}")
    else:
        do_kernel_test(10, 16, 256, trace=args.trace)
