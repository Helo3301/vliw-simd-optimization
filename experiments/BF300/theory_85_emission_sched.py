"""
BF200 Theory 1: Address-Tracking Branch Fusion

Instead of tracking idx and computing addr = forest_p + idx every gather round,
track addr directly. The branch formula becomes:

Current: idx = 2*idx + 1 + bit (3 VALU) + addr = forest_p + idx (1 VALU) = 4 VALU
Addr-tracking: addr = 2*addr + (1-fp) (1 FMA) + addr = addr + bit (1 ADD) = 3 VALU total
  (AND for bit extraction is still needed, so branch = 3 VALU, but no separate addr!)

Savings per desk:
  R3-R9: 7 rounds x 1 addr saved = 7
  R10: addr ready from R9 = 1
  R14: 1 addr saved
  R15: addr ready from R14 = 1
  Cost: 2 idx->addr conversions (after R2, after R13)
  Init: 1 VALU for v_1_minus_fp
Net: 10 - 2 = 8 per desk, minus 1 init = 8*32 - 1 = 255 VALU saved
"""

import random
import argparse
import sys
import os
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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


def _schedule_slots(slots: list[tuple[str, tuple]]) -> list[dict[str, list[tuple]]]:
    """Ready-set scheduler with emission-order-preserving heuristics."""
    
    n = len(slots)
    if n == 0:
        return []
    
    engines = [s[0] for s in slots]
    ops = [s[1] for s in slots]
    all_rw = [_slot_rw(engines[i], ops[i]) for i in range(n)]
    
    # Build dependency graph
    reg_writer = {}
    reg_readers = defaultdict(list)
    
    pred_set = [set() for _ in range(n)]
    succ_list = [[] for _ in range(n)]
    
    for i in range(n):
        reads_i, writes_i = all_rw[i]
        preds = set()
        
        for r in reads_i:
            if r in reg_writer:
                preds.add(reg_writer[r])
        
        for w in writes_i:
            if w in reg_writer:
                preds.add(reg_writer[w])
            for reader in reg_readers.get(w, []):
                preds.add(reader)
        
        pred_set[i] = preds
        for p in preds:
            succ_list[p].append(i)
        
        for r in reads_i:
            reg_readers[r].append(i)
        for w in writes_i:
            reg_writer[w] = i
            reg_readers[w] = []
    
    # Compute height (longest path from each node to any sink)
    # This serves as urgency metric
    height = [0] * n
    # Process in reverse topological order
    out_degree = [len(succ_list[i]) for i in range(n)]
    rev_queue = [i for i in range(n) if out_degree[i] == 0]
    rev_order = []
    while rev_queue:
        i = rev_queue.pop()
        rev_order.append(i)
        for p in pred_set[i]:
            out_degree[p] -= 1
            if out_degree[p] == 0:
                rev_queue.append(p)
    
    for i in rev_order:
        max_h = 0
        for s in succ_list[i]:
            max_h = max(max_h, height[s] + 1)
        height[i] = max_h
    
    # Initialize
    pred_count = [len(pred_set[i]) for i in range(n)]
    earliest_cycle = [0] * n
    scheduled = [False] * n
    
    # Ready set: ops with no unscheduled predecessors
    # Sort by: (-height, emission_order) for urgency-based scheduling
    import bisect
    
    ready = []  # sorted list of (priority_key, op_idx)
    for i in range(n):
        if pred_count[i] == 0:
            # Priority: higher height = more urgent. Break ties by emission order.
            ready.append((i, i))
    ready.sort()
    
    cycles_out = []
    usage_out = []
    current_cycle = 0
    ops_done = 0
    
    while ops_done < n:
        while len(cycles_out) <= current_cycle:
            cycles_out.append({})
            usage_out.append(defaultdict(int))
        
        # Pack as many ready ops into current_cycle as possible
        remaining = []
        just_done = []
        
        for key, idx in ready:
            if earliest_cycle[idx] > current_cycle:
                remaining.append((key, idx))
                continue
            
            eng = engines[idx]
            if usage_out[current_cycle][eng] < SLOT_LIMITS[eng]:
                cycles_out[current_cycle].setdefault(eng, []).append(ops[idx])
                usage_out[current_cycle][eng] += 1
                scheduled[idx] = True
                ops_done += 1
                just_done.append(idx)
            else:
                remaining.append((key, idx))
        
        # Notify successors
        newly_ready = []
        for idx in just_done:
            for s in succ_list[idx]:
                pred_count[s] -= 1
                if pred_count[s] == 0:
                    earliest_cycle[s] = current_cycle + 1
                    newly_ready.append((s, s))
        
        if newly_ready:
            newly_ready.sort()
            # Merge remaining and newly_ready (both sorted)
            merged = []
            i, j = 0, 0
            while i < len(remaining) and j < len(newly_ready):
                if remaining[i] <= newly_ready[j]:
                    merged.append(remaining[i])
                    i += 1
                else:
                    merged.append(newly_ready[j])
                    j += 1
            merged.extend(remaining[i:])
            merged.extend(newly_ready[j:])
            ready = merged
        else:
            ready = remaining
        
        current_cycle += 1
    
    return [c for c in cycles_out if c]



class KernelBuilderA1:
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

        fast_init_vars = [("n_nodes", 1), ("forest_values_p", 4), ("inp_indices_p", 5), ("inp_values_p", 6)]
        for var_name, _ in fast_init_vars:
            self.alloc_scratch(var_name)
        for var_name, idx in fast_init_vars:
            self.emit("load", ("const", tmp_scalar, idx))
            self.emit("load", ("load", self.scratch[var_name], tmp_scalar))

        # v_zero removed (unused)
        v_one = self.scratch_vconst(1, "v_one")
        v_two = self.scratch_vconst(2, "v_two")
        v_three = self.scratch_vconst(3, "v_three")
        # Theory 22: v_n_nodes removed (unused in round processing)
        # v_n_nodes = self.alloc_vec("v_n_nodes")
        # self.emit("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]))

        v_forest_p = self.alloc_vec("v_forest_p")
        self.emit("valu", ("vbroadcast", v_forest_p, self.scratch["forest_values_p"]))

        # THEORY 1: Precompute v_1_minus_fp = 1 - forest_p for addr-tracking branch
        v_1_minus_fp = self.alloc_vec("v_1_minus_fp")
        self.emit("valu", ("-", v_1_minus_fp, v_one, v_forest_p))

        # THEORY 1b: Precompute v_fp_plus_1 = forest_p + 1 for combined branch+conversion
        v_fp_plus_1 = self.alloc_vec("v_fp_plus_1")
        self.emit("valu", ("+", v_fp_plus_1, v_forest_p, v_one))

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

        NUM_PRELOADED = 15  # tree[0]..tree[14] = levels 0,1,2,3
        v_tree = []
        for i in range(NUM_PRELOADED):
            v_node = self.alloc_vec(f"v_tree_{i}")
            v_tree.append(v_node)
            self.emit("alu", ("+", tmp_addr, self.scratch["forest_values_p"], self.scratch_const(i)))
            self.emit("load", ("load", tmp_scalar, tmp_addr))
            self.emit("valu", ("vbroadcast", v_node, tmp_scalar))

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
                'bit1': self.alloc_vec(f"v_bit1_{d}"),
            }
            desks.append(desk)

        offset_regs = [self.alloc_scratch(f"off_{d}") for d in range(NUM_DESKS)]
        addr_tmp = [self.alloc_scratch(f"addr_tmp_{i}") for i in range(32)]

        print(f"Scratch usage: {self.scratch_ptr} / {SCRATCH_SIZE}")

        self.emit("flow", ("pause",))

        def emit_hash_interleaved(group_desks):
            # Interleave desk order (even first, then odd) + per-desk hash (all stages per desk)
            gd = [group_desks[i] for i in [1, 0, 2, 3]]
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

        def emit_branch_idx_to_addr(desk_idx):
            """Combined branch + idx->addr conversion. Computes addr = fp + 2*idx + 1 + bit.
            Uses: addr = FMA(idx, 2, fp+1+bit). Doesn't update idx.
            Saves 1 VALU vs separate branch + conversion."""
            d = desks[desk_idx]
            self.emit("valu", ("&", d['tmp1'], d['val'], v_one))          # bit = val & 1
            self.emit("valu", ("+", d['tmp1'], v_fp_plus_1, d['tmp1']))    # t = fp + 1 + bit
            self.emit("valu", ("multiply_add", d['addr'], d['idx'], v_two, d['tmp1']))  # addr = 2*idx + t

        def emit_branch_idx_to_addr_save_bit2(desk_idx):
            """Same as emit_branch_idx_to_addr but also saves bit2 = val & 1 to tmp2.
            Used before level-3 preloaded rounds."""
            d = desks[desk_idx]
            self.emit("valu", ("&", d['tmp2'], d['val'], v_one))          # bit2 = val & 1 (saved to tmp2!)
            self.emit("valu", ("+", d['tmp1'], v_fp_plus_1, d['tmp2']))    # t = fp + 1 + bit
            self.emit("valu", ("multiply_add", d['addr'], d['idx'], v_two, d['tmp1']))  # addr = 2*idx + t

        def emit_xor_with_node(desk_idx, node_vec):
            d = desks[desk_idx]
            self.emit("valu", ("^", d['val'], d['val'], node_vec))

        # Fused rounds 0+1+2 with bit tracking + bit1 save for Theory 30
        def emit_rounds_0_1_2_fused(group_desks):
            # === Round 0 ===
            for d in group_desks:
                emit_xor_with_node(d, v_tree[0])
            emit_hash_interleaved(group_desks)
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("&", desk['bit0'], desk['val'], v_one))
                self.emit("valu", ("+", desk['idx'], v_one, desk['bit0']))

            # === Round 1 === (vselect for node selection: saves 1 VALU per desk)
            for d in group_desks:
                desk = desks[d]
                self.emit("flow", ("vselect", desk['node_val'], desk['bit0'], v_tree[2], v_tree[1]))
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("^", desk['val'], desk['val'], desk['node_val']))
            emit_hash_interleaved(group_desks)
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("&", desk['tmp1'], desk['val'], v_one))
                # THEORY 30: Save bit1 for level-3 vselect
                self.emit("flow", ("vselect", desk['bit1'], v_one, desk['tmp1'], desk['tmp1']))  # bit1 = tmp1 (flow copy)
                self.emit("valu", ("multiply_add", desk['idx'], desk['bit0'], v_two, v_three))
                self.emit("valu", ("+", desk['idx'], desk['idx'], desk['tmp1']))

            # === Round 2 === (3 vselect for node selection)
            for d in group_desks:
                desk = desks[d]
                self.emit("flow", ("vselect", desk['tmp2'], desk['tmp1'], v_tree[4], v_tree[3]))
                self.emit("flow", ("vselect", desk['node_val'], desk['tmp1'], v_tree[6], v_tree[5]))
                self.emit("flow", ("vselect", desk['node_val'], desk['bit0'], desk['node_val'], desk['tmp2']))
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("^", desk['val'], desk['val'], desk['node_val']))
            emit_hash_interleaved(group_desks)
            # Theory 1b + 30: Combined branch + idx->addr + save bit2
            for d in group_desks:
                emit_branch_idx_to_addr_save_bit2(d)

        # Fused rounds 11+12+13 with bit1 save for Theory 30
        def emit_rounds_11_12_13_fused(group_desks):
            # === Round 11 ===
            for d in group_desks:
                emit_xor_with_node(d, v_tree[0])
            emit_hash_interleaved(group_desks)
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("&", desk['bit0'], desk['val'], v_one))
                self.emit("valu", ("+", desk['idx'], v_one, desk['bit0']))

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
                self.emit("valu", ("&", desk['tmp1'], desk['val'], v_one))
                # THEORY 30: Save bit1 for level-3 vselect
                self.emit("flow", ("vselect", desk['bit1'], v_one, desk['tmp1'], desk['tmp1']))  # bit1 = tmp1 (flow copy)
                self.emit("valu", ("multiply_add", desk['idx'], desk['bit0'], v_two, v_three))
                self.emit("valu", ("+", desk['idx'], desk['idx'], desk['tmp1']))

            # === Round 13 === (3 vselect for node selection)
            for d in group_desks:
                desk = desks[d]
                self.emit("flow", ("vselect", desk['tmp2'], desk['tmp1'], v_tree[4], v_tree[3]))
                self.emit("flow", ("vselect", desk['node_val'], desk['tmp1'], v_tree[6], v_tree[5]))
                self.emit("flow", ("vselect", desk['node_val'], desk['bit0'], desk['node_val'], desk['tmp2']))
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("^", desk['val'], desk['val'], desk['node_val']))
            emit_hash_interleaved(group_desks)
            # Theory 1b + 30: Combined branch + idx->addr + save bit2
            for d in group_desks:
                emit_branch_idx_to_addr_save_bit2(d)

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

        def emit_round_level3_preloaded_v2(group_desks):
            """Round at level 3: 8-way vselect from tree[7..14] using bit0, bit1, bit2.
            bit2 is saved in desk['tmp2'] by emit_branch_idx_to_addr_save_bit2.
            bit1 is in desk['bit1'] (saved during fused rounds).
            bit0 is in desk['bit0'] (saved during R0/R11).
            Uses desk['idx'] as extra temp (not needed until R10/R11)."""
            for d in group_desks:
                desk = desks[d]
                bit0 = desk['bit0']
                bit1 = desk['bit1']
                bit2 = desk['tmp2']  # bit2 saved by emit_branch_idx_to_addr_save_bit2
                nv = desk['node_val']
                t1 = desk['idx']     # use idx as temp (free until R10/R11)
                t2 = desk['tmp1']    # tmp1 is free (was overwritten by branch)
                
                # 8 nodes tree[7..14], indexed by 7 + bit0*4 + bit1*2 + bit2
                # Level 0 (bit2): select within 4 consecutive pairs
                self.emit("flow", ("vselect", t1, bit2, v_tree[8], v_tree[7]))     # pair(0,0)
                self.emit("flow", ("vselect", t2, bit2, v_tree[10], v_tree[9]))     # pair(0,1)
                # Level 1 (bit1): select between pairs
                self.emit("flow", ("vselect", t1, bit1, t2, t1))                   # half(0)
                self.emit("flow", ("vselect", t2, bit2, v_tree[12], v_tree[11]))    # pair(1,0)
                self.emit("flow", ("vselect", nv, bit2, v_tree[14], v_tree[13]))    # pair(1,1)
                self.emit("flow", ("vselect", nv, bit1, nv, t2))                   # half(1)
                # Level 2 (bit0): final selection
                self.emit("flow", ("vselect", nv, bit0, nv, t1))                   # result
            
            # XOR
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("^", desk['val'], desk['val'], desk['node_val']))
            # Hash
            emit_hash_interleaved(group_desks)
            # Branch (addr-tracking: updates addr for next gather round)
            for d in group_desks:
                emit_branch_addr_tracking(d)

        # *** R10 with branch skip + addr-tracking ***
        def emit_round_10_optimized(group_desks):
            """Round 10: addr already ready from R9 addr-tracking. Skip branch, set idx=0."""
            # No addr compute needed - addr is ready from R9's addr-tracking branch!
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
            # Theory 21: R10 idx=0 XOR removed (R11 sets idx=1+bit0 without reading old idx)
            pass

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

            for d in range(NUM_DESKS):
                self.emit("load", ("vload", desks[d]['idx'], addr_tmp[d*2]))
                self.emit("load", ("vload", desks[d]['val'], addr_tmp[d*2+1]))

            GROUP_SIZE = 4
            num_full_groups = NUM_DESKS // GROUP_SIZE

            all_groups = []
            for g in range(num_full_groups):
                all_groups.append(list(range(g * GROUP_SIZE, (g + 1) * GROUP_SIZE)))

            for group_desks in all_groups:
                # Fused rounds 0+1+2 (R2 branch now directly produces addr)
                emit_rounds_0_1_2_fused(group_desks)

                # THEORY 30: Round 3 uses preloaded level-3 nodes
                emit_round_level3_preloaded_v2(group_desks)

                # Rounds 4-9: addr-tracking gather rounds
                for _rnd in range(4, 10):
                    emit_gather_round_addr_tracking(group_desks)

                # Round 10: addr ready from R9, skip branch, set idx=0
                emit_round_10_optimized(group_desks)

                # Fused rounds 11+12+13 (R13 branch now directly produces addr)
                emit_rounds_11_12_13_fused(group_desks)

                # THEORY 30: Round 14 uses preloaded level-3 nodes  
                emit_round_level3_preloaded_v2(group_desks)

                # Round 15: final, addr ready from R14 (no branch, no addr compute)
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

    kb = KernelBuilderA1()
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
    print("vs B4-2 baseline (1558): ", 1558 - machine.cycle)
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
