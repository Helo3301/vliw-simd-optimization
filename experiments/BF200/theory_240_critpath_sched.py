"""
BF200 Theory 240: Critical-path list scheduler

Based on Theory 229 (best: 1,398 cycles).
Key insight: The greedy ASAP scheduler processes ops in emission order.
A critical-path scheduler assigns priority based on the longest path
from each operation to the end of the DAG. Operations on the critical
path are scheduled first, reducing overall schedule length.

The scheduler build a dependency DAG, computes longest-remaining-path
for each node, then at each cycle picks the ready operation with the
highest priority (longest remaining path).

This should help especially in the drain phase where the current scheduler
might not optimally order the last group's operations.
"""

import random
import argparse
import sys
import os
from collections import defaultdict
import heapq

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


def _schedule_slots_critpath(slots: list[tuple[str, tuple]]) -> list[dict[str, list[tuple]]]:
    """Critical-path list scheduler.

    1. Build dependency DAG
    2. Compute longest-remaining-path for each node
    3. At each cycle, schedule ready ops with highest priority first
    """
    n = len(slots)
    if n == 0:
        return []

    # Pre-compute reads/writes for each op
    rw = [_slot_rw(engine, slot) for engine, slot in slots]

    # Build dependency edges: successors[i] = list of ops that depend on i
    # predecessors_count[i] = number of unscheduled predecessors
    # For each scratch address, track last writer and last readers

    last_writer = {}  # addr -> op_idx
    last_readers = defaultdict(set)  # addr -> set of op_idx

    successors = [[] for _ in range(n)]
    predecessors = [set() for _ in range(n)]

    for i, (engine, slot) in enumerate(slots):
        reads, writes = rw[i]

        deps = set()

        # RAW: read after write dependency
        for addr in reads:
            if addr in last_writer:
                deps.add(last_writer[addr])

        # WAW: write after write dependency
        # WAR: write after read dependency
        for addr in writes:
            if addr in last_writer:
                deps.add(last_writer[addr])
            for reader in last_readers.get(addr, set()):
                deps.add(reader)

        for dep in deps:
            if dep != i:
                successors[dep].append(i)
                predecessors[i].add(dep)

        # Update tracking
        for addr in reads:
            last_readers[addr].add(i)
        for addr in writes:
            last_writer[addr] = i
            # Clear old readers since this write overwrites
            last_readers[addr] = set()

    # Compute longest path from each node to any sink (no successors)
    # Use reverse topological order
    longest_path = [1] * n  # Each op takes 1 cycle

    # Topological sort by processing in reverse order of dependencies
    # Since ops were emitted in a valid order, we can just iterate in reverse
    for i in range(n - 1, -1, -1):
        max_succ = 0
        for j in successors[i]:
            max_succ = max(max_succ, longest_path[j])
        longest_path[i] = 1 + max_succ

    # List scheduling with priority = longest remaining path
    cycles = []
    usage = []
    scheduled_cycle = [-1] * n
    ready_time = [0] * n  # earliest cycle this op can start

    # Compute initial ready time based on predecessors
    unscheduled_preds = [len(predecessors[i]) for i in range(n)]

    # Ready queue: (negative priority, op_idx) -- use negative for max-heap via min-heap
    ready_queue = []

    for i in range(n):
        if unscheduled_preds[i] == 0:
            heapq.heappush(ready_queue, (-longest_path[i], i))

    def ensure_cycle(cycle):
        while len(cycles) <= cycle:
            cycles.append({})
            usage.append(defaultdict(int))

    current_cycle = 0
    scheduled_count = 0

    while scheduled_count < n:
        # Try to schedule ready ops at current_cycle
        deferred = []
        any_scheduled = False

        while ready_queue:
            neg_prio, op_idx = heapq.heappop(ready_queue)
            engine = slots[op_idx][0]
            slot = slots[op_idx][1]

            # Check if op is ready at current_cycle
            if ready_time[op_idx] > current_cycle:
                deferred.append((neg_prio, op_idx))
                continue

            # Check if engine has capacity
            ensure_cycle(current_cycle)
            limit = SLOT_LIMITS[engine]
            if usage[current_cycle][engine] >= limit:
                deferred.append((neg_prio, op_idx))
                continue

            # Schedule it!
            cycles[current_cycle].setdefault(engine, []).append(slot)
            usage[current_cycle][engine] += 1
            scheduled_cycle[op_idx] = current_cycle
            scheduled_count += 1
            any_scheduled = True

            # Update successors
            for j in successors[op_idx]:
                unscheduled_preds[j] -= 1
                # Update ready time: successor can start at current_cycle + 1
                ready_time[j] = max(ready_time[j], current_cycle + 1)
                if unscheduled_preds[j] == 0:
                    heapq.heappush(ready_queue, (-longest_path[j], j))

        # Put deferred ops back
        for item in deferred:
            heapq.heappush(ready_queue, item)

        current_cycle += 1

    return [c for c in cycles if c]


# Also keep the original scheduler for comparison
def _schedule_slots_greedy(slots: list[tuple[str, tuple]]) -> list[dict[str, list[tuple]]]:
    cycles: list[dict[str, list[tuple]]] = []
    usage: list[dict[str, int]] = []
    ready_time: dict[int, int] = defaultdict(int)
    last_write: dict[int, int] = defaultdict(lambda: -1)
    last_read: dict[int, int] = defaultdict(lambda: -1)

    def ensure_cycle(cycle: int) -> None:
        while len(cycles) <= cycle:
            cycles.append({})
            usage.append(defaultdict(int))

    def find_cycle(engine: str, earliest: int) -> int:
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


class KernelBuilderA1:
    """Theory 240: Critical-path scheduler with Theory 229 emission"""
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

    def build_kernel(self, forest_height: int, n_nodes: int, batch_size: int, rounds: int,
                     use_critpath: bool = True):
        tmp_scalar = self.alloc_scratch("tmp_scalar")
        tmp_addr = self.alloc_scratch("tmp_addr")

        fast_init_vars = [("n_nodes", 1), ("forest_values_p", 4), ("inp_indices_p", 5), ("inp_values_p", 6)]
        for var_name, _ in fast_init_vars:
            self.alloc_scratch(var_name)
        for var_name, idx in fast_init_vars:
            self.emit("load", ("const", tmp_scalar, idx))
            self.emit("load", ("load", self.scratch[var_name], tmp_scalar))

        v_zero = self.scratch_vconst(0, "v_zero")
        v_one = self.scratch_vconst(1, "v_one")
        v_two = self.scratch_vconst(2, "v_two")
        v_n_nodes = self.alloc_vec("v_n_nodes")
        self.emit("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]))

        v_forest_p = self.alloc_vec("v_forest_p")
        self.emit("valu", ("vbroadcast", v_forest_p, self.scratch["forest_values_p"]))

        v_1_minus_fp = self.alloc_vec("v_1_minus_fp")
        self.emit("valu", ("-", v_1_minus_fp, v_one, v_forest_p))

        v_fp_plus_1 = self.alloc_vec("v_fp_plus_1")
        self.emit("valu", ("+", v_fp_plus_1, v_forest_p, v_one))

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

        NUM_PRELOADED = 15
        v_tree = []
        for i in range(NUM_PRELOADED):
            v_node = self.alloc_vec(f"v_tree_{i}")
            v_tree.append(v_node)
            self.emit("alu", ("+", tmp_addr, self.scratch["forest_values_p"], self.scratch_const(i)))
            self.emit("load", ("load", tmp_scalar, tmp_addr))
            self.emit("valu", ("vbroadcast", v_node, tmp_scalar))

        DESKS_PER_BANK = 8
        NUM_BANKS = 2
        NUM_DESKS = DESKS_PER_BANK * NUM_BANKS

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

        offset_regs = [[self.alloc_scratch(f"off_b{b}_{d}") for d in range(DESKS_PER_BANK)] for b in range(NUM_BANKS)]
        addr_tmp = [[self.alloc_scratch(f"addr_tmp_b{b}_{i}") for i in range(DESKS_PER_BANK * 2)] for b in range(NUM_BANKS)]

        print(f"Scratch usage: {self.scratch_ptr} / {SCRATCH_SIZE}")

        self.emit("flow", ("pause",))

        def emit_hash_interleaved(group_desks):
            gd = group_desks[:]
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

        def emit_branch_addr_tracking(desk_idx):
            d = desks[desk_idx]
            self.emit("valu", ("&", d['tmp1'], d['val'], v_one))
            self.emit("valu", ("multiply_add", d['addr'], d['addr'], v_two, v_1_minus_fp))
            self.emit("valu", ("+", d['addr'], d['addr'], d['tmp1']))

        def emit_xor_with_node(desk_idx, node_vec):
            d = desks[desk_idx]
            self.emit("valu", ("^", d['val'], d['val'], node_vec))

        def emit_rounds_0_1_2_3_fused(group_desks):
            for d in group_desks:
                emit_xor_with_node(d, v_tree[0])
            emit_hash_interleaved(group_desks)
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("&", desk['bit0'], desk['val'], v_one))
            for d in group_desks:
                desk = desks[d]
                self.emit("flow", ("vselect", desk['node_val'], desk['bit0'], v_tree[2], v_tree[1]))
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("^", desk['val'], desk['val'], desk['node_val']))
            emit_hash_interleaved(group_desks)
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("&", desk['bit1'], desk['val'], v_one))
            for d in group_desks:
                desk = desks[d]
                self.emit("flow", ("vselect", desk['tmp2'], desk['bit1'], v_tree[4], v_tree[3]))
                self.emit("flow", ("vselect", desk['node_val'], desk['bit1'], v_tree[6], v_tree[5]))
                self.emit("flow", ("vselect", desk['node_val'], desk['bit0'], desk['node_val'], desk['tmp2']))
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("^", desk['val'], desk['val'], desk['node_val']))
            emit_hash_interleaved(group_desks)
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("&", desk['idx'], desk['val'], v_one))
            for d in group_desks:
                desk = desks[d]
                self.emit("flow", ("vselect", desk['tmp2'],     desk['idx'], v_tree[8],  v_tree[7]))
                self.emit("flow", ("vselect", desk['node_val'], desk['idx'], v_tree[10], v_tree[9]))
                self.emit("flow", ("vselect", desk['tmp2'],     desk['bit1'], desk['node_val'], desk['tmp2']))
                self.emit("flow", ("vselect", desk['node_val'], desk['idx'], v_tree[12], v_tree[11]))
                self.emit("flow", ("vselect", desk['addr'],     desk['idx'], v_tree[14], v_tree[13]))
                self.emit("flow", ("vselect", desk['node_val'], desk['bit1'], desk['addr'], desk['node_val']))
                self.emit("flow", ("vselect", desk['node_val'], desk['bit0'], desk['node_val'], desk['tmp2']))
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("^", desk['val'], desk['val'], desk['node_val']))
            emit_hash_interleaved(group_desks)
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("&", desk['tmp1'], desk['val'], v_one))
                self.emit("valu", ("multiply_add", desk['addr'], desk['bit0'], v_two, desk['bit1']))
                self.emit("valu", ("multiply_add", desk['addr'], desk['addr'], v_two, desk['idx']))
                self.emit("valu", ("multiply_add", desk['addr'], desk['addr'], v_two, desk['tmp1']))
                self.emit("valu", ("+", desk['addr'], desk['addr'], v_fp_plus_15))

        def emit_rounds_11_12_13_14_fused(group_desks):
            for d in group_desks:
                emit_xor_with_node(d, v_tree[0])
            emit_hash_interleaved(group_desks)
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("&", desk['bit0'], desk['val'], v_one))
            for d in group_desks:
                desk = desks[d]
                self.emit("flow", ("vselect", desk['node_val'], desk['bit0'], v_tree[2], v_tree[1]))
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("^", desk['val'], desk['val'], desk['node_val']))
            emit_hash_interleaved(group_desks)
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("&", desk['bit1'], desk['val'], v_one))
            for d in group_desks:
                desk = desks[d]
                self.emit("flow", ("vselect", desk['tmp2'], desk['bit1'], v_tree[4], v_tree[3]))
                self.emit("flow", ("vselect", desk['node_val'], desk['bit1'], v_tree[6], v_tree[5]))
                self.emit("flow", ("vselect", desk['node_val'], desk['bit0'], desk['node_val'], desk['tmp2']))
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("^", desk['val'], desk['val'], desk['node_val']))
            emit_hash_interleaved(group_desks)
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("&", desk['idx'], desk['val'], v_one))
            for d in group_desks:
                desk = desks[d]
                self.emit("flow", ("vselect", desk['tmp2'],     desk['idx'], v_tree[8],  v_tree[7]))
                self.emit("flow", ("vselect", desk['node_val'], desk['idx'], v_tree[10], v_tree[9]))
                self.emit("flow", ("vselect", desk['tmp2'],     desk['bit1'], desk['node_val'], desk['tmp2']))
                self.emit("flow", ("vselect", desk['node_val'], desk['idx'], v_tree[12], v_tree[11]))
                self.emit("flow", ("vselect", desk['addr'],     desk['idx'], v_tree[14], v_tree[13]))
                self.emit("flow", ("vselect", desk['node_val'], desk['bit1'], desk['addr'], desk['node_val']))
                self.emit("flow", ("vselect", desk['node_val'], desk['bit0'], desk['node_val'], desk['tmp2']))
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("^", desk['val'], desk['val'], desk['node_val']))
            emit_hash_interleaved(group_desks)
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("&", desk['tmp1'], desk['val'], v_one))
                self.emit("valu", ("multiply_add", desk['addr'], desk['bit0'], v_two, desk['bit1']))
                self.emit("valu", ("multiply_add", desk['addr'], desk['addr'], v_two, desk['idx']))
                self.emit("valu", ("multiply_add", desk['addr'], desk['addr'], v_two, desk['tmp1']))
                self.emit("valu", ("+", desk['addr'], desk['addr'], v_fp_plus_15))

        def emit_gather_round_addr_tracking(group_desks):
            for d in group_desks:
                desk = desks[d]
                for lane in range(VLEN):
                    self.emit("load", ("load", desk['node_val'] + lane, desk['addr'] + lane))
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("^", desk['val'], desk['val'], desk['node_val']))
            emit_hash_interleaved(group_desks)
            for d in group_desks:
                emit_branch_addr_tracking(d)

        def emit_round_10_optimized(group_desks):
            for d in group_desks:
                desk = desks[d]
                for lane in range(VLEN):
                    self.emit("load", ("load", desk['node_val'] + lane, desk['addr'] + lane))
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("^", desk['val'], desk['val'], desk['node_val']))
            emit_hash_interleaved(group_desks)

        def emit_round_15_final_interleaved(group_desks):
            for d in group_desks:
                desk = desks[d]
                for lane in range(VLEN):
                    self.emit("load", ("load", desk['node_val'] + lane, desk['addr'] + lane))
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("^", desk['val'], desk['val'], desk['node_val']))
            emit_hash_interleaved(group_desks)

        def emit_all_rounds_for_group(group_desks):
            emit_rounds_0_1_2_3_fused(group_desks)
            for _rnd in range(4, 10):
                emit_gather_round_addr_tracking(group_desks)
            emit_round_10_optimized(group_desks)
            emit_rounds_11_12_13_14_fused(group_desks)
            emit_round_15_final_interleaved(group_desks)

        def emit_tile_load(tile_idx, bank_idx):
            tile_offset = tile_idx * DESKS_PER_BANK * VLEN
            bank_desk_start = bank_idx * DESKS_PER_BANK
            for d in range(DESKS_PER_BANK):
                self.emit("load", ("const", offset_regs[bank_idx][d], tile_offset + d * VLEN))
            for d in range(DESKS_PER_BANK):
                self.emit("alu", ("+", addr_tmp[bank_idx][d*2], self.scratch["inp_indices_p"], offset_regs[bank_idx][d]))
                self.emit("alu", ("+", addr_tmp[bank_idx][d*2+1], self.scratch["inp_values_p"], offset_regs[bank_idx][d]))
            for d in range(DESKS_PER_BANK):
                global_desk = bank_desk_start + d
                self.emit("load", ("vload", desks[global_desk]['idx'], addr_tmp[bank_idx][d*2]))
                self.emit("load", ("vload", desks[global_desk]['val'], addr_tmp[bank_idx][d*2+1]))

        def emit_tile_store(bank_idx):
            bank_desk_start = bank_idx * DESKS_PER_BANK
            for d in range(DESKS_PER_BANK):
                global_desk = bank_desk_start + d
                self.emit("store", ("vstore", addr_tmp[bank_idx][d*2], desks[global_desk]['idx']))
                self.emit("store", ("vstore", addr_tmp[bank_idx][d*2+1], desks[global_desk]['val']))

        NUM_TILES = batch_size // (DESKS_PER_BANK * VLEN)
        GROUP_SIZE = 4
        num_groups = DESKS_PER_BANK // GROUP_SIZE

        for pair in range(NUM_TILES // 2):
            tile_a = pair * 2
            tile_b = pair * 2 + 1
            emit_tile_load(tile_a, 0)
            emit_tile_load(tile_b, 1)
            for g in range(num_groups):
                group_desks_a = list(range(0 + g * GROUP_SIZE, 0 + (g + 1) * GROUP_SIZE))
                emit_all_rounds_for_group(group_desks_a)
                group_desks_b = list(range(DESKS_PER_BANK + g * GROUP_SIZE,
                                          DESKS_PER_BANK + (g + 1) * GROUP_SIZE))
                emit_all_rounds_for_group(group_desks_b)
            emit_tile_store(0)
            emit_tile_store(1)

        phases = []
        current_phase = []
        for engine, slot in self.slots:
            if engine == "flow" and slot == ("pause",):
                phases.append(current_phase)
                current_phase = []
            else:
                current_phase.append((engine, slot))
        phases.append(current_phase)

        scheduler = _schedule_slots_critpath if use_critpath else _schedule_slots_greedy

        self.instrs = []
        for i, phase in enumerate(phases):
            if phase:
                print(f"Scheduling phase {i} ({len(phase)} ops)...")
                phase_instrs = scheduler(phase)
                self.instrs.extend(phase_instrs)
            if i < len(phases) - 1:
                self.instrs.append({"flow": [("pause",)]})

        self.instrs.append({"flow": [("pause",)]})

        valu_count = sum(1 for e, s in self.slots if e == "valu")
        print(f"Total slots: {len(self.slots)}, VALU ops: {valu_count}, Cycles: {len(self.instrs)}")


BASELINE = 147734


def do_kernel_test(forest_height: int, rounds: int, batch_size: int, seed: int = 123,
                   trace: bool = False, prints: bool = False, check: bool = False,
                   use_critpath: bool = True):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilderA1()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds,
                    use_critpath=use_critpath)

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
    return machine.cycle


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--greedy", action="store_true", help="Use greedy scheduler instead of critpath")
    args = parser.parse_args()

    use_critpath = not args.greedy

    if args.check:
        cycles = do_kernel_test(10, 16, 256, check=True, use_critpath=use_critpath)
        print(f"Correctness check {'PASSED' if cycles else 'FAILED'}! Cycles: {cycles}")
    else:
        do_kernel_test(10, 16, 256, trace=args.trace, use_critpath=use_critpath)
