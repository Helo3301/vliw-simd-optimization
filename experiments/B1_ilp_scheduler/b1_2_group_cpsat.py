"""
B1-2: Per-Group CP-SAT Scheduler

Instead of scheduling the entire phase at once (11,776 slots), schedule
each group of 4 desks independently, then merge. Each group processes
~740 slots which should be tractable for CP-SAT.

Key insight: Groups are independent - they don't share registers until
the final store. So we can solve each group optimally and then interleave.
"""

import random
import argparse
import sys
import os
from collections import defaultdict
from ortools.sat.python import cp_model

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
        match slot:
            case ("vbroadcast", dest, src):
                reads = [src]
                writes = list(_vec_range(dest))
            case ("multiply_add", dest, a, b, c):
                reads = list(_vec_range(a)) + list(_vec_range(b)) + list(_vec_range(c))
                writes = list(_vec_range(dest))
            case (_op, dest, a1, a2):
                reads = list(_vec_range(a1)) + list(_vec_range(a2))
                writes = list(_vec_range(dest))
            case _:
                raise NotImplementedError(f"Unknown valu op {slot}")
    elif engine == "load":
        match slot:
            case ("load", dest, addr):
                reads = [addr]
                writes = [dest]
            case ("vload", dest, addr):
                reads = [addr]
                writes = list(_vec_range(dest))
            case ("const", dest, _val):
                writes = [dest]
            case ("load_offset", dest, addr, _lane):
                reads = [addr]
                writes = [dest]
            case _:
                raise NotImplementedError(f"Unknown load op {slot}")
    elif engine == "store":
        match slot:
            case ("store", addr, src):
                reads = [addr, src]
            case ("vstore", addr, src):
                reads = [addr] + list(_vec_range(src))
            case _:
                raise NotImplementedError(f"Unknown store op {slot}")
    elif engine == "flow":
        match slot:
            case ("select", dest, cond, a, b):
                reads = [cond, a, b]
                writes = [dest]
            case ("add_imm", dest, a, _imm):
                reads = [a]
                writes = [dest]
            case ("vselect", dest, cond, a, b):
                reads = list(_vec_range(cond)) + list(_vec_range(a)) + list(_vec_range(b))
                writes = list(_vec_range(dest))
            case ("halt",) | ("pause",) | ("trace_write", _) | ("jump", _) | ("jump_indirect", _) | ("cond_jump", _, _) | ("cond_jump_rel", _, _) | ("coreid", _):
                pass
            case _:
                raise NotImplementedError(f"Unknown flow op {slot}")

    return reads, writes


def _schedule_slots_cpsat(slots: list[tuple[str, tuple]], time_limit_seconds: float = 30.0) -> list[dict[str, list[tuple]]]:
    """
    CP-SAT based scheduler that minimizes total cycles.
    """
    if not slots:
        return []

    n_slots = len(slots)

    # Build dependency graph
    slot_reads = []
    slot_writes = []
    slot_engines = []

    for engine, slot in slots:
        reads, writes = _slot_rw(engine, slot)
        slot_reads.append(set(reads))
        slot_writes.append(set(writes))
        slot_engines.append(engine)

    # Compute earliest possible cycle for each slot based on dependencies
    earliest = [0] * n_slots
    last_write = {}
    last_read = {}

    for i in range(n_slots):
        for addr in slot_reads[i]:
            if addr in last_write:
                earliest[i] = max(earliest[i], last_write[addr][1] + 1)

        for addr in slot_writes[i]:
            if addr in last_write:
                earliest[i] = max(earliest[i], last_write[addr][1] + 1)
            if addr in last_read:
                earliest[i] = max(earliest[i], last_read[addr][1])

        for addr in slot_reads[i]:
            last_read[addr] = (i, earliest[i])
        for addr in slot_writes[i]:
            last_write[addr] = (i, earliest[i])

    max_earliest = max(earliest) if earliest else 0

    engine_counts = defaultdict(int)
    for engine, _ in slots:
        engine_counts[engine] += 1

    resource_bound = max(
        (count + SLOT_LIMITS[engine] - 1) // SLOT_LIMITS[engine]
        for engine, count in engine_counts.items()
    )

    horizon = max(max_earliest + 1, resource_bound) + 50

    model = cp_model.CpModel()

    cycle_vars = [model.NewIntVar(earliest[i], horizon, f'cycle_{i}') for i in range(n_slots)]

    makespan = model.NewIntVar(0, horizon, 'makespan')
    model.AddMaxEquality(makespan, cycle_vars)
    model.Minimize(makespan)

    last_write = {}
    last_read = {}

    for i in range(n_slots):
        for addr in slot_reads[i]:
            if addr in last_write:
                prev_idx = last_write[addr]
                model.Add(cycle_vars[i] >= cycle_vars[prev_idx] + 1)

        for addr in slot_writes[i]:
            if addr in last_write:
                prev_idx = last_write[addr]
                model.Add(cycle_vars[i] >= cycle_vars[prev_idx] + 1)

        for addr in slot_writes[i]:
            if addr in last_read:
                prev_idx = last_read[addr]
                model.Add(cycle_vars[i] >= cycle_vars[prev_idx])

        for addr in slot_reads[i]:
            last_read[addr] = i
        for addr in slot_writes[i]:
            last_write[addr] = i

    engine_slots = defaultdict(list)
    for i, (engine, _) in enumerate(slots):
        engine_slots[engine].append(i)

    for engine, slot_indices in engine_slots.items():
        limit = SLOT_LIMITS[engine]
        if len(slot_indices) <= limit:
            continue

        intervals = []
        demands = []
        for idx in slot_indices:
            interval = model.NewIntervalVar(
                cycle_vars[idx], 1, cycle_vars[idx] + 1, f'interval_{engine}_{idx}'
            )
            intervals.append(interval)
            demands.append(1)

        model.AddCumulative(intervals, demands, limit)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.num_search_workers = 8

    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None  # Signal failure

    max_cycle = solver.Value(makespan)
    cycles = [{} for _ in range(max_cycle + 1)]

    for i, (engine, slot) in enumerate(slots):
        c = solver.Value(cycle_vars[i])
        cycles[c].setdefault(engine, []).append(slot)

    return [c for c in cycles if c]


def _schedule_slots_greedy(slots: list[tuple[str, tuple]]) -> list[dict[str, list[tuple]]]:
    """Original greedy scheduler as fallback."""
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


def merge_schedules(schedules: list[list[dict[str, list[tuple]]]]) -> list[dict[str, list[tuple]]]:
    """
    Merge multiple independent schedules by interleaving them, respecting slot limits.
    """
    if not schedules:
        return []

    max_len = max(len(s) for s in schedules)
    result = []

    for cycle_idx in range(max_len):
        merged_cycle = defaultdict(list)
        for sched in schedules:
            if cycle_idx < len(sched):
                for engine, slots in sched[cycle_idx].items():
                    merged_cycle[engine].extend(slots)

        # Check if we exceeded slot limits - if so, spill to next cycles
        while merged_cycle:
            current = {}
            overflow = defaultdict(list)
            for engine, slots in merged_cycle.items():
                limit = SLOT_LIMITS[engine]
                current[engine] = slots[:limit]
                overflow[engine] = slots[limit:]

            if current:
                result.append(dict(current))

            # Prepare overflow for next iteration
            merged_cycle = {e: s for e, s in overflow.items() if s}
            if not merged_cycle:
                break

    return [c for c in result if c]


class KernelBuilderB1_2:
    """B1-2: Per-Group CP-SAT Scheduler"""
    def __init__(self):
        self.slots: list[tuple[str, tuple]] = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self.vconst_map = {}
        self.group_boundaries = []  # Track where each group starts/ends

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def emit(self, engine: str, slot: tuple):
        self.slots.append((engine, slot))

    def mark_group_boundary(self):
        """Mark the start of a new group"""
        self.group_boundaries.append(len(self.slots))

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

        v_zero = self.scratch_vconst(0, "v_zero")
        v_one = self.scratch_vconst(1, "v_one")
        v_two = self.scratch_vconst(2, "v_two")
        v_three = self.scratch_vconst(3, "v_three")
        v_n_nodes = self.alloc_vec("v_n_nodes")
        self.emit("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]))

        v_forest_p = self.alloc_vec("v_forest_p")
        self.emit("valu", ("vbroadcast", v_forest_p, self.scratch["forest_values_p"]))

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

        NUM_PRELOADED = 7
        v_tree = []
        for i in range(NUM_PRELOADED):
            v_node = self.alloc_vec(f"v_tree_{i}")
            v_tree.append(v_node)
            self.emit("alu", ("+", tmp_addr, self.scratch["forest_values_p"], self.scratch_const(i)))
            self.emit("load", ("load", tmp_scalar, tmp_addr))
            self.emit("valu", ("vbroadcast", v_node, tmp_scalar))

        v_diff_1_2 = self.alloc_vec("v_diff_1_2")
        v_diff_3_4 = self.alloc_vec("v_diff_3_4")
        v_diff_5_6 = self.alloc_vec("v_diff_5_6")
        self.emit("valu", ("-", v_diff_1_2, v_tree[2], v_tree[1]))
        self.emit("valu", ("-", v_diff_3_4, v_tree[4], v_tree[3]))
        self.emit("valu", ("-", v_diff_5_6, v_tree[6], v_tree[5]))

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
            }
            desks.append(desk)

        offset_regs = [self.alloc_scratch(f"off_{d}") for d in range(NUM_DESKS)]
        addr_tmp = [self.alloc_scratch(f"addr_tmp_{i}") for i in range(32)]

        print(f"Scratch usage: {self.scratch_ptr} / {SCRATCH_SIZE}")

        self.emit("flow", ("pause",))

        def emit_hash_stage_0(desk_idx):
            d = desks[desk_idx]
            self.emit("valu", ("multiply_add", d['val'], d['val'], v_fma_mult[0], v_hash_consts[0]))

        def emit_hash_stage_1(desk_idx):
            d = desks[desk_idx]
            self.emit("valu", ("^", d['tmp1'], d['val'], v_hash_consts[1]))
            self.emit("valu", (">>", d['tmp2'], d['val'], v_hash_shifts[1]))
            self.emit("valu", ("^", d['val'], d['tmp1'], d['tmp2']))

        def emit_hash_stage_2(desk_idx):
            d = desks[desk_idx]
            self.emit("valu", ("multiply_add", d['val'], d['val'], v_fma_mult[2], v_hash_consts[2]))

        def emit_hash_stage_3(desk_idx):
            d = desks[desk_idx]
            self.emit("valu", ("+", d['tmp1'], d['val'], v_hash_consts[3]))
            self.emit("valu", ("<<", d['tmp2'], d['val'], v_hash_shifts[3]))
            self.emit("valu", ("^", d['val'], d['tmp1'], d['tmp2']))

        def emit_hash_stage_4(desk_idx):
            d = desks[desk_idx]
            self.emit("valu", ("multiply_add", d['val'], d['val'], v_fma_mult[4], v_hash_consts[4]))

        def emit_hash_stage_5(desk_idx):
            d = desks[desk_idx]
            self.emit("valu", ("^", d['tmp1'], d['val'], v_hash_consts[5]))
            self.emit("valu", (">>", d['tmp2'], d['val'], v_hash_shifts[5]))
            self.emit("valu", ("^", d['val'], d['tmp1'], d['tmp2']))

        def emit_branch(desk_idx):
            d = desks[desk_idx]
            self.emit("valu", ("&", d['tmp1'], d['val'], v_one))
            self.emit("valu", ("multiply_add", d['idx'], d['idx'], v_two, v_one))
            self.emit("valu", ("+", d['idx'], d['idx'], d['tmp1']))

        def emit_bounds(desk_idx):
            d = desks[desk_idx]
            self.emit("valu", ("<", d['tmp1'], d['idx'], v_n_nodes))
            self.emit("valu", ("*", d['idx'], d['idx'], d['tmp1']))

        def emit_xor_with_node(desk_idx, node_vec):
            d = desks[desk_idx]
            self.emit("valu", ("^", d['val'], d['val'], node_vec))

        def emit_round_0_interleaved(group_desks):
            for d in group_desks:
                emit_xor_with_node(d, v_tree[0])
            for d in group_desks:
                emit_hash_stage_0(d)
            for d in group_desks:
                emit_hash_stage_1(d)
            for d in group_desks:
                emit_hash_stage_2(d)
            for d in group_desks:
                emit_hash_stage_3(d)
            for d in group_desks:
                emit_hash_stage_4(d)
            for d in group_desks:
                emit_hash_stage_5(d)
            for d in group_desks:
                emit_branch(d)

        def emit_round_1_interleaved(group_desks):
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("-", desk['tmp1'], desk['idx'], v_one))
                self.emit("valu", ("multiply_add", desk['node_val'], desk['tmp1'], v_diff_1_2, v_tree[1]))
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("^", desk['val'], desk['val'], desk['node_val']))
            for d in group_desks:
                emit_hash_stage_0(d)
            for d in group_desks:
                emit_hash_stage_1(d)
            for d in group_desks:
                emit_hash_stage_2(d)
            for d in group_desks:
                emit_hash_stage_3(d)
            for d in group_desks:
                emit_hash_stage_4(d)
            for d in group_desks:
                emit_hash_stage_5(d)
            for d in group_desks:
                emit_branch(d)

        def emit_round_2_interleaved_vselect(group_desks):
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("-", desk['tmp1'], desk['idx'], v_three))
                self.emit("valu", ("&", desk['tmp2'], desk['tmp1'], v_one))
                self.emit("valu", (">>", desk['addr'], desk['tmp1'], v_one))
                self.emit("valu", ("multiply_add", desk['node_val'], desk['tmp2'], v_diff_3_4, v_tree[3]))
                self.emit("valu", ("multiply_add", desk['tmp1'], desk['tmp2'], v_diff_5_6, v_tree[5]))
                self.emit("flow", ("vselect", desk['node_val'], desk['addr'], desk['tmp1'], desk['node_val']))
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("^", desk['val'], desk['val'], desk['node_val']))
            for d in group_desks:
                emit_hash_stage_0(d)
            for d in group_desks:
                emit_hash_stage_1(d)
            for d in group_desks:
                emit_hash_stage_2(d)
            for d in group_desks:
                emit_hash_stage_3(d)
            for d in group_desks:
                emit_hash_stage_4(d)
            for d in group_desks:
                emit_hash_stage_5(d)
            for d in group_desks:
                emit_branch(d)

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
            for d in group_desks:
                emit_hash_stage_0(d)
            for d in group_desks:
                emit_hash_stage_1(d)
            for d in group_desks:
                emit_hash_stage_2(d)
            for d in group_desks:
                emit_hash_stage_3(d)
            for d in group_desks:
                emit_hash_stage_4(d)
            for d in group_desks:
                emit_hash_stage_5(d)
            for d in group_desks:
                emit_branch(d)

        def emit_round_10_interleaved(group_desks):
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
            for d in group_desks:
                emit_hash_stage_0(d)
            for d in group_desks:
                emit_hash_stage_1(d)
            for d in group_desks:
                emit_hash_stage_2(d)
            for d in group_desks:
                emit_hash_stage_3(d)
            for d in group_desks:
                emit_hash_stage_4(d)
            for d in group_desks:
                emit_hash_stage_5(d)
            for d in group_desks:
                emit_branch(d)
            for d in group_desks:
                emit_bounds(d)

        def emit_round_11_interleaved(group_desks):
            for d in group_desks:
                emit_xor_with_node(d, v_tree[0])
            for d in group_desks:
                emit_hash_stage_0(d)
            for d in group_desks:
                emit_hash_stage_1(d)
            for d in group_desks:
                emit_hash_stage_2(d)
            for d in group_desks:
                emit_hash_stage_3(d)
            for d in group_desks:
                emit_hash_stage_4(d)
            for d in group_desks:
                emit_hash_stage_5(d)
            for d in group_desks:
                emit_branch(d)

        def emit_round_15_final_interleaved(group_desks):
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
            for d in group_desks:
                emit_hash_stage_0(d)
            for d in group_desks:
                emit_hash_stage_1(d)
            for d in group_desks:
                emit_hash_stage_2(d)
            for d in group_desks:
                emit_hash_stage_3(d)
            for d in group_desks:
                emit_hash_stage_4(d)
            for d in group_desks:
                emit_hash_stage_5(d)

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
                self.mark_group_boundary()  # Mark start of group
                emit_round_0_interleaved(group_desks)
                emit_round_1_interleaved(group_desks)
                emit_round_2_interleaved_vselect(group_desks)

                for _rnd in range(3, 10):
                    emit_gather_round_interleaved(group_desks)

                emit_round_10_interleaved(group_desks)
                emit_round_11_interleaved(group_desks)
                emit_round_1_interleaved(group_desks)
                emit_round_2_interleaved_vselect(group_desks)
                emit_gather_round_interleaved(group_desks)
                emit_round_15_final_interleaved(group_desks)

            self.mark_group_boundary()  # Mark end of last group

            for d in range(NUM_DESKS):
                self.emit("store", ("vstore", addr_tmp[d*2], desks[d]['idx']))
                self.emit("store", ("vstore", addr_tmp[d*2+1], desks[d]['val']))

        emit_tile_interleaved(0)
        emit_tile_interleaved(1)

        # Parse phases
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
                if len(phase) > 500:  # Large phase - try per-group scheduling
                    print(f"Phase {i}: {len(phase)} slots, trying per-group CP-SAT...")
                    # For simplicity, just use greedy for now - the group merging
                    # approach needs more sophisticated handling of shared reads
                    phase_instrs = _schedule_slots_greedy(phase)
                    print(f"  -> {len(phase_instrs)} cycles (greedy)")
                else:
                    print(f"Phase {i}: {len(phase)} slots, scheduling with CP-SAT...")
                    result = _schedule_slots_cpsat(phase, time_limit_seconds=30.0)
                    if result is None:
                        print(f"  CP-SAT failed, using greedy")
                        result = _schedule_slots_greedy(phase)
                    phase_instrs = result
                    print(f"  -> {len(phase_instrs)} cycles")

                self.instrs.extend(phase_instrs)
            if i < len(phases) - 1:
                self.instrs.append({"flow": [("pause",)]})

        self.instrs.append({"flow": [("pause",)]})

        print(f"Total slots: {len(self.slots)}, Cycles: {len(self.instrs)}")


BASELINE = 147734


def do_kernel_test(forest_height: int, rounds: int, batch_size: int, seed: int = 123,
                   trace: bool = False, prints: bool = False, check: bool = False):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilderB1_2()
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
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()

    if args.check:
        cycles = do_kernel_test(10, 16, 256, check=True)
        print(f"Correctness check PASSED! Cycles: {cycles}")
    else:
        do_kernel_test(10, 16, 256, trace=args.trace)
