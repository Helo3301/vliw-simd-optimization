"""
# Experiment H115: Load Slot Optimization

**GOAL:** Verify and optimize load slot utilization in gather operations.

**HYPOTHESIS:** We have 2 load slots per cycle. In gather operations, we do 256 individual
loads (32 desks * 8 lanes). That's 128 cycles minimum just for loads. Let's verify we're
utilizing both slots fully.

**APPROACH:**
1. Add analysis to count load ops per cycle
2. Reorder gather emissions to ensure 2 loads per cycle are always used
3. Consider if vload can replace some scalar loads for idx/val loading

**BASELINE:** H105 = 1,843 cycles
**TARGET:** 1,790 cycles (82.5x speedup)
"""

import random
import unittest
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
    """Get read and write addresses for a slot."""
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


def _schedule_slots(slots: list[tuple[str, tuple]], analyze_loads: bool = False) -> list[dict[str, list[tuple]]]:
    """Automatically schedule operations into VLIW bundles respecting dependencies."""
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

    result = [c for c in cycles if c]

    # Analyze load utilization if requested
    if analyze_loads:
        load_stats = analyze_load_utilization(result)
        return result, load_stats

    return result


def analyze_load_utilization(cycles: list[dict[str, list[tuple]]]) -> dict:
    """Analyze load slot utilization per cycle."""
    stats = {
        'total_cycles': len(cycles),
        'cycles_with_loads': 0,
        'cycles_with_1_load': 0,
        'cycles_with_2_loads': 0,
        'total_load_ops': 0,
        'total_scalar_loads': 0,
        'total_vloads': 0,
        'total_consts': 0,
        'wasted_load_slots': 0,  # Cycles with 1 load that could have had 2
        'load_slot_utilization': 0.0,
    }

    for cycle_ops in cycles:
        load_ops = cycle_ops.get('load', [])
        num_loads = len(load_ops)

        if num_loads > 0:
            stats['cycles_with_loads'] += 1
            stats['total_load_ops'] += num_loads

            for op in load_ops:
                if op[0] == 'load':
                    stats['total_scalar_loads'] += 1
                elif op[0] == 'vload':
                    stats['total_vloads'] += 1
                elif op[0] == 'const':
                    stats['total_consts'] += 1

            if num_loads == 1:
                stats['cycles_with_1_load'] += 1
                stats['wasted_load_slots'] += 1
            elif num_loads >= 2:
                stats['cycles_with_2_loads'] += 1

    # Calculate utilization (2 slots available per cycle with loads)
    if stats['cycles_with_loads'] > 0:
        max_possible_loads = stats['cycles_with_loads'] * 2
        stats['load_slot_utilization'] = stats['total_load_ops'] / max_possible_loads * 100

    return stats


def analyze_all_units(cycles: list[dict[str, list[tuple]]]) -> dict:
    """Analyze utilization of ALL execution units per cycle."""
    stats = {
        'total_cycles': len(cycles),
        'alu': {'ops': 0, 'cycles_used': 0, 'max_per_cycle': 0},
        'valu': {'ops': 0, 'cycles_used': 0, 'max_per_cycle': 0},
        'load': {'ops': 0, 'cycles_used': 0, 'max_per_cycle': 0},
        'store': {'ops': 0, 'cycles_used': 0, 'max_per_cycle': 0},
        'flow': {'ops': 0, 'cycles_used': 0, 'max_per_cycle': 0},
        'empty_cycles': 0,
        'bottleneck_unit': None,
        'bottleneck_cycles': 0,
    }

    unit_usage_distribution = {unit: defaultdict(int) for unit in ['alu', 'valu', 'load', 'store', 'flow']}

    for cycle_ops in cycles:
        cycle_has_ops = False
        for unit in ['alu', 'valu', 'load', 'store', 'flow']:
            ops = cycle_ops.get(unit, [])
            num_ops = len(ops)
            if num_ops > 0:
                cycle_has_ops = True
                stats[unit]['ops'] += num_ops
                stats[unit]['cycles_used'] += 1
                stats[unit]['max_per_cycle'] = max(stats[unit]['max_per_cycle'], num_ops)
                unit_usage_distribution[unit][num_ops] += 1

        if not cycle_has_ops:
            stats['empty_cycles'] += 1

    # Determine bottleneck (unit that's most often the limiting factor)
    for unit in ['alu', 'valu', 'load', 'store']:
        limit = SLOT_LIMITS[unit]
        at_limit_cycles = sum(count for usage, count in unit_usage_distribution[unit].items() if usage >= limit)
        if at_limit_cycles > stats['bottleneck_cycles']:
            stats['bottleneck_cycles'] = at_limit_cycles
            stats['bottleneck_unit'] = unit

    stats['unit_usage_distribution'] = unit_usage_distribution
    return stats


def print_all_unit_stats(stats: dict):
    """Print utilization statistics for all execution units."""
    print("\n=== EXECUTION UNIT UTILIZATION ANALYSIS ===")
    print(f"Total cycles: {stats['total_cycles']}")
    print(f"Empty cycles: {stats['empty_cycles']}")
    print()

    for unit in ['alu', 'valu', 'load', 'store', 'flow']:
        unit_stats = stats[unit]
        limit = SLOT_LIMITS.get(unit, 'N/A')
        print(f"{unit.upper():6s}: {unit_stats['ops']:5d} ops, {unit_stats['cycles_used']:5d} cycles used, "
              f"max {unit_stats['max_per_cycle']}/cycle (limit: {limit})")

        # Show distribution
        dist = stats['unit_usage_distribution'][unit]
        if dist:
            dist_str = ", ".join(f"{usage}x:{count}" for usage, count in sorted(dist.items()))
            print(f"         Distribution: {dist_str}")

    print()
    if stats['bottleneck_unit']:
        print(f"BOTTLENECK: {stats['bottleneck_unit'].upper()} ({stats['bottleneck_cycles']} cycles at limit)")
    print("=" * 45)


def print_load_stats(stats: dict):
    """Print load utilization statistics."""
    print("\n=== LOAD SLOT UTILIZATION ANALYSIS ===")
    print(f"Total cycles: {stats['total_cycles']}")
    print(f"Cycles with load ops: {stats['cycles_with_loads']}")
    print(f"  - With 1 load: {stats['cycles_with_1_load']} (wasted slot potential)")
    print(f"  - With 2 loads: {stats['cycles_with_2_loads']} (fully utilized)")
    print(f"Total load operations: {stats['total_load_ops']}")
    print(f"  - Scalar loads: {stats['total_scalar_loads']}")
    print(f"  - Vector loads: {stats['total_vloads']}")
    print(f"  - Constants: {stats['total_consts']}")
    print(f"Wasted load slots: {stats['wasted_load_slots']}")
    print(f"Load slot utilization: {stats['load_slot_utilization']:.1f}%")
    print("=" * 40)


class KernelBuilderH115:
    """
    H115: Load slot optimization - maximize 2 loads per cycle in gather operations.

    Key changes from H105:
    1. Interleave gather loads from different desks to maximize slot usage
    2. Ensure addr computation completes early enough for load pairing
    3. Add load utilization analysis
    """
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
        """Add operation to flat list for later scheduling."""
        self.slots.append((engine, slot))

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, f"Out of scratch space: {self.scratch_ptr}"
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

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Build kernel using 32 desks per tile with optimized load slot utilization.
        """
        tmp_scalar = self.alloc_scratch("tmp_scalar")
        tmp_addr = self.alloc_scratch("tmp_addr")

        # Load header values
        init_vars = ["rounds", "n_nodes", "batch_size", "forest_height",
                     "forest_values_p", "inp_indices_p", "inp_values_p"]
        for v in init_vars:
            self.alloc_scratch(v)
        for i, v in enumerate(init_vars):
            self.emit("load", ("const", tmp_scalar, i))
            self.emit("load", ("load", self.scratch[v], tmp_scalar))

        # Vector constants
        v_zero = self.scratch_vconst(0, "v_zero")
        v_one = self.scratch_vconst(1, "v_one")
        v_two = self.scratch_vconst(2, "v_two")
        v_three = self.scratch_vconst(3, "v_three")  # For 4-way selection
        v_n_nodes = self.alloc_vec("v_n_nodes")
        self.emit("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]))

        # Precompute forest_values_p as vector (eliminates per-gather vbroadcast)
        v_forest_p = self.alloc_vec("v_forest_p")
        self.emit("valu", ("vbroadcast", v_forest_p, self.scratch["forest_values_p"]))

        # Hash constants (FMA for stages 0, 2, 4)
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

        # Preload only tree nodes 0-6
        NUM_PRELOADED = 7
        v_tree = []
        for i in range(NUM_PRELOADED):
            v_node = self.alloc_vec(f"v_tree_{i}")
            v_tree.append(v_node)
            self.emit("alu", ("+", tmp_addr, self.scratch["forest_values_p"], self.scratch_const(i)))
            self.emit("load", ("load", tmp_scalar, tmp_addr))
            self.emit("valu", ("vbroadcast", v_node, tmp_scalar))

        # Precompute tree differences for selection
        v_diff_1_2 = self.alloc_vec("v_diff_1_2")
        v_diff_3_4 = self.alloc_vec("v_diff_3_4")
        v_diff_5_6 = self.alloc_vec("v_diff_5_6")
        self.emit("valu", ("-", v_diff_1_2, v_tree[2], v_tree[1]))
        self.emit("valu", ("-", v_diff_3_4, v_tree[4], v_tree[3]))
        self.emit("valu", ("-", v_diff_5_6, v_tree[6], v_tree[5]))

        # Allocate per-desk vectors (32 desks)
        NUM_DESKS = 32
        desks = []
        for d in range(NUM_DESKS):
            desk = {
                'idx': self.alloc_vec(f"v_idx_{d}"),
                'val': self.alloc_vec(f"v_val_{d}"),
                'node_val': self.alloc_vec(f"v_node_{d}"),
                'addr': self.alloc_vec(f"v_addr_{d}"),
            }
            desks.append(desk)

        # SHARED TEMP REGISTERS
        NUM_SHARED_TEMPS = 16
        shared_temps = [self.alloc_vec(f"v_shared_tmp_{i}") for i in range(NUM_SHARED_TEMPS)]

        def get_temps(desk_idx):
            group = desk_idx // 4
            t1_idx = group * 2
            t2_idx = group * 2 + 1
            return shared_temps[t1_idx], shared_temps[t2_idx]

        # Offset addresses
        offset_regs = [self.alloc_scratch(f"off_{d}") for d in range(NUM_DESKS)]
        addr_tmp = [self.alloc_scratch(f"addr_tmp_{i}") for i in range(4)]

        scratch_before = self.scratch_ptr
        print(f"Scratch usage: {self.scratch_ptr} / {SCRATCH_SIZE}")
        print(f"  - H115: Load slot optimization experiment")

        # Pause before main computation
        self.emit("flow", ("pause",))

        # Desk groupings for temp conflict avoidance
        group_0 = list(range(0, NUM_DESKS, 4))
        group_1 = list(range(1, NUM_DESKS, 4))
        group_2 = list(range(2, NUM_DESKS, 4))
        group_3 = list(range(3, NUM_DESKS, 4))

        def emit_hash_stages_all_desks():
            """Emit hash stages for all 32 desks in a double-pumped fashion."""
            # Stage 0 (FMA)
            for d in range(NUM_DESKS):
                self.emit("valu", ("multiply_add", desks[d]['val'], desks[d]['val'], v_fma_mult[0], v_hash_consts[0]))

            # Stage 1 (3 ops)
            for group in [group_0, group_1, group_2, group_3]:
                for d in group:
                    tmp1, tmp2 = get_temps(d)
                    self.emit("valu", ("^", tmp1, desks[d]['val'], v_hash_consts[1]))
                for d in group:
                    tmp1, tmp2 = get_temps(d)
                    self.emit("valu", (">>", tmp2, desks[d]['val'], v_hash_shifts[1]))
                for d in group:
                    tmp1, tmp2 = get_temps(d)
                    self.emit("valu", ("^", desks[d]['val'], tmp1, tmp2))

            # Stage 2 (FMA)
            for d in range(NUM_DESKS):
                self.emit("valu", ("multiply_add", desks[d]['val'], desks[d]['val'], v_fma_mult[2], v_hash_consts[2]))

            # Stage 3 (3 ops)
            for group in [group_0, group_1, group_2, group_3]:
                for d in group:
                    tmp1, tmp2 = get_temps(d)
                    self.emit("valu", ("+", tmp1, desks[d]['val'], v_hash_consts[3]))
                for d in group:
                    tmp1, tmp2 = get_temps(d)
                    self.emit("valu", ("<<", tmp2, desks[d]['val'], v_hash_shifts[3]))
                for d in group:
                    tmp1, tmp2 = get_temps(d)
                    self.emit("valu", ("^", desks[d]['val'], tmp1, tmp2))

            # Stage 4 (FMA)
            for d in range(NUM_DESKS):
                self.emit("valu", ("multiply_add", desks[d]['val'], desks[d]['val'], v_fma_mult[4], v_hash_consts[4]))

            # Stage 5 (3 ops)
            for group in [group_0, group_1, group_2, group_3]:
                for d in group:
                    tmp1, tmp2 = get_temps(d)
                    self.emit("valu", ("^", tmp1, desks[d]['val'], v_hash_consts[5]))
                for d in group:
                    tmp1, tmp2 = get_temps(d)
                    self.emit("valu", (">>", tmp2, desks[d]['val'], v_hash_shifts[5]))
                for d in group:
                    tmp1, tmp2 = get_temps(d)
                    self.emit("valu", ("^", desks[d]['val'], tmp1, tmp2))

        def emit_hash_stages(desk_idx):
            """Emit hash stages for a single desk."""
            d = desks[desk_idx]
            tmp1, tmp2 = get_temps(desk_idx)
            for hi in range(6):
                if hi in v_fma_mult:
                    self.emit("valu", ("multiply_add", d['val'], d['val'], v_fma_mult[hi], v_hash_consts[hi]))
                elif hi == 1:
                    self.emit("valu", ("^", tmp1, d['val'], v_hash_consts[hi]))
                    self.emit("valu", (">>", tmp2, d['val'], v_hash_shifts[hi]))
                    self.emit("valu", ("^", d['val'], tmp1, tmp2))
                elif hi == 3:
                    self.emit("valu", ("+", tmp1, d['val'], v_hash_consts[hi]))
                    self.emit("valu", ("<<", tmp2, d['val'], v_hash_shifts[hi]))
                    self.emit("valu", ("^", d['val'], tmp1, tmp2))
                elif hi == 5:
                    self.emit("valu", ("^", tmp1, d['val'], v_hash_consts[hi]))
                    self.emit("valu", (">>", tmp2, d['val'], v_hash_shifts[hi]))
                    self.emit("valu", ("^", d['val'], tmp1, tmp2))

        def emit_branch(desk_idx):
            d = desks[desk_idx]
            tmp1, _ = get_temps(desk_idx)
            self.emit("valu", ("&", tmp1, d['val'], v_one))
            self.emit("valu", ("multiply_add", d['idx'], d['idx'], v_two, v_one))
            self.emit("valu", ("+", d['idx'], d['idx'], tmp1))

        def emit_bounds(desk_idx):
            d = desks[desk_idx]
            tmp1, _ = get_temps(desk_idx)
            self.emit("valu", ("<", tmp1, d['idx'], v_n_nodes))
            self.emit("valu", ("*", d['idx'], d['idx'], tmp1))

        def emit_gather_optimized(round_num: int, with_bounds: bool = False):
            """
            H115 OPTIMIZATION: Emit gather operations with optimized load pairing.

            Key insight: Each gather needs 256 loads (32 desks * 8 lanes).
            With 2 load slots, minimum is 128 cycles just for loads.

            Strategy: Use the original sequential emit order - the scheduler
            already achieves 100% load utilization. The interleaved approach
            actually created worse dependency chains.
            """
            # Compute addresses for ALL desks first
            for d in range(NUM_DESKS):
                self.emit("valu", ("+", desks[d]['addr'], v_forest_p, desks[d]['idx']))

            # Original sequential order - scheduler handles pairing
            for d in range(NUM_DESKS):
                for lane in range(VLEN):
                    self.emit("load", ("load", desks[d]['node_val'] + lane, desks[d]['addr'] + lane))

            # XOR for all desks
            for d in range(NUM_DESKS):
                self.emit("valu", ("^", desks[d]['val'], desks[d]['val'], desks[d]['node_val']))

            # Double-pumped hash stages
            emit_hash_stages_all_desks()

            # Branch for all desks
            for d in range(NUM_DESKS):
                emit_branch(d)

            # Bounds check if needed
            if with_bounds:
                for d in range(NUM_DESKS):
                    emit_bounds(d)

        def emit_tile():
            """Emit all operations for the single tile (all 256 batch elements)."""
            # Load offsets for all 32 desks
            for d in range(NUM_DESKS):
                self.emit("load", ("const", offset_regs[d], d * VLEN))

            # Compute load addresses and load idx/val for all desks
            for d_batch in range(0, NUM_DESKS, 2):
                d0, d1 = d_batch, d_batch + 1
                self.emit("alu", ("+", addr_tmp[0], self.scratch["inp_indices_p"], offset_regs[d0]))
                self.emit("alu", ("+", addr_tmp[1], self.scratch["inp_values_p"], offset_regs[d0]))
                self.emit("alu", ("+", addr_tmp[2], self.scratch["inp_indices_p"], offset_regs[d1]))
                self.emit("alu", ("+", addr_tmp[3], self.scratch["inp_values_p"], offset_regs[d1]))
                self.emit("load", ("vload", desks[d0]['idx'], addr_tmp[0]))
                self.emit("load", ("vload", desks[d0]['val'], addr_tmp[1]))
                self.emit("load", ("vload", desks[d1]['idx'], addr_tmp[2]))
                self.emit("load", ("vload", desks[d1]['val'], addr_tmp[3]))

            # ===== ROUNDS =====
            # Round 0: All indices = 0, use tree[0] directly
            for d in range(NUM_DESKS):
                self.emit("valu", ("^", desks[d]['val'], desks[d]['val'], v_tree[0]))
            emit_hash_stages_all_desks()
            for d in range(NUM_DESKS):
                emit_branch(d)

            # Round 1: Indices in {1, 2}, use arithmetic selection
            for group in [group_0, group_1, group_2, group_3]:
                for d in group:
                    tmp1, _ = get_temps(d)
                    self.emit("valu", ("-", tmp1, desks[d]['idx'], v_one))
                for d in group:
                    tmp1, _ = get_temps(d)
                    self.emit("valu", ("multiply_add", desks[d]['node_val'], tmp1, v_diff_1_2, v_tree[1]))
            for d in range(NUM_DESKS):
                self.emit("valu", ("^", desks[d]['val'], desks[d]['val'], desks[d]['node_val']))
            emit_hash_stages_all_desks()
            for d in range(NUM_DESKS):
                emit_branch(d)

            # Round 2: Indices in {3, 4, 5, 6}, use 4-way arithmetic selection
            for d in range(NUM_DESKS):
                tmp1, tmp2 = get_temps(d)
                self.emit("valu", ("-", tmp1, desks[d]['idx'], v_three))
                self.emit("valu", ("&", tmp2, tmp1, v_one))
                self.emit("valu", (">>", desks[d]['addr'], tmp1, v_one))
                self.emit("valu", ("multiply_add", desks[d]['node_val'], tmp2, v_diff_3_4, v_tree[3]))
                self.emit("valu", ("multiply_add", tmp1, tmp2, v_diff_5_6, v_tree[5]))
                self.emit("valu", ("-", tmp2, tmp1, desks[d]['node_val']))
                self.emit("valu", ("multiply_add", desks[d]['node_val'], desks[d]['addr'], tmp2, desks[d]['node_val']))
                self.emit("valu", ("^", desks[d]['val'], desks[d]['val'], desks[d]['node_val']))
                emit_hash_stages(d)
                emit_branch(d)

            # Rounds 3-9: Gather without bounds (using optimized gather)
            for _round in range(3, 10):
                emit_gather_optimized(_round, with_bounds=False)

            # Round 10: Gather WITH bounds
            emit_gather_optimized(10, with_bounds=True)

            # Round 11: All indices = 0 after wrap, use tree[0] directly
            for d in range(NUM_DESKS):
                self.emit("valu", ("^", desks[d]['val'], desks[d]['val'], v_tree[0]))
            emit_hash_stages_all_desks()
            for d in range(NUM_DESKS):
                emit_branch(d)

            # Round 12: Indices in {1, 2}, use arithmetic selection
            for group in [group_0, group_1, group_2, group_3]:
                for d in group:
                    tmp1, _ = get_temps(d)
                    self.emit("valu", ("-", tmp1, desks[d]['idx'], v_one))
                for d in group:
                    tmp1, _ = get_temps(d)
                    self.emit("valu", ("multiply_add", desks[d]['node_val'], tmp1, v_diff_1_2, v_tree[1]))
            for d in range(NUM_DESKS):
                self.emit("valu", ("^", desks[d]['val'], desks[d]['val'], desks[d]['node_val']))
            emit_hash_stages_all_desks()
            for d in range(NUM_DESKS):
                emit_branch(d)

            # Round 13: Indices in {3, 4, 5, 6}, use 4-way arithmetic selection
            for d in range(NUM_DESKS):
                tmp1, tmp2 = get_temps(d)
                self.emit("valu", ("-", tmp1, desks[d]['idx'], v_three))
                self.emit("valu", ("&", tmp2, tmp1, v_one))
                self.emit("valu", (">>", desks[d]['addr'], tmp1, v_one))
                self.emit("valu", ("multiply_add", desks[d]['node_val'], tmp2, v_diff_3_4, v_tree[3]))
                self.emit("valu", ("multiply_add", tmp1, tmp2, v_diff_5_6, v_tree[5]))
                self.emit("valu", ("-", tmp2, tmp1, desks[d]['node_val']))
                self.emit("valu", ("multiply_add", desks[d]['node_val'], desks[d]['addr'], tmp2, desks[d]['node_val']))
                self.emit("valu", ("^", desks[d]['val'], desks[d]['val'], desks[d]['node_val']))
                emit_hash_stages(d)
                emit_branch(d)

            # Rounds 14-15: Gather without bounds (using optimized gather)
            for _round in range(14, 16):
                emit_gather_optimized(_round, with_bounds=False)

            # Store results
            for d_batch in range(0, NUM_DESKS, 2):
                d0, d1 = d_batch, d_batch + 1
                self.emit("alu", ("+", addr_tmp[0], self.scratch["inp_indices_p"], offset_regs[d0]))
                self.emit("alu", ("+", addr_tmp[1], self.scratch["inp_values_p"], offset_regs[d0]))
                self.emit("alu", ("+", addr_tmp[2], self.scratch["inp_indices_p"], offset_regs[d1]))
                self.emit("alu", ("+", addr_tmp[3], self.scratch["inp_values_p"], offset_regs[d1]))
                self.emit("store", ("vstore", addr_tmp[0], desks[d0]['idx']))
                self.emit("store", ("vstore", addr_tmp[1], desks[d0]['val']))
                self.emit("store", ("vstore", addr_tmp[2], desks[d1]['idx']))
                self.emit("store", ("vstore", addr_tmp[3], desks[d1]['val']))

        # Emit the single tile
        emit_tile()

        # Schedule operations in phases separated by pauses
        phases = []
        current_phase = []
        for engine, slot in self.slots:
            if engine == "flow" and slot == ("pause",):
                phases.append(current_phase)
                current_phase = []
            else:
                current_phase.append((engine, slot))
        phases.append(current_phase)

        # Schedule each phase with load analysis
        self.instrs = []
        total_load_stats = None

        for i, phase in enumerate(phases):
            if phase:
                result = _schedule_slots(phase, analyze_loads=True)
                phase_instrs, load_stats = result
                self.instrs.extend(phase_instrs)

                # Accumulate load stats (focus on main computation phase)
                if i == 1:  # Main computation phase (after first pause)
                    total_load_stats = load_stats
            if i < len(phases) - 1:
                self.instrs.append({"flow": [("pause",)]})

        # Add final pause
        self.instrs.append({"flow": [("pause",)]})

        print(f"Total slots: {len(self.slots)}, Cycles: {len(self.instrs)}")

        # Print load utilization stats
        if total_load_stats:
            print_load_stats(total_load_stats)

        # Comprehensive unit analysis on main phase
        all_unit_stats = analyze_all_units(self.instrs[:-1])  # Exclude final pause
        print_all_unit_stats(all_unit_stats)


BASELINE = 147734


def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
    check: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilderH115()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if check:
            assert (
                machine.mem[inp_values_p : inp_values_p + len(inp.values)]
                == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
            ), f"Incorrect result on round {i}"

    print("\n" + "=" * 50)
    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    print("=" * 50)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


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
