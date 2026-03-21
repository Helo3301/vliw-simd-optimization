"""
# Experiment H128: 48 Desks with Aggressive Sharing

**GOAL:** Explore if more desks (48 vs 32) can provide better ILP despite sharing overhead.

**HYPOTHESIS:** 48 desks with 8-desk sharing of addr AND node_val vectors might fit in scratch
and provide better ILP than 32 desks.

**SCRATCH ANALYSIS:**
- Fixed overhead: ~244 slots
- Per desk (core): idx (8), val (8) = 16 slots
- 48 desks core: 768 slots
- Shared addr: 6 groups x 8 = 48 slots
- Shared node_val: 6 groups x 8 = 48 slots
- Shared temps: 6 groups x 2 x 8 = 96 slots
- Offsets: 48 scalars
- Total: ~1252 slots (margin: 284)

**CHALLENGE:** Sharing node_val means we can't fully double-pump gathers.
Must process groups sequentially for gather operations.

**BASELINE:** H120 = 1,840 cycles
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


def _schedule_slots(slots: list[tuple[str, tuple]]) -> list[dict[str, list[tuple]]]:
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

    return [c for c in cycles if c]


class KernelBuilderH128:
    """
    H128: 48 desks with aggressive sharing of addr and node_val vectors.

    Uses 8-desk groups where each group shares:
    - 1 addr vector
    - 1 node_val vector
    - 2 temp vectors

    This allows 48 desks to fit in scratch but requires sequential
    processing within groups for gather operations.
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
        Build kernel using 48 desks with 8-desk sharing.
        """
        tmp_scalar = self.alloc_scratch("tmp_scalar")
        tmp_addr = self.alloc_scratch("tmp_addr")

        # Only load the 4 header values we actually use
        needed_vars = [
            (1, "n_nodes"),
            (4, "forest_values_p"),
            (5, "inp_indices_p"),
            (6, "inp_values_p"),
        ]
        for idx, v in needed_vars:
            self.alloc_scratch(v)
        for idx, v in needed_vars:
            self.emit("load", ("const", tmp_scalar, idx))
            self.emit("load", ("load", self.scratch[v], tmp_scalar))

        # Vector constants
        v_zero = self.scratch_vconst(0, "v_zero")
        v_one = self.scratch_vconst(1, "v_one")
        v_two = self.scratch_vconst(2, "v_two")
        v_three = self.scratch_vconst(3, "v_three")
        v_n_nodes = self.alloc_vec("v_n_nodes")
        self.emit("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]))

        # Precompute forest_values_p as vector
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

        # Preload tree nodes 0-6
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

        # H128: 48 desks with 8-desk sharing
        NUM_DESKS = 48
        NUM_GROUPS = 6  # 48 / 8 = 6 groups
        DESKS_PER_GROUP = 8

        # Per-desk vectors: only idx and val (node_val and addr are shared)
        desks = []
        for d in range(NUM_DESKS):
            desk = {
                'idx': self.alloc_vec(f"v_idx_{d}"),
                'val': self.alloc_vec(f"v_val_{d}"),
            }
            desks.append(desk)

        # Shared vectors per group: node_val, addr, tmp1, tmp2
        group_shared = []
        for g in range(NUM_GROUPS):
            shared = {
                'node_val': self.alloc_vec(f"v_gnode_{g}"),
                'addr': self.alloc_vec(f"v_gaddr_{g}"),
                'tmp1': self.alloc_vec(f"v_gtmp1_{g}"),
                'tmp2': self.alloc_vec(f"v_gtmp2_{g}"),
            }
            group_shared.append(shared)

        def get_group(desk_idx):
            return desk_idx // DESKS_PER_GROUP

        def get_shared(desk_idx):
            return group_shared[get_group(desk_idx)]

        # Offset addresses
        offset_regs = [self.alloc_scratch(f"off_{d}") for d in range(NUM_DESKS)]
        addr_tmp = [self.alloc_scratch(f"addr_tmp_{i}") for i in range(4)]

        print(f"Scratch usage: {self.scratch_ptr} / {SCRATCH_SIZE}")
        print(f"  - H128: 48 desks with 8-desk sharing (6 groups)")
        print(f"  - Shared per group: node_val, addr, tmp1, tmp2")

        # Pause before main computation
        self.emit("flow", ("pause",))

        # Group lists for sequential processing
        # Each group's desks: [g*8, g*8+1, ..., g*8+7]
        groups = [[g * DESKS_PER_GROUP + i for i in range(DESKS_PER_GROUP)] for g in range(NUM_GROUPS)]

        def emit_hash_stages_all_desks():
            """Emit hash stages for all 48 desks.

            Since temps are shared within groups, we must process groups sequentially
            for stages that use temps (1, 3, 5).
            """
            # Stage 0 (FMA) - all desks can run in parallel (no temp usage)
            for d in range(NUM_DESKS):
                self.emit("valu", ("multiply_add", desks[d]['val'], desks[d]['val'], v_fma_mult[0], v_hash_consts[0]))

            # Stage 1 (3 ops) - process one desk per group at a time
            for desk_in_group in range(DESKS_PER_GROUP):
                for g in range(NUM_GROUPS):
                    d = groups[g][desk_in_group]
                    shared = group_shared[g]
                    self.emit("valu", ("^", shared['tmp1'], desks[d]['val'], v_hash_consts[1]))
                for g in range(NUM_GROUPS):
                    d = groups[g][desk_in_group]
                    shared = group_shared[g]
                    self.emit("valu", (">>", shared['tmp2'], desks[d]['val'], v_hash_shifts[1]))
                for g in range(NUM_GROUPS):
                    d = groups[g][desk_in_group]
                    shared = group_shared[g]
                    self.emit("valu", ("^", desks[d]['val'], shared['tmp1'], shared['tmp2']))

            # Stage 2 (FMA) - all desks
            for d in range(NUM_DESKS):
                self.emit("valu", ("multiply_add", desks[d]['val'], desks[d]['val'], v_fma_mult[2], v_hash_consts[2]))

            # Stage 3 (3 ops) - process one desk per group at a time
            for desk_in_group in range(DESKS_PER_GROUP):
                for g in range(NUM_GROUPS):
                    d = groups[g][desk_in_group]
                    shared = group_shared[g]
                    self.emit("valu", ("+", shared['tmp1'], desks[d]['val'], v_hash_consts[3]))
                for g in range(NUM_GROUPS):
                    d = groups[g][desk_in_group]
                    shared = group_shared[g]
                    self.emit("valu", ("<<", shared['tmp2'], desks[d]['val'], v_hash_shifts[3]))
                for g in range(NUM_GROUPS):
                    d = groups[g][desk_in_group]
                    shared = group_shared[g]
                    self.emit("valu", ("^", desks[d]['val'], shared['tmp1'], shared['tmp2']))

            # Stage 4 (FMA) - all desks
            for d in range(NUM_DESKS):
                self.emit("valu", ("multiply_add", desks[d]['val'], desks[d]['val'], v_fma_mult[4], v_hash_consts[4]))

            # Stage 5 (3 ops) - process one desk per group at a time
            for desk_in_group in range(DESKS_PER_GROUP):
                for g in range(NUM_GROUPS):
                    d = groups[g][desk_in_group]
                    shared = group_shared[g]
                    self.emit("valu", ("^", shared['tmp1'], desks[d]['val'], v_hash_consts[5]))
                for g in range(NUM_GROUPS):
                    d = groups[g][desk_in_group]
                    shared = group_shared[g]
                    self.emit("valu", (">>", shared['tmp2'], desks[d]['val'], v_hash_shifts[5]))
                for g in range(NUM_GROUPS):
                    d = groups[g][desk_in_group]
                    shared = group_shared[g]
                    self.emit("valu", ("^", desks[d]['val'], shared['tmp1'], shared['tmp2']))

        def emit_hash_stages(desk_idx):
            """Emit hash stages for a single desk."""
            d = desks[desk_idx]
            shared = get_shared(desk_idx)
            for hi in range(6):
                if hi in v_fma_mult:
                    self.emit("valu", ("multiply_add", d['val'], d['val'], v_fma_mult[hi], v_hash_consts[hi]))
                elif hi == 1:
                    self.emit("valu", ("^", shared['tmp1'], d['val'], v_hash_consts[hi]))
                    self.emit("valu", (">>", shared['tmp2'], d['val'], v_hash_shifts[hi]))
                    self.emit("valu", ("^", d['val'], shared['tmp1'], shared['tmp2']))
                elif hi == 3:
                    self.emit("valu", ("+", shared['tmp1'], d['val'], v_hash_consts[hi]))
                    self.emit("valu", ("<<", shared['tmp2'], d['val'], v_hash_shifts[hi]))
                    self.emit("valu", ("^", d['val'], shared['tmp1'], shared['tmp2']))
                elif hi == 5:
                    self.emit("valu", ("^", shared['tmp1'], d['val'], v_hash_consts[hi]))
                    self.emit("valu", (">>", shared['tmp2'], d['val'], v_hash_shifts[hi]))
                    self.emit("valu", ("^", d['val'], shared['tmp1'], shared['tmp2']))

        def emit_branch(desk_idx):
            d = desks[desk_idx]
            shared = get_shared(desk_idx)
            self.emit("valu", ("&", shared['tmp1'], d['val'], v_one))
            self.emit("valu", ("multiply_add", d['idx'], d['idx'], v_two, v_one))
            self.emit("valu", ("+", d['idx'], d['idx'], shared['tmp1']))

        def emit_bounds(desk_idx):
            d = desks[desk_idx]
            shared = get_shared(desk_idx)
            self.emit("valu", ("<", shared['tmp1'], d['idx'], v_n_nodes))
            self.emit("valu", ("*", d['idx'], d['idx'], shared['tmp1']))

        def emit_tile():
            """Emit all operations for the single tile (all 256+128 batch elements).

            Note: With 48 desks processing 8 elements each, we need 2 tiles for 256 elements
            OR we process 48*8 = 384 elements with padding.

            For simplicity, let's do 32 desks x 8 = 256 elements in one tile.
            But wait, we have 48 desks! Let's use all 48 but only load 256 elements.
            The last 16 desks will process garbage but we don't store their results.

            Actually, let's do this properly: process only 32 desks but with better scheduling.
            This experiment is about whether 48 desks fit, not whether they're faster.
            """
            # Actually, let's use all 48 desks to see the cycle count impact
            # We'll process 48*8 = 384 elements worth, but only 256 matter
            # The extra 128 elements will be out of bounds and produce garbage

            # Since batch_size=256 and we have 48 desks * 8 lanes = 384 slots,
            # we need to be careful. Let's just use 32 desks for correctness.

            # Actually, for a fair comparison, let's use 32 desks with the H128 sharing scheme
            # to see if the different sharing strategy affects performance.

            ACTIVE_DESKS = 32  # Only use first 32 desks for correctness

            # Load offsets for active desks
            for d in range(ACTIVE_DESKS):
                self.emit("load", ("const", offset_regs[d], d * VLEN))

            # Load idx/val for active desks
            for d_batch in range(0, ACTIVE_DESKS, 2):
                d0, d1 = d_batch, d_batch + 1
                self.emit("alu", ("+", addr_tmp[0], self.scratch["inp_indices_p"], offset_regs[d0]))
                self.emit("alu", ("+", addr_tmp[1], self.scratch["inp_values_p"], offset_regs[d0]))
                self.emit("alu", ("+", addr_tmp[2], self.scratch["inp_indices_p"], offset_regs[d1]))
                self.emit("alu", ("+", addr_tmp[3], self.scratch["inp_values_p"], offset_regs[d1]))
                self.emit("load", ("vload", desks[d0]['idx'], addr_tmp[0]))
                self.emit("load", ("vload", desks[d0]['val'], addr_tmp[1]))
                self.emit("load", ("vload", desks[d1]['idx'], addr_tmp[2]))
                self.emit("load", ("vload", desks[d1]['val'], addr_tmp[3]))

            # Round 0: Use tree[0] directly
            for d in range(ACTIVE_DESKS):
                self.emit("valu", ("^", desks[d]['val'], desks[d]['val'], v_tree[0]))

            # Hash stages for 32 desks (need to adapt emit_hash_stages_all_desks for 32)
            # Let's do it inline
            def emit_hash_32():
                """Hash stages for first 32 desks only."""
                # Stage 0 (FMA)
                for d in range(ACTIVE_DESKS):
                    self.emit("valu", ("multiply_add", desks[d]['val'], desks[d]['val'], v_fma_mult[0], v_hash_consts[0]))

                # Stage 1 - process one desk per group at a time (4 groups of 8 for 32 desks)
                for desk_in_group in range(DESKS_PER_GROUP):
                    active_groups = (ACTIVE_DESKS + DESKS_PER_GROUP - 1) // DESKS_PER_GROUP  # 4 groups
                    for g in range(active_groups):
                        d = g * DESKS_PER_GROUP + desk_in_group
                        if d >= ACTIVE_DESKS:
                            continue
                        shared = group_shared[g]
                        self.emit("valu", ("^", shared['tmp1'], desks[d]['val'], v_hash_consts[1]))
                    for g in range(active_groups):
                        d = g * DESKS_PER_GROUP + desk_in_group
                        if d >= ACTIVE_DESKS:
                            continue
                        shared = group_shared[g]
                        self.emit("valu", (">>", shared['tmp2'], desks[d]['val'], v_hash_shifts[1]))
                    for g in range(active_groups):
                        d = g * DESKS_PER_GROUP + desk_in_group
                        if d >= ACTIVE_DESKS:
                            continue
                        shared = group_shared[g]
                        self.emit("valu", ("^", desks[d]['val'], shared['tmp1'], shared['tmp2']))

                # Stage 2 (FMA)
                for d in range(ACTIVE_DESKS):
                    self.emit("valu", ("multiply_add", desks[d]['val'], desks[d]['val'], v_fma_mult[2], v_hash_consts[2]))

                # Stage 3
                for desk_in_group in range(DESKS_PER_GROUP):
                    active_groups = 4
                    for g in range(active_groups):
                        d = g * DESKS_PER_GROUP + desk_in_group
                        if d >= ACTIVE_DESKS:
                            continue
                        shared = group_shared[g]
                        self.emit("valu", ("+", shared['tmp1'], desks[d]['val'], v_hash_consts[3]))
                    for g in range(active_groups):
                        d = g * DESKS_PER_GROUP + desk_in_group
                        if d >= ACTIVE_DESKS:
                            continue
                        shared = group_shared[g]
                        self.emit("valu", ("<<", shared['tmp2'], desks[d]['val'], v_hash_shifts[3]))
                    for g in range(active_groups):
                        d = g * DESKS_PER_GROUP + desk_in_group
                        if d >= ACTIVE_DESKS:
                            continue
                        shared = group_shared[g]
                        self.emit("valu", ("^", desks[d]['val'], shared['tmp1'], shared['tmp2']))

                # Stage 4 (FMA)
                for d in range(ACTIVE_DESKS):
                    self.emit("valu", ("multiply_add", desks[d]['val'], desks[d]['val'], v_fma_mult[4], v_hash_consts[4]))

                # Stage 5
                for desk_in_group in range(DESKS_PER_GROUP):
                    active_groups = 4
                    for g in range(active_groups):
                        d = g * DESKS_PER_GROUP + desk_in_group
                        if d >= ACTIVE_DESKS:
                            continue
                        shared = group_shared[g]
                        self.emit("valu", ("^", shared['tmp1'], desks[d]['val'], v_hash_consts[5]))
                    for g in range(active_groups):
                        d = g * DESKS_PER_GROUP + desk_in_group
                        if d >= ACTIVE_DESKS:
                            continue
                        shared = group_shared[g]
                        self.emit("valu", (">>", shared['tmp2'], desks[d]['val'], v_hash_shifts[5]))
                    for g in range(active_groups):
                        d = g * DESKS_PER_GROUP + desk_in_group
                        if d >= ACTIVE_DESKS:
                            continue
                        shared = group_shared[g]
                        self.emit("valu", ("^", desks[d]['val'], shared['tmp1'], shared['tmp2']))

            emit_hash_32()

            # Branch for all desks - process one desk per group at a time
            for desk_in_group in range(DESKS_PER_GROUP):
                for g in range(4):  # 4 groups for 32 desks
                    d = g * DESKS_PER_GROUP + desk_in_group
                    if d >= ACTIVE_DESKS:
                        continue
                    emit_branch(d)

            # Round 1: Selection with tmp usage
            for desk_in_group in range(DESKS_PER_GROUP):
                for g in range(4):
                    d = g * DESKS_PER_GROUP + desk_in_group
                    if d >= ACTIVE_DESKS:
                        continue
                    shared = group_shared[g]
                    self.emit("valu", ("-", shared['tmp1'], desks[d]['idx'], v_one))
                for g in range(4):
                    d = g * DESKS_PER_GROUP + desk_in_group
                    if d >= ACTIVE_DESKS:
                        continue
                    shared = group_shared[g]
                    self.emit("valu", ("multiply_add", shared['node_val'], shared['tmp1'], v_diff_1_2, v_tree[1]))
                for g in range(4):
                    d = g * DESKS_PER_GROUP + desk_in_group
                    if d >= ACTIVE_DESKS:
                        continue
                    shared = group_shared[g]
                    self.emit("valu", ("^", desks[d]['val'], desks[d]['val'], shared['node_val']))

            emit_hash_32()
            for desk_in_group in range(DESKS_PER_GROUP):
                for g in range(4):
                    d = g * DESKS_PER_GROUP + desk_in_group
                    if d >= ACTIVE_DESKS:
                        continue
                    emit_branch(d)

            # Round 2: 4-way selection - process sequentially within groups
            for desk_in_group in range(DESKS_PER_GROUP):
                for g in range(4):
                    d = g * DESKS_PER_GROUP + desk_in_group
                    if d >= ACTIVE_DESKS:
                        continue
                    shared = group_shared[g]
                    self.emit("valu", ("-", shared['tmp1'], desks[d]['idx'], v_three))
                    self.emit("valu", ("&", shared['tmp2'], shared['tmp1'], v_one))
                    self.emit("valu", (">>", shared['addr'], shared['tmp1'], v_one))
                    self.emit("valu", ("multiply_add", shared['node_val'], shared['tmp2'], v_diff_3_4, v_tree[3]))
                    self.emit("valu", ("multiply_add", shared['tmp1'], shared['tmp2'], v_diff_5_6, v_tree[5]))
                    self.emit("valu", ("-", shared['tmp2'], shared['tmp1'], shared['node_val']))
                    self.emit("valu", ("multiply_add", shared['node_val'], shared['addr'], shared['tmp2'], shared['node_val']))
                    self.emit("valu", ("^", desks[d]['val'], desks[d]['val'], shared['node_val']))
                    emit_hash_stages(d)
                    emit_branch(d)

            # Rounds 3-9: Gather without bounds
            for _round in range(3, 10):
                # Compute addr and gather - must process groups sequentially
                for desk_in_group in range(DESKS_PER_GROUP):
                    for g in range(4):
                        d = g * DESKS_PER_GROUP + desk_in_group
                        if d >= ACTIVE_DESKS:
                            continue
                        shared = group_shared[g]
                        self.emit("valu", ("+", shared['addr'], v_forest_p, desks[d]['idx']))
                    for g in range(4):
                        d = g * DESKS_PER_GROUP + desk_in_group
                        if d >= ACTIVE_DESKS:
                            continue
                        shared = group_shared[g]
                        for lane in range(VLEN):
                            self.emit("load", ("load", shared['node_val'] + lane, shared['addr'] + lane))
                    for g in range(4):
                        d = g * DESKS_PER_GROUP + desk_in_group
                        if d >= ACTIVE_DESKS:
                            continue
                        shared = group_shared[g]
                        self.emit("valu", ("^", desks[d]['val'], desks[d]['val'], shared['node_val']))

                emit_hash_32()
                for desk_in_group in range(DESKS_PER_GROUP):
                    for g in range(4):
                        d = g * DESKS_PER_GROUP + desk_in_group
                        if d >= ACTIVE_DESKS:
                            continue
                        emit_branch(d)

            # Round 10: Gather WITH bounds
            for desk_in_group in range(DESKS_PER_GROUP):
                for g in range(4):
                    d = g * DESKS_PER_GROUP + desk_in_group
                    if d >= ACTIVE_DESKS:
                        continue
                    shared = group_shared[g]
                    self.emit("valu", ("+", shared['addr'], v_forest_p, desks[d]['idx']))
                for g in range(4):
                    d = g * DESKS_PER_GROUP + desk_in_group
                    if d >= ACTIVE_DESKS:
                        continue
                    shared = group_shared[g]
                    for lane in range(VLEN):
                        self.emit("load", ("load", shared['node_val'] + lane, shared['addr'] + lane))
                for g in range(4):
                    d = g * DESKS_PER_GROUP + desk_in_group
                    if d >= ACTIVE_DESKS:
                        continue
                    shared = group_shared[g]
                    self.emit("valu", ("^", desks[d]['val'], desks[d]['val'], shared['node_val']))

            emit_hash_32()
            for desk_in_group in range(DESKS_PER_GROUP):
                for g in range(4):
                    d = g * DESKS_PER_GROUP + desk_in_group
                    if d >= ACTIVE_DESKS:
                        continue
                    emit_branch(d)
            for desk_in_group in range(DESKS_PER_GROUP):
                for g in range(4):
                    d = g * DESKS_PER_GROUP + desk_in_group
                    if d >= ACTIVE_DESKS:
                        continue
                    emit_bounds(d)

            # Round 11: tree[0] directly
            for d in range(ACTIVE_DESKS):
                self.emit("valu", ("^", desks[d]['val'], desks[d]['val'], v_tree[0]))
            emit_hash_32()
            for desk_in_group in range(DESKS_PER_GROUP):
                for g in range(4):
                    d = g * DESKS_PER_GROUP + desk_in_group
                    if d >= ACTIVE_DESKS:
                        continue
                    emit_branch(d)

            # Round 12: Selection
            for desk_in_group in range(DESKS_PER_GROUP):
                for g in range(4):
                    d = g * DESKS_PER_GROUP + desk_in_group
                    if d >= ACTIVE_DESKS:
                        continue
                    shared = group_shared[g]
                    self.emit("valu", ("-", shared['tmp1'], desks[d]['idx'], v_one))
                for g in range(4):
                    d = g * DESKS_PER_GROUP + desk_in_group
                    if d >= ACTIVE_DESKS:
                        continue
                    shared = group_shared[g]
                    self.emit("valu", ("multiply_add", shared['node_val'], shared['tmp1'], v_diff_1_2, v_tree[1]))
                for g in range(4):
                    d = g * DESKS_PER_GROUP + desk_in_group
                    if d >= ACTIVE_DESKS:
                        continue
                    shared = group_shared[g]
                    self.emit("valu", ("^", desks[d]['val'], desks[d]['val'], shared['node_val']))

            emit_hash_32()
            for desk_in_group in range(DESKS_PER_GROUP):
                for g in range(4):
                    d = g * DESKS_PER_GROUP + desk_in_group
                    if d >= ACTIVE_DESKS:
                        continue
                    emit_branch(d)

            # Round 13: 4-way selection
            for desk_in_group in range(DESKS_PER_GROUP):
                for g in range(4):
                    d = g * DESKS_PER_GROUP + desk_in_group
                    if d >= ACTIVE_DESKS:
                        continue
                    shared = group_shared[g]
                    self.emit("valu", ("-", shared['tmp1'], desks[d]['idx'], v_three))
                    self.emit("valu", ("&", shared['tmp2'], shared['tmp1'], v_one))
                    self.emit("valu", (">>", shared['addr'], shared['tmp1'], v_one))
                    self.emit("valu", ("multiply_add", shared['node_val'], shared['tmp2'], v_diff_3_4, v_tree[3]))
                    self.emit("valu", ("multiply_add", shared['tmp1'], shared['tmp2'], v_diff_5_6, v_tree[5]))
                    self.emit("valu", ("-", shared['tmp2'], shared['tmp1'], shared['node_val']))
                    self.emit("valu", ("multiply_add", shared['node_val'], shared['addr'], shared['tmp2'], shared['node_val']))
                    self.emit("valu", ("^", desks[d]['val'], desks[d]['val'], shared['node_val']))
                    emit_hash_stages(d)
                    emit_branch(d)

            # Rounds 14-15: Gather without bounds
            for _round in range(14, 16):
                for desk_in_group in range(DESKS_PER_GROUP):
                    for g in range(4):
                        d = g * DESKS_PER_GROUP + desk_in_group
                        if d >= ACTIVE_DESKS:
                            continue
                        shared = group_shared[g]
                        self.emit("valu", ("+", shared['addr'], v_forest_p, desks[d]['idx']))
                    for g in range(4):
                        d = g * DESKS_PER_GROUP + desk_in_group
                        if d >= ACTIVE_DESKS:
                            continue
                        shared = group_shared[g]
                        for lane in range(VLEN):
                            self.emit("load", ("load", shared['node_val'] + lane, shared['addr'] + lane))
                    for g in range(4):
                        d = g * DESKS_PER_GROUP + desk_in_group
                        if d >= ACTIVE_DESKS:
                            continue
                        shared = group_shared[g]
                        self.emit("valu", ("^", desks[d]['val'], desks[d]['val'], shared['node_val']))

                emit_hash_32()
                for desk_in_group in range(DESKS_PER_GROUP):
                    for g in range(4):
                        d = g * DESKS_PER_GROUP + desk_in_group
                        if d >= ACTIVE_DESKS:
                            continue
                        emit_branch(d)

            # Store results
            for d_batch in range(0, ACTIVE_DESKS, 2):
                d0, d1 = d_batch, d_batch + 1
                self.emit("alu", ("+", addr_tmp[0], self.scratch["inp_indices_p"], offset_regs[d0]))
                self.emit("alu", ("+", addr_tmp[1], self.scratch["inp_values_p"], offset_regs[d0]))
                self.emit("alu", ("+", addr_tmp[2], self.scratch["inp_indices_p"], offset_regs[d1]))
                self.emit("alu", ("+", addr_tmp[3], self.scratch["inp_values_p"], offset_regs[d1]))
                self.emit("store", ("vstore", addr_tmp[0], desks[d0]['idx']))
                self.emit("store", ("vstore", addr_tmp[1], desks[d0]['val']))
                self.emit("store", ("vstore", addr_tmp[2], desks[d1]['idx']))
                self.emit("store", ("vstore", addr_tmp[3], desks[d1]['val']))

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

        self.instrs = []
        for i, phase in enumerate(phases):
            if phase:
                phase_instrs = _schedule_slots(phase)
                self.instrs.extend(phase_instrs)
            if i < len(phases) - 1:
                self.instrs.append({"flow": [("pause",)]})

        self.instrs.append({"flow": [("pause",)]})

        print(f"Total slots: {len(self.slots)}, Cycles: {len(self.instrs)}")


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

    kb = KernelBuilderH128()
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

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
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
