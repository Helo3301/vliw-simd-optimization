"""
# Experiment H92: Combined Double-Pumped Hash + Cross-Tile Pipelining

**GOAL:** Combine the best optimizations from H87 and H90 to achieve cumulative benefits.

**APPROACH:** Use H87's double-pump structure as the base, but add cross-tile software
pipelining for gather rounds only. For non-gather rounds, process tiles sequentially
with double-pump.

**KEY INSIGHT:** The main benefit of H90's cross-tile approach is for gather rounds
where memory latency is the bottleneck. By issuing loads for both tiles before
computing for either, we maximize overlap between memory and compute.

For non-gather rounds (0, 1, 2, 11, 12, 13), the bottleneck is VALU, so H87's
double-pump approach is better.

**BASELINE:** H87 = 1,851 cycles, H90 = 1,867 cycles, H85 = 1,898 cycles
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


class KernelBuilderH92:
    """
    H92: Combined Double-Pumped Hash + Cross-Tile Pipelining
    Based on H87 with cross-tile software pipelining for gather rounds.
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
        Build kernel combining H87's double-pump with H90's cross-tile pipelining.
        """
        NUM_TILES = 2
        NUM_DESKS = 16

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

        # Preload tree nodes 0-14
        NUM_PRELOADED = 15
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

        # Allocate per-desk vectors for BOTH tiles (from H90)
        all_desks = []  # [tile][desk]
        for t in range(NUM_TILES):
            tile_desks = []
            for d in range(NUM_DESKS):
                desk = {
                    'idx': self.alloc_vec(f"v_idx_t{t}_d{d}"),
                    'val': self.alloc_vec(f"v_val_t{t}_d{d}"),
                    'node_val': self.alloc_vec(f"v_node_t{t}_d{d}"),
                    'addr': self.alloc_vec(f"v_addr_t{t}_d{d}"),
                }
                tile_desks.append(desk)
            all_desks.append(tile_desks)

        # Shared temp registers - 16 vectors, shared across tiles (from H87)
        NUM_SHARED_TEMPS = 16
        shared_temps = [self.alloc_vec(f"v_shared_tmp_{i}") for i in range(NUM_SHARED_TEMPS)]

        def get_temps(desk_idx):
            """Get temp registers for a desk within a tile."""
            pair = desk_idx // 2  # 8 pairs
            t1_idx = pair * 2
            t2_idx = pair * 2 + 1
            return shared_temps[t1_idx], shared_temps[t2_idx]

        # Offset addresses for both tiles
        offset_regs = []
        for t in range(NUM_TILES):
            tile_offsets = [self.alloc_scratch(f"off_t{t}_{d}") for d in range(NUM_DESKS)]
            offset_regs.append(tile_offsets)
        addr_tmp = [self.alloc_scratch(f"addr_tmp_{i}") for i in range(32)]

        print(f"Scratch usage: {self.scratch_ptr} / {SCRATCH_SIZE}")

        # Pause before main computation
        self.emit("flow", ("pause",))

        # Even and odd desks within a tile for double-pump pattern (from H87)
        even_desks = list(range(0, NUM_DESKS, 2))  # [0, 2, 4, 6, 8, 10, 12, 14]
        odd_desks = list(range(1, NUM_DESKS, 2))   # [1, 3, 5, 7, 9, 11, 13, 15]

        # Double-pumped hash stages for a single tile (from H87)
        def emit_hash_stages_all_desks(tile_idx):
            """Emit hash stages for all desks in a tile with double-pump."""
            desks = all_desks[tile_idx]

            # Stage 0 (FMA) - all desks
            for d in range(NUM_DESKS):
                self.emit("valu", ("multiply_add", desks[d]['val'], desks[d]['val'], v_fma_mult[0], v_hash_consts[0]))

            # Stage 1 (3 ops) - even then odd
            for d in even_desks:
                tmp1, tmp2 = get_temps(d)
                self.emit("valu", ("^", tmp1, desks[d]['val'], v_hash_consts[1]))
            for d in even_desks:
                tmp1, tmp2 = get_temps(d)
                self.emit("valu", (">>", tmp2, desks[d]['val'], v_hash_shifts[1]))
            for d in even_desks:
                tmp1, tmp2 = get_temps(d)
                self.emit("valu", ("^", desks[d]['val'], tmp1, tmp2))
            for d in odd_desks:
                tmp1, tmp2 = get_temps(d)
                self.emit("valu", ("^", tmp1, desks[d]['val'], v_hash_consts[1]))
            for d in odd_desks:
                tmp1, tmp2 = get_temps(d)
                self.emit("valu", (">>", tmp2, desks[d]['val'], v_hash_shifts[1]))
            for d in odd_desks:
                tmp1, tmp2 = get_temps(d)
                self.emit("valu", ("^", desks[d]['val'], tmp1, tmp2))

            # Stage 2 (FMA) - all desks
            for d in range(NUM_DESKS):
                self.emit("valu", ("multiply_add", desks[d]['val'], desks[d]['val'], v_fma_mult[2], v_hash_consts[2]))

            # Stage 3 (3 ops) - even then odd
            for d in even_desks:
                tmp1, tmp2 = get_temps(d)
                self.emit("valu", ("+", tmp1, desks[d]['val'], v_hash_consts[3]))
            for d in even_desks:
                tmp1, tmp2 = get_temps(d)
                self.emit("valu", ("<<", tmp2, desks[d]['val'], v_hash_shifts[3]))
            for d in even_desks:
                tmp1, tmp2 = get_temps(d)
                self.emit("valu", ("^", desks[d]['val'], tmp1, tmp2))
            for d in odd_desks:
                tmp1, tmp2 = get_temps(d)
                self.emit("valu", ("+", tmp1, desks[d]['val'], v_hash_consts[3]))
            for d in odd_desks:
                tmp1, tmp2 = get_temps(d)
                self.emit("valu", ("<<", tmp2, desks[d]['val'], v_hash_shifts[3]))
            for d in odd_desks:
                tmp1, tmp2 = get_temps(d)
                self.emit("valu", ("^", desks[d]['val'], tmp1, tmp2))

            # Stage 4 (FMA) - all desks
            for d in range(NUM_DESKS):
                self.emit("valu", ("multiply_add", desks[d]['val'], desks[d]['val'], v_fma_mult[4], v_hash_consts[4]))

            # Stage 5 (3 ops) - even then odd
            for d in even_desks:
                tmp1, tmp2 = get_temps(d)
                self.emit("valu", ("^", tmp1, desks[d]['val'], v_hash_consts[5]))
            for d in even_desks:
                tmp1, tmp2 = get_temps(d)
                self.emit("valu", (">>", tmp2, desks[d]['val'], v_hash_shifts[5]))
            for d in even_desks:
                tmp1, tmp2 = get_temps(d)
                self.emit("valu", ("^", desks[d]['val'], tmp1, tmp2))
            for d in odd_desks:
                tmp1, tmp2 = get_temps(d)
                self.emit("valu", ("^", tmp1, desks[d]['val'], v_hash_consts[5]))
            for d in odd_desks:
                tmp1, tmp2 = get_temps(d)
                self.emit("valu", (">>", tmp2, desks[d]['val'], v_hash_shifts[5]))
            for d in odd_desks:
                tmp1, tmp2 = get_temps(d)
                self.emit("valu", ("^", desks[d]['val'], tmp1, tmp2))

        def emit_hash_stages(tile_idx, desk_idx):
            """Emit hash stages for a single desk."""
            d = all_desks[tile_idx][desk_idx]
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

        def emit_branch(tile_idx, desk_idx):
            d = all_desks[tile_idx][desk_idx]
            tmp1, _ = get_temps(desk_idx)
            self.emit("valu", ("&", tmp1, d['val'], v_one))
            self.emit("valu", ("multiply_add", d['idx'], d['idx'], v_two, v_one))
            self.emit("valu", ("+", d['idx'], d['idx'], tmp1))

        def emit_bounds(tile_idx, desk_idx):
            d = all_desks[tile_idx][desk_idx]
            tmp1, _ = get_temps(desk_idx)
            self.emit("valu", ("<", tmp1, d['idx'], v_n_nodes))
            self.emit("valu", ("*", d['idx'], d['idx'], tmp1))

        # Helper for a complete tile round (from H87)
        def emit_tile_round_0(t):
            """Round 0: All indices = 0, use tree[0] directly."""
            for d in range(NUM_DESKS):
                self.emit("valu", ("^", all_desks[t][d]['val'], all_desks[t][d]['val'], v_tree[0]))
            emit_hash_stages_all_desks(t)
            for d in range(NUM_DESKS):
                emit_branch(t, d)

        def emit_tile_round_1(t):
            """Round 1: Indices in {1, 2}, use arithmetic selection."""
            for d in even_desks:
                tmp1, _ = get_temps(d)
                self.emit("valu", ("-", tmp1, all_desks[t][d]['idx'], v_one))
            for d in even_desks:
                tmp1, _ = get_temps(d)
                self.emit("valu", ("multiply_add", all_desks[t][d]['node_val'], tmp1, v_diff_1_2, v_tree[1]))
            for d in odd_desks:
                tmp1, _ = get_temps(d)
                self.emit("valu", ("-", tmp1, all_desks[t][d]['idx'], v_one))
            for d in odd_desks:
                tmp1, _ = get_temps(d)
                self.emit("valu", ("multiply_add", all_desks[t][d]['node_val'], tmp1, v_diff_1_2, v_tree[1]))
            for d in range(NUM_DESKS):
                self.emit("valu", ("^", all_desks[t][d]['val'], all_desks[t][d]['val'], all_desks[t][d]['node_val']))
            emit_hash_stages_all_desks(t)
            for d in range(NUM_DESKS):
                emit_branch(t, d)

        def emit_tile_round_2(t):
            """Round 2: Indices in {3, 4, 5, 6}, use 4-way arithmetic selection."""
            for d in range(NUM_DESKS):
                desk = all_desks[t][d]
                tmp1, tmp2 = get_temps(d)
                self.emit("valu", ("-", tmp1, desk['idx'], v_three))
                self.emit("valu", ("&", tmp2, tmp1, v_one))
                self.emit("valu", (">>", desk['addr'], tmp1, v_one))
                self.emit("valu", ("multiply_add", desk['node_val'], tmp2, v_diff_3_4, v_tree[3]))
                self.emit("valu", ("multiply_add", tmp1, tmp2, v_diff_5_6, v_tree[5]))
                self.emit("valu", ("-", tmp2, tmp1, desk['node_val']))
                self.emit("valu", ("multiply_add", desk['node_val'], desk['addr'], tmp2, desk['node_val']))
                self.emit("valu", ("^", desk['val'], desk['val'], desk['node_val']))
                emit_hash_stages(t, d)
                emit_branch(t, d)

        # ===== LOAD BOTH TILES' DATA AT START (from H90) =====
        for t in range(NUM_TILES):
            tile_offset = t * NUM_DESKS * VLEN
            # Load offsets
            for d in range(NUM_DESKS):
                self.emit("load", ("const", offset_regs[t][d], tile_offset + d * VLEN))
            # Compute addresses and load
            for d in range(NUM_DESKS):
                self.emit("alu", ("+", addr_tmp[d*2], self.scratch["inp_indices_p"], offset_regs[t][d]))
                self.emit("alu", ("+", addr_tmp[d*2+1], self.scratch["inp_values_p"], offset_regs[t][d]))
            for d in range(NUM_DESKS):
                self.emit("load", ("vload", all_desks[t][d]['idx'], addr_tmp[d*2]))
                self.emit("load", ("vload", all_desks[t][d]['val'], addr_tmp[d*2+1]))

        # ===== ROUNDS 0-2: Non-gather, process tiles sequentially with double-pump =====
        for t in range(NUM_TILES):
            emit_tile_round_0(t)
        for t in range(NUM_TILES):
            emit_tile_round_1(t)
        for t in range(NUM_TILES):
            emit_tile_round_2(t)

        # ===== ROUNDS 3-9: Gather - Cross-tile software pipelining =====
        # Issue ALL loads for BOTH tiles, then compute for tile 0, then tile 1
        for _round in range(3, 10):
            # Phase 1: Address calc and loads for both tiles
            for t in range(NUM_TILES):
                for d in range(NUM_DESKS):
                    desk = all_desks[t][d]
                    self.emit("valu", ("+", desk['addr'], v_forest_p, desk['idx']))
                for d in range(NUM_DESKS):
                    desk = all_desks[t][d]
                    for lane in range(VLEN):
                        self.emit("load", ("load", desk['node_val'] + lane, desk['addr'] + lane))
            # Phase 2: Compute for tile 0
            for d in range(NUM_DESKS):
                self.emit("valu", ("^", all_desks[0][d]['val'], all_desks[0][d]['val'], all_desks[0][d]['node_val']))
            emit_hash_stages_all_desks(0)
            for d in range(NUM_DESKS):
                emit_branch(0, d)
            # Phase 3: Compute for tile 1
            for d in range(NUM_DESKS):
                self.emit("valu", ("^", all_desks[1][d]['val'], all_desks[1][d]['val'], all_desks[1][d]['node_val']))
            emit_hash_stages_all_desks(1)
            for d in range(NUM_DESKS):
                emit_branch(1, d)

        # ===== ROUND 10: Gather with bounds - Cross-tile software pipelining =====
        # Phase 1: Address calc and loads for both tiles
        for t in range(NUM_TILES):
            for d in range(NUM_DESKS):
                desk = all_desks[t][d]
                self.emit("valu", ("+", desk['addr'], v_forest_p, desk['idx']))
            for d in range(NUM_DESKS):
                desk = all_desks[t][d]
                for lane in range(VLEN):
                    self.emit("load", ("load", desk['node_val'] + lane, desk['addr'] + lane))
        # Phase 2: Compute for tile 0
        for d in range(NUM_DESKS):
            self.emit("valu", ("^", all_desks[0][d]['val'], all_desks[0][d]['val'], all_desks[0][d]['node_val']))
        emit_hash_stages_all_desks(0)
        for d in range(NUM_DESKS):
            emit_branch(0, d)
        for d in range(NUM_DESKS):
            emit_bounds(0, d)
        # Phase 3: Compute for tile 1
        for d in range(NUM_DESKS):
            self.emit("valu", ("^", all_desks[1][d]['val'], all_desks[1][d]['val'], all_desks[1][d]['node_val']))
        emit_hash_stages_all_desks(1)
        for d in range(NUM_DESKS):
            emit_branch(1, d)
        for d in range(NUM_DESKS):
            emit_bounds(1, d)

        # ===== ROUNDS 11-13: Non-gather, process tiles sequentially =====
        for t in range(NUM_TILES):
            emit_tile_round_0(t)  # Round 11 same as round 0
        for t in range(NUM_TILES):
            emit_tile_round_1(t)  # Round 12 same as round 1
        for t in range(NUM_TILES):
            emit_tile_round_2(t)  # Round 13 same as round 2

        # ===== ROUNDS 14-15: Gather - Cross-tile software pipelining =====
        for _round in range(14, 16):
            # Phase 1: Address calc and loads for both tiles
            for t in range(NUM_TILES):
                for d in range(NUM_DESKS):
                    desk = all_desks[t][d]
                    self.emit("valu", ("+", desk['addr'], v_forest_p, desk['idx']))
                for d in range(NUM_DESKS):
                    desk = all_desks[t][d]
                    for lane in range(VLEN):
                        self.emit("load", ("load", desk['node_val'] + lane, desk['addr'] + lane))
            # Phase 2: Compute for tile 0
            for d in range(NUM_DESKS):
                self.emit("valu", ("^", all_desks[0][d]['val'], all_desks[0][d]['val'], all_desks[0][d]['node_val']))
            emit_hash_stages_all_desks(0)
            for d in range(NUM_DESKS):
                emit_branch(0, d)
            # Phase 3: Compute for tile 1
            for d in range(NUM_DESKS):
                self.emit("valu", ("^", all_desks[1][d]['val'], all_desks[1][d]['val'], all_desks[1][d]['node_val']))
            emit_hash_stages_all_desks(1)
            for d in range(NUM_DESKS):
                emit_branch(1, d)

        # Store results for both tiles
        for t in range(NUM_TILES):
            for d in range(NUM_DESKS):
                self.emit("alu", ("+", addr_tmp[d*2], self.scratch["inp_indices_p"], offset_regs[t][d]))
                self.emit("alu", ("+", addr_tmp[d*2+1], self.scratch["inp_values_p"], offset_regs[t][d]))
            for d in range(NUM_DESKS):
                self.emit("store", ("vstore", addr_tmp[d*2], all_desks[t][d]['idx']))
                self.emit("store", ("vstore", addr_tmp[d*2+1], all_desks[t][d]['val']))

        # Schedule operations
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

    kb = KernelBuilderH92()
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
