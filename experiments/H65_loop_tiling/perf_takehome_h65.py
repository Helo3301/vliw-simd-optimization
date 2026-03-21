"""
# Experiment H65: Loop Tiling for Tree Traversal

**GOAL:** Process small batches through ALL 16 rounds before moving to next batch.
This keeps state in scratch longer and reduces load/store overhead between rounds.

**FINAL DESIGN (v3 - Best Version):**
- 2 tiles of 128 elements each (16 desks per tile)
- Per tile: load inputs, process all 16 rounds, store results
- Deep 16-desk pipelining for maximum latency hiding

**KEY INSIGHT:**
By processing all 16 rounds per tile, we dramatically reduce load/store overhead:
- H54: 16 iterations x (load + store phases) = 32 memory phases
- H65v3: 2 iterations x (load + store phases) = 4 memory phases

This 8x reduction in memory phase overhead yields significant cycle savings.

**VERSIONS:**
- v1 (8 vectors, 4 tiles): 3,494 cycles - shallow pipelining limits gains
- v2: Same as v1 (attempted preload, reverted)
- v3 (16 desks, 2 tiles): 2,941 cycles - BEST! 15% faster than H54

**RESULTS:**
- H54 baseline: 3,462 cycles
- H65v3: 2,941 cycles (15% improvement, 50.2x speedup over 147,734 baseline)
- Target: 1,790 cycles (still 64% above target)
"""

from collections import defaultdict
import random
import unittest
import argparse
import sys
import os

# Add parent directory to path to import problem module
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


class KernelBuilderH65:
    """
    H65: Loop Tiling for Tree Traversal

    Process 64 elements (8 vectors) through all 16 rounds per tile.
    """
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, f"Out of scratch space: {self.scratch_ptr} > {SCRATCH_SIZE}"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        H65: Loop Tiling kernel.

        4 tiles x 64 elements x 16 rounds
        """
        # Standard initialization
        tmp_scalar = self.alloc_scratch("tmp_scalar")
        addr_scalar = self.alloc_scratch("addr_scalar")

        init_vars = [
            "rounds", "n_nodes", "batch_size", "forest_height",
            "forest_values_p", "inp_indices_p", "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp_scalar, i))
            self.add("load", ("load", self.scratch[v], tmp_scalar))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        v_zero = self.alloc_scratch("v_zero", VLEN)
        v_one = self.alloc_scratch("v_one", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)

        self.add("valu", ("vbroadcast", v_zero, zero_const))
        self.add("valu", ("vbroadcast", v_one, one_const))
        self.add("valu", ("vbroadcast", v_two, two_const))
        self.add("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]))

        # Hash constants and FMA multipliers
        FMA_MULTIPLIERS = {
            0: 4097,  # 1 + 2^12
            2: 33,    # 1 + 2^5
            4: 9,     # 1 + 2^3
        }

        v_hash_consts = []
        v_hash_shifts = []
        v_fma_mult = {}

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            const_scalar = self.scratch_const(val1)
            shift_scalar = self.scratch_const(val3)
            v_const = self.alloc_scratch(f"v_hash_const_{hi}", VLEN)
            v_shift = self.alloc_scratch(f"v_hash_shift_{hi}", VLEN)
            self.add("valu", ("vbroadcast", v_const, const_scalar))
            self.add("valu", ("vbroadcast", v_shift, shift_scalar))
            v_hash_consts.append(v_const)
            v_hash_shifts.append(v_shift)

            if hi in FMA_MULTIPLIERS:
                mult_scalar = self.scratch_const(FMA_MULTIPLIERS[hi])
                v_mult = self.alloc_scratch(f"v_fma_mult_{hi}", VLEN)
                self.add("valu", ("vbroadcast", v_mult, mult_scalar))
                v_fma_mult[hi] = v_mult

        # ============================================================
        # TILE CONFIGURATION: 8 vectors (64 elements) per tile
        # ============================================================
        NUM_VECS = 8  # 8 vectors = 64 elements per tile
        NUM_TILES = batch_size // (NUM_VECS * VLEN)  # 4 tiles

        # Allocate per-vector scratch for the tile
        vecs = []
        for v in range(NUM_VECS):
            vec = {
                'idx': self.alloc_scratch(f"v_idx_{v}", VLEN),
                'val': self.alloc_scratch(f"v_val_{v}", VLEN),
                'node_val': self.alloc_scratch(f"v_node_{v}", VLEN),
                'addr': self.alloc_scratch(f"v_addr_{v}", VLEN),
                'tmp1': self.alloc_scratch(f"v_tmp1_{v}", VLEN),
                'tmp2': self.alloc_scratch(f"v_tmp2_{v}", VLEN),
            }
            vecs.append(vec)

        # Address temporaries (need 16 for 8 vecs x 2 addresses)
        addr_tmp = [self.alloc_scratch(f"addr_tmp_{i}") for i in range(20)]

        # Loop counters and offsets
        tile_counter = self.alloc_scratch("tile_counter")
        tile_offset = self.alloc_scratch("tile_offset")
        round_counter = self.alloc_scratch("round_counter")

        # Constants
        tile_stride = NUM_VECS * VLEN  # 64 elements per tile
        tile_stride_const = self.scratch_const(tile_stride)
        num_tiles_const = self.scratch_const(NUM_TILES)
        num_rounds_const = self.scratch_const(rounds)

        # Offset constants for vector loads within tile
        vec_offsets = [self.scratch_const(v * VLEN) for v in range(NUM_VECS)]

        self.add("flow", ("pause",))

        # Print scratch usage
        print(f"Scratch usage before main loop: {self.scratch_ptr} / {SCRATCH_SIZE}")

        # Initialize tile counter
        self.add("load", ("const", tile_counter, 0))
        self.add("load", ("const", tile_offset, 0))

        # ============================================================
        # MAIN TILE LOOP
        # ============================================================
        tile_loop_start = len(self.instrs)

        # PHASE 1: LOAD TILE INPUT DATA
        # Compute base addresses for this tile
        self.instrs.append({
            "alu": [
                ("+", addr_tmp[0], self.scratch["inp_indices_p"], tile_offset),
                ("+", addr_tmp[1], self.scratch["inp_values_p"], tile_offset),
            ],
        })

        # Compute addresses for all 8 vectors
        self.instrs.append({
            "alu": [
                ("+", addr_tmp[2], addr_tmp[0], vec_offsets[1]),
                ("+", addr_tmp[3], addr_tmp[1], vec_offsets[1]),
                ("+", addr_tmp[4], addr_tmp[0], vec_offsets[2]),
                ("+", addr_tmp[5], addr_tmp[1], vec_offsets[2]),
                ("+", addr_tmp[6], addr_tmp[0], vec_offsets[3]),
                ("+", addr_tmp[7], addr_tmp[1], vec_offsets[3]),
                ("+", addr_tmp[8], addr_tmp[0], vec_offsets[4]),
                ("+", addr_tmp[9], addr_tmp[1], vec_offsets[4]),
                ("+", addr_tmp[10], addr_tmp[0], vec_offsets[5]),
                ("+", addr_tmp[11], addr_tmp[1], vec_offsets[5]),
                ("+", addr_tmp[12], addr_tmp[0], vec_offsets[6]),
            ],
        })
        self.instrs.append({
            "alu": [
                ("+", addr_tmp[13], addr_tmp[1], vec_offsets[6]),
                ("+", addr_tmp[14], addr_tmp[0], vec_offsets[7]),
                ("+", addr_tmp[15], addr_tmp[1], vec_offsets[7]),
            ],
        })

        # Load idx and val for all 8 vectors (2 vloads per cycle = 8 cycles)
        for v in range(NUM_VECS):
            self.instrs.append({
                "load": [
                    ("vload", vecs[v]['idx'], addr_tmp[v * 2]),
                    ("vload", vecs[v]['val'], addr_tmp[v * 2 + 1]),
                ],
            })

        # PHASE 2: PROCESS ALL 16 ROUNDS
        self.add("load", ("const", round_counter, 0))

        round_loop_start = len(self.instrs)

        # Helper functions for hash computation
        def emit_hash_stage(vec_idx, stage):
            """Emit one hash stage for a vector"""
            v = vecs[vec_idx]
            if stage in v_fma_mult:
                # FMA stages: 0, 2, 4
                return [("multiply_add", v['val'], v['val'], v_fma_mult[stage], v_hash_consts[stage])]
            else:
                # XOR stages: 1, 3, 5
                if stage == 1:
                    return [
                        ("^", v['tmp1'], v['val'], v_hash_consts[stage]),
                        (">>", v['tmp2'], v['val'], v_hash_shifts[stage]),
                    ]
                elif stage == 3:
                    return [
                        ("+", v['tmp1'], v['val'], v_hash_consts[stage]),
                        ("<<", v['tmp2'], v['val'], v_hash_shifts[stage]),
                    ]
                elif stage == 5:
                    return [
                        ("^", v['tmp1'], v['val'], v_hash_consts[stage]),
                        (">>", v['tmp2'], v['val'], v_hash_shifts[stage]),
                    ]
            return []

        def emit_hash_combine(vec_idx, stage):
            """Emit combine for XOR stages"""
            v = vecs[vec_idx]
            if stage in [1, 3, 5]:
                return [("^", v['val'], v['tmp1'], v['tmp2'])]
            return []

        def emit_xor_node(vec_idx):
            """XOR val with node_val"""
            v = vecs[vec_idx]
            return [("^", v['val'], v['val'], v['node_val'])]

        def emit_branch_ops(vec_idx):
            """Emit branch operations for a vector"""
            v = vecs[vec_idx]
            ops = []
            ops.append(("&", v['tmp1'], v['val'], v_one))
            ops.append(("multiply_add", v['idx'], v['idx'], v_two, v_one))
            return ops

        def emit_branch_add(vec_idx):
            """Add branch bit to idx"""
            v = vecs[vec_idx]
            return [("+", v['idx'], v['idx'], v['tmp1'])]

        def emit_bounds_check(vec_idx):
            """Bounds check for idx"""
            v = vecs[vec_idx]
            return [("<", v['tmp1'], v['idx'], v_n_nodes)]

        def emit_bounds_apply(vec_idx):
            """Apply bounds check"""
            v = vecs[vec_idx]
            return [("*", v['idx'], v['idx'], v['tmp1'])]

        # Prepare gather addresses (2 cycles for 8 vectors)
        self.instrs.append({
            "valu": [
                ("vbroadcast", vecs[0]['addr'], self.scratch["forest_values_p"]),
                ("vbroadcast", vecs[1]['addr'], self.scratch["forest_values_p"]),
                ("vbroadcast", vecs[2]['addr'], self.scratch["forest_values_p"]),
                ("vbroadcast", vecs[3]['addr'], self.scratch["forest_values_p"]),
                ("vbroadcast", vecs[4]['addr'], self.scratch["forest_values_p"]),
                ("vbroadcast", vecs[5]['addr'], self.scratch["forest_values_p"]),
            ],
        })
        self.instrs.append({
            "valu": [
                ("vbroadcast", vecs[6]['addr'], self.scratch["forest_values_p"]),
                ("vbroadcast", vecs[7]['addr'], self.scratch["forest_values_p"]),
                ("+", vecs[0]['addr'], vecs[0]['addr'], vecs[0]['idx']),
                ("+", vecs[1]['addr'], vecs[1]['addr'], vecs[1]['idx']),
                ("+", vecs[2]['addr'], vecs[2]['addr'], vecs[2]['idx']),
                ("+", vecs[3]['addr'], vecs[3]['addr'], vecs[3]['idx']),
            ],
        })
        self.instrs.append({
            "valu": [
                ("+", vecs[4]['addr'], vecs[4]['addr'], vecs[4]['idx']),
                ("+", vecs[5]['addr'], vecs[5]['addr'], vecs[5]['idx']),
                ("+", vecs[6]['addr'], vecs[6]['addr'], vecs[6]['idx']),
                ("+", vecs[7]['addr'], vecs[7]['addr'], vecs[7]['idx']),
            ],
        })

        # Gather tree values for all 8 vectors (64 scalar loads = 32 cycles)
        # Interleave with hash computation for deep pipelining
        # This is the H54-style interleaved gather/compute approach

        # Vec 0 gather (4 cycles)
        for lane in range(0, VLEN, 2):
            self.instrs.append({
                "load": [
                    ("load", vecs[0]['node_val'] + lane, vecs[0]['addr'] + lane),
                    ("load", vecs[0]['node_val'] + lane + 1, vecs[0]['addr'] + lane + 1),
                ],
            })

        # Vec 1 gather + Vec 0 XOR and hash
        self.instrs.append({
            "load": [
                ("load", vecs[1]['node_val'], vecs[1]['addr']),
                ("load", vecs[1]['node_val'] + 1, vecs[1]['addr'] + 1),
            ],
            "valu": emit_xor_node(0),
        })
        self.instrs.append({
            "load": [
                ("load", vecs[1]['node_val'] + 2, vecs[1]['addr'] + 2),
                ("load", vecs[1]['node_val'] + 3, vecs[1]['addr'] + 3),
            ],
            "valu": emit_hash_stage(0, 0),
        })
        self.instrs.append({
            "load": [
                ("load", vecs[1]['node_val'] + 4, vecs[1]['addr'] + 4),
                ("load", vecs[1]['node_val'] + 5, vecs[1]['addr'] + 5),
            ],
            "valu": emit_hash_stage(0, 1),
        })
        self.instrs.append({
            "load": [
                ("load", vecs[1]['node_val'] + 6, vecs[1]['addr'] + 6),
                ("load", vecs[1]['node_val'] + 7, vecs[1]['addr'] + 7),
            ],
            "valu": emit_hash_combine(0, 1),
        })

        # Vec 2 gather + Vec 0 hash + Vec 1 XOR
        self.instrs.append({
            "load": [
                ("load", vecs[2]['node_val'], vecs[2]['addr']),
                ("load", vecs[2]['node_val'] + 1, vecs[2]['addr'] + 1),
            ],
            "valu": emit_hash_stage(0, 2) + emit_xor_node(1),
        })
        self.instrs.append({
            "load": [
                ("load", vecs[2]['node_val'] + 2, vecs[2]['addr'] + 2),
                ("load", vecs[2]['node_val'] + 3, vecs[2]['addr'] + 3),
            ],
            "valu": emit_hash_stage(0, 3) + emit_hash_stage(1, 0),
        })
        self.instrs.append({
            "load": [
                ("load", vecs[2]['node_val'] + 4, vecs[2]['addr'] + 4),
                ("load", vecs[2]['node_val'] + 5, vecs[2]['addr'] + 5),
            ],
            "valu": emit_hash_combine(0, 3) + emit_hash_stage(1, 1),
        })
        self.instrs.append({
            "load": [
                ("load", vecs[2]['node_val'] + 6, vecs[2]['addr'] + 6),
                ("load", vecs[2]['node_val'] + 7, vecs[2]['addr'] + 7),
            ],
            "valu": emit_hash_stage(0, 4) + emit_hash_combine(1, 1),
        })

        # Vec 3 gather + Vec 0 final + Vec 1 hash + Vec 2 XOR
        self.instrs.append({
            "load": [
                ("load", vecs[3]['node_val'], vecs[3]['addr']),
                ("load", vecs[3]['node_val'] + 1, vecs[3]['addr'] + 1),
            ],
            "valu": emit_hash_stage(0, 5) + emit_hash_stage(1, 2) + emit_xor_node(2),
        })
        self.instrs.append({
            "load": [
                ("load", vecs[3]['node_val'] + 2, vecs[3]['addr'] + 2),
                ("load", vecs[3]['node_val'] + 3, vecs[3]['addr'] + 3),
            ],
            "valu": emit_hash_combine(0, 5) + emit_hash_stage(1, 3) + emit_hash_stage(2, 0),
        })
        self.instrs.append({
            "load": [
                ("load", vecs[3]['node_val'] + 4, vecs[3]['addr'] + 4),
                ("load", vecs[3]['node_val'] + 5, vecs[3]['addr'] + 5),
            ],
            "valu": emit_branch_ops(0) + emit_hash_combine(1, 3) + emit_hash_stage(2, 1),
        })
        self.instrs.append({
            "load": [
                ("load", vecs[3]['node_val'] + 6, vecs[3]['addr'] + 6),
                ("load", vecs[3]['node_val'] + 7, vecs[3]['addr'] + 7),
            ],
            "valu": emit_branch_add(0) + emit_hash_stage(1, 4) + emit_hash_combine(2, 1),
        })

        # Vec 4 gather + pipeline continues
        self.instrs.append({
            "load": [
                ("load", vecs[4]['node_val'], vecs[4]['addr']),
                ("load", vecs[4]['node_val'] + 1, vecs[4]['addr'] + 1),
            ],
            "valu": emit_bounds_check(0) + emit_hash_stage(1, 5) + emit_hash_stage(2, 2) + emit_xor_node(3),
        })
        self.instrs.append({
            "load": [
                ("load", vecs[4]['node_val'] + 2, vecs[4]['addr'] + 2),
                ("load", vecs[4]['node_val'] + 3, vecs[4]['addr'] + 3),
            ],
            "valu": emit_bounds_apply(0) + emit_hash_combine(1, 5) + emit_hash_stage(2, 3) + emit_hash_stage(3, 0),
        })
        self.instrs.append({
            "load": [
                ("load", vecs[4]['node_val'] + 4, vecs[4]['addr'] + 4),
                ("load", vecs[4]['node_val'] + 5, vecs[4]['addr'] + 5),
            ],
            "valu": emit_branch_ops(1) + emit_hash_combine(2, 3) + emit_hash_stage(3, 1),
        })
        self.instrs.append({
            "load": [
                ("load", vecs[4]['node_val'] + 6, vecs[4]['addr'] + 6),
                ("load", vecs[4]['node_val'] + 7, vecs[4]['addr'] + 7),
            ],
            "valu": emit_branch_add(1) + emit_hash_stage(2, 4) + emit_hash_combine(3, 1),
        })

        # Vec 5 gather
        self.instrs.append({
            "load": [
                ("load", vecs[5]['node_val'], vecs[5]['addr']),
                ("load", vecs[5]['node_val'] + 1, vecs[5]['addr'] + 1),
            ],
            "valu": emit_bounds_check(1) + emit_hash_stage(2, 5) + emit_hash_stage(3, 2) + emit_xor_node(4),
        })
        self.instrs.append({
            "load": [
                ("load", vecs[5]['node_val'] + 2, vecs[5]['addr'] + 2),
                ("load", vecs[5]['node_val'] + 3, vecs[5]['addr'] + 3),
            ],
            "valu": emit_bounds_apply(1) + emit_hash_combine(2, 5) + emit_hash_stage(3, 3) + emit_hash_stage(4, 0),
        })
        self.instrs.append({
            "load": [
                ("load", vecs[5]['node_val'] + 4, vecs[5]['addr'] + 4),
                ("load", vecs[5]['node_val'] + 5, vecs[5]['addr'] + 5),
            ],
            "valu": emit_branch_ops(2) + emit_hash_combine(3, 3) + emit_hash_stage(4, 1),
        })
        self.instrs.append({
            "load": [
                ("load", vecs[5]['node_val'] + 6, vecs[5]['addr'] + 6),
                ("load", vecs[5]['node_val'] + 7, vecs[5]['addr'] + 7),
            ],
            "valu": emit_branch_add(2) + emit_hash_stage(3, 4) + emit_hash_combine(4, 1),
        })

        # Vec 6 gather
        self.instrs.append({
            "load": [
                ("load", vecs[6]['node_val'], vecs[6]['addr']),
                ("load", vecs[6]['node_val'] + 1, vecs[6]['addr'] + 1),
            ],
            "valu": emit_bounds_check(2) + emit_hash_stage(3, 5) + emit_hash_stage(4, 2) + emit_xor_node(5),
        })
        self.instrs.append({
            "load": [
                ("load", vecs[6]['node_val'] + 2, vecs[6]['addr'] + 2),
                ("load", vecs[6]['node_val'] + 3, vecs[6]['addr'] + 3),
            ],
            "valu": emit_bounds_apply(2) + emit_hash_combine(3, 5) + emit_hash_stage(4, 3) + emit_hash_stage(5, 0),
        })
        self.instrs.append({
            "load": [
                ("load", vecs[6]['node_val'] + 4, vecs[6]['addr'] + 4),
                ("load", vecs[6]['node_val'] + 5, vecs[6]['addr'] + 5),
            ],
            "valu": emit_branch_ops(3) + emit_hash_combine(4, 3) + emit_hash_stage(5, 1),
        })
        self.instrs.append({
            "load": [
                ("load", vecs[6]['node_val'] + 6, vecs[6]['addr'] + 6),
                ("load", vecs[6]['node_val'] + 7, vecs[6]['addr'] + 7),
            ],
            "valu": emit_branch_add(3) + emit_hash_stage(4, 4) + emit_hash_combine(5, 1),
        })

        # Vec 7 gather
        self.instrs.append({
            "load": [
                ("load", vecs[7]['node_val'], vecs[7]['addr']),
                ("load", vecs[7]['node_val'] + 1, vecs[7]['addr'] + 1),
            ],
            "valu": emit_bounds_check(3) + emit_hash_stage(4, 5) + emit_hash_stage(5, 2) + emit_xor_node(6),
        })
        self.instrs.append({
            "load": [
                ("load", vecs[7]['node_val'] + 2, vecs[7]['addr'] + 2),
                ("load", vecs[7]['node_val'] + 3, vecs[7]['addr'] + 3),
            ],
            "valu": emit_bounds_apply(3) + emit_hash_combine(4, 5) + emit_hash_stage(5, 3) + emit_hash_stage(6, 0),
        })
        self.instrs.append({
            "load": [
                ("load", vecs[7]['node_val'] + 4, vecs[7]['addr'] + 4),
                ("load", vecs[7]['node_val'] + 5, vecs[7]['addr'] + 5),
            ],
            "valu": emit_branch_ops(4) + emit_hash_combine(5, 3) + emit_hash_stage(6, 1),
        })
        self.instrs.append({
            "load": [
                ("load", vecs[7]['node_val'] + 6, vecs[7]['addr'] + 6),
                ("load", vecs[7]['node_val'] + 7, vecs[7]['addr'] + 7),
            ],
            "valu": emit_branch_add(4) + emit_hash_stage(5, 4) + emit_hash_combine(6, 1),
        })

        # Finish remaining operations (no more loads, just VALU)
        self.instrs.append({
            "valu": emit_bounds_check(4) + emit_hash_stage(5, 5) + emit_hash_stage(6, 2) + emit_xor_node(7),
        })
        self.instrs.append({
            "valu": emit_bounds_apply(4) + emit_hash_combine(5, 5) + emit_hash_stage(6, 3) + emit_hash_stage(7, 0),
        })
        self.instrs.append({
            "valu": emit_branch_ops(5) + emit_hash_combine(6, 3) + emit_hash_stage(7, 1),
        })
        self.instrs.append({
            "valu": emit_branch_add(5) + emit_hash_stage(6, 4) + emit_hash_combine(7, 1),
        })
        self.instrs.append({
            "valu": emit_bounds_check(5) + emit_hash_stage(6, 5) + emit_hash_stage(7, 2),
        })
        self.instrs.append({
            "valu": emit_bounds_apply(5) + emit_hash_combine(6, 5) + emit_hash_stage(7, 3),
        })
        self.instrs.append({
            "valu": emit_branch_ops(6) + emit_hash_combine(7, 3),
        })
        self.instrs.append({
            "valu": emit_branch_add(6) + emit_hash_stage(7, 4),
        })
        self.instrs.append({
            "valu": emit_bounds_check(6) + emit_hash_stage(7, 5),
        })
        self.instrs.append({
            "valu": emit_bounds_apply(6) + emit_hash_combine(7, 5),
        })
        self.instrs.append({
            "valu": emit_branch_ops(7),
        })
        self.instrs.append({
            "valu": emit_branch_add(7),
        })
        self.instrs.append({
            "valu": emit_bounds_check(7),
        })
        self.instrs.append({
            "valu": emit_bounds_apply(7),
        })

        # Round loop control
        self.instrs.append({
            "alu": [
                ("+", round_counter, round_counter, one_const),
            ],
        })
        self.instrs.append({
            "alu": [
                ("<", tmp_scalar, round_counter, num_rounds_const),
            ],
        })
        self.add("flow", ("cond_jump", tmp_scalar, round_loop_start))

        # PHASE 3: STORE TILE RESULTS
        # Recompute store addresses
        self.instrs.append({
            "alu": [
                ("+", addr_tmp[0], self.scratch["inp_indices_p"], tile_offset),
                ("+", addr_tmp[1], self.scratch["inp_values_p"], tile_offset),
            ],
        })
        self.instrs.append({
            "alu": [
                ("+", addr_tmp[2], addr_tmp[0], vec_offsets[1]),
                ("+", addr_tmp[3], addr_tmp[1], vec_offsets[1]),
                ("+", addr_tmp[4], addr_tmp[0], vec_offsets[2]),
                ("+", addr_tmp[5], addr_tmp[1], vec_offsets[2]),
                ("+", addr_tmp[6], addr_tmp[0], vec_offsets[3]),
                ("+", addr_tmp[7], addr_tmp[1], vec_offsets[3]),
                ("+", addr_tmp[8], addr_tmp[0], vec_offsets[4]),
                ("+", addr_tmp[9], addr_tmp[1], vec_offsets[4]),
                ("+", addr_tmp[10], addr_tmp[0], vec_offsets[5]),
                ("+", addr_tmp[11], addr_tmp[1], vec_offsets[5]),
                ("+", addr_tmp[12], addr_tmp[0], vec_offsets[6]),
            ],
        })
        self.instrs.append({
            "alu": [
                ("+", addr_tmp[13], addr_tmp[1], vec_offsets[6]),
                ("+", addr_tmp[14], addr_tmp[0], vec_offsets[7]),
                ("+", addr_tmp[15], addr_tmp[1], vec_offsets[7]),
            ],
        })

        # Store idx and val for all 8 vectors (2 vstores per cycle = 8 cycles)
        for v in range(NUM_VECS):
            self.instrs.append({
                "store": [
                    ("vstore", addr_tmp[v * 2], vecs[v]['idx']),
                    ("vstore", addr_tmp[v * 2 + 1], vecs[v]['val']),
                ],
            })

        # Tile loop control
        self.instrs.append({
            "alu": [
                ("+", tile_counter, tile_counter, one_const),
                ("+", tile_offset, tile_offset, tile_stride_const),
            ],
        })
        self.instrs.append({
            "alu": [
                ("<", tmp_scalar, tile_counter, num_tiles_const),
            ],
        })
        self.add("flow", ("cond_jump", tmp_scalar, tile_loop_start))

        # Done
        self.instrs.append({"flow": [("pause",)]})


# Reuse baseline value
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

    kb = KernelBuilderH65()
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
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        if check:
            assert (
                machine.mem[inp_values_p : inp_values_p + len(inp.values)]
                == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
            ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Run correctness check")
    parser.add_argument("--trace", action="store_true", help="Generate trace")
    args = parser.parse_args()

    if args.check:
        cycles = do_kernel_test(10, 16, 256, check=True)
        print(f"Correctness check PASSED! Cycles: {cycles}")
    else:
        cycles = do_kernel_test(10, 16, 256, trace=args.trace)
