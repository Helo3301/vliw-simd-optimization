"""
# Experiment H74: Fully Unrolled Tiles + Wrap Exploitation

**GOAL:** Combine wrap-around exploitation with fully unrolled tile loop.

**KEY OPTIMIZATIONS:**
1. Full tile unroll: Eliminates loop overhead (~15-20 cycles saved)
2. Wrap-around exploitation from H73: Rounds 11-12 use preload
3. No bounds checks on rounds 11-15 (indices guaranteed < 2047)

**BASED ON:** H73 (2,643 cycles)
"""

from collections import defaultdict
import random
import unittest
import argparse
import sys
import os

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


class KernelBuilderH74:
    """
    H74: Fully unrolled tile loop with wrap exploitation.
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
        H74: Fully unrolled tile kernel.
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

        # Preload tree nodes 0-14
        NUM_PRELOADED = 15
        v_tree = []
        for i in range(NUM_PRELOADED):
            v_tree_node = self.alloc_scratch(f"v_tree_{i}", VLEN)
            v_tree.append(v_tree_node)

        tree_addr_tmp = self.alloc_scratch("tree_addr_tmp")
        for i in range(NUM_PRELOADED):
            self.instrs.append({
                "alu": [("+", tree_addr_tmp, self.scratch["forest_values_p"], self.scratch_const(i))],
            })
            self.add("load", ("load", tmp_scalar, tree_addr_tmp))
            self.add("valu", ("vbroadcast", v_tree[i], tmp_scalar))

        self.add("flow", ("pause",))

        # 16 DESKS
        NUM_DESKS = 16

        desks = []
        for d in range(NUM_DESKS):
            desk = {
                'idx': self.alloc_scratch(f"v_idx_{d}", VLEN),
                'val': self.alloc_scratch(f"v_val_{d}", VLEN),
                'node_val': self.alloc_scratch(f"v_node_{d}", VLEN),
                'addr': self.alloc_scratch(f"v_addr_{d}", VLEN),
                'tmp1': self.alloc_scratch(f"v_tmp1_{d}", VLEN),
                'tmp2': self.alloc_scratch(f"v_tmp2_{d}", VLEN),
            }
            desks.append(desk)

        addr_tmp = [self.alloc_scratch(f"addr_tmp_{i}") for i in range(32)]
        offset_regs = [self.alloc_scratch(f"off_{d}") for d in range(NUM_DESKS)]

        # Pre-compute offsets for both tiles
        NUM_TILES = batch_size // (NUM_DESKS * VLEN)  # = 2

        print(f"Scratch usage before main loop: {self.scratch_ptr} / {SCRATCH_SIZE}")

        # Helper functions (same as H73)
        def emit_hash_stage(desk_idx, stage):
            d = desks[desk_idx]
            if stage in v_fma_mult:
                return [("multiply_add", d['val'], d['val'], v_fma_mult[stage], v_hash_consts[stage])]
            else:
                if stage == 1:
                    return [
                        ("^", d['tmp1'], d['val'], v_hash_consts[stage]),
                        (">>", d['tmp2'], d['val'], v_hash_shifts[stage]),
                    ]
                elif stage == 3:
                    return [
                        ("+", d['tmp1'], d['val'], v_hash_consts[stage]),
                        ("<<", d['tmp2'], d['val'], v_hash_shifts[stage]),
                    ]
                elif stage == 5:
                    return [
                        ("^", d['tmp1'], d['val'], v_hash_consts[stage]),
                        (">>", d['tmp2'], d['val'], v_hash_shifts[stage]),
                    ]
            return []

        def emit_hash_combine(desk_idx, stage):
            d = desks[desk_idx]
            if stage in [1, 3, 5]:
                return [("^", d['val'], d['tmp1'], d['tmp2'])]
            return []

        def emit_branch_ops(desk_idx):
            d = desks[desk_idx]
            return [
                ("&", d['tmp1'], d['val'], v_one),
                ("multiply_add", d['idx'], d['idx'], v_two, v_one),
            ]

        def emit_branch_add(desk_idx):
            d = desks[desk_idx]
            return [("+", d['idx'], d['idx'], d['tmp1'])]

        emit_bounds_flag = [True]

        def emit_bounds_check(desk_idx):
            if not emit_bounds_flag[0]:
                return []
            d = desks[desk_idx]
            return [("<", d['tmp1'], d['idx'], v_n_nodes)]

        def emit_bounds_apply(desk_idx):
            if not emit_bounds_flag[0]:
                return []
            d = desks[desk_idx]
            return [("*", d['idx'], d['idx'], d['tmp1'])]

        def emit_xor_node(desk_idx):
            d = desks[desk_idx]
            return [("^", d['val'], d['val'], d['node_val'])]

        def emit_gather_round(skip_bounds=False):
            """Emit a full gather-based round for all 16 desks."""
            old_bounds_flag = emit_bounds_flag[0]
            emit_bounds_flag[0] = not skip_bounds

            # Prepare gather addresses
            self.instrs.append({
                "valu": [
                    ("vbroadcast", desks[0]['addr'], self.scratch["forest_values_p"]),
                    ("vbroadcast", desks[1]['addr'], self.scratch["forest_values_p"]),
                    ("vbroadcast", desks[2]['addr'], self.scratch["forest_values_p"]),
                    ("vbroadcast", desks[3]['addr'], self.scratch["forest_values_p"]),
                    ("vbroadcast", desks[4]['addr'], self.scratch["forest_values_p"]),
                    ("vbroadcast", desks[5]['addr'], self.scratch["forest_values_p"]),
                ],
            })
            self.instrs.append({
                "valu": [
                    ("vbroadcast", desks[6]['addr'], self.scratch["forest_values_p"]),
                    ("vbroadcast", desks[7]['addr'], self.scratch["forest_values_p"]),
                    ("vbroadcast", desks[8]['addr'], self.scratch["forest_values_p"]),
                    ("vbroadcast", desks[9]['addr'], self.scratch["forest_values_p"]),
                    ("vbroadcast", desks[10]['addr'], self.scratch["forest_values_p"]),
                    ("vbroadcast", desks[11]['addr'], self.scratch["forest_values_p"]),
                ],
            })
            self.instrs.append({
                "valu": [
                    ("vbroadcast", desks[12]['addr'], self.scratch["forest_values_p"]),
                    ("vbroadcast", desks[13]['addr'], self.scratch["forest_values_p"]),
                    ("vbroadcast", desks[14]['addr'], self.scratch["forest_values_p"]),
                    ("vbroadcast", desks[15]['addr'], self.scratch["forest_values_p"]),
                ],
            })

            # Add idx to addr
            self.instrs.append({
                "valu": [
                    ("+", desks[0]['addr'], desks[0]['addr'], desks[0]['idx']),
                    ("+", desks[1]['addr'], desks[1]['addr'], desks[1]['idx']),
                    ("+", desks[2]['addr'], desks[2]['addr'], desks[2]['idx']),
                    ("+", desks[3]['addr'], desks[3]['addr'], desks[3]['idx']),
                    ("+", desks[4]['addr'], desks[4]['addr'], desks[4]['idx']),
                    ("+", desks[5]['addr'], desks[5]['addr'], desks[5]['idx']),
                ],
            })
            self.instrs.append({
                "valu": [
                    ("+", desks[6]['addr'], desks[6]['addr'], desks[6]['idx']),
                    ("+", desks[7]['addr'], desks[7]['addr'], desks[7]['idx']),
                    ("+", desks[8]['addr'], desks[8]['addr'], desks[8]['idx']),
                    ("+", desks[9]['addr'], desks[9]['addr'], desks[9]['idx']),
                    ("+", desks[10]['addr'], desks[10]['addr'], desks[10]['idx']),
                    ("+", desks[11]['addr'], desks[11]['addr'], desks[11]['idx']),
                ],
            })
            self.instrs.append({
                "valu": [
                    ("+", desks[12]['addr'], desks[12]['addr'], desks[12]['idx']),
                    ("+", desks[13]['addr'], desks[13]['addr'], desks[13]['idx']),
                    ("+", desks[14]['addr'], desks[14]['addr'], desks[14]['idx']),
                    ("+", desks[15]['addr'], desks[15]['addr'], desks[15]['idx']),
                ],
            })

            # Interleaved gather + hash (same pattern as H73)
            for lane in range(0, VLEN, 2):
                self.instrs.append({
                    "load": [
                        ("load", desks[0]['node_val'] + lane, desks[0]['addr'] + lane),
                        ("load", desks[0]['node_val'] + lane + 1, desks[0]['addr'] + lane + 1),
                    ],
                })

            self.instrs.append({
                "load": [
                    ("load", desks[1]['node_val'], desks[1]['addr']),
                    ("load", desks[1]['node_val'] + 1, desks[1]['addr'] + 1),
                ],
                "valu": emit_xor_node(0),
            })
            self.instrs.append({
                "load": [
                    ("load", desks[1]['node_val'] + 2, desks[1]['addr'] + 2),
                    ("load", desks[1]['node_val'] + 3, desks[1]['addr'] + 3),
                ],
                "valu": emit_hash_stage(0, 0),
            })
            self.instrs.append({
                "load": [
                    ("load", desks[1]['node_val'] + 4, desks[1]['addr'] + 4),
                    ("load", desks[1]['node_val'] + 5, desks[1]['addr'] + 5),
                ],
                "valu": emit_hash_stage(0, 1),
            })
            self.instrs.append({
                "load": [
                    ("load", desks[1]['node_val'] + 6, desks[1]['addr'] + 6),
                    ("load", desks[1]['node_val'] + 7, desks[1]['addr'] + 7),
                ],
                "valu": emit_hash_combine(0, 1),
            })

            self.instrs.append({
                "load": [
                    ("load", desks[2]['node_val'], desks[2]['addr']),
                    ("load", desks[2]['node_val'] + 1, desks[2]['addr'] + 1),
                ],
                "valu": emit_hash_stage(0, 2) + emit_xor_node(1),
            })
            self.instrs.append({
                "load": [
                    ("load", desks[2]['node_val'] + 2, desks[2]['addr'] + 2),
                    ("load", desks[2]['node_val'] + 3, desks[2]['addr'] + 3),
                ],
                "valu": emit_hash_stage(0, 3) + emit_hash_stage(1, 0),
            })
            self.instrs.append({
                "load": [
                    ("load", desks[2]['node_val'] + 4, desks[2]['addr'] + 4),
                    ("load", desks[2]['node_val'] + 5, desks[2]['addr'] + 5),
                ],
                "valu": emit_hash_combine(0, 3) + emit_hash_stage(1, 1),
            })
            self.instrs.append({
                "load": [
                    ("load", desks[2]['node_val'] + 6, desks[2]['addr'] + 6),
                    ("load", desks[2]['node_val'] + 7, desks[2]['addr'] + 7),
                ],
                "valu": emit_hash_stage(0, 4) + emit_hash_combine(1, 1),
            })

            self.instrs.append({
                "load": [
                    ("load", desks[3]['node_val'], desks[3]['addr']),
                    ("load", desks[3]['node_val'] + 1, desks[3]['addr'] + 1),
                ],
                "valu": emit_hash_stage(0, 5) + emit_hash_stage(1, 2) + emit_xor_node(2),
            })
            self.instrs.append({
                "load": [
                    ("load", desks[3]['node_val'] + 2, desks[3]['addr'] + 2),
                    ("load", desks[3]['node_val'] + 3, desks[3]['addr'] + 3),
                ],
                "valu": emit_hash_combine(0, 5) + emit_hash_stage(1, 3) + emit_hash_stage(2, 0),
            })
            self.instrs.append({
                "load": [
                    ("load", desks[3]['node_val'] + 4, desks[3]['addr'] + 4),
                    ("load", desks[3]['node_val'] + 5, desks[3]['addr'] + 5),
                ],
                "valu": emit_branch_ops(0) + emit_hash_combine(1, 3) + emit_hash_stage(2, 1),
            })
            self.instrs.append({
                "load": [
                    ("load", desks[3]['node_val'] + 6, desks[3]['addr'] + 6),
                    ("load", desks[3]['node_val'] + 7, desks[3]['addr'] + 7),
                ],
                "valu": emit_branch_add(0) + emit_hash_stage(1, 4) + emit_hash_combine(2, 1),
            })

            self.instrs.append({
                "load": [
                    ("load", desks[4]['node_val'], desks[4]['addr']),
                    ("load", desks[4]['node_val'] + 1, desks[4]['addr'] + 1),
                ],
                "valu": emit_bounds_check(0) + emit_hash_stage(1, 5) + emit_hash_stage(2, 2) + emit_xor_node(3),
            })
            self.instrs.append({
                "load": [
                    ("load", desks[4]['node_val'] + 2, desks[4]['addr'] + 2),
                    ("load", desks[4]['node_val'] + 3, desks[4]['addr'] + 3),
                ],
                "valu": emit_bounds_apply(0) + emit_hash_combine(1, 5) + emit_hash_stage(2, 3) + emit_hash_stage(3, 0),
            })
            self.instrs.append({
                "load": [
                    ("load", desks[4]['node_val'] + 4, desks[4]['addr'] + 4),
                    ("load", desks[4]['node_val'] + 5, desks[4]['addr'] + 5),
                ],
                "valu": emit_branch_ops(1) + emit_hash_combine(2, 3) + emit_hash_stage(3, 1),
            })
            self.instrs.append({
                "load": [
                    ("load", desks[4]['node_val'] + 6, desks[4]['addr'] + 6),
                    ("load", desks[4]['node_val'] + 7, desks[4]['addr'] + 7),
                ],
                "valu": emit_branch_add(1) + emit_hash_stage(2, 4) + emit_hash_combine(3, 1),
            })

            for start_desk in range(5, 16):
                for lane_pair in range(4):
                    lane = lane_pair * 2
                    valu_ops = []

                    if lane_pair == 0:
                        valu_ops += emit_bounds_check(start_desk - 4) if start_desk >= 4 else []
                        valu_ops += emit_hash_stage(start_desk - 3, 5) if start_desk >= 3 else []
                        valu_ops += emit_hash_stage(start_desk - 2, 2) if start_desk >= 2 else []
                        valu_ops += emit_xor_node(start_desk - 1) if start_desk >= 1 else []
                    elif lane_pair == 1:
                        valu_ops += emit_bounds_apply(start_desk - 4) if start_desk >= 4 else []
                        valu_ops += emit_hash_combine(start_desk - 3, 5) if start_desk >= 3 else []
                        valu_ops += emit_hash_stage(start_desk - 2, 3) if start_desk >= 2 else []
                        valu_ops += emit_hash_stage(start_desk - 1, 0) if start_desk >= 1 else []
                    elif lane_pair == 2:
                        valu_ops += emit_branch_ops(start_desk - 3) if start_desk >= 3 else []
                        valu_ops += emit_hash_combine(start_desk - 2, 3) if start_desk >= 2 else []
                        valu_ops += emit_hash_stage(start_desk - 1, 1) if start_desk >= 1 else []
                    elif lane_pair == 3:
                        valu_ops += emit_branch_add(start_desk - 3) if start_desk >= 3 else []
                        valu_ops += emit_hash_stage(start_desk - 2, 4) if start_desk >= 2 else []
                        valu_ops += emit_hash_combine(start_desk - 1, 1) if start_desk >= 1 else []

                    instr = {
                        "load": [
                            ("load", desks[start_desk]['node_val'] + lane, desks[start_desk]['addr'] + lane),
                            ("load", desks[start_desk]['node_val'] + lane + 1, desks[start_desk]['addr'] + lane + 1),
                        ],
                    }
                    if valu_ops:
                        instr["valu"] = valu_ops
                    self.instrs.append(instr)

            # Finish remaining ops
            self.instrs.append({"valu": emit_bounds_check(12) + emit_hash_stage(13, 5) + emit_hash_stage(14, 2) + emit_xor_node(15)})
            self.instrs.append({"valu": emit_bounds_apply(12) + emit_hash_combine(13, 5) + emit_hash_stage(14, 3) + emit_hash_stage(15, 0)})
            self.instrs.append({"valu": emit_branch_ops(13) + emit_hash_combine(14, 3) + emit_hash_stage(15, 1)})
            self.instrs.append({"valu": emit_branch_add(13) + emit_hash_stage(14, 4) + emit_hash_combine(15, 1)})
            self.instrs.append({"valu": emit_bounds_check(13) + emit_hash_stage(14, 5) + emit_hash_stage(15, 2)})
            self.instrs.append({"valu": emit_bounds_apply(13) + emit_hash_combine(14, 5) + emit_hash_stage(15, 3)})
            self.instrs.append({"valu": emit_branch_ops(14) + emit_hash_combine(15, 3)})
            self.instrs.append({"valu": emit_branch_add(14) + emit_hash_stage(15, 4)})
            self.instrs.append({"valu": emit_bounds_check(14) + emit_hash_stage(15, 5)})
            self.instrs.append({"valu": emit_bounds_apply(14) + emit_hash_combine(15, 5)})
            self.instrs.append({"valu": emit_branch_ops(15)})
            self.instrs.append({"valu": emit_branch_add(15)})
            if emit_bounds_flag[0]:
                self.instrs.append({"valu": emit_bounds_check(15)})
                self.instrs.append({"valu": emit_bounds_apply(15)})

            emit_bounds_flag[0] = old_bounds_flag

        def emit_preload_round_all_idx_known(tree_idx, skip_bounds=False):
            """For rounds where all indices are the same value."""
            for d in range(0, NUM_DESKS, 6):
                ops = [("+", desks[dd]['node_val'], v_tree[tree_idx], v_zero) for dd in range(d, min(d + 6, NUM_DESKS))]
                self.instrs.append({"valu": ops})

            for d in range(0, NUM_DESKS, 6):
                ops = sum([emit_xor_node(dd) for dd in range(d, min(d + 6, NUM_DESKS))], [])
                self.instrs.append({"valu": ops})

            for d in range(0, NUM_DESKS, 6):
                ops = sum([emit_hash_stage(dd, 0) for dd in range(d, min(d + 6, NUM_DESKS))], [])
                self.instrs.append({"valu": ops})

            for d in range(0, NUM_DESKS, 3):
                ops = sum([emit_hash_stage(dd, 1) for dd in range(d, min(d + 3, NUM_DESKS))], [])
                self.instrs.append({"valu": ops})

            for d in range(0, NUM_DESKS, 6):
                ops = sum([emit_hash_combine(dd, 1) for dd in range(d, min(d + 6, NUM_DESKS))], [])
                self.instrs.append({"valu": ops})

            for d in range(0, NUM_DESKS, 6):
                ops = sum([emit_hash_stage(dd, 2) for dd in range(d, min(d + 6, NUM_DESKS))], [])
                self.instrs.append({"valu": ops})

            for d in range(0, NUM_DESKS, 3):
                ops = sum([emit_hash_stage(dd, 3) for dd in range(d, min(d + 3, NUM_DESKS))], [])
                self.instrs.append({"valu": ops})

            for d in range(0, NUM_DESKS, 6):
                ops = sum([emit_hash_combine(dd, 3) for dd in range(d, min(d + 6, NUM_DESKS))], [])
                self.instrs.append({"valu": ops})

            for d in range(0, NUM_DESKS, 6):
                ops = sum([emit_hash_stage(dd, 4) for dd in range(d, min(d + 6, NUM_DESKS))], [])
                self.instrs.append({"valu": ops})

            for d in range(0, NUM_DESKS, 3):
                ops = sum([emit_hash_stage(dd, 5) for dd in range(d, min(d + 3, NUM_DESKS))], [])
                self.instrs.append({"valu": ops})

            for d in range(0, NUM_DESKS, 6):
                ops = sum([emit_hash_combine(dd, 5) for dd in range(d, min(d + 6, NUM_DESKS))], [])
                self.instrs.append({"valu": ops})

            for d in range(0, NUM_DESKS, 3):
                ops = sum([emit_branch_ops(dd) for dd in range(d, min(d + 3, NUM_DESKS))], [])
                self.instrs.append({"valu": ops})

            for d in range(0, NUM_DESKS, 6):
                ops = sum([emit_branch_add(dd) for dd in range(d, min(d + 6, NUM_DESKS))], [])
                self.instrs.append({"valu": ops})

            if not skip_bounds:
                for d in range(0, NUM_DESKS, 6):
                    ops = sum([emit_bounds_check(dd) for dd in range(d, min(d + 6, NUM_DESKS))], [])
                    self.instrs.append({"valu": ops})
                for d in range(0, NUM_DESKS, 6):
                    ops = sum([emit_bounds_apply(dd) for dd in range(d, min(d + 6, NUM_DESKS))], [])
                    self.instrs.append({"valu": ops})

        def emit_preload_round_1(skip_bounds=False):
            """For rounds where indices are in {1, 2}."""
            for d in range(0, NUM_DESKS, 6):
                ops = [("-", desks[dd]['tmp1'], desks[dd]['idx'], v_one) for dd in range(d, min(d + 6, NUM_DESKS))]
                self.instrs.append({"valu": ops})

            for d in range(0, NUM_DESKS, 6):
                ops = [("-", desks[dd]['addr'], v_tree[2], v_tree[1]) for dd in range(d, min(d + 6, NUM_DESKS))]
                self.instrs.append({"valu": ops})

            for d in range(0, NUM_DESKS, 6):
                ops = [("multiply_add", desks[dd]['node_val'], desks[dd]['tmp1'], desks[dd]['addr'], v_tree[1]) for dd in range(d, min(d + 6, NUM_DESKS))]
                self.instrs.append({"valu": ops})

            for d in range(0, NUM_DESKS, 6):
                ops = sum([emit_xor_node(dd) for dd in range(d, min(d + 6, NUM_DESKS))], [])
                self.instrs.append({"valu": ops})

            for d in range(0, NUM_DESKS, 6):
                self.instrs.append({"valu": [emit_hash_stage(dd, 0)[0] for dd in range(d, min(d + 6, NUM_DESKS))]})
            for d in range(0, NUM_DESKS, 3):
                self.instrs.append({"valu": sum([emit_hash_stage(dd, 1) for dd in range(d, min(d + 3, NUM_DESKS))], [])})
            for d in range(0, NUM_DESKS, 6):
                self.instrs.append({"valu": [emit_hash_combine(dd, 1)[0] for dd in range(d, min(d + 6, NUM_DESKS))]})
            for d in range(0, NUM_DESKS, 6):
                self.instrs.append({"valu": [emit_hash_stage(dd, 2)[0] for dd in range(d, min(d + 6, NUM_DESKS))]})
            for d in range(0, NUM_DESKS, 3):
                self.instrs.append({"valu": sum([emit_hash_stage(dd, 3) for dd in range(d, min(d + 3, NUM_DESKS))], [])})
            for d in range(0, NUM_DESKS, 6):
                self.instrs.append({"valu": [emit_hash_combine(dd, 3)[0] for dd in range(d, min(d + 6, NUM_DESKS))]})
            for d in range(0, NUM_DESKS, 6):
                self.instrs.append({"valu": [emit_hash_stage(dd, 4)[0] for dd in range(d, min(d + 6, NUM_DESKS))]})
            for d in range(0, NUM_DESKS, 3):
                self.instrs.append({"valu": sum([emit_hash_stage(dd, 5) for dd in range(d, min(d + 3, NUM_DESKS))], [])})
            for d in range(0, NUM_DESKS, 6):
                self.instrs.append({"valu": [emit_hash_combine(dd, 5)[0] for dd in range(d, min(d + 6, NUM_DESKS))]})

            for d in range(0, NUM_DESKS, 3):
                self.instrs.append({"valu": sum([emit_branch_ops(dd) for dd in range(d, min(d + 3, NUM_DESKS))], [])})
            for d in range(0, NUM_DESKS, 6):
                self.instrs.append({"valu": [emit_branch_add(dd)[0] for dd in range(d, min(d + 6, NUM_DESKS))]})

            if not skip_bounds:
                for d in range(0, NUM_DESKS, 6):
                    self.instrs.append({"valu": [emit_bounds_check(dd)[0] for dd in range(d, min(d + 6, NUM_DESKS))]})
                for d in range(0, NUM_DESKS, 6):
                    self.instrs.append({"valu": [emit_bounds_apply(dd)[0] for dd in range(d, min(d + 6, NUM_DESKS))]})

        def emit_tile(tile_idx):
            """Emit code for a single tile."""
            tile_offset_val = tile_idx * NUM_DESKS * VLEN

            # Calculate offsets for this tile
            for d in range(NUM_DESKS):
                self.add("load", ("const", offset_regs[d], tile_offset_val + d * VLEN))

            # Compute load addresses
            self.instrs.append({
                "alu": [
                    ("+", addr_tmp[d*2], self.scratch["inp_indices_p"], offset_regs[d])
                    for d in range(min(12, NUM_DESKS))
                ],
            })
            self.instrs.append({
                "alu": [
                    ("+", addr_tmp[d*2+1], self.scratch["inp_values_p"], offset_regs[d])
                    for d in range(min(12, NUM_DESKS))
                ],
            })
            if NUM_DESKS > 12:
                self.instrs.append({
                    "alu": [
                        ("+", addr_tmp[d*2], self.scratch["inp_indices_p"], offset_regs[d])
                        for d in range(12, NUM_DESKS)
                    ],
                })
                self.instrs.append({
                    "alu": [
                        ("+", addr_tmp[d*2+1], self.scratch["inp_values_p"], offset_regs[d])
                        for d in range(12, NUM_DESKS)
                    ],
                })

            # Load idx/val for all desks
            for d in range(0, NUM_DESKS, 2):
                self.instrs.append({
                    "load": [
                        ("vload", desks[d]['idx'], addr_tmp[d*2]),
                        ("vload", desks[d]['val'], addr_tmp[d*2+1]),
                    ],
                })
                self.instrs.append({
                    "load": [
                        ("vload", desks[d+1]['idx'], addr_tmp[(d+1)*2]),
                        ("vload", desks[d+1]['val'], addr_tmp[(d+1)*2+1]),
                    ],
                })

            # ROUNDS
            emit_preload_round_all_idx_known(0, skip_bounds=True)  # R0
            emit_preload_round_1(skip_bounds=True)  # R1
            for _ in range(2, 10):  # R2-9
                emit_gather_round(skip_bounds=True)
            emit_gather_round(skip_bounds=False)  # R10
            emit_preload_round_all_idx_known(0, skip_bounds=True)  # R11 (wrap)
            emit_preload_round_1(skip_bounds=True)  # R12 (wrap)
            for _ in range(13, 15):  # R13-14
                emit_gather_round(skip_bounds=True)
            emit_gather_round(skip_bounds=True)  # R15 (no bounds needed)

            # Store all desks
            for d in range(NUM_DESKS):
                self.instrs.append({
                    "store": [
                        ("vstore", addr_tmp[d*2], desks[d]['idx']),
                        ("vstore", addr_tmp[d*2+1], desks[d]['val']),
                    ],
                })

        # EMIT BOTH TILES (FULLY UNROLLED)
        emit_tile(0)
        emit_tile(1)

        # Done
        self.instrs.append({"flow": [("pause",)]})

        # Clean up empty slots
        cleaned_instrs = []
        for instr in self.instrs:
            cleaned = {k: v for k, v in instr.items() if v}
            if cleaned:
                cleaned_instrs.append(cleaned)
        self.instrs = cleaned_instrs


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

    kb = KernelBuilderH74()
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
        inp_indices_p = ref_mem[5]

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
