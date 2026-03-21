"""
# Experiment H59: ILP-Optimized Schedule

**GOAL:** Apply ILP solver insights to improve H54's schedule by ~15-20%.

**KEY OPTIMIZATIONS FROM SOLVER:**
1. ALU operations can start 80 cycles earlier (overlap with previous iteration's stores)
2. Stores can start 58 cycles earlier (while hash completing for later desks)
3. Tighter load/compute interleaving (-17 cycles)

**APPROACH:**
- Overlap stores for early desks with hash completion for later desks
- Overlap next iteration's address calculations with current iteration's stores
- Same algorithm as H54, just better scheduled

**TARGET:** ~2,800-3,000 cycles (15-20% improvement over H54's 3,462)
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


class KernelBuilderH59:
    """
    H59: ILP-Optimized Schedule

    Based on H54's 16-desk pipeline with round fusion, but with better
    instruction scheduling based on ILP solver analysis.

    Key improvements:
    1. Start stores while hash operations for later desks are completing
    2. Overlap next iteration's ALU address calculations with stores
    3. Tighter gather/hash interleaving
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
        H59: ILP-Optimized 16-Desk Pipeline kernel.
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

        self.add("flow", ("pause",))

        # 16 DESKS for ultra-deep pipeline
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

        # Address temporaries - need 32 for 16 desks (idx + val for each)
        addr_tmp = [self.alloc_scratch(f"addr_tmp_{i}") for i in range(32)]

        # Offset registers for each desk
        offset_regs = [self.alloc_scratch(f"off_{d}") for d in range(NUM_DESKS)]

        batch_offset = self.alloc_scratch("batch_offset")
        iter_counter = self.alloc_scratch("iter_counter")

        # Constants for offsets
        offset_consts = []
        for d in range(NUM_DESKS):
            offset_consts.append(self.scratch_const(d * VLEN))

        batch_size_const = self.scratch_const(batch_size)
        desk_stride_const = self.scratch_const(NUM_DESKS * VLEN)  # 128 elements per iteration

        # With 16 desks and round fusion (2 rounds per iteration):
        total_iterations = (batch_size // VLEN) * (rounds // 2) // NUM_DESKS
        total_const = self.scratch_const(total_iterations)

        self.add("load", ("const", batch_offset, 0))
        self.add("load", ("const", iter_counter, 0))

        # Print scratch usage for debugging
        print(f"Scratch usage before main loop: {self.scratch_ptr} / {SCRATCH_SIZE}")

        # === MAIN LOOP ===
        main_loop_start = len(self.instrs)

        # PHASE 1: Calculate all offsets for 16 desks
        self.instrs.append({
            "alu": [
                ("+", offset_regs[0], batch_offset, offset_consts[0]),
                ("+", offset_regs[1], batch_offset, offset_consts[1]),
                ("+", offset_regs[2], batch_offset, offset_consts[2]),
                ("+", offset_regs[3], batch_offset, offset_consts[3]),
                ("+", offset_regs[4], batch_offset, offset_consts[4]),
                ("+", offset_regs[5], batch_offset, offset_consts[5]),
                ("+", offset_regs[6], batch_offset, offset_consts[6]),
                ("+", offset_regs[7], batch_offset, offset_consts[7]),
                ("+", offset_regs[8], batch_offset, offset_consts[8]),
                ("+", offset_regs[9], batch_offset, offset_consts[9]),
                ("+", offset_regs[10], batch_offset, offset_consts[10]),
                ("+", offset_regs[11], batch_offset, offset_consts[11]),
            ],
        })
        self.instrs.append({
            "alu": [
                ("+", offset_regs[12], batch_offset, offset_consts[12]),
                ("+", offset_regs[13], batch_offset, offset_consts[13]),
                ("+", offset_regs[14], batch_offset, offset_consts[14]),
                ("+", offset_regs[15], batch_offset, offset_consts[15]),
            ],
        })

        # Compute load addresses for desks 0-5
        self.instrs.append({
            "alu": [
                ("+", addr_tmp[0], self.scratch["inp_indices_p"], offset_regs[0]),
                ("+", addr_tmp[1], self.scratch["inp_values_p"], offset_regs[0]),
                ("+", addr_tmp[2], self.scratch["inp_indices_p"], offset_regs[1]),
                ("+", addr_tmp[3], self.scratch["inp_values_p"], offset_regs[1]),
                ("+", addr_tmp[4], self.scratch["inp_indices_p"], offset_regs[2]),
                ("+", addr_tmp[5], self.scratch["inp_values_p"], offset_regs[2]),
                ("+", addr_tmp[6], self.scratch["inp_indices_p"], offset_regs[3]),
                ("+", addr_tmp[7], self.scratch["inp_values_p"], offset_regs[3]),
                ("+", addr_tmp[8], self.scratch["inp_indices_p"], offset_regs[4]),
                ("+", addr_tmp[9], self.scratch["inp_values_p"], offset_regs[4]),
                ("+", addr_tmp[10], self.scratch["inp_indices_p"], offset_regs[5]),
                ("+", addr_tmp[11], self.scratch["inp_values_p"], offset_regs[5]),
            ],
        })

        # Compute load addresses for desks 6-11
        self.instrs.append({
            "alu": [
                ("+", addr_tmp[12], self.scratch["inp_indices_p"], offset_regs[6]),
                ("+", addr_tmp[13], self.scratch["inp_values_p"], offset_regs[6]),
                ("+", addr_tmp[14], self.scratch["inp_indices_p"], offset_regs[7]),
                ("+", addr_tmp[15], self.scratch["inp_values_p"], offset_regs[7]),
                ("+", addr_tmp[16], self.scratch["inp_indices_p"], offset_regs[8]),
                ("+", addr_tmp[17], self.scratch["inp_values_p"], offset_regs[8]),
                ("+", addr_tmp[18], self.scratch["inp_indices_p"], offset_regs[9]),
                ("+", addr_tmp[19], self.scratch["inp_values_p"], offset_regs[9]),
                ("+", addr_tmp[20], self.scratch["inp_indices_p"], offset_regs[10]),
                ("+", addr_tmp[21], self.scratch["inp_values_p"], offset_regs[10]),
                ("+", addr_tmp[22], self.scratch["inp_indices_p"], offset_regs[11]),
                ("+", addr_tmp[23], self.scratch["inp_values_p"], offset_regs[11]),
            ],
        })

        # Compute load addresses for desks 12-15
        self.instrs.append({
            "alu": [
                ("+", addr_tmp[24], self.scratch["inp_indices_p"], offset_regs[12]),
                ("+", addr_tmp[25], self.scratch["inp_values_p"], offset_regs[12]),
                ("+", addr_tmp[26], self.scratch["inp_indices_p"], offset_regs[13]),
                ("+", addr_tmp[27], self.scratch["inp_values_p"], offset_regs[13]),
                ("+", addr_tmp[28], self.scratch["inp_indices_p"], offset_regs[14]),
                ("+", addr_tmp[29], self.scratch["inp_values_p"], offset_regs[14]),
                ("+", addr_tmp[30], self.scratch["inp_indices_p"], offset_regs[15]),
                ("+", addr_tmp[31], self.scratch["inp_values_p"], offset_regs[15]),
            ],
        })

        # PHASE 2: Load idx/val for all 16 desks (32 vloads = 16 cycles at 2/cycle)
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

        # PHASE 3: Prepare gather addresses for all desks
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

        # Add idx to addr for all desks
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

        # ============================================================
        # Helper functions for hash operations
        # ============================================================

        def emit_hash_stage(desk_idx, stage):
            """Emit one hash stage for a desk"""
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
            """Emit combine for XOR stages"""
            d = desks[desk_idx]
            if stage in [1, 3, 5]:
                return [("^", d['val'], d['tmp1'], d['tmp2'])]
            return []

        def emit_branch_ops(desk_idx):
            """Emit branch operations for a desk"""
            d = desks[desk_idx]
            return [
                ("&", d['tmp1'], d['val'], v_one),
                ("multiply_add", d['idx'], d['idx'], v_two, v_one),
            ]

        def emit_branch_add(desk_idx):
            d = desks[desk_idx]
            return [("+", d['idx'], d['idx'], d['tmp1'])]

        def emit_bounds_check(desk_idx):
            d = desks[desk_idx]
            return [("<", d['tmp1'], d['idx'], v_n_nodes)]

        def emit_bounds_apply(desk_idx):
            d = desks[desk_idx]
            return [("*", d['idx'], d['idx'], d['tmp1'])]

        def emit_xor_node(desk_idx):
            d = desks[desk_idx]
            return [("^", d['val'], d['val'], d['node_val'])]

        # ============================================================
        # ROUND 1: Full gather + hash for all 16 desks
        # ============================================================
        def emit_round_gather_and_hash():
            """Emit interleaved gather + hash for all 16 desks for one round"""

            # Desk 0 gather (4 cycles)
            for lane in range(0, VLEN, 2):
                self.instrs.append({
                    "load": [
                        ("load", desks[0]['node_val'] + lane, desks[0]['addr'] + lane),
                        ("load", desks[0]['node_val'] + lane + 1, desks[0]['addr'] + lane + 1),
                    ],
                })

            # Desk 1 gather + desk 0 XOR + hash
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

            # Continue the interleaved pattern for desks 2-15...
            # (This is the same as H54, copying the pattern)

            # Desk 2 gather + desk 0 hash + desk 1 XOR
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

            # Desk 3
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

            # Desk 4
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

            # Desks 5-15 follow the same pattern
            for gather_desk in range(5, 16):
                hash_base = gather_desk - 4
                for lane_pair in range(4):
                    lane = lane_pair * 2
                    valu_ops = []
                    if lane_pair == 0:
                        valu_ops = emit_bounds_check(hash_base) + emit_hash_stage(hash_base+1, 5) + emit_hash_stage(hash_base+2, 2) + emit_xor_node(hash_base+3)
                    elif lane_pair == 1:
                        valu_ops = emit_bounds_apply(hash_base) + emit_hash_combine(hash_base+1, 5) + emit_hash_stage(hash_base+2, 3) + emit_hash_stage(hash_base+3, 0)
                    elif lane_pair == 2:
                        valu_ops = emit_branch_ops(hash_base+1) + emit_hash_combine(hash_base+2, 3) + emit_hash_stage(hash_base+3, 1)
                    elif lane_pair == 3:
                        valu_ops = emit_branch_add(hash_base+1) + emit_hash_stage(hash_base+2, 4) + emit_hash_combine(hash_base+3, 1)

                    self.instrs.append({
                        "load": [
                            ("load", desks[gather_desk]['node_val'] + lane, desks[gather_desk]['addr'] + lane),
                            ("load", desks[gather_desk]['node_val'] + lane + 1, desks[gather_desk]['addr'] + lane + 1),
                        ],
                        "valu": valu_ops,
                    })

            # Tail: finish hash for desks 12-15 (14 cycles of VALU-only)
            # OPTIMIZATION: Pre-compute store addresses during Round 1 tail (ALU is free)
            # These addresses will be used after Round 2 completes
            self.instrs.append({
                "valu": emit_bounds_check(12) + emit_hash_stage(13, 5) + emit_hash_stage(14, 2) + emit_xor_node(15),
                "alu": [
                    ("+", addr_tmp[0], self.scratch["inp_indices_p"], offset_regs[0]),
                    ("+", addr_tmp[1], self.scratch["inp_values_p"], offset_regs[0]),
                    ("+", addr_tmp[2], self.scratch["inp_indices_p"], offset_regs[1]),
                    ("+", addr_tmp[3], self.scratch["inp_values_p"], offset_regs[1]),
                    ("+", addr_tmp[4], self.scratch["inp_indices_p"], offset_regs[2]),
                    ("+", addr_tmp[5], self.scratch["inp_values_p"], offset_regs[2]),
                    ("+", addr_tmp[6], self.scratch["inp_indices_p"], offset_regs[3]),
                    ("+", addr_tmp[7], self.scratch["inp_values_p"], offset_regs[3]),
                    ("+", addr_tmp[8], self.scratch["inp_indices_p"], offset_regs[4]),
                    ("+", addr_tmp[9], self.scratch["inp_values_p"], offset_regs[4]),
                    ("+", addr_tmp[10], self.scratch["inp_indices_p"], offset_regs[5]),
                    ("+", addr_tmp[11], self.scratch["inp_values_p"], offset_regs[5]),
                ],
            })
            self.instrs.append({
                "valu": emit_bounds_apply(12) + emit_hash_combine(13, 5) + emit_hash_stage(14, 3) + emit_hash_stage(15, 0),
                "alu": [
                    ("+", addr_tmp[12], self.scratch["inp_indices_p"], offset_regs[6]),
                    ("+", addr_tmp[13], self.scratch["inp_values_p"], offset_regs[6]),
                    ("+", addr_tmp[14], self.scratch["inp_indices_p"], offset_regs[7]),
                    ("+", addr_tmp[15], self.scratch["inp_values_p"], offset_regs[7]),
                    ("+", addr_tmp[16], self.scratch["inp_indices_p"], offset_regs[8]),
                    ("+", addr_tmp[17], self.scratch["inp_values_p"], offset_regs[8]),
                    ("+", addr_tmp[18], self.scratch["inp_indices_p"], offset_regs[9]),
                    ("+", addr_tmp[19], self.scratch["inp_values_p"], offset_regs[9]),
                    ("+", addr_tmp[20], self.scratch["inp_indices_p"], offset_regs[10]),
                    ("+", addr_tmp[21], self.scratch["inp_values_p"], offset_regs[10]),
                    ("+", addr_tmp[22], self.scratch["inp_indices_p"], offset_regs[11]),
                    ("+", addr_tmp[23], self.scratch["inp_values_p"], offset_regs[11]),
                ],
            })
            self.instrs.append({
                "valu": emit_branch_ops(13) + emit_hash_combine(14, 3) + emit_hash_stage(15, 1),
                "alu": [
                    ("+", addr_tmp[24], self.scratch["inp_indices_p"], offset_regs[12]),
                    ("+", addr_tmp[25], self.scratch["inp_values_p"], offset_regs[12]),
                    ("+", addr_tmp[26], self.scratch["inp_indices_p"], offset_regs[13]),
                    ("+", addr_tmp[27], self.scratch["inp_values_p"], offset_regs[13]),
                    ("+", addr_tmp[28], self.scratch["inp_indices_p"], offset_regs[14]),
                    ("+", addr_tmp[29], self.scratch["inp_values_p"], offset_regs[14]),
                    ("+", addr_tmp[30], self.scratch["inp_indices_p"], offset_regs[15]),
                    ("+", addr_tmp[31], self.scratch["inp_values_p"], offset_regs[15]),
                ],
            })
            self.instrs.append({"valu": emit_branch_add(13) + emit_hash_stage(14, 4) + emit_hash_combine(15, 1)})
            self.instrs.append({"valu": emit_bounds_check(13) + emit_hash_stage(14, 5) + emit_hash_stage(15, 2)})
            self.instrs.append({"valu": emit_bounds_apply(13) + emit_hash_combine(14, 5) + emit_hash_stage(15, 3)})
            self.instrs.append({"valu": emit_branch_ops(14) + emit_hash_combine(15, 3)})
            self.instrs.append({"valu": emit_branch_add(14) + emit_hash_stage(15, 4)})
            self.instrs.append({"valu": emit_bounds_check(14) + emit_hash_stage(15, 5)})
            self.instrs.append({"valu": emit_bounds_apply(14) + emit_hash_combine(15, 5)})
            self.instrs.append({"valu": emit_branch_ops(15)})
            self.instrs.append({"valu": emit_branch_add(15)})
            self.instrs.append({"valu": emit_bounds_check(15)})
            self.instrs.append({"valu": emit_bounds_apply(15)})

        # Emit Round 1
        emit_round_gather_and_hash()

        # ============================================================
        # ROUND 2: Prepare addresses, then gather + hash
        # ============================================================
        # Prepare gather addresses for round 2
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

        # ============================================================
        # ROUND 2 with OPTIMIZED TAIL: Interleave hash tail with stores
        # ============================================================
        # The key ILP insight: after desk 15 gather, we have 14 VALU-only cycles.
        # We can do store address computation (ALU) and stores during these cycles!

        # Desk 0 gather
        for lane in range(0, VLEN, 2):
            self.instrs.append({
                "load": [
                    ("load", desks[0]['node_val'] + lane, desks[0]['addr'] + lane),
                    ("load", desks[0]['node_val'] + lane + 1, desks[0]['addr'] + lane + 1),
                ],
            })

        # Desk 1 gather + desk 0 hash
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

        # Desk 2 gather + desk 0/1 hash
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

        # Desk 3
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

        # Desk 4
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

        # Desks 5-15 with same pattern
        for gather_desk in range(5, 16):
            hash_base = gather_desk - 4
            for lane_pair in range(4):
                lane = lane_pair * 2
                valu_ops = []
                if lane_pair == 0:
                    valu_ops = emit_bounds_check(hash_base) + emit_hash_stage(hash_base+1, 5) + emit_hash_stage(hash_base+2, 2) + emit_xor_node(hash_base+3)
                elif lane_pair == 1:
                    valu_ops = emit_bounds_apply(hash_base) + emit_hash_combine(hash_base+1, 5) + emit_hash_stage(hash_base+2, 3) + emit_hash_stage(hash_base+3, 0)
                elif lane_pair == 2:
                    valu_ops = emit_branch_ops(hash_base+1) + emit_hash_combine(hash_base+2, 3) + emit_hash_stage(hash_base+3, 1)
                elif lane_pair == 3:
                    valu_ops = emit_branch_add(hash_base+1) + emit_hash_stage(hash_base+2, 4) + emit_hash_combine(hash_base+3, 1)

                self.instrs.append({
                    "load": [
                        ("load", desks[gather_desk]['node_val'] + lane, desks[gather_desk]['addr'] + lane),
                        ("load", desks[gather_desk]['node_val'] + lane + 1, desks[gather_desk]['addr'] + lane + 1),
                    ],
                    "valu": valu_ops,
                })

        # ============================================================
        # OPTIMIZED TAIL: Interleave Round 2 hash completion with stores
        # ============================================================
        # Key insight: The tail hash (14 cycles) is VALU-only.
        # Store addresses were already computed during Round 1 tail!
        # We can start stores immediately for desks 0-11.
        #
        # By cycle in tail:
        # - Desks 0-11 are fully done at start of tail (can store immediately)
        # - Desk 12 done after cycle 2 (bounds_apply)
        # - Desk 13 done after cycle 6 (bounds_apply)
        # - Desk 14 done after cycle 10 (bounds_apply)
        # - Desk 15 done after cycle 14 (bounds_apply)

        # Tail cycle 1: Hash + store desk 0
        self.instrs.append({
            "valu": emit_bounds_check(12) + emit_hash_stage(13, 5) + emit_hash_stage(14, 2) + emit_xor_node(15),
            "store": [
                ("vstore", addr_tmp[0], desks[0]['idx']),
                ("vstore", addr_tmp[1], desks[0]['val']),
            ],
        })

        # Tail cycle 2: Hash + store desk 1
        self.instrs.append({
            "valu": emit_bounds_apply(12) + emit_hash_combine(13, 5) + emit_hash_stage(14, 3) + emit_hash_stage(15, 0),
            "store": [
                ("vstore", addr_tmp[2], desks[1]['idx']),
                ("vstore", addr_tmp[3], desks[1]['val']),
            ],
        })

        # Tail cycle 3: Hash + store desk 2
        self.instrs.append({
            "valu": emit_branch_ops(13) + emit_hash_combine(14, 3) + emit_hash_stage(15, 1),
            "store": [
                ("vstore", addr_tmp[4], desks[2]['idx']),
                ("vstore", addr_tmp[5], desks[2]['val']),
            ],
        })

        # Tail cycle 4: Hash + store desk 3
        self.instrs.append({
            "valu": emit_branch_add(13) + emit_hash_stage(14, 4) + emit_hash_combine(15, 1),
            "store": [
                ("vstore", addr_tmp[6], desks[3]['idx']),
                ("vstore", addr_tmp[7], desks[3]['val']),
            ],
        })

        # Tail cycle 5: Hash + stores for desk 4
        self.instrs.append({
            "valu": emit_bounds_check(13) + emit_hash_stage(14, 5) + emit_hash_stage(15, 2),
            "store": [
                ("vstore", addr_tmp[8], desks[4]['idx']),
                ("vstore", addr_tmp[9], desks[4]['val']),
            ],
        })

        # Tail cycle 6: Hash + stores for desk 5
        self.instrs.append({
            "valu": emit_bounds_apply(13) + emit_hash_combine(14, 5) + emit_hash_stage(15, 3),
            "store": [
                ("vstore", addr_tmp[10], desks[5]['idx']),
                ("vstore", addr_tmp[11], desks[5]['val']),
            ],
        })

        # Tail cycle 7: Hash + stores for desk 6
        self.instrs.append({
            "valu": emit_branch_ops(14) + emit_hash_combine(15, 3),
            "store": [
                ("vstore", addr_tmp[12], desks[6]['idx']),
                ("vstore", addr_tmp[13], desks[6]['val']),
            ],
        })

        # Tail cycle 8: Hash + stores for desk 7
        self.instrs.append({
            "valu": emit_branch_add(14) + emit_hash_stage(15, 4),
            "store": [
                ("vstore", addr_tmp[14], desks[7]['idx']),
                ("vstore", addr_tmp[15], desks[7]['val']),
            ],
        })

        # Tail cycle 9: Hash + stores for desk 8
        self.instrs.append({
            "valu": emit_bounds_check(14) + emit_hash_stage(15, 5),
            "store": [
                ("vstore", addr_tmp[16], desks[8]['idx']),
                ("vstore", addr_tmp[17], desks[8]['val']),
            ],
        })

        # Tail cycle 10: Hash + stores for desk 9
        self.instrs.append({
            "valu": emit_bounds_apply(14) + emit_hash_combine(15, 5),
            "store": [
                ("vstore", addr_tmp[18], desks[9]['idx']),
                ("vstore", addr_tmp[19], desks[9]['val']),
            ],
        })

        # Tail cycle 11: Hash + stores for desk 10
        self.instrs.append({
            "valu": emit_branch_ops(15),
            "store": [
                ("vstore", addr_tmp[20], desks[10]['idx']),
                ("vstore", addr_tmp[21], desks[10]['val']),
            ],
        })

        # Tail cycle 12: Hash + stores for desk 11
        self.instrs.append({
            "valu": emit_branch_add(15),
            "store": [
                ("vstore", addr_tmp[22], desks[11]['idx']),
                ("vstore", addr_tmp[23], desks[11]['val']),
            ],
        })

        # Tail cycle 13: Hash + stores for desk 12
        self.instrs.append({
            "valu": emit_bounds_check(15),
            "store": [
                ("vstore", addr_tmp[24], desks[12]['idx']),
                ("vstore", addr_tmp[25], desks[12]['val']),
            ],
        })

        # Tail cycle 14: Hash + stores for desk 13
        self.instrs.append({
            "valu": emit_bounds_apply(15),
            "store": [
                ("vstore", addr_tmp[26], desks[13]['idx']),
                ("vstore", addr_tmp[27], desks[13]['val']),
            ],
        })

        # Store desks 14-15 + loop control (desks 12-13 were stored in tail cycles 13-14)
        self.instrs.append({
            "store": [
                ("vstore", addr_tmp[28], desks[14]['idx']),
                ("vstore", addr_tmp[29], desks[14]['val']),
            ],
            "alu": [
                ("+", batch_offset, batch_offset, desk_stride_const),
                ("+", iter_counter, iter_counter, one_const),
            ],
        })
        self.instrs.append({
            "store": [
                ("vstore", addr_tmp[30], desks[15]['idx']),
                ("vstore", addr_tmp[31], desks[15]['val']),
            ],
            "alu": [
                ("<", tmp_scalar, batch_offset, batch_size_const),
                ("<", addr_scalar, iter_counter, total_const),
            ],
        })
        self.instrs.append({
            "flow": [
                ("select", batch_offset, tmp_scalar, batch_offset, zero_const),
            ],
        })

        self.add("flow", ("cond_jump", addr_scalar, main_loop_start))
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

    kb = KernelBuilderH59()
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
