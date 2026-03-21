"""
# Experiment H41: Full Level-Aware Optimization with Path Tracking

**KEY INSIGHT:**
Tree level is DETERMINISTIC from round number:
- Round 0: Level 0 (tree[0]) - all lanes access same node
- Round 1: Level 1 (tree[1] or tree[2]) - 2 possible nodes
- Round 2: Level 2 (tree[3..6]) - 4 possible nodes
- ...
- Round 10: Level 10 (back to random access)
- Round 11+: Wraps around

**STRATEGY:**
1. Preload tree levels 0-3 (15 nodes: 1+2+4+8 = 15)
2. For rounds 0-3, use broadcast/vselect instead of gather
3. For rounds 4+, use standard gather with 8-desk overlap

**CYCLE SAVINGS:**
- Round 0: Save 4 gather cycles per batch (use broadcast)
- Rounds 1-3: Save ~2-3 cycles per batch (use vselect trees)
- Rounds 4+: Standard gather

With 32 batches and 16 rounds:
- Round 0: 32 batches * 4 cycles saved = 128 cycles
- Rounds 1-3 (x4 with wrap): 32 * 4 * 2 = 256 cycles saved
- Total potential savings: ~384 cycles

From 4,062 -> ~3,700 cycles, ~9% improvement
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


class KernelBuilderH41:
    """
    H41: Level-aware optimization with preloaded tree levels.
    Uses 8-desk pipeline with round fusion.
    For level 0 rounds, broadcasts tree[0] instead of gather.
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
        H41: 8-desk with level-aware optimization for level 0.
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

        # Preload tree[0] as a vector (for level 0 rounds)
        tree_0_scalar = self.alloc_scratch("tree_0_scalar")
        v_tree_0 = self.alloc_scratch("v_tree_0", VLEN)
        self.add("load", ("load", tree_0_scalar, self.scratch["forest_values_p"]))
        self.add("valu", ("vbroadcast", v_tree_0, tree_0_scalar))

        self.add("flow", ("pause",))

        # 8 DESKS for deep pipeline
        NUM_DESKS = 8

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

        # Address temporaries
        addr_tmp = [self.alloc_scratch(f"addr_tmp_{i}") for i in range(16)]

        # Offset registers for each desk
        offset_regs = [self.alloc_scratch(f"off_{d}") for d in range(NUM_DESKS)]

        batch_offset = self.alloc_scratch("batch_offset")
        iter_counter = self.alloc_scratch("iter_counter")
        round_counter = self.alloc_scratch("round_counter")

        # Constants for offsets
        offset_consts = []
        for d in range(NUM_DESKS):
            offset_consts.append(self.scratch_const(d * VLEN))

        batch_size_const = self.scratch_const(batch_size)
        desk_stride_const = self.scratch_const(NUM_DESKS * VLEN)
        rounds_const = self.scratch_const(rounds)
        num_desks_const = self.scratch_const(NUM_DESKS)
        height_plus_one = self.scratch_const(forest_height + 1)  # 11 for level wrap

        # Total iterations for inner loop (per round): batch_size / (NUM_DESKS * VLEN)
        inner_iters = batch_size // (NUM_DESKS * VLEN)
        inner_iters_const = self.scratch_const(inner_iters)

        self.add("load", ("const", batch_offset, 0))
        self.add("load", ("const", iter_counter, 0))
        self.add("load", ("const", round_counter, 0))

        print(f"Scratch usage before main loop: {self.scratch_ptr} / {SCRATCH_SIZE}")

        # Helper functions
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

        def emit_bounds_check(desk_idx):
            d = desks[desk_idx]
            return [("<", d['tmp1'], d['idx'], v_n_nodes)]

        def emit_bounds_apply(desk_idx):
            d = desks[desk_idx]
            return [("*", d['idx'], d['idx'], d['tmp1'])]

        def emit_xor_node(desk_idx):
            d = desks[desk_idx]
            return [("^", d['val'], d['val'], d['node_val'])]

        def emit_full_hash_and_branch(desk_idx):
            """Emit full hash + branch operations for one desk"""
            ops = []
            ops.extend(emit_hash_stage(desk_idx, 0))  # FMA
            ops.extend(emit_hash_stage(desk_idx, 1))  # XOR prep
            ops.extend(emit_hash_combine(desk_idx, 1))  # XOR combine
            ops.extend(emit_hash_stage(desk_idx, 2))  # FMA
            ops.extend(emit_hash_stage(desk_idx, 3))  # Add prep
            ops.extend(emit_hash_combine(desk_idx, 3))  # XOR combine
            ops.extend(emit_hash_stage(desk_idx, 4))  # FMA
            ops.extend(emit_hash_stage(desk_idx, 5))  # XOR prep
            ops.extend(emit_hash_combine(desk_idx, 5))  # XOR combine
            ops.extend(emit_branch_ops(desk_idx))
            ops.extend(emit_branch_add(desk_idx))
            ops.extend(emit_bounds_check(desk_idx))
            ops.extend(emit_bounds_apply(desk_idx))
            return ops

        # ======================================================================
        # OUTER LOOP: Iterate over rounds
        # ======================================================================
        round_loop_start = len(self.instrs)

        # Reset batch offset at start of each round
        self.add("load", ("const", batch_offset, 0))
        self.add("load", ("const", iter_counter, 0))

        # ======================================================================
        # BATCH LOOP: Process batches within each round
        # ======================================================================
        batch_loop_start = len(self.instrs)

        # PHASE 1: Calculate all offsets for 8 desks
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
            ],
        })

        # Compute load addresses for desks 0-3
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
            ],
        })

        # Compute load addresses for desks 4-7
        self.instrs.append({
            "alu": [
                ("+", addr_tmp[8], self.scratch["inp_indices_p"], offset_regs[4]),
                ("+", addr_tmp[9], self.scratch["inp_values_p"], offset_regs[4]),
                ("+", addr_tmp[10], self.scratch["inp_indices_p"], offset_regs[5]),
                ("+", addr_tmp[11], self.scratch["inp_values_p"], offset_regs[5]),
                ("+", addr_tmp[12], self.scratch["inp_indices_p"], offset_regs[6]),
                ("+", addr_tmp[13], self.scratch["inp_values_p"], offset_regs[6]),
                ("+", addr_tmp[14], self.scratch["inp_indices_p"], offset_regs[7]),
                ("+", addr_tmp[15], self.scratch["inp_values_p"], offset_regs[7]),
            ],
        })

        # PHASE 2: Load idx/val for all 8 desks
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

        # PHASE 3: Check if we're in round 0 (level 0)
        # For round 0, we can use the preloaded tree[0] instead of gather
        level_0_check = self.alloc_scratch("level_0_check")
        self.add("alu", ("==", level_0_check, round_counter, zero_const))

        # Branch to level-0 optimized path if round_counter == 0
        level_0_opt_start = len(self.instrs) + 1
        self.add("flow", ("cond_jump", level_0_check, level_0_opt_start))

        # ======================================================================
        # STANDARD PATH: Gather + Hash for rounds 1+
        # ======================================================================

        # Prepare gather addresses for all desks
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
            ],
        })

        # Interleaved gather + hash for 8 desks
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

        # Desk 2 gather + desk 0-1 hash
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

        # Desk 3 gather + desk 0-2 hash
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

        # Desk 4 gather + desk 0-3 operations
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

        # Desk 5 gather + desk 1-4 operations
        self.instrs.append({
            "load": [
                ("load", desks[5]['node_val'], desks[5]['addr']),
                ("load", desks[5]['node_val'] + 1, desks[5]['addr'] + 1),
            ],
            "valu": emit_bounds_check(1) + emit_hash_stage(2, 5) + emit_hash_stage(3, 2) + emit_xor_node(4),
        })
        self.instrs.append({
            "load": [
                ("load", desks[5]['node_val'] + 2, desks[5]['addr'] + 2),
                ("load", desks[5]['node_val'] + 3, desks[5]['addr'] + 3),
            ],
            "valu": emit_bounds_apply(1) + emit_hash_combine(2, 5) + emit_hash_stage(3, 3) + emit_hash_stage(4, 0),
        })
        self.instrs.append({
            "load": [
                ("load", desks[5]['node_val'] + 4, desks[5]['addr'] + 4),
                ("load", desks[5]['node_val'] + 5, desks[5]['addr'] + 5),
            ],
            "valu": emit_branch_ops(2) + emit_hash_combine(3, 3) + emit_hash_stage(4, 1),
        })
        self.instrs.append({
            "load": [
                ("load", desks[5]['node_val'] + 6, desks[5]['addr'] + 6),
                ("load", desks[5]['node_val'] + 7, desks[5]['addr'] + 7),
            ],
            "valu": emit_branch_add(2) + emit_hash_stage(3, 4) + emit_hash_combine(4, 1),
        })

        # Desk 6 gather + desk 2-5 operations
        self.instrs.append({
            "load": [
                ("load", desks[6]['node_val'], desks[6]['addr']),
                ("load", desks[6]['node_val'] + 1, desks[6]['addr'] + 1),
            ],
            "valu": emit_bounds_check(2) + emit_hash_stage(3, 5) + emit_hash_stage(4, 2) + emit_xor_node(5),
        })
        self.instrs.append({
            "load": [
                ("load", desks[6]['node_val'] + 2, desks[6]['addr'] + 2),
                ("load", desks[6]['node_val'] + 3, desks[6]['addr'] + 3),
            ],
            "valu": emit_bounds_apply(2) + emit_hash_combine(3, 5) + emit_hash_stage(4, 3) + emit_hash_stage(5, 0),
        })
        self.instrs.append({
            "load": [
                ("load", desks[6]['node_val'] + 4, desks[6]['addr'] + 4),
                ("load", desks[6]['node_val'] + 5, desks[6]['addr'] + 5),
            ],
            "valu": emit_branch_ops(3) + emit_hash_combine(4, 3) + emit_hash_stage(5, 1),
        })
        self.instrs.append({
            "load": [
                ("load", desks[6]['node_val'] + 6, desks[6]['addr'] + 6),
                ("load", desks[6]['node_val'] + 7, desks[6]['addr'] + 7),
            ],
            "valu": emit_branch_add(3) + emit_hash_stage(4, 4) + emit_hash_combine(5, 1),
        })

        # Desk 7 gather + desk 3-6 operations
        self.instrs.append({
            "load": [
                ("load", desks[7]['node_val'], desks[7]['addr']),
                ("load", desks[7]['node_val'] + 1, desks[7]['addr'] + 1),
            ],
            "valu": emit_bounds_check(3) + emit_hash_stage(4, 5) + emit_hash_stage(5, 2) + emit_xor_node(6),
        })
        self.instrs.append({
            "load": [
                ("load", desks[7]['node_val'] + 2, desks[7]['addr'] + 2),
                ("load", desks[7]['node_val'] + 3, desks[7]['addr'] + 3),
            ],
            "valu": emit_bounds_apply(3) + emit_hash_combine(4, 5) + emit_hash_stage(5, 3) + emit_hash_stage(6, 0),
        })
        self.instrs.append({
            "load": [
                ("load", desks[7]['node_val'] + 4, desks[7]['addr'] + 4),
                ("load", desks[7]['node_val'] + 5, desks[7]['addr'] + 5),
            ],
            "valu": emit_branch_ops(4) + emit_hash_combine(5, 3) + emit_hash_stage(6, 1),
        })
        self.instrs.append({
            "load": [
                ("load", desks[7]['node_val'] + 6, desks[7]['addr'] + 6),
                ("load", desks[7]['node_val'] + 7, desks[7]['addr'] + 7),
            ],
            "valu": emit_branch_add(4) + emit_hash_stage(5, 4) + emit_hash_combine(6, 1),
        })

        # After desk 7 gather complete, finish remaining operations for desks 4-7
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

        # Jump to store
        store_phase = len(self.instrs)
        self.add("flow", ("jump", 0))  # Placeholder, will patch
        store_jump_instr = len(self.instrs) - 1

        # ======================================================================
        # LEVEL 0 PATH: Use preloaded tree[0]
        # ======================================================================
        level_0_opt_start_actual = len(self.instrs)

        # Patch the conditional jump
        self.instrs[level_0_opt_start - 1] = {"flow": [("cond_jump", level_0_check, level_0_opt_start_actual)]}

        # For level 0, all lanes access tree[0], which we have preloaded as v_tree_0
        # Just copy v_tree_0 to node_val for all desks and proceed
        self.instrs.append({
            "valu": [
                ("+", desks[0]['node_val'], v_tree_0, v_zero),  # Copy v_tree_0
                ("+", desks[1]['node_val'], v_tree_0, v_zero),
                ("+", desks[2]['node_val'], v_tree_0, v_zero),
                ("+", desks[3]['node_val'], v_tree_0, v_zero),
                ("+", desks[4]['node_val'], v_tree_0, v_zero),
                ("+", desks[5]['node_val'], v_tree_0, v_zero),
            ],
        })
        self.instrs.append({
            "valu": [
                ("+", desks[6]['node_val'], v_tree_0, v_zero),
                ("+", desks[7]['node_val'], v_tree_0, v_zero),
            ],
        })

        # Now do XOR + hash for all 8 desks
        # This is faster than gather since we don't need load ops
        for d in range(NUM_DESKS):
            self.instrs.append({"valu": emit_xor_node(d)})
            self.instrs.append({"valu": emit_hash_stage(d, 0)})
            self.instrs.append({"valu": emit_hash_stage(d, 1)})
            self.instrs.append({"valu": emit_hash_combine(d, 1)})
            self.instrs.append({"valu": emit_hash_stage(d, 2)})
            self.instrs.append({"valu": emit_hash_stage(d, 3)})
            self.instrs.append({"valu": emit_hash_combine(d, 3)})
            self.instrs.append({"valu": emit_hash_stage(d, 4)})
            self.instrs.append({"valu": emit_hash_stage(d, 5)})
            self.instrs.append({"valu": emit_hash_combine(d, 5)})
            self.instrs.append({"valu": emit_branch_ops(d)})
            self.instrs.append({"valu": emit_branch_add(d)})
            self.instrs.append({"valu": emit_bounds_check(d)})
            self.instrs.append({"valu": emit_bounds_apply(d)})

        # ======================================================================
        # STORE PHASE: Store all 8 desks
        # ======================================================================
        store_phase_actual = len(self.instrs)

        # Patch the jump from standard path
        self.instrs[store_jump_instr] = {"flow": [("jump", store_phase_actual)]}

        # Compute store addresses
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
            ],
        })
        self.instrs.append({
            "alu": [
                ("+", addr_tmp[8], self.scratch["inp_indices_p"], offset_regs[4]),
                ("+", addr_tmp[9], self.scratch["inp_values_p"], offset_regs[4]),
                ("+", addr_tmp[10], self.scratch["inp_indices_p"], offset_regs[5]),
                ("+", addr_tmp[11], self.scratch["inp_values_p"], offset_regs[5]),
                ("+", addr_tmp[12], self.scratch["inp_indices_p"], offset_regs[6]),
                ("+", addr_tmp[13], self.scratch["inp_values_p"], offset_regs[6]),
                ("+", addr_tmp[14], self.scratch["inp_indices_p"], offset_regs[7]),
                ("+", addr_tmp[15], self.scratch["inp_values_p"], offset_regs[7]),
            ],
        })

        # Store all 8 desks
        for d in range(NUM_DESKS):
            self.instrs.append({
                "store": [
                    ("vstore", addr_tmp[d*2], desks[d]['idx']),
                    ("vstore", addr_tmp[d*2+1], desks[d]['val']),
                ],
            })

        # Batch loop control
        self.instrs.append({
            "alu": [
                ("+", batch_offset, batch_offset, desk_stride_const),
                ("+", iter_counter, iter_counter, one_const),
            ],
        })
        self.instrs.append({
            "alu": [
                ("<", tmp_scalar, iter_counter, inner_iters_const),
            ],
        })

        self.add("flow", ("cond_jump", tmp_scalar, batch_loop_start))

        # Round loop control
        self.instrs.append({
            "alu": [
                ("+", round_counter, round_counter, one_const),
            ],
        })
        self.instrs.append({
            "alu": [
                ("<", tmp_scalar, round_counter, rounds_const),
            ],
        })

        self.add("flow", ("cond_jump", tmp_scalar, round_loop_start))
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

    kb = KernelBuilderH41()
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
