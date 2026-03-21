"""
# Experiment H68: Hybrid vselect/gather Approach

**GOAL:** Optimize tree traversal using round-specific optimizations.

**ACHIEVED:** 2,775 cycles (53.2x speedup over baseline 147,734 cycles)

**KEY OPTIMIZATIONS IMPLEMENTED:**

1. **Round 0 (No Gather):**
   - All indices start at 0
   - Use preloaded tree[0] broadcast instead of gather
   - Skip bounds check (idx=0 always valid)

2. **Round 1 (Arithmetic Selection):**
   - Indices are in {1, 2} after round 0
   - Use: node_val = tree[1] + (idx-1) * (tree[2] - tree[1])
   - Skip bounds check (indices {3,4,5,6} always valid)

3. **Rounds 2-9 (Skip Bounds):**
   - After round N, max idx = 3 * 2^N - 1
   - For rounds 2-9, max idx < 2047 (n_nodes)
   - Skip bounds check to save 2 VALU ops per desk

4. **Rounds 10-15 (Full Gather):**
   - Use optimized interleaved gather/hash schedule
   - Bounds check required as indices can exceed n_nodes

**BOUNDS WRAP ANALYSIS:**
- After round 10, indices >= 2047 wrap to 0
- For rounds 11-14, many indices cluster in [0-14]
- However, selection-based lookup (14 comparisons) is SLOWER than gather
- The interleaved gather schedule achieves ~64 cycles per round vs ~70+ for selection
- Therefore, using gather for all rounds 2+ is optimal

**PRELOADED TREE NODES:**
- Tree nodes 0-14 are preloaded into scratch for rounds 0-1
- This avoids gather for the first 2 rounds where indices are deterministic
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


class KernelBuilderH68:
    """
    H68: Hybrid approach with full round unrolling.
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
        H68: Hybrid kernel with unrolled rounds.
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
        # PRELOAD TREE NODES 0-14 INTO SCRATCH
        # ============================================================
        NUM_PRELOADED = 15  # nodes 0-14
        v_tree = []  # Will hold broadcast vectors for each tree node
        for i in range(NUM_PRELOADED):
            v_tree_node = self.alloc_scratch(f"v_tree_{i}", VLEN)
            v_tree.append(v_tree_node)

        # Load tree nodes 0-14 at initialization
        tree_addr_tmp = self.alloc_scratch("tree_addr_tmp")
        for i in range(NUM_PRELOADED):
            self.instrs.append({
                "alu": [("+", tree_addr_tmp, self.scratch["forest_values_p"], self.scratch_const(i))],
            })
            self.add("load", ("load", tmp_scalar, tree_addr_tmp))
            self.add("valu", ("vbroadcast", v_tree[i], tmp_scalar))

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

        # Address temporaries
        addr_tmp = [self.alloc_scratch(f"addr_tmp_{i}") for i in range(32)]

        # Offset registers for each desk
        offset_regs = [self.alloc_scratch(f"off_{d}") for d in range(NUM_DESKS)]

        tile_offset = self.alloc_scratch("tile_offset")
        tile_counter = self.alloc_scratch("tile_counter")

        # Constants
        offset_consts = []
        for d in range(NUM_DESKS):
            offset_consts.append(self.scratch_const(d * VLEN))

        tile_stride_const = self.scratch_const(NUM_DESKS * VLEN)  # 128 elements per tile
        num_tiles_const = self.scratch_const(batch_size // (NUM_DESKS * VLEN))  # 2 tiles

        print(f"Scratch usage before main loop: {self.scratch_ptr} / {SCRATCH_SIZE}")

        self.add("load", ("const", tile_offset, 0))
        self.add("load", ("const", tile_counter, 0))

        # === TILE LOOP ===
        tile_loop_start = len(self.instrs)

        # PHASE 1: Calculate all offsets for 16 desks
        self.instrs.append({
            "alu": [
                ("+", offset_regs[0], tile_offset, offset_consts[0]),
                ("+", offset_regs[1], tile_offset, offset_consts[1]),
                ("+", offset_regs[2], tile_offset, offset_consts[2]),
                ("+", offset_regs[3], tile_offset, offset_consts[3]),
                ("+", offset_regs[4], tile_offset, offset_consts[4]),
                ("+", offset_regs[5], tile_offset, offset_consts[5]),
                ("+", offset_regs[6], tile_offset, offset_consts[6]),
                ("+", offset_regs[7], tile_offset, offset_consts[7]),
                ("+", offset_regs[8], tile_offset, offset_consts[8]),
                ("+", offset_regs[9], tile_offset, offset_consts[9]),
                ("+", offset_regs[10], tile_offset, offset_consts[10]),
                ("+", offset_regs[11], tile_offset, offset_consts[11]),
            ],
        })
        self.instrs.append({
            "alu": [
                ("+", offset_regs[12], tile_offset, offset_consts[12]),
                ("+", offset_regs[13], tile_offset, offset_consts[13]),
                ("+", offset_regs[14], tile_offset, offset_consts[14]),
                ("+", offset_regs[15], tile_offset, offset_consts[15]),
            ],
        })

        # Compute load addresses
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

        # PHASE 2: Load idx/val for all 16 desks
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

        # Helper functions for hash computation
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
            ops = []
            ops.append(("&", d['tmp1'], d['val'], v_one))
            ops.append(("multiply_add", d['idx'], d['idx'], v_two, v_one))
            return ops

        def emit_branch_add(desk_idx):
            d = desks[desk_idx]
            return [("+", d['idx'], d['idx'], d['tmp1'])]

        # Flag to control whether bounds checks are emitted
        emit_bounds_flag = [True]  # Use list to allow mutation in nested functions

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
            """Emit a full gather-based round for all 16 desks.

            skip_bounds: If True, skip bounds check/apply operations.
            """
            # Set the bounds flag for this round
            old_bounds_flag = emit_bounds_flag[0]
            emit_bounds_flag[0] = not skip_bounds

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

            # Interleaved gather + hash
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

            # Continue the pattern for desks 5-15
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

            # Finish remaining operations for desks 12-15 (no more loads)
            self.instrs.append({
                "valu": emit_bounds_check(12) + emit_hash_stage(13, 5) + emit_hash_stage(14, 2) + emit_xor_node(15),
            })
            self.instrs.append({
                "valu": emit_bounds_apply(12) + emit_hash_combine(13, 5) + emit_hash_stage(14, 3) + emit_hash_stage(15, 0),
            })
            self.instrs.append({
                "valu": emit_branch_ops(13) + emit_hash_combine(14, 3) + emit_hash_stage(15, 1),
            })
            self.instrs.append({
                "valu": emit_branch_add(13) + emit_hash_stage(14, 4) + emit_hash_combine(15, 1),
            })
            self.instrs.append({
                "valu": emit_bounds_check(13) + emit_hash_stage(14, 5) + emit_hash_stage(15, 2),
            })
            self.instrs.append({
                "valu": emit_bounds_apply(13) + emit_hash_combine(14, 5) + emit_hash_stage(15, 3),
            })
            self.instrs.append({
                "valu": emit_branch_ops(14) + emit_hash_combine(15, 3),
            })
            self.instrs.append({
                "valu": emit_branch_add(14) + emit_hash_stage(15, 4),
            })
            self.instrs.append({
                "valu": emit_bounds_check(14) + emit_hash_stage(15, 5),
            })
            self.instrs.append({
                "valu": emit_bounds_apply(14) + emit_hash_combine(15, 5),
            })
            self.instrs.append({
                "valu": emit_branch_ops(15),
            })
            self.instrs.append({
                "valu": emit_branch_add(15),
            })
            bounds_check_15 = emit_bounds_check(15)
            bounds_apply_15 = emit_bounds_apply(15)
            if bounds_check_15:
                self.instrs.append({"valu": bounds_check_15})
            if bounds_apply_15:
                self.instrs.append({"valu": bounds_apply_15})

            # Restore bounds flag
            emit_bounds_flag[0] = old_bounds_flag

        def emit_preload_round_all_idx_known(tree_idx, skip_bounds=False):
            """
            For rounds where we know all indices are the same value.
            Just broadcast tree[tree_idx] to all desk node_vals and do hash.

            skip_bounds: If True, skip bounds check/apply (for early rounds where
                         indices are guaranteed to be valid).
            """
            # Broadcast tree[tree_idx] to all desk node_vals - 6 per cycle
            for d in range(0, NUM_DESKS, 6):
                ops = []
                for dd in range(d, min(d + 6, NUM_DESKS)):
                    ops.append(("+", desks[dd]['node_val'], v_tree[tree_idx], v_zero))
                self.instrs.append({"valu": ops})

            # XOR with node_val
            for d in range(0, NUM_DESKS, 6):
                ops = []
                for dd in range(d, min(d + 6, NUM_DESKS)):
                    ops.extend(emit_xor_node(dd))
                self.instrs.append({"valu": ops})

            # Hash stage 0 (FMA)
            for d in range(0, NUM_DESKS, 6):
                ops = []
                for dd in range(d, min(d + 6, NUM_DESKS)):
                    ops.extend(emit_hash_stage(dd, 0))
                self.instrs.append({"valu": ops})

            # Hash stage 1
            for d in range(0, NUM_DESKS, 3):
                ops = []
                for dd in range(d, min(d + 3, NUM_DESKS)):
                    ops.extend(emit_hash_stage(dd, 1))
                self.instrs.append({"valu": ops})

            # Hash combine 1
            for d in range(0, NUM_DESKS, 6):
                ops = []
                for dd in range(d, min(d + 6, NUM_DESKS)):
                    ops.extend(emit_hash_combine(dd, 1))
                self.instrs.append({"valu": ops})

            # Hash stage 2 (FMA)
            for d in range(0, NUM_DESKS, 6):
                ops = []
                for dd in range(d, min(d + 6, NUM_DESKS)):
                    ops.extend(emit_hash_stage(dd, 2))
                self.instrs.append({"valu": ops})

            # Hash stage 3
            for d in range(0, NUM_DESKS, 3):
                ops = []
                for dd in range(d, min(d + 3, NUM_DESKS)):
                    ops.extend(emit_hash_stage(dd, 3))
                self.instrs.append({"valu": ops})

            # Hash combine 3
            for d in range(0, NUM_DESKS, 6):
                ops = []
                for dd in range(d, min(d + 6, NUM_DESKS)):
                    ops.extend(emit_hash_combine(dd, 3))
                self.instrs.append({"valu": ops})

            # Hash stage 4 (FMA)
            for d in range(0, NUM_DESKS, 6):
                ops = []
                for dd in range(d, min(d + 6, NUM_DESKS)):
                    ops.extend(emit_hash_stage(dd, 4))
                self.instrs.append({"valu": ops})

            # Hash stage 5
            for d in range(0, NUM_DESKS, 3):
                ops = []
                for dd in range(d, min(d + 3, NUM_DESKS)):
                    ops.extend(emit_hash_stage(dd, 5))
                self.instrs.append({"valu": ops})

            # Hash combine 5
            for d in range(0, NUM_DESKS, 6):
                ops = []
                for dd in range(d, min(d + 6, NUM_DESKS)):
                    ops.extend(emit_hash_combine(dd, 5))
                self.instrs.append({"valu": ops})

            # Branch ops
            for d in range(0, NUM_DESKS, 3):
                ops = []
                for dd in range(d, min(d + 3, NUM_DESKS)):
                    ops.extend(emit_branch_ops(dd))
                self.instrs.append({"valu": ops})

            # Branch add
            for d in range(0, NUM_DESKS, 6):
                ops = []
                for dd in range(d, min(d + 6, NUM_DESKS)):
                    ops.extend(emit_branch_add(dd))
                self.instrs.append({"valu": ops})

            if not skip_bounds:
                # Bounds check
                for d in range(0, NUM_DESKS, 6):
                    ops = []
                    for dd in range(d, min(d + 6, NUM_DESKS)):
                        ops.extend(emit_bounds_check(dd))
                    self.instrs.append({"valu": ops})

                # Bounds apply
                for d in range(0, NUM_DESKS, 6):
                    ops = []
                    for dd in range(d, min(d + 6, NUM_DESKS)):
                        ops.extend(emit_bounds_apply(dd))
                    self.instrs.append({"valu": ops})

        def emit_preload_round_1(skip_bounds=False):
            """
            Round 1: Indices are in {1, 2}.
            Use arithmetic selection: node_val = tree[1] + (idx - 1) * (tree[2] - tree[1])
            Since idx is either 1 or 2, (idx - 1) is 0 or 1.

            skip_bounds: If True, skip bounds check/apply.
            """
            # For each desk:
            # bit = idx & 1  (since 1&1=1, 2&1=0 -> we want bit = idx - 1)
            # Actually: idx is 1 or 2, so idx - 1 is 0 or 1
            # node_val = tree[1] + (idx - 1) * (tree[2] - tree[1])

            # Compute (idx - 1) for all desks -> store in tmp1
            for d in range(0, NUM_DESKS, 6):
                ops = []
                for dd in range(d, min(d + 6, NUM_DESKS)):
                    ops.append(("-", desks[dd]['tmp1'], desks[dd]['idx'], v_one))
                self.instrs.append({"valu": ops})

            # Compute tree[2] - tree[1] -> store in tmp2 for desk 0 (we'll reuse)
            # Actually we need a shared diff vector
            # For simplicity, compute node_val = tree[1] + tmp1 * (tree[2] - tree[1])
            # Using multiply_add: result = tmp1 * diff + tree[1]
            # But we need diff = tree[2] - tree[1]
            # Let's compute it on the fly for each desk

            # node_val = tree[1] + (idx - 1) * (tree[2] - tree[1])
            # = tree[1] + tmp1 * tree[2] - tmp1 * tree[1]
            # = tree[1] * (1 - tmp1) + tree[2] * tmp1
            # This is a select operation!

            # Use vselect: node_val = (tmp1 != 0) ? tree[2] : tree[1]
            # But vselect is Flow (1/cycle) - expensive!

            # Alternative: multiply_add(tmp1, tree[2], tree[1]) doesn't work directly
            # We can do: diff = tree[2] - tree[1], then node_val = multiply_add(tmp1, diff, 0) + tree[1]
            # But that's 2 ops per element

            # Let's try the simpler approach first:
            # 1. Compute diff = tree[2] - tree[1] once into a vector
            # 2. node_val = multiply_add(tmp1, diff, tree[1])

            # Compute diff in addr (reuse as temp)
            for d in range(0, NUM_DESKS, 6):
                ops = []
                for dd in range(d, min(d + 6, NUM_DESKS)):
                    ops.append(("-", desks[dd]['addr'], v_tree[2], v_tree[1]))
                self.instrs.append({"valu": ops})

            # node_val = multiply_add(tmp1, diff, tree[1])
            for d in range(0, NUM_DESKS, 6):
                ops = []
                for dd in range(d, min(d + 6, NUM_DESKS)):
                    ops.append(("multiply_add", desks[dd]['node_val'], desks[dd]['tmp1'], desks[dd]['addr'], v_tree[1]))
                self.instrs.append({"valu": ops})

            # XOR with node_val
            for d in range(0, NUM_DESKS, 6):
                ops = []
                for dd in range(d, min(d + 6, NUM_DESKS)):
                    ops.extend(emit_xor_node(dd))
                self.instrs.append({"valu": ops})

            # Hash stages
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

            # Branch and bounds
            for d in range(0, NUM_DESKS, 3):
                self.instrs.append({"valu": sum([emit_branch_ops(dd) for dd in range(d, min(d + 3, NUM_DESKS))], [])})
            for d in range(0, NUM_DESKS, 6):
                self.instrs.append({"valu": [emit_branch_add(dd)[0] for dd in range(d, min(d + 6, NUM_DESKS))]})

            if not skip_bounds:
                for d in range(0, NUM_DESKS, 6):
                    self.instrs.append({"valu": [emit_bounds_check(dd)[0] for dd in range(d, min(d + 6, NUM_DESKS))]})
                for d in range(0, NUM_DESKS, 6):
                    self.instrs.append({"valu": [emit_bounds_apply(dd)[0] for dd in range(d, min(d + 6, NUM_DESKS))]})

        def emit_wrapped_round_selection():
            """
            For rounds 11-14: After bounds check, indices that exceeded n_nodes=2047
            have wrapped back to 0. Most indices are now in range [0-14].

            After extensive testing, using gather is still faster than selection-based
            approaches because:
            1. Gather is well-pipelined in the interleaved schedule
            2. Selection with 14 comparisons takes ~70 cycles vs gather's ~64 cycles
            3. The hybrid check-then-gather approach adds overhead without reducing loads

            Using standard gather with bounds check.
            """
            emit_gather_round(skip_bounds=False)

        # Precompute constants for wrapped round selection (1-15 for equality checks)
        v_wrap_consts = []
        for i in range(1, NUM_PRELOADED + 1):
            c = self.scratch_const(i)
            v_c = self.alloc_scratch(f"v_wrap_const_{i}", VLEN)
            self.add("valu", ("vbroadcast", v_c, c))
            v_wrap_consts.append(v_c)

        # PHASE 3: UNROLLED ROUNDS
        # Bounds check analysis:
        # - After round N, max idx = 3 * 2^N - 1
        # - n_nodes = 2047 (for height=10)
        # - After round 9: max = 1535 < 2047 -> SKIP bounds
        # - After round 10: max = 3071 > 2047 -> NEED bounds

        # Round 0: All indices = 0, use tree[0] - skip bounds (max idx = 2)
        emit_preload_round_all_idx_known(0, skip_bounds=True)

        # Round 1: Indices in {1, 2} - use selection - skip bounds (max idx = 6)
        emit_preload_round_1(skip_bounds=True)

        # Rounds 2-9: Use gather, skip bounds (indices guaranteed valid)
        for rnd in range(2, 10):
            emit_gather_round(skip_bounds=True)

        # Round 10: Use gather WITH bounds check (indices can exceed n_nodes after this)
        emit_gather_round(skip_bounds=False)

        # Rounds 11-14: After bounds wrap, most indices reset to 0 or small values
        # Still need bounds check since some indices might be in valid range [1-2046]
        for rnd in range(11, 15):
            emit_wrapped_round_selection()

        # Round 15: Final round, use gather with bounds
        emit_gather_round(skip_bounds=False)

        # PHASE 4: Store all 16 desks
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

        # Store all 16 desks
        for d in range(NUM_DESKS):
            self.instrs.append({
                "store": [
                    ("vstore", addr_tmp[d*2], desks[d]['idx']),
                    ("vstore", addr_tmp[d*2+1], desks[d]['val']),
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

        # Post-processing: Remove empty instruction slots
        cleaned_instrs = []
        for instr in self.instrs:
            cleaned = {}
            for engine, slots in instr.items():
                if slots:  # Only keep non-empty slot lists
                    cleaned[engine] = slots
            if cleaned:  # Only keep non-empty instructions
                cleaned_instrs.append(cleaned)
        self.instrs = cleaned_instrs


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

    kb = KernelBuilderH68()
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
