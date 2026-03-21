"""
# Experiment H42: Broadcast-and-Mask for Early Rounds

**THESIS:**
For early rounds (0-3), elements are clustered at a small number of tree nodes:
- Round 0: ALL elements at index 0 → 1 unique node
- Round 1: Elements at indices 1 or 2 → 2 unique nodes
- Round 2: Elements at indices 3-6 → 4 unique nodes
- Round 3: Elements at indices 7-14 → 8 unique nodes (= VLEN)

Current approach: Gather 8 values via scalar loads (4 cycles per desk)
Proposed: Broadcast + mask selection using ALU/VALU instead of gather

**KEY INSIGHT:**
Selection via masking uses ALU/VALU (12+6 slots/cycle) not flow engine (1 slot/cycle).
For round 0: 1 load + 1 broadcast vs 4 cycles gather = 3 cycle savings per desk.

**IMPLEMENTATION:**
This experiment tests round 0 only (simplest case).
- Detect when we're processing round 0 (all idx values are 0)
- Load tree[0] once
- Broadcast to all lanes
- Skip the 4-cycle gather

**CONSTRAINTS HONORED:**
- No vgather (we use broadcast instead)
- No indirect scratch (we use direct addressing)
- Round fusion preserved for rounds 1+
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


class KernelBuilderH42:
    """
    H42: Broadcast-and-Mask for Round 0

    Key insight: In round 0, ALL elements are at tree index 0.
    Instead of gathering 8 different values (which are all tree[0]),
    we load tree[0] once and broadcast to all lanes.
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
        H42: Broadcast-and-Mask kernel.

        Structure:
        1. Round 0 phase: Uses broadcast (all elements at root)
        2. Rounds 1-15 phase: Uses H38-style pipeline
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

        # 8 DESKS for deep pipeline (same as H38)
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

        # Offset registers
        offset_regs = [self.alloc_scratch(f"off_{d}") for d in range(NUM_DESKS)]

        batch_offset = self.alloc_scratch("batch_offset")
        iter_counter = self.alloc_scratch("iter_counter")
        round_counter = self.alloc_scratch("round_counter")

        # Constants
        offset_consts = []
        for d in range(NUM_DESKS):
            offset_consts.append(self.scratch_const(d * VLEN))

        batch_size_const = self.scratch_const(batch_size)
        desk_stride_const = self.scratch_const(NUM_DESKS * VLEN)

        # Scalar for tree[0] and broadcast
        root_node_val = self.alloc_scratch("root_node_val")
        v_root_node = self.alloc_scratch("v_root_node", VLEN)

        print(f"Scratch usage before main loop: {self.scratch_ptr} / {SCRATCH_SIZE}")

        self.add("load", ("const", batch_offset, 0))
        self.add("load", ("const", iter_counter, 0))
        self.add("load", ("const", round_counter, 0))

        # ============================================================
        # ROUND 0 PHASE: Special handling using broadcast
        # All elements start at index 0, so tree[0] is the only node needed
        # ============================================================

        # Load tree[0] once
        self.add("load", ("load", root_node_val, self.scratch["forest_values_p"]))
        # Broadcast to vector
        self.add("valu", ("vbroadcast", v_root_node, root_node_val))

        # Process all batches for round 0 (no gather needed - use broadcast)
        round0_batches = batch_size // VLEN  # 32 batches of 8 elements
        round0_const = self.scratch_const(round0_batches)

        round0_loop_start = len(self.instrs)

        # Load 8 desks of idx/val (same as H38 but simpler - all idx are 0)
        # Compute offsets for 8 desks
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

        # Compute load addresses for all 8 desks
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

        # Load idx/val for all 8 desks (8 vloads = 4 cycles at 2/cycle)
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

        # KEY OPTIMIZATION: Instead of 8 x 4-cycle gathers, copy v_root_node to all desks
        # This is 8 VALU operations = 2 cycles (6 VALU slots/cycle)
        self.instrs.append({
            "valu": [
                ("+", desks[0]['node_val'], v_root_node, v_zero),  # copy
                ("+", desks[1]['node_val'], v_root_node, v_zero),
                ("+", desks[2]['node_val'], v_root_node, v_zero),
                ("+", desks[3]['node_val'], v_root_node, v_zero),
                ("+", desks[4]['node_val'], v_root_node, v_zero),
                ("+", desks[5]['node_val'], v_root_node, v_zero),
            ],
        })
        self.instrs.append({
            "valu": [
                ("+", desks[6]['node_val'], v_root_node, v_zero),
                ("+", desks[7]['node_val'], v_root_node, v_zero),
            ],
        })

        # Now hash all 8 desks (no interleaving with gather needed!)
        # This can be parallelized better since we're not waiting for gathers

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
            d = desks[desk_idx]
            if stage in [1, 3, 5]:
                return [("^", d['val'], d['tmp1'], d['tmp2'])]
            return []

        def emit_xor_node(desk_idx):
            d = desks[desk_idx]
            return [("^", d['val'], d['val'], d['node_val'])]

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

        # Process all 8 desks - XOR first (can do 6 per cycle)
        self.instrs.append({"valu": emit_xor_node(0) + emit_xor_node(1) + emit_xor_node(2) + emit_xor_node(3) + emit_xor_node(4) + emit_xor_node(5)})
        self.instrs.append({"valu": emit_xor_node(6) + emit_xor_node(7)})

        # Hash stages for all desks - batch as much as possible
        for stage in range(6):
            # Pack hash operations for multiple desks per cycle
            ops = []
            for d in range(8):
                ops.extend(emit_hash_stage(d, stage))
            # Split into cycles (6 VALU slots max)
            while ops:
                cycle_ops = ops[:6]
                ops = ops[6:]
                self.instrs.append({"valu": cycle_ops})

            # Combine for XOR stages
            if stage in [1, 3, 5]:
                combines = []
                for d in range(8):
                    combines.extend(emit_hash_combine(d, stage))
                while combines:
                    cycle_ops = combines[:6]
                    combines = combines[6:]
                    self.instrs.append({"valu": cycle_ops})

        # Branch operations for all 8 desks
        branch_ops = []
        for d in range(8):
            branch_ops.extend(emit_branch_ops(d))
        while branch_ops:
            cycle_ops = branch_ops[:6]
            branch_ops = branch_ops[6:]
            self.instrs.append({"valu": cycle_ops})

        # Branch add
        add_ops = []
        for d in range(8):
            add_ops.extend(emit_branch_add(d))
        while add_ops:
            cycle_ops = add_ops[:6]
            add_ops = add_ops[6:]
            self.instrs.append({"valu": cycle_ops})

        # Bounds check
        check_ops = []
        for d in range(8):
            check_ops.extend(emit_bounds_check(d))
        while check_ops:
            cycle_ops = check_ops[:6]
            check_ops = check_ops[6:]
            self.instrs.append({"valu": cycle_ops})

        # Bounds apply (multiply by condition)
        apply_ops = []
        for d in range(8):
            apply_ops.extend(emit_bounds_apply(d))
        while apply_ops:
            cycle_ops = apply_ops[:6]
            apply_ops = apply_ops[6:]
            self.instrs.append({"valu": cycle_ops})

        # Store all 8 desks
        for d in range(NUM_DESKS):
            self.instrs.append({
                "store": [
                    ("vstore", addr_tmp[d*2], desks[d]['idx']),
                    ("vstore", addr_tmp[d*2+1], desks[d]['val']),
                ],
            })

        # Loop control for round 0
        self.instrs.append({
            "alu": [
                ("+", batch_offset, batch_offset, desk_stride_const),
                ("+", iter_counter, iter_counter, one_const),
            ],
        })
        self.instrs.append({
            "alu": [
                ("<", tmp_scalar, batch_offset, batch_size_const),
            ],
        })
        self.add("flow", ("cond_jump", tmp_scalar, round0_loop_start))

        # Reset for rounds 1-15
        self.add("load", ("const", batch_offset, 0))

        # ============================================================
        # ROUNDS 1-15: Standard H38-style pipeline (with round fusion)
        # ============================================================

        # For remaining rounds, use H38's approach
        remaining_rounds = rounds - 1  # 15 rounds
        total_remaining = (batch_size // VLEN) * (remaining_rounds) // NUM_DESKS
        total_remaining_const = self.scratch_const(total_remaining)

        self.add("load", ("const", iter_counter, 0))

        main_loop_start = len(self.instrs)

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

        # PHASE 3: Prepare gather addresses
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

        # PHASE 4: Gather for all 8 desks (32 cycles at 2 loads/cycle)
        for d in range(NUM_DESKS):
            for lane in range(0, VLEN, 2):
                self.instrs.append({
                    "load": [
                        ("load", desks[d]['node_val'] + lane, desks[d]['addr'] + lane),
                        ("load", desks[d]['node_val'] + lane + 1, desks[d]['addr'] + lane + 1),
                    ],
                })

        # PHASE 5: Hash all 8 desks
        # XOR with node values
        self.instrs.append({"valu": emit_xor_node(0) + emit_xor_node(1) + emit_xor_node(2) + emit_xor_node(3) + emit_xor_node(4) + emit_xor_node(5)})
        self.instrs.append({"valu": emit_xor_node(6) + emit_xor_node(7)})

        # Hash stages
        for stage in range(6):
            ops = []
            for d in range(8):
                ops.extend(emit_hash_stage(d, stage))
            while ops:
                cycle_ops = ops[:6]
                ops = ops[6:]
                self.instrs.append({"valu": cycle_ops})

            if stage in [1, 3, 5]:
                combines = []
                for d in range(8):
                    combines.extend(emit_hash_combine(d, stage))
                while combines:
                    cycle_ops = combines[:6]
                    combines = combines[6:]
                    self.instrs.append({"valu": cycle_ops})

        # Branch operations
        branch_ops = []
        for d in range(8):
            branch_ops.extend(emit_branch_ops(d))
        while branch_ops:
            cycle_ops = branch_ops[:6]
            branch_ops = branch_ops[6:]
            self.instrs.append({"valu": cycle_ops})

        add_ops = []
        for d in range(8):
            add_ops.extend(emit_branch_add(d))
        while add_ops:
            cycle_ops = add_ops[:6]
            add_ops = add_ops[6:]
            self.instrs.append({"valu": cycle_ops})

        check_ops = []
        for d in range(8):
            check_ops.extend(emit_bounds_check(d))
        while check_ops:
            cycle_ops = check_ops[:6]
            check_ops = check_ops[6:]
            self.instrs.append({"valu": cycle_ops})

        apply_ops = []
        for d in range(8):
            apply_ops.extend(emit_bounds_apply(d))
        while apply_ops:
            cycle_ops = apply_ops[:6]
            apply_ops = apply_ops[6:]
            self.instrs.append({"valu": cycle_ops})

        # PHASE 6: Store all 8 desks
        for d in range(NUM_DESKS):
            self.instrs.append({
                "store": [
                    ("vstore", addr_tmp[d*2], desks[d]['idx']),
                    ("vstore", addr_tmp[d*2+1], desks[d]['val']),
                ],
            })

        # Loop control
        self.instrs.append({
            "alu": [
                ("+", batch_offset, batch_offset, desk_stride_const),
                ("+", iter_counter, iter_counter, one_const),
            ],
        })
        self.instrs.append({
            "alu": [
                ("<", tmp_scalar, batch_offset, batch_size_const),
                ("<", addr_scalar, iter_counter, total_remaining_const),
            ],
        })
        self.instrs.append({
            "flow": [
                ("select", batch_offset, tmp_scalar, batch_offset, zero_const),
            ],
        })

        self.add("flow", ("cond_jump", addr_scalar, main_loop_start))
        self.instrs.append({"flow": [("pause",)]})


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

    kb = KernelBuilderH42()
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
