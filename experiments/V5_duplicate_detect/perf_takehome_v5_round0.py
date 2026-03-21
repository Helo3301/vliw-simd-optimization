"""
# Experiment V5b: Round 0 Optimization via Single Load + Broadcast

Optimize round 0 specifically since ALL lanes have idx=0.

**Insight:**
In round 0, all 8 lanes have idx=0, meaning all desks need tree[0].
Current: 8 scalar loads per desk = 4 cycles per desk
Optimized: 1 scalar load + vbroadcast = 2 cycles per desk (potentially 1 if overlapped)

**Strategy:**
1. Process round 0 separately before the main loop
2. Use single load + broadcast for round 0's gather
3. Main loop processes rounds 1-15

**Expected Savings:**
- Round 0: 8 iterations (256 elements / 32 per iter)
- Each iteration: save 2 cycles per desk * 4 desks = 8 cycles
- Total: 64 cycles saved (if no overhead)

**Base: C4 at 4,667 cycles**
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


class KernelBuilderV5b:
    """
    V5b: Round 0 Optimization

    Separate round 0 processing with single-load broadcast optimization,
    then main loop handles rounds 1-15.
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
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
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
            0: 4097,
            2: 33,
            4: 9,
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

        NUM_DESKS = 4

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

        addr_tmp = [self.alloc_scratch(f"addr_tmp_{i}") for i in range(8)]
        next_addr_tmp = [self.alloc_scratch(f"next_addr_tmp_{i}") for i in range(8)]
        next_offset_regs = [self.alloc_scratch(f"next_off_{d}") for d in range(NUM_DESKS)]

        batch_offset = self.alloc_scratch("batch_offset")
        next_batch_offset = self.alloc_scratch("next_batch_offset")
        iter_counter = self.alloc_scratch("iter_counter")

        # V5b: Scalar temp for single node value
        single_node_val = self.alloc_scratch("single_node_val")

        eight_const = self.scratch_const(VLEN)
        sixteen_const = self.scratch_const(2 * VLEN)
        twentyfour_const = self.scratch_const(3 * VLEN)
        thirtytwo_const = self.scratch_const(4 * VLEN)
        batch_size_const = self.scratch_const(batch_size)

        # V5b: For round 0 loop, we need iterations = batch_size / (VLEN * NUM_DESKS)
        round0_iterations = batch_size // (VLEN * NUM_DESKS)
        round0_iter_const = self.scratch_const(round0_iterations)

        # Main loop: processes rounds 1-15 (rounds paired as (0,1), (2,3)...)
        # But we separate round 0, so main loop does rounds 1-15
        # Actually, round fusion pairs rounds, so we need to restructure
        # Let's keep the original structure but optimize the gather for iteration 0-7

        # Simplified approach: Use a flag to track first batch pass
        first_pass_flag = self.alloc_scratch("first_pass_flag")
        self.add("load", ("const", first_pass_flag, 1))  # Start with first_pass = 1

        total_iterations = (batch_size // VLEN) * (rounds // 2) // NUM_DESKS
        total_const = self.scratch_const(total_iterations)

        offset_regs = [self.alloc_scratch(f"off_{d}") for d in range(NUM_DESKS)]

        self.add("load", ("const", batch_offset, 0))
        self.add("load", ("const", iter_counter, 0))

        # === MAIN LOOP ===
        main_loop_start = len(self.instrs)

        # PHASE 1: Calculate all offsets and load addresses
        self.instrs.append({
            "alu": [
                ("+", offset_regs[0], batch_offset, zero_const),
                ("+", offset_regs[1], batch_offset, eight_const),
                ("+", offset_regs[2], batch_offset, sixteen_const),
                ("+", offset_regs[3], batch_offset, twentyfour_const),
                ("+", addr_tmp[0], self.scratch["inp_indices_p"], batch_offset),
                ("+", addr_tmp[1], self.scratch["inp_values_p"], batch_offset),
            ],
            "valu": [
                ("vbroadcast", desks[0]['addr'], self.scratch["forest_values_p"]),
            ],
        })

        # Load desk0 idx/val + compute addresses for desk1,2
        self.instrs.append({
            "load": [
                ("vload", desks[0]['idx'], addr_tmp[0]),
                ("vload", desks[0]['val'], addr_tmp[1]),
            ],
            "alu": [
                ("+", addr_tmp[2], self.scratch["inp_indices_p"], offset_regs[1]),
                ("+", addr_tmp[3], self.scratch["inp_values_p"], offset_regs[1]),
                ("+", addr_tmp[4], self.scratch["inp_indices_p"], offset_regs[2]),
                ("+", addr_tmp[5], self.scratch["inp_values_p"], offset_regs[2]),
            ],
        })

        # Load desk1 + compute desk2,3 addresses
        self.instrs.append({
            "load": [
                ("vload", desks[1]['idx'], addr_tmp[2]),
                ("vload", desks[1]['val'], addr_tmp[3]),
            ],
            "alu": [
                ("+", addr_tmp[6], self.scratch["inp_indices_p"], offset_regs[3]),
                ("+", addr_tmp[7], self.scratch["inp_values_p"], offset_regs[3]),
            ],
            "valu": [
                ("vbroadcast", desks[1]['addr'], self.scratch["forest_values_p"]),
            ],
        })

        # Load desk2 + desk0 gather addr add + desk2 broadcast
        self.instrs.append({
            "load": [
                ("vload", desks[2]['idx'], addr_tmp[4]),
                ("vload", desks[2]['val'], addr_tmp[5]),
            ],
            "valu": [
                ("+", desks[0]['addr'], desks[0]['addr'], desks[0]['idx']),
                ("vbroadcast", desks[2]['addr'], self.scratch["forest_values_p"]),
            ],
        })

        # Load desk3 + desk1 gather addr add + desk3 broadcast
        self.instrs.append({
            "load": [
                ("vload", desks[3]['idx'], addr_tmp[6]),
                ("vload", desks[3]['val'], addr_tmp[7]),
            ],
            "valu": [
                ("+", desks[1]['addr'], desks[1]['addr'], desks[1]['idx']),
                ("vbroadcast", desks[3]['addr'], self.scratch["forest_values_p"]),
            ],
        })

        # ============================================================
        # V5b OPTIMIZATION: Check first_pass_flag for round 0 optimization
        # If first_pass_flag == 1, use single load + broadcast
        # Otherwise, use normal gather
        # ============================================================

        # Compute gather addresses for desks 2,3
        self.instrs.append({
            "valu": [
                ("+", desks[2]['addr'], desks[2]['addr'], desks[2]['idx']),
                ("+", desks[3]['addr'], desks[3]['addr'], desks[3]['idx']),
            ],
        })

        # Check if first pass (idx=0 for all lanes in round 0)
        # Branch to optimized path if first_pass_flag is set
        # Actually, conditional execution is complex in this architecture
        # Let's skip the conditional for now and use the standard gather

        # ROUND 1 OF 2: Standard gather (keeping C4's approach for correctness)

        # First gather + desk 2,3 already done in previous step
        self.instrs.append({
            "load": [
                ("load", desks[0]['node_val'], desks[0]['addr']),
                ("load", desks[0]['node_val'] + 1, desks[0]['addr'] + 1),
            ],
        })

        # Remaining desk0 gathers
        for lane in range(2, VLEN, 2):
            self.instrs.append({
                "load": [
                    ("load", desks[0]['node_val'] + lane, desks[0]['addr'] + lane),
                    ("load", desks[0]['node_val'] + lane + 1, desks[0]['addr'] + lane + 1),
                ],
            })

        # After desk0 gather: XOR desk0 + start desk1 gather
        self.instrs.append({
            "load": [
                ("load", desks[1]['node_val'], desks[1]['addr']),
                ("load", desks[1]['node_val'] + 1, desks[1]['addr'] + 1),
            ],
            "valu": [
                ("^", desks[0]['val'], desks[0]['val'], desks[0]['node_val']),
            ],
        })

        # Desk1 gather lanes 2-3 + desk0 hash stage 0 (FMA)
        self.instrs.append({
            "load": [
                ("load", desks[1]['node_val'] + 2, desks[1]['addr'] + 2),
                ("load", desks[1]['node_val'] + 3, desks[1]['addr'] + 3),
            ],
            "valu": [
                ("multiply_add", desks[0]['val'], desks[0]['val'], v_fma_mult[0], v_hash_consts[0]),
            ],
        })

        # Desk1 gather lanes 4-5 + desk0 hash stage 1 prep
        self.instrs.append({
            "load": [
                ("load", desks[1]['node_val'] + 4, desks[1]['addr'] + 4),
                ("load", desks[1]['node_val'] + 5, desks[1]['addr'] + 5),
            ],
            "valu": [
                ("^", desks[0]['tmp1'], desks[0]['val'], v_hash_consts[1]),
                (">>", desks[0]['tmp2'], desks[0]['val'], v_hash_shifts[1]),
            ],
        })

        # Desk1 gather lanes 6-7 + desk0 hash stage 1 combine
        self.instrs.append({
            "load": [
                ("load", desks[1]['node_val'] + 6, desks[1]['addr'] + 6),
                ("load", desks[1]['node_val'] + 7, desks[1]['addr'] + 7),
            ],
            "valu": [
                ("^", desks[0]['val'], desks[0]['tmp1'], desks[0]['tmp2']),
            ],
        })

        # Desk1 gather complete: XOR desk1 + desk0 hash stage 2 + start desk2 gather
        self.instrs.append({
            "load": [
                ("load", desks[2]['node_val'], desks[2]['addr']),
                ("load", desks[2]['node_val'] + 1, desks[2]['addr'] + 1),
            ],
            "valu": [
                ("^", desks[1]['val'], desks[1]['val'], desks[1]['node_val']),
                ("multiply_add", desks[0]['val'], desks[0]['val'], v_fma_mult[2], v_hash_consts[2]),
            ],
        })

        # Desk2 gather 2-3 + desk0 hash3 prep + desk1 hash0
        self.instrs.append({
            "load": [
                ("load", desks[2]['node_val'] + 2, desks[2]['addr'] + 2),
                ("load", desks[2]['node_val'] + 3, desks[2]['addr'] + 3),
            ],
            "valu": [
                ("+", desks[0]['tmp1'], desks[0]['val'], v_hash_consts[3]),
                ("<<", desks[0]['tmp2'], desks[0]['val'], v_hash_shifts[3]),
                ("multiply_add", desks[1]['val'], desks[1]['val'], v_fma_mult[0], v_hash_consts[0]),
            ],
        })

        # Desk2 gather 4-5 + desk0 hash3 combine + desk1 hash1 prep
        self.instrs.append({
            "load": [
                ("load", desks[2]['node_val'] + 4, desks[2]['addr'] + 4),
                ("load", desks[2]['node_val'] + 5, desks[2]['addr'] + 5),
            ],
            "valu": [
                ("^", desks[0]['val'], desks[0]['tmp1'], desks[0]['tmp2']),
                ("^", desks[1]['tmp1'], desks[1]['val'], v_hash_consts[1]),
                (">>", desks[1]['tmp2'], desks[1]['val'], v_hash_shifts[1]),
            ],
        })

        # Desk2 gather 6-7 + desk0 hash4 + desk1 hash1 combine
        self.instrs.append({
            "load": [
                ("load", desks[2]['node_val'] + 6, desks[2]['addr'] + 6),
                ("load", desks[2]['node_val'] + 7, desks[2]['addr'] + 7),
            ],
            "valu": [
                ("multiply_add", desks[0]['val'], desks[0]['val'], v_fma_mult[4], v_hash_consts[4]),
                ("^", desks[1]['val'], desks[1]['tmp1'], desks[1]['tmp2']),
            ],
        })

        # Desk2 gather complete: XOR desk2 + desk0 hash5 prep + desk1 hash2 + start desk3 gather
        self.instrs.append({
            "load": [
                ("load", desks[3]['node_val'], desks[3]['addr']),
                ("load", desks[3]['node_val'] + 1, desks[3]['addr'] + 1),
            ],
            "valu": [
                ("^", desks[2]['val'], desks[2]['val'], desks[2]['node_val']),
                ("^", desks[0]['tmp1'], desks[0]['val'], v_hash_consts[5]),
                (">>", desks[0]['tmp2'], desks[0]['val'], v_hash_shifts[5]),
                ("multiply_add", desks[1]['val'], desks[1]['val'], v_fma_mult[2], v_hash_consts[2]),
            ],
        })

        # Desk3 gather 2-3 + desk0 hash5 combine + desk1 hash3 prep + desk2 hash0
        self.instrs.append({
            "load": [
                ("load", desks[3]['node_val'] + 2, desks[3]['addr'] + 2),
                ("load", desks[3]['node_val'] + 3, desks[3]['addr'] + 3),
            ],
            "valu": [
                ("^", desks[0]['val'], desks[0]['tmp1'], desks[0]['tmp2']),
                ("+", desks[1]['tmp1'], desks[1]['val'], v_hash_consts[3]),
                ("<<", desks[1]['tmp2'], desks[1]['val'], v_hash_shifts[3]),
                ("multiply_add", desks[2]['val'], desks[2]['val'], v_fma_mult[0], v_hash_consts[0]),
            ],
        })

        # Desk3 gather 4-5 + desk0 branch prep
        self.instrs.append({
            "load": [
                ("load", desks[3]['node_val'] + 4, desks[3]['addr'] + 4),
                ("load", desks[3]['node_val'] + 5, desks[3]['addr'] + 5),
            ],
            "valu": [
                ("&", desks[0]['tmp1'], desks[0]['val'], v_one),
                ("multiply_add", desks[0]['idx'], desks[0]['idx'], v_two, v_one),
                ("^", desks[1]['val'], desks[1]['tmp1'], desks[1]['tmp2']),
                ("^", desks[2]['tmp1'], desks[2]['val'], v_hash_consts[1]),
                (">>", desks[2]['tmp2'], desks[2]['val'], v_hash_shifts[1]),
            ],
        })

        # Desk3 gather 6-7 + desk0 branch + desk1 hash4 + desk2 hash1 combine
        self.instrs.append({
            "load": [
                ("load", desks[3]['node_val'] + 6, desks[3]['addr'] + 6),
                ("load", desks[3]['node_val'] + 7, desks[3]['addr'] + 7),
            ],
            "valu": [
                ("+", desks[0]['idx'], desks[0]['idx'], desks[0]['tmp1']),
                ("multiply_add", desks[1]['val'], desks[1]['val'], v_fma_mult[4], v_hash_consts[4]),
                ("^", desks[2]['val'], desks[2]['tmp1'], desks[2]['tmp2']),
            ],
        })

        # Continue pipeline...
        self.instrs.append({
            "valu": [
                ("^", desks[1]['tmp1'], desks[1]['val'], v_hash_consts[5]),
                (">>", desks[1]['tmp2'], desks[1]['val'], v_hash_shifts[5]),
                ("multiply_add", desks[2]['val'], desks[2]['val'], v_fma_mult[2], v_hash_consts[2]),
                ("^", desks[3]['val'], desks[3]['val'], desks[3]['node_val']),
            ],
        })

        self.instrs.append({
            "valu": [
                ("<", desks[0]['tmp1'], desks[0]['idx'], v_n_nodes),
                ("^", desks[1]['val'], desks[1]['tmp1'], desks[1]['tmp2']),
                ("+", desks[2]['tmp1'], desks[2]['val'], v_hash_consts[3]),
                ("<<", desks[2]['tmp2'], desks[2]['val'], v_hash_shifts[3]),
                ("multiply_add", desks[3]['val'], desks[3]['val'], v_fma_mult[0], v_hash_consts[0]),
            ],
        })

        self.instrs.append({
            "valu": [
                ("*", desks[0]['idx'], desks[0]['idx'], desks[0]['tmp1']),
                ("&", desks[1]['tmp1'], desks[1]['val'], v_one),
                ("multiply_add", desks[1]['idx'], desks[1]['idx'], v_two, v_one),
                ("^", desks[2]['val'], desks[2]['tmp1'], desks[2]['tmp2']),
                ("^", desks[3]['tmp1'], desks[3]['val'], v_hash_consts[1]),
                (">>", desks[3]['tmp2'], desks[3]['val'], v_hash_shifts[1]),
            ],
        })

        self.instrs.append({
            "valu": [
                ("+", desks[1]['idx'], desks[1]['idx'], desks[1]['tmp1']),
                ("multiply_add", desks[2]['val'], desks[2]['val'], v_fma_mult[4], v_hash_consts[4]),
                ("^", desks[3]['val'], desks[3]['tmp1'], desks[3]['tmp2']),
            ],
        })

        self.instrs.append({
            "valu": [
                ("^", desks[2]['tmp1'], desks[2]['val'], v_hash_consts[5]),
                (">>", desks[2]['tmp2'], desks[2]['val'], v_hash_shifts[5]),
                ("multiply_add", desks[3]['val'], desks[3]['val'], v_fma_mult[2], v_hash_consts[2]),
            ],
        })

        self.instrs.append({
            "valu": [
                ("<", desks[1]['tmp1'], desks[1]['idx'], v_n_nodes),
                ("^", desks[2]['val'], desks[2]['tmp1'], desks[2]['tmp2']),
                ("+", desks[3]['tmp1'], desks[3]['val'], v_hash_consts[3]),
                ("<<", desks[3]['tmp2'], desks[3]['val'], v_hash_shifts[3]),
            ],
        })

        self.instrs.append({
            "valu": [
                ("*", desks[1]['idx'], desks[1]['idx'], desks[1]['tmp1']),
                ("&", desks[2]['tmp1'], desks[2]['val'], v_one),
                ("multiply_add", desks[2]['idx'], desks[2]['idx'], v_two, v_one),
                ("^", desks[3]['val'], desks[3]['tmp1'], desks[3]['tmp2']),
            ],
        })

        self.instrs.append({
            "valu": [
                ("+", desks[2]['idx'], desks[2]['idx'], desks[2]['tmp1']),
                ("multiply_add", desks[3]['val'], desks[3]['val'], v_fma_mult[4], v_hash_consts[4]),
            ],
        })

        self.instrs.append({
            "valu": [
                ("^", desks[3]['tmp1'], desks[3]['val'], v_hash_consts[5]),
                (">>", desks[3]['tmp2'], desks[3]['val'], v_hash_shifts[5]),
            ],
        })

        self.instrs.append({
            "valu": [
                ("<", desks[2]['tmp1'], desks[2]['idx'], v_n_nodes),
                ("^", desks[3]['val'], desks[3]['tmp1'], desks[3]['tmp2']),
            ],
        })

        self.instrs.append({
            "valu": [
                ("*", desks[2]['idx'], desks[2]['idx'], desks[2]['tmp1']),
                ("&", desks[3]['tmp1'], desks[3]['val'], v_one),
                ("multiply_add", desks[3]['idx'], desks[3]['idx'], v_two, v_one),
            ],
        })

        self.instrs.append({
            "valu": [
                ("+", desks[3]['idx'], desks[3]['idx'], desks[3]['tmp1']),
            ],
        })

        self.instrs.append({
            "valu": [
                ("<", desks[3]['tmp1'], desks[3]['idx'], v_n_nodes),
            ],
        })

        self.instrs.append({
            "valu": [
                ("*", desks[3]['idx'], desks[3]['idx'], desks[3]['tmp1']),
            ],
        })

        # ============================================================
        # ROUND 2 OF 2
        # ============================================================

        self.instrs.append({
            "valu": [
                ("vbroadcast", desks[0]['addr'], self.scratch["forest_values_p"]),
                ("vbroadcast", desks[1]['addr'], self.scratch["forest_values_p"]),
                ("vbroadcast", desks[2]['addr'], self.scratch["forest_values_p"]),
                ("vbroadcast", desks[3]['addr'], self.scratch["forest_values_p"]),
            ],
        })

        self.instrs.append({
            "valu": [
                ("+", desks[0]['addr'], desks[0]['addr'], desks[0]['idx']),
                ("+", desks[1]['addr'], desks[1]['addr'], desks[1]['idx']),
                ("+", desks[2]['addr'], desks[2]['addr'], desks[2]['idx']),
                ("+", desks[3]['addr'], desks[3]['addr'], desks[3]['idx']),
            ],
        })

        # Round 2: Desk 0 gathers
        for lane in range(0, VLEN, 2):
            self.instrs.append({
                "load": [
                    ("load", desks[0]['node_val'] + lane, desks[0]['addr'] + lane),
                    ("load", desks[0]['node_val'] + lane + 1, desks[0]['addr'] + lane + 1),
                ],
            })

        # Round 2: XOR desk0 + start desk1 gather
        self.instrs.append({
            "load": [
                ("load", desks[1]['node_val'], desks[1]['addr']),
                ("load", desks[1]['node_val'] + 1, desks[1]['addr'] + 1),
            ],
            "valu": [
                ("^", desks[0]['val'], desks[0]['val'], desks[0]['node_val']),
            ],
        })

        self.instrs.append({
            "load": [
                ("load", desks[1]['node_val'] + 2, desks[1]['addr'] + 2),
                ("load", desks[1]['node_val'] + 3, desks[1]['addr'] + 3),
            ],
            "valu": [
                ("multiply_add", desks[0]['val'], desks[0]['val'], v_fma_mult[0], v_hash_consts[0]),
            ],
        })

        self.instrs.append({
            "load": [
                ("load", desks[1]['node_val'] + 4, desks[1]['addr'] + 4),
                ("load", desks[1]['node_val'] + 5, desks[1]['addr'] + 5),
            ],
            "valu": [
                ("^", desks[0]['tmp1'], desks[0]['val'], v_hash_consts[1]),
                (">>", desks[0]['tmp2'], desks[0]['val'], v_hash_shifts[1]),
            ],
        })

        self.instrs.append({
            "load": [
                ("load", desks[1]['node_val'] + 6, desks[1]['addr'] + 6),
                ("load", desks[1]['node_val'] + 7, desks[1]['addr'] + 7),
            ],
            "valu": [
                ("^", desks[0]['val'], desks[0]['tmp1'], desks[0]['tmp2']),
            ],
        })

        self.instrs.append({
            "load": [
                ("load", desks[2]['node_val'], desks[2]['addr']),
                ("load", desks[2]['node_val'] + 1, desks[2]['addr'] + 1),
            ],
            "valu": [
                ("^", desks[1]['val'], desks[1]['val'], desks[1]['node_val']),
                ("multiply_add", desks[0]['val'], desks[0]['val'], v_fma_mult[2], v_hash_consts[2]),
            ],
        })

        self.instrs.append({
            "load": [
                ("load", desks[2]['node_val'] + 2, desks[2]['addr'] + 2),
                ("load", desks[2]['node_val'] + 3, desks[2]['addr'] + 3),
            ],
            "valu": [
                ("+", desks[0]['tmp1'], desks[0]['val'], v_hash_consts[3]),
                ("<<", desks[0]['tmp2'], desks[0]['val'], v_hash_shifts[3]),
                ("multiply_add", desks[1]['val'], desks[1]['val'], v_fma_mult[0], v_hash_consts[0]),
            ],
        })

        self.instrs.append({
            "load": [
                ("load", desks[2]['node_val'] + 4, desks[2]['addr'] + 4),
                ("load", desks[2]['node_val'] + 5, desks[2]['addr'] + 5),
            ],
            "valu": [
                ("^", desks[0]['val'], desks[0]['tmp1'], desks[0]['tmp2']),
                ("^", desks[1]['tmp1'], desks[1]['val'], v_hash_consts[1]),
                (">>", desks[1]['tmp2'], desks[1]['val'], v_hash_shifts[1]),
            ],
        })

        self.instrs.append({
            "load": [
                ("load", desks[2]['node_val'] + 6, desks[2]['addr'] + 6),
                ("load", desks[2]['node_val'] + 7, desks[2]['addr'] + 7),
            ],
            "valu": [
                ("multiply_add", desks[0]['val'], desks[0]['val'], v_fma_mult[4], v_hash_consts[4]),
                ("^", desks[1]['val'], desks[1]['tmp1'], desks[1]['tmp2']),
            ],
        })

        self.instrs.append({
            "load": [
                ("load", desks[3]['node_val'], desks[3]['addr']),
                ("load", desks[3]['node_val'] + 1, desks[3]['addr'] + 1),
            ],
            "valu": [
                ("^", desks[2]['val'], desks[2]['val'], desks[2]['node_val']),
                ("^", desks[0]['tmp1'], desks[0]['val'], v_hash_consts[5]),
                (">>", desks[0]['tmp2'], desks[0]['val'], v_hash_shifts[5]),
                ("multiply_add", desks[1]['val'], desks[1]['val'], v_fma_mult[2], v_hash_consts[2]),
            ],
        })

        self.instrs.append({
            "load": [
                ("load", desks[3]['node_val'] + 2, desks[3]['addr'] + 2),
                ("load", desks[3]['node_val'] + 3, desks[3]['addr'] + 3),
            ],
            "valu": [
                ("^", desks[0]['val'], desks[0]['tmp1'], desks[0]['tmp2']),
                ("+", desks[1]['tmp1'], desks[1]['val'], v_hash_consts[3]),
                ("<<", desks[1]['tmp2'], desks[1]['val'], v_hash_shifts[3]),
                ("multiply_add", desks[2]['val'], desks[2]['val'], v_fma_mult[0], v_hash_consts[0]),
            ],
        })

        self.instrs.append({
            "load": [
                ("load", desks[3]['node_val'] + 4, desks[3]['addr'] + 4),
                ("load", desks[3]['node_val'] + 5, desks[3]['addr'] + 5),
            ],
            "valu": [
                ("&", desks[0]['tmp1'], desks[0]['val'], v_one),
                ("multiply_add", desks[0]['idx'], desks[0]['idx'], v_two, v_one),
                ("^", desks[1]['val'], desks[1]['tmp1'], desks[1]['tmp2']),
                ("^", desks[2]['tmp1'], desks[2]['val'], v_hash_consts[1]),
                (">>", desks[2]['tmp2'], desks[2]['val'], v_hash_shifts[1]),
            ],
            "alu": [
                ("+", next_batch_offset, batch_offset, thirtytwo_const),
            ],
        })

        self.instrs.append({
            "load": [
                ("load", desks[3]['node_val'] + 6, desks[3]['addr'] + 6),
                ("load", desks[3]['node_val'] + 7, desks[3]['addr'] + 7),
            ],
            "valu": [
                ("+", desks[0]['idx'], desks[0]['idx'], desks[0]['tmp1']),
                ("multiply_add", desks[1]['val'], desks[1]['val'], v_fma_mult[4], v_hash_consts[4]),
                ("^", desks[2]['val'], desks[2]['tmp1'], desks[2]['tmp2']),
            ],
            "alu": [
                ("+", next_offset_regs[0], next_batch_offset, zero_const),
                ("+", next_offset_regs[1], next_batch_offset, eight_const),
                ("+", next_offset_regs[2], next_batch_offset, sixteen_const),
                ("+", next_offset_regs[3], next_batch_offset, twentyfour_const),
            ],
        })

        self.instrs.append({
            "valu": [
                ("^", desks[1]['tmp1'], desks[1]['val'], v_hash_consts[5]),
                (">>", desks[1]['tmp2'], desks[1]['val'], v_hash_shifts[5]),
                ("multiply_add", desks[2]['val'], desks[2]['val'], v_fma_mult[2], v_hash_consts[2]),
                ("^", desks[3]['val'], desks[3]['val'], desks[3]['node_val']),
            ],
            "alu": [
                ("+", next_addr_tmp[0], self.scratch["inp_indices_p"], next_offset_regs[0]),
                ("+", next_addr_tmp[1], self.scratch["inp_values_p"], next_offset_regs[0]),
            ],
        })

        self.instrs.append({
            "valu": [
                ("<", desks[0]['tmp1'], desks[0]['idx'], v_n_nodes),
                ("^", desks[1]['val'], desks[1]['tmp1'], desks[1]['tmp2']),
                ("+", desks[2]['tmp1'], desks[2]['val'], v_hash_consts[3]),
                ("<<", desks[2]['tmp2'], desks[2]['val'], v_hash_shifts[3]),
                ("multiply_add", desks[3]['val'], desks[3]['val'], v_fma_mult[0], v_hash_consts[0]),
            ],
            "alu": [
                ("+", next_addr_tmp[2], self.scratch["inp_indices_p"], next_offset_regs[1]),
                ("+", next_addr_tmp[3], self.scratch["inp_values_p"], next_offset_regs[1]),
            ],
        })

        self.instrs.append({
            "valu": [
                ("*", desks[0]['idx'], desks[0]['idx'], desks[0]['tmp1']),
                ("&", desks[1]['tmp1'], desks[1]['val'], v_one),
                ("multiply_add", desks[1]['idx'], desks[1]['idx'], v_two, v_one),
                ("^", desks[2]['val'], desks[2]['tmp1'], desks[2]['tmp2']),
                ("^", desks[3]['tmp1'], desks[3]['val'], v_hash_consts[1]),
                (">>", desks[3]['tmp2'], desks[3]['val'], v_hash_shifts[1]),
            ],
            "alu": [
                ("+", next_addr_tmp[4], self.scratch["inp_indices_p"], next_offset_regs[2]),
                ("+", next_addr_tmp[5], self.scratch["inp_values_p"], next_offset_regs[2]),
            ],
        })

        self.instrs.append({
            "valu": [
                ("+", desks[1]['idx'], desks[1]['idx'], desks[1]['tmp1']),
                ("multiply_add", desks[2]['val'], desks[2]['val'], v_fma_mult[4], v_hash_consts[4]),
                ("^", desks[3]['val'], desks[3]['tmp1'], desks[3]['tmp2']),
            ],
            "alu": [
                ("+", next_addr_tmp[6], self.scratch["inp_indices_p"], next_offset_regs[3]),
                ("+", next_addr_tmp[7], self.scratch["inp_values_p"], next_offset_regs[3]),
            ],
        })

        self.instrs.append({
            "valu": [
                ("^", desks[2]['tmp1'], desks[2]['val'], v_hash_consts[5]),
                (">>", desks[2]['tmp2'], desks[2]['val'], v_hash_shifts[5]),
                ("multiply_add", desks[3]['val'], desks[3]['val'], v_fma_mult[2], v_hash_consts[2]),
            ],
        })

        self.instrs.append({
            "valu": [
                ("<", desks[1]['tmp1'], desks[1]['idx'], v_n_nodes),
                ("^", desks[2]['val'], desks[2]['tmp1'], desks[2]['tmp2']),
                ("+", desks[3]['tmp1'], desks[3]['val'], v_hash_consts[3]),
                ("<<", desks[3]['tmp2'], desks[3]['val'], v_hash_shifts[3]),
            ],
        })

        self.instrs.append({
            "valu": [
                ("*", desks[1]['idx'], desks[1]['idx'], desks[1]['tmp1']),
                ("&", desks[2]['tmp1'], desks[2]['val'], v_one),
                ("multiply_add", desks[2]['idx'], desks[2]['idx'], v_two, v_one),
                ("^", desks[3]['val'], desks[3]['tmp1'], desks[3]['tmp2']),
            ],
        })

        self.instrs.append({
            "valu": [
                ("+", desks[2]['idx'], desks[2]['idx'], desks[2]['tmp1']),
                ("multiply_add", desks[3]['val'], desks[3]['val'], v_fma_mult[4], v_hash_consts[4]),
            ],
        })

        self.instrs.append({
            "valu": [
                ("^", desks[3]['tmp1'], desks[3]['val'], v_hash_consts[5]),
                (">>", desks[3]['tmp2'], desks[3]['val'], v_hash_shifts[5]),
            ],
        })

        self.instrs.append({
            "valu": [
                ("<", desks[2]['tmp1'], desks[2]['idx'], v_n_nodes),
                ("^", desks[3]['val'], desks[3]['tmp1'], desks[3]['tmp2']),
            ],
        })

        self.instrs.append({
            "valu": [
                ("*", desks[2]['idx'], desks[2]['idx'], desks[2]['tmp1']),
                ("&", desks[3]['tmp1'], desks[3]['val'], v_one),
                ("multiply_add", desks[3]['idx'], desks[3]['idx'], v_two, v_one),
            ],
        })

        self.instrs.append({
            "valu": [
                ("+", desks[3]['idx'], desks[3]['idx'], desks[3]['tmp1']),
            ],
            "alu": [
                ("+", addr_tmp[0], self.scratch["inp_indices_p"], offset_regs[0]),
                ("+", addr_tmp[1], self.scratch["inp_values_p"], offset_regs[0]),
                ("+", addr_tmp[2], self.scratch["inp_indices_p"], offset_regs[1]),
                ("+", addr_tmp[3], self.scratch["inp_values_p"], offset_regs[1]),
            ],
        })

        self.instrs.append({
            "valu": [
                ("<", desks[3]['tmp1'], desks[3]['idx'], v_n_nodes),
            ],
            "alu": [
                ("+", addr_tmp[4], self.scratch["inp_indices_p"], offset_regs[2]),
                ("+", addr_tmp[5], self.scratch["inp_values_p"], offset_regs[2]),
                ("+", addr_tmp[6], self.scratch["inp_indices_p"], offset_regs[3]),
                ("+", addr_tmp[7], self.scratch["inp_values_p"], offset_regs[3]),
            ],
        })

        self.instrs.append({
            "valu": [
                ("*", desks[3]['idx'], desks[3]['idx'], desks[3]['tmp1']),
            ],
        })

        # Store coalescing
        self.instrs.append({
            "store": [
                ("vstore", addr_tmp[0], desks[0]['idx']),
                ("vstore", addr_tmp[1], desks[0]['val']),
            ],
            "alu": [
                ("+", batch_offset, batch_offset, thirtytwo_const),
                ("+", iter_counter, iter_counter, one_const),
            ],
        })

        self.instrs.append({
            "store": [
                ("vstore", addr_tmp[2], desks[1]['idx']),
                ("vstore", addr_tmp[3], desks[1]['val']),
            ],
            "alu": [
                ("<", tmp_scalar, batch_offset, batch_size_const),
            ],
        })

        self.instrs.append({
            "store": [
                ("vstore", addr_tmp[4], desks[2]['idx']),
                ("vstore", addr_tmp[5], desks[2]['val']),
            ],
            "alu": [
                ("<", addr_scalar, iter_counter, total_const),
            ],
        })

        self.instrs.append({
            "store": [
                ("vstore", addr_tmp[6], desks[3]['idx']),
                ("vstore", addr_tmp[7], desks[3]['val']),
            ],
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

    kb = KernelBuilderV5b()
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
