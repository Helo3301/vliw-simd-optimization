"""
# Experiment H31: Single-Batch Full Fusion

**Goal:** Process 32 elements (4 desks x 8 VLEN) through ALL 16 rounds without intermediate stores.

**Strategy:**
1. For the FIRST batch only (elements 0-31), process all 16 rounds in-register:
   - Load idx[0-31] and val[0-31] once at start
   - Loop 16 times: gather -> XOR -> hash -> branch (no intermediate store)
   - Store idx[0-31] and val[0-31] once at end
2. For remaining batches (32-255), use standard C4 approach

**Key insight:** By eliminating 15 out of 16 store/load pairs for the first batch,
we should see significant cycle reduction if register pressure doesn't cause spills.

**Expected:**
- First batch: ~400-600 cycles for 32 elements if theory correct
- First batch: ~800+ cycles if register pressure causes issues
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


class KernelBuilderH31:
    """
    H31: Single-Batch Full Fusion

    Process the first batch (32 elements) through all 16 rounds without intermediate stores.
    Remaining batches use standard C4 approach (2-round fusion).
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
        """
        H31: Single-Batch Full Fusion kernel.
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

        # Use 4 desks for interleaving
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
        offset_regs = [self.alloc_scratch(f"off_{d}") for d in range(NUM_DESKS)]

        batch_offset = self.alloc_scratch("batch_offset")
        iter_counter = self.alloc_scratch("iter_counter")
        round_counter = self.alloc_scratch("round_counter")

        eight_const = self.scratch_const(VLEN)
        sixteen_const = self.scratch_const(2 * VLEN)
        twentyfour_const = self.scratch_const(3 * VLEN)
        thirtytwo_const = self.scratch_const(4 * VLEN)
        batch_size_const = self.scratch_const(batch_size)
        rounds_const = self.scratch_const(rounds)

        self.add("load", ("const", batch_offset, 0))
        self.add("load", ("const", iter_counter, 0))

        # ============================================================
        # PHASE 1: FIRST BATCH - FULL 16-ROUND FUSION
        # Load once, process 16 rounds, store once
        # ============================================================

        # Calculate offsets and load initial data for first batch
        self.instrs.append({
            "alu": [
                ("+", offset_regs[0], batch_offset, zero_const),
                ("+", offset_regs[1], batch_offset, eight_const),
                ("+", offset_regs[2], batch_offset, sixteen_const),
                ("+", offset_regs[3], batch_offset, twentyfour_const),
                ("+", addr_tmp[0], self.scratch["inp_indices_p"], batch_offset),
                ("+", addr_tmp[1], self.scratch["inp_values_p"], batch_offset),
            ],
        })

        # Compute remaining load addresses
        self.instrs.append({
            "alu": [
                ("+", addr_tmp[2], self.scratch["inp_indices_p"], offset_regs[1]),
                ("+", addr_tmp[3], self.scratch["inp_values_p"], offset_regs[1]),
                ("+", addr_tmp[4], self.scratch["inp_indices_p"], offset_regs[2]),
                ("+", addr_tmp[5], self.scratch["inp_values_p"], offset_regs[2]),
                ("+", addr_tmp[6], self.scratch["inp_indices_p"], offset_regs[3]),
                ("+", addr_tmp[7], self.scratch["inp_values_p"], offset_regs[3]),
            ],
        })

        # Load all 4 desks (idx and val)
        self.instrs.append({
            "load": [
                ("vload", desks[0]['idx'], addr_tmp[0]),
                ("vload", desks[0]['val'], addr_tmp[1]),
            ],
        })
        self.instrs.append({
            "load": [
                ("vload", desks[1]['idx'], addr_tmp[2]),
                ("vload", desks[1]['val'], addr_tmp[3]),
            ],
        })
        self.instrs.append({
            "load": [
                ("vload", desks[2]['idx'], addr_tmp[4]),
                ("vload", desks[2]['val'], addr_tmp[5]),
            ],
        })
        self.instrs.append({
            "load": [
                ("vload", desks[3]['idx'], addr_tmp[6]),
                ("vload", desks[3]['val'], addr_tmp[7]),
            ],
        })

        # Initialize round counter for first batch
        self.add("load", ("const", round_counter, 0))

        # === FIRST BATCH ROUND LOOP START ===
        first_batch_loop_start = len(self.instrs)

        # Compute gather addresses for all desks
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

        # Gather for desk 0
        for lane in range(0, VLEN, 2):
            self.instrs.append({
                "load": [
                    ("load", desks[0]['node_val'] + lane, desks[0]['addr'] + lane),
                    ("load", desks[0]['node_val'] + lane + 1, desks[0]['addr'] + lane + 1),
                ],
            })

        # XOR desk0 + start desk1 gather
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

        # Desk2 gather 6-7 + desk0 hash4 (FMA) + desk1 hash1 combine
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

        # Desk3 gather 4-5 + desk0 branch prep (AND, FMA)
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

        # Desk3 gather 6-7 + desk0 branch add + desk1 hash4 + desk2 hash1 combine
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

        # desk0 bounds check + desk1 hash5 prep + desk2 hash2 + desk3 XOR
        self.instrs.append({
            "valu": [
                ("^", desks[1]['tmp1'], desks[1]['val'], v_hash_consts[5]),
                (">>", desks[1]['tmp2'], desks[1]['val'], v_hash_shifts[5]),
                ("multiply_add", desks[2]['val'], desks[2]['val'], v_fma_mult[2], v_hash_consts[2]),
                ("^", desks[3]['val'], desks[3]['val'], desks[3]['node_val']),
            ],
        })

        # Desk0 bounds + desk1 hash5 combine + desk2 hash3 prep + desk3 hash0
        self.instrs.append({
            "valu": [
                ("<", desks[0]['tmp1'], desks[0]['idx'], v_n_nodes),
                ("^", desks[1]['val'], desks[1]['tmp1'], desks[1]['tmp2']),
                ("+", desks[2]['tmp1'], desks[2]['val'], v_hash_consts[3]),
                ("<<", desks[2]['tmp2'], desks[2]['val'], v_hash_shifts[3]),
                ("multiply_add", desks[3]['val'], desks[3]['val'], v_fma_mult[0], v_hash_consts[0]),
            ],
        })

        # Desk0 vselect bypass + desk1 branch prep with FMA
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

        # Desk1 branch add + desk2 hash4 + desk3 hash1 combine
        self.instrs.append({
            "valu": [
                ("+", desks[1]['idx'], desks[1]['idx'], desks[1]['tmp1']),
                ("multiply_add", desks[2]['val'], desks[2]['val'], v_fma_mult[4], v_hash_consts[4]),
                ("^", desks[3]['val'], desks[3]['tmp1'], desks[3]['tmp2']),
            ],
        })

        # desk2 hash5 prep + desk3 hash2
        self.instrs.append({
            "valu": [
                ("^", desks[2]['tmp1'], desks[2]['val'], v_hash_consts[5]),
                (">>", desks[2]['tmp2'], desks[2]['val'], v_hash_shifts[5]),
                ("multiply_add", desks[3]['val'], desks[3]['val'], v_fma_mult[2], v_hash_consts[2]),
            ],
        })

        # Desk1 bounds + desk2 hash5 combine + desk3 hash3 prep
        self.instrs.append({
            "valu": [
                ("<", desks[1]['tmp1'], desks[1]['idx'], v_n_nodes),
                ("^", desks[2]['val'], desks[2]['tmp1'], desks[2]['tmp2']),
                ("+", desks[3]['tmp1'], desks[3]['val'], v_hash_consts[3]),
                ("<<", desks[3]['tmp2'], desks[3]['val'], v_hash_shifts[3]),
            ],
        })

        # Desk1 vselect bypass + desk2 branch prep with FMA + desk3 hash3 combine
        self.instrs.append({
            "valu": [
                ("*", desks[1]['idx'], desks[1]['idx'], desks[1]['tmp1']),
                ("&", desks[2]['tmp1'], desks[2]['val'], v_one),
                ("multiply_add", desks[2]['idx'], desks[2]['idx'], v_two, v_one),
                ("^", desks[3]['val'], desks[3]['tmp1'], desks[3]['tmp2']),
            ],
        })

        # Desk2 branch add + desk3 hash4
        self.instrs.append({
            "valu": [
                ("+", desks[2]['idx'], desks[2]['idx'], desks[2]['tmp1']),
                ("multiply_add", desks[3]['val'], desks[3]['val'], v_fma_mult[4], v_hash_consts[4]),
            ],
        })

        # desk3 hash5 prep
        self.instrs.append({
            "valu": [
                ("^", desks[3]['tmp1'], desks[3]['val'], v_hash_consts[5]),
                (">>", desks[3]['tmp2'], desks[3]['val'], v_hash_shifts[5]),
            ],
        })

        # Desk2 bounds + desk3 hash5 combine
        self.instrs.append({
            "valu": [
                ("<", desks[2]['tmp1'], desks[2]['idx'], v_n_nodes),
                ("^", desks[3]['val'], desks[3]['tmp1'], desks[3]['tmp2']),
            ],
        })

        # Desk2 vselect bypass + desk3 branch prep with FMA
        self.instrs.append({
            "valu": [
                ("*", desks[2]['idx'], desks[2]['idx'], desks[2]['tmp1']),
                ("&", desks[3]['tmp1'], desks[3]['val'], v_one),
                ("multiply_add", desks[3]['idx'], desks[3]['idx'], v_two, v_one),
            ],
        })

        # Desk3 branch add
        self.instrs.append({
            "valu": [
                ("+", desks[3]['idx'], desks[3]['idx'], desks[3]['tmp1']),
            ],
        })

        # Desk3 bounds
        self.instrs.append({
            "valu": [
                ("<", desks[3]['tmp1'], desks[3]['idx'], v_n_nodes),
            ],
        })

        # Desk3 vselect bypass + increment round counter
        self.instrs.append({
            "valu": [
                ("*", desks[3]['idx'], desks[3]['idx'], desks[3]['tmp1']),
            ],
            "alu": [
                ("+", round_counter, round_counter, one_const),
            ],
        })

        # Check if we need more rounds (round_counter < 16)
        self.instrs.append({
            "alu": [
                ("<", tmp_scalar, round_counter, rounds_const),
            ],
        })

        # Jump back if more rounds needed
        self.add("flow", ("cond_jump", tmp_scalar, first_batch_loop_start))

        # === FIRST BATCH LOOP END - Store results ===
        # Compute store addresses
        self.instrs.append({
            "alu": [
                ("+", addr_tmp[0], self.scratch["inp_indices_p"], offset_regs[0]),
                ("+", addr_tmp[1], self.scratch["inp_values_p"], offset_regs[0]),
                ("+", addr_tmp[2], self.scratch["inp_indices_p"], offset_regs[1]),
                ("+", addr_tmp[3], self.scratch["inp_values_p"], offset_regs[1]),
                ("+", addr_tmp[4], self.scratch["inp_indices_p"], offset_regs[2]),
                ("+", addr_tmp[5], self.scratch["inp_values_p"], offset_regs[2]),
            ],
        })
        self.instrs.append({
            "alu": [
                ("+", addr_tmp[6], self.scratch["inp_indices_p"], offset_regs[3]),
                ("+", addr_tmp[7], self.scratch["inp_values_p"], offset_regs[3]),
            ],
        })

        # Store all desks
        self.instrs.append({
            "store": [
                ("vstore", addr_tmp[0], desks[0]['idx']),
                ("vstore", addr_tmp[1], desks[0]['val']),
            ],
        })
        self.instrs.append({
            "store": [
                ("vstore", addr_tmp[2], desks[1]['idx']),
                ("vstore", addr_tmp[3], desks[1]['val']),
            ],
        })
        self.instrs.append({
            "store": [
                ("vstore", addr_tmp[4], desks[2]['idx']),
                ("vstore", addr_tmp[5], desks[2]['val']),
            ],
        })
        self.instrs.append({
            "store": [
                ("vstore", addr_tmp[6], desks[3]['idx']),
                ("vstore", addr_tmp[7], desks[3]['val']),
            ],
        })

        # ============================================================
        # PHASE 2: REMAINING BATCHES (32-255) - Use C4 approach
        # Standard 2-round fusion with intermediate stores
        # ============================================================

        # Move batch_offset to next batch
        self.add("load", ("const", batch_offset, 32))  # Start at element 32
        self.add("load", ("const", iter_counter, 0))

        # For remaining batches, use C4-style 2-round fusion
        # Total iterations = ((batch_size - 32) // VLEN) * (rounds // 2) // NUM_DESKS
        # = (224 / 8) * 8 / 4 = 28 * 8 / 4 = 56
        remaining_batch_size = batch_size - 32
        total_iterations = (remaining_batch_size // VLEN) * (rounds // 2) // NUM_DESKS
        total_const = self.scratch_const(total_iterations)

        # Pre-allocate address registers for next iteration
        next_addr_tmp = [self.alloc_scratch(f"next_addr_tmp_{i}") for i in range(8)]
        next_offset_regs = [self.alloc_scratch(f"next_off_{d}") for d in range(NUM_DESKS)]
        next_batch_offset = self.alloc_scratch("next_batch_offset")

        # === MAIN LOOP for remaining batches ===
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

        # Load desk1 + compute desk2,3 addresses + desk1 gather addr broadcast
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
        # ROUND 1 OF 2: INTERLEAVED GATHER + HASH PIPELINE
        # ============================================================

        # First gather + desk 2,3 address add
        self.instrs.append({
            "load": [
                ("load", desks[0]['node_val'], desks[0]['addr']),
                ("load", desks[0]['node_val'] + 1, desks[0]['addr'] + 1),
            ],
            "valu": [
                ("+", desks[2]['addr'], desks[2]['addr'], desks[2]['idx']),
                ("+", desks[3]['addr'], desks[3]['addr'], desks[3]['idx']),
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

        # After desk0 gather complete: XOR desk0 + start desk1 gather
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

        # Desk1 gather complete: XOR desk1 + desk0 hash stage 2 (FMA) + start desk2 gather
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

        # Desk2 gather 6-7 + desk0 hash4 (FMA) + desk1 hash1 combine
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

        # Desk3 gather 6-7 + desk0 branch add + desk1 hash4 + desk2 hash1 combine
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

        # desk0 branch done, continue with hashes + desk3 XOR
        self.instrs.append({
            "valu": [
                ("^", desks[1]['tmp1'], desks[1]['val'], v_hash_consts[5]),
                (">>", desks[1]['tmp2'], desks[1]['val'], v_hash_shifts[5]),
                ("multiply_add", desks[2]['val'], desks[2]['val'], v_fma_mult[2], v_hash_consts[2]),
                ("^", desks[3]['val'], desks[3]['val'], desks[3]['node_val']),
            ],
        })

        # Desk0 bounds check + desk1 hash5 combine + desk2 hash3 prep + desk3 hash0
        self.instrs.append({
            "valu": [
                ("<", desks[0]['tmp1'], desks[0]['idx'], v_n_nodes),
                ("^", desks[1]['val'], desks[1]['tmp1'], desks[1]['tmp2']),
                ("+", desks[2]['tmp1'], desks[2]['val'], v_hash_consts[3]),
                ("<<", desks[2]['tmp2'], desks[2]['val'], v_hash_shifts[3]),
                ("multiply_add", desks[3]['val'], desks[3]['val'], v_fma_mult[0], v_hash_consts[0]),
            ],
        })

        # Desk0 vselect bypass + desk1 branch prep with FMA
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

        # Desk1 branch add + desk2 hash4 + desk3 hash1 combine
        self.instrs.append({
            "valu": [
                ("+", desks[1]['idx'], desks[1]['idx'], desks[1]['tmp1']),
                ("multiply_add", desks[2]['val'], desks[2]['val'], v_fma_mult[4], v_hash_consts[4]),
                ("^", desks[3]['val'], desks[3]['tmp1'], desks[3]['tmp2']),
            ],
        })

        # desk2 hash5 prep + desk3 hash2
        self.instrs.append({
            "valu": [
                ("^", desks[2]['tmp1'], desks[2]['val'], v_hash_consts[5]),
                (">>", desks[2]['tmp2'], desks[2]['val'], v_hash_shifts[5]),
                ("multiply_add", desks[3]['val'], desks[3]['val'], v_fma_mult[2], v_hash_consts[2]),
            ],
        })

        # Desk1 bounds + desk2 hash5 combine + desk3 hash3 prep
        self.instrs.append({
            "valu": [
                ("<", desks[1]['tmp1'], desks[1]['idx'], v_n_nodes),
                ("^", desks[2]['val'], desks[2]['tmp1'], desks[2]['tmp2']),
                ("+", desks[3]['tmp1'], desks[3]['val'], v_hash_consts[3]),
                ("<<", desks[3]['tmp2'], desks[3]['val'], v_hash_shifts[3]),
            ],
        })

        # Desk1 vselect bypass + desk2 branch prep with FMA + desk3 hash3 combine
        self.instrs.append({
            "valu": [
                ("*", desks[1]['idx'], desks[1]['idx'], desks[1]['tmp1']),
                ("&", desks[2]['tmp1'], desks[2]['val'], v_one),
                ("multiply_add", desks[2]['idx'], desks[2]['idx'], v_two, v_one),
                ("^", desks[3]['val'], desks[3]['tmp1'], desks[3]['tmp2']),
            ],
        })

        # Desk2 branch add + desk3 hash4
        self.instrs.append({
            "valu": [
                ("+", desks[2]['idx'], desks[2]['idx'], desks[2]['tmp1']),
                ("multiply_add", desks[3]['val'], desks[3]['val'], v_fma_mult[4], v_hash_consts[4]),
            ],
        })

        # desk3 hash5 prep
        self.instrs.append({
            "valu": [
                ("^", desks[3]['tmp1'], desks[3]['val'], v_hash_consts[5]),
                (">>", desks[3]['tmp2'], desks[3]['val'], v_hash_shifts[5]),
            ],
        })

        # Desk2 bounds + desk3 hash5 combine
        self.instrs.append({
            "valu": [
                ("<", desks[2]['tmp1'], desks[2]['idx'], v_n_nodes),
                ("^", desks[3]['val'], desks[3]['tmp1'], desks[3]['tmp2']),
            ],
        })

        # Desk2 vselect bypass + desk3 branch prep with FMA
        self.instrs.append({
            "valu": [
                ("*", desks[2]['idx'], desks[2]['idx'], desks[2]['tmp1']),
                ("&", desks[3]['tmp1'], desks[3]['val'], v_one),
                ("multiply_add", desks[3]['idx'], desks[3]['idx'], v_two, v_one),
            ],
        })

        # Desk3 branch add
        self.instrs.append({
            "valu": [
                ("+", desks[3]['idx'], desks[3]['idx'], desks[3]['tmp1']),
            ],
        })

        # Desk3 bounds
        self.instrs.append({
            "valu": [
                ("<", desks[3]['tmp1'], desks[3]['idx'], v_n_nodes),
            ],
        })

        # Desk3 vselect bypass
        self.instrs.append({
            "valu": [
                ("*", desks[3]['idx'], desks[3]['idx'], desks[3]['tmp1']),
            ],
        })

        # ============================================================
        # ROUND 2 OF 2 - NO INTERMEDIATE STORE/LOAD
        # ============================================================

        # Compute gather addresses for round 2
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

        # Round 2: After desk0 gather complete: XOR desk0 + start desk1 gather
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

        # Desk2 gather 6-7 + desk0 hash4 (FMA) + desk1 hash1 combine
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

        # Round 2 Desk3 gather 4-5 + desk0 branch prep + precompute next batch_offset
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

        # Round 2 Desk3 gather 6-7 + desk0 branch add + desk1 hash4 + desk2 hash1 combine
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

        # Continue round 2 processing with address precomputation
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

        # Round 2: Desk0 bounds check
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

        # Round 2 Desk0 vselect bypass + desk1 branch prep with FMA
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

        # Round 2 Desk1 branch add + desk2 hash4 + desk3 hash1 combine
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

        # Round 2: desk2 hash5 prep + desk3 hash2
        self.instrs.append({
            "valu": [
                ("^", desks[2]['tmp1'], desks[2]['val'], v_hash_consts[5]),
                (">>", desks[2]['tmp2'], desks[2]['val'], v_hash_shifts[5]),
                ("multiply_add", desks[3]['val'], desks[3]['val'], v_fma_mult[2], v_hash_consts[2]),
            ],
        })

        # Round 2: Desk1 bounds + desk2 hash5 combine + desk3 hash3 prep
        self.instrs.append({
            "valu": [
                ("<", desks[1]['tmp1'], desks[1]['idx'], v_n_nodes),
                ("^", desks[2]['val'], desks[2]['tmp1'], desks[2]['tmp2']),
                ("+", desks[3]['tmp1'], desks[3]['val'], v_hash_consts[3]),
                ("<<", desks[3]['tmp2'], desks[3]['val'], v_hash_shifts[3]),
            ],
        })

        # Round 2 Desk1 vselect bypass + desk2 branch prep with FMA + desk3 hash3 combine
        self.instrs.append({
            "valu": [
                ("*", desks[1]['idx'], desks[1]['idx'], desks[1]['tmp1']),
                ("&", desks[2]['tmp1'], desks[2]['val'], v_one),
                ("multiply_add", desks[2]['idx'], desks[2]['idx'], v_two, v_one),
                ("^", desks[3]['val'], desks[3]['tmp1'], desks[3]['tmp2']),
            ],
        })

        # Round 2 Desk2 branch add + desk3 hash4
        self.instrs.append({
            "valu": [
                ("+", desks[2]['idx'], desks[2]['idx'], desks[2]['tmp1']),
                ("multiply_add", desks[3]['val'], desks[3]['val'], v_fma_mult[4], v_hash_consts[4]),
            ],
        })

        # Round 2: desk3 hash5 prep
        self.instrs.append({
            "valu": [
                ("^", desks[3]['tmp1'], desks[3]['val'], v_hash_consts[5]),
                (">>", desks[3]['tmp2'], desks[3]['val'], v_hash_shifts[5]),
            ],
        })

        # Round 2: Desk2 bounds + desk3 hash5 combine
        self.instrs.append({
            "valu": [
                ("<", desks[2]['tmp1'], desks[2]['idx'], v_n_nodes),
                ("^", desks[3]['val'], desks[3]['tmp1'], desks[3]['tmp2']),
            ],
        })

        # Round 2 Desk2 vselect bypass + desk3 branch prep with FMA
        self.instrs.append({
            "valu": [
                ("*", desks[2]['idx'], desks[2]['idx'], desks[2]['tmp1']),
                ("&", desks[3]['tmp1'], desks[3]['val'], v_one),
                ("multiply_add", desks[3]['idx'], desks[3]['idx'], v_two, v_one),
            ],
        })

        # Round 2 Desk3 branch add + compute store addresses
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

        # Round 2: Desk3 bounds + continue store address computation
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

        # Round 2: Desk3 vselect bypass
        self.instrs.append({
            "valu": [
                ("*", desks[3]['idx'], desks[3]['idx'], desks[3]['tmp1']),
            ],
        })

        # ============================================================
        # STORE COALESCING
        # ============================================================

        # Store desk0 + update batch_offset and iter_counter
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

        # Store desk1 + check batch_offset < batch_size
        self.instrs.append({
            "store": [
                ("vstore", addr_tmp[2], desks[1]['idx']),
                ("vstore", addr_tmp[3], desks[1]['val']),
            ],
            "alu": [
                ("<", tmp_scalar, batch_offset, batch_size_const),
            ],
        })

        # Store desk2 + check iteration counter
        self.instrs.append({
            "store": [
                ("vstore", addr_tmp[4], desks[2]['idx']),
                ("vstore", addr_tmp[5], desks[2]['val']),
            ],
            "alu": [
                ("<", addr_scalar, iter_counter, total_const),
            ],
        })

        # Store desk3 + wrap-around select
        # For remaining batches (32-255), wrap to 32 instead of 0
        self.instrs.append({
            "store": [
                ("vstore", addr_tmp[6], desks[3]['idx']),
                ("vstore", addr_tmp[7], desks[3]['val']),
            ],
            "flow": [
                ("select", batch_offset, tmp_scalar, batch_offset, thirtytwo_const),
            ],
        })

        # LOOP CONTROL
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

    kb = KernelBuilderH31()
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
