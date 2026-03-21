"""
# Experiment H15: Aggressive Loop Unrolling (2x)

This builds on H7H10 (5,947 cycles) by unrolling the main loop 2x.

**Current Loop Structure (H7H10):**
- 128 iterations total
- Each iteration processes 4 desks at offsets 0, 8, 16, 24
- Loop overhead per iteration: counter increment, bounds check, conditional jump

**Unrolled Structure (H15):**
- 64 iterations total
- Each iteration processes 8 effective batches:
  - First batch: desks 0-3 at offsets 0, 8, 16, 24
  - Second batch: desks 0-3 at offsets 32, 40, 48, 56
- Loop overhead cut in half

**Loop Overhead per Iteration (before unrolling):**
1. ALU: batch_offset += 32, iter_counter += 1  (~1 cycle)
2. ALU: compare batch_offset < batch_size     (~1 cycle)
3. FLOW: select for batch wrap               (~1 cycle)
4. ALU: compare iter_counter < total         (~1 cycle)
5. FLOW: cond_jump                           (~1 cycle)
Total: ~5 cycles per iteration

**Expected Savings:**
- Before: 128 iterations x 5 cycles = 640 cycles overhead
- After: 64 iterations x 5 cycles = 320 cycles overhead
- Savings: ~320 cycles

**Expected:** ~5,627 cycles (5,947 - 320)
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


class KernelBuilderH15:
    """
    H15: 2x Loop Unrolling on top of H7H10

    The main loop is unrolled 2x to reduce loop overhead.
    Instead of processing 4 desks per iteration (128 iterations),
    we process 8 effective batches per iteration (64 iterations).
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

    def emit_desk_pipeline(self, desks, desk_indices, offset_regs, addr_tmp,
                           v_zero, v_one, v_two, v_n_nodes,
                           v_hash_consts, v_hash_shifts, v_fma_mult,
                           is_second_batch=False):
        """
        Emit the processing pipeline for 4 desks.
        This is the main work of each "batch" - we call it twice per loop iteration.
        """
        # PHASE 1: Calculate all offsets and load addresses
        # Pack ALU operations heavily
        self.instrs.append({
            "alu": [
                ("+", addr_tmp[0], self.scratch["inp_indices_p"], offset_regs[0]),
                ("+", addr_tmp[1], self.scratch["inp_values_p"], offset_regs[0]),
            ],
        })

        # Interleaved pattern: Start loading while computing addresses
        # Cycle: Load desk0 idx/val + compute addresses for desk1,2
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

        # Load desk1 + compute desk2,3 addresses + start desk0 gather addr
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
                ("vbroadcast", desks[0]['addr'], self.scratch["forest_values_p"]),
            ],
        })

        # Load desk2 + desk0 gather addr add + desk1 gather addr broadcast
        self.instrs.append({
            "load": [
                ("vload", desks[2]['idx'], addr_tmp[4]),
                ("vload", desks[2]['val'], addr_tmp[5]),
            ],
            "valu": [
                ("+", desks[0]['addr'], desks[0]['addr'], desks[0]['idx']),
                ("vbroadcast", desks[1]['addr'], self.scratch["forest_values_p"]),
            ],
        })

        # Load desk3 + desk1 gather addr add + desk2,3 broadcasts
        self.instrs.append({
            "load": [
                ("vload", desks[3]['idx'], addr_tmp[6]),
                ("vload", desks[3]['val'], addr_tmp[7]),
            ],
            "valu": [
                ("+", desks[1]['addr'], desks[1]['addr'], desks[1]['idx']),
                ("vbroadcast", desks[2]['addr'], self.scratch["forest_values_p"]),
                ("vbroadcast", desks[3]['addr'], self.scratch["forest_values_p"]),
            ],
        })

        # Desk 2,3 gather address add
        self.instrs.append({
            "valu": [
                ("+", desks[2]['addr'], desks[2]['addr'], desks[2]['idx']),
                ("+", desks[3]['addr'], desks[3]['addr'], desks[3]['idx']),
            ],
        })

        # ============================================================
        # INTERLEAVED GATHER + HASH PIPELINE
        # ============================================================

        # ---- Desk 0 gathers (4 cycles) ----
        for lane in range(0, VLEN, 2):
            self.instrs.append({
                "load": [
                    ("load", desks[0]['node_val'] + lane, desks[0]['addr'] + lane),
                    ("load", desks[0]['node_val'] + lane + 1, desks[0]['addr'] + lane + 1),
                ],
            })

        # ---- After desk0 gather complete: XOR desk0 + start desk1 gather ----
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

        # Desk1 gather lanes 4-5 + desk0 hash stage 1 prep (stage 1 is XOR-based)
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

        # ---- Desk1 gather complete: XOR desk1 + desk0 hash stage 2 (FMA) + start desk2 gather ----
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
                # Stage 3 is ("+", 0xD3A2646C, "^", "<<", 9) - op1 is +, op2 is ^
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
                # Stage 3 combine: op2 is ^
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

        # ---- Desk2 gather complete: XOR desk2 + desk0 hash5 prep + desk1 hash2 + start desk3 gather ----
        self.instrs.append({
            "load": [
                ("load", desks[3]['node_val'], desks[3]['addr']),
                ("load", desks[3]['node_val'] + 1, desks[3]['addr'] + 1),
            ],
            "valu": [
                ("^", desks[2]['val'], desks[2]['val'], desks[2]['node_val']),
                # Stage 5 is ("^", ..., "^", ">>", 16) - op1 and op2 are both ^
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

        # Desk3 gather 4-5 + desk0 branch prep (AND, MUL) + desk1 hash3 combine + desk2 hash1 prep
        self.instrs.append({
            "load": [
                ("load", desks[3]['node_val'] + 4, desks[3]['addr'] + 4),
                ("load", desks[3]['node_val'] + 5, desks[3]['addr'] + 5),
            ],
            "valu": [
                ("&", desks[0]['tmp1'], desks[0]['val'], v_one),
                ("*", desks[0]['idx'], desks[0]['idx'], v_two),
                ("^", desks[1]['val'], desks[1]['tmp1'], desks[1]['tmp2']),
                ("^", desks[2]['tmp1'], desks[2]['val'], v_hash_consts[1]),
                (">>", desks[2]['tmp2'], desks[2]['val'], v_hash_shifts[1]),
            ],
        })

        # Desk3 gather 6-7 + desk0 branch (add 1, add tmp1) + desk1 hash4 + desk2 hash1 combine
        self.instrs.append({
            "load": [
                ("load", desks[3]['node_val'] + 6, desks[3]['addr'] + 6),
                ("load", desks[3]['node_val'] + 7, desks[3]['addr'] + 7),
            ],
            "valu": [
                ("+", desks[0]['idx'], desks[0]['idx'], v_one),
                ("multiply_add", desks[1]['val'], desks[1]['val'], v_fma_mult[4], v_hash_consts[4]),
                ("^", desks[2]['val'], desks[2]['tmp1'], desks[2]['tmp2']),
            ],
        })

        # Continue desk0 branch + complete pipeline
        self.instrs.append({
            "valu": [
                ("+", desks[0]['idx'], desks[0]['idx'], desks[0]['tmp1']),
                # Desk1 hash5 prep
                ("^", desks[1]['tmp1'], desks[1]['val'], v_hash_consts[5]),
                (">>", desks[1]['tmp2'], desks[1]['val'], v_hash_shifts[5]),
                # Desk2 hash2
                ("multiply_add", desks[2]['val'], desks[2]['val'], v_fma_mult[2], v_hash_consts[2]),
                # Desk3 XOR
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

        # H7H10 CHANGE: Replace desk0 vselect with VALU multiply
        self.instrs.append({
            "valu": [
                # H10 optimization: vselect -> multiply
                ("*", desks[0]['idx'], desks[0]['idx'], desks[0]['tmp1']),
                # desk1 branch prep
                ("&", desks[1]['tmp1'], desks[1]['val'], v_one),
                ("*", desks[1]['idx'], desks[1]['idx'], v_two),
                # desk2 hash3 combine
                ("^", desks[2]['val'], desks[2]['tmp1'], desks[2]['tmp2']),
                # desk3 hash1 prep
                ("^", desks[3]['tmp1'], desks[3]['val'], v_hash_consts[1]),
                (">>", desks[3]['tmp2'], desks[3]['val'], v_hash_shifts[1]),
            ],
        })

        # Desk1 branch (add 1) + desk2 hash4 + desk3 hash1 combine
        self.instrs.append({
            "valu": [
                ("+", desks[1]['idx'], desks[1]['idx'], v_one),
                ("multiply_add", desks[2]['val'], desks[2]['val'], v_fma_mult[4], v_hash_consts[4]),
                ("^", desks[3]['val'], desks[3]['tmp1'], desks[3]['tmp2']),
            ],
        })

        # Desk1 branch (add tmp1) + desk2 hash5 prep + desk3 hash2
        self.instrs.append({
            "valu": [
                ("+", desks[1]['idx'], desks[1]['idx'], desks[1]['tmp1']),
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

        # H7H10 CHANGE: Replace desk1 vselect with VALU multiply + desk2 branch prep + desk3 hash3 combine
        self.instrs.append({
            "valu": [
                # H10 optimization: vselect -> multiply
                ("*", desks[1]['idx'], desks[1]['idx'], desks[1]['tmp1']),
                # desk2 branch prep
                ("&", desks[2]['tmp1'], desks[2]['val'], v_one),
                ("*", desks[2]['idx'], desks[2]['idx'], v_two),
                # desk3 hash3 combine
                ("^", desks[3]['val'], desks[3]['tmp1'], desks[3]['tmp2']),
            ],
        })

        # Desk2 branch (add 1) + desk3 hash4
        self.instrs.append({
            "valu": [
                ("+", desks[2]['idx'], desks[2]['idx'], v_one),
                ("multiply_add", desks[3]['val'], desks[3]['val'], v_fma_mult[4], v_hash_consts[4]),
            ],
        })

        # Desk2 branch (add tmp1) + desk3 hash5 prep
        self.instrs.append({
            "valu": [
                ("+", desks[2]['idx'], desks[2]['idx'], desks[2]['tmp1']),
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

        # H7H10 CHANGE: Replace desk2 vselect with VALU multiply + desk3 branch prep
        self.instrs.append({
            "valu": [
                # H10 optimization: vselect -> multiply
                ("*", desks[2]['idx'], desks[2]['idx'], desks[2]['tmp1']),
                # desk3 branch prep
                ("&", desks[3]['tmp1'], desks[3]['val'], v_one),
                ("*", desks[3]['idx'], desks[3]['idx'], v_two),
            ],
        })

        # Desk3 branch (add 1)
        self.instrs.append({
            "valu": [
                ("+", desks[3]['idx'], desks[3]['idx'], v_one),
            ],
        })

        # Desk3 branch (add tmp1)
        self.instrs.append({
            "valu": [
                ("+", desks[3]['idx'], desks[3]['idx'], desks[3]['tmp1']),
            ],
        })

        # Desk3 bounds + start computing store addresses for desks 0,1,2 (not desk3 yet)
        self.instrs.append({
            "valu": [
                ("<", desks[3]['tmp1'], desks[3]['idx'], v_n_nodes),
            ],
            "alu": [
                ("+", addr_tmp[0], self.scratch["inp_indices_p"], offset_regs[0]),
                ("+", addr_tmp[1], self.scratch["inp_values_p"], offset_regs[0]),
                ("+", addr_tmp[2], self.scratch["inp_indices_p"], offset_regs[1]),
                ("+", addr_tmp[3], self.scratch["inp_values_p"], offset_regs[1]),
                ("+", addr_tmp[4], self.scratch["inp_indices_p"], offset_regs[2]),
                ("+", addr_tmp[5], self.scratch["inp_values_p"], offset_regs[2]),
            ],
        })

        # H7H10 CHANGE: Replace desk3 vselect with VALU multiply + compute desk3 store addresses
        self.instrs.append({
            "valu": [
                # H10 optimization: vselect -> multiply
                ("*", desks[3]['idx'], desks[3]['idx'], desks[3]['tmp1']),
            ],
            "alu": [
                ("+", addr_tmp[6], self.scratch["inp_indices_p"], offset_regs[3]),
                ("+", addr_tmp[7], self.scratch["inp_values_p"], offset_regs[3]),
            ],
        })

        # STORE PHASE - Pack stores together (2 per cycle)

        # Interleave stores (2 per cycle)
        self.instrs.append({"store": [("vstore", addr_tmp[0], desks[0]['idx']), ("vstore", addr_tmp[1], desks[0]['val'])]})
        self.instrs.append({"store": [("vstore", addr_tmp[2], desks[1]['idx']), ("vstore", addr_tmp[3], desks[1]['val'])]})
        self.instrs.append({"store": [("vstore", addr_tmp[4], desks[2]['idx']), ("vstore", addr_tmp[5], desks[2]['val'])]})
        self.instrs.append({"store": [("vstore", addr_tmp[6], desks[3]['idx']), ("vstore", addr_tmp[7], desks[3]['val'])]})

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        H15: 2x Unrolled loop kernel.
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
        batch_offset = self.alloc_scratch("batch_offset")
        iter_counter = self.alloc_scratch("iter_counter")

        eight_const = self.scratch_const(VLEN)
        sixteen_const = self.scratch_const(2 * VLEN)
        twentyfour_const = self.scratch_const(3 * VLEN)
        thirtytwo_const = self.scratch_const(4 * VLEN)
        forty_const = self.scratch_const(5 * VLEN)
        fortyeight_const = self.scratch_const(6 * VLEN)
        fiftysix_const = self.scratch_const(7 * VLEN)
        sixtyfour_const = self.scratch_const(8 * VLEN)  # H15: Doubled increment
        batch_size_const = self.scratch_const(batch_size)

        # H15: Halved iterations (64 instead of 128)
        total_iterations = (batch_size // VLEN) * rounds // NUM_DESKS // 2
        total_const = self.scratch_const(total_iterations)

        # Offset registers for first batch (0, 8, 16, 24)
        offset_regs_batch1 = [self.alloc_scratch(f"off_b1_{d}") for d in range(NUM_DESKS)]
        # Offset registers for second batch (32, 40, 48, 56)
        offset_regs_batch2 = [self.alloc_scratch(f"off_b2_{d}") for d in range(NUM_DESKS)]

        self.add("load", ("const", batch_offset, 0))
        self.add("load", ("const", iter_counter, 0))

        # === MAIN LOOP with 2x UNROLLING ===
        main_loop_start = len(self.instrs)

        # === FIRST BATCH: offsets 0, 8, 16, 24 ===
        # Calculate offsets for first batch
        self.instrs.append({
            "alu": [
                ("+", offset_regs_batch1[0], batch_offset, zero_const),
                ("+", offset_regs_batch1[1], batch_offset, eight_const),
                ("+", offset_regs_batch1[2], batch_offset, sixteen_const),
                ("+", offset_regs_batch1[3], batch_offset, twentyfour_const),
            ],
        })

        # Emit the pipeline for first batch
        self.emit_desk_pipeline(
            desks, list(range(NUM_DESKS)), offset_regs_batch1, addr_tmp,
            v_zero, v_one, v_two, v_n_nodes,
            v_hash_consts, v_hash_shifts, v_fma_mult,
            is_second_batch=False
        )

        # === SECOND BATCH: offsets 32, 40, 48, 56 ===
        # Calculate offsets for second batch
        self.instrs.append({
            "alu": [
                ("+", offset_regs_batch2[0], batch_offset, thirtytwo_const),
                ("+", offset_regs_batch2[1], batch_offset, forty_const),
                ("+", offset_regs_batch2[2], batch_offset, fortyeight_const),
                ("+", offset_regs_batch2[3], batch_offset, fiftysix_const),
            ],
        })

        # Emit the pipeline for second batch
        self.emit_desk_pipeline(
            desks, list(range(NUM_DESKS)), offset_regs_batch2, addr_tmp,
            v_zero, v_one, v_two, v_n_nodes,
            v_hash_consts, v_hash_shifts, v_fma_mult,
            is_second_batch=True
        )

        # LOOP CONTROL (only once per 2 batches now!)
        # H15: Increment batch_offset by 64 instead of 32
        self.instrs.append({
            "alu": [
                ("+", batch_offset, batch_offset, sixtyfour_const),
                ("+", iter_counter, iter_counter, one_const),
            ],
        })
        self.instrs.append({
            "alu": [("<", tmp_scalar, batch_offset, batch_size_const)],
        })
        self.add("flow", ("select", batch_offset, tmp_scalar, batch_offset, zero_const))
        self.instrs.append({
            "alu": [("<", tmp_scalar, iter_counter, total_const)],
        })
        self.add("flow", ("cond_jump", tmp_scalar, main_loop_start))
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

    kb = KernelBuilderH15()
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
