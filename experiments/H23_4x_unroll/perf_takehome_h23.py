"""
# Experiment H23: 4x Loop Unrolling (Optimized)

Building on C4 (4,667 cycles), this tests aggressive loop unrolling with
optimized inter-iteration overlap.

**C4 Baseline:**
- Processes 2 rounds per iteration (round fusion)
- Uses 4 desks for interleaving
- Total iterations = 64 (8 double-rounds * 8 batches / 4 desks)

**H23 Strategy (Revised):**
- Keep the C4 structure but overlap store/setup phases between unrolled copies
- Use 4 sets of offset registers to pre-compute addresses for all unroll copies
- Interleave stores with address computation for next copy
- Only do loop control once per 4 original iterations

**Key Changes from naive unroll:**
1. Pre-allocate all offset registers upfront (4 sets of 4 = 16 offset regs)
2. Compute all batch offsets at loop start
3. Overlap stores with next iteration setup (like C4's address pipelining)

**Expected:** Better pipeline utilization, reduced stalls.
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


class KernelBuilderH23:
    """
    H23: 4x Loop Unrolling with Optimized Overlap

    Key optimization: Pre-compute all 4 sets of offsets at loop start,
    then overlap stores with address computation between unroll copies.
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

    def emit_double_round_with_stores(self, desks, v_hash_consts, v_hash_shifts, v_fma_mult,
                                       v_zero, v_one, v_two, v_n_nodes,
                                       offset_regs, addr_tmp, store_offset_regs, store_addr_tmp,
                                       is_last_unroll=False, next_offset_regs=None, next_base_offset=None,
                                       eight_const=None, sixteen_const=None, twentyfour_const=None,
                                       inp_indices_p=None, inp_values_p=None):
        """
        Emit a complete double-round processing with stores at the end.
        If not last unroll, overlap stores with address setup for next unroll.
        """
        # PHASE 1: Load idx/val for all 4 desks (offset_regs and addr_tmp already computed)

        # Load desk0 idx/val + compute addresses for desk1,2
        self.instrs.append({
            "load": [
                ("vload", desks[0]['idx'], addr_tmp[0]),
                ("vload", desks[0]['val'], addr_tmp[1]),
            ],
            "alu": [
                ("+", addr_tmp[2], inp_indices_p, offset_regs[1]),
                ("+", addr_tmp[3], inp_values_p, offset_regs[1]),
                ("+", addr_tmp[4], inp_indices_p, offset_regs[2]),
                ("+", addr_tmp[5], inp_values_p, offset_regs[2]),
            ],
            "valu": [
                ("vbroadcast", desks[0]['addr'], self.scratch["forest_values_p"]),
            ],
        })

        # Load desk1 + compute desk3 addresses + desk1 gather addr broadcast
        self.instrs.append({
            "load": [
                ("vload", desks[1]['idx'], addr_tmp[2]),
                ("vload", desks[1]['val'], addr_tmp[3]),
            ],
            "alu": [
                ("+", addr_tmp[6], inp_indices_p, offset_regs[3]),
                ("+", addr_tmp[7], inp_values_p, offset_regs[3]),
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

        # Remaining desk0 gathers (3 cycles)
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

        # Desk3 gather 4-5 + desk0 branch prep with FMA
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

        # Desk3 gather 6-7 + desk0 branch (add tmp1 only) + desk1 hash4 + desk2 hash1 combine
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

        # Desk1 branch (add tmp1 only) + desk2 hash4 + desk3 hash1 combine
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

        # Desk2 branch (add tmp1 only) + desk3 hash4
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

        # Desk3 branch (add tmp1 only)
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

        # Compute gather addresses for round 2 (all desks)
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

        # Round 2 Desk3 gather 4-5 + desk0 branch prep with FMA
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

        # Round 2 Desk3 gather 6-7 + desk0 branch (add tmp1 only) + desk1 hash4 + desk2 hash1 combine
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

        # Continue round 2 processing
        self.instrs.append({
            "valu": [
                ("^", desks[1]['tmp1'], desks[1]['val'], v_hash_consts[5]),
                (">>", desks[1]['tmp2'], desks[1]['val'], v_hash_shifts[5]),
                ("multiply_add", desks[2]['val'], desks[2]['val'], v_fma_mult[2], v_hash_consts[2]),
                ("^", desks[3]['val'], desks[3]['val'], desks[3]['node_val']),
            ],
        })

        # Round 2: Desk0 bounds check + desk1 hash5 combine + desk2 hash3 prep + desk3 hash0
        self.instrs.append({
            "valu": [
                ("<", desks[0]['tmp1'], desks[0]['idx'], v_n_nodes),
                ("^", desks[1]['val'], desks[1]['tmp1'], desks[1]['tmp2']),
                ("+", desks[2]['tmp1'], desks[2]['val'], v_hash_consts[3]),
                ("<<", desks[2]['tmp2'], desks[2]['val'], v_hash_shifts[3]),
                ("multiply_add", desks[3]['val'], desks[3]['val'], v_fma_mult[0], v_hash_consts[0]),
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
        })

        # Round 2 Desk1 branch (add tmp1 only) + desk2 hash4 + desk3 hash1 combine
        self.instrs.append({
            "valu": [
                ("+", desks[1]['idx'], desks[1]['idx'], desks[1]['tmp1']),
                ("multiply_add", desks[2]['val'], desks[2]['val'], v_fma_mult[4], v_hash_consts[4]),
                ("^", desks[3]['val'], desks[3]['tmp1'], desks[3]['tmp2']),
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

        # Round 2 Desk2 branch (add tmp1 only) + desk3 hash4
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

        # Round 2 Desk3 branch (add tmp1 only)
        self.instrs.append({
            "valu": [
                ("+", desks[3]['idx'], desks[3]['idx'], desks[3]['tmp1']),
            ],
        })

        # Round 2: Desk3 bounds
        self.instrs.append({
            "valu": [
                ("<", desks[3]['tmp1'], desks[3]['idx'], v_n_nodes),
            ],
        })

        # Round 2: Desk3 vselect bypass + start computing store addresses
        if not is_last_unroll:
            self.instrs.append({
                "valu": [
                    ("*", desks[3]['idx'], desks[3]['idx'], desks[3]['tmp1']),
                ],
                "alu": [
                    ("+", store_addr_tmp[0], inp_indices_p, store_offset_regs[0]),
                    ("+", store_addr_tmp[1], inp_values_p, store_offset_regs[0]),
                    ("+", store_addr_tmp[2], inp_indices_p, store_offset_regs[1]),
                    ("+", store_addr_tmp[3], inp_values_p, store_offset_regs[1]),
                    # Also start computing next iteration's offsets
                    ("+", next_offset_regs[0], next_base_offset, self.const_map[0]),
                    ("+", next_offset_regs[1], next_base_offset, eight_const),
                ],
            })

            # Stores + continue next iteration offset computation
            self.instrs.append({
                "store": [
                    ("vstore", store_addr_tmp[0], desks[0]['idx']),
                    ("vstore", store_addr_tmp[1], desks[0]['val']),
                ],
                "alu": [
                    ("+", store_addr_tmp[4], inp_indices_p, store_offset_regs[2]),
                    ("+", store_addr_tmp[5], inp_values_p, store_offset_regs[2]),
                    ("+", store_addr_tmp[6], inp_indices_p, store_offset_regs[3]),
                    ("+", store_addr_tmp[7], inp_values_p, store_offset_regs[3]),
                    ("+", next_offset_regs[2], next_base_offset, sixteen_const),
                    ("+", next_offset_regs[3], next_base_offset, twentyfour_const),
                ],
            })

            self.instrs.append({
                "store": [
                    ("vstore", store_addr_tmp[2], desks[1]['idx']),
                    ("vstore", store_addr_tmp[3], desks[1]['val']),
                ],
                "alu": [
                    # Compute first load address for next iteration
                    ("+", addr_tmp[0], inp_indices_p, next_base_offset),
                    ("+", addr_tmp[1], inp_values_p, next_base_offset),
                ],
            })

            self.instrs.append({
                "store": [
                    ("vstore", store_addr_tmp[4], desks[2]['idx']),
                    ("vstore", store_addr_tmp[5], desks[2]['val']),
                ],
            })

            self.instrs.append({
                "store": [
                    ("vstore", store_addr_tmp[6], desks[3]['idx']),
                    ("vstore", store_addr_tmp[7], desks[3]['val']),
                ],
            })
        else:
            # Last unroll - just compute store addresses and store
            self.instrs.append({
                "valu": [
                    ("*", desks[3]['idx'], desks[3]['idx'], desks[3]['tmp1']),
                ],
                "alu": [
                    ("+", store_addr_tmp[0], inp_indices_p, store_offset_regs[0]),
                    ("+", store_addr_tmp[1], inp_values_p, store_offset_regs[0]),
                    ("+", store_addr_tmp[2], inp_indices_p, store_offset_regs[1]),
                    ("+", store_addr_tmp[3], inp_values_p, store_offset_regs[1]),
                ],
            })

            self.instrs.append({
                "alu": [
                    ("+", store_addr_tmp[4], inp_indices_p, store_offset_regs[2]),
                    ("+", store_addr_tmp[5], inp_values_p, store_offset_regs[2]),
                    ("+", store_addr_tmp[6], inp_indices_p, store_offset_regs[3]),
                    ("+", store_addr_tmp[7], inp_values_p, store_offset_regs[3]),
                ],
            })

            self.instrs.append({
                "store": [
                    ("vstore", store_addr_tmp[0], desks[0]['idx']),
                    ("vstore", store_addr_tmp[1], desks[0]['val']),
                ],
            })

            self.instrs.append({
                "store": [
                    ("vstore", store_addr_tmp[2], desks[1]['idx']),
                    ("vstore", store_addr_tmp[3], desks[1]['val']),
                ],
            })

            self.instrs.append({
                "store": [
                    ("vstore", store_addr_tmp[4], desks[2]['idx']),
                    ("vstore", store_addr_tmp[5], desks[2]['val']),
                ],
            })

            self.instrs.append({
                "store": [
                    ("vstore", store_addr_tmp[6], desks[3]['idx']),
                    ("vstore", store_addr_tmp[7], desks[3]['val']),
                ],
            })

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        H23: 4x Unrolled kernel with optimized overlap
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

        # Allocate addr_tmp and store_addr_tmp (separate to allow overlap)
        addr_tmp = [self.alloc_scratch(f"addr_tmp_{i}") for i in range(8)]
        store_addr_tmp = [self.alloc_scratch(f"store_addr_tmp_{i}") for i in range(8)]

        # Allocate 4 sets of offset registers for 4 unroll copies
        all_offset_regs = []
        for u in range(4):
            offset_regs = [self.alloc_scratch(f"off_{u}_{d}") for d in range(NUM_DESKS)]
            all_offset_regs.append(offset_regs)

        # Base offsets for each unroll copy
        base_offsets = [self.alloc_scratch(f"base_off_{u}") for u in range(4)]

        batch_offset = self.alloc_scratch("batch_offset")
        iter_counter = self.alloc_scratch("iter_counter")

        eight_const = self.scratch_const(VLEN)
        sixteen_const = self.scratch_const(2 * VLEN)
        twentyfour_const = self.scratch_const(3 * VLEN)
        thirtytwo_const = self.scratch_const(4 * VLEN)
        sixtyfour_const = self.scratch_const(8 * VLEN)
        ninetysix_const = self.scratch_const(12 * VLEN)
        one_twenty_eight_const = self.scratch_const(16 * VLEN)
        batch_size_const = self.scratch_const(batch_size)

        # H23: With 4x unroll, we have 16 loop iterations instead of 64
        total_iterations = (batch_size // VLEN) * (rounds // 2) // NUM_DESKS // 4
        total_const = self.scratch_const(total_iterations)

        self.add("load", ("const", batch_offset, 0))
        self.add("load", ("const", iter_counter, 0))

        inp_indices_p = self.scratch["inp_indices_p"]
        inp_values_p = self.scratch["inp_values_p"]

        # === MAIN LOOP with 4x UNROLLING ===
        main_loop_start = len(self.instrs)

        # Compute all 4 base offsets at loop start
        self.instrs.append({
            "alu": [
                ("+", base_offsets[0], batch_offset, zero_const),
                ("+", base_offsets[1], batch_offset, thirtytwo_const),
                ("+", base_offsets[2], batch_offset, sixtyfour_const),
                ("+", base_offsets[3], batch_offset, ninetysix_const),
            ],
        })

        # Compute all offset registers for unroll copy 0
        self.instrs.append({
            "alu": [
                ("+", all_offset_regs[0][0], base_offsets[0], zero_const),
                ("+", all_offset_regs[0][1], base_offsets[0], eight_const),
                ("+", all_offset_regs[0][2], base_offsets[0], sixteen_const),
                ("+", all_offset_regs[0][3], base_offsets[0], twentyfour_const),
                ("+", addr_tmp[0], inp_indices_p, base_offsets[0]),
                ("+", addr_tmp[1], inp_values_p, base_offsets[0]),
            ],
        })

        # ============ UNROLL COPY 0 ============
        self.emit_double_round_with_stores(
            desks, v_hash_consts, v_hash_shifts, v_fma_mult,
            v_zero, v_one, v_two, v_n_nodes,
            all_offset_regs[0], addr_tmp, all_offset_regs[0], store_addr_tmp,
            is_last_unroll=False, next_offset_regs=all_offset_regs[1],
            next_base_offset=base_offsets[1],
            eight_const=eight_const, sixteen_const=sixteen_const,
            twentyfour_const=twentyfour_const,
            inp_indices_p=inp_indices_p, inp_values_p=inp_values_p
        )

        # ============ UNROLL COPY 1 ============
        self.emit_double_round_with_stores(
            desks, v_hash_consts, v_hash_shifts, v_fma_mult,
            v_zero, v_one, v_two, v_n_nodes,
            all_offset_regs[1], addr_tmp, all_offset_regs[1], store_addr_tmp,
            is_last_unroll=False, next_offset_regs=all_offset_regs[2],
            next_base_offset=base_offsets[2],
            eight_const=eight_const, sixteen_const=sixteen_const,
            twentyfour_const=twentyfour_const,
            inp_indices_p=inp_indices_p, inp_values_p=inp_values_p
        )

        # ============ UNROLL COPY 2 ============
        self.emit_double_round_with_stores(
            desks, v_hash_consts, v_hash_shifts, v_fma_mult,
            v_zero, v_one, v_two, v_n_nodes,
            all_offset_regs[2], addr_tmp, all_offset_regs[2], store_addr_tmp,
            is_last_unroll=False, next_offset_regs=all_offset_regs[3],
            next_base_offset=base_offsets[3],
            eight_const=eight_const, sixteen_const=sixteen_const,
            twentyfour_const=twentyfour_const,
            inp_indices_p=inp_indices_p, inp_values_p=inp_values_p
        )

        # ============ UNROLL COPY 3 (last) ============
        self.emit_double_round_with_stores(
            desks, v_hash_consts, v_hash_shifts, v_fma_mult,
            v_zero, v_one, v_two, v_n_nodes,
            all_offset_regs[3], addr_tmp, all_offset_regs[3], store_addr_tmp,
            is_last_unroll=True, next_offset_regs=None,
            next_base_offset=None,
            eight_const=eight_const, sixteen_const=sixteen_const,
            twentyfour_const=twentyfour_const,
            inp_indices_p=inp_indices_p, inp_values_p=inp_values_p
        )

        # ============ LOOP CONTROL ============
        self.instrs.append({
            "alu": [
                ("+", batch_offset, batch_offset, one_twenty_eight_const),
                ("+", iter_counter, iter_counter, one_const),
            ],
        })

        self.instrs.append({
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
C4_CYCLES = 4667


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

    kb = KernelBuilderH23()
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
    print(f"Speedup over C4 ({C4_CYCLES}): ", C4_CYCLES / machine.cycle)
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
