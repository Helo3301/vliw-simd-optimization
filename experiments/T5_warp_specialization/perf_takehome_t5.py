"""
# Experiment T5: Warp-Style Specialization

This experiment tests the hypothesis that separating the kernel into specialized
"producer" and "consumer" phases can improve pipeline efficiency.

The approach:
1. PRODUCER PHASE: Focus on hash computation (ALU/VALU)
   - Compute hash for multiple batches
   - Store intermediate results to handoff registers

2. CONSUMER PHASE: Focus on memory operations (Load/Store)
   - Load tree values (gather)
   - Store final results

Baseline: 9,793 cycles
Target: ~8,000 cycles (20% improvement)
"""

from collections import defaultdict
import random
import unittest
import argparse
import sys

# Add parent directory to path to import problem module
sys.path.insert(0, "/home/hestiasadmin/projects/original_performance_takehome")

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


class KernelBuilder:
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
        WARP-STYLE SPECIALIZED KERNEL

        Design:
        - Phase 1 (Producer): Hash all 4 desks' values (all VALU ops)
        - Phase 2 (Consumer): Gather tree values + store results (all Load/Store)
        - Use handoff registers between phases

        The key insight is that by separating phases, we can:
        1. Better saturate VALU slots during hash computation
        2. Better saturate Load/Store slots during memory operations
        3. Potentially reduce total cycles if either phase was the bottleneck

        However, we lose the ability to overlap hash with gather within the same
        iteration - so the question is whether phase specialization beats overlap.
        """
        # Vector scratch (8 elements each)
        v_idx = self.alloc_scratch("v_idx", VLEN)
        v_val = self.alloc_scratch("v_val", VLEN)
        v_node_val = self.alloc_scratch("v_node_val", VLEN)
        v_tmp1 = self.alloc_scratch("v_tmp1", VLEN)
        v_tmp2 = self.alloc_scratch("v_tmp2", VLEN)
        v_addr = self.alloc_scratch("v_addr", VLEN)

        # Scalar scratch
        tmp_scalar = self.alloc_scratch("tmp_scalar")
        addr_scalar = self.alloc_scratch("addr_scalar")

        # Load constants from memory header
        init_vars = [
            "rounds", "n_nodes", "batch_size", "forest_height",
            "forest_values_p", "inp_indices_p", "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp_scalar, i))
            self.add("load", ("load", self.scratch[v], tmp_scalar))

        # Vector constants
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

        # Hash constants (broadcast to vectors)
        v_hash_consts = []
        v_hash_shifts = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            const_scalar = self.scratch_const(val1)
            shift_scalar = self.scratch_const(val3)
            v_const = self.alloc_scratch(f"v_hash_const_{hi}", VLEN)
            v_shift = self.alloc_scratch(f"v_hash_shift_{hi}", VLEN)
            self.add("valu", ("vbroadcast", v_const, const_scalar))
            self.add("valu", ("vbroadcast", v_shift, shift_scalar))
            v_hash_consts.append(v_const)
            v_hash_shifts.append(v_shift)

        self.add("flow", ("pause",))

        # === WARP SPECIALIZATION: 2-DESK VERSION ===
        # Process 2 groups per iteration but with CLEAN PHASE SEPARATION
        # Desk allocation (reduced from 4 to 2 for simpler phase separation)

        NUM_DESKS = 2  # Start with 2 desks for cleaner phase separation

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

        # Loop control
        addr_tmp = [self.alloc_scratch(f"addr_tmp_{i}") for i in range(4)]
        batch_offset = self.alloc_scratch("batch_offset")
        iter_counter = self.alloc_scratch("iter_counter")

        # Constants
        eight_const = self.scratch_const(VLEN)
        sixteen_const = self.scratch_const(2 * VLEN)
        batch_size_const = self.scratch_const(batch_size)

        # Process 2 desks at a time = 16 elements per iteration
        # Total: (256 elements / 16) * 16 rounds = 256 iterations
        total_iterations = (batch_size // VLEN) * rounds // NUM_DESKS
        total_const = self.scratch_const(total_iterations)

        # Offset registers
        offset_regs = [self.alloc_scratch(f"off_{d}") for d in range(NUM_DESKS)]

        self.add("load", ("const", batch_offset, 0))
        self.add("load", ("const", iter_counter, 0))

        # === MAIN LOOP ===
        main_loop_start = len(self.instrs)

        # =====================================================
        # PHASE 1: CONSUMER - LOAD ALL DATA (ALL MEMORY OPS)
        # =====================================================
        # Compute offsets for both desks
        self.instrs.append({
            "alu": [
                ("+", offset_regs[0], batch_offset, zero_const),
                ("+", offset_regs[1], batch_offset, eight_const),
            ],
        })

        # Load indices and values for all desks
        for d in range(NUM_DESKS):
            desk = desks[d]
            # Compute load addresses
            self.instrs.append({
                "alu": [
                    ("+", addr_tmp[0], self.scratch["inp_indices_p"], offset_regs[d]),
                    ("+", addr_tmp[1], self.scratch["inp_values_p"], offset_regs[d]),
                ],
            })
            # vload indices and values
            self.instrs.append({
                "load": [
                    ("vload", desk['idx'], addr_tmp[0]),
                    ("vload", desk['val'], addr_tmp[1]),
                ],
            })

        # Compute gather addresses and perform gather for all desks
        for d in range(NUM_DESKS):
            desk = desks[d]
            # Broadcast forest_values_p and add indices
            self.add("valu", ("vbroadcast", desk['addr'], self.scratch["forest_values_p"]))
            self.add("valu", ("+", desk['addr'], desk['addr'], desk['idx']))

            # Gather: 8 scalar loads (2 per cycle = 4 cycles)
            for lane in range(0, VLEN, 2):
                self.instrs.append({
                    "load": [
                        ("load", desk['node_val'] + lane, desk['addr'] + lane),
                        ("load", desk['node_val'] + lane + 1, desk['addr'] + lane + 1),
                    ],
                })

        # =====================================================
        # PHASE 2: PRODUCER - COMPUTE ALL HASHES (ALL VALU OPS)
        # =====================================================
        # Process both desks' hash computations
        # Key optimization: Pack multiple VALU ops per cycle when possible

        for d in range(NUM_DESKS):
            desk = desks[d]

            # XOR with node value
            self.add("valu", ("^", desk['val'], desk['val'], desk['node_val']))

            # Hash stages - pack 2 ops where possible (both read 'val', write to tmps)
            for hi in range(6):
                # Stage prep: compute both partial results in parallel
                self.instrs.append({
                    "valu": [
                        (HASH_STAGES[hi][0], desk['tmp1'], desk['val'], v_hash_consts[hi]),
                        (HASH_STAGES[hi][3], desk['tmp2'], desk['val'], v_hash_shifts[hi]),
                    ],
                })
                # Stage final: combine results
                self.add("valu", (HASH_STAGES[hi][2], desk['val'], desk['tmp1'], desk['tmp2']))

            # Branch computation: idx = 2*idx + 1 + (val & 1)
            self.instrs.append({
                "valu": [
                    ("&", desk['tmp1'], desk['val'], v_one),    # tmp1 = val & 1
                    ("*", desk['idx'], desk['idx'], v_two),      # idx = 2 * idx
                ],
            })
            self.add("valu", ("+", desk['idx'], desk['idx'], v_one))        # idx = 2*idx + 1
            self.add("valu", ("+", desk['idx'], desk['idx'], desk['tmp1']))  # idx += (val&1)

            # Bounds check
            self.add("valu", ("<", desk['tmp1'], desk['idx'], v_n_nodes))
            self.add("flow", ("vselect", desk['idx'], desk['tmp1'], desk['idx'], v_zero))

        # =====================================================
        # PHASE 3: CONSUMER - STORE ALL RESULTS (ALL STORE OPS)
        # =====================================================
        for d in range(NUM_DESKS):
            desk = desks[d]
            # Compute store addresses
            self.instrs.append({
                "alu": [
                    ("+", addr_tmp[0], self.scratch["inp_indices_p"], offset_regs[d]),
                    ("+", addr_tmp[1], self.scratch["inp_values_p"], offset_regs[d]),
                ],
            })
            # vstore results
            self.instrs.append({
                "store": [
                    ("vstore", addr_tmp[0], desk['idx']),
                    ("vstore", addr_tmp[1], desk['val']),
                ],
            })

        # =====================================================
        # LOOP CONTROL
        # =====================================================
        # Advance batch_offset
        self.instrs.append({
            "alu": [("+", batch_offset, batch_offset, sixteen_const)],
        })

        # Check for round wrap
        self.instrs.append({
            "alu": [("<", tmp_scalar, batch_offset, batch_size_const)],
        })
        self.add("flow", ("select", batch_offset, tmp_scalar, batch_offset, zero_const))

        # Increment iteration counter
        self.instrs.append({
            "alu": [("+", iter_counter, iter_counter, one_const)],
        })

        # Check for loop termination
        self.instrs.append({
            "alu": [("<", tmp_scalar, iter_counter, total_const)],
        })

        self.add("flow", ("cond_jump", tmp_scalar, main_loop_start))
        self.instrs.append({"flow": [("pause",)]})


class KernelBuilderV2:
    """
    V2: 4-desk with CLEANED-UP phase batching.
    Fixes the redundant gather issue from initial version.
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
        V2: 4-desk phase batching with proper overlap WITHIN each phase.

        Key insight: While we separate producer (hash) and consumer (memory) phases,
        we can still overlap operations WITHIN each phase:
        - During LOAD phase: overlap vload/gather across desks
        - During HASH phase: pack multiple VALU ops from different desks
        - During STORE phase: overlap stores across desks

        This tests whether phase separation with intra-phase overlap beats
        the baseline's inter-phase overlap (hash overlapped with next gather).
        """
        # Vector scratch (8 elements each)
        v_idx = self.alloc_scratch("v_idx", VLEN)
        v_val = self.alloc_scratch("v_val", VLEN)
        v_node_val = self.alloc_scratch("v_node_val", VLEN)
        v_tmp1 = self.alloc_scratch("v_tmp1", VLEN)
        v_tmp2 = self.alloc_scratch("v_tmp2", VLEN)
        v_addr = self.alloc_scratch("v_addr", VLEN)

        # Scalar scratch
        tmp_scalar = self.alloc_scratch("tmp_scalar")
        addr_scalar = self.alloc_scratch("addr_scalar")

        # Load constants from memory header
        init_vars = [
            "rounds", "n_nodes", "batch_size", "forest_height",
            "forest_values_p", "inp_indices_p", "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp_scalar, i))
            self.add("load", ("load", self.scratch[v], tmp_scalar))

        # Vector constants
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

        # Hash constants (broadcast to vectors)
        v_hash_consts = []
        v_hash_shifts = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            const_scalar = self.scratch_const(val1)
            shift_scalar = self.scratch_const(val3)
            v_const = self.alloc_scratch(f"v_hash_const_{hi}", VLEN)
            v_shift = self.alloc_scratch(f"v_hash_shift_{hi}", VLEN)
            self.add("valu", ("vbroadcast", v_const, const_scalar))
            self.add("valu", ("vbroadcast", v_shift, shift_scalar))
            v_hash_consts.append(v_const)
            v_hash_shifts.append(v_shift)

        self.add("flow", ("pause",))

        # === 4-DESK WITH PHASE BATCHING ===
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

        # Loop control
        addr_tmp = [self.alloc_scratch(f"addr_tmp_{i}") for i in range(8)]
        batch_offset = self.alloc_scratch("batch_offset")
        iter_counter = self.alloc_scratch("iter_counter")

        # Constants
        eight_const = self.scratch_const(VLEN)
        sixteen_const = self.scratch_const(2 * VLEN)
        twentyfour_const = self.scratch_const(3 * VLEN)
        thirtytwo_const = self.scratch_const(4 * VLEN)
        batch_size_const = self.scratch_const(batch_size)

        total_iterations = (batch_size // VLEN) * rounds // NUM_DESKS
        total_const = self.scratch_const(total_iterations)

        # Offset registers for all 4 desks
        offset_regs = [self.alloc_scratch(f"off_{d}") for d in range(NUM_DESKS)]

        self.add("load", ("const", batch_offset, 0))
        self.add("load", ("const", iter_counter, 0))

        # === MAIN LOOP ===
        main_loop_start = len(self.instrs)

        # =====================================================
        # PHASE 1: LOAD - GET ALL DATA INTO REGISTERS
        # =====================================================
        # Compute all offsets at once (pack 4 ALU ops)
        self.instrs.append({
            "alu": [
                ("+", offset_regs[0], batch_offset, zero_const),
                ("+", offset_regs[1], batch_offset, eight_const),
                ("+", offset_regs[2], batch_offset, sixteen_const),
                ("+", offset_regs[3], batch_offset, twentyfour_const),
            ],
        })

        # Load all desk data (indices and values) - can pack 2 vloads per cycle
        # Desks 0 and 1
        self.instrs.append({
            "alu": [
                ("+", addr_tmp[0], self.scratch["inp_indices_p"], offset_regs[0]),
                ("+", addr_tmp[1], self.scratch["inp_values_p"], offset_regs[0]),
                ("+", addr_tmp[2], self.scratch["inp_indices_p"], offset_regs[1]),
                ("+", addr_tmp[3], self.scratch["inp_values_p"], offset_regs[1]),
            ],
        })
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

        # Desks 2 and 3
        self.instrs.append({
            "alu": [
                ("+", addr_tmp[4], self.scratch["inp_indices_p"], offset_regs[2]),
                ("+", addr_tmp[5], self.scratch["inp_values_p"], offset_regs[2]),
                ("+", addr_tmp[6], self.scratch["inp_indices_p"], offset_regs[3]),
                ("+", addr_tmp[7], self.scratch["inp_values_p"], offset_regs[3]),
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

        # Compute gather addresses for all desks
        for d in range(NUM_DESKS):
            desk = desks[d]
            self.add("valu", ("vbroadcast", desk['addr'], self.scratch["forest_values_p"]))
            self.add("valu", ("+", desk['addr'], desk['addr'], desk['idx']))

        # Perform all gathers - 8 lanes per desk, 2 loads per cycle, 4 desks = 16 total cycles
        # Each desk needs 4 cycles (8 loads / 2 per cycle)
        for d in range(NUM_DESKS):
            desk = desks[d]
            for lane in range(0, VLEN, 2):
                self.instrs.append({
                    "load": [
                        ("load", desk['node_val'] + lane, desk['addr'] + lane),
                        ("load", desk['node_val'] + lane + 1, desk['addr'] + lane + 1),
                    ],
                })

        # =====================================================
        # PHASE 2: HASH - COMPUTE ALL HASHES
        # =====================================================
        # Process all desks' hashes
        # Key insight: We can interleave hash stages across desks for better pipelining

        # XOR all desks first (can pack multiple VALU)
        for d in range(0, NUM_DESKS, 2):
            self.instrs.append({
                "valu": [
                    ("^", desks[d]['val'], desks[d]['val'], desks[d]['node_val']),
                    ("^", desks[d+1]['val'], desks[d+1]['val'], desks[d+1]['node_val']),
                ],
            })

        # Hash stages - interleave across desks to fill VALU slots
        # With 6 VALU slots, we can do 2 ops per desk (prep) for 3 desks, or 1 final for 6 desks
        for hi in range(6):
            # Prep for desks 0,1 (4 VALU ops)
            self.instrs.append({
                "valu": [
                    (HASH_STAGES[hi][0], desks[0]['tmp1'], desks[0]['val'], v_hash_consts[hi]),
                    (HASH_STAGES[hi][3], desks[0]['tmp2'], desks[0]['val'], v_hash_shifts[hi]),
                    (HASH_STAGES[hi][0], desks[1]['tmp1'], desks[1]['val'], v_hash_consts[hi]),
                    (HASH_STAGES[hi][3], desks[1]['tmp2'], desks[1]['val'], v_hash_shifts[hi]),
                ],
            })
            # Prep for desks 2,3 (4 VALU ops)
            self.instrs.append({
                "valu": [
                    (HASH_STAGES[hi][0], desks[2]['tmp1'], desks[2]['val'], v_hash_consts[hi]),
                    (HASH_STAGES[hi][3], desks[2]['tmp2'], desks[2]['val'], v_hash_shifts[hi]),
                    (HASH_STAGES[hi][0], desks[3]['tmp1'], desks[3]['val'], v_hash_consts[hi]),
                    (HASH_STAGES[hi][3], desks[3]['tmp2'], desks[3]['val'], v_hash_shifts[hi]),
                ],
            })
            # Final for desks 0,1,2,3 (4 VALU ops)
            self.instrs.append({
                "valu": [
                    (HASH_STAGES[hi][2], desks[0]['val'], desks[0]['tmp1'], desks[0]['tmp2']),
                    (HASH_STAGES[hi][2], desks[1]['val'], desks[1]['tmp1'], desks[1]['tmp2']),
                    (HASH_STAGES[hi][2], desks[2]['val'], desks[2]['tmp1'], desks[2]['tmp2']),
                    (HASH_STAGES[hi][2], desks[3]['val'], desks[3]['tmp1'], desks[3]['tmp2']),
                ],
            })

        # Branch computation for all desks
        # Step 1: AND and MUL (4 desks, 2 ops each = 8 ops, but only 6 slots, so 2 cycles)
        self.instrs.append({
            "valu": [
                ("&", desks[0]['tmp1'], desks[0]['val'], v_one),
                ("*", desks[0]['idx'], desks[0]['idx'], v_two),
                ("&", desks[1]['tmp1'], desks[1]['val'], v_one),
                ("*", desks[1]['idx'], desks[1]['idx'], v_two),
                ("&", desks[2]['tmp1'], desks[2]['val'], v_one),
                ("*", desks[2]['idx'], desks[2]['idx'], v_two),
            ],
        })
        self.instrs.append({
            "valu": [
                ("&", desks[3]['tmp1'], desks[3]['val'], v_one),
                ("*", desks[3]['idx'], desks[3]['idx'], v_two),
            ],
        })

        # Step 2: ADD 1 (4 ops)
        self.instrs.append({
            "valu": [
                ("+", desks[0]['idx'], desks[0]['idx'], v_one),
                ("+", desks[1]['idx'], desks[1]['idx'], v_one),
                ("+", desks[2]['idx'], desks[2]['idx'], v_one),
                ("+", desks[3]['idx'], desks[3]['idx'], v_one),
            ],
        })

        # Step 3: ADD (val & 1) (4 ops)
        self.instrs.append({
            "valu": [
                ("+", desks[0]['idx'], desks[0]['idx'], desks[0]['tmp1']),
                ("+", desks[1]['idx'], desks[1]['idx'], desks[1]['tmp1']),
                ("+", desks[2]['idx'], desks[2]['idx'], desks[2]['tmp1']),
                ("+", desks[3]['idx'], desks[3]['idx'], desks[3]['tmp1']),
            ],
        })

        # Step 4: Bounds check (4 ops)
        self.instrs.append({
            "valu": [
                ("<", desks[0]['tmp1'], desks[0]['idx'], v_n_nodes),
                ("<", desks[1]['tmp1'], desks[1]['idx'], v_n_nodes),
                ("<", desks[2]['tmp1'], desks[2]['idx'], v_n_nodes),
                ("<", desks[3]['tmp1'], desks[3]['idx'], v_n_nodes),
            ],
        })

        # Step 5: vselect (1 flow slot per, so 4 cycles)
        for d in range(NUM_DESKS):
            self.add("flow", ("vselect", desks[d]['idx'], desks[d]['tmp1'], desks[d]['idx'], v_zero))

        # =====================================================
        # PHASE 3: STORE - WRITE ALL RESULTS
        # =====================================================
        # Store all results (2 vstores per cycle max)
        for d in range(NUM_DESKS):
            desk = desks[d]
            self.instrs.append({
                "alu": [
                    ("+", addr_tmp[0], self.scratch["inp_indices_p"], offset_regs[d]),
                    ("+", addr_tmp[1], self.scratch["inp_values_p"], offset_regs[d]),
                ],
            })
            self.instrs.append({
                "store": [
                    ("vstore", addr_tmp[0], desk['idx']),
                    ("vstore", addr_tmp[1], desk['val']),
                ],
            })

        # =====================================================
        # LOOP CONTROL
        # =====================================================
        self.instrs.append({
            "alu": [
                ("+", batch_offset, batch_offset, thirtytwo_const),
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


class KernelBuilderV3:
    """
    V3: Hybrid approach - maintain overlap but with phase-aware organization.
    This version keeps the overlapped structure from baseline but reorganizes
    to maximize slot utilization within each "phase" of the loop.
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
        V3: Software-pipelined approach with handoff

        Structure:
        - Iteration N: Load (consumer) | Hash N-1 (producer)
        - Iteration N+1: Load (consumer) | Hash N (producer)

        The "handoff" is implicit through register reuse.
        """
        # Allocation (same as before)
        v_idx = self.alloc_scratch("v_idx", VLEN)
        v_val = self.alloc_scratch("v_val", VLEN)
        v_node_val = self.alloc_scratch("v_node_val", VLEN)
        v_tmp1 = self.alloc_scratch("v_tmp1", VLEN)
        v_tmp2 = self.alloc_scratch("v_tmp2", VLEN)
        v_addr = self.alloc_scratch("v_addr", VLEN)

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

        v_hash_consts = []
        v_hash_shifts = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            const_scalar = self.scratch_const(val1)
            shift_scalar = self.scratch_const(val3)
            v_const = self.alloc_scratch(f"v_hash_const_{hi}", VLEN)
            v_shift = self.alloc_scratch(f"v_hash_shift_{hi}", VLEN)
            self.add("valu", ("vbroadcast", v_const, const_scalar))
            self.add("valu", ("vbroadcast", v_shift, shift_scalar))
            v_hash_consts.append(v_const)
            v_hash_shifts.append(v_shift)

        self.add("flow", ("pause",))

        # TWO-DESK PIPELINED: While hashing desk A, load for desk B
        NUM_DESKS = 2

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

        addr_tmp = [self.alloc_scratch(f"addr_tmp_{i}") for i in range(4)]
        batch_offset = self.alloc_scratch("batch_offset")
        iter_counter = self.alloc_scratch("iter_counter")

        eight_const = self.scratch_const(VLEN)
        sixteen_const = self.scratch_const(2 * VLEN)
        batch_size_const = self.scratch_const(batch_size)

        total_iterations = (batch_size // VLEN) * rounds // NUM_DESKS
        total_const = self.scratch_const(total_iterations)

        offset_regs = [self.alloc_scratch(f"off_{d}") for d in range(NUM_DESKS)]

        self.add("load", ("const", batch_offset, 0))
        self.add("load", ("const", iter_counter, 0))

        # PROLOGUE: Load first desk's data
        d0 = desks[0]
        self.instrs.append({
            "alu": [
                ("+", offset_regs[0], batch_offset, zero_const),
                ("+", addr_tmp[0], self.scratch["inp_indices_p"], batch_offset),
                ("+", addr_tmp[1], self.scratch["inp_values_p"], batch_offset),
            ],
        })
        self.instrs.append({
            "load": [
                ("vload", d0['idx'], addr_tmp[0]),
                ("vload", d0['val'], addr_tmp[1]),
            ],
        })
        self.add("valu", ("vbroadcast", d0['addr'], self.scratch["forest_values_p"]))
        self.add("valu", ("+", d0['addr'], d0['addr'], d0['idx']))
        for lane in range(0, VLEN, 2):
            self.instrs.append({
                "load": [
                    ("load", d0['node_val'] + lane, d0['addr'] + lane),
                    ("load", d0['node_val'] + lane + 1, d0['addr'] + lane + 1),
                ],
            })

        # === MAIN LOOP ===
        main_loop_start = len(self.instrs)

        # PIPELINED: While processing desk 0, load desk 1; while processing desk 1, load desk 0 (next iter)

        for current_desk in range(NUM_DESKS):
            curr = desks[current_desk]
            next_desk = (current_desk + 1) % NUM_DESKS
            next_d = desks[next_desk]
            is_last = (current_desk == NUM_DESKS - 1)

            # If last desk, compute next iteration's offset
            if is_last:
                self.instrs.append({
                    "alu": [("+", batch_offset, batch_offset, sixteen_const)],
                })
                self.instrs.append({
                    "alu": [("<", tmp_scalar, batch_offset, batch_size_const)],
                })
                self.add("flow", ("select", batch_offset, tmp_scalar, batch_offset, zero_const))

            # Compute next desk's offset
            next_offset = eight_const if not is_last else zero_const
            self.instrs.append({
                "alu": [
                    ("+", offset_regs[next_desk], batch_offset, next_offset if not is_last else zero_const),
                    ("+", addr_tmp[2], self.scratch["inp_indices_p"], offset_regs[current_desk]),
                    ("+", addr_tmp[3], self.scratch["inp_values_p"], offset_regs[current_desk]),
                ],
            })

            # XOR + compute next desk's load addresses
            self.instrs.append({
                "valu": [("^", curr['val'], curr['val'], curr['node_val'])],
                "alu": [
                    ("+", addr_tmp[0], self.scratch["inp_indices_p"], batch_offset if is_last else offset_regs[next_desk]),
                    ("+", addr_tmp[1], self.scratch["inp_values_p"], batch_offset if is_last else offset_regs[next_desk]),
                ],
            })

            # Hash stage 0 + vload next desk
            self.instrs.append({
                "valu": [
                    (HASH_STAGES[0][0], curr['tmp1'], curr['val'], v_hash_consts[0]),
                    (HASH_STAGES[0][3], curr['tmp2'], curr['val'], v_hash_shifts[0]),
                ],
                "load": [
                    ("vload", next_d['idx'], addr_tmp[0]),
                    ("vload", next_d['val'], addr_tmp[1]),
                ],
            })
            self.add("valu", (HASH_STAGES[0][2], curr['val'], curr['tmp1'], curr['tmp2']))

            # Hash stage 1 + broadcast for next's gather
            self.instrs.append({
                "valu": [
                    (HASH_STAGES[1][0], curr['tmp1'], curr['val'], v_hash_consts[1]),
                    (HASH_STAGES[1][3], curr['tmp2'], curr['val'], v_hash_shifts[1]),
                    ("vbroadcast", next_d['addr'], self.scratch["forest_values_p"]),
                ],
            })
            self.instrs.append({
                "valu": [
                    (HASH_STAGES[1][2], curr['val'], curr['tmp1'], curr['tmp2']),
                    ("+", next_d['addr'], next_d['addr'], next_d['idx']),
                ],
            })

            # Hash stages 2-5 + gather for next
            for hi in range(2, 6):
                lane = (hi - 2) * 2
                self.instrs.append({
                    "valu": [
                        (HASH_STAGES[hi][0], curr['tmp1'], curr['val'], v_hash_consts[hi]),
                        (HASH_STAGES[hi][3], curr['tmp2'], curr['val'], v_hash_shifts[hi]),
                    ],
                    "load": [
                        ("load", next_d['node_val'] + lane, next_d['addr'] + lane),
                        ("load", next_d['node_val'] + lane + 1, next_d['addr'] + lane + 1),
                    ],
                })
                self.add("valu", (HASH_STAGES[hi][2], curr['val'], curr['tmp1'], curr['tmp2']))

            # Branch computation + store current
            self.instrs.append({
                "valu": [
                    ("&", curr['tmp1'], curr['val'], v_one),
                    ("*", curr['idx'], curr['idx'], v_two),
                ],
            })
            self.add("valu", ("+", curr['idx'], curr['idx'], v_one))
            self.add("valu", ("+", curr['idx'], curr['idx'], curr['tmp1']))
            self.add("valu", ("<", curr['tmp1'], curr['idx'], v_n_nodes))
            self.add("flow", ("vselect", curr['idx'], curr['tmp1'], curr['idx'], v_zero))

            self.instrs.append({
                "store": [
                    ("vstore", addr_tmp[2], curr['idx']),
                    ("vstore", addr_tmp[3], curr['val']),
                ],
            })

        # Loop control
        self.instrs.append({
            "alu": [("+", iter_counter, iter_counter, one_const)],
        })
        self.instrs.append({
            "alu": [("<", tmp_scalar, iter_counter, total_const)],
        })
        self.add("flow", ("cond_jump", tmp_scalar, main_loop_start))
        self.instrs.append({"flow": [("pause",)]})


class KernelBuilderV4:
    """
    V4: True warp-style specialization with 6 VALU slot maximization.

    The key insight: the VALU engine has 6 slots but we're typically only using 2.
    If we batch hash operations across 3 desks, we can use all 6 slots:
    - 2 slots per desk (stage prep: op1 and op3)
    - 3 desks = 6 slots fully utilized

    This requires reorganizing to process hash stages across multiple desks
    before moving to the next stage.
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
        V4: 6-desk with maximized VALU utilization.

        Strategy:
        1. Use 6 desks (48 elements per iteration)
        2. During hash: pack 6 VALU ops (2 from each of 3 desks' stage prep)
        3. During load: maximize 2 load slots
        4. Maintain overlap between phases for different desk groups

        The key change: instead of processing desks sequentially, we process
        hash stages in parallel across desk pairs.
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

        v_hash_consts = []
        v_hash_shifts = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            const_scalar = self.scratch_const(val1)
            shift_scalar = self.scratch_const(val3)
            v_const = self.alloc_scratch(f"v_hash_const_{hi}", VLEN)
            v_shift = self.alloc_scratch(f"v_hash_shift_{hi}", VLEN)
            self.add("valu", ("vbroadcast", v_const, const_scalar))
            self.add("valu", ("vbroadcast", v_shift, shift_scalar))
            v_hash_consts.append(v_const)
            v_hash_shifts.append(v_shift)

        self.add("flow", ("pause",))

        # Use 4 desks (same as baseline) but with better VALU packing
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
        batch_size_const = self.scratch_const(batch_size)

        total_iterations = (batch_size // VLEN) * rounds // NUM_DESKS
        total_const = self.scratch_const(total_iterations)

        offset_regs = [self.alloc_scratch(f"off_{d}") for d in range(NUM_DESKS)]

        self.add("load", ("const", batch_offset, 0))
        self.add("load", ("const", iter_counter, 0))

        # === MAIN LOOP ===
        main_loop_start = len(self.instrs)

        # PHASE 1: LOAD - Same as V2 but optimized
        self.instrs.append({
            "alu": [
                ("+", offset_regs[0], batch_offset, zero_const),
                ("+", offset_regs[1], batch_offset, eight_const),
                ("+", offset_regs[2], batch_offset, sixteen_const),
                ("+", offset_regs[3], batch_offset, twentyfour_const),
            ],
        })

        # Load indices/values with better address packing
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

        # vloads (2 per cycle max)
        self.instrs.append({"load": [("vload", desks[0]['idx'], addr_tmp[0]), ("vload", desks[0]['val'], addr_tmp[1])]})
        self.instrs.append({"load": [("vload", desks[1]['idx'], addr_tmp[2]), ("vload", desks[1]['val'], addr_tmp[3])]})
        self.instrs.append({"load": [("vload", desks[2]['idx'], addr_tmp[4]), ("vload", desks[2]['val'], addr_tmp[5])]})

        self.instrs.append({
            "alu": [
                ("+", addr_tmp[6], self.scratch["inp_indices_p"], offset_regs[3]),
                ("+", addr_tmp[7], self.scratch["inp_values_p"], offset_regs[3]),
            ],
        })
        self.instrs.append({"load": [("vload", desks[3]['idx'], addr_tmp[6]), ("vload", desks[3]['val'], addr_tmp[7])]})

        # Compute gather addresses for all desks (pack 2 valu ops per cycle)
        self.instrs.append({
            "valu": [
                ("vbroadcast", desks[0]['addr'], self.scratch["forest_values_p"]),
                ("vbroadcast", desks[1]['addr'], self.scratch["forest_values_p"]),
            ],
        })
        self.instrs.append({
            "valu": [
                ("+", desks[0]['addr'], desks[0]['addr'], desks[0]['idx']),
                ("+", desks[1]['addr'], desks[1]['addr'], desks[1]['idx']),
            ],
        })
        self.instrs.append({
            "valu": [
                ("vbroadcast", desks[2]['addr'], self.scratch["forest_values_p"]),
                ("vbroadcast", desks[3]['addr'], self.scratch["forest_values_p"]),
            ],
        })
        self.instrs.append({
            "valu": [
                ("+", desks[2]['addr'], desks[2]['addr'], desks[2]['idx']),
                ("+", desks[3]['addr'], desks[3]['addr'], desks[3]['idx']),
            ],
        })

        # Perform all gathers - 4 desks x 4 cycles = 16 cycles
        for d in range(NUM_DESKS):
            for lane in range(0, VLEN, 2):
                self.instrs.append({
                    "load": [
                        ("load", desks[d]['node_val'] + lane, desks[d]['addr'] + lane),
                        ("load", desks[d]['node_val'] + lane + 1, desks[d]['addr'] + lane + 1),
                    ],
                })

        # PHASE 2: HASH - Use 6 VALU slots by batching across desks
        # XOR all desks (pack 4)
        self.instrs.append({
            "valu": [
                ("^", desks[0]['val'], desks[0]['val'], desks[0]['node_val']),
                ("^", desks[1]['val'], desks[1]['val'], desks[1]['node_val']),
                ("^", desks[2]['val'], desks[2]['val'], desks[2]['node_val']),
                ("^", desks[3]['val'], desks[3]['val'], desks[3]['node_val']),
            ],
        })

        # Hash stages - KEY OPTIMIZATION: pack 6 VALU ops per cycle
        # Each stage has 3 parts: prep1 (op1), prep2 (op3), final (op2)
        # We can pack prep1+prep2 for 3 desks = 6 ops
        for hi in range(6):
            # Prep for desks 0,1,2 (6 VALU ops)
            self.instrs.append({
                "valu": [
                    (HASH_STAGES[hi][0], desks[0]['tmp1'], desks[0]['val'], v_hash_consts[hi]),
                    (HASH_STAGES[hi][3], desks[0]['tmp2'], desks[0]['val'], v_hash_shifts[hi]),
                    (HASH_STAGES[hi][0], desks[1]['tmp1'], desks[1]['val'], v_hash_consts[hi]),
                    (HASH_STAGES[hi][3], desks[1]['tmp2'], desks[1]['val'], v_hash_shifts[hi]),
                    (HASH_STAGES[hi][0], desks[2]['tmp1'], desks[2]['val'], v_hash_consts[hi]),
                    (HASH_STAGES[hi][3], desks[2]['tmp2'], desks[2]['val'], v_hash_shifts[hi]),
                ],
            })
            # Prep for desk 3 + final for desks 0,1,2 (6 VALU ops)
            self.instrs.append({
                "valu": [
                    (HASH_STAGES[hi][0], desks[3]['tmp1'], desks[3]['val'], v_hash_consts[hi]),
                    (HASH_STAGES[hi][3], desks[3]['tmp2'], desks[3]['val'], v_hash_shifts[hi]),
                    (HASH_STAGES[hi][2], desks[0]['val'], desks[0]['tmp1'], desks[0]['tmp2']),
                    (HASH_STAGES[hi][2], desks[1]['val'], desks[1]['tmp1'], desks[1]['tmp2']),
                    (HASH_STAGES[hi][2], desks[2]['val'], desks[2]['tmp1'], desks[2]['tmp2']),
                ],
            })
            # Final for desk 3 (1 VALU op)
            self.add("valu", (HASH_STAGES[hi][2], desks[3]['val'], desks[3]['tmp1'], desks[3]['tmp2']))

        # Branch computation for all desks - pack well
        # AND and MUL for all 4 (8 ops, need 2 cycles at 6 max)
        self.instrs.append({
            "valu": [
                ("&", desks[0]['tmp1'], desks[0]['val'], v_one),
                ("*", desks[0]['idx'], desks[0]['idx'], v_two),
                ("&", desks[1]['tmp1'], desks[1]['val'], v_one),
                ("*", desks[1]['idx'], desks[1]['idx'], v_two),
                ("&", desks[2]['tmp1'], desks[2]['val'], v_one),
                ("*", desks[2]['idx'], desks[2]['idx'], v_two),
            ],
        })
        self.instrs.append({
            "valu": [
                ("&", desks[3]['tmp1'], desks[3]['val'], v_one),
                ("*", desks[3]['idx'], desks[3]['idx'], v_two),
            ],
        })

        # ADD 1 for all (4 ops)
        self.instrs.append({
            "valu": [
                ("+", desks[0]['idx'], desks[0]['idx'], v_one),
                ("+", desks[1]['idx'], desks[1]['idx'], v_one),
                ("+", desks[2]['idx'], desks[2]['idx'], v_one),
                ("+", desks[3]['idx'], desks[3]['idx'], v_one),
            ],
        })

        # ADD (val&1) for all (4 ops)
        self.instrs.append({
            "valu": [
                ("+", desks[0]['idx'], desks[0]['idx'], desks[0]['tmp1']),
                ("+", desks[1]['idx'], desks[1]['idx'], desks[1]['tmp1']),
                ("+", desks[2]['idx'], desks[2]['idx'], desks[2]['tmp1']),
                ("+", desks[3]['idx'], desks[3]['idx'], desks[3]['tmp1']),
            ],
        })

        # Fix desk 2's idx (we added v_two instead of v_one above - bug)
        # Actually let me fix that by using the correct constant

        # Bounds check for all (4 ops)
        self.instrs.append({
            "valu": [
                ("<", desks[0]['tmp1'], desks[0]['idx'], v_n_nodes),
                ("<", desks[1]['tmp1'], desks[1]['idx'], v_n_nodes),
                ("<", desks[2]['tmp1'], desks[2]['idx'], v_n_nodes),
                ("<", desks[3]['tmp1'], desks[3]['idx'], v_n_nodes),
            ],
        })

        # vselect for all (4 cycles - flow limited to 1/cycle)
        for d in range(NUM_DESKS):
            self.add("flow", ("vselect", desks[d]['idx'], desks[d]['tmp1'], desks[d]['idx'], v_zero))

        # PHASE 3: STORE
        for d in range(NUM_DESKS):
            self.instrs.append({
                "alu": [
                    ("+", addr_tmp[0], self.scratch["inp_indices_p"], offset_regs[d]),
                    ("+", addr_tmp[1], self.scratch["inp_values_p"], offset_regs[d]),
                ],
            })
            self.instrs.append({
                "store": [
                    ("vstore", addr_tmp[0], desks[d]['idx']),
                    ("vstore", addr_tmp[1], desks[d]['val']),
                ],
            })

        # LOOP CONTROL
        self.instrs.append({
            "alu": [
                ("+", batch_offset, batch_offset, thirtytwo_const),
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


class KernelBuilderV5:
    """
    V5: BEST OF BOTH WORLDS - V4's VALU packing + baseline's hash-gather overlap.

    Strategy:
    - Use 4 desks like V4
    - Pack hash operations across desks (6 VALU slots)
    - During later hash stages, overlap with gathering for NEXT iteration

    This should combine the 11.8% gain from V4 with additional gains from overlap.
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
        V5: Hash-gather overlap with VALU packing.

        Key changes from V4:
        1. During hash stages 2-5, overlap with loading next iteration's data
        2. Use pipelined structure: current hash + next gather

        This requires 8 desks worth of registers (4 current + 4 next).
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

        v_hash_consts = []
        v_hash_shifts = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            const_scalar = self.scratch_const(val1)
            shift_scalar = self.scratch_const(val3)
            v_const = self.alloc_scratch(f"v_hash_const_{hi}", VLEN)
            v_shift = self.alloc_scratch(f"v_hash_shift_{hi}", VLEN)
            self.add("valu", ("vbroadcast", v_const, const_scalar))
            self.add("valu", ("vbroadcast", v_shift, shift_scalar))
            v_hash_consts.append(v_const)
            v_hash_shifts.append(v_shift)

        self.add("flow", ("pause",))

        # TWO GROUPS of 4 desks: A (current iteration) and B (next iteration for overlap)
        NUM_DESKS = 4

        # Group A: desks for current iteration hash
        desks_a = []
        for d in range(NUM_DESKS):
            desk = {
                'idx': self.alloc_scratch(f"v_idx_a{d}", VLEN),
                'val': self.alloc_scratch(f"v_val_a{d}", VLEN),
                'node_val': self.alloc_scratch(f"v_node_a{d}", VLEN),
                'addr': self.alloc_scratch(f"v_addr_a{d}", VLEN),
                'tmp1': self.alloc_scratch(f"v_tmp1_a{d}", VLEN),
                'tmp2': self.alloc_scratch(f"v_tmp2_a{d}", VLEN),
            }
            desks_a.append(desk)

        # Group B: desks for next iteration gather (will swap with A)
        desks_b = []
        for d in range(NUM_DESKS):
            desk = {
                'idx': self.alloc_scratch(f"v_idx_b{d}", VLEN),
                'val': self.alloc_scratch(f"v_val_b{d}", VLEN),
                'node_val': self.alloc_scratch(f"v_node_b{d}", VLEN),
                'addr': self.alloc_scratch(f"v_addr_b{d}", VLEN),
                'tmp1': self.alloc_scratch(f"v_tmp1_b{d}", VLEN),
                'tmp2': self.alloc_scratch(f"v_tmp2_b{d}", VLEN),
            }
            desks_b.append(desk)

        addr_tmp = [self.alloc_scratch(f"addr_tmp_{i}") for i in range(8)]
        batch_offset = self.alloc_scratch("batch_offset")
        batch_offset_next = self.alloc_scratch("batch_offset_next")
        iter_counter = self.alloc_scratch("iter_counter")

        eight_const = self.scratch_const(VLEN)
        sixteen_const = self.scratch_const(2 * VLEN)
        twentyfour_const = self.scratch_const(3 * VLEN)
        thirtytwo_const = self.scratch_const(4 * VLEN)
        batch_size_const = self.scratch_const(batch_size)

        total_iterations = (batch_size // VLEN) * rounds // NUM_DESKS
        total_const = self.scratch_const(total_iterations)

        offset_regs_a = [self.alloc_scratch(f"off_a{d}") for d in range(NUM_DESKS)]
        offset_regs_b = [self.alloc_scratch(f"off_b{d}") for d in range(NUM_DESKS)]

        self.add("load", ("const", batch_offset, 0))
        self.add("load", ("const", iter_counter, 0))

        # PROLOGUE: Load first iteration into desks_a
        self.instrs.append({
            "alu": [
                ("+", offset_regs_a[0], batch_offset, zero_const),
                ("+", offset_regs_a[1], batch_offset, eight_const),
                ("+", offset_regs_a[2], batch_offset, sixteen_const),
                ("+", offset_regs_a[3], batch_offset, twentyfour_const),
            ],
        })

        # Load all desk A data
        self.instrs.append({
            "alu": [
                ("+", addr_tmp[0], self.scratch["inp_indices_p"], offset_regs_a[0]),
                ("+", addr_tmp[1], self.scratch["inp_values_p"], offset_regs_a[0]),
                ("+", addr_tmp[2], self.scratch["inp_indices_p"], offset_regs_a[1]),
                ("+", addr_tmp[3], self.scratch["inp_values_p"], offset_regs_a[1]),
                ("+", addr_tmp[4], self.scratch["inp_indices_p"], offset_regs_a[2]),
                ("+", addr_tmp[5], self.scratch["inp_values_p"], offset_regs_a[2]),
            ],
        })
        self.instrs.append({"load": [("vload", desks_a[0]['idx'], addr_tmp[0]), ("vload", desks_a[0]['val'], addr_tmp[1])]})
        self.instrs.append({"load": [("vload", desks_a[1]['idx'], addr_tmp[2]), ("vload", desks_a[1]['val'], addr_tmp[3])]})
        self.instrs.append({"load": [("vload", desks_a[2]['idx'], addr_tmp[4]), ("vload", desks_a[2]['val'], addr_tmp[5])]})
        self.instrs.append({
            "alu": [
                ("+", addr_tmp[6], self.scratch["inp_indices_p"], offset_regs_a[3]),
                ("+", addr_tmp[7], self.scratch["inp_values_p"], offset_regs_a[3]),
            ],
        })
        self.instrs.append({"load": [("vload", desks_a[3]['idx'], addr_tmp[6]), ("vload", desks_a[3]['val'], addr_tmp[7])]})

        # Compute gather addresses for A
        self.instrs.append({
            "valu": [
                ("vbroadcast", desks_a[0]['addr'], self.scratch["forest_values_p"]),
                ("vbroadcast", desks_a[1]['addr'], self.scratch["forest_values_p"]),
            ],
        })
        self.instrs.append({
            "valu": [
                ("+", desks_a[0]['addr'], desks_a[0]['addr'], desks_a[0]['idx']),
                ("+", desks_a[1]['addr'], desks_a[1]['addr'], desks_a[1]['idx']),
            ],
        })
        self.instrs.append({
            "valu": [
                ("vbroadcast", desks_a[2]['addr'], self.scratch["forest_values_p"]),
                ("vbroadcast", desks_a[3]['addr'], self.scratch["forest_values_p"]),
            ],
        })
        self.instrs.append({
            "valu": [
                ("+", desks_a[2]['addr'], desks_a[2]['addr'], desks_a[2]['idx']),
                ("+", desks_a[3]['addr'], desks_a[3]['addr'], desks_a[3]['idx']),
            ],
        })

        # Gather for A
        for d in range(NUM_DESKS):
            for lane in range(0, VLEN, 2):
                self.instrs.append({
                    "load": [
                        ("load", desks_a[d]['node_val'] + lane, desks_a[d]['addr'] + lane),
                        ("load", desks_a[d]['node_val'] + lane + 1, desks_a[d]['addr'] + lane + 1),
                    ],
                })

        # === MAIN LOOP ===
        # Structure: Hash A | Load B -> Store A | Gather B -> Hash B | Load A (next) -> Store B | Gather A (next)
        # Simplified: we alternate between processing A and B, with overlap

        main_loop_start = len(self.instrs)

        # For simplicity, let's just do the hash+overlap for A, then store A, then swap roles
        # This is a simplified version of full software pipelining

        # Compute next iteration offset
        self.instrs.append({
            "alu": [("+", batch_offset_next, batch_offset, thirtytwo_const)],
        })
        self.instrs.append({
            "alu": [("<", tmp_scalar, batch_offset_next, batch_size_const)],
        })
        self.add("flow", ("select", batch_offset_next, tmp_scalar, batch_offset_next, zero_const))

        # Compute B's offsets
        self.instrs.append({
            "alu": [
                ("+", offset_regs_b[0], batch_offset_next, zero_const),
                ("+", offset_regs_b[1], batch_offset_next, eight_const),
                ("+", offset_regs_b[2], batch_offset_next, sixteen_const),
                ("+", offset_regs_b[3], batch_offset_next, twentyfour_const),
            ],
        })

        # XOR for A (start hash)
        self.instrs.append({
            "valu": [
                ("^", desks_a[0]['val'], desks_a[0]['val'], desks_a[0]['node_val']),
                ("^", desks_a[1]['val'], desks_a[1]['val'], desks_a[1]['node_val']),
                ("^", desks_a[2]['val'], desks_a[2]['val'], desks_a[2]['node_val']),
                ("^", desks_a[3]['val'], desks_a[3]['val'], desks_a[3]['node_val']),
            ],
        })

        # Hash stage 0 for A + compute B's load addresses
        self.instrs.append({
            "valu": [
                (HASH_STAGES[0][0], desks_a[0]['tmp1'], desks_a[0]['val'], v_hash_consts[0]),
                (HASH_STAGES[0][3], desks_a[0]['tmp2'], desks_a[0]['val'], v_hash_shifts[0]),
                (HASH_STAGES[0][0], desks_a[1]['tmp1'], desks_a[1]['val'], v_hash_consts[0]),
                (HASH_STAGES[0][3], desks_a[1]['tmp2'], desks_a[1]['val'], v_hash_shifts[0]),
                (HASH_STAGES[0][0], desks_a[2]['tmp1'], desks_a[2]['val'], v_hash_consts[0]),
                (HASH_STAGES[0][3], desks_a[2]['tmp2'], desks_a[2]['val'], v_hash_shifts[0]),
            ],
            "alu": [
                ("+", addr_tmp[0], self.scratch["inp_indices_p"], offset_regs_b[0]),
                ("+", addr_tmp[1], self.scratch["inp_values_p"], offset_regs_b[0]),
            ],
        })

        # Hash stage 0 desk 3 prep + final for 0,1,2 + vload B0
        self.instrs.append({
            "valu": [
                (HASH_STAGES[0][0], desks_a[3]['tmp1'], desks_a[3]['val'], v_hash_consts[0]),
                (HASH_STAGES[0][3], desks_a[3]['tmp2'], desks_a[3]['val'], v_hash_shifts[0]),
                (HASH_STAGES[0][2], desks_a[0]['val'], desks_a[0]['tmp1'], desks_a[0]['tmp2']),
                (HASH_STAGES[0][2], desks_a[1]['val'], desks_a[1]['tmp1'], desks_a[1]['tmp2']),
                (HASH_STAGES[0][2], desks_a[2]['val'], desks_a[2]['tmp1'], desks_a[2]['tmp2']),
            ],
            "load": [
                ("vload", desks_b[0]['idx'], addr_tmp[0]),
                ("vload", desks_b[0]['val'], addr_tmp[1]),
            ],
        })

        # Hash stage 0 desk 3 final + compute B1 addresses + vload B1
        self.instrs.append({
            "valu": [
                (HASH_STAGES[0][2], desks_a[3]['val'], desks_a[3]['tmp1'], desks_a[3]['tmp2']),
            ],
            "alu": [
                ("+", addr_tmp[2], self.scratch["inp_indices_p"], offset_regs_b[1]),
                ("+", addr_tmp[3], self.scratch["inp_values_p"], offset_regs_b[1]),
            ],
        })

        # Continue hash stages 1-5 with overlap
        for hi in range(1, 6):
            # Prep for desks 0,1,2 + vload B
            if hi == 1:
                load_desk_idx = 1
            elif hi == 2:
                load_desk_idx = 2
            elif hi == 3:
                load_desk_idx = 3
            else:
                load_desk_idx = None  # No more vloads needed

            if load_desk_idx is not None and load_desk_idx < NUM_DESKS:
                self.instrs.append({
                    "valu": [
                        (HASH_STAGES[hi][0], desks_a[0]['tmp1'], desks_a[0]['val'], v_hash_consts[hi]),
                        (HASH_STAGES[hi][3], desks_a[0]['tmp2'], desks_a[0]['val'], v_hash_shifts[hi]),
                        (HASH_STAGES[hi][0], desks_a[1]['tmp1'], desks_a[1]['val'], v_hash_consts[hi]),
                        (HASH_STAGES[hi][3], desks_a[1]['tmp2'], desks_a[1]['val'], v_hash_shifts[hi]),
                        (HASH_STAGES[hi][0], desks_a[2]['tmp1'], desks_a[2]['val'], v_hash_consts[hi]),
                        (HASH_STAGES[hi][3], desks_a[2]['tmp2'], desks_a[2]['val'], v_hash_shifts[hi]),
                    ],
                    "load": [
                        ("vload", desks_b[load_desk_idx]['idx'], addr_tmp[load_desk_idx*2]),
                        ("vload", desks_b[load_desk_idx]['val'], addr_tmp[load_desk_idx*2+1]),
                    ],
                })
            else:
                self.instrs.append({
                    "valu": [
                        (HASH_STAGES[hi][0], desks_a[0]['tmp1'], desks_a[0]['val'], v_hash_consts[hi]),
                        (HASH_STAGES[hi][3], desks_a[0]['tmp2'], desks_a[0]['val'], v_hash_shifts[hi]),
                        (HASH_STAGES[hi][0], desks_a[1]['tmp1'], desks_a[1]['val'], v_hash_consts[hi]),
                        (HASH_STAGES[hi][3], desks_a[1]['tmp2'], desks_a[1]['val'], v_hash_shifts[hi]),
                        (HASH_STAGES[hi][0], desks_a[2]['tmp1'], desks_a[2]['val'], v_hash_consts[hi]),
                        (HASH_STAGES[hi][3], desks_a[2]['tmp2'], desks_a[2]['val'], v_hash_shifts[hi]),
                    ],
                })

            # Compute next desk B addresses if needed
            if load_desk_idx is not None and load_desk_idx + 1 < NUM_DESKS:
                self.instrs.append({
                    "alu": [
                        ("+", addr_tmp[(load_desk_idx+1)*2], self.scratch["inp_indices_p"], offset_regs_b[load_desk_idx+1]),
                        ("+", addr_tmp[(load_desk_idx+1)*2+1], self.scratch["inp_values_p"], offset_regs_b[load_desk_idx+1]),
                    ],
                })

            # Prep desk 3 + final for 0,1,2
            self.instrs.append({
                "valu": [
                    (HASH_STAGES[hi][0], desks_a[3]['tmp1'], desks_a[3]['val'], v_hash_consts[hi]),
                    (HASH_STAGES[hi][3], desks_a[3]['tmp2'], desks_a[3]['val'], v_hash_shifts[hi]),
                    (HASH_STAGES[hi][2], desks_a[0]['val'], desks_a[0]['tmp1'], desks_a[0]['tmp2']),
                    (HASH_STAGES[hi][2], desks_a[1]['val'], desks_a[1]['tmp1'], desks_a[1]['tmp2']),
                    (HASH_STAGES[hi][2], desks_a[2]['val'], desks_a[2]['tmp1'], desks_a[2]['tmp2']),
                ],
            })

            # Final for desk 3
            self.add("valu", (HASH_STAGES[hi][2], desks_a[3]['val'], desks_a[3]['tmp1'], desks_a[3]['tmp2']))

        # Compute gather addresses for B while doing A's branch
        self.instrs.append({
            "valu": [
                ("&", desks_a[0]['tmp1'], desks_a[0]['val'], v_one),
                ("*", desks_a[0]['idx'], desks_a[0]['idx'], v_two),
                ("&", desks_a[1]['tmp1'], desks_a[1]['val'], v_one),
                ("*", desks_a[1]['idx'], desks_a[1]['idx'], v_two),
                ("vbroadcast", desks_b[0]['addr'], self.scratch["forest_values_p"]),
                ("vbroadcast", desks_b[1]['addr'], self.scratch["forest_values_p"]),
            ],
        })

        self.instrs.append({
            "valu": [
                ("&", desks_a[2]['tmp1'], desks_a[2]['val'], v_one),
                ("*", desks_a[2]['idx'], desks_a[2]['idx'], v_two),
                ("&", desks_a[3]['tmp1'], desks_a[3]['val'], v_one),
                ("*", desks_a[3]['idx'], desks_a[3]['idx'], v_two),
                ("+", desks_b[0]['addr'], desks_b[0]['addr'], desks_b[0]['idx']),
                ("+", desks_b[1]['addr'], desks_b[1]['addr'], desks_b[1]['idx']),
            ],
        })

        self.instrs.append({
            "valu": [
                ("+", desks_a[0]['idx'], desks_a[0]['idx'], v_one),
                ("+", desks_a[1]['idx'], desks_a[1]['idx'], v_one),
                ("+", desks_a[2]['idx'], desks_a[2]['idx'], v_one),
                ("+", desks_a[3]['idx'], desks_a[3]['idx'], v_one),
                ("vbroadcast", desks_b[2]['addr'], self.scratch["forest_values_p"]),
                ("vbroadcast", desks_b[3]['addr'], self.scratch["forest_values_p"]),
            ],
        })

        self.instrs.append({
            "valu": [
                ("+", desks_a[0]['idx'], desks_a[0]['idx'], desks_a[0]['tmp1']),
                ("+", desks_a[1]['idx'], desks_a[1]['idx'], desks_a[1]['tmp1']),
                ("+", desks_a[2]['idx'], desks_a[2]['idx'], desks_a[2]['tmp1']),
                ("+", desks_a[3]['idx'], desks_a[3]['idx'], desks_a[3]['tmp1']),
                ("+", desks_b[2]['addr'], desks_b[2]['addr'], desks_b[2]['idx']),
                ("+", desks_b[3]['addr'], desks_b[3]['addr'], desks_b[3]['idx']),
            ],
        })

        # Bounds check for A + gather for B
        self.instrs.append({
            "valu": [
                ("<", desks_a[0]['tmp1'], desks_a[0]['idx'], v_n_nodes),
                ("<", desks_a[1]['tmp1'], desks_a[1]['idx'], v_n_nodes),
                ("<", desks_a[2]['tmp1'], desks_a[2]['idx'], v_n_nodes),
                ("<", desks_a[3]['tmp1'], desks_a[3]['idx'], v_n_nodes),
            ],
            "load": [
                ("load", desks_b[0]['node_val'] + 0, desks_b[0]['addr'] + 0),
                ("load", desks_b[0]['node_val'] + 1, desks_b[0]['addr'] + 1),
            ],
        })

        # vselect for A + gather for B (interleaved)
        gather_lane = 2
        for d in range(NUM_DESKS):
            desk_b_idx = gather_lane // 8
            lane_in_desk = gather_lane % 8
            if desk_b_idx < NUM_DESKS and lane_in_desk < VLEN:
                self.instrs.append({
                    "flow": [("vselect", desks_a[d]['idx'], desks_a[d]['tmp1'], desks_a[d]['idx'], v_zero)],
                    "load": [
                        ("load", desks_b[desk_b_idx]['node_val'] + lane_in_desk, desks_b[desk_b_idx]['addr'] + lane_in_desk),
                        ("load", desks_b[desk_b_idx]['node_val'] + lane_in_desk + 1, desks_b[desk_b_idx]['addr'] + lane_in_desk + 1),
                    ],
                })
            else:
                self.add("flow", ("vselect", desks_a[d]['idx'], desks_a[d]['tmp1'], desks_a[d]['idx'], v_zero))
            gather_lane += 2

        # Store A + continue gather B
        for d in range(NUM_DESKS):
            desk_b_idx = gather_lane // 8
            lane_in_desk = gather_lane % 8
            self.instrs.append({
                "alu": [
                    ("+", addr_tmp[0], self.scratch["inp_indices_p"], offset_regs_a[d]),
                    ("+", addr_tmp[1], self.scratch["inp_values_p"], offset_regs_a[d]),
                ],
            })
            if desk_b_idx < NUM_DESKS and lane_in_desk < VLEN:
                self.instrs.append({
                    "store": [
                        ("vstore", addr_tmp[0], desks_a[d]['idx']),
                        ("vstore", addr_tmp[1], desks_a[d]['val']),
                    ],
                    "load": [
                        ("load", desks_b[desk_b_idx]['node_val'] + lane_in_desk, desks_b[desk_b_idx]['addr'] + lane_in_desk),
                        ("load", desks_b[desk_b_idx]['node_val'] + lane_in_desk + 1, desks_b[desk_b_idx]['addr'] + lane_in_desk + 1),
                    ],
                })
            else:
                self.instrs.append({
                    "store": [
                        ("vstore", addr_tmp[0], desks_a[d]['idx']),
                        ("vstore", addr_tmp[1], desks_a[d]['val']),
                    ],
                })
            gather_lane += 2

        # Finish remaining gathers for B
        while gather_lane < NUM_DESKS * VLEN:
            desk_b_idx = gather_lane // 8
            lane_in_desk = gather_lane % 8
            if desk_b_idx < NUM_DESKS and lane_in_desk < VLEN:
                self.instrs.append({
                    "load": [
                        ("load", desks_b[desk_b_idx]['node_val'] + lane_in_desk, desks_b[desk_b_idx]['addr'] + lane_in_desk),
                        ("load", desks_b[desk_b_idx]['node_val'] + lane_in_desk + 1, desks_b[desk_b_idx]['addr'] + lane_in_desk + 1),
                    ],
                })
            gather_lane += 2

        # Swap A and B (copy B to A for next iteration)
        # Actually, simpler: just update batch_offset and re-copy from B to A
        # This adds overhead but keeps logic simpler

        # Copy B to A
        for d in range(NUM_DESKS):
            # Copy idx
            for lane in range(VLEN):
                self.instrs.append({
                    "alu": [("+", desks_a[d]['idx'] + lane, desks_b[d]['idx'] + lane, zero_const)],
                })
            # Copy val
            for lane in range(VLEN):
                self.instrs.append({
                    "alu": [("+", desks_a[d]['val'] + lane, desks_b[d]['val'] + lane, zero_const)],
                })
            # Copy node_val
            for lane in range(VLEN):
                self.instrs.append({
                    "alu": [("+", desks_a[d]['node_val'] + lane, desks_b[d]['node_val'] + lane, zero_const)],
                })
            # Copy addr
            for lane in range(VLEN):
                self.instrs.append({
                    "alu": [("+", desks_a[d]['addr'] + lane, desks_b[d]['addr'] + lane, zero_const)],
                })

        # Copy offset_regs
        for d in range(NUM_DESKS):
            self.instrs.append({
                "alu": [("+", offset_regs_a[d], offset_regs_b[d], zero_const)],
            })

        # Update batch_offset
        self.instrs.append({
            "alu": [("+", batch_offset, batch_offset_next, zero_const)],
        })

        # Loop control
        self.instrs.append({
            "alu": [("+", iter_counter, iter_counter, one_const)],
        })
        self.instrs.append({
            "alu": [("<", tmp_scalar, iter_counter, total_const)],
        })
        self.add("flow", ("cond_jump", tmp_scalar, main_loop_start))
        self.instrs.append({"flow": [("pause",)]})


BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
    version: int = 1,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}, version={version}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    if version == 1:
        kb = KernelBuilder()
    elif version == 2:
        kb = KernelBuilderV2()
    elif version == 3:
        kb = KernelBuilderV3()
    elif version == 4:
        kb = KernelBuilderV4()
    elif version == 5:
        kb = KernelBuilderV5()
    else:
        raise ValueError(f"Unknown version {version}")

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
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline (147734): ", BASELINE / machine.cycle)
    print("Speedup over current best (9793): ", 9793 / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_kernel_v1(self):
        """Test V1: 2-desk phase separation"""
        do_kernel_test(10, 16, 256, version=1)

    def test_kernel_v2(self):
        """Test V2: 4-desk phase batching"""
        do_kernel_test(10, 16, 256, version=2)

    def test_kernel_v3(self):
        """Test V3: 2-desk pipelined hybrid"""
        do_kernel_test(10, 16, 256, version=3)

    def test_kernel_v4(self):
        """Test V4: 6-VALU slot maximization"""
        do_kernel_test(10, 16, 256, version=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Run correctness tests")
    parser.add_argument("--version", type=int, default=1, help="Kernel version (1, 2, or 3)")
    parser.add_argument("--all", action="store_true", help="Test all versions")
    args = parser.parse_args()

    if args.check:
        # Run tests
        unittest.main(argv=[''], exit=False)
    elif args.all:
        print("=" * 60)
        print("Testing all versions")
        print("=" * 60)
        for v in [1, 2, 3, 4, 5]:
            print(f"\n{'='*60}")
            print(f"Version {v}")
            print("=" * 60)
            try:
                cycles = do_kernel_test(10, 16, 256, version=v)
            except Exception as e:
                print(f"Error: {e}")
    else:
        do_kernel_test(10, 16, 256, version=args.version)
