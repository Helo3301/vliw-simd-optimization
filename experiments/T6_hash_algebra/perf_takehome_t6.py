"""
# Experiment T6: Algebraic Strength Reduction on Hash

This is a COPY of perf_takehome.py with FMA (multiply_add) optimization
applied to hash stages that can be algebraically reduced.

Stage 0: val = (val + C0) + (val << 12) = val * 4097 + C0
Stage 2: val = (val + C2) + (val << 5) = val * 33 + C2
Stage 4: val = (val + C4) + (val << 3) = val * 9 + C4

These stages can use multiply_add instead of 3 operations.
"""

from collections import defaultdict
import random
import unittest
import sys
import os

# Add parent directory to path to import problem.py
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


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

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

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            # OPTIMIZATION: Pack the two independent operations together
            # tmp1 = val op1 const  AND  tmp2 = val op3 shift  (both read val, independent)
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_packed(self, slots: list[tuple[str, tuple]]):
        """
        VLIW packing: combine independent operations into single cycles.
        This version packs consecutive ALU ops that don't have dependencies.
        """
        instrs = []
        i = 0
        while i < len(slots):
            engine, slot = slots[i]

            # Skip debug instructions (they don't count as cycles)
            if engine == "debug":
                instrs.append({engine: [slot]})
                i += 1
                continue

            # Try to pack multiple ALU ops together
            if engine == "alu" and i + 1 < len(slots) and slots[i + 1][0] == "alu":
                # Check if we can pack these two ALU ops
                # They're packable if the second doesn't read the first's destination
                slot1 = slot
                slot2 = slots[i + 1][1]
                dest1 = slot1[1]  # destination of first op

                # Check if slot2 reads dest1 (positions 2 and 3 are operands)
                reads_dest1 = (slot2[2] == dest1 or slot2[3] == dest1)

                if not reads_dest1:
                    # Pack them together!
                    instrs.append({"alu": [slot1, slot2]})
                    i += 2
                    continue

            # Default: single slot per instruction
            instrs.append({engine: [slot]})
            i += 1

        return instrs

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        SIMD vectorized implementation processing 8 elements at a time.

        T6 OPTIMIZATION: Uses multiply_add (FMA) for hash stages 0, 2, 4 where
        algebraic reduction is possible:
        - Stage 0: val = (val + C0) + (val << 12) = val * 4097 + C0
        - Stage 2: val = (val + C2) + (val << 5) = val * 33 + C2
        - Stage 4: val = (val + C4) + (val << 3) = val * 9 + C4
        """
        # Vector scratch (8 elements each)
        v_idx = self.alloc_scratch("v_idx", VLEN)
        v_val = self.alloc_scratch("v_val", VLEN)
        v_node_val = self.alloc_scratch("v_node_val", VLEN)
        v_tmp1 = self.alloc_scratch("v_tmp1", VLEN)
        v_tmp2 = self.alloc_scratch("v_tmp2", VLEN)
        v_addr = self.alloc_scratch("v_addr", VLEN)  # for gather addresses

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

        # Vector constants (broadcast scalars to vectors)
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

        # Hash constants (broadcast each to vector)
        # T6: Pre-compute FMA multipliers for stages 0, 2, 4
        # Stage 0: val * 4097 + C0 (4097 = 1 + 2^12)
        # Stage 2: val * 33 + C2 (33 = 1 + 2^5)
        # Stage 4: val * 9 + C4 (9 = 1 + 2^3)
        FMA_MULTIPLIERS = {
            0: 4097,  # 1 + 2^12
            2: 33,    # 1 + 2^5
            4: 9,     # 1 + 2^3
        }

        v_hash_consts = []
        v_hash_shifts = []
        v_fma_multipliers = {}  # For FMA-optimizable stages

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            const_scalar = self.scratch_const(val1)
            shift_scalar = self.scratch_const(val3)
            v_const = self.alloc_scratch(f"v_hash_const_{hi}", VLEN)
            v_shift = self.alloc_scratch(f"v_hash_shift_{hi}", VLEN)
            self.add("valu", ("vbroadcast", v_const, const_scalar))
            self.add("valu", ("vbroadcast", v_shift, shift_scalar))
            v_hash_consts.append(v_const)
            v_hash_shifts.append(v_shift)

            # For FMA-optimizable stages, also broadcast the multiplier
            if hi in FMA_MULTIPLIERS:
                mult_scalar = self.scratch_const(FMA_MULTIPLIERS[hi])
                v_mult = self.alloc_scratch(f"v_fma_mult_{hi}", VLEN)
                self.add("valu", ("vbroadcast", v_mult, mult_scalar))
                v_fma_multipliers[hi] = v_mult

        self.add("flow", ("pause",))

        # === FOUR DESKS - UNROLLED BY 4 ===
        # Process 4 groups per loop iteration, overlapping hash with gather

        # Allocate 4 sets of vector registers
        desks = []
        for d in range(4):
            desk = {
                'idx': self.alloc_scratch(f"v_idx_{d}", VLEN),
                'val': self.alloc_scratch(f"v_val_{d}", VLEN),
                'node_val': self.alloc_scratch(f"v_node_{d}", VLEN),
                'addr': self.alloc_scratch(f"v_addr_{d}", VLEN),
                'tmp1': self.alloc_scratch(f"v_tmp1_{d}", VLEN),
                'tmp2': self.alloc_scratch(f"v_tmp2_{d}", VLEN),
            }
            desks.append(desk)

        # Scalar temps and offsets
        addr_tmp = [self.alloc_scratch(f"addr_tmp_{i}") for i in range(4)]
        batch_offset = self.alloc_scratch("batch_offset")
        iter_counter = self.alloc_scratch("iter_counter")

        eight_const = self.scratch_const(VLEN)
        thirtytwo_const = self.scratch_const(4 * VLEN)
        batch_size_const = self.scratch_const(batch_size)
        total_iterations = (batch_size // VLEN) * rounds // 4  # Divided by 4 since we unroll
        total_const = self.scratch_const(total_iterations)

        self.add("load", ("const", batch_offset, 0))
        self.add("load", ("const", iter_counter, 0))

        # Offset constants for packing
        offset_regs = [self.alloc_scratch(f"off_{d}") for d in range(4)]
        sixteen_const = self.scratch_const(2*VLEN)
        twentyfour_const = self.scratch_const(3*VLEN)

        # Dedicated registers for desk 0's pre-loaded addresses (to avoid conflicts with addr_tmp)
        addr_d0_idx = self.alloc_scratch("addr_d0_idx")
        addr_d0_val = self.alloc_scratch("addr_d0_val")

        # Compute offsets + desk 0 addresses for FIRST iteration (before loop)
        d0 = desks[0]
        self.instrs.append({
            "alu": [
                ("+", offset_regs[0], batch_offset, zero_const),
                ("+", offset_regs[1], batch_offset, eight_const),
                ("+", offset_regs[2], batch_offset, sixteen_const),
                ("+", offset_regs[3], batch_offset, twentyfour_const),
                ("+", addr_d0_idx, self.scratch["inp_indices_p"], batch_offset),
                ("+", addr_d0_val, self.scratch["inp_values_p"], batch_offset),
            ],
        })

        # FIRST ITERATION PROLOGUE - runs once before the loop
        # Load desk 0 + broadcast for gather
        self.instrs.append({
            "load": [
                ("vload", d0['idx'], addr_d0_idx),
                ("vload", d0['val'], addr_d0_val),
            ],
            "valu": [
                ("vbroadcast", d0['addr'], self.scratch["forest_values_p"]),
            ],
        })

        # Compute gather addresses
        self.add("valu", ("+", d0['addr'], d0['addr'], d0['idx']))

        # Gather for desk 0 - pack 2 loads per cycle
        for lane in range(0, VLEN, 2):
            self.instrs.append({"load": [
                ("load", d0['node_val'] + lane, d0['addr'] + lane),
                ("load", d0['node_val'] + lane + 1, d0['addr'] + lane + 1),
            ]})

        # MAIN LOOP STARTS HERE - desk 0 data ready (from prologue for iter 1, from desk 3 for iter 2+)
        main_loop_start = len(self.instrs)

        # Helper function to emit hash for a desk with T6 FMA optimization
        def emit_hash_stage_fma(curr, hi, v_hash_consts, v_hash_shifts, v_fma_multipliers):
            """
            Emit optimized hash stage using FMA where applicable.
            Returns list of instruction dicts to append.
            """
            op1, val1, op2, op3, val3 = HASH_STAGES[hi]

            # Check if this stage can use FMA
            # FMA works when: op1 == '+' and op2 == '+' (additive stages)
            if hi in v_fma_multipliers:
                # FMA optimization: val = val * multiplier + constant
                # Single instruction instead of 3!
                return [("multiply_add", curr['val'], curr['val'], v_fma_multipliers[hi], v_hash_consts[hi])]
            else:
                # Standard hash: 2 parallel ops + 1 combine
                # tmp1 = val op1 const, tmp2 = val op3 shift, then val = tmp1 op2 tmp2
                return None  # Signal to use standard approach

        # For desks 0, 1, 2: hash current while gathering next
        for d in range(3):
            curr = desks[d]
            next_d = desks[d + 1]

            # XOR for current + Compute load addresses for next (packed - different engines)
            self.instrs.append({
                "valu": [("^", curr['val'], curr['val'], curr['node_val'])],
                "alu": [
                    ("+", addr_tmp[2], self.scratch["inp_indices_p"], offset_regs[d + 1]),
                    ("+", addr_tmp[3], self.scratch["inp_values_p"], offset_regs[d + 1]),
                ],
            })

            # Hash stage 0 - T6 OPTIMIZATION: Use FMA!
            # val = val * 4097 + C0 (single multiply_add)
            self.instrs.append({
                "valu": [
                    ("multiply_add", curr['val'], curr['val'], v_fma_multipliers[0], v_hash_consts[0]),
                ],
                "load": [
                    ("vload", next_d['idx'], addr_tmp[2]),
                    ("vload", next_d['val'], addr_tmp[3]),
                ],
            })
            # No extra cycle needed - FMA replaces 3 ops with 1!

            # Hash stage 1 - Standard (XOR-based, cannot FMA) + broadcast for next's gather addr
            self.instrs.append({"valu": [
                (HASH_STAGES[1][0], curr['tmp1'], curr['val'], v_hash_consts[1]),
                (HASH_STAGES[1][3], curr['tmp2'], curr['val'], v_hash_shifts[1]),
                ("vbroadcast", next_d['addr'], self.scratch["forest_values_p"]),
            ]})
            # Hash1 final + compute next's gather addresses (both depend on prev cycle, independent)
            self.instrs.append({"valu": [
                (HASH_STAGES[1][2], curr['val'], curr['tmp1'], curr['tmp2']),
                ("+", next_d['addr'], next_d['addr'], next_d['idx']),
            ]})

            # Hash stage 2 - T6 OPTIMIZATION: Use FMA!
            # val = val * 33 + C2 + gather for next
            self.instrs.append({
                "valu": [
                    ("multiply_add", curr['val'], curr['val'], v_fma_multipliers[2], v_hash_consts[2]),
                ],
                "load": [
                    ("load", next_d['node_val'] + 0, next_d['addr'] + 0),
                    ("load", next_d['node_val'] + 1, next_d['addr'] + 1),
                ],
            })

            # Hash stage 3 - Standard (XOR combine, cannot FMA) + gather
            self.instrs.append({
                "valu": [
                    (HASH_STAGES[3][0], curr['tmp1'], curr['val'], v_hash_consts[3]),
                    (HASH_STAGES[3][3], curr['tmp2'], curr['val'], v_hash_shifts[3]),
                ],
                "load": [
                    ("load", next_d['node_val'] + 2, next_d['addr'] + 2),
                    ("load", next_d['node_val'] + 3, next_d['addr'] + 3),
                ],
            })
            self.add("valu", (HASH_STAGES[3][2], curr['val'], curr['tmp1'], curr['tmp2']))

            # Hash stage 4 - T6 OPTIMIZATION: Use FMA!
            # val = val * 9 + C4 + gather for next
            self.instrs.append({
                "valu": [
                    ("multiply_add", curr['val'], curr['val'], v_fma_multipliers[4], v_hash_consts[4]),
                ],
                "load": [
                    ("load", next_d['node_val'] + 4, next_d['addr'] + 4),
                    ("load", next_d['node_val'] + 5, next_d['addr'] + 5),
                ],
            })

            # Hash stage 5 - Standard (XOR-based, cannot FMA) + last gather
            self.instrs.append({
                "valu": [
                    (HASH_STAGES[5][0], curr['tmp1'], curr['val'], v_hash_consts[5]),
                    (HASH_STAGES[5][3], curr['tmp2'], curr['val'], v_hash_shifts[5]),
                ],
                "load": [
                    ("load", next_d['node_val'] + 6, next_d['addr'] + 6),
                    ("load", next_d['node_val'] + 7, next_d['addr'] + 7),
                ],
            })
            self.add("valu", (HASH_STAGES[5][2], curr['val'], curr['tmp1'], curr['tmp2']))

            # Branch for current - SIMPLIFIED
            # Key insight: offset = is_even ? 1 : 2 = 1 + (val & 1)
            # So idx_next = 2*idx + 1 + (val & 1), eliminating the ==0 check and vselect
            self.instrs.append({
                "valu": [
                    ("&", curr['tmp1'], curr['val'], v_one),  # tmp1 = val & 1
                    ("*", curr['idx'], curr['idx'], v_two),    # idx = 2 * idx
                ],
                "alu": [
                    ("+", addr_tmp[0], self.scratch["inp_indices_p"], offset_regs[d]),
                    ("+", addr_tmp[1], self.scratch["inp_values_p"], offset_regs[d]),
                ],
            })
            # idx = 2*idx + 1 + (val & 1) in two steps
            self.add("valu", ("+", curr['idx'], curr['idx'], v_one))       # idx = 2*idx + 1
            self.add("valu", ("+", curr['idx'], curr['idx'], curr['tmp1']))  # idx += (val&1)
            # Bounds check and wrap
            self.add("valu", ("<", curr['tmp1'], curr['idx'], v_n_nodes))
            self.add("flow", ("vselect", curr['idx'], curr['tmp1'], curr['idx'], v_zero))

            # Store - for desk 2, pack with desk 3's XOR
            if d == 2:
                d3 = desks[3]
                self.instrs.append({
                    "store": [
                        ("vstore", addr_tmp[0], curr['idx']),
                        ("vstore", addr_tmp[1], curr['val']),
                    ],
                    "valu": [("^", d3['val'], d3['val'], d3['node_val'])],
                })
            else:
                self.instrs.append({"store": [
                    ("vstore", addr_tmp[0], curr['idx']),
                    ("vstore", addr_tmp[1], curr['val']),
                ]})

        # Process desk 3 - XOR already done (packed with desk 2 store)
        d3 = desks[3]

        # Advance batch_offset for next iteration WHILE hashing desk 3
        # Hash stage 0 - T6 OPTIMIZATION: Use FMA! + update batch_offset
        self.instrs.append({
            "valu": [
                ("multiply_add", d3['val'], d3['val'], v_fma_multipliers[0], v_hash_consts[0]),
            ],
            "alu": [
                ("+", batch_offset, batch_offset, thirtytwo_const),
            ],
        })

        # Hash stage 1 - Standard + check batch_offset wrap
        self.instrs.append({
            "valu": [
                (HASH_STAGES[1][0], d3['tmp1'], d3['val'], v_hash_consts[1]),
                (HASH_STAGES[1][3], d3['tmp2'], d3['val'], v_hash_shifts[1]),
            ],
            "alu": [
                ("<", tmp_scalar, batch_offset, batch_size_const),
            ],
        })
        self.add("valu", (HASH_STAGES[1][2], d3['val'], d3['tmp1'], d3['tmp2']))

        # Allocate scratch for speculative next-iteration computation
        spec_batch_offset = self.alloc_scratch("spec_batch_offset")

        # Hash stage 2 - T6 OPTIMIZATION: Use FMA! + select speculative batch_offset + desk 3 store addr
        self.instrs.append({
            "valu": [
                ("multiply_add", d3['val'], d3['val'], v_fma_multipliers[2], v_hash_consts[2]),
            ],
            "flow": [
                ("select", spec_batch_offset, tmp_scalar, batch_offset, zero_const),
            ],
            "alu": [
                ("+", addr_tmp[0], self.scratch["inp_indices_p"], offset_regs[3]),
                ("+", addr_tmp[1], self.scratch["inp_values_p"], offset_regs[3]),
            ],
        })

        # Compute speculative desk 0 addresses (moved from after hash stage 2 final)
        self.instrs.append({
            "alu": [
                ("+", addr_d0_idx, self.scratch["inp_indices_p"], spec_batch_offset),
                ("+", addr_d0_val, self.scratch["inp_values_p"], spec_batch_offset),
            ],
        })

        # Hash stage 3 - Standard + speculative vload for next iter desk 0
        self.instrs.append({
            "valu": [
                (HASH_STAGES[3][0], d3['tmp1'], d3['val'], v_hash_consts[3]),
                (HASH_STAGES[3][3], d3['tmp2'], d3['val'], v_hash_shifts[3]),
            ],
            "load": [
                ("vload", d0['idx'], addr_d0_idx),
                ("vload", d0['val'], addr_d0_val),
            ],
        })

        # Hash stage 3 final + vbroadcast for next iter gather
        self.instrs.append({
            "valu": [
                (HASH_STAGES[3][2], d3['val'], d3['tmp1'], d3['tmp2']),
                ("vbroadcast", d0['addr'], self.scratch["forest_values_p"]),
            ],
        })

        # Hash stage 4 - T6 OPTIMIZATION: Use FMA! + compute gather addresses for next iter
        self.instrs.append({
            "valu": [
                ("multiply_add", d3['val'], d3['val'], v_fma_multipliers[4], v_hash_consts[4]),
                ("+", d0['addr'], d0['addr'], d0['idx']),
            ],
        })

        # First gather loads for next iter
        self.instrs.append({
            "load": [
                ("load", d0['node_val'] + 0, d0['addr'] + 0),
                ("load", d0['node_val'] + 1, d0['addr'] + 1),
            ],
        })

        # Hash stage 5 prep + more gather loads
        self.instrs.append({
            "valu": [
                (HASH_STAGES[5][0], d3['tmp1'], d3['val'], v_hash_consts[5]),
                (HASH_STAGES[5][3], d3['tmp2'], d3['val'], v_hash_shifts[5]),
            ],
            "load": [
                ("load", d0['node_val'] + 2, d0['addr'] + 2),
                ("load", d0['node_val'] + 3, d0['addr'] + 3),
            ],
        })

        # Hash stage 5 final + more gather loads
        self.instrs.append({
            "valu": [(HASH_STAGES[5][2], d3['val'], d3['tmp1'], d3['tmp2'])],
            "load": [
                ("load", d0['node_val'] + 4, d0['addr'] + 4),
                ("load", d0['node_val'] + 5, d0['addr'] + 5),
            ],
        })

        # Branch for desk 3 + final gather loads for next iter + wrap batch_offset
        self.instrs.append({
            "valu": [
                ("&", d3['tmp1'], d3['val'], v_one),  # tmp1 = val & 1
                ("*", d3['idx'], d3['idx'], v_two),    # idx = 2 * idx
            ],
            "flow": [
                ("select", batch_offset, tmp_scalar, batch_offset, zero_const),
            ],
            "load": [
                ("load", d0['node_val'] + 6, d0['addr'] + 6),
                ("load", d0['node_val'] + 7, d0['addr'] + 7),
            ],
        })
        # idx = 2*idx + 1 + pack with next iter offset[0]
        # (addr_d0 already computed speculatively using spec_batch_offset)
        self.instrs.append({
            "valu": [("+", d3['idx'], d3['idx'], v_one)],
            "alu": [("+", offset_regs[0], batch_offset, zero_const)],
        })
        # idx += (val&1) + pack with next iter offset[1]
        self.instrs.append({
            "valu": [("+", d3['idx'], d3['idx'], d3['tmp1'])],
            "alu": [("+", offset_regs[1], batch_offset, eight_const)],
        })
        # bounds check + pack with next iter offset[2,3]
        self.instrs.append({
            "valu": [("<", d3['tmp1'], d3['idx'], v_n_nodes)],
            "alu": [
                ("+", offset_regs[2], batch_offset, sixteen_const),
                ("+", offset_regs[3], batch_offset, twentyfour_const),
            ],
        })

        # Pack vselect + iter++ (addr compute already done in hash stage 2)
        self.instrs.append({
            "flow": [("vselect", d3['idx'], d3['tmp1'], d3['idx'], v_zero)],
            "alu": [
                ("+", iter_counter, iter_counter, one_const),
            ],
        })

        # Store + compute comparison (store reads idx which vselect wrote in prev cycle - OK)
        self.instrs.append({
            "store": [
                ("vstore", addr_tmp[0], d3['idx']),
                ("vstore", addr_tmp[1], d3['val']),
            ],
            "alu": [
                ("<", tmp_scalar, iter_counter, total_const),
            ],
        })

        # Loop jump
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
    check: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Run correctness check")
    parser.add_argument("--trace", action="store_true", help="Generate trace")
    args = parser.parse_args()

    if args.check:
        cycles = do_kernel_test(10, 16, 256, check=True)
        print(f"Correctness check PASSED! Cycles: {cycles}")
    else:
        cycles = do_kernel_test(10, 16, 256, trace=args.trace)
