"""
# Experiment T4: 8-Desk Deep Pipeline - Sequential with Better Overlap

This version keeps the sequential 8-desk structure but optimizes the
cross-iteration overlap more aggressively.

With 8 desks, we have more hash cycles during desk 7's processing to
overlap with desk 0's gather of the NEXT iteration.

Key insight: Desk 7's hash provides 12 cycles of VALU work during which
we can do a LOT of load operations for desk 0 of the next iteration.

Expected: Some improvement over the baseline 9,793 cycles
"""

from collections import defaultdict
import random
import unittest
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
        8-DESK DEEP PIPELINE - Sequential with Better Cross-Iteration Overlap

        Same structure as original 4-desk but with 8 desks.
        Process desks 0-6 with hash/gather overlap on next desk.
        Process desk 7 with hash/gather overlap on desk 0 of NEXT iteration.
        """

        # ============================================
        # REGISTER ALLOCATION
        # ============================================

        tmp_scalar = self.alloc_scratch("tmp_scalar")

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

        # ============================================
        # 8-DESK REGISTER ALLOCATION
        # ============================================

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

        addr_tmp = [self.alloc_scratch(f"addr_tmp_{i}") for i in range(4)]
        batch_offset = self.alloc_scratch("batch_offset")
        iter_counter = self.alloc_scratch("iter_counter")

        eight_const = self.scratch_const(VLEN)
        sixteen_const = self.scratch_const(2*VLEN)
        twentyfour_const = self.scratch_const(3*VLEN)
        thirtytwo_const = self.scratch_const(4*VLEN)
        forty_const = self.scratch_const(5*VLEN)
        fortyeight_const = self.scratch_const(6*VLEN)
        fiftysix_const = self.scratch_const(7*VLEN)
        sixtyfour_const = self.scratch_const(8*VLEN)

        batch_size_const = self.scratch_const(batch_size)
        total_iterations = (batch_size // VLEN) * rounds // NUM_DESKS
        total_const = self.scratch_const(total_iterations)

        offset_regs = [self.alloc_scratch(f"off_{d}") for d in range(NUM_DESKS)]

        addr_d0_idx = self.alloc_scratch("addr_d0_idx")
        addr_d0_val = self.alloc_scratch("addr_d0_val")
        spec_batch_offset = self.alloc_scratch("spec_batch_offset")

        self.add("load", ("const", batch_offset, 0))
        self.add("load", ("const", iter_counter, 0))

        # ============================================
        # PROLOGUE
        # ============================================

        d0 = desks[0]

        self.instrs.append({
            "alu": [
                ("+", offset_regs[0], batch_offset, zero_const),
                ("+", offset_regs[1], batch_offset, eight_const),
                ("+", offset_regs[2], batch_offset, sixteen_const),
                ("+", offset_regs[3], batch_offset, twentyfour_const),
                ("+", offset_regs[4], batch_offset, thirtytwo_const),
                ("+", offset_regs[5], batch_offset, forty_const),
                ("+", offset_regs[6], batch_offset, fortyeight_const),
                ("+", offset_regs[7], batch_offset, fiftysix_const),
                ("+", addr_d0_idx, self.scratch["inp_indices_p"], batch_offset),
                ("+", addr_d0_val, self.scratch["inp_values_p"], batch_offset),
            ],
        })

        self.instrs.append({
            "load": [
                ("vload", d0['idx'], addr_d0_idx),
                ("vload", d0['val'], addr_d0_val),
            ],
            "valu": [
                ("vbroadcast", d0['addr'], self.scratch["forest_values_p"]),
            ],
        })

        self.add("valu", ("+", d0['addr'], d0['addr'], d0['idx']))

        for lane in range(0, VLEN, 2):
            self.instrs.append({"load": [
                ("load", d0['node_val'] + lane, d0['addr'] + lane),
                ("load", d0['node_val'] + lane + 1, d0['addr'] + lane + 1),
            ]})

        # ============================================
        # MAIN LOOP
        # ============================================
        main_loop_start = len(self.instrs)

        # Process desks 0-6 with hash/gather overlap
        for d in range(7):
            curr = desks[d]
            next_d = desks[d + 1]

            # XOR + load addresses for next
            self.instrs.append({
                "valu": [("^", curr['val'], curr['val'], curr['node_val'])],
                "alu": [
                    ("+", addr_tmp[2], self.scratch["inp_indices_p"], offset_regs[d + 1]),
                    ("+", addr_tmp[3], self.scratch["inp_values_p"], offset_regs[d + 1]),
                ],
            })

            # Hash 0 + vload next
            self.instrs.append({
                "valu": [
                    (HASH_STAGES[0][0], curr['tmp1'], curr['val'], v_hash_consts[0]),
                    (HASH_STAGES[0][3], curr['tmp2'], curr['val'], v_hash_shifts[0]),
                ],
                "load": [
                    ("vload", next_d['idx'], addr_tmp[2]),
                    ("vload", next_d['val'], addr_tmp[3]),
                ],
            })
            self.add("valu", (HASH_STAGES[0][2], curr['val'], curr['tmp1'], curr['tmp2']))

            # Hash 1 + broadcast
            self.instrs.append({"valu": [
                (HASH_STAGES[1][0], curr['tmp1'], curr['val'], v_hash_consts[1]),
                (HASH_STAGES[1][3], curr['tmp2'], curr['val'], v_hash_shifts[1]),
                ("vbroadcast", next_d['addr'], self.scratch["forest_values_p"]),
            ]})
            self.instrs.append({"valu": [
                (HASH_STAGES[1][2], curr['val'], curr['tmp1'], curr['tmp2']),
                ("+", next_d['addr'], next_d['addr'], next_d['idx']),
            ]})

            # Hash 2-5 + gather
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

            # Branch + store addr
            self.instrs.append({
                "valu": [
                    ("&", curr['tmp1'], curr['val'], v_one),
                    ("*", curr['idx'], curr['idx'], v_two),
                ],
                "alu": [
                    ("+", addr_tmp[0], self.scratch["inp_indices_p"], offset_regs[d]),
                    ("+", addr_tmp[1], self.scratch["inp_values_p"], offset_regs[d]),
                ],
            })
            self.add("valu", ("+", curr['idx'], curr['idx'], v_one))
            self.add("valu", ("+", curr['idx'], curr['idx'], curr['tmp1']))
            self.add("valu", ("<", curr['tmp1'], curr['idx'], v_n_nodes))
            self.add("flow", ("vselect", curr['idx'], curr['tmp1'], curr['idx'], v_zero))

            # Store + XOR for d6->d7
            if d == 6:
                d7 = desks[7]
                self.instrs.append({
                    "store": [
                        ("vstore", addr_tmp[0], curr['idx']),
                        ("vstore", addr_tmp[1], curr['val']),
                    ],
                    "valu": [("^", d7['val'], d7['val'], d7['node_val'])],
                })
            else:
                self.instrs.append({"store": [
                    ("vstore", addr_tmp[0], curr['idx']),
                    ("vstore", addr_tmp[1], curr['val']),
                ]})

        # ============================================
        # DESK 7 with next-iteration prefetch
        # ============================================
        d7 = desks[7]
        d0 = desks[0]

        # Hash 0 + batch update
        self.instrs.append({
            "valu": [
                (HASH_STAGES[0][0], d7['tmp1'], d7['val'], v_hash_consts[0]),
                (HASH_STAGES[0][3], d7['tmp2'], d7['val'], v_hash_shifts[0]),
            ],
            "alu": [("+", batch_offset, batch_offset, sixtyfour_const)],
        })
        self.add("valu", (HASH_STAGES[0][2], d7['val'], d7['tmp1'], d7['tmp2']))

        # Hash 1 + wrap check
        self.instrs.append({
            "valu": [
                (HASH_STAGES[1][0], d7['tmp1'], d7['val'], v_hash_consts[1]),
                (HASH_STAGES[1][3], d7['tmp2'], d7['val'], v_hash_shifts[1]),
            ],
            "alu": [("<", tmp_scalar, batch_offset, batch_size_const)],
        })
        self.add("valu", (HASH_STAGES[1][2], d7['val'], d7['tmp1'], d7['tmp2']))

        # Hash 2 + spec offset + store addr
        self.instrs.append({
            "valu": [
                (HASH_STAGES[2][0], d7['tmp1'], d7['val'], v_hash_consts[2]),
                (HASH_STAGES[2][3], d7['tmp2'], d7['val'], v_hash_shifts[2]),
            ],
            "flow": [("select", spec_batch_offset, tmp_scalar, batch_offset, zero_const)],
            "alu": [
                ("+", addr_tmp[0], self.scratch["inp_indices_p"], offset_regs[7]),
                ("+", addr_tmp[1], self.scratch["inp_values_p"], offset_regs[7]),
            ],
        })

        # Hash 2 final + d0 addr
        self.instrs.append({
            "valu": [(HASH_STAGES[2][2], d7['val'], d7['tmp1'], d7['tmp2'])],
            "alu": [
                ("+", addr_d0_idx, self.scratch["inp_indices_p"], spec_batch_offset),
                ("+", addr_d0_val, self.scratch["inp_values_p"], spec_batch_offset),
            ],
        })

        # Hash 3 + vload d0
        self.instrs.append({
            "valu": [
                (HASH_STAGES[3][0], d7['tmp1'], d7['val'], v_hash_consts[3]),
                (HASH_STAGES[3][3], d7['tmp2'], d7['val'], v_hash_shifts[3]),
            ],
            "load": [
                ("vload", d0['idx'], addr_d0_idx),
                ("vload", d0['val'], addr_d0_val),
            ],
        })

        # Hash 3 final + broadcast
        self.instrs.append({
            "valu": [
                (HASH_STAGES[3][2], d7['val'], d7['tmp1'], d7['tmp2']),
                ("vbroadcast", d0['addr'], self.scratch["forest_values_p"]),
            ],
        })

        # Hash 4 + gather addr
        self.instrs.append({
            "valu": [
                (HASH_STAGES[4][0], d7['tmp1'], d7['val'], v_hash_consts[4]),
                (HASH_STAGES[4][3], d7['tmp2'], d7['val'], v_hash_shifts[4]),
                ("+", d0['addr'], d0['addr'], d0['idx']),
            ],
        })

        # Hash 4 final + gather 0-1
        self.instrs.append({
            "valu": [(HASH_STAGES[4][2], d7['val'], d7['tmp1'], d7['tmp2'])],
            "load": [
                ("load", d0['node_val'] + 0, d0['addr'] + 0),
                ("load", d0['node_val'] + 1, d0['addr'] + 1),
            ],
        })

        # Hash 5 + gather 2-3
        self.instrs.append({
            "valu": [
                (HASH_STAGES[5][0], d7['tmp1'], d7['val'], v_hash_consts[5]),
                (HASH_STAGES[5][3], d7['tmp2'], d7['val'], v_hash_shifts[5]),
            ],
            "load": [
                ("load", d0['node_val'] + 2, d0['addr'] + 2),
                ("load", d0['node_val'] + 3, d0['addr'] + 3),
            ],
        })

        # Hash 5 final + gather 4-5
        self.instrs.append({
            "valu": [(HASH_STAGES[5][2], d7['val'], d7['tmp1'], d7['tmp2'])],
            "load": [
                ("load", d0['node_val'] + 4, d0['addr'] + 4),
                ("load", d0['node_val'] + 5, d0['addr'] + 5),
            ],
        })

        # Branch + gather 6-7 + wrap
        self.instrs.append({
            "valu": [
                ("&", d7['tmp1'], d7['val'], v_one),
                ("*", d7['idx'], d7['idx'], v_two),
            ],
            "flow": [("select", batch_offset, tmp_scalar, batch_offset, zero_const)],
            "load": [
                ("load", d0['node_val'] + 6, d0['addr'] + 6),
                ("load", d0['node_val'] + 7, d0['addr'] + 7),
            ],
        })

        # idx + 1 + offset 0-3
        self.instrs.append({
            "valu": [("+", d7['idx'], d7['idx'], v_one)],
            "alu": [
                ("+", offset_regs[0], batch_offset, zero_const),
                ("+", offset_regs[1], batch_offset, eight_const),
                ("+", offset_regs[2], batch_offset, sixteen_const),
                ("+", offset_regs[3], batch_offset, twentyfour_const),
            ],
        })

        # idx + tmp1 + offset 4-7
        self.instrs.append({
            "valu": [("+", d7['idx'], d7['idx'], d7['tmp1'])],
            "alu": [
                ("+", offset_regs[4], batch_offset, thirtytwo_const),
                ("+", offset_regs[5], batch_offset, forty_const),
                ("+", offset_regs[6], batch_offset, fortyeight_const),
                ("+", offset_regs[7], batch_offset, fiftysix_const),
            ],
        })

        # bounds + iter++
        self.instrs.append({
            "valu": [("<", d7['tmp1'], d7['idx'], v_n_nodes)],
            "alu": [("+", iter_counter, iter_counter, one_const)],
        })

        # vselect + compare
        self.instrs.append({
            "flow": [("vselect", d7['idx'], d7['tmp1'], d7['idx'], v_zero)],
            "alu": [("<", tmp_scalar, iter_counter, total_const)],
        })

        # store
        self.instrs.append({
            "store": [
                ("vstore", addr_tmp[0], d7['idx']),
                ("vstore", addr_tmp[1], d7['val']),
            ],
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
    check: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
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

    if check:
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
        print("CORRECTNESS CHECK PASSED!")
    else:
        for _ in reference_kernel2(mem, value_trace):
            machine.run()

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    print(f"Scratch usage: {kb.scratch_ptr} / {SCRATCH_SIZE}")
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Run correctness check")
    parser.add_argument("--trace", action="store_true", help="Generate trace file")
    parser.add_argument("--prints", action="store_true", help="Print debug info")
    args = parser.parse_args()

    if args.check or args.trace or args.prints:
        do_kernel_test(10, 16, 256, trace=args.trace, prints=args.prints, check=args.check)
    else:
        do_kernel_test(10, 16, 256)
