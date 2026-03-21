"""
T1 Experiment: Modulo-Scheduled Multi-Iteration Pipeline

This experiment implements an 8-desk pipeline to process more elements per loop
iteration, reducing loop overhead and enabling more overlap opportunities.

Baseline: 9,793 cycles with 4 desks
Target: ~6,000-7,000 cycles with 8 desks (1.4-1.6x improvement)
"""

from collections import defaultdict
import random
import sys
import unittest

sys.path.insert(0, '/home/hestiasadmin/projects/original_performance_takehome')

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


class EightDeskPipelineBuilder:
    """
    8-desk pipeline that processes 64 elements per loop iteration.

    Structure similar to baseline 4-desk, but doubled:
    - Process desk 0 while gathering desk 1
    - Process desk 1 while gathering desk 2
    - ...
    - Process desk 7 while gathering desk 0 (next iteration)

    This reduces loop overhead by half (64 iterations vs 128) and
    provides the same overlap benefits.
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
        assert self.scratch_ptr <= SCRATCH_SIZE, f"Out of scratch space at {self.scratch_ptr}"
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
        8-desk pipeline following the same pattern as the baseline 4-desk version.
        """

        NUM_DESKS = 8

        # Allocate 8 sets of vector registers
        desks = []
        for d in range(NUM_DESKS):
            desk = {
                'idx': self.alloc_scratch(f"d{d}_idx", VLEN),
                'val': self.alloc_scratch(f"d{d}_val", VLEN),
                'node_val': self.alloc_scratch(f"d{d}_node", VLEN),
                'tmp1': self.alloc_scratch(f"d{d}_tmp1", VLEN),
                'tmp2': self.alloc_scratch(f"d{d}_tmp2", VLEN),
                'addr': self.alloc_scratch(f"d{d}_addr", VLEN),
            }
            desks.append(desk)

        # Scalar temps
        tmp_scalar = self.alloc_scratch("tmp_scalar")
        addr_tmp = [self.alloc_scratch(f"addr_tmp_{i}") for i in range(4)]

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

        # Scalar constants
        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)
        eight_const = self.scratch_const(VLEN)

        # Vector constants
        v_zero = self.alloc_scratch("v_zero", VLEN)
        v_one = self.alloc_scratch("v_one", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)

        self.add("valu", ("vbroadcast", v_zero, zero_const))
        self.add("valu", ("vbroadcast", v_one, one_const))
        self.add("valu", ("vbroadcast", v_two, two_const))
        self.add("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]))

        # Hash constants
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

        # Offset tracking - separate offset for each desk
        offset_regs = [self.alloc_scratch(f"offset_{d}") for d in range(NUM_DESKS)]
        batch_offset = self.alloc_scratch("batch_offset")
        iter_counter = self.alloc_scratch("iter_counter")

        # Constants for offset computation
        sixteen_const = self.scratch_const(2 * VLEN)
        twentyfour_const = self.scratch_const(3 * VLEN)
        thirtytwo_const = self.scratch_const(4 * VLEN)
        forty_const = self.scratch_const(5 * VLEN)
        fortyeight_const = self.scratch_const(6 * VLEN)
        fiftysix_const = self.scratch_const(7 * VLEN)
        sixtyfour_const = self.scratch_const(8 * VLEN)
        batch_size_const = self.scratch_const(batch_size)

        # Total iterations: 512 groups / 8 = 64
        total_iterations = ((batch_size // VLEN) * rounds) // NUM_DESKS
        total_const = self.scratch_const(total_iterations)

        self.add("load", ("const", batch_offset, 0))
        self.add("load", ("const", iter_counter, 0))

        # Compute initial offsets
        self.instrs.append({"alu": [
            ("+", offset_regs[0], batch_offset, zero_const),
            ("+", offset_regs[1], batch_offset, eight_const),
            ("+", offset_regs[2], batch_offset, sixteen_const),
            ("+", offset_regs[3], batch_offset, twentyfour_const),
            ("+", offset_regs[4], batch_offset, thirtytwo_const),
            ("+", offset_regs[5], batch_offset, forty_const),
        ]})
        self.instrs.append({"alu": [
            ("+", offset_regs[6], batch_offset, fortyeight_const),
            ("+", offset_regs[7], batch_offset, fiftysix_const),
        ]})

        # Dedicated registers for desk 0's addresses (for speculative load)
        addr_d0_idx = self.alloc_scratch("addr_d0_idx")
        addr_d0_val = self.alloc_scratch("addr_d0_val")

        # Compute desk 0's addresses for first iteration
        self.instrs.append({"alu": [
            ("+", addr_d0_idx, self.scratch["inp_indices_p"], batch_offset),
            ("+", addr_d0_val, self.scratch["inp_values_p"], batch_offset),
        ]})

        # PROLOGUE: Load desk 0 + broadcast for gather
        d0 = desks[0]
        self.instrs.append({
            "load": [
                ("vload", d0['idx'], addr_d0_idx),
                ("vload", d0['val'], addr_d0_val),
            ],
            "valu": [
                ("vbroadcast", d0['addr'], self.scratch["forest_values_p"]),
            ],
        })

        # Compute gather addresses for desk 0
        self.add("valu", ("+", d0['addr'], d0['addr'], d0['idx']))

        # Gather for desk 0
        for lane in range(0, VLEN, 2):
            self.instrs.append({"load": [
                ("load", d0['node_val'] + lane, d0['addr'] + lane),
                ("load", d0['node_val'] + lane + 1, d0['addr'] + lane + 1),
            ]})

        # MAIN LOOP
        main_loop_start = len(self.instrs)

        # Process desks 0-6 with overlap to next desk
        for d in range(NUM_DESKS - 1):
            curr = desks[d]
            next_d = desks[d + 1]

            # XOR for current + compute load addresses for next
            self.instrs.append({
                "valu": [("^", curr['val'], curr['val'], curr['node_val'])],
                "alu": [
                    ("+", addr_tmp[0], self.scratch["inp_indices_p"], offset_regs[d + 1]),
                    ("+", addr_tmp[1], self.scratch["inp_values_p"], offset_regs[d + 1]),
                ],
            })

            # Hash stage 0 + vload for next desk
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

            # Hash stage 1 + broadcast gather addr for next desk
            self.instrs.append({"valu": [
                (HASH_STAGES[1][0], curr['tmp1'], curr['val'], v_hash_consts[1]),
                (HASH_STAGES[1][3], curr['tmp2'], curr['val'], v_hash_shifts[1]),
                ("vbroadcast", next_d['addr'], self.scratch["forest_values_p"]),
            ]})
            self.instrs.append({"valu": [
                (HASH_STAGES[1][2], curr['val'], curr['tmp1'], curr['tmp2']),
                ("+", next_d['addr'], next_d['addr'], next_d['idx']),
            ]})

            # Hash stages 2-5 + gather for next desk
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

            # Branch for current desk
            self.instrs.append({
                "valu": [
                    ("&", curr['tmp1'], curr['val'], v_one),
                    ("*", curr['idx'], curr['idx'], v_two),
                ],
                "alu": [
                    ("+", addr_tmp[2], self.scratch["inp_indices_p"], offset_regs[d]),
                    ("+", addr_tmp[3], self.scratch["inp_values_p"], offset_regs[d]),
                ],
            })
            self.add("valu", ("+", curr['idx'], curr['idx'], v_one))
            self.add("valu", ("+", curr['idx'], curr['idx'], curr['tmp1']))
            self.add("valu", ("<", curr['tmp1'], curr['idx'], v_n_nodes))
            self.add("flow", ("vselect", curr['idx'], curr['tmp1'], curr['idx'], v_zero))

            # Store for current desk
            self.instrs.append({"store": [
                ("vstore", addr_tmp[2], curr['idx']),
                ("vstore", addr_tmp[3], curr['val']),
            ]})

        # Process desk 7 - similar to baseline's desk 3 handling with speculative load
        d7 = desks[7]

        # XOR
        self.add("valu", ("^", d7['val'], d7['val'], d7['node_val']))

        # Hash stage 0 + update batch_offset
        self.instrs.append({
            "valu": [
                (HASH_STAGES[0][0], d7['tmp1'], d7['val'], v_hash_consts[0]),
                (HASH_STAGES[0][3], d7['tmp2'], d7['val'], v_hash_shifts[0]),
            ],
            "alu": [
                ("+", batch_offset, batch_offset, sixtyfour_const),
            ],
        })
        self.add("valu", (HASH_STAGES[0][2], d7['val'], d7['tmp1'], d7['tmp2']))

        # Hash stage 1 + check batch_offset wrap
        self.instrs.append({
            "valu": [
                (HASH_STAGES[1][0], d7['tmp1'], d7['val'], v_hash_consts[1]),
                (HASH_STAGES[1][3], d7['tmp2'], d7['val'], v_hash_shifts[1]),
            ],
            "alu": [
                ("<", tmp_scalar, batch_offset, batch_size_const),
            ],
        })
        self.add("valu", (HASH_STAGES[1][2], d7['val'], d7['tmp1'], d7['tmp2']))

        # Allocate register for speculative batch offset
        spec_batch_offset = self.alloc_scratch("spec_batch_offset")

        # Hash stage 2 + select speculative batch_offset + compute store addr for desk 7
        self.instrs.append({
            "valu": [
                (HASH_STAGES[2][0], d7['tmp1'], d7['val'], v_hash_consts[2]),
                (HASH_STAGES[2][3], d7['tmp2'], d7['val'], v_hash_shifts[2]),
            ],
            "flow": [
                ("select", spec_batch_offset, tmp_scalar, batch_offset, zero_const),
            ],
            "alu": [
                ("+", addr_tmp[2], self.scratch["inp_indices_p"], offset_regs[7]),
                ("+", addr_tmp[3], self.scratch["inp_values_p"], offset_regs[7]),
            ],
        })

        # Hash stage 2 final + compute speculative desk 0 addresses
        self.instrs.append({
            "valu": [(HASH_STAGES[2][2], d7['val'], d7['tmp1'], d7['tmp2'])],
            "alu": [
                ("+", addr_d0_idx, self.scratch["inp_indices_p"], spec_batch_offset),
                ("+", addr_d0_val, self.scratch["inp_values_p"], spec_batch_offset),
            ],
        })

        # Hash stage 3 + speculative vload for next iter desk 0
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

        # Hash stage 3 final + vbroadcast for next iter gather
        self.instrs.append({
            "valu": [
                (HASH_STAGES[3][2], d7['val'], d7['tmp1'], d7['tmp2']),
                ("vbroadcast", d0['addr'], self.scratch["forest_values_p"]),
            ],
        })

        # Hash stage 4 + compute gather addresses for next iter
        self.instrs.append({
            "valu": [
                (HASH_STAGES[4][0], d7['tmp1'], d7['val'], v_hash_consts[4]),
                (HASH_STAGES[4][3], d7['tmp2'], d7['val'], v_hash_shifts[4]),
                ("+", d0['addr'], d0['addr'], d0['idx']),
            ],
        })

        # Hash stage 4 final + first gather loads for next iter
        self.instrs.append({
            "valu": [(HASH_STAGES[4][2], d7['val'], d7['tmp1'], d7['tmp2'])],
            "load": [
                ("load", d0['node_val'] + 0, d0['addr'] + 0),
                ("load", d0['node_val'] + 1, d0['addr'] + 1),
            ],
        })

        # Hash stage 5 + more gather loads
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

        # Hash stage 5 final + more gather loads
        self.instrs.append({
            "valu": [(HASH_STAGES[5][2], d7['val'], d7['tmp1'], d7['tmp2'])],
            "load": [
                ("load", d0['node_val'] + 4, d0['addr'] + 4),
                ("load", d0['node_val'] + 5, d0['addr'] + 5),
            ],
        })

        # Branch for desk 7 + final gather loads + wrap batch_offset
        self.instrs.append({
            "valu": [
                ("&", d7['tmp1'], d7['val'], v_one),
                ("*", d7['idx'], d7['idx'], v_two),
            ],
            "flow": [
                ("select", batch_offset, tmp_scalar, batch_offset, zero_const),
            ],
            "load": [
                ("load", d0['node_val'] + 6, d0['addr'] + 6),
                ("load", d0['node_val'] + 7, d0['addr'] + 7),
            ],
        })

        # idx = 2*idx + 1 + update offset[0]
        self.instrs.append({
            "valu": [("+", d7['idx'], d7['idx'], v_one)],
            "alu": [("+", offset_regs[0], batch_offset, zero_const)],
        })

        # idx += (val&1) + update offset[1]
        self.instrs.append({
            "valu": [("+", d7['idx'], d7['idx'], d7['tmp1'])],
            "alu": [("+", offset_regs[1], batch_offset, eight_const)],
        })

        # Bounds check + update offsets 2-5
        self.instrs.append({
            "valu": [("<", d7['tmp1'], d7['idx'], v_n_nodes)],
            "alu": [
                ("+", offset_regs[2], batch_offset, sixteen_const),
                ("+", offset_regs[3], batch_offset, twentyfour_const),
                ("+", offset_regs[4], batch_offset, thirtytwo_const),
                ("+", offset_regs[5], batch_offset, forty_const),
            ],
        })

        # vselect + update offsets 6-7 + iter++
        self.instrs.append({
            "flow": [("vselect", d7['idx'], d7['tmp1'], d7['idx'], v_zero)],
            "alu": [
                ("+", offset_regs[6], batch_offset, fortyeight_const),
                ("+", offset_regs[7], batch_offset, fiftysix_const),
                ("+", iter_counter, iter_counter, one_const),
            ],
        })

        # Store desk 7 + loop comparison
        self.instrs.append({
            "store": [
                ("vstore", addr_tmp[2], d7['idx']),
                ("vstore", addr_tmp[3], d7['val']),
            ],
            "alu": [
                ("<", tmp_scalar, iter_counter, total_const),
            ],
        })

        # Loop jump
        self.add("flow", ("cond_jump", tmp_scalar, main_loop_start))

        self.instrs.append({"flow": [("pause",)]})


KernelBuilder = EightDeskPipelineBuilder


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
        machine.enable_debug = True

    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print("Got:     ", machine.mem[inp_values_p : inp_values_p + len(inp.values)][:16])
            print("Expected:", ref_mem[inp_values_p : inp_values_p + len(inp.values)][:16])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)

    def test_kernel_check(self):
        do_kernel_test(10, 16, 256, check=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Run correctness check")
    parser.add_argument("--trace", action="store_true", help="Generate trace")
    args = parser.parse_args()

    if args.check:
        do_kernel_test(10, 16, 256, check=True)
    else:
        do_kernel_test(10, 16, 256, trace=args.trace)
