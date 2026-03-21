"""
# Experiment H10: Flow Unit vselect Bypass

This experiment attempts to bypass the flow unit bottleneck for vselect operations
by replacing them with VALU multiplication operations.

Background:
- The flow unit only allows 1 operation per cycle
- vselect takes 4 cycles (one per desk) because it uses the flow unit
- VALU can process up to 6 operations per cycle

The Insight:
- vselect(dest, cond, idx, v_zero) selects idx if cond != 0, else v_zero
- The condition comes from: cond = (idx < n_nodes), producing 0 or 1
- This can be replaced with: dest = idx * cond
  - When cond = 1: idx * 1 = idx (correct)
  - When cond = 0: idx * 0 = 0 (correct, same as v_zero)

Expected gains:
- Before: 4 vselect operations = 4 cycles (flow limited to 1/cycle)
- After: 4 multiply operations = 1 cycle (VALU can do 4+ per cycle)
- Savings: 3 cycles per iteration x (128 iterations) = ~384 cycles

Baseline: T6+T5 achieves 7,995 cycles
Target: ~7,600 cycles
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


class KernelBuilderH10:
    """
    H10: vselect bypass using VALU multiplication.

    Key optimization: Replace flow-unit vselect with VALU multiply.
    vselect(dest, cond, idx, zero) -> dest = idx * cond

    This works because:
    - cond from (<) comparison is 0 or 1
    - idx * 1 = idx (when idx < n_nodes)
    - idx * 0 = 0 (when idx >= n_nodes)
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
        H10: vselect bypass - replace flow vselect with VALU multiply.
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

        # Use 4 desks
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

        # PHASE 1: LOAD
        self.instrs.append({
            "alu": [
                ("+", offset_regs[0], batch_offset, zero_const),
                ("+", offset_regs[1], batch_offset, eight_const),
                ("+", offset_regs[2], batch_offset, sixteen_const),
                ("+", offset_regs[3], batch_offset, twentyfour_const),
            ],
        })

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

        # Compute gather addresses
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

        # Perform all gathers
        for d in range(NUM_DESKS):
            for lane in range(0, VLEN, 2):
                self.instrs.append({
                    "load": [
                        ("load", desks[d]['node_val'] + lane, desks[d]['addr'] + lane),
                        ("load", desks[d]['node_val'] + lane + 1, desks[d]['addr'] + lane + 1),
                    ],
                })

        # PHASE 2: HASH
        # XOR all desks
        self.instrs.append({
            "valu": [
                ("^", desks[0]['val'], desks[0]['val'], desks[0]['node_val']),
                ("^", desks[1]['val'], desks[1]['val'], desks[1]['node_val']),
                ("^", desks[2]['val'], desks[2]['val'], desks[2]['node_val']),
                ("^", desks[3]['val'], desks[3]['val'], desks[3]['node_val']),
            ],
        })

        # Hash stages
        for hi in range(6):
            if hi in FMA_MULTIPLIERS:
                self.instrs.append({
                    "valu": [
                        ("multiply_add", desks[0]['val'], desks[0]['val'], v_fma_mult[hi], v_hash_consts[hi]),
                        ("multiply_add", desks[1]['val'], desks[1]['val'], v_fma_mult[hi], v_hash_consts[hi]),
                        ("multiply_add", desks[2]['val'], desks[2]['val'], v_fma_mult[hi], v_hash_consts[hi]),
                        ("multiply_add", desks[3]['val'], desks[3]['val'], v_fma_mult[hi], v_hash_consts[hi]),
                    ],
                })
            else:
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
                self.instrs.append({
                    "valu": [
                        (HASH_STAGES[hi][0], desks[3]['tmp1'], desks[3]['val'], v_hash_consts[hi]),
                        (HASH_STAGES[hi][3], desks[3]['tmp2'], desks[3]['val'], v_hash_shifts[hi]),
                        (HASH_STAGES[hi][2], desks[0]['val'], desks[0]['tmp1'], desks[0]['tmp2']),
                        (HASH_STAGES[hi][2], desks[1]['val'], desks[1]['tmp1'], desks[1]['tmp2']),
                        (HASH_STAGES[hi][2], desks[2]['val'], desks[2]['tmp1'], desks[2]['tmp2']),
                    ],
                })
                self.add("valu", (HASH_STAGES[hi][2], desks[3]['val'], desks[3]['tmp1'], desks[3]['tmp2']))

        # Branch computation
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

        self.instrs.append({
            "valu": [
                ("+", desks[0]['idx'], desks[0]['idx'], v_one),
                ("+", desks[1]['idx'], desks[1]['idx'], v_one),
                ("+", desks[2]['idx'], desks[2]['idx'], v_one),
                ("+", desks[3]['idx'], desks[3]['idx'], v_one),
            ],
        })

        self.instrs.append({
            "valu": [
                ("+", desks[0]['idx'], desks[0]['idx'], desks[0]['tmp1']),
                ("+", desks[1]['idx'], desks[1]['idx'], desks[1]['tmp1']),
                ("+", desks[2]['idx'], desks[2]['idx'], desks[2]['tmp1']),
                ("+", desks[3]['idx'], desks[3]['idx'], desks[3]['tmp1']),
            ],
        })

        # Bounds check - produces 0 or 1
        self.instrs.append({
            "valu": [
                ("<", desks[0]['tmp1'], desks[0]['idx'], v_n_nodes),
                ("<", desks[1]['tmp1'], desks[1]['idx'], v_n_nodes),
                ("<", desks[2]['tmp1'], desks[2]['idx'], v_n_nodes),
                ("<", desks[3]['tmp1'], desks[3]['idx'], v_n_nodes),
            ],
        })

        # H10 OPTIMIZATION: Replace vselect with VALU multiply
        # vselect(dest, cond, idx, v_zero) -> dest = idx * cond
        # This works because cond is 0 or 1:
        # - cond=1 (idx < n_nodes): idx * 1 = idx
        # - cond=0 (idx >= n_nodes): idx * 0 = 0
        #
        # Before: 4 flow operations = 4 cycles
        # After: 4 VALU operations = 1 cycle (can pack all 4)
        self.instrs.append({
            "valu": [
                ("*", desks[0]['idx'], desks[0]['idx'], desks[0]['tmp1']),
                ("*", desks[1]['idx'], desks[1]['idx'], desks[1]['tmp1']),
                ("*", desks[2]['idx'], desks[2]['idx'], desks[2]['tmp1']),
                ("*", desks[3]['idx'], desks[3]['idx'], desks[3]['tmp1']),
            ],
        })

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

    kb = KernelBuilderH10()
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
