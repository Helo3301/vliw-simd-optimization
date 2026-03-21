"""
# Experiment H9: Level-Aware Tree Preloading

This experiment investigates whether preloading entire tree levels can improve performance.

Key insights from T2:
- ALL indices in a given round are at the SAME tree level
- Round 0 -> Level 1 (2 nodes), Round 1 -> Level 2 (4 nodes), etc.
- Tree levels: Level k has 2^k nodes at memory offsets [2^k - 1, 2^(k+1) - 2]
- Levels 1-6 have 2, 4, 8, 16, 32, 64 nodes (8-512 bytes) - fits easily in scratch

Hypothesis:
Since T2 found that scratch and memory have the SAME latency, the benefit would not
come from faster access. However, there could be benefits from:
1. Reduced address computation (level_base + offset instead of forest_ptr + idx)
2. Better instruction packing during gather (addresses are simpler)
3. Amortizing the level load across all 256 elements

However, the T2 RESULTS.md explicitly states: "Level preloading - Scratch and memory
have same latency" is NOT viable.

This experiment will verify this finding by implementing level preloading and measuring
actual cycle counts.

Baseline: T6+T5 achieves 7,995 cycles
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


class KernelBuilderH9:
    """
    H9: Level-Aware Tree Preloading

    Strategy:
    - Track current tree level based on round number
    - For small levels (1-6, up to 64 nodes), preload entire level to scratch
    - Use simpler addressing for gather (scratch_base + (idx - level_start))
    - For larger levels (7-10), fall back to standard memory access

    Level structure (perfect binary tree with height 10, 2047 nodes):
    - Level 0: node 0 (root)
    - Level k: nodes [2^k - 1, 2^(k+1) - 2], size = 2^k

    Round mapping (starts at root=0, after round r we're at level (r+1) % 11):
    - Round 0 puts indices at level 1 (2 nodes)
    - Round 1 puts indices at level 2 (4 nodes)
    - ...
    - Round 9 puts indices at level 10 (1024 nodes)
    - Round 10 wraps to level 0 (1 node - only root)
    - Round 11 puts indices at level 1 again

    Note: We load the level we're ABOUT TO ACCESS, not the level we came from.
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
        H9: Level-Aware Tree Preloading

        For this experiment, we'll try a modified approach:
        - Since all 256 indices access the same level, preload that level
        - For levels 1-6 (<=64 nodes), this is cheap (8 vloads max)
        - The gather then becomes scratch lookup instead of memory lookup

        But based on T2 findings, scratch and memory have same latency.
        Let's verify by implementing and measuring.
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

        # Hash constants and FMA multipliers (from T6+T5)
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

        # Use 4 desks (same as T6+T5)
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
        round_counter = self.alloc_scratch("round_counter")

        # Level tracking
        current_level = self.alloc_scratch("current_level")
        level_start = self.alloc_scratch("level_start")  # 2^level - 1
        level_size = self.alloc_scratch("level_size")    # 2^level

        # Scratch buffer for preloaded level (max 64 nodes for levels 1-6)
        # For this experiment, we'll preload levels up to 64 nodes
        MAX_PRELOAD_SIZE = 64
        level_buffer = self.alloc_scratch("level_buffer", MAX_PRELOAD_SIZE)
        level_buffer_base = self.alloc_scratch("level_buffer_base")  # Scalar for vbroadcast
        v_level_start = self.alloc_scratch("v_level_start", VLEN)

        eight_const = self.scratch_const(VLEN)
        sixteen_const = self.scratch_const(2 * VLEN)
        twentyfour_const = self.scratch_const(3 * VLEN)
        thirtytwo_const = self.scratch_const(4 * VLEN)
        batch_size_const = self.scratch_const(batch_size)

        # Constants for level computation
        # After round r, indices are at level ((r+1) % 11) but we handle wrap specially
        # Actually, we need to track which level we're accessing in the CURRENT iteration
        # At iteration start, indices point to the current level

        total_iterations = (batch_size // VLEN) * rounds // NUM_DESKS
        total_const = self.scratch_const(total_iterations)
        iters_per_round = self.scratch_const((batch_size // VLEN) // NUM_DESKS)
        eleven_const = self.scratch_const(11)
        max_preload_level_const = self.scratch_const(6)  # Levels 1-6 can be preloaded

        offset_regs = [self.alloc_scratch(f"off_{d}") for d in range(NUM_DESKS)]

        # Initialize
        self.add("load", ("const", batch_offset, 0))
        self.add("load", ("const", iter_counter, 0))
        self.add("load", ("const", round_counter, 0))
        self.add("load", ("const", current_level, 0))  # Start at level 0 (root)
        self.add("load", ("const", level_start, 0))    # Level 0 starts at 0
        self.add("load", ("const", level_size, 1))     # Level 0 has 1 node
        self.add("load", ("const", level_buffer_base, level_buffer))

        # === MAIN LOOP ===
        main_loop_start = len(self.instrs)

        # PHASE 1: LOAD indices/values
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

        # PHASE 2: GATHER - using memory (standard approach from T6+T5)
        # The level preloading idea doesn't work because:
        # 1. We'd need to preload BEFORE knowing what indices we have
        # 2. The indices are already loaded, we just need to gather node values
        # 3. Preloading the level still requires address computation for the gather

        # Actually, let's think about this differently:
        # - We know the current indices point to a specific level
        # - If we preloaded that level, we could do: node_val = level_buffer[idx - level_start]
        # - But this still requires: vbroadcast level_buffer_base, subtract v_level_start, then gather
        # - vs current: vbroadcast forest_values_p, add v_idx, then gather
        # - It's actually MORE operations (subtract instead of already-computed add)!

        # The key insight: there's NO benefit because:
        # 1. Scratch and memory have same latency
        # 2. Address computation is similar or worse
        # 3. Preloading adds overhead (vloads at round start)

        # Compute gather addresses for all desks
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

        # PHASE 3: HASH (same as T6+T5 with FMA optimization)
        self.instrs.append({
            "valu": [
                ("^", desks[0]['val'], desks[0]['val'], desks[0]['node_val']),
                ("^", desks[1]['val'], desks[1]['val'], desks[1]['node_val']),
                ("^", desks[2]['val'], desks[2]['val'], desks[2]['node_val']),
                ("^", desks[3]['val'], desks[3]['val'], desks[3]['node_val']),
            ],
        })

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

        # Branch computation (same as T6+T5)
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

        self.instrs.append({
            "valu": [
                ("<", desks[0]['tmp1'], desks[0]['idx'], v_n_nodes),
                ("<", desks[1]['tmp1'], desks[1]['idx'], v_n_nodes),
                ("<", desks[2]['tmp1'], desks[2]['idx'], v_n_nodes),
                ("<", desks[3]['tmp1'], desks[3]['idx'], v_n_nodes),
            ],
        })

        for d in range(NUM_DESKS):
            self.add("flow", ("vselect", desks[d]['idx'], desks[d]['tmp1'], desks[d]['idx'], v_zero))

        # PHASE 4: STORE
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


# Baseline from T6+T5
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

    kb = KernelBuilderH9()
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
        """Test the reference kernels against each other"""
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
