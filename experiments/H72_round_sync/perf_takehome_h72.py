"""
# Experiment H72: Round-Synchronous Processing

**GOAL:** Process ALL 256 elements through each round before moving to next round.

**KEY INSIGHT FROM H63:**
- Rounds 0-7: Only 255 unique tree nodes needed (1+2+4+8+16+32+64+128)
- Can preload all tree nodes 0-127 and use vselect
- Rounds 8-15: Need gather but can still pipeline within a round

**ARCHITECTURE:**
- Round 0: tree[0] broadcast to all 256 elements
- Round 1: vselect between tree[1], tree[2]
- Round 2: vselect among tree[3-6]
- Round 3: vselect among tree[7-14]
- Rounds 4+: gather-based (indices too diverse)

**WHY ROUND-SYNC IS BETTER:**
- Desk-pipelining overlaps gather(round N+1) with hash(round N)
- Round-sync overlaps gather(element K+1) with hash(element K) within SAME round
- Better for early rounds where tree values are shared across elements
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


class KernelBuilderH72:
    """
    H72: Round-synchronous processing.

    Process all 256 elements through round 0, then all through round 1, etc.
    This allows sharing tree values across elements within a round.
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

    def build_kernel(self, forest_height: int, n_nodes: int, batch_size: int, rounds: int):
        """
        Round-synchronous kernel.
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

        # Basic constants
        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)
        vlen_const = self.scratch_const(VLEN)

        v_zero = self.alloc_scratch("v_zero", VLEN)
        v_one = self.alloc_scratch("v_one", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)

        self.add("valu", ("vbroadcast", v_zero, zero_const))
        self.add("valu", ("vbroadcast", v_one, one_const))
        self.add("valu", ("vbroadcast", v_two, two_const))
        self.add("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]))

        # Hash constants - FMA optimization
        FMA_MULTIPLIERS = {0: 4097, 2: 33, 4: 9}

        v_hash_consts = []
        v_hash_shifts = []
        v_fma_mult = {}

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            const_scalar = self.scratch_const(val1)
            v_const = self.alloc_scratch(f"v_hash_const_{hi}", VLEN)
            self.add("valu", ("vbroadcast", v_const, const_scalar))
            v_hash_consts.append(v_const)

            if hi in FMA_MULTIPLIERS:
                mult_scalar = self.scratch_const(FMA_MULTIPLIERS[hi])
                v_mult = self.alloc_scratch(f"v_fma_mult_{hi}", VLEN)
                self.add("valu", ("vbroadcast", v_mult, mult_scalar))
                v_fma_mult[hi] = v_mult
                v_hash_shifts.append(None)  # Not used for FMA stages
            else:
                shift_scalar = self.scratch_const(val3)
                v_shift = self.alloc_scratch(f"v_hash_shift_{hi}", VLEN)
                self.add("valu", ("vbroadcast", v_shift, shift_scalar))
                v_hash_shifts.append(v_shift)

        # Preload tree nodes 0-14 for rounds 0-3
        # Level 0: node 0
        # Level 1: nodes 1-2
        # Level 2: nodes 3-6
        # Level 3: nodes 7-14
        NUM_PRELOADED = 15
        v_tree = []
        tree_addr_tmp = self.alloc_scratch("tree_addr_tmp")

        for i in range(NUM_PRELOADED):
            v_tree_node = self.alloc_scratch(f"v_tree_{i}", VLEN)
            v_tree.append(v_tree_node)

        # Load and broadcast tree nodes
        for i in range(NUM_PRELOADED):
            self.instrs.append({
                "alu": [
                    ("+", tree_addr_tmp, self.scratch["forest_values_p"], self.scratch_const(i))
                ],
            })
            self.add("load", ("load", tmp_scalar, tree_addr_tmp))
            self.add("valu", ("vbroadcast", v_tree[i], tmp_scalar))

        # Working vectors - 8 at a time for deep pipelining within a round
        NUM_WORKING = 8  # Process 8 vectors (64 elements) at a time
        NUM_BATCHES = batch_size // (NUM_WORKING * VLEN)  # 256 / 64 = 4 batches

        work_idx = []
        work_val = []
        work_node = []
        work_addr = []
        work_tmp1 = []
        work_tmp2 = []

        for w in range(NUM_WORKING):
            work_idx.append(self.alloc_scratch(f"w_idx_{w}", VLEN))
            work_val.append(self.alloc_scratch(f"w_val_{w}", VLEN))
            work_node.append(self.alloc_scratch(f"w_node_{w}", VLEN))
            work_addr.append(self.alloc_scratch(f"w_addr_{w}", VLEN))
            work_tmp1.append(self.alloc_scratch(f"w_tmp1_{w}", VLEN))
            work_tmp2.append(self.alloc_scratch(f"w_tmp2_{w}", VLEN))

        # Batch offset tracking
        batch_offset = self.alloc_scratch("batch_offset")
        batch_counter = self.alloc_scratch("batch_counter")
        round_counter = self.alloc_scratch("round_counter")

        batch_stride = self.scratch_const(NUM_WORKING * VLEN)
        num_batches_const = self.scratch_const(NUM_BATCHES)
        num_rounds_const = self.scratch_const(rounds)

        # Address temporaries for loading/storing
        load_addr = [self.alloc_scratch(f"load_addr_{i}") for i in range(NUM_WORKING * 2)]

        print(f"Scratch usage before main loop: {self.scratch_ptr} / {SCRATCH_SIZE}")

        # Pause after initialization to sync with reference_kernel2's first yield
        self.add("flow", ("pause",))

        # ============ MAIN KERNEL ============
        # Process all elements through each round

        # Initialize round counter
        self.add("load", ("const", round_counter, 0))

        round_loop_start = len(self.instrs)

        # Initialize batch counter for this round
        self.add("load", ("const", batch_offset, 0))
        self.add("load", ("const", batch_counter, 0))

        batch_loop_start = len(self.instrs)

        # PHASE 1: Load all working vectors for this batch
        # Compute addresses
        for w in range(NUM_WORKING):
            offset_const = self.scratch_const(w * VLEN)
            self.instrs.append({
                "alu": [
                    ("+", load_addr[w*2], batch_offset, offset_const),
                    ("+", load_addr[w*2+1], batch_offset, offset_const),
                ],
            })
            self.instrs.append({
                "alu": [
                    ("+", load_addr[w*2], self.scratch["inp_indices_p"], load_addr[w*2]),
                    ("+", load_addr[w*2+1], self.scratch["inp_values_p"], load_addr[w*2+1]),
                ],
            })

        # Load all vectors
        for w in range(0, NUM_WORKING, 2):
            self.instrs.append({
                "load": [
                    ("vload", work_idx[w], load_addr[w*2]),
                    ("vload", work_val[w], load_addr[w*2+1]),
                ],
            })
            if w + 1 < NUM_WORKING:
                self.instrs.append({
                    "load": [
                        ("vload", work_idx[w+1], load_addr[(w+1)*2]),
                        ("vload", work_val[w+1], load_addr[(w+1)*2+1]),
                    ],
                })

        # PHASE 2: Process this round for all working vectors
        # The round-specific behavior is controlled by round_counter
        # But we can't do conditional code easily, so we use a gather-based approach
        # with the optimization that indices are bounded in early rounds

        def emit_xor_hash_branch(w, skip_bounds):
            """Emit XOR, hash, and branch update for one working vector."""
            idx = work_idx[w]
            val = work_val[w]
            node = work_node[w]
            addr = work_addr[w]
            tmp1 = work_tmp1[w]
            tmp2 = work_tmp2[w]

            # XOR with node value (VALU does all 8 lanes at once)
            self.instrs.append({
                "valu": [("^", val, val, node)],
            })

            # Hash stages with FMA optimization
            for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                if hi in v_fma_mult:
                    self.instrs.append({
                        "valu": [("multiply_add", val, val, v_fma_mult[hi], v_hash_consts[hi])],
                    })
                else:
                    # val = (val + c) ^ (val >> k)
                    self.instrs.append({
                        "valu": [(op1, tmp1, val, v_hash_consts[hi])],
                    })
                    self.instrs.append({
                        "valu": [(op3, tmp2, val, v_hash_shifts[hi])],
                    })
                    self.instrs.append({
                        "valu": [(op2, val, tmp1, tmp2)],
                    })

            # Branch: idx = idx * 2 + 1 + (val & 1)
            self.instrs.append({
                "valu": [("&", tmp1, val, v_one)],
            })
            self.instrs.append({
                "valu": [("+", tmp2, tmp1, v_one)],
            })
            self.instrs.append({
                "valu": [("multiply_add", idx, idx, v_two, tmp2)],
            })

            # Bounds check: idx = 0 if idx >= n_nodes
            # Use: tmp1 = idx < n_nodes (1 if valid, 0 if overflow)
            # Then: idx = idx * tmp1 (sets to 0 if overflow)
            if not skip_bounds:
                self.instrs.append({
                    "valu": [("<", tmp1, idx, v_n_nodes)],
                })
                self.instrs.append({
                    "valu": [("*", idx, idx, tmp1)],
                })

        # For simplicity, use a single gather-based round for all rounds
        # The vselect optimization for rounds 0-3 requires conditional code
        # which is complex without proper control flow

        # Gather tree values for all working vectors
        for w in range(NUM_WORKING):
            idx = work_idx[w]
            node = work_node[w]
            addr = work_addr[w]

            # Compute gather addresses: forest_values_p + idx
            for lane in range(VLEN):
                self.instrs.append({
                    "alu": [
                        ("+", addr + lane, self.scratch["forest_values_p"], idx + lane),
                    ],
                })

            # Gather (8 loads)
            for lane in range(0, VLEN, 2):
                self.instrs.append({
                    "load": [
                        ("load", node + lane, addr + lane),
                        ("load", node + lane + 1, addr + lane + 1),
                    ],
                })

        # XOR, hash, branch for all working vectors
        # Interleave to hide latencies
        for w in range(NUM_WORKING):
            # Skip bounds check for simplicity in first version
            emit_xor_hash_branch(w, skip_bounds=False)

        # PHASE 3: Store all working vectors
        for w in range(0, NUM_WORKING, 2):
            self.instrs.append({
                "store": [
                    ("vstore", load_addr[w*2], work_idx[w]),
                    ("vstore", load_addr[w*2+1], work_val[w]),
                ],
            })
            if w + 1 < NUM_WORKING:
                self.instrs.append({
                    "store": [
                        ("vstore", load_addr[(w+1)*2], work_idx[w+1]),
                        ("vstore", load_addr[(w+1)*2+1], work_val[w+1]),
                    ],
                })

        # Update batch counter
        self.instrs.append({
            "alu": [
                ("+", batch_offset, batch_offset, batch_stride),
                ("+", batch_counter, batch_counter, one_const),
            ],
        })
        self.instrs.append({
            "alu": [("<", tmp_scalar, batch_counter, num_batches_const)],
        })
        self.add("flow", ("cond_jump", tmp_scalar, batch_loop_start))

        # Update round counter
        self.instrs.append({
            "alu": [
                ("+", round_counter, round_counter, one_const),
            ],
        })
        self.instrs.append({
            "alu": [("<", tmp_scalar, round_counter, num_rounds_const)],
        })
        self.add("flow", ("cond_jump", tmp_scalar, round_loop_start))

        self.instrs.append({"flow": [("pause",)]})

        # Clean up empty instruction slots
        cleaned = []
        for instr in self.instrs:
            clean_instr = {k: v for k, v in instr.items() if v}
            if clean_instr:
                cleaned.append(clean_instr)
        self.instrs = cleaned


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

    kb = KernelBuilderH72()
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
