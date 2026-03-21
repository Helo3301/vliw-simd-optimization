"""Debug trace to find where MATH1 diverges from reference."""
import random
import sys
sys.path.insert(0, ".")

from problem import (
    Tree, Input, build_mem_image, reference_kernel2, Machine, N_CORES, HASH_STAGES
)

def myhash(val):
    """Reference hash function."""
    val = val % (2**32)
    for op1, val1, op2, op3, val3 in HASH_STAGES:
        if op1 == "+":
            val = (val + val1) % (2**32)
        else:
            val = val ^ val1
        if op2 == "<<":
            tmp = (val << val3) % (2**32)
        else:
            tmp = val >> val3
        if op3 == "+":
            val = (val + tmp) % (2**32)
        else:
            val = val ^ tmp
    return val

def trace_reference(forest, inp, rounds, element_idx=0):
    """Trace single element through reference implementation."""
    forest_values_p = 7  # From memory layout
    n_nodes = len(forest.values)

    idx = inp.indices[element_idx]
    val = inp.values[element_idx]

    print(f"Initial: idx={idx}, val={val}")
    print()

    for rnd in range(min(rounds, 3)):  # Just trace first 3 rounds
        node_val = forest.values[idx]
        print(f"Round {rnd}:")
        print(f"  idx={idx}, node_val={node_val}")

        val = myhash(val ^ node_val)
        print(f"  after hash: val={val}")

        branch_bit = 1 if val % 2 == 0 else 2  # NOTE: this is opposite of what I expected!
        old_idx = idx
        idx = 2 * idx + branch_bit
        print(f"  branch: old_idx={old_idx}, branch_bit={branch_bit}, new_idx={idx}")

        if idx >= n_nodes:
            print(f"  WRAP: idx={idx} >= n_nodes={n_nodes}, wrapping to 0")
            idx = 0
        print()

def trace_math1(forest, inp, rounds, element_idx=0):
    """Trace single element through MATH1 address representation."""
    forest_values_p = 7
    n_nodes = len(forest.values)

    idx = inp.indices[element_idx]
    val = inp.values[element_idx]
    addr = forest_values_p + idx

    # Precompute constant
    neg_forest_p_plus_1 = (1 - forest_values_p) % (2**32)
    print(f"Precomputed: 1 - forest_p = {neg_forest_p_plus_1} (mod 2^32)")

    print(f"Initial: idx={idx}, addr={addr}, val={val}")
    print()

    for rnd in range(min(rounds, 3)):
        computed_idx = addr - forest_values_p
        node_val = forest.values[computed_idx]
        print(f"Round {rnd}:")
        print(f"  addr={addr}, computed_idx={computed_idx}, node_val={node_val}")

        val = myhash(val ^ node_val)
        print(f"  after hash: val={val}")

        # MATH1 branch formula
        branch_bit = val & 1  # 0 or 1
        offset = (neg_forest_p_plus_1 + branch_bit) % (2**32)
        old_addr = addr
        addr = (2 * addr + offset) % (2**32)
        new_idx = addr - forest_values_p

        # Reference formula for comparison
        ref_branch = 1 if val % 2 == 0 else 2
        ref_idx = 2 * computed_idx + ref_branch

        print(f"  branch_bit={branch_bit}, offset={offset}")
        print(f"  MATH1: old_addr={old_addr}, new_addr={addr}, new_idx={new_idx}")
        print(f"  REF:   expected new_idx={ref_idx}")

        if new_idx != ref_idx:
            print(f"  *** MISMATCH! ***")
        print()

if __name__ == "__main__":
    random.seed(123)
    forest = Tree.generate(10)
    inp = Input.generate(forest, 256, 16)

    print("="*60)
    print("REFERENCE IMPLEMENTATION")
    print("="*60)
    trace_reference(forest, inp, 16, element_idx=0)

    print("="*60)
    print("MATH1 ADDRESS REPRESENTATION")
    print("="*60)
    trace_math1(forest, inp, 16, element_idx=0)
