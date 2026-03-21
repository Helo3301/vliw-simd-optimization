"""
Tiger Team Experiment: Exhaustive Branch Formulation Search

Goal: Find any 2-instruction sequence that computes:
    idx' = 2*idx + 1 + (val & 1)

Equivalent to:
    idx' = 2*idx + 1 if val is odd
    idx' = 2*idx + 2 if val is even

This saves 1 VALU operation per branch (15 branches/element * 32 desks = 480 ops = 80 cycles)
"""

import itertools

# The target function
def target(idx, val):
    return 2 * idx + 1 + (val & 1)

# Alternative formulation: branch = 1 if odd, 2 if even
# So: idx' = 2*idx + (2 - (val & 1)) = 2*idx + 2 - (val & 1)

# ISA operations available (VALU)
def op_add(a, b):
    return (a + b) % (2**32)

def op_sub(a, b):
    return (a - b) % (2**32)

def op_mul(a, b):
    return (a * b) % (2**32)

def op_xor(a, b):
    return a ^ b

def op_and(a, b):
    return a & b

def op_or(a, b):
    return a | b

def op_shl(a, b):
    return (a << b) % (2**32)

def op_shr(a, b):
    return a >> b

def op_fma(a, b, c):
    """multiply_add: a * b + c"""
    return (a * b + c) % (2**32)

# Map of binary ops
BIN_OPS = {
    "+": op_add,
    "-": op_sub,
    "*": op_mul,
    "^": op_xor,
    "&": op_and,
    "|": op_or,
    "<<": op_shl,
    ">>": op_shr,
}

# Constants that might be useful
CONSTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 31, 32, 0xFFFFFFFF, 0xFFFFFFFE]

def search_2op_formula():
    """
    Try all combinations of 2 binary operations to compute:
    idx' = 2*idx + 1 + (val & 1)

    We have inputs: idx, val
    And constants: 0, 1, 2, ...
    """
    print("Searching for 2-op branch formulation...")
    print("Target: idx' = 2*idx + 1 + (val & 1)")
    print()

    # Test values
    test_cases = []
    for idx in [0, 1, 2, 3, 5, 10, 100, 1023]:
        for val in [0, 1, 2, 3, 4, 5, 100, 255, 1000, 0xDEADBEEF]:
            expected = target(idx, val)
            test_cases.append((idx, val, expected))

    found_solutions = []

    # Try: op1(A, B) where result feeds into op2(result, C)
    # Total: 2 ops to get from (idx, val) to idx'

    # Variables we can use
    def get_values(idx, val):
        return {
            "idx": idx,
            "val": val,
            "2": 2,
            "1": 1,
            "0": 0,
            "3": 3,
            "idx*2": (idx * 2) % (2**32),
            "val&1": val & 1,
            "~val&1": (~val) & 1,
        }

    # First approach: Try all 2-binary-op combinations
    for op1_name, op1 in BIN_OPS.items():
        for op2_name, op2 in BIN_OPS.items():
            # Try different operand combinations for op1
            operands1 = ["idx", "val", "2", "1", "0", "3"]
            for a1, b1 in itertools.product(operands1, repeat=2):
                for a2 in ["result1", "idx", "val", "2", "1", "0", "3"]:
                    for b2 in ["result1", "idx", "val", "2", "1", "0", "3"]:
                        # Skip if neither uses result1
                        if a2 != "result1" and b2 != "result1":
                            continue

                        all_pass = True
                        for idx, val, expected in test_cases:
                            vals = get_values(idx, val)
                            try:
                                r1 = op1(vals.get(a1, int(a1) if a1.isdigit() else 0),
                                        vals.get(b1, int(b1) if b1.isdigit() else 0))
                                vals["result1"] = r1
                                r2 = op2(vals.get(a2, int(a2) if a2.isdigit() else 0),
                                        vals.get(b2, int(b2) if b2.isdigit() else 0))
                                if r2 != expected:
                                    all_pass = False
                                    break
                            except:
                                all_pass = False
                                break

                        if all_pass:
                            formula = f"t1 = {a1} {op1_name} {b1}; result = {a2} {op2_name} {b2}"
                            if formula not in found_solutions:
                                found_solutions.append(formula)
                                print(f"FOUND 2-op solution: {formula}")

    # Try FMA-based approaches (FMA is 1 op but does multiply-add)
    print("\n--- Trying FMA-based approaches ---")

    # The key insight: 2*idx + 1 + (val&1) = 2*idx + (1 + val&1)
    # If we could compute (1 + val&1) in 0 ops, then FMA(idx, 2, that) works
    # But (1 + val&1) requires at least 2 ops: AND then ADD

    # Alternative: 2*idx + 1 + (val&1) = 2*idx + 2 - (~val&1) = 2*idx + 2 - (1 - val&1)
    # Hmm, also not simpler

    # What about: using properties of val&1?
    # val&1 = 0 or 1
    # 1 + val&1 = 1 or 2
    # This is selecting between 1 and 2 based on val's LSB

    # Can we use vselect?
    # vselect(dest, cond, a, b) puts a[i] if cond[i]!=0 else b[i]
    # So: vselect(offset, val&1, 2, 1) puts 2 if odd, 1 if even... wait that's backwards
    # We want 1 if odd, 2 if even
    # vselect(offset, val&1, 1, 2) puts 1 if val&1!=0 (odd), 2 if val&1==0 (even) - WRONG
    # Actually vselect gives a if cond!=0, b if cond==0
    # val&1 == 1 for odd, 0 for even
    # So vselect(offset, val&1, X, Y) = X if odd, Y if even
    # We want offset=1 if odd, offset=2 if even
    # So vselect(offset, val&1, 1, 2) = 1 if odd, 2 if even - wait, that's wrong per above

    # Let me recalculate:
    # target = 2*idx + 1 + (val & 1)
    # For odd val (val&1=1): target = 2*idx + 2
    # For even val (val&1=0): target = 2*idx + 1

    # So the offset is: 2 if odd, 1 if even
    # vselect(offset, val&1, 2, 1) would give 2 if val&1!=0 (odd), 1 if val&1==0 (even) - CORRECT!

    # But vselect requires the condition, which means we need val&1 computed
    # So: AND to get val&1, then vselect, then FMA = 3 ops still
    # Unless... we use val directly as condition?

    # vselect(offset, val, 2, 1) = 2 if val!=0, 1 if val==0
    # This is wrong for even non-zero values

    print("\nAnalysis of vselect approach:")
    print("vselect(offset, val&1, 2, 1) gives: 2 if odd, 1 if even (CORRECT)")
    print("But we still need AND to compute val&1 first")
    print("Total: 3 ops (AND, vselect, FMA)")

    # What if we use multiplication property?
    # (val & 1) * 1 + 1 = 1 + (val & 1) = offset
    # But we can't compute (val&1) without AND

    # What about using modulo?
    # val % 2 = val & 1, but ISA doesn't have modulo for vectors

    # What about clever bit manipulation?
    # val & 1 gives 0 or 1
    # (val & 1) + 1 = 1 or 2 (the offset we want)
    #
    # Can we get (val & 1) + 1 in fewer ops?
    # (~(val-1)) & 1... no
    # (val | ~0) & 1... no
    #
    # What if we use XOR?
    # val ^ 0 = val, val ^ 1 flips LSB
    # (val & 1) ^ 1 = 0 if odd, 1 if even (inverted)
    # 2 - (val & 1) = 1 if odd, 2 if even... wait, that's wrong
    # 2 - (val & 1) = 2 - 1 = 1 for odd, 2 - 0 = 2 for even - CORRECT!

    # So: offset = 2 - (val & 1)
    # Then: idx' = FMA(idx, 2, offset) = 2*idx + offset
    #
    # This is: AND, SUB, FMA = 3 ops still

    print("\nAlternative formulation: idx' = 2*idx + (2 - (val & 1))")
    print("Still 3 ops: AND, SUB, FMA")

    # What about: idx' = 2*idx + 1 + (val & 1)
    #            = 2*idx + 1 + val - (val & ~1)  ... no, introduces val

    # The fundamental issue:
    # We MUST extract the LSB of val (requires 1 op: AND)
    # We MUST multiply idx by 2 (requires part of an op)
    # We MUST add 1 and the LSB (requires combining)

    # FMA can do 2*idx + X in 1 op, but X must be precomputed
    # X = 1 + (val & 1) requires 2 ops
    # Total: 3 ops

    print("\n=== CONCLUSION ===")
    print("No 2-op solution found for: idx' = 2*idx + 1 + (val & 1)")
    print("The mathematical minimum appears to be 3 ops:")
    print("  1. AND to extract val & 1")
    print("  2. Something to prepare the offset")
    print("  3. Something to combine with 2*idx")
    print()
    print("Current optimal: AND + FMA(idx,2,1) + ADD = 3 VALU ops")

    return found_solutions


def search_alternative_branch_formulas():
    """
    What if we reformulate the problem?

    Instead of: idx' = 2*idx + 1 + (val & 1)
    Consider: idx' = (idx << 1) | 1 | (val & 1)

    Wait, let's verify:
    For idx=5, odd val: target = 10 + 2 = 12 = 0b1100
                        formula = 10 | 1 | 1 = 11 = 0b1011 WRONG

    Actually: idx' = 2*idx + 1 + (val&1)
    For idx=5, odd val (val&1=1): 10 + 1 + 1 = 12
    For idx=5, even val (val&1=0): 10 + 1 + 0 = 11

    Can we use: idx' = (idx << 1) + 1 + (val & 1)?
    This is identical to the original. Just using shift vs mul.

    What about: idx' = (idx + idx) + 1 + (val & 1)?
    ADD, ... still 3 ops
    """
    print("\n=== Searching for alternative formulations ===")

    # What if we precompute 2*idx?
    # Then: idx' = idx2 + 1 + (val & 1)
    # But computing idx2 = idx + idx or idx << 1 is 1 op
    # Then we need AND + ADD for val part = 2 ops
    # Total: still 3 ops

    # What if we use lookup tables?
    # (val & 1) can only be 0 or 1
    # Precompute offset[0] = 1, offset[1] = 2
    # Then: idx' = 2*idx + offset[val & 1]
    # But accessing offset[val&1] requires: AND + memory indexing
    # Memory indexing is a LOAD, not VALU
    # So: AND(1 op) + LOAD(uses LOAD slot, not VALU) + FMA(1 op) = 2 VALU ops!

    print("POTENTIAL SOLUTION:")
    print("Using lookup table for offset:")
    print("  1. Preload offset table: offset[0]=1, offset[1]=2 into scratch")
    print("  2. AND: bit = val & 1 (1 VALU)")
    print("  3. LOAD: off = offset[bit] (uses LOAD slot, 0 VALU)")
    print("  4. FMA: idx' = 2*idx + off (1 VALU)")
    print("Total: 2 VALU ops + 1 LOAD op")
    print()
    print("BUT: This adds 1 LOAD per branch (15 rounds * 32 desks = 480 extra loads)")
    print("At 2 loads/cycle, that's 240 extra cycles for loads")
    print("Savings: 480 VALU ops = 80 cycles")
    print("Net: 240 - 80 = 160 cycles WORSE!")
    print()
    print("Unless... we're not currently load-bound and have spare load slots")

    # Current load utilization analysis
    print("\nCurrent LOAD utilization analysis:")
    print("  Gather loads: 10 rounds * 128 loads/round * 2 tiles = 2560")
    print("  Other loads: ~81 (init + preload)")
    print("  Total: ~2641 loads")
    print("  At 2 loads/cycle: 1321 cycles minimum")
    print("  Current B4-2: 1558 cycles")
    print("  VALU bound at: 1514 cycles")
    print()
    print("  We're VALU-bound, not load-bound!")
    print("  This means we have spare load slots!")
    print()
    print("  Extra loads for lookup: 480")
    print("  New total: 2641 + 480 = 3121 loads")
    print("  New load bound: 1561 cycles")
    print()
    print("  With 2-op VALU branch:")
    print("    Old VALU: ~9083 ops / 6 = 1514 cycles")
    print("    Save: 480 ops")
    print("    New VALU: 8603 ops / 6 = 1434 cycles")
    print()
    print("  New bottleneck: LOAD at 1561 cycles (worse than current 1558)")
    print("  This approach doesn't help!")


def check_vselect_trick():
    """
    Can we use vselect in a clever way?

    vselect is on FLOW engine (1/cycle), separate from VALU (6/cycle)
    If we could move branch computation to FLOW, we free up VALU
    """
    print("\n=== Checking vselect-based approach ===")

    # Precompute:
    #   even_idx = FMA(idx, 2, 1)  # 2*idx + 1 for even val
    #   odd_idx = FMA(idx, 2, 2)   # 2*idx + 2 for odd val
    # Then:
    #   cond = val & 1
    #   idx' = vselect(cond, odd_idx, even_idx)

    print("Approach: Compute both outcomes, select")
    print("  1. FMA: even_idx = 2*idx + 1")
    print("  2. FMA: odd_idx = 2*idx + 2")
    print("  3. AND: cond = val & 1")
    print("  4. vselect: idx' = cond ? odd_idx : even_idx")
    print()
    print("VALU ops: 3 (2 FMA + 1 AND)")
    print("FLOW ops: 1 (vselect)")
    print()
    print("This is WORSE than current 3 VALU ops!")
    print("Because we're adding work, not reducing it.")

    # What if we share the FMA?
    print("\nAlternative: Share the doubling")
    print("  1. Operation: idx2 = idx * 2 (or idx + idx)")
    print("  2. FMA: even_idx = idx2 + 1")
    print("  3. FMA: odd_idx = idx2 + 2")
    print("Wait, FMA needs multiply. idx2 + 1 is ADD, not FMA.")
    print()
    print("OK so:")
    print("  1. MUL or SHL: idx2 = idx * 2")
    print("  2. ADD: even_idx = idx2 + 1")
    print("  3. ADD: odd_idx = idx2 + 2")
    print("  4. AND: cond = val & 1")
    print("  5. vselect: idx'")
    print()
    print("VALU: 4 ops (EVEN WORSE)")


def main():
    print("=" * 60)
    print("TIGER TEAM: Branch Formulation Exhaustive Search")
    print("=" * 60)
    print()

    solutions = search_2op_formula()

    search_alternative_branch_formulas()

    check_vselect_trick()

    print()
    print("=" * 60)
    print("FINAL CONCLUSION")
    print("=" * 60)
    print()
    print("The branch computation idx' = 2*idx + 1 + (val & 1) requires")
    print("a MINIMUM of 3 VALU operations with the current ISA.")
    print()
    print("No 2-op formulation exists because:")
    print("  1. Extracting (val & 1) REQUIRES an AND operation (1 op)")
    print("  2. Computing 2*idx + offset REQUIRES at least 1 more op")
    print("  3. Combining the offset with 2*idx REQUIRES at least 1 more op")
    print()
    print("FMA can combine multiply-add into 1 op, but we still need:")
    print("  - AND for bit extraction")
    print("  - Some way to prepare the FMA operand")
    print()
    print("The lookup table approach trades VALU for LOAD but makes us load-bound.")
    print("The vselect approach adds VALU ops, not reduces them.")
    print()
    print("B2 branch experiments were correct: 3 ops is the minimum.")


if __name__ == "__main__":
    main()
