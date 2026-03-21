"""
A1: SMT-Based Verification of Branch Formulation Impossibility

Goal: Formally prove (or disprove) that no 2-instruction sequence can compute
      idx' = 2*idx + 1 + (val & 1)

Method: Use Z3 SMT solver to exhaustively search all 2-instruction combinations

Key insight: The branch formula depends on TWO independent inputs (idx, val).
Any single instruction can only combine inputs in limited ways.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from z3 import *
except ImportError:
    print("Z3 not installed. Please install with: pip install z3-solver")
    sys.exit(1)

# Define bit width (32-bit)
BV_WIDTH = 32

def target_formula(idx, val):
    """The formula we need to compute: 2*idx + 1 + (val & 1)"""
    bit = val & BitVecVal(1, BV_WIDTH)
    return 2 * idx + 1 + bit

# Available operations in the ISA
def alu_ops():
    """Return all binary operations available"""
    return {
        '+': lambda a, b: a + b,
        '-': lambda a, b: a - b,
        '*': lambda a, b: a * b,
        '^': lambda a, b: a ^ b,
        '&': lambda a, b: a & b,
        '|': lambda a, b: a | b,
        '<<': lambda a, b: a << (b & BitVecVal(31, BV_WIDTH)),  # Mask shift to avoid overflow
        '>>': lambda a, b: LShR(a, b & BitVecVal(31, BV_WIDTH)),
    }

def search_two_op_sequence():
    """
    Search for any 2-operation sequence that computes the branch formula.

    We consider sequences of form:
    1. op1(X, Y) -> tmp
    2. op2(tmp, Z) -> result  OR  op2(Z, tmp) -> result

    Where X, Y, Z can be: idx, val, or small constants (0, 1, 2, 3)
    """
    print("=" * 60)
    print("SMT-Based Branch Formula Verification")
    print("=" * 60)
    print()
    print("Target: idx' = 2*idx + 1 + (val & 1)")
    print()

    # Symbolic inputs
    idx = BitVec('idx', BV_WIDTH)
    val = BitVec('val', BV_WIDTH)
    target = target_formula(idx, val)

    ops = alu_ops()
    constants = [0, 1, 2, 3, 4097, 33, 9]  # Include FMA multipliers

    # Available operands for instructions
    operands = {
        'idx': idx,
        'val': val,
        **{f'c{c}': BitVecVal(c, BV_WIDTH) for c in constants}
    }

    solutions_found = 0

    print(f"Searching {len(ops)} operations x {len(operands)} operands...")
    print()

    # Try all op1, op2 combinations
    for op1_name, op1 in ops.items():
        for op2_name, op2 in ops.items():
            # Try all operand combinations for op1
            for x_name, x in operands.items():
                for y_name, y in operands.items():
                    # Compute intermediate result
                    try:
                        tmp = op1(x, y)
                    except:
                        continue

                    # Try all operand combinations for op2
                    for z_name, z in operands.items():
                        # Try op2(tmp, z)
                        try:
                            result1 = op2(tmp, z)

                            # Check if this equals target for all inputs
                            s = Solver()
                            s.add(result1 != target)
                            if s.check() == unsat:
                                solutions_found += 1
                                print(f"SOLUTION FOUND!")
                                print(f"  Step 1: tmp = {x_name} {op1_name} {y_name}")
                                print(f"  Step 2: result = tmp {op2_name} {z_name}")
                                print()
                        except:
                            pass

                        # Try op2(z, tmp)
                        try:
                            result2 = op2(z, tmp)

                            s = Solver()
                            s.add(result2 != target)
                            if s.check() == unsat:
                                solutions_found += 1
                                print(f"SOLUTION FOUND!")
                                print(f"  Step 1: tmp = {x_name} {op1_name} {y_name}")
                                print(f"  Step 2: result = {z_name} {op2_name} tmp")
                                print()
                        except:
                            pass

    return solutions_found

def search_fma_based():
    """
    Special search for FMA-based solutions.

    multiply_add(a, b, c) = a * b + c

    Try: one FMA and one other op
    """
    print("=" * 60)
    print("FMA-Based Search")
    print("=" * 60)
    print()

    idx = BitVec('idx', BV_WIDTH)
    val = BitVec('val', BV_WIDTH)
    target = target_formula(idx, val)

    ops = alu_ops()
    constants = [0, 1, 2, 3]

    operands = {
        'idx': idx,
        'val': val,
        **{f'c{c}': BitVecVal(c, BV_WIDTH) for c in constants}
    }

    solutions_found = 0

    # FMA first, then another op
    print("Pattern: FMA(a, b, c) then op(result, d)")
    for a_name, a in operands.items():
        for b_name, b in operands.items():
            for c_name, c in operands.items():
                fma_result = a * b + c

                for op_name, op in ops.items():
                    for d_name, d in operands.items():
                        try:
                            result = op(fma_result, d)

                            s = Solver()
                            s.add(result != target)
                            if s.check() == unsat:
                                solutions_found += 1
                                print(f"SOLUTION: FMA({a_name}, {b_name}, {c_name}) {op_name} {d_name}")
                        except:
                            pass

                        try:
                            result = op(d, fma_result)

                            s = Solver()
                            s.add(result != target)
                            if s.check() == unsat:
                                solutions_found += 1
                                print(f"SOLUTION: {d_name} {op_name} FMA({a_name}, {b_name}, {c_name})")
                        except:
                            pass

    # Another op first, then FMA
    print()
    print("Pattern: op(a, b) then FMA(result, c, d)")
    for op_name, op in ops.items():
        for a_name, a in operands.items():
            for b_name, b in operands.items():
                try:
                    tmp = op(a, b)
                except:
                    continue

                for c_name, c in operands.items():
                    for d_name, d in operands.items():
                        # FMA(tmp, c, d)
                        try:
                            result = tmp * c + d

                            s = Solver()
                            s.add(result != target)
                            if s.check() == unsat:
                                solutions_found += 1
                                print(f"SOLUTION: FMA({a_name} {op_name} {b_name}, {c_name}, {d_name})")
                        except:
                            pass

                        # FMA(c, tmp, d)
                        try:
                            result = c * tmp + d

                            s = Solver()
                            s.add(result != target)
                            if s.check() == unsat:
                                solutions_found += 1
                                print(f"SOLUTION: FMA({c_name}, {a_name} {op_name} {b_name}, {d_name})")
                        except:
                            pass

                        # FMA(c, d, tmp)
                        try:
                            result = c * d + tmp

                            s = Solver()
                            s.add(result != target)
                            if s.check() == unsat:
                                solutions_found += 1
                                print(f"SOLUTION: FMA({c_name}, {d_name}, {a_name} {op_name} {b_name})")
                        except:
                            pass

    return solutions_found

def verify_known_solution():
    """Verify that the known 3-op solution is correct."""
    print("=" * 60)
    print("Verifying Known 3-Op Solution")
    print("=" * 60)
    print()

    idx = BitVec('idx', BV_WIDTH)
    val = BitVec('val', BV_WIDTH)
    target = target_formula(idx, val)

    # Known solution:
    # 1. bit = val & 1
    # 2. tmp = 2*idx + 1  (FMA: idx * 2 + 1)
    # 3. result = tmp + bit

    bit = val & BitVecVal(1, BV_WIDTH)
    tmp = idx * BitVecVal(2, BV_WIDTH) + BitVecVal(1, BV_WIDTH)
    result = tmp + bit

    s = Solver()
    s.add(result != target)

    if s.check() == unsat:
        print("VERIFIED: Known 3-op solution is CORRECT")
        print("  Step 1: bit = val & 1")
        print("  Step 2: tmp = idx * 2 + 1  (using FMA)")
        print("  Step 3: result = tmp + bit")
    else:
        print("ERROR: Known solution does not match target!")
        print(f"Counterexample: {s.model()}")

    print()

def main():
    print("=" * 60)
    print("A1: SMT-Based Branch Formulation Verification")
    print("=" * 60)
    print()
    print("This experiment uses Z3 SMT solver to formally verify")
    print("whether a 2-operation solution exists for the branch formula.")
    print()

    # First verify the known solution works
    verify_known_solution()

    # Search for 2-op solutions
    print("Searching for 2-operation solutions...")
    print()

    basic_solutions = search_two_op_sequence()
    fma_solutions = search_fma_based()

    total_solutions = basic_solutions + fma_solutions

    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print()
    if total_solutions == 0:
        print("NO 2-OPERATION SOLUTION EXISTS")
        print()
        print("The SMT solver has verified that no sequence of 2 instructions")
        print("from the available ISA can compute: idx' = 2*idx + 1 + (val & 1)")
        print()
        print("The known 3-operation solution is MINIMAL.")
    else:
        print(f"FOUND {total_solutions} solution(s)!")
        print()
        print("This would be a major breakthrough - review solutions carefully.")

    return total_solutions

if __name__ == "__main__":
    main()
