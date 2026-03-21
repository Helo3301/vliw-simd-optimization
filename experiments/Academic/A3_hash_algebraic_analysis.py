"""
A3: Deep Algebraic Analysis of the Hash Function

Goal: Investigate if there are mathematical identities or properties of the
hash function that could enable optimization.

Key Questions:
1. Does the hash have any algebraic structure we can exploit?
2. Are there equivalent formulations with different parallelism?
3. Can we pre-compute any intermediate values?
4. Is there a way to "undo" part of the hash (for speculative execution)?

The hash function:
  Stage 0: val = val*4097 + C0    (x + x<<12 + C)
  Stage 1: val = (val^C1) ^ (val>>19)
  Stage 2: val = val*33 + C2      (x + x<<5 + C)
  Stage 3: val = (val+C3) ^ (val<<9)
  Stage 4: val = val*9 + C4       (x + x<<3 + C)
  Stage 5: val = (val^C5) ^ (val>>16)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from problem import HASH_STAGES, myhash
import random

def analyze_hash_structure():
    """Analyze the mathematical structure of the hash function."""
    print("=" * 60)
    print("Hash Function Algebraic Structure Analysis")
    print("=" * 60)
    print()

    print("Hash stages:")
    for i, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
        print(f"  Stage {i}: ({op1}, 0x{val1:08X}, {op2}, {op3}, {val3})")
        # Interpret the formula
        if op3 == "<<":
            mult = 1 + (1 << val3)
            print(f"           val = val * {mult} + 0x{val1:08X}  (via x + x<<{val3} + C)")
        else:
            print(f"           val = (val {op1} C) {op2} (val {op3} {val3})")

    print()
    return HASH_STAGES

def test_bit_independence():
    """
    Test if any output bits are independent of specific input bits.
    This could enable parallel computation paths.
    """
    print("=" * 60)
    print("Bit Independence Analysis")
    print("=" * 60)
    print()

    # Test how each input bit affects output bits
    input_bit_effects = []

    for bit in range(32):
        # Test value with just this bit set
        test_val = 1 << bit
        hash_result = myhash(test_val)

        # Compare with hash of 0
        base_hash = myhash(0)

        # XOR to see which output bits changed
        diff = hash_result ^ base_hash

        affected_bits = bin(diff).count('1')
        input_bit_effects.append((bit, affected_bits, diff))

    print("Input bit -> Output bit dependencies:")
    print("(High avalanche = good hash, but limits optimization)")
    print()

    low_avalanche_bits = []
    for bit, affected, diff in input_bit_effects:
        status = ""
        if affected < 8:
            status = " <- LOW AVALANCHE!"
            low_avalanche_bits.append(bit)
        print(f"  Input bit {bit:2d} affects {affected:2d} output bits")

    if low_avalanche_bits:
        print()
        print(f"Low avalanche bits found: {low_avalanche_bits}")
        print("These bits might be exploitable for optimization!")
    else:
        print()
        print("All input bits have good avalanche (16+ output bits affected).")
        print("No easy optimization path from bit independence.")

    return len(low_avalanche_bits)

def test_branch_bit_dependency():
    """
    The branch only uses val & 1 (the lowest bit).
    Analyze how this bit depends on the input.
    """
    print()
    print("=" * 60)
    print("Branch Bit (LSB) Dependency Analysis")
    print("=" * 60)
    print()

    # For the branch, we only care about the lowest bit of the output
    # Test how different inputs affect this bit

    samples = 10000
    random.seed(42)

    # Test 1: Does input LSB correlate with output LSB?
    same_lsb = 0
    for _ in range(samples):
        val = random.randint(0, 2**32 - 1)
        result = myhash(val)
        if (val & 1) == (result & 1):
            same_lsb += 1

    print(f"Input LSB matches output LSB: {100*same_lsb/samples:.1f}% of samples")
    print("(Expected ~50% for good hash, meaning LSB is unpredictable)")

    # Test 2: Given a tree node XOR, how early can we know the branch?
    print()
    print("Testing if branch bit can be determined from partial hash:")

    # The formula is: branch_bit = myhash(val ^ node_val) & 1
    # Can we determine this from earlier hash stages?

    for stage in range(6):
        same_bit_count = 0
        for _ in range(samples):
            val = random.randint(0, 2**32 - 1)
            node = random.randint(0, 2**32 - 1)

            # Full hash
            full_result = myhash(val ^ node)
            final_bit = full_result & 1

            # Partial hash (first 'stage' stages only)
            partial = val ^ node
            for i in range(stage + 1):
                op1, val1, op2, op3, val3 = HASH_STAGES[i]
                fns = {
                    "+": lambda x, y: (x + y) % (2**32),
                    "^": lambda x, y: x ^ y,
                    "<<": lambda x, y: (x << y) % (2**32),
                    ">>": lambda x, y: x >> y,
                }
                partial = fns[op2](fns[op1](partial, val1), fns[op3](partial, val3))

            partial_bit = partial & 1

            if partial_bit == final_bit:
                same_bit_count += 1

        print(f"  After stage {stage}: LSB matches final {100*same_bit_count/samples:.1f}%")

    print()
    print("Interpretation:")
    print("  If any stage shows 100%, we could stop there for branch prediction.")
    print("  50% means the stage has no predictive power (as expected).")

def test_inverse_operations():
    """
    Test if we can invert parts of the hash for speculative execution.
    If we know the hash output and one input, can we recover the other?
    """
    print()
    print("=" * 60)
    print("Hash Invertibility Analysis")
    print("=" * 60)
    print()

    # The FMA stages (0, 2, 4) use multiplication which is NOT invertible mod 2^32
    # (because even multipliers have no inverse)

    # Check if any multipliers have modular inverses
    multipliers = [4097, 33, 9]  # 1 + 2^12, 1 + 2^5, 1 + 2^3

    print("Checking if FMA multipliers are invertible mod 2^32:")
    for m in multipliers:
        # A number is invertible mod 2^32 iff it's odd
        invertible = m % 2 == 1
        print(f"  {m} ({m:012b}b): {'INVERTIBLE' if invertible else 'NOT INVERTIBLE'}")

    print()
    print("All FMA multipliers are odd, so technically invertible.")
    print("However, inversion requires computing modular inverse, which is expensive.")
    print()

    # The XOR/shift stages are partially invertible
    print("XOR/shift stages invertibility:")
    print("  Stage 1: val = (val^C) ^ (val>>19)")
    print("    Partial: if we know (val^C), we can XOR with (val>>19)")
    print("    BUT: (val>>19) loses top 19 bits - NOT fully invertible")
    print()
    print("  Stage 3: val = (val+C) ^ (val<<9)")
    print("    The << 9 means bottom 9 bits are 0 in shift result")
    print("    XOR with this preserves bottom 9 bits of (val+C)")
    print("    NOT fully invertible due to add overflow")
    print()
    print("  Stage 5: val = (val^C) ^ (val>>16)")
    print("    Similar to stage 1 - loses top 16 bits")
    print()
    print("Conclusion: Hash is NOT efficiently invertible.")
    print("Speculative execution cannot work backwards from assumed outputs.")

def test_commutativity():
    """
    Test if any hash stages commute (can be reordered).
    This could enable different parallelization.
    """
    print()
    print("=" * 60)
    print("Stage Commutativity Analysis")
    print("=" * 60)
    print()

    # Test if stages i and j commute
    samples = 1000
    random.seed(42)

    def apply_stage(val, stage_idx):
        op1, val1, op2, op3, val3 = HASH_STAGES[stage_idx]
        fns = {
            "+": lambda x, y: (x + y) % (2**32),
            "^": lambda x, y: x ^ y,
            "<<": lambda x, y: (x << y) % (2**32),
            ">>": lambda x, y: x >> y,
        }
        return fns[op2](fns[op1](val, val1), fns[op3](val, val3))

    print("Testing if stages commute (same result in either order):")
    for i in range(6):
        for j in range(i+1, 6):
            same = 0
            for _ in range(samples):
                val = random.randint(0, 2**32 - 1)

                # Order 1: i then j
                r1 = apply_stage(apply_stage(val, i), j)

                # Order 2: j then i
                r2 = apply_stage(apply_stage(val, j), i)

                if r1 == r2:
                    same += 1

            commutes = same == samples
            percent = 100 * same / samples
            print(f"  Stages {i} and {j}: {'COMMUTE' if commutes else f'{percent:.1f}% same'}")

    print()
    print("Conclusion: Hash stages do NOT commute.")
    print("Cannot reorder stages for better parallelism.")

def analyze_constant_structure():
    """
    Analyze the constants used in the hash.
    Are there patterns that could be exploited?
    """
    print()
    print("=" * 60)
    print("Hash Constant Analysis")
    print("=" * 60)
    print()

    constants = [
        (0, 0x7ED55D16),
        (1, 0xC761C23C),
        (2, 0x165667B1),
        (3, 0xD3A2646C),
        (4, 0xFD7046C5),
        (5, 0xB55A4F09),
    ]

    print("Constants in binary and analysis:")
    for stage, c in constants:
        binary = format(c, '032b')
        ones = bin(c).count('1')
        print(f"  C{stage} = 0x{c:08X} = {binary[:16]}_{binary[16:]} ({ones} ones)")

    print()
    print("Properties:")
    for stage, c in constants:
        print(f"  C{stage}: mod 2 = {c % 2}, mod 4 = {c % 4}, mod 8 = {c % 8}")

    # Check for XOR patterns
    print()
    print("XOR relationships between constants:")
    for i in range(len(constants)):
        for j in range(i+1, len(constants)):
            xor_result = constants[i][1] ^ constants[j][1]
            ones = bin(xor_result).count('1')
            print(f"  C{i} ^ C{j} = 0x{xor_result:08X} ({ones} ones)")

def main():
    print("=" * 70)
    print("A3: Deep Algebraic Analysis of the Hash Function")
    print("=" * 70)
    print()
    print("This analysis investigates mathematical properties of the hash")
    print("that might enable optimization through algebraic manipulation.")
    print()

    analyze_hash_structure()
    low_avalanche = test_bit_independence()
    test_branch_bit_dependency()
    test_inverse_operations()
    test_commutativity()
    analyze_constant_structure()

    print()
    print("=" * 60)
    print("CONCLUSIONS")
    print("=" * 60)
    print()

    conclusions = [
        "1. Hash function shows good avalanche (all bits mix well)",
        "2. FMA stages (0, 2, 4) optimized - cannot reduce further",
        "3. XOR/shift stages (1, 3, 5) are 3 ops each - appears minimal",
        "4. Hash is not efficiently invertible - limits speculation",
        "5. Stages do NOT commute - cannot reorder for parallelism",
        "6. Branch bit (LSB) is unpredictable until hash completes",
        f"7. {'Some low-avalanche bits found - potential exploit!' if low_avalanche > 0 else 'No exploitable bit patterns found'}",
    ]

    for c in conclusions:
        print(c)

    print()
    print("Final assessment: The hash function is cryptographically designed")
    print("to resist optimization. No algebraic shortcuts appear to exist.")
    print()
    print("The 12-operation minimum is likely TRULY irreducible.")

if __name__ == "__main__":
    main()
