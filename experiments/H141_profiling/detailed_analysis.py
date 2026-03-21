"""
Detailed per-round analysis of H140 kernel to identify specific optimization opportunities.
"""

import sys
import os
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from problem import SLOT_LIMITS, VLEN, HASH_STAGES

def analyze_hash_stages():
    """Analyze the hash computation."""
    print("=" * 80)
    print("HASH STAGE ANALYSIS")
    print("=" * 80)
    print()
    print("Hash stages (6 total):")
    for i, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
        print(f"  Stage {i}: val = (val {op1} {hex(val1)}) {op2} (val {op3} {val3})")

    # Current implementation uses:
    # FMA stages (0, 2, 4): 1 multiply_add each = 1 VALU op
    # Non-FMA stages (1, 3, 5):
    #   Stage 1: XOR, shift_right, XOR = 3 VALU ops
    #   Stage 3: ADD, shift_left, XOR = 3 VALU ops
    #   Stage 5: XOR, shift_right, XOR = 3 VALU ops

    print()
    print("VALU ops per hash stage in H140 implementation:")
    print("  Stage 0 (FMA): 1 multiply_add")
    print("  Stage 1: 3 ops (^, >>, ^)")
    print("  Stage 2 (FMA): 1 multiply_add")
    print("  Stage 3: 3 ops (+, <<, ^)")
    print("  Stage 4 (FMA): 1 multiply_add")
    print("  Stage 5: 3 ops (^, >>, ^)")
    print("  Total: 12 VALU ops per desk per round")


def analyze_round_types():
    """Analyze operations per round type."""
    print()
    print("=" * 80)
    print("ROUND TYPE ANALYSIS")
    print("=" * 80)
    print()

    # Round types in H140:
    # Round 0: XOR with tree[0], hash, branch = 1 + 12 + 3 = 16 VALU ops
    # Round 1: Arithmetic selection (2-way), XOR, hash, branch = 2 + 1 + 12 + 3 = 18 VALU ops
    # Round 2: 4-way selection, XOR, hash, branch = 8 + 1 + 12 + 3 = 24 VALU ops
    # Rounds 3-9 (gather): addr calc, XOR, hash, branch = 1 + 1 + 12 + 3 = 17 VALU ops (per desk)
    # Round 10 (gather+bounds): = 17 + 2 = 19 VALU ops
    # Round 11: same as round 0 = 16 VALU ops
    # Round 12: same as round 1 = 18 VALU ops
    # Round 13: same as round 2 = 24 VALU ops
    # Round 14 (gather): = 17 VALU ops
    # Round 15 (final, no branch): = 17 - 3 = 14 VALU ops

    round_valu = {
        0: 16,
        1: 18,
        2: 24,
        3: 17, 4: 17, 5: 17, 6: 17, 7: 17, 8: 17, 9: 17,
        10: 19,
        11: 16,
        12: 18,
        13: 24,
        14: 17,
        15: 14,
    }

    round_loads = {
        0: 0,
        1: 0,
        2: 0,
        3: 8, 4: 8, 5: 8, 6: 8, 7: 8, 8: 8, 9: 8,
        10: 8,
        11: 0,
        12: 0,
        13: 0,
        14: 8,
        15: 8,
    }

    print("VALU ops per round (per desk):")
    total_valu_per_desk = 0
    for r in range(16):
        total_valu_per_desk += round_valu[r]
        print(f"  Round {r:2d}: {round_valu[r]:2d} VALU ops")
    print(f"  Total per desk: {total_valu_per_desk} VALU ops")

    print()
    print("LOADs per round (per desk):")
    total_loads_per_desk = 0
    for r in range(16):
        total_loads_per_desk += round_loads[r]
        print(f"  Round {r:2d}: {round_loads[r]:2d} loads (gather)")
    print(f"  Total per desk: {total_loads_per_desk} loads")

    # 16 desks, 2 tiles
    num_desks = 16
    num_tiles = 2

    print()
    print("=" * 80)
    print("TOTAL OPERATION COUNT VERIFICATION")
    print("=" * 80)
    print()

    # Setup operations (approximate)
    setup_valu = 24  # vbroadcast ops
    setup_loads = 54  # const ops + header loads + tree preloads

    # Per-tile operations
    tile_loads = 32  # vload for idx/val (16 desks * 2)
    tile_stores = 32  # vstore for idx/val (16 desks * 2)

    # Per-desk operations (16 rounds)
    desk_valu = total_valu_per_desk
    desk_loads = total_loads_per_desk

    total_valu = setup_valu + num_tiles * num_desks * desk_valu
    total_loads = setup_loads + num_tiles * (tile_loads + num_desks * desk_loads)
    total_stores = num_tiles * tile_stores

    print(f"Calculated totals:")
    print(f"  VALU: {setup_valu} (setup) + {num_tiles} * {num_desks} * {desk_valu} = {total_valu}")
    print(f"  LOAD: {setup_loads} (setup) + {num_tiles} * ({tile_loads} + {num_desks} * {desk_loads}) = {total_loads}")
    print(f"  STORE: {num_tiles} * {tile_stores} = {total_stores}")


def analyze_optimization_opportunities():
    """Identify specific optimization opportunities."""
    print()
    print("=" * 80)
    print("OPTIMIZATION OPPORTUNITIES")
    print("=" * 80)
    print()

    # Target: save 66 cycles = ~396 VALU ops

    print("1. HASH COMPUTATION OPTIMIZATION")
    print("-" * 40)
    print("   Current: 12 VALU ops per hash")
    print("   - 3 FMA stages: 3 ops")
    print("   - 3 non-FMA stages: 9 ops (3 each)")
    print()
    print("   Potential: Reduce non-FMA stages")
    print("   - Could potentially reduce shift+xor patterns")
    print("   - Each VALU op saved * 32 desks * 16 rounds = 512 ops")
    print()

    print("2. BRANCH COMPUTATION OPTIMIZATION")
    print("-" * 40)
    print("   Current: 3 VALU ops per branch")
    print("   - idx = idx * 2 + 1 + (val & 1)")
    print("   - Uses: multiply_add, &, +")
    print()
    print("   Note: Round 15 already skips branch (H133 optimization)")
    print("   - Savings: 3 * 32 = 96 VALU ops already saved")
    print()

    print("3. BOUNDS CHECK OPTIMIZATION")
    print("-" * 40)
    print("   Current: 2 VALU ops at round 10")
    print("   - < comparison, * to zero out")
    print()
    print("   Round 10 bounds are critical - no easy savings here")
    print()

    print("4. SELECTION LOGIC OPTIMIZATION")
    print("-" * 40)
    print("   Round 1 (2-way): 2 VALU ops (-, multiply_add)")
    print("   Round 2 (4-way): 8 VALU ops")
    print("   Rounds 11-13: Same pattern")
    print()
    print("   4-way selection breakdown:")
    print("   - -, &, >>, multiply_add, multiply_add, -, multiply_add")
    print("   Could potentially optimize with different selection strategy")
    print()

    print("5. POTENTIAL SAVINGS CALCULATION")
    print("-" * 40)
    print()

    # Target: 66 cycles to save
    # Need ~396 VALU ops to save

    # Option A: Reduce hash ops per round
    # If we save 1 VALU op per hash: 32 desks * 16 rounds = 512 VALU ops = ~85 cycles

    # Option B: Optimize 4-way selection
    # If we save 2 VALU ops per 4-way: 32 desks * 2 rounds * 2 tiles = 256 VALU ops = ~42 cycles

    print("Option A: Reduce 1 VALU op per hash stage")
    print("  32 desks * 16 rounds = 512 VALU ops saved = ~85 cycles")
    print()

    print("Option B: Reduce 2 VALU ops in 4-way selection")
    print("  32 desks * 4 uses (R2, R13 x2 tiles) = 128 VALU ops saved = ~21 cycles")
    print()

    print("Option C: Optimize gather rounds (reduce addr calc)")
    print("  Currently 1 VALU op for addr calc per gather")
    print("  Could potentially batch or reuse")
    print()


def main():
    analyze_hash_stages()
    analyze_round_types()
    analyze_optimization_opportunities()


if __name__ == "__main__":
    main()
