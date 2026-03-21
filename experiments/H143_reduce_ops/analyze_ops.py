"""
Analyze ops breakdown in H140 kernel
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from problem import HASH_STAGES, VLEN

# Count ops per operation type

# Hash stages analysis:
# Stage 0: FMA (val = val * 4097 + const) -> 1 VALU
# Stage 1: XOR, SHIFT, XOR -> 3 VALU
# Stage 2: FMA (val = val * 33 + const) -> 1 VALU
# Stage 3: ADD, SHIFT, XOR -> 3 VALU
# Stage 4: FMA (val = val * 9 + const) -> 1 VALU
# Stage 5: XOR, SHIFT, XOR -> 3 VALU
# Total hash: 12 VALU ops

# Branch computation:
# AND, FMA, ADD -> 3 VALU ops

# Bounds check:
# LT, MUL -> 2 VALU ops

# Per round type analysis:

def count_round_ops():
    """Count ops for each round type"""

    hash_ops = 12  # 3 FMA + 9 other
    branch_ops = 3  # AND, FMA, ADD
    bounds_ops = 2  # LT, MUL

    # Round 0: idx=0 fixed, use tree[0]
    # XOR + hash + branch = 1 + 12 + 3 = 16 VALU
    round_0 = {
        'xor': 1,
        'hash': hash_ops,
        'branch': branch_ops,
        'total_valu': 1 + hash_ops + branch_ops,
        'loads': 0,  # tree[0] preloaded
    }

    # Round 1: idx in {1,2}, arithmetic selection
    # SUB, FMA, XOR + hash + branch = 3 + 12 + 3 = 18 VALU
    round_1 = {
        'selection': 2,  # SUB + FMA
        'xor': 1,
        'hash': hash_ops,
        'branch': branch_ops,
        'total_valu': 2 + 1 + hash_ops + branch_ops,
        'loads': 0,
    }

    # Round 2: idx in {3,4,5,6}, 4-way selection
    # SUB, AND, SHIFT, FMA, FMA, SUB, FMA, XOR + hash + branch = 8 + 12 + 3 = 23 VALU
    round_2 = {
        'selection': 7,  # SUB, AND, SHIFT, 2xFMA, SUB, FMA
        'xor': 1,
        'hash': hash_ops,
        'branch': branch_ops,
        'total_valu': 7 + 1 + hash_ops + branch_ops,
        'loads': 0,
    }

    # Rounds 3-9, 14: gather rounds (no bounds)
    # ADD (addr), XOR + hash + branch = 1 + 1 + 12 + 3 = 17 VALU
    gather_round = {
        'addr': 1,  # ADD
        'xor': 1,
        'hash': hash_ops,
        'branch': branch_ops,
        'total_valu': 1 + 1 + hash_ops + branch_ops,
        'loads': VLEN,  # 8 scalar loads for gather
    }

    # Round 10: gather WITH bounds
    # ADD (addr), XOR + hash + branch + bounds = 1 + 1 + 12 + 3 + 2 = 19 VALU
    round_10 = {
        'addr': 1,
        'xor': 1,
        'hash': hash_ops,
        'branch': branch_ops,
        'bounds': bounds_ops,
        'total_valu': 1 + 1 + hash_ops + branch_ops + bounds_ops,
        'loads': VLEN,
    }

    # Round 11: same as round 0
    round_11 = round_0.copy()

    # Round 12: same as round 1
    round_12 = round_1.copy()

    # Round 13: same as round 2
    round_13 = round_2.copy()

    # Round 15 (final): gather without branch
    # ADD (addr), XOR + hash = 1 + 1 + 12 = 14 VALU
    round_15 = {
        'addr': 1,
        'xor': 1,
        'hash': hash_ops,
        'branch': 0,  # Skip in final round
        'total_valu': 1 + 1 + hash_ops,
        'loads': VLEN,
    }

    return {
        'round_0': round_0,
        'round_1': round_1,
        'round_2': round_2,
        'gather': gather_round,  # rounds 3-9, 14
        'round_10': round_10,
        'round_11': round_11,
        'round_12': round_12,
        'round_13': round_13,
        'round_15': round_15,
    }

def print_analysis():
    rounds = count_round_ops()

    print("=" * 60)
    print("OPS BREAKDOWN PER ROUND TYPE")
    print("=" * 60)

    for name, data in rounds.items():
        print(f"\n{name}:")
        for key, val in data.items():
            print(f"  {key}: {val}")

    # Total per desk (16 rounds)
    # 2 x round_0_like (rounds 0, 11)
    # 2 x round_1_like (rounds 1, 12)
    # 2 x round_2_like (rounds 2, 13)
    # 8 x gather_round (rounds 3-9, 14)
    # 1 x round_10
    # 1 x round_15

    total_valu = (
        2 * rounds['round_0']['total_valu'] +
        2 * rounds['round_1']['total_valu'] +
        2 * rounds['round_2']['total_valu'] +
        8 * rounds['gather']['total_valu'] +
        1 * rounds['round_10']['total_valu'] +
        1 * rounds['round_15']['total_valu']
    )

    total_loads = (
        8 * rounds['gather']['loads'] +
        1 * rounds['round_10']['loads'] +
        1 * rounds['round_15']['loads']
    )

    print("\n" + "=" * 60)
    print("TOTAL PER DESK (16 rounds)")
    print("=" * 60)
    print(f"Total VALU ops: {total_valu}")
    print(f"Total load ops (gathers): {total_loads}")

    # Total for all 16 desks x 2 tiles
    num_desks = 16
    num_tiles = 2
    print(f"\nFor {num_desks} desks x {num_tiles} tiles:")
    print(f"Total VALU ops: {total_valu * num_desks * num_tiles}")
    print(f"Total load ops: {total_loads * num_desks * num_tiles}")

    # Bottleneck analysis
    print("\n" + "=" * 60)
    print("BOTTLENECK ANALYSIS")
    print("=" * 60)
    print(f"VALU limit: 6 per cycle")
    print(f"Load limit: 2 per cycle")

    total_valu_all = total_valu * num_desks * num_tiles
    total_loads_all = total_loads * num_desks * num_tiles

    min_cycles_valu = total_valu_all / 6
    min_cycles_load = total_loads_all / 2

    print(f"\nMinimum cycles (VALU bound): {min_cycles_valu:.1f}")
    print(f"Minimum cycles (Load bound): {min_cycles_load:.1f}")

    # Potential savings
    print("\n" + "=" * 60)
    print("POTENTIAL SAVINGS")
    print("=" * 60)

    # If we reduce branch from 3 to 2 ops:
    branch_savings = 1 * 14 * num_desks * num_tiles  # 14 rounds with branch
    print(f"If branch 3->2 ops: save {branch_savings} VALU = {branch_savings/6:.1f} cycles")

    # If we reduce 4-way selection from 7 to 5 ops:
    select4_savings = 2 * 2 * num_desks * num_tiles  # 2 rounds with 4-way
    print(f"If 4-way select 7->5 ops: save {select4_savings} VALU = {select4_savings/6:.1f} cycles")

    # If we reduce bounds from 2 to 1 op:
    bounds_savings = 1 * 1 * num_desks * num_tiles  # 1 round with bounds
    print(f"If bounds 2->1 ops: save {bounds_savings} VALU = {bounds_savings/6:.1f} cycles")

if __name__ == "__main__":
    print_analysis()
