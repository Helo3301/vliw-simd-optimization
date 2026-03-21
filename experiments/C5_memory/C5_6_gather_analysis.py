"""
C5-6: Analysis of Gather Rounds - Can We Reduce Them?

After 5 experiments showing memory access patterns don't help, let's analyze
why and what would actually help.

The core problem:
- VALU bound: 1,514 cycles (9,083 ops / 6 per cycle)
- LOAD bound: 1,321 cycles (2,641 ops / 2 per cycle)
- B4-2 achieved: 1,558 cycles

VALU is the bottleneck, not LOADs. Memory access patterns cannot improve
performance unless they reduce VALU operations.

Analysis of round structure:
- Rounds 0, 1, 2: Fused (preloaded tree[0-6])
- Rounds 3-9: 7 gather rounds (random tree access)
- Round 10: Gather + bounds check
- Rounds 11, 12, 13: Fused (after wrap, preloaded tree[0-6])
- Round 14: Gather
- Round 15: Final gather (no branch)

Total gather rounds: 10 (rounds 3-10, 14, 15)

Each gather round requires:
- 1 VALU for addr = forest_p + idx
- 8 LOADs (per desk)
- 1 VALU for XOR
- 12 VALU for hash
- 3 VALU for branch (except round 15)

That's 17 VALU per gather round (14 for round 15), plus LOADs.

Could we fuse more gather rounds?
- Problem: After hash, we need to LOAD the next node, which depends on branch result
- Branch result depends on hash output (bit = val & 1)
- This is a true data dependency - no way around it

The only way to reduce gather operations:
1. Preload more tree nodes (but tree has 2047 nodes, can't fit in scratch)
2. Reduce number of rounds (but that's the algorithm specification)
3. Find a mathematical identity that skips rounds (none known)

Conclusion: Gather rounds are irreducible given the algorithm.

Let's verify B4-2 is actually optimal by counting operations:
"""

import random
import argparse
import sys
import os
from collections import defaultdict

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


def analyze_b4_2_operations():
    """Analyze operation counts from B4-2"""

    # Per-desk operation counts
    per_desk = {
        # Rounds 0-2 fused (preloaded nodes)
        'rounds_0_2': {
            'xor': 3,  # One per round
            'hash': 3 * 12,  # 12 ops per hash
            'branch': 2 * 3,  # Rounds 0 and 1 have branch-like ops (bit extraction + idx)
            # Round 2 has 4-way selection: 2 FMA + 1 vselect (on FLOW)
            'selection': 2 * 1 + 2 * 2,  # 2 FMAs for low/high pairs in R2
            'idx_update': 2 + 2 + 3,  # bit extract + idx updates
        },
        # Rounds 3-9: 7 gather rounds
        'rounds_3_9': {
            'addr': 7,  # addr = forest_p + idx
            'xor': 7,
            'hash': 7 * 12,
            'branch': 7 * 3,
        },
        # Round 10: gather + bounds
        'round_10': {
            'addr': 1,
            'xor': 1,
            'hash': 12,
            'branch': 3,
            'bounds': 2,  # compare + conditional zero
        },
        # Rounds 11-13 fused (after wrap)
        'rounds_11_13': {
            'xor': 3,
            'hash': 3 * 12,
            'branch': 2 * 3,  # Similar to rounds 0-2
            'selection': 2 * 1 + 2 * 2,
            'idx_update': 2 + 2 + 3,
        },
        # Round 14: gather
        'round_14': {
            'addr': 1,
            'xor': 1,
            'hash': 12,
            'branch': 3,
        },
        # Round 15: final gather (no branch)
        'round_15': {
            'addr': 1,
            'xor': 1,
            'hash': 12,
        },
    }

    # Sum up VALU operations per desk
    valu_per_desk = 0
    for phase, ops in per_desk.items():
        phase_total = sum(ops.values())
        print(f"{phase}: {ops}")
        print(f"  Total: {phase_total} VALU")
        valu_per_desk += phase_total

    print(f"\nTotal VALU per desk: {valu_per_desk}")

    # For 32 desks
    num_desks = 32
    total_valu = valu_per_desk * num_desks
    print(f"Total VALU for {num_desks} desks: {total_valu}")

    # Add setup overhead
    setup_valu = 27  # Broadcasts, diffs, etc.
    total_valu_with_setup = total_valu + setup_valu
    print(f"Total VALU with setup: {total_valu_with_setup}")

    # VALU bound
    valu_bound = (total_valu_with_setup + 5) // 6  # Ceiling division
    print(f"VALU bound: {valu_bound} cycles")

    # Load operations
    loads = {
        'input_idx': 32,  # vloads for idx
        'input_val': 32,  # vloads for val
        'tree_preload': 7,  # scalar loads for tree[0-6]
        'gathers': 10 * 32 * 8,  # 10 gather rounds * 32 desks * 8 lanes
        'setup': 10,  # constants etc
    }
    total_loads = sum(loads.values())
    load_bound = (total_loads + 1) // 2
    print(f"\nTotal LOADs: {total_loads}")
    print(f"LOAD bound: {load_bound} cycles")

    print(f"\nTheoretical minimum: max({valu_bound}, {load_bound}) = {max(valu_bound, load_bound)} cycles")
    print(f"B4-2 achieved: 1558 cycles")
    print(f"Gap: {1558 - max(valu_bound, load_bound)} cycles")


if __name__ == "__main__":
    analyze_b4_2_operations()

    # Also run B4-2 baseline to verify
    print("\n--- Verification Run ---")

    # Import and run B4-2
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from B4_round_fusion.B4_2_full_early_rounds_fusion import do_kernel_test
    do_kernel_test(10, 16, 256, check=True)
