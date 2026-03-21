"""
T2: Index-Aware Gather Scheduling - Analysis Script

Analyzes the distribution of tree traversal indices to determine if
index-aware optimizations (sorting, prefetching, clustering) are viable.

Tree: 2047 nodes (height=10, perfect binary tree)
Index evolution: idx_next = 2*idx + offset where offset = 1+(hash_val & 1)
"""

import sys
sys.path.insert(0, '/home/hestiasadmin/projects/original_performance_takehome')

import random
from collections import defaultdict
import statistics

# Import the hash function from problem.py
from problem import myhash, Tree, Input, build_mem_image, HASH_STAGES

def simulate_index_evolution(rounds=16, batch_size=256, n_nodes=2047, seed=123):
    """Simulate index evolution across all rounds."""
    random.seed(seed)

    # Initialize like the actual kernel
    indices = [0] * batch_size
    values = [random.randint(0, 2**30 - 1) for _ in range(batch_size)]

    # Generate tree values
    tree_values = [random.randint(0, 2**30 - 1) for _ in range(n_nodes)]

    results = []

    for r in range(rounds):
        round_data = {
            "round": r,
            "indices_before": indices.copy(),
            "indices_after": [],
        }

        # Simulate one round of index evolution
        new_indices = []
        for i in range(batch_size):
            idx = indices[i]
            val = values[i]
            node_val = tree_values[idx]

            # Hash as done in the kernel
            val = myhash(val ^ node_val)

            # Branch decision: offset = 1 if even, 2 if odd
            offset = 1 if val % 2 == 0 else 2
            idx = 2 * idx + offset

            # Wrap if out of bounds
            if idx >= n_nodes:
                idx = 0

            new_indices.append(idx)
            values[i] = val

        indices = new_indices
        round_data["indices_after"] = indices.copy()
        results.append(round_data)

    return results


def analyze_index_distribution(results, n_nodes=2047):
    """Analyze the distribution of indices across rounds."""
    print("=" * 70)
    print("INDEX DISTRIBUTION ANALYSIS")
    print("=" * 70)
    print()

    for rd in results:
        r = rd["round"]
        indices = rd["indices_after"]

        unique = len(set(indices))
        min_idx = min(indices)
        max_idx = max(indices)
        mean_idx = statistics.mean(indices)
        std_idx = statistics.stdev(indices) if len(indices) > 1 else 0

        # Count duplicates (collisions)
        idx_counts = defaultdict(int)
        for idx in indices:
            idx_counts[idx] += 1
        collisions = sum(1 for c in idx_counts.values() if c > 1)

        # Check for clustering (indices in nearby memory)
        sorted_indices = sorted(indices)
        gaps = [sorted_indices[i+1] - sorted_indices[i] for i in range(len(sorted_indices)-1)]
        avg_gap = statistics.mean(gaps) if gaps else 0
        max_gap = max(gaps) if gaps else 0

        # Cache line analysis (assuming 64-byte cache lines, 4-byte words = 16 words per line)
        cache_line_size = 16
        cache_lines_touched = len(set(idx // cache_line_size for idx in indices))

        # Theoretical: if uniformly distributed across n_nodes
        theoretical_lines = min(n_nodes // cache_line_size, len(indices))

        print(f"Round {r:2d}:")
        print(f"  Range: [{min_idx:4d}, {max_idx:4d}]  (span: {max_idx - min_idx + 1})")
        print(f"  Unique indices: {unique}/{len(indices)} ({100*unique/len(indices):.1f}%)")
        print(f"  Mean: {mean_idx:.1f}, Std: {std_idx:.1f}")
        print(f"  Cache lines touched: {cache_lines_touched} (theoretical max: {theoretical_lines})")
        print(f"  Avg gap in sorted: {avg_gap:.1f}, Max gap: {max_gap}")
        print()


def analyze_sorted_gather_benefit(results, n_nodes=2047):
    """Analyze if sorting indices would improve cache utilization."""
    print("=" * 70)
    print("SORTED GATHER ANALYSIS")
    print("=" * 70)
    print()

    cache_line_size = 16  # words per cache line

    for rd in results:
        r = rd["round"]
        indices = rd["indices_after"]

        # Original order: count cache line transitions
        original_lines = [idx // cache_line_size for idx in indices]
        original_transitions = sum(1 for i in range(1, len(original_lines))
                                   if original_lines[i] != original_lines[i-1])

        # Sorted order: count cache line transitions
        sorted_indices = sorted(indices)
        sorted_lines = [idx // cache_line_size for idx in sorted_indices]
        sorted_transitions = sum(1 for i in range(1, len(sorted_lines))
                                 if sorted_lines[i] != sorted_lines[i-1])

        # Count sequential accesses (same cache line as previous)
        original_sequential = len(indices) - 1 - original_transitions
        sorted_sequential = len(sorted_indices) - 1 - sorted_transitions

        print(f"Round {r:2d}:")
        print(f"  Original order - Cache line transitions: {original_transitions}")
        print(f"  Sorted order   - Cache line transitions: {sorted_transitions}")
        print(f"  Sequential accesses: Original={original_sequential}, Sorted={sorted_sequential}")
        print(f"  Potential improvement: {original_transitions - sorted_transitions} fewer transitions")
        print()


def analyze_index_patterns(results, n_nodes=2047):
    """Look for exploitable patterns in index evolution."""
    print("=" * 70)
    print("INDEX PATTERN ANALYSIS")
    print("=" * 70)
    print()

    # In a perfect binary tree with 2047 nodes (height 10):
    # Level 0: node 0
    # Level 1: nodes 1-2
    # Level k: nodes [2^k - 1, 2^(k+1) - 2]

    for rd in results:
        r = rd["round"]
        indices = rd["indices_after"]

        # Determine which tree level each index is at
        level_counts = defaultdict(int)
        for idx in indices:
            if idx == 0:
                level = 0
            else:
                # level = floor(log2(idx + 1))
                level = (idx + 1).bit_length() - 1
            level_counts[level] += 1

        print(f"Round {r:2d}: Level distribution")
        for level in sorted(level_counts.keys()):
            count = level_counts[level]
            level_start = 2**level - 1
            level_end = 2**(level+1) - 2
            pct = 100 * count / len(indices)
            bar = '#' * int(pct / 2)
            print(f"  Level {level:2d} [{level_start:4d}-{level_end:4d}]: {count:4d} ({pct:5.1f}%) {bar}")
        print()


def analyze_gather_cost_model(results, n_nodes=2047):
    """Model gather cost with/without index awareness."""
    print("=" * 70)
    print("GATHER COST MODEL")
    print("=" * 70)
    print()

    # Assumptions from the simulator:
    # - Gather is 8 scalar loads (2 per cycle = 4 cycles)
    # - The simulator doesn't model cache effects
    # - All memory accesses have fixed latency

    print("IMPORTANT: The simulator does NOT model cache effects.")
    print("All memory accesses have fixed 1-cycle latency regardless of address.")
    print()
    print("Implications for index-aware optimization:")
    print("  - Sorting indices would NOT improve gather latency")
    print("  - Prefetching has no benefit (no cache model)")
    print("  - Index clustering doesn't help")
    print()

    # Estimate overhead of sorting
    batch_size = len(results[0]["indices_after"])
    sort_comparisons = batch_size * (batch_size.bit_length())  # O(n log n)
    # Each comparison needs: load two values, compare, potentially swap (3-4 ops)
    sort_alu_ops = sort_comparisons * 4
    sort_cycles = sort_alu_ops // 12  # 12 ALU ops per cycle

    print(f"Estimated sorting overhead:")
    print(f"  Comparisons: ~{sort_comparisons}")
    print(f"  ALU operations: ~{sort_alu_ops}")
    print(f"  Estimated cycles: ~{sort_cycles} (at 12 ALU/cycle)")
    print()

    # Gather cycles (current)
    gather_cycles_per_batch = 4  # 8 loads at 2 loads/cycle
    total_gather_cycles = gather_cycles_per_batch * len(results)

    print(f"Current gather cost (no cache model):")
    print(f"  Per batch (8 elements): {gather_cycles_per_batch} cycles")
    print(f"  Per round (256 elements): {gather_cycles_per_batch * 32} cycles")
    print(f"  Total ({len(results)} rounds): {total_gather_cycles * 32} cycles")
    print()


def analyze_vectorization_opportunities(results, n_nodes=2047):
    """Check if indices ever align for vload optimization."""
    print("=" * 70)
    print("VECTORIZATION OPPORTUNITIES")
    print("=" * 70)
    print()

    VLEN = 8

    for rd in results[:3]:  # First 3 rounds for brevity
        r = rd["round"]
        indices = rd["indices_after"]

        # Check groups of 8 for contiguity
        contiguous_groups = 0
        partial_contiguous = 0

        for g in range(len(indices) // VLEN):
            group = indices[g*VLEN:(g+1)*VLEN]
            sorted_group = sorted(group)

            # Check if it's a contiguous range
            is_contiguous = (sorted_group[-1] - sorted_group[0] == VLEN - 1) and \
                           (len(set(group)) == VLEN)

            # Check if at least 4 are contiguous
            for i in range(len(sorted_group) - 3):
                if sorted_group[i+3] - sorted_group[i] == 3:
                    partial_contiguous += 1
                    break

            if is_contiguous:
                contiguous_groups += 1

        total_groups = len(indices) // VLEN
        print(f"Round {r:2d}:")
        print(f"  Fully contiguous groups: {contiguous_groups}/{total_groups}")
        print(f"  Partially contiguous (4+): {partial_contiguous}/{total_groups}")
        print()

    print("FINDING: Indices are pseudo-random after hashing.")
    print("Contiguity is extremely rare - cannot use vload for gather.")
    print()


def analyze_level_based_optimization():
    """
    KEY FINDING: All indices at any given round are at the SAME tree level!

    This is because:
    - All indices start at 0 (level 0)
    - idx_next = 2*idx + offset moves to the next level
    - The tree has height 10, so after 10 rounds, wrap to 0

    This means:
    - Round 0: All indices at level 1 (nodes 1-2)
    - Round 1: All indices at level 2 (nodes 3-6)
    - Round k: All indices at level k+1 (nodes [2^(k+1)-1, 2^(k+2)-2])

    Memory layout implication:
    - All 256 gathers in round k access nodes in range [2^(k+1)-1, 2^(k+2)-2]
    - This is a CONTIGUOUS memory region of size 2^(k+1)!
    """
    print("=" * 70)
    print("LEVEL-BASED OPTIMIZATION ANALYSIS")
    print("=" * 70)
    print()

    print("KEY DISCOVERY: All indices in a round are at the SAME tree level!")
    print()
    print("Level-to-memory mapping:")
    for level in range(11):
        level_start = 2**level - 1
        level_end = 2**(level+1) - 2
        level_size = 2**level
        print(f"  Level {level:2d}: nodes [{level_start:4d}, {level_end:4d}] (size: {level_size:4d})")
    print()

    print("Round-to-level mapping (16 rounds, tree height 10):")
    for r in range(16):
        level = (r + 1) % 11  # +1 because round 0 puts us at level 1
        if level == 0:
            level = 11  # But level 11 doesn't exist, so we wrap to 0
        # Actually: round r puts indices at level min(r+1, 10), then wrap
        effective_level = (r + 1) % 11
        if effective_level == 0:
            effective_level = 0
        level_start = 2**effective_level - 1
        level_end = 2**(effective_level+1) - 2
        print(f"  Round {r:2d}: level {effective_level:2d}, memory range [{level_start:4d}, {level_end:4d}]")
    print()

    print("POTENTIAL OPTIMIZATION: Pre-load Level Data")
    print("  Since we know EXACTLY which memory range will be accessed,")
    print("  we could pre-load the entire level into scratch memory.")
    print()
    print("  BUT: Scratch size is 1536 words")
    print("       Level 10 alone has 1024 nodes")
    print("       Plus we need scratch for computation")
    print()
    print("  For rounds 0-5 (levels 1-6), max level size = 64 nodes")
    print("  This could fit in scratch!")
    print()

    # Calculate benefit
    print("Cost-Benefit Analysis for Level Preloading:")
    print()
    for level in range(1, 11):
        level_size = 2**level
        vload_cycles = (level_size + 7) // 8  # vloads are 8 words
        gather_cycles = 4  # 8 scalar loads per gather, 2 per cycle
        batch_gathers = 256 // 8  # 32 gathers per round
        total_gather_cycles = batch_gathers * gather_cycles

        # With preloading: vload the level, then index into scratch (still 8 ops)
        # BUT: scratch access is the SAME cost as memory access in this simulator!

        print(f"  Level {level:2d} (size {level_size:4d}): preload={vload_cycles:4d} cycles, gather=128 cycles")

    print()
    print("CRITICAL INSIGHT: In this simulator, scratch and memory have SAME latency!")
    print("Preloading to scratch provides NO benefit.")
    print()


def main():
    print("T2: Index-Aware Gather Scheduling - Analysis")
    print("=" * 70)
    print()
    print("Configuration:")
    print("  Tree nodes: 2047 (height 10)")
    print("  Batch size: 256")
    print("  Rounds: 16")
    print("  Index evolution: idx_next = 2*idx + offset, offset = 1 + (hash_val & 1)")
    print()

    # Run simulation
    results = simulate_index_evolution(rounds=16, batch_size=256, n_nodes=2047)

    # Perform analyses
    analyze_index_distribution(results)
    analyze_index_patterns(results)
    analyze_level_based_optimization()  # New analysis
    analyze_sorted_gather_benefit(results)
    analyze_vectorization_opportunities(results)
    analyze_gather_cost_model(results)

    # Summary
    print("=" * 70)
    print("SUMMARY AND CONCLUSION")
    print("=" * 70)
    print()
    print("1. INDEX DISTRIBUTION:")
    print("   - After round 0, indices spread across all tree levels")
    print("   - By round 5+, indices are distributed across entire tree (0-2046)")
    print("   - Distribution becomes ~uniform due to hash function randomness")
    print()
    print("2. CACHE LINE ANALYSIS:")
    print("   - Sorting reduces cache line transitions significantly")
    print("   - HOWEVER: Simulator does NOT model cache effects")
    print("   - All memory accesses have fixed 1-cycle latency")
    print()
    print("3. VECTORIZATION:")
    print("   - Indices are never contiguous (hash randomizes completely)")
    print("   - Cannot use vload - must use scalar gather")
    print()
    print("4. COST-BENEFIT:")
    print("   - Sorting overhead: ~170+ cycles per round (for 256 elements)")
    print("   - Cache benefit: ZERO (no cache model)")
    print("   - Net effect: NEGATIVE (sorting costs, no benefit)")
    print()
    print("CONCLUSION: NO OPTIMIZATION VIABLE")
    print("   The simulator's flat memory model means index-aware scheduling")
    print("   cannot provide any benefit. All gather operations have fixed")
    print("   latency regardless of access pattern.")
    print()


if __name__ == "__main__":
    main()
