"""
Analyze the instruction schedule to understand dependency chains and stalls.
"""

import sys
import os
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from problem import SLOT_LIMITS, VLEN

# Import the kernel builder
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "H140_h82_combined"))
from perf_takehome_h140 import KernelBuilderH140


def analyze_cycle_distribution(instrs):
    """Analyze how operations are distributed across cycles."""
    print("=" * 80)
    print("CYCLE DISTRIBUTION ANALYSIS")
    print("=" * 80)
    print()

    # Track slot usage per cycle
    valu_dist = defaultdict(int)
    load_dist = defaultdict(int)
    combined_dist = defaultdict(int)

    for cycle, bundle in enumerate(instrs):
        valu_count = len(bundle.get('valu', []))
        load_count = len(bundle.get('load', []))

        valu_dist[valu_count] += 1
        load_dist[load_count] += 1
        combined_dist[(valu_count, load_count)] += 1

    print("VALU slot distribution:")
    for slots in range(7):
        count = valu_dist[slots]
        pct = count / len(instrs) * 100 if len(instrs) > 0 else 0
        bar = "#" * int(pct / 2)
        print(f"  {slots} slots: {count:4d} cycles ({pct:5.1f}%) {bar}")

    print()
    print("LOAD slot distribution:")
    for slots in range(3):
        count = load_dist[slots]
        pct = count / len(instrs) * 100 if len(instrs) > 0 else 0
        bar = "#" * int(pct / 2)
        print(f"  {slots} slots: {count:4d} cycles ({pct:5.1f}%) {bar}")

    print()
    print("Combined (VALU, LOAD) distribution (top 10):")
    sorted_combined = sorted(combined_dist.items(), key=lambda x: -x[1])[:10]
    for (valu, load), count in sorted_combined:
        pct = count / len(instrs) * 100 if len(instrs) > 0 else 0
        print(f"  VALU={valu}, LOAD={load}: {count:4d} cycles ({pct:5.1f}%)")


def analyze_stall_patterns(instrs):
    """Analyze patterns that might indicate stalls."""
    print()
    print("=" * 80)
    print("POTENTIAL STALL PATTERN ANALYSIS")
    print("=" * 80)
    print()

    # Look for cycles where we're below capacity on both VALU and LOAD
    # This indicates dependency stalls
    stall_candidates = []

    for cycle, bundle in enumerate(instrs):
        valu_count = len(bundle.get('valu', []))
        load_count = len(bundle.get('load', []))
        alu_count = len(bundle.get('alu', []))
        store_count = len(bundle.get('store', []))
        flow_count = len(bundle.get('flow', []))

        total_slots = valu_count + load_count + alu_count + store_count + flow_count

        # Potential stall: neither engine at capacity
        if valu_count < 6 and load_count < 2 and total_slots < 6:
            stall_candidates.append((cycle, valu_count, load_count, total_slots))

    print(f"Cycles with potential dependency stalls: {len(stall_candidates)}")
    print("(Cycles where VALU<6, LOAD<2, and total<6)")
    print()

    if stall_candidates:
        print("First 20 stall candidates:")
        for cycle, valu, load, total in stall_candidates[:20]:
            print(f"  Cycle {cycle:4d}: VALU={valu}, LOAD={load}, total={total}")


def analyze_phase_boundaries(instrs):
    """Analyze phases separated by pause instructions."""
    print()
    print("=" * 80)
    print("PHASE ANALYSIS"  )
    print("=" * 80)
    print()

    phases = []
    current_phase_start = 0

    for cycle, bundle in enumerate(instrs):
        if 'flow' in bundle:
            for slot in bundle['flow']:
                if slot[0] == 'pause':
                    phases.append((current_phase_start, cycle))
                    current_phase_start = cycle + 1

    # Don't forget the last phase
    if current_phase_start < len(instrs):
        phases.append((current_phase_start, len(instrs) - 1))

    print(f"Number of phases: {len(phases)}")
    for i, (start, end) in enumerate(phases):
        duration = end - start + 1
        print(f"  Phase {i}: cycles {start}-{end} ({duration} cycles)")


def analyze_load_gather_patterns(instrs):
    """Analyze the gather (scalar load) patterns."""
    print()
    print("=" * 80)
    print("GATHER PATTERN ANALYSIS")
    print("=" * 80)
    print()

    # Count cycles with scalar loads (gathers)
    gather_cycles = []

    for cycle, bundle in enumerate(instrs):
        if 'load' in bundle:
            scalar_loads = sum(1 for slot in bundle['load']
                             if slot[0] == 'load')
            if scalar_loads > 0:
                gather_cycles.append((cycle, scalar_loads))

    print(f"Cycles with scalar loads (gathers): {len(gather_cycles)}")

    # Distribution of scalar loads per cycle
    load_dist = defaultdict(int)
    for cycle, count in gather_cycles:
        load_dist[count] += 1

    print("Scalar loads per cycle distribution:")
    for count, cycles in sorted(load_dist.items()):
        print(f"  {count} loads: {cycles} cycles")

    # Analyze consecutive gather patterns
    consecutive_runs = []
    current_run = 0
    run_start = None

    for cycle in range(len(instrs)):
        bundle = instrs[cycle]
        has_scalar_load = 'load' in bundle and any(
            slot[0] == 'load' for slot in bundle['load'])

        if has_scalar_load:
            if current_run == 0:
                run_start = cycle
            current_run += 1
        else:
            if current_run > 0:
                consecutive_runs.append((run_start, current_run))
            current_run = 0

    if current_run > 0:
        consecutive_runs.append((run_start, current_run))

    print(f"\nConsecutive gather runs: {len(consecutive_runs)}")
    run_lengths = [r[1] for r in consecutive_runs]
    if run_lengths:
        print(f"  Average length: {sum(run_lengths)/len(run_lengths):.1f}")
        print(f"  Max length: {max(run_lengths)}")
        print(f"  Total cycles with gathers: {sum(run_lengths)}")


def analyze_valu_saturation(instrs):
    """Analyze VALU saturation patterns."""
    print()
    print("=" * 80)
    print("VALU SATURATION ANALYSIS")
    print("=" * 80)
    print()

    # Find runs of VALU saturation
    saturated_runs = []
    current_run = 0
    run_start = None

    for cycle in range(len(instrs)):
        bundle = instrs[cycle]
        valu_count = len(bundle.get('valu', []))

        if valu_count == 6:
            if current_run == 0:
                run_start = cycle
            current_run += 1
        else:
            if current_run > 0:
                saturated_runs.append((run_start, current_run))
            current_run = 0

    if current_run > 0:
        saturated_runs.append((run_start, current_run))

    print(f"Consecutive VALU saturation runs: {len(saturated_runs)}")
    run_lengths = [r[1] for r in saturated_runs]
    if run_lengths:
        print(f"  Average length: {sum(run_lengths)/len(run_lengths):.1f}")
        print(f"  Max length: {max(run_lengths)}")
        print(f"  Total saturated cycles: {sum(run_lengths)}")

    # Show longest runs
    if saturated_runs:
        sorted_runs = sorted(saturated_runs, key=lambda x: -x[1])[:10]
        print("\nLongest VALU saturation runs:")
        for start, length in sorted_runs:
            print(f"  Cycles {start}-{start+length-1}: {length} cycles")


def main():
    print("=" * 80)
    print("H140 Instruction Schedule Analysis")
    print("=" * 80)
    print()

    # Build the kernel
    kb = KernelBuilderH140()
    kb.build_kernel(forest_height=10, n_nodes=2047, batch_size=256, rounds=16)

    total_cycles = len(kb.instrs)
    print(f"Total cycles: {total_cycles}")
    print()

    analyze_cycle_distribution(kb.instrs)
    analyze_stall_patterns(kb.instrs)
    analyze_phase_boundaries(kb.instrs)
    analyze_load_gather_patterns(kb.instrs)
    analyze_valu_saturation(kb.instrs)


if __name__ == "__main__":
    main()
