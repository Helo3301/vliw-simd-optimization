"""
Profile H140 kernel to analyze operation counts, slot utilization, and identify bottlenecks.
"""

import sys
import os
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from problem import SLOT_LIMITS, VLEN

# Import the kernel builder
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "H140_h82_combined"))
from perf_takehome_h140 import KernelBuilderH140


def count_operations(slots):
    """Count operations by engine type."""
    counts = defaultdict(int)
    op_counts = defaultdict(lambda: defaultdict(int))

    for engine, slot in slots:
        counts[engine] += 1
        op_name = slot[0] if isinstance(slot, tuple) else str(slot)
        op_counts[engine][op_name] += 1

    return counts, op_counts


def analyze_scheduled_bundles(instrs):
    """Analyze the scheduled instruction bundles."""
    cycle_stats = []
    engine_totals = defaultdict(int)

    for cycle_idx, bundle in enumerate(instrs):
        cycle_usage = {}
        for engine, slots in bundle.items():
            cycle_usage[engine] = len(slots)
            engine_totals[engine] += len(slots)
        cycle_stats.append(cycle_usage)

    return cycle_stats, engine_totals


def calculate_utilization(cycle_stats, total_cycles):
    """Calculate slot utilization statistics."""
    utilization = {}

    for engine, limit in SLOT_LIMITS.items():
        if engine == "debug":
            continue

        slots_used = sum(stat.get(engine, 0) for stat in cycle_stats)
        max_possible = limit * total_cycles
        actual_avg = slots_used / total_cycles if total_cycles > 0 else 0
        utilization[engine] = {
            'total_used': slots_used,
            'max_possible': max_possible,
            'utilization_pct': (slots_used / max_possible * 100) if max_possible > 0 else 0,
            'avg_per_cycle': actual_avg,
            'limit': limit
        }

    return utilization


def find_bottleneck_cycles(cycle_stats, threshold=0.9):
    """Find cycles where an engine is near its limit."""
    bottleneck_cycles = defaultdict(list)

    for cycle_idx, usage in enumerate(cycle_stats):
        for engine, count in usage.items():
            if engine == "debug":
                continue
            limit = SLOT_LIMITS[engine]
            if count >= limit * threshold:
                bottleneck_cycles[engine].append((cycle_idx, count, limit))

    return bottleneck_cycles


def analyze_dependency_chain(slots):
    """Analyze potential dependency chains (simplified)."""
    # This is a simplified analysis - tracks data dependencies
    writes = defaultdict(list)
    reads = defaultdict(list)

    for idx, (engine, slot) in enumerate(slots):
        if engine == "alu":
            _op, dest, a1, a2 = slot
            reads[a1].append(idx)
            reads[a2].append(idx)
            writes[dest].append(idx)
        elif engine == "valu":
            if slot[0] == "vbroadcast":
                _, dest, src = slot
                reads[src].append(idx)
                for i in range(VLEN):
                    writes[dest + i].append(idx)
            elif slot[0] == "multiply_add":
                _, dest, a, b, c = slot
                for i in range(VLEN):
                    reads[a + i].append(idx)
                    reads[b + i].append(idx)
                    reads[c + i].append(idx)
                    writes[dest + i].append(idx)
            else:
                _, dest, a1, a2 = slot
                for i in range(VLEN):
                    reads[a1 + i].append(idx)
                    reads[a2 + i].append(idx)
                    writes[dest + i].append(idx)
        elif engine == "load":
            if slot[0] in ("load", "load_offset"):
                dest = slot[1]
                addr = slot[2]
                reads[addr].append(idx)
                writes[dest].append(idx)
            elif slot[0] == "vload":
                _, dest, addr = slot
                reads[addr].append(idx)
                for i in range(VLEN):
                    writes[dest + i].append(idx)
            elif slot[0] == "const":
                _, dest, _ = slot
                writes[dest].append(idx)
        elif engine == "store":
            if slot[0] == "store":
                _, addr, src = slot
                reads[addr].append(idx)
                reads[src].append(idx)
            elif slot[0] == "vstore":
                _, addr, src = slot
                reads[addr].append(idx)
                for i in range(VLEN):
                    reads[src + i].append(idx)

    return writes, reads


def main():
    print("=" * 80)
    print("H140 VLIW SIMD Kernel Profiling Analysis")
    print("=" * 80)
    print()

    # Build the kernel
    kb = KernelBuilderH140()
    kb.build_kernel(forest_height=10, n_nodes=2047, batch_size=256, rounds=16)

    total_cycles = len(kb.instrs)
    total_slots = len(kb.slots)

    print(f"Total cycles: {total_cycles}")
    print(f"Total operations (slots): {total_slots}")
    print(f"Average slots per cycle: {total_slots / total_cycles:.2f}")
    print()

    # Count operations by engine
    print("=" * 80)
    print("OPERATION COUNTS BY ENGINE")
    print("=" * 80)
    counts, op_counts = count_operations(kb.slots)

    for engine in ['valu', 'load', 'store', 'alu', 'flow']:
        count = counts.get(engine, 0)
        limit = SLOT_LIMITS[engine]
        theoretical_min = count / limit
        print(f"\n{engine.upper()}: {count} ops (limit: {limit}/cycle)")
        print(f"  Theoretical minimum cycles: {theoretical_min:.1f}")
        if op_counts[engine]:
            print("  Breakdown:")
            for op, cnt in sorted(op_counts[engine].items(), key=lambda x: -x[1]):
                print(f"    {op}: {cnt}")

    print()
    print("=" * 80)
    print("THEORETICAL BOUNDS")
    print("=" * 80)

    bounds = {}
    for engine in ['valu', 'load', 'store', 'alu', 'flow']:
        count = counts.get(engine, 0)
        limit = SLOT_LIMITS[engine]
        bound = count / limit
        bounds[engine] = bound
        print(f"{engine.upper()} bound: {count} / {limit} = {bound:.1f} cycles")

    bottleneck = max(bounds.items(), key=lambda x: x[1])
    print(f"\nBottleneck engine: {bottleneck[0].upper()} ({bottleneck[1]:.1f} cycles)")
    print(f"Actual cycles: {total_cycles}")
    print(f"Overhead vs theoretical: {total_cycles - bottleneck[1]:.1f} cycles ({(total_cycles / bottleneck[1] - 1) * 100:.1f}%)")

    print()
    print("=" * 80)
    print("SLOT UTILIZATION ANALYSIS")
    print("=" * 80)

    cycle_stats, engine_totals = analyze_scheduled_bundles(kb.instrs)
    utilization = calculate_utilization(cycle_stats, total_cycles)

    total_slots_possible = 0
    total_slots_used = 0

    for engine in ['valu', 'load', 'store', 'alu', 'flow']:
        u = utilization[engine]
        total_slots_possible += u['max_possible']
        total_slots_used += u['total_used']
        print(f"\n{engine.upper()}:")
        print(f"  Total used: {u['total_used']}")
        print(f"  Max possible: {u['max_possible']} ({u['limit']}/cycle x {total_cycles} cycles)")
        print(f"  Utilization: {u['utilization_pct']:.1f}%")
        print(f"  Average per cycle: {u['avg_per_cycle']:.2f} / {u['limit']}")

    print(f"\nOverall slot utilization: {total_slots_used}/{total_slots_possible} ({total_slots_used/total_slots_possible*100:.1f}%)")

    print()
    print("=" * 80)
    print("BOTTLENECK CYCLE ANALYSIS")
    print("=" * 80)

    bottleneck_cycles = find_bottleneck_cycles(cycle_stats, threshold=0.8)

    for engine in ['valu', 'load', 'store', 'alu', 'flow']:
        cycles = bottleneck_cycles.get(engine, [])
        print(f"\n{engine.upper()}: {len(cycles)} cycles at >=80% capacity")
        if cycles and len(cycles) <= 10:
            for c, count, limit in cycles[:10]:
                print(f"  Cycle {c}: {count}/{limit}")
        elif cycles:
            print(f"  (showing first 10 of {len(cycles)})")
            for c, count, limit in cycles[:10]:
                print(f"  Cycle {c}: {count}/{limit}")

    # Count cycles at full capacity
    print()
    print("=" * 80)
    print("CYCLES AT FULL CAPACITY (100%)")
    print("=" * 80)

    full_capacity_cycles = find_bottleneck_cycles(cycle_stats, threshold=1.0)
    for engine in ['valu', 'load', 'store', 'alu', 'flow']:
        cycles = full_capacity_cycles.get(engine, [])
        print(f"{engine.upper()}: {len(cycles)} cycles at 100% capacity")

    print()
    print("=" * 80)
    print("IDLE SLOT ANALYSIS")
    print("=" * 80)

    for engine in ['valu', 'load', 'store', 'alu', 'flow']:
        u = utilization[engine]
        idle_slots = u['max_possible'] - u['total_used']
        print(f"{engine.upper()}: {idle_slots} idle slots ({idle_slots/total_cycles:.1f}/cycle average)")

    print()
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    valu_util = utilization['valu']['utilization_pct']
    load_util = utilization['load']['utilization_pct']

    print(f"\n1. VALU utilization: {valu_util:.1f}%")
    if valu_util > 90:
        print("   -> VALU is saturated. Reducing VALU ops is critical.")
    elif valu_util > 70:
        print("   -> VALU is moderately utilized. Some room for optimization.")
    else:
        print("   -> VALU has headroom. Consider using VALU instead of other engines.")

    print(f"\n2. LOAD utilization: {load_util:.1f}%")
    if load_util > 90:
        print("   -> LOAD is saturated. Reducing memory access is critical.")
    else:
        print("   -> LOAD has headroom.")

    # Calculate potential savings
    valu_ops = counts.get('valu', 0)
    load_ops = counts.get('load', 0)

    print(f"\n3. Potential cycle savings:")
    print(f"   - Reducing 6 VALU ops = ~1 cycle saved (current: {valu_ops} ops)")
    print(f"   - Reducing 2 LOAD ops = ~1 cycle saved (current: {load_ops} ops)")

    # Target analysis
    target_cycles = 1579
    current_cycles = total_cycles
    cycles_to_save = current_cycles - target_cycles

    print(f"\n4. Target analysis:")
    print(f"   Current: {current_cycles} cycles")
    print(f"   Target: {target_cycles} cycles")
    print(f"   Need to save: {cycles_to_save} cycles")

    if valu_util > load_util:
        valu_ops_to_save = cycles_to_save * 6
        print(f"   -> If VALU-bound: need to eliminate ~{valu_ops_to_save} VALU ops")
    else:
        load_ops_to_save = cycles_to_save * 2
        print(f"   -> If LOAD-bound: need to eliminate ~{load_ops_to_save} LOAD ops")


if __name__ == "__main__":
    main()
