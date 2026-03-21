"""
H59: Analyze optimal schedule for actual H54 kernel

Uses the extracted H54 instructions from h54_instrs.json
"""

import json
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from ortools.sat.python import cp_model


# Constants
SLOT_LIMITS = {
    "alu": 12,
    "valu": 6,
    "load": 2,
    "store": 2,
    "flow": 1,
}

VLEN = 8


@dataclass
class Operation:
    """Represents a single operation."""
    op_id: int
    cycle: int
    engine: str
    slot: list
    dest: Optional[int]
    srcs: List[int]
    latency: int = 1


def parse_slot(engine: str, slot: list) -> Tuple[Optional[int], List[int]]:
    """Parse a slot to extract destination and source addresses."""
    if not slot:
        return None, []

    op = slot[0]

    if engine == 'load':
        if op == 'const':
            return slot[1], []
        elif op == 'load':
            return slot[1], [slot[2]]
        elif op == 'vload':
            return slot[1], [slot[2]]
        return None, []

    elif engine == 'store':
        if op in ['vstore', 'store']:
            return None, [slot[1], slot[2]]
        return None, []

    elif engine == 'valu':
        if op == 'vbroadcast':
            return slot[1], [slot[2]]
        elif op == 'multiply_add':
            return slot[1], [slot[2], slot[3], slot[4]]
        elif len(slot) == 4:
            return slot[1], [slot[2], slot[3]]
        return None, []

    elif engine == 'alu':
        if len(slot) == 4:
            return slot[1], [slot[2], slot[3]]
        return None, []

    elif engine == 'flow':
        if op == 'pause':
            return None, []
        elif op == 'cond_jump':
            return None, [slot[1]]
        elif op == 'select':
            return slot[1], [slot[2], slot[3], slot[4]]
        return None, []

    return None, []


def get_dest_range(dest: Optional[int], engine: str, op_type: str) -> List[int]:
    """Get all destination addresses."""
    if dest is None or dest == -1:
        return []
    if engine == "valu" or (engine == "load" and op_type == "vload"):
        return list(range(dest, dest + VLEN))
    return [dest]


def get_source_ranges(srcs: List[int], engine: str) -> List[int]:
    """Get all source addresses."""
    addrs = []
    for src in srcs:
        if engine == "valu":
            addrs.extend(range(src, src + VLEN))
        else:
            addrs.append(src)
    return addrs


def extract_operations(instrs: List[dict], start_cycle: int, end_cycle: int) -> Tuple[List[Operation], List[Tuple[int, int]]]:
    """Extract operations and dependencies."""
    operations = []
    dependencies = []
    last_def: Dict[int, int] = {}

    for cycle_idx in range(start_cycle, end_cycle):
        instr = instrs[cycle_idx]
        for engine, slots in instr.items():
            if engine == 'debug':
                continue
            for slot in slots:
                op_id = len(operations)
                dest, srcs = parse_slot(engine, slot)
                op_type = slot[0] if slot else None

                op = Operation(
                    op_id=op_id,
                    cycle=cycle_idx - start_cycle,
                    engine=engine,
                    slot=slot,
                    dest=dest,
                    srcs=srcs,
                    latency=1
                )
                operations.append(op)

                # Dependencies
                src_addrs = get_source_ranges(srcs, engine)
                for addr in src_addrs:
                    if addr in last_def:
                        writer_id = last_def[addr]
                        if writer_id != op_id:
                            dependencies.append((writer_id, op_id))

                # Update last_def
                dest_addrs = get_dest_range(dest, engine, op_type)
                for addr in dest_addrs:
                    last_def[addr] = op_id

    dependencies = list(set(dependencies))
    return operations, dependencies


def build_cpsat_model(
    operations: List[Operation],
    dependencies: List[Tuple[int, int]],
    horizon: int
) -> Tuple[cp_model.CpModel, Dict[int, any], any]:
    """Build CP-SAT model."""
    model = cp_model.CpModel()

    starts = {}
    ends = {}
    intervals = {}

    for op in operations:
        starts[op.op_id] = model.NewIntVar(0, horizon, f'start_{op.op_id}')
        ends[op.op_id] = model.NewIntVar(0, horizon + 1, f'end_{op.op_id}')
        model.Add(ends[op.op_id] == starts[op.op_id] + op.latency)
        intervals[op.op_id] = model.NewIntervalVar(
            starts[op.op_id], op.latency, ends[op.op_id], f'interval_{op.op_id}'
        )

    makespan = model.NewIntVar(0, horizon + 1, 'makespan')

    # Dependency constraints
    for src_id, dst_id in dependencies:
        model.Add(starts[dst_id] >= ends[src_id])

    # Resource constraints
    for engine, limit in SLOT_LIMITS.items():
        engine_intervals = [intervals[op.op_id] for op in operations if op.engine == engine]
        if engine_intervals:
            model.AddCumulative(engine_intervals, [1] * len(engine_intervals), limit)

    # Makespan
    for op in operations:
        model.Add(makespan >= ends[op.op_id])

    model.Minimize(makespan)
    return model, starts, makespan


def count_by_engine(operations: List[Operation]) -> Dict[str, int]:
    """Count operations by engine."""
    counts = defaultdict(int)
    for op in operations:
        counts[op.engine] += 1
    return dict(counts)


def compute_theoretical_min(operations: List[Operation]) -> int:
    """Compute theoretical minimum cycles."""
    counts = count_by_engine(operations)
    mins = {}
    for engine, limit in SLOT_LIMITS.items():
        if engine in counts:
            mins[engine] = (counts[engine] + limit - 1) // limit
        else:
            mins[engine] = 0
    return max(mins.values()) if mins else 0


def main():
    print("=" * 70)
    print("H59: Optimal Schedule Analysis for H54 Kernel")
    print("=" * 70)

    # Load extracted instructions
    print("\n[Step 1] Loading H54 instructions...")
    with open('/home/hestiasadmin/projects/original_performance_takehome/experiments/H59_ilp_schedule/h54_instrs.json', 'r') as f:
        data = json.load(f)

    instrs = data['instructions']
    loop_start = data['loop_start']
    loop_end = data['loop_end']
    loop_cycles = data['loop_cycles']

    print(f"  Total instructions: {len(instrs)}")
    print(f"  Main loop: cycles {loop_start} to {loop_end}")
    print(f"  Main loop size: {loop_cycles} cycles")

    # Extract operations
    print("\n[Step 2] Extracting operations...")
    operations, dependencies = extract_operations(instrs, loop_start, loop_end)
    print(f"  Operations: {len(operations)}")
    print(f"  Dependencies: {len(dependencies)}")

    # Analyze current schedule
    print("\n[Step 3] Analyzing current schedule...")
    counts = count_by_engine(operations)
    print(f"\n  Operation counts by engine:")
    for engine, count in sorted(counts.items()):
        print(f"    {engine:6s}: {count:4d} ops")

    print(f"\n  Current makespan: {loop_cycles} cycles")

    utilization = {
        engine: counts.get(engine, 0) / (loop_cycles * SLOT_LIMITS[engine]) * 100
        for engine in SLOT_LIMITS
    }
    print(f"\n  Resource utilization:")
    for engine in SLOT_LIMITS:
        print(f"    {engine:6s}: {utilization[engine]:5.1f}%")

    # Theoretical minimum
    print("\n[Step 4] Computing theoretical minimum...")
    theoretical_min = compute_theoretical_min(operations)
    print(f"  Theoretical minimum (resource-bound): {theoretical_min} cycles")

    # Solve
    print("\n[Step 5] Solving for optimal schedule...")
    horizon = loop_cycles + 20
    print(f"  Building CP-SAT model with {len(operations)} ops, {len(dependencies)} deps")
    print(f"  Horizon: {horizon}")

    model, starts, makespan = build_cpsat_model(operations, dependencies, horizon)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 300
    solver.parameters.num_search_workers = 8

    print("  Solving with 5-minute time limit...")
    start_time = time.time()
    status = solver.Solve(model)
    solve_time = time.time() - start_time

    print(f"\n  Solver status: {solver.StatusName(status)}")
    print(f"  Solve time: {solve_time:.2f}s")

    results = {
        'current_makespan': loop_cycles,
        'theoretical_min': theoretical_min,
        'operations': len(operations),
        'dependencies': len(dependencies),
        'counts': counts,
        'utilization': utilization,
        'solver_status': solver.StatusName(status),
        'solve_time': solve_time,
    }

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        optimal = solver.Value(makespan)
        print(f"\n  Results:")
        print(f"    Current H54 makespan:  {loop_cycles} cycles")
        print(f"    Optimal makespan:      {optimal} cycles")
        print(f"    Theoretical minimum:   {theoretical_min} cycles")

        improvement = (loop_cycles - optimal) / loop_cycles * 100
        print(f"\n    Improvement potential: {loop_cycles - optimal} cycles ({improvement:.1f}%)")

        gap = (optimal - theoretical_min) / theoretical_min * 100 if theoretical_min > 0 else 0
        print(f"    Gap from theory:       {optimal - theoretical_min} cycles ({gap:.1f}%)")

        results['optimal_makespan'] = optimal
        results['improvement_cycles'] = loop_cycles - optimal
        results['improvement_pct'] = improvement
        results['gap_to_theory_cycles'] = optimal - theoretical_min
        results['gap_to_theory_pct'] = gap

        # Analyze schedule differences
        schedule = {op.op_id: solver.Value(starts[op.op_id]) for op in operations}

        moved_earlier = [(op, schedule[op.op_id] - op.cycle) for op in operations if schedule[op.op_id] < op.cycle]
        moved_later = [(op, schedule[op.op_id] - op.cycle) for op in operations if schedule[op.op_id] > op.cycle]
        unchanged = [op for op in operations if schedule[op.op_id] == op.cycle]

        print(f"\n  Schedule analysis:")
        print(f"    Operations moved earlier: {len(moved_earlier)}")
        print(f"    Operations moved later:   {len(moved_later)}")
        print(f"    Operations unchanged:     {len(unchanged)}")

        # Movement by engine
        print("\n  Average movement by engine:")
        for engine in SLOT_LIMITS:
            engine_ops = [(op, schedule[op.op_id] - op.cycle) for op in operations if op.engine == engine]
            if engine_ops:
                avg = sum(diff for _, diff in engine_ops) / len(engine_ops)
                print(f"    {engine:6s}: {avg:+.1f} cycles")

    else:
        print("  No feasible solution found.")

    # Save results
    with open('/home/hestiasadmin/projects/original_performance_takehome/experiments/H59_ilp_schedule/results_h54.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to results_h54.json")

    # Project total kernel improvement
    print("\n" + "=" * 70)
    print("TOTAL KERNEL PROJECTION")
    print("=" * 70)

    total_h54 = 3462
    iterations = 16
    setup_overhead = total_h54 - (iterations * loop_cycles)

    print(f"\n  H54 total cycles: {total_h54}")
    print(f"  Iterations: {iterations}")
    print(f"  Loop cycles/iter: {loop_cycles}")
    print(f"  Setup overhead: {setup_overhead} cycles")

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        optimal_total = iterations * optimal + setup_overhead
        total_improvement = total_h54 - optimal_total
        total_improvement_pct = total_improvement / total_h54 * 100

        print(f"\n  Projected optimal total: {optimal_total} cycles")
        print(f"  Projected improvement: {total_improvement} cycles ({total_improvement_pct:.1f}%)")
        print(f"  Theoretical floor: {iterations * theoretical_min + setup_overhead} cycles")

        results['total_h54'] = total_h54
        results['projected_optimal_total'] = optimal_total
        results['total_improvement_cycles'] = total_improvement
        results['total_improvement_pct'] = total_improvement_pct

        # Save updated results
        with open('/home/hestiasadmin/projects/original_performance_takehome/experiments/H59_ilp_schedule/results_h54.json', 'w') as f:
            json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    results = main()
