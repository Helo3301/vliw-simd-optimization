"""
T3: Constraint-Based Optimal Scheduling - Optimal Scheduler using OR-Tools

This script models the VLIW scheduling problem as a Constraint Programming (CP) problem
and solves for the optimal schedule.

Variables:
- cycle[op]: When operation op executes

Constraints:
- Dependencies: cycle[b] >= cycle[a] + 1 for all (a -> b) dependencies
- Resources: For each cycle c and slot type t, sum of ops using t at cycle c <= capacity[t]

Objective: Minimize makespan (max cycle across all operations)
"""

import sys
sys.path.insert(0, '/home/hestiasadmin/projects/original_performance_takehome')

import json
from collections import defaultdict
from ortools.sat.python import cp_model

from problem import SLOT_LIMITS


def load_analysis_data():
    """Load the analysis data from the previous step."""
    with open("/home/hestiasadmin/projects/original_performance_takehome/experiments/T3_constraint_solver/analysis_data.json", "r") as f:
        return json.load(f)


def solve_optimal_schedule(ops, deps, max_cycles=None, time_limit=60):
    """
    Solve for optimal schedule using CP-SAT solver.

    Args:
        ops: List of operation dicts with id, engine, etc.
        deps: Dict mapping op_id -> list of (dependent_op_id, latency)
        max_cycles: Upper bound on makespan (default: 2x current)
        time_limit: Time limit in seconds

    Returns:
        (makespan, cycle_assignments) or (None, None) if no solution
    """
    model = cp_model.CpModel()

    n_ops = len(ops)
    if max_cycles is None:
        max_cycles = n_ops * 2  # Conservative upper bound

    # Variables: cycle[op_id] = when operation executes
    cycle = {}
    for op in ops:
        cycle[op["id"]] = model.NewIntVar(0, max_cycles - 1, f'cycle_{op["id"]}')

    # Makespan variable
    makespan = model.NewIntVar(0, max_cycles - 1, 'makespan')

    # Dependency constraints
    for src_id_str, dep_list in deps.items():
        src_id = int(src_id_str)
        for dst_id, latency in dep_list:
            if src_id < n_ops and dst_id < n_ops:
                model.Add(cycle[dst_id] >= cycle[src_id] + latency)

    # Makespan definition
    for op in ops:
        model.Add(makespan >= cycle[op["id"]])

    # Resource constraints: For each cycle and engine type, limit concurrent ops
    # We need to express: at most SLOT_LIMITS[engine] ops of type engine at each cycle

    # Group operations by engine
    ops_by_engine = defaultdict(list)
    for op in ops:
        ops_by_engine[op["engine"]].append(op["id"])

    # For each engine and each possible cycle, create constraint
    for engine, op_ids in ops_by_engine.items():
        if engine == "debug":
            continue
        limit = SLOT_LIMITS[engine]

        for c in range(max_cycles):
            # at_cycle[op_id] is 1 if op_id is scheduled at cycle c
            at_cycle_vars = []
            for op_id in op_ids:
                at_cycle = model.NewBoolVar(f'op{op_id}_at_cycle{c}')
                model.Add(cycle[op_id] == c).OnlyEnforceIf(at_cycle)
                model.Add(cycle[op_id] != c).OnlyEnforceIf(at_cycle.Not())
                at_cycle_vars.append(at_cycle)

            # Sum of at_cycle vars must be <= limit
            model.Add(sum(at_cycle_vars) <= limit)

    # Objective: minimize makespan
    model.Minimize(makespan)

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = 4

    print(f"Solving with {n_ops} operations, max_cycles={max_cycles}...")
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL:
        print(f"OPTIMAL solution found!")
        result_makespan = solver.Value(makespan) + 1  # +1 because cycles are 0-indexed
        cycle_assignments = {op["id"]: solver.Value(cycle[op["id"]]) for op in ops}
        return result_makespan, cycle_assignments
    elif status == cp_model.FEASIBLE:
        print(f"Feasible (but not proven optimal) solution found.")
        result_makespan = solver.Value(makespan) + 1
        cycle_assignments = {op["id"]: solver.Value(cycle[op["id"]]) for op in ops}
        return result_makespan, cycle_assignments
    else:
        print(f"No solution found. Status: {solver.StatusName(status)}")
        return None, None


def solve_with_binary_search(ops, deps, lower_bound, upper_bound, time_limit_per=30):
    """
    Use binary search to find minimum feasible makespan.
    This can be faster than letting the solver optimize directly.
    """
    model_cache = {}

    def is_feasible(target_makespan):
        """Check if a schedule with given makespan is feasible."""
        model = cp_model.CpModel()

        n_ops = len(ops)

        # Variables
        cycle = {}
        for op in ops:
            cycle[op["id"]] = model.NewIntVar(0, target_makespan - 1, f'cycle_{op["id"]}')

        # Dependency constraints
        for src_id_str, dep_list in deps.items():
            src_id = int(src_id_str)
            for dst_id, latency in dep_list:
                if src_id < n_ops and dst_id < n_ops:
                    model.Add(cycle[dst_id] >= cycle[src_id] + latency)

        # Resource constraints
        ops_by_engine = defaultdict(list)
        for op in ops:
            ops_by_engine[op["engine"]].append(op["id"])

        for engine, op_ids in ops_by_engine.items():
            if engine == "debug":
                continue
            limit = SLOT_LIMITS[engine]

            for c in range(target_makespan):
                at_cycle_vars = []
                for op_id in op_ids:
                    at_cycle = model.NewBoolVar(f'op{op_id}_at_cycle{c}')
                    model.Add(cycle[op_id] == c).OnlyEnforceIf(at_cycle)
                    model.Add(cycle[op_id] != c).OnlyEnforceIf(at_cycle.Not())
                    at_cycle_vars.append(at_cycle)

                model.Add(sum(at_cycle_vars) <= limit)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit_per
        solver.parameters.num_search_workers = 4

        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            cycle_assignments = {op["id"]: solver.Value(cycle[op["id"]]) for op in ops}
            return True, cycle_assignments
        return False, None

    best_assignments = None

    while lower_bound < upper_bound:
        mid = (lower_bound + upper_bound) // 2
        print(f"  Trying makespan = {mid}...", end=" ")

        feasible, assignments = is_feasible(mid)

        if feasible:
            print("FEASIBLE")
            upper_bound = mid
            best_assignments = assignments
        else:
            print("INFEASIBLE")
            lower_bound = mid + 1

    return lower_bound, best_assignments


def analyze_optimal_schedule(ops, cycle_assignments):
    """Analyze the optimal schedule to understand resource usage."""
    max_cycle = max(cycle_assignments.values())

    # Group ops by cycle and engine
    schedule_by_cycle = defaultdict(lambda: defaultdict(list))
    for op in ops:
        c = cycle_assignments[op["id"]]
        schedule_by_cycle[c][op["engine"]].append(op)

    print(f"\nOptimal schedule analysis:")
    print(f"  Makespan: {max_cycle + 1} cycles")

    # Resource utilization
    utilization = {engine: [] for engine in SLOT_LIMITS if engine != "debug"}
    for c in range(max_cycle + 1):
        for engine in utilization:
            count = len(schedule_by_cycle[c][engine])
            utilization[engine].append(count)

    print("\n  Average resource utilization:")
    for engine in ["alu", "valu", "load", "store", "flow"]:
        avg = sum(utilization[engine]) / (max_cycle + 1)
        max_use = max(utilization[engine]) if utilization[engine] else 0
        limit = SLOT_LIMITS[engine]
        print(f"    {engine:5s}: avg={avg:.2f}/{limit}, max={max_use}/{limit}")

    return schedule_by_cycle


def main():
    """Main optimization routine."""
    print("=" * 70)
    print("T3: Constraint-Based Optimal Scheduling - Optimal Scheduler")
    print("=" * 70)

    # Load analysis data
    data = load_analysis_data()
    ops = data["operations"]
    deps = data["dependencies"]

    print(f"\nLoaded {len(ops)} operations with {sum(len(v) for v in deps.values())} dependencies")
    print(f"Current schedule: {data['loop_cycles']} cycles")
    print(f"Lower bound: {data['lower_bound']} cycles")

    # Use binary search to find optimal makespan
    print("\n" + "=" * 70)
    print("BINARY SEARCH FOR OPTIMAL MAKESPAN")
    print("=" * 70)

    lower = data["lower_bound"]
    upper = data["loop_cycles"]

    optimal_makespan, cycle_assignments = solve_with_binary_search(
        ops, deps, lower, upper, time_limit_per=30
    )

    if cycle_assignments is None:
        print("\nFailed to find any feasible solution!")
        return

    print(f"\nOptimal makespan found: {optimal_makespan} cycles")

    # Analyze the optimal schedule
    schedule_by_cycle = analyze_optimal_schedule(ops, cycle_assignments)

    # Compare to current
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"  Current schedule: {data['loop_cycles']} cycles")
    print(f"  Optimal schedule: {optimal_makespan} cycles")
    print(f"  Improvement: {data['loop_cycles'] - optimal_makespan} cycles ({100*(data['loop_cycles'] - optimal_makespan)/data['loop_cycles']:.1f}%)")
    print(f"  Theoretical lower bound: {data['lower_bound']} cycles")
    print(f"  Gap from theoretical: {optimal_makespan - data['lower_bound']} cycles")

    # Save results
    results = {
        "current_cycles": data["loop_cycles"],
        "optimal_cycles": optimal_makespan,
        "lower_bound": data["lower_bound"],
        "improvement_cycles": data["loop_cycles"] - optimal_makespan,
        "improvement_percent": 100 * (data["loop_cycles"] - optimal_makespan) / data["loop_cycles"],
        "cycle_assignments": cycle_assignments,
        "schedule_by_cycle": {
            c: {
                engine: [{"id": op["id"], "op_type": op["op_type"]} for op in ops_list]
                for engine, ops_list in engines.items()
            }
            for c, engines in schedule_by_cycle.items()
        }
    }

    with open("/home/hestiasadmin/projects/original_performance_takehome/experiments/T3_constraint_solver/optimal_schedule.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\nOptimal schedule saved to optimal_schedule.json")

    # If improvement is significant, suggest next steps
    if results["improvement_percent"] > 5:
        print("\n" + "=" * 70)
        print("RECOMMENDATION")
        print("=" * 70)
        print(f"  The current schedule is {results['improvement_percent']:.1f}% suboptimal.")
        print("  A new kernel implementation using the optimal schedule could")
        print(f"  potentially reduce total cycles from ~9,793 to ~{int(9793 * optimal_makespan / data['loop_cycles'])}.")

    return results


if __name__ == "__main__":
    main()
