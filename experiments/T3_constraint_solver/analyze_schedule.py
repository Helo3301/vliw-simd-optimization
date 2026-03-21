"""
T3: Constraint-Based Optimal Scheduling - Schedule Analysis

This script:
1. Extracts all operations from one loop iteration of the current kernel
2. Builds a dependency graph
3. Identifies current slot utilization per cycle
4. Reports findings on critical path and resource usage
"""

import sys
sys.path.insert(0, '/home/hestiasadmin/projects/original_performance_takehome')

from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set, Any
import json

from problem import SLOT_LIMITS, VLEN, HASH_STAGES


@dataclass
class Operation:
    """Represents a single operation in the schedule."""
    id: int
    engine: str           # alu, valu, load, store, flow
    op_type: str          # The operation name (e.g., '+', 'vload', 'cond_jump')
    dest: int             # Destination scratch address (-1 if none)
    sources: List[int]    # Source scratch addresses
    cycle: int            # Cycle number when scheduled
    raw_slot: tuple       # Original slot tuple

    def __repr__(self):
        return f"Op{self.id}({self.engine}.{self.op_type}, dest={self.dest}, src={self.sources}, cycle={self.cycle})"


def parse_slot(engine: str, slot: tuple) -> Tuple[int, List[int], str]:
    """
    Parse a slot tuple to extract destination, sources, and operation type.
    Returns (dest, sources, op_type).
    """
    op_type = slot[0]
    dest = -1
    sources = []

    if engine == "alu":
        # (op, dest, src1, src2)
        op_type, dest, src1, src2 = slot
        sources = [src1, src2]

    elif engine == "valu":
        if op_type == "vbroadcast":
            # (vbroadcast, dest, src)
            _, dest, src = slot
            sources = [src]
        elif op_type == "multiply_add":
            # (multiply_add, dest, a, b, c)
            _, dest, a, b, c = slot
            sources = [a, b, c]
        else:
            # (op, dest, src1, src2)
            _, dest, src1, src2 = slot
            sources = [src1, src2]

    elif engine == "load":
        if op_type == "load":
            # (load, dest, addr)
            _, dest, addr = slot
            sources = [addr]
        elif op_type == "load_offset":
            # (load_offset, dest, addr, offset)
            _, dest, addr, offset = slot
            sources = [addr]
        elif op_type == "vload":
            # (vload, dest, addr)
            _, dest, addr = slot
            sources = [addr]
        elif op_type == "const":
            # (const, dest, val)
            _, dest, val = slot
            sources = []

    elif engine == "store":
        if op_type == "store":
            # (store, addr, src)
            _, addr, src = slot
            dest = -1  # Stores don't have a scratch destination
            sources = [addr, src]
        elif op_type == "vstore":
            # (vstore, addr, src)
            _, addr, src = slot
            dest = -1
            sources = [addr, src]

    elif engine == "flow":
        if op_type == "select":
            # (select, dest, cond, a, b)
            _, dest, cond, a, b = slot
            sources = [cond, a, b]
        elif op_type == "vselect":
            # (vselect, dest, cond, a, b)
            _, dest, cond, a, b = slot
            sources = [cond, a, b]
        elif op_type == "cond_jump":
            # (cond_jump, cond, addr)
            _, cond, addr = slot
            sources = [cond]
        elif op_type in ("pause", "halt"):
            pass
        elif op_type == "add_imm":
            # (add_imm, dest, a, imm)
            _, dest, a, imm = slot
            sources = [a]

    return dest, sources, op_type


def extract_operations(kernel: List[Dict]) -> List[Operation]:
    """Extract all operations from a kernel, assigning cycle numbers."""
    ops = []
    op_id = 0

    for cycle, instr in enumerate(kernel):
        for engine, slots in instr.items():
            if engine == "debug":
                continue
            for slot in slots:
                dest, sources, op_type = parse_slot(engine, slot)
                ops.append(Operation(
                    id=op_id,
                    engine=engine,
                    op_type=op_type,
                    dest=dest,
                    sources=sources,
                    cycle=cycle,
                    raw_slot=slot
                ))
                op_id += 1

    return ops


def build_dependency_graph(ops: List[Operation]) -> Dict[int, List[Tuple[int, int]]]:
    """
    Build a dependency graph from operations.
    Returns a dict mapping op_id -> list of (dependent_op_id, latency).

    A dependency exists when op B reads from a location that op A writes to.
    We also handle vector operations (VLEN=8 consecutive addresses).
    """
    # Track last writer to each address range
    # For vectors, dest means dest..dest+VLEN-1

    dependencies = defaultdict(list)

    # For each address, track (op_id, is_vector)
    last_writer: Dict[int, Tuple[int, bool]] = {}

    def get_dest_range(op: Operation) -> List[int]:
        """Get all destination addresses for an operation."""
        if op.dest == -1:
            return []
        if op.engine == "valu" or (op.engine == "load" and op.op_type == "vload"):
            return list(range(op.dest, op.dest + VLEN))
        return [op.dest]

    def get_source_range(op: Operation) -> List[int]:
        """Get all source addresses for an operation (including vector expansion)."""
        addrs = []
        for src in op.sources:
            if op.engine == "valu":
                # Vector sources read VLEN addresses
                addrs.extend(range(src, src + VLEN))
            else:
                addrs.append(src)
        return addrs

    # Also track stores to memory (different namespace)
    # For simplicity, we'll treat store addresses as potential dependencies

    for op in ops:
        # Check dependencies: does this op read from something a previous op wrote?
        src_addrs = get_source_range(op)
        for addr in src_addrs:
            if addr in last_writer:
                writer_id, _ = last_writer[addr]
                if writer_id != op.id:
                    # RAW dependency with latency 1
                    dependencies[writer_id].append((op.id, 1))

        # Update last writer
        for addr in get_dest_range(op):
            last_writer[addr] = (op.id, op.engine == "valu")

    return dependencies


def compute_critical_path(ops: List[Operation], dependencies: Dict[int, List[Tuple[int, int]]]) -> Tuple[int, List[int]]:
    """
    Compute the critical path length (longest dependency chain).
    Returns (path_length, list of op_ids on critical path).
    """
    n = len(ops)
    # dist[i] = longest path starting from op i
    dist = [1] * n
    next_op = [-1] * n

    # Process in reverse topological order (by cycle, then by id)
    sorted_ops = sorted(ops, key=lambda o: (-o.cycle, -o.id))

    for op in sorted_ops:
        max_dist = 1
        best_next = -1
        for dep_id, latency in dependencies.get(op.id, []):
            if dist[dep_id] + latency > max_dist:
                max_dist = dist[dep_id] + latency
                best_next = dep_id
        dist[op.id] = max_dist
        next_op[op.id] = best_next

    # Find start of critical path
    start = max(range(n), key=lambda i: dist[i])
    path = [start]
    while next_op[path[-1]] != -1:
        path.append(next_op[path[-1]])

    return dist[start], path


def analyze_slot_utilization(kernel: List[Dict]) -> Dict[str, Dict]:
    """
    Analyze slot utilization per cycle.
    Returns statistics about resource usage.
    """
    stats = {
        "per_cycle": [],
        "totals": {engine: 0 for engine in SLOT_LIMITS if engine != "debug"},
        "max_usage": {engine: 0 for engine in SLOT_LIMITS if engine != "debug"},
        "avg_usage": {engine: 0.0 for engine in SLOT_LIMITS if engine != "debug"},
    }

    for cycle, instr in enumerate(kernel):
        cycle_usage = {}
        for engine in SLOT_LIMITS:
            if engine == "debug":
                continue
            usage = len(instr.get(engine, []))
            cycle_usage[engine] = usage
            stats["totals"][engine] += usage
            stats["max_usage"][engine] = max(stats["max_usage"][engine], usage)
        stats["per_cycle"].append(cycle_usage)

    num_cycles = len(kernel)
    for engine in stats["avg_usage"]:
        stats["avg_usage"][engine] = stats["totals"][engine] / max(1, num_cycles)

    return stats


def find_loop_boundaries(kernel: List[Dict]) -> Tuple[int, int]:
    """
    Find the main loop start and end indices.
    Look for cond_jump instruction that jumps backwards.
    """
    loop_start = None
    loop_end = None

    for i, instr in enumerate(kernel):
        if "flow" in instr:
            for slot in instr["flow"]:
                if slot[0] == "cond_jump":
                    _, cond, target = slot
                    if target < i:
                        loop_end = i
                        loop_start = target
                        break
        if loop_end is not None:
            break

    return loop_start, loop_end


def extract_loop_iteration(kernel: List[Dict], loop_start: int, loop_end: int) -> List[Dict]:
    """Extract a single loop iteration (inclusive of loop_start, inclusive of loop_end)."""
    return kernel[loop_start:loop_end + 1]


def main():
    """Main analysis routine."""
    print("=" * 70)
    print("T3: Constraint-Based Optimal Scheduling - Schedule Analysis")
    print("=" * 70)

    # Import and build the kernel
    from perf_takehome import KernelBuilder

    kb = KernelBuilder()
    kb.build_kernel(forest_height=10, n_nodes=2047, batch_size=256, rounds=16)
    kernel = kb.instrs

    print(f"\nTotal kernel size: {len(kernel)} instructions")

    # Find loop boundaries
    loop_start, loop_end = find_loop_boundaries(kernel)
    print(f"Main loop: cycles {loop_start} to {loop_end}")
    print(f"Loop body size: {loop_end - loop_start + 1} cycles")

    # Extract one iteration
    loop_body = extract_loop_iteration(kernel, loop_start, loop_end)

    # Extract operations
    ops = extract_operations(loop_body)
    print(f"\nOperations in one loop iteration: {len(ops)}")

    # Count by engine
    engine_counts = defaultdict(int)
    for op in ops:
        engine_counts[op.engine] += 1
    print("\nOperations by engine:")
    for engine, count in sorted(engine_counts.items()):
        print(f"  {engine}: {count}")

    # Build dependency graph
    deps = build_dependency_graph(ops)
    total_deps = sum(len(v) for v in deps.values())
    print(f"\nTotal dependencies: {total_deps}")

    # Compute critical path
    crit_len, crit_path = compute_critical_path(ops, deps)
    print(f"\nCritical path length: {crit_len} cycles")
    print("Critical path operations:")
    for op_id in crit_path[:10]:  # Show first 10
        op = ops[op_id]
        print(f"  {op}")
    if len(crit_path) > 10:
        print(f"  ... ({len(crit_path) - 10} more)")

    # Analyze slot utilization
    stats = analyze_slot_utilization(loop_body)
    print("\nSlot utilization in loop body:")
    print(f"  Total cycles: {len(loop_body)}")
    for engine in ["alu", "valu", "load", "store", "flow"]:
        limit = SLOT_LIMITS[engine]
        avg = stats["avg_usage"][engine]
        max_use = stats["max_usage"][engine]
        total = stats["totals"][engine]
        print(f"  {engine:5s}: total={total:3d}, avg={avg:.2f}/{limit}, max={max_use}/{limit}")

    # Calculate theoretical minimum cycles
    print("\n" + "=" * 70)
    print("THEORETICAL MINIMUM ANALYSIS")
    print("=" * 70)

    # Resource-bound lower bound: max(ops_of_type / capacity)
    resource_bound = 0
    for engine, count in engine_counts.items():
        limit = SLOT_LIMITS[engine]
        cycles_needed = (count + limit - 1) // limit  # ceiling division
        print(f"  {engine}: {count} ops / {limit} capacity = {cycles_needed} cycles minimum")
        resource_bound = max(resource_bound, cycles_needed)

    print(f"\nResource-bound lower bound: {resource_bound} cycles")
    print(f"Dependency-bound lower bound (critical path): {crit_len} cycles")
    print(f"Overall lower bound: {max(resource_bound, crit_len)} cycles")
    print(f"Current schedule: {len(loop_body)} cycles")
    print(f"Gap from optimal: {len(loop_body) - max(resource_bound, crit_len)} cycles ({100*(len(loop_body) - max(resource_bound, crit_len))/len(loop_body):.1f}%)")

    # Save detailed data for the optimizer
    analysis_data = {
        "loop_start": loop_start,
        "loop_end": loop_end,
        "loop_cycles": len(loop_body),
        "num_ops": len(ops),
        "engine_counts": dict(engine_counts),
        "critical_path_length": crit_len,
        "resource_bound": resource_bound,
        "lower_bound": max(resource_bound, crit_len),
        "gap_cycles": len(loop_body) - max(resource_bound, crit_len),
        "operations": [
            {
                "id": op.id,
                "engine": op.engine,
                "op_type": op.op_type,
                "dest": op.dest,
                "sources": op.sources,
                "cycle": op.cycle,
                "raw_slot": op.raw_slot
            }
            for op in ops
        ],
        "dependencies": {str(k): v for k, v in deps.items()},
    }

    with open("/home/hestiasadmin/projects/original_performance_takehome/experiments/T3_constraint_solver/analysis_data.json", "w") as f:
        json.dump(analysis_data, f, indent=2, default=str)

    print("\nAnalysis data saved to analysis_data.json")

    return analysis_data


if __name__ == "__main__":
    main()
