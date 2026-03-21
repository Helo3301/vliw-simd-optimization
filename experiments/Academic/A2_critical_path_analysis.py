"""
A2: Critical Path Analysis and Dependency Graph Optimization

Goal: Analyze the dependency graph of the computation to identify:
1. The critical path length
2. Opportunities for parallelism
3. Whether restructuring can reduce the critical path

Theory: The minimum cycle count is bounded by max(resource_bound, critical_path_bound).
If critical_path > resource_bound, we need to restructure to expose more parallelism.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple

# ISA slot limits
VALU_SLOTS = 6
LOAD_SLOTS = 2
STORE_SLOTS = 2
ALU_SLOTS = 12
FLOW_SLOTS = 1

@dataclass
class Operation:
    """Represents a single operation in the computation graph"""
    id: int
    engine: str  # 'valu', 'load', 'store', 'alu', 'flow'
    op_type: str  # '+', '*', 'load', 'vselect', etc.
    reads: List[int]  # IDs of operations this depends on
    description: str = ""

class DependencyGraph:
    """Builds and analyzes the dependency graph of the kernel"""

    def __init__(self):
        self.ops: Dict[int, Operation] = {}
        self.next_id = 0
        self.input_ids: Dict[str, int] = {}  # Name -> ID for inputs

    def add_input(self, name: str) -> int:
        """Add an input (no dependencies)"""
        op_id = self.next_id
        self.next_id += 1
        self.ops[op_id] = Operation(
            id=op_id,
            engine='input',
            op_type='input',
            reads=[],
            description=f"input:{name}"
        )
        self.input_ids[name] = op_id
        return op_id

    def add_op(self, engine: str, op_type: str, reads: List[int], desc: str = "") -> int:
        """Add an operation"""
        op_id = self.next_id
        self.next_id += 1
        self.ops[op_id] = Operation(
            id=op_id,
            engine=engine,
            op_type=op_type,
            reads=reads,
            description=desc
        )
        return op_id

    def compute_depths(self) -> Dict[int, int]:
        """Compute the depth (critical path length to reach) for each operation"""
        depths = {}

        def get_depth(op_id: int) -> int:
            if op_id in depths:
                return depths[op_id]

            op = self.ops[op_id]
            if op.engine == 'input':
                depths[op_id] = 0
            else:
                max_dep_depth = max((get_depth(r) for r in op.reads), default=0)
                depths[op_id] = max_dep_depth + 1

            return depths[op_id]

        for op_id in self.ops:
            get_depth(op_id)

        return depths

    def find_critical_path(self) -> Tuple[int, List[int]]:
        """Find the critical path and its length"""
        depths = self.compute_depths()
        max_depth = max(depths.values())

        # Find an operation at max depth
        critical_op = max(self.ops.keys(), key=lambda x: depths[x])

        # Trace back the critical path
        path = [critical_op]
        current = critical_op
        while self.ops[current].reads:
            # Find the predecessor with maximum depth
            pred = max(self.ops[current].reads, key=lambda x: depths[x])
            path.append(pred)
            current = pred

        path.reverse()
        return max_depth, path

    def count_by_engine(self) -> Dict[str, int]:
        """Count operations by engine"""
        counts = defaultdict(int)
        for op in self.ops.values():
            if op.engine != 'input':
                counts[op.engine] += 1
        return dict(counts)

    def compute_parallelism(self) -> Dict[int, Dict[str, int]]:
        """Compute available parallelism at each depth level"""
        depths = self.compute_depths()

        # Group ops by depth
        by_depth = defaultdict(lambda: defaultdict(int))
        for op_id, depth in depths.items():
            engine = self.ops[op_id].engine
            if engine != 'input':
                by_depth[depth][engine] += 1

        return dict(by_depth)

def build_single_round_graph(desk_id: int, round_id: int, is_gather: bool, graph: DependencyGraph) -> Dict[str, int]:
    """
    Build the dependency graph for a single round of computation.

    Returns dict of output operation IDs.
    """
    # Inputs from previous round (or initial load)
    idx = graph.add_input(f"idx_{desk_id}_{round_id}")
    val = graph.add_input(f"val_{desk_id}_{round_id}")

    # Phase 1: XOR with tree node
    if is_gather:
        # Gather round: Load tree[idx]
        addr = graph.add_op('valu', '+', [idx, graph.add_input(f"forest_p_{desk_id}")],
                           f"addr = idx + forest_p (d{desk_id} r{round_id})")
        node_val = graph.add_op('load', 'load', [addr],
                               f"node_val = tree[addr] (d{desk_id} r{round_id})")
    else:
        # Fused round: Use preloaded tree value
        node_val = graph.add_input(f"tree_preload_{desk_id}_{round_id}")

    xor_val = graph.add_op('valu', '^', [val, node_val],
                          f"val = val ^ node_val (d{desk_id} r{round_id})")

    # Phase 2: Hash (12 ops)
    hash_stages = [
        ('valu', 'multiply_add', [xor_val], "hash stage 0 (FMA)"),
        ('valu', '^', [], "hash stage 1a (XOR C)"),
        ('valu', '>>', [], "hash stage 1b (RSHIFT)"),
        ('valu', '^', [], "hash stage 1c (XOR)"),
        ('valu', 'multiply_add', [], "hash stage 2 (FMA)"),
        ('valu', '+', [], "hash stage 3a (ADD C)"),
        ('valu', '<<', [], "hash stage 3b (LSHIFT)"),
        ('valu', '^', [], "hash stage 3c (XOR)"),
        ('valu', 'multiply_add', [], "hash stage 4 (FMA)"),
        ('valu', '^', [], "hash stage 5a (XOR C)"),
        ('valu', '>>', [], "hash stage 5b (RSHIFT)"),
        ('valu', '^', [], "hash stage 5c (XOR)"),
    ]

    prev = xor_val
    hash_const = graph.add_input(f"hash_const_{desk_id}")

    for i, (engine, op_type, _, desc) in enumerate(hash_stages):
        if i == 0:
            reads = [prev, hash_const]
        else:
            reads = [prev]
            if i in [1, 4, 5, 9]:  # Stages that use constants
                reads.append(hash_const)

        prev = graph.add_op(engine, op_type, reads,
                           f"{desc} (d{desk_id} r{round_id})")

    hashed_val = prev

    # Phase 3: Branch computation (3 ops)
    bit = graph.add_op('valu', '&', [hashed_val, graph.add_input(f"one_{desk_id}")],
                      f"bit = val & 1 (d{desk_id} r{round_id})")
    tmp = graph.add_op('valu', 'multiply_add', [idx, graph.add_input(f"two_{desk_id}"),
                                                 graph.add_input(f"one_{desk_id}")],
                      f"tmp = 2*idx + 1 (d{desk_id} r{round_id})")
    new_idx = graph.add_op('valu', '+', [tmp, bit],
                          f"new_idx = tmp + bit (d{desk_id} r{round_id})")

    return {
        'idx': new_idx,
        'val': hashed_val
    }

def analyze_single_desk():
    """Analyze the critical path for a single desk processing all 16 rounds"""
    print("=" * 60)
    print("Single Desk Critical Path Analysis")
    print("=" * 60)
    print()

    graph = DependencyGraph()

    # Process 16 rounds
    prev_outputs = {'idx': graph.add_input("idx_0"), 'val': graph.add_input("val_0")}

    for r in range(16):
        # Rounds 0-2 and 11-13 are fused (no gather)
        # Rounds 3-10 and 14-15 are gather rounds
        is_gather = r >= 3 and r <= 10 or r >= 14

        # Simplified: just track critical path
        outputs = build_single_round_graph(0, r, is_gather, graph)
        prev_outputs = outputs

    # Analyze
    depths = graph.compute_depths()
    critical_length, critical_path = graph.find_critical_path()
    counts = graph.count_by_engine()
    parallelism = graph.compute_parallelism()

    print(f"Total operations: {sum(counts.values())}")
    print(f"Operations by engine: {counts}")
    print()
    print(f"Critical path length: {critical_length}")
    print()
    print("Critical path (first 20 ops):")
    for op_id in critical_path[:20]:
        op = graph.ops[op_id]
        print(f"  {op_id}: [{op.engine}] {op.description}")

    print()
    print("Parallelism by depth (first 30 levels):")
    for depth in sorted(parallelism.keys())[:30]:
        counts = parallelism[depth]
        print(f"  Depth {depth:3d}: {dict(counts)}")

    return critical_length, counts

def compute_resource_bound(valu_ops: int, load_ops: int) -> int:
    """Compute the resource-based lower bound"""
    valu_bound = (valu_ops + VALU_SLOTS - 1) // VALU_SLOTS
    load_bound = (load_ops + LOAD_SLOTS - 1) // LOAD_SLOTS
    return max(valu_bound, load_bound)

def main():
    print("=" * 70)
    print("A2: Critical Path Analysis and Dependency Graph Optimization")
    print("=" * 70)
    print()
    print("This analysis examines the fundamental dependency structure")
    print("to identify if restructuring could reduce the critical path.")
    print()

    critical_length, counts = analyze_single_desk()

    print()
    print("=" * 60)
    print("THEORETICAL BOUNDS ANALYSIS")
    print("=" * 60)
    print()

    # Scale to full kernel (32 desks)
    valu_per_desk = counts.get('valu', 0)
    load_per_desk = counts.get('load', 0)

    total_valu = valu_per_desk * 32
    total_load = load_per_desk * 32 + 64 + 64  # Plus initial loads and stores

    valu_bound = (total_valu + VALU_SLOTS - 1) // VALU_SLOTS
    load_bound = (total_load + LOAD_SLOTS - 1) // LOAD_SLOTS

    print(f"Per-desk VALU ops: {valu_per_desk}")
    print(f"Per-desk LOAD ops (gather): {load_per_desk}")
    print()
    print(f"Total VALU ops (32 desks): {total_valu}")
    print(f"Total LOAD ops: {total_load}")
    print()
    print(f"VALU resource bound: {valu_bound} cycles")
    print(f"LOAD resource bound: {load_bound} cycles")
    print()
    print(f"Critical path bound (single desk): {critical_length} cycles")
    print()

    if critical_length > valu_bound:
        print("FINDING: CRITICAL PATH LIMITED")
        print("The critical path exceeds the resource bound!")
        print("Restructuring to expose more parallelism could help.")
    else:
        print("FINDING: RESOURCE LIMITED")
        print("The resource bound exceeds the critical path.")
        print("We have enough parallelism; need fewer operations.")

    print()
    print("=" * 60)
    print("IMPLICATIONS FOR 1,363 TARGET")
    print("=" * 60)
    print()

    # Analysis for 1,363 target
    target_cycles = 1363
    max_valu_ops = target_cycles * VALU_SLOTS
    current_valu = 9083  # From theoretical analysis

    print(f"Target: {target_cycles} cycles")
    print(f"Maximum VALU ops at target: {max_valu_ops}")
    print(f"Current VALU ops: {current_valu}")
    print(f"Reduction needed: {current_valu - max_valu_ops} ops ({100 * (current_valu - max_valu_ops) / current_valu:.1f}%)")
    print()

    # B4-2 analysis
    b42_cycles = 1558
    b42_valu = 8608  # Approximate from fusion savings

    print(f"B4-2 result: {b42_cycles} cycles")
    print(f"B4-2 estimated VALU ops: {b42_valu}")
    print(f"B4-2 efficiency: {100 * b42_valu / (b42_cycles * VALU_SLOTS):.1f}%")
    print()

    # What would be needed
    print("To reach 1,363 cycles, we need one of:")
    print(f"  1. Reduce VALU ops to {max_valu_ops} (current: {current_valu})")
    print(f"  2. Achieve 100% VALU utilization (from current ~96%)")
    print(f"  3. Restructure to break the VALU bottleneck")
    print()
    print("Analysis suggests option 1 (operation reduction) is the only viable path.")

if __name__ == "__main__":
    main()
