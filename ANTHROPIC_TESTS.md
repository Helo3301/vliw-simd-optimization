# Anthropic Performance Challenge: Experimental Test Specifications

**Document Version:** 1.0
**Date:** 2026-01-24
**Baseline:** 9,793 cycles (15.1x speedup from original 147,734)
**Target:** < 2,164 cycles (Opus 4 threshold)

---

## Overview

This document specifies experiments to test 6 theories derived from comparing our current implementation against academic literature on VLIW scheduling, software pipelining, and memory optimization.

### Theory Summary

| ID | Theory | Expected Gain | Risk |
|----|--------|---------------|------|
| T1 | Modulo-Scheduled Multi-Iteration Pipeline | 2-3x | High |
| T2 | Index-Aware Gather Scheduling | 5-10% | Low |
| T3 | Constraint-Based Optimal Scheduling | 10-20% | Medium |
| T4 | 8-Desk Deep Pipeline | 50% | High |
| T5 | Warp-Style Specialization | 20-30% | High |
| T6 | Algebraic Strength Reduction on Hash | <5% | Low |

---

## Experiment T1: Modulo-Scheduled Multi-Iteration Pipeline

### Hypothesis

By overlapping operations across iteration boundaries (not just within iterations), we can approach the theoretical minimum Initiation Interval (II) of ~16 cycles per 32 elements.

### Background

Current approach processes iteration N completely before starting N+1. True software pipelining starts iteration N+1 while N is still in flight, using modulo scheduling to avoid resource conflicts.

**Theoretical Minimum II:**
```
II_resource = ceil(32 loads / 2 loads_per_cycle) = 16 cycles
II_recurrence = hash_latency = 12 cycles
II_min = max(16, 12) = 16 cycles
```

**Current:** ~82 cycles per iteration = 5.1x above minimum.

### Methodology

#### Phase 1: Analyze Current Schedule

1. Map all operations to (cycle, slot) pairs
2. Identify the critical path (longest dependency chain)
3. Compute current II and resource utilization per cycle

**Measurement:**
```python
def analyze_schedule(kernel):
    """Extract cycle-by-cycle resource utilization."""
    utilization = defaultdict(lambda: {"alu": 0, "valu": 0, "load": 0, "store": 0, "flow": 0})
    for cycle, ops in enumerate(kernel):
        for slot_type, slot_ops in ops.items():
            utilization[cycle][slot_type] = len(slot_ops)
    return utilization
```

#### Phase 2: Design Modulo Schedule

1. Set target II = 20 (conservative, 25% above minimum)
2. Create modulo resource reservation table (MRT) with II rows
3. Schedule operations using the MRT, wrapping around modulo II
4. Handle cross-iteration dependencies with explicit offset tracking

**Modulo Schedule Structure:**
```
MRT[cycle % II][slot] = operation_from_iteration[cycle // II]

Example with II=20:
Cycle 0:  Iter 0 gather[0:1]
Cycle 1:  Iter 0 gather[2:3]
...
Cycle 16: Iter 0 hash_final | Iter 1 gather[0:1]  # Overlap!
Cycle 17: Iter 0 branch     | Iter 1 gather[2:3]
...
Cycle 20: Iter 1 continues where Iter 0 started
```

#### Phase 3: Implement Kernel Generator

```python
def generate_modulo_kernel(II: int, num_iterations: int):
    """
    Generate kernel with modulo scheduling.

    Args:
        II: Initiation Interval (cycles between iteration starts)
        num_iterations: Total iterations (128 for our problem)

    Returns:
        List of cycle dictionaries
    """
    kernel = []

    # Prologue: ramp up pipeline
    for iter_offset in range(pipeline_depth):
        # Start iteration iter_offset at cycle iter_offset * II
        pass

    # Steady state: all stages active
    for cycle in range(steady_state_cycles):
        ops = {"alu": [], "valu": [], "load": [], "store": [], "flow": []}

        for stage in range(pipeline_depth):
            iter_num = (cycle // II) - stage
            if 0 <= iter_num < num_iterations:
                stage_ops = get_stage_ops(stage, cycle % II, iter_num)
                merge_ops(ops, stage_ops)

        kernel.append(ops)

    # Epilogue: drain pipeline
    for iter_offset in range(pipeline_depth):
        pass

    return kernel
```

#### Phase 4: Validation

1. Run with `check=True` to verify correctness
2. Compare cycles against baseline
3. Verify no resource conflicts (≤12 ALU, ≤6 VALU, ≤2 load, ≤2 store, ≤1 flow per cycle)

### Success Criteria

| Metric | Threshold | Stretch Goal |
|--------|-----------|--------------|
| Cycles | < 6,000 | < 4,000 |
| Speedup vs baseline | > 1.5x | > 2.5x |
| Correctness | 100% pass | 100% pass |
| II achieved | ≤ 25 | ≤ 20 |

### Risks

1. **Register pressure:** Multiple iterations in flight = multiple copies of all variables
2. **Dependency tracking:** Cross-iteration dependencies are complex to manage
3. **Prologue/epilogue overhead:** May dominate for small iteration counts

### Fallback

If full modulo scheduling is too complex, try **partial overlap**: start iteration N+1's gather while N's final hash stages are computing, without full pipelining.

---

## Experiment T2: Index-Aware Gather Scheduling

### Hypothesis

Tree traversal indices follow predictable patterns (binary tree structure) that can be exploited for better cache utilization or gather scheduling.

### Background

The tree has 2047 nodes. Index evolution: `idx_next = 2*idx + offset` where offset ∈ {1,2}.

**Pattern Analysis:**
- Round 0: indices start at 0
- Round 1: indices ∈ {1, 2}
- Round 2: indices ∈ {3, 4, 5, 6}
- Round k: indices ∈ [2^k - 1, 2^(k+1) - 2]

After 16 rounds, indices wrap (mod 2047) so pattern breaks down.

### Methodology

#### Phase 1: Index Distribution Analysis

```python
def analyze_index_distribution(rounds=16, batch_size=256):
    """Analyze where indices land after each round."""
    indices = [0] * batch_size
    values = [i for i in range(batch_size)]  # Simplified

    distribution = []
    for round in range(rounds):
        # Simulate hash and branch
        for i in range(batch_size):
            val = hash(values[i])
            offset = 1 + (val & 1)
            indices[i] = (2 * indices[i] + offset) % 2047

        # Record distribution
        distribution.append({
            "round": round,
            "min_idx": min(indices),
            "max_idx": max(indices),
            "unique": len(set(indices)),
            "clusters": find_clusters(indices),
        })

    return distribution
```

#### Phase 2: Cluster-Aware Gather

If indices cluster, we can:
1. Sort batch elements by current index
2. Gather in sorted order (better cache line utilization)
3. Unsort results

**Cost-Benefit Analysis:**
- Sort cost: O(n log n) comparisons ≈ 256 * 8 = 2048 comparisons
- At 2 comparisons per cycle (ALU), sorting costs ~1000 cycles
- Benefit: Only worthwhile if gather improves by >1000 cycles

#### Phase 3: Prefetch Experiment

If the simulator supports memory prefetching or has cache modeling:

```python
def gather_with_prefetch(indices, tree_values, prefetch_distance=4):
    """
    Prefetch upcoming gather addresses while processing current ones.
    """
    results = []
    for i in range(0, len(indices), 8):
        # Prefetch next batch
        if i + 8 < len(indices):
            for j in range(8):
                prefetch(tree_values + indices[i + 8 + j])

        # Gather current batch
        for j in range(8):
            results.append(load(tree_values + indices[i + j]))

    return results
```

#### Phase 4: Measurement

Run baseline and modified versions, compare:
1. Total cycles
2. Cycles spent in gather operations specifically
3. Cache hit rate (if observable)

### Success Criteria

| Metric | Threshold | Notes |
|--------|-----------|-------|
| Cycles | < 9,300 | 5% improvement |
| Gather cycles | Measurable reduction | Compare before/after |
| Overhead | < benefit | Sorting/prefetch cost must be covered |

### Risks

1. Simulator may not model cache effects
2. Sorting overhead may exceed benefits
3. Index distribution may be too uniform to exploit

### Fallback

If no cache effects, document findings and move on. This experiment is low-risk, low-reward.

---

## Experiment T3: Constraint-Based Optimal Scheduling

### Hypothesis

Our manual VLIW packing missed optimization opportunities. A constraint solver can find the provably optimal schedule.

### Background

From arXiv:1804.02452, the scheduling problem can be modeled as:

**Variables:**
- `cycle[op]`: When operation `op` executes
- `slot[op, type]`: Binary - does `op` use slot `type`?

**Constraints:**
- Dependencies: `cycle[b] ≥ cycle[a] + latency[a]` for all (a→b)
- Resources: `Σ slot[op, type] ≤ capacity[type]` for each cycle and type
- Slot assignment: Each op uses exactly one slot type

**Objective:** Minimize `max(cycle[op])`

### Methodology

#### Phase 1: Extract Dependency Graph

```python
def extract_dependencies(kernel_ops):
    """
    Extract data dependencies from kernel operations.

    Returns:
        List of (source_op, dest_op, latency) tuples
    """
    dependencies = []

    # Track last write to each register
    last_writer = {}

    for op_id, op in enumerate(kernel_ops):
        dest, *sources = op

        # Add dependencies from sources
        for src in sources:
            if src in last_writer:
                dependencies.append((last_writer[src], op_id, 1))

        # Update last writer
        last_writer[dest] = op_id

    return dependencies
```

#### Phase 2: Build ILP Model

```python
from ortools.linear_solver import pywraplp

def build_scheduling_ilp(ops, dependencies, slot_capacities):
    """
    Build ILP model for optimal scheduling.
    """
    solver = pywraplp.Solver.CreateSolver('SCIP')

    max_cycles = len(ops) * 2  # Upper bound

    # Variables
    cycle = {}
    for op_id in range(len(ops)):
        cycle[op_id] = solver.IntVar(0, max_cycles, f'cycle_{op_id}')

    makespan = solver.IntVar(0, max_cycles, 'makespan')

    # Dependency constraints
    for src, dst, latency in dependencies:
        solver.Add(cycle[dst] >= cycle[src] + latency)

    # Makespan definition
    for op_id in range(len(ops)):
        solver.Add(makespan >= cycle[op_id])

    # Resource constraints (linearized)
    # For each cycle c and slot type t:
    #   Σ (cycle[op] == c AND type[op] == t) ≤ capacity[t]
    # This requires auxiliary binary variables...

    # Objective
    solver.Minimize(makespan)

    return solver, cycle, makespan
```

#### Phase 3: Apply to One Loop Iteration

1. Extract all operations from one loop iteration (current: ~200 ops)
2. Build dependency graph
3. Solve ILP for optimal schedule
4. Compare against current heuristic schedule

#### Phase 4: Full Kernel Generation

If ILP finds better schedule for one iteration:
1. Regenerate full kernel using optimal operation ordering
2. Validate correctness
3. Measure total cycles

### Success Criteria

| Metric | Threshold | Notes |
|--------|-----------|-------|
| ILP solve time | < 60 seconds | Per iteration |
| Cycles saved | > 5% | Worth the complexity |
| Optimal gap | 0% | ILP should find provably optimal |

### Risks

1. ILP may be too slow for full kernel
2. Resource constraints make model complex
3. May only save a few cycles if we're already close to optimal

### Fallback

If ILP is too slow, try:
1. Constraint propagation (CP) instead of ILP
2. Local search starting from current solution
3. Analyze ILP solution for patterns we can manually apply

---

## Experiment T4: 8-Desk Deep Pipeline

### Hypothesis

Doubling the unroll factor from 4 to 8 desks enables more overlap opportunities and approaches the theoretical minimum cycles.

### Background

**Current (4-desk):**
- 4 desks × 8 elements = 32 elements per iteration
- 128 iterations total
- ~82 cycles per iteration
- Register usage: ~640 words

**Proposed (8-desk):**
- 8 desks × 8 elements = 64 elements per iteration
- 64 iterations total
- Expected: ~100-120 cycles per iteration (but processing 2x elements)
- Register usage: ~1280 words (fits in 1536 scratch)

### Methodology

#### Phase 1: Register Budget Analysis

```python
def calculate_register_budget(num_desks):
    """Calculate register requirements for N-desk pipeline."""
    per_desk = {
        "indices": 8,      # Vector of current indices
        "values": 8,       # Vector of current values
        "node_vals": 8,    # Vector of gathered tree values
        "hash_temps": 16,  # Temporary registers for hash stages
        "branch_temps": 8, # Temporary registers for branch computation
        "addresses": 8,    # Computed memory addresses
    }

    per_desk_total = sum(per_desk.values())  # ~56 per desk

    global_regs = {
        "batch_offset": 1,
        "loop_counter": 1,
        "constants": 10,   # Hash constants, tree pointers, etc.
        "misc": 10,
    }

    total = num_desks * per_desk_total + sum(global_regs.values())

    return {
        "per_desk": per_desk_total,
        "global": sum(global_regs.values()),
        "total": total,
        "available": 1536,
        "utilization": total / 1536,
    }

# Results:
# 4 desks: 56*4 + 22 = 246 words (16% utilization)
# 8 desks: 56*8 + 22 = 470 words (31% utilization)
# 16 desks: 56*16 + 22 = 918 words (60% utilization)
```

#### Phase 2: Overlap Analysis

With 8 desks, we can overlap:
- Desk 4's gather with Desk 0's hash (same iteration)
- Desk 5's gather with Desk 1's hash
- Desk 6's gather with Desk 2's hash
- Desk 7's gather with Desk 3's hash
- Then Desk 0's gather (next iter) with Desk 4's hash
- etc.

**Pipeline Structure:**
```
Cycle range | Desk 0-3 activity      | Desk 4-7 activity
------------|------------------------|------------------------
0-15        | Gather + Hash stage 1  | (idle - prologue)
16-31       | Hash stages 2-6        | Gather + Hash stage 1
32-47       | Branch + Store         | Hash stages 2-6
48-63       | Gather (next iter)     | Branch + Store
...         | (pattern repeats)      | ...
```

#### Phase 3: Implementation

```python
def generate_8desk_kernel():
    """Generate kernel with 8-desk pipeline."""
    kernel = []

    # Register allocation
    desk_regs = allocate_desk_registers(8)

    # Prologue: start desks 0-3
    kernel.extend(generate_prologue_4desk(desk_regs[0:4]))

    # Main loop: 64 iterations (each processes 64 elements)
    loop_start = len(kernel)

    for phase in range(4):  # 4 phases per iteration
        if phase == 0:
            # Desks 0-3: gather + hash1 | Desks 4-7: hash 2-6
            pass
        elif phase == 1:
            # Desks 0-3: hash 2-6 | Desks 4-7: branch + store
            pass
        elif phase == 2:
            # Desks 0-3: branch + store | Desks 4-7: gather + hash1
            pass
        elif phase == 3:
            # Desks 0-3: (prep next) | Desks 4-7: hash 2-6
            pass

    # Loop control
    kernel.extend(generate_loop_control(loop_start))

    # Epilogue: drain desks 4-7
    kernel.extend(generate_epilogue_4desk(desk_regs[4:8]))

    return kernel
```

#### Phase 4: Validation and Measurement

1. Verify correctness with `check=True`
2. Measure total cycles
3. Calculate cycles per element
4. Compare against 4-desk baseline

### Success Criteria

| Metric | Threshold | Stretch Goal |
|--------|-----------|--------------|
| Cycles | < 6,500 | < 5,000 |
| Cycles/element | < 1.6 | < 1.2 |
| Register usage | < 1200 | < 1000 |
| Correctness | 100% | 100% |

### Risks

1. Increased complexity may introduce bugs
2. 8-desk structure may not fit cleanly in iteration count (4096 / 64 = 64 iterations)
3. Register spilling if budget exceeded

### Fallback

If 8 desks is too complex, try 6 desks as intermediate step.

---

## Experiment T5: Warp-Style Specialization

### Hypothesis

Separating the kernel into specialized "producer" and "consumer" phases, inspired by GPU warp specialization, can improve pipeline efficiency.

### Background

From arXiv:2512.18134:
> "Warp specialization assigns different warps to different pipeline stages"

In our VLIW context, we don't have warps, but we can:
1. Dedicate certain cycles to "production" (hash computation)
2. Dedicate certain cycles to "consumption" (gather, store)
3. Use explicit data handoff between phases

### Methodology

#### Phase 1: Characterize Current Mixing

Analyze current kernel to see how producer/consumer operations are mixed:

```python
def categorize_operations(kernel):
    """Categorize ops as producer (compute) or consumer (memory)."""
    stats = {"producer_only": 0, "consumer_only": 0, "mixed": 0}

    for cycle in kernel:
        has_compute = len(cycle.get("alu", [])) + len(cycle.get("valu", [])) > 0
        has_memory = len(cycle.get("load", [])) + len(cycle.get("store", [])) > 0

        if has_compute and has_memory:
            stats["mixed"] += 1
        elif has_compute:
            stats["producer_only"] += 1
        elif has_memory:
            stats["consumer_only"] += 1

    return stats
```

#### Phase 2: Design Specialized Phases

**Producer Phase (Cycles 0-N):**
- Focus: Hash computation
- Slots used: ALU (12), VALU (6)
- Output: Computed values and next indices
- Store to "handoff" registers

**Consumer Phase (Cycles N-M):**
- Focus: Memory operations
- Slots used: Load (2), Store (2)
- Input: Read from "handoff" registers
- Perform gather and store

#### Phase 3: Implementation

```python
def generate_specialized_kernel():
    """Generate kernel with phase specialization."""
    kernel = []

    # Handoff buffer registers
    handoff_start = 1400  # Use high scratch addresses

    for iteration in range(num_iterations):
        # === PRODUCER PHASE ===
        # Compute hash for all 4 desks, store results to handoff
        for desk in range(4):
            # Hash computation (all VALU)
            kernel.extend(generate_hash_only(desk))

        # Store computed indices to handoff buffer
        kernel.extend(generate_handoff_store(handoff_start))

        # === CONSUMER PHASE ===
        # Load indices from handoff, perform gather, store results
        kernel.extend(generate_handoff_load(handoff_start))

        for desk in range(4):
            # Gather and store (all Load/Store)
            kernel.extend(generate_gather_store_only(desk))

    return kernel
```

#### Phase 4: Measurement

Compare:
1. Total cycles
2. ALU/VALU utilization during producer phase
3. Load/Store utilization during consumer phase
4. Handoff overhead

### Success Criteria

| Metric | Threshold | Notes |
|--------|-----------|-------|
| Cycles | < 8,000 | 20% improvement |
| Producer utilization | > 80% | ALU+VALU usage |
| Consumer utilization | > 80% | Load+Store usage |
| Handoff overhead | < 10% | Extra cycles for data movement |

### Risks

1. Handoff overhead may exceed benefits
2. Phase separation may reduce overlap opportunities
3. May conflict with other optimizations (T1, T4)

### Fallback

If full specialization hurts, try partial specialization: only separate the most constrained phases.

---

## Experiment T6: Algebraic Strength Reduction on Hash

### Hypothesis

The 6-stage hash function may have algebraic properties that allow optimization.

### Background

Each hash stage is:
```
val = (val op1 const) op2 (val op3 shift)
```

Where op1, op2, op3 ∈ {+, ^} and shift is a constant.

### Methodology

#### Phase 1: Algebraic Analysis

```python
def analyze_hash_algebra():
    """Analyze algebraic properties of hash stages."""
    stages = [
        ("+", 0x7ED55D16, "+", "<<", 12),
        ("^", 0xC761C23C, "^", ">>", 19),
        ("+", 0x165667B1, "+", "<<", 5),
        ("+", 0xD3A2646C, "^", "<<", 9),
        ("+", 0xFD7046C5, "+", "<<", 3),
        ("^", 0xB55A4F09, "^", ">>", 16),
    ]

    # Check for:
    # 1. Associativity exploitation
    # 2. Common subexpressions
    # 3. Strength reduction opportunities
    # 4. Lookup table feasibility

    # Analysis results...
    return analysis
```

#### Phase 2: Stage Fusion

Can we combine stages?

```
Stage 0: val = (val + C0) + (val << 12)
       = val + C0 + (val << 12)
       = val * (1 + 2^12) + C0
       = val * 4097 + C0

Stage 1: val = (val ^ C1) ^ (val >> 19)
       = Cannot simplify (XOR doesn't distribute)
```

**Finding:** Stages with XOR cannot be algebraically simplified. Stages with only + can potentially use multiply-add.

#### Phase 3: FMA Exploitation

Check if `multiply_add` instruction helps:

```python
# Stage 0 alternative using FMA:
# val = val * 4097 + C0
# But wait - does our VALU support multiply_add?

# From ISA:
# ("multiply_add", dest, a, b, c) -> dest[i] = a[i] * b[i] + c[i]

# So Stage 0 could be:
# vbroadcast(const_4097, 4097)
# vbroadcast(const_C0, 0x7ED55D16)
# multiply_add(val, val, const_4097, const_C0)  # 1 cycle instead of 2!
```

#### Phase 4: Implementation

```python
def generate_optimized_hash():
    """Generate hash with FMA optimization for applicable stages."""
    ops = []

    # Stage 0: val = val * 4097 + C0 (FMA)
    ops.append({"valu": [("multiply_add", "val", "val", "const_4097", "const_C0")]})

    # Stage 1: val = (val ^ C1) ^ (val >> 19) (no change)
    ops.append({"valu": [("^", "tmp1", "val", "const_C1"), (">>", "tmp2", "val", 19)]})
    ops.append({"valu": [("^", "val", "tmp1", "tmp2")]})

    # ... continue for remaining stages

    return ops
```

#### Phase 5: Measurement

1. Count cycles for optimized vs. original hash
2. Verify hash produces identical results
3. Measure total kernel improvement

### Success Criteria

| Metric | Threshold | Notes |
|--------|-----------|-------|
| Hash cycles | Reduced by ≥2 | Per element |
| Total cycles | < 9,500 | ~3% improvement |
| Hash correctness | 100% | Bit-exact match |

### Risks

1. FMA may not be faster than separate ops (depends on architecture)
2. Only some stages can be optimized
3. Register pressure from constants

### Fallback

If no algebraic optimization works, document findings and confirm hash is already optimal.

---

## Experimental Protocol

### Execution Order

Run experiments in order of expected impact × inverse complexity:

1. **T6** (Low complexity, quick validation)
2. **T2** (Low complexity, may inform other experiments)
3. **T3** (Medium complexity, provides optimality insights)
4. **T4** (High impact, medium complexity)
5. **T5** (Medium impact, high complexity)
6. **T1** (Highest potential, highest complexity)

### Measurement Protocol

For each experiment:

1. **Baseline measurement:**
   ```bash
   python perf_takehome.py --check  # Verify correctness
   python perf_takehome.py          # Record cycles
   ```

2. **Implementation**

3. **Post-implementation measurement:**
   ```bash
   python perf_takehome.py --check  # Verify correctness first!
   python perf_takehome.py          # Record cycles
   ```

4. **Documentation:**
   - Cycles before/after
   - Speedup achieved
   - Key insights
   - Unexpected findings

### Success Aggregation

| Scenario | Cumulative Cycles | Threshold |
|----------|-------------------|-----------|
| Baseline | 9,793 | - |
| +T6 only | ~9,500 | - |
| +T4 | ~5,000 | - |
| +T1 | ~2,500 | Opus 4 threshold: 2,164 |
| +T3 | ~2,100 | ✓ Target achieved |

---

## Appendix A: Register Map (Current)

```
Scratch addresses (current 4-desk implementation):
0-7:     desk0_indices (vector)
8-15:    desk0_values (vector)
16-23:   desk0_node_vals (vector)
24-31:   desk0_temps (hash)
32-39:   desk1_indices
...
160-167: Constants (hash)
168-175: Pointers (inp_indices_p, inp_values_p, forest_values_p)
176-183: Loop control (batch_offset, counter, etc.)
```

## Appendix B: Cycle Breakdown (Current)

```
Per iteration (32 elements):
  Desk 0: 15 cycles
    - Gather: 4 cycles (overlapped)
    - Hash: 12 cycles (overlapped with gather)
    - Branch: 3 cycles
    - Store: 2 cycles
  Desk 1: 15 cycles (similar)
  Desk 2: 16 cycles
  Desk 3: 20 cycles (includes speculative load)
  Loop control: 2 cycles

Total: ~82 cycles per iteration
128 iterations × 82 = 10,496 cycles
+ Prologue/epilogue: ~700 cycles
= ~11,200 cycles (theoretical)

Actual: 9,793 cycles (better due to packing)
```

## Appendix C: References

1. arXiv:1804.02452 - Combinatorial Register Allocation and Instruction Scheduling
2. arXiv:1409.7628 - Survey on Combinatorial Register Allocation
3. arXiv:2512.18134 - Software Pipelining and Warp Specialization for GPUs
4. arXiv:1811.03743 - Spatter: Gather/Scatter Performance Tool
5. arXiv:1911.03991 - DNN for Loop Unrolling Factor Estimation
