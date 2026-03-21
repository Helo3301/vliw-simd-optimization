# H59: Constraint-Based Optimal Schedule Design

## Executive Summary

This design document explores using Integer Linear Programming (ILP) or Constraint Programming (CP) to find a provably optimal instruction schedule for the VLIW SIMD kernel. **The key insight**: the current H54 schedule was hand-crafted using heuristics; a solver might find hidden parallelism or better instruction packing.

**Key Findings**:
1. **The problem is tractable** for our kernel size (~250-400 operations per iteration)
2. **Expected improvement potential**: 5-15% cycle reduction (based on Unison paper results)
3. **Primary opportunity**: Better ALU/VALU slot packing during load-bound phases
4. **Risk**: The load slot bottleneck (2/cycle) may already force near-optimal scheduling

---

## 1. Background: Why Constraint-Based Scheduling?

### 1.1 The Scheduling Problem

In VLIW architectures, the compiler/programmer must:
1. Respect data dependencies between operations
2. Fit operations into limited slots per cycle
3. Minimize total cycles

This is NP-hard in general, but tractable for moderate-sized basic blocks.

### 1.2 Prior Art: Unison Framework

The Unison framework (arxiv.org/abs/1804.02452) demonstrated:
- Combined register allocation + instruction scheduling via ILP
- 1.1-10% speedup on Hexagon DSP (VLIW architecture)
- Practical solving times for basic blocks up to ~500 operations

### 1.3 Our Hypothesis

H54's hand-crafted schedule may leave performance on the table:
- Sub-optimal operation packing in cycles with slack
- Unnecessary serialization between independent operations
- Non-optimal interleaving of gather and compute phases

---

## 2. Operation Analysis of H54

### 2.1 Main Loop Structure

The H54 main loop (per iteration) consists of:

```
PHASE 1: Address Calculation (ALU-heavy)
  - 16 offset calculations: offset_regs[d] = batch_offset + offset_consts[d]
  - 32 address calculations for input loads
  - Cycles: 5 (current), could be 3 (optimal)

PHASE 2: Input Loading (LOAD-bound)
  - 32 vloads (16 desks x 2 vectors each)
  - Cycles: 16 (at 2/cycle) - HARD MINIMUM

PHASE 3: Gather Address Prep (VALU-heavy)
  - 16 vbroadcasts of forest_values_p
  - 16 vector additions (addr = addr + idx)
  - Cycles: 6 (current) - could overlap with Phase 2

PHASE 4: Round 1 Gather + Hash (INTERLEAVED)
  - 128 scalar loads (16 desks x 8 lanes)
  - 19 VALU ops per desk x 16 desks = 304 VALU ops
  - Cycles: 64 load + overlapped compute

PHASE 5: Round 2 (same as Round 1)

PHASE 6: Store (STORE-bound)
  - 32 vstores
  - Cycles: 16 (at 2/cycle) - HARD MINIMUM

PHASE 7: Loop Control
  - 2-3 ALU ops + 1 flow
  - Cycles: 3
```

### 2.2 Operation Counts per Iteration

| Category | Operation | Count | Slot Type |
|----------|-----------|-------|-----------|
| Address calc | ALU add | 48 | ALU |
| Input load | vload | 32 | Load |
| Addr prep | vbroadcast | 16x2 | VALU |
| Addr prep | vadd | 16x2 | VALU |
| Gather | scalar load | 256 | Load |
| XOR | vxor | 32 | VALU |
| Hash FMA | multiply_add | 96 | VALU |
| Hash XOR ops | vxor, vshift | 192 | VALU |
| Hash combine | vxor | 96 | VALU |
| Branch calc | vand, vadd, vmul | 160 | VALU |
| Bounds check | vcmp, vmul | 64 | VALU |
| Store | vstore | 32 | Store |
| Loop control | ALU | 4 | ALU |
| Loop control | flow | 2 | Flow |

**Total: ~1,028 slot-operations per iteration**

### 2.3 Resource Utilization Analysis

Current H54 achieves ~217 cycles per iteration (3,462 / 16 iterations).

Theoretical minimums (if resources were not limited):
- Load-bound: (32 + 256) / 2 = 144 cycles
- Store-bound: 32 / 2 = 16 cycles
- VALU-bound: ~704 ops / 6 slots = 118 cycles

**Critical path**: Load operations (144 cycles minimum per iteration)

### 2.4 Slack Analysis

In 217 cycles with 144 load-cycles:
- ALU slack: 217 * 12 - 52 = 2,552 unused ALU slots
- VALU slack: 217 * 6 - 704 = 598 unused VALU slots
- Store slack: (217 - 16) * 2 = 402 unused store slots

**Key insight**: We have massive ALU slack and significant VALU slack during load phases.

---

## 3. ILP Formulation

### 3.1 Decision Variables

For each operation `i`:
- `t[i]` = start time (cycle) of operation i, integer >= 0
- `r[i,j]` = 1 if operation i uses resource slot j, 0 otherwise (only if we need explicit slot assignment)

### 3.2 Sets and Parameters

```
OPERATIONS = {op_0, op_1, ..., op_n}
RESOURCES = {ALU, VALU, LOAD, STORE, FLOW}
SLOTS[r] = max slots per cycle for resource r
LATENCY[i] = latency of operation i (typically 1 for this architecture)
DEPS = {(i, j) : operation j depends on result of operation i}
RESOURCE[i] = which resource operation i uses
```

### 3.3 Constraints

**1. Dependency Constraints**:
For each (i, j) in DEPS:
```
t[j] >= t[i] + LATENCY[i]
```

**2. Resource Constraints** (using cumulative scheduling):
For each resource r and each cycle c:
```
sum(x[i,c] for i where RESOURCE[i] == r) <= SLOTS[r]
```
where `x[i,c] = 1` if operation i executes at cycle c.

Alternatively, using time-indexed formulation:
```
For all c: sum_{i: RESOURCE[i]=r} [t[i] == c] <= SLOTS[r]
```

**3. Makespan Variable**:
```
makespan >= t[i] + LATENCY[i] for all i
```

### 3.4 Objective Function

```
minimize makespan
```

### 3.5 Problem Size Estimate

For one iteration:
- Variables: ~1,028 operations x 1 start time = 1,028 integers
- Binary variables for time-indexed: 1,028 x 250 (cycles) = 257,000 binaries
- Dependency constraints: ~1,500 (sequential dependencies)
- Resource constraints: 250 cycles x 5 resources = 1,250

**This is tractable for modern ILP solvers (Gurobi, CPLEX, CBC).**

---

## 4. Dependency Graph Analysis

### 4.1 Critical Path Identification

The longest dependency chain per desk:

```
vload(idx) -> vadd(addr) -> load(node_val[0..7]) -> vxor ->
FMA0 -> XOR1 -> FMA2 -> XOR3 -> FMA4 -> XOR5 ->
branch_calc -> bounds_check -> vstore
```

Length: 1 + 1 + 8 + 1 + 6 + 4 + 2 + 1 = 24 cycles minimum per desk

For 16 desks with 2 rounds: 24 * 2 = 48 cycles (if perfectly parallelized)

**Gap**: Current 217 cycles vs 48 cycle critical path = 4.5x overhead

### 4.2 Parallelism Available

At any moment, independent operations across desks can execute:
- 16 parallel desk pipelines
- Each at different stages of their hash computation

The ILP can find the optimal interleaving.

### 4.3 Bottleneck Analysis

The true bottleneck is load bandwidth:
- 288 loads per iteration / 2 slots = 144 cycles minimum
- Plus 16 store cycles = 160 cycles
- Plus ~5 cycles setup/loop

**Theoretical floor: ~165-170 cycles per iteration**

Current: 217 cycles (28% above theoretical floor)

---

## 5. Implementation Approach

### 5.1 Using PuLP (Open Source)

```python
from pulp import *

def build_schedule_ilp(operations, dependencies, resources):
    """
    Build ILP for optimal scheduling.

    operations: list of (op_id, resource_type, latency)
    dependencies: list of (src_op, dst_op)
    resources: dict of resource -> max_slots
    """
    # Create problem
    prob = LpProblem("VLIW_Schedule", LpMinimize)

    # Time horizon (conservative upper bound)
    T = len(operations) * 2

    # Variables: x[i,t] = 1 if op i starts at time t
    x = LpVariable.dicts("x",
                         ((i, t) for i in range(len(operations))
                          for t in range(T)),
                         cat='Binary')

    # Makespan variable
    makespan = LpVariable("makespan", lowBound=0, cat='Integer')

    # Objective: minimize makespan
    prob += makespan

    # Each operation executes exactly once
    for i in range(len(operations)):
        prob += lpSum(x[i,t] for t in range(T)) == 1

    # Dependency constraints
    for src, dst in dependencies:
        src_lat = operations[src][2]
        for t in range(T):
            # If src starts at t, dst must start at t + latency or later
            prob += lpSum(x[dst, t2] for t2 in range(t + src_lat)) <= \
                   T * (1 - x[src, t])

    # Resource constraints
    for t in range(T):
        for res, limit in resources.items():
            ops_using_res = [i for i, (_, r, _) in enumerate(operations)
                           if r == res]
            prob += lpSum(x[i, t] for i in ops_using_res) <= limit

    # Makespan constraint
    for i, (_, _, lat) in enumerate(operations):
        for t in range(T):
            prob += makespan >= (t + lat) * x[i, t]

    return prob, x, makespan
```

### 5.2 Using OR-Tools CP-SAT (Google)

```python
from ortools.sat.python import cp_model

def build_schedule_cpsat(operations, dependencies, resources, horizon):
    """
    Build CP-SAT model for optimal scheduling.
    """
    model = cp_model.CpModel()

    # Start time variables
    starts = {}
    for i, (op_id, res, lat) in enumerate(operations):
        starts[i] = model.NewIntVar(0, horizon, f'start_{op_id}')

    # Makespan
    makespan = model.NewIntVar(0, horizon, 'makespan')

    # Dependencies
    for src, dst in dependencies:
        src_lat = operations[src][2]
        model.Add(starts[dst] >= starts[src] + src_lat)

    # Resource constraints using cumulative
    for res, limit in resources.items():
        intervals = []
        for i, (op_id, r, lat) in enumerate(operations):
            if r == res:
                interval = model.NewIntervalVar(
                    starts[i], lat, starts[i] + lat, f'interval_{op_id}')
                intervals.append(interval)

        # Cumulative constraint: at most 'limit' operations at any time
        demands = [1] * len(intervals)
        model.AddCumulative(intervals, demands, limit)

    # Makespan
    for i, (_, _, lat) in enumerate(operations):
        model.Add(makespan >= starts[i] + lat)

    # Objective
    model.Minimize(makespan)

    return model, starts, makespan
```

### 5.3 Extracting Operations from H54

To build the dependency graph, we need to:

1. **Parse the instruction stream**: Extract all operations from `kb.instrs`
2. **Build dependency graph**: Track def-use chains through scratch addresses
3. **Classify by resource**: ALU, VALU, Load, Store, Flow

```python
def extract_operations(instrs):
    """
    Extract operations and build dependency graph.
    """
    operations = []
    last_def = {}  # scratch_addr -> operation index
    dependencies = []

    for cycle_idx, instr in enumerate(instrs):
        for engine, slots in instr.items():
            for slot in slots:
                op_id = len(operations)

                # Determine resource type and latency
                if engine == 'alu':
                    res, lat = 'ALU', 1
                elif engine == 'valu':
                    res, lat = 'VALU', 1
                elif engine == 'load':
                    res, lat = 'LOAD', 1
                elif engine == 'store':
                    res, lat = 'STORE', 1
                elif engine == 'flow':
                    res, lat = 'FLOW', 1

                operations.append((op_id, res, lat, slot))

                # Add dependencies based on operands
                dest, *srcs = parse_slot(slot)
                for src in srcs:
                    if src in last_def:
                        dependencies.append((last_def[src], op_id))

                # Update last_def for destinations
                if dest is not None:
                    last_def[dest] = op_id

    return operations, dependencies
```

---

## 6. Expected Results and Limitations

### 6.1 Potential Improvements

Based on Unison results (1.1-10% on VLIW):

| Scenario | Expected Improvement | Resulting Cycles |
|----------|---------------------|------------------|
| Conservative | 5% | 3,289 |
| Moderate | 10% | 3,116 |
| Optimistic | 15% | 2,943 |

### 6.2 Fundamental Limitations

The ILP cannot overcome hardware constraints:
- **Load bottleneck**: 288 loads at 2/cycle = 144 cycles minimum
- **Store requirement**: 32 stores at 2/cycle = 16 cycles
- **Total floor**: ~160 cycles per iteration = 2,560 cycles total

**Maximum possible improvement**: 3,462 -> 2,560 = 26% (if schedule is perfect)

### 6.3 Why We May Already Be Near Optimal

H54's interleaved gather-hash schedule already exploits:
- Overlapping compute with load latency
- Pipelined desk processing
- Efficient loop structure

The 28% gap to theoretical minimum may be due to:
- Necessary serialization (data dependencies)
- Control flow overhead
- Setup/cleanup phases

---

## 7. Implementation Plan

### 7.1 Phase 1: Operation Extraction

1. Write parser for H54 instruction stream
2. Build def-use dependency graph
3. Validate correctness against hand-counted operations

### 7.2 Phase 2: ILP Model

1. Implement PuLP/OR-Tools model
2. Start with single iteration (simplest scope)
3. Add constraints incrementally
4. Validate against known-good schedule

### 7.3 Phase 3: Solve and Analyze

1. Run solver with time limit (5-10 minutes)
2. Extract optimal schedule
3. Compare to H54 hand-crafted schedule
4. Identify specific improvements

### 7.4 Phase 4: Code Generation

1. Generate new instruction sequence from optimal schedule
2. Implement as H59 kernel
3. Validate correctness
4. Measure actual cycle count

---

## 8. Tractability Analysis

### 8.1 Problem Size

For one iteration of H54:
- ~1,000 operations
- ~1,500 dependency edges
- ~250 cycle time horizon

### 8.2 Solving Time Estimates

Based on Unison benchmarks:
- Basic blocks of 500 ops: seconds to minutes
- Basic blocks of 1000 ops: minutes to hours

Our problem should solve in **5-30 minutes** with CBC/CPLEX.

### 8.3 Simplifications to Improve Tractability

If full problem is too slow:
1. **Decompose by phase**: Solve each phase independently
2. **Fix partial schedule**: Keep known-good parts, optimize slack
3. **Reduce time horizon**: Use tighter bounds from heuristic solution
4. **Symmetry breaking**: Fix desk ordering within equivalent phases

---

## 9. Conclusions and Recommendations

### 9.1 Feasibility: YES

The constraint-based scheduling problem is tractable for our kernel size.

### 9.2 Expected Benefit: MODERATE (5-15%)

The load slot bottleneck limits improvement potential, but there is likely 5-15% improvement available in better slot packing.

### 9.3 Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Solver timeout | Medium | High | Decompose problem, use time limits |
| Minimal improvement | Medium | Medium | Analyze before full implementation |
| Incorrect schedule | Low | High | Validate against simulation |
| Code generation complexity | High | Medium | Start with manual translation |

### 9.4 Recommendation

**Proceed with Phase 1-3 (analysis only)** before committing to full implementation:

1. Extract operations and dependencies from H54
2. Build ILP model
3. Solve and compare to current schedule
4. **Only proceed to Phase 4 if improvement > 10%**

If ILP shows <5% improvement potential, the hand-crafted schedule is likely near-optimal given the load bottleneck.

---

## Appendix A: Hash Operation Dependencies

Each hash stage depends on the previous:

```
val_0 = val ^ node_val           (XOR)
val_1 = val_0 * 4097 + 0x7ED55D16  (FMA stage 0)
val_2 = (val_1 ^ 0xC761C23C) ^ (val_1 >> 19)  (XOR stage 1)
val_3 = val_2 * 33 + 0x165667B1  (FMA stage 2)
val_4 = (val_3 + 0xD3A2646C) ^ (val_3 << 9)   (XOR stage 3)
val_5 = val_4 * 9 + 0xFD7046C5   (FMA stage 4)
val_6 = (val_5 ^ 0xB55A4F09) ^ (val_5 >> 16)  (XOR stage 5)
```

Critical path through hash: 7 dependent operations (minimum 7 cycles per hash)

## Appendix B: Dependency Graph Visualization

```
        vload(idx) -----> vadd(addr) -----> load(node[0])
             \                                   \
              \                                   \---> load(node[7])
               \                                              |
                vload(val) --------------------------> vxor(val, node)
                                                              |
                                                         FMA_0
                                                              |
                                                         XOR_1
                                                              |
                                                         FMA_2
                                                              |
                                                         XOR_3
                                                              |
                                                         FMA_4
                                                              |
                                                         XOR_5
                                                              |
                                                       branch_calc
                                                              |
                                                       bounds_check
                                                              |
                                                         vstore
```

## Appendix C: Resource Utilization Target

Ideal utilization per iteration (217 cycles):
```
Resource | Capacity | Used  | Utilization
---------|----------|-------|------------
ALU      | 2,604    | 52    | 2.0%
VALU     | 1,302    | 704   | 54.1%
Load     | 434      | 288   | 66.4%
Store    | 434      | 32    | 7.4%
Flow     | 217      | 2     | 0.9%
```

**Observation**: ALU is massively underutilized. Could we convert some VALU ops to use ALU lanes? This would require ISA investigation.

## Appendix D: Quick Win Analysis

Before full ILP implementation, check for obvious improvements:

1. **Phase 3 overlap with Phase 2**: Can we start vbroadcasts before all vloads complete?
2. **Store overlap with Round 2 cleanup**: Can stores begin before all hash ops finish?
3. **ALU utilization during gathers**: Can we precompute next iteration addresses?

These heuristic improvements might capture most of the ILP potential without solver overhead.
