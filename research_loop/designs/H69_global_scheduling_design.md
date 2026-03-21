# H69: Global Instruction Scheduling Design

## Executive Summary

**Current Best**: H68v2 at 2,775 cycles (53.2x speedup)
**Target**: 1,790 cycles (82x speedup)
**Gap**: 985 cycles (55% above target)

**Key Insight**: The fundamental difference between our approach and PR #22's 1,299 cycle solution is **architectural**:

| Aspect | Our Approach (H68) | PR #22 Approach |
|--------|-------------------|-----------------|
| Scheduling | Per-iteration, local | **Global**, all operations |
| Desk count | 16 (deep pipeline) | Likely 4-8 (simpler) |
| Round processing | 16 desks x 2 rounds | **Cross-round interleaving** |
| Operation packing | Manual, heuristic | **Greedy list scheduler** |

This document designs a **global instruction scheduling** approach that fundamentally restructures how we emit and pack VLIW bundles.

---

## 1. Problem Analysis: Why Local Scheduling Fails

### 1.1 Current H68 Approach

```python
# H68 Structure (simplified)
for tile in [0, 1]:                    # 2 tiles of 128 elements
    load_idx_val_for_16_desks()        # 16 cycles
    for round in range(16):
        for desk in range(16):         # Deep pipeline
            # Tightly scheduled per-desk:
            emit_gather(desk)          # 4 cycles
            emit_hash(desk)            # Interleaved with next desk's gather
            emit_branch(desk)
        # Finish remaining operations
    store_idx_val_for_16_desks()       # 16 cycles
```

**Problem**: Each desk's operations are scheduled **locally** within the iteration. The scheduler tries to overlap desk N's hash with desk N+1's gather, but:
1. Dependencies within a desk are rigid (gather -> XOR -> hash -> branch -> gather)
2. Limited look-ahead: only sees one round at a time
3. No cross-round optimization opportunities

### 1.2 What PR #22 Does Differently

Based on the analysis, PR #22 likely uses:

```python
# PR #22 Structure (hypothesized)
operations = []
for round in range(16):
    for element_batch in range(batch_size // VLEN):
        operations.extend([
            Op("gather", round, batch, depends_on=[...]),
            Op("xor", round, batch, depends_on=[gather]),
            Op("hash_stage_0", round, batch, depends_on=[xor]),
            # ... all hash stages
            Op("branch", round, batch, depends_on=[hash_5]),
        ])

# Global greedy list scheduler
scheduled_program = pack_vliw_bundles(operations)
```

**Key Difference**: All operations across ALL rounds and elements are emitted as a flat list with dependencies, then a greedy scheduler packs them into VLIW cycles.

---

## 2. Global Scheduling Algorithm

### 2.1 Operation Graph Construction

**Step 1**: Define all operations for the kernel

For 256 elements, 16 rounds, processing in vectors of 8:
- 32 vectors per round
- 16 rounds
- ~20 operations per vector-round (gather + hash + branch)
- Total: 32 x 16 x 20 = **10,240 operations**

**Step 2**: Encode dependencies

```
gather[r,v] -> xor[r,v] -> hash_0[r,v] -> hash_1[r,v] -> ... -> branch[r,v] -> gather[r+1,v]
```

The key insight: **operations from different vectors/rounds are independent** until they share resources!

### 2.2 Greedy List Scheduler

```python
def greedy_schedule(operations, slot_limits):
    """
    Pack operations into VLIW cycles respecting slot limits and dependencies.

    slot_limits = {
        "load": 2,
        "store": 2,
        "alu": 12,
        "valu": 6,
        "flow": 1
    }
    """
    ready_queue = [op for op in operations if op.dependencies_met()]
    schedule = []
    current_cycle = []
    current_slots = {engine: 0 for engine in slot_limits}

    while ready_queue or current_cycle:
        # Try to add operations to current cycle
        for op in list(ready_queue):
            if current_slots[op.engine] < slot_limits[op.engine]:
                current_cycle.append(op)
                current_slots[op.engine] += 1
                ready_queue.remove(op)

        # Commit cycle and advance
        schedule.append(current_cycle)
        update_dependencies(current_cycle)  # Mark outputs as available
        ready_queue.extend(newly_ready_operations())
        current_cycle = []
        current_slots = {engine: 0 for engine in slot_limits}

    return schedule
```

### 2.3 Why Global Beats Local

**Example**: Processing 2 vectors through 2 rounds

**Local scheduling (H68 style)**:
```
Cycle 1: gather[r0,v0]  # Use 2 load slots
Cycle 2: gather[r0,v0]  # Still loading
Cycle 3: gather[r0,v0]
Cycle 4: gather[r0,v0], xor[r0,v0]  # Hash starts
Cycle 5: hash_0[r0,v0], gather[r0,v1]  # Next vector starts
... (sequential pattern continues)
```

**Global scheduling**:
```
Cycle 1: gather[r0,v0], gather[r0,v1]  # Use ALL 2 load slots immediately!
Cycle 2: gather[r0,v0], gather[r0,v1]
Cycle 3: gather[r0,v0], gather[r0,v1]
Cycle 4: gather[r0,v0], gather[r0,v1], xor[r0,v0], xor[r0,v1]
Cycle 5: hash_0[r0,v0], hash_0[r0,v1], gather[r1,v0]...
```

**Key advantage**: Global scheduling can pack operations from **different rounds** into the same cycle, achieving better slot utilization.

---

## 3. Round-Centric vs Element-Centric Processing

### 3.1 Current Element-Centric (H68)

```
Process 128 elements through round 0
Process 128 elements through round 1
...
Process 128 elements through round 15
Store results
Process next 128 elements...
```

**Characteristics**:
- Each element completes all rounds before memory write
- Large scratch footprint (16 desks x 6 vectors each = 768 vector registers)
- Deep pipeline hides gather latency

### 3.2 Proposed Round-Centric

```
Process ALL 256 elements through round 0
Process ALL 256 elements through round 1
...
Process ALL 256 elements through round 15
Store results
```

**Characteristics**:
- All elements at same round = shared tree level accesses
- Smaller scratch footprint per "work unit"
- Can exploit index locality better

**However**: Round-centric has the broadcast overhead problem (from H60 analysis).

### 3.3 Hybrid Round-Centric with Tiling

**Proposed Architecture**:

```
For tile in [0, 1, 2, 3]:  # 4 tiles of 64 elements each
    For round in [0..15]:
        For vector in tile*8..(tile+1)*8:  # 8 vectors per tile
            emit_operation_graph(round, vector)
    Store tile results
```

**Why 4 tiles?**
- 64 elements = 8 vectors = manageable scratch (~400 words per tile)
- 4 tiles x 16 rounds = 64 round-tiles to schedule globally
- Smaller scope than full 256 elements reduces broadcast overhead

---

## 4. Desk Count Analysis

### 4.1 Current H68: 16 Desks

**Pros**:
- Very deep pipeline (16 gathers in flight)
- Excellent latency hiding
- Proven to work (2,775 cycles)

**Cons**:
- 16 desks x 48 words = 768 words scratch (50% of total)
- Complex scheduling
- Local optimization only

### 4.2 Proposed: 4-8 Desks with Global Scheduling

**With 4 desks** (32 elements in flight):
- Scratch: 4 desks x 48 words = 192 words (12.5% of total)
- More room for tree preloading, temporaries
- Simpler operation graph

**With 8 desks** (64 elements in flight):
- Scratch: 8 desks x 48 words = 384 words (25% of total)
- Good balance of parallelism and simplicity

**Key insight**: With global scheduling, fewer desks can achieve similar throughput because:
1. Operations from different rounds fill idle slots
2. Better overall utilization of VALU (currently ~50%)
3. Load bottleneck (2/cycle) is unchanged

### 4.3 Theoretical Load Analysis

**Problem size**: 256 elements x 16 rounds = 4,096 element-rounds

**Load requirements** (with level preloading):
- Rounds 0-3: ~32 loads (selection from preloaded)
- Rounds 4-10: 7 rounds x 256 gathers = 1,792 gathers = 896 cycles
- Rounds 11-14: Depends on bounds wrap exploitation
- Round 15: 256 gathers = 128 cycles

**Minimum gather cycles**: ~1,024 cycles (without optimization)

**With round 11-14 optimization**: If indices cluster at 0, we could use preloaded values:
- Rounds 11-14: ~128 cycles (broadcast + select)
- Total: 896 + 128 + 128 = 1,152 cycles minimum for node access

---

## 5. Memory Model for Global Scheduling

### 5.1 Scratch Layout

**Fixed allocations** (always present):
```
| Region | Start | Size | Purpose |
|--------|-------|------|---------|
| Constants | 0 | 50 | Hash constants, multipliers |
| Tree preload | 50 | 128 | Tree nodes 0-14 (broadcast) |
| Temporaries | 178 | 100 | Scalar temps, addresses |
```

**Per-tile allocations** (reused across tiles):
```
| Region | Start | Size | Purpose |
|--------|-------|------|---------|
| idx_vecs | 278 | 64 | 8 vectors x 8 lanes |
| val_vecs | 342 | 64 | 8 vectors x 8 lanes |
| node_vecs | 406 | 64 | 8 vectors x 8 lanes |
| tmp1_vecs | 470 | 64 | 8 vectors x 8 lanes |
| tmp2_vecs | 534 | 64 | 8 vectors x 8 lanes |
| addr_vecs | 598 | 64 | 8 vectors x 8 lanes |
```

**Total**: ~662 words (43% of 1,536)

This leaves **874 words free** for additional optimizations.

### 5.2 Preloaded Tree Nodes

**Nodes 0-14** cover:
- Round 0: All at index 0 (tree[0])
- Round 1: Indices {1, 2} (tree[1], tree[2])
- Round 2: Indices {3, 4, 5, 6} (tree[3-6])
- Round 3: Indices {7-14} (tree[7-14])

**After bounds wrap (rounds 11-14)**:
Most indices reset to 0, then spread to 1-14 over subsequent rounds.

**Preload cost**: 15 scalar loads + 15 vbroadcasts = ~15 cycles (one-time)

---

## 6. Operation Emission Strategy

### 6.1 Operation Types

```python
class Op:
    engine: str  # "load", "store", "alu", "valu", "flow"
    type: str    # "gather_lane", "vbroadcast", "fma", etc.
    round: int
    vector: int
    lane: int = None  # For per-lane operations
    dependencies: List[Op]
```

**Per-vector operations** (for a single round):
```
1. gather_prepare: vbroadcast forest_values_p to addr_vec (1 VALU)
2. gather_add: addr_vec = addr_vec + idx_vec (1 VALU)
3. gather_load: 8x load node_val_vec[lane] from addr_vec[lane] (8 loads, 4 cycles)
4. xor: val_vec = val_vec ^ node_val_vec (1 VALU or 8 ALU)
5. hash_stage_0: val = val * 4097 + const (1 VALU FMA)
6. hash_stage_1a: tmp1 = val ^ const (1 VALU)
7. hash_stage_1b: tmp2 = val >> shift (1 VALU)
8. hash_stage_1c: val = tmp1 ^ tmp2 (1 VALU)
... (repeat for stages 2-5)
12. branch_and: tmp1 = val & 1 (1 VALU)
13. branch_fma: idx = idx * 2 + 1 (1 VALU FMA)
14. branch_add: idx = idx + tmp1 (1 VALU)
15. bounds_cmp: tmp1 = idx < n_nodes (1 VALU)
16. bounds_mul: idx = idx * tmp1 (1 VALU)
```

**Total**: ~20 operations per vector-round

### 6.2 Dependency Graph

```
round_r_vec_v:
  gather_prepare -> gather_add -> gather_load[0..7]

  gather_load[7] -> xor -> hash_0 -> hash_1a, hash_1b
  hash_1a, hash_1b -> hash_1c -> hash_2 -> ...
  ... -> hash_5c -> branch_and, branch_fma
  branch_and, branch_fma -> branch_add -> bounds_cmp -> bounds_mul

  bounds_mul[round_r] -> gather_prepare[round_r+1]  # Cross-round dependency
```

### 6.3 Round Boundaries

**Key observation**: Round r+1's gather depends on round r's bounds check completing.

However, operations from **different vectors** in round r+1 can start as soon as their corresponding vector's round r is done.

This enables **cross-vector and cross-round interleaving**:
```
Cycle N:   vec0/r0/hash_5, vec1/r0/hash_4, vec2/r0/hash_3, vec0/r1/gather[0,1]
Cycle N+1: vec0/r0/branch, vec1/r0/hash_5, vec2/r0/hash_4, vec0/r1/gather[2,3]
```

---

## 7. Expected Cycle Count

### 7.1 Theoretical Minimum

**Bottleneck analysis**:

| Resource | Operations | Slots/cycle | Minimum Cycles |
|----------|-----------|-------------|----------------|
| Load | 4,096 gathers (8 lanes each) | 2 | 2,048 |
| VALU | ~20 ops x 512 vectors | 6 | 1,707 |
| Store | 64 vectors x 2 (idx, val) | 2 | 64 |
| Flow | Minimal | 1 | ~32 |

**Without optimization**: Load-bound at 2,048 cycles.

**With level preloading** (rounds 0-3 + 11-14 use selection):
- Rounds needing gather: 4-10, 15 = 8 rounds
- Gathers: 8 rounds x 32 vectors x 4 cycles = 1,024 cycles
- Plus selection overhead for 8 rounds x 32 vectors x 2 VALU = 512 ops = 86 cycles

**Theoretical floor with optimization**: ~1,100-1,200 cycles

### 7.2 Realistic Estimate

Accounting for:
- Setup/teardown: 30 cycles
- Loop overhead: 20 cycles
- Dependency stalls: 10% overhead
- Memory pipeline latency: 50 cycles

**Conservative estimate**: 1,400-1,600 cycles

**Optimistic estimate**: 1,200-1,400 cycles

This aligns with PR #22's 1,299 cycles!

### 7.3 Why H68 Achieves 2,775 Instead

**H68's inefficiencies**:
1. **Local scheduling**: Cannot interleave across rounds
2. **All rounds use gather**: No level preloading optimization
3. **16-desk overhead**: Setup/switching costs
4. **No bounds wrap exploitation**: Rounds 11-14 do full gathers

**Gap breakdown**:
- Gather savings (rounds 0-3 + 11-14): ~800 cycles
- Better slot utilization: ~200 cycles
- Reduced overhead: ~100 cycles
- **Total savings**: ~1,100 cycles

2,775 - 1,100 = 1,675 cycles (close to target!)

---

## 8. Implementation Approach

### 8.1 Phase 1: Operation Graph Generator

```python
def emit_operations(batch_size, rounds):
    """Generate all operations with dependencies."""
    ops = []
    for r in range(rounds):
        for v in range(batch_size // VLEN):
            ops.extend(emit_round_vector_ops(r, v))
    return ops

def emit_round_vector_ops(round, vector):
    """Generate operations for one vector through one round."""
    ops = []

    # Gather (or selection for preloaded rounds)
    if round < 4 or (round >= 11 and round < 15):
        ops.extend(emit_selection_ops(round, vector))
    else:
        ops.extend(emit_gather_ops(round, vector))

    # XOR and hash
    ops.extend(emit_xor_hash_ops(round, vector))

    # Branch and bounds
    ops.extend(emit_branch_ops(round, vector))

    return ops
```

### 8.2 Phase 2: Greedy List Scheduler

```python
def schedule(operations):
    """Pack operations into VLIW cycles."""
    scheduled = []
    completed = set()

    while len(completed) < len(operations):
        cycle = {}
        slot_counts = {"load": 0, "store": 0, "alu": 0, "valu": 0, "flow": 0}

        for op in operations:
            if op in completed:
                continue
            if not all(dep in completed for dep in op.dependencies):
                continue
            if slot_counts[op.engine] >= SLOT_LIMITS[op.engine]:
                continue

            # Add to current cycle
            if op.engine not in cycle:
                cycle[op.engine] = []
            cycle[op.engine].append(op.to_instruction())
            slot_counts[op.engine] += 1
            completed.add(op)

        scheduled.append(cycle)

    return scheduled
```

### 8.3 Phase 3: Optimization Passes

**Priority ordering**: Schedule high-dependency operations first
```python
def priority(op):
    return len(dependents(op))  # More dependents = higher priority
```

**Latency-aware scheduling**: Account for multi-cycle operations
```python
def cycle_ready(op, current_cycle, completed_cycles):
    for dep in op.dependencies:
        if completed_cycles[dep] + dep.latency > current_cycle:
            return False
    return True
```

---

## 9. Risk Assessment

### 9.1 High Confidence

1. **Level preloading works**: Proven in H68 for rounds 0-1
2. **Global scheduling improves utilization**: Standard compiler technique
3. **Bounds wrap happens**: Mathematically verified

### 9.2 Medium Confidence

1. **8 rounds can use selection**: Needs verification of index distribution
2. **Greedy scheduler produces near-optimal results**: May need tuning
3. **Scratch fits**: ~662 words seems safe, but complex to verify

### 9.3 Lower Confidence

1. **Achieves 1,299 cycles**: May require additional tricks not identified
2. **Implementation complexity**: Global scheduling is harder to debug
3. **Correctness**: More operations = more opportunities for bugs

---

## 10. Comparison with Alternative Approaches

### 10.1 vs H68 (Current Best)

| Aspect | H68 | H69 |
|--------|-----|-----|
| Desk count | 16 | 4-8 |
| Scheduling | Local | Global |
| Rounds with gather | 14 | 8 |
| Expected cycles | 2,775 | ~1,400 |
| Complexity | Medium | High |

### 10.2 vs Pure Round-Synchronized

| Aspect | Round-Sync | H69 |
|--------|-----------|-----|
| Element grouping | All 256 | 64 per tile |
| Broadcast overhead | High | Lower |
| Scratch pressure | Very high | Moderate |
| Expected cycles | 2,000+ | ~1,400 |

### 10.3 vs H65 Loop Tiling

| Aspect | H65 | H69 |
|--------|-----|-----|
| Tile size | 32 elements | 64 elements |
| Scheduling | Per-tile local | Global across tiles |
| Level preloading | Partial | Full |
| Expected cycles | 2,600 | ~1,400 |

---

## 11. Implementation Roadmap

### Step 1: Infrastructure (Day 1)
- Create operation class with dependency tracking
- Implement basic greedy scheduler
- Test on small example (4 elements, 2 rounds)

### Step 2: Operation Graph (Day 2)
- Implement full operation emission
- Add level preloading for rounds 0-3
- Verify dependency correctness

### Step 3: Optimization (Day 3)
- Add bounds wrap exploitation for rounds 11-14
- Tune scheduler priorities
- Profile slot utilization

### Step 4: Validation (Day 4)
- Compare against reference kernel
- Measure actual cycles
- Debug any correctness issues

---

## 12. Conclusion

### 12.1 The Core Insight

PR #22's 1,299 cycle solution fundamentally differs from our approach:
- **Global scheduling** instead of local per-desk optimization
- **Fewer desks** (likely 4-8) with better cross-round interleaving
- **Full exploitation** of level preloading and bounds wrap

### 12.2 Expected Outcome

With H69's global scheduling approach:
- **Best case**: 1,200-1,300 cycles (matching PR #22)
- **Realistic case**: 1,400-1,600 cycles (50% improvement over H68)
- **Worst case**: 2,000-2,200 cycles (still significant improvement)

### 12.3 Key Success Factors

1. **Accurate dependency modeling**: All cross-round dependencies must be correct
2. **Efficient scheduler**: Greedy with good priority function
3. **Level preloading**: Must work for rounds 0-3 and 11-14
4. **Correct bounds wrap analysis**: Verify index distribution

### 12.4 The Path to Sub-1500 Cycles

```
H68 (2,775 cycles)
    |
    v  [Add global scheduling]
H69v1 (~2,200 cycles)
    |
    v  [Add level preloading for rounds 0-3]
H69v2 (~1,800 cycles)
    |
    v  [Add bounds wrap for rounds 11-14]
H69v3 (~1,400 cycles)
    |
    v  [Tune scheduler, optimize hot paths]
H69v4 (~1,200 cycles) = TARGET ACHIEVED
```

---

## Appendix A: ISA Reference

| Engine | Slots/Cycle | Key Operations |
|--------|-------------|----------------|
| Load | 2 | load (scalar), vload (8 contiguous), const |
| Store | 2 | store (scalar), vstore (8 contiguous) |
| VALU | 6 | vbroadcast, multiply_add, +, -, *, ^, &, <<, >> |
| ALU | 12 | Same as VALU but scalar (can do 8 lanes as 8 ops) |
| Flow | 1 | vselect, select, cond_jump, jump |

**Key insight**: ALU has 12 slots. XOR and other bitwise ops can run on ALU, freeing VALU for hash stages.

---

## Appendix B: Operation Count Summary

For 256 elements, 16 rounds:

| Operation Type | Count | Engine | Cycles (serial) |
|---------------|-------|--------|-----------------|
| Gather (8 rounds) | 2,048 | Load | 1,024 |
| Selection (8 rounds) | 512 | VALU | 86 |
| XOR | 512 | ALU | 43 |
| Hash stages | 3,072 | VALU | 512 |
| Branch | 1,024 | VALU | 171 |
| Bounds check | 1,024 | VALU | 171 |
| **Total** | | | **~2,000** (serial) |

With perfect parallelism: **~1,100 cycles** (load-bound)

---

## Appendix C: Scratch Layout Diagram

```
Address: 0                                                    1535
         |--Constants--|--Tree Preload--|--Temps--|--Per-Tile State--|--Free--|
         |    50       |      128       |   100   |       384        |  874   |
```

**Per-Tile State** (reused across 4 tiles):
- idx_vecs: 8 vectors x 8 lanes = 64 words
- val_vecs: 64 words
- node_vecs: 64 words
- tmp1_vecs: 64 words
- tmp2_vecs: 64 words
- addr_vecs: 64 words
- **Total**: 384 words

---

## Appendix D: File References

- H68 Implementation: `/home/hestiasadmin/projects/original_performance_takehome/experiments/H68_hybrid_vselect/perf_takehome_h68.py`
- H67 Analysis: `/home/hestiasadmin/projects/original_performance_takehome/research_loop/designs/H67_claude_solution_analysis.md`
- H63 Missing Trick: `/home/hestiasadmin/projects/original_performance_takehome/research_loop/designs/H63_missing_trick_analysis.md`
- ISA Reference: `/home/hestiasadmin/projects/original_performance_takehome/problem.py`
