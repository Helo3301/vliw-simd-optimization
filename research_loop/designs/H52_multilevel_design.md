# H52: Multi-Level Jump Design Document

## Hypothesis Summary

**Goal**: Reduce cycle count by computing two tree levels per index update instead of one.

**Current Best**: H38 at 4,062 cycles
**Target**: 1,790 cycles
**Required Speedup**: 2.27x

## Proposed Optimization

### Current Single-Level Approach (H38)
```python
# Round N:
val_N = hash(val ^ node_val[idx])
bit_N = val_N & 1
idx = idx * 2 + 1 + bit_N

# Round N+1:
val_N1 = hash(val_N ^ node_val[idx])
bit_N1 = val_N1 & 1
idx = idx * 2 + 1 + bit_N1
```

### Proposed Multi-Level Approach
```python
# Compute two levels at once:
idx_new = idx * 4 + 3 + bit0 * 2 + bit1
```

Where the mapping is:
- Left-Left (bit0=0, bit1=0): idx*4 + 3
- Left-Right (bit0=0, bit1=1): idx*4 + 4
- Right-Left (bit0=1, bit1=0): idx*4 + 5
- Right-Right (bit0=1, bit1=1): idx*4 + 6

Formula: `idx_new = idx*4 + 3 + bit0*2 + bit1`

## Detailed Feasibility Analysis

### 1. The Fundamental Dependency Chain Problem

**Critical Issue: Sequential Hash Dependencies**

The hash function creates an unavoidable sequential dependency:

```
Round N:
  val_N = hash(val_{N-1} ^ node_val[idx_N])
  bit_N = val_N & 1
  idx_{N+1} = idx_N * 2 + 1 + bit_N

Round N+1:
  val_{N+1} = hash(val_N ^ node_val[idx_{N+1}])  <-- DEPENDS ON idx_{N+1}
  bit_{N+1} = val_{N+1} & 1
  idx_{N+2} = idx_{N+1} * 2 + 1 + bit_{N+1}
```

To compute `bit_{N+1}`, we need:
1. `idx_{N+1}` (the result of round N)
2. `node_val[idx_{N+1}]` (a gather from memory)
3. `hash(val_N ^ node_val[idx_{N+1}])` (6 hash stages)
4. Extract bit: `val_{N+1} & 1`

**This chain cannot be parallelized** because each step depends on the previous step.

### 2. Cycle Analysis of Current Round Processing

From H38 implementation analysis:

**Per-round operations (single desk):**
1. Gather node_val (4 cycles - 8 loads at 2/cycle)
2. XOR val with node_val (1 VALU)
3. Hash stages 0-5 (approximately 9 VALU operations total)
4. Branch ops: AND, FMA for idx, bounds check, apply (4-5 VALU)

**Total per round: ~10-14 cycles (compute-bound)**

### 3. Why Multi-Level Cannot Work with Current Architecture

#### Attempt 1: Speculative Execution
**Idea**: Speculatively compute both left and right paths, select based on bit.

```python
# Speculative approach:
# Assume bit0 = 0 (left branch)
idx_left = idx * 2 + 1
val_left = hash(val ^ node_val[idx_left])

# Assume bit0 = 1 (right branch)
idx_right = idx * 2 + 2
val_right = hash(val ^ node_val[idx_right])

# Actual bit0 depends on current round's hash
bit0 = (hash(val ^ node_val[idx])) & 1
idx_next = select(bit0, idx_right, idx_left)
val_next = select(bit0, val_right, val_left)
```

**Problem**: This **doubles** the work (2 gathers, 2 hash chains, 2 branch computations) for each level. No speedup, potentially slowdown due to doubled memory pressure.

#### Attempt 2: Full 4-Way Speculation for Two Levels
```python
# 4 speculative paths for 2 levels:
# L-L, L-R, R-L, R-R
idx_LL = idx * 4 + 3
idx_LR = idx * 4 + 4
idx_RL = idx * 4 + 5
idx_RR = idx * 4 + 6

# 4 gathers, 4 hash chains
val_LL = hash(hash(val ^ node_val[idx*2+1]) ^ node_val[idx_LL])
val_LR = hash(hash(val ^ node_val[idx*2+1]) ^ node_val[idx_LR])
val_RL = hash(hash(val ^ node_val[idx*2+2]) ^ node_val[idx_RL])
val_RR = hash(hash(val ^ node_val[idx*2+2]) ^ node_val[idx_RR])
```

**Problem**: 4x the gathers (16 cycles instead of 4), 4x the hash chains, 4x the register pressure. Massive slowdown.

#### Attempt 3: Pipelined Look-Ahead
**Idea**: While processing round N, pre-compute bits for round N+2.

```
Cycle timeline:
Desk 0: Round 0 gather -> hash -> bit0
Desk 1: Round 1 gather -> hash -> bit1 (needs bit0 result first!)
...
```

**Problem**: Desk 1 cannot start its gather until Desk 0 completes, because the gather address depends on `idx` which depends on `bit0`. The "pipelining" in H38 works on **different batch elements**, not on sequential rounds of the same element.

### 4. Why H38's Desk Approach is Orthogonal

H38 uses 8 desks to process 8 **different batch elements** in parallel:
- Desk 0: batch element 0
- Desk 1: batch element 8
- Desk 2: batch element 16
- etc.

Each desk processes its own element's rounds independently. The parallelism comes from ILP across different elements, not from fusing rounds of the same element.

### 5. Mathematical Proof of Infeasibility

**Theorem**: Multi-level jump cannot reduce cycles for a single element's round processing.

**Proof**:
Let T_gather = 4 cycles (8 loads at 2/cycle)
Let T_hash = 9 cycles (6 hash stages with dependencies)
Let T_branch = 4 cycles (AND, FMA, bounds, apply)

For single-level (2 rounds):
- Total = 2 * (T_gather + T_hash + T_branch) = 2 * 17 = 34 cycles per element

For multi-level attempting both at once:
- Still need T_hash for round N to get bit0
- Still need T_gather for idx_{N+1} (depends on bit0)
- Still need T_hash for round N+1 to get bit1
- **Cannot parallelize these steps**

The dependency graph is:
```
val -> hash -> bit0 -> idx_{N+1} -> gather -> val' -> hash -> bit1 -> idx_{N+2}
```

This is a linear chain with no parallelizable opportunities.

### 6. Where Could Speedup Come From?

Given the 4,062 cycle count and target of 1,790 cycles, we need different approaches:

**Option A: Reduce Hash Latency**
- Use faster hash function (fewer stages?)
- Combine hash operations more aggressively with FMA

**Option B: Increase Batch Parallelism**
- More desks (16 desks? Limited by scratch space)
- Better interleaving of gather and compute

**Option C: Prefetching / Decoupled Execution**
- Speculatively prefetch likely next nodes
- Use tree structure for locality hints

**Option D: Algorithmic Changes**
- Reorder computation across batch elements
- Trade memory for compute

## Conclusion

### Verdict: INFEASIBLE

The Multi-Level Jump optimization (H52) is **not feasible** due to fundamental data dependencies in the algorithm:

1. **Hash result determines branch direction**: `bit = hash(val ^ node_val) & 1`
2. **Branch direction determines next gather address**: `idx_next = idx * 2 + 1 + bit`
3. **Next hash requires gathered value**: `val_next = hash(val ^ node_val[idx_next])`

These form an unavoidable sequential chain that cannot be parallelized or overlapped within a single element's processing.

### Alternative Recommendation

Instead of multi-level jump, consider:

1. **H53: 16-Desk Ultra-Deep Pipeline** - Double desk count again (if scratch allows)
2. **H54: Hash Stage Fusion** - Further optimize hash computation with more aggressive FMA usage
3. **H55: Decoupled Gather-Compute** - Separate gather scheduling from compute scheduling
4. **H56: Tree Locality Exploitation** - Use tree structure for cache-aware access patterns

## Appendix: Cycle Breakdown Analysis

### H38 Current Performance
- 32 main loop iterations (256 elements / 8 desks / 2 rounds per iter)
- Per iteration: ~127 cycles (estimate from code structure)
- Total: 32 * 127 = 4,064 cycles (matches observed 4,062)

### Theoretical Minimum (Gather-Bound)
- 16 rounds * 256 elements = 4,096 round-element pairs
- Each requires 1 gather (8 loads = 4 cycles)
- Minimum cycles if only gather-bound: 4,096 * 4 / parallelism
- With perfect 8-way parallelism: 4,096 * 4 / 8 = 2,048 cycles
- Still above target of 1,790

### Achieving Target 1,790
- Requires effective parallelism of: 4,096 * 4 / 1,790 = 9.15
- Need more than 8 desks OR reduce work per element
- Round fusion (already in H38) helps but not enough
