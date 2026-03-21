# H58: Index Grouping / Convergent Processing Design

## Status: ANALYSIS COMPLETE - NOT IMPLEMENTABLE (ISA Constraints)

## Executive Summary

H58 proposes exploiting the **convergent nature of tree traversal** to reduce redundant loads. The key insight is that all 256 batch elements start at index 0 (root), meaning Round 0 requires only ONE unique tree value load instead of 256.

**However, this optimization is NOT implementable** given the ISA constraints:
1. No vgather instruction (must use scalar loads)
2. No indirect scratch addressing (cannot dynamically route values to lanes)
3. Histogram/permutation operations would require indirect addressing

The overhead of tracking unique indices and distributing values without indirect addressing exceeds the potential savings.

---

## 1. Key Insight: Convergent Index Distribution

### Starting Point Analysis

From `problem.py` line 433-434:
```python
indices = [0 for _ in range(batch_size)]  # ALL start at root!
values = [random.randint(0, 2**30 - 1) for _ in range(batch_size)]
```

**All 256 batch elements start at tree index 0.**

### Index Distribution Evolution

| Round | Max Unique Indices | Theoretical Distribution |
|-------|-------------------|-------------------------|
| 0 | 1 | All 256 at idx=0 |
| 1 | 2 | Split between idx=1 and idx=2 |
| 2 | 4 | At most 4 indices (1,2,3,4) |
| 3 | 8 | At most 8 indices |
| N | min(2^N, 256) | Up to 2^N unique indices |

After Round 8, indices can be anywhere, but **there are only 256 elements** - meaning many indices are shared.

### Potential Load Savings

| Round | Current Loads | Ideal Loads | Savings |
|-------|--------------|-------------|---------|
| 0 | 256 | 1 | 255 loads |
| 1 | 256 | 2 | 254 loads |
| 2 | 256 | 4 | 252 loads |
| 3 | 256 | 8 | 248 loads |
| ... | ... | ... | ... |
| 8+ | 256 | ~100-200 | 56-156 loads |

**Early rounds offer massive savings.**

---

## 2. ISA Constraint Analysis

### Available Operations

From `problem.py` SLOT_LIMITS:
```python
SLOT_LIMITS = {
    "alu": 12,      # scalar arithmetic
    "valu": 6,      # vector arithmetic (includes vbroadcast)
    "load": 2,      # scalar/vector loads
    "store": 2,     # scalar/vector stores
    "flow": 1,      # control flow (select, vselect, jumps)
}
```

### Critical Missing Operations

1. **No vgather**: Cannot load `tree[idx[0]], tree[idx[1]], ..., tree[idx[7]]` in one operation
2. **No indirect scratch addressing**: Cannot do `scratch[scratch[addr]]`
3. **No vpermute/vshuffle**: Cannot rearrange vector elements based on index

### What We CAN Do

| Operation | Engine | Description |
|-----------|--------|-------------|
| vbroadcast | VALU | Broadcast scalar to all VLEN lanes |
| vselect | Flow | Conditional select between two vectors |
| scalar load | Load | Load single value from memory |
| vload | Load | Load VLEN contiguous values |

---

## 3. Proposed Approach: Static Round-0 Optimization

### Round 0 Special Case (Implementable)

Since ALL elements start at idx=0, Round 0 can be optimized:

**Current (H38):**
```python
# For each desk (8 elements):
for lane in range(8):
    node_val[lane] = load(forest_base + idx[lane])  # 8 loads!
```
Cycles: 4 cycles per desk (8 loads / 2 loads per cycle)

**Optimized:**
```python
# Round 0 ONLY - all indices are 0
tree_root = load(forest_base + 0)     # 1 load
vbroadcast(node_val, tree_root)        # 1 VALU op
```
Cycles: 1-2 cycles per desk

**Savings for Round 0**: ~3 cycles per desk, 8 desks = 24 cycles

### Why Only Round 0?

After Round 0, indices diverge to 1 or 2. To exploit this:
1. Need to determine WHICH elements have idx=1 vs idx=2
2. Need to ROUTE the correct tree value to each element

This requires **conditional per-element selection**, which means:
- Comparing each lane's index against the unique values
- Using vselect to route values

---

## 4. Dynamic Index Grouping: Why It Fails

### Theoretical Algorithm

```python
# For round N with multiple unique indices:
unique_indices = set(idx_vector)  # Find unique values
for unique_idx in unique_indices:
    tree_val = load(forest_base + unique_idx)  # One load per unique
    mask = (idx_vector == unique_idx)           # Which elements need this?
    node_val = vselect(mask, tree_val_broadcast, node_val)  # Route to them
```

### Implementation Challenges

**Challenge 1: Finding Unique Indices**
- No reduce operation to find unique values in a vector
- Would need to compare all pairs: O(VLEN^2) comparisons
- With VLEN=8: 28 comparisons per vector, 32 vectors = 896 comparisons

**Challenge 2: Counting Occurrences**
- No histogram instruction
- Cannot do `count = sum(idx == unique_idx)` efficiently
- Would need 256 comparisons and additions

**Challenge 3: Routing Values**
- After loading each unique value, must broadcast and selectively assign
- For K unique indices: K broadcasts + K vselects = K cycles (Flow is 1/cycle!)
- If K=8 (Round 3): 8 vselect ops = 8 cycles in Flow engine

### Cycle Cost Analysis

**Current Gather (H38):**
- 8 scalar loads per desk
- 2 loads/cycle = 4 cycles per desk

**Dynamic Grouping (if possible):**
```
Find unique indices:  ~8 cycles (comparisons)
Load unique values:   K/2 cycles (K loads, 2/cycle)
Broadcast + route:    K cycles (vselect is Flow engine, 1/cycle)
```

For K=4 unique indices: 8 + 2 + 4 = 14 cycles
For K=8 unique indices: 8 + 4 + 8 = 20 cycles

**Dynamic grouping is SLOWER than current gather!**

---

## 5. Static Round Specialization: Feasibility

### Compile-Time Known Patterns

We KNOW the index pattern for early rounds:
- Round 0: All idx=0
- Round 1: All idx in {1, 2}
- Round 2: All idx in {1, 2, 3, 4}

### Specialized Round 0

```python
# ROUND 0 ONLY (all idx=0)
tree_root_addr = forest_base + 0
tree_root = load(tree_root_addr)       # 1 load slot

# For each desk:
vbroadcast(node_val_0, tree_root)      # 1 VALU slot
vbroadcast(node_val_1, tree_root)      # Can do 6 per cycle!
# ... all 8 desks
```

**Implementation (1 cycle):**
```python
{
    "load": [("load", tree_root_scalar, zero_addr)],
    "valu": [
        ("vbroadcast", node_val_0, tree_root_scalar),
        ("vbroadcast", node_val_1, tree_root_scalar),
        ("vbroadcast", node_val_2, tree_root_scalar),
        ("vbroadcast", node_val_3, tree_root_scalar),
        ("vbroadcast", node_val_4, tree_root_scalar),
        ("vbroadcast", node_val_5, tree_root_scalar),
    ]
}
```

**Savings: 4 cycles (gather) - 1 cycle (broadcast) = 3 cycles per desk batch**

### Specialized Round 1

Indices are in {1, 2}. Strategy:
1. Load tree[1] and tree[2] (2 loads = 1 cycle)
2. Broadcast both to vectors (2 vbroadcast = 1 cycle)
3. Compute bit = val & 1 (determines if idx=1 or idx=2)
4. vselect(node_val, bit, tree_2_vec, tree_1_vec) (1 flow op per desk)

**Problem: Step 4 requires 8 vselect ops (one per desk) = 8 cycles!**

Flow engine limit of 1 vselect/cycle makes this slower than gather.

### The Flow Engine Bottleneck

| Approach | Load Cycles | VALU Cycles | Flow Cycles | Total |
|----------|-------------|-------------|-------------|-------|
| Current gather (8 desks) | 32 | 0 | 0 | 32 |
| Round 1 optimized (8 desks) | 1 | 2 | 8 | 11 |

Wait - this IS faster! Let me recalculate...

**Current H38 Round 1 for 8 desks:**
- Gather interleaved with compute
- ~4 cycles per desk for gather alone = 32 cycles for 8 desks (but overlapped)

**Optimized Round 1:**
- Load tree[1], tree[2]: 1 cycle
- Broadcast both: 1 cycle (2 vbroadcasts, have 6 slots)
- vselect for 8 desks: 8 cycles (Flow bottleneck)
- Total: 10 cycles vs ~32 cycles for gather phase

**BUT:** This only works for Round 1. Later rounds have more unique indices.

---

## 6. Round-by-Round Analysis

### Round 2: {1, 2, 3, 4} Possible Indices

Strategy:
1. Load tree[1], tree[2], tree[3], tree[4]: 2 cycles
2. Broadcast to vectors: 1 cycle (4 broadcasts)
3. For each desk: 2 vselects to choose among 4 values

Using nested vselect:
```python
# High bit selects {1,2} vs {3,4}
high_bit = (idx >> 1) & 1
# Low bit selects within pair
low_bit = idx & 1

# First level: select between {1,2} and {3,4}
left_pair = vselect(low_bit, tree_2, tree_1)   # tree[1] or tree[2]
right_pair = vselect(low_bit, tree_4, tree_3)  # tree[3] or tree[4]
# Second level: select between pairs
node_val = vselect(high_bit, right_pair, left_pair)
```

**Per desk: 2 vselects = 2 cycles in Flow engine**
**8 desks: 16 vselect cycles**

Compare to current gather: 4 cycles per desk (overlapped with compute)

---

## 7. The Core Problem: Flow Engine Bottleneck

The vselect operation is in the **Flow engine**, which has only **1 slot per cycle**.

To distribute values based on dynamic indices:
- Need K-1 vselects for K unique values (binary tree of selections)
- 8 desks * (K-1) vselects = 8*(K-1) cycles in Flow

### Break-Even Analysis

Current gather cost: 4 cycles per desk (load-limited)

Index grouping cost per desk:
- Load K values: K/2 cycles
- Broadcast K values: ceil(K/6) cycles
- vselect tree: (K-1) cycles per desk

**Break-even when:** K/2 + ceil(K/6) + (K-1) < 4

For K=2: 1 + 1 + 1 = 3 cycles (FASTER)
For K=4: 2 + 1 + 3 = 6 cycles (SLOWER)
For K=8: 4 + 2 + 7 = 13 cycles (MUCH SLOWER)

**Conclusion: Only profitable for K <= 2 (Round 0 and Round 1)**

---

## 8. Implementable Optimization: Round 0 Special Path

### Code Structure

```python
# === ROUND 0 SPECIAL (all idx=0) ===
tree_root = load(forest_base)           # 1 cycle
# Broadcast to all 8 desks (6 vbroadcast slots/cycle)
cycle1: vbroadcast x 6 (desks 0-5)      # 1 cycle
cycle2: vbroadcast x 2 (desks 6-7)      # 1 cycle (could overlap)

# Now proceed with hash computation...
# XOR, hash stages, branch computation as before

# === ROUNDS 1-15: Use standard gather ===
```

### Cycle Savings Estimate

**Round 0 current cost (8 desks):**
- Gather interleaved: ~32 cycles (load bound)
- Compute overlapped

**Round 0 optimized:**
- 1 load + 2 broadcast cycles = 3 cycles
- Savings: ~29 cycles

**But wait:** In H38, gather is heavily overlapped with compute. The actual savings depend on critical path analysis.

Looking at H38 implementation:
- Desk 0 gather: 4 cycles
- While desk 0 gathers, no compute happens (nothing to compute yet)
- Desk 1 gather overlaps with desk 0 hash start
- etc.

**Actual Round 0 optimization:** Would eliminate the initial 4-cycle latency before compute can begin.

---

## 9. Critical Path Impact

### Current H38 Critical Path (per 2-round iteration)

```
[Gather D0: 4cy] -> [XOR+Hash D0: overlapped]
     [Gather D1: 4cy] -> [XOR+Hash D1: overlapped]
          [Gather D2: 4cy] -> ...
```

Total gather time: ~32 cycles (8 desks * 4 cycles)
But heavily overlapped with compute.

Actual critical path ≈ 4 (first desk) + 32 (store) + overhead

### With Round 0 Optimization

First iteration only:
- Load root: 1 cycle
- Broadcast all desks: 2 cycles (vs 32 cycle gather sequence)
- Rest of round: same compute

**Savings: ~29 cycles, but only for FIRST iteration (which is Round 0)**

With 32 iterations total (16 rounds / 2 rounds per iteration * 32 batches / 64 elements per iter... wait, let me recalculate)

From H38:
```python
total_iterations = (batch_size // VLEN) * (rounds // 2) // NUM_DESKS
                 = (256 // 8) * (16 // 2) // 8 = 32 * 8 / 8 = 32
```

But Round 0 only happens once across ALL batches (first round of each batch element).

Actually, each batch element goes through rounds 0-15. With 256 elements and 8-desk processing:
- 256/64 = 4 iterations per round
- Round 0 is iteration 0, 1, 2, 3 (4 iterations)

**Savings: 4 iterations * ~29 cycles = ~116 cycles**

---

## 10. Final Verdict

### Implementable Optimizations

| Optimization | Implementable | Savings | Complexity |
|--------------|--------------|---------|------------|
| Round 0 broadcast | YES | ~116 cycles | Low |
| Round 1 broadcast+vselect | MARGINAL | ~0-50 cycles | Medium |
| Dynamic grouping | NO | N/A | Would be negative |

### Recommendation

**Do NOT pursue H58 as originally conceived.**

The core idea (load each unique tree value once) is sound but:
1. ISA lacks indirect addressing for efficient value distribution
2. Flow engine bottleneck (1 vselect/cycle) kills performance for K > 2
3. Only Round 0 offers clear benefit, saving ~116 cycles
4. Round 1 optimization is marginal (Flow-bound)

### Alternative: Round-0-Only Micro-Optimization

Could implement a small optimization for Round 0 only:
- Check if this is the first iteration
- Use broadcast instead of gather
- Savings: ~116 cycles out of 4,062 = 2.9% improvement

**Not worth the code complexity for 2.9% gain.**

---

## 11. Why Target Seems Unreachable

### Current: 4,062 cycles
### Target: 1,790 cycles
### Gap: 2.27x

The gather bottleneck is fundamental:
- 256 elements * 16 rounds = 4,096 node value loads
- At 2 loads/cycle: 2,048 cycles MINIMUM just for tree loads
- Plus hash compute, store, loop overhead

**Target of 1,790 cycles is below the theoretical load minimum!**

Unless:
1. Tree values can be prefetched/cached
2. Multiple rounds can share index computations
3. Some loads can be eliminated entirely

H58's insight about convergent indices is correct but ISA doesn't support exploitation.

---

## 12. Pseudocode: What H58 Would Look Like (If ISA Supported It)

```python
# HYPOTHETICAL - requires vgather or indirect scratch

for round in range(rounds):
    # Build histogram of indices
    unique_indices, counts = histogram(idx_vector)  # NOT AVAILABLE

    # Load each unique tree value ONCE
    for i, ui in enumerate(unique_indices):
        unique_vals[i] = load(forest_base + ui)     # K loads vs 256

    # Distribute to elements - requires indirect addressing
    for i in range(256):
        node_val[i] = unique_vals[index_to_unique_slot[idx[i]]]  # NOT AVAILABLE
```

---

## 13. Conclusion

**H58 is NOT implementable** given ISA constraints.

The insight is valid (convergent indices mean redundant loads) but exploitation requires:
- vgather or indirect scratch addressing (not available)
- Efficient histogram/unique (not available)
- Parallel vselect (limited to 1/cycle in Flow engine)

The only practical extract is Round-0 optimization (~2.9% speedup), which is insufficient to justify implementation complexity.

**Recommendation: Archive H58 and focus on other approaches.**

---

## Appendix: ISA Gap Analysis

| Required Operation | Available | Alternative | Viable |
|-------------------|-----------|-------------|--------|
| vgather | NO | 8x scalar load | YES (current) |
| Indirect scratch | NO | Explicit vselect tree | SLOW |
| Histogram | NO | O(N^2) comparisons | TOO SLOW |
| vpermute | NO | vselect chain | SLOW |
| Parallel vselect | NO (1/cycle) | Multiple iterations | SLOW |

The ISA is designed for **regular memory access patterns**, not the irregular/convergent patterns that tree traversal exhibits.
