# H66: Level Preloading with 16-Desk Loop Tiling

## Executive Summary

**Objective**: Combine H65v3's successful 16-desk loop tiling architecture (2,941 cycles) with level preloading and arithmetic value selection from H63 analysis to achieve sub-1,790 cycles.

**Current Best**: H65v3 = 2,941 cycles
**Target**: 1,790 cycles (39% reduction needed)

**Key Insight**: All 256 elements start at tree index 0. Early rounds (0-7) only access at most 255 unique tree nodes total. By preloading these nodes and using arithmetic selection instead of gathers, we can eliminate the majority of loads in the first 8 rounds.

---

## 1. Memory Budget Analysis

### 1.1 H65v3 Current Scratch Usage

From perf_takehome_h65_v3.py:

| Component | Size (words) | Notes |
|-----------|--------------|-------|
| Standard variables | 10 | tmp_scalar, addr_scalar, rounds, etc. |
| Constants | ~20 | zero, one, two, offsets, etc. |
| Vector constants (v_zero, v_one, v_two, v_n_nodes) | 32 | 4 x VLEN |
| Hash constants (6 stages x 2 vectors) | 96 | v_hash_consts, v_hash_shifts |
| FMA multipliers (3 stages) | 24 | v_fma_mult for stages 0, 2, 4 |
| **16 Desks** (6 vectors each) | 768 | idx, val, node_val, addr, tmp1, tmp2 |
| Address temporaries | 32 | addr_tmp for load/store computations |
| Offset registers | 16 | off_0 through off_15 |
| Loop counters | 3 | tile_offset, tile_counter, round_counter |
| Offset constants | 16 | offset_consts for desk offsets |
| Other constants | ~10 | tile_stride, num_tiles, num_rounds |
| **TOTAL** | **~1,027 words** | 67% of 1,536 |

**Available for preloading**: 1,536 - 1,027 = **509 words**

### 1.2 Level Preloading Requirements

Tree levels 0-7 contain 255 nodes total:

| Level | Nodes | Indices | Words Needed |
|-------|-------|---------|--------------|
| 0 | 1 | 0 | 1 |
| 1 | 2 | 1-2 | 2 |
| 2 | 4 | 3-6 | 4 |
| 3 | 8 | 7-14 | 8 |
| 4 | 16 | 15-30 | 16 |
| 5 | 32 | 31-62 | 32 |
| 6 | 64 | 63-126 | 64 |
| 7 | 128 | 127-254 | 128 |
| **TOTAL** | **255** | 0-254 | **255** |

**Verdict**: 255 words < 509 available words. **FITS!**

### 1.3 Revised Scratch Budget with Preloading

| Component | Size (words) | Notes |
|-----------|--------------|-------|
| H65v3 base requirements | 1,027 | As calculated above |
| Tree level preload buffer | 255 | Levels 0-7 |
| **TOTAL** | **1,282 words** | 83% of 1,536 |

**Headroom**: 254 words remaining for additional optimizations.

---

## 2. Round-by-Round Strategy

### 2.1 Overview

The kernel processes 128 elements (16 desks, 2 tiles) through 16 rounds:

| Round | Max Unique Indices | Strategy |
|-------|-------------------|----------|
| 0 | 1 (all at index 0) | Broadcast preloaded tree[0] |
| 1 | 2 (indices 1-2) | Arithmetic selection from preload |
| 2 | 4 (indices 3-6) | Arithmetic selection from preload |
| 3 | 8 (indices 7-14) | Arithmetic selection from preload |
| 4 | 16 (indices 15-30) | Arithmetic selection from preload |
| 5 | 32 (indices 31-62) | Arithmetic selection from preload |
| 6 | 64 (indices 63-126) | Arithmetic selection from preload |
| 7 | 128 (indices 127-254) | Arithmetic selection from preload |
| 8-15 | Up to 256 | Standard gather (indices exceed preload) |

### 2.2 Round 0 (All at Index 0)

**Situation**: All 256 elements have idx=0.

**Approach**:
1. Load tree[0] into a scalar (1 load)
2. Broadcast to all 16 desk node_val vectors (16 vbroadcasts = 3 cycles at 6/cycle)

**Cost**: 1 load + 3 VALU cycles = **4 cycles** (vs ~64 cycles for 16 gathers in H65v3)

**Savings**: ~60 cycles

### 2.3 Rounds 1-7 (Arithmetic Selection)

**Key Technique**: Instead of gathering individual tree values per element, use arithmetic value selection.

For round N, elements can be at indices in levels 0-N. We preloaded all these values. The question is: how to efficiently select the correct value per element?

**Arithmetic Selection Formula**:
```
result = T1 + bit * (T2 - T1)
```
Where:
- T1, T2 are two possible values
- bit is 0 or 1 (extracted from idx)

This uses VALU (6 ops/cycle) instead of vselect (Flow, 1 op/cycle).

**Binary Tree Selection**:
For round N, we need to select from 2^N possible values using N bits from the element's index.

**Example - Round 2 (4 values at indices 3,4,5,6)**:
```
idx in {3,4,5,6} maps to local offsets {0,1,2,3}
local_idx = idx - 3

bit0 = local_idx & 1
bit1 = (local_idx >> 1) & 1

// Binary tree selection
// Level 1: pairs (0,1) and (2,3)
val_01 = tree[3] + bit0 * (tree[4] - tree[3])
val_23 = tree[5] + bit0 * (tree[6] - tree[5])

// Level 2: final selection
result = val_01 + bit1 * (val_23 - val_01)
```

**Cost per round N (for 16 desks)**:
- Bit extraction: N ops per desk
- Selection tree: (N) levels x 2 ops (sub + FMA) per desk = 2N ops per desk
- Total: 3N ops per desk x 16 desks = 48N ops
- At 6 VALU ops/cycle: 8N cycles

| Round | Selection Ops | Selection Cycles |
|-------|---------------|------------------|
| 1 | 48 | 8 |
| 2 | 96 | 16 |
| 3 | 144 | 24 |
| 4 | 192 | 32 |
| 5 | 240 | 40 |
| 6 | 288 | 48 |
| 7 | 336 | 56 |
| **Total R1-7** | **1,344** | **224** |

### 2.4 Rounds 8-15 (Standard Gather)

For rounds 8-15, indices can span up to 256 unique values. The preloaded 255 nodes no longer cover all possibilities, so we fall back to H65v3's gather approach.

**Per-Round Cost (H65v3 baseline)**:
- Gather: 64 cycles (16 desks x 4 gather cycles, interleaved)
- Hash + branch: ~30 cycles (overlapped with gather)
- Total: ~94 cycles per round (from current 2,941 / 16 / 2 ~= 92)

**Rounds 8-15 Total**: 8 x 94 = **752 cycles**

---

## 3. Expected Cycle Count Estimate

### 3.1 Phase-by-Phase Breakdown

#### Phase 1: Initialization (~50 cycles)
Same as H65v3 - initialize constants, hash parameters, desk structures.

#### Phase 2: Preload Tree Levels 0-7 (16 cycles)
```
255 values / 8 (VLEN) = 32 vloads
32 vloads / 2 (loads/cycle) = 16 cycles
```

#### Phase 3: Tile Loop (2 tiles)

**Per-Tile Structure**:
1. Load input indices/values for 16 desks (16 cycles)
2. Process rounds 0-7 with preloaded selection
3. Process rounds 8-15 with standard gathers
4. Store output indices/values for 16 desks (16 cycles)

**Round 0**: 4 cycles (broadcast)
**Rounds 1-7**: Selection + hash + branch

For each desk, per round:
- Selection: 8N cycles / 16 desks = 0.5N cycles (amortized)
- Hash: 2 cycles (12 ops, 6 ops/cycle, with overlap)
- Branch: 0.5 cycles (3 ops, overlapped)

Wait - we need to recalculate more carefully. The 48N ops for 16 desks means:

**Revised Per-Round Costs (Rounds 1-7)**:

| Round | Selection Ops | + Hash Ops (192) | + Branch Ops (48) | Total | Cycles |
|-------|---------------|------------------|-------------------|-------|--------|
| 1 | 48 | 192 | 48 | 288 | 48 |
| 2 | 96 | 192 | 48 | 336 | 56 |
| 3 | 144 | 192 | 48 | 384 | 64 |
| 4 | 192 | 192 | 48 | 432 | 72 |
| 5 | 240 | 192 | 48 | 480 | 80 |
| 6 | 288 | 192 | 48 | 528 | 88 |
| 7 | 336 | 192 | 48 | 576 | 96 |
| **Total R0-7** | | | | | **508** |

Hmm, this is SLOWER than H65v3's gather approach for later rounds (6-7).

### 3.2 Re-evaluating the Trade-off

**H65v3 Per-Round Cost** (from actual implementation):
- Gather: 4 cycles per desk (8 scalar loads for VLEN=8)
- With 16 desks interleaved: 64 cycles per round
- Hash computation overlapped with next gather: ~30 cycles
- Total: ~94 cycles per round, BUT with deep pipelining this is reduced

Looking at H65v3's 2,941 cycles:
- 2 tiles, 16 rounds per tile = 32 tile-rounds
- 2,941 / 32 = ~92 cycles per tile-round
- Minus setup/teardown, effective per-round: ~90 cycles

**Key Issue**: Arithmetic selection for late rounds (5-7) is actually MORE expensive than gathering!

| Round | Selection Cycles | H65v3 Gather Cycles | Winner |
|-------|-----------------|---------------------|--------|
| 0 | 4 | 64 | Selection (16x) |
| 1 | 48 | 64 | Selection (1.3x) |
| 2 | 56 | 64 | Selection (1.1x) |
| 3 | 64 | 64 | Tie |
| 4 | 72 | 64 | **Gather** |
| 5 | 80 | 64 | **Gather** |
| 6 | 88 | 64 | **Gather** |
| 7 | 96 | 64 | **Gather** |

**Conclusion**: Use selection ONLY for rounds 0-3, then switch to gathering.

### 3.3 Hybrid Strategy: Selection for Rounds 0-3, Gather for Rounds 4-15

**Revised Cycle Estimate**:

| Component | Cycles | Notes |
|-----------|--------|-------|
| Initialization | 50 | Same as H65v3 |
| Preload (levels 0-3: 15 nodes) | 2 | 15/8 = 2 vloads |
| Per-tile load input | 16 | 32 vloads for 16 desks |
| Round 0 (broadcast) | 4 | Broadcast preloaded tree[0] |
| Rounds 1-3 (selection) | 48+56+64 = 168 | Selection + hash |
| Rounds 4-15 (gather) | 12 x 64 = 768 | H65v3 approach |
| Per-tile store output | 16 | 32 vstores for 16 desks |
| Tile overhead | 5 | Loop control |

**Per-Tile**: 16 + 4 + 168 + 768 + 16 + 5 = **977 cycles**
**Two Tiles**: 977 x 2 = **1,954 cycles**
**Total**: 50 + 2 + 1,954 = **2,006 cycles**

**Still above target!** The gather phase (768 cycles x 2 = 1,536 cycles) dominates.

---

## 4. Deeper Analysis: Why H65v3 is Already Optimal for Rounds 4-15

Looking more carefully at H65v3's actual performance:

**H65v3 Interleaving Optimization**:
- 16 desks have their gathers interleaved with hash computations
- While desk 0 is loading, desks 1-3 are computing hash
- This hides most of the gather latency

**H65v3 Per-Round Structure** (from code analysis):
1. Prepare gather addresses (3 VALU cycles for 16 desks)
2. Interleaved gather + hash (see code lines 400-615)
   - Each desk takes 4 cycles for gather
   - But overlapped: 16 desks * 4 cycles = 64, BUT pipelined to ~50 cycles
3. Finish remaining operations (14 VALU cycles)
4. Loop control (3 cycles)

**Actual per-round in H65v3**: ~70 cycles (not 64+30 = 94)

This explains 2,941 cycles:
- Setup: 100 cycles
- 2 tiles x (16 load + 16 x 70 round + 16 store + 10 overhead) = 2 x 1,162 = 2,324 cycles
- Total: 100 + 2,324 + 517 overhead = 2,941 cycles

Wait, that doesn't add up. Let me recalculate:
- 2,941 cycles / 2 tiles = 1,470 cycles per tile
- 1,470 / 16 rounds = ~92 cycles per round

So H65v3 achieves ~92 cycles per round with full pipelining.

---

## 5. Revised Strategy: Maximize Early-Round Savings

### 5.1 The Real Opportunity

The only way to beat H65v3 significantly is to reduce early round costs substantially:

**Round 0 Opportunity**:
- H65v3: ~92 cycles (full pipeline)
- With broadcast: 4 cycles + hash overlap = ~30 cycles
- **Savings: 62 cycles per tile = 124 cycles total**

**Rounds 1-3 Opportunity**:
If we can make selection + hash faster than gather + hash:
- H65v3: 92 cycles per round
- Selection approach needs to be < 92 cycles

For round 1 (2 values):
- Selection: 1 level x 2 ops = 2 ops per desk = 32 ops total = 6 cycles
- Hash: 192 ops = 32 cycles
- Branch: 48 ops = 8 cycles
- Total: 46 cycles < 92 cycles. **WIN!**

For round 2 (4 values):
- Selection: 2 levels x 2 ops = 4 ops per desk = 64 ops = 11 cycles
- Hash + branch: 40 cycles
- Total: 51 cycles < 92 cycles. **WIN!**

For round 3 (8 values):
- Selection: 3 levels x 2 ops = 6 ops per desk = 96 ops = 16 cycles
- Hash + branch: 40 cycles
- Total: 56 cycles < 92 cycles. **WIN!**

For round 4 (16 values):
- Selection: 4 levels x 2 ops = 8 ops per desk = 128 ops = 22 cycles
- Hash + branch: 40 cycles
- Total: 62 cycles < 92 cycles. **WIN!**

For round 5 (32 values):
- Selection: 5 levels x 2 ops = 10 ops per desk = 160 ops = 27 cycles
- Hash + branch: 40 cycles
- Total: 67 cycles < 92 cycles. **WIN!**

For round 6 (64 values):
- Selection: 6 levels x 2 ops = 12 ops per desk = 192 ops = 32 cycles
- Hash + branch: 40 cycles
- Total: 72 cycles < 92 cycles. **WIN!**

For round 7 (128 values):
- Selection: 7 levels x 2 ops = 14 ops per desk = 224 ops = 38 cycles
- Hash + branch: 40 cycles
- Total: 78 cycles < 92 cycles. **WIN!**

### 5.2 Complete Savings Calculation

| Round | H65v3 Cycles | With Selection | Savings |
|-------|--------------|----------------|---------|
| 0 | 92 | 30 | 62 |
| 1 | 92 | 46 | 46 |
| 2 | 92 | 51 | 41 |
| 3 | 92 | 56 | 36 |
| 4 | 92 | 62 | 30 |
| 5 | 92 | 67 | 25 |
| 6 | 92 | 72 | 20 |
| 7 | 92 | 78 | 14 |
| **Total R0-7** | **736** | **462** | **274** |

**Per Tile Savings**: 274 cycles
**Two Tiles Savings**: 548 cycles
**Estimated Total**: 2,941 - 548 = **2,393 cycles**

---

## 6. Further Optimization: Preload Address Computation

### 6.1 Current H65v3 Gather Overhead

H65v3 computes gather addresses each round:
```
For each desk:
  vbroadcast forest_values_p
  vadd addr, addr, idx  # Add current index to base
```

This is 2 VALU ops per desk = 32 ops per round = 6 cycles.

### 6.2 With Preloading, Skip Address Computation for Rounds 0-7

Since we use preloaded values, we don't need to compute gather addresses for rounds 0-7.

**Additional savings per round**: 6 cycles x 8 rounds = 48 cycles per tile = **96 cycles total**

**Revised Estimate**: 2,393 - 96 = **2,297 cycles**

---

## 7. Optimizing Rounds 8-15

### 7.1 Better Pipelining for Gather Rounds

H65v3 gathers are already well-pipelined, but we might gain more by:

1. **Precomputing tree level offsets**: Store level base addresses in scratch
2. **Batched address computation**: Compute all 16 desk addresses in parallel
3. **Overlap gather with selection**: Start next round's selection while finishing current gather

### 7.2 Potential Further Savings

If we can reduce rounds 8-15 from 92 to 85 cycles each:
- Savings: 7 cycles x 8 rounds x 2 tiles = **112 cycles**
- New estimate: 2,297 - 112 = **2,185 cycles**

---

## 8. Final Cycle Estimate

### 8.1 Optimistic Scenario

| Component | Cycles | Notes |
|-----------|--------|-------|
| Initialization | 50 | Constants, hash params |
| Preload levels 0-7 | 16 | 32 vloads at 2/cycle |
| Tile 1 load | 16 | 32 vloads for 16 desks |
| Tile 1 rounds 0-7 | 462 | Selection + hash |
| Tile 1 rounds 8-15 | 680 | 8 x 85 cycles |
| Tile 1 store | 16 | 32 vstores |
| Tile 2 (same) | 1,174 | Same as tile 1 |
| Tile overhead | 20 | Loop control |
| **TOTAL** | **2,434 cycles** |

### 8.2 Conservative Scenario

If selection overhead is higher than estimated (more data movement needed):

| Component | Cycles | Notes |
|-----------|--------|-------|
| Initialization | 50 | |
| Preload levels 0-7 | 16 | |
| Per-tile load/store | 64 | 2 x 32 |
| Rounds 0-7 x 2 tiles | 1,000 | ~63 cycles per round |
| Rounds 8-15 x 2 tiles | 1,400 | 88 cycles per round |
| Overhead | 40 | |
| **TOTAL** | **2,570 cycles** |

### 8.3 Expected Range

**2,400 - 2,600 cycles** (18-22% improvement over H65v3's 2,941)

**Gap to Target**: 600-800 cycles still needed

---

## 9. Implementation Approach

### 9.1 Phase 1: Add Preload Buffer

```python
# After desk allocation
v_tree_level = []
for level in range(8):
    level_size = 2**level
    v_tree_level.append(self.alloc_scratch(f"v_tree_L{level}", level_size))
# Total: 1 + 2 + 4 + 8 + 16 + 32 + 64 + 128 = 255 words
```

### 9.2 Phase 2: Preload Tree Levels

```python
# Before tile loop, after initialization
tree_base = self.scratch["forest_values_p"]

# Load level 0 (1 value)
self.add("load", ("load", v_tree_level[0], tree_base))

# Load levels 1-7 using vloads
offset = 1
for level in range(1, 8):
    level_size = 2**level
    level_base = offset
    for i in range(0, level_size, VLEN):
        addr = self.scratch_const(level_base + i)
        self.add("alu", ("+", addr_tmp, tree_base, addr))
        self.add("load", ("vload", v_tree_level[level] + i, addr_tmp))
    offset += level_size
```

### 9.3 Phase 3: Selection for Rounds 0-7

**Round 0**:
```python
# Broadcast tree[0] to all desks
for d in range(NUM_DESKS):
    self.add("valu", ("vbroadcast", desks[d]['node_val'], v_tree_level[0]))
```

**Rounds 1-7** (binary selection tree):
```python
def select_from_level(desk, round_num):
    """Select tree value for desk based on its index, from preloaded level."""
    level = round_num
    level_base = v_tree_level[level]

    # Compute local index within level
    # idx at this round is in range [2^level - 1, 2^(level+1) - 2]
    # local_idx = idx - (2^level - 1)

    # Extract bits and do binary selection
    for bit in range(level):
        # Extract bit from local_idx
        # Select between pair of values
        # Store intermediate in tmp
        pass

    # Final result in node_val
```

### 9.4 Phase 4: Fall Back to H65v3 for Rounds 8-15

Keep the existing interleaved gather + hash approach from H65v3.

---

## 10. Risks and Mitigations

### 10.1 Risk: Selection Overhead Higher Than Expected

**Mitigation**: Profile individual rounds. If round 5-7 selection is slower than gather, switch threshold.

### 10.2 Risk: Scratch Pressure

**Mitigation**: We have 254 words headroom. If needed, reduce preload to levels 0-5 (63 nodes).

### 10.3 Risk: Register Pressure During Selection

**Mitigation**: Reuse tmp1/tmp2 vectors from hash computation (not needed during selection phase).

---

## 11. Conclusion

### 11.1 Expected Outcome

H66 with level preloading should achieve **2,400-2,600 cycles**, an improvement of 12-18% over H65v3's 2,941 cycles.

### 11.2 Gap Analysis

| Design | Cycles | vs H65v3 | vs Target (1,790) |
|--------|--------|----------|-------------------|
| H65v3 | 2,941 | baseline | +64% |
| H66 (optimistic) | 2,400 | -18% | +34% |
| H66 (conservative) | 2,600 | -12% | +45% |
| Target | 1,790 | -39% | baseline |

### 11.3 What Would Be Needed for 1,790 Cycles?

To reach 1,790 cycles, we need ~600 more cycles of savings beyond H66. This would require:

1. **Perfect gather elimination**: Zero-cost tree access for ALL rounds (not just 0-7)
2. **Hash computation reduction**: Halving the 192 ops per round
3. **Completely hidden memory latency**: All loads overlapped with compute

These optimizations are likely beyond what level preloading alone can achieve. Additional algorithmic changes (different tree representation, batched hashing, etc.) would be needed.

### 11.4 Recommendation

Implement H66 to validate the 18% improvement estimate. If successful, this provides a strong foundation for further optimization. The remaining 600-cycle gap may require fundamentally different approaches beyond the current ISA utilization strategy.

---

## Appendix A: Scratch Memory Map

| Address Range | Size | Contents |
|---------------|------|----------|
| 0-9 | 10 | Scalar variables |
| 10-29 | 20 | Constants |
| 30-61 | 32 | v_zero, v_one, v_two, v_n_nodes |
| 62-157 | 96 | Hash constants |
| 158-181 | 24 | FMA multipliers |
| 182-949 | 768 | 16 Desks (6 vectors each) |
| 950-981 | 32 | Address temporaries |
| 982-997 | 16 | Offset registers |
| 998-1000 | 3 | Loop counters |
| 1001-1016 | 16 | Offset constants |
| 1017-1026 | 10 | Other constants |
| **1027-1281** | **255** | **Tree level preload buffer** |
| 1282-1535 | 254 | Available headroom |

---

## Appendix B: Binary Selection Tree Algorithm

For round N, selecting from 2^N preloaded values:

```
Input: idx (element's current tree index)
       preload[0..2^N-1] (preloaded tree values for level N)
Output: tree_val (the correct tree value for this element)

# Compute local index within level
level_start = 2^N - 1
local_idx = idx - level_start

# Binary selection tree
for bit = 0 to N-1:
    mask = 1 << bit
    bit_val = (local_idx >> bit) & 1

    # For each pair of values, select based on bit
    for pair in range(2^(N-bit-1)):
        v0 = selection[pair*2]
        v1 = selection[pair*2 + 1]
        selection[pair] = v0 + bit_val * (v1 - v0)  # Arithmetic select

tree_val = selection[0]
```

**VALU Implementation**:
```
vsub tmp, v1, v0          # tmp = v1 - v0
multiply_add result, bit, tmp, v0  # result = v0 + bit * (v1 - v0)
```

This uses 2 VALU ops per selection level, 2N ops total per element.

---

## Appendix C: Key Files

- H65v3 implementation: `/home/hestiasadmin/projects/original_performance_takehome/experiments/H65_loop_tiling/perf_takehome_h65_v3.py`
- H63 analysis: `/home/hestiasadmin/projects/original_performance_takehome/research_loop/designs/H63_missing_trick_analysis.md`
- H65 design: `/home/hestiasadmin/projects/original_performance_takehome/research_loop/designs/H65_loop_tiling_design.md`
