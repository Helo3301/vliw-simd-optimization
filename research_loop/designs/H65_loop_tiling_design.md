# H65: Loop Tiling for VLIW SIMD Tree Traversal

## Executive Summary

**Hypothesis**: Instead of processing all 256 elements through 2 rounds repeatedly (H54's approach), use loop tiling to process smaller element groups (32 or 64 elements) through ALL 16 rounds before moving to the next group. This keeps tree values and element state in scratch longer, enabling better temporal locality and reducing redundant loads.

**Key Insight from H63**: All 256 elements start at index 0, and early rounds have bounded index diversity. By processing smaller batches through more rounds, we can keep preloaded tree levels in scratch across multiple rounds per batch.

**Current Status**:
- H54: 3,462 cycles (best current)
- H64: Round-synchronized level preloading (not yet implemented)
- H65: Loop tiling with reduced batch sizes per tile

---

## 1. Current H54 Approach vs Proposed H65

### H54 Loop Structure

```python
for iteration in 0..8:                    # 8 iterations
    for desk in 0..16:                    # 16 desks (128 elements)
        for round in 0..1:                # 2 rounds fused
            load input indices/values
            load tree node
            compute hash
            compute next index
        store results
    batch_offset += 128
```

**Characteristics**:
- Process 128 elements (16 desks, 16 vectors)
- 2 rounds per iteration (round fusion)
- Repeat 8 times for 16 total rounds
- Each iteration: load → process 2 rounds → store
- Memory traffic: Input loaded/stored 8 times

### Proposed H65 Loop Tiling

```python
for tile in 0..8:                         # 8 tiles
    preload_tree_levels(0..7, scratch)    # 255 values once per tile
    for round in 0..15:                   # All 16 rounds
        for element in tile*32..(tile+1)*32:  # 32 elements per tile
            load idx/val (if not in scratch)
            get tree node (from preload or gather)
            compute hash
            compute next index
    store results for tile
    batch_offset += 32
```

**Characteristics**:
- Process 32 elements (4 vectors) per tile
- Tile processes ALL 16 rounds before moving to next tile
- Preload tree levels 0-7 once per tile (255 values = 32 vloads = 16 cycles)
- 8 tiles × 16 rounds = 128 tile-rounds total
- Memory traffic: Input loaded/stored 8 times (same as H54)
- Tree preload overhead: 16 cycles × 8 tiles = 128 cycles

---

## 2. Scratch Budget Analysis for Loop Tiling

### Per-Tile Memory Requirements

**For 4 vectors (32 elements):**

| Component | Size (words) | Notes |
|-----------|--------------|-------|
| idx_vec (4 vectors) | 32 | Current indices (4 × 8) |
| val_vec (4 vectors) | 32 | Current hash values |
| node_val_vec (4 vectors) | 32 | Loaded tree node |
| addr_vec (4 vectors) | 32 | Compute addresses |
| tmp1_vec (4 vectors) | 32 | Temporary for hash |
| tmp2_vec (4 vectors) | 32 | Temporary for hash |
| **Per-tile state** | **192** | |

**Tree Level Preloading (once per tile):**

| Level | Nodes | vload cycles | Words |
|-------|-------|--------------|-------|
| 0 | 1 | 1 | 8 |
| 1-2 | 2 | 1 | 8 |
| 3-6 | 4 | 1 | 8 |
| 7-14 | 8 | 1 | 8 |
| 15-30 | 16 | 2 | 16 |
| 31-62 | 32 | 4 | 32 |
| 63-126 | 64 | 8 | 64 |
| 127-254 | 128 | 16 | 128 |
| **Levels 0-7 Total** | **255** | **32 vloads** | **272** |

**Constants and Temporaries:**

| Component | Words |
|-----------|-------|
| Hash constants | 24 |
| FMA multipliers | 3 |
| Tree pointers (4 temp addr) | 4 |
| Loop counters | 2 |
| Utility constants | ~10 |
| **Subtotal** | **~43** |

**TOTAL SCRATCH BUDGET:**
- Tile state: 192 words
- Tree preload: 272 words
- Constants: 43 words
- **Total: 507 words (33% of 1,536 SCRATCH_SIZE)**

**Verdict**: ✓ FITS COMFORTABLY IN SCRATCH

This is much better than H54's 1,010 words and leaves 1,029 words free for additional optimization.

---

## 3. Optimal Tile Size Analysis

### Trade-off: Tile Size vs Preload Overhead

Let's evaluate different tile sizes:

**32 Elements (4 vectors) - PROPOSED**

| Metric | Value |
|--------|-------|
| Tiles | 8 |
| Vectors per tile | 4 |
| State per tile (words) | 192 |
| Tree preload overhead (cycles/tile) | 16 |
| Total preload cycles (8 tiles) | **128** |
| Scratch used | 507 (33%) |
| Parallelism (simultaneous elements) | 32 |

**64 Elements (8 vectors) - ALTERNATIVE**

| Metric | Value |
|--------|-------|
| Tiles | 4 |
| Vectors per tile | 8 |
| State per tile (words) | 384 |
| Tree preload overhead (cycles/tile) | 16 |
| Total preload cycles (4 tiles) | **64** |
| Scratch used | 699 (45%) |
| Parallelism (simultaneous elements) | 64 |

**128 Elements (16 vectors) - H54 EQUIVALENT**

| Metric | Value |
|--------|-------|
| Tiles | 2 |
| Vectors per tile | 16 |
| State per tile (words) | 768 |
| Tree preload overhead (cycles/tile) | 16 |
| Total preload cycles (2 tiles) | **32** |
| Scratch used | 1,011 (66%) |
| Parallelism (simultaneous elements) | 128 |

**256 Elements (32 vectors) - FULL BATCH**

| Metric | Value |
|--------|-------|
| Tiles | 1 |
| Vectors per tile | 32 |
| State per tile (words) | 1,536 |
| Tree preload overhead (cycles/tile) | 16 |
| Total preload cycles (1 tile) | **16** |
| Scratch used | 1,819 (118%) |
| Parallelism (simultaneous elements) | 256 |

**Verdict**:
- **32 elements**: Maximum benefit from preload staying in scratch longer, but 8 preload phases
- **64 elements**: Good balance, 4 preload phases, still fits comfortably
- **128 elements**: Becomes like H54, minimal tiling benefit
- **256 elements**: Exceeds scratch limits

**RECOMMENDED: 32 elements (4 vectors) per tile** for maximum locality benefits.

---

## 4. Cycle Estimate: H65 with 32-Element Tiles

### Phase Breakdown

#### SETUP PHASE
```
- Initialize constants: 20 cycles
- Initialize hash multipliers: 5 cycles
- Total: 25 cycles
```

#### FOR EACH TILE (8 tiles):

**A. Tree Preload (once per tile):**
- Load levels 0-7: 32 vloads = 16 cycles (at 2 loads/cycle)
- Store to scratch: overlap with next phase
- Subtotal: 16 cycles

**B. Main Loop (16 rounds × 32 elements):**

For each round:

1. **Load Input Data (for first 4 vectors)**
   - vload idx: 1 cycle
   - vload val: 1 cycle
   - (After first round, reuse previous round's idx/val)
   - Subtotal: 2 cycles (first round only)

2. **Get Tree Node**
   - Early rounds (0-7): Select from preloaded values (arithmetic selection)
     - 1 vsub (diff)
     - 1 multiply_add (selection)
     - Cost: 2 ops / 6 = 1 cycle (with pipelining)
   - Later rounds (8-15): Standard gather
     - 4 loads per vector (8 elements = 8 lanes)
     - Cost: 4 cycles per round

3. **Hash Computation**
   - 6 hash stages per element
   - 4 vectors × 8 elements per vector = 32 elements
   - Per-round cost:
     - Stages 0, 2, 4 (FMA): 3 ops
     - Stages 1, 3, 5 (XOR-based): 4 ops (per stage)
     - Total: 18 ops per vector = 72 ops per round
     - At 6 VALU ops/cycle: 12 cycles (can overlap with next phase)

4. **Branch Computation**
   - AND, FMA for idx calculation
   - Bounds check
   - Cost: 4 ops per vector = 16 ops = 3 cycles

**Per-Round Cost Breakdown:**

| Round | Preloaded? | Node Load | Hash | Branch | Total |
|-------|-----------|-----------|------|--------|-------|
| 0 | Yes | 1 | 12 | 3 | 16 |
| 1 | Yes | 1 | 12 | 3 | 16 |
| 2 | Yes | 1 | 12 | 3 | 16 |
| 3 | Yes | 1 | 12 | 3 | 16 |
| 4 | Yes | 1 | 12 | 3 | 16 |
| 5 | Yes | 1 | 12 | 3 | 16 |
| 6 | Yes | 1 | 12 | 3 | 16 |
| 7 | Yes | 1 | 12 | 3 | 16 |
| 8 | No | 4 | 12 | 3 | 19 |
| 9 | No | 4 | 12 | 3 | 19 |
| 10 | No | 4 | 12 | 3 | 19 |
| 11 | No | 4 | 12 | 3 | 19 |
| 12 | No | 4 | 12 | 3 | 19 |
| 13 | No | 4 | 12 | 3 | 19 |
| 14 | No | 4 | 12 | 3 | 19 |
| 15 | No | 4 | 12 | 3 | 19 |

**Subtotal per tile**:
- Rounds 0-7: 8 × 16 = 128 cycles
- Rounds 8-15: 8 × 19 = 152 cycles
- **Total per tile**: 280 cycles

**C. Store Phase (once per tile):**
- Compute store addresses: 3 cycles
- Store 4 vectors (32 vstore operations): 16 cycles (at 2/cycle)
- Subtotal: 19 cycles

**D. Tile Loop Overhead:**
- Tile iteration bookkeeping: 2 cycles

**Per-Tile Total**: 16 + 280 + 19 + 2 = **317 cycles**

#### TOTAL ESTIMATE

```
Setup:                25 cycles
Tile loop (8 tiles):  8 × 317 = 2,536 cycles
Final store/cleanup:  10 cycles
Loop termination:     5 cycles

TOTAL:               ~2,576 cycles
```

**Comparison**:
- H54: 3,462 cycles
- H65 estimate: 2,576 cycles
- **Improvement: 25.6% faster**

**vs Target**:
- Target: 1,790 cycles
- H65 estimate: 2,576 cycles
- **Gap: 786 cycles (44% above target)**

---

## 5. Why H65 Helps (vs H54)

### Better Cache/Scratch Locality

1. **Tree Values Stay in Scratch**
   - H54: Loads tree values, immediately uses, rarely reused
   - H65: Preloads levels 0-7 once per tile, reused across 16 rounds
   - Benefit: Reduced memory bandwidth

2. **Element State Consolidation**
   - H54: 16 desks, each with full state, context switch between rounds
   - H65: 4 vectors (32 elements), contiguous state block, no switching
   - Benefit: Better cache locality, fewer memory accesses

3. **Preload Efficiency**
   - H63 insight: Early rounds have bounded index diversity
   - H65 uses this: Preload levels 0-7 for ANY 32 elements
   - Benefit: Eliminate 8 × (128 + 32 + 64) = 2,304 gather operations

4. **Reduced Load/Store Iterations**
   - H54: 8 iterations (load/store input 8 times)
   - H65: 8 iterations (same), but WITHIN each iteration process all 16 rounds
   - Benefit: Better amortization of setup costs

### What H65 Does NOT Improve

- **Hash computation**: Still 256 elements × 16 rounds × ~12 ops
- **Branch computation**: Still required per element per round
- **Fundamental load bottleneck**: Rounds 8-15 still need 256 gathers per round

---

## 6. Comparison with H64 (Round-Synchronized Approach)

**H64 Hypothesis** (from H63):
- Synchronize all 256 elements per round
- Rounds 0-7: Use preloaded values + arithmetic selection (no gathers)
- Rounds 8-15: Fall back to desks with optimized pipelining

**H64 vs H65 Trade-offs**:

| Aspect | H64 | H65 |
|--------|-----|-----|
| Tile size | 256 (all elements) | 32 (per tile) |
| Preload scope | Once, shared by all | Once per tile, smaller scope |
| Early round cost (0-7) | Wavefront broadcast overhead | Arithmetic selection on smaller set |
| Later round cost (8-15) | Optimized pipelines | Standard gather on 32 elements |
| Scratch usage | ~1,600+ words | ~500 words |
| Synchronization overhead | Need to sync 256 elements | Tile-local, minimal overhead |
| Index diversity | Exploit early rounds | Exploit early rounds |

**H64 Advantages**:
- Potentially better for early rounds (broadcast all 256 at once)
- Simpler architecture (one synchronization point)

**H65 Advantages**:
- Better scratch utilization (leaves room for further optimization)
- Tighter inner loop (4 vectors vs 32 vectors)
- More flexible for varying batch sizes
- Reduces register pressure on individual operations

**Likely Result**: H65 and H64 have similar cycle performance, but H65 has better resource efficiency.

---

## 7. Key Architectural Decisions

### Decision 1: Tile Size = 32 Elements (4 Vectors)

**Justification**:
- Scratch budget: 507 words (33% of 1,536)
- Maintains 4:1 parallelism (sufficient for load/compute overlap)
- Preload cost (16 cycles) amortized over 32 elements × 16 rounds = 512 element-rounds
- Preload cost per element-round: 16 / 512 = 0.03 cycles

### Decision 2: Preload Tree Levels 0-7 (255 Nodes)

**Justification**:
- Covers rounds 0-7 (early, bounded index diversity)
- 255 nodes = 32 vloads = 16 cycles
- Leaves levels 8-10 for on-demand gather (these have 256-2,047 nodes, too large to preload)
- Aligns with H63's index convergence analysis

### Decision 3: Rounds 8-15 Use Standard Gather

**Justification**:
- At round 8, indices are fully dispersed (up to 256 unique values)
- Preloading all possible values exceeds scratch budget
- Standard gather is only option
- Represents 2,048 loads = 1,024 cycles minimum

### Decision 4: Maintain Load/Store Per Tile

**Justification**:
- Could optimize input indices/values once (shared across all rounds), but:
  - They get modified each round (new idx values)
  - Still need to store updated idx/val back to memory
  - Savings: ~5-10 cycles per tile, not significant
- Keep current approach for simplicity

---

## 8. Cycle-Accurate Estimation with Optimizations

### Aggressive Optimization Scenario

If we apply several micro-optimizations:

1. **Hash Pipelining Improvement**: Overlap hash stages with branch computation
   - Current: 12 + 3 = 15 cycles per round
   - Optimized: 12 cycles (branch computation overlaps entirely)
   - Saving: 3 cycles per round × 16 rounds × 8 tiles = 384 cycles

2. **Preload/Compute Overlap**: Preload while processing previous tile
   - Current: 16 cycles per tile
   - Overlapped: 0 cycles (amortized into previous tile)
   - Saving: 16 × 7 = 112 cycles (last tile must preload, can't overlap)

3. **Rounds 8-15 Optimization**: Better gather interleaving
   - Current: 4 cycles per gather
   - Optimized: 3.5 cycles (better desk scheduling)
   - Saving: 0.5 × 8 rounds × 32 elements/round × 8 tiles = 1,024 cycles

**Optimized Estimate**:
```
Base: 2,576 cycles
- Hash pipelining: 384 cycles
- Preload overlap: 112 cycles
- Gather optimization: 1,024 cycles

OPTIMIZED TOTAL: ~1,056 cycles
```

**BUT**: This requires:
1. Aggressive instruction scheduling (hard to achieve)
2. Perfect overlap of memory and compute (risky)
3. Advanced gather optimization beyond current understanding

**Realistic Optimized Range**: 2,200 - 2,400 cycles

---

## 9. Hypothesis Validation

### Key Question: Does Loop Tiling Actually Help?

**Hypothesis**: Process smaller batches through more rounds enables better reuse of preloaded tree values.

**Evidence For**:
- H63 shows early rounds have bounded indices (levels 0-7: 255 nodes)
- These 255 nodes can be kept in scratch for 16 rounds per tile
- Each element in the tile reuses these values multiple times
- Reduces total tree load bandwidth

**Evidence Against**:
- Preload overhead: 128 cycles total (16 × 8 tiles)
- H54 doesn't preload, saves these 128 cycles
- Net benefit: ~886 cycles saved - 128 cycles preload = 758 cycles net
- This matches our ~25% improvement estimate

**Risk**: The preload overhead might be higher than estimated if:
- vload has additional latency not accounted for
- Arithmetic selection takes more than 1 cycle per round
- Gather in rounds 8-15 cannot overlap with preload

### Comparison Point: What Would Break Even?

If H65 achieves exactly H54's 3,462 cycles:
- Preload cost: 128 cycles
- Tree reuse must save exactly 128 cycles over 8 tiles
- This means tree loads account for only 128 cycles in H54
- But H54 has ~2,048 tree loads minimum (256 × 8 rounds)
- At 2 loads/cycle: ~1,024 cycles minimum

**This suggests H65 will definitely improve over H54** (by at least the preload/gather savings).

---

## 10. Practical Implementation Challenges

### Challenge 1: Arithmetic Selection for Rounds 0-7

**Issue**: How to efficiently select from preloaded tree values using indices?

**Solution**:
```
For round N (N <= 7):
  - Need to select from 2^N values
  - Each element's index determines which value to pick
  - Use binary selection tree:
    - Extract bit[N-1] to bit[0] from element's current index
    - Level-by-level arithmetic selection
    - Cost: N levels × 2 ops (sub + multiply_add) = 2N ops per vector
    - At 6 VALU ops/cycle: 2N/6 ≈ N/3 cycles
```

**For round 3 (8 values, 3 bits)**:
- 3 levels of binary selection
- Per-vector cost: 6 ops = 1 cycle
- For 4 vectors: 4 cycles per round
- This matches our estimate

**For round 7 (128 values, 7 bits)**:
- 7 levels of binary selection
- Per-vector cost: 14 ops = 2.3 cycles
- For 4 vectors: ~10 cycles per round
- **This is HIGHER than our estimate!**

**Revised Estimate for Rounds 5-7**:

| Round | Unique Idx | Tree Levels | Selection Ops | Cycles (4 vec) |
|-------|-----------|-------------|---------------|----------------|
| 0 | 1 | 0 | 0 | 0 |
| 1 | 2 | 1 | 2 | 1 |
| 2 | 4 | 2 | 4 | 1 |
| 3 | 8 | 3 | 6 | 1 |
| 4 | 16 | 4 | 8 | 2 |
| 5 | 32 | 5 | 10 | 2 |
| 6 | 64 | 6 | 12 | 2 |
| 7 | 128 | 7 | 14 | 3 |

**Revised Per-Round Costs (Rounds 0-7)**:
- Rounds 0-4: 1 cycle selection + 12 cycles hash + 3 cycles branch = 16 cycles (matches)
- Round 5: 2 cycles selection + 12 cycles hash + 3 cycles branch = **17 cycles**
- Round 6: 2 cycles selection + 12 cycles hash + 3 cycles branch = **17 cycles**
- Round 7: 3 cycles selection + 12 cycles hash + 3 cycles branch = **18 cycles**

**Revised Total Per-Tile**:
- Rounds 0-4: 5 × 16 = 80 cycles
- Rounds 5-7: 3 × 17 = 51 cycles (+ 1 extra from round 7 = 52)
- Rounds 8-15: 8 × 19 = 152 cycles
- **Per-tile subtotal: 280 → 285 cycles**

**Revised Total Estimate**:
```
Setup:               25 cycles
Tile loop (8 tiles): 8 × (16 preload + 285 compute + 19 store + 2 overhead) = 8 × 322 = 2,576 cycles
Cleanup:             15 cycles

TOTAL:              ~2,616 cycles (vs 3,462 baseline, 24% improvement)
```

The improvement is robust to this refinement.

### Challenge 2: Register Pressure for Arithmetic Selection

**Issue**: Binary selection tree requires temporary values at each level.

**Solution**:
- Use tmp1, tmp2 registers already allocated for hash
- Reuse across rounds (tmp registers not needed during selection)
- No additional register pressure

### Challenge 3: Gather Scheduling for Rounds 8-15

**Issue**: 32 elements × 8 rounds = 256 gathers must be scheduled into 4-cycle windows.

**Solution**:
- Current H54 approach: 16 desks get 4 cycles per gather
- H65: 4 vectors get 4 cycles per gather (same concept, smaller scale)
- Should be achievable with existing techniques

---

## 11. Why H65 is Better than Pure H64

### H64 Issues (Hypothetical)

H64 would process all 256 elements through each round synchronously:

```
Round 0:
  - All 256 elements at idx=0
  - Load tree[0]: 1 load
  - Broadcast to 32 vectors: 6 broadcasts = 1 cycle
  - Cost: 2 cycles

Round 1:
  - Elements at idx ∈ {1, 2}
  - Load tree[1], tree[2]: 2 loads = 1 cycle
  - Broadcast to 32 vectors: 2 × 32 broadcasts = ~11 cycles
  - Arithmetic selection: 2 selections × 32 vectors = 64 ops = 11 cycles
  - Cost: 23 cycles (broadcast dominates!)
```

**Problem**: Broadcasting K values to 32 vectors costs K × 32 / 6 = 5.33K cycles.

For rounds 0-7:
- Round 0: 1 × 5.33 = 5 cycles
- Round 1: 2 × 5.33 = 11 cycles
- Round 2: 4 × 5.33 = 21 cycles
- Round 3: 8 × 5.33 = 43 cycles
- Round 4: 16 × 5.33 = 85 cycles (exceeds gather cost!)
- Round 5: 32 × 5.33 = 171 cycles
- Round 6: 64 × 5.33 = 341 cycles
- Round 7: 128 × 5.33 = 683 cycles

**Total for rounds 0-7 broadcasts alone: ~1,360 cycles**

This explains why H60's wavefront approach didn't work (see H60 analysis).

**H65 Avoids This Problem**:
- Only broadcasts to 4 vectors per tile (not 32)
- Broadcast cost per round: K × 4 / 6 = 0.67K cycles
- Round 5: 32 × 0.67 = 21 cycles (feasible)
- Round 7: 128 × 0.67 = 85 cycles (feasible)

By reducing the broadcast scope to per-tile (4 vectors), H65 avoids the broadcast bottleneck that limits H64.

---

## 12. Sensitivity Analysis

### What If Preload Overhead is Higher?

**Scenario**: vload has 2-cycle latency instead of 1-cycle throughput

- Preload cost per tile: 32 cycles instead of 16
- Total preload: 32 × 8 = 256 cycles
- Net change: +128 cycles
- **Total estimate: 2,744 cycles**
- Improvement vs H54: 20.7% (still significant)

### What If Arithmetic Selection is Slower?

**Scenario**: Selection takes 2 cycles per round instead of 1

- Rounds 0-4: 5 × 2 = 10 cycles selection (vs 5)
- Rounds 5-7: 3 × 2 = 6 cycles selection (vs 9)
- Net change per tile: 10 - 5 + 6 - 9 = +2 cycles
- Total change: 2 × 8 = 16 cycles
- **Total estimate: 2,632 cycles**
- Improvement vs H54: 23.9% (still significant)

### What If Gather in Rounds 8-15 Cannot Overlap?

**Scenario**: Gather phases have scheduling dependencies, cannot hide latency

- Rounds 8-15: 8 × 4 = 32 cycles per desk (not 19)
- Per tile: 8 × 32 = 256 cycles (not 152)
- Per-tile total: 16 + (80 + 52) + 256 + 19 = 423 cycles
- **Total estimate: 3,458 cycles**
- Improvement vs H54: 0% (essentially equivalent!)

This scenario suggests that H65 **requires good scheduling and overlap to be effective**.

---

## 13. Recommended Implementation Strategy

### Phase 1: Baseline H65 (32-element tiles)

**What to Implement**:
1. Loop structure with 8 tiles
2. Preload tree levels 0-7 once per tile
3. Rounds 0-7: Use arithmetic selection on preloaded values
4. Rounds 8-15: Standard gather (similar to H54)
5. Store phase once per tile

**Expected Cycles**: 2,600-2,700 (25% improvement)

**Effort**: Medium (new loop structure, but reuses existing patterns)

### Phase 2: Optimization (if Phase 1 doesn't hit target)

**Optimizations to Try**:
1. Hash pipelining: Overlap hash stages with branch computation
2. Better gather scheduling in rounds 8-15
3. Reduce preload overhead through scheduling tricks
4. Consider 64-element tiles if more parallelism helps

**Expected Additional Improvement**: 200-400 cycles

### Phase 3: Advanced (if still below target)

**If 1,790 target is truly required**:
1. Combine H65 with H64 strategies (sync early rounds across tiles?)
2. Explore algorithmic changes to hash computation
3. Deep ISA analysis for hidden parallelism

---

## 14. Conclusion

### Summary

**H65 Loop Tiling** offers a promising middle ground between H54 and H64:

1. **Mechanism**: Process 32 elements through all 16 rounds before moving to next tile
2. **Advantage**: Keeps preloaded tree levels 0-7 in scratch across many rounds
3. **Scratch Efficiency**: Uses only 33% of available scratch (vs H54's 66%)
4. **Expected Improvement**: 24-26% cycle reduction (2,600-2,700 cycles vs H54's 3,462)
5. **Gap to Target**: Still ~800 cycles away from 1,790 target

### Will H65 Reach 1,790 Cycles?

**Unlikely** unless combined with:
- H64's round synchronization techniques
- Aggressive micro-optimizations (hash pipelining, gather tricks)
- Undiscovered ISA features or architectural insights

**More Realistic Assessment**:
- H65 can achieve ~2,600 cycles (25% improvement)
- H65 + optimizations can achieve ~2,200-2,400 cycles (30-36% improvement)
- Reaching 1,790 cycles likely requires fundamentally different approach

### Key Insight

H63 revealed that all 256 elements start at index 0 and early rounds have bounded diversity. Both H64 and H65 exploit this, but:
- **H64**: Synchronize all 256 elements, suffer broadcast overhead at scale
- **H65**: Process smaller batches, avoid broadcast bottleneck, better scratch efficiency

**H65 is more practical for a ~25% improvement without excessive complexity.**

---

## Appendix A: Detailed Cycle Tables

### H65 Per-Tile Breakdown (32 elements, 4 vectors)

| Phase | Component | Cycles |
|-------|-----------|--------|
| Preload | Load levels 0-7 | 16 |
| Rounds 0-4 | Selection (5×1) + Hash (5×12) + Branch (5×3) | 80 |
| Rounds 5-7 | Selection (2+2+3) + Hash (3×12) + Branch (3×3) | 52 |
| Rounds 8-15 | Gather (8×4) + Hash (8×12) + Branch (8×3) | 152 |
| Store | Address computation + vstore | 19 |
| **Tile Overhead** | Loop bookkeeping | 2 |
| **PER-TILE TOTAL** | | **321 cycles** |

### H65 Total Estimate

| Phase | Cost |
|-------|------|
| Setup (constants) | 25 |
| 8 tiles × 321 cycles | 2,568 |
| Cleanup | 15 |
| **TOTAL** | **~2,608 cycles** |

### Comparison Matrix

| Design | Cycles | vs H54 | vs Target |
|--------|--------|--------|-----------|
| H54 | 3,462 | baseline | +1,672 |
| H65 | ~2,608 | -25% | +818 |
| H64 (est) | ~2,500 | -27% | +710 |
| Target | 1,790 | -48% | baseline |

---

## Appendix B: Key Files & References

- H54 baseline: `/home/hestiasadmin/projects/original_performance_takehome/experiments/H54_16desk/perf_takehome_h54.py`
- H63 analysis: `/home/hestiasadmin/projects/original_performance_takehome/research_loop/designs/H63_missing_trick_analysis.md`
- H60 wavefront: `/home/hestiasadmin/projects/original_performance_takehome/research_loop/designs/H60_wavefront_design.md`
- H61 unrolling: `/home/hestiasadmin/projects/original_performance_takehome/research_loop/designs/H61_loop_unrolling_design.md`

---
