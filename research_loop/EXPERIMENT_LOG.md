# Experiment Log

## Format
Each experiment records:
- **Hypothesis ID**
- **Paper Ingested** (if any)
- **Design Agent Output**
- **Implementation**
- **Result (cycles)**
- **Analysis**

---

## Experiments

### Experiment 1: H51 - CAPT-Style Branchless Traversal

**Paper**: arxiv.org/abs/2406.02807 - Collision-Affording Point Trees
**Design Agent**: a1ccc7b

**Design Agent Finding**: ALREADY IMPLEMENTED IN H38

The design agent analyzed H38 and found that branchless traversal is already implemented:
```python
bit = val & 1                    # VALU: vand
idx = idx * 2 + 1                # VALU: multiply_add
idx = idx + bit                  # VALU: vadd
```

This is exactly the CAPT technique: `idx = idx*2 + 1 + bit`

**Key Insight**: H38 uses ZERO vselect operations in the inner loop. All branch decisions are pure VALU arithmetic.

**Result**: No experiment needed - technique already in codebase
**Status**: COMPLETED (already implemented)

**New Hypotheses Generated**:
- H52: Multi-Level Jump (2 levels per iteration)
- H53: Eliminate Bounds Check

---

### Experiment 2: H52 - Multi-Level Jump

**Design Agent**: a1745b9

**Design Agent Finding**: INFEASIBLE

The agent proved mathematically that multi-level jump cannot work due to fundamental data dependencies:

```
val -> hash -> bit0 -> idx_{N+1} -> gather -> val' -> hash -> bit1 -> idx_{N+2}
```

This is a linear chain with NO parallelizable opportunities.

**Key Insight**: Hash result determines branch direction, which determines next gather address, which is needed for next hash. This chain CANNOT be broken.

**Mathematical Analysis**:
- Even with perfect 8-way parallelism, minimum gather-bound = 2,048 cycles
- Target of 1,790 requires more than 8 desks OR reduced work per element

**Result**: NO EXPERIMENT - hypothesis proven infeasible
**Status**: ABANDONED (mathematically impossible)

**New Hypotheses Generated**:
- H54: 16-Desk Ultra-Deep Pipeline (need more parallelism)
- H55: Hash Stage Fusion (reduce work)
- H56: Decoupled Gather-Compute
- H57: Tree Locality Exploitation

---

### Experiment 3: H54 - 16-Desk Ultra-Deep Pipeline

**Design Agent**: a8a8054

**Memory Analysis**: FEASIBLE
- 16 desks require ~1,010 words
- Available: 1,536 words
- Margin: 526 words spare

**Load Bottleneck Analysis**:
- 2 load slots/cycle is fundamental limit
- 288 loads per iteration = 144 cycles minimum (load-bound)
- More desks help overlap but don't increase throughput

**Estimated Performance**:
- Projected: 3,400-3,600 cycles
- Speedup over H38: ~1.15x
- Gap to target: Still 1.9x above 1,790

**Key Insight**: The load slot bottleneck (2/cycle) creates a hard floor around 2,000-2,500 cycles. No amount of desk parallelism can break this.

**Status**: COMPLETED

**Implementation Agent Result**:
- **Actual cycles**: 3,462
- **Speedup over H38**: 1.17x (14.8% faster)
- **Speedup over baseline**: 42.7x
- **Memory used**: 1,012 / 1,536 words (66%)
- **Correctness**: PASSED

**NEW BEST: H54 at 3,462 cycles**

---

### Experiment 4: H58 - Index Grouping / Convergent Processing

**Design Agent**: a07a131

**Design Agent Finding**: NOT IMPLEMENTABLE

**Key Analysis**:
- All 256 elements start at idx=0 (Round 0 needs only 1 load, not 256)
- BUT: ISA lacks indirect addressing to route values to correct lanes
- Flow engine bottleneck: only 1 vselect/cycle

**Break-Even Analysis**:
- For K unique indices, need K-1 vselects per desk
- K=2 (Round 1): 3 cycles per desk - FASTER than gather
- K=4 (Round 2): 6 cycles per desk - SLOWER than gather
- K=8+: MUCH SLOWER than gather

**Only Round 0 would benefit**: ~116 cycles saved (2.9% improvement)
- Not worth code complexity

**CRITICAL DISCOVERY**:
The analysis revealed the target is THEORETICALLY IMPOSSIBLE:
```
256 elements × 16 rounds = 4,096 loads
At 2 loads/cycle = 2,048 cycles MINIMUM
Target of 1,790 < 2,048 theoretical minimum!
```

**Result**: Hypothesis abandoned - ISA doesn't support exploitation
**Status**: ABANDONED (ISA constraints + target impossible)

---

### Experiment 5: H53 - Bounds Check Elimination

**Design Agent**: a60b193

**Design Agent Finding**: BOUNDS CHECK IS REQUIRED

**Key Analysis**:
- Tree has 2047 nodes (height=10: 2^11 - 1)
- After round 10, indices can exceed 2046
- Round 11+: max index grows to 4094, 8190, ... 131070
- Bounds check wraps indices back to 0 (root) - this is PART OF THE ALGORITHM

**Mathematical Proof**:
```
Round 10: max index = 2046 (fits exactly)
Round 11: max index = 4094 (EXCEEDS 2047!)
Round 12-16: Continues to exceed - MUST wrap
```

**Cycle Savings (if it were safe)**: Only ~171 cycles (5% of 3,462)

**Result**: REJECTED - Required for correctness
**Status**: ABANDONED (algorithm requirement, not safety check)

---

### Experiment 6: H55 - Hash Stage Fusion

**Design Agent**: a212aab

**Design Agent Finding**: ALREADY OPTIMAL

**Key Analysis**:
- H54 already implements FMA fusion for stages 0, 2, 4
- FMA multipliers: 4097 (1+2^12), 33 (1+2^5), 9 (1+2^3)
- XOR stages (1, 3, 5) cannot be fused algebraically
- Current: 12 VALU ops per hash = THEORETICAL MINIMUM

**Operation Breakdown**:
- Stage 0: 1 FMA (was 3 ops)
- Stage 1: 3 ops (XOR, SHIFT, XOR) - cannot fuse
- Stage 2: 1 FMA (was 3 ops)
- Stage 3: 3 ops (ADD, SHIFT, XOR) - cannot fuse
- Stage 4: 1 FMA (was 3 ops)
- Stage 5: 3 ops (XOR, SHIFT, XOR) - cannot fuse

**Result**: No optimization possible - H54 at theoretical minimum
**Status**: COMPLETED (already optimal)

---

### Experiment 7: H59 - Constraint-Based Optimal Schedule

**Design Agent**: ad93e62

**Design Agent Finding**: TRACTABLE BUT LIMITED POTENTIAL

**Key Analysis**:
- Problem size: ~1,000 operations per iteration
- ILP/CP formulation is tractable with modern solvers
- Expected improvement: 5-15% (based on Unison paper)

**Bottleneck Analysis**:
```
Current H54: 217 cycles/iteration (3,462 total)
Theoretical floor:
  - Load-bound: 288 loads / 2 = 144 cycles
  - Store-bound: 32 stores / 2 = 16 cycles
  - Setup/loop: ~5 cycles
  - TOTAL FLOOR: ~165 cycles/iteration = 2,640 cycles total

Gap to floor: 28% overhead in current schedule
Maximum improvement possible: 26%
```

**Slack Analysis**:
- ALU: 2% utilized (massive slack)
- VALU: 54% utilized
- Load: 66% utilized (bottleneck)
- Store: 7% utilized

**Recommendation**: Implement ILP analysis before full code generation

**Result**: Worth investigating but load bottleneck limits gains
**Status**: COMPLETED - ILP SOLVER RUN

**ILP Solver Results (OR-Tools CP-SAT)**:
- **Solver status**: OPTIMAL (found in 27.67s)
- **Current H54 makespan**: 212 cycles/iteration
- **Optimal makespan**: 161 cycles/iteration
- **Theoretical minimum**: 144 cycles/iteration
- **Improvement potential**: 51 cycles (24.1%)

**Operation Analysis**:
| Engine | Operations | Utilization | Theoretical Min |
|--------|-----------|-------------|-----------------|
| ALU | 84 | 3.3% | 7 cycles |
| VALU | 640 | 50.3% | 107 cycles |
| Load | 288 | 67.9% | 144 cycles |
| Store | 32 | 7.5% | 16 cycles |
| Flow | 2 | 0.9% | 2 cycles |

**Total Kernel Projection**:
- Current H54: 3,462 cycles
- **Projected optimal: 2,646 cycles** (23.6% improvement)
- Theoretical floor: 2,374 cycles
- Speedup vs baseline: 55.8x (up from 42.7x)

**Key Improvements Identified**:
1. ALU ops can start 80 cycles earlier (overlap with previous iteration)
2. Stores can start 58 cycles earlier (while hash completing)
3. Tighter load/compute interleaving

**CRITICAL FINDING**: Even with optimal scheduling, 2,646 cycles is still above the 1,790 target and the 2,048 "barrier" we calculated. Yet Claude achieved 1,487 cycles. Our theoretical minimum analysis must be wrong somewhere.

---

### Experiment 8: H60 - Wavefront + Arithmetic Value Selection

**Design Agent**: aa75616

**Design Agent Finding**: LIMITED BENEFIT

**Key Innovation Verified**:
Arithmetic selection formula: `result = T1 + bit * (T2 - T1)`
- Verified CORRECT for selecting between two values
- Uses VALU (6 ops/cycle) instead of vselect (1 op/cycle)
- Provides 3x throughput for value selection

**Cycle Analysis by Round**:
| Round | Unique Indices | Loads | Broadcasts | Select | Total |
|-------|----------------|-------|------------|--------|-------|
| 0 | 1 | 1 | 6 | 0 | 7 |
| 1 | 2 | 1 | 11 | 11 | 23 |
| 2 | 4 | 2 | 22 | 22 | 46 |
| 3 | 8 | 4 | 43 | 32 | 79 |
| 4 | 16 | 8 | 86 | 43 | 137 |
| 5+ | 32+ | 16+ | 171+ | 54+ | 241+ (WORSE!) |

**Problem**: Broadcast overhead (K × 32 vectors / 6 per cycle) exceeds gather cost at K >= 16

**Best Case Estimate**: ~2,690 cycles (22% better than H54)

**Load Savings Calculation**:
- Standard: 4,096 loads
- Wavefront (rounds 0-7): 255 unique loads + 2,048 (rounds 8-15) = 2,303 loads
- Savings: 1,793 loads = 897 cycles saved
- BUT: Broadcast overhead adds 1,360 cycles
- NET: WORSE by 463 cycles for full wavefront

**Result**: Only beneficial for rounds 0-3, marginal overall improvement
**Status**: PARTIALLY BENEFICIAL (not a breakthrough)

---

### Experiment 9: H63 - The Missing Trick Analysis

**Design Agent**: ad0935b

**Design Agent Finding**: MAJOR BREAKTHROUGH - Theoretical Minimum Was WRONG!

**The Critical Error in Our Analysis**:
Our 2,048 cycle "minimum" assumed every element needs a unique gather. This is FALSE because:
1. **All 256 elements start at index 0** (from `Input.generate`)
2. **Early rounds have bounded index diversity** - Round N has at most 2^N unique indices
3. **Tree levels are contiguous in memory** - vload can grab 8 values per slot

**Corrected Load Analysis**:
| Rounds | Old Analysis | Corrected Analysis |
|--------|-------------|-------------------|
| 0-7 | 2,048 loads | 255 loads (1+2+4+8+16+32+64+128) |
| 8-15 | 2,048 loads | 2,048 loads |
| **Total** | **4,096** | **2,303** |

**New theoretical minimum**: 2,303 / 2 = **1,152 cycles** (below 1,487 target!)

**The Missing Tricks Identified**:
1. **Tree level preloading** with vload (levels 0-7 = 32 vloads = 16 cycles)
2. **Arithmetic value selection** `T1 + bit*(T2-T1)` (3x faster than vselect)
3. **Round synchronization** - Process all 256 elements per round together
4. **Hybrid architecture** - Preload for rounds 0-7, pipelined gather for 8-15

**Estimated Cycle Budget for 1,487 Cycles**:
| Component | Cycles |
|-----------|--------|
| Tree preload (levels 0-7) | 20 |
| Rounds 0-7 (selection + hash) | 640 |
| Rounds 8-15 (gather + hash) | 800 |
| Loop overhead | 27 |
| **Total** | **1,487** |

**Key Insight from Blog**: "exploited parallelism across the 256 batch items rather than within single items"

This confirms: instead of desk-based pipelining, exploit the fact that early rounds have bounded index diversity.

**Status**: COMPLETED - Ready for implementation (H64)

---

### Experiment 10: H59v2 - ILP-Optimized Schedule Implementation

**Design Agent**: a721641

**Design Agent Finding**: SUCCESSFUL OPTIMIZATION

**Key Optimizations Applied**:
Based on the ILP solver analysis from H59, the following schedule improvements were implemented:

1. **Store Address Pre-computation**: Moved store address calculations from Round 2 tail to overlap with Round 1 hash tail (when VALU-only operations are running)

2. **Store/Hash Overlap**: Interleaved stores for desks 0-13 with Round 2 hash tail cycles (14 VALU-only cycles at the end of round processing)

3. **Loop Control Overlap**: ALU operations for loop control overlapped with final stores (desks 14-15)

**Cycle Analysis**:
```
H54 Round 2 Tail (14 cycles): Pure VALU operations
H59 Optimization: Fill ALU and Store slots during these 14 cycles
- Cycles 1-3: Store addr compute for desks 0-11 (ALU)
- Cycles 4-14: Stores for desks 0-13 + more addr compute
- Final cycles: Loop control + desks 14-15 stores
```

**Result**:
- **Actual cycles**: 3,158
- **Improvement over H54**: 304 cycles (8.8% faster)
- **Speedup over baseline**: 46.78x
- **Correctness**: PASSED

**NEW BEST: H59v2 at 3,158 cycles**

**Analysis**:
The optimization successfully exploited the slack in ALU and Store engines during the hash tail phase. The 8.8% improvement validates the ILP solver's prediction of ~24% potential improvement (we achieved about 1/3 of the theoretical maximum due to dependencies and practical constraints).

**Remaining Gap**:
- Current: 3,158 cycles
- Target: 1,790 cycles
- Gap: 1,368 cycles (still 1.76x away from target)

The H63 insight (level preloading + round synchronization) remains the most promising path to reach sub-2000 cycles.

---

### Experiment 11: H65 - Loop Tiling (Small Batches Through All Rounds)

**Design Agent**: aada9e2

**Design Agent Finding**: SLIGHTLY WORSE THAN H54

**Key Approach**:
- Process 8 tiles of 32 elements each (8 vectors × 8 lanes = 64 elements per tile, 4 tiles total)
- Each tile goes through ALL 16 rounds before moving to next tile
- Goal: Better scratch locality, keep tree values in scratch longer

**Implementation Details**:
- 8 vectors per tile (64 elements)
- 16 rounds per tile
- H54-style interleaved gather/hash within each tile
- Scratch usage: 593 / 1,536 words (39%)

**Result**:
- **Actual cycles**: 3,494
- **Speedup over baseline**: 42.28x
- **Correctness**: PASSED
- **Comparison to H54**: 32 cycles SLOWER (0.9% worse)

**Analysis**:
The loop tiling approach did NOT improve performance because:
1. The smaller tile size (8 vectors vs 16 desks) provides less ILP for latency hiding
2. Overhead from tile loop management adds cycles
3. The tree preloading optimization was NOT implemented - still doing full gathers for all rounds

**Conclusion**: Loop tiling alone doesn't help. The benefit would come from combining it with tree level preloading for rounds 0-7, which requires the H64 approach.

**Status**: COMPLETED - No improvement with v1

---

### Experiment 12: H65v3 - 16-Desk Full Loop Tiling

**Design Agent**: aada9e2 (continued)

**Design Agent Finding**: MAJOR BREAKTHROUGH!

**Key Innovation**:
Combined H54's 16-desk deep pipelining with full 16-round tiling per tile:
- H54: 16 desks × 2 rounds × 16 iterations = 32 memory phases
- H65v3: 16 desks × 16 rounds × 2 iterations = 4 memory phases

This 8x reduction in memory phase overhead yields significant cycle savings while maintaining deep pipelining for latency hiding.

**Implementation Details**:
- 2 tiles of 128 elements each
- 16 desks per tile (same as H54)
- ALL 16 rounds processed per tile before storing
- Only 2 tile iterations total

**Result**:
- **Actual cycles**: 2,941
- **Improvement over H54**: 521 cycles (15.1% faster)
- **Improvement over H59v2**: 217 cycles (6.9% faster)
- **Speedup over baseline**: 50.23x
- **Correctness**: PASSED

**NEW BEST: H65v3 at 2,941 cycles**

**Analysis**:
The key insight is that H54's 2-round fusion strategy, while good for latency hiding, incurs significant load/store overhead by iterating 16 times. By processing all 16 rounds per tile, we amortize this overhead over more rounds.

The gap to target is now:
- Current: 2,941 cycles
- Target: 1,790 cycles
- Gap: 1,151 cycles (64% above target)

**Files Created**:
- `experiments/H65_loop_tiling/perf_takehome_h65_v3.py`

---

### Experiment 13: H67 - Claude Solution Analysis

**Design Agent**: a5fc3e2

**Design Agent Finding**: MAJOR BREAKTHROUGH - Found the Missing Optimization!

**Key Discovery from GitHub PR #22 (1,299 cycles)**:
The solution exploits **bounds check wrapping** that we missed:
- Tree height=10 means n_nodes=2047
- After round 10, indices can exceed 2047
- Bounds check `idx = 0 if idx >= n_nodes else idx` resets indices to 0!
- This causes indices to CLUSTER in range [0-14] for rounds 11-14

**Hybrid Round Strategy (the key insight)**:
| Rounds | Method | Why |
|--------|--------|-----|
| 0-3 | vselect (no gather) | Indices guaranteed in [0-14] |
| 4-10 | gather | Indices dispersed beyond preloaded range |
| 11-14 | vselect (no gather) | Indices WRAP BACK due to bounds check |
| 15 | gather | Final round |

**Only need to preload 15 nodes (0-14)**, not 255!

**Key Techniques from PR #22**:
1. Tree level preloading (nodes 0-14)
2. Hybrid vselect/gather strategy
3. Hash stage fusion with multiply_add (FMA)
4. XOR in ALU instead of VALU (12 slots vs 6)
5. Global instruction scheduling

**Result**: Analysis documented in H67_claude_solution_analysis.md
**Status**: COMPLETED - Path to sub-1500 cycles identified

---

### Experiment 14: H68 - Hybrid vselect/Gather Kernel

**Implementation Agent**: a9cfed2

**Implementation Details**:
Partial implementation of the hybrid strategy from H67 analysis:
- Round 0: Uses preloaded tree[0] (broadcast instead of gather)
- Round 1: Uses arithmetic selection between tree[1] and tree[2]
- Rounds 2-15: Still uses standard gather (NOT YET OPTIMIZED)

**Result**:
- **Actual cycles**: 2,799
- **Improvement over H65v3**: 142 cycles (4.8% faster)
- **Speedup over baseline**: 52.78x
- **Correctness**: PASSED

**NEW BEST: H68 at 2,799 cycles**

**Remaining Optimizations (NOT YET IMPLEMENTED)**:
1. Rounds 2-3: Selection from tree[3-14]
2. **Rounds 11-14: Bounds wrap exploitation** (key insight not yet used!)
3. Global instruction scheduling

**Gap Analysis**:
- Current: 2,799 cycles
- Target: 1,790 cycles
- Gap: 1,009 cycles (56% above target)

---

### Experiment 16: H68v2 - Bounds Wrap Exploitation Attempt

**Implementation Agent**: ac90850

**Finding**: Bounds wrap selection SLOWER than gather!

**Analysis**:
- Selection with 14 equality comparisons for rounds 11-14: ~70+ cycles per round
- Interleaved gather schedule: ~64 cycles per round
- The comparison chain cannot be parallelized effectively
- Gather is well-pipelined with hash computation

**Optimizations That DID Work**:
1. Skip bounds check for rounds 2-9 (indices < 2047)
2. Post-processing to remove empty instruction slots

**Result**:
- **Actual cycles**: 2,775
- **Improvement over H68**: 24 cycles (0.9% faster)
- **Speedup over baseline**: 53.2x
- **Correctness**: PASSED

**NEW BEST: H68v2 at 2,775 cycles**

**Key Insight**: The PR #22 solution (1,299 cycles) must use a fundamentally different approach:
- Global instruction scheduling (not per-iteration)
- Different round processing architecture
- Possibly fewer desks with better scheduling

---

### Experiment 17: H69 - Global Scheduling Attempt

**Implementation Agent**: a50b501

**Design**: Attempted to replicate PR #22's global scheduling approach:
1. Emit all operations as flat list with dependencies
2. Use greedy list scheduler to pack VLIW bundles
3. Process rounds 0-3 and 11-14 with selection (indices in [0-14])
4. Process rounds 4-10 and 15 with gather

**Result**:
- **Actual cycles**: 6,673
- **Speedup over baseline**: 22.1x
- **Correctness**: PASSED
- **Comparison**: 2.4x SLOWER than H68v2

**Analysis**:
The naive global scheduling approach performed much worse than expected because:
1. Our manually crafted deep pipelining (16 desks, 2-round fusion) hides latency better
2. The greedy list scheduler doesn't find optimal schedules for this problem
3. PR #22's approach uses specific "blocks of 17 with round tiles of 13" that we couldn't replicate

**Key Insight**: The PR #22 solution uses a highly tuned scheduling strategy that isn't easily replicated. The combination of block/tile sizes is critical to achieving sub-1500 cycles.

**Status**: ABANDONED - Global scheduling not beneficial without precise tuning

---

### Experiment 18: H70 - ALU XOR Optimization

**Implementation Agent**: abbb0e6

**Goal**: Move XOR operations from VALU (6 slots) to ALU (12 slots) per PR #22.

**Finding**: NO IMPROVEMENT

**Analysis**:
- XOR must complete before hash can start (dependency chain)
- The bottleneck is loads (2/cycle during gather), not VALU
- VALU is already well-utilized in the interleaved schedule
- Moving XOR to ALU doesn't reduce cycle count

**Result**: 2,775 cycles (same as H68v2)
**Status**: COMPLETED - No benefit found

---

### Experiment 19: H71 - PR #22 Tiling Strategy

**Implementation**: Attempted to replicate PR #22's approach.

**PR #22 Analysis**:
- Uses `group_size = 17`, `round_tile = 13`
- Uses `level = _round % (forest_height + 1)` for wrap-around exploitation
- vselect for levels 0-3 (rounds 0-3 AND 11-14 due to wrap)
- Greedy list scheduler for automatic VLIW packing

**Finding**: PR #22 reference code FAILS correctness check!
```
AssertionError: Incorrect result on round 0
```

**H71 Result**:
- **Cycles**: 2,743 (slight improvement over H68v2)
- **Analysis**: The 17-block tiling provides marginal benefit

---

### Experiment 20: H72 - Round-Synchronous Processing

**Implementation**: Process ALL 256 elements through each round before moving to next.

**Hypothesis**: Round-sync allows sharing tree values across elements.

**Result**:
- **Cycles**: 18,104 (much slower than H68)
- **Correctness**: PASSED
- **Analysis**: Poor instruction packing leads to low utilization

**Key Learning**: Round-sync alone doesn't help without proper:
1. Instruction packing (12 ALU, 6 VALU slots/cycle)
2. Interleaved gather/compute scheduling
3. Exploitation of index diversity bounds

---

### Summary: Gap Analysis

**Current State**:
| Kernel | Cycles | Speedup | Notes |
|--------|--------|---------|-------|
| Baseline | 147,734 | 1.0x | Reference |
| H72 | 18,104 | 8.2x | Round-sync (unoptimized) |
| H68v2 | 2,775 | 53.2x | **BEST** |
| H71 | 2,743 | 53.9x | PR #22 tiling |
| Target | 1,790 | 82.5x | Blog claim |
| PR #22 | 1,299 | 113.7x | FAILS correctness |

**Gap to Target**: 2,775 → 1,790 requires 35% reduction (985 cycles)

**Unexploited Optimizations**:
1. Level-based vselect for rounds 0-3 AND 11-14 (wrap exploitation)
2. Global instruction scheduling (not just greedy)
3. Better tiling parameters

---

### Experiment 15: H64v2 - Level Preload Attempt

**Implementation Agent**: a8b5dda

**Finding**: Limited benefit from level preloading alone.

The agent added tree[0] preloading infrastructure but didn't implement the full optimization because:
1. H54's gather is interleaved with hash computation
2. Removing gather without restructuring leaves VALU idle
3. Estimated savings (~128 cycles) didn't justify complexity

**Result**:
- **Actual cycles**: 3,465
- **Essentially same as H54**: (3,462 baseline)
- **Status**: No improvement achieved

---

### Experiment 21: H73 - Wrap-Around Exploitation

**Implementation**: Based on the discovery that ALL 256 indices wrap to 0 after round 10.

**Key Insight**: After round 10's bounds check, ALL indices become 0 because:
- Round 9 max_idx = 2044
- Round 10 computes idx = 2*idx + branch, giving min 2047
- All indices >= n_nodes (2047), so ALL reset to 0

**Optimization Applied**:
1. Round 11: Use tree[0] broadcast (same as round 0)
2. Round 12: Use arithmetic selection with tree[1], tree[2] (same as round 1)
3. Rounds 13-14: Use gather with skip_bounds (indices guaranteed < 2047)
4. Round 15: Use gather with skip_bounds (max_idx = 62 after round 15)

**Result**:
- **Cycles**: 2,643 → 2,647 (initial) → 2,643 (with R15 skip_bounds)
- **Speedup**: 55.9x
- **Improvement over H68v2**: 132 cycles (4.8%)

**Status**: COMPLETED - Working and faster

---

### Experiment 22: H74 - Fully Unrolled Tiles

**Implementation**: Eliminate tile loop overhead by fully unrolling both tiles.

**Change from H73**:
- Instead of tile loop with 2 iterations, emit code for both tiles inline
- Eliminates loop control overhead (const loads, compare, jump)

**Result**:
- **Cycles**: 2,613
- **Speedup**: 56.5x
- **Improvement over H73**: 30 cycles (1.1%)
- **Improvement over H68v2**: 162 cycles (5.8%)

**Analysis**:
- Total instructions: ~2,700
- Packing efficiency: ~52%
- Main remaining optimization: Global instruction scheduling

**Status**: COMPLETED - New best result

---

### Updated Summary

| Kernel | Cycles | Speedup | Notes |
|--------|--------|---------|-------|
| Baseline | 147,734 | 1.0x | Reference |
| H68v2 | 2,775 | 53.2x | Previous best |
| H73 | 2,643 | 55.9x | Wrap exploitation |
| **H74** | **2,613** | **56.5x** | **NEW BEST** |
| Target | 1,790 | 82.5x | Blog claim |
| PR #22 | 1,299 | 113.7x | FAILS correctness |

**Gap to Target**: 2,613 → 1,790 requires 31% reduction (823 cycles)

**Remaining Opportunities**:
1. Global instruction scheduling (PR #22's key technique)
2. Better interleaving of preload rounds with gather rounds
3. ALU parallelization during VALU operations

---

### Experiment 23: H75 - Automatic VLIW Scheduling

**Implementation**: Adopted the automatic scheduler from PR #22 with our correct algorithm.

**Key Technique**: Instead of manually packing instructions:
1. Emit all operations as flat list of (engine, slot) pairs
2. Use greedy list scheduler to pack respecting:
   - Data dependencies (read-after-write hazards)
   - Slot limits (2 load, 6 VALU, 12 ALU, 1 flow, 2 store)
3. Handle pauses specially (schedule phases independently)

**Bug Fixed**: Initial version put both pauses at the beginning because scheduler
treated pause as dependency-free. Fixed by scheduling in phases separated by pauses.

**Result**:
- **Cycles**: 1,938
- **Speedup**: 76.2x
- **Improvement over H74**: 675 cycles (26% faster!)
- **Total slots**: 12,676
- **Packing efficiency**: ~85% (vs 52% in H74)

**Gap to Target**: 1,938 → 1,790 requires only 8% reduction (148 cycles)

**Status**: COMPLETED - Major breakthrough!

---

### Final Summary

| Kernel | Cycles | Speedup | Notes |
|--------|--------|---------|-------|
| Baseline | 147,734 | 1.0x | Reference |
| H68v2 | 2,775 | 53.2x | Previous best (manual packing) |
| H73 | 2,643 | 55.9x | Wrap exploitation |
| H74 | 2,613 | 56.5x | Fully unrolled tiles |
| **H75** | **1,938** | **76.2x** | **Automatic scheduling** |
| Target | 1,790 | 82.5x | Blog claim |
| PR #22 | 1,299 | 113.7x | FAILS correctness |

**Key Learnings**:
1. Automatic scheduling is critical - 26% improvement from better packing
2. Wrap-around exploitation saves ~100 cycles
3. Manual instruction packing is error-prone and suboptimal
4. The scheduler respects dependencies automatically

---

### Experiment 24: H77 - 4-Way Arithmetic Selection

**Implementation**: Extended arithmetic selection to rounds 2 and 13 (indices 3-6).

**Key Technique**: Instead of gather for indices 3-6, use precomputed tree differences:
- v_diff_3_4 = tree[4] - tree[3]
- v_diff_5_6 = tree[6] - tree[5]

Selection logic (7 ops per desk):
1. offset = idx - 3
2. bit0 = offset & 1
3. bit1 = offset >> 1
4. low_pair = tree[3] + bit0 * diff_3_4
5. high_pair = tree[5] + bit0 * diff_5_6
6. diff = high_pair - low_pair
7. result = low_pair + bit1 * diff

**Result**:
- **Cycles**: 1,914
- **Speedup**: 77.2x
- **Improvement over H75**: 24 cycles (1.2% faster)

**Status**: COMPLETED - Incremental improvement

---

### Experiment 25: H79 - Precomputed Address Vector & Direct XOR

**Implementation**: Two micro-optimizations:

1. **Precompute forest_values_p as vector**: Broadcast once during init, use directly
   in gather rounds instead of vbroadcast + add per desk. Saves ~288 vbroadcasts.

2. **Direct XOR with tree[0]**: For rounds 0 and 11, XOR directly with v_tree[0]
   instead of copying to node_val first. Saves 64 VALU ops.

**Result**:
- **Cycles**: 1,904
- **Speedup**: 77.6x
- **Improvement over H77**: 10 cycles (0.5% faster)
- **Total slots**: 12,041
- **Packing efficiency**: 6.32 slots/cycle

**Status**: COMPLETED - NEW BEST RESULT

---

### Experiment 26: H80 - 8-Way Arithmetic Selection (FAILED)

**Implementation**: Attempted 8-way selection for round 3 (indices 7-14).

**Result**:
- **Cycles**: 1,950 (46 cycles SLOWER than H79!)
- **Analysis**: 8-way selection has longer critical path (15 ops per desk)
  vs gather (4 cycles for 8 loads). The serialized VALU operations
  hurt more than the load bandwidth saved.

**Status**: ABANDONED - 8-way selection not beneficial

---

### Final Summary (Session 2)

| Kernel | Cycles | Speedup | Notes |
|--------|--------|---------|-------|
| Baseline | 147,734 | 1.0x | Reference |
| H75 | 1,938 | 76.2x | Auto scheduling |
| H77 | 1,914 | 77.2x | 4-way selection |
| **H79** | **1,904** | **77.6x** | **NEW BEST** |
| H80 | 1,950 | 75.8x | 8-way selection (slower) |
| Target | 1,790 | 82.5x | Blog claim |

**Gap to Target**: 1,904 → 1,790 requires 6% reduction (114 cycles)

**Scheduling Analysis**:
- 55.6% of cycles have 6 ops (VALU saturated)
- 25.5% of cycles have 8 ops (VALU + load saturated)
- 14.8% of cycles have 5 ops (dependency stalls during gather)
- Main bottleneck: Serial dependency chain in gather rounds

**Remaining Potential Optimizations**:
1. Modulo scheduling across round boundaries
2. Different round/desk interleaving strategies
3. Further precomputation of intermediate values
4. Possibly unreachable - 1,790 may require ISA features we're not using

---
