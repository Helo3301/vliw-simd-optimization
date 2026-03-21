# Hypothesis Queue

## Format
Each hypothesis has:
- **ID**: H{number}
- **Source**: Paper/concept that inspired it
- **Hypothesis**: What we think will help
- **Expected Impact**: Estimated cycle reduction
- **Status**: PENDING | DESIGNING | TESTING | COMPLETED | FAILED

---

## Active Queue

### H51: CAPT-Style Branchless Traversal (FROM PAPER)
- **Source**: arxiv.org/abs/2406.02807 - Collision-Affording Point Trees
- **Hypothesis**: Replace conditional branch selection with arithmetic: `idx = idx*2 + 1 + bit`
- **Expected Impact**: Eliminate vselect bottleneck
- **Status**: ALREADY IMPLEMENTED IN H38 - No experiment needed
- **Result**: H38 already uses branchless traversal (4 VALU ops, 0 vselect)

### H52: Multi-Level Jump (FROM H51 ANALYSIS)
- **Source**: H51 design analysis, section 6D
- **Hypothesis**: Instead of computing one tree level at a time, compute 2 levels at once: `idx = idx*4 + 3 + bit0*2 + bit1`. Requires pipelining hash bits.
- **Expected Impact**: Halve the number of tree descent iterations
- **Status**: ABANDONED - Proven mathematically infeasible (hash dependency chain)

### H53: Eliminate Bounds Check (FROM H51 ANALYSIS)
- **Source**: H51 design analysis, section 6A
- **Hypothesis**: Tree is complete up to height 10 (1023 nodes). If we guarantee indices stay valid, remove bounds check entirely.
- **Expected Impact**: Save 2 VALU cycles per desk per round
- **Status**: REJECTED - Bounds check is REQUIRED (indices exceed tree bounds after round 10)
- **Result**: Would break correctness, only ~5% savings even if safe

### H54: 16-Desk Ultra-Deep Pipeline (FROM H52 ANALYSIS)
- **Source**: H52 mathematical analysis - need parallelism > 8 to reach 1,790
- **Hypothesis**: Double desk count to 16 for more ILP. Mathematical minimum with 8 desks is 2,048 cycles.
- **Expected Impact**: Could approach 2,048 theoretical minimum
- **Status**: COMPLETED - NEW BEST: 3,462 cycles (14.8% improvement)
- **Result**: Memory fits (1,012/1,536 words), correctness verified

### H55: Hash Stage Fusion (FROM H52 ANALYSIS)
- **Source**: H52 cycle breakdown - hash is ~9 VALU ops
- **Hypothesis**: Fuse hash stages more aggressively with FMA to reduce total ops
- **Expected Impact**: Reduce compute per round
- **Status**: COMPLETED - H54 ALREADY OPTIMAL
- **Result**: 12 VALU ops/hash = theoretical minimum; FMA fusion already applied

### H56: Decoupled Gather-Compute (FROM H52 ANALYSIS)
- **Source**: H52 cycle analysis - separate gather scheduling from compute
- **Hypothesis**: Pre-schedule gathers to maximize load slot utilization
- **Expected Impact**: Better overlap of gather and compute
- **Status**: PENDING

### H58: Index Grouping / Convergent Processing (FROM PAPER)
- **Source**: arxiv.org/abs/1910.11141 - Batching Control-Intensive Programs
- **Hypothesis**: Group batch elements by their current tree index. In early rounds, ALL elements are at same index - process together with single load. As they diverge, form subgroups.
- **Expected Impact**: Reduce total loads in early rounds where index entropy is low
- **Status**: ABANDONED - ISA constraints prevent implementation
- **Result**: No indirect addressing, vselect bottleneck (1/cycle), only Round 0 would benefit (~2.9%)

### H59: Constraint-Based Optimal Schedule (FROM PAPER)
- **Source**: arxiv.org/abs/1804.02452 - Unison framework
- **Hypothesis**: Use ILP/constraint programming to find provably optimal instruction schedule given the ISA constraints and dependency graph.
- **Expected Impact**: Find any hidden parallelism we've missed
- **Status**: COMPLETED - NEW BEST: 3,158 cycles (8.8% improvement over H54)
- **Result**: ILP solver found 24% potential, implementation achieved 8.8% via store/hash overlap

### H60: Wavefront + Arithmetic Value Selection (NEW)
- **Source**: H58 analysis + arithmetic selection insight
- **Hypothesis**: Process all elements through each round (wavefront), use arithmetic `T1 + bit*(T2-T1)` instead of vselect
- **Expected Impact**: Reduce loads via shared tree values, bypass Flow engine bottleneck
- **Status**: ANALYZED - Limited benefit
- **Result**: Formula verified correct. Broadcast overhead kills approach for rounds 4+. Best case ~2,690 cycles (22% better)

### H42: Index Memoization
- **Source**: Database hash join literature
- **Hypothesis**: In early rounds, many elements share the same tree index. Load each unique index once, broadcast to all elements needing it.
- **Expected Impact**: 20-40% reduction in loads for rounds 0-5
- **Status**: PENDING

### H43: Radix-Sorted Batch Processing
- **Source**: Database radix partitioning
- **Hypothesis**: Sort batch elements by their current index before each round. Elements with same index processed together, single load serves multiple elements.
- **Expected Impact**: Potentially 2x reduction if index overlap is high
- **Status**: PENDING

### H44: Entropy-Aware Round Fusion
- **Source**: Information theory
- **Hypothesis**: Fuse more rounds when entropy is low (early rounds), fewer when high (late rounds). Adaptive fusion depth.
- **Expected Impact**: Better register utilization in early rounds
- **Status**: PENDING

### H45: Sorting Network for Branch Resolution
- **Source**: Bitonic sort, oblivious algorithms
- **Hypothesis**: Replace conditional branch selection with sorting network that always executes both paths and selects result. Removes branch dependency.
- **Expected Impact**: Convert control dependency to data dependency
- **Status**: PENDING

### H46: Wavefront Processing
- **Source**: Systolic arrays, DSP literature
- **Hypothesis**: Process tree as wavefront - all level-0 accesses first across all batches, then level-1, etc. Maximizes temporal locality.
- **Expected Impact**: Dramatic load reduction due to index reuse
- **Status**: PENDING

### H47: State Machine Minimization
- **Source**: Automata theory
- **Hypothesis**: The hash computation is an FSM. Minimize states to find shorter equivalent computation.
- **Expected Impact**: Unknown - depends on hash structure
- **Status**: PENDING

### H48: Speculative Index Precomputation
- **Source**: Branch prediction literature
- **Hypothesis**: Compute both possible next indices (left and right child) speculatively, select after hash completes. Hides hash latency.
- **Expected Impact**: Overlap hash with next gather setup
- **Status**: PENDING

### H49: Bloom Filter Skip
- **Source**: Database query optimization
- **Hypothesis**: Build bloom filter of "interesting" tree nodes. Skip loads for nodes not in filter.
- **Expected Impact**: Only useful if many nodes are never visited
- **Status**: PENDING

### H50: Tree Linearization (Eytzinger Layout)
- **Source**: Cache-oblivious algorithms
- **Hypothesis**: Reorder tree in memory using Eytzinger layout for better cache behavior. Left child at 2i, right at 2i+1 becomes sequential.
- **Expected Impact**: Better memory access patterns
- **Status**: PENDING

### H63: The Missing Trick Analysis
- **Source**: Blog showing Claude Opus 4.5 achieved 1,487 cycles (below our "theoretical minimum")
- **Hypothesis**: Our 2,048 cycle minimum analysis was WRONG. There's a trick we're missing.
- **Expected Impact**: Understand how sub-2000 is achievable
- **Status**: COMPLETED - MAJOR BREAKTHROUGH
- **Result**: All 256 elements start at idx=0. Rounds 0-7 have bounded index diversity. Tree levels are contiguous. vload can preload levels 0-7 in 16 cycles. Corrected minimum: 1,152 cycles.

### H64: Level-Based Preloading + Round Synchronization
- **Source**: H63 analysis
- **Hypothesis**: Preload tree levels 0-7 with vloads (16 cycles), use arithmetic selection for rounds 0-7, fall back to pipelined gather for rounds 8-15
- **Expected Impact**: Target sub-2000 cycles, potentially reaching 1,400-1,600
- **Status**: IMPLEMENTING

### H65: Loop Tiling (Small Batches Through All Rounds)
- **Source**: H63 alternative approach
- **Hypothesis**: Process smaller batches (32 elements) through ALL 16 rounds before moving to next batch. Keep tree values in scratch longer.
- **Expected Impact**: Better scratch locality, reduced reloads
- **Status**: COMPLETED - NO IMPROVEMENT
- **Result**: 3,494 cycles (0.9% slower than H54). Tiling alone doesn't help without tree preloading.

---

## Completed Hypotheses

### H63: The Missing Trick Analysis - COMPLETED
- Found the critical insight: our theoretical minimum was wrong
- Corrected minimum: ~1,152 cycles (not 2,048)
- Path to sub-2000: level preloading + arithmetic selection + round synchronization

---

## Failed/Abandoned Hypotheses

(None yet in this loop)
