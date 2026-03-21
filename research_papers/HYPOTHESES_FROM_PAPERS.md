# Hypotheses Generated from ArXiv Papers

## Paper Insights Summary

### From SIMD R-tree Paper (2309.16913v1)
- **Key insight:** Data layout matters for SIMD - packed arrays per attribute (D1) outperforms traditional layout (D0)
- **Key insight:** Software prefetching with BFS queue hides cache miss latency
- **Key insight:** Compress store with masks enables efficient queue operations
- **Speedup achieved:** Up to 9x with vectorization

### From ILP Paper (1909.06559v1)
- **Key insight:** Tomasulo's algorithm for out-of-order execution via register renaming
- **Key insight:** IBM 360/91 fetched from BOTH paths of branch (speculative)
- **Key insight:** WAW hazards overcome with register renaming

### From Loop Unrolling Papers (1911.03991v1, 1402.0671v1)
- **Key insight:** Optimal unrolling factor varies by program - DNN achieves ~15% improvement
- **Key insight:** Multi-pipeline ASIP shows 15% avg improvement, up to 30% max with unrolling
- **Key insight:** ILP exposure is main advantage of loop unrolling

### From TreeFuser Paper (1904.07061v1)
- **Key insight:** Traversal fusion reduces number of passes
- **Key insight:** Code motion can restructure traversal order (post-order to pre-order)
- **Key insight:** Type-specific partial fusion allows fine-grained optimization

## New Hypotheses

### H23: 4x Loop Unrolling
**Based on:** Loop unrolling papers showing optimal factor varies
**Hypothesis:** Processing 4 double-rounds per loop iteration may reduce loop overhead further
**Expected improvement:** 3-5% (halve loop overhead again)

### H24: Memory Access Reordering (Interleaved Tree Layout)
**Based on:** R-tree paper's D1 layout (packed arrays per attribute)
**Hypothesis:** If tree nodes were laid out differently, gather operations might have better locality
**Challenge:** Tree structure is given, but we can experiment with access patterns

### H25: Speculative Dual-Path Loading
**Based on:** IBM 360/91's dual-path branch speculation
**Hypothesis:** Load BOTH children of each node (left and right), then discard unused one
**Expected impact:** 2x loads but eliminates branch-dependent latency
**Trade-off:** Memory bandwidth vs latency

### H26: Depth-First to Level-First Reordering
**Based on:** R-tree BFS queue insight + our H22 level-aware finding
**Hypothesis:** Reorder batch elements so same tree levels are processed contiguously
**Expected improvement:** Better cache/scratchpad locality

### H27: Combined 4x Unroll + Round Fusion
**Based on:** All papers emphasizing combined optimizations
**Hypothesis:** 4 double-rounds per loop body = 8 rounds without intermediate stores
**Challenge:** Register pressure

### H28: Arithmetic Simplification in Branch Prediction
**Based on:** ILP paper's focus on reducing dependencies
**Hypothesis:** Pre-compute branch direction masks for multiple rounds
**Expected improvement:** Reduce ALU dependency chain

## Testing Priority
1. H23 (4x Unroll) - Low risk, builds on proven H15
2. H25 (Dual-Path) - Novel approach, high potential impact
3. H27 (4x Unroll + Fusion) - Ambitious combination
4. H28 (Branch Mask) - Targets ALU dependency chain
