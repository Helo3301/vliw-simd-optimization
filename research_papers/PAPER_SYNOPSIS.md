# ArXiv Paper Research Synopsis

**Date:** 2026-01-24
**Session:** Autonomous Research Loop for VLIW SIMD Optimization

## Papers Downloaded

Downloaded 29 papers from arXiv covering:
- VLIW instruction scheduling
- SIMD gather/scatter optimization
- Software pipelining
- Tree traversal vectorization
- Loop unrolling optimization
- Register allocation
- ILP (Instruction Level Parallelism)

## Most Relevant Papers

### 1. SIMD R-tree Paper (2309.16913v1)
**Title:** SIMD-ified R-tree Query Processing and Optimization

**Key Insights:**
- **Data Layout D1 (packed arrays)** outperforms traditional row-major layout for SIMD
- **Software prefetching with BFS queue** hides cache miss latency
- **Compress store with masks** enables efficient queue operations
- Achieved **up to 9x speedup** with vectorization

**Applicable to Our Problem:**
- We're doing tree traversal which has similar access patterns
- Data layout considerations may help
- Prefetching concept maps to our level-aware optimization (H22)

### 2. ILP Paper (1909.06559v1)
**Title:** Instructional Level Parallelism

**Key Insights:**
- **Tomasulo's algorithm** for out-of-order execution via register renaming
- **IBM 360/91 dual-path branch speculation** - fetch from BOTH paths
- WAW hazards overcome with register renaming

**Applicable to Our Problem:**
- H25 (dual-path loading) is directly inspired by this
- We already use pseudo-register-renaming with multiple desks

### 3. Loop Unrolling Papers (1911.03991v1, 1402.0671v1)
**Key Insights:**
- **Optimal unrolling factor varies by program** - DNN predicts ~15% improvement
- Multi-pipeline ASIP shows **15% avg improvement, up to 30% max** with unrolling
- **ILP exposure** is main advantage of loop unrolling

**Applicable to Our Problem:**
- H23 (4x unrolling) directly applies this
- We already have 2x round fusion (H12)

### 4. TreeFuser Paper (1904.07061v1)
**Title:** Sound, Fine-Grained Traversal Fusion for Heterogeneous Trees

**Key Insights:**
- **Traversal fusion** reduces number of passes
- **Code motion** can restructure traversal order (post-order to pre-order)
- **Type-specific partial fusion** allows fine-grained optimization

**Applicable to Our Problem:**
- Round fusion (H12) is already doing this
- Could explore 4-round fusion (H27)

## Technique Frequency Analysis

From all papers:
```
dependency: 24 mentions
parallelism: 17 mentions
vectorization: 16 mentions
register blocking: 11 mentions
gather: 8 mentions
speculative: 6 mentions
software pipelining: 6 mentions
fusion: 5 mentions
loop unrolling: 5 mentions
tiling: 5 mentions
coalescing: 5 mentions
scatter: 4 mentions
latency hiding: 4 mentions
double buffering: 4 mentions
```

## Hypotheses Generated from Papers

### H23: 4x Loop Unrolling
**Based on:** Loop unrolling papers showing optimal factor varies
**Status:** Testing

### H25: Speculative Dual-Path Loading
**Based on:** IBM 360/91's dual-path branch speculation
**Status:** Testing

### H27: 4x Unroll + Round Fusion Combo
**Based on:** Combined techniques from multiple papers
**Status:** Pending

### H28: Pre-computed Branch Masks
**Based on:** ILP paper's focus on reducing dependencies
**Status:** Testing

## Key Architectural Constraints

From problem.py:
- **Load slots:** 2 per cycle (no vgather!)
- **Store slots:** 2 per cycle
- **VALU slots:** 6 per cycle
- **ALU slots:** 12 per cycle
- **Flow slot:** 1 per cycle
- **VLEN:** 8 elements
- **Scratch size:** 1536 words

## Research Conclusions

1. **No vgather instruction** - This is the fundamental bottleneck. All papers assume gather operations exist.

2. **Gather operations are hidden** - Our V1-V5 experiments proved that gather overhead is already masked by VALU work.

3. **Near architectural limit** - C4 at 4,667 cycles is approximately 43.9% of theoretical minimum. Further improvements are limited.

4. **Loop overhead is small** - With 64 iterations, loop control is only ~256 cycles total. 4x unrolling saves at most 192 cycles.

5. **Dual-path loading doubles memory bandwidth** - May not help if we're already compute-bound.

## Files

```
research_papers/
├── autonomous_research.py   # Paper download and analysis script
├── research_log.md          # Download log
├── HYPOTHESES_FROM_PAPERS.md # Generated hypotheses
├── PAPER_SYNOPSIS.md        # This file
└── *.txt / *.pdf            # Downloaded papers
```
