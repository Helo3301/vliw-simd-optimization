# Genetic Algorithm Evolution Log

## Mission
Evolve kernel structures to find sub-1,400 cycle solutions for the VLIW SIMD tree traversal kernel.

- **Target:** 1,363 cycles
- **Starting best:** 1,645 cycles (H140)
- **Gap to close:** 282 cycles (17.2%)

---

## Final Results

| Metric | Value |
|--------|-------|
| **Best achieved** | **1,638 cycles** |
| Best genome | GROUP_SIZE=1, NUM_PRELOADED=7 |
| Improvement over H140 | 7 cycles (0.4%) |
| Speedup over baseline | 90.19x |
| Target achieved | **NO** (275 cycles short) |

---

## Evolution Phases

### Phase 1: Parameter Sweep GA (100 generations)
Simple genetic algorithm tuning GROUP_SIZE and NUM_PRELOADED parameters.

- Population size: 15
- Mutation rate: 40%
- Selection: Tournament (size 3)
- Elitism: Keep top 2

**Convergence:** Generation 1 found optimum, no improvement through 100 generations.

### Phase 2: Systematic Parameter Sweep
Exhaustive grid search over all parameter combinations.

**Parameters tested:**
- GROUP_SIZE: 1, 2, 3, 4, 5, 6, 8, 16
- NUM_PRELOADED: 3, 5, 7, 9, 11, 15

**Top 10 Results:**

| Rank | Cycles | GROUP_SIZE | NUM_PRELOADED |
|------|--------|------------|---------------|
| 1 | 1638 | 1 | 7 |
| 2 | 1639 | 1 | 9 |
| 3 | 1641 | 1 | 11 |
| 4 | 1644 | 6 | 7 |
| 5 | 1645 | 1 | 15 |
| 6 | 1645 | 4 | 7 |
| 7 | 1645 | 6 | 9 |
| 8 | 1646 | 4 | 9 |
| 9 | 1647 | 6 | 11 |
| 10 | 1648 | 2 | 7 |

**Key Finding:** GROUP_SIZE=1 consistently outperforms larger group sizes.

### Phase 3: Structural Mutation Variants

| Variant | Description | Result |
|---------|-------------|--------|
| V1: Deep Interleave | 6-desk batch emission | Scratch overflow / incorrect |
| V2: Reduced Ops | GROUP_SIZE=1, all optimizations | 1,638 cycles |
| V3: 6-Wide VALU | Match VALU slot count | 1,644 cycles |

---

## Theoretical Analysis

### Why 1,363 is Unreachable with Current Algorithms

From THEORETICAL_MINIMUM_PROOF.md:

| Bound | Cycles | Notes |
|-------|--------|-------|
| **VALU-bound minimum** | **1,514** | 9,083 ops / 6 slots |
| Target | 1,363 | Requires 8,178 ops max |
| Gap | 905 ops | 10% reduction needed |

**VALU Operations Breakdown (per desk, 32 desks total):**

| Component | Ops/desk | Total | Status |
|-----------|----------|-------|--------|
| Hash | 192 | 6,144 | IRREDUCIBLE |
| XOR | 16 | 512 | IRREDUCIBLE |
| Branch | 45 | 1,440 | Reducible to 30? |
| Bounds | 2 | 64 | Fixed |
| Selection | 18 | 576 | Partially reducible |
| Address | 10 | 320 | Fixed |
| **Total** | **283** | **9,056** | + 27 setup |

**To reach 1,363 cycles, would need:**
1. 2-op branch formulation (saves 480 ops) - **UNPROVEN**
2. Optimized 4-way selection (saves 128 ops)
3. Unknown algorithmic improvement (saves ~300 ops)

### Scheduler Efficiency

| Metric | Value |
|--------|-------|
| Achieved | 1,638 cycles |
| Theoretical | 1,514 cycles |
| Overhead | 124 cycles (8.2%) |

The 8.2% overhead is due to:
- Data dependencies preventing perfect VALU packing
- Load/store operations
- Non-VALU operations (ALU, FLOW)

---

## Evolutionary Dead Ends

1. **GROUP_SIZE > 1:** Always worse due to reduced scheduler flexibility
2. **NUM_PRELOADED < 7:** Causes correctness failures
3. **NUM_PRELOADED > 7:** Marginal benefit, wastes scratch
4. **Different interleaving:** Scheduler already optimizes well
5. **6-wide VALU batches:** Worse than GROUP_SIZE=1

---

## Conclusion

**The GA has converged to a local optimum at 1,638 cycles.**

The evolution explored:
- 100 generations of genetic evolution
- 48 parameter combinations (systematic sweep)
- 3 structural variants

All paths led to the same conclusion: **without reducing the total VALU operation count, sub-1,400 cycles is impossible**.

The gap to 1,363 cycles represents algorithmic improvements beyond what we've discovered:
- Novel branch computation (2 ops instead of 3)
- Undiscovered operation fusions
- Completely different kernel structure

---

## Files Generated

| File | Description |
|------|-------------|
| `experiments/GA_evolution/ga_framework.py` | GA framework |
| `experiments/GA_evolution/ga_runner.py` | Parameter evolution |
| `experiments/GA_evolution/systematic_sweep.py` | Grid search |
| `experiments/GA_evolution/kernel_v2_reduced_ops.py` | Best variant |
| `research_swarm/GA_SWEEP_RESULTS.md` | Full results |

---

## Recommendations for Future Work

1. **Formal verification** of 2-op branch impossibility
2. **ILP-based scheduler** to eliminate remaining 8.2% overhead
3. **ISA analysis** for undiscovered operation combinations
4. **Alternative algorithms** that reduce total operations
