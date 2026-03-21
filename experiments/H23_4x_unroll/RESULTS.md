# H23: 4x Loop Unrolling Results

## Summary

| Metric | Value |
|--------|-------|
| H23 Cycles | 4,653 |
| C4 Baseline | 4,667 |
| Improvement | 14 cycles (0.30%) |
| Speedup over C4 | 1.003x |
| Speedup over original baseline | 31.75x |

## Hypothesis

Loop unrolling reduces loop control overhead (increment, compare, branch) by processing multiple iterations per loop body. With 4x unrolling, we reduce loop iterations from 64 to 16.

## Implementation Details

### C4 Baseline Structure
- Processes 2 rounds per iteration (round fusion from H12)
- Uses 4 desks for interleaving
- Total iterations: 64 (8 double-rounds * 8 batches / 4 desks)
- Loop overhead per iteration: ~5-6 cycles (offset calculations, counter increment, compare, branch)

### H23 Changes
1. **4x Unrolled Loop Body**: Each loop iteration processes 4 sets of batches
2. **Pre-computed Base Offsets**: All 4 base offsets computed at loop start
3. **Overlapped Setup**: Store operations overlapped with next unroll copy's address computation
4. **Separate Offset Register Sets**: 4 sets of offset registers (16 total) to avoid conflicts

### Key Optimizations in Optimized Version
- Overlap final VALU operation (desk3 vselect bypass) with store address computation
- Pre-compute next unroll copy's offset registers during stores
- Pipeline first load address computation with stores

## Analysis

### Why Marginal Improvement?

The 4x unrolling provides only a **0.3% improvement** because:

1. **C4 Already Well-Optimized**: C4 has aggressive pipelining with store coalescing (H13) and address pipelining (H14) that already hides much of the loop overhead

2. **Inter-Copy Setup Overhead**: Each unrolled copy still needs:
   - 1 cycle for offset calculations for 4 desks
   - Address computations for vloads
   - Store address computations

3. **Code Size Trade-off**: 4x unrolling quadruples code size, increasing instruction memory pressure

4. **Diminishing Returns**: With only 64 original iterations and ~5-6 cycles loop overhead:
   - Total loop overhead: ~320-384 cycles
   - Savings from 4x unroll: ~240-288 cycles theoretically
   - But inter-copy setup costs most of it back

### Loop Overhead Breakdown

| Component | Original (64 iters) | 4x Unrolled (16 iters) |
|-----------|---------------------|------------------------|
| Counter increment | 64 | 16 |
| Counter compare | 64 | 16 |
| Batch offset update | 64 | 16 |
| Batch offset compare | 64 | 16 |
| Select (wrap-around) | 64 | 16 |
| Conditional jump | 64 | 16 |
| Inter-copy setup | 0 | 48 (3 copies * 16 iters) |

The inter-copy setup overhead (computing new offset registers and prepping next load addresses) consumes much of the savings.

## Conclusion

**H23 Result: MARGINAL IMPROVEMENT (+0.3%)**

4x loop unrolling provides a small but measurable improvement over C4. The technique works, but C4's existing optimizations (store coalescing, address pipelining) already minimize the impact of loop overhead.

This suggests we may be approaching the optimization limit for this kernel structure. Further gains would likely require:
- Algorithmic changes (different hash computation)
- Memory access pattern optimization
- Increased parallelism (more desks, if register pressure allows)

## Verification

```
$ python3.11 perf_takehome_h23.py --check
forest_height=10, rounds=16, batch_size=256
CYCLES:  4653
Speedup over baseline:  31.750268643885665
Speedup over C4 (4667):  1.0030088115194498
Correctness check PASSED! Cycles: 4653
```
