# Experiment H19: 8-Desk Extreme ILP for VLIW SIMD

## Summary

**Hypothesis:** 8 desks with H7-style aggressive interleaving (NOT sequential like T4) could provide better ILP by having more independent operations to pack per cycle.

**Result:** NEGATIVE - 8 desks does not improve over 4 desks.

## Performance Results

| Metric | Value |
|--------|-------|
| H19 Cycles | 6,334 |
| H7 Cycles (4-desk) | 6,203 |
| T4 Cycles (8-desk sequential) | 9,792 |
| Baseline | 147,734 |
| H19 Speedup vs Baseline | 23.3x |
| H7 Speedup vs Baseline | 23.8x |
| H19 vs H7 | -2.1% (REGRESSION) |
| H19 vs T4 | +35.3% (improvement over sequential 8-desk) |

## Analysis

### Why H19 Failed to Improve Over H7

1. **Sequential 4-desk blocks don't add ILP**
   - H19 runs two 4-desk blocks sequentially per iteration
   - Each block follows H7's interleaving pattern independently
   - No overlap between the two blocks

2. **Extra overhead**
   - Loading 8 desks' idx/val at start: 8 vload cycles (4 extra vs H7)
   - Two separate gather/hash pipelines
   - Two separate store phases (8 stores vs 4)

3. **Gather bottleneck**
   - Each gather takes 4 cycles (8 lanes, 2 loads/cycle)
   - The machine has only 2 load slots per cycle
   - More desks can't overlap their gathers efficiently

### Why T4's Sequential Approach Failed (9,792 cycles)

T4 processed desks 0-7 fully sequentially:
- Complete desk0 (gather, hash, branch, store)
- Complete desk1 (gather, hash, branch, store)
- ...and so on

This is 2x slower because:
- No interleaving between desks
- All VALU slots idle during gather phases
- All load slots idle during hash phases

### Why H7's 4-Desk Approach is Optimal

H7's 4-desk interleaving achieves good balance:
- While desk0 gathers, no VALU work available yet
- While desk1 gathers, desk0 hashes (VALU busy)
- While desk2 gathers, desk0-1 hash (VALU full)
- While desk3 gathers, desk0-2 hash/branch (VALU full)

With 4 desks, the gather and hash phases overlap well. Adding more desks doesn't help because:
- Load slots are already fully utilized during gather
- VALU slots are already well-utilized during hash
- The pipeline is already "full" with 4 desks

## Conclusion

**H19 disproves the hypothesis that 8 desks provide better ILP than 4 desks.**

Key insight: The bottleneck in this kernel is the gather phase (2 loads/cycle for 8 lanes = 4 cycles per desk). True 8-desk interleaving would require overlapping the store phase of early desks with the gather phase of later desks, but:

1. Store and load engines operate independently but both need memory bandwidth
2. The pipeline depth with 4 desks already fills available execution slots
3. 8 desks just doubles the work without improving parallelism

The 4-desk H7 approach remains optimal at 6,203 cycles.

## Files

- Kernel: `/home/hestiasadmin/projects/original_performance_takehome/experiments/H19_8desk_ilp/perf_takehome_h19.py`
- Correctness: PASSED
- Scratch usage: 588 / 1536 (38%)
