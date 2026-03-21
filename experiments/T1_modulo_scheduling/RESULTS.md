# T1 Experiment Results: Modulo-Scheduled Multi-Iteration Pipeline

## Summary

This experiment attempted to implement modulo scheduling to overlap multiple loop iterations. The goal was to reduce the effective Initiation Interval (II) from ~82 cycles to ~20 cycles, achieving a 2-4x speedup.

## Results

| Implementation | Cycles | Speedup vs Original | Notes |
|---------------|--------|---------------------|-------|
| Original baseline (147,734 cycles) | - | 1.0x | Unoptimized |
| Current baseline (4-desk) | 9,793 | 15.1x | Already highly optimized |
| 8-desk pipeline (T1) | 9,858 | 15.0x | Slightly worse than baseline |

**Outcome: No improvement achieved.** The 8-desk pipeline implementation is 0.7% slower than the baseline.

## Analysis

### Why Modulo Scheduling Didn't Help

1. **Baseline is already well-optimized**: The current 4-desk implementation already overlaps gather operations with hash computation. Adding more desks doesn't increase parallelism because the bottleneck is elsewhere.

2. **Limited by memory bandwidth**: With 2 load slots and 2 store slots per cycle, the memory operations are already maximally utilized. More desks just spreads the work without reducing total cycles.

3. **Register pressure**: 8 desks require 8x the vector registers for idx, val, node_val, tmp1, tmp2, addr. This doesn't improve throughput.

4. **Loop overhead**: Each desk in the loop body has fixed overhead (XOR, 6 hash stages, branch, store). More desks means more total instructions.

### What True Modulo Scheduling Would Require

True modulo scheduling would need:
- Start iteration N+1 while iteration N is still in flight
- Multiple iterations sharing the SAME register bank with proper renaming
- A modulo resource reservation table (MRT) to track conflicts

The challenge is that each iteration WRITES back to memory, creating a data dependency. Iteration N+1 cannot read its data until iteration N writes it (for the same batch offset).

However, since we process the same batch across multiple rounds, there IS opportunity for round-level overlap:
- Round R, element i reads input at offset i
- Round R computes and writes to offset i
- Round R+1, element i reads the UPDATED value at offset i

This means we could potentially overlap Round R's final stages with Round R+1's initial stages for DIFFERENT elements.

### Attempted Approaches

1. **8-desk sequential**: Process 8 groups per loop instead of 4. Result: Same cycles, just fewer loop iterations.

2. **Phase overlap**: While desks 0-3 hash, desks 4-7 gather. Result: Bug-prone due to store offset tracking; when fixed, no performance benefit because the overlap was already happening at the desk level.

3. **Cross-iteration overlap**: Start next iteration's gather during current iteration's hash. Result: Data dependencies prevent this for the same element.

## Lessons Learned

1. **The baseline is near-optimal for this architecture**: The 4-desk pipeline with speculative loading achieves very good utilization of all execution slots.

2. **Memory bandwidth is the real bottleneck**:
   - Each vector group needs: 2 vloads + 8 scalar loads (gather) + 2 vstores = 12 memory ops
   - At 2 load + 2 store per cycle, minimum is 4 cycles for gather + 1 for store = 5 cycles per group
   - Current: ~19 cycles per group (includes hash overhead)

3. **Further optimization requires architectural changes**:
   - A vgather instruction would reduce gather from 4 cycles to 1
   - More load/store slots would increase memory bandwidth
   - Neither is available in this architecture

## Recommendations

For future optimization attempts:

1. **Focus on reducing per-group overhead**: The hash stage takes 12 cycles (6 stages x 2 cycles). Can any stages be merged or eliminated?

2. **Consider batch-level reorganization**: Process all elements through one round before starting the next round could reduce round-switching overhead.

3. **Explore algebraic optimizations**: Can the hash function be simplified? (See T6 experiment)

## Files

- `perf_takehome_t1.py`: 8-desk pipeline implementation (correct but not faster)

## Conclusion

The T1 experiment demonstrates that the current 4-desk baseline is already well-optimized for the given architecture constraints. Modulo scheduling across iterations doesn't help because:
1. Memory bandwidth is the bottleneck, not compute
2. Data dependencies between rounds prevent true iteration-level parallelism
3. The existing desk-level overlap already achieves most of the available parallelism

The theoretical minimum II of ~16 cycles is not achievable without architectural changes (vgather instruction) or algorithmic changes (different hash function).
