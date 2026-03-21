# Experiment T4: 8-Desk Deep Pipeline - Results

## Summary

| Metric | Value |
|--------|-------|
| Baseline (4-desk) | 9,793 cycles |
| T4 (8-desk) | 9,792 cycles |
| Improvement | ~0% (1 cycle) |
| Expected | 50% (5,000-6,500 cycles) |
| Status | Did not achieve expected improvement |

## Experiment Details

### Configuration
- 8 desks (doubled from 4)
- 64 elements per iteration (8 desks x 8 VLEN)
- 64 loop iterations (reduced from 128)
- Scratch usage: 560 words (36% of 1,536)

### Implementation Approaches Tried

1. **Simple Sequential 8-Desk**: Process desks 0-7 sequentially with hash/gather overlap between adjacent desks. Same structure as 4-desk but with 8 desks.

2. **Dual-Stream Processing**: Attempted to process pairs (0,4), (1,5), (2,6), (3,7) with shared VALU operations. Failed correctness due to complex data dependencies.

3. **Interleaved Hash/Gather**: Attempted to pack two desks' hash operations into the same VALU cycles. Theoretically sound but didn't yield improvements.

## Analysis

### Why 8-Desk Did Not Improve Over 4-Desk

The key finding is that the cycle count scales linearly with work done:
- 4-desk: 128 iterations x ~76 cycles = ~9,728 base cycles
- 8-desk: 64 iterations x ~153 cycles = ~9,792 base cycles

The ratio of 153/76 = 2.01x means we're doing approximately 2x the work per iteration but taking 2x the time. There is no net improvement because:

1. **Same Overlap Pattern**: With 8 desks, we still overlap desk N's hash with desk N+1's gather. The pattern is the same, just repeated more times per iteration.

2. **VALU Bottleneck**: Each hash stage requires a sequential dependency chain:
   ```
   tmp1 = val op1 const  \
   tmp2 = val op3 shift  / parallel
   val = tmp1 op2 tmp2     \ sequential
   ```
   We cannot compute two different desks' hash stages in the same cycles because each stage depends on the previous.

3. **Memory Bandwidth Unchanged**: We still do 2 loads per cycle for gather. With 8 desks, we do 8 gathers (32 loads total) instead of 4 gathers (16 loads total). The overlap remains the same proportionally.

4. **Loop Overhead Reduction is Minimal**: Going from 128 to 64 iterations saves ~5 cycles per iteration x 64 = ~320 cycles. This is offset by additional offset computation overhead with 8 desks.

### The Real Bottleneck

Per-desk processing takes approximately:
- XOR: 1 cycle
- Hash stages 0-1: 5 cycles (with some packing)
- Hash stages 2-5 + gather overlap: 8 cycles
- Branch: 4 cycles
- Store: 1 cycle
- **Total: ~19 cycles per desk**

With 8 desks, this is 152 cycles minimum. The overlap between desk 7 and desk 0 of the next iteration saves ~4-6 cycles, giving us ~146-148 cycles per iteration. Observed: 153 cycles (includes loop control overhead).

### What Would Be Needed for 50% Improvement

To achieve 5,000 cycles (50% improvement), we would need approximately 78 cycles per iteration (64 elements). That's ~1.2 cycles per element. Currently we're at ~2.4 cycles per element.

Potential approaches that were NOT attempted:
1. **Multi-Iteration Modulo Scheduling (T1)**: Start iteration N+1 while iteration N is still in flight
2. **16+ Desk Ultra-Deep Pipeline**: More desks = better amortization, but diminishing returns
3. **Algorithmic Changes**: Restructure the hash to reduce stage dependencies

## Conclusions

The T4 experiment demonstrates that simply increasing the unroll factor (number of desks) does not improve performance when:
- The same overlap pattern is applied
- Sequential dependencies within each desk's processing remain
- Memory bandwidth per cycle is unchanged

The key insight is that the 4-desk baseline has already captured most of the hash/gather overlap benefit. Going to 8 desks just doubles the work per iteration without improving the fundamental cycle efficiency.

## Files

- `perf_takehome_t4.py` - 8-desk implementation (working, correct)
- Scratch usage: 560/1536 (36%)
- Correctness: PASSED

## Recommendations for Future Experiments

1. **T1 (Modulo Scheduling)** is the most promising path - overlap across iteration boundaries, not just desk boundaries
2. Consider hybrid approaches: 4 desks with 2-deep iteration pipelining
3. Investigate whether the flow unit (only 1 slot) is a bottleneck for vselect operations
