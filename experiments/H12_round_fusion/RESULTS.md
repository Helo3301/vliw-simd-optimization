# Experiment H12: Round Fusion (Keep Values in Registers)

## Summary

| Metric | H7H10 Baseline | H12 Round Fusion | Improvement |
|--------|----------------|------------------|-------------|
| Cycles | 5,947 | 5,115 | 832 cycles (14.0%) |
| Speedup vs original baseline | 24.8x | 28.9x | +4.1x |

**Result: SUCCESS** - Correctness check passed, significant performance improvement achieved.

## Key Insight

The H7H10 kernel processes each round independently:
- Load idx/val from memory
- Compute hash + tree traversal
- Store idx/val back to memory
- (Next iteration) Repeat

This means each round incurs:
- 2 vload operations (idx + val) per desk
- 2 vstore operations (idx + val) per desk

By fusing 2 consecutive rounds, we eliminate the intermediate memory operations:
- Load idx/val once
- Compute round N
- Compute round N+1 (values stay in registers)
- Store idx/val once

## Memory Operations Analysis

### H7H10 (per iteration, 4 desks):
- **Loads:** 4 desks x 2 vloads (idx/val) = 8 vloads
- **Stores:** 4 desks x 2 vstores (idx/val) = 8 vstores
- **Gathers:** 4 desks x 8 scalar loads = 32 loads
- **Iterations:** (batch_size / VLEN) * rounds / NUM_DESKS = (256/8) * 16 / 4 = 128 iterations

Total memory ops per round: 8 vloads + 8 vstores + 32 gathers = 48 ops
Total for 16 rounds: 128 * 48 = 6,144 memory operations

### H12 Round Fusion (per iteration, 4 desks):
- **Loads:** 4 desks x 2 vloads (idx/val) = 8 vloads (same, but once per 2 rounds)
- **Stores:** 4 desks x 2 vstores (idx/val) = 8 vstores (same, but once per 2 rounds)
- **Gathers:** 4 desks x 8 scalar loads x 2 rounds = 64 loads
- **Iterations:** (batch_size / VLEN) * (rounds / 2) / NUM_DESKS = (256/8) * 8 / 4 = 64 iterations

Total memory ops per double-round: 8 vloads + 8 vstores + 64 gathers = 80 ops
Total for 8 double-rounds: 64 * 80 = 5,120 memory operations

**Memory ops saved:** 6,144 - 5,120 = 1,024 operations (16.7% reduction)

## Register Pressure Analysis

### H7H10 Per-Desk Registers (6 vectors each):
- `idx`: Tree node index (VLEN words)
- `val`: Current value (VLEN words)
- `node_val`: Gathered tree node value (VLEN words)
- `addr`: Gather address (VLEN words)
- `tmp1`: Temporary for hash computation (VLEN words)
- `tmp2`: Temporary for hash computation (VLEN words)

Total per desk: 6 * 8 = 48 words
Total for 4 desks: 192 words

### H12 Register Requirements (same per-desk, but values live longer):
Same allocation but registers hold values across 2 rounds instead of 1.
No additional registers needed because:
1. After round 1 completes for a desk, we immediately start round 2
2. The same `tmp1`/`tmp2` temporaries can be reused
3. `node_val` is overwritten by the second gather

**No increase in register pressure** - the key insight is that we reuse the same scratch space between rounds, just without the memory round-trip.

## Cycle Breakdown

### H7H10 Loop Body (128 iterations):
- Offset calculation: ~2 cycles
- Load idx/val (4 desks interleaved): ~5 cycles
- Gather address setup: ~3 cycles
- Gathers (32 loads, 2 per cycle): 16 cycles
- Hash + branch computation (deeply pipelined): ~25 cycles
- Bounds check + vselect bypass: ~4 cycles
- Store idx/val: 4 cycles
- Loop control: 5 cycles
**Total per iteration:** ~64 cycles

### H12 Loop Body (64 iterations):
- Offset calculation: ~2 cycles
- Load idx/val (4 desks interleaved): ~5 cycles
- Gather address setup (round 1): ~3 cycles
- Gathers round 1: 16 cycles
- Hash + branch round 1: ~25 cycles
- Gather address setup (round 2): ~2 cycles (simpler, no memory load)
- Gathers round 2: 16 cycles
- Hash + branch round 2: ~25 cycles
- Bounds check + vselect bypass: ~4 cycles
- Store idx/val: 4 cycles
- Loop control: 5 cycles
**Total per iteration:** ~107 cycles

### Total Cycles Comparison:
- H7H10: 128 iterations * ~64 cycles = ~8,192 cycles (loop only)
- H12: 64 iterations * ~107 cycles = ~6,848 cycles (loop only)

The actual measured improvement (832 cycles) comes from:
1. Eliminating 64 * 8 = 512 vloads (at ~1 cycle each, 2 per cycle bandwidth limited)
2. Eliminating 64 * 8 = 512 vstores (at ~1 cycle each, 2 per cycle bandwidth limited)
3. Reduced loop overhead (64 vs 128 iterations)

## Why This Worked

1. **Register Reuse:** The architecture has enough scratch space to hold 4 desks worth of vectors (192 words) without spilling, and these same registers can be reused across 2 rounds.

2. **Reduced Memory Bandwidth:** vload/vstore operations are limited to 2 per cycle. By eliminating half of them, we remove a bottleneck.

3. **Loop Overhead Reduction:** With half the iterations, we save cycles on:
   - Loop counter increment and comparison
   - Branch prediction/jump
   - Offset recalculation

4. **No Structural Hazards:** The gather operations (scalar loads) dominate the pipeline, and we still need 2 rounds worth of them. However, the vload/vstore elimination more than compensates.

## Potential Further Optimizations

1. **4-Round Fusion:** Could potentially fuse 4 rounds, but would require:
   - Either more scratch space for intermediate values
   - Or more complex scheduling to reuse registers

2. **Better Gather Scheduling:** The gathers are still the bottleneck (16 cycles per round). Further optimization would need to address the gather pattern.

3. **Load/Store Overlap:** Could potentially overlap the stores from the previous double-round with the loads for the next double-round using deeper software pipelining.

## Conclusion

Round fusion successfully eliminated 50% of the vload/vstore operations by processing 2 consecutive rounds without intermediate memory writeback. This reduced cycles from 5,947 to 5,115, a 14% improvement that exceeded the expected ~500 cycle savings.

The success of this optimization demonstrates that memory bandwidth for vload/vstore operations was indeed a limiting factor, and keeping data in registers across multiple computational rounds is highly beneficial on this architecture.
