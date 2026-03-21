# Experiment H15: Aggressive Loop Unrolling (2x)

## Summary

| Metric | Value |
|--------|-------|
| Baseline (H7H10) | 5,947 cycles |
| H15 Result | 5,758 cycles |
| Cycles Saved | 189 cycles |
| Improvement | 3.2% |

## Approach

Unrolled the main loop 2x by processing two batches of 4 desks per iteration instead of one.

### Before (H7H10):
- 128 total iterations
- Each iteration: process desks at offsets 0, 8, 16, 24
- Loop overhead per iteration: ~5 cycles
  - ALU: batch_offset += 32, iter_counter += 1 (1 cycle)
  - ALU: compare batch_offset < batch_size (1 cycle)
  - FLOW: select for batch wrap (1 cycle)
  - ALU: compare iter_counter < total (1 cycle)
  - FLOW: cond_jump (1 cycle)

### After (H15):
- 64 total iterations
- Each iteration:
  - First batch: desks at offsets 0, 8, 16, 24
  - Second batch: desks at offsets 32, 40, 48, 56
- Loop overhead only once per 2 batches

## Analysis

### Expected vs Actual Savings

**Expected:** 320 cycles (64 eliminated iterations x 5 cycles/iteration)

**Actual:** 189 cycles

**Discrepancy Analysis:**

1. **Loop overhead is likely less than 5 cycles:**
   - Some ALU operations may be packed together (6 ops per cycle limit)
   - The original loop control uses packed ALU in one instruction

2. **Added overhead from 2x unrolling:**
   - Additional offset calculation for second batch (+1 instruction)
   - Additional constants needed (40, 48, 56, 64) (+4 load const operations in init)

3. **Estimated actual loop overhead:**
   - Savings = 189 cycles / 64 eliminated iterations = ~3 cycles per iteration
   - This suggests the original loop overhead was ~3 cycles, not 5

### Code Size Impact

| Metric | H7H10 | H15 |
|--------|-------|-----|
| Main loop body (instructions) | ~47 | ~96 (2x body + overhead) |
| Additional constants | 0 | 4 (40, 48, 56, 64) |
| Additional offset registers | 0 | 4 (off_b2_0 through off_b2_3) |

The code size approximately doubled for the main loop body, as expected from 2x unrolling.

### Cycles Saved Per Eliminated Iteration

```
Cycles saved: 189
Eliminated iterations: 64
Per-iteration savings: 189 / 64 = 2.95 cycles
```

This suggests the actual loop overhead in H7H10 was approximately **3 cycles per iteration**, consisting of:
- 1 cycle: ALU operations (batch_offset += 32, iter_counter += 1, batch comparison) - packed
- 1 cycle: FLOW select for batch wrap
- 1 cycle: ALU comparison + FLOW cond_jump

## Conclusion

Loop unrolling 2x provides a modest 3.2% improvement (189 cycles). The savings are less than the theoretical maximum because:

1. H7H10 already has well-packed loop control (3 cycles, not 5)
2. The overhead of computing second batch offsets partially offsets the gains

### Verdict: SUCCESS

While the improvement is smaller than expected, 189 cycles is still meaningful:
- **H15 achieves 5,758 cycles** (down from 5,947)
- **Speedup: 24.84x to 25.66x** over the original baseline (147,734 cycles)

### Further Optimization Ideas

1. **4x unrolling:** Would save ~94 more cycles (half of current overhead)
2. **Eliminating batch wrap check:** If batch_size is always 256 and we process 64 elements per unrolled iteration, the batch wrap happens every 4 iterations - could be unrolled entirely
3. **Interleaving between batches:** Instead of sequential processing, interleave computations from both batches to better utilize functional units
