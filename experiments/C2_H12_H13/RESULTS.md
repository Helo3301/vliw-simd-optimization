# C2: Combined H12 (Round Fusion) + H13 (Store Coalescing)

## Results

| Metric | Value |
|--------|-------|
| **Cycle Count** | 4,859 |
| **Improvement over H12** | 5.0% (256 cycles saved) |
| **Improvement over H13** | 10.6% (576 cycles saved) |
| **Speedup over Baseline (147,734)** | 30.4x |

## Baseline Comparisons

| Implementation | Cycles | Speedup vs Baseline |
|----------------|--------|---------------------|
| H12 (Round Fusion) | 5,115 | 28.9x |
| H13 (Store Coalescing) | 5,435 | 27.2x |
| **C2 (Combined)** | **4,859** | **30.4x** |

## Techniques Combined

### H12: Round Fusion
- Process 2 consecutive rounds without intermediate memory writeback
- Keeps tree indices in registers between rounds
- Eliminates 50% of vload/vstore pairs per desk
- Loop processes 2 rounds per iteration (8 double-rounds instead of 16 single-rounds)

### H13: Store Coalescing
- Overlaps stores with ALU operations (they use different execution engines)
- Store engine (2 slots) and ALU engine (12 slots) are independent
- Overlaps desk3 bounds check with store address calculations
- Overlaps stores with loop control operations (batch_offset update, iter_counter increment)

## What Worked

1. **Complementary Optimizations**: H12's round fusion reduces the total number of store operations (from 8 stores per single-round to 8 stores per double-round), while H13's store coalescing improves the efficiency of those remaining stores.

2. **Store-ALU Overlap**: By integrating H13's pattern of overlapping stores with:
   - Desk3's final bounds check and vselect bypass computation
   - Store address calculations
   - Loop control ALU operations (batch_offset, iter_counter updates)
   - The batch_offset comparison and select for wrap-around

3. **Preserved Register Locality**: H12's key benefit (keeping idx/val in registers across two rounds) is fully preserved, eliminating intermediate vload/vstore pairs.

## What Didn't Work (or wasn't attempted)

1. **No Double-Buffering Across Iterations**: We could potentially start loading the next iteration's data while storing the current iteration's results, but this would require additional scratch registers and more complex control flow.

2. **Limited Store Overlap with Round 2 Computation**: The stores must happen after round 2 completes, so there's limited opportunity to overlap stores with compute work (only the final desk3 bounds/vselect can overlap).

## Cycle Breakdown Analysis

With H12 alone (5,115 cycles) processing 64 double-round iterations:
- Each iteration: ~80 cycles average

With C2 (4,859 cycles):
- Each iteration: ~76 cycles average
- Savings: ~4 cycles per iteration from store coalescing overlap

The improvement comes from eliminating pure store cycles by overlapping them with:
- 6 ALU operations for store address calculation
- Desk3's final vselect bypass computation
- Loop control operations
