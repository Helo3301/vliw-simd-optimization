# H144 Iteration Order Experiments - Results

## Objective
Test different orderings of desks/rounds to see if the scheduler can find better packing and improve cycle count.

## Baseline
- **H140**: 1,645 cycles (groups of 4 desks, each group through all 16 rounds, sequential order)
- **Target**: 1,579 cycles (need to save 66 cycles)

## Experiments and Results

| Experiment | Description | Cycles | Delta vs H140 | Result |
|------------|-------------|--------|---------------|--------|
| H144-A | Reverse group order (groups 3,2,1,0 instead of 0,1,2,3) | **1,636** | -9 | IMPROVED |
| H144-B | Interleaved tiles (tile0-group0, tile1-group0, tile0-group1...) | 1,739 | +94 | Worse (extra load/store overhead) |
| H144-C | Odd/even desk interleaving (0,2,4,6 then 1,3,5,7) | 1,645 | 0 | Same |
| H144-D | Smaller group size (2 instead of 4) | 1,648 | +3 | Slightly worse |
| H144-E | Larger group size (8 instead of 4) | 1,728 | +83 | Much worse |
| H144-F | Reverse desk order within groups (3,2,1,0 in each group) | **1,644** | -1 | Slightly improved |
| H144-G | Combined: reverse groups AND reverse desks within groups | **1,635** | -10 | BEST RESULT |

## Best Result: H144-G (Combined Reverse)

**File:** `perf_h144_combined_reverse.py`

**Approach:**
- Reverse the order of groups: process groups [12-15], [8-11], [4-7], [0-3] instead of [0-3], [4-7], [8-11], [12-15]
- Within each group, reverse desk order: process desks [15,14,13,12], [11,10,9,8], etc.

**Result:** 1,635 cycles (10 cycles saved vs H140 baseline)

## Analysis

### What worked:
1. **Reverse group order (-9 cycles)**: Processing later groups first allows the scheduler to find better overlapping patterns between operations. This likely helps because dependencies at the start of a group can overlap with the tail of the previous group.

2. **Reverse desk order within groups (-1 cycle)**: Minor improvement, but shows that even small ordering changes can affect scheduling.

3. **Combination (-10 cycles)**: The improvements are roughly additive, suggesting these are independent optimizations in how operations get scheduled.

### What didn't work:
1. **Interleaved tiles (+94 cycles)**: The additional load/store operations needed to switch between tiles within the same group dramatically hurt performance.

2. **Odd/even interleaving (0 cycles)**: No benefit - the scheduler handles this ordering equivalently.

3. **Different group sizes**:
   - Size 2 (+3 cycles): Too small, not enough work to overlap within group
   - Size 8 (+83 cycles): Too large, creates long dependency chains that hurt scheduling

## Gap to Target

- **Current best (H144-G):** 1,635 cycles
- **Target:** 1,579 cycles
- **Remaining gap:** 56 cycles

The iteration order optimizations provided a modest 10-cycle improvement, but are not sufficient to reach the target. Further algorithmic or structural changes would be needed to achieve the 1,579 cycle target.

## Recommendations

1. **Use H144-G's ordering** (reverse groups + reverse desks) as the new baseline for future experiments.

2. **Focus on other optimization vectors** since iteration order changes alone cannot bridge the 56-cycle gap:
   - Reduce operation count (algorithmic improvements)
   - Better register allocation
   - Improved memory access patterns
   - Pipeline optimization

3. **The greedy scheduler is order-sensitive** - future work should consider this when designing operation sequences.
