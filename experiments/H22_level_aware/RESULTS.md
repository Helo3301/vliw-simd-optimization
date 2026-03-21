# H22: Level-Aware Gathering - Results

## Summary

**Final Cycle Count:** 4,622 cycles
**Base (C4):** 4,667 cycles
**Improvement:** 45 cycles (0.96% reduction)
**Speedup over original baseline (147,734):** 31.96x

## Hypothesis

All batch elements are at the SAME tree level during each round:
- Round 0: ALL 256 elements access index 0 (1 unique address)
- Round 1: ALL elements access indices 1 or 2 (2 unique addresses)
- Round 2: ALL elements access indices 3-6 (4 unique addresses)
- Round 3: ALL elements access indices 7-14 (8 unique addresses = VLEN)

Tree levels are stored contiguously in memory:
- Level L starts at index (2^L - 1)
- Level L has 2^L nodes stored contiguously

The optimization: For early rounds where the level has <= 8 nodes, pre-load all tree values for that level and use selection operations instead of scattered gathers.

## Implementation

The implementation optimizes the first double-round (rounds 0-1):

**Round 0 (Level 0, 1 value):**
- Pre-load tree[0] during initialization
- Broadcast to all VLEN elements
- Replace 256 scattered scalar loads with 1 load + broadcast
- Theoretical savings: ~120 cycles (256 loads -> 1 load)

**Round 1 (Level 1, 2 values):**
- Pre-load tree[1] and tree[2] during initialization
- Use vselect to choose based on idx value (1 or 2)
- Replace 256 scattered scalar loads with 2 loads + vselect operations
- Theoretical savings: ~100 cycles

**Remaining rounds (2-15):**
- Use standard C4 approach with full gather operations
- These rounds access levels with 4+ nodes per batch element

## Actual Savings Analysis

The actual savings of 45 cycles is lower than the theoretical maximum for several reasons:

1. **Sequential vselect operations:** The flow engine can only do 1 vselect per cycle, requiring 4 cycles for all 4 desks in round 1.

2. **Data dependency overhead:** Had to split AND and ADD operations into separate cycles due to VLIW read-before-write semantics. The original code had:
   ```python
   ("&", tmp1, val, v_one),
   ("+", idx, v_one, tmp1),  # reads tmp1 BEFORE AND writes to it!
   ```
   This required fixing to ensure correct operation.

3. **Less aggressive pipelining:** The level-aware code for rounds 0-1 is written more straightforwardly, without the aggressive interleaving of C4's gather+hash pipeline.

4. **Initialization overhead:** Pre-loading tree values for levels 0 and 1 adds a few cycles to the setup phase.

## Cycle Breakdown

| Phase | Cycles |
|-------|--------|
| Initialization (including level preloads) | ~59 |
| First double-round (rounds 0-1) x 8 iterations | ~64 per iter = 512 |
| Remaining 7 double-rounds (C4 style) | ~581 per double-round = 4,067 |
| **Total** | **4,622** |

## Conclusions

1. **Hypothesis validated:** Level-aware gathering works and produces correct results.

2. **Marginal improvement:** The 0.96% improvement is modest because:
   - The gather operations are already pipelined well in C4
   - vselect operations still require multiple cycles
   - Only 2 of 16 rounds benefit from this optimization

3. **Potential improvements:**
   - Extend to rounds 2-3 (levels 2-3) for additional savings
   - More aggressive pipelining of the level-aware code
   - Explore scratch-based gather for level 3 (8 values = VLEN)

4. **Trade-offs:**
   - Code complexity increased significantly
   - Maintenance burden higher with specialized code paths
   - Benefits diminish for deeper tree levels

## Files

- `perf_takehome_h22.py` - Main implementation
- `RESULTS.md` - This file
