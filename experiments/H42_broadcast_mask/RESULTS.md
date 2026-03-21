# H42: Broadcast-and-Mask Experiment Results

## Thesis
For early rounds where elements are clustered, replace gather operations with broadcast+mask:
- Round 0: All elements at tree[0] → 1 load + broadcast vs 4-cycle gather
- Selection via ALU/VALU masking (18 slots/cycle) instead of flow engine (1 slot/cycle)

## Results

| Experiment | Cycles | vs Baseline | vs H38 | Notes |
|------------|--------|-------------|--------|-------|
| Baseline   | 147,734 | 1.0x | - | Reference |
| H38        | 4,062 | 36.4x | 1.0x | Previous best (8-desk + round fusion) |
| H42        | 5,752 | 25.7x | 0.71x | Broke pipeline - serialized operations |
| H42b       | 4,784 | 30.9x | 0.85x | Round 0 optimized, but no round fusion |
| **H42c**   | **3,992** | **37.0x** | **1.017x** | **NEW BEST!** Surgical broadcast integration |

## Key Findings

### 1. The broadcast idea works (correctness validated)
- Round 0 with broadcast instead of gather produces correct results
- All 256 elements do start at tree index 0 (confirmed)

### 2. Integration matters more than the idea itself
- H42: Broke pipeline → 41% slower
- H42b: Partial integration → 18% slower
- H42c: Surgical integration → **1.75% FASTER** (new best!)

### 3. The two-loop approach works
- Loop 1: Rounds 0-1 with broadcast for R0
- Loop 2: Rounds 2-15 with full H38 pipeline
- Clean separation avoids breaking round fusion

### 4. Actual savings vs expected
- Expected: ~120 cycles (30 cycles × 4 iterations)
- Actual: 70 cycles
- Difference: Round 0 hash is serialized (no gather to overlap with)

## H42c Architecture

```
PROLOGUE: Load tree[0], broadcast to v_root_node
    │
LOOP 1 (4 iterations, rounds 0-1):
    ├─ Round 0: BROADCAST (2 cycles) + hash
    └─ Round 1: Normal gather + hash interleaved
    │
LOOP 2 (28 iterations, rounds 2-15):
    └─ Exact H38 code with round fusion
```

## Lessons Learned

1. **Surgical changes beat structural rewrites** - H42c only modified the first loop
2. **Two-loop approach preserves optimization** - Each loop can be independently optimized
3. **The broadcast works** - 70 cycles saved with minimal code changes

## Conclusion

**Thesis: VALIDATED**

The broadcast concept works when properly integrated:
- ✓ Round 0 broadcast replaces 32-cycle gather with 2-cycle copy
- ✓ Two-loop structure preserves H38's optimizations for rounds 2-15
- ✓ **New best: 3,992 cycles (37.0x speedup)**

Next steps for further improvement:
- Apply broadcast to round 1 (2 unique nodes)
- Apply to rounds 2-3 (4 and 8 unique nodes)
- Potential additional savings: ~100-200 cycles
