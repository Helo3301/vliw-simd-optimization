# BF4: Branch Computation Optimization Results

**Agent:** Breadth-First Agent 4
**Baseline:** experiments/A1_r10_skip/perf_takehome_a1.py (1,548 cycles)
**Date:** 2026-01-25

## Summary

Tested 10 theories for optimizing branch computation (idx = 2*idx + 1 + bit).
**None of the theories yielded improvements over the baseline.**

## Test Results

| Theory | Description | Cycles | Delta | Status |
|--------|-------------|--------|-------|--------|
| 31 | Shift instead of FMA: (idx << 1) \| 1 \| bit | 1,599 | +51 | WORSE |
| 32 | Subtract: idx*2 + 2 - (1 - bit) | 1,597 | +49 | WORSE |
| 33 | Precompute both children, select | 1,549 | +1 | WORSE |
| 34 | Store 2*idx instead of idx | N/A | N/A | ANALYZED - WORSE |
| 35 | Store idx+1 (offset addressing) | N/A | N/A | ANALYZED - WORSE |
| 36 | Use vselect for branch | 1,549 | +1 | WORSE |
| 37 | Compute bit during hash stage | 1,548 | +0 | SAME |
| 38 | Batch 2 rounds at once | N/A | N/A | ANALYZED - NO BENEFIT |
| 39 | Lookup table for small idx | N/A | N/A | ANALYZED - NO BENEFIT |
| 40 | Skip branch in round 0 (idx=0 known) | 1,548 | +0 | SAME |

## Detailed Analysis

### Theory 31: Shift Instead of FMA
**Result: WORSE (+51 cycles)**

Attempted to replace:
```
FMA: idx = idx*2 + 1
ADD: idx = idx + bit
```
With:
```
SHIFT: idx = idx << 1
OR:    idx = idx | 1
ADD:   idx = idx + bit
```
**Why worse:** Adds an extra operation (4 ops vs 3 ops). VALU FMA is highly efficient on this architecture.

### Theory 32: Subtract Method
**Result: WORSE (+49 cycles)**

Used: `idx*2 + 2 - (1 - bit)` instead of `idx*2 + 1 + bit`

**Why worse:** Requires computing `1 - bit` as an extra operation, adding overhead.

### Theory 33 & 36: Precompute Both Children + Select
**Result: WORSE (+1 cycle)**

Computed both possible children:
```
left = 2*idx + 1
right = 2*idx + 2
idx = vselect(bit, right, left)
```

**Why worse:** 4 operations (FMA, ADD, vselect) vs 3 operations (AND, FMA, ADD). The vselect adds a flow slot dependency.

### Theory 34: Store 2*idx Between Rounds
**Result: ANALYZED - WORSE**

Would store `doubled_idx = 2*idx` to avoid multiplication in branch.

**Why worse:**
- Address calculation needs: `addr = forest_p + doubled_idx/2`
- Requires extra shift operation on every gather
- Net negative: adds more operations than it saves

### Theory 35: Store idx+1 (Offset Addressing)
**Result: ANALYZED - WORSE**

Would store `idx_plus_1 = idx + 1` to simplify branch math.

**Why worse:**
- Every address calculation becomes: `addr = forest_p + idx_plus_1 - 1`
- Subtraction required on every gather round
- Branch becomes: `idx_plus_1*2 - 1 + bit` (still 3 ops, plus address overhead)

### Theory 37: Compute Bit During Hash
**Result: SAME (0 cycles)**

Attempted to pipeline the AND operation with the final hash XOR.

**Why no improvement:** The scheduler already places these operations optimally. The AND depends on the final hash value, so true pipelining isn't possible.

### Theory 38: Batch 2 Rounds
**Result: ANALYZED - NO BENEFIT**

Attempted to compute branches for 2 rounds at once.

**Why no benefit:** Sequential dependency:
```
idx1 = branch(idx0, bit0)
idx2 = branch(idx1, bit1)
```
Cannot parallelize because idx2 depends on idx1.

### Theory 39: Lookup Table for Small idx
**Result: ANALYZED - NO BENEFIT**

Would use preloaded tables: `children[idx] = (left_child, right_child)`

**Why no benefit:**
- Already using FMA-based early round fusion for first 3 levels
- Vector lookup would require expensive scatter/gather
- No faster than current arithmetic approach for SIMD

### Theory 40: Skip Round 0 Branch
**Result: SAME (0 cycles)**

Round 0 always has idx=0, so branch result is always 1 or 2.

**Why no improvement:** The baseline already optimizes this case:
```
bit0 = val & 1
idx = 1 + bit0  // 2 ops instead of 3
```
This is already implemented in the fused round 0+1+2 code.

## Conclusions

1. **The current branch implementation is optimal** for this architecture:
   - 3 ops: AND (extract bit), FMA (2*idx+1), ADD (add bit)
   - Uses efficient FMA instruction
   - No better alternative found

2. **Shift-based alternatives are worse** because they require more operations.

3. **Select-based alternatives are worse** because vselect adds flow slot dependency.

4. **Storage transformations (34, 35) add overhead** elsewhere in the pipeline.

5. **Pipelining attempts (37) show the scheduler is already optimal**.

6. **Special-case optimizations (40) are already implemented** in the baseline.

## Recommendation

**No changes recommended.** The branch computation is at its theoretical minimum of 3 operations:
1. Extract bit (AND)
2. Compute base (FMA)
3. Add direction (ADD)

Further optimization would require architectural changes (e.g., a custom branch instruction).
