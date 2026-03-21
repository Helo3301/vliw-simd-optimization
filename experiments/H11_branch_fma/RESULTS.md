# Experiment H11: Branch Computation FMA Optimization

## Summary
**Result: SUCCESS** - 5,819 cycles (down from 5,947 in H7H10, a savings of 128 cycles)

## What Changed

The branch computation in the kernel computes the next index as:
```python
idx = 2 * idx + (1 if val % 2 == 0 else 2)
# Which is equivalent to:
idx = 2 * idx + 1 + (val & 1)
```

In H7H10, this was implemented as 4 operations per desk:
1. `tmp1 = val & 1` (AND - extract branch bit)
2. `idx = idx * 2` (MUL - double the index)
3. `idx = idx + 1` (ADD - add 1 in a separate cycle)
4. `idx = idx + tmp1` (ADD - add branch bit)

**H11 Optimization:** Replace steps 2 and 3 with a single FMA (fused multiply-add):
1. `tmp1 = val & 1` (AND - extract branch bit)
2. `idx = multiply_add(idx, 2, 1)` (FMA - compute `idx*2 + 1` in one operation)
3. `idx = idx + tmp1` (ADD - add branch bit)

This saves 1 VALU operation per desk per iteration.

## Cycle Breakdown

### H7H10 (Before) - 5,947 cycles
- Main loop iterations: 128 (= 256/8 * 16 / 4 desks)
- Branch computation per desk: MUL + ADD(+1) + ADD(+tmp1) = 3 VALU operations
- Desk3's `+1` operation was in its own dedicated cycle

### H11 (After) - 5,819 cycles
- Main loop iterations: 128
- Branch computation per desk: FMA + ADD(+tmp1) = 2 VALU operations
- Desk3's `+1` cycle is ELIMINATED (folded into FMA)

### Analysis of Savings
- **Per-iteration savings:** 1 cycle (desk3's standalone `+1` cycle eliminated)
- **Total iterations:** 128
- **Expected savings:** 128 cycles
- **Actual savings:** 5,947 - 5,819 = 128 cycles

The savings match exactly because:
- Desks 0, 1, 2: Their `+1` operations shared cycles with other hash operations, so removing them only reduced VALU pressure but didn't eliminate cycles
- Desk 3: Its `+1` operation was in a standalone cycle, so removing it saves exactly 1 cycle per iteration

## Why It Worked

1. **FMA is a single VALU operation** that computes `a*b + c` in one cycle
2. **The branch formula `idx*2 + 1`** maps perfectly to FMA with a=idx, b=2, c=1
3. **Desk3's standalone cycle** was the key: since it only contained the `+1` operation, eliminating it saved exactly 1 cycle per loop iteration
4. **VLIW semantics** allowed us to move the branch bit addition (`+tmp1`) to fill the freed VALU slots in other cycles

## Code Changes

For each desk (0, 1, 2, 3):
- Changed: `("*", desk['idx'], desk['idx'], v_two)` to `("multiply_add", desk['idx'], desk['idx'], v_two, v_one)`
- Removed: `("+", desk['idx'], desk['idx'], v_one)` from its separate cycle
- Moved: `("+", desk['idx'], desk['idx'], desk['tmp1'])` earlier to fill the slot where `+1` was

## Performance Improvement

| Metric | H7H10 | H11 | Improvement |
|--------|-------|-----|-------------|
| Cycles | 5,947 | 5,819 | 128 cycles (2.15%) |
| Speedup vs Baseline | 24.84x | 25.39x | +0.55x |

## Conclusion

The FMA optimization successfully reduced cycles by eliminating the standalone `+1` cycle for desk3. While the VALU operation count reduction was 512 ops total (4 desks * 1 op * 128 iterations), the actual cycle savings was 128 because most cycles weren't bottlenecked solely on VALU operations - the main win came from eliminating desk3's dedicated `+1` cycle.
