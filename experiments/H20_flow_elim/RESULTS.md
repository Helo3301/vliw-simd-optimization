# H20: Flow Unit Complete Elimination - Results

## Summary

| Metric | Value |
|--------|-------|
| Baseline (C4) | 4,667 cycles |
| H20 Result | 4,667 cycles |
| Improvement | 0 cycles (0%) |
| Correctness | PASSED |

## Hypothesis

The Flow unit is limited to 1 op/cycle, creating a serialization bottleneck. By replacing Flow operations (specifically `select`) with arithmetic equivalents using ALU, we can avoid this bottleneck.

## Flow Operations in C4

| Operation | Location | Can Eliminate? | Action Taken |
|-----------|----------|----------------|--------------|
| `pause` | Initialization | No (required for sync) | Kept |
| `select` | Loop wrap-around | Yes | Replaced with ALU multiply |
| `cond_jump` | Loop control | No (required for branch) | Kept |
| `pause` | End | No (required for sync) | Kept |

## Optimization Applied

**Original (C4, line 981):**
```python
"flow": [
    ("select", batch_offset, tmp_scalar, batch_offset, zero_const),
],
```

**New (H20):**
```python
"alu": [
    ("*", batch_offset, batch_offset, tmp_scalar),
],
```

**Why this is equivalent:**
- `select(dest, cond, a, b)` = `a if cond else b`
- When `tmp_scalar = 1` (batch_offset < batch_size): keep batch_offset
- When `tmp_scalar = 0` (batch_offset >= batch_size): use 0
- `batch_offset * tmp_scalar` produces the same result:
  - `batch_offset * 1 = batch_offset`
  - `batch_offset * 0 = 0`

## Why No Cycle Improvement?

The optimization is **correct** but does not improve performance because:

1. **Parallel Execution**: In C4, the `select` operation was already executing in parallel with store operations in the same instruction bundle:
   ```python
   self.instrs.append({
       "store": [
           ("vstore", addr_tmp[6], desks[3]['idx']),
           ("vstore", addr_tmp[7], desks[3]['val']),
       ],
       "flow": [
           ("select", batch_offset, tmp_scalar, batch_offset, zero_const),
       ],
   })
   ```

2. **Store Unit Bottleneck**: The store unit (limited to 2 ops/cycle) was the bottleneck in that cycle, not the flow unit. The cycle completes when ALL engines finish their work.

3. **No Latency Reduction**: Moving `select` -> ALU `multiply` doesn't reduce the store latency, so the cycle count remains the same.

## Lessons Learned

1. **Flow unit bottleneck is limited**: The flow unit's 1 op/cycle limit only matters when:
   - Flow operations are on the critical path
   - Flow operations cannot be parallelized with slower engines

2. **C4 already optimized this**: The C4 implementation cleverly placed the `select` in a cycle dominated by stores, effectively hiding the flow unit's cost.

3. **ALU capacity is abundant**: Moving scalar operations to ALU (12 slots) is always safe capacity-wise, but only helps if it enables better parallelism or reduces critical path length.

## Conclusion

**Hypothesis REJECTED**: While the optimization is correct and removes a Flow unit dependency, it provides no performance benefit in C4's context because the Flow operation was already hidden behind store latency. The critical path was not affected.

The 4% gain seen in H10 (vselect -> VALU multiply) likely occurred because that `vselect` was on the critical path in its context, whereas this scalar `select` was not.
