# Experiment H10: Flow Unit vselect Bypass

## Summary

Successfully bypassed the flow unit bottleneck for vselect operations by replacing them with VALU multiplication operations.

## Results

| Metric | Value |
|--------|-------|
| Baseline (T6+T5) | 7,995 cycles |
| H10 Result | 7,611 cycles |
| Cycles Saved | 384 cycles |
| Improvement | 4.8% |
| Speedup over original baseline (147,734) | 19.41x |

## The Optimization

### Problem
The flow unit only allows 1 operation per cycle. The original code used 4 `vselect` operations (one per desk), requiring 4 cycles:

```python
# Before: 4 cycles (flow limited to 1/cycle)
for d in range(NUM_DESKS):
    self.add("flow", ("vselect", desks[d]['idx'], desks[d]['tmp1'], desks[d]['idx'], v_zero))
```

### Solution
Replaced `vselect` with VALU multiplication, allowing all 4 operations in a single cycle:

```python
# After: 1 cycle (VALU can do 6 ops/cycle)
self.instrs.append({
    "valu": [
        ("*", desks[0]['idx'], desks[0]['idx'], desks[0]['tmp1']),
        ("*", desks[1]['idx'], desks[1]['idx'], desks[1]['tmp1']),
        ("*", desks[2]['idx'], desks[2]['idx'], desks[2]['tmp1']),
        ("*", desks[3]['idx'], desks[3]['idx'], desks[3]['tmp1']),
    ],
})
```

### Why This Works

The `vselect` operation was:
```
vselect(dest, cond, idx, v_zero)  
// If cond != 0: dest = idx
// If cond == 0: dest = v_zero (which is 0)
```

The condition comes from `cond = (idx < n_nodes)`, which produces either 0 or 1.

This can be expressed as simple multiplication:
```
dest = idx * cond
// When cond = 1 (idx < n_nodes): idx * 1 = idx
// When cond = 0 (idx >= n_nodes): idx * 0 = 0
```

### Cycle Savings Calculation

- Iterations per loop: 128 (batch_size=256 / VLEN=8 / desks=4 * rounds=16)
- Savings per iteration: 3 cycles (4 cycles -> 1 cycle)
- Total expected savings: 384 cycles
- Actual savings: 384 cycles (7,995 - 7,611 = 384)

The actual savings exactly match the expected savings.

## Correctness

The correctness check passes:
```
$ python3.11 experiments/H10_vselect_bypass/perf_takehome_h10.py --check
Correctness check PASSED! Cycles: 7611
```

## Key Insight

The general pattern `vselect(dest, cond, a, b)` can sometimes be replaced with VALU operations when:
1. The condition produces 0 or 1 (not arbitrary non-zero values)
2. One of the branches is 0

Formula: `dest = a * cond + b * (1 - cond)`

When b = 0: `dest = a * cond`

This optimization applies whenever you need conditional selection with a zero fallback and a 0/1 condition.

## Files

- `perf_takehome_h10.py`: Optimized kernel with vselect bypass
- `RESULTS.md`: This file
