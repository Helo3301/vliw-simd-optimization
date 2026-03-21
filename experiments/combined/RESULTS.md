# Experiment T6+T5: Combined FMA + Warp-Style VALU Packing

## Summary

This experiment combines the best optimizations from:
- **T5 V4**: 6-VALU slot packing across multiple desks (8,758 cycles)
- **T6**: FMA (multiply_add) for hash stages 0, 2, 4 (8,514 cycles)

## Results

| Configuration | Cycles | Improvement vs Baseline |
|---------------|--------|-------------------------|
| Baseline      | 9,793  | -                       |
| T5 V4         | 8,758  | 10.6%                   |
| T6            | 8,514  | 13.1%                   |
| **T6+T5**     | **7,995** | **18.4%**            |

**Final Result: 7,995 cycles (18.4% improvement over baseline)**

## Optimization Details

### Hash Stage Analysis

The hash function has 6 stages with different algebraic properties:

| Stage | Original Operation              | Optimization | Result |
|-------|--------------------------------|--------------|--------|
| 0     | val = (val + C0) + (val << 12) | FMA: val * 4097 + C0 | 1 cycle |
| 1     | val = (val ^ C1) ^ (val >> 13) | 6-slot packing | 3 cycles |
| 2     | val = (val + C2) + (val << 5)  | FMA: val * 33 + C2 | 1 cycle |
| 3     | val = (val ^ C3) ^ (val >> 6)  | 6-slot packing | 3 cycles |
| 4     | val = (val + C4) + (val << 3)  | FMA: val * 9 + C4 | 1 cycle |
| 5     | val = (val ^ C5) ^ (val >> 16) | 6-slot packing | 3 cycles |

### FMA Stages (0, 2, 4)

For stages with addition operations, we can use algebraic strength reduction:
- `val + C + (val << N)` = `val * (1 + 2^N) + C`

This allows a single `multiply_add(dest, val, multiplier, constant)` instruction
instead of 3 separate operations (op1, op3, op2).

With 4 desks, we can pack all 4 FMA operations into a single cycle:
```python
self.instrs.append({
    "valu": [
        ("multiply_add", desks[0]['val'], desks[0]['val'], v_fma_mult[hi], v_hash_consts[hi]),
        ("multiply_add", desks[1]['val'], desks[1]['val'], v_fma_mult[hi], v_hash_consts[hi]),
        ("multiply_add", desks[2]['val'], desks[2]['val'], v_fma_mult[hi], v_hash_consts[hi]),
        ("multiply_add", desks[3]['val'], desks[3]['val'], v_fma_mult[hi], v_hash_consts[hi]),
    ],
})
```

### Non-FMA Stages (1, 3, 5) - 6-VALU Packing

For XOR-based stages that cannot use FMA, we use T5's 6-slot packing:
- Cycle 1: prep (op1, op3) for desks 0, 1, 2 = 6 VALU ops
- Cycle 2: prep for desk 3 + combine (op2) for desks 0, 1, 2 = 5 VALU ops
- Cycle 3: combine for desk 3 = 1 VALU op

### Cycle Count Analysis

Per-iteration breakdown (4 desks, 32 elements):
- Load phase: ~23 cycles (same as T5 V4)
- Hash phase: ~15 cycles (reduced from 19 with FMA)
  - XOR all desks: 1 cycle
  - FMA stages (0, 2, 4): 3 cycles (1 each)
  - Non-FMA stages (1, 3, 5): 9 cycles (3 each)
  - Branch computation: 6 cycles
- Store phase: ~8 cycles
- Loop control: ~5 cycles

## Interaction Effects

The optimizations are partially additive:

| Metric | Expected | Actual | Notes |
|--------|----------|--------|-------|
| T5 V4 improvement | 10.6% | - | VALU packing |
| T6 improvement | 13.1% | - | FMA optimization |
| Combined (if additive) | ~21% | 18.4% | Some overlap exists |

The slight difference from fully additive gains is expected because:
1. Both optimizations target the same hash phase
2. T6's FMA replaces operations that T5 was packing
3. The improvements compound rather than add directly

## Correctness Verification

```
$ python3.11 experiments/combined/perf_takehome_t6t5.py --check
forest_height=10, rounds=16, batch_size=256
CYCLES:  7995
Speedup over baseline:  18.47829893683552
Correctness check PASSED! Cycles: 7995
```

## Conclusion

The combined T6+T5 optimization achieves **7,995 cycles**, an **18.4% improvement**
over the baseline of 9,793 cycles. This represents a significant speedup by:

1. Using FMA to reduce 3-cycle hash stages to 1-cycle operations (stages 0, 2, 4)
2. Using 6-VALU slot packing to maximize throughput for non-FMA stages (1, 3, 5)

The result falls within the expected range of 7,500-8,000 cycles and demonstrates
that the two optimizations can be effectively combined.
