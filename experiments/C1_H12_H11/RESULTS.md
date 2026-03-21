# C1: Combined H12 (Round Fusion) + H11 (Branch FMA) Results

## Performance Summary

| Metric | Value |
|--------|-------|
| **Cycle Count** | 4,987 |
| **H12 Baseline** | 5,115 cycles |
| **Improvement over H12** | 128 cycles (2.5%) |
| **Improvement over H11** | 832 cycles (14.3%) |
| **Speedup over original baseline (147,734)** | 29.6x |

## Techniques Combined

### H12: Round Fusion (Base)
- Process 2 consecutive rounds without intermediate memory writeback
- Keeps tree indices in registers between rounds
- Eliminates 50% of vload/vstore pairs per desk
- Original performance: 5,115 cycles

### H11: Branch FMA Optimization (Added)
- Uses `multiply_add(idx, idx, v_two, v_one)` for branch computation
- Replaces 2 separate operations (MUL + ADD) with 1 FMA instruction
- Original performance: 5,819 cycles (single-round iteration)

## What Worked

1. **Additive optimization benefits**: The two optimizations target different aspects of the pipeline:
   - H12 reduces memory traffic (load/store operations)
   - H11 reduces VALU operations in branch computation
   - These improvements stack together

2. **Branch computation simplification**: For each desk, the branch computation changed from:
   ```
   Before (H12):
     ("*", desk['idx'], desk['idx'], v_two)        # idx = idx * 2
     ("+", desk['idx'], desk['idx'], v_one)        # idx = idx + 1
     ("+", desk['idx'], desk['idx'], desk['tmp1']) # idx = idx + branch

   After (C1):
     ("multiply_add", desk['idx'], desk['idx'], v_two, v_one)  # idx = idx*2 + 1
     ("+", desk['idx'], desk['idx'], desk['tmp1'])             # idx = idx + branch
   ```

3. **Cycle savings analysis**:
   - H12 has 8 iterations (double-rounds) x 4 desks = 32 branch sequences per round-pair
   - Each round in the pair saves 1 VALU op per desk
   - 8 iterations x 2 rounds x 4 desks = 64 total FMA opportunities
   - Observed savings: ~128 cycles, consistent with expectations

## Notes

- The combination was straightforward since H12 and H11 optimize orthogonal aspects
- H12's round fusion structure was preserved as the base
- H11's FMA optimization was applied to all 8 branch computation sequences (4 desks x 2 rounds)
- The correctness test passes, confirming the optimization doesn't affect functional behavior
