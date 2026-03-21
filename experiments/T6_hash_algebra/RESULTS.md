# Experiment T6: Algebraic Strength Reduction on Hash

## Summary

**Status:** SUCCESS

| Metric | Value |
|--------|-------|
| Baseline cycles | 9,793 |
| New cycles | 8,514 |
| Cycles saved | 1,279 |
| Speedup | 1.15x (15% improvement) |
| Correctness | PASSED |

## Hypothesis

The 6-stage hash function has algebraic properties that allow optimization. Specifically, stages using only addition (not XOR) can be converted to a single FMA (fused multiply-add) operation.

## Analysis

### Hash Stage Structure

Each hash stage computes: `val = (val op1 const) op2 (val op3 shift)`

The 6 stages are:
```
Stage 0: val = (val + 0x7ED55D16) + (val << 12)  -> op2 = +
Stage 1: val = (val ^ 0xC761C23C) ^ (val >> 19)  -> op2 = ^
Stage 2: val = (val + 0x165667B1) + (val << 5)   -> op2 = +
Stage 3: val = (val + 0xD3A2646C) ^ (val << 9)   -> op2 = ^
Stage 4: val = (val + 0xFD7046C5) + (val << 3)   -> op2 = +
Stage 5: val = (val ^ 0xB55A4F09) ^ (val >> 16)  -> op2 = ^
```

### Algebraic Reduction

For stages 0, 2, and 4 (where op2 = '+'):
```
val = (val + C) + (val << N)
    = val + C + val * 2^N
    = val * (1 + 2^N) + C
```

This can be computed with a single `multiply_add` instruction!

| Stage | Original | Optimized |
|-------|----------|-----------|
| 0 | `(val + C0) + (val << 12)` | `val * 4097 + C0` |
| 2 | `(val + C2) + (val << 5)` | `val * 33 + C2` |
| 4 | `(val + C4) + (val << 3)` | `val * 9 + C4` |

### Why Stages 1, 3, 5 Cannot Be Optimized

XOR does not distribute over addition, so:
- `(val ^ C) ^ (val >> N)` cannot be algebraically simplified
- These stages still require 3 operations (2 parallel + 1 combine)

## Implementation

### Changes Made

1. Pre-computed FMA multipliers as vector constants:
   - `v_fma_mult_0` = 4097 (broadcast)
   - `v_fma_mult_2` = 33 (broadcast)
   - `v_fma_mult_4` = 9 (broadcast)

2. Replaced hash stages 0, 2, 4 with single `multiply_add` instruction:
   ```python
   # Original (3 cycles):
   self.instrs.append({"valu": [
       ("+", curr['tmp1'], curr['val'], v_hash_consts[0]),
       ("<<", curr['tmp2'], curr['val'], v_hash_shifts[0]),
   ]})
   self.add("valu", ("+", curr['val'], curr['tmp1'], curr['tmp2']))

   # Optimized (1 cycle):
   self.instrs.append({"valu": [
       ("multiply_add", curr['val'], curr['val'], v_fma_multipliers[0], v_hash_consts[0]),
   ]})
   ```

3. Adjusted instruction packing to account for the freed-up VALU slots.

### Cycles Saved Per Iteration

Each desk processes 6 hash stages. With FMA optimization:
- Stages 0, 2, 4: 1 cycle each (was 2-3 depending on packing)
- Stages 1, 3, 5: 2-3 cycles each (unchanged)

Per desk: ~3 cycles saved
Per 4-desk iteration: ~12 cycles saved
Total for 128 iterations: ~1,536 cycles theoretical

Actual savings: 1,279 cycles (some overlap was already achieved)

## What Worked

1. **FMA instruction exists and is efficient** - The simulator's `multiply_add` instruction computes `a * b + c` in a single cycle, enabling the optimization.

2. **Algebraic reduction is correct** - The transformation `(val + C) + (val << N) = val * (1 + 2^N) + C` produces bit-identical results.

3. **No additional register pressure** - We need 3 extra vector constants (the multipliers), but scratch space has ample room.

4. **Better than expected improvement** - Initial estimate was <5% improvement; actual was 15%.

## What Didn't Work / Limitations

1. **Cannot optimize XOR stages** - Stages 1, 3, 5 use XOR which doesn't have the same algebraic properties as addition. No FMA-style optimization exists for these.

2. **Some overlap reduction** - The original code packed the two parallel operations (add + shift) with loads from other operations. With FMA taking only one VALU slot, some of this parallelism is lost, reducing the net gain.

3. **Stage 3 cannot use FMA** - Even though stage 3 uses addition for op1 (`val + C`), its op2 is XOR, so it cannot benefit from FMA.

## Key Insights

1. **Strength reduction on hash functions** - Classic compiler optimization (replacing expensive operations with cheaper equivalents) applies to hash functions. The identity `x + x*2^N = x*(1+2^N)` converts 2 additions into 1 multiply-add.

2. **Simulator supports FMA natively** - This is key. Without hardware FMA support, this optimization would be meaningless.

3. **Diminishing returns on micro-optimizations** - This 15% speedup is significant but we're approaching theoretical limits. Further gains require architectural changes (more parallelism, better overlap scheduling).

4. **Unexpected magnitude of improvement** - The T6 specification predicted <5% gain, but actual was 15%. This suggests the baseline was not optimally packed, and the FMA freed up slots that enabled better overall scheduling.

## Files Modified

- Created: `experiments/T6_hash_algebra/perf_takehome_t6.py` (modified copy of original)
- Original `perf_takehome.py` unchanged

## Verification

```bash
cd /home/hestiasadmin/projects/original_performance_takehome
python3.11 experiments/T6_hash_algebra/perf_takehome_t6.py --check
# Output: Correctness check PASSED! Cycles: 8514
```
