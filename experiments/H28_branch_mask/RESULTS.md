# H28: Pre-computed Branch Masks - Results

## Hypothesis

The current implementation computes branch direction (`val & 1`) after each hash computation, creating a dependency chain. We hypothesized that pre-computing branch information earlier in the pipeline could reduce the critical path.

## Implementation

The H28 implementation pre-computes the `idx*2+1` base value during hash stage 4 (before hash stage 5 completes), storing it in a separate register (`idx_base`). After the hash completes:
1. Extract branch bit: `tmp1 = val & 1`
2. Compute final index: `idx = idx_base + tmp1`

This is compared to the C4 baseline approach:
1. Same cycle: `tmp1 = val & 1` AND `idx = idx * 2 + 1` (parallel operations)
2. Next cycle: `idx = idx + tmp1`

## Results

| Metric | C4 Baseline | H28 |
|--------|-------------|-----|
| Cycles | 4,667 | 4,667 |
| Speedup over initial baseline (147,734) | 31.66x | 31.66x |

## Analysis

**No performance improvement observed.**

The H28 optimization provides the same cycle count as C4 (4,667 cycles). The pre-computation of `idx*2+1` one cycle earlier does not reduce the critical path because:

1. **Already parallelized in C4**: The C4 baseline already computes `idx*2+1` and `val & 1` in the same cycle using different VALU slots. Both operations are independent and can run in parallel.

2. **Sequential dependency remains**: Regardless of when `idx*2+1` is computed, we still need:
   - One cycle to extract the branch bit after the hash completes
   - One cycle to add the branch bit to the base index

3. **Work redistribution, not reduction**: Moving the `idx*2+1` FMA operation from cycle N to cycle N-1 simply shifts 1 VALU operation between cycles without reducing the total cycle count.

4. **VALU slot utilization**: In cycles where H28 pre-computes `idx_base`, the C4 baseline uses those same slots for the same operations (just in the next cycle). The total VALU work is identical.

## Key Insights

The branch computation in this kernel is already well-optimized in C4:
- The hash function is the true bottleneck (6 stages with dependencies)
- Branch direction extraction (`val & 1`) must wait for the final hash result
- The `idx*2+1` computation can be parallelized with the AND operation

To further reduce the critical path, optimizations would need to:
- Reduce the number of hash stages (not possible without changing the algorithm)
- Find a way to determine the final LSB earlier (complex due to carry propagation in hash stages)
- Further increase parallelism between desks or reduce inter-iteration dependencies

## Conclusion

**Hypothesis H28 is REJECTED.** Pre-computing the branch base (`idx*2+1`) earlier does not provide performance benefits because the C4 baseline already exploits the available parallelism by computing it in the same cycle as the branch bit extraction.

## Test Output

```
forest_height=10, rounds=16, batch_size=256
CYCLES:  4667
Speedup over baseline:  31.655024641097064
Correctness check PASSED! Cycles: 4667
```
