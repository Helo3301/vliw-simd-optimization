# H142: GROUP_SIZE Sweep Results

## Summary

Testing different GROUP_SIZE values for the interleaved round processing optimization.

**Goal:** Find if GROUP_SIZE=4 is actually optimal, or if another value is better.

## Results

| GROUP_SIZE | Cycles | vs H140 (1645) | Correct |
|------------|--------|----------------|---------|
| 2 | 1648 | +3 (worse) | yes |
| 3 | 1653 | +8 (worse) | yes |
| 4 | 1645 | baseline | yes |
| 5 | 1755 | +110 (worse) | yes |
| 6 | 1644 | **-1 (better)** | yes |
| 8 | 1728 | +83 (worse) | yes |

## Analysis

**Winner: GROUP_SIZE=6 with 1,644 cycles (1 cycle improvement)**

The results show that GROUP_SIZE affects performance in a non-linear way:

1. **GROUP_SIZE=6**: Best at 1,644 cycles - saves 1 cycle over baseline
2. **GROUP_SIZE=4**: Second best at 1,645 cycles (current baseline)
3. **GROUP_SIZE=2**: Third best at 1,648 cycles (+3 cycles)
4. **GROUP_SIZE=3**: Fourth at 1,653 cycles (+8 cycles)
5. **GROUP_SIZE=8**: Fifth at 1,728 cycles (+83 cycles)
6. **GROUP_SIZE=5**: Worst at 1,755 cycles (+110 cycles)

The pattern suggests:
- Divisors of 16 (NUM_DESKS) work better: 2, 4, 8 are divisors; 6 is close
- GROUP_SIZE=5 causes uneven groups (3 groups of 5 + 1 group of 1), explaining poor performance
- GROUP_SIZE=8 splits into only 2 groups, which may reduce instruction-level parallelism

## Conclusion

GROUP_SIZE=6 achieves 1,644 cycles - a marginal 1-cycle improvement over the baseline.

**Note:** The improvement is minimal (0.06%). The target is 1,579 cycles, so we still need to save 65 more cycles beyond this finding.

## Files

- `test_group_2.py` - GROUP_SIZE=2 variant
- `test_group_3.py` - GROUP_SIZE=3 variant
- `test_group_5.py` - GROUP_SIZE=5 variant
- `test_group_6.py` - GROUP_SIZE=6 variant (best)
- `test_group_8.py` - GROUP_SIZE=8 variant
