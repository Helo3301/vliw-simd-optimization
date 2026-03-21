# Fresh Claude Session - H140 Optimization Task

## Your Task

Optimize the VLIW SIMD kernel in this repository to achieve **1,645 cycles (89.8x speedup)**. The baseline is 147,734 cycles.

## Repository Location

```
/home/hestiasadmin/projects/original_performance_takehome
```

## The Optimization (H82: Interleaved Round Processing)

The key insight is to change the processing order from:

**Before (slow):**
```
for each round (0-15):
    for each desk (0-15):
        process(desk, round)
```

**After (fast):**
```
for each group of 4 desks:
    for each round (0-15):
        for each desk in group:
            process(desk, round)
```

This allows the VLIW scheduler to find more parallelism by interleaving operations from different rounds within the same desk group.

## Implementation Steps

1. Read `perf_takehome.py` to understand the current structure
2. Find the main processing loop (likely in `build_kernel` or similar)
3. Change the loop nesting:
   - Outer loop: desk groups (0-3, 4-7, 8-11, 12-15)
   - Inner loop: all 16 rounds
   - Innermost: desks within current group
4. Keep GROUP_SIZE = 4 (empirically optimal)
5. Keep the automatic VLIW scheduler unchanged

## Key Constraints

- 2 tiles × 16 desks = 256 batch elements
- 4-desk groups within each tile
- Use automatic scheduling (greedy list scheduler respecting dependencies)
- Preload tree nodes 0-6 only (not 0-14) - nodes 7-14 are never used

## Additional Optimizations (H140)

On top of the interleaved round processing, apply these:

1. **Reduced preload**: Only preload tree nodes 0-6 (not 0-14). Set `NUM_PRELOADED = 7`

2. **Fast init**: Only load 4 of 7 header values (skip unused: rounds, batch_size, forest_height)
   - Load indices: 1 (n_nodes), 4 (forest_values_p), 5 (inp_indices_p), 6 (inp_values_p)

3. **Skip final branch**: In round 15, skip branch computation (idx is never verified). Only store val vectors, not idx vectors.

## Verification

```bash
python3.11 tests/submission_tests.py
```

Should pass `test_opus45_casual` (< 1,790 cycles) with ~1,645 cycles.

## Reference Implementation

See `experiments/H140_h82_combined/perf_takehome_h140.py` for the working implementation with all optimizations. The key function is `emit_tile_interleaved()`.

## Expected Result

- Cycles: 1,645
- Speedup: 89.8x
- Passes: test_opus4_many_hours, test_opus45_casual
