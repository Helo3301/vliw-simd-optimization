# Experiment H14: Parallel Address Pipeline (ALU/VALU Overlap)

## Summary

This experiment explores using ALU to precompute addresses while VALU is busy with hash computation, achieving better utilization of the parallel execution units.

## Results

| Version | Cycles | Improvement |
|---------|--------|-------------|
| H7H10 Baseline | 5,947 | - |
| H14 Final | 5,307 | 640 cycles (10.8%) |

## Key Optimizations

### 1. Store/Loop Control Overlap (384 cycles saved)

The original code had separate cycles for:
- Store desk0
- Store desk1
- Store desk2
- Store desk3
- batch_offset += 32, iter_counter += 1
- batch_offset < batch_size comparison
- select (wrap-around)
- iter_counter < total comparison
- cond_jump

We merged ALU operations with store operations:
- Store desk0 + ALU (batch_offset update, iter_counter update)
- Store desk1 + ALU (batch_offset < batch_size comparison)
- Store desk2 + ALU (iter_counter < total comparison)
- Store desk3 + Flow (select wrap-around)
- cond_jump

This saves 3 cycles per iteration = 384 cycles total.

### 2. Iter Counter Comparison Overlap (128 cycles saved)

Originally, the iter_counter comparison happened after all stores. We moved it to overlap with Store desk2 by using a separate scratch register (`addr_scalar`) to avoid conflicting with the batch_offset comparison result.

This saves 1 cycle per iteration = 128 cycles total.

### 3. Desk 2,3 Address Add Merge with Gather (128 cycles saved)

Originally, computing desk 2,3 gather addresses was a separate VALU-only cycle. We merged it with the first desk0 gather cycle:

Before:
```
Cycle N: VALU: desk2,3 addr add
Cycle N+1: Load: desk0 gather lanes 0-1
```

After:
```
Cycle N: Load: desk0 gather lanes 0-1 | VALU: desk2,3 addr add
```

This saves 1 cycle per iteration = 128 cycles total.

## ALU Utilization Analysis

### Before H14 (H7H10)
- Hash computation phases: ALU completely idle
- Load phases: ALU mostly idle except for address computation
- Store phases: ALU idle

### After H14
- Hash computation phases: ALU computes next iteration's addresses (precomputation)
- Load phases: ALU computes current iteration's addresses
- Store phases: ALU performs loop control operations

### Precomputation (Not Fully Utilized)

We precompute next iteration's addresses during the hash phase:
- `next_batch_offset = batch_offset + 32`
- `next_offset_regs[0..3]`
- `next_addr_tmp[0..7]`

However, these precomputed values are NOT fully utilized because:
1. The wrap-around logic (`batch_offset = batch_offset >= batch_size ? 0 : batch_offset`) makes precomputed values invalid when wrap occurs
2. Double-buffering would require additional complexity

The precomputation code is present but currently serves to fill ALU slots during VALU-heavy phases without providing direct cycle savings.

## What Did NOT Work

### Using Precomputed Addresses
- Challenge: Wrap-around invalidates precomputed addresses 1 out of every 8 iterations
- Solution attempted: None - the complexity of handling the special case would likely negate any savings

### Merging First ALU Cycle
- Challenge: First loop cycle computes addr_tmp[0,1] which are needed immediately for vloads
- Data dependency prevents merging with subsequent load operations

### Moving More Work to Load-Only Gather Cycles
- The desk0 gather has 3 load-only cycles (lanes 2-7)
- No useful ALU/VALU work can be moved here because all required computations either:
  - Depend on the gather results (hash operations)
  - Have already been computed (addresses)
  - Are already being computed in parallel (next iteration precomputation happens during hash)

## Architecture Insights

- **ALU**: 12 scalar slots per cycle - heavily underutilized during hash computation
- **VALU**: 6 vector slots per cycle - fully utilized during hash computation
- **Load**: 2 slots per cycle - often a bottleneck during gathers
- **Store**: 2 slots per cycle
- **Flow**: 1 slot per cycle

The key insight is that Store cycles don't use ALU, allowing loop control operations to be overlapped with stores.

## Files

- `perf_takehome_h14.py`: Implementation with all optimizations
- `RESULTS.md`: This file

## Correctness

All tests pass:
```
python3.11 experiments/H14_addr_pipeline/perf_takehome_h14.py --check
Correctness check PASSED! Cycles: 5307
```

## Conclusion

H14 achieves a 10.8% improvement over H7H10 (5,947 -> 5,307 cycles) by:
1. Overlapping ALU loop control with Store operations
2. Merging VALU operations with Load operations

The theoretical goal of using ALU for address precomputation during VALU-heavy hash phases is partially achieved but provides limited direct cycle savings due to data dependencies and wrap-around handling complexity.

The main wins come from better packing of existing operations into cycles where different execution units have slack capacity.
