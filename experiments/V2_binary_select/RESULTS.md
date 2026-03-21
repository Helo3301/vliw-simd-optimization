# V2: Hierarchical Binary Selection - Experiment Results

## Summary

**Cycle Count: 4,667 cycles** (matches C4 baseline)

- Speedup over original baseline (147,734): 31.66x
- Comparison to C4: 1.0x (identical performance)

## The Concept

The original goal was to replace gather operations with binary selection:

For a tree level with N values, use log2(N) binary selection stages:
```
Step 1: Select within pairs based on bit0 (N/2 selections)
Step 2: Select from pair results based on bit1 (N/4 selections)
Step 3: Continue until final selection
```

Selection formula: `result = a + (b-a) * cond` when cond is 0 or 1

## Analysis

### Why Binary Selection Doesn't Help

For 8 values (level 3 of the tree):
- 3 bit extractions (AND, shift, AND operations)
- Step 1: 4 selections = 12 VALU ops (sub, mul, add each)
- Step 2: 2 selections = 6 VALU ops
- Step 3: 1 selection = 3 VALU ops
- **Total: ~24 VALU ops = 4 cycles minimum** (at 6 VALU/cycle)

Compare to gather: **8 loads at 2 loads/cycle = 4 cycles**

The binary selection approach is NOT faster for small tree levels. The gather operation and binary selection have similar throughput.

### Tree Level Considerations

The tree has height 10 (2047 nodes total):
- Level 0 (root): 1 node, idx=0
- Level 1: 2 nodes, idx in [1,2]
- Level 2: 4 nodes, idx in [3,6]
- Level 3: 8 nodes, idx in [7,14]
- ...
- Level 10: 1024 nodes, idx in [1023,2046]

For level 10 with 1024 values, binary selection would need:
- 10 levels of selection
- ~60+ VALU operations
- vs 512 cycles for gather (1024 loads / 2 per cycle)

Binary selection could be advantageous for very large levels, but the overhead of preloading all 1024 values as broadcast vectors makes this impractical.

### Actual Implementation

The current V2 implementation does NOT use binary selection. Instead, it:
1. Matches the C4 structure exactly
2. Uses interleaved gather + hash pipeline
3. Employs round fusion (2 rounds per iteration)
4. Uses FMA for branch computation
5. Overlaps stores with loop control

## Key Learnings

1. **Gather is efficient**: The simulated gather (8 scalar loads) takes 4 cycles. Binary selection also takes ~4 cycles for 8 values. No advantage.

2. **Memory bandwidth vs compute**: The bottleneck is memory access patterns, not compute. Binary selection trades memory for compute, but the architecture already has sufficient compute capacity.

3. **Pipelining is key**: The real optimizations in C4 come from:
   - Overlapping gather with hash computation
   - Round fusion to avoid intermediate stores
   - Address pre-computation during VALU phases

4. **Binary selection might help if**:
   - There was memory latency (not modeled here)
   - The tree values could be cached in registers
   - The level being accessed was known at compile time

## Conclusion

The hierarchical binary selection approach is theoretically interesting but provides no performance advantage for this architecture and problem size. The C4 implementation's success comes from careful instruction scheduling and pipelining, not from avoiding gather operations.

## Files

- `perf_takehome_v2.py` - Implementation (essentially matches C4)
- `RESULTS.md` - This analysis document
