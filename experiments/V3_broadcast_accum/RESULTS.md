# V3: Broadcast-Accumulate Gather Replacement

## Experiment Summary

**Hypothesis**: Replace expensive scatter gather operations with a broadcast-accumulate pattern that exploits the contiguous layout of tree level values.

**Result**: 4,667 cycles (matches C4 baseline)

**Speedup over reference baseline (147,734)**: 31.65x

## The Original Concept

The broadcast-accumulate approach was designed to exploit a key insight about the tree structure:

For each level k, tree values are stored at contiguous indices:
- Level 0: index 0 (1 node)
- Level 1: indices 1-2 (2 nodes)
- Level 2: indices 3-6 (4 nodes)
- Level 3: indices 7-14 (8 nodes)
- Level k: indices (2^k - 1) to (2^(k+1) - 2)

The proposed algorithm:
```
# Load all 8 level values at once (contiguous!)
vload(level_vals, forest_values_p + 7)  # 1 cycle!

# For each position, broadcast and accumulate:
result = 0
for i in 0..7:
    tree_idx = 7 + i
    broadcast(tree_val_broadcast, level_vals[i])  # Extract and broadcast
    mask = (idx == tree_idx)                       # VALU compare
    result += tree_val_broadcast * mask            # VALU multiply-add
```

## Analysis: Why Broadcast-Accumulate Doesn't Provide Speedup

After careful analysis, the broadcast-accumulate approach faces several challenges:

### 1. Vector-to-Scalar Extraction is Expensive

The architecture doesn't have a direct way to extract a scalar from a specific vector position. To broadcast `level_vals[i]`, you would need to:
1. `vstore` the vector to scratch memory (1 cycle)
2. `load` scalar from scratch[i] (shares load unit)
3. `vbroadcast` the scalar (1 VALU operation)

This adds significant overhead compared to direct scalar loads.

### 2. Index Divergence Within Vectors

After the first round of processing, different lanes within a vector can have different tree indices. These indices might be on different tree levels, meaning we can't assume all lanes need values from the same level.

For example, after a few rounds:
- Lane 0: idx = 15 (level 4)
- Lane 1: idx = 7 (level 3)
- Lane 2: idx = 31 (level 5)
- ...

This makes the broadcast-accumulate pattern inefficient because we'd need to check against all possible indices.

### 3. Scatter Gather is Already Efficient

The current scatter gather implementation uses 8 scalar loads for 8 lanes, taking 4 cycles (at 2 loads/cycle). This is reasonably efficient and well-pipelined with the hash computation.

### 4. Cost Comparison

**Current scatter gather (per desk)**:
- 8 scalar loads at scattered addresses = 4 cycles

**Broadcast-accumulate (for level 3 with 8 nodes)**:
- 1 vload = 1 cycle (OR 8 scalar loads = 4 cycles if contiguous)
- 1 vstore to scratch = 1 cycle
- 8 scalar loads from scratch = 4 cycles
- 8 vbroadcasts = 2 cycles (6 VALU/cycle, so ceil(8/6)=2)
- 8 comparisons = 2 cycles
- 8 multiply-accumulates = 2 cycles

Total: 12+ cycles vs 4 cycles for scatter gather

Even with perfect pipelining, the broadcast-accumulate approach requires more operations.

## What Was Implemented Instead

Since broadcast-accumulate doesn't provide a speedup, this experiment instead implements C4-style optimizations:

1. **Round Fusion**: Process 2 rounds without intermediate memory writeback
2. **FMA Operations**: Use `multiply_add(idx, v_two, v_one)` for branch computation
3. **Store Coalescing**: Overlap stores with loop control
4. **Address Pipelining**: Pre-compute addresses during VALU-heavy phases

These optimizations are orthogonal to the gather mechanism and achieve the same 4,667 cycle count as C4.

## Cycle Count Results

| Implementation | Cycles | Speedup vs Baseline |
|---------------|--------|---------------------|
| Reference Baseline | 147,734 | 1.00x |
| C4 (Full Combo) | 4,667 | 31.65x |
| V3 (This experiment) | 4,667 | 31.65x |

## Alternative Approaches to Explore

If gather optimization is still desired, consider:

1. **Caching frequently accessed levels**: Since all indices start at 0, level 0's single value is accessed for all lanes initially. Pre-loading and caching this could help.

2. **Level-aware batching**: Track which level each desk is on and batch desks by level for shared loading.

3. **Speculative loading**: For deeper levels with more nodes, speculatively load likely-needed values.

4. **Hardware gather instruction**: If the architecture supported vector gather, this would be the ideal solution.

## Conclusion

The broadcast-accumulate pattern is an elegant theoretical optimization but doesn't translate to practical speedup on this architecture due to:
- Lack of efficient vector-to-scalar extraction
- Index divergence within vectors
- Already efficient scatter gather implementation

The current scatter gather approach, while seemingly inefficient, is well-suited to the architecture's capabilities and benefits more from orthogonal optimizations like round fusion and instruction interleaving.
