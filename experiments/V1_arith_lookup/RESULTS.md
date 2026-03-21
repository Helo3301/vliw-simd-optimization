# V1: Arithmetic Table Lookup - Results

## Concept

The idea was to replace scattered scalar loads (vgather) with arithmetic operations:

1. **Pre-load tree level contiguously**: Use vload to load an entire tree level (e.g., level 3 has 8 values at indices 7-14)
2. **Use arithmetic to select**: Instead of 8 scalar loads, use compare-multiply-accumulate:
   ```
   result = 0
   for i in 0..7:
       mask = (offset == i)  # VALU compare -> 0 or 1
       result += level_val[i] * mask  # VALU multiply-add
   ```

## Implementation

Created two variants:
- `perf_takehome_v1.py`: Full version with pre-loaded level values (still uses standard gather)
- `perf_takehome_v1_level0.py`: Simplified version focusing on level 0 optimization

## Results

| Implementation | Cycles | Speedup vs Baseline | vs C4 (4,667) |
|---------------|--------|---------------------|---------------|
| C4 (baseline) | 4,667  | 31.65x              | 1.000x        |
| V1 (full)     | 4,966  | 29.75x              | 0.940x        |
| V1 (level0)   | 4,925  | 30.00x              | 0.948x        |

## Analysis

### Why This Approach Didn't Work

1. **Cycle Cost Analysis**:
   - Standard gather: 8 scalar loads at 2/cycle = 4 cycles
   - Arithmetic lookup for level 3: 8 compare + 8 multiply + 8 add = 24 VALU ops
   - At 6 VALU/cycle = 4 cycles minimum
   - **Result**: Arithmetic lookup is NOT faster than gather for VLEN-sized levels

2. **VALU Contention**:
   - The C4 kernel already uses VALU heavily for hash computations
   - During each gather, the VALU is busy computing hashes for previous desks
   - Adding arithmetic lookup would compete for the same VALU slots
   - **Result**: Can't overlap arithmetic lookup with existing work

3. **Overhead Costs**:
   - Pre-loading level values requires setup cycles
   - Broadcasting each level value to a vector takes additional VALU cycles
   - These costs are amortized over many iterations, but they add up
   - **Result**: Initialization overhead (about 40 extra cycles for level preloading)

4. **Level Divergence Problem**:
   - After the first few rounds, different lanes are at different tree levels
   - To use arithmetic lookup, we'd need to detect which level each lane is in
   - This requires additional comparison operations per lane
   - **Result**: Complex level detection negates any potential savings

### When Arithmetic Lookup COULD Work

1. **If VALU were much faster** (e.g., 12 ops/cycle instead of 6):
   - The 24 VALU ops would take only 2 cycles vs 4 cycles for gather
   - But with current slot limits, this isn't beneficial

2. **For very small levels** (level 0-1):
   - Level 0 (1 value): Just broadcast, no arithmetic needed
   - Level 1 (2 values): 6 VALU ops = 1 cycle vs 4 cycles gather
   - But we only visit these levels briefly, so limited overall impact

3. **If Load engine were more constrained**:
   - Current architecture allows 2 loads/cycle
   - If it were 1 load/cycle, gather would take 8 cycles
   - Then 4-cycle arithmetic lookup would be 2x faster

4. **With true SIMD shuffle/permute operations**:
   - If the ISA had a permute instruction (like x86 VPERMPS)
   - Could select per-lane values in 1-2 cycles
   - But this architecture doesn't have such instructions

## Conclusion

The arithmetic table lookup approach is **not beneficial** for this architecture because:

1. The arithmetic cost matches the gather cost for VLEN-sized levels
2. The VALU is already heavily utilized during gather (hash computation)
3. Initialization overhead adds cycles without proportional benefit
4. Level divergence makes the approach complex for later rounds

The C4 implementation with interleaved desk processing remains the best approach because it hides gather latency by overlapping with hash computations across multiple desks.

## Files

- `perf_takehome_v1.py`: Full implementation with level pre-loading
- `perf_takehome_v1_level0.py`: Simplified version focusing on level 0

## Key Takeaway

For this VLIW architecture, **memory coalescing and instruction interleaving** are more effective than **computation substitution**. The bottleneck is not the Load engine (2 ops/cycle) but rather the overall instruction scheduling and data dependencies.
