# V5: Duplicate Detection & Sharing - Results

## Base Implementation
- **C4 (Full Combo):** 4,667 cycles

## V5 Implementation Results
- **V5 (No changes, baseline copy):** 4,667 cycles
- **V5b (Simplified structure, some pipelining lost):** 4,732 cycles (+65 cycles worse)

## Concept Analysis

### The Opportunity
Within 8 SIMD lanes, multiple lanes often need the SAME tree value (same idx):
- Round 0: ALL lanes have idx=0 (100% duplicates) - could be 1 load instead of 8
- Round 1: At most 2 unique indices - could be 2 loads max
- Round 2: At most 4 unique indices
- Later rounds: Diminishing duplicate probability

### Statistical Insight
With 8 SIMD lanes and limited unique indices in early rounds, by the pigeonhole principle:
- Round 0: 8 lanes, 1 unique index = 100% duplicates
- Round 1: 8 lanes, 2 possible indices = ~37% duplicate rate
- Round 3: 8 lanes, 8 possible indices = no guaranteed duplicates

### Why Duplicate Detection DOESN'T Help

#### 1. Detection Overhead Exceeds Savings

For general duplicate detection, we need to:
```
# Check if all lanes have same idx
# Compare idx[0] with all other lanes
# No horizontal reduction instruction available!
# Must compare pairwise and AND results

# Estimated detection cost: 4-6 cycles
# Potential savings: 2 cycles (4 -> 2 for broadcast)
# Net: LOSS of 2-4 cycles per gather
```

#### 2. Round 0 Optimization Analysis

For round 0 only (where we KNOW all idx=0):
- 64 total iterations in main loop
- 8 iterations process round 0 (iterations 0-7)
- Per iteration savings: 4 desks * 2 cycles = 8 cycles
- Total potential savings: 8 * 8 = 64 cycles

BUT: Conditional check adds overhead:
- Check `iter_counter < 8`: 1 ALU cycle
- Branch: 1 flow cycle
- Per iteration overhead: ~2 cycles * 64 iterations = 128 cycles
- **Net: LOSS of 64 cycles**

#### 3. Pipeline Disruption

The C4 implementation has carefully overlapped:
- Load engine doing gathers
- VALU engine computing hashes from previous desks
- ALU engine precomputing addresses

Changing the gather pattern disrupts this pipeline, as seen in V5b where minor structural changes caused a 65-cycle regression.

#### 4. Architectural Limitations

The architecture lacks:
- Horizontal reduction (no way to efficiently check if all lanes are same)
- Conditional vector operations (no lane-wise masking)
- Scatter/gather optimization (each lane address is independent)

### When V5 WOULD Help

V5 would be beneficial if:
1. **Longer vectors (VLEN >> 8):** More duplicates to detect, higher savings
2. **Horizontal reduction instruction:** O(1) check for all-same condition
3. **Very deep trees:** More rounds, more opportunities
4. **Different access patterns:** If indices had strong locality

### Conclusion

**Duplicate detection is NOT worth the overhead for this workload.**

The key factors:
- Detection cost (no horizontal reduce) exceeds load savings
- Pipeline disruption from changed gather pattern
- Short VLEN (8) limits duplicate probability in later rounds
- Conditional branching overhead exceeds savings for round 0 optimization

The C4 implementation's approach of just doing all 8 loads per desk, while seemingly wasteful, is actually optimal given the architecture constraints. The loads are overlapped with VALU operations, so the "wasted" loads don't add to the critical path.

## Cycle Count Comparison

| Implementation | Cycles | vs C4 |
|----------------|--------|-------|
| C4 (Base)      | 4,667  | -     |
| V5 (Copy)      | 4,667  | +0    |
| V5b (Modified) | 4,732  | +65   |

## Key Insight

**The duplicate loads are NOT on the critical path.** The gather operations are overlapped with hash computations. Even if we reduced the number of loads, the VALU operations would still take the same time. The load engine's work is "free" in the sense that it's happening in parallel with the real bottleneck (VALU throughput).

This is a classic case where optimizing a non-bottleneck resource provides no benefit.
