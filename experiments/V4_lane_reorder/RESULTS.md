# V4: Lane Reordering / Broadcast Optimization Results

## Concept

The core insight behind V4 was that batch elements could be reorganized to align SIMD lanes with tree indices, potentially enabling vectorized loads instead of scalar gathers.

**Key observations about the tree traversal:**
1. **Round 0:** ALL 256 batch elements start with `idx=0`
2. **Round 1:** Elements have `idx` in {1, 2} (only 2 possibilities)
3. **Round 2:** Elements have `idx` in {3, 4, 5, 6} (4 possibilities)
4. **Round 3:** Elements have `idx` in {7-14} (8 possibilities = VLEN!)

## Implementations Tested

### V4 (perf_takehome_v4.py)
A simplified version matching C4's structure with minor modifications.

**Result:** 4,672 cycles (5 cycles SLOWER than C4's 4,667)

The overhead of loading unused tree values negated any benefit.

### V4b (perf_takehome_v4_broadcast.py)
Attempted to use broadcast for round 0 instead of gather:
- First loop: Process rounds 0-1 with broadcast for round 0
- Second loop: Process rounds 2-15 with standard gather

**Result:** 4,710 cycles (43 cycles slower than C4) + INCORRECT RESULTS

The implementation complexity introduced bugs and did not achieve the expected savings.

## Analysis: Why V4 Doesn't Help

### Theoretical Savings (Round 0 Broadcast)
- Round 0 processes 256 elements in 8 iterations (32 elements/iteration with 4 desks)
- Each iteration currently does 4 desks * 4 cycles gather = 16 cycles for round 0
- Broadcast could replace this with 1 cycle
- **Maximum theoretical savings:** ~120 cycles for round 0 only

### Why This Doesn't Work in Practice

1. **C4's interleaving already hides gather latency**
   - The gather operations are overlapped with hash computations
   - The 4 cycles per desk are hidden by the VALU pipeline
   - Eliminating gathers doesn't reduce the critical path

2. **Loop separation overhead**
   - Splitting into two loops adds control flow overhead
   - First loop needs its own store phase
   - Second loop needs additional setup

3. **Hash computation is the bottleneck**
   - 6 hash stages * 2 operations/stage = 12 VALU operations per round
   - With 4 desks interleaved, VALU is always busy
   - Gather operations fit into "bubbles" where VALU would be idle

4. **VLEN=8 limitation**
   - Only round 3 has exactly 8 possible indices (matching VLEN)
   - Even if we could vectorize the tree load, it's only 1 round out of 16

## Partitioning/Bucketing Analysis

The original V4 concept involved **reorganizing batch elements** so elements with the same tree index end up in the same SIMD lane.

**Problems:**
1. **Partitioning cost:** Sorting/bucketing 256 elements by their index requires O(batch_size) work per round
2. **Dynamic nature:** The partition changes every round as indices diverge
3. **No SIMD benefit:** Even after partitioning, we still need to gather from different tree locations

## Conclusion

**V4 is a dead end.** The broadcast optimization for round 0 provides negligible benefit because:
- C4 already hides gather latency through instruction-level parallelism
- The hash computation pipeline is the actual bottleneck
- Architectural complexity increases code complexity without payoff

**The gather operation is NOT the performance bottleneck** - it's hidden by the VALU pipeline. Further optimization should focus on:
- Reducing hash computation cycles
- Better memory access patterns for batch data (already optimized in C4)
- Exploring instruction-level optimizations within the VALU pipeline

## Cycle Counts Summary

| Implementation | Cycles | vs C4 |
|---------------|--------|-------|
| C4 (baseline) | 4,667 | 1.00x |
| V4 (simplified) | 4,672 | 0.999x (slower) |
| V4b (broadcast attempt) | 4,710 | 0.991x (slower, buggy) |

**Recommendation:** Abandon V4 approach and focus optimization efforts elsewhere.
