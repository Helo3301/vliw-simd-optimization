# Experiment H136: Speculative Dual-Address Computation

## Summary

**Baseline:** H120 = 1,840 cycles (80.3x speedup)
**Target:** 1,790 cycles (82.5x speedup)
**Gap:** 50 cycles

**Result:** No improvement achieved. All variants remained at 1,840 cycles.

## Hypothesis

For gather rounds, we compute `addr = forest_p + idx` AFTER the branch computation. The idx depends on the hash result. If we could precompute both possible next addresses during the hash computation and then select the correct one, we might be able to break the dependency chain.

## Approaches Tested

### V1: Speculative Address During Hash
- Compute `spec_base = 2*idx + 1` during hash stages
- After hash, compute `new_idx = spec_base + (val & 1)`
- Then `new_addr = forest_p + new_idx`

**Result:** 1,844 cycles (+4 cycles)
**Analysis:** Added extra VALU operations that couldn't be hidden within existing slack. The scheduler was already efficient.

### V2: Speculative Address During Gather
- Try to overlap address computation with the gather phase (128 cycles of load-bound operation)
- Issue: Can't use addr register for speculation because it's needed for the current gather

**Result:** 1,840 cycles (no change from baseline)
**Analysis:** Functionally identical to baseline after fixing the addr overwrite bug.

### V3: Precompute idx+1 During Hash
- Precompute `idx_plus_one = idx + 1` during hash
- Branch becomes: `new_idx = idx + idx_plus_one + bit`
- This equals `2*idx + 1 + bit`

**Result:** 1,840 cycles (no change)
**Analysis:** Extra ops hidden in scheduler slack, but no net gain.

### V4: Address-Centric FMA Computation
- Precompute `forest_p_plus_1 = forest_p + 1`
- Use FMA: `addr = FMA(idx, 2, forest_p_plus_1) + bit`
- Recover: `idx = addr - forest_p`

**Result:** 1,840 cycles (no change)
**Analysis:** The idx recovery step cancels any savings from fused FMA.

## Root Cause Analysis

The fundamental issue is that the address computation is NOT on the critical path. The critical path is:

```
Gather (128 cycles) -> XOR -> Hash (15 ops) -> Branch (3 ops) -> Addr (1 op)
                                                                     |
                                                            [Next round's gather]
```

The gather phase dominates at 128 cycles. During gather, VALU is idle, but we cannot start computing the next address until:
1. Gather completes (need node_val)
2. XOR completes (need val ^ node_val)
3. Hash completes (need hash result for branch bit)
4. Branch completes (need new idx)

This dependency chain cannot be broken because:
- `new_idx = 2*old_idx + 1 + (hash_val & 1)`
- `hash_val` depends on `old_val ^ old_node_val`
- `old_node_val` comes from gather

The only way to break this would be to compute addresses for ALL possible future paths, but that's exponential (2^N paths for N rounds ahead).

## Key Insights

1. **The scheduler is already efficient:** Adding operations during "idle" periods doesn't help if those periods aren't truly idle from a dependency perspective.

2. **Dependency chains are the bottleneck:** The address computation is cheap (1 op) but must wait for the hash result.

3. **Gather dominates:** At 128 cycles, the gather phase is the real bottleneck. Optimizing address computation saves at most a few cycles per round.

4. **No free lunch with speculation:** Any speculative computation that requires recovery later adds net operations.

## Conclusion

The hypothesis that speculative dual-address computation could save 50 cycles is incorrect. The address computation is not a significant bottleneck, and the dependency on the hash result cannot be circumvented. The current H120 implementation at 1,840 cycles is already well-optimized for address handling.

To achieve the target of 1,790 cycles, optimization efforts should focus on:
1. Reducing gather latency (unlikely without hardware changes)
2. Reducing hash computation (already FMA-optimized)
3. Reducing total number of rounds/operations (algorithmic change)
4. Better cross-round pipelining (explored in H16, found too complex)
