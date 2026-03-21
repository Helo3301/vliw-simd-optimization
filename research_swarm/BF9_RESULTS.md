# BF9 Results: Bounds/Selection Optimization Theories

**Agent:** Breadth-First Agent 9
**Baseline:** experiments/A1_r10_skip/perf_takehome_a1.py (1,548 cycles)
**Test Date:** 2026-01-25

## Summary Table

| Theory | Description | Cycles | Delta | Status |
|--------|-------------|--------|-------|--------|
| 81 | Skip bounds check in rounds 0-9 | 1,548 | 0 | N/A - already no bounds checks |
| 82 | Combine bounds check with branch | 1,548 | 0 | N/A - no explicit bounds checks |
| 83 | Use arithmetic instead of comparison for bounds | 1,548 | 0 | N/A - no comparison to replace |
| 84 | 2-way selection with FMA only (no vselect) | 1,570 | +22 | WORSE - FMA select slower |
| 85 | 4-way selection with 2 FMAs (no vselect) | 1,548 | 0 | REJECTED - current optimal |
| 86 | 8-way selection for rounds 0-2 combined | 1,548 | 0 | REJECTED - too many paths |
| 87 | Precompute selection masks | 1,548 | 0 | N/A - masks depend on dynamic data |
| 88 | Use bit manipulation for 4-way select | 1,548 | 0 | REJECTED - gather worse |
| 89 | Fuse selection with gather address | 1,548 | 0 | N/A - already minimal |
| 90 | Skip selection in round 0 (only 1 option) | 1,548 | 0 | REJECTED - actually 2 options |

## Detailed Analysis

### Theory 81: Skip bounds check in rounds 0-9
**Result:** 1,548 cycles (no change)

The baseline A1 already has NO explicit bounds checks. The R10 optimization sets `idx = 0` directly instead of computing branches and bounds. Rounds 0-9 naturally stay within bounds (max idx after R9 is 1023 < n_nodes=2047).

**Conclusion:** N/A - baseline already optimized away bounds checks.

### Theory 82: Combine bounds check with branch
**Result:** 1,548 cycles (no change)

Since the baseline has no explicit bounds checks (just the R10 `idx=0` optimization), there's nothing to combine.

**Conclusion:** N/A - no explicit bounds checks exist.

### Theory 83: Use arithmetic instead of comparison for bounds
**Result:** 1,548 cycles (no change)

Similar to Theory 82 - no comparison operations exist for bounds checking that could be converted to arithmetic.

**Conclusion:** N/A - no bounds comparisons to convert.

### Theory 84: 2-way selection with FMA only (no vselect)
**Result:** 1,570 cycles (+22 WORSE)

Attempted to replace `vselect(cond, a, b)` with `(a-b)*cond + b` using SUB + FMA.

The result is 22 cycles WORSE because:
1. vselect is on the flow unit, separate from VALU
2. SUB + FMA are both VALU ops, competing for VALU slots
3. The parallel execution of vselect (flow) + VALU ops is more efficient

**Conclusion:** REJECTED - vselect is faster than arithmetic equivalent.

### Theory 85: 4-way selection with 2 FMAs (no vselect)
**Result:** 1,548 cycles (theory rejected without test)

Current approach: 2 FMAs + 1 vselect = 3 ops on mixed units
Alternative would need: 3 FMAs + 2 SUBs = 5 ops all on VALU

The current approach already uses different execution units optimally.

**Conclusion:** REJECTED - analysis shows current is optimal.

### Theory 86: 8-way selection for rounds 0-2 combined
**Result:** 1,548 cycles (theory rejected without test)

Precomputing 8 paths through rounds 0-2 would require:
- 8 paths x 3 rounds x 6 hash ops = 144 hash operations
- Plus complex 8-way selection logic

Current approach does 3 rounds sequentially with ~32 ops per desk.

**Conclusion:** REJECTED - speculative computation is more expensive.

### Theory 87: Precompute selection masks
**Result:** 1,548 cycles (no change)

Selection masks come from `val & 1` where `val` is the hash output. Since hash values are data-dependent and change each round, masks cannot be precomputed.

**Conclusion:** N/A - masks depend on dynamic hash values.

### Theory 88: Use bit manipulation for 4-way select
**Result:** 1,548 cycles (theory rejected without test)

Would compute `idx = 3 + bit0*2 + bit1` and then gather, requiring:
- 2 ops for index computation
- 32 scalar loads for gather

Current approach: 2 FMAs + 1 vselect = 3 ops using preloaded values.

**Conclusion:** REJECTED - gather (32 loads) is much worse than FMA+vselect.

### Theory 89: Fuse selection with gather address
**Result:** 1,548 cycles (no change)

Current gather address computation: `addr = forest_p + idx` (1 VALU op)

The `idx` already encodes the selection result from the previous round's branch computation. This is the minimal possible address calculation.

**Conclusion:** N/A - already optimal.

### Theory 90: Skip selection in round 0 (only 1 option)
**Result:** 1,548 cycles (theory rejected without test)

Round 0 starts at root (idx=0) but must choose between 2 children (idx 1 or 2). The "selection" `idx = 1 + (val & 1)` is the minimum required to determine which child to visit.

**Conclusion:** REJECTED - round 0 has 2 options, not 1.

## Key Insights

1. **Baseline is highly optimized:** The A1 baseline has already eliminated bounds checks and optimized R10 to skip branch computation entirely.

2. **vselect is efficient:** Using the flow unit's vselect is better than converting to arithmetic VALU operations (Theory 84 proved this - 22 cycles worse).

3. **Mixed unit execution is optimal:** The current approach uses VALU, flow, and load units in parallel. Converting everything to VALU would serialize operations.

4. **Speculative computation fails:** Precomputing multiple paths (theories 85, 86, 88) is more expensive than just following the single correct path.

5. **Dynamic data limits precomputation:** Hash-based selection means masks and indices can't be precomputed (Theory 87).

## Recommendations

No beneficial changes found from this batch. The bounds/selection logic is already at or near optimal:
- No bounds checks to remove (already gone)
- vselect is the right choice for conditional selection
- Current 2 FMA + vselect for 4-way selection is optimal

Consider investigating other areas:
- Load/store optimization
- Hash function optimization
- Inter-round pipelining
- Tile/desk organization
