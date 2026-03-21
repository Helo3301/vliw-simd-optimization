# H17: 4-Round Fusion Results

## Summary

| Metric | Value |
|--------|-------|
| Cycle Count | 4,698 |
| H12 Baseline (2-round) | 5,115 |
| Improvement over H12 | 417 cycles (8.2%) |
| Original Baseline | 147,734 |
| Speedup over baseline | 31.4x |

## Hypothesis

H12's 2-round fusion saved 14% by eliminating 50% of vload/vstore operations. The hypothesis was that 4-round fusion would:
- Eliminate 75% of vload/vstore (vs 50%)
- Reduce iterations from 64 to 32
- Process 4 consecutive rounds with values in registers
- Expected savings: ~500 additional cycles

## Results

**Hypothesis: VALIDATED**

The implementation achieved 417 cycles savings (expected ~500), providing an 8.2% improvement over H12's 2-round fusion.

### Memory Operation Analysis

| Configuration | Loads per batch element | Stores per batch element |
|--------------|-------------------------|--------------------------|
| No fusion (baseline) | 16 | 16 |
| 2-round fusion (H12) | 8 | 8 |
| 4-round fusion (H17) | 4 | 4 |

By processing 4 rounds per memory access cycle:
- Each batch element is loaded once for 4 rounds instead of once for 2 rounds
- Total vload/vstore operations reduced by another 50% vs H12

### Cycle Breakdown

The loop now processes:
- 32 iterations total (was 64 in H12)
- Each iteration: Load -> Hash(R1) -> Hash(R2) -> Hash(R3) -> Hash(R4) -> Store
- 4 desks interleaved per iteration

### Register Pressure Analysis

**Challenge identified:** Tracking values across 4 rounds requires careful register management.

**Solution implemented:**
- 4 desks, each with 6 vector registers:
  - `idx`: Current tree index (8 elements)
  - `val`: Current hash value (8 elements)
  - `node_val`: Gathered node values (8 elements)
  - `addr`: Gather addresses (8 elements)
  - `tmp1`, `tmp2`: Temporary registers for hash computation (16 elements)

Total vector scratch used per desk: 48 elements
Total across 4 desks: 192 elements

Plus hash constants (6 stages x 2 vectors each = 96 elements) and broadcast constants.

**Verdict:** The architecture's scratch space was sufficient. No register spilling required.

### What Worked

1. **Amortized memory overhead:** The 4-round fusion successfully amortizes the vload/vstore cost across more computation
2. **Instruction scheduling:** The interleaved desk pattern from H12 scaled well to 4 rounds
3. **Register reuse:** The same `idx`, `val`, `tmp1`, `tmp2` registers are reused across all 4 rounds within an iteration

### What Didn't Work as Expected

1. **Slightly below expected savings:** Achieved 417 cycles vs expected 500 cycles
   - Reason: Additional address computation overhead for 4 gather phases per iteration
   - Each round requires `vbroadcast` + `add` for gather address setup (2 cycles per round x 4 = 8 cycles overhead)

2. **Code size increased significantly:** The unrolled 4-round structure results in a much larger kernel
   - Mitigation: Used helper methods (`_emit_round_n`, `_emit_final_round`) to reduce code duplication

### Potential Further Optimizations

1. **8-round fusion:** Would reduce vload/vstore to 2 per batch element, but:
   - Code size would be enormous
   - Register pressure approaching limits
   - Diminishing returns (would save ~200 additional cycles at most)

2. **Better address computation pipelining:** The gather address setup could potentially overlap more with computation

## Conclusion

4-round fusion is a worthwhile optimization over 2-round fusion, providing an 8.2% additional speedup. The register pressure concerns did not materialize as the architecture has sufficient scratch space. The main trade-off is increased code complexity and size.

The technique demonstrates that for memory-bound kernels, reducing memory operations through round fusion provides significant performance gains, even when the reduction follows a diminishing-returns curve.
