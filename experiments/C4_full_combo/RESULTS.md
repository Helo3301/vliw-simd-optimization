# C4: Full Combo Results (H12 + H11 + H13 + H14)

## Summary

| Metric | Value |
|--------|-------|
| **C4 Cycle Count** | **4,667 cycles** |
| **Speedup over Baseline (147,734)** | **31.66x** |
| **Improvement over H12 (5,115)** | **448 cycles (8.76% faster)** |

## Individual Implementation Benchmarks

| Implementation | Cycles | Technique |
|---------------|--------|-----------|
| H12 (Round Fusion) | 5,115 | 2 rounds per iteration, no intermediate store/load |
| H11 (Branch FMA) | 5,819 | FMA for branch: multiply_add(idx, 2, 1) |
| H13 (Store Coalescing) | 5,435 | Overlap stores with ALU operations |
| H14 (Address Pipeline) | 5,307 | Precompute addresses with ALU during VALU phases |
| **C4 (Combined)** | **4,667** | All optimizations combined |

## Optimization Breakdown

### H12: Round Fusion (Base)
- **Contribution**: Largest impact - eliminates 50% of vload/vstore pairs
- **Mechanism**: Process 2 rounds without intermediate memory writeback
- **Integration**: Forms the structural foundation of C4

### H11: Branch FMA
- **Contribution**: Reduces branch computation by 1 cycle per desk
- **Mechanism**: Replaces `idx = idx * 2; idx = idx + 1` with `idx = multiply_add(idx, 2, 1)`
- **Integration**: Applied to all 8 branch computations (4 desks x 2 rounds)

### H13: Store Coalescing
- **Contribution**: Eliminates pure store cycles by overlapping with ALU
- **Mechanism**: Batch stores with loop control operations
- **Integration**: Applied at end of double-rounds, stores overlap with:
  - batch_offset and iter_counter updates
  - Comparison operations for loop control
  - wrap-around select for batch_offset

### H14: Address Pipeline
- **Contribution**: Utilizes idle ALU slots during VALU-heavy phases
- **Mechanism**: Precompute next iteration's addresses while hash computes
- **Integration**: During round 2's hash phases, ALU computes:
  - next_batch_offset
  - next_offset_regs[0..3]
  - next_addr_tmp[0..7]

## Optimization Interactions

### Synergies
1. **H12 + H13**: Round fusion doubles the amount of useful work between stores, making store coalescing more impactful
2. **H12 + H14**: More VALU phases means more opportunities for ALU address pipelining
3. **H11 + all**: Branch FMA savings compound since it's applied in both rounds

### Trade-offs
1. **Register Pressure**: H12 requires keeping values in registers across two rounds
2. **Code Size**: Combined kernel is significantly larger due to duplicated round logic
3. **H14 Complexity**: Pre-computed addresses must be carefully coordinated with loop control

## Cycle Accounting

### Per Double-Round Iteration (approximate breakdown)

| Phase | Cycles (est.) | Description |
|-------|--------------|-------------|
| Load Phase | ~6 | Load all 4 desks (overlapped) |
| Round 1 Gather + Hash | ~16 | Interleaved gather/hash for 4 desks |
| Round 1 Branch | ~10 | Branch computation with FMA |
| Round 2 Setup | ~2 | Address recomputation |
| Round 2 Gather + Hash | ~16 | Interleaved gather/hash for 4 desks |
| Round 2 Branch | ~10 | Branch computation with FMA |
| Store + Loop Control | ~4 | Coalesced stores |
| **Total per iteration** | **~64** | Processing 32 elements x 2 rounds |

### Total Iterations
- batch_size=256, VLEN=8, rounds=16, NUM_DESKS=4
- total_iterations = (256 / 8) * (16 / 2) / 4 = 64 iterations
- 64 iterations * ~64 cycles + ~570 cycles setup = ~4,667 cycles

## Conclusion

The C4 full combo achieves the best performance by:
1. Building on H12's round fusion as the structural foundation
2. Systematically applying H11's branch FMA to all 8 branch points
3. Leveraging H13's store coalescing at the end of each double-round
4. Filling VALU-heavy phases with H14's address pipelining

The 448-cycle improvement over H12 alone (8.76%) demonstrates that the other optimizations contribute meaningfully even when round fusion is already in place.
