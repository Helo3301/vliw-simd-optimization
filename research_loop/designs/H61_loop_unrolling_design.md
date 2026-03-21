# H61: Loop Unrolling Analysis

## Hypothesis

Loop unrolling reduces loop overhead and might expose more instruction-level parallelism (ILP). Could 2x or 4x unrolling bring us closer to the 1,790 cycle target from 3,462?

## Loop Overhead Measurement

### H54 Loop Structure (per iteration)

H54 executes 16 iterations with this structure:

**Phase 1: Offset Calculation** (5 cycles)
- Compute 16 desk offsets: 2 cycles (12 ALU ops per cycle)
- Compute 32 load addresses: 3 cycles (12 ALU ops per cycle)

**Phase 2: Load Input Data** (16 cycles)
- Load 32 vector values (idx/val pairs) at 2 per cycle
- Dependency: waits on Phase 1

**Phase 3: Prepare Gather Addresses** (6 cycles)
- Broadcast forest_values_p: 3 cycles (6 VALU ops per cycle)
- Add indices to addresses: 3 cycles (6 VALU ops per cycle)

**Phase 4: Gather + Hash (Round 1)** (~43 cycles)
- Interleaved gather (4 cycles per desk) + hash pipeline
- Load bottleneck: 2 slots/cycle limits progress

**Phase 5: Gather + Hash (Round 2)** (~43 cycles)
- Same pattern as Round 1 (2 rounds fused per iteration)

**Phase 6: Store Phase** (19 cycles)
- Compute store addresses: 3 cycles
- Store 32 values: 16 cycles (at 2 per cycle)

**Phase 7: Loop Control** (4 cycles)
- Increment batch_offset and iter_counter: 1 cycle
- Comparisons: 1 cycle
- Flow control and conditional jump: 2 cycles

### Summary

**Pure Loop Overhead: ~12 cycles per iteration**
- Offset computation: 5 cycles
- Store address computation: 3 cycles
- Loop control: 4 cycles

**Total per iteration: ~136 cycles (estimated)**
- Compute: 73 cycles
- Memory+Control: 63 cycles

**Measured total: 3,462 cycles = 216 cycles/iteration**
- Difference: 80 cycles/iteration overhead from memory latency and stalls

## Unrolling Analysis

### 2x Unrolling (8 iterations instead of 16)

**Savings:**
- Loop control: 16 × 4 cycles = 64 cycles → 8 × 4 = 32 cycles → Save 32 cycles
- Offset computation: 16 × 5 cycles = 80 cycles → 8 × 5 = 40 cycles → Save 40 cycles
- Store address computation: 16 × 3 cycles = 48 cycles → 8 × 3 = 24 cycles → Save 24 cycles
- **Total savings: 96 cycles (2.8% improvement)**

**Scratch Impact:**
- Current: 16 desks × 48 words + constants = ~1,010 words (fits in 1,536)
- 2x unroll (32 desks): 32 × 48 words + constants = ~1,700 words (EXCEEDS limit!)
- **Verdict: TOO MUCH SCRATCH MEMORY**

### 4x Unrolling (4 iterations instead of 16)

**Savings:**
- Loop control: 64 cycles → 16 cycles → Save 48 cycles
- Offset computation: 80 cycles → 20 cycles → Save 60 cycles
- Store address computation: 48 cycles → 12 cycles → Save 36 cycles
- **Total savings: 144 cycles (4.2% improvement)**

**Scratch Impact:**
- 4x unroll (64 desks): 64 × 48 words + constants = ~3,200+ words (CATASTROPHIC!)
- **Verdict: IMPOSSIBLE due to scratch limits**

## Load Bottleneck Analysis

The real bottleneck is **not loop overhead** but **load bandwidth**.

- H54 measures: 3,462 cycles
- Theoretical minimum (perfect scheduling): ~2,000-2,100 cycles
- Actual gap: 1,300+ cycles (37% wasted on latency/stalls)

**Load constraints:**
- 2 slots per cycle maximum
- Tree traversal is sequential (dependent loads)
- Each gather takes ~4 cycles minimum per desk
- 16 desks × 2 rounds = 32 desk-traversals

**With 2x unrolling:**
- 8 iterations instead of 16
- Same total work
- Loop overhead savings: 96 cycles
- **Estimated improvement: 96 cycles → 3,366 cycles (0.5% improvement)**
- **Still 1,576 cycles away from target**

## Round Fusion Alternative

Current H54 already fuses 2 rounds per iteration. Could we fuse MORE rounds?

**4 rounds per iteration:**
- Would reduce iterations from 16 to 8
- Saves loop overhead (as in 2x unrolling): ~96 cycles
- Requires more scratch for intermediate state
- Current desk footprint: 48 words (idx, val, node_val, addr, tmp1, tmp2)
- **4 rounds would need: intermediate results for 3 extra rounds in flight**
- Estimated scratch: ~16 desks × (48 + 30) = ~1,250 words (fits!)
- **This might be worth exploring**

**8 rounds per iteration:**
- Would reduce iterations to 4
- Saves loop overhead: ~192 cycles
- Scratch explosion: 16 desks × (48 + 60) = ~1,728 words (borderline!)
- Reduces ILP opportunity (fewer independent traversals)
- **Likely not beneficial due to reduced parallelism**

## Conclusion

**Loop unrolling is NOT the right optimization for H54.**

1. **Loop overhead is minimal** (~12 cycles/iteration, 3-5% of total)
2. **Scratch memory is tight** - unrolling requires 2x desks, which breaks memory limits
3. **The bottleneck is load latency** - we need better latency hiding, not fewer loops
4. **2x unrolling saves only 96 cycles** (2.8%) but requires 1,700 words scratch (1,164 over budget)
5. **4x unrolling is impossible** - exceeds scratch by 1,664 words

### Recommendation: NOT WORTH PURSUING

Instead, focus on:
- **H62: Better load scheduling** to hide gather latency
- **H63: Constraint-aware gathering** to reduce dependent load chains
- **H64: Memory prefetching** or reordering to improve cache behavior
- **H65: Round fusion beyond 2** (4-8 rounds) with careful scratch management

The target of 1,790 cycles requires ~50% cycle reduction, which unrolling cannot deliver.
