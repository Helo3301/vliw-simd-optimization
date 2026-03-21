# H21: Speculative Gather Preloading - Results

## Cycle Count
- **H21 Result:** 4,667 cycles
- **C4 Baseline:** 4,667 cycles
- **Change:** 0 cycles (0% improvement)

## Hypothesis Testing

### Original Hypothesis
During the hash phase, the Load engine is often idle. We could:
1. Predict next iteration's gather addresses (batch_offset + 32 is predictable)
2. Start gathering next iteration's tree values during current iteration's hash
3. Hide gather latency behind hash computation

### Expected Savings
- Could hide ~8-12 cycles of gather latency per iteration

### Actual Result
**No improvement achieved.**

## Analysis

### Why the Hypothesis Failed

1. **Load Engine Utilization is High During Critical Phases**
   - During Round 1, the Load engine performs tree gathers for desks 0-3 sequentially
   - Each desk takes 4 cycles (8 lanes at 2 loads/cycle)
   - Total: 16 cycles of continuous Load activity for tree gathers per round
   - The hash computations are already well-overlapped with these gathers

2. **VALU-Only Phases Have Limited Idle Load Cycles**
   - After desk3's tree gather completes, there are ~12 VALU-only cycles
   - However, during these cycles, we CAN compute next iteration's addresses (ALU ops)
   - The C4 baseline already does this (H14 Address Pipeline optimization)

3. **Speculation Overhead Exceeds Benefits**
   - To actually USE speculative preloads, we'd need:
     a. Conditional logic to check if speculation is valid (1 cycle)
     b. Either use pre-loaded data or load fresh
     c. Handle wrap-around cases (1 in 8 iterations)
   - This conditional overhead would exceed any savings

4. **Architecture Limitations**
   - No hardware support for speculative execution
   - vselect and select operations still cost cycles
   - Can't "cancel" speculative loads if wrap-around occurs

### What Was Tested

1. **Speculative Address Computation** - Already done by C4/H14
   - Computing next_batch_offset, next_offset_regs, next_addr_tmp during VALU phases
   - This is "free" since ALU slots are available during VALU-heavy phases

2. **Speculative Input Preloading** - Not beneficial
   - Loading next iteration's idx/val into separate buffers during idle Load cycles
   - Would require conditional logic to use preloaded data vs. fresh loads
   - The 4 cycles of vloads at the start of each iteration are well-pipelined

### Key Insight
The C4 kernel is already highly optimized. The gather latency is mostly hidden by:
- Interleaving 4 desks' operations
- Overlapping desk N's hash computation with desk N+1's tree gathers
- Pipelining address computation with ALU during VALU phases

The "idle" Load cycles exist, but using them speculatively would require:
- Separate scratch buffers for speculative data (memory overhead)
- Conditional logic to select between speculative and fresh data (cycle overhead)
- Complex handling of wrap-around cases

## Speculation Hit Rate Analysis
If implemented:
- **Hit rate:** 87.5% (7 of 8 iterations would benefit)
- **Miss rate:** 12.5% (wrap-around every 8 iterations)
- But the overhead of checking speculation validity and selecting data sources would likely cost more than the 4 cycles saved per speculation hit.

## Conclusion
The speculative gather preloading hypothesis is **not viable** for this architecture because:
1. The speculation overhead exceeds potential benefits
2. The C4 kernel already achieves good Load/VALU overlap through desk interleaving
3. Address computation pipelining (H14) already uses idle ALU cycles effectively

**Recommendation:** Focus on other optimization approaches such as:
- Further hash stage fusion
- Different desk interleaving patterns
- Algorithmic improvements to the hash function
