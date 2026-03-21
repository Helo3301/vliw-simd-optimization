# H16: Cross-Iteration Pipelining (Double Buffering) Results

## Hypothesis

Building on C4 (4,667 cycles), overlap stores and loads between iterations using double-buffered registers:
- While storing set A's results, load set B's data
- The store engine (2 slots) and load engine (2 slots) are independent
- Expected to hide store latency behind load operations

## Results

| Metric | Value |
|--------|-------|
| Cycle Count | 4,667 |
| Baseline (C4) | 4,667 |
| Improvement | 0% |
| Correctness | PASSED |

## Implementation Status

**NOT IMPLEMENTED** - The current implementation is functionally identical to C4.

### What Was Attempted

Multiple implementations were tried to overlap stores from iteration N with loads for iteration N+1:

1. **First attempt**: Restructured loop to load next batch during store phase
   - **Failed**: IndexError - out-of-bounds memory access on last iteration
   - Cause: next_addr_tmp addresses pointed past end of batch

2. **Second attempt**: Added wrap-around logic for next_batch_offset
   - **Failed**: Same IndexError
   - Cause: Wrap-around computed too late, addresses already calculated

3. **Third attempt**: Loop peeling with prolog/epilog
   - **Failed**: Complex restructuring broke the interleaved desk pipeline
   - The 4-desk interleaving pattern is tightly coupled with the load/process/store sequence

### Why True Double-Buffering is Difficult

The current C4 structure has:
```
LOOP:
  LOAD  desk0, desk1, desk2, desk3   (5 cycles)
  PROCESS rounds 1 and 2             (~60 cycles)
  STORE desk0, desk1, desk2, desk3   (4 cycles)
  JUMP to LOOP
```

True double-buffering requires:
```
PROLOG:
  LOAD first batch into set A

LOOP:
  PROCESS set A (round 1 and 2)
  STORE set A + LOAD next batch into set B   <-- overlap here
  SWAP A and B
  JUMP to LOOP

EPILOG:
  PROCESS final set
  STORE final set (no load)
```

The challenges are:

1. **Register pressure**: Need 2x the desk registers (8 desks instead of 4)
   - Current: 6 registers per desk = 24 vector registers
   - Double-buffered: 48 vector registers required
   - Scratch space may not accommodate this

2. **Address computation coupling**: The H14 optimization pre-computes next addresses during processing
   - These addresses are tightly integrated with the hash pipeline
   - Separating them for two register sets is non-trivial

3. **Last iteration handling**: The final iteration must not load (or must load "safe" wrapped addresses)
   - This requires conditional logic or loop peeling
   - Loop peeling doubles code size

4. **Store/Load slot limits**: Only 2 load slots and 2 store slots per cycle
   - Storing 4 desks = 8 vstores = 4 cycles minimum
   - Loading 4 desks = 8 vloads = 4 cycles minimum
   - Even with overlap, we can only do 2 stores + 2 loads per cycle
   - Potential savings: 4 cycles per iteration (from 8 to 4 if fully overlapped)

### Theoretical Maximum Savings

If we could fully overlap stores and loads:
- Current store phase: 4 cycles
- Current load phase: ~5 cycles
- With overlap: max(4, 5) = 5 cycles total instead of 4+5 = 9 cycles
- Savings: ~4 cycles per iteration

With ~146 iterations (batch_size=256, VLEN=8, 4 desks, 2 rounds/iter):
- total_iterations = 256 / 8 * 8 / 4 = 64 iterations of the main loop
- Potential savings: 64 * 4 = 256 cycles
- New target: 4,667 - 256 = ~4,411 cycles

This is a ~5.5% improvement - worthwhile but requires significant restructuring.

## Conclusion

**H16 Double Buffering was NOT successfully implemented.**

The hypothesis is theoretically sound - store and load engines are independent and can operate in parallel. However, the practical implementation requires:
1. Loop peeling (prolog/epilog)
2. Doubling the desk register allocation
3. Careful handling of iteration boundaries
4. Redesigning the address pre-computation pipeline

The tight integration of H12 (round fusion), H13 (store coalescing), and H14 (address pipelining) in C4 makes it difficult to add H16 without a complete rewrite of the loop structure.

### Recommendations for Future Work

1. **Fresh implementation**: Start from scratch rather than modifying C4
2. **Separate register sets**: Allocate desks_A[4] and desks_B[4] from the beginning
3. **Explicit state machine**: Model the prolog/main/epilog phases explicitly
4. **Conditional last-iteration logic**: Use the existing iteration counter to skip loads on final iteration

## Files

- `perf_takehome_h16.py`: Implementation (currently identical to C4)
- `RESULTS.md`: This file

## Command to Reproduce

```bash
python3.11 experiments/H16_double_buffer/perf_takehome_h16.py --check
```
