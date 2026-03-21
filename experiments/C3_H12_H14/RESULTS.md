# C3: Combined H12 (Round Fusion) + H14 (Parallel Address Pipeline)

## Results Summary

| Metric | Value |
|--------|-------|
| **Cycle Count** | 4,795 |
| **H12 Baseline** | 5,115 |
| **Improvement over H12** | 320 cycles (6.26%) |
| **Original Baseline** | 147,734 |
| **Speedup over Original** | 30.81x |

## Techniques Combined

### H12: Round Fusion
- Process 2 consecutive rounds without intermediate memory writeback
- Keeps tree indices in registers between rounds
- Eliminates 50% of vload/vstore pairs per desk
- Total iterations reduced from 16 rounds to 8 "double-rounds"

### H14: Parallel Address Pipeline
- Uses ALU to pre-compute next iteration's base addresses while VALU computes hash stages
- Moves vbroadcast operations earlier where possible
- Overlaps desk 2,3 gather address adds with first desk0 gather
- Overlaps store phase with loop control ALU operations

## What Worked

1. **Address Pre-computation During Round 1 Hash Phase**: During the VALU-heavy hash computation of round 1, the ALU was mostly idle. H14's technique of computing next_batch_offset and next_offset_regs during this time carried over well.

2. **Early vbroadcast Operations**: Moving vbroadcast operations earlier in the setup phase (from H14) allowed better pipelining of the initial desk address computation.

3. **Desk 2,3 Address Overlap with First Gather**: The H14 optimization of computing desk 2,3 gather addresses during the first desk0 gather cycle was preserved in the combined kernel.

4. **Store Phase ALU Overlap**: H14's technique of overlapping loop control ALU operations with the store phase was preserved, saving cycles at the end of each iteration.

5. **Combined Benefits**: The round fusion (H12) cuts the number of memory round-trips in half, while the address pipeline (H14) improves ALU/VALU utilization during the remaining operations.

## What Could Be Improved

1. **Round 2 Address Pipeline**: The H14 address pre-computation is only applied to round 1's hash phase. Round 2's hash phase could also benefit from similar ALU overlap, though it would require pre-computing addresses for the next loop iteration.

2. **Further Fusion**: Processing 3 or 4 rounds without memory writeback could yield additional savings, though register pressure becomes a concern.

3. **Round 2 Gather Address Computation**: Currently uses 2 VALU cycles for round 2 gather address computation (vbroadcast + add). This could potentially be overlapped with the tail of round 1's branch computation.

## Cycle Breakdown

The combined kernel processes 2 rounds per iteration with 8 total iterations:
- Setup: ~37 cycles (one-time)
- Per iteration: ~59 cycles x 8 = 472 cycles (estimated)
- Total: ~4,795 cycles

Compared to H12's 5,115 cycles, the 320 cycle savings come primarily from:
- Better ALU/VALU overlap during hash phases
- Earlier vbroadcast operations
- Overlapped store and loop control operations
