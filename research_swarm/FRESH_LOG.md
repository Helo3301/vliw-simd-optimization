# Fresh Start Optimization Log

## Target: 1,363 cycles

## Progress Summary
| Iteration | Cycles | Speedup vs Target | Notes |
|-----------|--------|-------------------|-------|
| 2 | 15,992 | 11.7x | Basic vectorized |
| 4 | 10,104 | 7.4x | 2-batch interleaving |
| 6 | 8,231 | 6.0x | 8-batch processing |
| 7 | 7,463 | 5.5x | 16-batch processing |
| 8 | 7,047 | 5.2x | Arithmetic wrap |
| 11 | 7,094 | 5.2x | Clean 16-batch |
| 13 | 6,302 | 4.6x | 2-round fusion |
| 14 | 5,906 | 4.3x | 4-round fusion |
| 15 | 5,708 | 4.2x | 8-round fusion |
| 16 | 5,609 | 4.1x | Full 16-round fusion |
| 19 | 5,392 | 4.0x | Level-aware round 0 |
| 20 | 5,232 | 3.8x | Level-aware rounds 0-1 (vselect) |
| 21 | 5,213 | 3.8x | Level-aware rounds 0-1 (arithmetic) |
| 22 | 5,213 | 3.8x | Same as 21 (cleanup) |
| 23 | 5,252 | 3.9x | Incomplete round 2 setup (WORSE) |
| 24-27 | 5,213 | 3.8x | Various pipelining attempts (no improvement) |

## Current Bottleneck Analysis (7,094 cycles)
Per half (16 batches) = ~220 cycles:
- Load idx/val: 16 cycles
- Compute addresses: 16 cycles
- **Gather: 64 cycles** <- BOTTLENECK (45% of time)
- Move to vectors: 16 cycles
- XOR: 3 cycles
- Hash: 54 cycles (6 stages x 9)
- Next index: 12 cycles
- Wrap: 6 cycles
- Store: 16 cycles
- Pointer update: 16 cycles

220 cycles x 2 halves x 16 rounds = 7,040 cycles (close!)

## Iteration 1-2: Basic Implementation
- **Result**: 15,992 cycles (correct)
- **Analysis**:
  - Processing 256 elements in batches of 8 (VLEN)
  - 16 rounds
  - Each batch iteration ~31 cycles estimated

### Bottleneck Analysis:
Looking at the inner loop:
- 1 cycle: Load v_idx, v_val (vload x2)
- 1 cycle: Compute 8 addresses
- 4 cycles: Gather 8 node values (2 loads/cycle)
- 1 cycle: Copy to vector
- 1 cycle: XOR
- 12 cycles: Hash (6 stages x 2 cycles each)
- 4 cycles: Index computation
- 1 cycle: Store
- 3 cycles: Loop overhead
= ~28 cycles per batch

With 32 batches x 16 rounds = 512 iterations
512 x 28 = 14,336 cycles (close to actual)

### Key Bottlenecks:
1. **Gather is expensive**: 4 cycles for 8 loads
2. **Hash takes 12 cycles**: 6 stages x 2 cycles
3. **Only 1 flow slot** limits control

### Ideas for Next Iterations:
1. **Pipeline multiple batches**: Process 2+ batches in parallel
2. **Overlap gather with hash**: Start next gather while hashing
3. **Use more ALU slots**: 12 ALU slots available, use for hash
4. **Fuse hash stages**: Can we combine operations?
5. **Eliminate loop overhead**: Unroll loops

## Iteration 3: Analysis of Resource Utilization

Per-cycle resources:
- ALU: 12 slots
- VALU: 6 slots
- Load: 2 slots
- Store: 2 slots
- Flow: 1 slot

Current utilization in hash:
- Using 2 VALU slots per stage (could use 6)
- Not using ALU at all during hash
- Not loading anything during hash (4 cycles wasted)

**Key insight**: During hash computation, we have:
- 4 free VALU slots
- 12 free ALU slots
- 2 free load slots

We should overlap gather of next batch with hash of current batch!
