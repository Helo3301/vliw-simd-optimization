# Experiment H7: Aggressive Cross-Desk Interleaving

## Summary

This experiment implements the key insight from T3's constraint solver: interleave operations from all 4 desks so that while one desk is hashing, another is gathering, another is doing branch computation, etc.

**Result: 6,203 cycles** (22.4% improvement over T6+T5's 7,995 cycles)

| Metric | Value |
|--------|-------|
| H7 cycles | 6,203 |
| T6+T5 cycles (previous best) | 7,995 |
| Improvement over T6+T5 | 22.4% |
| Baseline cycles | 9,793 |
| Improvement over baseline | 36.7% |
| T3 theoretical optimal | ~4,352 (34 cycles/iteration * 128) |
| Gap from theoretical | 42.5% |

## Approach

### Key Insight from T3

The T3 constraint solver proved that 34 cycles per loop iteration is achievable vs the ~62 cycles in sequential processing. The current code processes desks sequentially:
```
desk0: gather -> hash(6 stages) -> branch -> store
desk1: gather -> hash(6 stages) -> branch -> store
...
```

The optimal schedule interleaves operations so that in any given cycle, we have operations from multiple desks at different pipeline stages.

### Pipeline Structure

H7 creates a software pipeline where:
- **Gather phase** for desk N+1 overlaps with **hash phase** for desk N
- **Hash stages** from multiple desks execute in the same cycle
- **Branch computation** overlaps with late hash stages of following desks

Example of interleaved execution:
```
Cycle X:   desk0_hash_stage4(FMA) + desk1_hash_stage1_combine
Cycle X+1: desk0_hash_stage5_prep  + desk1_hash_stage2(FMA) + desk2_XOR + desk3_gather_lanes_0-1
```

### Combined Optimizations

H7 combines:
1. **T6's FMA optimization** - Hash stages 0, 2, 4 use `multiply_add` (single cycle instead of 3)
2. **T5's 6-VALU packing** - Pack up to 6 VALU ops per cycle from different desks
3. **Aggressive interleaving** - Operations from all 4 desks in every cycle

## Implementation Details

### Phase 1: Load (5-6 cycles)
- Load indices and values for all 4 desks
- Compute gather addresses
- Pack ALU operations (12 slots available)
- Overlap vloads (2 per cycle) with VALU broadcasts

### Phase 2: Interleaved Gather + Hash (~25 cycles)
This is where the major gains come from:
- Start desk0 gather (4 cycles)
- While gathering desk1, start desk0 XOR and hash
- While gathering desk2, continue desk0 hash and start desk1 hash
- And so on...

Key patterns:
- **FMA stages (0, 2, 4)**: Single multiply_add instruction per desk
- **Non-FMA stages (1, 3, 5)**: 2-cycle pattern (prep -> combine)
- **Branch**: 4 ops (AND, MUL, ADD, ADD) can be spread across 2 cycles

### Phase 3: Store + Loop Control (~10 cycles)
- Compute store addresses (packed ALU)
- Execute vstores (2 per cycle max)
- Loop counter management and conditional jump

## Loop Body Analysis

| Phase | Cycles | Description |
|-------|--------|-------------|
| Load + addresses | 6 | All 4 desks loaded, addresses computed |
| Gather desk0 | 4 | 8 lanes at 2 loads/cycle |
| Interleaved gather+hash | 16 | Desks 1-3 gather overlapped with hash stages |
| Hash epilog | 8 | Drain pipeline for desks 2-3 |
| Store | 6 | Addresses + 4 vstores |
| Loop control | 5 | Counter, compare, branch |
| **Total** | **~45** | Per iteration (approximate) |

Actual measured: 6,203 cycles / 128 iterations = **48.5 cycles per iteration**

## Why Not Yet Optimal?

T3 proved 34 cycles is achievable, we're at ~48.5. The remaining gap is due to:

1. **Sequential vselect**: Flow engine limited to 1 op/cycle, so 4 vselects = 4 cycles minimum
2. **Gather serialization**: Gathers must complete before XOR can start for each desk
3. **Store phase**: Can't fully overlap stores with next iteration's loads
4. **Pipeline fill/drain**: Some cycles at edges aren't fully packed

### Potential Further Improvements

1. **Software pipelining across iterations**: Start next batch's loads while storing current batch
2. **8-desk unrolling**: More independent operations to schedule
3. **Register pressure optimization**: Better allocation to reduce conflicts

## Files

- `perf_takehome_h7.py` - Implementation with aggressive interleaving
- `RESULTS.md` - This document

## Verification

```
$ python3.11 experiments/H7_interleaving/perf_takehome_h7.py --check
forest_height=10, rounds=16, batch_size=256
CYCLES:  6203
Speedup over baseline:  23.816540383685314
Correctness check PASSED! Cycles: 6203
```

## Conclusions

H7 demonstrates that aggressive cross-desk interleaving can yield significant performance gains (22.4% over T6+T5). The approach validates T3's finding that interleaving operations is key to reducing cycles.

However, there's still a gap from the theoretical optimal (34 cycles). Closing this gap would require:
1. Even more aggressive software pipelining (across iterations, not just within)
2. Careful register allocation to avoid conflicts
3. Potentially 8-desk unrolling to provide more scheduling freedom

The current implementation achieves ~48.5 cycles per iteration vs the theoretical 34 cycles, capturing about 50% of the potential improvement identified by T3.
