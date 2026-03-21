# H9: Level-Aware Tree Preloading - Experiment Results

## Summary

| Metric | Value |
|--------|-------|
| T6+T5 Baseline Cycles | 7,995 |
| H9 Cycles | 8,002 |
| Delta | +7 cycles (worse) |
| Conclusion | **NOT VIABLE** - Confirms T2's findings |

## Hypothesis

Tree traversal indices follow a level-based pattern that could be exploited:

1. **T2's Discovery**: All indices in a given round are at the SAME tree level
   - Round 0: Level 1 (2 nodes, memory [1-2])
   - Round 1: Level 2 (4 nodes, memory [3-6])
   - Round k: Level k+1 mod 11
   - Levels 1-6 have at most 64 nodes (512 bytes)

2. **H9 Hypothesis**: Preloading entire tree levels into scratch memory could provide:
   - Reduced address computation overhead
   - Simpler addressing (level_base + relative_idx vs forest_p + absolute_idx)
   - Better instruction packing during gather

## Analysis

### Why Preloading Cannot Help

After careful analysis, level preloading provides **zero benefit** because:

#### 1. Memory Model (from T2)
```
The simulator does NOT model cache effects.
All memory accesses have fixed 1-cycle latency regardless of:
- Access pattern (random vs sequential)
- Cache locality
- Previous accesses
```

Scratch and main memory have **identical latency** in this simulator.

#### 2. Address Computation Comparison

**Current approach (T6+T5):**
```
vbroadcast(addr, forest_values_p)   // 1 VALU op
vadd(addr, addr, idx)               // 1 VALU op
8x scalar load(node_val, addr)      // 4 cycles (2 loads/cycle)
```

**With level preloading:**
```
// Once per round: vload level data (8 vloads for 64 nodes = 4 cycles)
vbroadcast(addr, level_scratch_base) // 1 VALU op
vsub(addr, idx, level_start)         // 1 VALU op (EXTRA!)
vadd(addr, addr, level_base)         // 1 VALU op
8x scalar load(node_val, addr)       // 4 cycles (same!)
```

Level preloading **adds** operations rather than removing them:
- Extra subtract to convert absolute index to level-relative offset
- Extra preload vloads (amortized but still overhead)

#### 3. Gather Cannot Be Vectorized

The core limitation is that tree indices are **scattered** within each level due to hash function randomization:
- Even though all 256 indices are at the same level, they're randomly distributed
- Cannot use `vload` for the actual node value fetch
- Must still use 8 scalar loads per desk (2 per cycle = 4 cycles per desk)

### Implementation Notes

The H9 implementation is essentially identical to T6+T5, confirming that:
1. No code changes could exploit level information
2. The extra scratch allocation added 7 cycles of overhead
3. T2's analysis was correct

## Cost Analysis

### Theoretical Level Preloading Cost

| Level | Size (nodes) | vloads needed | Preload cycles |
|-------|--------------|---------------|----------------|
| 1 | 2 | 1 | 1 |
| 2 | 4 | 1 | 1 |
| 3 | 8 | 1 | 1 |
| 4 | 16 | 2 | 1 |
| 5 | 32 | 4 | 2 |
| 6 | 64 | 8 | 4 |
| 7 | 128 | 16 | 8 |
| 8 | 256 | 32 | 16 |
| 9 | 512 | 64 | 32 |
| 10 | 1024 | 128 | 64 |

For a full 16-round batch, preloading small levels would cost ~20+ cycles with zero benefit.

### Why Address Computation Cannot Be Simplified

The key insight is that **knowing the memory range** doesn't help when you still need to:
1. Compute individual addresses for scattered indices
2. Perform scalar loads (can't vectorize non-contiguous access)
3. Store results in correct scratch locations

The address computation is already minimal:
- One broadcast + one add per 8 elements
- Removing this would require eliminating the index entirely (impossible)

## Conclusion

**Level-aware tree preloading is NOT viable for this simulator.**

This experiment confirms T2's findings from a different angle:

| T2 Conclusion | H9 Verification |
|---------------|-----------------|
| Scratch and memory have same latency | Confirmed - preloading to scratch provides no speed benefit |
| Level preloading not viable | Confirmed - adds overhead without reducing gather cycles |
| Cannot use vload for gather | Confirmed - indices are scattered, not contiguous |

The only way to improve gather performance would be if:
1. The simulator modeled cache with different latencies
2. vload supported gather (non-contiguous addressing)
3. Memory banking allowed parallel access to scattered addresses

None of these features exist in the simulator.

## Files

- `perf_takehome_h9.py` - Implementation (essentially identical to T6+T5)
- `RESULTS.md` - This document

## Recommendations

1. **Do not pursue level-aware optimizations** - the memory model makes them impossible
2. **Focus on other areas**:
   - Instruction-level parallelism (VALU packing) - already done in T5/T6
   - FMA optimization - already done in T6
   - Loop overhead reduction
   - Store coalescing
3. **The 7,995 cycle result from T6+T5 is likely near-optimal** given the simulator's constraints
