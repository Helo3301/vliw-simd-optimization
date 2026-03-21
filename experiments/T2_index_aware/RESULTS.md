# T2: Index-Aware Gather Scheduling - Experiment Results

## Summary

| Metric | Value |
|--------|-------|
| Baseline Cycles | 9,793 |
| New Cycles | N/A (no optimization implemented) |
| Speedup | N/A |
| Conclusion | **NOT VIABLE** |

## Hypothesis

Tree traversal indices follow predictable patterns (binary tree structure) that could be exploited for better cache utilization or gather scheduling through:
- Index sorting before gather
- Prefetching
- Clustering awareness

## Analysis Methodology

Created `/experiments/T2_index_aware/analyze_indices.py` which:
1. Simulates index evolution across 16 rounds with batch_size=256
2. Analyzes index distribution patterns
3. Identifies tree level clustering
4. Models cache line behavior
5. Estimates sorting overhead costs

## Key Findings

### 1. Index Pattern Discovery

**Major Finding:** All indices in a given round are at the SAME tree level.

This occurs because:
- All indices start at 0 (root, level 0)
- `idx_next = 2*idx + offset` always moves to the next level
- Tree has height 10, so after round 9, indices wrap to level 0

Round-to-level mapping:
```
Round  0: level  1, memory range [   1,    2]   (2 nodes)
Round  1: level  2, memory range [   3,    6]   (4 nodes)
Round  2: level  3, memory range [   7,   14]   (8 nodes)
...
Round  9: level 10, memory range [1023, 2046]   (1024 nodes)
Round 10: level  0, memory range [   0,    0]   (1 node - wrap!)
Round 11: level  1, memory range [   1,    2]   (pattern repeats)
```

### 2. Index Distribution Within Levels

Within each level, indices are pseudo-randomly distributed due to the hash function:

| Round | Unique Indices | Level Size | Utilization |
|-------|---------------|------------|-------------|
| 0-2   | 2, 4, 8       | 2, 4, 8    | 100% (all nodes hit) |
| 3-5   | 16, 32, 63    | 16, 32, 64 | 98-100% |
| 6-9   | 110, 173, 208, 228 | 128, 256, 512, 1024 | 22-86% |

### 3. Cache Line Analysis (If Cache Were Modeled)

Sorting would significantly reduce cache line transitions:

| Round | Original Transitions | Sorted Transitions | Potential Savings |
|-------|---------------------|-------------------|-------------------|
| 5     | 203                 | 4                 | 199 |
| 7     | 241                 | 16                | 225 |
| 9     | 251                 | 61                | 190 |

### 4. Simulator Memory Model

**CRITICAL FINDING:** The simulator does NOT model cache effects.

From `problem.py`:
```python
def load(self, core, *slot):
    match slot:
        case ("load", dest, addr):
            self.scratch_write[dest] = self.mem[core.scratch[addr]]
```

All memory accesses have identical 1-cycle latency regardless of:
- Access pattern (random vs sequential)
- Cache locality
- Previous accesses

### 5. Cost-Benefit Analysis

**Sorting Overhead:**
- Estimated comparisons: ~2,304 (n log n for n=256)
- Estimated ALU operations: ~9,216
- Estimated cycles: ~768 per round at 12 ALU/cycle
- Total for 16 rounds: ~12,000+ cycles

**Potential Cache Benefit:**
- In real hardware: Significant (fewer cache misses)
- In this simulator: **ZERO** (no cache model)

**Net Result: NEGATIVE** (sorting costs cycles, provides zero benefit)

### 6. Other Optimization Attempts Considered

| Optimization | Why Not Viable |
|--------------|----------------|
| Index sorting | No cache model - no benefit |
| Prefetching | No cache model - no benefit |
| Level preloading | Scratch and memory have same latency |
| vload for gather | Indices are non-contiguous (hash randomizes) |

## Conclusion

**Index-aware gather scheduling is NOT viable for this simulator.**

The simulator's flat memory model with uniform access latency means:
1. All gather patterns perform identically
2. Sorting indices only adds overhead
3. No cache-aware optimizations can help
4. Index clustering knowledge cannot be exploited

The only theoretical benefit would come if the simulator modeled:
- Cache hierarchy with different latencies
- Memory banking/conflicts
- Prefetch instructions

Since none of these are modeled, the current gather implementation (scalar loads, 2 per cycle) is already optimal within the simulator's constraints.

## Recommendation

Move on to other experiments (T1, T3, T4, T5, T6) which focus on:
- Instruction-level parallelism (viable)
- VLIW packing optimization (viable)
- Pipeline scheduling (viable)
- Algebraic optimizations (potentially viable)

These don't rely on cache effects that the simulator doesn't model.

## Files

- `analyze_indices.py` - Analysis script with all findings
- `RESULTS.md` - This document
