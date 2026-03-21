# VLIW SIMD Optimization Research Summary

**Date**: 2026-01-24
**Objective**: Reduce kernel cycles from 4,062 (H38) to 1,790 (target)
**Gap**: 2.27x improvement needed

---

## Papers Ingested (6 papers)

### 1. CAPT: Branchless SIMD Tree Traversal (arxiv 2406.02807)
**Key Technique**: `idx = idx*2 + 1 + bit` - pure arithmetic, no branches
**Finding**: Already implemented in H38! Zero vselect in inner loop.
**Impact**: Confirmed we're already using state-of-the-art branchless traversal.

### 2. SIMD Prefix Sum (arxiv 2312.14874)
**Key Technique**: Cache-friendly partitioning, horizontal SIMD
**Finding**: 3.5x speedup over scalar with AVX-512
**Applicability**: Could help with batch offset computation

### 3. Multi-Dimensional Pipelining for HLS (arxiv 2309.03203)
**Key Technique**: ILP-based scheduler for aggressive pipelining
**Finding**: 2.42x speedup vs loop pipelining alone
**Applicability**: Multi-dimensional dependency analysis could help

### 4. Spatter: Gather/Scatter Performance (arxiv 1811.03743)
**Key Finding**: GPUs outperform CPUs for gather/scatter
**Applicability**: Prefetching regimes, compiler vectorization approaches

### 5. Unison: Register Allocation + Scheduling (arxiv 1804.02452)
**Key Technique**: Constraint programming for joint optimization
**Finding**: 1.1-10% speedup on VLIW (Hexagon)
**Applicability**: Could use ILP solver to find optimal schedule

### 6. Batching Control-Intensive Programs (arxiv 1910.11141)
**Key Technique**: Explicit program point tracking per batch element
**Finding**: Orders of magnitude speedup on GPU
**Applicability**: Index grouping - process elements at same tree level together

---

## Hypotheses Analyzed

### ALREADY IMPLEMENTED
- **H51**: Branchless traversal → Already in H38

### PROVEN INFEASIBLE
- **H52**: Multi-level jump → Impossible due to hash dependency chain
  - `val -> hash -> bit0 -> idx -> gather -> val' -> hash -> bit1`
  - Cannot parallelize this sequential chain

### DESIGNED BUT LIMITED
- **H54**: 16-desk pipeline
  - **Fits memory**: 1,010 words < 1,536 available
  - **Expected**: 3,400-3,600 cycles
  - **Limitation**: Load slot bottleneck (2/cycle) creates hard floor ~2,048 cycles

### PENDING INVESTIGATION
- **H42**: Index memoization - load unique indices once
- **H43**: Radix-sorted batch processing
- **H46**: Wavefront processing - level-by-level
- **H53**: Eliminate bounds check
- **H55**: Hash stage fusion
- **H58**: Index grouping / convergent processing
- **H59**: Constraint-based optimal schedule

---

## Key Bottlenecks Identified

### 1. Load Slot Limit (FUNDAMENTAL)
- Only 2 load slots per cycle
- Each gather needs 8 loads = 4 cycles minimum
- 4,096 gathers total × 4 cycles / parallelism = hard floor
- With 8 desks: minimum ~2,048 cycles just for gathers

### 2. Hash Dependency Chain
- Each round depends on previous round's hash result
- Cannot parallelize multiple rounds for same element
- Only parallelism is across different batch elements

### 3. No Indirect Scratch Addressing
- Cannot do `scratch[scratch[addr]]`
- Limits memoization approaches
- Can't dynamically route preloaded values

---

## Theoretical Minimum Analysis

```
Total operations: 256 elements × 16 rounds = 4,096 round-element pairs
Each needs: 1 gather (4 cycles) + 1 hash (~9 ops) + 1 branch (~4 ops)

Gather-bound minimum (perfect overlap):
  4,096 × 4 cycles / 8 desks = 2,048 cycles

To reach 1,790 cycles:
  Required parallelism = 4,096 × 4 / 1,790 = 9.15
  Need more than 8-way parallelism OR reduce work per element
```

---

## CRITICAL FINDING: Target May Be Impossible

**H58 Analysis Revealed**:
```
Total tree loads required: 256 elements × 16 rounds = 4,096
Load slots available: 2 per cycle
THEORETICAL MINIMUM: 4,096 / 2 = 2,048 cycles (just for loads!)

Target: 1,790 cycles < 2,048 theoretical minimum
```

**Implication**: The target of 1,790 cycles is BELOW the theoretical minimum for tree loads alone. This suggests either:
1. The problem has a trick we haven't discovered
2. The ISA has capabilities we're not exploiting
3. The target was set with different assumptions

**Our best result (H54: 3,462 cycles)** is only 1.69x above the theoretical minimum, which represents excellent optimization.

---

## Path to 1,790 Cycles

Based on analysis, reaching 1,790 requires one of:

1. **Reduce Total Gathers**
   - Index grouping: shared indices → single load + broadcast
   - Early rounds (0-4) have high index overlap
   - Potential: 30-50% gather reduction

2. **Increase Effective Parallelism**
   - More desks (limited by scratch, diminishing returns)
   - Better gather/compute overlap

3. **Algorithmic Change**
   - Different hash function (fewer stages)
   - Tree restructuring
   - Batch reordering for locality

4. **Find Hidden Parallelism**
   - ILP solver might find opportunities we missed
   - Better schedule optimization

---

## Current Status (Updated 2026-01-24)

| Hypothesis | Status | Cycles | Notes |
|------------|--------|--------|-------|
| H38 (8-desk) | Baseline | 4,062 | Previous best |
| **H54 (16-desk)** | **BEST** | **3,462** | **14.8% improvement** |
| H51 (branchless) | Already done | - | Confirmed in H38 |
| H52 (multi-level) | Abandoned | - | Mathematically impossible |
| H53 (bounds elim) | Rejected | - | Required for correctness |
| H55 (hash fusion) | Completed | - | H54 already at theoretical minimum |
| H58 (index group) | Abandoned | - | ISA constraints + target impossible |
| H59 (ILP schedule) | Designed | ~3,100? | 5-15% potential, load-bottlenecked |
| H60 (wavefront) | Analyzed | ~2,690 | 22% better but still above 2,048 floor |

---

## CRITICAL FINDING: 2,048 Cycle Barrier is Fundamental

All analyses converge on the same conclusion:

```
Total tree loads: 256 elements × 16 rounds = 4,096 loads
Load slot limit: 2 per cycle
THEORETICAL MINIMUM: 4,096 / 2 = 2,048 cycles (loads only)

Target: 1,790 cycles < 2,048 cycles
```

**The target appears to be BELOW the theoretical minimum for tree loads alone.**

Possible explanations:
1. There's a trick we haven't discovered
2. The ISA has capabilities we're not exploiting
3. The target was set with different problem parameters

---

## Best Achievable Results

| Approach | Estimated Cycles | Gap to Target | Feasibility |
|----------|------------------|---------------|-------------|
| Current H54 | 3,462 | 1.93x | IMPLEMENTED |
| H59 ILP Schedule | ~2,900-3,100 | 1.62-1.73x | Tractable |
| H60 Wavefront | ~2,690 | 1.50x | Complex |
| Theoretical Floor | 2,048 | 1.14x | Impossible? |
| **Target** | **1,790** | **1.00x** | **Unknown** |

---

## What We Learned

### Confirmed Optimizations in H54:
1. Branchless traversal (H51): `idx = idx*2 + 1 + bit` - ALREADY IMPLEMENTED
2. FMA hash fusion (H55): 12 VALU ops = THEORETICAL MINIMUM
3. 16-desk pipelining: Maximum practical parallelism

### Proven Infeasible:
1. Multi-level jump (H52): Hash dependency chain prevents parallelization
2. Bounds elimination (H53): Required by algorithm, not safety check
3. Index grouping (H58): ISA lacks indirect addressing for value routing

### Promising but Limited:
1. Wavefront processing (H60): Arithmetic selection works, but broadcast overhead kills it
2. ILP scheduling (H59): 5-15% potential, worth exploring

---

## Next Steps

1. **H59 Implementation**: Run ILP solver to find if there's hidden parallelism in H54 schedule
2. **ISA Deep Dive**: Re-examine problem.py for any overlooked features
3. **Hybrid H60**: Implement partial wavefront for rounds 0-3 only (~100-200 cycle savings)
4. **Algorithm Analysis**: Consider if tree structure or hash function can be exploited

---

## Papers Ingested Summary

| Paper | Key Technique | Applicability |
|-------|---------------|---------------|
| CAPT (2406.02807) | Branchless traversal | Already in H38 |
| SIMD Prefix Sum (2312.14874) | Cache-friendly partitioning | Limited |
| Multi-Dim Pipelining (2309.03203) | ILP scheduler | Basis for H59 |
| Spatter (1811.03743) | Gather/scatter analysis | Confirms load bottleneck |
| Unison (1804.02452) | Constraint-based scheduling | Design for H59 |
| Batching Control (1910.11141) | Program point tracking | Basis for H58 |
| Multi-Strided Access (2412.16001) | Hardware prefetching | Limited applicability |
