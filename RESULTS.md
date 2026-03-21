# Phase 4: Testing Results - Final Summary

## Progress Summary

| Version | Cycles | Speedup | Status |
|---------|--------|---------|--------|
| Baseline | 147,734 | 1.0x | Starting point |
| H1: VLIW pack hash | 123,158 | 1.2x | ✓ |
| H2: SIMD (8-wide) | 23,117 | 6.4x | ✓ |
| H2+VLIW: Packed SIMD | 17,997 | 8.2x | ✓ |
| H3: Loops + Packed | 17,460 | 8.5x | ✓ Pass threshold 2 |
| 4-desk unroll | 16,185 | 9.1x | ✓ |
| Hash/gather overlap | 13,881 | 10.6x | ✓ |
| Packed branch | 13,497 | 10.9x | ✓ |
| Desk 3 overlap fix | 12,985 | 11.4x | ✓ |
| Branch simplify | 12,473 | 11.8x | ✓ |
| Prologue pack | 11,834 | 12.5x | ✓ |
| XOR+addr pack | 11,450 | 12.9x | ✓ |
| Hash1+addr pack | 11,066 | 13.4x | ✓ |
| Store+XOR pack | 10,938 | 13.5x | ✓ |
| Loop control pack | 10,682 | 13.8x | ✓ |
| Offset precompute | 10,555 | 14.0x | ✓ |
| **Speculative desk0 load** | **9,793** | **15.1x** | ✓ **Best** |

## Thresholds Passed

| Threshold | Target | Our Result | Status |
|-----------|--------|------------|--------|
| Better than baseline | < 147,734 | 9,793 | ✓ PASS |
| Updated starting point | < 18,532 | 9,793 | ✓ PASS |
| Opus 4 (many hours) | < 2,164 | 9,793 | ✗ Need 4.5x more |
| Opus 4.5 casual | < 1,790 | 9,793 | ✗ |
| Opus 4.5 (11hr) | < 1,487 | 9,793 | ✗ |
| Best AI | < 1,363 | 9,793 | ✗ |

## Key Optimizations Implemented

### 1. SIMD Vectorization (6.4x improvement)
- Process 8 batch elements simultaneously with valu/vload/vstore
- Reduced loop iterations from 4096 to 512

### 2. 4-Desk Unroll with Hash/Gather Overlap (~2x improvement)
- Process 4 vector groups (32 elements) per loop iteration
- Overlap desk N's hash stages 2-5 with desk N+1's gather
- Each desk's hash takes 12+ cycles, gather takes 4 cycles - perfect overlap

### 3. Aggressive VLIW Packing (~1.4x improvement)
- Pack XOR (VALU) with address computation (ALU)
- Pack hash finals with gather address computation
- Pack branch operations with store address computation
- Pack vselect with loop control
- Pack store with loop comparison

### 4. Branch Simplification (~1.1x improvement)
- Replace `is_even = (val & 1) == 0; offset = is_even ? 1 : 2`
- With `offset = 1 + (val & 1)` - eliminates comparison and vselect

### 5. Speculative Loading (~1.08x improvement)
- During desk 3's hash stages 2-5, speculatively load desk 0 for next iteration
- Use wrap check from hash stage 1 to select correct batch_offset
- Completely eliminates prologue overhead for iterations 2+

## Architecture Analysis

### Current Structure (82.5 cycles per loop iteration)
```
For each iteration (processes 32 elements):
  Desk 0: XOR + hash + gather(desk1) + branch + store  (~15 cycles)
  Desk 1: XOR + hash + gather(desk2) + branch + store  (~15 cycles)
  Desk 2: XOR + hash + gather(desk3) + branch + store  (~16 cycles)
  Desk 3: XOR + hash + speculative_load(desk0) + branch + store + loop  (~20 cycles)
  First iter only: +6 cycles for prologue
```

### Bottleneck Analysis
- **Gather**: 4 cycles per desk (8 loads, 2/cycle) - fundamental limit
- **Hash**: 12 cycles per desk (6 stages × 2 cycles), partially overlapped with gather
- **Branch**: 5 cycles per desk (could reduce with FMA if available)
- **Store**: 2 cycles per desk

### Why ~2,000 Cycles Isn't Achievable With Current Approach

To reach < 2,164 cycles (16.9 cycles per loop iteration):
- Current: 82.5 cycles for 32 elements = 2.6 cycles/element
- Target: 16.9 cycles for 32 elements = 0.53 cycles/element

The theoretical minimum with 4-desk structure:
- 4 desks × 4 cycles gather = 16 cycles (if perfectly overlapped)
- But gather for desk 0 can't overlap with desk 3's hash (different iterations)
- Plus branch/store overhead

### What Would Be Needed for < 2,000 Cycles

1. **8+ Desk Unroll**: More desks = more overlap opportunities
2. **Multi-Iteration Pipeline**: Overlap gather for iteration N with hash for iteration N-1
3. **Deep Register Management**: Keep 4+ vector groups in flight simultaneously
4. **Restructured Loop**: Top-tested loop to allow packing cond_jump with other ops

The best AI solutions (1,363 cycles ≈ 10.6 cycles/iter) likely use:
- 16+ desks unrolled
- 4-deep pipeline across iterations
- Complete overlap of all gather with hash across iteration boundaries

## Lessons Learned

1. **VLIW Packing Matters**: Each packed operation saves ~128 cycles (once per iteration)
2. **Gather Is The Bottleneck**: 8 random memory accesses at 2/cycle = 4 cycles minimum
3. **Overlap Is Key**: Hash/gather overlap provides the largest improvement
4. **Dependencies Are Tricky**: Write-after-read semantics prevent some desired packings
5. **Deeper Pipeline = More Speedup**: But also more complexity and register pressure

## Final Numbers
- **Baseline**: 147,734 cycles
- **Our Best**: 9,793 cycles
- **Speedup**: 15.1x
- **Tests Passed**: 2/8 thresholds
