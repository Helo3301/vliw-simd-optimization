# Experiment Results Summary

**Original:** 147,734 cycles
**Baseline:** 9,793 cycles (15.1x speedup)
**Target:** < 2,164 cycles (Opus 4 threshold)

## Final Results - All Experiments

| Experiment | Cycles | vs Baseline | Status |
|------------|--------|-------------|--------|
| **C4: Full Combo (H12+H11+H13+H14)** | **4,667** | **+52.3%** | **🏆 BEST** |
| C3: H12+H14 | 4,795 | +51.0% | ✅ Success |
| C2: H12+H13 | 4,859 | +50.4% | ✅ Success |
| C1: H12+H11 | 4,987 | +49.1% | ✅ Success |
| H12: Round Fusion | 5,115 | +47.8% | ✅ Success |
| H14: Address Pipeline | 5,307 | +45.8% | ✅ Success |
| H13: Store Coalescing | 5,435 | +44.5% | ✅ Success |
| H15: Loop Unroll 2x | 5,758 | +41.2% | ✅ Success |
| H11: Branch FMA | 5,819 | +40.6% | ✅ Success |
| H7+H10 Combined | 5,947 | +39.3% | ✅ Success |
| H7: Cross-desk Interleaving | 6,203 | +36.7% | ✅ Success |
| H10: vselect Bypass | 7,611 | +22.3% | ✅ Success |
| T6+T5 Combined | 7,995 | +18.4% | ✅ Success |
| T6: Hash Algebra (FMA) | 8,514 | +13.1% | ✅ Success |
| T5 V4: Warp Specialization | 8,758 | +10.6% | ✅ Success |
| T4: 8-Desk Pipeline | 9,792 | ~0% | ❌ No benefit |
| T1: Modulo Scheduling | 9,858 | -0.7% | ❌ No benefit |
| H9: Level Preloading | 8,002 | -0.1% | ❌ Not viable |
| H8: XOR Optimization | N/A | - | ❌ Not viable |
| T2: Index-Aware Gather | N/A | - | ❌ Not viable |
| H16: Double Buffering | 4,667 | 0% | ❌ Not implemented |
| T3: Constraint Solver | 34 (theory) | 55% suboptimal | 📊 Analysis |

## Optimization Journey

```
147,734 (original)
    ↓ 15.1x speedup (baseline optimizations)
  9,793 (baseline)
    ↓ T6: FMA optimization
  8,514 (-13.1%)
    ↓ + T5: VALU packing
  7,995 (-18.4%)
    ↓ H7: Cross-desk interleaving (BREAKTHROUGH!)
  6,203 (-36.7%)
    ↓ + H10: vselect bypass
  5,947 (-39.3%)
    ↓ H12: Round Fusion (BREAKTHROUGH!)
  5,115 (-47.8%)
    ↓ C4: Full Combo (H12+H11+H13+H14)
  4,667 (-52.3%)  ← CURRENT BEST
```

## Total Speedup

| Metric | Value |
|--------|-------|
| Original → Best | **31.7x speedup** |
| Cycles reduced | 147,734 → 4,667 |
| Improvement | 96.8% |
| Gap to target | 2.16x (need 2,164) |

## Round 4: Combination Testing

### C4: Full Combo (H12+H11+H13+H14) 🏆 NEW BEST
- **Cycles:** 4,667 (448 cycles saved vs H12)
- **Improvement over H12:** 8.76%
- **Techniques combined:**
  - H12: Round fusion (2 rounds without memory writeback)
  - H11: Branch FMA (`multiply_add(idx, 2, 1)`)
  - H13: Store coalescing (overlap stores with ALU work)
  - H14: Address pipelining (pre-compute addresses during VALU work)
- **Key insight:** All four optimizations are orthogonal and compound well

### C3: H12+H14 (Round Fusion + Address Pipeline)
- **Cycles:** 4,795 (320 cycles saved vs H12)
- **Improvement over H12:** 6.26%
- **Why it works:** Address pre-computation fills idle ALU cycles during hash phases

### C2: H12+H13 (Round Fusion + Store Coalescing)
- **Cycles:** 4,859 (256 cycles saved vs H12)
- **Improvement over H12:** 5.0%
- **Why it works:** Fewer stores from fusion + better overlap of remaining stores

### C1: H12+H11 (Round Fusion + Branch FMA)
- **Cycles:** 4,987 (128 cycles saved vs H12)
- **Improvement over H12:** 2.5%
- **Why it works:** 8 branch computations (4 desks × 2 rounds) each save 1 cycle

## Round 3 Experiments (H11-H15)

### H12: Round Fusion (+14.0% vs H7H10)
- **Key insight:** Process 2 consecutive rounds without intermediate memory writeback
- **Implementation:** Keep tree indices in registers, only write back every 2 rounds
- **Benefit:** Eliminates 50% of store/load cycles between rounds
- **Result:** 5,115 cycles (832 cycles saved)

### H14: Parallel Address Pipeline (+10.8% vs H7H10)
- **Key insight:** ALU sits idle while VALU computes hash stages
- **Implementation:** Use ALU to pre-compute next iteration's base addresses
- **Benefit:** Overlaps address calculation with hash computation
- **Result:** 5,307 cycles (640 cycles saved)

### H13: Store Coalescing (+8.6% vs H7H10)
- **Key insight:** Stores happen serially at end of each round
- **Implementation:** Batch all stores and overlap with next iteration's loads
- **Benefit:** Better memory bandwidth utilization
- **Result:** 5,435 cycles (512 cycles saved)

### H15: Loop Unrolling 2x (+3.2% vs H7H10)
- **Key insight:** Loop overhead (bounds check, counter update) adds ~5 cycles/iteration
- **Implementation:** Process 2 iterations per loop body
- **Benefit:** Halves loop control overhead
- **Result:** 5,758 cycles (189 cycles saved)

### H11: Branch FMA (+2.15% vs H7H10)
- **Key insight:** Branch computation `idx = 2*idx + 1 + (val & 1)` uses 3 ops
- **Implementation:** Use `multiply_add(idx, idx, 2, 1)` then add branch bit
- **Benefit:** Reduces branch computation from 3 to 2 cycles
- **Result:** 5,819 cycles (128 cycles saved)

### H16: Cross-Iteration Pipelining (NOT IMPLEMENTED)
- **Hypothesis:** Overlap stores from iteration N with loads for iteration N+1
- **Status:** Failed to implement - too complex
- **Challenges:**
  1. Requires loop peeling (prolog loads first batch, epilog stores last without loading)
  2. Needs 2x register sets (8 desks instead of 4) for double-buffering
  3. Last iteration must not load or must use safe wrapped addresses
  4. Tight coupling of H12/H13/H14 makes restructuring difficult
- **Theoretical savings:** ~256 cycles (4 cycles × 64 iterations)
- **Result:** Reverted to C4 baseline (4,667 cycles)

## What Worked (Rounds 1-2)

### H7: Cross-Desk Interleaving (+36.7%)
- **Key insight:** T3 proved optimal is 34 cycles/iteration vs ~62
- **Implementation:** Interleave operations from ALL 4 desks simultaneously
- **Example:** While desk0 gathers, desk1 hashes stage 2, desk2 hashes stage 4, desk3 stores
- **Result:** Keeps all engines (VALU, Load, ALU) busy every cycle

### H10: vselect Bypass (+4.1% on top of H7)
- **Key insight:** Flow unit limited to 1 op/cycle, VALU has 6 slots
- **Implementation:** Replace `vselect(idx, cond, idx, zero)` with `idx * cond`
- **Math:** When cond=1: idx×1=idx. When cond=0: idx×0=0
- **Result:** Pack 4 vselects into 1 VALU cycle instead of 4 flow cycles

### T6: FMA Optimization (+13.1%)
- **Key insight:** `(val + C) + (val << N) = val * (1 + 2^N) + C`
- **Implementation:** Use `multiply_add` for hash stages 0, 2, 4
- **Result:** Replace 3 ops with 1 FMA instruction

### T5: VALU Packing (+10.6%)
- **Key insight:** VALU utilization was only 0.87/6 slots
- **Implementation:** Pack 6 VALU ops per cycle across desks
- **Result:** Improved utilization to 0.98/cycle

## What Didn't Work

| Experiment | Why It Failed |
|------------|---------------|
| T4: 8-Desk | Linear scaling - 2x desks = 2x time per iteration |
| T1: Modulo | Memory bandwidth bottleneck, not compute |
| H9: Level Preload | Scratch and memory have same latency |
| H8: XOR Opt | No algebraic reduction possible for XOR |
| T2: Index-Aware | Simulator has flat memory model (no cache) |
| H16: Double Buffer | Loop restructuring too complex - needs prolog/epilog pattern |

## Files

```
experiments/
├── RESULTS_SUMMARY.md           # This file
├── C4_full_combo/               # BEST: 4,667 cycles
├── C3_H12_H14/                  # 4,795 cycles
├── C2_H12_H13/                  # 4,859 cycles
├── C1_H12_H11/                  # 4,987 cycles
├── H12_round_fusion/            # 5,115 cycles
├── H14_addr_pipeline/           # 5,307 cycles
├── H13_store_coalesce/          # 5,435 cycles
├── H15_unroll_2x/               # 5,758 cycles
├── H11_branch_fma/              # 5,819 cycles
├── H16_double_buffer/           # 4,667 cycles (NOT IMPLEMENTED - same as C4)
├── H7H10_combined/              # 5,947 cycles
├── H7_interleaving/             # 6,203 cycles
├── H10_vselect_bypass/          # 7,611 cycles
├── combined/                    # T6+T5: 7,995 cycles
├── T1_modulo_scheduling/
├── T2_index_aware/
├── T3_constraint_solver/
├── T4_8desk_pipeline/
├── T5_warp_specialization/
├── T6_hash_algebra/
├── H8_xor_opt/
└── H9_level_preload/
```

## Remaining Gap Analysis

**Current:** 4,667 cycles
**Target:** 2,164 cycles
**Gap:** 2.16x

To reach target would require:
1. Achieving T3's theoretical 34 cycles/iteration (currently ~36)
2. Further reducing memory access overhead
3. Exploiting any remaining ALU/VALU idle slots
4. Possible: vgather instruction (not available)

## Diminishing Returns Analysis

| From | To | Cycles Saved | Effort |
|------|-----|--------------|--------|
| Original → Baseline | 147,734 → 9,793 | 137,941 | Initial optimizations |
| Baseline → H7H10 | 9,793 → 5,947 | 3,846 | Cross-desk interleaving |
| H7H10 → H12 | 5,947 → 5,115 | 832 | Round fusion |
| H12 → C4 | 5,115 → 4,667 | 448 | Full combination |

Each round yields smaller improvements as we approach the architectural limit.
