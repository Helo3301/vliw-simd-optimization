# H143: Reduce Total Operations Experiment

## Goal
Find algorithmic changes that reduce total ops, thereby reducing cycles.

**Starting point:** H140 baseline at 1,645 cycles (11,908 total slots)
**Target:** 1,579 cycles (need to save 66 cycles)

## Analysis

### Operations Breakdown (from analyze_ops.py)

| Round Type | VALU Ops | Loads | Notes |
|------------|----------|-------|-------|
| Round 0    | 16       | 0     | XOR + hash + branch |
| Round 1    | 18       | 0     | 2-way selection + XOR + hash + branch |
| Round 2    | 23       | 0     | 4-way selection (7 ops) + XOR + hash + branch |
| Gather (3-9,14) | 17  | 8     | addr + XOR + hash + branch |
| Round 10   | 19       | 8     | gather + bounds check |
| Round 11   | 16       | 0     | same as round 0 |
| Round 12   | 18       | 0     | same as round 1 |
| Round 13   | 23       | 0     | same as round 2 |
| Round 15   | 14       | 8     | gather + XOR + hash (no branch) |

**Total per desk (16 rounds):** 283 VALU ops, 80 loads
**Total for 16 desks x 2 tiles:** 9,056 VALU ops, 2,560 loads

### Bottleneck Analysis

- VALU limit: 6 per cycle
- Load limit: 2 per cycle

**Theoretical minimum:**
- VALU bound: 9,056 / 6 = **1,509 cycles**
- Load bound: 2,560 / 2 = **1,280 cycles**

We are VALU-bound, so reducing VALU ops should help.

### Potential Savings Identified

1. **Branch optimization (3 -> 2 ops):** Could save ~75 cycles (448 VALU / 6)
2. **4-way selection (7 -> 5 ops):** Could save ~21 cycles (128 VALU / 6)
3. **Bounds check (2 -> 1 op):** Could save ~5 cycles (32 VALU / 6)

## Experiments Conducted

### H143a: vselect for 4-way selection
**Approach:** Replace 7 VALU ops with 5 VALU + 1 FLOW (vselect) for 4-way node selection in rounds 2/13.

**Result:** 1,618 cycles (-27 from H140)

**Analysis:** The vselect instruction uses a FLOW slot instead of 2 VALU ops. Since FLOW has only 1 slot per cycle but doesn't compete with VALU, this effectively reduces VALU pressure.

### H143b: vselect + larger group size (8)
**Approach:** Combine vselect optimization with larger group size (8 instead of 4).

**Result:** 1,715 cycles (+70 from H140)

**Analysis:** Larger groups create longer dependency chains that hurt scheduling.

### H143c: vselect + smaller group size (2)
**Approach:** Combine vselect optimization with smaller group size (2).

**Result:** 1,626 cycles (-19 from H140)

**Analysis:** Smaller groups also hurt - less opportunity for ILP across desks.

### H143d: vselect + stage-interleaved hash
**Approach:** Instead of emitting all hash stages for one desk then the next, emit all stage-0 for the group, then all stage-1, etc. This allows better overlap across desks.

**Result:** 1,613 cycles (-32 from H140)

**Analysis:** Stage-interleaved emission allows the scheduler to better utilize VALU slots by overlapping operations from different desks.

### H143e: vselect + stage-interleaved + group size 3
**Approach:** Try group size 3 (matching 6 VALU slots / 2 ops per desk).

**Result:** 1,634 cycles (-11 from H140)

**Analysis:** Group size 4 remains optimal.

## Summary of Results

| Experiment | Cycles | Change from H140 | Key Optimization |
|------------|--------|------------------|------------------|
| H140 (baseline) | 1,645 | - | - |
| H143a | 1,618 | -27 | vselect for 4-way |
| H143b | 1,715 | +70 | vselect + group 8 (worse) |
| H143c | 1,626 | -19 | vselect + group 2 |
| **H143d** | **1,613** | **-32** | **vselect + stage-interleaved** |
| H143e | 1,634 | -11 | vselect + stage-interleaved + group 3 |

## Best Result: H143d

**Cycles:** 1,613 (32 cycles saved from H140)
**Total slots:** 11,844 (64 fewer than H140)

### Optimizations in H143d:
1. **vselect for 4-way selection:** Saves 2 VALU per 4-way round (rounds 2/13)
2. **Stage-interleaved hash emission:** Better ILP by emitting same hash stage across group before proceeding to next stage

## Remaining Gap

**Target:** 1,579 cycles
**Achieved:** 1,613 cycles
**Gap:** 34 cycles

### Ideas Not Yet Explored:
1. **Branch optimization:** The 3-op branch (AND, FMA, ADD) is hard to reduce without new instructions
2. **Hash stage merging:** Algebraic simplification doesn't help due to dependency chains
3. **Better scheduling algorithm:** Current greedy scheduler may miss global optima
4. **More aggressive precomputation:** Already precomputing tree differences
5. **Different tile/desk organization:** Would require architectural changes

### Theoretical Limit Analysis:
With 11,844 slots and 6 VALU per cycle + 2 load per cycle, the theoretical minimum is around 1,509 cycles (VALU-bound). Our best at 1,613 shows the scheduler is achieving about 93% of theoretical VALU utilization (9,056 / (1,613 * 6) = 93.6%).

Further gains would require either:
- Reducing total VALU ops (algorithmic changes)
- Better scheduling (compiler improvements)
- Overlapping with other engines (already near saturation)
