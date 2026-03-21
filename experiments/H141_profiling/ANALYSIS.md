# H140 VLIW SIMD Kernel Profiling Analysis

## Executive Summary

The H140 kernel achieves **1,645 cycles** (89.8x speedup over baseline of 147,734 cycles).
To reach the target of **1,579 cycles**, we need to save **66 cycles**.

**Key Finding:** The kernel is **VALU-bound** with 92% VALU utilization. The theoretical
minimum based on VALU operations is 1,513.8 cycles, meaning there is ~131 cycles of
overhead (8.7%) from dependencies and scheduling inefficiencies.

---

## 1. Operation Counts by Engine

| Engine | Total Ops | Limit/Cycle | Theoretical Min Cycles |
|--------|-----------|-------------|------------------------|
| **VALU** | 9,083 | 6 | **1,513.8** |
| LOAD | 2,689 | 2 | 1,344.5 |
| STORE | 64 | 2 | 32.0 |
| ALU | 71 | 12 | 5.9 |
| FLOW | 1 | 1 | 1.0 |

### VALU Operation Breakdown
```
^         : 3,072 (33.8%)  - XOR operations
multiply_add: 2,272 (25.0%)  - FMA operations
+         : 1,312 (14.4%)  - Addition
>>        : 1,088 (12.0%)  - Right shift
&         :   544 ( 6.0%)  - AND (branch/selection)
<<        :   512 ( 5.6%)  - Left shift
-         :   195 ( 2.1%)  - Subtraction (selection)
<         :    32 ( 0.4%)  - Comparison (bounds)
*         :    32 ( 0.4%)  - Multiply (bounds)
vbroadcast:    24 ( 0.3%)  - Setup broadcasts
```

### LOAD Operation Breakdown
```
load      : 2,571 (95.6%)  - Scalar loads (gathers)
vload     :    64 ( 2.4%)  - Vector loads
const     :    54 ( 2.0%)  - Constant loads
```

---

## 2. Theoretical Bounds Analysis

```
VALU bound:  9,083 / 6  = 1,513.8 cycles  <-- BOTTLENECK
LOAD bound:  2,689 / 2  = 1,344.5 cycles
STORE bound:    64 / 2  =    32.0 cycles
ALU bound:      71 / 12 =     5.9 cycles
FLOW bound:       1 / 1 =     1.0 cycles
```

**Bottleneck:** VALU is the limiting factor at 1,513.8 theoretical minimum cycles.

**Overhead:** Actual 1,645 cycles - Theoretical 1,513.8 cycles = **131.2 cycles overhead (8.7%)**

This overhead comes from:
1. Data dependencies preventing full parallel scheduling
2. Phase boundary (pause) overhead
3. Setup operations that can't be parallelized

---

## 3. Slot Utilization Analysis

| Engine | Slots Used | Max Possible | Utilization | Avg/Cycle |
|--------|------------|--------------|-------------|-----------|
| **VALU** | 9,083 | 9,870 | **92.0%** | 5.52/6 |
| LOAD | 2,689 | 3,290 | 81.7% | 1.63/2 |
| STORE | 64 | 3,290 | 1.9% | 0.04/2 |
| ALU | 71 | 19,740 | 0.4% | 0.04/12 |
| FLOW | 2 | 1,645 | 0.1% | 0.00/1 |

**Total:** 11,909 slots used out of 37,835 possible (31.5% overall utilization)

### Idle Slots Available
```
VALU:    787 idle slots (0.5/cycle avg)
LOAD:    601 idle slots (0.4/cycle avg)
STORE: 3,226 idle slots (2.0/cycle avg) - completely unused
ALU:  19,669 idle slots (12.0/cycle avg) - nearly unused
FLOW:  1,643 idle slots (1.0/cycle avg) - nearly unused
```

**Key Insight:** STORE and ALU engines are almost completely idle. If work could be
moved from VALU to ALU (for scalar operations) or STORE could be used for caching,
there might be opportunities.

---

## 4. Cycle Distribution Analysis

### VALU Saturation
```
6 slots (100%): 1,316 cycles (80.0%)  <-- Most cycles are VALU-saturated
5 slots ( 83%):    67 cycles ( 4.1%)
4 slots ( 67%):   147 cycles ( 8.9%)
3 slots ( 50%):    74 cycles ( 4.5%)
0-2 slots:         41 cycles ( 2.5%)
```

**80% of cycles are running at full VALU capacity (6/6 slots used).**

### LOAD Saturation
```
2 slots (100%): 1,343 cycles (81.6%)
1 slot  ( 50%):     3 cycles ( 0.2%)
0 slots (  0%):   299 cycles (18.2%)
```

### Combined Distribution
```
VALU=6, LOAD=2: 1,055 cycles (64.1%)  <-- Dominant pattern
VALU=6, LOAD=0:   261 cycles (15.9%)
VALU=4, LOAD=2:   142 cycles ( 8.6%)
VALU=3, LOAD=2:    69 cycles ( 4.2%)
```

---

## 5. Critical Path Analysis

### Phase Structure
The kernel has **2 phases** separated by pause:
- **Phase 0 (Setup):** Cycles 0-20 (21 cycles)
- **Phase 1 (Main computation):** Cycles 21-1644 (1,624 cycles)

### VALU Saturation Runs
The longest continuous VALU saturation run spans **1,216 cycles** (cycles 33-1248),
representing the main computational loop through rounds 0-13 approximately.

### Dependency Stalls
Only **40 cycles** show potential dependency stalls (VALU<6, LOAD<2, total<6).
Most of these are in:
- Setup phase (cycles 15-20)
- Final rounds (cycles 1562-1587)

This indicates the scheduler is doing a good job packing operations, but there
are structural dependencies that prevent full utilization.

---

## 6. Per-Round Operation Analysis

### VALU Operations per Round (per desk)
```
Round  0: 16 ops  (tree[0] XOR, hash, branch)
Round  1: 18 ops  (2-way selection, XOR, hash, branch)
Round  2: 24 ops  (4-way selection, XOR, hash, branch)
Rounds 3-9: 17 ops each  (gather addr, XOR, hash, branch)
Round 10: 19 ops  (gather, hash, branch, bounds)
Round 11: 16 ops  (tree[0] XOR, hash, branch)
Round 12: 18 ops  (2-way selection, XOR, hash, branch)
Round 13: 24 ops  (4-way selection, XOR, hash, branch)
Round 14: 17 ops  (gather, XOR, hash, branch)
Round 15: 14 ops  (gather, XOR, hash - NO branch)
```

**Total per desk:** 285 VALU ops
**Total for 32 desks:** 9,120 VALU ops (matches ~9,083 observed with setup)

### Hash Computation Breakdown (12 VALU ops per hash)
```
Stage 0 (FMA): val = (val + C) + (val << 12)  -> 1 multiply_add
Stage 1:       val = (val ^ C) ^ (val >> 19)  -> 3 ops (^, >>, ^)
Stage 2 (FMA): val = (val + C) + (val << 5)   -> 1 multiply_add
Stage 3:       val = (val + C) ^ (val << 9)   -> 3 ops (+, <<, ^)
Stage 4 (FMA): val = (val + C) + (val << 3)   -> 1 multiply_add
Stage 5:       val = (val ^ C) ^ (val >> 16)  -> 3 ops (^, >>, ^)
```

### Branch Computation (3 VALU ops)
```
idx = idx * 2 + 1 + (val & 1)
Operations: multiply_add, &, +
```

---

## 7. Optimization Recommendations

### Target: Save 66 cycles (~396 VALU ops)

### Option A: Reduce Hash Stage Operations
**Impact: ~85 cycles saved per VALU op eliminated**

The non-FMA hash stages (1, 3, 5) each use 3 VALU ops:
```
Stage 1: tmp1 = val ^ C; tmp2 = val >> 19; val = tmp1 ^ tmp2
Stage 3: tmp1 = val + C; tmp2 = val << 9;  val = tmp1 ^ tmp2
Stage 5: tmp1 = val ^ C; tmp2 = val >> 16; val = tmp1 ^ tmp2
```

If we could fuse or eliminate 1 op per non-FMA stage:
- 1 op saved * 3 stages * 32 desks * 16 rounds = 1,536 VALU ops saved
- **~256 cycles saved** (more than needed!)

However, the hash function is fixed, so this may not be possible without
breaking correctness.

### Option B: Optimize 4-Way Selection (Rounds 2, 13)
**Impact: ~21 cycles per 2 ops saved**

Current 4-way selection uses 8 VALU ops. If reduced to 6:
- 2 ops saved * 32 desks * 4 uses = 256 VALU ops saved
- **~42 cycles saved**

Potential approach: Use lookup table or different arithmetic.

### Option C: Reduce Branch Operations
**Impact: Limited - Round 15 already optimized**

Branch uses 3 ops: multiply_add, &, +. Round 15 already skips branch.
Further savings would require skipping bounds check (risky).

### Option D: Leverage Idle ALU Slots
**Impact: Variable**

The ALU engine has 19,669 idle slots. If scalar operations from VALU setup
could be moved to ALU, this could reduce VALU pressure.

Currently only 71 ALU ops are used (address calculations).

### Option E: Reduce Gather Overhead
**Impact: ~10-20 cycles**

1,293 cycles involve scalar loads (gathers). Each gather round needs:
- 1 VALU op for address calculation: `addr = forest_p + idx`
- 8 scalar loads

If address calculation could be batched or eliminated:
- 1 op * 10 gather rounds * 32 desks = 320 VALU ops
- **~53 cycles saved**

### Option F: Optimize Selection Logic
**Impact: ~30-40 cycles**

2-way selection (rounds 1, 12) uses 2 VALU ops
4-way selection (rounds 2, 13) uses 8 VALU ops

Total selection overhead: (2*2 + 8*2) * 32 = 640 VALU ops = ~107 cycles

If 4-way could be reduced to 6 ops using different approach:
- 2 * 2 * 32 = 128 VALU ops saved = **~21 cycles**

---

## 8. Summary

| Metric | Current | Target | Gap |
|--------|---------|--------|-----|
| Cycles | 1,645 | 1,579 | 66 |
| VALU ops | 9,083 | ~8,687 | ~396 |
| VALU utilization | 92.0% | 91.4% | -0.6% |

**Most Promising Optimizations:**
1. **Reduce gather address calculation** (~53 cycles)
2. **Optimize 4-way selection** (~21 cycles)
3. **Move scalar setup to ALU** (variable)

Combined, these could potentially save 74+ cycles, exceeding the 66-cycle target.

---

## 9. Files Generated

- `/home/hestiasadmin/projects/original_performance_takehome/experiments/H141_profiling/profile_h140.py` - Main profiling script
- `/home/hestiasadmin/projects/original_performance_takehome/experiments/H141_profiling/detailed_analysis.py` - Per-round analysis
- `/home/hestiasadmin/projects/original_performance_takehome/experiments/H141_profiling/schedule_analysis.py` - Schedule pattern analysis
- `/home/hestiasadmin/projects/original_performance_takehome/experiments/H141_profiling/ANALYSIS.md` - This document
