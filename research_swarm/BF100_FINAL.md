# BF100 Optimization Sweep - Final Report

## Summary

Tested 116 distinct optimization theories on the A1 R10 Branch Skip kernel (baseline: 1,548 cycles).

**Best result: 1,536 cycles (12 cycle improvement, verified correct)**

Winning kernel saved at: `/home/hestiasadmin/projects/original_performance_takehome/experiments/BF100/theory_15_WIN.py`

## All Results

### Category 1: Scheduling Variations (Theories 1-15)

| # | Theory | Cycles | Delta | Correct | Notes |
|---|--------|--------|-------|---------|-------|
| 1 | Reverse desk order in hash stages | 1,548 | 0 | YES | No effect |
| 2 | Interleave loads by lane across desks | 1,700 | -152 | YES | Much worse |
| 3 | Reverse desk order in XOR+branch | 1,548 | 0 | YES | No effect |
| 4 | Reverse desk order in hash | 1,548 | 0 | YES | No effect |
| 5 | Interleave desk order (even/odd) | 1,548 | 0 | YES | No effect |
| 6 | FMA stages first then XOR stages | 1,554 | -6 | NO | Breaks hash dependencies |
| 7 | **Per-desk hash (no interleaving)** | **1,546** | **+2** | **YES** | **WIN: per-desk better** |
| 8 | Split hash phases (0-2 then 3-5) | 1,548 | 0 | YES | No effect |
| 9 | Lane-first load order | 1,700 | -152 | YES | Much worse |
| 10 | Pairs load order | 1,651 | -103 | YES | Worse |
| 11 | Reverse group processing order | 1,552 | -4 | YES | Slightly worse |
| 12 | Interleave group processing | 1,548 | 0 | YES | No effect |
| 13 | Reverse hash + reverse groups | 1,552 | -4 | YES | Slightly worse |
| 14 | Per-desk hash + pairs load | 1,681 | -133 | YES | Worse |
| 15 | **Interleave desk + per-desk hash** | **1,536** | **+12** | **YES** | **BEST WIN** |

### Category 2: Combinations with T15 Winner (Theories 16-30)

| # | Theory | Cycles | Delta | Correct | Notes |
|---|--------|--------|-------|---------|-------|
| 16 | T15 + reverse groups | 1,544 | +4 | YES | WIN |
| 17 | T15 + interleave groups | 1,536 | +12 | YES | WIN (same as T15) |
| 18 | Reverse hash + per_desk | 1,548 | 0 | YES | No effect |
| 19 | Per-desk hash + reverse groups | 1,536 | +12 | YES | WIN |
| 20 | Per-desk hash + interleave groups | 1,546 | +2 | YES | WIN |
| 21 | Split phases + interleave desk | 1,548 | 0 | YES | No effect |
| 22 | Split phases + reverse desk | 1,548 | 0 | YES | No effect |
| 23 | T15 + GROUP_SIZE=3 (w/ remainder) | 1,571 | -23 | YES | Worse |
| 24 | T15 + GROUP_SIZE=2 | 1,555 | -7 | YES | Worse |
| 25 | T15 + GROUP_SIZE=5 (w/ remainder) | 1,661 | -113 | YES | Worse |
| 26 | T15 + GROUP_SIZE=8 | 1,622 | -74 | YES | Worse |
| 27 | T15 + GROUP_SIZE=6 (w/ remainder) | 1,563 | -15 | YES | Worse |
| 28 | T15 + GROUP_SIZE=1 | 1,568 | -20 | YES | Worse |
| 29 | Per-desk + GS3 (w/ remainder) | 1,571 | -23 | YES | Worse |
| 30 | Per-desk + GS2 | 1,555 | -7 | YES | Worse |

### Category 3: Group Size Variations (Theories 31-55)

| # | Theory | Cycles | Delta | Correct | Notes |
|---|--------|--------|-------|---------|-------|
| 31 | GROUP_SIZE=3 (w/ remainder) | 1,571 | -23 | YES | Worse |
| 32 | **GROUP_SIZE=2** | **1,536** | **+12** | **YES** | **WIN** |
| 33 | GROUP_SIZE=5 (w/ remainder) | 1,665 | -117 | YES | Much worse |
| 34 | GROUP_SIZE=6 (w/ remainder) | 1,571 | -23 | YES | Worse |
| 35 | GROUP_SIZE=8 | 1,627 | -79 | YES | Worse |
| 36 | GROUP_SIZE=1 | 1,568 | -20 | YES | Worse |
| 37 | GROUP_SIZE=16 | 1,828 | -280 | YES | Much worse |
| 38 | Interleave desk + GS3 | 1,571 | -23 | YES | Worse |
| 39 | **Interleave desk + GS2** | **1,536** | **+12** | **YES** | **WIN** |
| 40 | Reverse desk + GS3 | 1,571 | -23 | YES | Worse |
| 41 | Per-desk + GS5 | 1,656 | -108 | YES | Worse |
| 42 | Per-desk + GS8 | 1,627 | -79 | YES | Worse |
| 43 | Per-desk + GS6 | 1,556 | -8 | YES | Worse |
| 44 | Per-desk + GS1 | 1,568 | -20 | YES | Worse |
| 45 | Per-desk + GS16 | 1,828 | -280 | YES | Much worse |
| 46 | Split + GS3 | 1,571 | -23 | YES | Worse |
| 47 | **Split + GS2** | **1,536** | **+12** | **YES** | **WIN** |
| 48 | Split + GS5 | 1,665 | -117 | YES | Worse |
| 49 | FMA first + GS3 | 1,455 | +93 | NO | Breaks correctness |
| 50 | FMA first + GS2 | 1,562 | -14 | NO | Breaks correctness |
| 51 | T15 + lane_first load | 1,700 | -152 | YES | Much worse |
| 52 | T15 + pairs load | 1,681 | -133 | YES | Worse |
| 53 | Per-desk + lane_first | 1,700 | -152 | YES | Much worse |
| 54 | Per-desk + pairs | 1,681 | -133 | YES | Worse |
| 55 | Interleave all | 1,651 | -103 | YES | Worse |

### Category 4: Desk Count Variations (Theories 56-65)

All failed correctness because changing desk count without adjusting tile count breaks batch size requirements.

| # | Theory | Cycles | Correct | Notes |
|---|--------|--------|---------|-------|
| 56-65 | Various desk counts (8,12,20,24) | 837-1,236 | NO | Wrong batch size |

### Category 5: Cross-Combinations (Theories 66-100)

| # | Theory | Cycles | Delta | Correct | Notes |
|---|--------|--------|-------|---------|-------|
| 66 | T15 + GS3 + reverse groups | 1,571 | -23 | YES | Worse |
| 67 | T15 + GS3 + interleave groups | 1,571 | -23 | YES | Worse |
| 68 | T15 + GS2 + reverse groups | 1,550 | -2 | YES | Worse |
| 69 | T15 + GS2 + interleave groups | 1,555 | -7 | YES | Worse |
| 70 | Reverse + per_desk + GS3 | 1,571 | -23 | YES | Worse |
| 71 | **Reverse + per_desk + GS2** | **1,537** | **+11** | **YES** | **WIN** |
| 72-80 | Various combos | 1,448-1,656 | various | mixed | Mixed results |
| 81-94 | Desk+group with wrong tile counts | Various | N/A | NO | Invalid |
| 95 | **T15 + GS4 + reverse groups** | **1,544** | **+4** | **YES** | **WIN** |
| 96 | **T15 + GS4 + interleave groups** | **1,536** | **+12** | **YES** | **WIN** |
| 97 | Split + interleave + GS3 | 1,571 | -23 | YES | Worse |
| 98 | **Split + interleave + GS2** | **1,536** | **+12** | **YES** | **WIN** |
| 99 | FMA first + interleave + GS3 | 1,455 | +93 | NO | Incorrect |
| 100 | FMA first + interleave + GS2 | 1,562 | -14 | NO | Incorrect |

### Category 6: Novel/Structural (Theories 101-116)

| # | Theory | Cycles | Delta | Correct | Notes |
|---|--------|--------|-------|---------|-------|
| 101 | Reverse desk in hash (standalone) | 1,548 | 0 | YES | No effect |
| 102 | Critical path scheduler | 1,825 | -277 | YES | Much worse |
| 103 | Pairs-stage hash pattern | 1,536 | +12 | YES | WIN (same floor) |
| 104 | Overlap branch+addr emission | 1,536 | +12 | YES | Same floor |
| 105 | Interleave XOR+branch emission | 1,536 | +12 | YES | Same floor |
| 106 | Branch per desk immediate | 1,536 | +12 | YES | Same floor |
| 107 | Replace vselect with arithmetic | 1,562 | -14 | YES | Worse: +VALU ops |
| 108 | Group pair interleaving | 1,639 | -91 | YES | Worse |
| 109 | 4 tiles of 8 desks | 1,614 | -66 | YES | Worse |
| 110 | 1 tile of 32 desks | N/A | N/A | N/A | Doesn't fit scratch |
| 111 | NUM_PRELOADED=3 | ERROR | N/A | ERROR | Fused rounds need 7 |
| 112 | NUM_PRELOADED=15 | 1,555 | -7 | YES | Worse |
| 113 | Hash pattern: per-desk stride (0,2,1,3) | 1,536 | +12 | YES | Same floor |
| 114 | Hash pattern: per-desk 3,1,0,2 | 1,536 | +12 | YES | Same floor |
| 115 | Hash pattern: per-desk forward | 1,546 | +2 | YES | WIN (small) |
| 116 | Per-desk + interleave + reverse groups | 1,544 | +4 | YES | WIN |

## Top 5 Best Results

| Rank | Theory | Cycles | Improvement | Key Change |
|------|--------|--------|-------------|------------|
| 1 | T15: Interleave desk + per-desk hash | 1,536 | +12 cycles | Hash emission order |
| 1 | T17: T15 + interleave groups | 1,536 | +12 cycles | Same mechanism |
| 1 | T19: Per-desk + reverse groups | 1,536 | +12 cycles | Same mechanism |
| 1 | T32: GROUP_SIZE=2 (baseline hash) | 1,536 | +12 cycles | Group size change |
| 1 | T96: T15 + GS4 + interleave groups | 1,536 | +12 cycles | Combined |

(8 additional configurations also achieve 1,536 -- all equivalent)

## Patterns Observed

### What Works
1. **Per-desk hash emission** is the single most impactful change. Instead of emitting hash stage N for all 4 desks, then stage N+1, emit all 12 stages for desk 0, then all 12 for desk 1, etc. This allows the scheduler to better overlap independent dependency chains.

2. **Non-sequential desk ordering** (interleave, stride) provides a small additional benefit when combined with per-desk hash. The interleave pattern (0,2,1,3 instead of 0,1,2,3) helps the scheduler pack VALU ops more efficiently.

3. **GROUP_SIZE=2** independently achieves the same 1,536 cycles even with the baseline hash emission order. Smaller groups have shorter hash phases, reducing scheduling gaps.

### What Doesn't Work
1. **Changing load emission order** (lane-first, pairs) is always worse by 100-150 cycles. The current desk-first order (all 8 lanes per desk, then next desk) is optimal.

2. **Critical path scheduler** is dramatically worse (+277 cycles). The greedy list scheduler is near-optimal for this workload.

3. **Replacing vselect with arithmetic** adds VALU ops without saving enough flow cycles to compensate.

4. **Group interleaving** (processing rounds from different groups alternately) hurts locality and worsens scheduling.

5. **Larger group sizes** (6, 8, 16) are always worse due to increased scheduling pressure on a single group's hash phase.

6. **Non-divisible group sizes** (3, 5, 6 for 16 desks) create unbalanced remainders that worsen overall scheduling.

### The 1,536 Floor Analysis

The current kernel has:
- **8,507 VALU ops** requiring minimum ceil(8507/6) = **1,418 cycles**
- **2,689 load ops** requiring minimum ceil(2689/2) = **1,345 cycles**
- **65 flow ops** (1 pause + 64 vselect)
- **64 store ops**

The actual 1,536 cycles show **81.2% of cycles achieve full 6 VALU/cycle utilization**. The 118-cycle gap above the VALU minimum comes from:
- Hash dependency chains (12 sequential ops with 3 parallel pairs)
- Load-VALU overlap gaps at tile/group boundaries
- Flow engine serialization (vselect at 1/cycle)

To go below 1,536, one would need to either:
1. **Reduce total VALU ops** (requires algorithmic changes, not just scheduling)
2. **Reduce load count** (e.g., by extending round fusion to cover more levels)
3. **Find a fundamentally different execution structure** (not explored)

## Recommendations

1. **Deploy Theory 15** (1,536 cycles) as the new best kernel. It improves on A1 by 12 cycles (0.8% improvement) with a simple hash emission order change.

2. **Future work** should focus on operation count reduction, not scheduling:
   - Extend round fusion beyond R0+1+2 and R11+12+13 (preload more tree levels)
   - Find algebraic hash optimizations (unlikely -- proven 12-op minimum)
   - Explore custom scheduling heuristics tuned to this specific workload

3. **The target of 1,363 cycles is below the theoretical VALU minimum of 1,418** and cannot be achieved without reducing the total number of VALU operations. The current 8,507 VALU ops appear to be at or near the algebraic minimum for 16 rounds of tree traversal with this hash function.

## Files

- Winning kernel: `/home/hestiasadmin/projects/original_performance_takehome/experiments/BF100/theory_15_WIN.py`
- Test framework: `/home/hestiasadmin/projects/original_performance_takehome/experiments/BF100/run_theories.py`
- Hash pattern sweep: `/home/hestiasadmin/projects/original_performance_takehome/experiments/BF100/sweep_hash_patterns.py`
- Fine-grained sweep: `/home/hestiasadmin/projects/original_performance_takehome/experiments/BF100/sweep_fine.py`
- Structural sweep: `/home/hestiasadmin/projects/original_performance_takehome/experiments/BF100/sweep_structural.py`
