# BF100 Optimization Sweep Progress

**Baseline**: 1,548 cycles (A1 R10 Branch Skip)
**Best Found**: 1,536 cycles (Theory 15 and many equivalent variants)
**Target**: < 1,363 cycles (likely below theoretical VALU minimum of 1,418)

## Results Table (Theories 1-40)

| # | Theory | Cycles | vs A1 | Correct? | Notes |
|---|--------|--------|-------|----------|-------|
| 1 | Reverse desk order in hash stages | 1,548 | 0 | YES | No change |
| 2 | Interleave loads by lane across desks | 1,700 | -152 | YES | WORSE - delays vector readiness |
| 3 | Reverse desk order in XOR+branch | 1,548 | 0 | YES | No change |
| 4 | Reverse desk order in hash | 1,548 | 0 | YES | No change |
| 5 | Interleave desk order (even/odd) | 1,548 | 0 | YES | No change |
| 6 | FMA stages first, then XOR stages | 1,554 | -6 | NO | Breaks deps |
| 7 | All hash per desk (no interleaving) | 1,546 | +2 | YES | WIN |
| 8 | Split hash phases (0-2 then 3-5) | 1,548 | 0 | YES | No change |
| 9 | Lane-first load order | 1,700 | -152 | YES | WORSE |
| 10 | Pairs load order | 1,651 | -103 | YES | WORSE |
| 11 | Reverse group processing order | 1,552 | -4 | YES | WORSE |
| 12 | Interleave group processing | 1,548 | 0 | YES | No change |
| 13 | Reverse hash + reverse groups | 1,552 | -4 | YES | WORSE |
| 14 | Per-desk hash + pairs load | 1,681 | -133 | YES | WORSE |
| 15 | **Interleave desk + per-desk hash** | **1,536** | **+12** | **YES** | **BEST WIN** |
| 16 | T15 + reverse groups | 1,544 | +4 | YES | WIN |
| 17 | T15 + interleave groups | 1,536 | +12 | YES | WIN |
| 18 | Reverse hash + per_desk | 1,548 | 0 | YES | No change |
| 19 | Per-desk hash + reverse groups | 1,536 | +12 | YES | WIN |
| 20 | Per-desk hash + interleave groups | 1,546 | +2 | YES | WIN |
| 21 | Split phases + interleave desk | 1,548 | 0 | YES | No change |
| 22 | Split phases + reverse desk | 1,548 | 0 | YES | No change |
| 23 | T15 + GROUP_SIZE=3 (with remainder) | 1,571 | -23 | YES | WORSE |
| 24 | T15 + GROUP_SIZE=2 | 1,555 | -7 | YES | WORSE |
| 25 | T15 + GROUP_SIZE=5 (with remainder) | 1,661 | -113 | YES | WORSE |
| 26 | T15 + GROUP_SIZE=8 | 1,622 | -74 | YES | WORSE |
| 27 | T15 + GROUP_SIZE=6 (with remainder) | 1,563 | -15 | YES | WORSE |
| 28 | T15 + GROUP_SIZE=1 | 1,568 | -20 | YES | WORSE |
| 29 | Per-desk + GS3 (with remainder) | 1,571 | -23 | YES | WORSE |
| 30 | Per-desk + GS2 | 1,555 | -7 | YES | WORSE |
| 31 | GS3 (with remainder) | 1,571 | -23 | YES | WORSE |
| 32 | GROUP_SIZE=2 | 1,536 | +12 | YES | WIN |
| 33 | GS5 (with remainder) | 1,665 | -117 | YES | WORSE |
| 34 | GS6 (with remainder) | 1,571 | -23 | YES | WORSE |
| 35 | GROUP_SIZE=8 | 1,627 | -79 | YES | WORSE |
| 36 | GROUP_SIZE=1 | 1,568 | -20 | YES | WORSE |
| 37 | GROUP_SIZE=16 | 1,828 | -280 | YES | WORSE |
| 38 | Interleave desk + GS3 | 1,571 | -23 | YES | WORSE (with remainder) |
| 39 | Interleave desk + GS2 | 1,536 | +12 | YES | WIN |
| 40 | Reverse desk + GS3 | 1,571 | -23 | YES | WORSE |

## Results Table (Theories 41-80)

| # | Theory | Cycles | vs A1 | Correct? | Notes |
|---|--------|--------|-------|----------|-------|
| 41 | Per-desk + GS5 | 1,656 | -108 | YES | WORSE |
| 42 | Per-desk + GS8 | 1,627 | -79 | YES | WORSE |
| 43 | Per-desk + GS6 | 1,556 | -8 | YES | WORSE |
| 44 | Per-desk + GS1 | 1,568 | -20 | YES | WORSE |
| 45 | Per-desk + GS16 | 1,828 | -280 | YES | WORSE |
| 46 | Split + GS3 | 1,571 | -23 | YES | WORSE |
| 47 | Split + GS2 | 1,536 | +12 | YES | WIN |
| 48 | Split + GS5 | 1,665 | -117 | YES | WORSE |
| 49 | FMA first + GS3 | 1,455 | +93 | NO | Incorrect |
| 50 | FMA first + GS2 | 1,562 | -14 | NO | Incorrect |
| 51 | T15 + lane_first load | 1,700 | -152 | YES | WORSE |
| 52 | T15 + pairs load | 1,681 | -133 | YES | WORSE |
| 53 | Per-desk + lane_first | 1,700 | -152 | YES | WORSE |
| 54 | Per-desk + pairs | 1,681 | -133 | YES | WORSE |
| 55 | Interleave all | 1,651 | -103 | YES | WORSE |
| 56-65 | Desk count variations (8,12,20,24) | Various | N/A | NO | Invalid: wrong batch size |
| 66 | T15 + GS3 + reverse groups | 1,571 | -23 | YES | WORSE |
| 67 | T15 + GS3 + interleave groups | 1,571 | -23 | YES | WORSE |
| 68 | T15 + GS2 + reverse groups | 1,550 | -2 | YES | WORSE |
| 69 | T15 + GS2 + interleave groups | 1,555 | -7 | YES | WORSE |
| 70 | Reverse + per_desk + GS3 | 1,571 | -23 | YES | WORSE |
| 71 | Reverse + per_desk + GS2 | 1,537 | +11 | YES | WIN |
| 72 | T15 + GS5 + reverse | 1,661 | -113 | YES | WORSE |
| 73 | T15 + GS5 + interleave | 1,661 | -113 | YES | WORSE |
| 74-80 | Various combos | 1,537-1,656 | Various | YES | Mixed results |

## Results Table (Theories 81-116)

| # | Theory | Cycles | vs A1 | Correct? | Notes |
|---|--------|--------|-------|----------|-------|
| 81-94 | Desk+group combos with 8/12/20 desks | Various | N/A | NO | Invalid batch sizes |
| 95 | T15 + GS4 + reverse groups | 1,544 | +4 | YES | WIN |
| 96 | T15 + GS4 + interleave groups | 1,536 | +12 | YES | WIN |
| 97 | Split + interleave + GS3 | 1,571 | -23 | YES | WORSE |
| 98 | Split + interleave + GS2 | 1,536 | +12 | YES | WIN |
| 99 | FMA first + interleave + GS3 | 1,455 | +93 | NO | Incorrect |
| 100 | FMA first + interleave + GS2 | 1,562 | -14 | NO | Incorrect |
| 101 | Reverse desk in hash (standalone) | 1,548 | 0 | YES | Same as T4 |
| 102 | Critical path scheduler | 1,825 | -277 | YES | MUCH WORSE |
| 103 | Pairs-stage hash | 1,536 | +12 | YES | Same as T15 |
| 104-106 | Structural variants | 1,536 | +12 | YES | All hit 1,536 floor |
| 107 | vselect replaced with arithmetic | 1,562 | -14 | YES | WORSE: +128 VALU |
| 108 | Group pair interleaving | 1,639 | -91 | YES | WORSE |
| 109 | 4 tiles of 8 desks | 1,614 | -66 | YES | WORSE |
| 110 | 1 tile of 32 desks | N/A | N/A | N/A | Won't fit in scratch |
| 111 | NUM_PRELOADED=3 | ERROR | N/A | ERR | Fused rounds need 7 |
| 112 | NUM_PRELOADED=15 | 1,555 | -7 | YES | WORSE |
| 116 | Per-desk + interleave desk + reverse groups | 1,544 | +4 | YES | WIN |

## Progress Checkpoint (after 100+ theories)

**Best result: 1,536 cycles** achieved by many equivalent configurations.
All require either:
- Per-desk hash emission (all 12 stages per desk before moving to next desk)
- Or pairs/alternate hash emission (small groups of desks)
- AND some form of desk reordering (interleave, stride, non-sequential)

The 1,536 floor appears to be the scheduling minimum for the current operation count (8,507 VALU ops).
