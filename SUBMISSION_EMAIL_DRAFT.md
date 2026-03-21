# Submission Email Draft

**To:** performance-recruiting@anthropic.com
**Subject:** Performance Take-Home: ~1,315 cycles (112x speedup)

---

Hi,

I'm submitting my solution to the original performance take-home. The kernel achieves **~1,311–1,316 cycles** (varies by SA run) — a **112x speedup** over the 147,734-cycle baseline, passing all 9 submission tests including the < 1,363 threshold.

## Validation

```
$ git diff origin/main tests/
(empty — tests unchanged)

$ python3.11 tests/submission_tests.py
Ran 9 tests in 430.455s
OK
Scratch usage: 1360 / 1536
Total slots: 10785, VALU ops: 7745, Cycles: 1316
Speedup over baseline: 112.3x
```

Note: The scheduler uses simulated annealing, so cycle count varies slightly between builds (observed range: 1,311–1,316). All runs pass the < 1,363 threshold.

## Approach Summary

The optimization happened in three phases:

**1. Algorithmic restructuring (147,734 → ~1,400 cycles)**
- Level-3 tree fusion: preload tree[0–14] into vector registers, replace gather ops with vselect cascades for rounds 0–3 and 11–14
- 16-desk interleaving with 2-tile processing
- Address-tracking branch: track scatter address directly (`addr = 2*addr + (1-fp) + bit`)
- Deferred index computation via FMA chain at round boundaries

**2. Operation reduction (~1,400 → 1,388 cycles)**
- R10–R11 XOR fold: precompute `HASH_CONST_5 ^ tree[0]`, eliminating 31 VALU ops

**3. Instruction scheduling (1,388 → 1,315 cycles)**
- Replaced the greedy list scheduler with priority-based topological sort + simulated annealing
- Key insight: to keep the 6-wide VALU pipeline saturated, schedule non-VALU ops that feed long VALU dependency chains *before* the VALU ops themselves
- SA with block reversals on the topological order discovers non-obvious reorderings the greedy scheduler misses

Final kernel: 10,785 ops (7,745 VALU), ~1,311–1,316 cycles. Theoretical VALU-bound minimum is ceil(7745/6) = 1,291. Scheduling overhead is ~20–25 cycles (~1.9%).

## What's attached

- `perf_takehome.py` — the optimized kernel (914 lines)
- `OPTIMIZATION_REPORT.md` — detailed writeup of the full journey, what worked, what didn't, and kernel architecture
- Resume

## Context

I used Claude Code (Opus 4.5) as a tool throughout — for generating experiment variants, running automated searches, and implementing the SA scheduler. The algorithmic insights, experimental strategy, and direction came from me. The OPTIMIZATION_REPORT.md documents the full process in detail.

---

[NAME]
[CONTACT INFO]
