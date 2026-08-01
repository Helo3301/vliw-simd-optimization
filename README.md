# VLIW SIMD Optimization: 1,139 cycles

**129.7x** faster than the 147,734-cycle baseline. 9/9 submission tests pass.

This repository contains my optimized solution to [Anthropic's VLIW SIMD performance
take-home](https://github.com/anthropics/original_performance_takehome). The kernel walks a binary
tree for 16 rounds, hashing each batch element against the node it lands on and branching on a bit
of the hash.

## Results

| Metric | Value |
|--------|-------|
| Cycles | **1,139** |
| Baseline | 147,734 |
| Speedup | **129.7x** |
| Submission tests | 9/9 passing |
| Binding engine | load — 2,120 ops at 2/cycle = 1,060-cycle floor (93.1% occupied) |

Against the published reference points, all of which are for the 2-hour variant starting at 18,532
cycles:

| Reference | Cycles |
|-----------|--------|
| Claude Opus 4, many hours in the harness | 2,164 |
| Claude Opus 4.5, casual session ≈ best human in 2h | 1,790 |
| Claude Opus 4.5, 11.5h in the harness | 1,487 |
| Claude Opus 4.5, improved harness | 1,363 |
| **This kernel** | **1,139** |

## Correctness

`reference_kernel2` writes both `inp_values` and `inp_indices` every round, but
`tests/submission_tests.py` only compares `inp_values`. An earlier version of this kernel therefore
passed every test while writing **wrong values to all 256 final indices**. That is fixed: round 15
now recovers the index it branches to from the tracked gather address,

```
idx_next = 2*idx + 1 + bit = 2*addr + (1 - 2*forest_p) + bit
```

Verified against the reference on **both** arrays across 20 seeds. To check this yourself, compare
full memory rather than just the values slice:

```python
for ref in reference_kernel2(mem): pass
ip, vp, n = ref[5], ref[6], 256
assert machine.mem[vp:vp+n] == ref[vp:vp+n]   # values  (what the tests check)
assert machine.mem[ip:ip+n] == ref[ip:ip+n]   # indices (what they don't)
```

Note that the kernel is specialized to the tested configuration in ways that predate this work:
level-3 tree fusion assumes every element starts at the root (which `Input.generate` guarantees),
and round 10's wrap-to-zero is assumed rather than tested.

## Key techniques

### Algebraic: hash stages 2+3 fuse into 3 ops

Stage 3 of the hash is `(s2 + C3) ^ (s2 << 9)`, where stage 2 gives `s2 = a*33 + C2`. Both operands
are affine in `a` over Z/2³², so each is a single `multiply_add` and `s2` never has to exist:

```
t1  = a*33    + (C2 + C3)              == s2 + C3
t2  = a*16896 + ((C2 << 9) mod 2^32)   == s2 << 9
out = t1 ^ t2
```

11 VALU ops per hash instead of 12, and the two FMAs are mutually independent so the dependency
chain shortens as well. −509 VALU ops. A previous writeup called the 12-op hash "irreducible —
defined by the ISA's hash function"; that was the wrong premise.

### Structural: spill vector work onto the idle ALU

VALU was binding at 6 slots/cycle. The ALU has **12** slots/cycle and was carrying 91 ops — roughly
14,000 idle slots. A VALU slot is exactly 8 lanes of the work one ALU slot does on one lane, so any
elementwise vector op can be re-emitted as `VLEN` scalar ALU ops (`emit_velem`). Branch-bit
extraction, the node XOR and the address adds moved across. That trades 8/12 of an ALU cycle for
1/6 of a VALU cycle — worse per op, free while the ALU is empty. −1,184 VALU ops.

Together these took VALU from 7,745 to 6,085 and moved the binding engine from VALU to load.

### Scheduling and pipelining

Once the ALU spill moved the bottleneck to the load engine, the useful question became: **which
cycles issue zero loads, and why is each one empty?** At 1,204 cycles, 123 did, in three clusters —
and each cluster had a different cause.

- **The fused rounds contain no gathers at all**, and round-block-major emission marched all 16
  desks through them simultaneously, so the load engine idled across the full width of the kernel.
  Groups now march in lockstep only within a chunk, and chunks run one after another, so one
  chunk's fused rounds overlap another chunk's gathers (`PTH_EMIT_CHUNK`, default 2). The
  tile-0/tile-1 seam disappears entirely. **−54 cycles.**
- **The tree preload was serialised by register reuse.** All 14 nodes went through one shared
  `tmp_addr`/`tmp_scalar` pair, so node *i+1*'s address add had to wait on node *i*'s broadcast — a
  14-deep false dependency landing squarely in the pipeline fill, where nothing else can run.
  Independent registers cost 28 of the 141 free scratch words. **−6 cycles.**
- **`const` issues on the load engine**, and per-desk element offsets were 16 `const` ops per tile —
  32 load slots spent on an arithmetic progression. One const now seeds each tile and the ALU, which
  has slack, walks the rest. `const` ops 51 → 22. **−5 cycles.**

Also: load-aware priority orderings (the inherited scheduler ranked by downstream VALU count,
correct only while VALU bound); dead-op elimination by backward liveness, which removes 32 per-desk
index vloads that level-3 fusion overwrites before any read; and an annealer sped up by keeping an
incremental position table instead of rebuilding all *n* positions per iteration.

**Annealing now contributes nothing** — same cycle count with or without it, against 77 cycles on a
previous attempt. Once load binds rather than VALU, no reordering conjures load slots that do not
exist. Its budget is sized as insurance, which keeps the build at a few minutes.

## What was tried and rejected

| Idea | Result |
|------|--------|
| Cycle-driven list scheduler, critical-path priority | 2,076 cycles; even program-order priority gave 1,379 against 1,218. Aggressive reordering fights the anti-dependency web created by per-desk register reuse. |
| Constant-operand vselects → `multiply_add` against precomputed tree differences | 1,213–1,246 across all 8 level splits, against 1,211. Flow fell 705 → 257 exactly as predicted and it still lost: flow was saturated but never *binding*, and the replacement FMAs sit on the critical path the vselects were off. |
| Fusing round 4 (only 16 distinct nodes) to cut 256 gather loads | Blocked on scratch. A 16-way select does fit in 3 temps — the last level-A FMA can write into the b3 bit register itself, since this ISA performs all reads before all writes — but the deferred address chain still needs b3 afterwards, and every reordering comes out one register short. |
| 8 desks per tile instead of 16, to free scratch for the above | Frees 520 words (875/1536) and costs 112 cycles (1,262 vs 1,150) against at most ~80 recoverable. Net loss. |
| Non-uniform chunk widths (`1,3`, `1,1,2`, `2,1,1`, `1,2,1`, `3,1`, `1,1,1,1`) | Best ties uniform `2,2`; most are worse. The residual slack at chunk transitions is not reachable by rearranging chunk widths. |

## Where the remaining time goes

1,139 cycles against a 1,060-cycle load floor, with the load engine 93.1% occupied. The 79-cycle
gap is now mostly structural rather than schedulable:

- **~34 cycles of pipeline fill.** The first gather issues at cycle 72, because no gather address
  exists until rounds 0–3 have hashed, and four sequential hash chains is ~44 cycles of pure
  latency. Through that window VALU runs at 4.25 of 6 slots — it is waiting on dependencies, not on
  capacity, so no scheduler can compress it.
- **13 cycles of drain**, the mirror image: hash, final index, store after the last gather.
- **~30 cycles of slack at chunk transitions**, the only genuinely recoverable part, and it resisted
  every chunk arrangement tried.

Moving the floor itself means cutting gather loads. 2,048 of the 2,120 loads are gathers — 8 rounds
× 256 elements — and the only reducible ones are rounds 4 and 15, which land on just 16 distinct
nodes. Both routes to that are priced out above.

## Reproduction

```bash
git diff origin/main tests/            # empty — tests unmodified
python3.11 tests/submission_tests.py   # 9/9, ~4 min (scheduler build)
python3.11 perf_takehome.py --check
```

Python 3.11+ required (`match` syntax in `problem.py`). `problem.py` and `tests/` are unmodified
from Anthropic's repository; all changes are confined to `perf_takehome.py`.

Tuning knobs are environment variables (`PTH_ALU_BIT`, `PTH_ALU_XOR`, `PTH_ALU_ADD`,
`PTH_DL_THRESH`, `PTH_SA_ITERS`, `PTH_NO_SA`); the defaults are the tuned values.

## Deep dive

`OPTIMIZATION_REPORT.md` covers the earlier phase of this work in detail. Its headline figure of
1,311 cycles does not reproduce — the official test on that commit gives **1,316** — and its claim
that the hash is irreducible at 12 VALU ops is superseded by the stage 2+3 fusion above.

## License

The optimized kernel and documentation are my original work. The ISA simulator (`problem.py`) and
test suite come from [Anthropic's original repository](https://github.com/anthropics/original_performance_takehome).
