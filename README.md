# VLIW SIMD Optimization: 1,138 cycles

**129.8x** faster than the 147,734-cycle baseline. 9/9 submission tests pass.

This repository contains my optimized solution to [Anthropic's VLIW SIMD performance
take-home](https://github.com/anthropics/original_performance_takehome). The kernel walks a binary
tree for 16 rounds, hashing each batch element against the node it lands on and branching on a bit
of the hash.

## Results

| Metric | Value |
|--------|-------|
| Cycles | **1,138** |
| Baseline | 147,734 |
| Speedup | **129.8x** |
| Submission tests | 9/9 passing |
| Binding engine | load — 2,106 ops at 2/cycle = 1,053-cycle floor |

Against the published reference points, all of which are for the 2-hour variant starting at 18,532
cycles:

| Reference | Cycles |
|-----------|--------|
| Claude Opus 4, many hours in the harness | 2,164 |
| Claude Opus 4.5, casual session ≈ best human in 2h | 1,790 |
| Claude Opus 4.5, 11.5h in the harness | 1,487 |
| Claude Opus 4.5, improved harness | 1,363 |
| **This kernel** | **1,138** |

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
- **The tree preload didn't need scalar loads at all.** `tree[0..15]` is 16 contiguous words, so two
  `vload`s fetch every node the fused rounds use — and words inside a loaded vector are ordinary
  scratch cells, so `vbroadcast` takes each one directly as its scalar source. This replaced 15
  scalar loads plus 15 address adds (which had also been serialised through one shared
  `tmp_addr`/`tmp_scalar` pair, a 14-deep false dependency landing in the pipeline fill). Load-engine
  ops −13, and ~40 scratch words freed. **−7 cycles** across both steps.
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
| Spilling hash ops to the ALU (as the branch-bit / node-XOR / address adds were) | Monotonically worse: 1,141 / 1,148 / 1,156 / 1,182 / 1,220 as the offloaded fraction rises. The earlier spills sit at round boundaries with slack; the hash shifts sit **on the critical path**, and one VALU op becomes 8 ALU ops that must all retire before the dependent XOR. The ALU is a throughput reservoir, not a latency substitute. |
| Fusing round 4 (only 16 distinct nodes) to cut 256 gather loads | Dead on economics, not just registers. It trades −254 loads (floor 1,053 → 926) for +280 VALU (floor 1,014 → **1,061**) and +224 flow (→930), so the binding floor *regresses*. Absorbing the VALU cost needs exactly the ALU spill the row above rules out. |
| The "C5 deferred fold" — restructure hash stage 5 so the next round's node XOR folds into the constant | Built and verified correct. It does work: stage 5 drops from 3 VALU ops to 2 (`w = s4 ^ (s4>>16)`, with C5 absorbed into pre-XORed `C5^tree[j]` constants and the branch bit coming out complemented), taking the hash to **10 VALU ops — its true ISA floor**. VALU 6,083 → 5,910. **Zero cycles**, because VALU is not the binding engine and its floor was already 39 cycles under load's. Not kept: real algebra, dead headroom. |
| Partial level-4 fusion — fuse round 4's 16-way node choice for a *fraction* of desks, sized to the flow engine's idle capacity | Built and verified correct at 8 fractions; **monotonically worse**, 1,139 → 1,313. The floor model was right and irrelevant: at 16/32 desks fused it moved the bound from 1,054 to **992** exactly as predicted (load 2,107→1,983, flow 706→946), and actual cycles rose 46 while starved cycles doubled, 88 → 196. Flow is 1 slot/cycle **globally**, so a 15-vselect cascade is 15 cycles of serial latency no other desk can overlap, landing in round 4's dependency chain between the address computation and the hash. The gather it replaces is 8 loads at 2/cycle that interleave freely across desks. Idle flow is not spare capacity; it is a 1-wide serial resource. |
| Deferring each chunk's round-15 gather to fill the tail | Identical 1,138 — the scheduler was already placing those gathers optimally. |
| Ramped chunk plans (`1,3`, `1,1,2`, `1,2,1`, `1,1,1,1`, `2,1,1`, `1,2`, `1,2,2`, …) | Best ties uniform width 2; most are worse. A narrower leading chunk does reach its first gather sooner, but its shorter gather phase then fails to hide the next chunk's fill. |
| Cycle-driven list scheduler, critical-path priority | 2,076 cycles; even program-order priority gave 1,379 against 1,218. Aggressive reordering fights the anti-dependency web from per-desk register reuse. |
| Constant-operand vselects → `multiply_add` on precomputed tree differences | 1,213–1,246 across all 8 level splits, against 1,211. Flow fell 705 → 257 exactly as predicted and it still lost: flow was saturated but never *binding*. |
| 8 desks per tile instead of 16, to free scratch | Frees 520 words but costs 112 cycles against at most ~80 recoverable. Net loss. |

## Where the remaining time goes

Earlier versions of this section quoted a "floor" that was really a conditional: *given this op
mix*, some engine needs N cycles. That is not a bound on the problem, and it moved every time the op
mix did. Here are bounds that are actually derived, weakest scope first.

**≥854 cycles, for any program.** The hash is 512 vector-hashes of 10 VALU ops each (10 is the ISA
floor: stages 0/2/4 are one `multiply_add`, stages 2+3 fuse into 3, stage 5 folds to 2, and stage 1's
`(a^C1)^(a>>19)` cannot fold its constant in either direction). 5,120 ÷ 6 slots = 854. This alone
refutes any target below it — including the 768 implied by counting only the unavoidable gathers.

**~1,044 cycles, for any program, by joint placement.** Solving for the best split of every work
class across the engines that can host it — rather than reading each engine's floor off the current
mix — gives 1,044 whether rounds 4 and 15 are gathered (load-bound) or fused (ALU-bound). The two
strategies land within one cycle of each other, which is why every attempt to trade between them
came out neutral-to-negative.

**1,100 cycles, for this op set under any schedule.** Using only the dependency graph: the earliest
dependency-feasible cycle of each load, sorted, gives `max over k of (est_k + (N−k)/2)` = 1,087, plus
a 13-cycle tail after the last gather. The binding term is at k=42 — after the first 42 loads clear
the pipeline fill at cycle 55, the remaining 2,064 still need 1,032 cycles at 2/cycle.

**1,138 achieved**, so 38 cycles of headroom against the tightest bound.

Those 38 are not scattered inefficiency. 97.8% of gather loads issue *later* than their earliest
legal cycle and 624 are ≥10 cycles late — the load engine carries a deep backlog nearly everywhere,
which is what a saturated bottleneck looks like. The residual is the moments the queue runs dry: the
pipeline fill (55 cycles, already inside the bound, and irreducible since no gather address exists
until four hash chains retire) and the program's tail, where the last chunk runs fused rounds 11–14
with no gathers left anywhere to issue — its own round 15 depends on its round 14.

Reordering cannot help a queue that is already backed up, which is why five further priority
orderings, nine chunk plans and a cycle-driven list scheduler all failed to move it.

The annealer agrees. Raising its budget 16x, from 1,500 iterations to 25,000, moves the result from
1,138 to **1,137** — one cycle, for 44 minutes of build time instead of 4. The search space is that
flat, which is what being 38 cycles from a hard bound looks like when the residual is structural
rather than reorderable. `PTH_SA_ITERS=25000` reproduces it if you want the cycle; the default is
deliberately the fast build.

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
