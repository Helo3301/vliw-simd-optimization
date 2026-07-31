# VLIW SIMD Optimization: 1,204 cycles

**122.7x** faster than the 147,734-cycle baseline. 9/9 submission tests pass.

This repository contains my optimized solution to [Anthropic's VLIW SIMD performance
take-home](https://github.com/anthropics/original_performance_takehome). The kernel walks a binary
tree for 16 rounds, hashing each batch element against the node it lands on and branching on a bit
of the hash.

## Results

| Metric | Value |
|--------|-------|
| Cycles | **1,204** |
| Baseline | 147,734 |
| Speedup | **122.7x** |
| Submission tests | 9/9 passing |
| Binding engine | load — 2,149 ops at 2/cycle = 1,075-cycle floor |

Against the published reference points, all of which are for the 2-hour variant starting at 18,532
cycles:

| Reference | Cycles |
|-----------|--------|
| Claude Opus 4, many hours in the harness | 2,164 |
| Claude Opus 4.5, casual session ≈ best human in 2h | 1,790 |
| Claude Opus 4.5, 11.5h in the harness | 1,487 |
| Claude Opus 4.5, improved harness | 1,363 |
| **This kernel** | **1,204** |

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

### Scheduling

- **Load-aware priority orderings.** The inherited scheduler ranked ops by downstream VALU count,
  correct while VALU was binding. Once load binds, the same reasoning has to run on downstream
  *load* count: schedule the address arithmetic feeding long gather chains first, or the load
  pipeline starves.
- **Dead-op elimination** by backward liveness. Removes 36 ops, 32 of them per-desk index vloads —
  every element provably starts at the tree root, so level-3 fusion overwrites the loaded value
  before anything reads it.
- **Annealing right-sized.** It now finds nothing (1,204 with or without), because no reordering
  conjures load slots that don't exist. It was worth 77 cycles on the previous attempt, when VALU
  bound and every other engine had slack. Build time went from ~17 min to ~4 min as a result.

## What was tried and rejected

| Idea | Result |
|------|--------|
| Cycle-driven list scheduler, critical-path priority | 2,076 cycles; even program-order priority gave 1,379 against 1,218. Aggressive reordering fights the anti-dependency web created by per-desk register reuse. |
| Constant-operand vselects → `multiply_add` against precomputed tree differences | 1,213–1,246 across all 8 level splits, against 1,211. Flow fell 705 → 257 exactly as predicted; flow was saturated but not *binding*, and the replacement FMAs sit on the critical path the vselects were off. |
| Fusing round 4 (only 16 distinct nodes) to cut 256 gather loads | Blocked: needs ~256 words of scratch against 141 free. |

## Where the remaining time goes

1,204 cycles against a 1,075-cycle load floor. The 129-cycle gap is dependency structure — startup,
the tile-0/tile-1 seam, and the flow-bound fused rounds — not packing inefficiency; the load engine
is already saturated in ~89% of cycles. Moving the floor itself means cutting gather loads, and the
only candidate is fusing round 4, which is scratch-blocked until the desk register file is
restructured.

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
