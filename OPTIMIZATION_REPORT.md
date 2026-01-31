# Optimization Report: VLIW SIMD Tree Traversal + Hash Kernel

**Final result: 1,311 cycles** (verified correct across 20+ random seeds)
**Speedup: 112.7x** over 147,734-cycle baseline
**Submission tests: 9/9 passing** (including the `< 1363` threshold)

```
Thresholds beaten:
  147,734  baseline                         112.7x
   18,532  updated starting point            14.1x
    2,164  Opus 4 many hours                  1.65x
    1,790  Opus 4.5 casual / best human 2hr   1.37x
    1,579  Opus 4.5 2hr harness               1.20x
    1,548  Sonnet 4.5 many hours              1.18x
    1,487  Opus 4.5 11.5hr harness            1.13x
    1,363  Opus 4.5 improved harness          1.04x
```

Validation:
```bash
git diff origin/main tests/          # empty — tests untouched
python3.11 tests/submission_tests.py  # 9/9 pass, 1311 cycles
```

---

## 1. Problem Description

Optimize a VLIW SIMD kernel that performs parallel tree traversal with hash
computation. Each batch element walks a binary tree for 16 rounds, hashing its
value with the tree node at each step and branching left/right based on a bit
extracted from the hash.

**Parameters:** `forest_height=10, rounds=16, batch_size=256, VLEN=8`

**ISA resources per cycle:**
| Engine | Slots/cycle | Notes |
|--------|-------------|-------|
| ALU    | 12          | Scalar arithmetic |
| VALU   | 6           | Vector ops (FMA, XOR, shifts, AND, ADD) |
| Load   | 2           | Loads from memory/scratch |
| Store  | 2           | Stores to memory/scratch |
| Flow   | 1           | vselect, jumps, conditionals — **bottleneck** |

**Constraints:** SCRATCH_SIZE=1536, N_CORES=1, no vgather instruction.

---

## 2. The Optimization Journey

### Phase 1: Algorithmic Breakthroughs (sessions Jan 24-25)

Starting from the 147,734-cycle naive baseline, the kernel was restructured
through a series of major algorithmic changes. Key milestones:

| Cycles | Speedup | Technique |
|--------|---------|-----------|
| 147,734 | 1.0x | Naive baseline |
| ~18,000 | ~8x | Round fusion (2 rounds per iteration) |
| ~5,200 | ~28x | 16-desk interleaving + 2-tile processing |
| 4,062 | 36x | 8-desk pipeline variant |
| ~1,558 | 95x | Level-3 tree fusion (preload tree[0-14]) |
| 1,400 | 106x | ALU-based tree index computation |
| 1,390 | 106x | Deferred idx computation via FMA chain |

#### Key algorithmic innovations:

**Level-3 Tree Fusion** — The critical insight that all 256 batch elements
start at the same tree root. At round R, elements are at tree level R, which
has only 2^R nodes. For rounds 0-3, there are at most 15 unique nodes. By
preloading tree[0] through tree[14] into vector registers and using vselect
cascades instead of gather operations, we eliminate expensive per-element
scatter-gather loads for the first 4 rounds. This is repeated for rounds 11-14
using the same 15 nodes (since the tree is binary, levels repeat).

**Addr-tracking branch** — Instead of maintaining a tree index and converting
it to an address each round, the kernel tracks the scatter address directly:
`addr = 2*addr + (1-fp) + bit`. This eliminates the idx-to-addr conversion
(saves VALU ops).

**Deferred idx computation** — For the fused rounds 0-3, instead of building
the tree index incrementally at each round (R0: idx from bit0, R1: idx from
bit0+bit1, etc.), all 4 bits are collected and the final address is computed
once at R3 via an FMA chain:
```
s1 = 2*bit0 + bit1
s2 = 2*s1 + bit2
addr = FMA(s2, 2, bit3) + (fp + 15)
```
This saves 128 VALU ops.

**ALU-based tree indices** — Tree indices 3-14 are computed via scalar ALU
addition chains from constants 1 and 2, freeing 12 load slots for other work.

### Phase 2: Overnight Exhaustive Search (Jan 25-26, ~8 hours)

With the algorithmic structure locked in at 1,390 cycles, an automated search
explored 200+ experiments across 60+ agent processes. Categories tested:

| Category | Experiments | Cycle Range | Finding |
|----------|------------|-------------|---------|
| Hash desk orderings | All 24 permutations | 1,389-1,400 | [1,0,3,2] best at 1,389 |
| Group processing orders | All 24 permutations | 1,389-1,401 | [0,1,2,3] tied for best |
| Group sizes (2,8,16) | 4 experiments | FAIL | Fused rounds hardcoded for size 4 |
| Hash emission strategies | 8 variants | All 1,389 | Scheduler compensates |
| Gather restructuring | 3 variants | 1,391-1,394 | No improvement |
| **R10-R11 XOR fold** | **1 experiment** | **1,388** | **Only improvement found** |
| XOR fold + all desk orders | 24 experiments | All 1,388 | Fold makes ordering irrelevant |
| Cross-group interleaving | 5 experiments | 1,423-1,525 | Much worse |
| Flow-based branch | 1 experiment | 1,414 | Saved 191 VALU but flow bottleneck |
| Structural: round fusion, hash algebra, speculative exec, tiling, etc. | ~100+ | All >= 1,388 | No improvement |

**R10-R11 XOR fold (H67):** The one improvement found overnight. Precomputes
`v_c5_xor_t0 = HASH_CONST_5 ^ tree[0]` during initialization, then uses this
in round 10's final hash stage instead of HASH_CONST_5. This folds round 11's
XOR-with-tree[0] into round 10's hash output. Net savings: 31 VALU ops
(32 desk-iterations × 1 XOR − 1 precompute). VALU: 7,779 → 7,748.

**Exhaustive search conclusion:** Parameter-level and emission-order changes
plateau at 1,388 with the greedy list scheduler. The remaining 96-cycle gap
to the theoretical minimum of 1,292 (ceil(7748/6)) is scheduling overhead.

### Phase 3: Scheduler Revolution (Jan 26, "50 shots")

The breakthrough insight: **the greedy list scheduler was the bottleneck, not
the kernel operations.** The original scheduler processes ops in emission order,
placing each at the earliest legal cycle. This is fast but leaves cycles empty
when a better reordering would pack more ops per cycle.

#### Shot-by-shot progression:

| Shot | Cycles | Delta | Technique |
|------|--------|-------|-----------|
| 0 | 1,388 | — | Baseline (greedy scheduler) |
| 5 | 1,384 | -4 | Topological sort with different priority |
| 13 | 1,381 | -7 | VALU-first priority (schedule non-VALU first to clear deps) |
| 18 | 1,350 | -38 | Downstream-VALU priority (threshold=50) |
| 19 | 1,332 | -56 | Downstream-VALU priority (threshold=25) |
| 85 | **1,311** | **-77** | **Simulated annealing on topological order** |

#### The winning scheduler (`_schedule_slots`):

**Step 1: Dependency graph construction.** Build a DAG of all ~10,000 ops
based on read-after-write and write-after-read/write hazards.

**Step 2: Compute downstream VALU count.** For each op, compute the maximum
number of VALU ops on any path from that op to a sink. This estimates how
"urgent" it is to schedule an op — ops that feed long VALU chains should be
scheduled early to avoid starvation.

**Step 3: Priority-based topological sort.** Try 4 priority strategies:
1. VALU-first (schedule VALU ops before others)
2. Downstream-VALU priority (threshold=25) — schedule non-VALU ops that feed
   long VALU chains first, keeping the VALU pipeline fed
3. ALU-first — schedule ALU ops early to clear scalar dependencies
4. Downstream-VALU (threshold=28) — wider variant

Strategy #2 was the key insight: **to keep the VALU pipeline busy, you need
to schedule the non-VALU ops that feed it BEFORE the VALU ops themselves.**
The greedy scheduler treated all ops equally and often left VALU slots empty
because load/ALU operations that produce VALU inputs hadn't been scheduled yet.

**Step 4: Simulated annealing.** Starting from each priority ordering, apply
SA with block reversals:
- Randomly select a contiguous block of ops (size 2-16)
- Reverse the block
- Check if the reversal respects all dependency constraints
- Accept if it improves cycle count, or probabilistically accept worse
  solutions (temperature-based)
- Feed each accepted reordering to the original greedy scheduler

The SA discovers non-obvious reorderings that no priority function finds,
squeezing out the final cycles of scheduling overhead.

**Step 5: Chain refinement.** Take the best SA result and run additional SA
passes with different seeds to escape local optima.

---

## 3. Kernel Architecture (Final Version)

### 3.1 Data Layout

```
Scratch memory (1,385 / 1,536 words used):
  - 16 desks × 8 vector registers = 128 vectors (1,024 words)
  - Vector constants: hash constants, shifts, FMA multipliers
  - Scalar scratch: tmp_scalar, tmp_addr, forest_values_p, etc.
  - 15 tree node vectors (tree[0] through tree[14])
  - Precomputed: v_c5_xor_t0 = HASH_CONST_5 ^ tree[0]
```

Each desk holds 8 VLEN-wide registers:
```
idx       — tree index (used in fused rounds)
val       — hash accumulator
node_val  — loaded tree node value
addr      — scatter-gather address (used in gather rounds)
tmp1, tmp2 — temporaries for hash computation
bit0, bit1 — extracted bits for deferred idx computation
```

### 3.2 Processing Pipeline (per tile)

```
For each tile (2 tiles, 8 desks each):
  For each group (4 groups of 4 desks):
    Rounds 0-3 (FUSED):
      R0: XOR(val, tree[0]) → hash → extract bit0
      R1: vselect(bit0, tree[2], tree[1]) → XOR → hash → extract bit1
      R2: 3-vselect cascade → XOR → hash → extract bit2
      R3: 7-vselect cascade → XOR → hash → deferred addr (5 FMA ops)

    Rounds 4-9 (GATHER):
      Each round: 8 loads per desk → XOR → 12-op hash → 3-op branch

    Round 10 (OPTIMIZED):
      Gather → XOR → hash with C5^tree[0] fold → skip branch (R11 handles)

    Rounds 11-14 (FUSED):
      Same as R0-R3 but using R10's hash output as starting value
      R11: XOR already folded into R10's hash via v_c5_xor_t0

    Round 15 (FINAL):
      Gather → XOR → hash → extract final bit into idx

  Store all 16 desks (idx, val) back to memory
```

### 3.3 Hash Function

The hash is a fixed 6-stage function applied per element per round:
```
Stage 0: val = FMA(val, 4097, C0)           — multiply_add
Stage 1: val = (val ^ C1) ^ (val >> S1)     — XOR + shift + XOR
Stage 2: val = FMA(val, 33, C2)             — multiply_add
Stage 3: val = (val + C3) ^ (val << S3)     — ADD + shift + XOR
Stage 4: val = FMA(val, 9, C4)              — multiply_add
Stage 5: val = (val ^ C5) ^ (val >> S5)     — XOR + shift + XOR
```
Total: 12 VALU ops per desk per round. 12 × 16 rounds × 32 desk-iterations
= 6,144 VALU ops. This is irreducible — defined by the ISA's hash function.

### 3.4 Operation Count

```
Total ops:      10,785
  VALU:          7,745 (71.8%) — binding constraint
  Load:          2,157 (20.0%)
  Store:           256 (2.4%)
  ALU:             338 (3.1%)
  Flow:            289 (2.7%)

Theoretical minimum (VALU-bound): ceil(7745/6) = 1,291 cycles
Actual with SA scheduler:         1,311 cycles
Scheduling overhead:              20 cycles (1.5%)
```

---

## 4. What Didn't Work

### 4.1 Dead Ends

**Cross-group interleaving (1,423-1,525):** Attempting to interleave operations
from different groups destroyed locality that the greedy scheduler relies on.

**Flow-based branch (1,414):** Replacing the 3-VALU branch (AND + FMA + ADD)
with 2-VALU + 1-FLOW (AND + vselect + FMA) saved 191 VALU ops but shifted
the bottleneck to the flow engine (1 slot/cycle). Net negative.

**Level-4 fusion (1,472):** Extending the vselect cascade to round 4 requires
15 vselects, overwhelming the flow engine. Correct but slower.

**Group size changes (FAIL):** The fused round functions are hardcoded for
4-desk groups. Changing group size requires rewriting the vselect cascades.

**Genetic algorithms, constraint solvers, ILP scheduling:** Several agents
attempted these approaches. None produced valid or improved kernels within
the available time.

### 4.2 Why 1,311 Might Be Near-Optimal

The scheduling overhead is now just 20 cycles (1.5%). The remaining gap to the
theoretical minimum of 1,291 likely comes from:
- Data dependency chains that prevent perfect packing
- The flow engine (1 slot/cycle) occasionally serializing vselect cascades
- Load-to-use latencies in gather rounds

Further improvement likely requires reducing total VALU ops (e.g., finding
algebraic shortcuts in the hash function or sharing computation across rounds).

---

## 5. Key Files

| File | Description |
|------|-------------|
| `perf_takehome.py` | Final optimized kernel (914 lines, 1,311 cycles) |
| `problem.py` | ISA simulator and reference implementation (DO NOT MODIFY) |
| `tests/submission_tests.py` | Official submission tests (DO NOT MODIFY) |
| `/tmp/overnight_results/log.txt` | 100-hypothesis overnight search log |
| `/tmp/overnight_results/summary.txt` | Overnight analysis and conclusions |
| `/tmp/shots_log.txt` | 50-shot scheduler optimization log |
| `/tmp/exp{1-106}.py` | Individual experiment files |

---

## 6. Approach Summary

The optimization can be understood in three layers:

1. **Algorithm** (147,734 → ~1,400): Level-3 tree fusion, round fusion,
   16-desk interleaving, addr-tracking, deferred idx computation, ALU tree
   indices, interleaved tile-0 loading. These reduce the total operations and
   create parallelism the scheduler can exploit.

2. **Operation reduction** (1,400 → 1,388): R10-R11 XOR fold, removing unused
   variables. Small wins from algebraic simplification.

3. **Scheduling** (1,388 → 1,311): Priority-based topological reordering with
   downstream-VALU analysis + simulated annealing. This was the single largest
   improvement in the final phase, recovering 77 of the 96 wasted scheduling
   cycles (80% of the overhead eliminated).

The lesson: once the algorithmic structure is locked in, the quality of
instruction scheduling dominates. A smarter scheduler found 5.5% more
performance from the same operations that the greedy scheduler couldn't pack.

---

## 7. Reproduction

```bash
# Verify tests are untouched
git diff origin/main tests/

# Run submission tests (build takes ~2-6 min due to SA, cached after first run)
python3.11 tests/submission_tests.py

# Quick correctness check
python3.11 perf_takehome.py --check
```

Note: Python 3.11+ is required (`match` syntax in problem.py). The system
default `python3` may be 3.9.

---

*Generated by Claude Opus 4.5 — January 26, 2026*
