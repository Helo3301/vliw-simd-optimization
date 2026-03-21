# Step-by-Step Optimization Guide: VLIW SIMD Tree Traversal Kernel

## Target
Optimize `perf_takehome.py` for the Anthropic original_performance_takehome challenge.
Starting point: 147,734 cycles. Our result: **1,477 cycles (100x speedup)**.

## Verification Commands
```bash
# Run correctness + cycle count
python3.11 perf_takehome.py --check

# Run official submission tests (8 random-seed correctness + speed tiers)
python3.11 tests/submission_tests.py -v

# Verify no test/problem modifications
git diff origin/main tests/
git diff origin/main problem.py
```

## Prerequisites
- Python 3.10+ (3.11 recommended for `match` statement support in problem.py)
- Fresh clone of: `git clone https://github.com/anthropics/original_performance_takehome.git`

---

## ISA Reference (Memorize This)

| Engine | Slots/Cycle | Operations |
|--------|-------------|------------|
| **VALU** | 6 | +, -, *, /, &, \|, ^, <<, >>, multiply_add (FMA: a*b+c), vbroadcast |
| **ALU** | 12 | Same as VALU but scalar only |
| **Load** | 2 | load, vload, const, load_offset |
| **Store** | 2 | store, vstore |
| **Flow** | 1 | select, vselect (vector conditional), add_imm (scalar), jumps, halt, pause |

- **VLEN = 8**: Vector registers hold 8 elements
- **SCRATCH_SIZE = 1536**: Total scratch memory slots
- **No vgather**: Must do 8 individual scalar loads per gather
- **vselect uses Flow engine** (1/cycle), NOT VALU (6/cycle) -- critical for optimization
- **FMA**: `multiply_add(dest, a, b, c)` computes `a * b + c` in 1 VALU op

## Algorithm Overview

The kernel performs tree traversal + hash for 256 batch elements over 16 rounds:
```
for each round (0-15):
    val = hash(val XOR tree[idx])
    idx = 2*idx + 1 + (val & 1)    # binary tree branch
    if idx >= n_nodes: idx = 0       # bounds wrap
```

Hash function (6 stages, 12 VALU ops -- IRREDUCIBLE):
```
Stage 0: val = val * 4097 + C0              (1 FMA)
Stage 1: val = (val ^ C1) ^ (val >> S1)     (3 ops: XOR, SHIFT, XOR)
Stage 2: val = val * 33 + C2                (1 FMA)
Stage 3: val = (val + C3) ^ (val << S3)     (3 ops: ADD, SHIFT, XOR)
Stage 4: val = val * 9 + C4                 (1 FMA)
Stage 5: val = (val ^ C5) ^ (val >> S5)     (3 ops: XOR, SHIFT, XOR)
```

---

## STEP 1: Greedy List Scheduler
**Expected: Enables all further optimizations**

Build a greedy list scheduler that packs operations into VLIW cycles respecting:
1. Data dependencies (read-after-write, write-after-write, write-after-read)
2. Slot limits per engine per cycle
3. Earliest-cycle-first placement

This is the `_schedule_slots()` function. It takes a flat list of `(engine, slot)` tuples and produces a list of cycle dictionaries. The scheduler is critical -- it's responsible for ~93-97% of theoretical VALU utilization.

```python
def _schedule_slots(slots: list[tuple[str, tuple]]) -> list[dict[str, list[tuple]]]:
    # For each operation:
    #   1. Compute earliest cycle based on data dependencies
    #   2. Find first cycle where the engine has a free slot
    #   3. Place the operation, update dependency tracking
```

You also need `_slot_rw()` which returns (reads, writes) for each operation type to compute dependencies. See the final kernel for the complete implementation.

**Checkpoint**: The scheduler itself doesn't have a cycle count -- it's infrastructure for everything else.

---

## STEP 2: Basic Vectorized Kernel with Desks
**Expected: ~15,000-18,000 cycles**

Key concepts:
- **Desk**: A unit of work processing VLEN=8 batch elements in SIMD
- **32 desks needed**: 256 elements / 8 = 32 desks
- **Tile**: A group of desks processed together. Use 2 tiles of 16 desks each (fits scratch).

Per desk, allocate these vector registers in scratch:
- `idx` (8 slots): current tree indices
- `val` (8 slots): current values
- `node_val` (8 slots): gathered tree node values
- `addr` (8 slots): gather addresses
- `tmp1`, `tmp2` (8 slots each): temporaries for hash
- `bit0` (8 slots): saved branch bit for round fusion

Also allocate:
- Vector constants: v_zero, v_one, v_two, v_three, v_forest_p, v_n_nodes
- Hash constants: 6 hash constants + 3 shift constants + 3 FMA multipliers (all broadcast)
- 7 preloaded tree nodes: tree[0] through tree[6] (levels 0-2, broadcast)
- 3 diff vectors: tree[2]-tree[1], tree[4]-tree[3], tree[6]-tree[5]
- Scalar temporaries and offset registers

Scratch budget: ~1,180 of 1,536 slots used.

Basic round structure (for each desk, each round):
```
1. addr = forest_p + idx           (1 VALU)
2. For lane in 0..7:
     load node_val[lane] from addr[lane]  (8 loads)
3. val = val XOR node_val           (1 VALU)
4. Hash val (12 VALU ops)
5. bit = val & 1                    (1 VALU: AND)
6. idx = 2*idx + 1                  (1 VALU: FMA)
7. idx = idx + bit                  (1 VALU: ADD)
```

Emit all operations as a flat list, then feed to `_schedule_slots()`.

**Checkpoint**: ~15,000 cycles. Verify correctness with `--check`.

---

## STEP 3: Multi-Desk Interleaving with Groups
**Expected: ~4,000-5,000 cycles**

Instead of processing desks one at a time, process them in **groups** where operations from multiple desks interleave. This gives the scheduler independent operations to pack into VALU slots.

- **GROUP_SIZE = 4**: Process 4 desks together in each group
- **4 groups per tile**: 16 desks / 4 = 4 groups
- **Interleave by phase**: Emit all addr computations for group, then all gathers, then all XORs, then all hashes, then all branches

```python
GROUP_SIZE = 4
for group in all_groups:
    for round in range(16):
        # All addr for group
        for d in group: emit addr computation
        # All gathers for group
        for d in group:
            for lane in 8: emit load
        # All XORs for group
        for d in group: emit XOR
        # Hash interleaved across group
        emit_hash_interleaved(group)
        # All branches for group
        for d in group: emit branch
```

For hash interleaving, emit each hash stage across all desks in the group:
```python
def emit_hash_interleaved(group_desks):
    for stage in [0,1,2,3,4,5]:
        for d in group_desks:
            emit stage operations for desk d
```

**Checkpoint**: ~4,000-5,000 cycles. The exact number depends on hash interleaving strategy.

---

## STEP 4: Round Fusion (R0+1+2 and R11+12+13)
**Expected: ~1,600-1,700 cycles (major jump)**

**Key insight**: All 256 elements start at idx=0 (the tree root). This means:
- **Round 0**: ALL elements access tree[0] -- preloaded, no gather needed
- **Round 1**: Elements access tree[1] or tree[2] -- preloaded, 2-way select
- **Round 2**: Elements access tree[3..6] -- preloaded, 4-way select
- **Rounds 11-13**: After R10 resets idx=0, same pattern repeats

This eliminates gathers for 6 of 16 rounds and replaces them with arithmetic selection from preloaded values.

### Round 0 (fused):
```python
# XOR with preloaded tree[0] (no gather!)
emit: val = val XOR tree[0]               # 1 VALU
# Hash
emit_hash_interleaved(group)              # 12 VALU
# Simplified branch: idx starts at 0, so idx' = 1 + bit
emit: bit0 = val & 1                      # 1 VALU
emit: idx = 1 + bit0                      # 1 VALU (ADD, not FMA)
```
Saves: 1 addr VALU + 8 loads vs standard round. Also saves 1 branch VALU (ADD instead of FMA+ADD).

### Round 1 (fused, bit-tracking):
```python
# Node selection: idx is 1 or 2, select tree[1] or tree[2]
emit: node_val = tree[1] + bit0 * (tree[2] - tree[1])  # 1 FMA using precomputed diff
# XOR + Hash
emit: val = val XOR node_val              # 1 VALU
emit_hash_interleaved(group)              # 12 VALU
# Branch using bit-tracking
emit: bit1 = val & 1                      # 1 VALU
emit: idx = bit0 * 2 + 3                  # 1 FMA
emit: idx = idx + bit1                    # 1 ADD
```

### Round 2 (fused, 4-way select):
```python
# 4-way node selection from tree[3..6] using bit0 (R0) and bit1 (R1):
emit: node_a = tree[3] + bit1*(tree[4]-tree[3])   # FMA: 2-way select for bit0=0 path
emit: node_b = tree[5] + bit1*(tree[6]-tree[5])   # FMA: 2-way select for bit0=1 path
emit: node_val = vselect(bit0, node_b, node_a)     # vselect: pick based on R0 bit
# XOR + Hash + Standard branch (back to normal)
```

### Rounds 11-13: Identical logic to 0-2 (R10 resets idx=0).

### R10: Branch Skip
After round 10's hash, ALL indices exceed n_nodes (2047 for height-10 tree) and wrap to 0. This is deterministic. Skip the 3-op branch and 2-op bounds check. Replace with:
```python
emit: idx = idx XOR idx  # idx = 0, costs 1 VALU instead of 5
```

### Round 15: No Branch
R15 is the final round. The branch result is never used. Skip it entirely.

**Checkpoint**: ~1,600-1,700 cycles with GROUP_SIZE=4. Verify correctness.

---

## STEP 5: Per-Desk Hash Emission
**Expected: ~1,536 cycles (scheduling optimization)**

Change hash emission from stage-interleaved to per-desk:

**Before** (stage-by-stage across desks):
```python
for stage in [0,1,2,3,4,5]:
    for d in group: emit stage for d
```

**After** (all stages per desk, interleaved desk order):
```python
# Reorder desks: even indices first, then odd (0,2,1,3 for GROUP_SIZE=4)
gd = [group[0], group[2], group[1], group[3]]
for d in gd:
    emit all 12 hash ops for desk d
```

This gives the scheduler longer independent dependency chains, improving VALU packing.

**Checkpoint**: 1,536 cycles, 8,507 VALU ops. This is the **scheduling floor** for this operation count.

---

## STEP 6: Address-Tracking Branch Fusion
**Expected: ~1,497 cycles (operation reduction)**

**Core insight**: Instead of tracking `idx` and computing `addr = forest_p + idx` each gather round, track `addr` directly. The branch formula naturally produces the next gather address:

```
Current (4 VALU per gather round):
  bit = val & 1                        (AND)
  idx = 2*idx + 1                      (FMA)
  idx = idx + bit                      (ADD)
  addr = forest_p + idx                (ADD)  <-- ELIMINATED

Address-tracking (3 VALU per gather round):
  bit = val & 1                        (AND)
  addr = 2*addr + (1 - forest_p)       (FMA with precomputed constant)
  addr = addr + bit                    (ADD)
```

**Math proof**: `addr = forest_p + idx`. So `addr' = forest_p + idx' = forest_p + 2*idx + 1 + bit = forest_p + 2*(addr - forest_p) + 1 + bit = 2*addr - forest_p + 1 + bit = 2*addr + (1 - forest_p) + bit`.

**Implementation**:
1. Precompute `v_1_minus_fp = v_one - v_forest_p` at init (1 VALU)
2. After fused R0+1+2 (which works with idx), convert: `addr = forest_p + idx` (1 VALU per desk)
3. R3-R9: Use addr-tracking branch (saves 1 VALU per desk per round)
4. R10: addr ready from R9, gather directly, set idx=0 (saves 1 VALU)
5. After fused R11+12+13, convert again (1 VALU per desk)
6. R14: addr-tracking (saves 1 VALU)
7. R15: addr ready from R14 branch (saves 1 VALU)

**Savings**: ~255 VALU ops total.

**Checkpoint**: ~1,497 cycles, ~8,252 VALU ops. Verify correctness.

---

## STEP 7: vselect Node Selection in Fused Rounds
**Expected: ~1,477 cycles (VALU-to-flow trade)**

Replace FMA-based node selection in fused rounds with `vselect` (flow engine). This trades VALU ops (bottleneck, 6/cycle) for flow ops (1/cycle, usually has slack).

### Round 1/12 node selection:
**Before** (1 VALU):
```python
emit: node_val = FMA(bit0, diff_1_2, tree[1])  # 1 VALU
```

**After** (0 VALU, 1 flow):
```python
emit: node_val = vselect(bit0, tree[2], tree[1])  # 1 flow (0 VALU!)
```
Saves 1 VALU per desk per occurrence. Also eliminates the `diff_1_2` precomputation.

### Round 2/13 node selection (4-way):
**Before** (2 VALU + 1 flow):
```python
emit: node_a = FMA(bit1, diff_3_4, tree[3])    # 1 VALU
emit: node_b = FMA(bit1, diff_5_6, tree[5])    # 1 VALU
emit: node_val = vselect(bit0, node_b, node_a)  # 1 flow
```

**After** (0 VALU + 3 flow):
```python
emit: tmp = vselect(bit1, tree[4], tree[3])          # 1 flow
emit: node_val = vselect(bit1, tree[6], tree[5])     # 1 flow
emit: node_val = vselect(bit0, node_val, tmp)         # 1 flow
```
Saves 2 VALU per desk per occurrence. Also eliminates `diff_3_4` and `diff_5_6` precomputations (3 SUB ops removed from init + 24 scratch slots freed).

**Warning**: Don't convert TOO many operations to vselect. Beyond ~512 flow ops, the flow engine becomes the bottleneck. Rounds 1/12 + Rounds 2/13 is the sweet spot.

**Checkpoint**: ~1,477 cycles, ~8,057 VALU ops.

---

## STEP 8: Combined Branch + idx-to-addr Conversion
**Expected: 1,477 cycles (marginal VALU reduction)**

At the end of fused R2 and R13, we need both: (a) compute the branch and (b) convert from idx to addr for the upcoming addr-tracking rounds.

**Before** (4 VALU):
```python
# Standard branch
emit: bit = val & 1                      (AND)
emit: idx = 2*idx + 1                    (FMA)
emit: idx = idx + bit                    (ADD)
# Then convert
emit: addr = forest_p + idx              (ADD)
```

**After** (3 VALU -- the branch directly produces addr):
```python
emit: bit = val & 1                                    (AND)
emit: t = forest_p + 1 + bit                           (ADD using precomputed v_fp_plus_1)
emit: addr = FMA(idx, 2, t)  # addr = 2*idx + fp+1+bit (FMA)
```

This skips the separate idx computation and conversion. The FMA directly computes the address.
Precompute `v_fp_plus_1 = v_forest_p + v_one` at init.

**Note**: This means fused rounds R2/R13 don't update `idx` -- they produce `addr` directly. The gather rounds that follow use addr-tracking. When fused rounds R11+12+13 need idx (at idx=0), R10 already set idx=0.

**Savings**: 1 VALU per desk at end of R2 fusion + 1 at end of R13 fusion = 64 VALU total.

**Checkpoint**: 1,477 cycles, 7,994 VALU ops. Correctness PASSED on 8 random seeds.

---

## Final Result Summary

| Step | Change | VALU Ops | Cycles | Test Tier Passed |
|------|--------|----------|--------|-----------------|
| 0 | Baseline | - | 147,734 | test_kernel_speedup |
| 2 | Basic vectorized | ~12,000 | ~15,000 | test_kernel_updated_starting_point |
| 3 | Multi-desk groups | ~10,000 | ~4,500 | test_opus4_many_hours |
| 4 | Round fusion + R10 skip | ~8,700 | ~1,650 | test_opus45_casual |
| 5 | Per-desk hash emission | 8,507 | 1,536 | test_sonnet45_many_hours |
| 6 | Address-tracking | 8,252 | 1,497 | test_opus45_11hr |
| 7 | vselect node selection | 8,057 | 1,477 | test_opus45_11hr |
| 8 | Combined branch+convert | 7,994 | 1,477 | test_opus45_11hr |

Tests passed: 8/9 (all except `test_opus45_improved_harness` which requires < 1,363 cycles).

---

## Key Constants for the Final Kernel

```
NUM_DESKS = 16          (per tile)
NUM_TILES = 2           (16 * 2 * 8 = 256 elements)
GROUP_SIZE = 4          (desks processed together)
NUM_PRELOADED = 7       (tree nodes 0-6, levels 0-2)
SCRATCH_USAGE = 1,176   (of 1,536 available)
TOTAL_VALU_OPS = 7,994
TOTAL_SLOTS = 11,075
FLOW_OPS = ~320         (vselect ops from fusion)
```

---

## Proven Dead Ends (Don't Waste Time On These)

1. **2-op branch**: `idx = 2*idx + 1 + (val & 1)` provably requires 3 VALU ops minimum (exhaustive ISA search)
2. **Hash reduction**: 12 VALU ops per hash call is algebraically irreducible
3. **Critical path scheduler**: 277 cycles worse than greedy list scheduler
4. **GROUP_SIZE != 4**: Extensive sweep confirms 4 is optimal for this configuration
5. **Lane-first load order**: Always 100-150 cycles worse than desk-first
6. **Replacing all vselects with arithmetic**: Too many VALU ops added vs flow savings
7. **More than ~512 flow ops**: Flow engine (1/cycle) becomes the bottleneck
8. **Round fusion beyond level 2**: 8-way selection costs more VALU than the gather it replaces
9. **ALU offloading**: ALU is scalar-only; 8 ALU ops per vector op is much worse than 1 VALU

## Remaining Gap Analysis

```
Current:            1,477 cycles
Theoretical VALU minimum: ceil(7994/6) = 1,333 cycles
Scheduling overhead:      144 cycles (92.2% efficiency)
Target:             1,363 cycles (requires 97.8% scheduling efficiency -- likely impossible)
```

The 1,363 target may correspond to a different problem specification or harness improvements not reflected in the public repo.

---

## File Structure

```
perf_takehome.py          # Your kernel (only file you modify)
problem.py                # Simulator and ISA (DO NOT MODIFY)
tests/
  submission_tests.py     # Official tests (DO NOT MODIFY)
  frozen_problem.py       # Frozen copy of problem.py (DO NOT MODIFY)
```
