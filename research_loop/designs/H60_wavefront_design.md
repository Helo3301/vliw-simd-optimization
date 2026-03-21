# H60: Wavefront Processing with Arithmetic Value Selection

## Executive Summary

This design proposes a **fundamentally different processing model** that could theoretically break the 2,048 cycle barrier by exploiting **index convergence** in tree traversal combined with **arithmetic value selection** to bypass the Flow engine bottleneck.

**Key Innovation**: Use VALU-based arithmetic formulas instead of vselect (Flow engine) to route values, enabling 6 operations/cycle instead of 1 operation/cycle.

---

## 1. Arithmetic Value Selection: Correctness Verification

### The Formula

Given:
- `T1 = tree[1]`, `T2 = tree[2]`
- `bit` in {0, 1} (determines which child to take)
- Want: `node_val = T1 if bit == 0 else T2`

**Arithmetic Solution**:
```
result = T1 + bit * (T2 - T1)
```

**Verification**:
- When `bit = 0`: `result = T1 + 0 * (T2 - T1) = T1` (correct)
- When `bit = 1`: `result = T1 + 1 * (T2 - T1) = T1 + T2 - T1 = T2` (correct)

**Alternative Form** (equivalent):
```
result = T1 * (1 - bit) + T2 * bit
```

### ISA Implementation

Using available VALU operations:
```python
# Step 1: Compute difference
diff = vsub(T2, T1)                    # diff = T2 - T1

# Step 2: Compute result using multiply_add
result = multiply_add(bit, diff, T1)   # result = bit * diff + T1
                                       #        = bit * (T2 - T1) + T1
                                       #        = T1 + bit * (T2 - T1)
```

**VALU Cost**: 2 operations per selection (vsub + multiply_add)
**VALU Throughput**: 6 slots/cycle
**Effective Selection Rate**: 3 selections/cycle (vs 1 vselect/cycle in Flow)

**THE FORMULA IS CORRECT and provides 3x throughput improvement over vselect!**

---

## 2. Wavefront Processing Model

### Current Model (Desk Pipelining)

Process different batch elements in parallel:
```
Iteration 1: Elements 0-127 through Round 0
Iteration 2: Elements 0-127 through Round 1
... (interleaved across 16 desks)
```

Each element does its own gather: 4,096 total tree loads.

### Proposed Model (Wavefront Processing)

Process ALL 256 elements through each round together:
```
Round 0: ALL elements at idx=0 -> Load tree[0] ONCE, broadcast to ALL
Round 1: ALL elements at idx in {1,2} -> Load tree[1] AND tree[2], arithmetic select
Round 2: ALL elements at idx in {1,2,3,4} -> Load 4 values, multi-level select
...
```

### Index Distribution by Round

| Round | Unique Indices | Tree Loads Required | Max Index Value |
|-------|----------------|---------------------|-----------------|
| 0     | 1              | 1                   | 0               |
| 1     | 2              | 2                   | 2               |
| 2     | 4              | 4                   | 4               |
| 3     | 8              | 8                   | 10              |
| 4     | 16             | 16                  | 22              |
| 5     | 32             | 32                  | 46              |
| 6     | 64             | 64                  | 94              |
| 7     | 128            | 128                 | 190             |
| 8     | 256            | 256                 | 382             |
| 9-15  | 256            | 256 each            | varies          |

**Key Insight**: Early rounds (0-7) have bounded unique indices. Round 8+ has 256 unique indices (same as batch size).

---

## 3. Detailed Round-by-Round Design

### Round 0: Single Value Broadcast

All 256 elements start at idx=0.

**Operations**:
1. Load tree[0]: 1 load slot
2. Broadcast to all 32 vectors (256 elements / 8 VLEN): 6 vbroadcast slots x 6 cycles

**Cost**:
- Loads: 1 (1 cycle)
- VALU: 32 broadcasts / 6 per cycle = 6 cycles
- **Total: 7 cycles**

**Current (H54)**: ~64 cycles for 16 desks x 4 loads each

**Savings: ~57 cycles**

### Round 1: Two-Value Arithmetic Selection

Indices are in {1, 2}. The bit that determines left (1) vs right (2) is `bit = val & 1` from Round 0.

**Operations**:
1. Load tree[1] and tree[2]: 2 loads = 1 cycle
2. Broadcast both to vector temps: 2 x 32 vectors = 64 broadcasts
3. Compute selection for each vector:
   - `diff = tree[2]_vec - tree[1]_vec` (1 vsub per vector)
   - `result = multiply_add(bit_vec, diff, tree[1]_vec)` (1 FMA per vector)

**Cost**:
- Loads: 2 (1 cycle)
- Broadcasts: 64 / 6 = 11 cycles (can overlap with loads after first)
- Arithmetic select: 32 vsub + 32 FMA = 64 ops / 6 = 11 cycles
- **Total: ~23 cycles**

**Current (H54)**: ~64 cycles for gather phase
**Savings: ~41 cycles**

### Round 2: Four-Value Selection Tree

Indices in {1, 2, 3, 4}. Need 2-level selection tree.

**Approach**:
```
# Level 1: Select within pairs using bit0
pair_1_2 = T1 + bit0 * (T2 - T1)    # Select between tree[1] and tree[2]
pair_3_4 = T3 + bit0 * (T4 - T3)    # Select between tree[3] and tree[4]

# Level 2: Select between pairs using bit1
result = pair_1_2 + bit1 * (pair_3_4 - pair_1_2)
```

**Cost Analysis**:
- Loads: 4 values (2 cycles)
- Broadcasts: 4 x 32 = 128 (22 cycles)
- Level 1 select: 64 ops (11 cycles)
- Level 2 select: 64 ops (11 cycles)
- **Total: ~46 cycles**

### General Pattern for Round N (N <= 7)

Unique indices: 2^N
Selection tree depth: N levels

**Cost Components**:
1. **Loads**: 2^N values / 2 per cycle = 2^(N-1) cycles
2. **Broadcasts**: 2^N x 32 vectors / 6 per cycle = 2^N x 32 / 6 cycles
3. **Selection**: N levels x 32 vectors x 2 ops / 6 per cycle

| Round | Loads (cy) | Broadcasts (cy) | Selection (cy) | Total (cy) |
|-------|------------|-----------------|----------------|------------|
| 0     | 1          | 6               | 0              | 7          |
| 1     | 1          | 11              | 11             | 23         |
| 2     | 2          | 22              | 22             | 46         |
| 3     | 4          | 43              | 32             | 79         |
| 4     | 8          | 86              | 43             | 137        |
| 5     | 16         | 171             | 54             | 241        |
| 6     | 32         | 342             | 64             | 438        |
| 7     | 64         | 683             | 75             | 822        |

### Rounds 8-15: The Problem

At Round 8+, unique indices can be up to 256 (same as batch size).

**Challenge**: 256 loads x 256 broadcasts = massive overhead

This is where wavefront processing **breaks down** for later rounds.

---

## 4. Hybrid Strategy: Wavefront + Desk Pipeline

### Proposal

Use wavefront processing for early rounds (0-4 or 0-5) where index count is small.
Use desk pipelining for later rounds (5+ or 6+) where indices diverge.

**Why Round 5 Cutoff?**
- Round 5 has 32 unique indices
- 32 loads = 16 cycles (still faster than 64 cycles for gather)
- Broadcasts become expensive at 32 x 32 = 1024 ops

**Alternative: Round 4 Cutoff**
- Round 4 has 16 unique indices
- More balanced load/broadcast/select ratio

### Cycle Estimate: Hybrid Approach

**Wavefront Rounds 0-4** (5 rounds):
Sum from table: 7 + 23 + 46 + 79 + 137 = 292 cycles

**Plus Hash Computation** for rounds 0-4:
- 256 elements x 5 rounds x ~12 VALU ops = 15,360 ops
- At 6 ops/cycle = 2,560 cycles (but overlapped with loads)
- Effective: ~256 x 5 x 2 = 2,560 / 6 = 427 cycles (highly parallelized)

**Desk Pipeline Rounds 5-15** (11 rounds):
Using H54 approach: 11/16 x 3,462 = ~2,380 cycles (rough estimate)

**Total Estimate**: 292 + 427 + 2,380 = ~3,100 cycles

**This is WORSE than H54 (3,462 cycles) for Round 5+ because:**
- The broadcast overhead dominates
- 32 broadcasts x 32 vectors = 1,024 ops = 171 cycles just for broadcasts

---

## 5. Critical Issue: Broadcast Overhead

The fundamental problem with wavefront processing:

**Each unique tree value must be broadcast to ALL 256 elements (32 vectors)**

At 6 broadcasts/cycle, broadcasting K unique values takes:
```
K x 32 / 6 = K x 5.33 cycles
```

This quickly exceeds the gather cost of H54:
- H54 gather for all elements: ~64 cycles (16 desks x 4 cycles)
- Wavefront broadcast for K=8: 8 x 5.33 = 43 cycles (competitive)
- Wavefront broadcast for K=16: 16 x 5.33 = 85 cycles (worse!)
- Wavefront broadcast for K=32: 32 x 5.33 = 171 cycles (much worse!)

**The approach becomes uncompetitive at K >= 16 (Round 4).**

---

## 6. Alternative: Partial Wavefront for Round 0-3 Only

### Optimized Early Rounds

| Round | Unique | Loads | Broadcasts | Select | Total | H54 Equivalent |
|-------|--------|-------|------------|--------|-------|----------------|
| 0     | 1      | 1     | 6          | 0      | 7     | ~64            |
| 1     | 2      | 1     | 11         | 11     | 23    | ~64            |
| 2     | 4      | 2     | 22         | 22     | 46    | ~64            |
| 3     | 8      | 4     | 43         | 32     | 79    | ~64            |
| **Sum** |      | **8** | **82**     | **65** | **155** | **~256**     |

**Savings from Rounds 0-3**: 256 - 155 = 101 cycles

**Remaining Rounds 4-15** (12 rounds):
12/16 x 3,462 = 2,597 cycles (proportional estimate)

**Total**: 155 + 2,597 + hash_overhead = ~2,900 cycles

Still not reaching 1,790 target.

---

## 7. Why The 2,048 Barrier Exists

### Fundamental Load Constraint

The problem requires:
- 256 elements x 16 rounds = 4,096 tree node lookups
- Each lookup requires 1 load
- 2 loads/cycle maximum
- **MINIMUM: 4,096 / 2 = 2,048 cycles just for loads**

### How Wavefront Could Break It

Wavefront processing can reduce TOTAL loads by reusing values:

| Round | Standard Loads | Wavefront Loads |
|-------|----------------|-----------------|
| 0     | 256            | 1               |
| 1     | 256            | 2               |
| 2     | 256            | 4               |
| 3     | 256            | 8               |
| 4     | 256            | 16              |
| 5     | 256            | 32              |
| 6     | 256            | 64              |
| 7     | 256            | 128             |
| 8-15  | 256 x 8 = 2048 | 256 x 8 = 2048  |
| **Total** | **4,096**  | **2,303**       |

**Potential load reduction**: 4,096 - 2,303 = 1,793 loads saved!

At 2 loads/cycle, this is **897 cycles saved**.

**Minimum with wavefront**: 2,303 / 2 = 1,152 cycles for loads alone

### But Broadcast Overhead Kills It

For rounds 0-7, we save 2,048 - 255 = 1,793 loads.
But we add: 1 + 2 + 4 + 8 + 16 + 32 + 64 + 128 = 255 unique values to broadcast.
Each broadcast to 32 vectors: 255 x 32 / 6 = 1,360 cycles of broadcasts!

**Net effect**: Save 897 load cycles, add 1,360 broadcast cycles = **WORSE by 463 cycles**

---

## 8. The Real Breakthrough: Selective Wavefront + In-Place Registers

### Key Observation

The broadcast overhead comes from needing to distribute values to ALL 32 vectors.

**What if we process fewer vectors per wavefront?**

### 8-Vector Wavefront (8 desks worth)

Process 8 vectors (64 elements) as a wavefront:
- Round 0: 1 load, 8 broadcasts (2 cycles)
- Round 1: 2 loads, 16 broadcasts (3 cycles), 16 select ops (3 cycles)
- Round 2: 4 loads, 32 broadcasts (6 cycles), 32 select ops (6 cycles)

Total for rounds 0-2: 2 + 6 + 12 = 20 cycles for 64 elements
H54 equivalent: 64/128 x ~64 = 32 cycles (for 2 rounds)

Wait, H54 does 128 elements x 2 rounds per iteration. Let me recalculate.

H54 per iteration: 128 elements x 2 rounds = 256 element-rounds
Cost: ~216 cycles (3,462 / 16 iterations)

Wavefront for 64 elements x 3 rounds = 192 element-rounds
Cost: 20 cycles (significantly cheaper)

**But we still need rounds 3-15 for these elements!**

---

## 9. Cycle Count Estimate: Best Case Wavefront

### Assumptions
- Wavefront for rounds 0-3 only (8 unique indices max)
- Desk pipeline for rounds 4-15 (12 rounds)
- Perfect overlap of broadcasts with selection ops

### Wavefront Phase (Rounds 0-3, all 256 elements)

Round 0: 1 load, 6 broadcast cycles, 0 select = 7 cycles
Round 1: 1 load, 11 broadcast cycles, 11 select cycles (parallel) = 12 cycles
Round 2: 2 loads, 22 broadcast cycles, 22 select cycles (parallel) = 24 cycles
Round 3: 4 loads, 43 broadcast cycles, 32 select cycles (parallel) = 47 cycles

Hash per round: 256 elements x 12 ops = 3,072 ops / 6 = 512 cycles per round
But highly parallelized across vectors.

**Subtotal for rounds 0-3**: ~90 + 512 x 4/overlap_factor = ~90 + 200 = ~290 cycles

### Desk Pipeline Phase (Rounds 4-15, 12 rounds)

H54 does 16 rounds in 3,462 cycles.
Scaling: 12/16 x 3,462 = 2,597 cycles

**But wait** - H54 also includes load/store overhead for reloading idx/val each iteration.
For wavefront, we keep idx/val in registers throughout.

Better estimate: 12/16 x (3,462 - setup_overhead) = ~2,400 cycles

### Total Estimate

290 + 2,400 = **~2,690 cycles**

**This is 22% faster than H54 (3,462) but still above 2,048 barrier.**

---

## 10. Can We Reach 1,790 Cycles?

### What Would Be Required

Target: 1,790 cycles
Current best: 3,462 cycles
Required improvement: 1.93x

### Path 1: Reduce Total Work

The problem has 256 x 16 = 4,096 element-round pairs.
Each needs: gather (or selection) + hash + branch.

At 1,790 cycles: 1,790 / 4,096 = 0.44 cycles per element-round.

**This requires sub-cycle throughput!**

### Path 2: Massive Parallelism

H54 uses 16 desks (128 elements at once, 32% of batch).
To hit 1,790: would need ~36 desks (288 elements, more than batch size!).

**Not feasible.**

### Path 3: Algorithmic Change

The hash function has 6 stages, each requiring VALU ops.
If hash could be simplified: would reduce compute significantly.

**Not allowed by problem constraints.**

### Path 4: Different ISA Interpretation

Perhaps there's a hidden ISA feature we're not exploiting?
- vgather (not available)
- Wider VLEN (fixed at 8)
- More load slots (fixed at 2)

**No hidden features found.**

---

## 11. Conclusion

### Arithmetic Value Selection: VALID

The formula `result = T1 + bit * (T2 - T1)` correctly selects between two values using VALU operations, bypassing the Flow engine bottleneck.

**Throughput**: 3 selections/cycle (vs 1 vselect/cycle)

### Wavefront Processing: LIMITED BENEFIT

The approach reduces total loads but adds broadcast overhead:
- **Beneficial for rounds 0-3** (1-8 unique indices)
- **Neutral for round 4** (16 unique indices)
- **Detrimental for rounds 5+** (32+ unique indices)

### Estimated Improvement

Best-case hybrid approach: ~2,690 cycles (22% faster than H54's 3,462)

### Cannot Reach 1,790 Cycles

**The 2,048 cycle barrier is fundamental.**

Even with perfect wavefront processing eliminating all redundant loads:
- Minimum loads: 2,303 / 2 = 1,152 cycles
- Hash computation: 256 x 16 x 12 ops / 6 = 8,192 cycles (before overlap)
- Even with 50% overlap: 4,096 cycles for hash alone

**The target of 1,790 cycles appears to be below the theoretical minimum** unless:
1. Hash function can be simplified (algorithm change)
2. There's an ISA feature we haven't discovered
3. The target was set with different problem parameters

---

## 12. Recommended Next Steps

1. **Implement Partial Wavefront (Rounds 0-3)**
   - Expected improvement: ~200-400 cycles
   - Complexity: Medium
   - Risk: Low

2. **Profile H54 for Remaining Slack**
   - Are there empty slots in VALU during load phases?
   - Can we overlap more compute with loads?

3. **Explore Hash Optimization**
   - Can any hash stages be combined or eliminated?
   - Are there algebraic identities we can exploit?

4. **Re-examine ISA for Hidden Features**
   - Review problem.py for any overlooked instructions
   - Check for indirect addressing modes

---

## Appendix A: Cycle Count Formulas

### Wavefront Round N Cost

```
loads = 2^N / 2 cycles
broadcasts = 2^N * 32 / 6 cycles
selection = N * 32 * 2 / 6 cycles
total = loads + max(broadcasts, selection)
```

### Comparison Point

H54 per 2-round iteration: 3,462 / 16 = 216 cycles for 128 elements

### Load Minimum

Standard: 4,096 / 2 = 2,048 cycles
Wavefront: (1+2+4+8+16+32+64+128+256*8) / 2 = 2,303 / 2 = 1,152 cycles

---

## Appendix B: ISA Quick Reference

| Engine | Slots/Cycle | Key Operations |
|--------|-------------|----------------|
| Load   | 2           | load, vload, const |
| Store  | 2           | store, vstore |
| VALU   | 6           | vbroadcast, multiply_add, +, -, *, ^, &, <<, >> |
| ALU    | 12          | scalar arithmetic |
| Flow   | 1           | vselect, select, jumps |

VLEN = 8 (vector length)
SCRATCH_SIZE = 1,536 words
