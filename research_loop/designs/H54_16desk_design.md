# H54: 16-Desk Ultra-Deep Pipeline Design

## Executive Summary

This design document analyzes the feasibility of doubling the desk count from 8 to 16 for improved instruction-level parallelism. The analysis concludes that **16 desks will fit in scratch memory** but provides **diminishing returns** due to the load slot bottleneck. The estimated cycle count is approximately **3,400-3,600 cycles** (vs 4,062 for H38), still well above the 1,790 target.

---

## 1. Scratch Memory Analysis

### 1.1 Current H38 Memory Layout (8 Desks)

From the H38 implementation, per-desk allocation:
```
Per desk: idx(8) + val(8) + node_val(8) + addr(8) + tmp1(8) + tmp2(8) = 48 words
8 desks: 48 x 8 = 384 words
```

Additional allocations in H38:
- Init variables: 7 scalars (7 words)
- Scalar temps: 2 words (tmp_scalar, addr_scalar)
- Vector constants: v_zero, v_one, v_two, v_n_nodes = 4 x 8 = 32 words
- Hash constants: 6 stages x (const + shift) = 12 vectors = 96 words
- FMA multipliers: 3 stages x 8 = 24 words
- Offset constants: 8 scalars = 8 words
- Address temporaries: 16 scalars = 16 words
- Offset registers: 8 scalars = 8 words
- Loop control: 2 scalars = 2 words
- Miscellaneous const scalars: ~15 words

**H38 Total**: ~592 words (reported as ~634 in comments, actual ~592)

### 1.2 Projected 16-Desk Memory Requirements

**Per-desk allocations (16 desks)**:
```
Per desk: idx(8) + val(8) + node_val(8) + addr(8) + tmp1(8) + tmp2(8) = 48 words
16 desks: 48 x 16 = 768 words
```

**Constant storage (unchanged)**:
- Init variables: 7 words
- Scalar temps: 2 words
- Vector constants: 32 words
- Hash constants: 96 words
- FMA multipliers: 24 words
- Loop control: 2 words
- Miscellaneous const scalars: ~15 words

**Scaling-dependent allocations**:
- Offset constants: 16 scalars = 16 words
- Address temporaries: 32 scalars = 32 words (for 16 pairs of idx/val addresses)
- Offset registers: 16 scalars = 16 words

### 1.3 Total for 16 Desks

| Category | H38 (8 desk) | H54 (16 desk) |
|----------|--------------|---------------|
| Desk vectors | 384 | 768 |
| Constants | ~178 | ~178 |
| Address temps | 16 | 32 |
| Offset regs/consts | 16 | 32 |
| **Total** | **~594** | **~1010** |

**Result: 1,010 words < 1,536 words. 16 desks fit!**

Available margin: ~526 words for any additional optimizations.

---

## 2. Architecture Bottleneck Analysis

### 2.1 Load Slot Bottleneck (Critical!)

The ISA constraint of **2 load slots per cycle** is the fundamental bottleneck:

**Gather operation**: Each desk needs 8 scalar loads to gather node values
- 8 loads at 2/cycle = 4 cycles per desk gather
- 16 desks x 4 cycles = 64 cycles minimum for all gathers

**Input loading**: Each desk needs idx and val vectors (2 vloads)
- 16 desks x 2 vloads = 32 vloads
- At 2/cycle = 16 cycles

**Total minimum load cycles per iteration**: 64 + 16 = 80 cycles

### 2.2 Store Slot Analysis

**Output storing**: Each desk needs idx and val vectors (2 vstores)
- 16 desks x 2 vstores = 32 vstores
- At 2/cycle = 16 cycles per iteration

### 2.3 VALU Analysis

Compute per desk per round:
- XOR with node_val: 1 op
- 6 hash stages: ~10 ops (3 FMA + 7 component ops)
- Branch computation: 4 ops
- Bounds check: 2 ops
- Address preparation: 2 ops

Total: ~19 VALU ops per desk per round

For 16 desks x 2 rounds = 608 VALU ops per iteration
At 6 slots/cycle = 102 cycles minimum compute

**However**, VALU ops can be overlapped with loads. With 16 desks, we have much better overlap potential.

---

## 3. Cycle Count Estimation

### 3.1 H38 Structure Analysis

H38 achieves 4,062 cycles for:
- batch_size = 256
- rounds = 16
- 8 desks with 2-round fusion
- total_iterations = (256/8) * (16/2) / 8 = 32

Cycles per iteration in H38: ~127 cycles

### 3.2 Projected H54 Structure

With 16 desks:
- total_iterations = (256/8) * (16/2) / 16 = 16

**Per-iteration breakdown for 16 desks**:

1. **Address calculation**: 3-4 cycles (ALU parallel)
2. **Input loading**: 16 vloads at 2/cycle = 8 cycles
3. **Address prep for gather**: 4 cycles (vbroadcast + vadd for 16 desks)
4. **Gather Phase Round 1**:
   - 16 desks x 4 cycles = 64 cycles
   - Heavily overlapped with hash computation
   - Effective: ~48 cycles (with compute overlap)
5. **Hash/Branch cleanup Round 1**: ~14 cycles (for trailing desks)
6. **Address prep Round 2**: 4 cycles
7. **Gather Phase Round 2**: ~48 cycles (with overlap)
8. **Hash/Branch cleanup Round 2**: ~14 cycles
9. **Store address calculation**: 2 cycles
10. **Output storing**: 16 vstores x 2/cycle = 16 cycles (can be 32 if idx+val separate)
11. **Loop control**: 3 cycles

**Estimated per-iteration**: ~175-200 cycles

**Total for 16 iterations**: 16 x 190 = ~3,040 cycles

Adding setup overhead (~200 cycles): **~3,240 cycles**

### 3.3 Conservative Estimate with Realistic Overlap

Considering:
- Not all VALU ops can perfectly overlap with loads
- Pipeline stalls from data dependencies
- ALU contention with address calculations

**Realistic estimate: 3,400-3,600 cycles**

Speedup over H38: 4,062 / 3,500 = 1.16x

---

## 4. Comparison: Why Not More Desks?

### 4.1 Maximum Desk Count

Memory constraint: 1,536 words available

Formula: `48 * desks + 178 + 2*desks + 2*desks = 1536`
         `52 * desks = 1358`
         `desks = 26.1`

**Maximum desks that fit: 26**

### 4.2 Diminishing Returns Analysis

| Desks | Memory Used | Gather Cycles | Est. Total Cycles |
|-------|-------------|---------------|-------------------|
| 8     | 594         | 32            | 4,062 (actual)    |
| 16    | 1,010       | 64            | ~3,400-3,600      |
| 20    | 1,218       | 80            | ~3,300-3,500      |
| 24    | 1,426       | 96            | ~3,200-3,400      |

Beyond 16 desks, the load bottleneck means more desks provide minimal benefit:
- More gathers to execute (2 loads/cycle limit)
- More iterations means less amortization of setup costs
- Code complexity increases substantially

---

## 5. Alternative Approaches to Reach 1,790 Cycles

The 1,790 cycle target implies ~14x parallelism (147,734 / 10,500 per iteration).

### 5.1 Why Current Approach Cannot Reach Target

With the load slot limit (2/cycle), the theoretical minimum for gather operations:
- 256 elements x 16 rounds x 1 gather per element = 4,096 total gathers
- Each gather = 8 loads = 4 cycles
- Total gather cycles = 4,096 x 4 / parallelism_factor

Even with perfect 8-way batching: 4,096 x 4 / 8 = 2,048 cycles minimum just for gathers.

### 5.2 Potential Strategies Beyond More Desks

1. **Data Locality Optimization**: Pre-sort indices to increase cache hits (not applicable in this model)

2. **Algorithmic Change**: Different hash function with fewer operations

3. **Round Fusion > 2**: Fuse 4 rounds per iteration instead of 2
   - Reduces load/store overhead
   - Increases register pressure significantly

4. **Speculative Execution**: Pre-compute both branches of tree traversal
   - Doubles compute but could hide latency better
   - Memory requirements explode

---

## 6. Implementation Approach for H54

### 6.1 Key Changes from H38

1. **Double desk allocation**: 16 desks instead of 8
2. **Extended address temp arrays**: 32 instead of 16
3. **Adjusted loop count**: 16 iterations instead of 32
4. **Deeper pipeline interleaving**: More aggressive overlap of gather with hash

### 6.2 Pipeline Schedule

```
Cycle | Load Slots | VALU Slots | Notes
------+------------+------------+------------------------
0-7   | vload 0-15 | -          | Load all idx/val pairs
8-11  | gather[0]  | -          | Desk 0 gather (4 cyc)
12-15 | gather[1]  | xor/hash 0 | Desk 1 gather + desk 0 process
16-19 | gather[2]  | hash 0-1   | Desk 2 gather + desk 0-1 process
20-23 | gather[3]  | hash 0-2   | Desk 3 gather + desk 0-2 process
...   | ...        | ...        | Continue interleaving
60-63 | gather[15] | hash 11-14 | Desk 15 gather + desk 11-14 process
64-77 | -          | hash 12-15 | Finish desk 12-15 processing
```

### 6.3 Code Structure

```python
NUM_DESKS = 16

# Desk allocation (768 words)
desks = []
for d in range(NUM_DESKS):
    desk = {
        'idx': alloc_scratch(f"v_idx_{d}", VLEN),
        'val': alloc_scratch(f"v_val_{d}", VLEN),
        'node_val': alloc_scratch(f"v_node_{d}", VLEN),
        'addr': alloc_scratch(f"v_addr_{d}", VLEN),
        'tmp1': alloc_scratch(f"v_tmp1_{d}", VLEN),
        'tmp2': alloc_scratch(f"v_tmp2_{d}", VLEN),
    }
    desks.append(desk)

# Address temps (32 words)
addr_tmp = [alloc_scratch(f"addr_tmp_{i}") for i in range(32)]

# Main loop: 16 iterations instead of 32
total_iterations = (batch_size // VLEN) * (rounds // 2) // NUM_DESKS  # = 16
```

---

## 7. Conclusions and Recommendations

### 7.1 Feasibility: YES

16 desks fit in 1,536 words with 526 words to spare.

### 7.2 Expected Performance

- **Estimated cycles**: 3,400-3,600
- **Speedup over H38**: ~1.15x
- **Gap to target**: Still 1.9x above 1,790 target

### 7.3 Recommendation

**Implement H54 as an incremental improvement**, but recognize it will not reach the target. The fundamental load slot bottleneck (2/cycle) creates a hard floor around 2,000-2,500 cycles that cannot be overcome by adding more desks.

To reach 1,790 cycles, a fundamentally different approach is needed:
- Algorithmic changes (different traversal pattern)
- Data structure changes (non-implicit tree)
- Instruction set exploitation (finding overlooked parallelism)

### 7.4 Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Memory overflow | Low | High | Validated: 1,010 < 1,536 |
| Pipeline stalls | Medium | Medium | Careful scheduling |
| Code complexity | High | Medium | Systematic helper functions |
| Correctness bugs | Medium | High | Incremental testing |

---

## Appendix A: Detailed Memory Map for 16 Desks

```
Address Range | Size  | Contents
--------------+-------+----------------------------------
0-1           | 2     | tmp_scalar, addr_scalar
2-8           | 7     | init vars (rounds, n_nodes, etc.)
9-40          | 32    | v_zero, v_one, v_two, v_n_nodes
41-136        | 96    | v_hash_consts[0-5], v_hash_shifts[0-5]
137-160       | 24    | v_fma_mult[0,2,4]
161-176       | 16    | offset_consts[0-15]
177-208       | 32    | addr_tmp[0-31]
209-224       | 16    | offset_regs[0-15]
225-226       | 2     | batch_offset, iter_counter
227-242       | 16    | misc constants
243-1010      | 768   | desks[0-15] (48 words each)
--------------+-------+----------------------------------
Total: ~1,010 words
Available: 526 words margin
```

## Appendix B: Load Slot Utilization Analysis

```
Phase           | Load Operations | Cycles
----------------+-----------------+--------
Input loading   | 32 vloads       | 16
Round 1 gather  | 128 loads       | 64
Round 2 gather  | 128 loads       | 64
----------------+-----------------+--------
Total per iter  | 288 loads       | 144 cycles (load-bound)
```

Note: With 2 load slots/cycle, 288 loads require minimum 144 cycles.
H54 target per iteration: ~190 cycles (32% overhead for compute/store/control).
