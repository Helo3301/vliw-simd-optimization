# H48: Speculative Index Precomputation Design

## Executive Summary

This document analyzes the feasibility of speculatively computing both possible next indices (left and right child) in parallel with the hash computation. The goal is to hide hash latency by having the next gather address ready immediately when the hash result determines the branch direction.

## Current Performance Context

- **Current best**: H54 at 3,462 cycles
- **Target**: 1,790 cycles
- **Gap**: 1,672 cycles (48% reduction needed)
- **Key bottleneck identified**: Hash computation creates latency before the next gather can be addressed

## Problem Analysis

### Current Critical Path in H54

The current flow for each desk per round:

```
Cycle N:   Load node_val via gather (4 cycles for 8 lanes at 2 loads/cycle)
Cycle N+4: XOR val with node_val
Cycle N+5: Begin hash computation (12 VALU ops over multiple cycles)
...
Cycle N+X: Hash complete, extract bit = hash_result & 1
Cycle N+X+1: Compute next_idx = idx*2 + 1 + bit
Cycle N+X+2: Compute gather_addr = forest_base + next_idx
Cycle N+X+3: Start next gather
```

The hash computation (12 VALU operations) creates a serial dependency:
- We cannot compute `next_idx` until we know `bit`
- We cannot start the next gather until we have `gather_addr`

### Hash Latency Breakdown (from H55 analysis)

Per hash computation (already optimized with FMA):
- Stage 0: 1 FMA operation
- Stage 1: 2 prep ops + 1 combine = 3 ops
- Stage 2: 1 FMA operation
- Stage 3: 2 prep ops + 1 combine = 3 ops
- Stage 4: 1 FMA operation
- Stage 5: 2 prep ops + 1 combine = 3 ops
- **Total: 12 VALU operations per hash**

With data dependencies between stages, the minimum hash latency is:
- 6 stages with dependencies
- Each non-FMA stage requires 2 cycles (prep + combine)
- FMA stages require 1 cycle each
- **Minimum hash latency: ~9-10 cycles** (when fully saturating VALU slots)

## Speculative Precomputation Proposal

### Core Idea

Instead of waiting for `bit = hash_result & 1`, speculatively compute BOTH possible outcomes:

```
1. Load node_val at current idx
2. IN PARALLEL with hash:
   - idx_left = idx*2 + 1      (child if bit=0)
   - idx_right = idx*2 + 2     (child if bit=1)
   - addr_left = forest_base + idx_left
   - addr_right = forest_base + idx_right
3. Compute hash (concurrent with step 2)
4. After hash: bit = hash_result & 1
5. SELECT correct idx/addr based on bit:
   - final_idx = bit ? idx_right : idx_left
   - final_addr = bit ? addr_right : addr_left
```

### Operation Analysis

**Current index computation (serial):**
```
1. AND: bit = hash_result & 1         (1 VALU)
2. FMA: tmp = idx*2 + 1               (1 VALU)
3. ADD: next_idx = tmp + bit          (1 VALU)
4. Bounds check: cmp = idx < n_nodes  (1 VALU)
5. Bounds apply: idx = idx * cmp      (1 VALU)
Total: 5 VALU operations, SERIAL after hash
```

**Speculative computation (parallel with hash):**
```
Parallel with hash stages 0-2:
1. FMA: idx_left = idx*2 + 1          (1 VALU)
2. ADD: idx_right = idx_left + 1      (1 VALU)  // or FMA: idx*2 + 2

Parallel with hash stages 3-5:
3. ADD: addr_left = forest_base + idx_left   (1 VALU)
4. ADD: addr_right = forest_base + idx_right (1 VALU)

After hash completes:
5. AND: bit = hash_result & 1         (1 VALU)
6. SELECT: final_idx = vselect(bit, idx_right, idx_left)   (1 flow)
7. SELECT: final_addr = vselect(bit, addr_right, addr_left) (1 flow)
8. Bounds check: cmp = final_idx < n_nodes  (1 VALU)
9. Bounds apply: final_idx = final_idx * cmp (1 VALU)

Total: 6 VALU + 2 flow operations
```

### Cycle Benefit Analysis

**Current H54 approach:**
- Hash latency: ~10 cycles (with interleaving)
- Index computation: 5 cycles AFTER hash completes
- Next gather can start: +5 cycles after hash

**Speculative approach:**
- Hash latency: ~10 cycles
- Index/addr computation: PARALLEL with hash (hidden)
- After hash: 3 cycles (AND + 2 SELECT + bounds)
- Next gather can start: +3 cycles after hash

**Potential savings per desk: 2-3 cycles**

## Register Pressure Analysis

### Current H54 per-desk registers:
```python
desk = {
    'idx': 8 words,      # vector index
    'val': 8 words,      # vector value
    'node_val': 8 words, # loaded tree node value
    'addr': 8 words,     # gather address
    'tmp1': 8 words,     # hash temporary
    'tmp2': 8 words,     # hash temporary
}
Total: 48 words per desk
```

### Speculative approach per-desk:
```python
desk = {
    'idx': 8 words,
    'val': 8 words,
    'node_val': 8 words,
    'addr': 8 words,
    'tmp1': 8 words,
    'tmp2': 8 words,
    'idx_left': 8 words,    # NEW: speculative left index
    'idx_right': 8 words,   # NEW: speculative right index
    'addr_left': 8 words,   # NEW: speculative left address
    'addr_right': 8 words,  # NEW: speculative right address
}
Total: 80 words per desk (+32 words)
```

### Scratch Budget Impact

**Current H54 (16 desks):**
- Per desk: 48 words
- 16 desks: 768 words
- Constants: ~178 words
- Temporaries: ~64 words
- **Total: ~1010 words (fits in 1536)**

**With speculation (16 desks):**
- Per desk: 80 words
- 16 desks: 1280 words
- Constants: ~178 words
- Temporaries: ~64 words
- **Total: ~1522 words (TIGHT fit in 1536)**

**Alternative: 12 desks with speculation:**
- Per desk: 80 words
- 12 desks: 960 words
- Constants: ~178 words
- Temporaries: ~64 words
- **Total: ~1202 words (comfortable fit)**

## Interaction with Desk Pipelining

### Current H54 Interleaving Strategy

H54 achieves high throughput by interleaving operations across 16 desks:

```
Cycle 1: Desk 0 gather
Cycle 2: Desk 0 gather + Desk 1 gather starts
...
Cycle 4: Desk 0 XOR + hash stage 0 | Desk 1 gather | Desk 2 gather
Cycle 5: Desk 0 hash stage 1 | Desk 1 XOR | Desk 2 gather | Desk 3 gather
...
```

The interleaving means each desk has ~16 cycles between when it needs the hash result and when the next iteration starts.

### Impact of Speculation on Pipeline

**Positive**: The speculative idx/addr computation can be scheduled during cycles when VALU slots are underutilized:
- During gather (2 load ops, 0 VALU)
- During address computation cycles

**Negative**: The extra SELECT operations add to the critical path AFTER hash completes.

**Neutral**: The speculative computation doesn't conflict with hash stages since it uses different source registers (idx vs val).

## Quantitative Cycle Savings Estimate

### Per-Iteration Analysis (16 desks, 2 rounds)

**Current H54 main loop body (approximate):**
```
Address setup: 5 cycles
Input vloads (16 desks): 16 cycles
Gather address prep: 6 cycles
Round 1 gather+hash: ~80 cycles (interleaved)
Round 1 to Round 2 address prep: 6 cycles
Round 2 gather+hash: ~80 cycles (interleaved)
Store address prep: 3 cycles
Output vstores (16 desks): 16 cycles
Loop control: 3 cycles
Total: ~215 cycles per iteration
```

With 16 iterations: 16 * 215 = 3,440 cycles (close to measured 3,462)

**Speculative approach savings:**
- Per desk per round: Save ~2 cycles on the critical path
- But only affects the LAST desk in the pipeline (others are hidden)
- Net benefit: ~2-4 cycles per round
- Per iteration (2 rounds): ~4-8 cycles
- Total (16 iterations): ~64-128 cycles

**Estimated new cycle count: 3,334 - 3,398 cycles**

This is a **~2-4% improvement**, meaningful but not transformative.

## Alternative: Speculative Memory Prefetch

A more aggressive approach would be to actually issue BOTH gather operations speculatively:

```
1. Compute both addr_left and addr_right
2. Issue gather to BOTH addresses (4 loads instead of 2 per cycle)
3. After hash: SELECT the correct node_val
```

**Problems:**
- Load unit can only do 2 ops/cycle
- Would double the gather bandwidth requirement
- With 16 desks, we're already load-bound

**Verdict**: Not feasible without architectural changes.

## Implementation Sketch

### Register Allocation
```python
desk = {
    'idx': self.alloc_scratch(f"v_idx_{d}", VLEN),
    'val': self.alloc_scratch(f"v_val_{d}", VLEN),
    'node_val': self.alloc_scratch(f"v_node_{d}", VLEN),
    'addr': self.alloc_scratch(f"v_addr_{d}", VLEN),
    'tmp1': self.alloc_scratch(f"v_tmp1_{d}", VLEN),
    'tmp2': self.alloc_scratch(f"v_tmp2_{d}", VLEN),
    # New speculative registers
    'idx_left': self.alloc_scratch(f"v_idx_l_{d}", VLEN),
    'idx_right': self.alloc_scratch(f"v_idx_r_{d}", VLEN),
    'addr_left': self.alloc_scratch(f"v_addr_l_{d}", VLEN),
    'addr_right': self.alloc_scratch(f"v_addr_r_{d}", VLEN),
}
```

### Speculative Computation Emission
```python
def emit_speculative_idx(desk_idx):
    """Emit speculative index computation - schedule parallel with early hash stages"""
    d = desks[desk_idx]
    return [
        ("multiply_add", d['idx_left'], d['idx'], v_two, v_one),   # idx*2 + 1
        ("+", d['idx_right'], d['idx_left'], v_one),               # idx*2 + 2
    ]

def emit_speculative_addr(desk_idx):
    """Emit speculative address computation - schedule parallel with later hash stages"""
    d = desks[desk_idx]
    return [
        ("+", d['addr_left'], v_forest_base, d['idx_left']),
        ("+", d['addr_right'], v_forest_base, d['idx_right']),
    ]

def emit_speculative_select(desk_idx):
    """Select correct path after hash completes"""
    d = desks[desk_idx]
    # bit is in d['tmp1'] from AND operation
    return []  # Use flow vselect instead

def emit_branch_ops_speculative(desk_idx):
    """Modified branch operations for speculative approach"""
    d = desks[desk_idx]
    return [
        ("&", d['tmp1'], d['val'], v_one),  # bit = hash_result & 1
    ]
    # vselect operations would go in flow engine
```

### Pipeline Integration

The speculative computation should be scheduled in cycles where VALU has spare capacity:

```python
# Example interleaving with speculation
self.instrs.append({
    "load": [
        ("load", desks[2]['node_val'], desks[2]['addr']),
        ("load", desks[2]['node_val'] + 1, desks[2]['addr'] + 1),
    ],
    "valu": emit_hash_stage(0, 2) + emit_xor_node(1) + emit_speculative_idx(0),
    # Note: emit_speculative_idx can run in parallel since it uses idx, not val
})
```

## Risk Assessment

### Low Risk:
- Register allocation: Tight but feasible with 16 desks, comfortable with 12
- Correctness: SELECT operation is straightforward

### Medium Risk:
- Pipeline disruption: Adding operations may shift scheduling in unexpected ways
- VALU contention: Speculative ops compete with hash ops for 6 VALU slots

### High Risk:
- Net negative: The extra operations might cost more than they save if VALU becomes the bottleneck
- Breaking interleaving: Current pipeline relies on specific timing

## Recommendations

### Finding 1: Marginal Benefit Expected
The speculative index precomputation offers only 2-4% cycle reduction (~64-128 cycles out of 3,462). This is because:
- H54's deep pipelining (16 desks) already hides most hash latency
- The savings only benefit the critical path at pipeline boundaries

### Finding 2: Register Pressure Concerns
Adding 32 words per desk significantly impacts scratch budget:
- With 16 desks: 1522/1536 words (97% usage, risky)
- Recommended: Reduce to 12-14 desks if implementing

### Finding 3: Limited Impact on Target Gap
- Current: 3,462 cycles
- Expected with H48: ~3,350 cycles
- Target: 1,790 cycles
- This brings us from 93.3% of gap to 87.0% of gap remaining

### Verdict: LOW PRIORITY

The speculative index precomputation is mathematically sound but offers limited practical benefit given H54's existing pipeline depth. The cycles saved are small compared to the target gap.

**Higher priority optimizations should focus on:**
1. Memory access patterns (gather coalescing)
2. Loop overhead reduction
3. Fundamentally different algorithmic approaches (wavefront, index grouping)

### When H48 Might Be Valuable:
- If reducing desk count (e.g., to 8 desks due to other constraints), speculation becomes more valuable
- If combined with other optimizations that reduce hash interleaving depth
- As a micro-optimization after larger gains are achieved

## Appendix: VALU Slot Utilization Analysis

### Current H54 Per-Cycle VALU Usage

During steady-state interleaved execution:

| Cycle Type | Operations | VALU Slots Used |
|------------|------------|-----------------|
| Hash FMA stage | 2-3 desks at FMA | 2-3 |
| Hash XOR prep | 2 ops per desk | 4 |
| Hash XOR combine | 1 op per desk | 2-3 |
| XOR node_val | 1 op per desk | 1-2 |
| Branch ops | 2 ops per desk | 2-3 |
| Bounds check | 2 ops per desk | 2 |

Average: 3-4 VALU slots used per cycle (out of 6)

### With Speculation Added

| Cycle Type | Additional Operations | New Total |
|------------|----------------------|-----------|
| Speculative idx | +2 per desk | 5-6 |
| Speculative addr | +2 per desk | 5-6 |
| Select (flow) | 0 VALU (uses flow) | same |

This approaches the 6-slot limit but doesn't exceed it, suggesting speculation is feasible from a VALU perspective.

## Conclusion

H48 Speculative Index Precomputation is a valid optimization technique that provides modest cycle savings (2-4%) by computing both possible branch targets in parallel with the hash function. However, given H54's deep 16-desk pipeline that already effectively hides hash latency through interleaving, the practical benefit is limited.

**Recommendation**: Mark as low priority. The 1,672-cycle gap to target requires more fundamental changes than this micro-optimization can provide. Consider implementing only after larger algorithmic improvements are exhausted or if circumstances (reduced desk count) increase its relative value.
