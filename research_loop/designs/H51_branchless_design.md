# H51: CAPT-Style Branchless Traversal Design

## Status: ANALYSIS COMPLETE - Already Implemented in H38

## Executive Summary

Upon detailed analysis of the H38 implementation, the CAPT-style branchless traversal is **already implemented**. The current implementation uses:
```python
idx_next = idx * 2 + 1 + bit
```
Where `bit = val & 1` (0 or 1 from hash).

This is achieved via VALU operations, NOT flow engine vselect.

---

## 1. Paper Insight (arxiv 2406.02807 - CAPT)

The CAPT paper proposes branchless tree traversal:
```
idx = idx * 2 + 1 + bit
```
Where:
- `bit in {0, 1}` determines left vs right child
- Left child: `idx * 2 + 1` (bit = 0)
- Right child: `idx * 2 + 2` (bit = 1)
- No conditional branch needed - pure arithmetic

---

## 2. Current Implementation Analysis (H38)

### Branch Operations in H38

From lines 356-378 of `perf_takehome_h38.py`:

```python
def emit_branch_ops(desk_idx):
    """Emit branch operations for a desk"""
    d = desks[desk_idx]
    ops = []
    # AND, FMA for idx, add branch bit, bounds check, multiply bypass
    ops.append(("&", d['tmp1'], d['val'], v_one))           # bit = val & 1
    ops.append(("multiply_add", d['idx'], d['idx'], v_two, v_one))  # idx = idx*2 + 1
    return ops

def emit_branch_add(desk_idx):
    """Add branch bit to idx"""
    d = desks[desk_idx]
    return [("+", d['idx'], d['idx'], d['tmp1'])]           # idx = idx + bit
```

This is **exactly** the CAPT-style branchless traversal:
1. `bit = val & 1` - VALU (extract branch direction from hash)
2. `idx = idx * 2 + 1` - VALU (multiply_add)
3. `idx = idx + bit` - VALU (vector add)

### Flow Engine Usage

The only `select` usage in H38 is for **loop control** (line 1007):
```python
("select", batch_offset, tmp_scalar, batch_offset, zero_const)
```

This is a scalar `select` (not `vselect`), used for batch wraparound - NOT for tree traversal.

### No vselect in Inner Loop

The inner tree traversal uses **zero vselect operations**. All branch decisions are arithmetic.

---

## 3. Exact VALU Operations for Branch Decision

### Current Implementation (H38)

| Cycle | Operation | Engine | Description |
|-------|-----------|--------|-------------|
| 1 | `vand(tmp1, val, v_one)` | VALU | bit = val & 1 |
| 1 | `multiply_add(idx, idx, v_two, v_one)` | VALU | idx = idx*2 + 1 |
| 2 | `vadd(idx, idx, tmp1)` | VALU | idx = idx + bit |
| 3 | `vlt(tmp1, idx, v_n_nodes)` | VALU | bounds_valid = idx < n_nodes |
| 4 | `vmul(idx, idx, tmp1)` | VALU | idx = idx * bounds_valid (clamp to 0 if OOB) |

### Total: 4 cycles for complete branch + bounds check

All 4 operations use VALU (6 slots/cycle), NOT flow engine (1 slot/cycle).

---

## 4. Cycle Count Analysis

### Theoretical Cost Comparison

| Approach | Operations | Engine | Cycles |
|----------|-----------|--------|--------|
| **Conditional vselect** | | | |
| - Compute left child | vmul, vadd | VALU | 1 |
| - Compute right child | vmul, vadd | VALU | 1 |
| - Compare hash bit | vand | VALU | 1 |
| - Select child (vselect) | vselect | Flow | 1 |
| - Bounds check (vselect) | vselect | Flow | 1 |
| **Total** | | | **5 cycles** |
| | | | |
| **Branchless (CAPT)** | | | |
| - Extract bit | vand | VALU | |
| - Compute idx*2+1 | multiply_add | VALU | 1 (parallel) |
| - Add bit | vadd | VALU | 1 |
| - Bounds check | vlt | VALU | 1 |
| - Bounds apply | vmul | VALU | 1 |
| **Total** | | | **4 cycles** |

### Savings: 1 cycle per desk per round

With 8 desks, 16 rounds, 32 iterations: `1 * 8 * 16 * 32 / 2 = 2048` cycles saved
(Divide by 2 because rounds are fused in pairs)

**But this is already implemented in H38!**

---

## 5. Current Bottleneck Analysis

Since branchless traversal is already implemented, the remaining bottlenecks are:

### Primary Bottleneck: Gather Latency
- 8 scalar loads per desk (VLEN=8)
- 2 load slots per cycle
- **Minimum 4 cycles per desk for gather**

### Secondary Bottleneck: Hash Computation
- 6 hash stages
- Mixed FMA and XOR operations
- ~6 cycles per desk (pipelined with gather)

### Loop Control Bottleneck
- Single scalar `select` for batch wraparound (1 cycle)
- `cond_jump` for loop continuation (1 cycle)

---

## 6. Potential Further Optimizations

Since H51's core idea is already implemented, consider these alternatives:

### A. Eliminate Bounds Check Entirely (Risky)

Current bounds check:
```python
tmp1 = idx < n_nodes
idx = idx * tmp1
```

If tree is power-of-2 complete and indices never overflow, we could remove this.
**Savings: 2 cycles per desk**

**Risk**: Incorrect results if tree is not complete.

### B. Fuse Bounds Check with Branch

Instead of separate bounds check, incorporate into index computation:
```python
# Current: idx = (idx*2+1+bit) * (idx*2+1+bit < n_nodes)
# Alternative: Use tree structure to guarantee valid indices
```

### C. Speculative Both-Child Prefetch

Compute both children, prefetch both, use whichever is needed:
```python
left_idx = idx * 2 + 1
right_idx = idx * 2 + 2
# Prefetch both, select after hash completes
```

**Potential gain**: Hides some hash latency
**Cost**: 2x load bandwidth

### D. Multi-Level Jump

For known hash sequences, compute multiple levels at once:
```python
# Instead of: idx = idx*2 + 1 + bit0, then idx = idx*2 + 1 + bit1
# Compute: idx = idx*4 + 3 + bit0*2 + bit1
```

**Requires**: Pre-computing next hash bits (pipeline)

---

## 7. Pseudocode for Current Inner Loop

```python
# Current H38 inner loop per desk (2 rounds fused)

# ROUND 1
# Gather node value (4 cycles - load bottleneck)
for lane in range(0, 8, 2):
    node_val[lane] = gather(forest + idx[lane])
    node_val[lane+1] = gather(forest + idx[lane+1])

# XOR and hash (overlapped with next desk's gather)
val = val ^ node_val                    # VALU: vxor

# 6 hash stages (FMA and XOR)
for stage in hash_stages:
    val = hash_step(val)                # VALU: multiply_add or vxor/vshift

# Branchless branch (THIS IS H51!)
bit = val & 1                           # VALU: vand
idx = idx * 2 + 1                       # VALU: multiply_add
idx = idx + bit                         # VALU: vadd

# Bounds check
valid = idx < n_nodes                   # VALU: vlt
idx = idx * valid                       # VALU: vmul (0 if OOB)

# Prepare next gather address
addr = forest_base + idx                # VALU: vadd

# ROUND 2 (same as round 1, no reload)
# ... repeat without memory reload between rounds
```

---

## 8. Conclusion

**H51 (CAPT-Style Branchless Traversal) is ALREADY IMPLEMENTED in H38.**

The current implementation achieves:
- Zero vselect operations in inner loop
- Pure VALU arithmetic for branch decisions
- 4-cycle branch+bounds per desk (vs theoretical 5 with vselect)

### Recommendations

1. **Mark H51 as COMPLETED** - core idea already in codebase
2. **Focus on other bottlenecks**:
   - Gather latency (4 cycles/desk minimum)
   - Loop overhead reduction
   - Deeper pipeline (more desks)
3. **Consider H51 variants** (A-D above) as new hypotheses

---

## 9. Implementation Status

| Component | Status | File |
|-----------|--------|------|
| Branchless traversal | IMPLEMENTED | H38, lines 356-378 |
| Bounds check via multiply | IMPLEMENTED | H38, lines 376-379 |
| Flow-free inner loop | IMPLEMENTED | H38 (only scalar select for loop control) |

**Current Best Performance**: 4,062 cycles (H38 with 8 desks)
**Target**: 1,790 cycles
**Gap**: 2.27x - requires fundamentally different approach

---

## Appendix: ISA Reference

| Engine | Slots/Cycle | Operations |
|--------|-------------|------------|
| VALU | 6 | vadd, vmul, vand, vor, vxor, vshift, multiply_add, vbroadcast |
| Flow | 1 | select, vselect, cond_jump, pause, halt |
| Load | 2 | load, vload, const |
| Store | 2 | store, vstore |
| ALU | 12 | scalar arithmetic |

The key insight is that replacing Flow operations (1/cycle) with VALU (6/cycle) provides 6x parallelism. H38 has already achieved this for branch operations.
