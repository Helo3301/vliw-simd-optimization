# H7H10: Combined Interleaving + vselect Bypass

## Summary

This experiment combines the two best previous optimizations:
- **H7** (6,203 cycles): Aggressive cross-desk interleaving with software pipelining
- **H10** (7,611 cycles): vselect bypass via VALU multiply

## Results

| Metric | Value |
|--------|-------|
| **Cycles** | 5,947 |
| **Baseline** | 147,734 |
| **Speedup over baseline** | 24.84x |
| **Improvement over H7** | 256 cycles (4.1% faster) |

## Key Optimizations

### H7 Foundation: Cross-Desk Interleaving
The kernel uses a software pipeline approach where 4 desks operate at different stages simultaneously:
- Desk 0 leads, others follow in staggered fashion
- While gathering data for one desk, hash computations run on others
- Overlaps load/valu/alu operations across instruction slots

### H10 Enhancement: vselect to Multiply Conversion
Replaced flow-unit `vselect` operations with VALU `multiply`:

```python
# Before (flow unit, 1 op/cycle):
("vselect", idx, cond, idx, v_zero)

# After (VALU, can run in parallel with ALU):
("*", idx, idx, cond)  # idx * cond = idx if cond=1, 0 if cond=0
```

The key insight: `vselect(dest, cond, idx, zero)` can be replaced with `dest = idx * cond` when:
- `cond` comes from a comparison operation (producing 0 or 1)
- The alternative value is zero

### Critical Improvement: ALU Overlap
The main cycle savings came from a subtle scheduling improvement. In H7:
- Desk3's `vselect` used the flow unit alone (1 cycle)
- Store address computation used the ALU alone (1 cycle)

In H7H10:
- Desk3's VALU multiply can run in parallel with ALU operations
- Merged the desk3 bounds check with store address computation for desks 0-2
- Merged the desk3 multiply with store address computation for desk3

This eliminated one instruction cycle per loop iteration (128 iterations = 256 cycles saved, accounting for 2 cycles per full batch of 4 desks).

## Code Changes

### Original H7 (4 separate vselect operations)
```python
# Desk0 vselect (with other VALU ops - no savings here)
"flow": [("vselect", desks[0]['idx'], desks[0]['tmp1'], desks[0]['idx'], v_zero)],
"valu": [... 5 other operations ...]

# Desk1 vselect (with other VALU ops - no savings here)
"flow": [("vselect", desks[1]['idx'], desks[1]['tmp1'], desks[1]['idx'], v_zero)],
"valu": [... 3 other operations ...]

# Desk2 vselect (with other VALU ops - no savings here)
"flow": [("vselect", desks[2]['idx'], desks[2]['tmp1'], desks[2]['idx'], v_zero)],
"valu": [... 2 other operations ...]

# Desk3 vselect - ALONE on its own cycle
self.add("flow", ("vselect", desks[3]['idx'], desks[3]['tmp1'], desks[3]['idx'], v_zero))

# Store address computation - separate cycles
self.instrs.append({"alu": [... 6 ALU ops for desks 0-2 ...]})
self.instrs.append({"alu": [... 2 ALU ops for desk 3 ...]})
```

### H7H10 (vselect replaced with VALU multiply + ALU overlap)
```python
# Desks 0, 1, 2: vselect -> multiply (still packed with other VALU ops)
# No cycle savings, but maintains correctness

# Desk3: Combined with ALU operations
self.instrs.append({
    "valu": [("<", desks[3]['tmp1'], desks[3]['idx'], v_n_nodes)],
    "alu": [... 6 ALU ops for desks 0-2 store addresses ...]
})
self.instrs.append({
    "valu": [("*", desks[3]['idx'], desks[3]['idx'], desks[3]['tmp1'])],
    "alu": [... 2 ALU ops for desk 3 store addresses ...]
})
```

## Why This Works

1. **Flow unit limitation**: Only 1 operation per cycle
2. **VALU capacity**: Can run up to 6 operations per cycle, AND can run in parallel with ALU
3. **Original H7**: Desk3's vselect was isolated on the flow unit, couldn't overlap with ALU
4. **H7H10**: VALU multiply allows overlap with ALU store address computation

## Verification

```bash
python3.11 experiments/H7H10_combined/perf_takehome_h7h10.py --check
# Output:
# forest_height=10, rounds=16, batch_size=256
# CYCLES:  5947
# Speedup over baseline:  24.841768959139063
# Correctness check PASSED! Cycles: 5947
```

## Conclusion

The combination achieves **5,947 cycles** - a new best result. The 256-cycle improvement over H7 came from:
1. Converting vselect to VALU multiply (enabling VALU+ALU parallelism)
2. Merging the formerly isolated desk3 bounds/select operations with store address computation

This demonstrates that even well-optimized code can benefit from micro-architectural knowledge about which functional units can execute in parallel.
