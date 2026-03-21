# Experiment H13: Store Coalescing with Overlap

## Summary

**Result: 5,435 cycles** (down from H7H10's 5,947 cycles)

- **Cycles saved:** 512 cycles per run
- **Improvement:** 8.6% faster than H7H10
- **Speedup over baseline:** 27.18x (vs H7H10's 24.84x)

## Store Pattern Comparison

### Before (H7H10 Pattern)
The original H7H10 implementation had 8 separate cycles at the end of each loop iteration:

```
Cycle N:   valu: desk3 bounds check
           alu:  compute store addresses 0-5
Cycle N+1: valu: desk3 multiply (vselect bypass)
           alu:  compute store addresses 6-7
Cycle N+2: store: desk0 idx, desk0 val
Cycle N+3: store: desk1 idx, desk1 val
Cycle N+4: store: desk2 idx, desk2 val
Cycle N+5: store: desk3 idx, desk3 val
Cycle N+6: alu: batch_offset += 32, iter_counter += 1
Cycle N+7: alu: tmp_scalar = batch_offset < batch_size
Cycle N+8: flow: select batch_offset
Cycle N+9: alu: tmp_scalar = iter_counter < total
Cycle N+10: flow: cond_jump
```

**Total: ~8-10 cycles for store + loop control phase**

### After (H13 Pattern)
The optimized H13 implementation overlaps stores with other operations:

```
Cycle N:   valu: desk3 bounds check
           alu:  compute store addresses 0-5 (6 ops)
Cycle N+1: valu: desk3 multiply (vselect bypass)
           alu:  compute store addresses 6-7
           store: desk0 idx, desk0 val  <-- OVERLAP!
Cycle N+2: store: desk1 idx, desk1 val
           alu:  batch_offset += 32, iter_counter += 1  <-- OVERLAP!
Cycle N+3: store: desk2 idx, desk2 val
           alu:  tmp_scalar = batch_offset < batch_size  <-- OVERLAP!
Cycle N+4: store: desk3 idx, desk3 val
           flow: select batch_offset  <-- OVERLAP!
Cycle N+5: alu: tmp_scalar = iter_counter < total
Cycle N+6: flow: cond_jump
```

**Total: ~6 cycles for store + loop control phase**

## Key Optimization Insights

### 1. Engine Independence
The VLIW architecture has independent execution engines:
- **ALU:** 12 slots per cycle
- **VALU:** 6 slots per cycle
- **Store:** 2 slots per cycle
- **Flow:** 1 slot per cycle

These engines execute in parallel. When a cycle only uses one engine, we're wasting potential parallelism.

### 2. Store-ALU Overlap
The key insight is that stores don't need ALU results - they just need:
1. The store address (computed by ALU beforehand)
2. The data to store (in scratch registers)

Once the addresses are computed, we can issue stores while ALU does other work.

### 3. Store-Flow Overlap
Similarly, we can overlap the final store (desk3) with the flow select operation, since select doesn't depend on the store result.

## Cycle Savings Breakdown

| Phase | H7H10 Cycles | H13 Cycles | Saved |
|-------|-------------|-----------|-------|
| Store desk0 | 1 (pure) | 0 (overlapped with valu/alu) | 1 |
| Store desk1 | 1 (pure) | 0 (overlapped with alu) | 1 |
| Store desk2 | 1 (pure) | 0 (overlapped with alu) | 1 |
| Store desk3 | 1 (pure) | 0 (overlapped with flow) | 1 |
| Loop control | 4 | 2 | 2 |
| **Total per iter** | **8** | **2** | **6** |

With 128 iterations total (batch_size=256, VLEN=8, 4 desks, 16 rounds):
- Total iterations = (256/8) * 16 / 4 = 128
- However, some overlap was already present, and there's loop overhead
- Actual measured savings: 512 cycles = ~4 cycles/iteration average

## Technical Notes

### Dependency Analysis
For stores to overlap with ALU, we must ensure:
1. Store addresses are computed before the store cycle
2. Data being stored is not modified by concurrent ALU ops
3. No RAW (read-after-write) hazards on store addresses

Our implementation satisfies all these constraints:
- `addr_tmp[0-7]` computed before first store
- `desks[*]['idx']` and `desks[*]['val']` are read-only during store phase
- Loop control uses `batch_offset` which is separate from store addresses

### Register Pressure
The optimization reuses existing scratch registers efficiently:
- `addr_tmp[0-7]` for store addresses (8 scalars)
- `offset_regs[0-3]` for desk offsets (4 scalars)
- No additional registers needed for the overlap

## Future Optimization Opportunities

1. **Double Buffering:** Use two sets of desk registers to overlap store of iteration N with load of iteration N+1
2. **Software Pipelining:** Further interleave phases to hide all latencies
3. **Loop Unrolling:** Combine multiple iterations to amortize loop overhead

## Conclusion

Store coalescing with overlap achieved a significant 8.6% improvement over H7H10 by:
1. Overlapping stores with VALU/ALU operations
2. Overlapping stores with flow operations
3. Reducing "pure store" cycles from 4 to 0

The total cycle count of 5,435 represents a 27.18x speedup over the baseline of 147,734 cycles.
