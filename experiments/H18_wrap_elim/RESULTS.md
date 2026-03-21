# H18: Wrap-Around Elimination Experiment Results

## Summary

| Metric | Value |
|--------|-------|
| Base Implementation | C4 (4,667 cycles) |
| H18 Cycles | 4,668 cycles |
| Delta | +1 cycle (regression) |
| Hypothesis | DISPROVEN |

## Hypothesis

The batch wrap-around logic in C4 uses a flow `select` operation:
```python
("select", batch_offset, tmp_scalar, batch_offset, zero_const)
```

The hypothesis was that this select:
- Requires a flow slot (limited to 1 per cycle)
- Adds comparison overhead
- Could be replaced with ALU-based bitwise AND

Since batch_size=256 is a power of 2:
- `batch_offset = (batch_offset + 32) % 256 = (batch_offset + 32) & 255`
- This uses ALU instead of flow

## Implementation

Replaced the select-based wrap-around with AND-based modulo:
```python
# Before (C4): In flow slot during store
("select", batch_offset, tmp_scalar, batch_offset, zero_const)

# After (H18): In ALU during round 2 processing
("+", next_batch_offset, batch_offset, thirtytwo_const),  # add 32
("&", next_batch_offset, next_batch_offset, batch_mask_const),  # & 255
```

## Results

**H18: 4,668 cycles** (vs C4: 4,667 cycles)

The optimization resulted in a 1-cycle regression, not an improvement.

## Analysis: Why the Hypothesis Failed

1. **The select was already "free"**: In C4, the select operation is in the flow slot during a cycle that also does store operations. Since flow and store are separate engines, they execute in parallel - the select costs nothing.

2. **The AND approach adds a dependency**: The AND-based approach requires:
   - Cycle N: `next_batch_offset = batch_offset + 32`
   - Cycle N+1: `next_batch_offset = next_batch_offset & 255`

   This creates a data dependency that may not overlap as well.

3. **ALU slots during key phases are already utilized**: The H14 optimization already uses ALU slots during the key phases for address precomputation. Adding more ALU operations competes for these slots.

4. **The select is optimally placed**: C4's select is placed in the store phase where:
   - Store engine is busy with vstores
   - ALU is doing counter updates
   - Flow slot is otherwise unused
   - This makes the select effectively free

## Key Insight

The original C4 implementation was already well-optimized. The select operation for batch wrap-around:
- Uses a flow slot that would otherwise be idle
- Runs in parallel with store operations
- Does not add to the critical path

Attempting to "optimize" by moving it to ALU:
- Uses ALU slots that could be needed elsewhere
- Adds a data dependency (ADD then AND)
- Results in slightly worse performance

## Conclusion

**DISPROVEN**: The select-based wrap-around is NOT a performance bottleneck. The flow unit has unused capacity during the store phase, making the select essentially free. The AND-based alternative does not provide any benefit and may even regress performance slightly.

This experiment validates that C4's scheduling is already near-optimal for the wrap-around logic. Future optimizations should focus on other areas of the kernel.

## Files

- Implementation: `perf_takehome_h18.py`
- Correctness: PASSED (`--check`)
