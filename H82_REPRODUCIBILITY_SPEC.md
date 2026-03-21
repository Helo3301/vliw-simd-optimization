# H82 Optimization: Reproducibility Specification

## Problem Context

You are optimizing a VLIW SIMD kernel for tree traversal with hash computation. The baseline achieves 147,734 cycles. Your target is to achieve **1,656 cycles (89.2x speedup)** using the technique described below.

## Architecture Constraints

- **VLIW slots per cycle:** 12 ALU, 6 VALU, 2 load, 2 store, 1 flow
- **VLEN:** 8 (vector length)
- **SCRATCH_SIZE:** 1536 slots
- **Batch size:** 256 elements
- **Rounds:** 16 tree traversal rounds per element
- **No vgather instruction** - must use scalar loads for gather operations

## The Key Optimization: Interleaved Round Processing

### The Insight

Most approaches process all desks through each round before moving to the next round:

```
Round 0: desk0, desk1, desk2, ..., desk15
Round 1: desk0, desk1, desk2, ..., desk15
...
Round 15: desk0, desk1, desk2, ..., desk15
```

**H82's approach:** Process groups of 4 desks through ALL 16 rounds before moving to the next group:

```
Group 0 (desks 0-3): Round 0, Round 1, ..., Round 15
Group 1 (desks 4-7): Round 0, Round 1, ..., Round 15
Group 2 (desks 8-11): Round 0, Round 1, ..., Round 15
Group 3 (desks 12-15): Round 0, Round 1, ..., Round 15
```

### Why This Works

1. **Better ILP:** The scheduler can overlap operations from different rounds within the same desk group
2. **Reduced dependency stalls:** Round N+1 depends on Round N for the same desk, but with 4 desks interleaved, the scheduler finds work to do while waiting
3. **Group size of 4 is optimal:** Tested 1-6, group size 4 achieved best results (aligns with 6 VALU slots/cycle)

## Implementation Details

### Structure

1. **2 tiles × 16 desks = 256 batch elements** (matches batch_size)
2. **4-desk groups** within each tile
3. **Per-desk vectors:** idx, val, node_val, addr (4 vectors × 8 lanes = 32 slots per desk)
4. **Shared temps:** 2 temp vectors shared between pairs of desks (tmp1, tmp2)

### Round Types

- **Rounds 0, 11:** Direct XOR with preloaded tree[0]
- **Rounds 1, 12:** 2-way arithmetic selection (indices 1-2)
- **Rounds 2, 13:** 4-way arithmetic selection (indices 3-6)
- **Rounds 3-9, 14-15:** Gather from memory (8 scalar loads per desk)
- **Round 10:** Gather with bounds check

### Wrap Exploitation

After round 10's bounds check, ALL indices that exceed n_nodes wrap to 0. This means:
- Rounds 11-13 use the same tree nodes as rounds 0-2
- Preload tree nodes 0-6 and reuse them

### Automatic Scheduling

Use a greedy list scheduler that:
1. Tracks read/write dependencies per scratch address
2. Places each operation at the earliest cycle where dependencies are satisfied
3. Respects slot limits (6 VALU, 2 load, etc.)

## Pseudocode

```python
def emit_tile_interleaved(tile_idx):
    # Load all desk idx/val vectors
    for d in range(16):
        vload(desks[d].idx, ...)
        vload(desks[d].val, ...)

    # Process in groups of 4
    GROUP_SIZE = 4
    for group_start in range(0, 16, GROUP_SIZE):
        group = range(group_start, min(group_start + GROUP_SIZE, 16))

        # All desks in group do round 0
        for d in group:
            emit_round_0(d)

        # All desks in group do round 1
        for d in group:
            emit_round_1(d)

        # ... continue for all 16 rounds ...

        for d in group:
            emit_round_15(d)

    # Store all results
    for d in range(16):
        vstore(desks[d].idx, ...)
        vstore(desks[d].val, ...)

# Process both tiles
emit_tile_interleaved(0)
emit_tile_interleaved(1)
```

## Verification

```bash
cd /home/hestiasadmin/projects/original_performance_takehome
python3.11 tests/submission_tests.py
```

Expected output:
- `test_opus4_many_hours` passes (< 2,164 cycles)
- `test_opus45_casual` passes (< 1,790 cycles)
- Cycles: ~1,656
- Speedup: ~89.2x

## Key Parameters Summary

| Parameter | Value |
|-----------|-------|
| Tiles | 2 |
| Desks per tile | 16 |
| Group size | 4 |
| Total batch | 256 |
| Preloaded tree nodes | 7 (indices 0-6) |
| Shared temps | 2 per desk pair |

## What NOT to Do

- Don't try to reorder rounds (dependency chain is strict)
- Don't use more than 4 desks per group (diminishing returns)
- Don't skip the automatic scheduler (it handles VLIW packing)
- Don't try to optimize the scheduler itself (it's already good enough)

## Expected Result

**1,656 cycles, 89.2x speedup over baseline**

This beats the Opus 4.5 casual target (1,790 cycles) by 134 cycles.
