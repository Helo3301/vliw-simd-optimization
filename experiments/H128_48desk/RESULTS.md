# Experiment H128: 48-Desk Configuration Analysis

## Hypothesis
If 32 desks helped (H96/H120), maybe 48 or 64 desks would provide more ILP.

## Analysis

### Memory Constraints
- SCRATCH_SIZE = 1536 slots
- VLEN = 8 (vector length)

Per-desk requirements in H120:
- idx: 8 slots
- val: 8 slots
- node_val: 8 slots
- addr: 8 slots
- Total: 32 slots per desk

Fixed overhead (constants, tree preloads, etc.): ~244 slots

### Configuration Feasibility

| Desks | Desk Vecs | Temps | Offsets | Total | Fits? |
|-------|-----------|-------|---------|-------|-------|
| 32    | 1024      | 128   | 32      | 1428  | YES   |
| 40    | 1280      | 160   | 40      | 1724  | NO    |
| 48    | 1536      | 192   | 48      | 2020  | NO    |
| 64    | 2048      | 256   | 64      | 2612  | NO    |

**64 desks need 2048 slots for desk vectors alone - exceeds total scratch!**

### Alternative: Aggressive Sharing

Sharing both `addr` and `node_val` between 8 desks in a group:
- 48 desks x 2 vecs (idx, val) = 768 slots
- 6 groups x 4 vecs (node_val, addr, tmp1, tmp2) = 192 slots
- Total: ~1252 slots (fits!)

But this creates severe data dependencies during gathers.

## Results

### H128a: 48 Desks with 8-Desk Aggressive Sharing
- **Cycles: 2393** (MUCH WORSE)
- Speedup: 61.7x
- Problem: Sequential gather processing within groups kills parallelism

### H128b: 2 Tiles of 16 Desks Each
- **Cycles: 1883** (43 cycles worse than baseline)
- Speedup: 78.5x
- Problem: Tile overhead + reduced ILP per tile

### Baseline H120: 32 Desks with 4-Desk Temp Sharing
- **Cycles: 1840** (CURRENT BEST)
- Speedup: 80.3x

## Conclusion

**48+ desks are not feasible** with the current architecture due to:

1. **Hard memory limit**: 64 desks need 2048 slots for desk vectors alone (exceeds 1536 total)
2. **Aggressive sharing kills parallelism**: Sharing node_val/addr across desks introduces dependencies that prevent effective scheduling
3. **Smaller tiles have overhead**: 2x16 desks performs worse than 1x32 desks

The 32-desk configuration with 4-desk temp sharing appears to be optimal for this problem size (256 batch elements = 32 desks x 8 lanes).

## Gap Analysis

- Current: 1,840 cycles
- Target: 1,790 cycles
- Gap: 50 cycles (2.7%)

To close this gap, we need algorithmic improvements, not more desks:
1. Reduce hash computation ops
2. Better scheduling of dependency chains
3. Exploit any unused ISA features
