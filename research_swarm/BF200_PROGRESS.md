# BF200 Progress Report

## Summary of Results

| Theory | Description | VALU Ops | Best Cycles | GROUP_SIZE | Status |
|--------|-------------|----------|-------------|------------|--------|
| Baseline (BF100 T15) | Original best | 8,507 | 1,536 | 4 | BASELINE |
| Theory 1 | Address-tracking branch fusion | 8,252 | 1,497 | 2 | WIN |
| Theory 10 | 2 vselect for R2/R13 node selection | 8,188 | 1,487 | 4 | WIN |
| Theory 10b | + vselect for R1/R12 node selection | 8,123 | 1,488 | 1 | WIN |
| Theory 10c | + 3 vselect for R2/R13, all diffs removed | 8,057 | 1,477 | 4 | WIN |
| Theory 1b | + Combined branch+conversion (10c base) | 7,994 | 1,477 | 4 | BEST |
| Theory 10d | + vselect R1/R12 branch (too much flow) | 7,992 | 1,482 | 2 | WORSE CYCLES |
| Theory 10e | + vselect R0/R11 idx (too much flow) | 7,993 | 1,489 | 4 | WORSE CYCLES |
| Theory 51 | Remove unused v_zero + v_n_nodes | -2 VALU | +2 cycles | varies | MARGINAL |

## Key Findings

### VALU Reduction Techniques (all verified correct):
1. **Address-tracking** (Theory 1): Track addr instead of idx during gather rounds. Saves 1 addr computation per gather round. Net: 255 VALU saved.
2. **vselect node selection** (Theory 10b/c): Replace FMA node lookups in fused rounds with vselect (flow engine). Saves 1-2 VALU per desk per round, at cost of flow ops.
3. **Combined branch+conversion** (Theory 1b): Merge the R2/R13 branch with idx-to-addr conversion. Saves 64 VALU total.
4. **Unused constant removal** (Theory 51): Remove v_zero and v_n_nodes broadcasts. Saves 2 VALU.

### Key Trade-offs Discovered:
- Replacing VALU with flow (vselect) is beneficial up to a point. Beyond ~512 flow ops, the flow bottleneck (1 per cycle) starts hurting scheduling.
- GROUP_SIZE=4 is optimal for the 10c/1b approach (balance of ILP and register pressure).
- Unused constant removal saves VALU but can hurt scheduling due to changed memory layout.

## Current Best Kernel
- **File**: theory_1b_gs4_WIN.py
- **VALU ops**: 7,994
- **Cycles**: 1,477
- **Improvement**: 59 cycles below previous best (1,536), 81 below A1 baseline (1,558)
