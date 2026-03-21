# H145 Combination Experiments Results

## Goal
Find synergies between experiments that individually didn't beat H140 (1,645 cycles).

Target: 1,579 cycles (need to save 66 cycles from H140)

## Baseline Performance
- H140: **1,645 cycles** (combines H82 + H105 + H120 + H133)

## Candidate Optimizations Tested

### Source Experiments (standalone performance)
| Experiment | Description | Standalone Cycles |
|------------|-------------|-------------------|
| H85 | Shared temp registers (8 pairs for 16 desks) | 1,898 |
| H87 | Double-pumped hash (batch ops by type, even-then-odd) | 1,851 |
| H134 | Inline diff computation | 1,839 |

### Combination Results

| Experiment | Combination | Cycles | Delta vs H140 | Result |
|------------|-------------|--------|---------------|--------|
| H145a | H140 + Shared Temps | 1,648 | +3 worse | FAILED |
| **H145b** | **H140 + Double-Pump Hash** | **1,643** | **-2 better** | **BEST** |
| H145c | H140 + Inline Diff | 1,674 | +29 worse | FAILED |
| H145d | H140 + Double-Pump + Shared Temps | 1,656 | +11 worse | FAILED |
| H145e | H140 + Aggressive Batch | 1,643 | -2 better | TIE |

## Analysis

### H145a: Shared Temps (FAILED)
- Saves 128 scratch slots (928 vs 1056)
- However, creates scheduling conflicts when desks in the same pair are processed together
- H140's group-of-4 interleaving means adjacent desks (0-1, 2-3) process together
- Shared temps between pairs 0-1, 2-3 causes register hazards
- **Conclusion:** Shared temps conflict with H82's interleaved grouping

### H145b: Double-Pumped Hash (BEST - 1,643 cycles)
- Batches hash operations by type: all FMAs together, then 3-op stages
- For 3-op stages, processes even desks (0,2) then odd (1,3) within group
- Works because even/odd desks use different tmp1/tmp2 registers
- **2 cycle improvement** - the batching exposes more ILP to the scheduler
- **Conclusion:** Double-pump complements H82's interleaving

### H145c: Inline Diff (FAILED)
- Computing diffs inline rather than precomputing adds 3 more operations per tile
- The scheduler can't hide these operations as well during interleaved processing
- H140's deep interleaving means diff computation can't overlap with loads
- **Conclusion:** Inline diffs conflict with interleaved round processing

### H145d: Double-Pump + Shared Temps (FAILED)
- Attempted to combine H145b's success with shared temps
- Shared temps cause conflicts even with double-pump ordering
- **Conclusion:** Shared temps fundamentally incompatible with H140's grouping

### H145e: Aggressive Batching (TIE)
- Extended double-pump to also batch branch operations
- Achieved same 1,643 cycles as H145b
- Additional batching doesn't expose more parallelism
- **Conclusion:** Hash double-pump captures all available benefit

## Key Insights

1. **H87's double-pump DOES synergize with H82's interleaving** (saves 2 cycles)

2. **Shared temps (H85) conflict with H82's group-of-4 strategy**
   - Groups 0-3 contain desk pairs 0-1, 2-3
   - These pairs share temps in H85
   - Processing them together causes write-after-write hazards

3. **Inline diff (H134) adds latency in interleaved context**
   - Without interleaving, inline diffs can overlap with other work
   - With interleaving, diffs are on critical path between rounds

4. **Not all optimizations compose**
   - H92 previously showed H87+H90 was worse (1,971 cycles)
   - H145d shows double-pump + shared temps is worse
   - Must test combinations empirically

## Best Result: H145b

**1,643 cycles** (2 cycles faster than H140's 1,645)

Still **64 cycles short** of target (1,579 cycles).

### What H145b Includes:
- H82: Interleaved round processing (groups of 4)
- H105: Reduced preload (7 nodes vs 15)
- H120: Fast init (4 header values)
- H133: Skip final branch
- **H87: Double-pumped hash stages**

### Next Steps to Reach Target:
To save 64 more cycles, would need major architectural changes:
1. Larger groups? (H82 found 4 was optimal)
2. Different tile strategy? (32 desks in one tile?)
3. Overlap tile 0/1 processing somehow?
4. Reduce gather latency? (currently 8 scalar loads per gather)
