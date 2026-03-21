# Fresh Claude Session - Complete H140 Implementation

## Your Task

Implement the optimized VLIW SIMD kernel to achieve **1,645 cycles (89.8x speedup)**. The baseline is 147,734 cycles.

## Repository

```
/home/hestiasadmin/projects/original_performance_takehome
```

## FASTEST PATH: Copy the Reference Implementation

**The complete working implementation is at:**
```
experiments/H140_h82_combined/perf_takehome_h140.py
```

To use it:
1. Copy the `KernelBuilderH140` class and helper functions (`_vec_range`, `_slot_rw`, `_schedule_slots`)
2. Rename `KernelBuilderH140` to `KernelBuilder` in `perf_takehome.py`
3. Run `python3.11 tests/submission_tests.py` to verify 1,645 cycles

**That's it.** The reference implementation is tested and working.

---

## Understanding the Optimizations (For Learning)

If you want to understand WHY it achieves 1,645 cycles, read on.

### The Critical Component: Automatic VLIW Scheduler

**Without the scheduler, you'll get ~4,000+ cycles.** This is the single most important piece.

The scheduler (`_schedule_slots()` function):
- Takes a flat list of `(engine, operation)` tuples
- Tracks read/write dependencies per scratch address
- Packs operations into VLIW bundles respecting slot limits (12 ALU, 6 VALU, 2 load, 2 store, 1 flow)
- Uses greedy "earliest possible cycle" placement

```python
# Operations are emitted to a list, NOT directly to instructions
self.slots.append((engine, slot))

# Later, scheduler packs them into VLIW bundles
phase_instrs = _schedule_slots(phase)
```

### Key Optimizations Applied

1. **Interleaved Round Processing (H82)** - THE BREAKTHROUGH (+183 cycles)
   - Process 4 desks through ALL 16 rounds, then next 4 desks
   - GROUP_SIZE = 4 is optimal
   - Allows scheduler to overlap operations from different rounds

2. **Arithmetic Selection (H77)** - Eliminates gathers for early rounds
   - Rounds 0, 11: XOR directly with tree[0]
   - Rounds 1, 12: 2-way arithmetic selection from tree[1,2]
   - Rounds 2, 13: 4-way arithmetic selection from tree[3-6]

3. **Reduced Tree Preload (H105)** - Only load tree nodes 0-6 (+7 cycles)

4. **Fast Init (H120)** - Only load 4 of 7 header values (+3 cycles)

5. **Skip Final Branch (H133)** - Don't compute idx in round 15 (+1 cycle)

### Structure

- **16 desks per tile × 2 tiles** = 256 batch elements (16 × 8 × 2)
- **4-desk groups** for interleaved processing
- **7 preloaded tree nodes** (0-6)
- **4 header values loaded** (n_nodes, forest_values_p, inp_indices_p, inp_values_p)

## Verification

```bash
python3.11 tests/submission_tests.py
```

Expected:
- Cycles: 1,645
- Speedup: 89.8x
- Passes: test_opus4_many_hours, test_opus45_casual

## Troubleshooting

### If you get ~3,600-4,000 cycles:
You're missing the automatic scheduler. Make sure:
1. Operations are emitted to a flat list (`self.slots.append((engine, slot))`)
2. Call `_schedule_slots()` to pack them into VLIW bundles
3. The scheduler handles all parallelism - don't manually order for ILP

### If you get ~2,000 cycles:
You have the scheduler but not interleaved round processing. Make sure:
1. GROUP_SIZE = 4
2. Loop order is: `for group → for round → for desk_in_group`

### If correctness checks fail:
Make sure you store BOTH idx and val vectors at the end (the test framework checks both).

## Reference Documents

1. `COMPLETE_OPTIMIZATION_CHAIN.md` - Detailed breakdown of all optimizations
2. `experiments/H140_h82_combined/perf_takehome_h140.py` - Working implementation
3. `RESULTS_SUMMARY.md` - Experiment history and results
