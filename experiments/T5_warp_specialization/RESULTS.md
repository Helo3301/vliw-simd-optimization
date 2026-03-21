# Experiment T5: Warp-Style Specialization Results

## Summary

**Baseline:** 9,793 cycles
**Best Result (V4):** 8,758 cycles
**Improvement:** 11.8% (1,035 cycles saved)
**Target (20%):** 8,000 cycles - NOT ACHIEVED

## Hypothesis

Separating the kernel into specialized "producer" (hash computation) and "consumer" (memory operations) phases, inspired by GPU warp specialization, would improve pipeline efficiency.

## Current Operation Mixing Analysis (Baseline)

| Metric | Value | Percentage |
|--------|-------|------------|
| Total cycles | 141 | 100% |
| Producer-only (ALU/VALU) | 66 | 46.8% |
| Consumer-only (Load/Store) | 46 | 32.6% |
| Mixed (both compute + memory) | 23 | 16.3% |
| Flow-only | 6 | 4.3% |

**Slot Utilization (Baseline):**
- ALU: 0.21/cycle (max 12)
- VALU: 0.87/cycle (max 6)
- Load: 0.64/cycle (max 2)
- Store: 0.06/cycle (max 2)
- Flow: 0.06/cycle (max 1)

Key insight: Only 16.3% of cycles actually mix compute with memory operations. The baseline already has significant phase separation.

## Implementations Tested

### V1: 2-Desk Phase Separation (16,179 cycles, -65% worse)

Clean phase separation with 2 desks. Complete producer/consumer isolation.

**Problem:** Fewer desks = less parallelism. No overlap opportunities. Dominated by memory phase.

### V2: 4-Desk Phase Batching (9,398 cycles, +4.2% better)

4 desks with phase-organized operations. Intra-phase packing but no inter-phase overlap.

**Improvement source:** Better VALU packing during hash phase. Pack 4 VALU ops per cycle.

### V3: 2-Desk Pipelined Hybrid (11,835 cycles, -21% worse)

Attempted to maintain overlap while processing 2 desks.

**Problem:** Reduced desk count hurts more than overlap helps.

### V4: 6-VALU Slot Maximization (8,758 cycles, +11.8% better) [BEST]

4 desks with maximized VALU slot packing. Each hash stage packs 6 VALU ops:
- Stage prep for desks 0,1,2 (6 ops)
- Stage prep for desk 3 + finals for 0,1,2 (5 ops)
- Final for desk 3 (1 op)

**Slot Utilization (V4):**
- VALU: 0.98/cycle (max 6), max per cycle: 6
- Mixed cycles: 0% (complete phase separation)

**Key Finding:** Better VALU packing compensates for lost overlap.

### V5: Full Overlap with Register Copies (23,761 cycles, -143% worse)

Attempted to combine V4 packing with cross-iteration overlap using dual register sets.

**Problem:** Copy overhead (transferring data between register sets) dominates any overlap benefit.

## Design of V4 (Best Version)

```
PHASE 1: LOAD (Consumer-only)
- Compute offsets for 4 desks
- vload indices/values for all desks
- Compute gather addresses
- Gather tree values (16 cycles for 32 loads)

PHASE 2: HASH (Producer-only)
- XOR with node values (4 VALU ops packed)
- For each hash stage (6 stages):
  * Prep desks 0,1,2 (6 VALU ops - FULL UTILIZATION)
  * Prep desk 3 + final 0,1,2 (5 VALU ops)
  * Final desk 3 (1 VALU op)
- Branch computation (packed 6 ops)
- Bounds check (packed 4 ops)
- vselect (4 cycles - flow-limited)

PHASE 3: STORE (Consumer-only)
- Store results for all 4 desks

LOOP CONTROL: Update counters and jump
```

## Handoff Analysis

In V4, the "handoff" between phases is implicit - data stays in the same registers:
- Load phase writes to desks[d]['idx'], desks[d]['val'], desks[d]['node_val']
- Hash phase reads from these, writes back to desks[d]['val'], desks[d]['idx']
- Store phase reads from these

No explicit handoff overhead because we use the same register set throughout each iteration.

V5 attempted explicit handoff with register copying - this proved extremely costly (8x more cycles than V4).

## Was Specialization Worth It?

**Partial Success:**

1. **VALU packing worked:** Going from 0.87 to 0.98 VALU ops/cycle saved cycles.

2. **Phase separation had mixed results:**
   - Losing the 16.3% mixed cycles (hash overlapped with gather) hurt.
   - But better VALU packing compensated.
   - Net: +11.8% improvement.

3. **Full overlap with handoff failed:** The overhead of moving data between register sets far exceeded the benefit of overlapping phases across iterations.

## Key Insights

1. **The baseline's mixing is limited:** Only 16.3% of cycles mix compute and memory. This limits how much overlap can help.

2. **VALU utilization was the bottleneck:** The baseline only used 0.87 VALU ops/cycle. V4's packing reaches 0.98.

3. **Register copying is expensive:** Any explicit handoff mechanism that requires copying registers will likely hurt performance.

4. **Desk count matters:** Reducing from 4 to 2 desks dramatically hurts performance, regardless of pipelining strategy.

## Conclusion

Warp-style specialization provides an 11.8% improvement (9,793 -> 8,758 cycles) but does not reach the 20% target (8,000 cycles).

The improvement comes primarily from **better VALU packing** during the hash phase, not from phase separation per se. The loss of hash-gather overlap is offset by packing 6 VALU ops per cycle instead of 2.

**Recommendation:** For further improvement, combine V4's VALU packing with the baseline's hash-gather overlap WITHOUT explicit register copying - this would require careful restructuring to maintain VALU packing while overlapping with the next iteration's gather.

## Commands Used

```bash
# Run correctness check
python3.11 experiments/T5_warp_specialization/perf_takehome_t5.py --check

# Run specific version
python3.11 experiments/T5_warp_specialization/perf_takehome_t5.py --version 4

# Run all versions
python3.11 experiments/T5_warp_specialization/perf_takehome_t5.py --all
```
