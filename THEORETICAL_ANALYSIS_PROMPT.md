# Theoretical Analysis Prompt for VLIW SIMD Optimization

Use this prompt to start a fresh Claude session focused on proving the theoretical minimum.

---

## Prompt

You are a theoretical computer scientist analyzing a VLIW SIMD kernel optimization problem.

**The Problem:**
- Repository: `/home/hestiasadmin/projects/original_performance_takehome`
- Current best: 1,613 cycles (H143d)
- Known achievable by others: 1,363 cycles
- Baseline: 147,734 cycles
- Unexplained gap: 250 cycles

**The Kernel:**
- 256 batch elements processed through 16 rounds
- Each round: tree lookup → XOR → 6-stage hash → branch computation
- ISA limits per cycle: 6 VALU, 2 load, 2 store, 12 ALU, 1 flow
- Vector length: 8 lanes

**Your Mission:**

1. **Prove the theoretical minimum cycle count from first principles:**
   - Count the irreducible operations (hash is 12 VALU minimum, proven algebraically)
   - Calculate engine-specific bounds (VALU_ops/6, LOAD_ops/2, etc.)
   - Determine which engine is the bottleneck

2. **Work backwards from 1,363 cycles:**
   - 1,363 × 6 = 8,178 max VALU ops if VALU-bound
   - Current implementation has ~9,056 VALU ops
   - Where can ~878 VALU ops be eliminated?

3. **Analyze the scheduling gap:**
   - Current greedy scheduler achieves ~93% VALU utilization
   - What would 100% utilization require?
   - Could an ILP/constraint solver achieve optimal scheduling?

4. **Identify specific algorithmic improvements:**
   - Branch computation: currently 3 ops (AND, FMA, ADD) - can it be 2?
   - Address calculation: can it be eliminated by storing addr instead of idx?
   - Round fusion: can operations be shared across rounds?

5. **Hypothesize what "improved test time compute harness" means:**
   - ILP-based scheduling?
   - Modulo scheduling / software pipelining?
   - Different kernel structure entirely?

**Key Files:**
- `problem.py` - ISA definition and reference kernel
- `experiments/H143_reduce_ops/perf_takehome_h143d.py` - Current best (1,613 cycles)
- `experiments/H141_profiling/ANALYSIS.md` - Profiling data
- `experiments/H146_theoretical_minimum/THEORY.md` - Existing theoretical analysis

**Deliverables:**
1. Mathematical proof of minimum cycle count
2. Breakdown of where cycles can be saved
3. Specific testable hypotheses ranked by potential impact
4. Recommendation for highest-ROI optimization to try first

Be rigorous. Show your math. Question every assumption. The goal is to understand WHY 1,363 is achievable, not just to get there by trial and error.

---

## Expected Insights

Based on prior analysis, you should find:
- Theoretical minimum: ~1,365 cycles (VALU-bound)
- Hash function: 12 ops irreducible (FMA already optimal for 3 stages)
- Branch: 3 ops currently, 2-op formulation unknown
- Scheduling overhead: ~100 cycles recoverable with optimal scheduler
- The T3 constraint solver experiment proved 55% improvement is possible with same ops

Use this as a starting point, but verify independently.
