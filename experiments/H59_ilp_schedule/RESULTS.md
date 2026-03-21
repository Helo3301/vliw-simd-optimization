# H59: Constraint-Based Optimal Schedule Analysis Results

## Executive Summary

**Key Finding**: The CP-SAT solver found that H54's hand-crafted schedule can be improved by **24.1%** per iteration (51 cycles), potentially reducing total kernel cycles from **3,462 to 2,646** (23.6% improvement).

| Metric | Value |
|--------|-------|
| Current H54 makespan (per iteration) | 212 cycles |
| Optimal makespan (per iteration) | 161 cycles |
| Theoretical minimum (resource-bound) | 144 cycles |
| Improvement potential | 51 cycles (24.1%) |
| Gap from theoretical minimum | 17 cycles (11.8%) |
| Solver status | OPTIMAL |
| Solve time | 27.67 seconds |

---

## 1. Problem Analysis

### 1.1 H54 Main Loop Characteristics

The H54 kernel processes 16 batches in parallel (16-desk pipeline) with round fusion (2 rounds per iteration).

**Operation counts per iteration:**
| Engine | Operations | Capacity | Theoretical Min Cycles |
|--------|-----------|----------|----------------------|
| ALU | 84 | 12/cycle | 7 cycles |
| VALU | 640 | 6/cycle | 107 cycles |
| Load | 288 | 2/cycle | 144 cycles |
| Store | 32 | 2/cycle | 16 cycles |
| Flow | 2 | 1/cycle | 2 cycles |

**Bottleneck**: Load operations (288 loads / 2 per cycle = 144 cycle minimum)

### 1.2 Current Resource Utilization

| Engine | Utilization |
|--------|-------------|
| ALU | 3.3% |
| VALU | 50.3% |
| Load | 67.9% |
| Store | 7.5% |
| Flow | 0.9% |

**Observation**: ALU is massively underutilized (3.3%), while load is the bottleneck at 67.9% utilization. There is significant room for better packing.

---

## 2. ILP Model Details

### 2.1 Model Structure

- **Variables**: 1,046 integer start-time variables (one per operation)
- **Dependencies**: 1,477 precedence constraints
- **Resource constraints**: Cumulative constraints for each engine type
- **Objective**: Minimize makespan

### 2.2 Solver Configuration

- **Solver**: Google OR-Tools CP-SAT
- **Time limit**: 300 seconds
- **Workers**: 8 parallel threads
- **Result**: OPTIMAL solution found in 27.67 seconds

---

## 3. Schedule Comparison

### 3.1 Operation Movement

The optimal schedule significantly rearranges operations:

| Category | Count |
|----------|-------|
| Operations moved earlier | 650 (62%) |
| Operations moved later | 377 (36%) |
| Operations unchanged | 19 (2%) |

### 3.2 Movement by Engine Type

| Engine | Avg Cycle Movement |
|--------|-------------------|
| ALU | -80.4 cycles (earlier) |
| VALU | -9.6 cycles (earlier) |
| Load | -17.2 cycles (earlier) |
| Store | -58.2 cycles (earlier) |
| Flow | -203.0 cycles (earlier) |

**Key insight**: ALU operations can be scheduled much earlier (-80 cycles on average), allowing better overlap with load-bound phases.

---

## 4. Total Kernel Projection

### 4.1 Current H54

- Total cycles: 3,462
- Iterations: 16
- Loop cycles per iteration: 212
- Setup overhead: 70 cycles

### 4.2 Projected Optimal

- Projected optimal total: **2,646 cycles**
- Improvement: **816 cycles (23.6%)**
- Theoretical floor: 2,374 cycles

### 4.3 Comparison to Baseline

| Metric | Current H54 | Optimal | Baseline |
|--------|-------------|---------|----------|
| Cycles | 3,462 | 2,646 | 147,734 |
| Speedup vs baseline | 42.7x | 55.8x | 1.0x |

---

## 5. Rescheduling Opportunities

Based on the solver output, key improvements come from:

### 5.1 Earlier ALU Operations

The offset and address calculations in PHASE 1 can start later in the previous iteration, overlapping with stores from the previous iteration.

### 5.2 Better Store Overlap

Stores (currently at end of iteration) can start earlier while hash operations for later desks are still completing.

### 5.3 Tighter Gather/Hash Interleaving

The current interleaving of gather and hash operations has gaps that can be filled with operations from other desks.

---

## 6. Implementation Recommendations

### 6.1 Proceed with Code Generation?

Given the **24% improvement potential**, it is worth implementing an optimized schedule.

However, key challenges:
1. The solver's schedule may not be directly translatable to readable code
2. Dependencies between iterations need careful handling
3. Scratch address allocation may need adjustment

### 6.2 Suggested Approach

1. **Phase 1**: Analyze the solver's schedule to identify the main restructuring patterns
2. **Phase 2**: Manually implement the key improvements:
   - Move ALU address calculations earlier
   - Start stores before all hash operations complete
   - Tighten gather/hash interleaving
3. **Phase 3**: Validate correctness and measure actual cycle count

### 6.3 Expected Practical Improvement

Due to implementation constraints, a realistic target is **15-20%** improvement (vs. the theoretical 24%), yielding:
- Target: ~2,800-3,000 cycles
- Speedup: 49-53x vs baseline

---

## 7. Technical Notes

### 7.1 Solver Details

```
Solver: OR-Tools CP-SAT v9.15.6755
Model size: 1,046 operations, 1,477 dependencies
Horizon: 232 cycles
Status: OPTIMAL
Solve time: 27.67s
```

### 7.2 Files Generated

- `h54_instrs.json`: Extracted H54 instruction stream
- `results_h54.json`: Detailed solver results
- `analyze_h54_optimal.py`: Analysis script
- `extract_h54.py`: Instruction extraction script

---

## 8. Conclusion

**The ILP analysis confirms significant optimization potential in the H54 schedule.**

The solver found a provably optimal schedule with 161 cycles vs. the current 212 cycles (24.1% improvement). This translates to a potential total kernel improvement from 3,462 to 2,646 cycles (23.6%).

The main sources of improvement are:
1. Earlier scheduling of ALU operations (-80 cycles average)
2. Better overlap of stores with hash computation (-58 cycles average)
3. Tighter load/compute interleaving (-17 cycles average)

**Recommendation**: Implement a new kernel (H60) based on these findings, targeting 15-20% improvement for a realistic outcome of ~2,800-3,000 cycles.
