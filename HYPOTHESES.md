# Phase 3: Optimization Hypotheses

## Hypothesis 1: Basic VLIW Packing (Hash Parallelism)
**Change:** Pack independent hash operations into same cycle
**Expected:** ~50% reduction in hash cycles (18 → 9 per element)
**Overall:** ~147K → ~100K cycles (~1.5x speedup)
**Complexity:** Easy
**Risk:** Low

```python
# Before: 3 cycles per hash stage
{"alu": [(op1, tmp1, val, const)]}  # cycle 1
{"alu": [(op3, tmp2, val, shift)]}  # cycle 2
{"alu": [(op2, val, tmp1, tmp2)]}   # cycle 3

# After: 2 cycles per hash stage
{"alu": [(op1, tmp1, val, const), (op3, tmp2, val, shift)]}  # cycle 1
{"alu": [(op2, val, tmp1, tmp2)]}                             # cycle 2
```

## Hypothesis 2: SIMD Vectorization (8-wide)
**Change:** Use valu/vload/vstore to process 8 elements at once
**Expected:** ~8x reduction in loop iterations
**Overall:** ~100K → ~15K cycles (with gather overhead)
**Complexity:** Medium
**Risk:** Medium (gather is tricky)

Key operations:
- `vload` for indices and values
- `valu` for all hash and branch ops
- `vselect` for conditional moves
- 8 scalar loads for tree values (gather)
- `vstore` for results

## Hypothesis 3: Loop Instead of Unroll
**Change:** Use jump instructions for round/batch loops
**Expected:** Minimal cycle change, enables other optimizations
**Overall:** Same cycles, smaller code, enables pipelining
**Complexity:** Medium
**Risk:** Low

```python
# Loop structure
batch_loop_start = current_instruction_index
# ... body ...
{"alu": [("+", counter, counter, eight)]}
{"alu": [("<", cond, counter, batch_size)]}
{"flow": [("cond_jump", cond, batch_loop_start)]}
```

## Hypothesis 4: Load/Compute Overlap
**Change:** Start loading next iteration's data while computing current
**Expected:** Hide some gather latency
**Overall:** ~15K → ~10K cycles
**Complexity:** Medium-Hard
**Risk:** Medium (scratch management)

```
Cycle N:   HASH(group A)  |  LOAD(group B indices)
Cycle N+1: HASH(group A)  |  LOAD(group B values)
Cycle N+2: HASH(group A)  |  GATHER(group B node_vals[0:1])
...
```

## Hypothesis 5: Software Pipeline (Multi-Group)
**Change:** Have multiple vector groups in flight simultaneously
**Expected:** Full overlap of gather/hash/store
**Overall:** ~10K → ~3K cycles
**Complexity:** Hard
**Risk:** High (complex scratch management)

Pipeline stages:
1. **LOAD**: vload indices, vload values
2. **SCATTER_PREP**: compute tree addresses
3. **GATHER**: 8 scalar loads for node_vals
4. **HASH**: 6-stage hash computation
5. **BRANCH**: compute next indices
6. **STORE**: vstore results

With 4+ stages in flight, amortize gather latency.

## Hypothesis 6: Interleave Hash Stages Across Elements
**Change:** While waiting for stage N result, start stage N of another element
**Expected:** Better ALU utilization
**Overall:** Minor improvement on top of other opts
**Complexity:** Medium
**Risk:** Low

## Hypothesis 7: Optimize Branch Computation
**Change:** Use bit tricks instead of modulo
**Expected:** Save 1-2 cycles per element
**Overall:** Minor
**Complexity:** Easy
**Risk:** Low

```python
# Before
{"alu": [("%", tmp, val, two)]}    # val % 2
{"alu": [("==", tmp, tmp, zero)]}  # == 0 ?

# After
{"alu": [("&", tmp, val, one)]}    # val & 1 (same as val % 2)
{"alu": [("^", tmp, tmp, one)]}    # flip: 0→1, 1→0 (is_even)
```

## Hypothesis 8: Fused idx Calculation
**Change:** Combine `2*idx + offset` more efficiently
**Expected:** Save 1 cycle per element
**Overall:** Minor
**Complexity:** Easy
**Risk:** Low

```python
# idx_next = 2*idx + (is_even ? 1 : 2)
# Equivalent: idx_next = 2*idx + 2 - is_even
# Or: idx_next = (idx << 1) | 1 | (1 - is_even)
```

## Testing Order (by impact × ease)

1. **H1: VLIW Packing** - Easy win, good baseline
2. **H2: SIMD** - Big impact, moderate work
3. **H7+H8: Branch opts** - Quick wins
4. **H3: Loops** - Enables H4/H5
5. **H4: Load/Compute Overlap** - Meaningful speedup
6. **H5: Full Pipeline** - Major speedup, most complex

## Success Criteria

| Threshold | Cycles | Speedup | Hypothesis Needed |
|-----------|--------|---------|-------------------|
| Baseline improvement | <147,734 | >1x | H1 alone |
| Updated starting point | <18,532 | >8x | H1 + H2 |
| Opus 4 (many hours) | <2,164 | >68x | H1-H4 |
| Opus 4.5 casual | <1,790 | >82x | H1-H5 (partial) |
| Opus 4.5 11hr | <1,487 | >99x | H1-H5 (full) |
| Best AI | <1,363 | >108x | All + fine-tuning |
