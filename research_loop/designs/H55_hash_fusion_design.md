# H55: Hash Stage Fusion Design

## Executive Summary

This document analyzes the hash computation in the VLIW SIMD tree traversal kernel to identify opportunities for algebraic optimization and FMA (fused multiply-add) fusion. The goal is to reduce the number of VALU operations required per hash computation.

## Current Performance Context

- **Current best**: 3,462 cycles (H54 with 16 desks)
- **Target**: 1,790 cycles
- **Gap**: 1,672 cycles (48% reduction needed)

## Hash Algorithm Analysis

### Reference Implementation

From `problem.py`, the hash algorithm uses 6 stages defined in `HASH_STAGES`:

```python
HASH_STAGES = [
    ("+", 0x7ED55D16, "+", "<<", 12),  # Stage 0
    ("^", 0xC761C23C, "^", ">>", 19),  # Stage 1
    ("+", 0x165667B1, "+", "<<", 5),   # Stage 2
    ("+", 0xD3A2646C, "^", "<<", 9),   # Stage 3
    ("+", 0xFD7046C5, "+", "<<", 3),   # Stage 4
    ("^", 0xB55A4F09, "^", ">>", 16),  # Stage 5
]
```

Each stage follows the pattern:
```python
a = r(fns[op2](r(fns[op1](a, val1)), r(fns[op3](a, val3))))
```

### Detailed Stage-by-Stage Breakdown

Let `a` be the input value.

**Stage 0:** `("+", 0x7ED55D16, "+", "<<", 12)`
```
t1 = a + 0x7ED55D16
t2 = a << 12
a' = t1 + t2 = (a + 0x7ED55D16) + (a << 12)
```
Operations: ADD, SHIFT, ADD = 3 ops naive

**Stage 1:** `("^", 0xC761C23C, "^", ">>", 19)`
```
t1 = a ^ 0xC761C23C
t2 = a >> 19
a' = t1 ^ t2 = (a ^ 0xC761C23C) ^ (a >> 19)
```
Operations: XOR, SHIFT, XOR = 3 ops

**Stage 2:** `("+", 0x165667B1, "+", "<<", 5)`
```
t1 = a + 0x165667B1
t2 = a << 5
a' = t1 + t2 = (a + 0x165667B1) + (a << 5)
```
Operations: ADD, SHIFT, ADD = 3 ops naive

**Stage 3:** `("+", 0xD3A2646C, "^", "<<", 9)`
```
t1 = a + 0xD3A2646C
t2 = a << 9
a' = t1 ^ t2 = (a + 0xD3A2646C) ^ (a << 9)
```
Operations: ADD, SHIFT, XOR = 3 ops

**Stage 4:** `("+", 0xFD7046C5, "+", "<<", 3)`
```
t1 = a + 0xFD7046C5
t2 = a << 3
a' = t1 + t2 = (a + 0xFD7046C5) + (a << 3)
```
Operations: ADD, SHIFT, ADD = 3 ops naive

**Stage 5:** `("^", 0xB55A4F09, "^", ">>", 16)`
```
t1 = a ^ 0xB55A4F09
t2 = a >> 16
a' = t1 ^ t2 = (a ^ 0xB55A4F09) ^ (a >> 16)
```
Operations: XOR, SHIFT, XOR = 3 ops

**Naive Total: 18 VALU operations per hash**

## FMA Fusion Opportunities

### Algebraic Transformation for ADD+SHIFT+ADD Stages

For stages 0, 2, and 4, the pattern is:
```
a' = (a + const) + (a << shift)
   = a + const + a * (2^shift)
   = a * (1 + 2^shift) + const
```

This is exactly a **FMA (fused multiply-add)** operation: `a' = a * multiplier + constant`

**FMA Multipliers:**
- Stage 0: `1 + 2^12 = 1 + 4096 = 4097`
- Stage 2: `1 + 2^5 = 1 + 32 = 33`
- Stage 4: `1 + 2^3 = 1 + 8 = 9`

### H54's Current Implementation

The H54 code already exploits FMA for stages 0, 2, and 4:

```python
FMA_MULTIPLIERS = {
    0: 4097,  # 1 + 2^12
    2: 33,    # 1 + 2^5
    4: 9,     # 1 + 2^3
}

def emit_hash_stage(desk_idx, stage):
    if stage in v_fma_mult:
        # FMA stages: 0, 2, 4
        return [("multiply_add", d['val'], d['val'], v_fma_mult[stage], v_hash_consts[stage])]
    else:
        # XOR stages: 1, 3, 5
        ...
```

**Current operation count with FMA:**
- Stage 0: 1 FMA
- Stage 1: 3 ops (XOR, SHIFT, XOR)
- Stage 2: 1 FMA
- Stage 3: 3 ops (ADD, SHIFT, XOR)
- Stage 4: 1 FMA
- Stage 5: 3 ops (XOR, SHIFT, XOR)

**Optimized Total: 12 VALU operations per hash**

## Analysis of Remaining Stages (1, 3, 5)

### Stage 1: `("^", const, "^", ">>", 19)`
```
a' = (a ^ const) ^ (a >> 19)
```
This cannot be fused into FMA because:
- XOR is not associative with multiplication
- No algebraic simplification possible

**Required ops: 3** (XOR prep, SHIFT, XOR combine)

### Stage 3: `("+", const, "^", "<<", 9)`
```
a' = (a + const) ^ (a << 9)
```
This cannot be fused because:
- The outer operation is XOR, not ADD
- XOR breaks the FMA pattern

**Required ops: 3** (ADD, SHIFT, XOR)

### Stage 5: `("^", const, "^", ">>", 16)`
```
a' = (a ^ const) ^ (a >> 16)
```
Same as stage 1 - no fusion possible.

**Required ops: 3** (XOR prep, SHIFT, XOR combine)

## Minimum Theoretical Operation Count

Given the algebraic constraints:

| Stage | Pattern | FMA Possible? | Min Ops |
|-------|---------|---------------|---------|
| 0 | (a + c) + (a << 12) | YES | 1 |
| 1 | (a ^ c) ^ (a >> 19) | NO | 3 |
| 2 | (a + c) + (a << 5) | YES | 1 |
| 3 | (a + c) ^ (a << 9) | NO | 3 |
| 4 | (a + c) + (a << 3) | YES | 1 |
| 5 | (a ^ c) ^ (a >> 16) | NO | 3 |

**Theoretical minimum: 12 VALU operations per hash**

**H54 already achieves this minimum!**

## Cycle Impact Analysis

### Current Hash Implementation in H54

Looking at the H54 emit functions:

```python
def emit_hash_stage(desk_idx, stage):
    d = desks[desk_idx]
    if stage in v_fma_mult:
        # FMA stages: 0, 2, 4 -> 1 VALU op
        return [("multiply_add", d['val'], d['val'], v_fma_mult[stage], v_hash_consts[stage])]
    else:
        # XOR stages: 1, 3, 5 -> 2 VALU ops (prep)
        if stage == 1:
            return [
                ("^", d['tmp1'], d['val'], v_hash_consts[stage]),
                (">>", d['tmp2'], d['val'], v_hash_shifts[stage]),
            ]
        elif stage == 3:
            return [
                ("+", d['tmp1'], d['val'], v_hash_consts[stage]),
                ("<<", d['tmp2'], d['val'], v_hash_shifts[stage]),
            ]
        elif stage == 5:
            return [
                ("^", d['tmp1'], d['val'], v_hash_consts[stage]),
                (">>", d['tmp2'], d['val'], v_hash_shifts[stage]),
            ]
    return []

def emit_hash_combine(desk_idx, stage):
    # XOR combine for stages 1, 3, 5 -> 1 VALU op
    d = desks[desk_idx]
    if stage in [1, 3, 5]:
        return [("^", d['val'], d['tmp1'], d['tmp2'])]
    return []
```

**Per-desk hash breakdown:**
- FMA stages (0, 2, 4): 3 ops total
- XOR stages (1, 3, 5): 2 prep + 1 combine = 3 ops each = 9 ops total
- **Total: 12 VALU ops per hash**

## Potential Further Optimizations

### 1. Parallel Shift and Constant Application

For stages 1, 3, 5, the prep operations (XOR/ADD + SHIFT) are already issued in parallel within the same cycle when possible. Looking at H54:

```python
("^", d['tmp1'], d['val'], v_hash_consts[stage]),
(">>", d['tmp2'], d['val'], v_hash_shifts[stage]),
```

These two operations can execute in the same cycle using 2 VALU slots.

### 2. Operation Packing Across Desks

H54 already packs operations across multiple desks per cycle:
```python
"valu": emit_hash_stage(0, 2) + emit_xor_node(1),  # 2 ops for hash + 1 for XOR
```

With 6 VALU slots and interleaved desks, hash stages from different desks are computed in parallel.

### 3. Speculative Alternative: Bit Manipulation Tricks

**Not recommended** - Would break mathematical correctness, but for academic interest:

If we could modify the hash algorithm (which we cannot), using `imul` with specific constants could potentially replace some shift-XOR patterns. However, the hash function's cryptographic properties depend on these specific operations.

## Bottleneck Analysis

### Why Can't We Go Below 12 Ops?

1. **XOR operations cannot be converted to FMA**: The algebraic identity `a*b + c` only works with addition, not XOR.

2. **Data dependencies**: Each hash stage depends on the previous stage's output, creating a serial chain.

3. **VALU slot limit**: With 6 VALU slots per cycle and 12 operations per hash:
   - Minimum hash latency = 2 cycles (if perfectly parallel)
   - With dependencies, actual latency is higher

### H54's Pipeline Strategy

H54 hides hash latency by interleaving desks:
- While desk 0 is computing hash stage 2, desk 1 is computing hash stage 0
- This keeps VALU slots saturated despite data dependencies

## Recommendations

### Finding 1: FMA Optimization Already Implemented
H54 already implements the optimal FMA fusion for stages 0, 2, and 4. No further algebraic optimization is possible for the hash function itself.

### Finding 2: Cycle Savings Not From Hash Fusion
The remaining 48% gap to target (1,672 cycles) cannot come from hash optimization since H54 is already at the theoretical minimum (12 ops/hash).

### Finding 3: Look Elsewhere for Gains
To reach 1,790 cycles, focus should be on:
1. **Load latency hiding**: More aggressive gather-compute overlap
2. **Store coalescing**: Batching memory writes
3. **Loop overhead reduction**: Reducing setup cycles per iteration
4. **Cache locality**: Better memory access patterns

### Finding 4: Potential Micro-Optimization
One small optimization could be combining the XOR prep operations:

Current (2 instructions):
```python
("^", d['tmp1'], d['val'], v_hash_consts[stage]),
(">>", d['tmp2'], d['val'], v_hash_shifts[stage]),
```

If the architecture supported a `xor_and_shift` compound operation, this could be 1 op. However, the ISA does not appear to support this.

## Cycle Count Verification

Per H54 for one iteration (16 desks, 2 rounds):

**Hash operations per desk per round:**
- 3 FMA ops (stages 0, 2, 4)
- 6 prep ops (stages 1, 3, 5 - 2 each)
- 3 combine ops (stages 1, 3, 5)
- Total: 12 VALU ops

**Per iteration:**
- 16 desks x 2 rounds x 12 ops = 384 VALU ops
- At 6 slots/cycle = 64 cycles minimum just for hash
- Plus gather, XOR, branch, bounds check, store...

The actual cycle count includes significant overhead from:
- Gather operations (bottlenecked by 2 load slots/cycle)
- Memory operations
- Loop control

## Conclusion

**H55 Hash Fusion Analysis Result: No Further Optimization Available**

The hash computation in H54 is already optimally implemented:
- FMA fusion applied to all applicable stages (0, 2, 4)
- XOR stages (1, 3, 5) cannot be further fused algebraically
- 12 VALU operations is the theoretical minimum

To reach the 1,790 cycle target, optimizations must focus on:
1. Memory subsystem (loads/stores)
2. Pipeline scheduling (better interleaving)
3. Loop structure (reduced overhead)
4. Alternative algorithmic approaches (index grouping, wavefront)

This analysis confirms that the hash computation itself is not the bottleneck for further optimization.
