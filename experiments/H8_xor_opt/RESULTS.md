# Experiment H8: XOR Hash Stage Optimization

## Summary

**Status:** NO OPTIMIZATION FOUND

| Metric | Value |
|--------|-------|
| Current best | 7,995 cycles (from T6+T5 combined) |
| Cycles after H8 | N/A (no optimization possible) |
| Improvement | None |

## Hypothesis

The non-FMA hash stages (1, 3, 5) each require 3 operations. Can these be reduced using:
- Carryless multiply or polynomial operations
- Precomputed constants
- Combined shift-XOR instructions
- Other specialized bitwise operations

## Analysis

### Hash Stage Structure

The 6 hash stages are:
```
Stage 0: val = (val + 0x7ED55D16) + (val << 12)  -> FMA optimized (T6)
Stage 1: val = (val ^ 0xC761C23C) ^ (val >> 19)  -> 3 ops (XOR, shift right, XOR)
Stage 2: val = (val + 0x165667B1) + (val << 5)   -> FMA optimized (T6)
Stage 3: val = (val + 0xD3A2646C) ^ (val << 9)   -> 3 ops (ADD, shift left, XOR)
Stage 4: val = (val + 0xFD7046C5) + (val << 3)   -> FMA optimized (T6)
Stage 5: val = (val ^ 0xB55A4F09) ^ (val >> 16)  -> 3 ops (XOR, shift right, XOR)
```

### Available VALU Operations

From `problem.py`, the simulator supports:
- **Special:** `vbroadcast`, `multiply_add` (FMA: a*b+c)
- **Standard ALU:** `+`, `-`, `*`, `//`, `cdiv`, `^`, `&`, `|`, `<<`, `>>`, `%`, `<`, `==`

**Missing operations that would help:**
- Carryless multiply (clmul) - could combine XOR chains
- Rotate instructions (ror/rol) - not helpful for this pattern anyway
- Ternary XOR (a ^ b ^ c in one op) - would directly optimize stages 1, 5
- Combined shift-XOR instruction - would reduce stage operations

### Why XOR Stages Cannot Be Optimized

#### Algebraic Analysis

**Stage 1 and 5 (Pure XOR):** `val = (val ^ C) ^ (val >> N)`

1. **Cannot precompute `C ^ (C >> N)`:**
   - The expression is `(val ^ C) ^ (val >> N)`
   - This equals `val ^ C ^ (val >> N)` (XOR is associative)
   - But `val >> N` depends on val, not C
   - So precomputing `C ^ (C >> N)` produces `C ^ (C >> N)`, not something that simplifies the formula

2. **No FMA-style reduction:**
   - FMA works because `(a + C) + (a << N) = a * (1 + 2^N) + C`
   - XOR doesn't have an equivalent identity: `(a ^ C) ^ (a >> N)` cannot be expressed as a single multiplication or addition

3. **XOR doesn't distribute over shift:**
   - `a ^ (b >> N)` cannot be rewritten as `(a ^ b) >> N` or similar
   - Right-shift loses bits, so there's no inverse operation

**Stage 3 (Mixed ADD/XOR):** `val = (val + C) ^ (val << N)`

1. **Mixing addition and XOR prevents simplification:**
   - `(val + C)` and `(val << N)` produce intermediate values that must be XORed
   - No algebraic identity combines ADD with XOR+shift

2. **Cannot use FMA:**
   - FMA requires the combine operation to be addition
   - Stage 3's combine operation is XOR

#### Theoretical Minimum Operations

For a pattern like `(val op1 C) op2 (val op3 N)`:
- Operation 1: `val op1 C` (needs val and C)
- Operation 2: `val op3 N` (needs val and N, can run parallel to op1)
- Operation 3: Combine results (needs results of both ops)

This is 2 parallel ops + 1 dependent op = **2 cycles minimum** (if ops 1 and 2 fit in one cycle).

The current T6+T5 implementation achieves this:
- Cycle 1: Prep for desks 0,1,2 (6 VALU ops: 2 ops x 3 desks)
- Cycle 2: Prep for desk 3 + combine for desks 0,1,2 (5 VALU ops)
- Cycle 3: Combine for desk 3 (1 VALU op)

This is **3 cycles for 4 desks**, which is 0.75 cycles per desk - already better than the theoretical 2-cycle minimum per desk when packing is considered.

### What Would Be Needed

To optimize XOR stages, the ISA would need one of:

1. **`multiply_xor` instruction:** `a * b ^ c` - doesn't exist
2. **Ternary XOR:** `a ^ b ^ c` - doesn't exist
3. **Carryless multiply (clmul):** Could express `a ^ (a >> N)` differently - doesn't exist
4. **Shift-XOR fusion:** `(a << N) ^ b` in one op - doesn't exist

### Current Implementation Efficiency

The T6+T5 combined implementation already achieves near-optimal packing:

| Stage Type | Ops per desk | Cycles for 4 desks | Avg cycles/desk |
|------------|--------------|--------------------|-----------------|
| FMA (0,2,4) | 1 | 1 (pack all 4) | 0.25 |
| Non-FMA (1,3,5) | 3 | 3 (via 6-slot packing) | 0.75 |

The non-FMA stages already use the 6-VALU slot limit efficiently.

## Conclusion

**No optimization is possible for the XOR-based hash stages (1, 3, 5)** within the current ISA because:

1. The instruction set lacks specialized bitwise operations (clmul, ternary XOR, shift-XOR fusion)
2. XOR lacks the algebraic properties that make FMA possible for addition stages
3. The current 6-VALU packing already achieves near-optimal cycles per desk

The theoretical minimum of 2 operations per stage cannot be achieved without ISA changes. The current implementation of 3 cycles per 4 desks (0.75 cycles/desk) is already excellent given the constraints.

## Potential Future Directions

If the ISA were extended, the following would help:
- **Ternary XOR (`xor3 dest, a, b, c`):** Would reduce stages 1, 5 to 2 ops (shift + xor3)
- **Shift-XOR fusion (`shr_xor dest, a, N, b`):** Would reduce to 2 ops (first XOR + shift-xor)
- **Carryless multiply:** Could potentially express some XOR patterns differently

## Additional Analysis: Alternative Approaches Considered

### 1. Using Scalar ALU Instead of VALU

The ALU has 12 slots per cycle vs VALU's 6 slots. Could we use scalar operations?

**Analysis:**
- VLEN = 8 (each VALU op processes 8 elements)
- To match 1 VALU op with scalar ALU: need 8 scalar ops
- For 4 desks, non-FMA stages need 12 VALU ops = 96 scalar ALU ops
- With 12 ALU slots: 96/12 = 8 cycles minimum
- Current VALU approach: 3 cycles
- **Conclusion: Scalar ALU would be 2.67x slower**

### 2. Restructuring Desk Grouping

Current pattern for 4 desks:
- Cycle 1: 6 VALU (prep for 3 desks)
- Cycle 2: 5 VALU (prep for 1 desk + combine for 3)
- Cycle 3: 1 VALU (combine for 1 desk)

Alternative patterns:
- 4+4+4: Would require 4 desks x 3 ops = 12 ops, but can only do 6/cycle = 2 cycles for prep, 1 for combine per desk = worse
- 3+3+3+3: 12 ops in 4 cycles = worse than current 3 cycles
- **Conclusion: Current 6+5+1 pattern is optimal given the data dependencies**

### 3. Interleaving with Other Engines

The current T6+T5 implementation already overlaps:
- ALU operations for address computation
- Load/store operations where possible
- Flow operations for branching

No additional interleaving opportunities were found for the hash XOR stages themselves.

### 4. Mathematical Reformulation Attempts

Attempted to find equivalent formulations:

**For `(val ^ C) ^ (val >> N)`:**
- Rewrite as `val ^ (C ^ (val >> N))` - still 3 ops
- Rewrite as `(val ^ (val >> N)) ^ C` - still 3 ops
- Look for bit-parallel tricks - none applicable
- Consider polynomial representation - no clmul available

**For `(val + C) ^ (val << N)` (stage 3):**
- Cannot factor out val (mixed operations)
- No identity exists for add-xor combinations
- Would need a `multiply_add_xor` instruction

### 5. Hardware Slot Limits Summary

```
| Engine | Slots/Cycle | Notes |
|--------|-------------|-------|
| ALU    | 12          | Scalar only |
| VALU   | 6           | Vector (VLEN=8) |
| Load   | 2           | Memory reads |
| Store  | 2           | Memory writes |
| Flow   | 1           | Control flow |
```

The 6-slot VALU limit is the fundamental bottleneck for hash stage optimization.

## Verification

Confirmed current best performance:
```bash
$ python3.11 experiments/combined/perf_takehome_t6t5.py --check
forest_height=10, rounds=16, batch_size=256
CYCLES:  7995
Speedup over baseline:  18.47829893683552
Correctness check PASSED! Cycles: 7995
```

## Files

- No implementation file created (no optimization found)
- This document: `experiments/H8_xor_opt/RESULTS.md`
