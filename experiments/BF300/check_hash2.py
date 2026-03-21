"""Check how hash is actually computed in reference kernel"""
import sys
sys.path.insert(0, '/home/hestiasadmin/projects/original_performance_takehome')
from problem import HASH_STAGES

# The hash stages are: 
# Each stage i = (op1, val1, op2, op3, val3)
# val = op1(val, val1)
# The reference kernel hash_value function:
# for op1, val1, op2, op3, val3 in HASH_STAGES:
#     value = _apply_op(op1, value, val1)
#     combined = _apply_op(op2, value, _apply_op(op3, value, val3))
#     value = combined

# Let me expand this manually:
print("Hash function expanded (per round):")
for i, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
    print(f"\n  Stage {i}: ({op1}, {val1}, {op2}, {op3}, {val3})")
    print(f"    Step 1: val = val {op1} {val1}")
    print(f"    Step 2: tmp = val {op3} {val3}")
    print(f"    Step 3: val = val {op2} tmp")

# What the kernel emits per stage pair:
# Stages 0,1 -> Stages 2,3 -> Stages 4,5
# FMA stages (0, 2, 4): val = val * mult + const
# XOR/shift stages (1, 3, 5):
#   tmp1 = val op2 const
#   tmp2 = val op3 shift
#   val = tmp1 ^ tmp2

# Current per-desk hash: 12 VALU (4 ops per stage pair * 3 pairs)
# Actually let me count exactly:
print("\n\nActual emission per desk:")
print("  FMA stage 0: multiply_add(val, val, mult0, const0)  -> 1 VALU")
print("  Stage 1: ^(tmp1, val, const1)                       -> 1 VALU")
print("           >>(tmp2, val, shift1)                       -> 1 VALU")  
print("           ^(val, tmp1, tmp2)                          -> 1 VALU")
print("  FMA stage 2: multiply_add(val, val, mult2, const2)  -> 1 VALU")
print("  Stage 3: +(tmp1, val, const3)                        -> 1 VALU")
print("           <<(tmp2, val, shift3)                       -> 1 VALU")
print("           ^(val, tmp1, tmp2)                          -> 1 VALU")
print("  FMA stage 4: multiply_add(val, val, mult4, const4)  -> 1 VALU")
print("  Stage 5: ^(tmp1, val, const5)                        -> 1 VALU")
print("           >>(tmp2, val, shift5)                       -> 1 VALU")
print("           ^(val, tmp1, tmp2)                          -> 1 VALU")
print("  Total: 12 VALU ops per desk per round")

# What's the chain depth?
# FMA -> tmp1/tmp2 (parallel) -> XOR -> FMA -> tmp1/tmp2 (parallel) -> XOR -> FMA -> tmp1/tmp2 -> XOR
# Depth: 3 FMA + 3 pair-ops (each 2 deep) = 3+6 = 9 deep
# But with 6 VALU/cycle and 4 desks:
# 12 ops * 4 desks = 48 ops, needs 8 cycles at 6/cycle
# But chain depth per desk is 9, so need >= 9 cycles per desk hash
# Unless we interleave across desks!

print("\n\nChain depth per desk hash: 9 (FMA->op->XOR->FMA->op->XOR->FMA->op->XOR)")
print("With 4 desks interleaved: can fit more in parallel")
print(f"48 ops / 6 per cycle = 8 cycles minimum (if no deps)")
print(f"But each desk has depth 9, so at least 9 cycles per desk unless interleaved")
