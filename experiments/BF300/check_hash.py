"""Check HASH_STAGES structure"""
import sys
sys.path.insert(0, '/home/hestiasadmin/projects/original_performance_takehome')
from problem import HASH_STAGES

print("HASH_STAGES:")
for i, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
    print(f"  Stage {i}: ({op1}, {val1}, {op2}, {op3}, {val3})")
    if op1 == 'multiply_add':
        print(f"    val = val * {val1 + 1} = val * mult + const  [FMA]")
    elif op1 == '+':
        print(f"    tmp = val + {val1}")
    elif op1 == '^':
        print(f"    tmp = val ^ {val1}")
    if op2 in ('^', '+'):
        print(f"    tmp2 = val {op2} ?")
    if op3 in ('>>', '<<'):
        print(f"    val = tmp {op3} {val3}")
