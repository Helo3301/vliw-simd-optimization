"""Analyze cycle-by-cycle breakdown of Theory 222"""
import sys
sys.path.insert(0, '/home/hestiasadmin/projects/original_performance_takehome')
sys.path.insert(0, '/home/hestiasadmin/projects/original_performance_takehome/experiments/BF300')

from theory_222_hashorder_WIN import KernelBuilderA1
from problem import *
import random

random.seed(123)
forest = Tree.generate(10)
inp = Input.generate(forest, 256, 16)
mem = build_mem_image(forest, inp)

kb = KernelBuilderA1()
kb.build_kernel(forest.height, len(forest.values), 256, 16)

# Analyze instruction stream
instrs = kb.instrs
total_cycles = len(instrs)
print(f"Total cycles: {total_cycles}")

# Count per-engine usage per cycle
valu_total = 0
load_total = 0
store_total = 0
flow_total = 0
alu_total = 0

valu_full = 0  # cycles with 6 VALU
load_full = 0  # cycles with 2 loads
flow_full = 0  # cycles with 1 flow
valu_zero = 0  # cycles with 0 VALU
load_zero = 0  # cycles with 0 loads
flow_zero = 0  # cycles with 0 flow

for cyc, instr in enumerate(instrs):
    nv = len(instr.get('valu', []))
    nl = len(instr.get('load', []))
    ns = len(instr.get('store', []))
    nf = len(instr.get('flow', []))
    na = len(instr.get('alu', []))
    
    valu_total += nv
    load_total += nl
    store_total += ns
    flow_total += nf
    alu_total += na
    
    if nv == 6: valu_full += 1
    if nl == 2: load_full += 1
    if nf == 1: flow_full += 1
    if nv == 0: valu_zero += 1
    if nl == 0: load_zero += 1
    if nf == 0: flow_zero += 1

print(f"\nEngine totals:")
print(f"  VALU: {valu_total} ops, min cycles: {(valu_total+5)//6} = {valu_total/6:.1f}")
print(f"  Load: {load_total} ops, min cycles: {(load_total+1)//2} = {load_total/2:.1f}")
print(f"  Store: {store_total} ops, min cycles: {(store_total+1)//2} = {store_total/2:.1f}")
print(f"  Flow: {flow_total} ops, min cycles: {flow_total}")
print(f"  ALU: {alu_total} ops, min cycles: {(alu_total+11)//12} = {alu_total/12:.1f}")

print(f"\nUtilization:")
print(f"  VALU full (6/6): {valu_full}/{total_cycles} ({100*valu_full/total_cycles:.1f}%)")
print(f"  VALU zero (0/6): {valu_zero}/{total_cycles} ({100*valu_zero/total_cycles:.1f}%)")
print(f"  Load full (2/2): {load_full}/{total_cycles} ({100*load_full/total_cycles:.1f}%)")
print(f"  Load zero (0/2): {load_zero}/{total_cycles} ({100*load_zero/total_cycles:.1f}%)")
print(f"  Flow full (1/1): {flow_full}/{total_cycles} ({100*flow_full/total_cycles:.1f}%)")
print(f"  Flow zero (0/1): {flow_zero}/{total_cycles} ({100*flow_zero/total_cycles:.1f}%)")

# Identify bottleneck cycles
valu_bound = 0  # cycles where VALU=6 and others could take more
load_bound = 0  # cycles where load=2 but VALU<6
flow_bound = 0  # cycles where flow=1 but VALU<6 and load<2
dep_bound = 0   # cycles where no engine is at limit (dependency stall)

for cyc, instr in enumerate(instrs):
    nv = len(instr.get('valu', []))
    nl = len(instr.get('load', []))
    nf = len(instr.get('flow', []))
    na = len(instr.get('alu', []))
    ns = len(instr.get('store', []))
    
    if nf == 1 and instr.get('flow', [None])[0] == ('pause',):
        continue
    
    if nv == 6:
        valu_bound += 1
    elif nl == 2 and nv < 6:
        load_bound += 1
    elif nf == 1 and nv < 6 and nl < 2:
        flow_bound += 1
    elif nv < 6 and nl < 2 and nf < 1:
        dep_bound += 1

print(f"\nBottleneck analysis:")
print(f"  VALU-bound (VALU=6): {valu_bound}")
print(f"  Load-bound (load=2, VALU<6): {load_bound}")
print(f"  Flow-bound (flow=1, VALU<6, load<2): {flow_bound}")
print(f"  Dep-stall (all < limit): {dep_bound}")
print(f"  Total: {valu_bound + load_bound + flow_bound + dep_bound}")

# Find sections - look for pause boundaries
pause_cycles = []
for cyc, instr in enumerate(instrs):
    if instr.get('flow', [None])[0] == ('pause',):
        pause_cycles.append(cyc)

print(f"\nPause at cycles: {pause_cycles}")
if len(pause_cycles) >= 2:
    print(f"  Init (before first pause): {pause_cycles[0]} cycles")
    print(f"  Tile 1: {pause_cycles[1] - pause_cycles[0] - 1} cycles")
    if len(pause_cycles) >= 3:
        print(f"  Tile 2: {pause_cycles[2] - pause_cycles[1] - 1} cycles")

# Histogram of VALU utilization
valu_hist = [0] * 7
for instr in instrs:
    nv = len(instr.get('valu', []))
    valu_hist[nv] += 1
print(f"\nVALU histogram:")
for i in range(7):
    bar = '#' * (valu_hist[i] // 5)
    print(f"  {i}/6: {valu_hist[i]:4d} {bar}")

# Load histogram
load_hist = [0] * 3
for instr in instrs:
    nl = len(instr.get('load', []))
    load_hist[nl] += 1
print(f"\nLoad histogram:")
for i in range(3):
    bar = '#' * (load_hist[i] // 5)
    print(f"  {i}/2: {load_hist[i]:4d} {bar}")

# Flow histogram
flow_hist = [0] * 2
for instr in instrs:
    nf = len(instr.get('flow', []))
    if nf < 2:
        flow_hist[nf] += 1
    else:
        flow_hist[1] += 1
print(f"\nFlow histogram:")
for i in range(2):
    bar = '#' * (flow_hist[i] // 5)
    print(f"  {i}/1: {flow_hist[i]:4d} {bar}")

