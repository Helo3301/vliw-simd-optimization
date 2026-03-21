"""Detailed analysis of theory_229_dualbank to find the 35-cycle gap"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from theory_229_dualbank import KernelBuilderA1
from problem import SLOT_LIMITS, VLEN, Tree, Input, build_mem_image
import random

random.seed(123)
forest = Tree.generate(10)
inp = Input.generate(forest, 256, 16)

kb = KernelBuilderA1()
kb.build_kernel(forest.height, len(forest.values), len(inp.indices), 16)

total_cycles = len(kb.instrs)
print(f"Total cycles: {total_cycles}")

# Count ops per engine
engine_counts = {}
for e, s in kb.slots:
    engine_counts[e] = engine_counts.get(e, 0) + 1
print(f"\nOp counts: {engine_counts}")
for e, c in sorted(engine_counts.items()):
    limit = SLOT_LIMITS[e]
    floor = (c + limit - 1) // limit
    print(f"  {e}: {c} ops, {limit}/cycle, floor={floor}")

# Cycle-by-cycle utilization
valu_util = []
load_util = []
flow_util = []
for i, instr in enumerate(kb.instrs):
    valu_util.append(len(instr.get('valu', [])))
    load_util.append(len(instr.get('load', [])))
    flow_util.append(len(instr.get('flow', [])))

total_valu = sum(valu_util)
valu_floor = (total_valu + 5) // 6
wasted_valu = total_cycles * 6 - total_valu
print(f"\nVALU floor: {valu_floor}")
print(f"Gap from floor: {total_cycles - valu_floor}")
print(f"Total wasted VALU slots: {wasted_valu}")

# Histogram
print("\n=== VALU utilization histogram ===")
for v in range(7):
    count = sum(1 for x in valu_util if x == v)
    if count > 0:
        print(f"  VALU={v}: {count} cycles ({100*count/total_cycles:.1f}%)")

# Where are VALU=0 cycles?
print("\n=== Cycles with VALU=0 ===")
zero_valu_cycles = [i for i in range(total_cycles) if valu_util[i] == 0]
print(f"  Count: {len(zero_valu_cycles)}")
if zero_valu_cycles:
    # Group into runs
    runs = []
    run_start = zero_valu_cycles[0]
    for i in range(1, len(zero_valu_cycles)):
        if zero_valu_cycles[i] != zero_valu_cycles[i-1] + 1:
            runs.append((run_start, zero_valu_cycles[i-1]))
            run_start = zero_valu_cycles[i]
    runs.append((run_start, zero_valu_cycles[-1]))
    for start, end in runs:
        length = end - start + 1
        print(f"    Cycles {start}-{end} ({length} cycles)")

# Where are low-VALU cycles?
print("\n=== Cycles with VALU<=2 (severely underutilized) ===")
low_valu_cycles = [i for i in range(total_cycles) if valu_util[i] <= 2]
print(f"  Count: {len(low_valu_cycles)}")
# Group into runs of 3+
runs = []
if low_valu_cycles:
    run_start = low_valu_cycles[0]
    for i in range(1, len(low_valu_cycles)):
        if low_valu_cycles[i] != low_valu_cycles[i-1] + 1:
            if low_valu_cycles[i-1] - run_start >= 2:
                runs.append((run_start, low_valu_cycles[i-1]))
            run_start = low_valu_cycles[i]
    if low_valu_cycles[-1] - run_start >= 2:
        runs.append((run_start, low_valu_cycles[-1]))
    for start, end in runs:
        length = end - start + 1
        avg_v = sum(valu_util[start:end+1]) / length
        avg_l = sum(load_util[start:end+1]) / length
        avg_f = sum(flow_util[start:end+1]) / length
        wasted = sum(6 - valu_util[j] for j in range(start, end+1))
        print(f"    Cycles {start}-{end} ({length} cy, avg VALU={avg_v:.1f}, load={avg_l:.1f}, flow={avg_f:.1f}, wasted={wasted})")

# Compute running wasted VALU count at each cycle
print("\n=== Cumulative wasted VALU at key points ===")
cumulative_wasted = 0
for i in range(total_cycles):
    cumulative_wasted += (6 - valu_util[i])
    if i in [20, 30, 40, 50, 100, 200, 400, 700, 1000, 1300, total_cycles-50, total_cycles-20, total_cycles-1]:
        print(f"  Cycle {i}: cumulative wasted VALU = {cumulative_wasted}")

# Init phase detail
print("\n=== Init phase (first 40 cycles) ===")
for i in range(min(40, total_cycles)):
    instr = kb.instrs[i]
    parts = []
    for e in ['alu', 'valu', 'load', 'store', 'flow']:
        n = len(instr.get(e, []))
        if n > 0:
            parts.append(f"{e}={n}")
    print(f"  Cycle {i:3d}: {', '.join(parts):30s} [VALU waste: {6-valu_util[i]}]")

# Drain phase detail
print(f"\n=== Drain phase (last 50 cycles, {total_cycles-50}-{total_cycles-1}) ===")
for i in range(max(0, total_cycles-50), total_cycles):
    instr = kb.instrs[i]
    parts = []
    for e in ['alu', 'valu', 'load', 'store', 'flow']:
        n = len(instr.get(e, []))
        if n > 0:
            parts.append(f"{e}={n}")
    print(f"  Cycle {i:3d}: {', '.join(parts):30s} [VALU waste: {6-valu_util[i]}]")

# Count transition bubbles in the body (groups transition)
print("\n=== Group transition analysis ===")
# Find where load bursts happen (vloads indicate tile/group transitions)
load_bursts = []
for i in range(total_cycles):
    if load_util[i] == 2 and (i == 0 or load_util[i-1] < 2):
        burst_start = i
        while i < total_cycles and load_util[i] == 2:
            i += 1
        load_bursts.append((burst_start, i-1))
print(f"  Load burst regions (load=2): {len(load_bursts)}")
for start, end in load_bursts:
    length = end - start + 1
    avg_v = sum(valu_util[start:end+1]) / length
    print(f"    Cycles {start}-{end} ({length} cy, avg VALU={avg_v:.1f})")
