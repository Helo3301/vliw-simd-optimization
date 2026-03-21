"""Analyze cycle-by-cycle utilization of theory_226_WIN"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from theory_226_WIN import KernelBuilderA1
from problem import SLOT_LIMITS, VLEN, Tree, Input, build_mem_image
import random

random.seed(123)
forest = Tree.generate(10)
inp = Input.generate(forest, 256, 16)

kb = KernelBuilderA1()
kb.build_kernel(forest.height, len(forest.values), len(inp.indices), 16)

# Count ops per engine
engine_counts = {}
for e, s in kb.slots:
    engine_counts[e] = engine_counts.get(e, 0) + 1

print("=== Op counts ===")
for e, c in sorted(engine_counts.items()):
    limit = SLOT_LIMITS[e]
    floor = (c + limit - 1) // limit
    print(f"  {e}: {c} ops, {limit}/cycle, floor={floor}")

total_cycles = len(kb.instrs)
print(f"\nTotal cycles: {total_cycles}")

# Count per-cycle utilization
valu_util = []
load_util = []
flow_util = []
alu_util = []
store_util = []

for i, instr in enumerate(kb.instrs):
    valu_util.append(len(instr.get('valu', [])))
    load_util.append(len(instr.get('load', [])))
    flow_util.append(len(instr.get('flow', [])))
    alu_util.append(len(instr.get('alu', [])))
    store_util.append(len(instr.get('store', [])))

# Find underutilized regions
print("\n=== VALU utilization histogram ===")
for v in range(7):
    count = sum(1 for x in valu_util if x == v)
    if count > 0:
        print(f"  VALU={v}: {count} cycles ({100*count/total_cycles:.1f}%)")

total_valu_used = sum(valu_util)
total_valu_capacity = total_cycles * 6
print(f"\nVALU used: {total_valu_used}/{total_valu_capacity} ({100*total_valu_used/total_valu_capacity:.1f}%)")
print(f"VALU wasted slots: {total_valu_capacity - total_valu_used}")

# Find load utilization
print("\n=== Load utilization histogram ===")
for v in range(3):
    count = sum(1 for x in load_util if x == v)
    if count > 0:
        print(f"  Load={v}: {count} cycles ({100*count/total_cycles:.1f}%)")

total_load_used = sum(load_util)
total_load_capacity = total_cycles * 2
print(f"\nLoad used: {total_load_used}/{total_load_capacity} ({100*total_load_used/total_load_capacity:.1f}%)")

# Find flow utilization
print("\n=== Flow utilization histogram ===")
for v in range(2):
    count = sum(1 for x in flow_util if x == v)
    if count > 0:
        print(f"  Flow={v}: {count} cycles ({100*count/total_cycles:.1f}%)")

total_flow_used = sum(flow_util)
print(f"\nFlow used: {total_flow_used}/{total_cycles} ({100*total_flow_used/total_cycles:.1f}%)")

# Find regions where VALU < 4 (underutilized)
print("\n=== Underutilized VALU regions (VALU < 4 for 3+ consecutive cycles) ===")
run_start = None
for i in range(total_cycles):
    if valu_util[i] < 4:
        if run_start is None:
            run_start = i
    else:
        if run_start is not None and i - run_start >= 3:
            run_len = i - run_start
            avg_valu = sum(valu_util[run_start:i]) / run_len
            avg_load = sum(load_util[run_start:i]) / run_len
            avg_flow = sum(flow_util[run_start:i]) / run_len
            wasted = sum(6 - valu_util[j] for j in range(run_start, i))
            print(f"  Cycles {run_start}-{i-1}: {run_len} cycles, avg VALU={avg_valu:.1f}, avg Load={avg_load:.1f}, avg Flow={avg_flow:.1f}, wasted VALU={wasted}")
        run_start = None
if run_start is not None and total_cycles - run_start >= 3:
    run_len = total_cycles - run_start
    avg_valu = sum(valu_util[run_start:total_cycles]) / run_len
    wasted = sum(6 - valu_util[j] for j in range(run_start, total_cycles))
    print(f"  Cycles {run_start}-{total_cycles-1}: {run_len} cycles, avg VALU={avg_valu:.1f}, wasted VALU={wasted}")

# Look at what's happening in cycles 0-50 (init)
print("\n=== Init phase (cycles 0-50) ===")
for i in range(min(50, total_cycles)):
    engines = []
    for e in ['alu', 'valu', 'load', 'store', 'flow']:
        n = len(kb.instrs[i].get(e, []))
        if n > 0:
            engines.append(f"{e}={n}")
    if valu_util[i] < 4:
        print(f"  Cycle {i}: {', '.join(engines)}")

# Find flow-heavy cycles
print("\n=== Flow-heavy regions (consecutive flow=1 cycles) ===")
flow_run_start = None
for i in range(total_cycles):
    if flow_util[i] >= 1:
        if flow_run_start is None:
            flow_run_start = i
    else:
        if flow_run_start is not None and i - flow_run_start >= 5:
            run_len = i - flow_run_start
            avg_valu = sum(valu_util[flow_run_start:i]) / run_len
            print(f"  Cycles {flow_run_start}-{i-1}: {run_len} flow cycles, avg VALU={avg_valu:.1f}")
        flow_run_start = None

# Compute theoretical floors
print("\n=== Theoretical analysis ===")
print(f"VALU floor: {(engine_counts.get('valu',0) + 5) // 6}")
print(f"Load floor: {(engine_counts.get('load',0) + 1) // 2}")
print(f"Flow floor: {engine_counts.get('flow',0)}")
print(f"Store floor: {(engine_counts.get('store',0) + 1) // 2}")
print(f"ALU floor: {(engine_counts.get('alu',0) + 11) // 12}")
print(f"Actual: {total_cycles}")
print(f"Gap from VALU floor: {total_cycles - (engine_counts.get('valu',0) + 5) // 6}")
