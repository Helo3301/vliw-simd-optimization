"""Analyze the tile boundary in theory_226_WIN to find overlap opportunities"""
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

total_cycles = len(kb.instrs)

# Find the tile boundary - look for the transition region
# The init phase is the pause at cycle ~27, so tile 1 starts at ~28
# Look for where stores happen (tile 1 end) and loads happen (tile 2 start)

print("=== Cycle-by-cycle detail around mid-point ===")
mid = total_cycles // 2
for i in range(max(0, mid - 20), min(total_cycles, mid + 20)):
    instr = kb.instrs[i]
    engines = []
    for e in ['alu', 'valu', 'load', 'store', 'flow']:
        n = len(instr.get(e, []))
        if n > 0:
            engines.append(f"{e}={n}")
            if e == 'flow':
                for s in instr['flow']:
                    if s[0] == 'pause':
                        engines.append('PAUSE')
    print(f"  Cycle {i}: {', '.join(engines)}")

# Find all store operations - they mark tile ends
print("\n=== Store locations ===")
store_cycles = []
for i, instr in enumerate(kb.instrs):
    if 'store' in instr:
        store_cycles.append(i)
        if len(store_cycles) <= 5 or len(store_cycles) > total_cycles - 5:
            print(f"  Cycle {i}: {len(instr['store'])} stores")

if store_cycles:
    print(f"  First store at cycle {store_cycles[0]}")
    print(f"  Last store at cycle {store_cycles[-1]}")
    print(f"  Total store cycles: {len(store_cycles)}")

# Find load-heavy regions (tile starts)
print("\n=== Load-heavy regions (load=2 with no VALU) ===")
for i, instr in enumerate(kb.instrs):
    vn = len(instr.get('valu', []))
    ln = len(instr.get('load', []))
    if ln == 2 and vn == 0 and i > 40:
        # This likely marks tile start
        print(f"  Cycle {i}: load={ln}, alu={len(instr.get('alu', []))}, store={len(instr.get('store', []))}")

# Now look at the very end to understand drain
print("\n=== Last 40 cycles (drain phase) ===")
for i in range(max(0, total_cycles - 40), total_cycles):
    instr = kb.instrs[i]
    engines = []
    for e in ['alu', 'valu', 'load', 'store', 'flow']:
        n = len(instr.get(e, []))
        if n > 0:
            engines.append(f"{e}={n}")
            if e == 'store':
                engines.append(f"(stores:{n})")
    print(f"  Cycle {i}: {', '.join(engines)}")

# Count ops by type at a more granular level
print("\n=== Total op budget breakdown ===")
valu_ops_by_type = {}
for e, s in kb.slots:
    if e == 'valu':
        valu_ops_by_type[s[0]] = valu_ops_by_type.get(s[0], 0) + 1

for op, count in sorted(valu_ops_by_type.items(), key=lambda x: -x[1]):
    print(f"  VALU {op}: {count}")

flow_ops_by_type = {}
for e, s in kb.slots:
    if e == 'flow':
        flow_ops_by_type[s[0]] = flow_ops_by_type.get(s[0], 0) + 1

print()
for op, count in sorted(flow_ops_by_type.items(), key=lambda x: -x[1]):
    print(f"  Flow {op}: {count}")
