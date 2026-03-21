"""Analyze group boundaries in theory_229 to find pipelining opportunities"""
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

# Track which desks are active in each cycle by looking at scratch addresses
# Desks 0-7 are bank A, 8-15 are bank B
# Each desk has 8 vec regs allocated: idx, val, node_val, addr, tmp1, tmp2, bit0, bit1
# We need to find which desk ranges are accessed per cycle

total_cycles = len(kb.instrs)

# Map scratch addresses to desks
desk_to_addrs = {}
for d in range(16):
    addrs = set()
    for key in ['idx', 'val', 'node_val', 'addr', 'tmp1', 'tmp2', 'bit0', 'bit1']:
        base = kb.scratch.get(f"v_{key}_{d}")
        if base is not None:
            for lane in range(VLEN):
                addrs.add(base + lane)
    desk_to_addrs[d] = addrs

# Also track stores for tile boundaries
store_cycles = []
for i, instr in enumerate(kb.instrs):
    if 'store' in instr:
        store_cycles.append(i)

print(f"Store cycles: {store_cycles[:20]}... ({len(store_cycles)} total)")

# Find active desks per cycle by checking which desk addresses are used
# This is expensive so let's just check VALU ops
def get_active_desks(instr):
    active = set()
    for engine in ['valu', 'load', 'store', 'flow']:
        for op in instr.get(engine, []):
            # Extract all register addresses from the op
            addrs = set()
            if engine == 'valu':
                if op[0] == 'vbroadcast':
                    for lane in range(VLEN):
                        addrs.add(op[1] + lane)
                elif op[0] == 'multiply_add':
                    for base in [op[1], op[2], op[3], op[4]]:
                        for lane in range(VLEN):
                            addrs.add(base + lane)
                else:
                    for base in [op[1], op[2], op[3]]:
                        for lane in range(VLEN):
                            addrs.add(base + lane)
            elif engine == 'flow':
                if op[0] == 'vselect':
                    for base in [op[1], op[2], op[3], op[4]]:
                        for lane in range(VLEN):
                            addrs.add(base + lane)
            elif engine == 'load':
                if op[0] == 'load':
                    addrs.add(op[1])
                elif op[0] == 'vload':
                    for lane in range(VLEN):
                        addrs.add(op[1] + lane)
            elif engine == 'store':
                if op[0] == 'vstore':
                    for lane in range(VLEN):
                        addrs.add(op[2] + lane)

            for d in range(16):
                if addrs & desk_to_addrs[d]:
                    active.add(d)
    return active

# Sample active desks at key cycle ranges
print("\n=== Active desks sampling ===")
# Check every 50 cycles in the body
for c in range(0, total_cycles, 50):
    desks = get_active_desks(kb.instrs[c])
    if desks:
        banks = {'A': sorted(d for d in desks if d < 8), 'B': sorted(d for d in desks if d >= 8)}
        print(f"  Cycle {c:4d}: Bank A desks={banks['A']}, Bank B desks={[d-8 for d in banks['B']]}")

# More detail around group transitions
print("\n=== Detailed active desks around transitions ===")
prev_desks = set()
for c in range(35, min(total_cycles, 800)):
    desks = get_active_desks(kb.instrs[c])
    if desks != prev_desks:
        banks = {'A': sorted(d for d in desks if d < 8), 'B': sorted(d for d in desks if d >= 8)}
        valu = len(kb.instrs[c].get('valu', []))
        if desks:  # Skip empty cycles
            # Check if this is a transition (new desk appeared or old disappeared)
            new_desks = desks - prev_desks
            gone_desks = prev_desks - desks
            if new_desks or gone_desks:
                print(f"  Cycle {c:4d}: VALU={valu} | +{sorted(new_desks)} -{sorted(gone_desks)} | Active: A={banks['A']} B={[d-8 for d in banks['B']]}")
        prev_desks = desks

# Count how many cycles each desk group is active
print("\n=== Desk activity spans ===")
for d in range(16):
    first_cycle = None
    last_cycle = None
    for c in range(total_cycles):
        desks = get_active_desks(kb.instrs[c])
        if d in desks:
            if first_cycle is None:
                first_cycle = c
            last_cycle = c
    if first_cycle is not None:
        print(f"  Desk {d:2d}: cycles {first_cycle:4d} - {last_cycle:4d} (span={last_cycle-first_cycle+1})")
