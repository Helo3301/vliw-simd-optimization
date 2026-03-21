"""Analyze the tail of the schedule in detail"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from theory_226_WIN import KernelBuilderA1, _schedule_slots
from problem import SLOT_LIMITS, VLEN
from collections import defaultdict
import random

random.seed(123)

kb = KernelBuilderA1()
from problem import Tree, Input
random.seed(123)
forest = Tree.generate(10)
inp = Input.generate(forest, 256, 16)
kb.build_kernel(forest.height, len(forest.values), len(inp.indices), 16)

total_cycles = len(kb.instrs)

# Count per-cycle VALU utilization in last 80 cycles
print("=== Last 80 cycles: detailed utilization ===")
wasted_in_tail = 0
for i in range(max(0, total_cycles - 80), total_cycles):
    instr = kb.instrs[i]
    vn = len(instr.get('valu', []))
    ln = len(instr.get('load', []))
    fn = len(instr.get('flow', []))
    sn = len(instr.get('store', []))
    an = len(instr.get('alu', []))
    wasted = 6 - vn
    if wasted > 0 and i >= total_cycles - 60:
        wasted_in_tail += wasted
    parts = []
    if vn: parts.append(f"V={vn}")
    if ln: parts.append(f"L={ln}")
    if fn: parts.append(f"F={fn}")
    if sn: parts.append(f"S={sn}")
    if an: parts.append(f"A={an}")
    print(f"  C{i}: {', '.join(parts):30s} wasted_valu={wasted}")

print(f"\nWasted VALU in last 60 cycles: {wasted_in_tail}")
print(f"Equivalent cycles: {wasted_in_tail / 6:.1f}")

# Now look at how many groups there are and their boundaries
# With GROUP_SIZE=4 and 16 desks, there are 4 groups per tile
# Each group processes 16 rounds independently
# Let's check: with 2 tiles * 4 groups = 8 groups total

# Count per-group VALU ops
# Each group has 242 VALU per desk * 4 desks = 968 VALU
# Floor per group: 968 / 6 = 162 cycles
print(f"\n=== Per-group analysis ===")
print(f"VALU per group: 242 * 4 = {242 * 4}")
print(f"Floor per group: {(242 * 4 + 5) // 6}")
print(f"8 groups * 162 = {8 * 162} cycles (hypothetical if perfectly packed)")
print(f"Actual: {total_cycles - 39 - 1} cycles (subtracting init and final pause)")
print(f"Gap per group: ~{(total_cycles - 40 - 8 * 162) / 8:.1f} cycles")

# The drain at the end is because the last group's operations can't fill
# all 6 VALU slots as they taper off.
# With more groups, the drain would be shorter per group but happen more times.

# Key question: what if we use GROUP_SIZE=2?
# Then 8 groups per tile, 16 total
# VALU per group: 242 * 2 = 484, floor: 81 cycles
# 16 groups * 81 = 1296 (vs 1297 VALU floor)
# But each group has its own ramp up/down overhead...

# What about GROUP_SIZE=8?
# 2 groups per tile, 4 total
# VALU per group: 242 * 8 = 1936, floor: 323 cycles
# 4 groups * 323 = 1292
# More overlap between groups since each group is bigger

# The real question: does the scheduler overlap consecutive groups?
# If group N's tail overlaps with group N+1's head, we save cycles.

# Let me check: with 8 groups (4 per tile * 2 tiles), how many cycles
# does each group "own" in the schedule?

# To do this properly, I'd need to track which slot belongs to which group.
# Let me count group boundaries by looking at the slot pattern.

# Each group starts with 4 XOR ops (round 0 xor with tree[0])
# which are VALU ops with v_tree[0] as operand.
