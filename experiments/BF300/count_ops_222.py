"""Count operations by type in Theory 222"""
import sys
sys.path.insert(0, '/home/hestiasadmin/projects/original_performance_takehome')
sys.path.insert(0, '/home/hestiasadmin/projects/original_performance_takehome/experiments/BF300')

from theory_222_hashorder_WIN import KernelBuilderA1
from problem import *
from collections import Counter
import random

random.seed(123)
forest = Tree.generate(10)
inp = Input.generate(forest, 256, 16)

kb = KernelBuilderA1()
kb.build_kernel(forest.height, len(forest.values), 256, 16)

# Count by engine and operation
engine_counts = Counter()
op_counts = Counter()
valu_op_counts = Counter()
flow_op_counts = Counter()
load_op_counts = Counter()

for engine, slot in kb.slots:
    engine_counts[engine] += 1
    op = slot[0]
    op_counts[(engine, op)] += 1
    if engine == 'valu':
        valu_op_counts[op] += 1
    elif engine == 'flow':
        flow_op_counts[op] += 1
    elif engine == 'load':
        load_op_counts[op] += 1

print("=== Engine Counts ===")
for engine, count in sorted(engine_counts.items()):
    print(f"  {engine}: {count}")

print("\n=== VALU Operations ===")
for op, count in sorted(valu_op_counts.items(), key=lambda x: -x[1]):
    print(f"  {op}: {count}")

print("\n=== Flow Operations ===")
for op, count in sorted(flow_op_counts.items(), key=lambda x: -x[1]):
    print(f"  {op}: {count}")

print("\n=== Load Operations ===")
for op, count in sorted(load_op_counts.items(), key=lambda x: -x[1]):
    print(f"  {op}: {count}")

# Per-round breakdown
# The structure is: init, then per-group:
#   emit_rounds_0_1_2_3_fused -> rounds 4-9 gather -> R10 -> rounds 11-14 fused -> R15 -> stores
# Let's figure out per-section VALU counts

# Count VALU in each function by tracking emissions
print("\n=== Per-section analysis ===")

# The kernel has 2 tiles, each with 4 groups of 4 desks = 16 desks
# Per group: fused_0123, 6 gathers (R4-R9), R10, fused_11_12_13_14, R15
# Let's compute from the known structure

TILES = 2
GROUPS = 4
DESKS_PER_GROUP = 4
D = DESKS_PER_GROUP

# Per-desk VALU counts in each section:
# fused_0123:
#   R0: XOR(1) + hash(12) + AND(1) = 14
#   R1: XOR(1) + hash(12) + AND(1) = 14
#   R2: XOR(1) + hash(12) + AND(1) = 14
#   R3: XOR(1) + hash(12) + AND(1) + deferred_addr(4+1=5) = 19
#   Total R0-R3: 14+14+14+19 = 61 per desk
print(f"Fused R0-R3 per desk: XOR+hash+AND = 14/round, R3 has +5 for addr = 61 VALU/desk")
fused_0123_per_group = 61 * D
print(f"  Per group: {fused_0123_per_group}")

# R4-R9 (6 gather rounds, addr-tracking):
#   XOR(1) + hash(12) + branch(3) = 16 per desk per round
# (branch = AND + FMA + ADD = 3 VALU)
gather_per_round_per_desk = 16
gather_per_group = gather_per_round_per_desk * D * 6
print(f"R4-R9 gather (6 rounds): 16 VALU/desk/round = {gather_per_group}/group")

# R10 (no branch):
#   XOR(1) + hash(12) = 13 per desk
r10_per_group = 13 * D
print(f"R10 (no branch): 13 VALU/desk = {r10_per_group}/group")

# Fused R11-R14: same as R0-R3
fused_1114_per_group = 61 * D
print(f"Fused R11-R14: same as R0-R3 = {fused_1114_per_group}/group")

# R15 (no branch):
#   XOR(1) + hash(12) = 13 per desk
r15_per_group = 13 * D
print(f"R15 (no branch): 13 VALU/desk = {r15_per_group}/group")

total_per_group = fused_0123_per_group + gather_per_group + r10_per_group + fused_1114_per_group + r15_per_group
total_per_tile = total_per_group * GROUPS
total = total_per_tile * TILES

print(f"\nTotal per group: {total_per_group}")
print(f"Total per tile: {total_per_tile}")
print(f"Total (2 tiles): {total}")

# Init VALU: vbroadcast and other setup
init_valu = valu_op_counts.get('vbroadcast', 0)
print(f"\nvbroadcast count (most are init): {init_valu}")
print(f"Expected total VALU: {total} + init")

# The actual count
print(f"Actual VALU: {engine_counts['valu']}")
print(f"Difference: {engine_counts['valu'] - total}")

# Flow operations analysis
print(f"\n=== Flow analysis ===")
print(f"Total flow: {engine_counts['flow']}")
# Per fused block: R1 has 1 vselect, R2 has 3 vselect, R3 has 7 vselect = 11 per desk
# Per fused block: 11 * 4 desks = 44 per group
# Per tile: 44 * 4 groups = 176 per tile
# 2 tiles = 352 vselects
# x2 for R0-R3 and R11-R14 = 704 vselects + 2 pauses
flow_vselects = flow_op_counts.get('vselect', 0)
print(f"vselect: {flow_vselects} (expected: 11*{D}*{GROUPS}*{TILES}*2 = {11*D*GROUPS*TILES*2})")
print(f"pause: {flow_op_counts.get('pause', 0)}")
