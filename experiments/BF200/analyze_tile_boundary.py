"""Find the exact tile boundary and check for overlap"""
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

# Count how many const ops there are in the slots (tile starts have const ops)
# Each tile starts with 16 const loads for offset_regs
const_count = 0
slot_idx_to_tile = {}
in_init = True
tile_boundaries = []

for i, (engine, slot) in enumerate(kb.slots):
    if engine == 'flow' and slot == ('pause',):
        in_init = False
        continue
    if not in_init and engine == 'load' and slot[0] == 'const':
        const_count += 1
        if const_count == 1 or const_count == 17:  # First const of tile 1 or tile 2
            tile_boundaries.append(i)

print(f"Total const loads (non-init): found tile boundary slots")

# More precise: find where tile 2's const loads start in the slot list
tile2_start = None
const_in_compute = 0
for i, (engine, slot) in enumerate(kb.slots):
    if engine == 'flow' and slot == ('pause',):
        continue
    if engine == 'load' and slot[0] == 'const':
        const_in_compute += 1
        if const_in_compute == 17:  # 17th const = first of tile 2 (tile 1 has 16)
            tile2_start = i
            print(f"Tile 2 starts at slot {i} (out of {len(kb.slots)} total)")
            break

# Count slots per tile
if tile2_start:
    tile1_slots = 0
    tile2_slots = 0
    past_pause = False
    for i, (e, s) in enumerate(kb.slots):
        if e == 'flow' and s == ('pause',):
            past_pause = True
            continue
        if past_pause:
            if i < tile2_start:
                tile1_slots += 1
            else:
                tile2_slots += 1
    print(f"Tile 1: {tile1_slots} slots")
    print(f"Tile 2: {tile2_slots} slots")

# Now find what scheduled cycle tile 2's first ops land on
# We need to map from slot index to scheduled cycle
# Since the scheduler processes slots in order, we can track this
# by running a modified scheduler that records slot-to-cycle mapping

from collections import defaultdict

def schedule_with_mapping(slots):
    cycles = []
    usage = []
    ready_time = defaultdict(int)
    last_write = defaultdict(lambda: -1)
    last_read = defaultdict(lambda: -1)
    slot_to_cycle = {}

    def ensure_cycle(cycle):
        while len(cycles) <= cycle:
            cycles.append({})
            usage.append(defaultdict(int))

    def find_cycle(engine, earliest):
        cycle = earliest
        limit = SLOT_LIMITS[engine]
        while True:
            ensure_cycle(cycle)
            if usage[cycle][engine] < limit:
                return cycle
            cycle += 1

    def _vec_range(base, length=VLEN):
        return range(base, base + length)

    def _slot_rw(engine, slot):
        reads = []
        writes = []
        if engine == "alu":
            _op, dest, a1, a2 = slot
            reads = [a1, a2]
            writes = [dest]
        elif engine == "valu":
            op = slot[0]
            if op == "vbroadcast":
                dest, src = slot[1], slot[2]
                reads = [src]
                writes = list(_vec_range(dest))
            elif op == "multiply_add":
                dest, a, b, c = slot[1], slot[2], slot[3], slot[4]
                reads = list(_vec_range(a)) + list(_vec_range(b)) + list(_vec_range(c))
                writes = list(_vec_range(dest))
            else:
                _op, dest, a1, a2 = slot
                reads = list(_vec_range(a1)) + list(_vec_range(a2))
                writes = list(_vec_range(dest))
        elif engine == "load":
            op = slot[0]
            if op == "load":
                dest, addr = slot[1], slot[2]
                reads = [addr]
                writes = [dest]
            elif op == "vload":
                dest, addr = slot[1], slot[2]
                reads = [addr]
                writes = list(_vec_range(dest))
            elif op == "const":
                dest = slot[1]
                writes = [dest]
            elif op == "load_offset":
                dest, addr, _lane = slot[1], slot[2], slot[3]
                reads = [addr]
                writes = [dest]
        elif engine == "store":
            op = slot[0]
            if op == "store":
                addr, src = slot[1], slot[2]
                reads = [addr, src]
            elif op == "vstore":
                addr, src = slot[1], slot[2]
                reads = [addr] + list(_vec_range(src))
        elif engine == "flow":
            op = slot[0]
            if op == "select":
                dest, cond, a, b = slot[1], slot[2], slot[3], slot[4]
                reads = [cond, a, b]
                writes = [dest]
            elif op == "add_imm":
                dest, a = slot[1], slot[2]
                reads = [a]
                writes = [dest]
            elif op == "vselect":
                dest, cond, a, b = slot[1], slot[2], slot[3], slot[4]
                reads = list(_vec_range(cond)) + list(_vec_range(a)) + list(_vec_range(b))
                writes = list(_vec_range(dest))
        return reads, writes

    for idx, (engine, slot) in enumerate(slots):
        reads, writes = _slot_rw(engine, slot)
        earliest = 0
        for addr in reads:
            earliest = max(earliest, ready_time[addr])
        for addr in writes:
            earliest = max(earliest, last_write[addr] + 1, last_read[addr])

        cycle = find_cycle(engine, earliest)
        ensure_cycle(cycle)
        cycles[cycle].setdefault(engine, []).append(slot)
        usage[cycle][engine] += 1
        slot_to_cycle[idx] = cycle

        for addr in reads:
            if last_read[addr] < cycle:
                last_read[addr] = cycle
        for addr in writes:
            last_write[addr] = cycle
            ready_time[addr] = cycle + 1

    return [c for c in cycles if c], slot_to_cycle

# Get the compute phase slots (after pause)
compute_slots = []
past_pause = False
slot_indices = []
for i, (e, s) in enumerate(kb.slots):
    if e == 'flow' and s == ('pause',):
        past_pause = True
        continue
    if past_pause:
        compute_slots.append((e, s))
        slot_indices.append(i)

scheduled, slot_to_cycle = schedule_with_mapping(compute_slots)

# Find the scheduled cycle for tile 2's first slot
# tile2_start is in original slot indices; convert to compute_slots index
if tile2_start:
    compute_tile2_start = slot_indices.index(tile2_start)
    tile2_first_cycle = slot_to_cycle[compute_tile2_start]

    # Find last cycle of tile 1
    tile1_last_cycle = max(slot_to_cycle[i] for i in range(compute_tile2_start))
    tile2_last_cycle = max(slot_to_cycle[i] for i in range(compute_tile2_start, len(compute_slots)))

    print(f"\nTile 1: cycles 0 - {tile1_last_cycle} (scheduled)")
    print(f"Tile 2: cycles {tile2_first_cycle} - {tile2_last_cycle} (scheduled)")
    print(f"Overlap region: cycles {tile2_first_cycle} - {tile1_last_cycle}")
    print(f"Overlap: {tile1_last_cycle - tile2_first_cycle + 1} cycles")

    # Check what tile 1 is doing during the overlap
    print(f"\n=== Overlap details ===")
    tile1_ops_in_overlap = 0
    tile2_ops_in_overlap = 0
    for i in range(len(compute_slots)):
        c = slot_to_cycle[i]
        if tile2_first_cycle <= c <= tile1_last_cycle:
            if i < compute_tile2_start:
                tile1_ops_in_overlap += 1
            else:
                tile2_ops_in_overlap += 1
    print(f"Tile 1 ops in overlap: {tile1_ops_in_overlap}")
    print(f"Tile 2 ops in overlap: {tile2_ops_in_overlap}")

    # Check how many tile 1 stores are in the overlap
    tile1_stores_in_overlap = 0
    for i in range(compute_tile2_start):
        e, s = compute_slots[i]
        c = slot_to_cycle[i]
        if c >= tile2_first_cycle and e == 'store':
            tile1_stores_in_overlap += 1
    print(f"Tile 1 stores in overlap: {tile1_stores_in_overlap}")
