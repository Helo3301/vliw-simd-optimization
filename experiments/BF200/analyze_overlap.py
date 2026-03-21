"""Analyze how tiles overlap in the schedule"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from theory_226_WIN import KernelBuilderA1, _schedule_slots
from problem import SLOT_LIMITS, VLEN, Tree, Input, build_mem_image
from collections import defaultdict
import random

random.seed(123)
forest = Tree.generate(10)
inp = Input.generate(forest, 256, 16)

kb = KernelBuilderA1()
kb.build_kernel(forest.height, len(forest.values), len(inp.indices), 16)

# Get compute phase slots
compute_slots = []
past_pause = False
for e, s in kb.slots:
    if e == 'flow' and s == ('pause',):
        past_pause = True
        continue
    if past_pause:
        compute_slots.append((e, s))

# Tile boundary at slot 5360 in compute_slots
tile_boundary = 5360

# Custom scheduler that tracks per-slot cycle mapping
def schedule_with_cycle_map(slots):
    cycles = []
    usage = []
    ready_time = defaultdict(int)
    last_write = defaultdict(lambda: -1)
    last_read = defaultdict(lambda: -1)
    slot_cycle = []

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
        reads, writes = [], []
        if engine == "alu":
            _op, dest, a1, a2 = slot
            reads = [a1, a2]; writes = [dest]
        elif engine == "valu":
            op = slot[0]
            if op == "vbroadcast":
                reads = [slot[2]]; writes = list(_vec_range(slot[1]))
            elif op == "multiply_add":
                reads = list(_vec_range(slot[2])) + list(_vec_range(slot[3])) + list(_vec_range(slot[4]))
                writes = list(_vec_range(slot[1]))
            else:
                reads = list(_vec_range(slot[2])) + list(_vec_range(slot[3]))
                writes = list(_vec_range(slot[1]))
        elif engine == "load":
            op = slot[0]
            if op == "load":
                reads = [slot[2]]; writes = [slot[1]]
            elif op == "vload":
                reads = [slot[2]]; writes = list(_vec_range(slot[1]))
            elif op == "const":
                writes = [slot[1]]
            elif op == "load_offset":
                reads = [slot[2]]; writes = [slot[1]]
        elif engine == "store":
            op = slot[0]
            if op == "store":
                reads = [slot[1], slot[2]]
            elif op == "vstore":
                reads = [slot[1]] + list(_vec_range(slot[2]))
        elif engine == "flow":
            op = slot[0]
            if op == "vselect":
                reads = list(_vec_range(slot[2])) + list(_vec_range(slot[3])) + list(_vec_range(slot[4]))
                writes = list(_vec_range(slot[1]))
            elif op in ("select",):
                reads = [slot[2], slot[3], slot[4]]; writes = [slot[1]]
            elif op == "add_imm":
                reads = [slot[2]]; writes = [slot[1]]
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
        slot_cycle.append(cycle)

        for addr in reads:
            if last_read[addr] < cycle:
                last_read[addr] = cycle
        for addr in writes:
            last_write[addr] = cycle
            ready_time[addr] = cycle + 1

    return cycles, slot_cycle

cycles, slot_cycle = schedule_with_cycle_map(compute_slots)

tile1_cycles = slot_cycle[:tile_boundary]
tile2_cycles = slot_cycle[tile_boundary:]

tile1_first = min(tile1_cycles)
tile1_last = max(tile1_cycles)
tile2_first = min(tile2_cycles)
tile2_last = max(tile2_cycles)

print(f"Tile 1: cycles {tile1_first} - {tile1_last}")
print(f"Tile 2: cycles {tile2_first} - {tile2_last}")
print(f"Overlap: cycles {tile2_first} - {tile1_last} = {tile1_last - tile2_first + 1} cycles")

# Analyze what's happening in the overlap
overlap_start = tile2_first
overlap_end = tile1_last

# Count tile 1 and tile 2 ops per engine in the overlap region
t1_overlap = defaultdict(int)
t2_overlap = defaultdict(int)

for i in range(tile_boundary):
    if overlap_start <= slot_cycle[i] <= overlap_end:
        e = compute_slots[i][0]
        t1_overlap[e] += 1

for i in range(tile_boundary, len(compute_slots)):
    if overlap_start <= slot_cycle[i] <= overlap_end:
        e = compute_slots[i][0]
        t2_overlap[e] += 1

print(f"\nIn overlap region ({overlap_end - overlap_start + 1} cycles):")
print(f"  Tile 1 ops: {dict(t1_overlap)}")
print(f"  Tile 2 ops: {dict(t2_overlap)}")

# Per-cycle breakdown of overlap region
print(f"\n=== Per-cycle in overlap region ===")
for c in range(overlap_start, min(overlap_end + 1, overlap_start + 50)):
    t1_engines = defaultdict(int)
    t2_engines = defaultdict(int)
    for i in range(tile_boundary):
        if slot_cycle[i] == c:
            t1_engines[compute_slots[i][0]] += 1
    for i in range(tile_boundary, len(compute_slots)):
        if slot_cycle[i] == c:
            t2_engines[compute_slots[i][0]] += 1

    t1_str = ", ".join(f"{e}={n}" for e, n in sorted(t1_engines.items()))
    t2_str = ", ".join(f"{e}={n}" for e, n in sorted(t2_engines.items()))
    total_valu = t1_engines.get('valu', 0) + t2_engines.get('valu', 0)
    if t1_str or t2_str:
        print(f"  Cycle {c}: T1=[{t1_str}] T2=[{t2_str}] total_valu={total_valu}")
