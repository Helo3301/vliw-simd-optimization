"""Find where tile 2 starts by looking at the slot emission pattern"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from theory_226_WIN import KernelBuilderA1
from problem import SLOT_LIMITS, VLEN, Tree, Input, build_mem_image
import random

random.seed(123)
forest = Tree.generate(10)
inp = Input.generate(forest, 256, 16)

kb = KernelBuilderA1()

# Patch to track tile boundaries
orig_build = kb.build_kernel
tile_start_slots = []

def patched_build(forest_height, n_nodes, batch_size, rounds):
    # Can't easily patch, let's count a different way
    pass

# Instead, let's count by looking at the const load values in the compute phase
# Tile 0 starts with const(offset_regs[0], 0), const(offset_regs[1], 8), etc.
# Tile 1 starts with const(offset_regs[0], 128), const(offset_regs[1], 136), etc.

# Build the kernel first
kb.build_kernel(forest.height, len(forest.values), len(inp.indices), 16)

# Find const loads in compute phase
past_pause = False
compute_start_slot = None
tile2_const_start = None
for i, (e, s) in enumerate(kb.slots):
    if e == 'flow' and s == ('pause',):
        past_pause = True
        compute_start_slot = i + 1
        continue
    if past_pause and e == 'load' and s[0] == 'const':
        # Tile 2 starts with const(offset_regs[0], 16*8=128) for 16 desks
        _, dest, val = s
        if val == 128:  # Tile 2's first offset (16 desks * 8 VLEN = 128)
            tile2_const_start = i
            print(f"Tile 2 first const at slot {i}, value={val}")
            break

if tile2_const_start is None:
    print("Could not find tile 2 boundary! Trying to find it by value pattern...")
    # Look for the pattern of const loads
    const_loads = []
    for i, (e, s) in enumerate(kb.slots):
        if e == 'load' and s[0] == 'const' and past_pause:
            const_loads.append((i, s[1], s[2]))
    for i, (slot_i, dest, val) in enumerate(const_loads[:40]):
        print(f"  Const at slot {slot_i}: dest={dest}, val={val}")
    # Also search for it
    for i, (slot_i, dest, val) in enumerate(const_loads):
        if val == 128:
            tile2_const_start = slot_i
            print(f"\nFound tile 2 at const_loads[{i}]: slot {slot_i}, val={val}")
            break

# Count total slots per tile
if tile2_const_start:
    tile1_count = tile2_const_start - compute_start_slot
    tile2_count = len(kb.slots) - tile2_const_start
    print(f"\nTile 1: {tile1_count} slots (slots {compute_start_slot}-{tile2_const_start-1})")
    print(f"Tile 2: {tile2_count} slots (slots {tile2_const_start}-{len(kb.slots)-1})")

    # Now figure out how many VALU/load/flow in each tile
    for tile_name, start, end in [("Tile 1", compute_start_slot, tile2_const_start),
                                   ("Tile 2", tile2_const_start, len(kb.slots))]:
        engine_counts = {}
        for j in range(start, end):
            e, s = kb.slots[j]
            engine_counts[e] = engine_counts.get(e, 0) + 1
        print(f"\n{tile_name} ops:")
        for e in ['alu', 'valu', 'load', 'store', 'flow']:
            print(f"  {e}: {engine_counts.get(e, 0)}")
