"""Compute exact VALU budget for theory_226"""
# Per desk per tile:
# Fused block 1 (R0-R3):
#   R0: XOR(1) + Hash(12) + AND_bit0(1) = 14
#   R1: XOR(1) + Hash(12) + AND_bit1(1) = 14
#   R2: XOR(1) + Hash(12) + AND_bit2(1) = 14
#   R3: XOR(1) + Hash(12) + AND_bit3(1) + FMA_chain(3)+ADD(1) = 18
# Subtotal R0-R3: 14+14+14+18 = 60

# Addr-tracking rounds (R4-R9, 6 rounds):
#   Each: XOR(1) + Hash(12) + Branch(AND+FMA+ADD=3) = 16
# Subtotal R4-R9: 6 * 16 = 96

# R10 (optimized, no branch):
#   XOR(1) + Hash(12) = 13
# Subtotal R10: 13

# Fused block 2 (R11-R14): same as R0-R3
# Subtotal R11-R14: 60

# R15 (final, no branch):
#   XOR(1) + Hash(12) = 13
# Subtotal R15: 13

# Per desk per tile TOTAL: 60 + 96 + 13 + 60 + 13 = 242

per_desk_per_tile = 242
num_desks = 16
num_tiles = 2

# Total computation VALU (not counting init)
compute_valu = per_desk_per_tile * num_desks * num_tiles
print(f"Compute VALU: {per_desk_per_tile} * {num_desks} * {num_tiles} = {compute_valu}")

# Init VALU: vbroadcasts and operations
# From the code:
# - 5 vbroadcasts for consts (v_zero, v_one, v_two, v_fifteen, and later ones)
# Actually let me count from the emit calls:
# scratch_vconst calls: 0, 1, 2, 15 = 4 values, each needs vbroadcast = 4
# v_n_nodes: vbroadcast = 1
# v_forest_p: vbroadcast = 1
# v_1_minus_fp: SUB = 1
# v_fp_plus_1: ADD = 1
# v_fp_plus_15: ADD = 1
# Hash consts: 6 vbroadcasts for val1 constants
# Hash shifts: 3 vbroadcasts for shift values (stages 1, 3, 5)
# FMA multipliers: 3 vbroadcasts (4097, 33, 9)
# Total hash vbroadcasts: 6 + 3 + 3 = 12
# Tree nodes: 15 vbroadcasts
# Total init: 4 + 1 + 1 + 1 + 1 + 1 + 12 + 15 = 36 (includes the SUB and ADDs)
# Wait, scratch_vconst for 0, 1, 2 = 3 values. Then later 15 is another.
# And the FMA mult 4097, 33, 9 each need their scalar const + vbroadcast = 2 each but scratch_vconst handles both.

# Actually let me just count: total is 7779 compute should be 242*16*2 = 7744
# So init overhead = 7779 - 7744 = 35 VALU ops
init_valu = 7779 - compute_valu
print(f"Init VALU: {init_valu}")
print(f"Total VALU: {compute_valu + init_valu} = {7779}")

# Now compute budget at different desk counts:
print("\n=== Desk count vs VALU vs Floor ===")
for num_d in [8, 10, 12, 14, 16, 20, 24, 32]:
    batch = 256
    vlen = 8
    n_tiles = batch // (num_d * vlen)
    if batch % (num_d * vlen) != 0:
        continue
    total = per_desk_per_tile * num_d * n_tiles + init_valu
    floor = (total + 5) // 6
    # Estimate flow ops
    # Per desk per tile: fused blocks have vselects
    # R1: 1 vselect per desk
    # R2: 3 vselects per desk
    # R3: 7 vselects per desk
    # R12: 1 vselect per desk
    # R13: 3 vselects per desk
    # R14: 7 vselects per desk
    # Total: 22 vselects per desk per tile
    vselects_per_desk_tile = 22
    total_flow = vselects_per_desk_tile * num_d * n_tiles + 1  # +1 for pause
    # Load ops
    # Per desk per tile:
    #   Init: 2 vloads = 2 loads (actually counted as 2 but each vload is 1 load op)
    #   R0-R3: no gathers (use preloaded tree)
    #   R4-R9: 6 rounds * 8 loads = 48
    #   R10: 8 loads
    #   R11-R14: no gathers
    #   R15: 8 loads
    #   Store: 2 vstores (not loads)
    # Per desk per tile loads: 48 + 8 + 8 = 64 gather loads + 2 vloads
    # Total loads = (64 + 2) * num_d * n_tiles + init_loads (~47)
    loads_per_desk_tile = 64 + 2  # gather loads + vloads
    init_loads = 47  # approximately
    total_loads = loads_per_desk_tile * num_d * n_tiles + init_loads
    load_floor = (total_loads + 1) // 2

    print(f"  {num_d} desks x {n_tiles} tiles: VALU={total}, floor={floor}, flow={total_flow}, load_floor={load_floor}")

# Focus on where the 105-cycle gap comes from
print("\n=== Gap analysis (16 desks, 2 tiles) ===")
print(f"VALU floor: 1297")
print(f"Actual: 1402")
print(f"Gap: 105 cycles")
print(f"Wasted VALU: 633 slots = 105.5 avg-cycles")
print()
print("Breakdown of wasted VALU by region:")
print("  Init (0-38): ~39 cycles, ~194 wasted VALU")
print("  Drain (1362-1401): ~40 cycles, ~141 wasted VALU")
print("  Mid body stalls: ~26 cycles scattered")
print()
print("To save 39 cycles:")
print("  Option A: Cut 234 VALU ops (reduces floor by 39)")
print("  Option B: Fill init/drain with useful VALU work")
print("  Option C: Overlap tile boundaries better")
print("  Option D: Reduce number of desks to reduce drain")
