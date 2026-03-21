"""Calculate scratch budget for various configurations"""
SCRATCH_SIZE = 1536
VLEN = 8

# Fixed overhead:
# tmp_scalar: 1
# tmp_addr: 1
# n_nodes, forest_values_p, inp_indices_p, inp_values_p: 4
# scalar consts: 0, 1, 2, 15 = 4 scalars
# v_zero, v_one, v_two, v_fifteen: 4 * 8 = 32
# v_n_nodes, v_forest_p, v_1_minus_fp, v_fp_plus_1, v_fp_plus_15: 5 * 8 = 40
# Hash consts: 6 scalars + 6*8 vectors = 6 + 48 = 54
# Hash shifts: 3 scalars + 3*8 vectors = 3 + 24 = 27
# FMA multipliers: 3 scalars + 3*8 vectors = 3 + 24 = 27
# Tree nodes: 15 * 8 = 120 (vectors) + 15 scalars from scratch_const(i) = 15 + 120 = 135
# Actually the tree nodes reuse scalar consts via scratch_const, but we need those too
# Let me count more carefully...

# Fixed scalars: tmp_scalar, tmp_addr, n_nodes, forest_values_p, inp_indices_p, inp_values_p = 6
# Scalar consts: 0, 1, 2, 15, and the tree indices 0..14 = 4 + 15 = 19 (but 0, 1, 2 overlap)
# Actually scratch_const(0), scratch_const(1), scratch_const(2), scratch_const(15), scratch_const(0..14)
# 0, 1, 2 are already allocated. 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 are new = 13 new
# Total scalar consts: 4 + 13 = 17 (values 0, 1, 2, 3, 4, 5, ..., 14, 15)
# But actually scratch_const caches, so: 0, 1, 2, 15, then 0 reused, 1 reused, 2 reused, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14
# New scalars: 0, 1, 2, 15, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 = 16 scalars
# Then hash val1 consts: depends on HASH_STAGES values, let's count from problem.py
# FMA mult consts: 4097, 33, 9 = 3 scalars
# Hash shift vals: depend on HASH_STAGES
# Let me just be precise...

# Actually, let me just compute from the code structure:
# Fixed scalar scratch:
fixed_scalars = 6  # tmp_scalar, tmp_addr, n_nodes, forest_values_p, inp_indices_p, inp_values_p

# Scalar consts (unique values): each scratch_const call allocates 1 scalar if not cached
# Values used: 0, 1, 2, 15 (from vconsts), 0-14 (tree indices), hash val1 values, hash shift values, FMA mults
# The hash-related values are from HASH_STAGES; let me just estimate
# From HASH_STAGES: 6 stages, each has val1 and val3
# Plus FMA multipliers: 4097, 33, 9
# These are all unique large numbers, so ~6 + 3 + 3 = 12 hash-related scalars
# Plus 0-15 = 16 small scalars
# Total scalar consts estimate: ~28 unique scalars
scalar_consts_est = 28

# Vector consts: v_zero, v_one, v_two, v_fifteen, v_n_nodes, v_forest_p,
# v_1_minus_fp, v_fp_plus_1, v_fp_plus_15, 6 hash consts, 3 hash shifts, 3 FMA mults
vec_consts = (5 + 4 + 6 + 3 + 3) * VLEN  # = 21 * 8 = 168

# Tree preloads: 15 * 8 = 120
tree_vecs = 15 * VLEN  # = 120

# Per desk: 8 vec registers * VLEN
per_desk_vecs = 8 * VLEN  # = 64

# Per bank: offset_regs + addr_tmp
# offset_regs: DESKS_PER_BANK scalars
# addr_tmp: DESKS_PER_BANK * 2 scalars

fixed_overhead = fixed_scalars + scalar_consts_est + vec_consts + tree_vecs
print(f"Fixed overhead: {fixed_overhead}")
print(f"  Scalars: {fixed_scalars + scalar_consts_est}")
print(f"  Vec consts: {vec_consts}")
print(f"  Tree preloads: {tree_vecs}")

print(f"\nSCRATCH_SIZE = {SCRATCH_SIZE}")
remaining = SCRATCH_SIZE - fixed_overhead
print(f"Remaining for desks + bank scaffolding: {remaining}")

for num_desks in [8, 10, 12, 14, 16, 18, 20, 24, 32]:
    batch = 256
    if batch % (num_desks * VLEN) != 0:
        continue
    n_tiles = batch // (num_desks * VLEN)

    desk_scratch = num_desks * per_desk_vecs

    # Bank scaffolding depends on banking strategy
    # For dual-bank: 2 banks of num_desks/2
    # offset_regs: num_desks/2 * 2 banks = num_desks
    # addr_tmp: num_desks/2 * 2 * 2 banks = num_desks * 2
    bank_scaffold = num_desks + num_desks * 2  # = 3 * num_desks

    # For single bank (theory_226 style):
    # offset_regs: num_desks
    # addr_tmp: num_desks * 2
    single_scaffold = num_desks + num_desks * 2  # = 3 * num_desks

    total = fixed_overhead + desk_scratch + bank_scaffold
    total_single = fixed_overhead + desk_scratch + single_scaffold
    fits = "OK" if total <= SCRATCH_SIZE else "EXCEEDS"

    print(f"\n  {num_desks:2d} desks x {n_tiles} tiles: desk_scratch={desk_scratch}, scaffold={bank_scaffold}, total={total} [{fits}]")
    print(f"     Remaining: {SCRATCH_SIZE - total}")
