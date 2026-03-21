"""Analyze VALU budget per round type to find reduction opportunities.

Goal: We need 35 fewer cycles. That means either:
(a) Reduce total VALU ops by 210 (= 35 * 6), OR
(b) Find a way to overlap more groups, OR
(c) Some combination

This script counts exactly what VALU operations each round type needs
per desk and where reductions might be possible.
"""

# Per-desk VALU ops per round type:
#
# Hash function: 12 VALU ops per desk per hash call
#   FMA(val, val, mult0, const0)      -- multiply_add
#   XOR(tmp1, val, const1)             -- ^
#   SHR(tmp2, val, shift1)             -- >>
#   XOR(val, tmp1, tmp2)               -- ^
#   FMA(val, val, mult2, const2)       -- multiply_add
#   ADD(tmp1, val, const3)             -- +
#   SHL(tmp2, val, shift3)             -- <<
#   XOR(val, tmp1, tmp2)               -- ^
#   FMA(val, val, mult4, const4)       -- multiply_add
#   XOR(tmp1, val, const5)             -- ^
#   SHR(tmp2, val, shift5)             -- >>
#   XOR(val, tmp1, tmp2)               -- ^
# Total: 12 VALU
#
# Branch + address tracking: 3 VALU per desk per gather round
#   AND(tmp1, val, v_one)              -- extract bit
#   FMA(addr, addr, v_two, v_1_minus_fp)  -- addr = 2*addr + (1-fp)
#   ADD(addr, addr, tmp1)              -- addr += bit
# Total: 3 VALU
#
# Gather round (with addr tracking):
#   8 loads (per lane) + 1 XOR + 12 hash + 3 branch = 16 VALU per desk
#
# Fused block R0-R3:
#   R0: 1 XOR + 12 hash + 1 AND = 14 VALU per desk
#   R1: 0 vselect (flow) + 1 XOR + 12 hash + 1 AND = 14 VALU per desk
#   R2: 0 vselect (flow) + 1 XOR + 12 hash + 1 AND = 14 VALU per desk
#   R3: 0 vselect (flow) + 1 XOR + 12 hash + 1 AND + 4 FMA/ADD = 18 VALU per desk
#   Total R0-R3: 60 VALU per desk
#
# Gather rounds R4-R9 (6 rounds): 16 * 6 = 96 VALU per desk
#
# R10 (gather, no branch): 8 loads + 1 XOR + 12 hash = 13 VALU per desk
#
# Fused block R11-R14: same as R0-R3 = 60 VALU per desk
#
# R15 (gather, no branch, no addr): 8 loads + 1 XOR + 12 hash = 13 VALU per desk
#
# Total per desk per tile: 60 + 96 + 13 + 60 + 13 = 242 VALU
# Total for 16 desks * 2 tiles: 16 * 2 * 242 = 7,744 VALU
# Plus init: ~35 VALU (vbroadcasts etc)
# Total: ~7,779 VALU
# Floor: ceil(7779/6) = 1,297 cycles

# Now: where can we save VALU?
#
# Option 1: Eliminate the AND(tmp1, val, v_one) -- extract low bit
#   This is needed in: R0, R1, R2, R3 (within fused blocks) = 4 per desk per fused block
#   Also in gather rounds: 1 per desk per round
#   Total: 4 + 6 + 4 = 14 per desk per tile
#   But we NEED the low bit for branching... unless we can use FMA differently
#
# Option 2: Reduce hash stages from 6 to fewer
#   Each hash pair (op+shift) is 4 VALU. 3 pairs = 12.
#   If we can fuse ops differently (FMA replaces 2 ops in stages 0,2,4)
#   But we already use FMA! The FMA replaces what would be separate multiply + add.
#   Current: FMA, XOR, SHR, XOR = 4 ops per pair (for pairs 1,3,5)
#   And: FMA = 1 op replacing multiply + add (for pairs 0,2,4)
#   So we have: 3 * (1+3) = 12 VALU. Can't reduce further.
#
# Option 3: Eliminate address tracking ops in some rounds
#   In R3 and R14, we compute addr from bits using FMA chain:
#     AND(tmp1, val, v_one)  -- 1 VALU
#     FMA(addr, bit0, v_two, bit1) -- 1 VALU
#     FMA(addr, addr, v_two, idx)  -- 1 VALU
#     FMA(addr, addr, v_two, tmp1) -- 1 VALU
#     ADD(addr, addr, v_fp_plus_15) -- 1 VALU
#   = 5 VALU per desk. Already very efficient.
#
# Option 4: Skip bounds check (idx wrapping)
#   The reference does: idx = 0 if idx >= n_nodes else idx
#   Our kernel handles this via addr tracking which naturally stays in range
#   for levels 0-3 (preloaded). For gather rounds, we load from computed addr.
#   We already DON'T do bounds checking in gather rounds -- the tree has 2^11-1 = 2047 nodes.
#   Height 10 means levels 0-10. After 10 rounds of branching from root,
#   we'd reach level 10 (leaves). But we have 16 rounds!
#   Rounds 11-14 are the second fused block -- they start from root again?
#   No -- R10 doesn't reset idx. R11-14 is another fused block because
#   the tree wrapping means we hit root.
#
# Wait -- let me re-examine the kernel structure:
# R0-R3: Fused (use preloaded tree nodes for levels 0-3)
# R4-R9: Gather rounds (6 rounds, traverse levels 4-9)
# R10: Gather round (no addr tracking - about to enter fused again?)
# R11-R14: Fused (tree levels 0-3 again because wrapping!)
# R15: Final gather (no addr tracking needed for output)
#
# After R10 we're at level 10 (the deepest). The next branch goes to
# level 11 which doesn't exist (idx >= n_nodes), so wraps to 0.
# That's why R11 starts with XOR against tree[0] again!
# The addr tracking in R10 is skipped because we KNOW the wrap happens.
# And then R11-R14 repeat the fused block for levels 0-3.
# R15 does one more gather at level 4.
#
# After R15, we're at level 5 (index ~31-62). This idx gets stored.
# The next round of 16 starts from this idx.
#
# The trick: after 16 rounds, forest_height=10 means:
# R0-R3: levels 0-3 (preloaded, fused)
# R4-R9: levels 4-9 (gathered)
# R10: level 10 (last real level)
# R11: wraps to root (level 0), R11-R14: levels 0-3 (fused)
# R15: level 4 (gathered)
# Final idx at level 5 depth

# OK so the VALU budget is pretty tight. Let me think about the drain differently.
#
# The drain is 72 cycles of overhead. The last group (desks 12-15) finishes
# without overlapping work to fill the VALU slots.
#
# What if we create ARTIFICIAL work to fill the drain?
# Can we start loading/preparing the NEXT tile pair during the drain?
#
# In theory_229, the structure is:
# Pair 0: load(tile0,tile1) -> compute(A_G0, B_G0, A_G1, B_G1) -> store(tile0,tile1)
# Pair 1: load(tile2,tile3) -> compute(A_G0, B_G0, A_G1, B_G1) -> store(tile2,tile3)
#
# What if we overlap pair 1's loads with pair 0's stores?
# The stores use store engine (2/cycle), loads use load engine (2/cycle).
# If we emit pair 1 loads BEFORE pair 0 stores, the scheduler would overlap them.
# But pair 1 needs the same scratch addresses (same desks!)...
# Unless we use DIFFERENT desks for pair 1's early loads.
#
# Actually wait - the stores write to memory, and the loads for pair 1 read from
# memory too. There's no memory ordering hazard because different tiles access
# different memory addresses.
#
# The real constraint: the vload for pair 1 writes to the same desk scratch
# as pair 0's vstore reads from. We'd need to store pair 0 BEFORE loading pair 1
# into the same registers.
#
# Unless... we have extra scratch for a small "prefetch buffer" for pair 1.

print("""
=== VALU Budget Analysis ===

Per-desk per-tile VALU breakdown:
  R0:  XOR(1) + Hash(12) + AND(1)               = 14
  R1:  XOR(1) + Hash(12) + AND(1)               = 14
  R2:  XOR(1) + Hash(12) + AND(1)               = 14
  R3:  XOR(1) + Hash(12) + AND(1) + FMA_chain(4) = 18
  R4:  XOR(1) + Hash(12) + Branch(3)             = 16
  R5:  XOR(1) + Hash(12) + Branch(3)             = 16
  R6:  XOR(1) + Hash(12) + Branch(3)             = 16
  R7:  XOR(1) + Hash(12) + Branch(3)             = 16
  R8:  XOR(1) + Hash(12) + Branch(3)             = 16
  R9:  XOR(1) + Hash(12) + Branch(3)             = 16
  R10: XOR(1) + Hash(12)                         = 13
  R11: XOR(1) + Hash(12) + AND(1)               = 14
  R12: XOR(1) + Hash(12) + AND(1)               = 14
  R13: XOR(1) + Hash(12) + AND(1)               = 14
  R14: XOR(1) + Hash(12) + AND(1) + FMA_chain(4) = 18
  R15: XOR(1) + Hash(12)                         = 13

  Total per desk per tile: 242 VALU

  16 desks * 2 tile-pairs: 7,744 VALU
  Init (vbroadcasts, etc): ~35 VALU
  Total: ~7,779 VALU
  Floor: ceil(7779/6) = 1,297 cycles

  Current: 1,398 cycles
  Target:  1,363 cycles
  Gap:     35 cycles

  Overhead above floor: 101 cycles
  Need to reduce to: 66 cycles overhead
""")

# Potential approaches to reduce overhead:
print("""
=== Approaches to Close 35-Cycle Gap ===

1. INCREASE GROUP OVERLAP (reduce drain)
   - Currently 4 groups per tile-pair, interleaved A_G0, B_G0, A_G1, B_G1
   - Drain = ~72 cycles (last group finishing alone)
   - What if we had 3 groups per bank (6 total) with smaller group size?
   - GROUP_SIZE=3 with 6 groups? But 8 desks / 3 = 2.67 (not integer)
   - GROUP_SIZE=2 with 8 groups? Very small groups...

2. OVERLAP TILE PAIRS (pipeline across pairs)
   - Emit pair 1's loads before pair 0's stores complete
   - Need separate scratch for early loads (prefetch buffer)

3. REDUCE VALU OPS
   - Each AND(val, v_one) extracts low bit. Could we use the FMA result
     directly? E.g. if val is guaranteed to be computed just before...
   - Actually, the AND is needed because we need bit0/bit1 for the vselect
     cascades in the fused blocks. These bits drive the tree level selection.

4. 3-GROUP ASYMMETRIC (NEW IDEA)
   - Bank A: 5 desks in group 0, 3 desks in group 1 = 8 desks
   - Bank B: 5 desks in group 0, 3 desks in group 1 = 8 desks
   - Total: 4 groups but asymmetric sizes
   - Group of 5: throughput 10 cycles for hash (5*2) vs 9 latency = no bubbles!
   - Group of 3: throughput 6 cycles < 9 latency = 3 cycle bubble, but short group
   - The G0(5) groups finish later, G1(3) groups finish earlier
   - This creates a longer overlap between groups

5. FILL DRAIN WITH STORES (move stores earlier)
   - Currently stores happen AFTER all compute
   - What if we store finished desks immediately after their last round?
   - As soon as desk 0's round 15 is done, store it, freeing VALU cycles
   - This doesn't add VALU work but may change scheduling to overlap better
""")
