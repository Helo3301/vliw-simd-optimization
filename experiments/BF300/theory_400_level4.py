"""
Theory 400: Level-4 tree preloading with R0-R4 and R11-R15 fusion.

Preload 31 tree nodes (levels 0-4). Use 15-vselect cascade for level 4.
Fuse R0-R4 (5 rounds without gathers) and R11-R14+R15 (5 rounds).

R15 is the final round: no branch needed. But we still need the XOR+hash.
For R11-R15: fuse R11-R14 (same as before) + R15 (level 4 gather + XOR + hash, no branch).
Wait, R15 currently goes to level 15 of the tree (gather address), not level 4.
Level 4 is only for rounds that go to depth 4.

Actually, the tree has 10 levels (height=10). Each round descends 1 level.
- R0: level 0 (root, idx=0)
- R1: level 1 (2 nodes)
- R2: level 2 (4 nodes)
- R3: level 3 (8 nodes)  
- R4: level 4 (16 nodes) <-- CAN preload
- R5-R9: levels 5-9 (32-512 nodes, too many to preload)
- R10: level 10 = wraps to 0 (if idx >= n_nodes, idx = 0)
  Actually R10: idx could be >= n_nodes, so it wraps. After wrap, idx=0 again.
- R11: level 0 again (same as R0)
- R12: level 1 
- R13: level 2
- R14: level 3
- R15: level 4 <-- CAN preload

So YES, R15 goes to level 4! If I preload 31 nodes, I can use vselect for R4 AND R15.

For R15: no branch needed (final round), just XOR + hash.
For R4: need vselect + XOR + hash + deferred addr computation.

This means:
- R0-R4: fused with 5-round deferred addr computation (5 bits)
- R5-R9: gather rounds (5 rounds instead of 6!)
- R10: special (no branch)
- R11-R15: fused with 5 rounds (R15 = no branch, just XOR + hash)

We save 2 gather rounds (R4 and R15 become fused). That's huge!
- Loads saved: 2 * 32 * 4 * 2 = 512 loads
- VALU saved: 2 * (3 branch) * 4 * 4 * 2 = 192 VALU from branches
  Plus we save 2 * 4 gather rounds per group = 8 XOR+hash rounds?
  No, we still do XOR+hash in the fused rounds. We just avoid the gather loads.

Actually, the savings come from:
1. No loads needed for R4 and R15 (they use preloaded vselect)
2. No addr computation for R4 and R15 (computed in deferred block)
3. R4 branch replaces 3 VALU (AND+FMA+ADD) with bit extraction (1 AND) and deferred addr

Wait, let me reconsider. In Theory 222:
- R0-R3 fused: 4 rounds, no gathers, uses vselect cascade, deferred 4-bit addr
- R4-R9: 6 gather rounds with addr-tracking branch
- R10: gather, no branch
- R11-R14 fused: 4 rounds, no gathers, same as R0-R3
- R15: gather, no branch

In Theory 400:
- R0-R4 fused: 5 rounds, no gathers, uses 15-vselect for R4, deferred 5-bit addr
- R5-R9: 5 gather rounds (SAVE 1 gather round per block!)
- R10: gather, no branch
- R11-R15 fused: 5 rounds, R15 uses vselect, no branch after R15

Savings per tile per group:
- 2 fewer gather rounds: saves 2 * 32 = 64 loads per group
- 2 fewer branch ops: saves 2 * 3 = 6 VALU per desk per block = 6*4*2 = 48 VALU
- But deferred addr goes from 5 to 6 ops: costs 1*4*2 = 8 more VALU
- Net VALU saved per tile: (48-8)*4 groups = 160 VALU per tile, 320 total
- Net loads saved: 64*4*2 = 512 loads

But flow cost:
- R4 level 4: 15 vselects/desk * 4 desks * 4 groups * 2 tiles = 480 flow
- R15 level 4: 15 vselects/desk * 4 desks * 4 groups * 2 tiles = 480 flow
- Total new flow: 480 + 480 = 960 more flow
- Current flow: 704
- New flow: 1664
- That's 1664 flow ops = 1664 cycles just for flow!

This is way too many flow ops. 1664 > current 1400 cycles total.

Hmm. Unless... I can reduce the vselect count for level 4.

For an 8-node selection (level 3), we need 7 vselects.
For a 16-node selection (level 4), we need 15 vselects.
The structure is: 2 subtrees of 8 nodes (7 vselects each) + 1 final select = 15.

Wait, but we already have bit0, bit1, bit2 from R0-R2. For R4 we also have bit3. 
The vselect cascade for R3 uses 7 vselects with bit0, bit1, bit2.
For R4 (level 4), we need to select from 16 nodes using bit0, bit1, bit2, bit3.

7 vselects (using bit0, bit1) give us 1 of 4 subtrees.
Then for each subtree, 3 more vselects (using bit2, bit3) give 1 of 4 nodes.
Total: 7 + 3 + 1 = 11? No...

Let me think step by step:
- 16 nodes at level 4: tree[15..30]
- Need to select: node = tree[15 + 8*bit0 + 4*bit1 + 2*bit2 + bit3 + bit4_not_needed]
  Wait, R4 has 5 rounds of bits: bit0 (from R0), bit1 (R1), bit2 (R2), bit3 (R3), bit4 (R4).
  After R4 hash, we extract bit4.
  Level 4 has 16 nodes, indexed by 4 bits: bit0, bit1, bit2, bit3.
  
  Hmm, level 4 node index = 2^4 - 1 + 8*bit0 + 4*bit1 + 2*bit2 + bit3 = 15 + 8*bit0 + 4*bit1 + 2*bit2 + bit3.
  That's 4 bits selecting from 16 nodes. We need bit0..bit3, NOT bit4.
  bit4 is the hash output of R4, used for the NEXT branch (R5).

So the vselect cascade for R4 is:
- Using bit0, bit1, bit2, bit3 (known from R0, R1, R2, R3)
- Select 1 of 16 = 15 vselects

But bit3 is the hash output of R3. In Theory 222, bit3 = R3's `val & 1`, stored in `tmp1`.
But after the R3 hash, the deferred addr computation overwrites tmp1!

In Theory 222's R3:
1. Extract bit2 into idx
2. 7-vselect cascade for R3 node
3. XOR with selected node
4. Hash
5. Extract bit3 = val & 1 -> tmp1
6. Deferred addr: FMA(addr, bit0, 2, bit1), FMA(addr, addr, 2, idx), FMA(addr, addr, 2, tmp1), ADD(addr, addr, fp+15)

After step 6, bit3 is lost (tmp1 was used). But for the Theory 400 R4 cascade, I need bit3!

So I need to save bit3 somewhere. I have bit0, bit1 registers. I used idx for bit2. 
Where to put bit3? I could use addr (before it's needed for R5).

Let me just save bit3 in addr before the deferred computation... but the deferred computation writes to addr. Hmm.

Actually, the approach changes: in Theory 400, R4 is FUSED, not gathered. So I don't need to compute the gather address after R3. Instead, after R3 hash I extract bit3, do the R4 vselect cascade, then after R4 hash I extract bit4 and compute the deferred addr from bits 0-4.

The deferred addr computation becomes:
addr = fp + 31 + 16*bit0 + 8*bit1 + 4*bit2 + 2*bit3 + bit4

Using FMA chain:
s = FMA(bit0, 2, bit1) = 2*bit0 + bit1
s = FMA(s, 2, bit2)    = 4*bit0 + 2*bit1 + bit2  
s = FMA(s, 2, bit3)    = 8*bit0 + 4*bit1 + 2*bit2 + bit3
s = FMA(s, 2, bit4)    = 16*bit0 + 8*bit1 + 4*bit2 + 2*bit3 + bit4
addr = s + fp + 31

That's 5 FMA + 1 ADD + 1 AND = 7 VALU per desk.
Vs Theory 222's 4 FMA + 1 ADD + 1 AND = 6 VALU per desk.
So 1 more VALU per desk.

But I also need places to store bit3 and bit4. Currently I have:
- bit0: dedicated register
- bit1: dedicated register  
- bit2: stored in idx
- bit3: ??? Need another register!

I could use node_val (it gets overwritten by the vselect cascade anyway).
Or I could use addr (not needed until the deferred computation at the end).

Let me use addr for bit3.

For the 15-vselect cascade in R4, I need bit0, bit1, bit2 (=idx), bit3 (=addr).
This uses 15 flow ops per desk.

Let me calculate the total flow ops with this approach:
- R0: 0 flow
- R1: 1 vselect/desk (2-way)
- R2: 3 vselects/desk (4-way)
- R3: 7 vselects/desk (8-way)
- R4: 15 vselects/desk (16-way)
Total per fused block per desk: 1 + 3 + 7 + 15 = 26 vselects
Per group: 26 * 4 = 104
Per tile: 104 * 4 = 416
Two fused blocks (R0-R4 + R11-R15): 416 * 2 = 832 per tile
Two tiles: 832 * 2 = 1664 vselects + 1 pause = 1665 flow ops

Hmm, that's what I calculated before. 1665 cycles minimum just for flow.
That's way more than the current 1400. This approach won't work.

UNLESS: I can somehow reduce the vselect count for the 16-node selection.

Idea: What if I do the 16-node selection hierarchically reusing intermediate results?
- First level (bit3): Split 16 into 2 groups of 8. Select sub-group.
  Use 7 vselects with bit0,bit1,bit2 (same as R3!) but on different tree nodes.
  Wait, the R3 cascade already selects from tree[7..14]. The R4 cascade would select
  from tree[15..30]. The STRUCTURE is different, not the bit pattern.
  
Actually, the 15-vselect cascade for 16 nodes is structured as:
  Layer 1 (bit3): 8 vselects pairing nodes (0,1),(2,3),... -> 8 results
  Layer 2 (bit2): 4 vselects pairing layer-1 results -> 4 results  
  Layer 3 (bit1): 2 vselects -> 2 results
  Layer 4 (bit0): 1 vselect -> final result
  Total: 8+4+2+1 = 15 vselects.

Can I use the tree structure differently? The binary tree has a natural 2-way split.
At level 4, node index = 15 + path (4-bit number).
  - bit0 splits: left subtree (tree[15..22]) vs right subtree (tree[23..30])
  - bit1 splits each 4-node group
  - etc.

Currently R3's 7-vselect cascade selects from 8 nodes using 3 bits.
R4 would need to select from 16 nodes using 4 bits.

Alternative: Can I reuse R3's result? R3's cascade already narrowed down to 1 of 8 subtrees based on bit0,bit1,bit2. For R4, I need to go one level deeper using bit3.

So after the R3 cascade gives me the node at level 3, for R4 I just need:
  left_child = tree[2*R3_idx + 1]
  right_child = tree[2*R3_idx + 2]
  R4_node = vselect(bit3, right_child, left_child)

But I don't know R3_idx per-lane! I know the NODE VALUE (from the vselect), not the index.

Unless I precompute pairs of level-4 children for each level-3 parent...

Actually! Here's the insight: I can build the R4 cascade from the R3 cascade!
R3 cascade: using bit0, bit1, bit2, select from tree[7..14] -> 8 possible results
For each of those 8 nodes at level 3, there are 2 children at level 4.

So I can do:
  For the left child (bit3=0): select from tree[15,17,19,21,23,25,27,29] using bit0,bit1,bit2
  For the right child (bit3=1): select from tree[16,18,20,22,24,26,28,30] using bit0,bit1,bit2
  Final: vselect(bit3, right_result, left_result)

Each of the two 8-node selections takes 7 vselects. Plus 1 final = 15 total. Same count.

BUT: if I've already computed the R3 cascade intermediate values, can I REUSE them?

The R3 cascade for tree[7..14]:
  bit2: (7,8),(9,10),(11,12),(13,14) -> 4 pairs -> sel7_8, sel9_10, sel11_12, sel13_14
  bit1: (sel7_8, sel9_10), (sel11_12, sel13_14) -> sel7_10, sel11_14
  bit0: (sel7_10, sel11_14) -> final

For R4 left children tree[15,17,19,21,23,25,27,29]:
  bit2: (15,17),(19,21),(23,25),(27,29) -> different from R3!
  
Hmm, the bit patterns are different because tree[15..30] has a different layout than tree[7..14].

Let me think differently. At level 4, the node for bit pattern b0b1b2b3 is:
  tree[15 + 8*b0 + 4*b1 + 2*b2 + b3]

I can structure the cascade as:
  Inner: for each b3 value (0,1), do a 3-bit cascade using b0,b1,b2 on 8 nodes
  Outer: select between the two results using b3

For b3=0: nodes are tree[15,19,23,27,17,21,25,29] = tree[15+4*b1+2*b2+8*b0] -> not consecutive
Actually: tree[15 + 8*b0 + 4*b1 + 2*b2 + 0] for b3=0 -> tree[15+8b0+4b1+2b2]
For b3=1: tree[15 + 8*b0 + 4*b1 + 2*b2 + 1] = tree[16+8b0+4b1+2b2]

So for b3=0: tree[15], tree[17], tree[19], tree[21], tree[23], tree[25], tree[27], tree[29] = tree[15,17,19,21,23,25,27,29]
For b3=1: tree[16], tree[18], tree[20], tree[22], tree[24], tree[26], tree[28], tree[30]

Each group has 8 nodes to select from using 3 bits (b0,b1,b2). That's 7 vselects each.
Plus 1 final b3 select. Total: 7+7+1 = 15.

Can I share work between the two groups? They use the SAME bits (b0,b1,b2) but different data.
The intermediate results are different. No sharing possible.

OK so 15 vselects per desk per R4 cascade is unavoidable. Let me think of another way.

What if I preload both children of each level-3 node? That's 16 pairs = 32 vectors.
No, that's way too much scratch.

I think the level-4 fusion is not viable due to flow bottleneck.
Let me try a different approach entirely.

WHAT IF: I only fuse R0-R4 on the SECOND block (R11-R15)?
R15 doesn't need a branch! So R15's fused version just needs:
  level-4 vselect cascade (15 vselects/desk)
  XOR
  hash
  NO branch, NO addr computation!

For R0-R3 fused (as current): 11 vselects/desk
For R11-R15 fused (new): 11 + 15 = 26 vselects/desk

Total flow per desk per tile: 11*2 + 15 = 37 (was 11*2=22)
Per group: 37*4 = 148 (was 88)
Per tile: 148*4 = 592 (was 352)
Two tiles: 1184 (was 704)
Plus 1 pause = 1185.

VALU savings from fusing R15:
- Save 0 VALU branch (R15 already had no branch)
- Save 0 hash (still need hash)
- Save 0 XOR (still need XOR)
- The only savings are loads! 32 loads per group * 4 groups * 2 tiles = 256 loads saved.

So we go from 2193 loads to 1937 loads. Load floor: 969 (was 1097).
VALU stays at 7779. VALU floor: 1297.
Flow: 1185.

Max(1297, 1185, 969) = 1297. So VALU-bound still.
Overhead would be 1297 + overhead. If overhead is ~100, we'd be at ~1397.
But with fewer loads, there should be LESS load-bound overhead.

Currently: 143 load-bound cycles. Saving 256 loads eliminates ~128 load-bound cycles.
But those cycles still need to exist for flow ops. We go from 706 flow to 1185 flow.
The flow-bound cycles would increase.

Let me just estimate: max(1297, 1185) + some_overhead.
If overhead drops from 103 to ~70 (due to better load-flow-VALU balance), total ~1367.
That's close to our target of <1363!

Let me try this. I'll implement R11-R15 fused (with level-4 vselect for R15).
"""
print("Theory 400 analysis complete. The R11-R15 fusion with level-4 preloading looks promising.")
print("Expected: ~1367 cycles (flow=1185, VALU=1297 floor)")
print("Need: <1363")
print("Let me implement it.")
