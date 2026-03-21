"""
C2-2: Selective Broadcast Based on Idx Uniformity

C2-1 showed that 8-way arithmetic selection (using VALU + vselect) is WORSE than
gather loads because VALU is already the bottleneck.

Key insight: The problem is LOAD bound (79%), not VALU bound. Adding more VALU ops
to save loads doesn't help because:
1. At 2 loads/cycle, 8 loads take 4 cycles
2. At 6 VALU/cycle, 8+ VALU ops take more than 1 cycle
3. The scheduler can overlap VALU with loads, so VALU ops are "free" up to a point

New strategy: Instead of replacing loads with VALU, try to reduce TOTAL load count
by detecting when all lanes in a desk have the same idx.

For round 3: All 8 lanes start with same history (same bit0, bit1 from rounds 0-2)
Wait... no, different lanes have different initial values, so they hash differently.

Actually, within a desk:
- Round 0: All lanes hit tree[0] (same)
- After hash: Different values (hash is per-lane)
- Round 1: Different bits -> idx in {1,2}
- Round 2: idx in {3,4,5,6}
- Round 3: idx in {7..14}

So even at round 3, different lanes likely have different indices.

What if we batch gathers at the GROUP level instead of DESK level?
- Group = 4 desks x 8 lanes = 32 elements
- For round 3, we have 8 possible indices (7-14)
- Could we count unique indices and do deduplicated loads?

This requires:
1. Collect all 32 indices from a group
2. Find unique indices
3. Load each unique index once
4. Scatter to the correct lanes

But the architecture doesn't support scatter operations easily.

Alternative: What if we trade more scratch space for batched loads?

Actually let's go back to basics. The current code does:
- Round 3-9, 14: emit_gather_round_interleaved
- Each gather round: 4 desks x 8 lanes = 32 loads per group, 8 groups = 256 loads/tile
- 10 gather rounds x 2 tiles = 5120 gather operations but at 8 lanes/desk
- Wait, let me recount...

Per group (4 desks):
- 4 desks x 8 lanes = 32 loads
- emit_gather_round_interleaved loops over group_desks (4 desks), each does 8 loads

Per tile (4 groups):
- 4 groups x 32 loads = 128 loads

Per iteration (2 tiles):
- 2 tiles x 128 loads = 256 loads per gather round

Total gather rounds: 7 (rounds 3-9) + 1 (round 10) + 1 (round 14) + 1 (round 15) = 10

Wait, rounds 3-9 is 7 rounds, 10, 14, 15 that's 10 gather rounds per tile iteration.
256 loads per round x 10 rounds = 2560 total gather loads.
At 2 loads/cycle = 1280 cycles minimum just for gathers.

Current B4-2 = 1558 cycles. Theoretical minimum = 1280.
Gap = 278 cycles.

To get below 1400, we need to save at least 158 cycles.
If we could eliminate even 1 gather round per element, that's 256 loads = 128 cycles saved!

Idea: What if we fused round 3 into the 0-1-2 fusion similar to how B4-2 works?

Round 2 ends with idx in {3,4,5,6}, which are consecutive.
Round 3's branch produces idx in {7,8,9,10,11,12,13,14}.

After round 2:
- idx = 3 + 2*bit0 + bit1 where bit0, bit1 are saved
- Round 3 XORs with tree[idx] where idx in {3,4,5,6}

We already do the 4-way select for round 2. The issue is after XOR and hash,
we get a new value and need to branch to compute idx for round 3.

The branch formula: idx_new = 2*idx + 1 + bit
For idx in {3,4,5,6}:
- idx=3: new_idx = 7 + bit (7 or 8)
- idx=4: new_idx = 9 + bit (9 or 10)
- idx=5: new_idx = 11 + bit (11 or 12)
- idx=6: new_idx = 13 + bit (13 or 14)

So after round 2's branch, idx is in {7..14} = level 3.
Round 3 must XOR with tree[idx] where idx in {7..14}.

If we've preloaded tree[7..14], we can do 8-way selection.
But C2-1 showed this is WORSE because of VALU pressure.

New idea: What if we pipeline the preloading?
While one group is doing its gather round, preload tree nodes for the next group.

Actually... the tree nodes are the SAME for all groups. The issue is that different
elements (lanes) within a group need different tree nodes.

Let me think about this differently. The real question is:
Can we reduce the NUMBER of gather operations while maintaining correctness?

What if we restructured the data layout?
Instead of indices being scattered, what if we sorted elements by their index?

No, that would require dynamic sorting which is expensive.

What about speculative execution?
For level 3 (8 nodes), we could:
1. Load ALL 8 nodes into vector registers at the start
2. Use arithmetic selection (expensive, as C2-1 showed)

What about hierarchical broadcast?
For round 3 with idx in {7..14}:
- Preload tree[7..14] as scalars
- Broadcast each to a vector register
- Use vselect chain

But this is exactly what C2-1 does, and it's worse.

The fundamental problem is that VALU is saturated. Adding VALU ops to save loads
doesn't help because the loads can run in parallel with VALU anyway.

New approach: Focus on reducing TOTAL operations, not just loads.

What if we could prove that certain paths through the tree are impossible,
and skip those computations?

Or: What if we increased parallelism by processing MORE elements at once?
Currently we have 4 desks per group. What if we had 2 desks per group but
processed more groups simultaneously?

Let me try a different optimization: reducing the hash computation overhead.

Actually, let me re-examine the baseline. Maybe there's room to improve
the scheduling without adding new ops.
"""

import random
import argparse
import sys
import os
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


def _vec_range(base: int, length: int = VLEN) -> range:
    return range(base, base + length)


def _slot_rw(engine: str, slot: tuple) -> tuple[list[int], list[int]]:
    reads: list[int] = []
    writes: list[int] = []

    if engine == "alu":
        _op, dest, a1, a2 = slot
        reads = [a1, a2]
        writes = [dest]
    elif engine == "valu":
        if slot[0] == "vbroadcast":
            dest, src = slot[1], slot[2]
            reads = [src]
            writes = list(_vec_range(dest))
        elif slot[0] == "multiply_add":
            dest, a, b, c = slot[1], slot[2], slot[3], slot[4]
            reads = list(_vec_range(a)) + list(_vec_range(b)) + list(_vec_range(c))
            writes = list(_vec_range(dest))
        else:
            _op, dest, a1, a2 = slot
            reads = list(_vec_range(a1)) + list(_vec_range(a2))
            writes = list(_vec_range(dest))
    elif engine == "load":
        if slot[0] == "load":
            dest, addr = slot[1], slot[2]
            reads = [addr]
            writes = [dest]
        elif slot[0] == "vload":
            dest, addr = slot[1], slot[2]
            reads = [addr]
            writes = list(_vec_range(dest))
        elif slot[0] == "const":
            dest, _val = slot[1], slot[2]
            writes = [dest]
        elif slot[0] == "load_offset":
            dest, addr, _lane = slot[1], slot[2], slot[3]
            reads = [addr]
            writes = [dest]
        else:
            raise NotImplementedError(f"Unknown load op {slot}")
    elif engine == "store":
        if slot[0] == "store":
            addr, src = slot[1], slot[2]
            reads = [addr, src]
        elif slot[0] == "vstore":
            addr, src = slot[1], slot[2]
            reads = [addr] + list(_vec_range(src))
        else:
            raise NotImplementedError(f"Unknown store op {slot}")
    elif engine == "flow":
        if slot[0] == "select":
            dest, cond, a, b = slot[1], slot[2], slot[3], slot[4]
            reads = [cond, a, b]
            writes = [dest]
        elif slot[0] == "add_imm":
            dest, a, _imm = slot[1], slot[2], slot[3]
            reads = [a]
            writes = [dest]
        elif slot[0] == "vselect":
            dest, cond, a, b = slot[1], slot[2], slot[3], slot[4]
            reads = list(_vec_range(cond)) + list(_vec_range(a)) + list(_vec_range(b))
            writes = list(_vec_range(dest))
        elif slot[0] in ("halt", "pause"):
            pass
        elif slot[0] == "trace_write":
            pass
        elif slot[0] in ("jump", "jump_indirect"):
            pass
        elif slot[0] in ("cond_jump", "cond_jump_rel"):
            pass
        elif slot[0] == "coreid":
            pass
        else:
            raise NotImplementedError(f"Unknown flow op {slot}")

    return reads, writes


def _schedule_slots(slots: list[tuple[str, tuple]]) -> list[dict[str, list[tuple]]]:
    cycles: list[dict[str, list[tuple]]] = []
    usage: list[dict[str, int]] = []
    ready_time: dict[int, int] = defaultdict(int)
    last_write: dict[int, int] = defaultdict(lambda: -1)
    last_read: dict[int, int] = defaultdict(lambda: -1)

    def ensure_cycle(cycle: int) -> None:
        while len(cycles) <= cycle:
            cycles.append({})
            usage.append(defaultdict(int))

    def find_cycle(engine: str, earliest: int) -> int:
        cycle = earliest
        limit = SLOT_LIMITS[engine]
        while True:
            ensure_cycle(cycle)
            if usage[cycle][engine] < limit:
                return cycle
            cycle += 1

    for engine, slot in slots:
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

        for addr in reads:
            if last_read[addr] < cycle:
                last_read[addr] = cycle
        for addr in writes:
            last_write[addr] = cycle
            ready_time[addr] = cycle + 1

    return [c for c in cycles if c]


class KernelBuilderC2_2:
    """
    C2-2: Analysis experiment - try fusing round 3 into 0-1-2 using extended preloading.

    Key insight: Round 3 accesses tree[7..14] which is only 8 nodes.
    If we can do 8-way selection efficiently, we eliminate one gather round.

    The challenge: 8-way selection needs ~7 vselects or equivalent VALU work.
    C2-1 showed this is worse because VALU is saturated.

    New approach: Try using multiply_add cascades instead of vselect.

    For 8 values v[0..7] and 3 bits b2,b1,b0:
    - Level 0: p01 = v[0] + b0*(v[1]-v[0]), p23 = v[2] + b0*(v[3]-v[2]), ...
    - Level 1: p0123 = p01 + b1*(p23-p01), p4567 = p45 + b1*(p67-p45)
    - Level 2: result = p0123 + b2*(p4567-p0123)

    This needs:
    - 4 differences precomputed (or computed inline)
    - 4 multiply_adds for level 0
    - 2 differences
    - 2 multiply_adds for level 1
    - 1 difference
    - 1 multiply_add for level 2

    Total: 7 differences + 7 multiply_adds = 14 VALU ops

    Compared to:
    - 32 loads (8 per desk x 4 desks) = 16 cycles of load (at 2/cycle)
    - 4 addr computations

    14 VALU ops at 6/cycle = ~3 cycles if perfectly scheduled.
    But we're already VALU-saturated, so these add to existing pressure.

    This doesn't seem promising. Let me try a different angle.

    What if we prefetch tree nodes during the hash computation?
    The hash takes many VALU ops. During that time, we could be loading
    the next round's tree nodes.

    Actually, the current code already does this implicitly via the scheduler.

    Let me try: What if we process desks in a different order to maximize
    load overlap?
    """

    def __init__(self):
        self.slots: list[tuple[str, tuple]] = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self.vconst_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def emit(self, engine: str, slot: tuple):
        self.slots.append((engine, slot))

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE
        return addr

    def alloc_vec(self, name=None):
        return self.alloc_scratch(name, VLEN)

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name or f"c_{val}")
            self.emit("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def scratch_vconst(self, val, name=None):
        if val not in self.vconst_map:
            scalar = self.scratch_const(val)
            addr = self.alloc_vec(name or f"v_{val}")
            self.emit("valu", ("vbroadcast", addr, scalar))
            self.vconst_map[val] = addr
        return self.vconst_map[val]

    def build_kernel(self, forest_height: int, n_nodes: int, batch_size: int, rounds: int):
        tmp_scalar = self.alloc_scratch("tmp_scalar")
        tmp_addr = self.alloc_scratch("tmp_addr")

        fast_init_vars = [("n_nodes", 1), ("forest_values_p", 4), ("inp_indices_p", 5), ("inp_values_p", 6)]
        for var_name, _ in fast_init_vars:
            self.alloc_scratch(var_name)
        for var_name, idx in fast_init_vars:
            self.emit("load", ("const", tmp_scalar, idx))
            self.emit("load", ("load", self.scratch[var_name], tmp_scalar))

        v_zero = self.scratch_vconst(0, "v_zero")
        v_one = self.scratch_vconst(1, "v_one")
        v_two = self.scratch_vconst(2, "v_two")
        v_three = self.scratch_vconst(3, "v_three")
        v_n_nodes = self.alloc_vec("v_n_nodes")
        self.emit("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]))

        v_forest_p = self.alloc_vec("v_forest_p")
        self.emit("valu", ("vbroadcast", v_forest_p, self.scratch["forest_values_p"]))

        FMA_MULTIPLIERS = {0: 4097, 2: 33, 4: 9}
        v_hash_consts = []
        v_hash_shifts = []
        v_fma_mult = {}

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            v_const = self.scratch_vconst(val1, f"v_hash_const_{hi}")
            v_hash_consts.append(v_const)
            if hi in FMA_MULTIPLIERS:
                v_fma_mult[hi] = self.scratch_vconst(FMA_MULTIPLIERS[hi], f"v_fma_mult_{hi}")
                v_hash_shifts.append(None)
            else:
                v_shift = self.scratch_vconst(val3, f"v_hash_shift_{hi}")
                v_hash_shifts.append(v_shift)

        NUM_PRELOADED = 7
        v_tree = []
        for i in range(NUM_PRELOADED):
            v_node = self.alloc_vec(f"v_tree_{i}")
            v_tree.append(v_node)
            self.emit("alu", ("+", tmp_addr, self.scratch["forest_values_p"], self.scratch_const(i)))
            self.emit("load", ("load", tmp_scalar, tmp_addr))
            self.emit("valu", ("vbroadcast", v_node, tmp_scalar))

        v_diff_1_2 = self.alloc_vec("v_diff_1_2")
        v_diff_3_4 = self.alloc_vec("v_diff_3_4")
        v_diff_5_6 = self.alloc_vec("v_diff_5_6")
        self.emit("valu", ("-", v_diff_1_2, v_tree[2], v_tree[1]))
        self.emit("valu", ("-", v_diff_3_4, v_tree[4], v_tree[3]))
        self.emit("valu", ("-", v_diff_5_6, v_tree[6], v_tree[5]))

        NUM_DESKS = 16
        desks = []
        for d in range(NUM_DESKS):
            desk = {
                'idx': self.alloc_vec(f"v_idx_{d}"),
                'val': self.alloc_vec(f"v_val_{d}"),
                'node_val': self.alloc_vec(f"v_node_{d}"),
                'addr': self.alloc_vec(f"v_addr_{d}"),
                'tmp1': self.alloc_vec(f"v_tmp1_{d}"),
                'tmp2': self.alloc_vec(f"v_tmp2_{d}"),
                'bit0': self.alloc_vec(f"v_bit0_{d}"),
            }
            desks.append(desk)

        offset_regs = [self.alloc_scratch(f"off_{d}") for d in range(NUM_DESKS)]
        addr_tmp = [self.alloc_scratch(f"addr_tmp_{i}") for i in range(32)]

        print(f"Scratch usage: {self.scratch_ptr} / {SCRATCH_SIZE}")

        self.emit("flow", ("pause",))

        def emit_hash_interleaved(group_desks):
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("multiply_add", desk['val'], desk['val'], v_fma_mult[0], v_hash_consts[0]))
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("^", desk['tmp1'], desk['val'], v_hash_consts[1]))
                self.emit("valu", (">>", desk['tmp2'], desk['val'], v_hash_shifts[1]))
                self.emit("valu", ("^", desk['val'], desk['tmp1'], desk['tmp2']))
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("multiply_add", desk['val'], desk['val'], v_fma_mult[2], v_hash_consts[2]))
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("+", desk['tmp1'], desk['val'], v_hash_consts[3]))
                self.emit("valu", ("<<", desk['tmp2'], desk['val'], v_hash_shifts[3]))
                self.emit("valu", ("^", desk['val'], desk['tmp1'], desk['tmp2']))
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("multiply_add", desk['val'], desk['val'], v_fma_mult[4], v_hash_consts[4]))
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("^", desk['tmp1'], desk['val'], v_hash_consts[5]))
                self.emit("valu", (">>", desk['tmp2'], desk['val'], v_hash_shifts[5]))
                self.emit("valu", ("^", desk['val'], desk['tmp1'], desk['tmp2']))

        def emit_branch(desk_idx):
            d = desks[desk_idx]
            self.emit("valu", ("&", d['tmp1'], d['val'], v_one))
            self.emit("valu", ("multiply_add", d['idx'], d['idx'], v_two, v_one))
            self.emit("valu", ("+", d['idx'], d['idx'], d['tmp1']))

        def emit_bounds(desk_idx):
            d = desks[desk_idx]
            self.emit("valu", ("<", d['tmp1'], d['idx'], v_n_nodes))
            self.emit("valu", ("*", d['idx'], d['idx'], d['tmp1']))

        def emit_xor_with_node(desk_idx, node_vec):
            d = desks[desk_idx]
            self.emit("valu", ("^", d['val'], d['val'], node_vec))

        def emit_rounds_0_1_2_fused(group_desks):
            for d in group_desks:
                emit_xor_with_node(d, v_tree[0])
            emit_hash_interleaved(group_desks)
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("&", desk['bit0'], desk['val'], v_one))
                self.emit("valu", ("+", desk['idx'], v_one, desk['bit0']))

            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("multiply_add", desk['node_val'], desk['bit0'], v_diff_1_2, v_tree[1]))
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("^", desk['val'], desk['val'], desk['node_val']))
            emit_hash_interleaved(group_desks)
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("&", desk['tmp1'], desk['val'], v_one))
                self.emit("valu", ("multiply_add", desk['idx'], desk['bit0'], v_two, v_three))
                self.emit("valu", ("+", desk['idx'], desk['idx'], desk['tmp1']))

            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("multiply_add", desk['node_val'], desk['tmp1'], v_diff_3_4, v_tree[3]))
                self.emit("valu", ("multiply_add", desk['tmp2'], desk['tmp1'], v_diff_5_6, v_tree[5]))
                self.emit("flow", ("vselect", desk['node_val'], desk['bit0'], desk['tmp2'], desk['node_val']))
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("^", desk['val'], desk['val'], desk['node_val']))
            emit_hash_interleaved(group_desks)
            for d in group_desks:
                emit_branch(d)

        def emit_rounds_11_12_13_fused(group_desks):
            for d in group_desks:
                emit_xor_with_node(d, v_tree[0])
            emit_hash_interleaved(group_desks)
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("&", desk['bit0'], desk['val'], v_one))
                self.emit("valu", ("+", desk['idx'], v_one, desk['bit0']))

            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("multiply_add", desk['node_val'], desk['bit0'], v_diff_1_2, v_tree[1]))
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("^", desk['val'], desk['val'], desk['node_val']))
            emit_hash_interleaved(group_desks)
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("&", desk['tmp1'], desk['val'], v_one))
                self.emit("valu", ("multiply_add", desk['idx'], desk['bit0'], v_two, v_three))
                self.emit("valu", ("+", desk['idx'], desk['idx'], desk['tmp1']))

            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("multiply_add", desk['node_val'], desk['tmp1'], v_diff_3_4, v_tree[3]))
                self.emit("valu", ("multiply_add", desk['tmp2'], desk['tmp1'], v_diff_5_6, v_tree[5]))
                self.emit("flow", ("vselect", desk['node_val'], desk['bit0'], desk['tmp2'], desk['node_val']))
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("^", desk['val'], desk['val'], desk['node_val']))
            emit_hash_interleaved(group_desks)
            for d in group_desks:
                emit_branch(d)

        # New approach: emit gathers with better interleaving
        # Instead of doing all addr computations, then all loads
        # interleave: addr_d0, load_d0_lane0-3, addr_d1, load_d1_lane0-3, ...
        def emit_gather_round_pipelined(group_desks):
            """Pipelined gather: overlap addr computation with loads"""
            # Emit all addr computations first (can execute in parallel)
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("+", desk['addr'], v_forest_p, desk['idx']))

            # Interleave loads across desks
            # This might help the scheduler overlap loads better
            for lane in range(VLEN):
                for d in group_desks:
                    desk = desks[d]
                    self.emit("load", ("load", desk['node_val'] + lane, desk['addr'] + lane))

            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("^", desk['val'], desk['val'], desk['node_val']))
            emit_hash_interleaved(group_desks)
            for d in group_desks:
                emit_branch(d)

        def emit_gather_round_interleaved(group_desks):
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("+", desk['addr'], v_forest_p, desk['idx']))
            for d in group_desks:
                desk = desks[d]
                for lane in range(VLEN):
                    self.emit("load", ("load", desk['node_val'] + lane, desk['addr'] + lane))
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("^", desk['val'], desk['val'], desk['node_val']))
            emit_hash_interleaved(group_desks)
            for d in group_desks:
                emit_branch(d)

        def emit_round_10_interleaved(group_desks):
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("+", desk['addr'], v_forest_p, desk['idx']))
            for d in group_desks:
                desk = desks[d]
                for lane in range(VLEN):
                    self.emit("load", ("load", desk['node_val'] + lane, desk['addr'] + lane))
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("^", desk['val'], desk['val'], desk['node_val']))
            emit_hash_interleaved(group_desks)
            for d in group_desks:
                emit_branch(d)
            for d in group_desks:
                emit_bounds(d)

        def emit_round_15_final_interleaved(group_desks):
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("+", desk['addr'], v_forest_p, desk['idx']))
            for d in group_desks:
                desk = desks[d]
                for lane in range(VLEN):
                    self.emit("load", ("load", desk['node_val'] + lane, desk['addr'] + lane))
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("^", desk['val'], desk['val'], desk['node_val']))
            emit_hash_interleaved(group_desks)

        def emit_tile_interleaved(tile_idx):
            tile_offset = tile_idx * NUM_DESKS * VLEN

            for d in range(NUM_DESKS):
                self.emit("load", ("const", offset_regs[d], tile_offset + d * VLEN))

            for d in range(NUM_DESKS):
                self.emit("alu", ("+", addr_tmp[d*2], self.scratch["inp_indices_p"], offset_regs[d]))
                self.emit("alu", ("+", addr_tmp[d*2+1], self.scratch["inp_values_p"], offset_regs[d]))

            for d in range(NUM_DESKS):
                self.emit("load", ("vload", desks[d]['idx'], addr_tmp[d*2]))
                self.emit("load", ("vload", desks[d]['val'], addr_tmp[d*2+1]))

            GROUP_SIZE = 4
            num_full_groups = NUM_DESKS // GROUP_SIZE

            all_groups = []
            for g in range(num_full_groups):
                all_groups.append(list(range(g * GROUP_SIZE, (g + 1) * GROUP_SIZE)))

            for group_desks in all_groups:
                emit_rounds_0_1_2_fused(group_desks)

                # Try pipelined gathers for rounds 3-9
                for _rnd in range(3, 10):
                    emit_gather_round_pipelined(group_desks)

                emit_round_10_interleaved(group_desks)
                emit_rounds_11_12_13_fused(group_desks)
                emit_gather_round_pipelined(group_desks)  # round 14
                emit_round_15_final_interleaved(group_desks)

            for d in range(NUM_DESKS):
                self.emit("store", ("vstore", addr_tmp[d*2], desks[d]['idx']))
                self.emit("store", ("vstore", addr_tmp[d*2+1], desks[d]['val']))

        emit_tile_interleaved(0)
        emit_tile_interleaved(1)

        phases = []
        current_phase = []
        for engine, slot in self.slots:
            if engine == "flow" and slot == ("pause",):
                phases.append(current_phase)
                current_phase = []
            else:
                current_phase.append((engine, slot))
        phases.append(current_phase)

        self.instrs = []
        for i, phase in enumerate(phases):
            if phase:
                phase_instrs = _schedule_slots(phase)
                self.instrs.extend(phase_instrs)
            if i < len(phases) - 1:
                self.instrs.append({"flow": [("pause",)]})

        self.instrs.append({"flow": [("pause",)]})

        print(f"Total slots: {len(self.slots)}, Cycles: {len(self.instrs)}")


def do_kernel_test(forest_height: int, rounds: int, batch_size: int, seed: int = 123,
                   trace: bool = False, prints: bool = False, check: bool = False):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilderC2_2()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)

    value_trace = {}
    machine = Machine(mem, kb.instrs, kb.debug_info(), n_cores=N_CORES, value_trace=value_trace, trace=trace)
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if check:
            assert (machine.mem[inp_values_p : inp_values_p + len(inp.values)]
                    == ref_mem[inp_values_p : inp_values_p + len(inp.values)]), f"Incorrect result on round {i}"

    print("CYCLES: ", machine.cycle)
    print("vs B4-2 baseline (1558): ", 1558 - machine.cycle)
    return machine.cycle


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()

    if args.check:
        cycles = do_kernel_test(10, 16, 256, check=True)
        print(f"Correctness check PASSED! Cycles: {cycles}")
    else:
        do_kernel_test(10, 16, 256, trace=args.trace)
