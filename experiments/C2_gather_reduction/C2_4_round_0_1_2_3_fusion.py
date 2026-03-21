"""
C2-4: Extended Round Fusion (0+1+2+3 and 11+12+13+14)

The key insight: If we can extend the fusion from rounds 0-2 to rounds 0-3,
we eliminate one ENTIRE gather round per group.

Current state after round 2:
- idx in {3,4,5,6} = tree level 2
- bit0, bit1 tracked from rounds 0,1

For round 3:
- XOR with tree[idx] where idx in {3,4,5,6}
- Hash
- Branch: new_idx = 2*idx + 1 + bit2

The trick: We already have the 4-way selection for tree[3..6] in round 2.
The result is stored in node_val. After XOR, we hash and branch.

After round 3's branch:
- idx in {7,8,9,10,11,12,13,14} = tree level 3

For round 4 (or rather, continuing the fusion):
- We'd need tree[7..14] = 8 nodes
- 8-way selection is expensive (C2-1 showed this)

But wait, let me check if we can do BETTER than 8-way selection.

After round 2's branch with bits (bit0, bit1, bit2):
idx = 2*(3 + 2*bit0 + bit1) + 1 + bit2
    = 6 + 4*bit0 + 2*bit1 + 1 + bit2
    = 7 + 4*bit0 + 2*bit1 + bit2

So idx in {7..14} = 7 + (4*bit0 + 2*bit1 + bit2)

The offset is offset = 4*bit0 + 2*bit1 + bit2 (a 3-bit number).

For 8-way selection, we need to select tree[7 + offset].

If we preload tree[7..14] as v_tree_7..v_tree_14, we can use:
1. Binary tree of vselects (7 vselects)
2. Arithmetic with differences (3 levels, 7 multiply_adds)

Both approaches need 7 FLOW or VALU ops.

Current gather: 1 addr computation + 8 loads = 1 VALU + 4 load-cycles

If VALU is the bottleneck, adding 7 VALU-ish ops is worse.
If LOAD is the bottleneck, saving 8 loads * (1/2 cycle) = 4 cycles might help.

Let's calculate total ops saved/added:
- Saved: 4 desks * 8 loads = 32 loads per group
- Added: 4 desks * ~12 VALU ops (bit extraction + selection) = 48 VALU ops

At 2 loads/cycle: 32 loads = 16 cycles saved (per group)
At 6 VALU/cycle: 48 VALU = 8 cycles added (per group)

Net: 16 - 8 = 8 cycles saved per group!
With 8 groups total: 64 cycles saved!

Let's implement this!

Strategy:
1. Preload tree[7..14] at initialization (8 scalar loads + 8 broadcasts)
2. Fuse rounds 0+1+2+3 using bit tracking
3. Same for rounds 11+12+13+14 after wrap

For 8-way selection, use hierarchical approach:
- bit0 selects between pairs: (7,8), (9,10), (11,12), (13,14)
- bit1 selects between quads: (7-10), (11-14)
- bit2 selects between the final two

Using multiply_add for 2-way selection:
pair_01 = tree[7] + bit2 * (tree[8] - tree[7])
pair_23 = tree[9] + bit2 * (tree[10] - tree[9])
pair_45 = tree[11] + bit2 * (tree[12] - tree[11])
pair_67 = tree[13] + bit2 * (tree[14] - tree[13])

quad_low = pair_01 + bit1 * (pair_23 - pair_01)
quad_high = pair_45 + bit1 * (pair_67 - pair_45)

result = quad_low + bit0 * (quad_high - quad_low)

This needs:
- 4 precomputed differences (for pairs)
- 4 multiply_adds (for pairs)
- 2 differences computed inline
- 2 multiply_adds (for quads)
- 1 difference computed inline
- 1 multiply_add (for result)

Total: 4 + 2 + 1 = 7 differences, 4 + 2 + 1 = 7 multiply_adds = 14 VALU ops per desk.

For 4 desks: 56 VALU ops per group for round 3 node selection.

But we SAVE:
- 4 addr computations (4 VALU ops)
- 32 loads (16 load-cycles)

Net VALU change: +56 - 4 = +52 VALU ops
Net load change: -32 loads = -16 cycles

At 6 VALU/cycle: +52 VALU = ~9 extra VALU cycles
At 2 loads/cycle: -32 loads = -16 cycles

If VALU and loads overlap well, net could be positive.
Let's test it!
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


class KernelBuilderC2_4:
    """C2-4: Extended round fusion (0-3 and 11-14) with 8-way preloaded selection"""

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
        v_four = self.scratch_vconst(4, "v_four")
        v_seven = self.scratch_vconst(7, "v_seven")
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

        # Preload tree nodes 0-14 (levels 0-3)
        NUM_PRELOADED = 15
        v_tree = []
        for i in range(NUM_PRELOADED):
            v_node = self.alloc_vec(f"v_tree_{i}")
            v_tree.append(v_node)
            self.emit("alu", ("+", tmp_addr, self.scratch["forest_values_p"], self.scratch_const(i)))
            self.emit("load", ("load", tmp_scalar, tmp_addr))
            self.emit("valu", ("vbroadcast", v_node, tmp_scalar))

        # Differences for level 1 (2-way selection in round 1)
        v_diff_1_2 = self.alloc_vec("v_diff_1_2")
        self.emit("valu", ("-", v_diff_1_2, v_tree[2], v_tree[1]))

        # Differences for level 2 (4-way selection in round 2)
        v_diff_3_4 = self.alloc_vec("v_diff_3_4")
        v_diff_5_6 = self.alloc_vec("v_diff_5_6")
        self.emit("valu", ("-", v_diff_3_4, v_tree[4], v_tree[3]))
        self.emit("valu", ("-", v_diff_5_6, v_tree[6], v_tree[5]))

        # Differences for level 3 (8-way selection in round 3)
        # For pairs: (7,8), (9,10), (11,12), (13,14)
        v_diff_7_8 = self.alloc_vec("v_diff_7_8")
        v_diff_9_10 = self.alloc_vec("v_diff_9_10")
        v_diff_11_12 = self.alloc_vec("v_diff_11_12")
        v_diff_13_14 = self.alloc_vec("v_diff_13_14")
        self.emit("valu", ("-", v_diff_7_8, v_tree[8], v_tree[7]))
        self.emit("valu", ("-", v_diff_9_10, v_tree[10], v_tree[9]))
        self.emit("valu", ("-", v_diff_11_12, v_tree[12], v_tree[11]))
        self.emit("valu", ("-", v_diff_13_14, v_tree[14], v_tree[13]))

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
                'bit0': self.alloc_vec(f"v_bit0_{d}"),  # Track bit from round 0
                'bit1': self.alloc_vec(f"v_bit1_{d}"),  # Track bit from round 1
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

        def emit_rounds_0_1_2_3_fused(group_desks):
            """
            Fused rounds 0+1+2+3 with bit tracking and 8-way selection for round 3
            """
            # === Round 0 ===
            for d in group_desks:
                emit_xor_with_node(d, v_tree[0])
            emit_hash_interleaved(group_desks)
            # Extract bit0 and compute idx1 = 1 + bit0
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("&", desk['bit0'], desk['val'], v_one))
                self.emit("valu", ("+", desk['idx'], v_one, desk['bit0']))

            # === Round 1 ===
            # node1 = tree[1] + bit0 * diff_1_2
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("multiply_add", desk['node_val'], desk['bit0'], v_diff_1_2, v_tree[1]))
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("^", desk['val'], desk['val'], desk['node_val']))
            emit_hash_interleaved(group_desks)
            # Extract bit1 and compute idx2 = 3 + 2*bit0 + bit1
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("&", desk['bit1'], desk['val'], v_one))  # Save bit1
                self.emit("valu", ("multiply_add", desk['idx'], desk['bit0'], v_two, v_three))
                self.emit("valu", ("+", desk['idx'], desk['idx'], desk['bit1']))

            # === Round 2 ===
            # 4-way selection from tree[3..6] using bit0 (saved) and bit1
            for d in group_desks:
                desk = desks[d]
                # bit1 is fresh from round 1
                # pair_low = tree[3] + bit1 * diff_3_4
                # pair_high = tree[5] + bit1 * diff_5_6
                self.emit("valu", ("multiply_add", desk['node_val'], desk['bit1'], v_diff_3_4, v_tree[3]))
                self.emit("valu", ("multiply_add", desk['tmp2'], desk['bit1'], v_diff_5_6, v_tree[5]))
                self.emit("flow", ("vselect", desk['node_val'], desk['bit0'], desk['tmp2'], desk['node_val']))
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("^", desk['val'], desk['val'], desk['node_val']))
            emit_hash_interleaved(group_desks)
            # Extract bit2 and compute idx3 = 7 + 4*bit0 + 2*bit1 + bit2
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("&", desk['tmp1'], desk['val'], v_one))  # bit2 in tmp1
                # idx3 = 7 + 4*bit0 + 2*bit1 + bit2
                self.emit("valu", ("multiply_add", desk['tmp2'], desk['bit0'], v_four, v_seven))  # 7 + 4*bit0
                self.emit("valu", ("multiply_add", desk['idx'], desk['bit1'], v_two, desk['tmp2']))  # + 2*bit1
                self.emit("valu", ("+", desk['idx'], desk['idx'], desk['tmp1']))  # + bit2

            # === Round 3 ===
            # 8-way selection from tree[7..14] using bit0, bit1, bit2
            # bit0 is in desk['bit0'], bit1 in desk['bit1'], bit2 in desk['tmp1']

            # Level 0: 4 pairs
            # pair_01 = tree[7] + bit2 * diff_7_8
            # pair_23 = tree[9] + bit2 * diff_9_10
            # pair_45 = tree[11] + bit2 * diff_11_12
            # pair_67 = tree[13] + bit2 * diff_13_14
            for d in group_desks:
                desk = desks[d]
                # bit2 is in tmp1
                # Use node_val and addr as temporaries
                self.emit("valu", ("multiply_add", desk['node_val'], desk['tmp1'], v_diff_7_8, v_tree[7]))   # pair_01
                self.emit("valu", ("multiply_add", desk['addr'], desk['tmp1'], v_diff_9_10, v_tree[9]))      # pair_23
                self.emit("valu", ("multiply_add", desk['tmp2'], desk['tmp1'], v_diff_11_12, v_tree[11]))    # pair_45

            # pair_67 and quad selections
            for d in group_desks:
                desk = desks[d]
                # Need one more temp for pair_67, then start selecting
                # We'll reuse carefully

                # First, select within pairs using bit1:
                # quad_low = pair_01 + bit1 * (pair_23 - pair_01)
                self.emit("valu", ("-", desk['bit1'], desk['addr'], desk['node_val']))  # OOPS, destroys bit1!

            # Let me restructure to preserve bits properly
            # RESTART round 3 selection with better register allocation

        def emit_rounds_0_1_2_3_fused_v2(group_desks):
            """
            Fused rounds 0+1+2+3 with careful bit tracking
            Need extra temp for 8-way selection
            """
            # === Round 0 ===
            for d in group_desks:
                emit_xor_with_node(d, v_tree[0])
            emit_hash_interleaved(group_desks)
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("&", desk['bit0'], desk['val'], v_one))
                self.emit("valu", ("+", desk['idx'], v_one, desk['bit0']))

            # === Round 1 ===
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("multiply_add", desk['node_val'], desk['bit0'], v_diff_1_2, v_tree[1]))
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("^", desk['val'], desk['val'], desk['node_val']))
            emit_hash_interleaved(group_desks)
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("&", desk['bit1'], desk['val'], v_one))
                self.emit("valu", ("multiply_add", desk['idx'], desk['bit0'], v_two, v_three))
                self.emit("valu", ("+", desk['idx'], desk['idx'], desk['bit1']))

            # === Round 2 ===
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("multiply_add", desk['node_val'], desk['bit1'], v_diff_3_4, v_tree[3]))
                self.emit("valu", ("multiply_add", desk['tmp2'], desk['bit1'], v_diff_5_6, v_tree[5]))
                self.emit("flow", ("vselect", desk['node_val'], desk['bit0'], desk['tmp2'], desk['node_val']))
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("^", desk['val'], desk['val'], desk['node_val']))
            emit_hash_interleaved(group_desks)
            # Extract bit2 into tmp1
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("&", desk['tmp1'], desk['val'], v_one))  # bit2

            # === Round 3 ===
            # 8-way selection: tree[7 + 4*bit0 + 2*bit1 + bit2]
            # We have: bit0 in desk['bit0'], bit1 in desk['bit1'], bit2 in desk['tmp1']

            # Strategy: Use hierarchical vselect
            # First level: bit2 selects between adjacent pairs
            # Second level: bit1 selects between pairs of pairs
            # Third level: bit0 selects final result

            # Level 0: Use bit2 to select within pairs
            # pair_0 = tree[7] + bit2 * diff_7_8   -> node_val
            # pair_1 = tree[9] + bit2 * diff_9_10  -> addr
            # pair_2 = tree[11] + bit2 * diff_11_12 -> tmp2
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("multiply_add", desk['node_val'], desk['tmp1'], v_diff_7_8, v_tree[7]))
                self.emit("valu", ("multiply_add", desk['addr'], desk['tmp1'], v_diff_9_10, v_tree[9]))
                self.emit("valu", ("multiply_add", desk['tmp2'], desk['tmp1'], v_diff_11_12, v_tree[11]))

            # pair_3 = tree[13] + bit2 * diff_13_14
            # Then: quad_low = vselect(bit1, pair_1, pair_0)
            #       quad_high = vselect(bit1, pair_3, pair_2)
            #       result = vselect(bit0, quad_high, quad_low)
            for d in group_desks:
                desk = desks[d]
                # pair_3 needs a temp - use val temporarily (will be overwritten by XOR later anyway)
                self.emit("valu", ("multiply_add", desk['val'], desk['tmp1'], v_diff_13_14, v_tree[13]))  # pair_3 in val

                # Now select:
                # quad_low = vselect(bit1, pair_1=addr, pair_0=node_val) -> node_val
                self.emit("flow", ("vselect", desk['node_val'], desk['bit1'], desk['addr'], desk['node_val']))
                # quad_high = vselect(bit1, pair_3=val, pair_2=tmp2) -> tmp2
                self.emit("flow", ("vselect", desk['tmp2'], desk['bit1'], desk['val'], desk['tmp2']))

                # Final: result = vselect(bit0, quad_high=tmp2, quad_low=node_val) -> node_val
                self.emit("flow", ("vselect", desk['node_val'], desk['bit0'], desk['tmp2'], desk['node_val']))

            # Now we need to restore val for XOR
            # Actually, val got clobbered with pair_3. We need to reload val!
            # This is a problem... let me rethink.

        def emit_rounds_0_1_2_3_fused_v3(group_desks):
            """
            Fused rounds 0+1+2+3 - use addr for pair_3 instead of val
            """
            # === Round 0 ===
            for d in group_desks:
                emit_xor_with_node(d, v_tree[0])
            emit_hash_interleaved(group_desks)
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("&", desk['bit0'], desk['val'], v_one))
                self.emit("valu", ("+", desk['idx'], v_one, desk['bit0']))

            # === Round 1 ===
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("multiply_add", desk['node_val'], desk['bit0'], v_diff_1_2, v_tree[1]))
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("^", desk['val'], desk['val'], desk['node_val']))
            emit_hash_interleaved(group_desks)
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("&", desk['bit1'], desk['val'], v_one))
                self.emit("valu", ("multiply_add", desk['idx'], desk['bit0'], v_two, v_three))
                self.emit("valu", ("+", desk['idx'], desk['idx'], desk['bit1']))

            # === Round 2 ===
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("multiply_add", desk['node_val'], desk['bit1'], v_diff_3_4, v_tree[3]))
                self.emit("valu", ("multiply_add", desk['tmp2'], desk['bit1'], v_diff_5_6, v_tree[5]))
                self.emit("flow", ("vselect", desk['node_val'], desk['bit0'], desk['tmp2'], desk['node_val']))
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("^", desk['val'], desk['val'], desk['node_val']))
            emit_hash_interleaved(group_desks)

            # Extract bit2 and save val for later
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("&", desk['tmp1'], desk['val'], v_one))  # bit2 in tmp1
                # val is needed for round 3 XOR

            # === Round 3 ===
            # 8-way selection needs 4 temps. We have: node_val, addr, tmp1, tmp2
            # bit2 is in tmp1, so we can only use node_val, addr, tmp2 as temps

            # Compute all 4 pair selections first, then combine
            # pair_0 = tree[7] + bit2 * diff_7_8
            # pair_1 = tree[9] + bit2 * diff_9_10
            # pair_2 = tree[11] + bit2 * diff_11_12
            # pair_3 = tree[13] + bit2 * diff_13_14

            # Then: quad_low = pair_0 + bit1 * (pair_1 - pair_0)
            #       quad_high = pair_2 + bit1 * (pair_3 - pair_2)
            #       result = quad_low + bit0 * (quad_high - quad_low)

            # Do pairs 0,1 and compute quad_low
            for d in group_desks:
                desk = desks[d]
                # pair_0 in node_val
                self.emit("valu", ("multiply_add", desk['node_val'], desk['tmp1'], v_diff_7_8, v_tree[7]))
                # pair_1 in addr
                self.emit("valu", ("multiply_add", desk['addr'], desk['tmp1'], v_diff_9_10, v_tree[9]))
                # quad_low = vselect(bit1, pair_1, pair_0) -> node_val
                self.emit("flow", ("vselect", desk['node_val'], desk['bit1'], desk['addr'], desk['node_val']))

            # Do pairs 2,3 and compute quad_high
            for d in group_desks:
                desk = desks[d]
                # pair_2 in addr (reusing)
                self.emit("valu", ("multiply_add", desk['addr'], desk['tmp1'], v_diff_11_12, v_tree[11]))
                # pair_3 in tmp2
                self.emit("valu", ("multiply_add", desk['tmp2'], desk['tmp1'], v_diff_13_14, v_tree[13]))
                # quad_high = vselect(bit1, pair_3, pair_2) -> addr
                self.emit("flow", ("vselect", desk['addr'], desk['bit1'], desk['tmp2'], desk['addr']))

            # Final selection
            for d in group_desks:
                desk = desks[d]
                # result = vselect(bit0, quad_high=addr, quad_low=node_val) -> node_val
                self.emit("flow", ("vselect", desk['node_val'], desk['bit0'], desk['addr'], desk['node_val']))

            # XOR val with selected node
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("^", desk['val'], desk['val'], desk['node_val']))

            # Hash and branch for round 3
            emit_hash_interleaved(group_desks)

            # Compute idx4 from idx3 using standard branch
            for d in group_desks:
                desk = desks[d]
                # idx3 = 7 + 4*bit0 + 2*bit1 + bit2, but we need to recompute it
                # Actually we didn't store idx3. Let me compute it now.
                self.emit("valu", ("multiply_add", desk['tmp2'], desk['bit0'], v_four, v_seven))
                self.emit("valu", ("multiply_add", desk['idx'], desk['bit1'], v_two, desk['tmp2']))
                self.emit("valu", ("+", desk['idx'], desk['idx'], desk['tmp1']))  # idx3

            for d in group_desks:
                emit_branch(d)

        def emit_rounds_11_12_13_14_fused(group_desks):
            """
            Fused rounds 11+12+13+14 (same as 0+1+2+3 after wrap)
            """
            # === Round 11 === (same as round 0)
            for d in group_desks:
                emit_xor_with_node(d, v_tree[0])
            emit_hash_interleaved(group_desks)
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("&", desk['bit0'], desk['val'], v_one))
                self.emit("valu", ("+", desk['idx'], v_one, desk['bit0']))

            # === Round 12 ===
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("multiply_add", desk['node_val'], desk['bit0'], v_diff_1_2, v_tree[1]))
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("^", desk['val'], desk['val'], desk['node_val']))
            emit_hash_interleaved(group_desks)
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("&", desk['bit1'], desk['val'], v_one))
                self.emit("valu", ("multiply_add", desk['idx'], desk['bit0'], v_two, v_three))
                self.emit("valu", ("+", desk['idx'], desk['idx'], desk['bit1']))

            # === Round 13 ===
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("multiply_add", desk['node_val'], desk['bit1'], v_diff_3_4, v_tree[3]))
                self.emit("valu", ("multiply_add", desk['tmp2'], desk['bit1'], v_diff_5_6, v_tree[5]))
                self.emit("flow", ("vselect", desk['node_val'], desk['bit0'], desk['tmp2'], desk['node_val']))
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("^", desk['val'], desk['val'], desk['node_val']))
            emit_hash_interleaved(group_desks)

            # Extract bit2
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("&", desk['tmp1'], desk['val'], v_one))

            # === Round 14 ===
            # 8-way selection (same as round 3)
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("multiply_add", desk['node_val'], desk['tmp1'], v_diff_7_8, v_tree[7]))
                self.emit("valu", ("multiply_add", desk['addr'], desk['tmp1'], v_diff_9_10, v_tree[9]))
                self.emit("flow", ("vselect", desk['node_val'], desk['bit1'], desk['addr'], desk['node_val']))

            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("multiply_add", desk['addr'], desk['tmp1'], v_diff_11_12, v_tree[11]))
                self.emit("valu", ("multiply_add", desk['tmp2'], desk['tmp1'], v_diff_13_14, v_tree[13]))
                self.emit("flow", ("vselect", desk['addr'], desk['bit1'], desk['tmp2'], desk['addr']))

            for d in group_desks:
                desk = desks[d]
                self.emit("flow", ("vselect", desk['node_val'], desk['bit0'], desk['addr'], desk['node_val']))

            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("^", desk['val'], desk['val'], desk['node_val']))

            emit_hash_interleaved(group_desks)

            # Compute idx after round 14
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("multiply_add", desk['tmp2'], desk['bit0'], v_four, v_seven))
                self.emit("valu", ("multiply_add", desk['idx'], desk['bit1'], v_two, desk['tmp2']))
                self.emit("valu", ("+", desk['idx'], desk['idx'], desk['tmp1']))

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
                # Fused rounds 0+1+2+3
                emit_rounds_0_1_2_3_fused_v3(group_desks)

                # Rounds 4-9: gather rounds
                for _rnd in range(4, 10):
                    emit_gather_round_interleaved(group_desks)

                # Round 10: last before wrap
                emit_round_10_interleaved(group_desks)

                # Fused rounds 11+12+13+14
                emit_rounds_11_12_13_14_fused(group_desks)

                # Round 15: final (no branch)
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

    kb = KernelBuilderC2_4()
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
