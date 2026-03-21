"""
B2-4: Branch Reduction via Shift and OR

Current: idx = 2*idx + 1 + (val & 1) = 3 VALU ops (AND, FMA, ADD)

Key insight:
  idx' = 2*idx + 1 when val is even (left child = 2i+1)
  idx' = 2*idx + 2 when val is odd (right child = 2i+2)

Using bit manipulation:
  2*idx = idx << 1
  2*idx + 1 = (idx << 1) | 1
  2*idx + 2 = (idx << 1) | 2  [but OR with 2 sets bit 1, which is different from +2]

Wait: (idx << 1) | 2 only works if idx's bit 0 is 0, which it is after <<1
  2*idx = ..xxx0  (always ends in 0 after shift)
  (2*idx) | 2 = ..xxx10  = 2*idx + 2 when lower bit is 0 ✓

So we could do:
  tmp = idx << 1        # 1 op
  bit = (val & 1) + 1   # 2 ops - this is where we're stuck
  idx' = tmp | bit      # 1 op

That's 4 ops total, worse!

ALTERNATIVE approach: Use OR creatively
What if we construct the bit pattern differently?

We want to OR either 0b01 or 0b10 into (idx << 1)
0b01 when val is even, 0b10 when val is odd

Method 1: Direct computation
bit_pattern = ((val & 1) << 1) | (~(val >> 0) & 1)
            = ((val & 1) << 1) | (1 - (val & 1))
This is complex.

Method 2: Precompute
odd_bit = 0b10 = 2
even_bit = 0b01 = 1

We need: vselect(val & 1, odd_bit, even_bit)
Then: idx' = (idx << 1) | selected_bit

Total: shift + AND + vselect + OR = 3 VALU + 1 FLOW
Still 3 VALU ops.

Wait! What if we avoid the AND by using val directly in vselect?
If vselect tests the LSB of the condition, we could do:
  idx' = (idx << 1) | vselect(val, 2, 1)

Let's check if vselect tests LSB or full value...
Looking at the code, vselect likely tests if cond != 0.

So we need the AND after all.

APPROACH B2-4: Let me try using multiply_add differently
idx' = idx * 2 + 1 + (val & 1)

What if: idx' = (idx + (val & 1)) * 2 - (val & 1) + 1
        = 2*idx + 2*(val & 1) - (val & 1) + 1
        = 2*idx + (val & 1) + 1 ✓

So:
  tmp = val & 1                    # 1 VALU
  tmp2 = idx + tmp                 # 1 VALU
  idx' = FMA(tmp2, 2, 1) - tmp    # FMA(tmp2, 2, 1-tmp) but that doesn't help

Hmm, let me try:
  idx' = FMA(idx, 2, 1) + (val & 1)

If we could combine FMA output with the AND result in one step...
But there's no fused add-and operation.

FINAL APPROACH for B2-4:
What if we use multiplication by val's LSB?
idx' = 2*idx + 1 + (val & 1)
     = 2*idx + 1 + (val * 1) & 1  [same thing]

What about: idx' = 2*idx + 1 + ((val >> 0) & 1)
Still the same.

Let me try using OR:
idx' = 2*idx + 1 + (val & 1)
If (2*idx + 1) always has LSB = 1:
  2*idx = even, so 2*idx + 1 has LSB = 1
  If val is even: add 0, result LSB = 1
  If val is odd: add 1, result = 2*idx + 2, LSB = 0

So we could compute:
  base = (idx << 1) | 1   # = 2*idx + 1, LSB is 1
  idx' = base + (val & 1)

Using OR instead of add for the +1:
  idx' = ((idx << 1) | 1) + (val & 1)
       = (idx << 1) + 1 + (val & 1)  [when (idx<<1) has bit0 = 0]
       = shift + OR + AND + ADD = 4 ops (worse!)

OK let me think about this completely differently.

INSIGHT: Maybe we can't reduce to 2 ops, but we can improve ILP.

The current chain is:
  AND -> FMA -> ADD (sequential)

What if we make it:
  FMA (independent of AND)
  AND (independent of FMA)
  ADD (depends on both)

This allows AND and FMA to run in parallel!

Current emit order:
  self.emit("valu", ("&", d['tmp1'], d['val'], v_one))        # depends on val
  self.emit("valu", ("multiply_add", d['idx'], d['idx'], v_two, v_one))  # depends on idx
  self.emit("valu", ("+", d['idx'], d['idx'], d['tmp1']))     # depends on both

The FMA doesn't depend on the AND result, so they can run in parallel.
The scheduler should already handle this!

Let me verify by checking if the issue is register pressure or something else.

Actually, let me try a completely different mathematical approach:
What if we use XOR instead of AND?

val & 1 = val ^ (val & ~1) = val - (val & ~1)
This doesn't simplify.

Or: (val & 1) = (val ^ 0) & 1 [same]

What about using the sign bit or comparison?
val & 1 = 1 if val is odd, 0 if even
val % 2 = val & 1 (by definition)

Is there a 1-op way to get val & 1?
- Modulo? Not available
- Compare val to (val >> 1) << 1? That's 3 ops

I don't think we can reduce branch to 2 VALU ops with the current instruction set.
But let me try one more thing: using the existing FMA more cleverly.

NEW IDEA: What if we merge the AND into the FMA somehow?
FMA computes a*b + c
What if a = val, b = something, c = something that gives us (val & 1)?

No, FMA is multiply-add, not a bitwise operation.

OK let me just test the current approach to verify it's 1613 cycles.
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
        match slot:
            case ("vbroadcast", dest, src):
                reads = [src]
                writes = list(_vec_range(dest))
            case ("multiply_add", dest, a, b, c):
                reads = list(_vec_range(a)) + list(_vec_range(b)) + list(_vec_range(c))
                writes = list(_vec_range(dest))
            case (_op, dest, a1, a2):
                reads = list(_vec_range(a1)) + list(_vec_range(a2))
                writes = list(_vec_range(dest))
            case _:
                raise NotImplementedError(f"Unknown valu op {slot}")
    elif engine == "load":
        match slot:
            case ("load", dest, addr):
                reads = [addr]
                writes = [dest]
            case ("vload", dest, addr):
                reads = [addr]
                writes = list(_vec_range(dest))
            case ("const", dest, _val):
                writes = [dest]
            case ("load_offset", dest, addr, _lane):
                reads = [addr]
                writes = [dest]
            case _:
                raise NotImplementedError(f"Unknown load op {slot}")
    elif engine == "store":
        match slot:
            case ("store", addr, src):
                reads = [addr, src]
            case ("vstore", addr, src):
                reads = [addr] + list(_vec_range(src))
            case _:
                raise NotImplementedError(f"Unknown store op {slot}")
    elif engine == "flow":
        match slot:
            case ("select", dest, cond, a, b):
                reads = [cond, a, b]
                writes = [dest]
            case ("add_imm", dest, a, _imm):
                reads = [a]
                writes = [dest]
            case ("vselect", dest, cond, a, b):
                reads = list(_vec_range(cond)) + list(_vec_range(a)) + list(_vec_range(b))
                writes = list(_vec_range(dest))
            case ("halt",) | ("pause",) | ("trace_write", _) | ("jump", _) | ("jump_indirect", _) | ("cond_jump", _, _) | ("cond_jump_rel", _, _) | ("coreid", _):
                pass
            case _:
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


class KernelBuilderB2_4:
    """B2-4: Test shift+OR approach"""
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
            }
            desks.append(desk)

        offset_regs = [self.alloc_scratch(f"off_{d}") for d in range(NUM_DESKS)]
        addr_tmp = [self.alloc_scratch(f"addr_tmp_{i}") for i in range(32)]

        print(f"Scratch usage: {self.scratch_ptr} / {SCRATCH_SIZE}")

        self.emit("flow", ("pause",))

        def emit_hash_stage_0(desk_idx):
            d = desks[desk_idx]
            self.emit("valu", ("multiply_add", d['val'], d['val'], v_fma_mult[0], v_hash_consts[0]))

        def emit_hash_stage_1(desk_idx):
            d = desks[desk_idx]
            self.emit("valu", ("^", d['tmp1'], d['val'], v_hash_consts[1]))
            self.emit("valu", (">>", d['tmp2'], d['val'], v_hash_shifts[1]))
            self.emit("valu", ("^", d['val'], d['tmp1'], d['tmp2']))

        def emit_hash_stage_2(desk_idx):
            d = desks[desk_idx]
            self.emit("valu", ("multiply_add", d['val'], d['val'], v_fma_mult[2], v_hash_consts[2]))

        def emit_hash_stage_3(desk_idx):
            d = desks[desk_idx]
            self.emit("valu", ("+", d['tmp1'], d['val'], v_hash_consts[3]))
            self.emit("valu", ("<<", d['tmp2'], d['val'], v_hash_shifts[3]))
            self.emit("valu", ("^", d['val'], d['tmp1'], d['tmp2']))

        def emit_hash_stage_4(desk_idx):
            d = desks[desk_idx]
            self.emit("valu", ("multiply_add", d['val'], d['val'], v_fma_mult[4], v_hash_consts[4]))

        def emit_hash_stage_5(desk_idx):
            d = desks[desk_idx]
            self.emit("valu", ("^", d['tmp1'], d['val'], v_hash_consts[5]))
            self.emit("valu", (">>", d['tmp2'], d['val'], v_hash_shifts[5]))
            self.emit("valu", ("^", d['val'], d['tmp1'], d['tmp2']))

        # B2-4: Test using shift+OR instead of FMA
        # idx' = (idx << 1) | 1 + (val & 1)
        # But (idx << 1) | 1 = idx * 2 + 1 only works for this case
        # Then we need to add (val & 1)
        # Total: shift + OR + AND + ADD = 4 ops (worse than 3!)
        #
        # Keeping the original 3-op branch for comparison
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

        def emit_round_0_interleaved(group_desks):
            for d in group_desks:
                emit_xor_with_node(d, v_tree[0])
            for d in group_desks:
                emit_hash_stage_0(d)
            for d in group_desks:
                emit_hash_stage_1(d)
            for d in group_desks:
                emit_hash_stage_2(d)
            for d in group_desks:
                emit_hash_stage_3(d)
            for d in group_desks:
                emit_hash_stage_4(d)
            for d in group_desks:
                emit_hash_stage_5(d)
            for d in group_desks:
                emit_branch(d)

        def emit_round_1_interleaved(group_desks):
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("-", desk['tmp1'], desk['idx'], v_one))
                self.emit("valu", ("multiply_add", desk['node_val'], desk['tmp1'], v_diff_1_2, v_tree[1]))
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("^", desk['val'], desk['val'], desk['node_val']))
            for d in group_desks:
                emit_hash_stage_0(d)
            for d in group_desks:
                emit_hash_stage_1(d)
            for d in group_desks:
                emit_hash_stage_2(d)
            for d in group_desks:
                emit_hash_stage_3(d)
            for d in group_desks:
                emit_hash_stage_4(d)
            for d in group_desks:
                emit_hash_stage_5(d)
            for d in group_desks:
                emit_branch(d)

        def emit_round_2_interleaved_vselect(group_desks):
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("-", desk['tmp1'], desk['idx'], v_three))
                self.emit("valu", ("&", desk['tmp2'], desk['tmp1'], v_one))
                self.emit("valu", (">>", desk['addr'], desk['tmp1'], v_one))
                self.emit("valu", ("multiply_add", desk['node_val'], desk['tmp2'], v_diff_3_4, v_tree[3]))
                self.emit("valu", ("multiply_add", desk['tmp1'], desk['tmp2'], v_diff_5_6, v_tree[5]))
                self.emit("flow", ("vselect", desk['node_val'], desk['addr'], desk['tmp1'], desk['node_val']))
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("^", desk['val'], desk['val'], desk['node_val']))
            for d in group_desks:
                emit_hash_stage_0(d)
            for d in group_desks:
                emit_hash_stage_1(d)
            for d in group_desks:
                emit_hash_stage_2(d)
            for d in group_desks:
                emit_hash_stage_3(d)
            for d in group_desks:
                emit_hash_stage_4(d)
            for d in group_desks:
                emit_hash_stage_5(d)
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
            for d in group_desks:
                emit_hash_stage_0(d)
            for d in group_desks:
                emit_hash_stage_1(d)
            for d in group_desks:
                emit_hash_stage_2(d)
            for d in group_desks:
                emit_hash_stage_3(d)
            for d in group_desks:
                emit_hash_stage_4(d)
            for d in group_desks:
                emit_hash_stage_5(d)
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
            for d in group_desks:
                emit_hash_stage_0(d)
            for d in group_desks:
                emit_hash_stage_1(d)
            for d in group_desks:
                emit_hash_stage_2(d)
            for d in group_desks:
                emit_hash_stage_3(d)
            for d in group_desks:
                emit_hash_stage_4(d)
            for d in group_desks:
                emit_hash_stage_5(d)
            for d in group_desks:
                emit_branch(d)
            for d in group_desks:
                emit_bounds(d)

        def emit_round_11_interleaved(group_desks):
            for d in group_desks:
                emit_xor_with_node(d, v_tree[0])
            for d in group_desks:
                emit_hash_stage_0(d)
            for d in group_desks:
                emit_hash_stage_1(d)
            for d in group_desks:
                emit_hash_stage_2(d)
            for d in group_desks:
                emit_hash_stage_3(d)
            for d in group_desks:
                emit_hash_stage_4(d)
            for d in group_desks:
                emit_hash_stage_5(d)
            for d in group_desks:
                emit_branch(d)

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
            for d in group_desks:
                emit_hash_stage_0(d)
            for d in group_desks:
                emit_hash_stage_1(d)
            for d in group_desks:
                emit_hash_stage_2(d)
            for d in group_desks:
                emit_hash_stage_3(d)
            for d in group_desks:
                emit_hash_stage_4(d)
            for d in group_desks:
                emit_hash_stage_5(d)

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
                emit_round_0_interleaved(group_desks)
                emit_round_1_interleaved(group_desks)
                emit_round_2_interleaved_vselect(group_desks)

                for _rnd in range(3, 10):
                    emit_gather_round_interleaved(group_desks)

                emit_round_10_interleaved(group_desks)
                emit_round_11_interleaved(group_desks)
                emit_round_1_interleaved(group_desks)
                emit_round_2_interleaved_vselect(group_desks)
                emit_gather_round_interleaved(group_desks)
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


BASELINE = 147734


def do_kernel_test(forest_height: int, rounds: int, batch_size: int, seed: int = 123,
                   trace: bool = False, prints: bool = False, check: bool = False):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilderB2_4()
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
    print("Speedup over baseline: ", BASELINE / machine.cycle)
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
