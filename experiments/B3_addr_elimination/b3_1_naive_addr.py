"""
# Experiment B3-1: Naive address storage to eliminate gather address computation

The idea: Store `addr = forest_p + idx` instead of just `idx`.
This eliminates the `addr = forest_p + idx` VALU op in gather rounds.

Current branch: idx = 2*idx + 1 + (val & 1)  -- 3 ops (multiply_add + and + add)

New branch with addr storage:
  addr = 2*(addr - forest_p) + 1 + (val & 1) + forest_p
      = 2*addr - 2*forest_p + 1 + (val & 1) + forest_p
      = 2*addr - forest_p + 1 + (val & 1)

So new branch:
  tmp1 = val & 1
  tmp2 = 2*addr - forest_p   (need to precompute -forest_p or use sub)
  addr = tmp2 + 1 + tmp1

Actually:
  tmp1 = val & 1
  addr = 2*addr + tmp1 + 1 - forest_p

Using multiply_add: addr = 2*addr + 1  (via multiply_add with v_one)
Then: addr = addr + tmp1
Then: addr = addr - forest_p

That's 4 ops vs current 3! But we save 1 op per gather round.
10 gather rounds * 1 op saved = 10 ops per desk
But branch runs 16 times, adding 16 extra ops per desk.
Net: +6 ops per desk - BAD

Let me try a different formulation:
  addr = 2*addr + (1 - forest_p) + (val & 1)

If we precompute v_one_minus_forest_p = 1 - forest_p, then:
  tmp1 = val & 1
  addr = multiply_add(addr, 2, v_one_minus_forest_p)  -- 2*addr + (1-forest_p)
  addr = addr + tmp1

That's still 3 ops, same as before! But we eliminate 10 gather addr ops.
Net: -10 ops per desk = GOOD

Wait, let me verify current branch more carefully from the code:
  self.emit("valu", ("&", d['tmp1'], d['val'], v_one))           # tmp1 = val & 1
  self.emit("valu", ("multiply_add", d['idx'], d['idx'], v_two, v_one))  # idx = 2*idx + 1
  self.emit("valu", ("+", d['idx'], d['idx'], d['tmp1']))        # idx = idx + tmp1

So current is 3 ops. New formulation also 3 ops, but we save 10 gather ops.
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


class KernelBuilderB3_1:
    """B3-1: Store addr instead of idx to eliminate gather address computation"""
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

        # NEW: Precompute v_neg_forest_p = -forest_p and v_one_minus_forest_p = 1 - forest_p
        # For branch: addr = 2*addr + (1 - forest_p) + (val & 1)
        v_neg_forest_p = self.alloc_vec("v_neg_forest_p")
        self.emit("valu", ("-", v_neg_forest_p, v_zero, v_forest_p))

        v_one_minus_forest_p = self.alloc_vec("v_one_minus_forest_p")
        self.emit("valu", ("+", v_one_minus_forest_p, v_one, v_neg_forest_p))

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
                'idx': self.alloc_vec(f"v_idx_{d}"),  # Now stores addr = forest_p + idx
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

        # Per-stage hash emission (for interleaving across desks)
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

        def emit_branch_addr(desk_idx):
            """New branch that works with addr = forest_p + idx stored in 'idx' register"""
            d = desks[desk_idx]
            # addr = 2*addr + (1 - forest_p) + (val & 1)
            self.emit("valu", ("&", d['tmp1'], d['val'], v_one))  # tmp1 = val & 1
            self.emit("valu", ("multiply_add", d['idx'], d['idx'], v_two, v_one_minus_forest_p))  # idx = 2*idx + (1-forest_p)
            self.emit("valu", ("+", d['idx'], d['idx'], d['tmp1']))  # idx = idx + tmp1

        def emit_bounds_addr(desk_idx):
            """Bounds check adjusted for addr storage"""
            d = desks[desk_idx]
            # We need to check if (addr - forest_p) < n_nodes
            # Original code did: idx = idx * (idx < n_nodes), which sets idx=0 if invalid
            # With addr storage, we want: addr = forest_p if invalid (so it points to tree[0])
            # If valid: keep addr. If invalid: set addr = forest_p
            # addr = valid ? addr : forest_p
            # Using: addr = addr * valid + forest_p * (1 - valid)
            #      = addr * valid + forest_p - forest_p * valid
            #      = valid * (addr - forest_p) + forest_p
            self.emit("valu", ("-", d['tmp1'], d['idx'], v_forest_p))  # tmp1 = idx (actual index)
            self.emit("valu", ("<", d['tmp2'], d['tmp1'], v_n_nodes))  # tmp2 = valid = (idx < n_nodes)
            # addr = tmp1 * tmp2 + forest_p = idx * valid + forest_p
            self.emit("valu", ("*", d['tmp1'], d['tmp1'], d['tmp2']))  # tmp1 = idx * valid
            self.emit("valu", ("+", d['idx'], d['tmp1'], v_forest_p))  # addr = tmp1 + forest_p

        def emit_xor_with_node(desk_idx, node_vec):
            d = desks[desk_idx]
            self.emit("valu", ("^", d['val'], d['val'], node_vec))

        def emit_convert_idx_to_addr(desk_idx):
            """Convert initial idx to addr = forest_p + idx"""
            d = desks[desk_idx]
            self.emit("valu", ("+", d['idx'], d['idx'], v_forest_p))

        def emit_round_0_interleaved(group_desks):
            """Emit round 0 with stage-interleaved hash for all desks in group"""
            # First, convert idx to addr for all desks
            for d in group_desks:
                emit_convert_idx_to_addr(d)
            # XOR with tree[0]
            for d in group_desks:
                emit_xor_with_node(d, v_tree[0])
            # Hash stage 0 (all desks)
            for d in group_desks:
                emit_hash_stage_0(d)
            # Hash stage 1 (all desks)
            for d in group_desks:
                emit_hash_stage_1(d)
            # Hash stage 2 (all desks)
            for d in group_desks:
                emit_hash_stage_2(d)
            # Hash stage 3 (all desks)
            for d in group_desks:
                emit_hash_stage_3(d)
            # Hash stage 4 (all desks)
            for d in group_desks:
                emit_hash_stage_4(d)
            # Hash stage 5 (all desks)
            for d in group_desks:
                emit_hash_stage_5(d)
            # Branch (all desks) - now using addr-based branch
            for d in group_desks:
                emit_branch_addr(d)

        def emit_round_1_interleaved(group_desks):
            # Selection (2-way) - need to convert addr to idx for selection then back
            for d in group_desks:
                desk = desks[d]
                # idx = addr - forest_p
                self.emit("valu", ("-", desk['tmp1'], desk['idx'], v_forest_p))  # tmp1 = idx
                self.emit("valu", ("-", desk['tmp2'], desk['tmp1'], v_one))  # tmp2 = idx - 1
                self.emit("valu", ("multiply_add", desk['node_val'], desk['tmp2'], v_diff_1_2, v_tree[1]))
            # XOR
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("^", desk['val'], desk['val'], desk['node_val']))
            # Hash stages interleaved
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
                emit_branch_addr(d)

        def emit_round_2_interleaved_vselect(group_desks):
            # 4-way selection setup - need idx = addr - forest_p
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("-", desk['addr'], desk['idx'], v_forest_p))  # addr = idx (temp)
                self.emit("valu", ("-", desk['tmp1'], desk['addr'], v_three))  # tmp1 = idx - 3
                self.emit("valu", ("&", desk['tmp2'], desk['tmp1'], v_one))
                self.emit("valu", (">>", desk['addr'], desk['tmp1'], v_one))
                self.emit("valu", ("multiply_add", desk['node_val'], desk['tmp2'], v_diff_3_4, v_tree[3]))
                self.emit("valu", ("multiply_add", desk['tmp1'], desk['tmp2'], v_diff_5_6, v_tree[5]))
                self.emit("flow", ("vselect", desk['node_val'], desk['addr'], desk['tmp1'], desk['node_val']))
            # XOR
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("^", desk['val'], desk['val'], desk['node_val']))
            # Hash stages interleaved
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
                emit_branch_addr(d)

        def emit_gather_round_interleaved(group_desks):
            # NO ADDRESS COMPUTATION NEEDED - idx already stores addr!
            # Gather loads using idx directly
            for d in group_desks:
                desk = desks[d]
                for lane in range(VLEN):
                    self.emit("load", ("load", desk['node_val'] + lane, desk['idx'] + lane))
            # XOR
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("^", desk['val'], desk['val'], desk['node_val']))
            # Hash stages interleaved
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
                emit_branch_addr(d)

        def emit_round_10_interleaved(group_desks):
            # Gather loads using idx directly (stores addr)
            for d in group_desks:
                desk = desks[d]
                for lane in range(VLEN):
                    self.emit("load", ("load", desk['node_val'] + lane, desk['idx'] + lane))
            # XOR
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("^", desk['val'], desk['val'], desk['node_val']))
            # Hash stages interleaved
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
                emit_branch_addr(d)
            # Bounds check
            for d in group_desks:
                emit_bounds_addr(d)

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
                emit_branch_addr(d)

        def emit_round_15_final_interleaved(group_desks):
            # Gather loads using idx directly (stores addr)
            for d in group_desks:
                desk = desks[d]
                for lane in range(VLEN):
                    self.emit("load", ("load", desk['node_val'] + lane, desk['idx'] + lane))
            # XOR
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("^", desk['val'], desk['val'], desk['node_val']))
            # Hash stages interleaved (no branch)
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

        def emit_convert_addr_to_idx(desk_idx):
            """Convert addr back to idx for store"""
            d = desks[desk_idx]
            self.emit("valu", ("-", d['idx'], d['idx'], v_forest_p))

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
                emit_round_1_interleaved(group_desks)  # Round 12 same as 1
                emit_round_2_interleaved_vselect(group_desks)  # Round 13 same as 2
                emit_gather_round_interleaved(group_desks)  # Round 14
                emit_round_15_final_interleaved(group_desks)

            # Convert addr back to idx before storing
            for d in range(NUM_DESKS):
                emit_convert_addr_to_idx(d)

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

    kb = KernelBuilderB3_1()
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
