"""
# Experiment H136 v4: Address-Centric Computation

**GOAL:** Reduce cycles by computing addr directly without maintaining idx separately.

**HYPOTHESIS:** Currently we maintain both idx and addr:
  - idx = 2*idx + 1 + bit  (3 ops)
  - addr = forest_p + idx  (1 op)
  - bounds: idx < n_nodes

If we precompute forest_p_plus_1 = forest_p + 1 and bounds_addr = forest_p + n_nodes,
we can compute addr directly:
  - addr = 2*idx + forest_p + 1 + bit = FMA(idx, 2, forest_p_plus_1) + bit  (2 ops)
  - idx = addr - forest_p  (1 op, only when needed)
  - bounds: addr < bounds_addr

For gather rounds without bounds, we only need addr:
  - addr = FMA(idx, 2, forest_p_plus_1) + bit  (2 ops)
  - idx_for_output = addr - forest_p  (can be deferred to end)

Wait, the issue is we need idx for the NEXT round's FMA:
  addr_next = FMA(idx_curr, 2, forest_p_plus_1) + bit

So we still need to recover idx. Let me think differently...

Actually, for gather rounds 3-9 without bounds:
  Current: idx = FMA(idx, 2, 1) + bit; addr = forest_p + idx  (3 ops + 1 op = 4 ops)
  New:     addr = FMA(idx, 2, forest_p_plus_1) + bit; idx = addr - forest_p  (2 ops + 1 op = 3 ops)

This saves 1 op per gather round! For 9 gather rounds x 32 desks = 288 VALU ops saved.
At 6 VALU/cycle, that's 48 cycles saved!

**BASELINE:** H120 = 1,840 cycles
**TARGET:** ~1,792 cycles
"""

import random
import unittest
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
    """Get read and write addresses for a slot."""
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
    """Automatically schedule operations into VLIW bundles respecting dependencies."""
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


class KernelBuilderH136v4:
    """
    H136 v4: Address-centric computation using FMA.
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
        assert self.scratch_ptr <= SCRATCH_SIZE, f"Out of scratch space: {self.scratch_ptr}"
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

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        tmp_scalar = self.alloc_scratch("tmp_scalar")
        tmp_addr = self.alloc_scratch("tmp_addr")

        needed_vars = [
            (1, "n_nodes"),
            (4, "forest_values_p"),
            (5, "inp_indices_p"),
            (6, "inp_values_p"),
        ]
        for idx, v in needed_vars:
            self.alloc_scratch(v)
        for idx, v in needed_vars:
            self.emit("load", ("const", tmp_scalar, idx))
            self.emit("load", ("load", self.scratch[v], tmp_scalar))

        v_zero = self.scratch_vconst(0, "v_zero")
        v_one = self.scratch_vconst(1, "v_one")
        v_two = self.scratch_vconst(2, "v_two")
        v_three = self.scratch_vconst(3, "v_three")
        v_n_nodes = self.alloc_vec("v_n_nodes")
        self.emit("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]))

        v_forest_p = self.alloc_vec("v_forest_p")
        self.emit("valu", ("vbroadcast", v_forest_p, self.scratch["forest_values_p"]))

        # H136 v4: Precompute forest_p + 1 for FMA-based addr computation
        v_forest_p_plus_1 = self.alloc_vec("v_forest_p_plus_1")
        self.emit("valu", ("+", v_forest_p_plus_1, v_forest_p, v_one))

        # Also precompute bounds threshold: n_nodes + forest_p
        v_bounds_addr = self.alloc_vec("v_bounds_addr")
        self.emit("valu", ("+", v_bounds_addr, v_n_nodes, v_forest_p))

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

        NUM_DESKS = 32
        desks = []
        for d in range(NUM_DESKS):
            desk = {
                'idx': self.alloc_vec(f"v_idx_{d}"),
                'val': self.alloc_vec(f"v_val_{d}"),
                'node_val': self.alloc_vec(f"v_node_{d}"),
                'addr': self.alloc_vec(f"v_addr_{d}"),
            }
            desks.append(desk)

        NUM_SHARED_TEMPS = 16
        shared_temps = [self.alloc_vec(f"v_shared_tmp_{i}") for i in range(NUM_SHARED_TEMPS)]

        def get_temps(desk_idx):
            group = desk_idx // 4
            t1_idx = group * 2
            t2_idx = group * 2 + 1
            return shared_temps[t1_idx], shared_temps[t2_idx]

        offset_regs = [self.alloc_scratch(f"off_{d}") for d in range(NUM_DESKS)]
        addr_tmp = [self.alloc_scratch(f"addr_tmp_{i}") for i in range(4)]

        print(f"Scratch usage: {self.scratch_ptr} / {SCRATCH_SIZE}")
        print(f"  - H136 v4: Address-centric FMA computation")

        self.emit("flow", ("pause",))

        group_0 = list(range(0, NUM_DESKS, 4))
        group_1 = list(range(1, NUM_DESKS, 4))
        group_2 = list(range(2, NUM_DESKS, 4))
        group_3 = list(range(3, NUM_DESKS, 4))

        def emit_hash_stages_all_desks():
            for d in range(NUM_DESKS):
                self.emit("valu", ("multiply_add", desks[d]['val'], desks[d]['val'], v_fma_mult[0], v_hash_consts[0]))

            for group in [group_0, group_1, group_2, group_3]:
                for d in group:
                    tmp1, tmp2 = get_temps(d)
                    self.emit("valu", ("^", tmp1, desks[d]['val'], v_hash_consts[1]))
                for d in group:
                    tmp1, tmp2 = get_temps(d)
                    self.emit("valu", (">>", tmp2, desks[d]['val'], v_hash_shifts[1]))
                for d in group:
                    tmp1, tmp2 = get_temps(d)
                    self.emit("valu", ("^", desks[d]['val'], tmp1, tmp2))

            for d in range(NUM_DESKS):
                self.emit("valu", ("multiply_add", desks[d]['val'], desks[d]['val'], v_fma_mult[2], v_hash_consts[2]))

            for group in [group_0, group_1, group_2, group_3]:
                for d in group:
                    tmp1, tmp2 = get_temps(d)
                    self.emit("valu", ("+", tmp1, desks[d]['val'], v_hash_consts[3]))
                for d in group:
                    tmp1, tmp2 = get_temps(d)
                    self.emit("valu", ("<<", tmp2, desks[d]['val'], v_hash_shifts[3]))
                for d in group:
                    tmp1, tmp2 = get_temps(d)
                    self.emit("valu", ("^", desks[d]['val'], tmp1, tmp2))

            for d in range(NUM_DESKS):
                self.emit("valu", ("multiply_add", desks[d]['val'], desks[d]['val'], v_fma_mult[4], v_hash_consts[4]))

            for group in [group_0, group_1, group_2, group_3]:
                for d in group:
                    tmp1, tmp2 = get_temps(d)
                    self.emit("valu", ("^", tmp1, desks[d]['val'], v_hash_consts[5]))
                for d in group:
                    tmp1, tmp2 = get_temps(d)
                    self.emit("valu", (">>", tmp2, desks[d]['val'], v_hash_shifts[5]))
                for d in group:
                    tmp1, tmp2 = get_temps(d)
                    self.emit("valu", ("^", desks[d]['val'], tmp1, tmp2))

        def emit_hash_stages(desk_idx):
            d = desks[desk_idx]
            tmp1, tmp2 = get_temps(desk_idx)
            for hi in range(6):
                if hi in v_fma_mult:
                    self.emit("valu", ("multiply_add", d['val'], d['val'], v_fma_mult[hi], v_hash_consts[hi]))
                elif hi == 1:
                    self.emit("valu", ("^", tmp1, d['val'], v_hash_consts[hi]))
                    self.emit("valu", (">>", tmp2, d['val'], v_hash_shifts[hi]))
                    self.emit("valu", ("^", d['val'], tmp1, tmp2))
                elif hi == 3:
                    self.emit("valu", ("+", tmp1, d['val'], v_hash_consts[hi]))
                    self.emit("valu", ("<<", tmp2, d['val'], v_hash_shifts[hi]))
                    self.emit("valu", ("^", d['val'], tmp1, tmp2))
                elif hi == 5:
                    self.emit("valu", ("^", tmp1, d['val'], v_hash_consts[hi]))
                    self.emit("valu", (">>", tmp2, d['val'], v_hash_shifts[hi]))
                    self.emit("valu", ("^", d['val'], tmp1, tmp2))

        def emit_branch(desk_idx):
            """Standard branch computation (for non-gather rounds)."""
            d = desks[desk_idx]
            tmp1, _ = get_temps(desk_idx)
            self.emit("valu", ("&", tmp1, d['val'], v_one))
            self.emit("valu", ("multiply_add", d['idx'], d['idx'], v_two, v_one))
            self.emit("valu", ("+", d['idx'], d['idx'], tmp1))

        def emit_branch_fma_addr(desk_idx):
            """
            H136 v4: Address-centric branch computation.

            New approach (3 ops total, same as before):
              tmp = val & 1
              addr = FMA(idx, 2, forest_p_plus_1) = 2*idx + forest_p + 1
              addr = addr + tmp

            Then recover idx = addr - forest_p when needed.

            Wait, this is still 3 ops. The "savings" would only come if we
            could eliminate the idx recovery step. Let's check when idx is actually used:
            - For gather rounds without bounds: idx is only needed for the next branch
            - For gather round with bounds (R10): idx is needed for bounds check
            - For arithmetic rounds: idx is used for tree selection
            - At the end: idx is stored back to memory

            So for gather rounds 3-9, we could keep just addr and recover idx
            only when transitioning to bounds/arithmetic.

            Actually, the issue is that the NEXT round's FMA needs idx:
              addr_next = FMA(idx_curr, 2, forest_p_plus_1) + bit

            So we can't avoid maintaining idx.

            Let's try a different approach: can we do branch + addr in 3 total ops
            instead of 4 (3 branch + 1 addr)?

            Current (4 ops):
              tmp = val & 1          # 1
              idx = FMA(idx, 2, 1)   # 2
              idx = idx + tmp        # 3
              addr = forest_p + idx  # 4

            Can we combine ops 2-4?
              tmp = val & 1
              addr = FMA(idx, 2, forest_p + 1 + tmp)  # Hmm, can't add tmp dynamically

            What about:
              tmp = val & 1
              addr = FMA(idx, 2, forest_p_plus_1)  # 2*idx + forest_p + 1
              addr = addr + tmp  # Final addr
              idx = addr - forest_p  # Recover idx

            That's still 4 ops. The issue is we need idx for the next iteration.

            WAIT - what if we store addr instead of idx, and recover idx at load/store time?
            - We need idx for bounds check: can compare addr < bounds_addr instead
            - We need idx for arithmetic selection: need to recover idx
            - We need idx for final store: need to recover idx

            For gather rounds without arithmetic, we could skip idx entirely:
              tmp = val & 1
              addr = FMA(idx, 2, forest_p_plus_1) + tmp  # This IS the new addr
              (Next gather uses addr directly)

            But we still need idx for the FMA. Unless we work with addr throughout:
              new_addr = 2*(old_addr - forest_p) + forest_p + 1 + bit
                       = 2*old_addr - 2*forest_p + forest_p + 1 + bit
                       = 2*old_addr - forest_p + 1 + bit

            Hmm, this is getting complicated. Let me just try the straightforward
            version and see if eliminating the separate addr computation helps.
            """
            d = desks[desk_idx]
            tmp1, _ = get_temps(desk_idx)
            # Get branch bit
            self.emit("valu", ("&", tmp1, d['val'], v_one))
            # addr = FMA(idx, 2, forest_p_plus_1) = 2*idx + forest_p + 1
            self.emit("valu", ("multiply_add", d['addr'], d['idx'], v_two, v_forest_p_plus_1))
            # addr = addr + bit
            self.emit("valu", ("+", d['addr'], d['addr'], tmp1))
            # Recover idx = addr - forest_p for next round's FMA
            self.emit("valu", ("-", d['idx'], d['addr'], v_forest_p))

        def emit_bounds_on_addr(desk_idx):
            """
            H136 v4: Bounds check using addr instead of idx.
            addr < bounds_addr  equivalent to  idx < n_nodes
            """
            d = desks[desk_idx]
            tmp1, _ = get_temps(desk_idx)
            # Check if addr < bounds_addr (= forest_p + n_nodes)
            self.emit("valu", ("<", tmp1, d['addr'], v_bounds_addr))
            # If out of bounds, reset addr to forest_p (idx = 0)
            self.emit("valu", ("*", d['addr'], d['addr'], tmp1))
            # Also reset idx = addr - forest_p = 0 when out of bounds
            # But if addr = 0 when out of bounds, idx = 0 - forest_p which is wrong!
            # We need: if out of bounds, addr = forest_p, idx = 0

            # Actually, the original code does: idx = idx * (idx < n_nodes)
            # So when out of bounds, idx becomes 0.
            # We need: addr = forest_p when idx = 0

            # Let me reconsider... Original:
            #   idx = idx * (idx < n_nodes)  // idx = 0 when out of bounds

            # With addr-centric:
            #   addr = addr * (addr < bounds_addr)???  No, this makes addr = 0

            # The issue is that when idx wraps to 0, addr should become forest_p, not 0.

            # Let's use select:
            #   in_bounds = addr < bounds_addr
            #   addr = in_bounds ? addr : forest_p
            #   idx = in_bounds ? idx : 0

            # But we don't have vselect implemented efficiently...

            # Actually, let's just stick with the original approach for bounds:
            #   idx = idx * (idx < n_nodes)
            #   addr = forest_p + idx  (recomputed)

        def emit_bounds(desk_idx):
            """Standard bounds check."""
            d = desks[desk_idx]
            tmp1, _ = get_temps(desk_idx)
            self.emit("valu", ("<", tmp1, d['idx'], v_n_nodes))
            self.emit("valu", ("*", d['idx'], d['idx'], tmp1))

        def emit_tile():
            for d in range(NUM_DESKS):
                self.emit("load", ("const", offset_regs[d], d * VLEN))

            for d_batch in range(0, NUM_DESKS, 2):
                d0, d1 = d_batch, d_batch + 1
                self.emit("alu", ("+", addr_tmp[0], self.scratch["inp_indices_p"], offset_regs[d0]))
                self.emit("alu", ("+", addr_tmp[1], self.scratch["inp_values_p"], offset_regs[d0]))
                self.emit("alu", ("+", addr_tmp[2], self.scratch["inp_indices_p"], offset_regs[d1]))
                self.emit("alu", ("+", addr_tmp[3], self.scratch["inp_values_p"], offset_regs[d1]))
                self.emit("load", ("vload", desks[d0]['idx'], addr_tmp[0]))
                self.emit("load", ("vload", desks[d0]['val'], addr_tmp[1]))
                self.emit("load", ("vload", desks[d1]['idx'], addr_tmp[2]))
                self.emit("load", ("vload", desks[d1]['val'], addr_tmp[3]))

            # Round 0
            for d in range(NUM_DESKS):
                self.emit("valu", ("^", desks[d]['val'], desks[d]['val'], v_tree[0]))
            emit_hash_stages_all_desks()
            for d in range(NUM_DESKS):
                emit_branch(d)

            # Round 1
            for group in [group_0, group_1, group_2, group_3]:
                for d in group:
                    tmp1, _ = get_temps(d)
                    self.emit("valu", ("-", tmp1, desks[d]['idx'], v_one))
                for d in group:
                    tmp1, _ = get_temps(d)
                    self.emit("valu", ("multiply_add", desks[d]['node_val'], tmp1, v_diff_1_2, v_tree[1]))
            for d in range(NUM_DESKS):
                self.emit("valu", ("^", desks[d]['val'], desks[d]['val'], desks[d]['node_val']))
            emit_hash_stages_all_desks()
            for d in range(NUM_DESKS):
                emit_branch(d)

            # Round 2
            for d in range(NUM_DESKS):
                tmp1, tmp2 = get_temps(d)
                self.emit("valu", ("-", tmp1, desks[d]['idx'], v_three))
                self.emit("valu", ("&", tmp2, tmp1, v_one))
                self.emit("valu", (">>", desks[d]['addr'], tmp1, v_one))
                self.emit("valu", ("multiply_add", desks[d]['node_val'], tmp2, v_diff_3_4, v_tree[3]))
                self.emit("valu", ("multiply_add", tmp1, tmp2, v_diff_5_6, v_tree[5]))
                self.emit("valu", ("-", tmp2, tmp1, desks[d]['node_val']))
                self.emit("valu", ("multiply_add", desks[d]['node_val'], desks[d]['addr'], tmp2, desks[d]['node_val']))
                self.emit("valu", ("^", desks[d]['val'], desks[d]['val'], desks[d]['node_val']))
                emit_hash_stages(d)
                emit_branch(d)

            # After Round 2, compute first gather addresses
            for d in range(NUM_DESKS):
                self.emit("valu", ("+", desks[d]['addr'], v_forest_p, desks[d]['idx']))

            # Rounds 3-9: Gather with FMA-based branch
            for _round in range(3, 10):
                # Gather
                for d in range(NUM_DESKS):
                    for lane in range(VLEN):
                        self.emit("load", ("load", desks[d]['node_val'] + lane, desks[d]['addr'] + lane))
                # XOR
                for d in range(NUM_DESKS):
                    self.emit("valu", ("^", desks[d]['val'], desks[d]['val'], desks[d]['node_val']))
                # Hash
                emit_hash_stages_all_desks()
                # H136 v4: FMA-based branch that computes addr directly
                for d in range(NUM_DESKS):
                    emit_branch_fma_addr(d)

            # Round 10 (with bounds)
            for d in range(NUM_DESKS):
                for lane in range(VLEN):
                    self.emit("load", ("load", desks[d]['node_val'] + lane, desks[d]['addr'] + lane))
            for d in range(NUM_DESKS):
                self.emit("valu", ("^", desks[d]['val'], desks[d]['val'], desks[d]['node_val']))
            emit_hash_stages_all_desks()
            # Use standard branch for round with bounds
            for d in range(NUM_DESKS):
                emit_branch(d)
            for d in range(NUM_DESKS):
                emit_bounds(d)

            # Round 11
            for d in range(NUM_DESKS):
                self.emit("valu", ("^", desks[d]['val'], desks[d]['val'], v_tree[0]))
            emit_hash_stages_all_desks()
            for d in range(NUM_DESKS):
                emit_branch(d)

            # Round 12
            for group in [group_0, group_1, group_2, group_3]:
                for d in group:
                    tmp1, _ = get_temps(d)
                    self.emit("valu", ("-", tmp1, desks[d]['idx'], v_one))
                for d in group:
                    tmp1, _ = get_temps(d)
                    self.emit("valu", ("multiply_add", desks[d]['node_val'], tmp1, v_diff_1_2, v_tree[1]))
            for d in range(NUM_DESKS):
                self.emit("valu", ("^", desks[d]['val'], desks[d]['val'], desks[d]['node_val']))
            emit_hash_stages_all_desks()
            for d in range(NUM_DESKS):
                emit_branch(d)

            # Round 13
            for d in range(NUM_DESKS):
                tmp1, tmp2 = get_temps(d)
                self.emit("valu", ("-", tmp1, desks[d]['idx'], v_three))
                self.emit("valu", ("&", tmp2, tmp1, v_one))
                self.emit("valu", (">>", desks[d]['addr'], tmp1, v_one))
                self.emit("valu", ("multiply_add", desks[d]['node_val'], tmp2, v_diff_3_4, v_tree[3]))
                self.emit("valu", ("multiply_add", tmp1, tmp2, v_diff_5_6, v_tree[5]))
                self.emit("valu", ("-", tmp2, tmp1, desks[d]['node_val']))
                self.emit("valu", ("multiply_add", desks[d]['node_val'], desks[d]['addr'], tmp2, desks[d]['node_val']))
                self.emit("valu", ("^", desks[d]['val'], desks[d]['val'], desks[d]['node_val']))
                emit_hash_stages(d)
                emit_branch(d)

            # After Round 13, compute gather addresses
            for d in range(NUM_DESKS):
                self.emit("valu", ("+", desks[d]['addr'], v_forest_p, desks[d]['idx']))

            # Rounds 14-15 with FMA-based branch
            for _round in range(14, 16):
                for d in range(NUM_DESKS):
                    for lane in range(VLEN):
                        self.emit("load", ("load", desks[d]['node_val'] + lane, desks[d]['addr'] + lane))
                for d in range(NUM_DESKS):
                    self.emit("valu", ("^", desks[d]['val'], desks[d]['val'], desks[d]['node_val']))
                emit_hash_stages_all_desks()
                if _round < 15:
                    for d in range(NUM_DESKS):
                        emit_branch_fma_addr(d)
                else:
                    # Last round - just update idx, don't need addr
                    for d in range(NUM_DESKS):
                        emit_branch(d)

            # Store results
            for d_batch in range(0, NUM_DESKS, 2):
                d0, d1 = d_batch, d_batch + 1
                self.emit("alu", ("+", addr_tmp[0], self.scratch["inp_indices_p"], offset_regs[d0]))
                self.emit("alu", ("+", addr_tmp[1], self.scratch["inp_values_p"], offset_regs[d0]))
                self.emit("alu", ("+", addr_tmp[2], self.scratch["inp_indices_p"], offset_regs[d1]))
                self.emit("alu", ("+", addr_tmp[3], self.scratch["inp_values_p"], offset_regs[d1]))
                self.emit("store", ("vstore", addr_tmp[0], desks[d0]['idx']))
                self.emit("store", ("vstore", addr_tmp[1], desks[d0]['val']))
                self.emit("store", ("vstore", addr_tmp[2], desks[d1]['idx']))
                self.emit("store", ("vstore", addr_tmp[3], desks[d1]['val']))

        emit_tile()

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


def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
    check: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilderH136v4()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if check:
            assert (
                machine.mem[inp_values_p : inp_values_p + len(inp.values)]
                == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
            ), f"Incorrect result on round {i}"

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
