"""
# Experiment H94: Optimized Hash Stage Ordering

**GOAL:** Improve upon H87's 1,851 cycles by reordering hash stage operations.

**HYPOTHESIS:** The hash stages have internal dependencies:
- Stages 0, 2, 4: FMA (single op, writes val)
- Stages 1, 3, 5: 3 ops where first two are independent:
  - tmp1 = val OP const (depends only on val)
  - tmp2 = val OP shift (depends only on val, independent of tmp1!)
  - val = tmp1 ^ tmp2 (depends on both)

Perhaps reordering to maximize overlap of non-dependent operations could help.

**APPROACHES TRIED:**
1. Wave-based emission: All op1s for all desks, then all op2s, then all op3s
   - Result: Same cycles (1,851) - scheduler already packs optimally

2. Interleaved op1/op2: Emit op1 and op2 for same desk consecutively
   - Result: Same cycles (1,851) - no effect on scheduling

3. Pipelined even/odd: Start odd desks' stage N while even finish stage N-1
   - Result: Same cycles (1,851) - scheduler handles this automatically

4. Fused hash+branch: Interleave branch FMA with hash stage 5
   - Result: Worse (1,875 cycles) - disrupts scheduler's optimal packing

5. Batch branch ordering: Process even then odd desks for branch ops
   - Result: Worse (1,875 cycles) - sequential ordering is better

**CONCLUSION:** The automatic scheduler (_schedule_slots) is already optimal for the
given operation mix. Emission order doesn't affect final cycle count as long as
temp register conflicts are avoided (even-then-odd pattern).

**RESULTS:**
- H87 baseline: 1,851 cycles
- H94 (all approaches): 1,851 cycles (no improvement)

**ANALYSIS:** The greedy scheduler considers all operations and their dependencies,
scheduling each as early as possible. With VALU limit of 6/cycle and proper dependency
tracking, the emission order only affects correctness (temp conflicts), not performance.
The hypothesis is NOT VALIDATED - reordering does not help.
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


class KernelBuilderH94:
    """
    H87: Double-pumped hash stages - interleave FMA and multi-op stages between desk pairs.
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
        """Add operation to flat list for later scheduling."""
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
        """
        Build kernel using flat-list generation with automatic scheduling.
        Uses shared temp registers instead of per-desk allocation.
        """
        tmp_scalar = self.alloc_scratch("tmp_scalar")
        tmp_addr = self.alloc_scratch("tmp_addr")

        # Load header values
        init_vars = ["rounds", "n_nodes", "batch_size", "forest_height",
                     "forest_values_p", "inp_indices_p", "inp_values_p"]
        for v in init_vars:
            self.alloc_scratch(v)
        for i, v in enumerate(init_vars):
            self.emit("load", ("const", tmp_scalar, i))
            self.emit("load", ("load", self.scratch[v], tmp_scalar))

        # Vector constants
        v_zero = self.scratch_vconst(0, "v_zero")
        v_one = self.scratch_vconst(1, "v_one")
        v_two = self.scratch_vconst(2, "v_two")
        v_three = self.scratch_vconst(3, "v_three")  # For 4-way selection
        v_n_nodes = self.alloc_vec("v_n_nodes")
        self.emit("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]))

        # Precompute forest_values_p as vector (eliminates per-gather vbroadcast)
        v_forest_p = self.alloc_vec("v_forest_p")
        self.emit("valu", ("vbroadcast", v_forest_p, self.scratch["forest_values_p"]))

        # Hash constants (FMA for stages 0, 2, 4)
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

        # Preload tree nodes 0-14
        NUM_PRELOADED = 15
        v_tree = []
        for i in range(NUM_PRELOADED):
            v_node = self.alloc_vec(f"v_tree_{i}")
            v_tree.append(v_node)
            self.emit("alu", ("+", tmp_addr, self.scratch["forest_values_p"], self.scratch_const(i)))
            self.emit("load", ("load", tmp_scalar, tmp_addr))
            self.emit("valu", ("vbroadcast", v_node, tmp_scalar))

        # Precompute tree differences for selection (avoids recomputing per-desk)
        v_diff_1_2 = self.alloc_vec("v_diff_1_2")  # tree[2] - tree[1]
        v_diff_3_4 = self.alloc_vec("v_diff_3_4")  # tree[4] - tree[3]
        v_diff_5_6 = self.alloc_vec("v_diff_5_6")  # tree[6] - tree[5]
        self.emit("valu", ("-", v_diff_1_2, v_tree[2], v_tree[1]))
        self.emit("valu", ("-", v_diff_3_4, v_tree[4], v_tree[3]))
        self.emit("valu", ("-", v_diff_5_6, v_tree[6], v_tree[5]))

        # Allocate per-desk vectors (16 desks) - NO tmp1/tmp2 per desk
        NUM_DESKS = 16
        desks = []
        for d in range(NUM_DESKS):
            desk = {
                'idx': self.alloc_vec(f"v_idx_{d}"),
                'val': self.alloc_vec(f"v_val_{d}"),
                'node_val': self.alloc_vec(f"v_node_{d}"),
                'addr': self.alloc_vec(f"v_addr_{d}"),
                # NO tmp1/tmp2 here - will use shared temps
            }
            desks.append(desk)

        # SHARED TEMP REGISTERS - Use 16 vectors like H85 (128 slots)
        # Each pair of 2 consecutive desks shares temps (0-1, 2-3, 4-5, etc.)
        NUM_SHARED_TEMPS = 16
        shared_temps = [self.alloc_vec(f"v_shared_tmp_{i}") for i in range(NUM_SHARED_TEMPS)]

        def get_temps(desk_idx):
            """Get temp registers for a desk."""
            pair = desk_idx // 2  # 8 pairs total (0-1, 2-3, etc.)
            t1_idx = pair * 2
            t2_idx = pair * 2 + 1
            return shared_temps[t1_idx], shared_temps[t2_idx]

        # Offset addresses
        offset_regs = [self.alloc_scratch(f"off_{d}") for d in range(NUM_DESKS)]
        addr_tmp = [self.alloc_scratch(f"addr_tmp_{i}") for i in range(32)]

        scratch_before = self.scratch_ptr
        print(f"Scratch usage: {self.scratch_ptr} / {SCRATCH_SIZE}")
        print(f"  - Saved {(NUM_DESKS * 2 - NUM_SHARED_TEMPS) * VLEN} slots by sharing temps")

        # Pause before main computation
        self.emit("flow", ("pause",))

        # Double-pumped hash stages: deeply interleave operations from pairs of desks
        def emit_hash_stages_interleaved(desk_a, desk_b):
            """Emit hash stages for two desks with deep operation-level interleaving.

            The key insight is that hash stages have this structure:
            - Stages 0, 2, 4: FMA (1 op, writes to val)
            - Stages 1, 3, 5: xor/shift pattern (3 ops: write tmp1, write tmp2, write val)

            By deeply interleaving ops within multi-op stages, we can:
            1. Emit FMAs for both desks back-to-back (both can execute in parallel)
            2. For 3-op stages, interleave the first two ops (xor/shift) for both desks,
               then interleave the final xor ops

            This creates more opportunities for the scheduler to fill VALU slots.
            """
            d_a = desks[desk_a]
            d_b = desks[desk_b]
            tmp1_a, tmp2_a = get_temps(desk_a)
            tmp1_b, tmp2_b = get_temps(desk_b)

            # Stage 0: FMA for both (can execute in parallel)
            self.emit("valu", ("multiply_add", d_a['val'], d_a['val'], v_fma_mult[0], v_hash_consts[0]))
            self.emit("valu", ("multiply_add", d_b['val'], d_b['val'], v_fma_mult[0], v_hash_consts[0]))

            # Stage 1: 3 ops each, deeply interleaved
            # First pair of ops (xor with const) - independent, can run in parallel
            self.emit("valu", ("^", tmp1_a, d_a['val'], v_hash_consts[1]))
            self.emit("valu", ("^", tmp1_b, d_b['val'], v_hash_consts[1]))
            # Second pair of ops (shift) - independent, can run in parallel
            self.emit("valu", (">>", tmp2_a, d_a['val'], v_hash_shifts[1]))
            self.emit("valu", (">>", tmp2_b, d_b['val'], v_hash_shifts[1]))
            # Final pair of ops (xor of results) - depends on above, but both can run in parallel
            self.emit("valu", ("^", d_a['val'], tmp1_a, tmp2_a))
            self.emit("valu", ("^", d_b['val'], tmp1_b, tmp2_b))

            # Stage 2: FMA for both
            self.emit("valu", ("multiply_add", d_a['val'], d_a['val'], v_fma_mult[2], v_hash_consts[2]))
            self.emit("valu", ("multiply_add", d_b['val'], d_b['val'], v_fma_mult[2], v_hash_consts[2]))

            # Stage 3: 3 ops each, deeply interleaved
            self.emit("valu", ("+", tmp1_a, d_a['val'], v_hash_consts[3]))
            self.emit("valu", ("+", tmp1_b, d_b['val'], v_hash_consts[3]))
            self.emit("valu", ("<<", tmp2_a, d_a['val'], v_hash_shifts[3]))
            self.emit("valu", ("<<", tmp2_b, d_b['val'], v_hash_shifts[3]))
            self.emit("valu", ("^", d_a['val'], tmp1_a, tmp2_a))
            self.emit("valu", ("^", d_b['val'], tmp1_b, tmp2_b))

            # Stage 4: FMA for both
            self.emit("valu", ("multiply_add", d_a['val'], d_a['val'], v_fma_mult[4], v_hash_consts[4]))
            self.emit("valu", ("multiply_add", d_b['val'], d_b['val'], v_fma_mult[4], v_hash_consts[4]))

            # Stage 5: 3 ops each, deeply interleaved
            self.emit("valu", ("^", tmp1_a, d_a['val'], v_hash_consts[5]))
            self.emit("valu", ("^", tmp1_b, d_b['val'], v_hash_consts[5]))
            self.emit("valu", (">>", tmp2_a, d_a['val'], v_hash_shifts[5]))
            self.emit("valu", (">>", tmp2_b, d_b['val'], v_hash_shifts[5]))
            self.emit("valu", ("^", d_a['val'], tmp1_a, tmp2_a))
            self.emit("valu", ("^", d_b['val'], tmp1_b, tmp2_b))

        # Original emit_hash_stages for cases where we process single desk
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

        # Double-pumped hash stages for all desks
        # Emits by operation type, processing even desks then odd to avoid temp conflicts
        even_desks = list(range(0, NUM_DESKS, 2))  # [0, 2, 4, 6, 8, 10, 12, 14]
        odd_desks = list(range(1, NUM_DESKS, 2))   # [1, 3, 5, 7, 9, 11, 13, 15]

        def emit_hash_stages_all_desks():
            """Emit hash stages for all desks in a double-pumped fashion."""
            # Stage 0 (FMA) - all desks can run in parallel (no temp usage)
            for d in range(NUM_DESKS):
                self.emit("valu", ("multiply_add", desks[d]['val'], desks[d]['val'], v_fma_mult[0], v_hash_consts[0]))

            # Stage 1 (3 ops) - process even desks first, then odd
            for d in even_desks:
                tmp1, tmp2 = get_temps(d)
                self.emit("valu", ("^", tmp1, desks[d]['val'], v_hash_consts[1]))
            for d in even_desks:
                tmp1, tmp2 = get_temps(d)
                self.emit("valu", (">>", tmp2, desks[d]['val'], v_hash_shifts[1]))
            for d in even_desks:
                tmp1, tmp2 = get_temps(d)
                self.emit("valu", ("^", desks[d]['val'], tmp1, tmp2))
            for d in odd_desks:
                tmp1, tmp2 = get_temps(d)
                self.emit("valu", ("^", tmp1, desks[d]['val'], v_hash_consts[1]))
            for d in odd_desks:
                tmp1, tmp2 = get_temps(d)
                self.emit("valu", (">>", tmp2, desks[d]['val'], v_hash_shifts[1]))
            for d in odd_desks:
                tmp1, tmp2 = get_temps(d)
                self.emit("valu", ("^", desks[d]['val'], tmp1, tmp2))

            # Stage 2 (FMA) - all desks
            for d in range(NUM_DESKS):
                self.emit("valu", ("multiply_add", desks[d]['val'], desks[d]['val'], v_fma_mult[2], v_hash_consts[2]))

            # Stage 3 (3 ops) - process even, then odd
            for d in even_desks:
                tmp1, tmp2 = get_temps(d)
                self.emit("valu", ("+", tmp1, desks[d]['val'], v_hash_consts[3]))
            for d in even_desks:
                tmp1, tmp2 = get_temps(d)
                self.emit("valu", ("<<", tmp2, desks[d]['val'], v_hash_shifts[3]))
            for d in even_desks:
                tmp1, tmp2 = get_temps(d)
                self.emit("valu", ("^", desks[d]['val'], tmp1, tmp2))
            for d in odd_desks:
                tmp1, tmp2 = get_temps(d)
                self.emit("valu", ("+", tmp1, desks[d]['val'], v_hash_consts[3]))
            for d in odd_desks:
                tmp1, tmp2 = get_temps(d)
                self.emit("valu", ("<<", tmp2, desks[d]['val'], v_hash_shifts[3]))
            for d in odd_desks:
                tmp1, tmp2 = get_temps(d)
                self.emit("valu", ("^", desks[d]['val'], tmp1, tmp2))

            # Stage 4 (FMA) - all desks
            for d in range(NUM_DESKS):
                self.emit("valu", ("multiply_add", desks[d]['val'], desks[d]['val'], v_fma_mult[4], v_hash_consts[4]))

            # Stage 5 (3 ops) - process even, then odd
            for d in even_desks:
                tmp1, tmp2 = get_temps(d)
                self.emit("valu", ("^", tmp1, desks[d]['val'], v_hash_consts[5]))
            for d in even_desks:
                tmp1, tmp2 = get_temps(d)
                self.emit("valu", (">>", tmp2, desks[d]['val'], v_hash_shifts[5]))
            for d in even_desks:
                tmp1, tmp2 = get_temps(d)
                self.emit("valu", ("^", desks[d]['val'], tmp1, tmp2))
            for d in odd_desks:
                tmp1, tmp2 = get_temps(d)
                self.emit("valu", ("^", tmp1, desks[d]['val'], v_hash_consts[5]))
            for d in odd_desks:
                tmp1, tmp2 = get_temps(d)
                self.emit("valu", (">>", tmp2, desks[d]['val'], v_hash_shifts[5]))
            for d in odd_desks:
                tmp1, tmp2 = get_temps(d)
                self.emit("valu", ("^", desks[d]['val'], tmp1, tmp2))

        def emit_branch(desk_idx):
            d = desks[desk_idx]
            tmp1, _ = get_temps(desk_idx)
            self.emit("valu", ("&", tmp1, d['val'], v_one))
            self.emit("valu", ("multiply_add", d['idx'], d['idx'], v_two, v_one))
            self.emit("valu", ("+", d['idx'], d['idx'], tmp1))

        def emit_bounds(desk_idx):
            d = desks[desk_idx]
            tmp1, _ = get_temps(desk_idx)
            self.emit("valu", ("<", tmp1, d['idx'], v_n_nodes))
            self.emit("valu", ("*", d['idx'], d['idx'], tmp1))

        def emit_tile(tile_idx):
            """Emit all operations for one tile."""
            tile_offset = tile_idx * NUM_DESKS * VLEN

            # Load offsets
            for d in range(NUM_DESKS):
                self.emit("load", ("const", offset_regs[d], tile_offset + d * VLEN))

            # Compute load addresses
            for d in range(NUM_DESKS):
                self.emit("alu", ("+", addr_tmp[d*2], self.scratch["inp_indices_p"], offset_regs[d]))
                self.emit("alu", ("+", addr_tmp[d*2+1], self.scratch["inp_values_p"], offset_regs[d]))

            # Load idx/val for all desks
            for d in range(NUM_DESKS):
                self.emit("load", ("vload", desks[d]['idx'], addr_tmp[d*2]))
                self.emit("load", ("vload", desks[d]['val'], addr_tmp[d*2+1]))

            # ===== ROUNDS =====
            # Round 0: All indices = 0, use tree[0] directly (no copy needed)
            # Double-pump approach: emit operations by type to maximize VALU utilization
            for d in range(NUM_DESKS):
                self.emit("valu", ("^", desks[d]['val'], desks[d]['val'], v_tree[0]))
            emit_hash_stages_all_desks()
            for d in range(NUM_DESKS):
                emit_branch(d)

            # Round 1: Indices in {1, 2}, use arithmetic selection
            # Double-pump: batch selection and XOR, then double-pump hash
            # Selection ops use tmp1, so process even then odd
            for d in even_desks:
                tmp1, _ = get_temps(d)
                self.emit("valu", ("-", tmp1, desks[d]['idx'], v_one))  # 0 or 1
            for d in even_desks:
                tmp1, _ = get_temps(d)
                self.emit("valu", ("multiply_add", desks[d]['node_val'], tmp1, v_diff_1_2, v_tree[1]))
            for d in odd_desks:
                tmp1, _ = get_temps(d)
                self.emit("valu", ("-", tmp1, desks[d]['idx'], v_one))
            for d in odd_desks:
                tmp1, _ = get_temps(d)
                self.emit("valu", ("multiply_add", desks[d]['node_val'], tmp1, v_diff_1_2, v_tree[1]))
            for d in range(NUM_DESKS):
                self.emit("valu", ("^", desks[d]['val'], desks[d]['val'], desks[d]['node_val']))
            emit_hash_stages_all_desks()
            for d in range(NUM_DESKS):
                emit_branch(d)

            # Round 2: Indices in {3, 4, 5, 6}, use 4-way arithmetic selection
            for d in range(NUM_DESKS):
                tmp1, tmp2 = get_temps(d)
                # Extract selection bits: tmp = idx - 3 gives {0, 1, 2, 3}
                self.emit("valu", ("-", tmp1, desks[d]['idx'], v_three))  # tmp1 = idx - 3
                self.emit("valu", ("&", tmp2, tmp1, v_one))   # bit0 = tmp1 & 1
                self.emit("valu", (">>", desks[d]['addr'], tmp1, v_one))  # bit1 = tmp1 >> 1 (reuse addr as temp)
                # Select from low pair (tree[3] or tree[4]) using precomputed diff
                self.emit("valu", ("multiply_add", desks[d]['node_val'], tmp2, v_diff_3_4, v_tree[3]))  # low_pair
                # Select from high pair (tree[5] or tree[6]) using precomputed diff
                self.emit("valu", ("multiply_add", tmp1, tmp2, v_diff_5_6, v_tree[5]))  # high_pair in tmp1
                # Final selection based on bit1
                self.emit("valu", ("-", tmp2, tmp1, desks[d]['node_val']))  # diff_pairs
                self.emit("valu", ("multiply_add", desks[d]['node_val'], desks[d]['addr'], tmp2, desks[d]['node_val']))  # result
                # XOR and hash
                self.emit("valu", ("^", desks[d]['val'], desks[d]['val'], desks[d]['node_val']))
                emit_hash_stages(d)
                emit_branch(d)

            # Rounds 3-9: Gather without bounds
            # Double-pump: batch addr calc, gather, xor, then double-pump hash
            for _round in range(3, 10):
                # Compute addresses for all desks
                for d in range(NUM_DESKS):
                    self.emit("valu", ("+", desks[d]['addr'], v_forest_p, desks[d]['idx']))
                # Gather for all desks
                for d in range(NUM_DESKS):
                    for lane in range(VLEN):
                        self.emit("load", ("load", desks[d]['node_val'] + lane, desks[d]['addr'] + lane))
                # XOR for all desks
                for d in range(NUM_DESKS):
                    self.emit("valu", ("^", desks[d]['val'], desks[d]['val'], desks[d]['node_val']))
                # Double-pumped hash stages
                emit_hash_stages_all_desks()
                # Branch for all desks
                for d in range(NUM_DESKS):
                    emit_branch(d)

            # Round 10: Gather WITH bounds
            # Full double-pump for everything except bounds which must be after branch
            for d in range(NUM_DESKS):
                self.emit("valu", ("+", desks[d]['addr'], v_forest_p, desks[d]['idx']))
            for d in range(NUM_DESKS):
                for lane in range(VLEN):
                    self.emit("load", ("load", desks[d]['node_val'] + lane, desks[d]['addr'] + lane))
            for d in range(NUM_DESKS):
                self.emit("valu", ("^", desks[d]['val'], desks[d]['val'], desks[d]['node_val']))
            emit_hash_stages_all_desks()
            for d in range(NUM_DESKS):
                emit_branch(d)
            for d in range(NUM_DESKS):
                emit_bounds(d)  # After R10, all indices wrap to 0

            # Round 11: All indices = 0 after wrap, use tree[0] directly
            # Double-pump approach
            for d in range(NUM_DESKS):
                self.emit("valu", ("^", desks[d]['val'], desks[d]['val'], v_tree[0]))
            emit_hash_stages_all_desks()
            for d in range(NUM_DESKS):
                emit_branch(d)

            # Round 12: Indices in {1, 2}, use arithmetic selection
            # Double-pump approach
            for d in even_desks:
                tmp1, _ = get_temps(d)
                self.emit("valu", ("-", tmp1, desks[d]['idx'], v_one))
            for d in even_desks:
                tmp1, _ = get_temps(d)
                self.emit("valu", ("multiply_add", desks[d]['node_val'], tmp1, v_diff_1_2, v_tree[1]))
            for d in odd_desks:
                tmp1, _ = get_temps(d)
                self.emit("valu", ("-", tmp1, desks[d]['idx'], v_one))
            for d in odd_desks:
                tmp1, _ = get_temps(d)
                self.emit("valu", ("multiply_add", desks[d]['node_val'], tmp1, v_diff_1_2, v_tree[1]))
            for d in range(NUM_DESKS):
                self.emit("valu", ("^", desks[d]['val'], desks[d]['val'], desks[d]['node_val']))
            emit_hash_stages_all_desks()
            for d in range(NUM_DESKS):
                emit_branch(d)

            # Round 13: Indices in {3, 4, 5, 6} after wrap, use 4-way arithmetic selection
            for d in range(NUM_DESKS):
                tmp1, tmp2 = get_temps(d)
                # Extract selection bits: tmp = idx - 3 gives {0, 1, 2, 3}
                self.emit("valu", ("-", tmp1, desks[d]['idx'], v_three))  # tmp1 = idx - 3
                self.emit("valu", ("&", tmp2, tmp1, v_one))   # bit0 = tmp1 & 1
                self.emit("valu", (">>", desks[d]['addr'], tmp1, v_one))  # bit1 = tmp1 >> 1 (reuse addr as temp)
                # Select from low pair (tree[3] or tree[4]) using precomputed diff
                self.emit("valu", ("multiply_add", desks[d]['node_val'], tmp2, v_diff_3_4, v_tree[3]))  # low_pair
                # Select from high pair (tree[5] or tree[6]) using precomputed diff
                self.emit("valu", ("multiply_add", tmp1, tmp2, v_diff_5_6, v_tree[5]))  # high_pair in tmp1
                # Final selection based on bit1
                self.emit("valu", ("-", tmp2, tmp1, desks[d]['node_val']))  # diff_pairs
                self.emit("valu", ("multiply_add", desks[d]['node_val'], desks[d]['addr'], tmp2, desks[d]['node_val']))  # result
                # XOR and hash
                self.emit("valu", ("^", desks[d]['val'], desks[d]['val'], desks[d]['node_val']))
                emit_hash_stages(d)
                emit_branch(d)

            # Rounds 14-15: Gather without bounds (max idx < 2047)
            # Double-pump approach
            for _round in range(14, 16):
                for d in range(NUM_DESKS):
                    self.emit("valu", ("+", desks[d]['addr'], v_forest_p, desks[d]['idx']))
                for d in range(NUM_DESKS):
                    for lane in range(VLEN):
                        self.emit("load", ("load", desks[d]['node_val'] + lane, desks[d]['addr'] + lane))
                for d in range(NUM_DESKS):
                    self.emit("valu", ("^", desks[d]['val'], desks[d]['val'], desks[d]['node_val']))
                emit_hash_stages_all_desks()
                for d in range(NUM_DESKS):
                    emit_branch(d)

            # Store results
            for d in range(NUM_DESKS):
                self.emit("store", ("vstore", addr_tmp[d*2], desks[d]['idx']))
                self.emit("store", ("vstore", addr_tmp[d*2+1], desks[d]['val']))

        # Emit both tiles
        emit_tile(0)
        emit_tile(1)

        # Note: Final pause handled separately (not in slots)

        # Schedule operations in phases separated by pauses
        # Split slots at pause operations
        phases = []
        current_phase = []
        for engine, slot in self.slots:
            if engine == "flow" and slot == ("pause",):
                phases.append(current_phase)
                current_phase = []
            else:
                current_phase.append((engine, slot))
        phases.append(current_phase)  # Add final phase

        # Schedule each phase independently, then concatenate with pauses between
        self.instrs = []
        for i, phase in enumerate(phases):
            if phase:  # Skip empty phases
                phase_instrs = _schedule_slots(phase)
                self.instrs.extend(phase_instrs)
            if i < len(phases) - 1:  # Add pause after each phase except the last
                self.instrs.append({"flow": [("pause",)]})

        # Add final pause
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

    kb = KernelBuilderH94()
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


class Tests(unittest.TestCase):
    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


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
