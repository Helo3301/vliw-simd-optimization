"""
# Experiment MATH1: Address Representation

**HYPOTHESIS:** Storing addr = forest_p + idx instead of idx will save operations.

**THEORY:**
- Current: 10 gather rounds × 1 VALU (addr = forest_p + idx) = 10 VALU per desk
- New: 0 VALU for address calc (addr already stored)
- Cost: Bounds check needs to convert back: +2 VALU
- Net: Save 8 VALU per desk = 256 VALU total = ~43 cycles

**CHANGES:**
- Store addr instead of idx in desk vectors
- Branch computes: addr' = 2*addr - forest_p + 1 + branch_bit
- Bounds check: idx = addr - forest_p; if idx >= n_nodes: addr = forest_p
- Gather: use addr directly (no calculation needed)
"""

import random
import unittest
from collections import defaultdict

from problem import (
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
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
    elif engine == "store":
        match slot:
            case ("store", addr, src):
                reads = [addr, src]
            case ("vstore", addr, src):
                reads = [addr] + list(_vec_range(src))
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


class KernelBuilder:
    """
    MATH1: Address representation experiment.
    Store addr = forest_p + idx instead of idx.
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

    def build_kernel(self, forest_height: int, n_nodes: int, batch_size: int, rounds: int):
        tmp_scalar = self.alloc_scratch("tmp_scalar")
        tmp_addr = self.alloc_scratch("tmp_addr")

        # Load header values
        for var_name, idx in [("n_nodes", 1), ("forest_values_p", 4), ("inp_indices_p", 5), ("inp_values_p", 6)]:
            self.alloc_scratch(var_name)
        for var_name, idx in [("n_nodes", 1), ("forest_values_p", 4), ("inp_indices_p", 5), ("inp_values_p", 6)]:
            self.emit("load", ("const", tmp_scalar, idx))
            self.emit("load", ("load", self.scratch[var_name], tmp_scalar))

        # Vector constants
        v_zero = self.scratch_vconst(0, "v_zero")
        v_one = self.scratch_vconst(1, "v_one")
        v_two = self.scratch_vconst(2, "v_two")
        v_three = self.scratch_vconst(3, "v_three")
        v_n_nodes = self.alloc_vec("v_n_nodes")
        self.emit("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]))

        # forest_values_p as vector
        v_forest_p = self.alloc_vec("v_forest_p")
        self.emit("valu", ("vbroadcast", v_forest_p, self.scratch["forest_values_p"]))

        # NEW: Precompute (1 - forest_p) for branch computation
        # addr' = 2*addr + (1 - forest_p + branch_bit) = 2*addr + offset
        # where offset = 1 - forest_p + branch_bit
        # Precompute: neg_forest_p_plus_1 = 1 - forest_p (as scalar, then broadcast)
        neg_forest_p_plus_1 = self.alloc_scratch("neg_forest_p_plus_1")
        self.emit("load", ("const", tmp_scalar, 1))
        self.emit("alu", ("-", neg_forest_p_plus_1, tmp_scalar, self.scratch["forest_values_p"]))
        v_neg_forest_p_plus_1 = self.alloc_vec("v_neg_forest_p_plus_1")
        self.emit("valu", ("vbroadcast", v_neg_forest_p_plus_1, neg_forest_p_plus_1))

        # Hash constants
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

        # Preload tree nodes 0-6
        NUM_PRELOADED = 7
        v_tree = []
        for i in range(NUM_PRELOADED):
            v_node = self.alloc_vec(f"v_tree_{i}")
            v_tree.append(v_node)
            self.emit("alu", ("+", tmp_addr, self.scratch["forest_values_p"], self.scratch_const(i)))
            self.emit("load", ("load", tmp_scalar, tmp_addr))
            self.emit("valu", ("vbroadcast", v_node, tmp_scalar))

        # Precompute tree[0] as addr for initial state
        # Initial idx = 0, so initial addr = forest_p + 0 = forest_p
        # (We'll set this per desk during load)

        # Tree differences for selection
        v_diff_1_2 = self.alloc_vec("v_diff_1_2")
        v_diff_3_4 = self.alloc_vec("v_diff_3_4")
        v_diff_5_6 = self.alloc_vec("v_diff_5_6")
        self.emit("valu", ("-", v_diff_1_2, v_tree[2], v_tree[1]))
        self.emit("valu", ("-", v_diff_3_4, v_tree[4], v_tree[3]))
        self.emit("valu", ("-", v_diff_5_6, v_tree[6], v_tree[5]))

        # Allocate per-desk vectors
        # KEY CHANGE: 'addr' stores forest_p + idx, not idx
        NUM_DESKS = 16
        desks = []
        for d in range(NUM_DESKS):
            desk = {
                'addr': self.alloc_vec(f"v_addr_{d}"),  # NOW: addr = forest_p + idx
                'val': self.alloc_vec(f"v_val_{d}"),
                'node_val': self.alloc_vec(f"v_node_{d}"),
                'tmp1': self.alloc_vec(f"v_tmp1_{d}"),
                'tmp2': self.alloc_vec(f"v_tmp2_{d}"),
                'idx': self.alloc_vec(f"v_idx_{d}"),  # Temp for bounds check and selection
            }
            desks.append(desk)

        offset_regs = [self.alloc_scratch(f"off_{d}") for d in range(NUM_DESKS)]
        addr_tmp = [self.alloc_scratch(f"addr_tmp_{i}") for i in range(32)]

        print(f"Scratch usage: {self.scratch_ptr} / {SCRATCH_SIZE}")

        self.emit("flow", ("pause",))

        # Helper: emit hash stages (same as before)
        def emit_hash_stages(desk_idx):
            d = desks[desk_idx]
            for hi in range(6):
                if hi in v_fma_mult:
                    self.emit("valu", ("multiply_add", d['val'], d['val'], v_fma_mult[hi], v_hash_consts[hi]))
                elif hi == 1:
                    self.emit("valu", ("^", d['tmp1'], d['val'], v_hash_consts[hi]))
                    self.emit("valu", (">>", d['tmp2'], d['val'], v_hash_shifts[hi]))
                    self.emit("valu", ("^", d['val'], d['tmp1'], d['tmp2']))
                elif hi == 3:
                    self.emit("valu", ("+", d['tmp1'], d['val'], v_hash_consts[hi]))
                    self.emit("valu", ("<<", d['tmp2'], d['val'], v_hash_shifts[hi]))
                    self.emit("valu", ("^", d['val'], d['tmp1'], d['tmp2']))
                elif hi == 5:
                    self.emit("valu", ("^", d['tmp1'], d['val'], v_hash_consts[hi]))
                    self.emit("valu", (">>", d['tmp2'], d['val'], v_hash_shifts[hi]))
                    self.emit("valu", ("^", d['val'], d['tmp1'], d['tmp2']))

        # NEW BRANCH: addr' = 2*addr + (1 - forest_p + branch_bit)
        def emit_branch_addr(desk_idx):
            d = desks[desk_idx]
            # branch_bit = val & 1
            self.emit("valu", ("&", d['tmp1'], d['val'], v_one))
            # offset = (1 - forest_p) + branch_bit
            self.emit("valu", ("+", d['tmp2'], v_neg_forest_p_plus_1, d['tmp1']))
            # addr' = 2*addr + offset = FMA(addr, 2, offset)
            self.emit("valu", ("multiply_add", d['addr'], d['addr'], v_two, d['tmp2']))

        # NEW BOUNDS CHECK: idx = addr - forest_p; if idx >= n_nodes: addr = forest_p
        def emit_bounds_addr(desk_idx):
            d = desks[desk_idx]
            # idx = addr - forest_p
            self.emit("valu", ("-", d['idx'], d['addr'], v_forest_p))
            # mask = idx < n_nodes (1 if in bounds, 0 if out)
            self.emit("valu", ("<", d['tmp1'], d['idx'], v_n_nodes))
            # idx_masked = idx * mask (0 if out of bounds)
            self.emit("valu", ("*", d['idx'], d['idx'], d['tmp1']))
            # addr = idx_masked + forest_p (= forest_p if out of bounds, = original addr if in bounds)
            self.emit("valu", ("+", d['addr'], d['idx'], v_forest_p))

        def emit_round_0(desk_idx):
            """Round 0: All addr = forest_p (idx=0), use tree[0] directly"""
            d = desks[desk_idx]
            self.emit("valu", ("^", d['val'], d['val'], v_tree[0]))
            emit_hash_stages(desk_idx)
            emit_branch_addr(desk_idx)

        def emit_round_1(desk_idx):
            """Round 1: idx in {1, 2}, need to compute idx from addr first"""
            d = desks[desk_idx]
            # idx = addr - forest_p
            self.emit("valu", ("-", d['idx'], d['addr'], v_forest_p))
            # selection: idx - 1 gives 0 or 1
            self.emit("valu", ("-", d['tmp1'], d['idx'], v_one))
            self.emit("valu", ("multiply_add", d['node_val'], d['tmp1'], v_diff_1_2, v_tree[1]))
            self.emit("valu", ("^", d['val'], d['val'], d['node_val']))
            emit_hash_stages(desk_idx)
            emit_branch_addr(desk_idx)

        def emit_round_2(desk_idx):
            """Round 2: idx in {3,4,5,6}, 4-way selection"""
            d = desks[desk_idx]
            # idx = addr - forest_p
            self.emit("valu", ("-", d['idx'], d['addr'], v_forest_p))
            # 4-way selection
            self.emit("valu", ("-", d['tmp1'], d['idx'], v_three))
            self.emit("valu", ("&", d['tmp2'], d['tmp1'], v_one))
            self.emit("valu", (">>", d['idx'], d['tmp1'], v_one))  # reuse idx as bit1
            self.emit("valu", ("multiply_add", d['node_val'], d['tmp2'], v_diff_3_4, v_tree[3]))
            self.emit("valu", ("multiply_add", d['tmp1'], d['tmp2'], v_diff_5_6, v_tree[5]))
            self.emit("valu", ("-", d['tmp2'], d['tmp1'], d['node_val']))
            self.emit("valu", ("multiply_add", d['node_val'], d['idx'], d['tmp2'], d['node_val']))
            self.emit("valu", ("^", d['val'], d['val'], d['node_val']))
            emit_hash_stages(desk_idx)
            emit_branch_addr(desk_idx)

        def emit_gather_round(desk_idx):
            """Gather rounds: addr is already the memory address! No calculation needed."""
            d = desks[desk_idx]
            # SAVINGS: No need to compute addr = forest_p + idx
            # addr is already forest_p + idx by our representation
            for lane in range(VLEN):
                self.emit("load", ("load", d['node_val'] + lane, d['addr'] + lane))
            self.emit("valu", ("^", d['val'], d['val'], d['node_val']))
            emit_hash_stages(desk_idx)
            emit_branch_addr(desk_idx)

        def emit_round_10(desk_idx):
            """Round 10: Gather WITH bounds check"""
            d = desks[desk_idx]
            for lane in range(VLEN):
                self.emit("load", ("load", d['node_val'] + lane, d['addr'] + lane))
            self.emit("valu", ("^", d['val'], d['val'], d['node_val']))
            emit_hash_stages(desk_idx)
            emit_branch_addr(desk_idx)
            emit_bounds_addr(desk_idx)

        def emit_round_11(desk_idx):
            """Round 11: All idx=0 after wrap, addr = forest_p"""
            d = desks[desk_idx]
            self.emit("valu", ("^", d['val'], d['val'], v_tree[0]))
            emit_hash_stages(desk_idx)
            emit_branch_addr(desk_idx)

        def emit_round_12(desk_idx):
            """Round 12: Same as round 1"""
            emit_round_1(desk_idx)

        def emit_round_13(desk_idx):
            """Round 13: Same as round 2"""
            emit_round_2(desk_idx)

        def emit_round_15_final(desk_idx):
            """Round 15: Final round - still need branch for final index"""
            d = desks[desk_idx]
            for lane in range(VLEN):
                self.emit("load", ("load", d['node_val'] + lane, d['addr'] + lane))
            self.emit("valu", ("^", d['val'], d['val'], d['node_val']))
            emit_hash_stages(desk_idx)
            emit_branch_addr(desk_idx)  # Need this for final idx

        def emit_tile_interleaved(tile_idx):
            tile_offset = tile_idx * NUM_DESKS * VLEN

            for d in range(NUM_DESKS):
                self.emit("load", ("const", offset_regs[d], tile_offset + d * VLEN))

            for d in range(NUM_DESKS):
                self.emit("alu", ("+", addr_tmp[d*2], self.scratch["inp_indices_p"], offset_regs[d]))
                self.emit("alu", ("+", addr_tmp[d*2+1], self.scratch["inp_values_p"], offset_regs[d]))

            # Load idx, then convert to addr = forest_p + idx
            for d in range(NUM_DESKS):
                self.emit("load", ("vload", desks[d]['idx'], addr_tmp[d*2]))
                self.emit("load", ("vload", desks[d]['val'], addr_tmp[d*2+1]))

            # Convert idx to addr = forest_p + idx
            for d in range(NUM_DESKS):
                self.emit("valu", ("+", desks[d]['addr'], v_forest_p, desks[d]['idx']))

            GROUP_SIZE = 4
            num_groups = NUM_DESKS // GROUP_SIZE

            for g in range(num_groups):
                group_desks = list(range(g * GROUP_SIZE, (g + 1) * GROUP_SIZE))

                for d in group_desks:
                    emit_round_0(d)
                for d in group_desks:
                    emit_round_1(d)
                for d in group_desks:
                    emit_round_2(d)
                for _rnd in range(3, 10):
                    for d in group_desks:
                        emit_gather_round(d)
                for d in group_desks:
                    emit_round_10(d)
                for d in group_desks:
                    emit_round_11(d)
                for d in group_desks:
                    emit_round_12(d)
                for d in group_desks:
                    emit_round_13(d)
                for d in group_desks:
                    emit_gather_round(d)  # Round 14
                for d in group_desks:
                    emit_round_15_final(d)

            # Store: need to convert addr back to idx first
            for d in range(NUM_DESKS):
                self.emit("valu", ("-", desks[d]['idx'], desks[d]['addr'], v_forest_p))

            for d in range(NUM_DESKS):
                self.emit("store", ("vstore", addr_tmp[d*2], desks[d]['idx']))
                self.emit("store", ("vstore", addr_tmp[d*2+1], desks[d]['val']))

        emit_tile_interleaved(0)
        emit_tile_interleaved(1)

        # Schedule
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
    check: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
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

    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        if check:
            inp_indices_p = ref_mem[5]
            inp_values_p = ref_mem[6]
            batch_size = ref_mem[2]
            assert (
                machine.mem[inp_values_p : inp_values_p + batch_size]
                == ref_mem[inp_values_p : inp_values_p + batch_size]
            ), f"Values mismatch on round {i}"
            assert (
                machine.mem[inp_indices_p : inp_indices_p + batch_size]
                == ref_mem[inp_indices_p : inp_indices_p + batch_size]
            ), f"Indices mismatch on round {i}"

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


if __name__ == "__main__":
    import sys
    check = "--check" in sys.argv
    cycles = do_kernel_test(10, 16, 256, check=check)
    if check:
        print("CORRECTNESS CHECK PASSED!")
