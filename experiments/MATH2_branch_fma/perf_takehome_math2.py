"""
# Experiment MATH2: Branch Computation with Single FMA

**HYPOTHESIS:** Can we compute idx' = 2*idx + 1 + branch_bit in fewer ops?

**CURRENT (H143d - 3 ops):**
1. tmp1 = val & 1
2. idx = idx * 2 + 1  (FMA)
3. idx = idx + tmp1

**ATTEMPT: 2 ops using FMA differently?**
- FMA computes a*b + c
- We need: idx' = 2*idx + 1 + (val & 1) = 2*idx + (1 + (val & 1))

**Approach A: Precompute offset vector**
- If we could precompute offset = 1 + (val & 1) BEFORE the FMA
- Then: idx' = 2*idx + offset (single FMA)
- But offset depends on val which depends on hash which depends on idx...

**Approach B: Use bit manipulation**
- (val & 1) is 0 or 1
- 1 + (val & 1) is 1 or 2
- Can we express this as: (val | 1) & 3? No: (0 | 1) & 3 = 1 ✓, (1 | 1) & 3 = 1 ✗

**Approach C: Two-stage with different order**
- Stage 1: tmp = 2*idx + 1
- Stage 2: idx = tmp + (val & 1)
This is still 3 ops (same as current).

**Approach D: Conditional add using select**
- idx_base = 2*idx + 1
- idx = select(val & 1, idx_base + 1, idx_base)
This is 3 ops: FMA, select, mask check

**CONCLUSION:** The 3-op branch seems minimal. But let's verify H143d is optimal
and explore if stage interleaving can hide branch latency better.

This experiment will verify current H143d baseline, then try reordering.
"""

import argparse
import random
from collections import defaultdict

import sys
sys.path.insert(0, ".")

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
    """Schedule operations into VLIW bundles respecting dependencies."""
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


class KernelBuilderMATH2:
    """
    MATH2: Verify branch computation is minimal.
    Try different interleaving patterns for branch ops.
    """
    def __init__(self):
        self.slots: list[tuple[str, tuple]] = []
        self.scratch = {}
        self.scratch_ptr = 0
        self.vec_ptr = 256
        self.const_map = {}
        self.vconst_map = {}

    def alloc_scratch(self, name: str) -> int:
        addr = self.scratch_ptr
        self.scratch_ptr += 1
        self.scratch[name] = addr
        return addr

    def alloc_vec(self, name: str) -> int:
        addr = self.vec_ptr
        self.vec_ptr += VLEN
        self.scratch[name] = addr
        return addr

    def scratch_const(self, val: int, name: str = None) -> int:
        if val not in self.const_map:
            addr = self.alloc_scratch(name or f"c_{val}")
            self.emit("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def scratch_vconst(self, val: int, name: str = None) -> int:
        if val not in self.vconst_map:
            scalar = self.scratch_const(val)
            addr = self.alloc_vec(name or f"v_{val}")
            self.emit("valu", ("vbroadcast", addr, scalar))
            self.vconst_map[val] = addr
        return self.vconst_map[val]

    def emit(self, engine: str, slot: tuple) -> None:
        self.slots.append((engine, slot))

    def debug_info(self) -> DebugInfo:
        return DebugInfo(scratch_map={v: (k, 1) for k, v in self.scratch.items()})

    def build_kernel(self, forest_height, n_nodes, batch_size, rounds):
        # Load memory layout pointers
        tmp_scalar = self.alloc_scratch("tmp_scalar")
        tmp_addr = self.alloc_scratch("tmp_addr")
        for i, name in enumerate(["rounds", "n_nodes", "batch_size", "forest_height",
                                  "forest_values_p", "inp_indices_p", "inp_values_p"]):
            self.scratch[name] = self.alloc_scratch(name)
            self.emit("load", ("const", tmp_scalar, i))
            self.emit("load", ("load", self.scratch[name], tmp_scalar))

        # Vector constants
        v_zero = self.scratch_vconst(0, "v_zero")
        v_one = self.scratch_vconst(1, "v_one")
        v_two = self.scratch_vconst(2, "v_two")
        v_three = self.scratch_vconst(3, "v_three")
        v_n_nodes = self.alloc_vec("v_n_nodes")
        self.emit("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]))

        v_forest_p = self.alloc_vec("v_forest_p")
        self.emit("valu", ("vbroadcast", v_forest_p, self.scratch["forest_values_p"]))

        # Hash constants with FMA multipliers
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

        # Tree differences for selection
        v_diff_1_2 = self.alloc_vec("v_diff_1_2")
        v_diff_3_4 = self.alloc_vec("v_diff_3_4")
        v_diff_5_6 = self.alloc_vec("v_diff_5_6")
        self.emit("valu", ("-", v_diff_1_2, v_tree[2], v_tree[1]))
        self.emit("valu", ("-", v_diff_3_4, v_tree[4], v_tree[3]))
        self.emit("valu", ("-", v_diff_5_6, v_tree[6], v_tree[5]))

        # Allocate per-desk vectors (same as H143d)
        NUM_DESKS = 16
        desks = []
        for d in range(NUM_DESKS):
            desk = {
                'idx': self.alloc_vec(f"v_idx_{d}"),
                'val': self.alloc_vec(f"v_val_{d}"),
                'addr': self.alloc_vec(f"v_addr_{d}"),
                'node_val': self.alloc_vec(f"v_node_{d}"),
                'tmp1': self.alloc_vec(f"v_tmp1_{d}"),
                'tmp2': self.alloc_vec(f"v_tmp2_{d}"),
            }
            desks.append(desk)

        offset_regs = [self.alloc_scratch(f"off_{d}") for d in range(NUM_DESKS)]
        addr_tmp = [self.alloc_scratch(f"addr_tmp_{i}") for i in range(32)]

        print(f"Scratch usage: {self.scratch_ptr + (self.vec_ptr - 256)} / {SCRATCH_SIZE}")

        self.emit("flow", ("pause",))

        # Hash stage emitters (same as H143d)
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

        # EXPERIMENT: Try different branch formulations

        # Original H143d branch (3 ops)
        def emit_branch_original(desk_idx):
            d = desks[desk_idx]
            self.emit("valu", ("&", d['tmp1'], d['val'], v_one))
            self.emit("valu", ("multiply_add", d['idx'], d['idx'], v_two, v_one))
            self.emit("valu", ("+", d['idx'], d['idx'], d['tmp1']))

        # Alternative: Compute branch bit first, then fused add
        # Still 3 ops but different dependency pattern
        def emit_branch_alt1(desk_idx):
            d = desks[desk_idx]
            # Branch bit + 1 first
            self.emit("valu", ("&", d['tmp1'], d['val'], v_one))
            self.emit("valu", ("+", d['tmp2'], d['tmp1'], v_one))  # offset = 1 + branch_bit
            # idx = 2*idx + offset
            self.emit("valu", ("multiply_add", d['idx'], d['idx'], v_two, d['tmp2']))

        # Use the alternative that computes offset first
        emit_branch = emit_branch_alt1

        def emit_bounds(desk_idx):
            d = desks[desk_idx]
            self.emit("valu", ("<", d['tmp1'], d['idx'], v_n_nodes))
            self.emit("valu", ("*", d['idx'], d['idx'], d['tmp1']))

        # Round emitters
        def emit_round_0_interleaved(group_desks):
            for d in group_desks:
                self.emit("valu", ("^", desks[d]['val'], desks[d]['val'], v_tree[0]))
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
                self.emit("valu", ("^", desks[d]['val'], desks[d]['val'], desks[d]['node_val']))
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

        def emit_round_2_interleaved(group_desks):
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("-", desk['tmp1'], desk['idx'], v_three))
                self.emit("valu", ("&", desk['tmp2'], desk['tmp1'], v_one))
                self.emit("valu", (">>", desk['idx'], desk['tmp1'], v_one))
                self.emit("valu", ("multiply_add", desk['node_val'], desk['tmp2'], v_diff_3_4, v_tree[3]))
                self.emit("valu", ("multiply_add", desk['tmp1'], desk['tmp2'], v_diff_5_6, v_tree[5]))
                self.emit("valu", ("-", desk['tmp2'], desk['tmp1'], desk['node_val']))
                self.emit("valu", ("multiply_add", desk['node_val'], desk['idx'], desk['tmp2'], desk['node_val']))
            for d in group_desks:
                self.emit("valu", ("^", desks[d]['val'], desks[d]['val'], desks[d]['node_val']))
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
                self.emit("valu", ("^", desks[d]['val'], desks[d]['val'], desks[d]['node_val']))
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
                self.emit("valu", ("^", desks[d]['val'], desks[d]['val'], desks[d]['node_val']))
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
            emit_round_0_interleaved(group_desks)

        def emit_round_12_interleaved(group_desks):
            emit_round_1_interleaved(group_desks)

        def emit_round_13_interleaved(group_desks):
            emit_round_2_interleaved(group_desks)

        def emit_round_15_final_interleaved(group_desks):
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("+", desk['addr'], v_forest_p, desk['idx']))
            for d in group_desks:
                desk = desks[d]
                for lane in range(VLEN):
                    self.emit("load", ("load", desk['node_val'] + lane, desk['addr'] + lane))
            for d in group_desks:
                self.emit("valu", ("^", desks[d]['val'], desks[d]['val'], desks[d]['node_val']))
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
                emit_branch(d)  # Need this for final idx

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
            num_groups = NUM_DESKS // GROUP_SIZE

            for g in range(num_groups):
                group_desks = list(range(g * GROUP_SIZE, (g + 1) * GROUP_SIZE))

                emit_round_0_interleaved(group_desks)
                emit_round_1_interleaved(group_desks)
                emit_round_2_interleaved(group_desks)
                for _ in range(3, 10):
                    emit_gather_round_interleaved(group_desks)
                emit_round_10_interleaved(group_desks)
                emit_round_11_interleaved(group_desks)
                emit_round_12_interleaved(group_desks)
                emit_round_13_interleaved(group_desks)
                emit_gather_round_interleaved(group_desks)  # Round 14
                emit_round_15_final_interleaved(group_desks)

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


def do_kernel_test(forest_height: int, rounds: int, batch_size: int, seed: int = 123,
                   trace: bool = False, check: bool = False):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilderMATH2()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)

    value_trace = {}
    machine = Machine(mem, kb.instrs, kb.debug_info(), n_cores=N_CORES, value_trace=value_trace, trace=trace)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()

    cycles = do_kernel_test(10, 16, 256, check=args.check, trace=args.trace)
    if args.check:
        print(f"CORRECTNESS CHECK PASSED!")
