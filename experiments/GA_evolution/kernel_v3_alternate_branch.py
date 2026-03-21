"""
GA Evolution Variant V3: Alternate Branch Computation
=====================================================

The branch computation is currently:
    idx' = 2 * idx + 1 + (val & 1)

This uses 3 operations:
1. val & 1 -> tmp
2. multiply_add: idx * 2 + 1 -> idx
3. idx + tmp -> idx

Alternative formulations to try:
1. idx' = (idx << 1) + 1 + (val & 1)
   - shift, add_imm, and, add = 4 ops (worse)

2. idx' = (idx << 1) | 1 | (val & 1)
   - Only works if (val & 1) doesn't already have bit 0 set
   - Actually: (idx << 1) always has bit 0 = 0
   - 1 | (val & 1) = 1 if val is even, 2 if val is odd (wrong)
   - Doesn't work

3. Using the fact that (val & 1) XOR 1 gives branch direction:
   - val even -> go right (idx * 2 + 1)
   - val odd -> go left (idx * 2 + 2)
   - So branch_offset = 1 + (val & 1)
   - idx' = idx * 2 + branch_offset = (idx << 1) + 1 + (val & 1)
   - Same as current

4. Try: idx' = (idx + idx) + 1 + (val & 1)
   - add, and, add, add_imm = 4 ops (worse)

5. Using multiply_add differently:
   - idx' = idx * 2 + (1 + (val & 1))
   - We need (1 + (val & 1)) as a vector
   - 1 + (val & 1) = 1 or 2
   - Could we precompute? No, val changes each round

Actually, let me try a different approach: what if we can reduce the
NUMBER of rounds that need branch computation?

Wait - after round 10, ALL indices wrap to 0. So rounds 11-13 have
predictable indices:
- Round 11: all idx = 0
- Round 12: all idx = 1 or 2
- Round 13: all idx = 3, 4, 5, or 6

These are already optimized (using arithmetic selection).

Let me try: Can we eliminate the branch computation for round 14?
No, after round 13 the indices are unpredictable again.

New idea: Can we batch the branch computations somehow?
If we had val0, val1, ... for multiple rounds, could we compute
all branches in parallel? No, because each round depends on the previous.

Let me try yet another approach: Can we use SIMD parallelism better?
What if we pack multiple tree traversals into a single vector lane?
No, we're already using 8-wide vectors.

After extensive analysis, the current 3-op branch seems optimal.

Let me try one more thing: What if we change the interleaving pattern
to better utilize the 6 VALU slots? The scheduler might be leaving
some slots empty.

This kernel tries: Emit operations for exactly 6 desks at a time
(matching VALU slot count) with carefully ordered operations.
"""

import sys
sys.path.insert(0, "/home/hestiasadmin/projects/original_performance_takehome")

import random
import argparse
from collections import defaultdict

from problem import (
    Engine, DebugInfo, SLOT_LIMITS, VLEN, N_CORES, SCRATCH_SIZE,
    Machine, Tree, Input, HASH_STAGES, build_mem_image, reference_kernel2,
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


class KernelBuilder:
    """
    6-wide VALU-aligned kernel.

    Emit operations in batches of 6 desks to match VALU slot count.
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

        fast_init_vars = [
            ("n_nodes", 1), ("forest_values_p", 4),
            ("inp_indices_p", 5), ("inp_values_p", 6),
        ]
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
        addr_tmp = [self.alloc_scratch(f"addr_tmp_{i}") for i in range(NUM_DESKS * 2)]

        self.emit("flow", ("pause",))

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

        def emit_branch(desk_idx):
            d = desks[desk_idx]
            self.emit("valu", ("&", d['tmp1'], d['val'], v_one))
            self.emit("valu", ("multiply_add", d['idx'], d['idx'], v_two, v_one))
            self.emit("valu", ("+", d['idx'], d['idx'], d['tmp1']))

        def emit_bounds(desk_idx):
            d = desks[desk_idx]
            self.emit("valu", ("<", d['tmp1'], d['idx'], v_n_nodes))
            self.emit("valu", ("*", d['idx'], d['idx'], d['tmp1']))

        def emit_round_0(desk_idx):
            d = desks[desk_idx]
            self.emit("valu", ("^", d['val'], d['val'], v_tree[0]))
            emit_hash_stages(desk_idx)
            emit_branch(desk_idx)

        def emit_round_1(desk_idx):
            d = desks[desk_idx]
            self.emit("valu", ("-", d['tmp1'], d['idx'], v_one))
            self.emit("valu", ("multiply_add", d['node_val'], d['tmp1'], v_diff_1_2, v_tree[1]))
            self.emit("valu", ("^", d['val'], d['val'], d['node_val']))
            emit_hash_stages(desk_idx)
            emit_branch(desk_idx)

        def emit_round_2(desk_idx):
            d = desks[desk_idx]
            self.emit("valu", ("-", d['tmp1'], d['idx'], v_three))
            self.emit("valu", ("&", d['tmp2'], d['tmp1'], v_one))
            self.emit("valu", (">>", d['addr'], d['tmp1'], v_one))
            self.emit("valu", ("multiply_add", d['node_val'], d['tmp2'], v_diff_3_4, v_tree[3]))
            self.emit("valu", ("multiply_add", d['tmp1'], d['tmp2'], v_diff_5_6, v_tree[5]))
            self.emit("valu", ("-", d['tmp2'], d['tmp1'], d['node_val']))
            self.emit("valu", ("multiply_add", d['node_val'], d['addr'], d['tmp2'], d['node_val']))
            self.emit("valu", ("^", d['val'], d['val'], d['node_val']))
            emit_hash_stages(desk_idx)
            emit_branch(desk_idx)

        def emit_gather_round(desk_idx):
            d = desks[desk_idx]
            self.emit("valu", ("+", d['addr'], v_forest_p, d['idx']))
            for lane in range(VLEN):
                self.emit("load", ("load", d['node_val'] + lane, d['addr'] + lane))
            self.emit("valu", ("^", d['val'], d['val'], d['node_val']))
            emit_hash_stages(desk_idx)
            emit_branch(desk_idx)

        def emit_round_10(desk_idx):
            d = desks[desk_idx]
            self.emit("valu", ("+", d['addr'], v_forest_p, d['idx']))
            for lane in range(VLEN):
                self.emit("load", ("load", d['node_val'] + lane, d['addr'] + lane))
            self.emit("valu", ("^", d['val'], d['val'], d['node_val']))
            emit_hash_stages(desk_idx)
            emit_branch(desk_idx)
            emit_bounds(desk_idx)

        def emit_round_11(desk_idx):
            d = desks[desk_idx]
            self.emit("valu", ("^", d['val'], d['val'], v_tree[0]))
            emit_hash_stages(desk_idx)
            emit_branch(desk_idx)

        def emit_round_12(desk_idx):
            d = desks[desk_idx]
            self.emit("valu", ("-", d['tmp1'], d['idx'], v_one))
            self.emit("valu", ("multiply_add", d['node_val'], d['tmp1'], v_diff_1_2, v_tree[1]))
            self.emit("valu", ("^", d['val'], d['val'], d['node_val']))
            emit_hash_stages(desk_idx)
            emit_branch(desk_idx)

        def emit_round_13(desk_idx):
            d = desks[desk_idx]
            self.emit("valu", ("-", d['tmp1'], d['idx'], v_three))
            self.emit("valu", ("&", d['tmp2'], d['tmp1'], v_one))
            self.emit("valu", (">>", d['addr'], d['tmp1'], v_one))
            self.emit("valu", ("multiply_add", d['node_val'], d['tmp2'], v_diff_3_4, v_tree[3]))
            self.emit("valu", ("multiply_add", d['tmp1'], d['tmp2'], v_diff_5_6, v_tree[5]))
            self.emit("valu", ("-", d['tmp2'], d['tmp1'], d['node_val']))
            self.emit("valu", ("multiply_add", d['node_val'], d['addr'], d['tmp2'], d['node_val']))
            self.emit("valu", ("^", d['val'], d['val'], d['node_val']))
            emit_hash_stages(desk_idx)
            emit_branch(desk_idx)

        def emit_round_15_final(desk_idx):
            d = desks[desk_idx]
            self.emit("valu", ("+", d['addr'], v_forest_p, d['idx']))
            for lane in range(VLEN):
                self.emit("load", ("load", d['node_val'] + lane, d['addr'] + lane))
            self.emit("valu", ("^", d['val'], d['val'], d['node_val']))
            emit_hash_stages(desk_idx)

        def emit_tile_6wide(tile_idx):
            """Process tile with 6-desk batches to match VALU width."""
            tile_offset = tile_idx * NUM_DESKS * VLEN

            for d in range(NUM_DESKS):
                self.emit("load", ("const", offset_regs[d], tile_offset + d * VLEN))
            for d in range(NUM_DESKS):
                self.emit("alu", ("+", addr_tmp[d*2], self.scratch["inp_indices_p"], offset_regs[d]))
                self.emit("alu", ("+", addr_tmp[d*2+1], self.scratch["inp_values_p"], offset_regs[d]))
            for d in range(NUM_DESKS):
                self.emit("load", ("vload", desks[d]['idx'], addr_tmp[d*2]))
                self.emit("load", ("vload", desks[d]['val'], addr_tmp[d*2+1]))

            # Process in groups of 6 (matching 6 VALU slots)
            GROUP_SIZE = 6
            all_desks = list(range(NUM_DESKS))

            for group_start in range(0, NUM_DESKS, GROUP_SIZE):
                group_desks = all_desks[group_start:group_start + GROUP_SIZE]

                # All rounds for this group
                for d in group_desks:
                    emit_round_0(d)
                for d in group_desks:
                    emit_round_1(d)
                for d in group_desks:
                    emit_round_2(d)
                for rnd in range(3, 10):
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
                    emit_gather_round(d)
                for d in group_desks:
                    emit_round_15_final(d)

            for d in range(NUM_DESKS):
                self.emit("store", ("vstore", addr_tmp[d*2], desks[d]['idx']))
                self.emit("store", ("vstore", addr_tmp[d*2+1], desks[d]['val']))

        emit_tile_6wide(0)
        emit_tile_6wide(1)

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
                self.instrs.extend(_schedule_slots(phase))
            if i < len(phases) - 1:
                self.instrs.append({"flow": [("pause",)]})
        self.instrs.append({"flow": [("pause",)]})


BASELINE = 147734


def do_kernel_test(forest_height: int, rounds: int, batch_size: int,
                   seed: int = 123, trace: bool = False, prints: bool = False,
                   check: bool = False):
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)

    value_trace = {}
    machine = Machine(mem, kb.instrs, kb.debug_info(), n_cores=N_CORES,
                      value_trace=value_trace, trace=trace)
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if check:
            assert (machine.mem[inp_values_p:inp_values_p + len(inp.values)]
                    == ref_mem[inp_values_p:inp_values_p + len(inp.values)]), f"Incorrect on round {i}"

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline:", BASELINE / machine.cycle)
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
