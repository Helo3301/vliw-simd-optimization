"""
Theory 84: 2-way selection with FMA only (no vselect)

Replace: vselect(cond, a, b)
With: a*cond + b*(1-cond) = a*cond + b - b*cond = (a-b)*cond + b

This uses 2 ops: SUB + FMA instead of vselect.
If FMA is faster than vselect on the flow unit, this could help.
"""

import random
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from problem import (
    SLOT_LIMITS, VLEN, N_CORES, SCRATCH_SIZE, Machine, Tree, Input,
    HASH_STAGES, build_mem_image, reference_kernel2, DebugInfo,
)
from collections import defaultdict


def _vec_range(base, length=VLEN):
    return range(base, base + length)


def _slot_rw(engine, slot):
    reads, writes = [], []
    if engine == "alu":
        _op, dest, a1, a2 = slot
        reads = [a1, a2]
        writes = [dest]
    elif engine == "valu":
        op = slot[0]
        if op == "vbroadcast":
            reads = [slot[2]]
            writes = list(_vec_range(slot[1]))
        elif op == "multiply_add":
            reads = list(_vec_range(slot[2])) + list(_vec_range(slot[3])) + list(_vec_range(slot[4]))
            writes = list(_vec_range(slot[1]))
        else:
            reads = list(_vec_range(slot[2])) + list(_vec_range(slot[3]))
            writes = list(_vec_range(slot[1]))
    elif engine == "load":
        op = slot[0]
        if op == "load":
            reads = [slot[2]]
            writes = [slot[1]]
        elif op == "vload":
            reads = [slot[2]]
            writes = list(_vec_range(slot[1]))
        elif op == "const":
            writes = [slot[1]]
    elif engine == "store":
        op = slot[0]
        if op == "store":
            reads = [slot[1], slot[2]]
        elif op == "vstore":
            reads = [slot[1]] + list(_vec_range(slot[2]))
    elif engine == "flow":
        op = slot[0]
        if op == "select":
            reads = [slot[2], slot[3], slot[4]]
            writes = [slot[1]]
        elif op == "vselect":
            reads = list(_vec_range(slot[2])) + list(_vec_range(slot[3])) + list(_vec_range(slot[4]))
            writes = list(_vec_range(slot[1]))
        elif op == "add_imm":
            reads = [slot[2]]
            writes = [slot[1]]
    return reads, writes


def _schedule_slots(slots):
    cycles, usage = [], []
    ready_time = defaultdict(int)
    last_write = defaultdict(lambda: -1)
    last_read = defaultdict(lambda: -1)

    def ensure_cycle(cycle):
        while len(cycles) <= cycle:
            cycles.append({})
            usage.append(defaultdict(int))

    def find_cycle(engine, earliest):
        cycle = earliest
        limit = SLOT_LIMITS[engine]
        while True:
            ensure_cycle(cycle)
            if usage[cycle][engine] < limit:
                return cycle
            cycle += 1

    for engine, slot in slots:
        reads, writes = _slot_rw(engine, slot)
        earliest = max([ready_time[a] for a in reads] + [last_write[a] + 1 for a in writes] + [last_read[a] for a in writes] + [0])
        cycle = find_cycle(engine, earliest)
        ensure_cycle(cycle)
        cycles[cycle].setdefault(engine, []).append(slot)
        usage[cycle][engine] += 1
        for addr in reads:
            last_read[addr] = max(last_read[addr], cycle)
        for addr in writes:
            last_write[addr] = cycle
            ready_time[addr] = cycle + 1
    return [c for c in cycles if c]


class KernelBuilder84:
    """Theory 84: Replace vselect with FMA-based selection"""
    def __init__(self):
        self.slots = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self.vconst_map = {}

    def emit(self, engine, slot):
        self.slots.append((engine, slot))

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
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

    def build_kernel(self, forest_height, n_nodes, batch_size, rounds):
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
                'tmp3': self.alloc_vec(f"v_tmp3_{d}"),  # Extra temp for FMA select
            }
            desks.append(desk)

        offset_regs = [self.alloc_scratch(f"off_{d}") for d in range(NUM_DESKS)]
        addr_tmp = [self.alloc_scratch(f"addr_tmp_{i}") for i in range(32)]

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

        def emit_xor_with_node(desk_idx, node_vec):
            d = desks[desk_idx]
            self.emit("valu", ("^", d['val'], d['val'], node_vec))

        # THEORY 84: FMA-based select
        # select(cond, a, b) -> (a-b)*cond + b
        def emit_fma_select(desk, dest, cond, a, b):
            """Replace vselect with (a-b)*cond + b using SUB + FMA"""
            self.emit("valu", ("-", desk['tmp3'], a, b))  # tmp3 = a - b
            self.emit("valu", ("multiply_add", dest, desk['tmp3'], cond, b))  # dest = tmp3*cond + b

        def emit_rounds_0_1_2_fused(group_desks):
            # Round 0
            for d in group_desks:
                emit_xor_with_node(d, v_tree[0])
            emit_hash_interleaved(group_desks)
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("&", desk['bit0'], desk['val'], v_one))
                self.emit("valu", ("+", desk['idx'], v_one, desk['bit0']))

            # Round 1
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

            # Round 2 - THEORY 84: use FMA select instead of vselect
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("multiply_add", desk['node_val'], desk['tmp1'], v_diff_3_4, v_tree[3]))
                self.emit("valu", ("multiply_add", desk['tmp2'], desk['tmp1'], v_diff_5_6, v_tree[5]))
                # FMA select: (tmp2 - node_val) * bit0 + node_val
                emit_fma_select(desk, desk['node_val'], desk['bit0'], desk['tmp2'], desk['node_val'])
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("^", desk['val'], desk['val'], desk['node_val']))
            emit_hash_interleaved(group_desks)
            for d in group_desks:
                emit_branch(d)

        def emit_rounds_11_12_13_fused(group_desks):
            # Round 11
            for d in group_desks:
                emit_xor_with_node(d, v_tree[0])
            emit_hash_interleaved(group_desks)
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("&", desk['bit0'], desk['val'], v_one))
                self.emit("valu", ("+", desk['idx'], v_one, desk['bit0']))

            # Round 12
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

            # Round 13 - THEORY 84: use FMA select
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("multiply_add", desk['node_val'], desk['tmp1'], v_diff_3_4, v_tree[3]))
                self.emit("valu", ("multiply_add", desk['tmp2'], desk['tmp1'], v_diff_5_6, v_tree[5]))
                emit_fma_select(desk, desk['node_val'], desk['bit0'], desk['tmp2'], desk['node_val'])
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

        def emit_round_10_optimized(group_desks):
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
                desk = desks[d]
                self.emit("valu", ("^", desk['idx'], desk['idx'], desk['idx']))

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
            tile_offset = tile_idx * 16 * VLEN
            for d in range(16):
                self.emit("load", ("const", offset_regs[d], tile_offset + d * VLEN))
            for d in range(16):
                self.emit("alu", ("+", addr_tmp[d*2], self.scratch["inp_indices_p"], offset_regs[d]))
                self.emit("alu", ("+", addr_tmp[d*2+1], self.scratch["inp_values_p"], offset_regs[d]))
            for d in range(16):
                self.emit("load", ("vload", desks[d]['idx'], addr_tmp[d*2]))
                self.emit("load", ("vload", desks[d]['val'], addr_tmp[d*2+1]))

            all_groups = [list(range(g * 4, (g + 1) * 4)) for g in range(4)]
            for group_desks in all_groups:
                emit_rounds_0_1_2_fused(group_desks)
                for _ in range(3, 10):
                    emit_gather_round_interleaved(group_desks)
                emit_round_10_optimized(group_desks)
                emit_rounds_11_12_13_fused(group_desks)
                emit_gather_round_interleaved(group_desks)
                emit_round_15_final_interleaved(group_desks)

            for d in range(16):
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
                self.instrs.extend(_schedule_slots(phase))
            if i < len(phases) - 1:
                self.instrs.append({"flow": [("pause",)]})
        self.instrs.append({"flow": [("pause",)]})


def test_theory():
    random.seed(123)
    forest = Tree.generate(10)
    inp = Input.generate(forest, 256, 16)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder84()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), 16)

    machine = Machine(mem, kb.instrs, DebugInfo(scratch_map=kb.scratch_debug), n_cores=N_CORES)
    for _ in reference_kernel2(mem, {}):
        machine.run()
    return machine.cycle


if __name__ == "__main__":
    cycles = test_theory()
    print(f"Theory 84 (2-way FMA select): {cycles} cycles")
    print(f"Delta from baseline (1548): {cycles - 1548}")
