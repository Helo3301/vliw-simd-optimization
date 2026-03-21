"""
Batch theory tester - generates variants of A1 kernel and tests them.
Each theory modifies the emission order or structure and checks cycles.
"""
import sys
import os
import random
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from problem import (
    Engine, DebugInfo, SLOT_LIMITS, VLEN, N_CORES, SCRATCH_SIZE,
    Machine, Tree, Input, HASH_STAGES, reference_kernel, build_mem_image, reference_kernel2,
)

def _vec_range(base, length=VLEN):
    return range(base, base + length)

def _slot_rw(engine, slot):
    reads = []
    writes = []
    if engine == "alu":
        _op, dest, a1, a2 = slot
        reads = [a1, a2]
        writes = [dest]
    elif engine == "valu":
        op = slot[0]
        if op == "vbroadcast":
            dest, src = slot[1], slot[2]
            reads = [src]
            writes = list(_vec_range(dest))
        elif op == "multiply_add":
            dest, a, b, c = slot[1], slot[2], slot[3], slot[4]
            reads = list(_vec_range(a)) + list(_vec_range(b)) + list(_vec_range(c))
            writes = list(_vec_range(dest))
        else:
            _op, dest, a1, a2 = slot
            reads = list(_vec_range(a1)) + list(_vec_range(a2))
            writes = list(_vec_range(dest))
    elif engine == "load":
        op = slot[0]
        if op == "load":
            dest, addr = slot[1], slot[2]
            reads = [addr]
            writes = [dest]
        elif op == "vload":
            dest, addr = slot[1], slot[2]
            reads = [addr]
            writes = list(_vec_range(dest))
        elif op == "const":
            dest = slot[1]
            writes = [dest]
        elif op == "load_offset":
            dest, addr, _lane = slot[1], slot[2], slot[3]
            reads = [addr]
            writes = [dest]
        else:
            raise NotImplementedError(f"Unknown load op {slot}")
    elif engine == "store":
        op = slot[0]
        if op == "store":
            addr, src = slot[1], slot[2]
            reads = [addr, src]
        elif op == "vstore":
            addr, src = slot[1], slot[2]
            reads = [addr] + list(_vec_range(src))
        else:
            raise NotImplementedError(f"Unknown store op {slot}")
    elif engine == "flow":
        op = slot[0]
        if op == "select":
            dest, cond, a, b = slot[1], slot[2], slot[3], slot[4]
            reads = [cond, a, b]
            writes = [dest]
        elif op == "add_imm":
            dest, a = slot[1], slot[2]
            reads = [a]
            writes = [dest]
        elif op == "vselect":
            dest, cond, a, b = slot[1], slot[2], slot[3], slot[4]
            reads = list(_vec_range(cond)) + list(_vec_range(a)) + list(_vec_range(b))
            writes = list(_vec_range(dest))
        elif op in ("halt", "pause", "trace_write", "jump", "jump_indirect", "cond_jump", "cond_jump_rel", "coreid"):
            pass
        else:
            raise NotImplementedError(f"Unknown flow op {slot}")
    return reads, writes

def _schedule_slots(slots):
    cycles = []
    usage = []
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
    def __init__(self, theory_config=None):
        self.slots = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self.vconst_map = {}
        self.config = theory_config or {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def emit(self, engine, slot):
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

    def build_kernel(self, forest_height, n_nodes, batch_size, rounds):
        cfg = self.config
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

        NUM_PRELOADED = cfg.get('num_preloaded', 7)
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

        NUM_DESKS = cfg.get('num_desks', 16)
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
        addr_tmp = [self.alloc_scratch(f"addr_tmp_{i}") for i in range(NUM_DESKS * 2)]

        GROUP_SIZE = cfg.get('group_size', 4)

        self.emit("flow", ("pause",))

        def emit_hash_interleaved(group_desks):
            hash_order = cfg.get('hash_desk_order', 'forward')
            gd = list(group_desks)
            if hash_order == 'reverse':
                gd = list(reversed(gd))
            elif hash_order == 'interleave':
                # even indices first, then odd
                gd = [gd[i] for i in range(0, len(gd), 2)] + [gd[i] for i in range(1, len(gd), 2)]

            hash_stage_order = cfg.get('hash_stage_order', 'normal')

            if hash_stage_order == 'fma_first':
                # Emit all FMA stages first, then XOR stages
                # Stage 0: FMA
                for d in gd:
                    desk = desks[d]
                    self.emit("valu", ("multiply_add", desk['val'], desk['val'], v_fma_mult[0], v_hash_consts[0]))
                # Stage 2: FMA
                for d in gd:
                    desk = desks[d]
                    self.emit("valu", ("multiply_add", desk['val'], desk['val'], v_fma_mult[2], v_hash_consts[2]))
                # Stage 4: FMA
                for d in gd:
                    desk = desks[d]
                    self.emit("valu", ("multiply_add", desk['val'], desk['val'], v_fma_mult[4], v_hash_consts[4]))
                # Stage 1: XOR chain
                for d in gd:
                    desk = desks[d]
                    self.emit("valu", ("^", desk['tmp1'], desk['val'], v_hash_consts[1]))
                    self.emit("valu", (">>", desk['tmp2'], desk['val'], v_hash_shifts[1]))
                    self.emit("valu", ("^", desk['val'], desk['tmp1'], desk['tmp2']))
                # Stage 3: XOR chain
                for d in gd:
                    desk = desks[d]
                    self.emit("valu", ("+", desk['tmp1'], desk['val'], v_hash_consts[3]))
                    self.emit("valu", ("<<", desk['tmp2'], desk['val'], v_hash_shifts[3]))
                    self.emit("valu", ("^", desk['val'], desk['tmp1'], desk['tmp2']))
                # Stage 5: XOR chain
                for d in gd:
                    desk = desks[d]
                    self.emit("valu", ("^", desk['tmp1'], desk['val'], v_hash_consts[5]))
                    self.emit("valu", (">>", desk['tmp2'], desk['val'], v_hash_shifts[5]))
                    self.emit("valu", ("^", desk['val'], desk['tmp1'], desk['tmp2']))
            elif hash_stage_order == 'per_desk':
                # Emit all 6 stages for desk 0, then all 6 for desk 1, etc.
                for d in gd:
                    desk = desks[d]
                    self.emit("valu", ("multiply_add", desk['val'], desk['val'], v_fma_mult[0], v_hash_consts[0]))
                    self.emit("valu", ("^", desk['tmp1'], desk['val'], v_hash_consts[1]))
                    self.emit("valu", (">>", desk['tmp2'], desk['val'], v_hash_shifts[1]))
                    self.emit("valu", ("^", desk['val'], desk['tmp1'], desk['tmp2']))
                    self.emit("valu", ("multiply_add", desk['val'], desk['val'], v_fma_mult[2], v_hash_consts[2]))
                    self.emit("valu", ("+", desk['tmp1'], desk['val'], v_hash_consts[3]))
                    self.emit("valu", ("<<", desk['tmp2'], desk['val'], v_hash_shifts[3]))
                    self.emit("valu", ("^", desk['val'], desk['tmp1'], desk['tmp2']))
                    self.emit("valu", ("multiply_add", desk['val'], desk['val'], v_fma_mult[4], v_hash_consts[4]))
                    self.emit("valu", ("^", desk['tmp1'], desk['val'], v_hash_consts[5]))
                    self.emit("valu", (">>", desk['tmp2'], desk['val'], v_hash_shifts[5]))
                    self.emit("valu", ("^", desk['val'], desk['tmp1'], desk['tmp2']))
            elif hash_stage_order == 'split_phases':
                # Stages 0-2 for all desks, then stages 3-5
                for d in gd:
                    desk = desks[d]
                    self.emit("valu", ("multiply_add", desk['val'], desk['val'], v_fma_mult[0], v_hash_consts[0]))
                for d in gd:
                    desk = desks[d]
                    self.emit("valu", ("^", desk['tmp1'], desk['val'], v_hash_consts[1]))
                    self.emit("valu", (">>", desk['tmp2'], desk['val'], v_hash_shifts[1]))
                    self.emit("valu", ("^", desk['val'], desk['tmp1'], desk['tmp2']))
                for d in gd:
                    desk = desks[d]
                    self.emit("valu", ("multiply_add", desk['val'], desk['val'], v_fma_mult[2], v_hash_consts[2]))
                # Now stages 3-5
                for d in gd:
                    desk = desks[d]
                    self.emit("valu", ("+", desk['tmp1'], desk['val'], v_hash_consts[3]))
                    self.emit("valu", ("<<", desk['tmp2'], desk['val'], v_hash_shifts[3]))
                    self.emit("valu", ("^", desk['val'], desk['tmp1'], desk['tmp2']))
                for d in gd:
                    desk = desks[d]
                    self.emit("valu", ("multiply_add", desk['val'], desk['val'], v_fma_mult[4], v_hash_consts[4]))
                for d in gd:
                    desk = desks[d]
                    self.emit("valu", ("^", desk['tmp1'], desk['val'], v_hash_consts[5]))
                    self.emit("valu", (">>", desk['tmp2'], desk['val'], v_hash_shifts[5]))
                    self.emit("valu", ("^", desk['val'], desk['tmp1'], desk['tmp2']))
            else:
                # Normal order (baseline)
                for d in gd:
                    desk = desks[d]
                    self.emit("valu", ("multiply_add", desk['val'], desk['val'], v_fma_mult[0], v_hash_consts[0]))
                for d in gd:
                    desk = desks[d]
                    self.emit("valu", ("^", desk['tmp1'], desk['val'], v_hash_consts[1]))
                    self.emit("valu", (">>", desk['tmp2'], desk['val'], v_hash_shifts[1]))
                    self.emit("valu", ("^", desk['val'], desk['tmp1'], desk['tmp2']))
                for d in gd:
                    desk = desks[d]
                    self.emit("valu", ("multiply_add", desk['val'], desk['val'], v_fma_mult[2], v_hash_consts[2]))
                for d in gd:
                    desk = desks[d]
                    self.emit("valu", ("+", desk['tmp1'], desk['val'], v_hash_consts[3]))
                    self.emit("valu", ("<<", desk['tmp2'], desk['val'], v_hash_shifts[3]))
                    self.emit("valu", ("^", desk['val'], desk['tmp1'], desk['tmp2']))
                for d in gd:
                    desk = desks[d]
                    self.emit("valu", ("multiply_add", desk['val'], desk['val'], v_fma_mult[4], v_hash_consts[4]))
                for d in gd:
                    desk = desks[d]
                    self.emit("valu", ("^", desk['tmp1'], desk['val'], v_hash_consts[5]))
                    self.emit("valu", (">>", desk['tmp2'], desk['val'], v_hash_shifts[5]))
                    self.emit("valu", ("^", desk['val'], desk['tmp1'], desk['tmp2']))

        def emit_branch(desk_idx):
            d = desks[desk_idx]
            branch_style = cfg.get('branch_style', 'normal')
            if branch_style == 'normal':
                self.emit("valu", ("&", d['tmp1'], d['val'], v_one))
                self.emit("valu", ("multiply_add", d['idx'], d['idx'], v_two, v_one))
                self.emit("valu", ("+", d['idx'], d['idx'], d['tmp1']))
            elif branch_style == 'shift':
                # Try: idx = (idx << 1) | 1 | (val & 1)  ... but that's wrong
                # idx' = idx*2 + 1 + (val & 1)
                # Using shift: idx<<1 + 1 + (val&1)
                self.emit("valu", ("&", d['tmp1'], d['val'], v_one))
                self.emit("valu", ("<<", d['idx'], d['idx'], v_one))  # idx = idx * 2
                self.emit("valu", ("+", d['idx'], d['idx'], v_one))   # idx = idx + 1
                self.emit("valu", ("+", d['idx'], d['idx'], d['tmp1']))  # idx = idx + bit
            else:
                self.emit("valu", ("&", d['tmp1'], d['val'], v_one))
                self.emit("valu", ("multiply_add", d['idx'], d['idx'], v_two, v_one))
                self.emit("valu", ("+", d['idx'], d['idx'], d['tmp1']))

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

        def emit_gather_round_interleaved(group_desks):
            load_order = cfg.get('load_order', 'desk_first')

            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("+", desk['addr'], v_forest_p, desk['idx']))

            if load_order == 'lane_first':
                for lane in range(VLEN):
                    for d in group_desks:
                        desk = desks[d]
                        self.emit("load", ("load", desk['node_val'] + lane, desk['addr'] + lane))
            elif load_order == 'pairs':
                # Emit loads in pairs: 2 per desk alternating
                for lane_start in range(0, VLEN, 2):
                    for d in group_desks:
                        desk = desks[d]
                        self.emit("load", ("load", desk['node_val'] + lane_start, desk['addr'] + lane_start))
                        self.emit("load", ("load", desk['node_val'] + lane_start + 1, desk['addr'] + lane_start + 1))
            else:
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
            load_order = cfg.get('load_order', 'desk_first')

            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("+", desk['addr'], v_forest_p, desk['idx']))

            if load_order == 'lane_first':
                for lane in range(VLEN):
                    for d in group_desks:
                        desk = desks[d]
                        self.emit("load", ("load", desk['node_val'] + lane, desk['addr'] + lane))
            elif load_order == 'pairs':
                for lane_start in range(0, VLEN, 2):
                    for d in group_desks:
                        desk = desks[d]
                        self.emit("load", ("load", desk['node_val'] + lane_start, desk['addr'] + lane_start))
                        self.emit("load", ("load", desk['node_val'] + lane_start + 1, desk['addr'] + lane_start + 1))
            else:
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
            load_order = cfg.get('load_order', 'desk_first')

            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("+", desk['addr'], v_forest_p, desk['idx']))

            if load_order == 'lane_first':
                for lane in range(VLEN):
                    for d in group_desks:
                        desk = desks[d]
                        self.emit("load", ("load", desk['node_val'] + lane, desk['addr'] + lane))
            elif load_order == 'pairs':
                for lane_start in range(0, VLEN, 2):
                    for d in group_desks:
                        desk = desks[d]
                        self.emit("load", ("load", desk['node_val'] + lane_start, desk['addr'] + lane_start))
                        self.emit("load", ("load", desk['node_val'] + lane_start + 1, desk['addr'] + lane_start + 1))
            else:
                for d in group_desks:
                    desk = desks[d]
                    for lane in range(VLEN):
                        self.emit("load", ("load", desk['node_val'] + lane, desk['addr'] + lane))

            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("^", desk['val'], desk['val'], desk['node_val']))
            emit_hash_interleaved(group_desks)

        def emit_tile_interleaved(tile_idx):
            num_tiles = cfg.get('num_tiles', 2)
            tile_offset = tile_idx * NUM_DESKS * VLEN

            for d in range(NUM_DESKS):
                self.emit("load", ("const", offset_regs[d], tile_offset + d * VLEN))
            for d in range(NUM_DESKS):
                self.emit("alu", ("+", addr_tmp[d*2], self.scratch["inp_indices_p"], offset_regs[d]))
                self.emit("alu", ("+", addr_tmp[d*2+1], self.scratch["inp_values_p"], offset_regs[d]))
            for d in range(NUM_DESKS):
                self.emit("load", ("vload", desks[d]['idx'], addr_tmp[d*2]))
                self.emit("load", ("vload", desks[d]['val'], addr_tmp[d*2+1]))

            num_full_groups = NUM_DESKS // GROUP_SIZE
            remainder = NUM_DESKS % GROUP_SIZE
            all_groups = []
            for g in range(num_full_groups):
                all_groups.append(list(range(g * GROUP_SIZE, (g + 1) * GROUP_SIZE)))
            if remainder > 0:
                all_groups.append(list(range(num_full_groups * GROUP_SIZE, NUM_DESKS)))

            group_order = cfg.get('group_order', 'forward')
            if group_order == 'reverse':
                all_groups = list(reversed(all_groups))
            elif group_order == 'interleave':
                even = [all_groups[i] for i in range(0, len(all_groups), 2)]
                odd = [all_groups[i] for i in range(1, len(all_groups), 2)]
                all_groups = []
                for e, o in zip(even, odd):
                    all_groups.append(e)
                    all_groups.append(o)
                if len(even) > len(odd):
                    all_groups.append(even[-1])

            for group_desks in all_groups:
                emit_rounds_0_1_2_fused(group_desks)

                gather_range = range(3, 10)
                for _rnd in gather_range:
                    emit_gather_round_interleaved(group_desks)

                emit_round_10_optimized(group_desks)
                emit_rounds_11_12_13_fused(group_desks)
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


def run_test(config, check=True, quiet=True):
    random.seed(123)
    forest = Tree.generate(10)
    inp = Input.generate(forest, 256, 16)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder(config)
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), 16)

    value_trace = {}
    machine = Machine(mem, kb.instrs, kb.debug_info(), n_cores=N_CORES, value_trace=value_trace, trace=False)
    machine.prints = False

    correct = True
    try:
        for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
            machine.run()
            inp_values_p = ref_mem[6]
            if check:
                assert (machine.mem[inp_values_p : inp_values_p + len(inp.values)]
                        == ref_mem[inp_values_p : inp_values_p + len(inp.values)]), f"Incorrect on round {i}"
    except (AssertionError, Exception) as e:
        correct = False
        if not quiet:
            print(f"ERROR: {e}")

    return machine.cycle, correct


if __name__ == "__main__":
    # Run all theories
    theories = {}

    # Scheduling variations (1-15)
    theories[1] = ("Baseline (verify)", {})
    theories[4] = ("Reverse desk order in hash", {'hash_desk_order': 'reverse'})
    theories[5] = ("Interleave desk order in hash (even/odd)", {'hash_desk_order': 'interleave'})
    theories[6] = ("FMA stages first, then XOR stages", {'hash_stage_order': 'fma_first'})
    theories[7] = ("All hash stages per desk (no interleaving)", {'hash_stage_order': 'per_desk'})
    theories[8] = ("Split hash phases (0-2 then 3-5)", {'hash_stage_order': 'split_phases'})
    theories[9] = ("Load order: lane first across desks", {'load_order': 'lane_first'})
    theories[10] = ("Load order: pairs (2 per desk)", {'load_order': 'pairs'})
    theories[11] = ("Reverse group processing order", {'group_order': 'reverse'})
    theories[12] = ("Interleave group processing", {'group_order': 'interleave'})
    theories[13] = ("Reverse hash + reverse groups", {'hash_desk_order': 'reverse', 'group_order': 'reverse'})
    theories[14] = ("Per-desk hash + pairs load", {'hash_stage_order': 'per_desk', 'load_order': 'pairs'})
    theories[15] = ("Interleave hash desk + per_desk hash", {'hash_desk_order': 'interleave', 'hash_stage_order': 'per_desk'})

    # Explore best win (theory 15) with other combinations (16-30)
    theories[16] = ("T15 + reverse groups", {'hash_desk_order': 'interleave', 'hash_stage_order': 'per_desk', 'group_order': 'reverse'})
    theories[17] = ("T15 + interleave groups", {'hash_desk_order': 'interleave', 'hash_stage_order': 'per_desk', 'group_order': 'interleave'})
    theories[18] = ("Reverse hash + per_desk", {'hash_desk_order': 'reverse', 'hash_stage_order': 'per_desk'})
    theories[19] = ("Per-desk hash + reverse groups", {'hash_stage_order': 'per_desk', 'group_order': 'reverse'})
    theories[20] = ("Per-desk hash + interleave groups", {'hash_stage_order': 'per_desk', 'group_order': 'interleave'})
    theories[21] = ("Split phases + interleave desk", {'hash_stage_order': 'split_phases', 'hash_desk_order': 'interleave'})
    theories[22] = ("Split phases + reverse desk", {'hash_stage_order': 'split_phases', 'hash_desk_order': 'reverse'})
    theories[23] = ("T15 + GROUP_SIZE=3", {'hash_desk_order': 'interleave', 'hash_stage_order': 'per_desk', 'group_size': 3})
    theories[24] = ("T15 + GROUP_SIZE=2", {'hash_desk_order': 'interleave', 'hash_stage_order': 'per_desk', 'group_size': 2})
    theories[25] = ("T15 + GROUP_SIZE=5", {'hash_desk_order': 'interleave', 'hash_stage_order': 'per_desk', 'group_size': 5})
    theories[26] = ("T15 + GROUP_SIZE=8", {'hash_desk_order': 'interleave', 'hash_stage_order': 'per_desk', 'group_size': 8})
    theories[27] = ("T15 + GROUP_SIZE=6", {'hash_desk_order': 'interleave', 'hash_stage_order': 'per_desk', 'group_size': 6})
    theories[28] = ("T15 + GROUP_SIZE=1", {'hash_desk_order': 'interleave', 'hash_stage_order': 'per_desk', 'group_size': 1})
    theories[29] = ("Per-desk hash + GROUP_SIZE=3", {'hash_stage_order': 'per_desk', 'group_size': 3})
    theories[30] = ("Per-desk hash + GROUP_SIZE=2", {'hash_stage_order': 'per_desk', 'group_size': 2})

    # Group size variations (31-40)
    theories[31] = ("GROUP_SIZE=3", {'group_size': 3})
    theories[32] = ("GROUP_SIZE=2", {'group_size': 2})
    theories[33] = ("GROUP_SIZE=5", {'group_size': 5})
    theories[34] = ("GROUP_SIZE=6", {'group_size': 6})
    theories[35] = ("GROUP_SIZE=8", {'group_size': 8})
    theories[36] = ("GROUP_SIZE=1", {'group_size': 1})
    theories[37] = ("GROUP_SIZE=16", {'group_size': 16})
    theories[38] = ("Interleave desk + GROUP_SIZE=3", {'hash_desk_order': 'interleave', 'group_size': 3})
    theories[39] = ("Interleave desk + GROUP_SIZE=2", {'hash_desk_order': 'interleave', 'group_size': 2})
    theories[40] = ("Reverse desk + GROUP_SIZE=3", {'hash_desk_order': 'reverse', 'group_size': 3})

    # More combos (41-55)
    theories[41] = ("Per-desk hash + GROUP_SIZE=5", {'hash_stage_order': 'per_desk', 'group_size': 5})
    theories[42] = ("Per-desk hash + GROUP_SIZE=8", {'hash_stage_order': 'per_desk', 'group_size': 8})
    theories[43] = ("Per-desk hash + GROUP_SIZE=6", {'hash_stage_order': 'per_desk', 'group_size': 6})
    theories[44] = ("Per-desk hash + GROUP_SIZE=1", {'hash_stage_order': 'per_desk', 'group_size': 1})
    theories[45] = ("Per-desk hash + GROUP_SIZE=16", {'hash_stage_order': 'per_desk', 'group_size': 16})
    theories[46] = ("Split phases + GROUP_SIZE=3", {'hash_stage_order': 'split_phases', 'group_size': 3})
    theories[47] = ("Split phases + GROUP_SIZE=2", {'hash_stage_order': 'split_phases', 'group_size': 2})
    theories[48] = ("Split phases + GROUP_SIZE=5", {'hash_stage_order': 'split_phases', 'group_size': 5})
    theories[49] = ("FMA first + GROUP_SIZE=3", {'hash_stage_order': 'fma_first', 'group_size': 3})
    theories[50] = ("FMA first + GROUP_SIZE=2", {'hash_stage_order': 'fma_first', 'group_size': 2})
    theories[51] = ("T15 + lane_first load", {'hash_desk_order': 'interleave', 'hash_stage_order': 'per_desk', 'load_order': 'lane_first'})
    theories[52] = ("T15 + pairs load", {'hash_desk_order': 'interleave', 'hash_stage_order': 'per_desk', 'load_order': 'pairs'})
    theories[53] = ("Per-desk + lane_first", {'hash_stage_order': 'per_desk', 'load_order': 'lane_first'})
    theories[54] = ("Per-desk + pairs", {'hash_stage_order': 'per_desk', 'load_order': 'pairs'})
    theories[55] = ("Interleave all: hash+groups+loads", {'hash_desk_order': 'interleave', 'group_order': 'interleave', 'load_order': 'pairs'})

    # Desk count variations (56-65)
    theories[56] = ("12 desks", {'num_desks': 12})
    theories[57] = ("20 desks", {'num_desks': 20})
    theories[58] = ("T15 + 12 desks", {'hash_desk_order': 'interleave', 'hash_stage_order': 'per_desk', 'num_desks': 12})
    theories[59] = ("T15 + 20 desks", {'hash_desk_order': 'interleave', 'hash_stage_order': 'per_desk', 'num_desks': 20})
    theories[60] = ("Per-desk + 12 desks", {'hash_stage_order': 'per_desk', 'num_desks': 12})
    theories[61] = ("Per-desk + 20 desks", {'hash_stage_order': 'per_desk', 'num_desks': 20})
    theories[62] = ("8 desks", {'num_desks': 8})
    theories[63] = ("T15 + 8 desks", {'hash_desk_order': 'interleave', 'hash_stage_order': 'per_desk', 'num_desks': 8})
    theories[64] = ("24 desks", {'num_desks': 24})
    theories[65] = ("T15 + 24 desks", {'hash_desk_order': 'interleave', 'hash_stage_order': 'per_desk', 'num_desks': 24})

    # Cross-combos with best configs (66-80)
    theories[66] = ("T15 + GS3 + reverse groups", {'hash_desk_order': 'interleave', 'hash_stage_order': 'per_desk', 'group_size': 3, 'group_order': 'reverse'})
    theories[67] = ("T15 + GS3 + interleave groups", {'hash_desk_order': 'interleave', 'hash_stage_order': 'per_desk', 'group_size': 3, 'group_order': 'interleave'})
    theories[68] = ("T15 + GS2 + reverse groups", {'hash_desk_order': 'interleave', 'hash_stage_order': 'per_desk', 'group_size': 2, 'group_order': 'reverse'})
    theories[69] = ("T15 + GS2 + interleave groups", {'hash_desk_order': 'interleave', 'hash_stage_order': 'per_desk', 'group_size': 2, 'group_order': 'interleave'})
    theories[70] = ("Reverse desk + per_desk + GS3", {'hash_desk_order': 'reverse', 'hash_stage_order': 'per_desk', 'group_size': 3})
    theories[71] = ("Reverse desk + per_desk + GS2", {'hash_desk_order': 'reverse', 'hash_stage_order': 'per_desk', 'group_size': 2})
    theories[72] = ("T15 + GS5 + reverse groups", {'hash_desk_order': 'interleave', 'hash_stage_order': 'per_desk', 'group_size': 5, 'group_order': 'reverse'})
    theories[73] = ("T15 + GS5 + interleave groups", {'hash_desk_order': 'interleave', 'hash_stage_order': 'per_desk', 'group_size': 5, 'group_order': 'interleave'})
    theories[74] = ("Per-desk + GS3 + reverse groups", {'hash_stage_order': 'per_desk', 'group_size': 3, 'group_order': 'reverse'})
    theories[75] = ("Per-desk + GS3 + interleave groups", {'hash_stage_order': 'per_desk', 'group_size': 3, 'group_order': 'interleave'})
    theories[76] = ("Per-desk + GS2 + reverse groups", {'hash_stage_order': 'per_desk', 'group_size': 2, 'group_order': 'reverse'})
    theories[77] = ("Per-desk + GS2 + interleave groups", {'hash_stage_order': 'per_desk', 'group_size': 2, 'group_order': 'interleave'})
    theories[78] = ("Per-desk + GS5 + reverse groups", {'hash_stage_order': 'per_desk', 'group_size': 5, 'group_order': 'reverse'})
    theories[79] = ("Per-desk + GS5 + interleave groups", {'hash_stage_order': 'per_desk', 'group_size': 5, 'group_order': 'interleave'})
    theories[80] = ("Per-desk + GS6 + reverse groups", {'hash_stage_order': 'per_desk', 'group_size': 6, 'group_order': 'reverse'})

    # More desk + group combos (81-90)
    theories[81] = ("T15 + 12 desks + GS3", {'hash_desk_order': 'interleave', 'hash_stage_order': 'per_desk', 'num_desks': 12, 'group_size': 3})
    theories[82] = ("T15 + 12 desks + GS4", {'hash_desk_order': 'interleave', 'hash_stage_order': 'per_desk', 'num_desks': 12, 'group_size': 4})
    theories[83] = ("T15 + 12 desks + GS6", {'hash_desk_order': 'interleave', 'hash_stage_order': 'per_desk', 'num_desks': 12, 'group_size': 6})
    theories[84] = ("T15 + 20 desks + GS5", {'hash_desk_order': 'interleave', 'hash_stage_order': 'per_desk', 'num_desks': 20, 'group_size': 5})
    theories[85] = ("T15 + 20 desks + GS4", {'hash_desk_order': 'interleave', 'hash_stage_order': 'per_desk', 'num_desks': 20, 'group_size': 4})
    theories[86] = ("T15 + 8 desks + GS2", {'hash_desk_order': 'interleave', 'hash_stage_order': 'per_desk', 'num_desks': 8, 'group_size': 2})
    theories[87] = ("T15 + 8 desks + GS4", {'hash_desk_order': 'interleave', 'hash_stage_order': 'per_desk', 'num_desks': 8, 'group_size': 4})
    theories[88] = ("Per-desk + 12 desks + GS3", {'hash_stage_order': 'per_desk', 'num_desks': 12, 'group_size': 3})
    theories[89] = ("Per-desk + 12 desks + GS4", {'hash_stage_order': 'per_desk', 'num_desks': 12, 'group_size': 4})
    theories[90] = ("Per-desk + 12 desks + GS6", {'hash_stage_order': 'per_desk', 'num_desks': 12, 'group_size': 6})

    # Final combos (91-100)
    theories[91] = ("Per-desk + 20 desks + GS5", {'hash_stage_order': 'per_desk', 'num_desks': 20, 'group_size': 5})
    theories[92] = ("Per-desk + 20 desks + GS4", {'hash_stage_order': 'per_desk', 'num_desks': 20, 'group_size': 4})
    theories[93] = ("Per-desk + 8 desks + GS2", {'hash_stage_order': 'per_desk', 'num_desks': 8, 'group_size': 2})
    theories[94] = ("Per-desk + 8 desks + GS4", {'hash_stage_order': 'per_desk', 'num_desks': 8, 'group_size': 4})
    theories[95] = ("T15 + GS4 + reverse groups", {'hash_desk_order': 'interleave', 'hash_stage_order': 'per_desk', 'group_size': 4, 'group_order': 'reverse'})
    theories[96] = ("T15 + GS4 + interleave groups", {'hash_desk_order': 'interleave', 'hash_stage_order': 'per_desk', 'group_size': 4, 'group_order': 'interleave'})
    theories[97] = ("Split + interleave desk + GS3", {'hash_stage_order': 'split_phases', 'hash_desk_order': 'interleave', 'group_size': 3})
    theories[98] = ("Split + interleave desk + GS2", {'hash_stage_order': 'split_phases', 'hash_desk_order': 'interleave', 'group_size': 2})
    theories[99] = ("FMA first + interleave desk + GS3", {'hash_stage_order': 'fma_first', 'hash_desk_order': 'interleave', 'group_size': 3})
    theories[100] = ("FMA first + interleave desk + GS2", {'hash_stage_order': 'fma_first', 'hash_desk_order': 'interleave', 'group_size': 2})

    print(f"{'#':>3} | {'Theory':<50} | {'Cycles':>6} | {'vs A1':>5} | {'OK?':>3}")
    print("-" * 80)

    results = {}
    wins = []
    for tid in sorted(theories.keys()):
        name, config = theories[tid]
        try:
            cycles, correct = run_test(config, check=True, quiet=True)
            delta = 1548 - cycles
            status = "YES" if correct else "NO"
            results[tid] = (name, cycles, delta, status)
            marker = " **WIN**" if delta > 0 and correct else ""
            if delta > 0 and correct:
                wins.append((tid, name, cycles, delta))
            print(f"{tid:>3} | {name:<50} | {cycles:>6} | {delta:>+5} | {status:>3}{marker}")
        except Exception as e:
            results[tid] = (name, -1, 0, f"ERR: {str(e)[:40]}")
            print(f"{tid:>3} | {name:<50} | {'ERROR':>6} | {'N/A':>5} | ERR: {str(e)[:40]}")

    print("\n=== WINS ===")
    for tid, name, cycles, delta in sorted(wins, key=lambda x: -x[3]):
        print(f"  #{tid}: {name} -> {cycles} cycles (improvement: {delta})")
    print(f"\nBest: {min((c for _, c, _, s in results.values() if s == 'YES' and c > 0), default='N/A')}")
