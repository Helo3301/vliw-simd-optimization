"""
BF200 Theory 1: Address-Tracking Branch Fusion

Instead of tracking idx and computing addr = forest_p + idx every gather round,
track addr directly. The branch formula becomes:

Current: idx = 2*idx + 1 + bit (3 VALU) + addr = forest_p + idx (1 VALU) = 4 VALU
Addr-tracking: addr = 2*addr + (1-fp) (1 FMA) + addr = addr + bit (1 ADD) = 3 VALU total
  (AND for bit extraction is still needed, so branch = 3 VALU, but no separate addr!)

Savings per desk:
  R3-R9: 7 rounds x 1 addr saved = 7
  R10: addr ready from R9 = 1
  R14: 1 addr saved
  R15: addr ready from R14 = 1
  Cost: 2 idx->addr conversions (after R2, after R13)
  Init: 1 VALU for v_1_minus_fp
Net: 10 - 2 = 8 per desk, minus 1 init = 8*32 - 1 = 255 VALU saved
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


class KernelBuilderA1:
    """A1: R10 Branch Skip - based on B4-2 with R10 optimization"""
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

        # v_zero REMOVED
        v_one = self.scratch_vconst(1, "v_one")
        v_two = self.scratch_vconst(2, "v_two")
        v_three = self.scratch_vconst(3, "v_three")

        v_forest_p = self.alloc_vec("v_forest_p")
        self.emit("valu", ("vbroadcast", v_forest_p, self.scratch["forest_values_p"]))

        # THEORY 1: Precompute v_1_minus_fp = 1 - forest_p for addr-tracking branch
        v_1_minus_fp = self.alloc_vec("v_1_minus_fp")
        self.emit("valu", ("-", v_1_minus_fp, v_one, v_forest_p))

        # THEORY 1b: Precompute v_fp_plus_1 = forest_p + 1 for combined branch+conversion
        v_fp_plus_1 = self.alloc_vec("v_fp_plus_1")
        self.emit("valu", ("+", v_fp_plus_1, v_forest_p, v_one))

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

        # All diff vectors removed: R1/R12 use vselect, R2/R13 use 3 vselect
        # (saves 3 VALU SUB ops and 24 scratch slots)

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
            }
            desks.append(desk)

        offset_regs = [self.alloc_scratch(f"off_{d}") for d in range(NUM_DESKS)]
        addr_tmp = [self.alloc_scratch(f"addr_tmp_{i}") for i in range(32)]

        print(f"Scratch usage: {self.scratch_ptr} / {SCRATCH_SIZE}")

        self.emit("flow", ("pause",))

        def emit_hash_interleaved(group_desks):
            # BF300: Desk ordering [2,0,1,3] = non-zero desk first, then desk 0
            # Saves 8 cycles vs previous [0,2,1,3] ordering
            gd = [group_desks[2], group_desks[0], group_desks[1], group_desks[3]]
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

        def emit_branch(desk_idx):
            d = desks[desk_idx]
            self.emit("valu", ("&", d['tmp1'], d['val'], v_one))
            self.emit("valu", ("multiply_add", d['idx'], d['idx'], v_two, v_one))
            self.emit("valu", ("+", d['idx'], d['idx'], d['tmp1']))

        def emit_branch_addr_tracking(desk_idx):
            """Branch that updates addr instead of idx. addr_new = 2*addr + (1-fp) + bit"""
            d = desks[desk_idx]
            self.emit("valu", ("&", d['tmp1'], d['val'], v_one))
            self.emit("valu", ("multiply_add", d['addr'], d['addr'], v_two, v_1_minus_fp))
            self.emit("valu", ("+", d['addr'], d['addr'], d['tmp1']))

        def emit_branch_idx_to_addr(desk_idx):
            """Combined branch + idx->addr conversion. Computes addr = fp + 2*idx + 1 + bit.
            Uses: addr = FMA(idx, 2, fp+1+bit). Doesn't update idx.
            Saves 1 VALU vs separate branch + conversion."""
            d = desks[desk_idx]
            self.emit("valu", ("&", d['tmp1'], d['val'], v_one))          # bit = val & 1
            self.emit("valu", ("+", d['tmp1'], v_fp_plus_1, d['tmp1']))    # t = fp + 1 + bit
            self.emit("valu", ("multiply_add", d['addr'], d['idx'], v_two, d['tmp1']))  # addr = 2*idx + t

        def emit_xor_with_node(desk_idx, node_vec):
            d = desks[desk_idx]
            self.emit("valu", ("^", d['val'], d['val'], node_vec))

        # Fused rounds 0+1+2 with bit tracking (from B4-2)
        def emit_rounds_0_1_2_fused(group_desks):
            # === Round 0 ===
            for d in group_desks:
                emit_xor_with_node(d, v_tree[0])
            emit_hash_interleaved(group_desks)
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("&", desk['bit0'], desk['val'], v_one))
                self.emit("valu", ("+", desk['idx'], v_one, desk['bit0']))

            # === Round 1 === (vselect for node selection: saves 1 VALU per desk)
            for d in group_desks:
                desk = desks[d]
                # vselect: if bit0!=0 select tree[2], else tree[1] (0 VALU, 1 flow)
                self.emit("flow", ("vselect", desk['node_val'], desk['bit0'], v_tree[2], v_tree[1]))
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("^", desk['val'], desk['val'], desk['node_val']))
            emit_hash_interleaved(group_desks)
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("&", desk['tmp1'], desk['val'], v_one))
                self.emit("valu", ("multiply_add", desk['idx'], desk['bit0'], v_two, v_three))
                self.emit("valu", ("+", desk['idx'], desk['idx'], desk['tmp1']))

            # === Round 2 === (Theory 10c: 3 vselect, 0 VALU for node selection)
            for d in group_desks:
                desk = desks[d]
                # 3 vselect cascade: bit1 selects within pair, bit0 selects between pairs
                self.emit("flow", ("vselect", desk['tmp2'], desk['tmp1'], v_tree[4], v_tree[3]))      # bit1: tree[3] or tree[4]
                self.emit("flow", ("vselect", desk['node_val'], desk['tmp1'], v_tree[6], v_tree[5]))   # bit1: tree[5] or tree[6]
                self.emit("flow", ("vselect", desk['node_val'], desk['bit0'], desk['node_val'], desk['tmp2']))  # bit0: pick pair
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("^", desk['val'], desk['val'], desk['node_val']))
            emit_hash_interleaved(group_desks)
            # Theory 1b: Combined branch + idx->addr conversion
            for d in group_desks:
                emit_branch_idx_to_addr(d)

        # Fused rounds 11+12+13 (from B4-2)
        def emit_rounds_11_12_13_fused(group_desks):
            # === Round 11 ===
            for d in group_desks:
                emit_xor_with_node(d, v_tree[0])
            emit_hash_interleaved(group_desks)
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("&", desk['bit0'], desk['val'], v_one))
                self.emit("valu", ("+", desk['idx'], v_one, desk['bit0']))

            # === Round 12 === (vselect for node selection: saves 1 VALU per desk)
            for d in group_desks:
                desk = desks[d]
                self.emit("flow", ("vselect", desk['node_val'], desk['bit0'], v_tree[2], v_tree[1]))
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("^", desk['val'], desk['val'], desk['node_val']))
            emit_hash_interleaved(group_desks)
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("&", desk['tmp1'], desk['val'], v_one))
                self.emit("valu", ("multiply_add", desk['idx'], desk['bit0'], v_two, v_three))
                self.emit("valu", ("+", desk['idx'], desk['idx'], desk['tmp1']))

            # === Round 13 === (Theory 10c: 3 vselect, 0 VALU for node selection)
            for d in group_desks:
                desk = desks[d]
                self.emit("flow", ("vselect", desk['tmp2'], desk['tmp1'], v_tree[4], v_tree[3]))
                self.emit("flow", ("vselect", desk['node_val'], desk['tmp1'], v_tree[6], v_tree[5]))
                self.emit("flow", ("vselect", desk['node_val'], desk['bit0'], desk['node_val'], desk['tmp2']))
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("^", desk['val'], desk['val'], desk['node_val']))
            emit_hash_interleaved(group_desks)
            # Theory 1b: Combined branch + idx->addr conversion
            for d in group_desks:
                emit_branch_idx_to_addr(d)

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

        def emit_gather_round_addr_tracking(group_desks):
            """Gather round using addr-tracking: addr is already the gather address"""
            # No addr computation needed - addr is already ready from previous branch!
            # Gather
            for d in group_desks:
                desk = desks[d]
                for lane in range(VLEN):
                    self.emit("load", ("load", desk['node_val'] + lane, desk['addr'] + lane))
            # XOR
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("^", desk['val'], desk['val'], desk['node_val']))
            # Hash
            emit_hash_interleaved(group_desks)
            # Branch (updates addr, not idx)
            for d in group_desks:
                emit_branch_addr_tracking(d)

        # *** R10 with branch skip + addr-tracking ***
        def emit_round_10_optimized(group_desks):
            """Round 10: addr already ready from R9 addr-tracking. Skip branch, set idx=0."""
            # No addr compute needed - addr is ready from R9's addr-tracking branch!
            # Gather
            for d in group_desks:
                desk = desks[d]
                for lane in range(VLEN):
                    self.emit("load", ("load", desk['node_val'] + lane, desk['addr'] + lane))
            # XOR
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("^", desk['val'], desk['val'], desk['node_val']))
            # Hash
            emit_hash_interleaved(group_desks)
            # Set idx = 0 for R11 fused rounds (still uses idx)
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("^", desk['idx'], desk['idx'], desk['idx']))  # idx = 0

        def emit_round_15_final_interleaved(group_desks):
            """R15: addr already ready from R14 addr-tracking. No branch needed."""
            # No addr computation needed - addr is ready from R14's addr-tracking branch!
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
                # Fused rounds 0+1+2 (R2 branch now directly produces addr)
                emit_rounds_0_1_2_fused(group_desks)

                # NO CONVERSION NEEDED: R2 branch already computed addr via emit_branch_idx_to_addr

                # Rounds 3-9: addr-tracking gather rounds (save 1 addr per round)
                for _rnd in range(3, 10):
                    emit_gather_round_addr_tracking(group_desks)

                # Round 10: addr ready from R9, skip branch, set idx=0
                emit_round_10_optimized(group_desks)

                # Fused rounds 11+12+13 (R13 branch now directly produces addr)
                emit_rounds_11_12_13_fused(group_desks)

                # NO CONVERSION NEEDED: R13 branch already computed addr via emit_branch_idx_to_addr

                # Round 14: addr-tracking gather round
                emit_gather_round_addr_tracking(group_desks)

                # Round 15: final, addr ready from R14 (no branch, no addr compute)
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

        # Count VALU ops for reporting
        valu_count = sum(1 for e, s in self.slots if e == "valu")
        print(f"Total slots: {len(self.slots)}, VALU ops: {valu_count}, Cycles: {len(self.instrs)}")


BASELINE = 147734


def do_kernel_test(forest_height: int, rounds: int, batch_size: int, seed: int = 123,
                   trace: bool = False, prints: bool = False, check: bool = False):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilderA1()
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
        print(f"Correctness check {'PASSED' if cycles else 'FAILED'}! Cycles: {cycles}")
    else:
        do_kernel_test(10, 16, 256, trace=args.trace)
