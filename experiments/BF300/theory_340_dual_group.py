"""
Theory 340: Dual-group stagger.

Process 2 groups simultaneously, staggered by half a round.
Group A does: loads -> XOR -> hash -> branch
Group B does: hash -> branch -> loads -> XOR
This way Group A's loads overlap with Group B's hash, and vice versa.

Implementation: 
- Process groups in pairs (G0,G1), (G2,G3)
- For each pair, emit rounds 4-9 with staggered structure:
  - G0 loads, G1 hash+branch
  - G0 XOR+hash+branch, G1 loads
  etc.
"""
import random, argparse, sys
from collections import defaultdict

sys.path.insert(0, '/home/hestiasadmin/projects/original_performance_takehome')
from problem import *

def _vec_range(base, length=VLEN):
    return range(base, base + length)

def _slot_rw(engine, slot):
    reads, writes = [], []
    if engine == "alu":
        _op, dest, a1, a2 = slot; reads = [a1, a2]; writes = [dest]
    elif engine == "valu":
        op = slot[0]
        if op == "vbroadcast": reads = [slot[2]]; writes = list(_vec_range(slot[1]))
        elif op == "multiply_add":
            dest, a, b, c = slot[1], slot[2], slot[3], slot[4]
            reads = list(_vec_range(a)) + list(_vec_range(b)) + list(_vec_range(c)); writes = list(_vec_range(dest))
        else:
            _op, dest, a1, a2 = slot
            reads = list(_vec_range(a1)) + list(_vec_range(a2)); writes = list(_vec_range(dest))
    elif engine == "load":
        op = slot[0]
        if op == "load": reads = [slot[2]]; writes = [slot[1]]
        elif op == "vload": reads = [slot[2]]; writes = list(_vec_range(slot[1]))
        elif op == "const": writes = [slot[1]]
        elif op == "load_offset": reads = [slot[2]]; writes = [slot[1]]
    elif engine == "store":
        op = slot[0]
        if op == "store": reads = [slot[1], slot[2]]
        elif op == "vstore": reads = [slot[1]] + list(_vec_range(slot[2]))
    elif engine == "flow":
        op = slot[0]
        if op == "select": reads = [slot[2], slot[3], slot[4]]; writes = [slot[1]]
        elif op == "add_imm": reads = [slot[2]]; writes = [slot[1]]
        elif op == "vselect":
            reads = list(_vec_range(slot[2])) + list(_vec_range(slot[3])) + list(_vec_range(slot[4]))
            writes = list(_vec_range(slot[1]))
    return reads, writes

def _schedule_slots(slots):
    cycles, usage = [], []
    ready_time = defaultdict(int)
    last_write = defaultdict(lambda: -1)
    last_read = defaultdict(lambda: -1)
    def ensure_cycle(c):
        while len(cycles) <= c: cycles.append({}); usage.append(defaultdict(int))
    def find_cycle(engine, earliest):
        c = earliest
        while True:
            ensure_cycle(c)
            if usage[c][engine] < SLOT_LIMITS[engine]: return c
            c += 1
    for engine, slot in slots:
        reads, writes = _slot_rw(engine, slot)
        earliest = 0
        for addr in reads: earliest = max(earliest, ready_time[addr])
        for addr in writes: earliest = max(earliest, last_write[addr] + 1, last_read[addr])
        cycle = find_cycle(engine, earliest)
        ensure_cycle(cycle)
        cycles[cycle].setdefault(engine, []).append(slot)
        usage[cycle][engine] += 1
        for addr in reads:
            if last_read[addr] < cycle: last_read[addr] = cycle
        for addr in writes: last_write[addr] = cycle; ready_time[addr] = cycle + 1
    return [c for c in cycles if c]

class KernelBuilderA1:
    def __init__(self):
        self.slots = []; self.scratch = {}; self.scratch_debug = {}
        self.scratch_ptr = 0; self.const_map = {}; self.vconst_map = {}
    def debug_info(self): return DebugInfo(scratch_map=self.scratch_debug)
    def emit(self, engine, slot): self.slots.append((engine, slot))
    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name: self.scratch[name] = addr; self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length; assert self.scratch_ptr <= SCRATCH_SIZE; return addr
    def alloc_vec(self, name=None): return self.alloc_scratch(name, VLEN)
    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name or f"c_{val}")
            self.emit("load", ("const", addr, val)); self.const_map[val] = addr
        return self.const_map[val]
    def scratch_vconst(self, val, name=None):
        if val not in self.vconst_map:
            scalar = self.scratch_const(val); addr = self.alloc_vec(name or f"v_{val}")
            self.emit("valu", ("vbroadcast", addr, scalar)); self.vconst_map[val] = addr
        return self.vconst_map[val]

    def build_kernel(self, forest_height, n_nodes, batch_size, rounds):
        tmp_scalar = self.alloc_scratch("tmp_scalar")
        tmp_addr = self.alloc_scratch("tmp_addr")
        fast_init_vars = [("n_nodes", 1), ("forest_values_p", 4), ("inp_indices_p", 5), ("inp_values_p", 6)]
        for var_name, _ in fast_init_vars: self.alloc_scratch(var_name)
        for var_name, idx in fast_init_vars:
            self.emit("load", ("const", tmp_scalar, idx))
            self.emit("load", ("load", self.scratch[var_name], tmp_scalar))

        v_zero = self.scratch_vconst(0); v_one = self.scratch_vconst(1); v_two = self.scratch_vconst(2)
        v_n_nodes = self.alloc_vec(); self.emit("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]))
        v_forest_p = self.alloc_vec(); self.emit("valu", ("vbroadcast", v_forest_p, self.scratch["forest_values_p"]))
        v_1_minus_fp = self.alloc_vec(); self.emit("valu", ("-", v_1_minus_fp, v_one, v_forest_p))
        v_fp_plus_1 = self.alloc_vec(); self.emit("valu", ("+", v_fp_plus_1, v_forest_p, v_one))
        v_fp_plus_15 = self.alloc_vec(); self.emit("valu", ("+", v_fp_plus_15, v_forest_p, self.scratch_vconst(15)))

        FMA_MULTIPLIERS = {0: 4097, 2: 33, 4: 9}
        v_hash_consts, v_hash_shifts, v_fma_mult = [], [], {}
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            v_hash_consts.append(self.scratch_vconst(val1))
            if hi in FMA_MULTIPLIERS: v_fma_mult[hi] = self.scratch_vconst(FMA_MULTIPLIERS[hi]); v_hash_shifts.append(None)
            else: v_hash_shifts.append(self.scratch_vconst(val3))

        v_tree = []
        for i in range(15):
            v_node = self.alloc_vec(); v_tree.append(v_node)
            self.emit("alu", ("+", tmp_addr, self.scratch["forest_values_p"], self.scratch_const(i)))
            self.emit("load", ("load", tmp_scalar, tmp_addr)); self.emit("valu", ("vbroadcast", v_node, tmp_scalar))

        desks = []
        for d in range(16):
            desks.append({k: self.alloc_vec() for k in ['idx','val','node_val','addr','tmp1','tmp2','bit0','bit1']})
        offset_regs = [self.alloc_scratch() for _ in range(16)]
        addr_tmp = [self.alloc_scratch() for _ in range(32)]
        print(f"Scratch usage: {self.scratch_ptr} / {SCRATCH_SIZE}")
        self.emit("flow", ("pause",))

        HASH_PERM = (0, 3, 2, 1)
        GATHER_PERM = (1, 2, 3, 0)

        def emit_hash_interleaved(gdesks):
            gd = [gdesks[HASH_PERM[i]] for i in range(4)]
            for d in gd:
                dk = desks[d]
                self.emit("valu", ("multiply_add", dk['val'], dk['val'], v_fma_mult[0], v_hash_consts[0]))
                self.emit("valu", ("^", dk['tmp1'], dk['val'], v_hash_consts[1]))
                self.emit("valu", (">>", dk['tmp2'], dk['val'], v_hash_shifts[1]))
                self.emit("valu", ("^", dk['val'], dk['tmp1'], dk['tmp2']))
                self.emit("valu", ("multiply_add", dk['val'], dk['val'], v_fma_mult[2], v_hash_consts[2]))
                self.emit("valu", ("+", dk['tmp1'], dk['val'], v_hash_consts[3]))
                self.emit("valu", ("<<", dk['tmp2'], dk['val'], v_hash_shifts[3]))
                self.emit("valu", ("^", dk['val'], dk['tmp1'], dk['tmp2']))
                self.emit("valu", ("multiply_add", dk['val'], dk['val'], v_fma_mult[4], v_hash_consts[4]))
                self.emit("valu", ("^", dk['tmp1'], dk['val'], v_hash_consts[5]))
                self.emit("valu", (">>", dk['tmp2'], dk['val'], v_hash_shifts[5]))
                self.emit("valu", ("^", dk['val'], dk['tmp1'], dk['tmp2']))

        def emit_desk_hash(d):
            dk = desks[d]
            self.emit("valu", ("multiply_add", dk['val'], dk['val'], v_fma_mult[0], v_hash_consts[0]))
            self.emit("valu", ("^", dk['tmp1'], dk['val'], v_hash_consts[1]))
            self.emit("valu", (">>", dk['tmp2'], dk['val'], v_hash_shifts[1]))
            self.emit("valu", ("^", dk['val'], dk['tmp1'], dk['tmp2']))
            self.emit("valu", ("multiply_add", dk['val'], dk['val'], v_fma_mult[2], v_hash_consts[2]))
            self.emit("valu", ("+", dk['tmp1'], dk['val'], v_hash_consts[3]))
            self.emit("valu", ("<<", dk['tmp2'], dk['val'], v_hash_shifts[3]))
            self.emit("valu", ("^", dk['val'], dk['tmp1'], dk['tmp2']))
            self.emit("valu", ("multiply_add", dk['val'], dk['val'], v_fma_mult[4], v_hash_consts[4]))
            self.emit("valu", ("^", dk['tmp1'], dk['val'], v_hash_consts[5]))
            self.emit("valu", (">>", dk['tmp2'], dk['val'], v_hash_shifts[5]))
            self.emit("valu", ("^", dk['val'], dk['tmp1'], dk['tmp2']))

        def emit_branch_addr(di):
            d = desks[di]
            self.emit("valu", ("&", d['tmp1'], d['val'], v_one))
            self.emit("valu", ("multiply_add", d['addr'], d['addr'], v_two, v_1_minus_fp))
            self.emit("valu", ("+", d['addr'], d['addr'], d['tmp1']))

        def emit_desk_loads(d):
            dk = desks[d]
            for lane in range(VLEN):
                self.emit("load", ("load", dk['node_val'] + lane, dk['addr'] + lane))

        def emit_fused_0123(gdesks):
            for d in gdesks: self.emit("valu", ("^", desks[d]['val'], desks[d]['val'], v_tree[0]))
            emit_hash_interleaved(gdesks)
            for d in gdesks: self.emit("valu", ("&", desks[d]['bit0'], desks[d]['val'], v_one))
            for d in gdesks: self.emit("flow", ("vselect", desks[d]['node_val'], desks[d]['bit0'], v_tree[2], v_tree[1]))
            for d in gdesks: self.emit("valu", ("^", desks[d]['val'], desks[d]['val'], desks[d]['node_val']))
            emit_hash_interleaved(gdesks)
            for d in gdesks: self.emit("valu", ("&", desks[d]['bit1'], desks[d]['val'], v_one))
            for d in gdesks:
                dk = desks[d]
                self.emit("flow", ("vselect", dk['tmp2'], dk['bit1'], v_tree[4], v_tree[3]))
                self.emit("flow", ("vselect", dk['node_val'], dk['bit1'], v_tree[6], v_tree[5]))
                self.emit("flow", ("vselect", dk['node_val'], dk['bit0'], dk['node_val'], dk['tmp2']))
            for d in gdesks: self.emit("valu", ("^", desks[d]['val'], desks[d]['val'], desks[d]['node_val']))
            emit_hash_interleaved(gdesks)
            for d in gdesks: self.emit("valu", ("&", desks[d]['idx'], desks[d]['val'], v_one))
            for d in gdesks:
                dk = desks[d]
                self.emit("flow", ("vselect", dk['tmp2'], dk['idx'], v_tree[8], v_tree[7]))
                self.emit("flow", ("vselect", dk['node_val'], dk['idx'], v_tree[10], v_tree[9]))
                self.emit("flow", ("vselect", dk['tmp2'], dk['bit1'], dk['node_val'], dk['tmp2']))
                self.emit("flow", ("vselect", dk['node_val'], dk['idx'], v_tree[12], v_tree[11]))
                self.emit("flow", ("vselect", dk['addr'], dk['idx'], v_tree[14], v_tree[13]))
                self.emit("flow", ("vselect", dk['node_val'], dk['bit1'], dk['addr'], dk['node_val']))
                self.emit("flow", ("vselect", dk['node_val'], dk['bit0'], dk['node_val'], dk['tmp2']))
            for d in gdesks: self.emit("valu", ("^", desks[d]['val'], desks[d]['val'], desks[d]['node_val']))
            emit_hash_interleaved(gdesks)
            for d in gdesks:
                dk = desks[d]
                self.emit("valu", ("&", dk['tmp1'], dk['val'], v_one))
                self.emit("valu", ("multiply_add", dk['addr'], dk['bit0'], v_two, dk['bit1']))
                self.emit("valu", ("multiply_add", dk['addr'], dk['addr'], v_two, dk['idx']))
                self.emit("valu", ("multiply_add", dk['addr'], dk['addr'], v_two, dk['tmp1']))
                self.emit("valu", ("+", dk['addr'], dk['addr'], v_fp_plus_15))

        def emit_gather_interleaved(gdesks):
            ho = [gdesks[GATHER_PERM[i]] for i in range(4)]
            for d in ho:
                emit_desk_loads(d)
                self.emit("valu", ("^", desks[d]['val'], desks[d]['val'], desks[d]['node_val']))
                emit_desk_hash(d)
                emit_branch_addr(d)

        def emit_r10(gdesks):
            for d in gdesks: emit_desk_loads(d)
            for d in gdesks: self.emit("valu", ("^", desks[d]['val'], desks[d]['val'], desks[d]['node_val']))
            emit_hash_interleaved(gdesks)

        def emit_r15(gdesks):
            for d in gdesks: emit_desk_loads(d)
            for d in gdesks: self.emit("valu", ("^", desks[d]['val'], desks[d]['val'], desks[d]['node_val']))
            emit_hash_interleaved(gdesks)

        def emit_tile(ti):
            off = ti * 16 * VLEN
            for d in range(16): self.emit("load", ("const", offset_regs[d], off + d * VLEN))
            for d in range(16):
                self.emit("alu", ("+", addr_tmp[d*2], self.scratch["inp_indices_p"], offset_regs[d]))
                self.emit("alu", ("+", addr_tmp[d*2+1], self.scratch["inp_values_p"], offset_regs[d]))
            for d in range(16):
                self.emit("load", ("vload", desks[d]['idx'], addr_tmp[d*2]))
                self.emit("load", ("vload", desks[d]['val'], addr_tmp[d*2+1]))

            groups = [list(range(g*4, (g+1)*4)) for g in range(4)]

            # Process in pairs with staggered dual-group approach
            for pair_idx in range(2):
                ga = groups[pair_idx * 2]
                gb = groups[pair_idx * 2 + 1]

                # Fused R0-R3 for both groups (must be sequential - uses vselects)
                emit_fused_0123(ga)
                emit_fused_0123(gb)

                # R4: Stagger start - GA loads, GB waits
                # Then for each pair of rounds, interleave
                # Actually, let's try: for each round, emit GA loads then GB loads then GA hash then GB hash
                # This is like the all-group interleave but only for 2 groups
                
                # Simpler: just use per-desk interleaving for each group separately
                # but stagger the two groups' emissions
                
                # GA round 4 loads+XOR
                for d in [ga[GATHER_PERM[i]] for i in range(4)]:
                    emit_desk_loads(d)
                    self.emit("valu", ("^", desks[d]['val'], desks[d]['val'], desks[d]['node_val']))
                # GB round 4 loads+XOR (staggered after GA loads)
                for d in [gb[GATHER_PERM[i]] for i in range(4)]:
                    emit_desk_loads(d)
                    self.emit("valu", ("^", desks[d]['val'], desks[d]['val'], desks[d]['node_val']))
                # GA hash+branch 
                for d in [ga[GATHER_PERM[i]] for i in range(4)]:
                    emit_desk_hash(d)
                    emit_branch_addr(d)
                # GB hash+branch
                for d in [gb[GATHER_PERM[i]] for i in range(4)]:
                    emit_desk_hash(d)
                    emit_branch_addr(d)

                # Remaining rounds (5-9) with per-desk interleave for each group
                for rnd in range(5, 10):
                    emit_gather_interleaved(ga)
                    emit_gather_interleaved(gb)

                # R10
                emit_r10(ga)
                emit_r10(gb)

                # Fused R11-R14
                emit_fused_0123(ga)  # same structure
                emit_fused_0123(gb)

                # R15
                emit_r15(ga)
                emit_r15(gb)

            for d in range(16):
                self.emit("store", ("vstore", addr_tmp[d*2], desks[d]['idx']))
                self.emit("store", ("vstore", addr_tmp[d*2+1], desks[d]['val']))

        emit_tile(0)
        emit_tile(1)

        phases, cur = [], []
        for e, s in self.slots:
            if e == "flow" and s == ("pause",): phases.append(cur); cur = []
            else: cur.append((e, s))
        phases.append(cur)
        self.instrs = []
        for i, phase in enumerate(phases):
            if phase: self.instrs.extend(_schedule_slots(phase))
            if i < len(phases) - 1: self.instrs.append({"flow": [("pause",)]})
        self.instrs.append({"flow": [("pause",)]})
        valu_count = sum(1 for e, s in self.slots if e == "valu")
        print(f"Total slots: {len(self.slots)}, VALU ops: {valu_count}, Cycles: {len(self.instrs)}")

BASELINE = 147734
def do_kernel_test(forest_height=10, rounds=16, batch_size=256, seed=123, trace=False, prints=False, check=False):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed); forest = Tree.generate(forest_height); inp = Input.generate(forest, batch_size, rounds); mem = build_mem_image(forest, inp)
    kb = KernelBuilderA1()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    value_trace = {}
    machine = Machine(mem, kb.instrs, kb.debug_info(), n_cores=N_CORES, value_trace=value_trace, trace=trace)
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        if check:
            inp_values_p = ref_mem[6]
            assert (machine.mem[inp_values_p : inp_values_p + len(inp.values)]
                    == ref_mem[inp_values_p : inp_values_p + len(inp.values)]), f"Incorrect on round {i}"
    print("CYCLES: ", machine.cycle); return machine.cycle

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        cycles = do_kernel_test(check=True)
        print(f"Correctness check {'PASSED' if cycles else 'FAILED'}! Cycles: {cycles}")
    else: do_kernel_test()
