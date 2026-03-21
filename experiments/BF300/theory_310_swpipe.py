"""
Theory 310: Software pipelining - overlap loads of round N+1 with hash of round N.

Instead of:
  R4: loads -> XOR -> hash -> branch
  R5: loads -> XOR -> hash -> branch

Do:
  Prologue: R4 loads
  R4-R8: XOR -> hash -> branch -> R(N+1) loads  [loads overlap with hash]
  R9: XOR -> hash -> branch (no loads after)

The key: after emitting the branch (which updates addr), immediately emit
the next round's loads before emitting the next round's XOR/hash.
This gives the scheduler more opportunity to schedule loads during hash computation.

Actually, the problem is that the BRANCH produces the addr needed for the NEXT loads.
So loads for R5 depend on the branch of R4. We can't start R5 loads until R4 branch completes.

BUT: within R4, the branch only needs 3 VALU ops AFTER hash. The hash takes 12 VALU/desk * 4 desks = 48 VALU.
With the branch at the end, the loads for R5 start after all 48+12 = ~60 VALU ops (plus 32 loads for R4).

What if we emit the branch for desk 0 BEFORE the hash for desk 3?
Then desk 0's loads for R5 can start during desk 3's hash.

Software pipeline the DESKS within a round:
  desk0: loads -> XOR -> hash -> branch
  desk1: loads -> XOR -> hash -> branch  
  desk2: loads -> XOR -> hash -> branch
  desk3: loads -> XOR -> hash -> branch -> (desk0 next round loads start here)

But currently we interleave hash across desks for better VALU utilization.
Let me try per-desk sequential processing instead of interleaved.
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

        v_zero = self.scratch_vconst(0, "v_zero")
        v_one = self.scratch_vconst(1, "v_one")
        v_two = self.scratch_vconst(2, "v_two")
        v_n_nodes = self.alloc_vec("v_n_nodes")
        self.emit("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]))

        v_forest_p = self.alloc_vec("v_forest_p")
        self.emit("valu", ("vbroadcast", v_forest_p, self.scratch["forest_values_p"]))
        v_1_minus_fp = self.alloc_vec("v_1_minus_fp")
        self.emit("valu", ("-", v_1_minus_fp, v_one, v_forest_p))
        v_fp_plus_1 = self.alloc_vec("v_fp_plus_1")
        self.emit("valu", ("+", v_fp_plus_1, v_forest_p, v_one))
        v_fp_plus_15 = self.alloc_vec("v_fp_plus_15")
        self.emit("valu", ("+", v_fp_plus_15, v_forest_p, self.scratch_vconst(15, "v_fifteen")))

        FMA_MULTIPLIERS = {0: 4097, 2: 33, 4: 9}
        v_hash_consts, v_hash_shifts, v_fma_mult = [], [], {}
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            v_hash_consts.append(self.scratch_vconst(val1, f"v_hash_const_{hi}"))
            if hi in FMA_MULTIPLIERS:
                v_fma_mult[hi] = self.scratch_vconst(FMA_MULTIPLIERS[hi], f"v_fma_mult_{hi}")
                v_hash_shifts.append(None)
            else:
                v_hash_shifts.append(self.scratch_vconst(val3, f"v_hash_shift_{hi}"))

        NUM_PRELOADED = 15
        v_tree = []
        for i in range(NUM_PRELOADED):
            v_node = self.alloc_vec(f"v_tree_{i}"); v_tree.append(v_node)
            self.emit("alu", ("+", tmp_addr, self.scratch["forest_values_p"], self.scratch_const(i)))
            self.emit("load", ("load", tmp_scalar, tmp_addr))
            self.emit("valu", ("vbroadcast", v_node, tmp_scalar))

        NUM_DESKS = 16
        desks = []
        for d in range(NUM_DESKS):
            desks.append({
                'idx': self.alloc_vec(f"v_idx_{d}"), 'val': self.alloc_vec(f"v_val_{d}"),
                'node_val': self.alloc_vec(f"v_node_{d}"), 'addr': self.alloc_vec(f"v_addr_{d}"),
                'tmp1': self.alloc_vec(f"v_tmp1_{d}"), 'tmp2': self.alloc_vec(f"v_tmp2_{d}"),
                'bit0': self.alloc_vec(f"v_bit0_{d}"), 'bit1': self.alloc_vec(f"v_bit1_{d}"),
            })
        offset_regs = [self.alloc_scratch(f"off_{d}") for d in range(NUM_DESKS)]
        addr_tmp = [self.alloc_scratch(f"addr_tmp_{i}") for i in range(32)]
        print(f"Scratch usage: {self.scratch_ptr} / {SCRATCH_SIZE}")
        self.emit("flow", ("pause",))

        def emit_hash_per_desk(desk_idx):
            """Hash a single desk (12 VALU ops)"""
            desk = desks[desk_idx]
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

        def emit_hash_interleaved(group_desks):
            gd = [group_desks[0], group_desks[3], group_desks[2], group_desks[1]]
            for d in gd: emit_hash_per_desk(d)

        def emit_branch_addr_tracking(desk_idx):
            d = desks[desk_idx]
            self.emit("valu", ("&", d['tmp1'], d['val'], v_one))
            self.emit("valu", ("multiply_add", d['addr'], d['addr'], v_two, v_1_minus_fp))
            self.emit("valu", ("+", d['addr'], d['addr'], d['tmp1']))

        def emit_xor_with_node(desk_idx, node_vec):
            self.emit("valu", ("^", desks[desk_idx]['val'], desks[desk_idx]['val'], node_vec))

        def emit_loads_for_desk(desk_idx):
            desk = desks[desk_idx]
            for lane in range(VLEN):
                self.emit("load", ("load", desk['node_val'] + lane, desk['addr'] + lane))

        def emit_rounds_0_1_2_3_fused(group_desks):
            for d in group_desks: emit_xor_with_node(d, v_tree[0])
            emit_hash_interleaved(group_desks)
            for d in group_desks: self.emit("valu", ("&", desks[d]['bit0'], desks[d]['val'], v_one))
            for d in group_desks: self.emit("flow", ("vselect", desks[d]['node_val'], desks[d]['bit0'], v_tree[2], v_tree[1]))
            for d in group_desks: self.emit("valu", ("^", desks[d]['val'], desks[d]['val'], desks[d]['node_val']))
            emit_hash_interleaved(group_desks)
            for d in group_desks: self.emit("valu", ("&", desks[d]['bit1'], desks[d]['val'], v_one))
            for d in group_desks:
                desk = desks[d]
                self.emit("flow", ("vselect", desk['tmp2'], desk['bit1'], v_tree[4], v_tree[3]))
                self.emit("flow", ("vselect", desk['node_val'], desk['bit1'], v_tree[6], v_tree[5]))
                self.emit("flow", ("vselect", desk['node_val'], desk['bit0'], desk['node_val'], desk['tmp2']))
            for d in group_desks: self.emit("valu", ("^", desks[d]['val'], desks[d]['val'], desks[d]['node_val']))
            emit_hash_interleaved(group_desks)
            for d in group_desks: self.emit("valu", ("&", desks[d]['idx'], desks[d]['val'], v_one))
            for d in group_desks:
                desk = desks[d]
                self.emit("flow", ("vselect", desk['tmp2'],     desk['idx'], v_tree[8],  v_tree[7]))
                self.emit("flow", ("vselect", desk['node_val'], desk['idx'], v_tree[10], v_tree[9]))
                self.emit("flow", ("vselect", desk['tmp2'],     desk['bit1'], desk['node_val'], desk['tmp2']))
                self.emit("flow", ("vselect", desk['node_val'], desk['idx'], v_tree[12], v_tree[11]))
                self.emit("flow", ("vselect", desk['addr'],     desk['idx'], v_tree[14], v_tree[13]))
                self.emit("flow", ("vselect", desk['node_val'], desk['bit1'], desk['addr'], desk['node_val']))
                self.emit("flow", ("vselect", desk['node_val'], desk['bit0'], desk['node_val'], desk['tmp2']))
            for d in group_desks: self.emit("valu", ("^", desks[d]['val'], desks[d]['val'], desks[d]['node_val']))
            emit_hash_interleaved(group_desks)
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("&", desk['tmp1'], desk['val'], v_one))
                self.emit("valu", ("multiply_add", desk['addr'], desk['bit0'], v_two, desk['bit1']))
                self.emit("valu", ("multiply_add", desk['addr'], desk['addr'], v_two, desk['idx']))
                self.emit("valu", ("multiply_add", desk['addr'], desk['addr'], v_two, desk['tmp1']))
                self.emit("valu", ("+", desk['addr'], desk['addr'], v_fp_plus_15))

        def emit_rounds_11_12_13_14_fused(group_desks):
            for d in group_desks: emit_xor_with_node(d, v_tree[0])
            emit_hash_interleaved(group_desks)
            for d in group_desks: self.emit("valu", ("&", desks[d]['bit0'], desks[d]['val'], v_one))
            for d in group_desks: self.emit("flow", ("vselect", desks[d]['node_val'], desks[d]['bit0'], v_tree[2], v_tree[1]))
            for d in group_desks: self.emit("valu", ("^", desks[d]['val'], desks[d]['val'], desks[d]['node_val']))
            emit_hash_interleaved(group_desks)
            for d in group_desks: self.emit("valu", ("&", desks[d]['bit1'], desks[d]['val'], v_one))
            for d in group_desks:
                desk = desks[d]
                self.emit("flow", ("vselect", desk['tmp2'], desk['bit1'], v_tree[4], v_tree[3]))
                self.emit("flow", ("vselect", desk['node_val'], desk['bit1'], v_tree[6], v_tree[5]))
                self.emit("flow", ("vselect", desk['node_val'], desk['bit0'], desk['node_val'], desk['tmp2']))
            for d in group_desks: self.emit("valu", ("^", desks[d]['val'], desks[d]['val'], desks[d]['node_val']))
            emit_hash_interleaved(group_desks)
            for d in group_desks: self.emit("valu", ("&", desks[d]['idx'], desks[d]['val'], v_one))
            for d in group_desks:
                desk = desks[d]
                self.emit("flow", ("vselect", desk['tmp2'],     desk['idx'], v_tree[8],  v_tree[7]))
                self.emit("flow", ("vselect", desk['node_val'], desk['idx'], v_tree[10], v_tree[9]))
                self.emit("flow", ("vselect", desk['tmp2'],     desk['bit1'], desk['node_val'], desk['tmp2']))
                self.emit("flow", ("vselect", desk['node_val'], desk['idx'], v_tree[12], v_tree[11]))
                self.emit("flow", ("vselect", desk['addr'],     desk['idx'], v_tree[14], v_tree[13]))
                self.emit("flow", ("vselect", desk['node_val'], desk['bit1'], desk['addr'], desk['node_val']))
                self.emit("flow", ("vselect", desk['node_val'], desk['bit0'], desk['node_val'], desk['tmp2']))
            for d in group_desks: self.emit("valu", ("^", desks[d]['val'], desks[d]['val'], desks[d]['node_val']))
            emit_hash_interleaved(group_desks)
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("&", desk['tmp1'], desk['val'], v_one))
                self.emit("valu", ("multiply_add", desk['addr'], desk['bit0'], v_two, desk['bit1']))
                self.emit("valu", ("multiply_add", desk['addr'], desk['addr'], v_two, desk['idx']))
                self.emit("valu", ("multiply_add", desk['addr'], desk['addr'], v_two, desk['tmp1']))
                self.emit("valu", ("+", desk['addr'], desk['addr'], v_fp_plus_15))

        def emit_gather_rounds_pipelined(group_desks, start_round, end_round):
            """Software pipelined gather rounds: emit loads early, overlap with previous hash"""
            gd = group_desks
            # Emit desk-by-desk: after desk D's branch, emit desk D's loads for next round
            # This allows loads to overlap with other desks' hash computation
            
            for rnd in range(start_round, end_round + 1):
                is_last = (rnd == end_round)
                
                # Loads (addr already ready from previous branch)
                for d in gd:
                    emit_loads_for_desk(d)
                
                # XOR
                for d in gd:
                    self.emit("valu", ("^", desks[d]['val'], desks[d]['val'], desks[d]['node_val']))
                
                if not is_last:
                    # Hash + branch with early next-round loads
                    # Emit hash desk-by-desk, with branch immediately after each desk
                    # After branch, emit that desk's loads for next round
                    hash_order = [gd[0], gd[3], gd[2], gd[1]]
                    for d in hash_order:
                        emit_hash_per_desk(d)
                        emit_branch_addr_tracking(d)
                else:
                    # Last round: no branch needed for R10
                    emit_hash_interleaved(gd)

        def emit_round_15_final(group_desks):
            for d in group_desks:
                emit_loads_for_desk(d)
            for d in group_desks:
                self.emit("valu", ("^", desks[d]['val'], desks[d]['val'], desks[d]['node_val']))
            emit_hash_interleaved(group_desks)

        def emit_tile_interleaved(tile_idx):
            tile_offset = tile_idx * NUM_DESKS * VLEN
            for d in range(NUM_DESKS): self.emit("load", ("const", offset_regs[d], tile_offset + d * VLEN))
            for d in range(NUM_DESKS):
                self.emit("alu", ("+", addr_tmp[d*2], self.scratch["inp_indices_p"], offset_regs[d]))
                self.emit("alu", ("+", addr_tmp[d*2+1], self.scratch["inp_values_p"], offset_regs[d]))
            for d in range(NUM_DESKS):
                self.emit("load", ("vload", desks[d]['idx'], addr_tmp[d*2]))
                self.emit("load", ("vload", desks[d]['val'], addr_tmp[d*2+1]))

            GROUP_SIZE = 4
            all_groups = [list(range(g*GROUP_SIZE, (g+1)*GROUP_SIZE)) for g in range(NUM_DESKS//GROUP_SIZE)]

            for gd in all_groups:
                emit_rounds_0_1_2_3_fused(gd)
                # Pipelined gather rounds 4-10
                emit_gather_rounds_pipelined(gd, 4, 10)
                emit_rounds_11_12_13_14_fused(gd)
                emit_round_15_final(gd)

            for d in range(NUM_DESKS):
                self.emit("store", ("vstore", addr_tmp[d*2], desks[d]['idx']))
                self.emit("store", ("vstore", addr_tmp[d*2+1], desks[d]['val']))

        emit_tile_interleaved(0)
        emit_tile_interleaved(1)

        phases, current_phase = [], []
        for engine, slot in self.slots:
            if engine == "flow" and slot == ("pause",):
                phases.append(current_phase); current_phase = []
            else: current_phase.append((engine, slot))
        phases.append(current_phase)
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
    random.seed(seed)
    forest = Tree.generate(forest_height); inp = Input.generate(forest, batch_size, rounds); mem = build_mem_image(forest, inp)
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
    print("CYCLES: ", machine.cycle)
    return machine.cycle

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    if args.check:
        cycles = do_kernel_test(check=True)
        print(f"Correctness check {'PASSED' if cycles else 'FAILED'}! Cycles: {cycles}")
    else: do_kernel_test(trace=args.trace)
