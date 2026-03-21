"""Sweep desk orders for Theory 330 interleaved gather"""
import sys, random, itertools
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

def test_perm(gather_perm, hash_perm):
    """Test with given gather desk order and hash desk order"""
    
    class KB:
        def __init__(self):
            self.slots = []; self.scratch = {}; self.scratch_debug = {}
            self.scratch_ptr = 0; self.const_map = {}; self.vconst_map = {}
        def debug_info(self): return DebugInfo(scratch_map=self.scratch_debug)
        def emit(self, engine, slot): self.slots.append((engine, slot))
        def alloc_scratch(self, name=None, length=1):
            addr = self.scratch_ptr
            if name: self.scratch[name] = addr; self.scratch_debug[addr] = (name, length)
            self.scratch_ptr += length; return addr
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
    
    kb = KB()
    tmp_scalar = kb.alloc_scratch("tmp_scalar"); tmp_addr = kb.alloc_scratch("tmp_addr")
    for vn, _ in [("n_nodes", 1), ("forest_values_p", 4), ("inp_indices_p", 5), ("inp_values_p", 6)]: kb.alloc_scratch(vn)
    for vn, idx in [("n_nodes", 1), ("forest_values_p", 4), ("inp_indices_p", 5), ("inp_values_p", 6)]:
        kb.emit("load", ("const", tmp_scalar, idx)); kb.emit("load", ("load", kb.scratch[vn], tmp_scalar))
    
    v_zero = kb.scratch_vconst(0); v_one = kb.scratch_vconst(1); v_two = kb.scratch_vconst(2)
    v_n_nodes = kb.alloc_vec(); kb.emit("valu", ("vbroadcast", v_n_nodes, kb.scratch["n_nodes"]))
    v_forest_p = kb.alloc_vec(); kb.emit("valu", ("vbroadcast", v_forest_p, kb.scratch["forest_values_p"]))
    v_1_minus_fp = kb.alloc_vec(); kb.emit("valu", ("-", v_1_minus_fp, v_one, v_forest_p))
    v_fp_plus_1 = kb.alloc_vec(); kb.emit("valu", ("+", v_fp_plus_1, v_forest_p, v_one))
    v_fp_plus_15 = kb.alloc_vec(); kb.emit("valu", ("+", v_fp_plus_15, v_forest_p, kb.scratch_vconst(15)))
    
    FMA_MULTIPLIERS = {0: 4097, 2: 33, 4: 9}
    v_hash_consts, v_hash_shifts, v_fma_mult = [], [], {}
    for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
        v_hash_consts.append(kb.scratch_vconst(val1))
        if hi in FMA_MULTIPLIERS: v_fma_mult[hi] = kb.scratch_vconst(FMA_MULTIPLIERS[hi]); v_hash_shifts.append(None)
        else: v_hash_shifts.append(kb.scratch_vconst(val3))
    
    v_tree = []
    for i in range(15):
        v_node = kb.alloc_vec(); v_tree.append(v_node)
        kb.emit("alu", ("+", tmp_addr, kb.scratch["forest_values_p"], kb.scratch_const(i)))
        kb.emit("load", ("load", tmp_scalar, tmp_addr)); kb.emit("valu", ("vbroadcast", v_node, tmp_scalar))
    
    desks = []
    for d in range(16):
        desks.append({k: kb.alloc_vec() for k in ['idx','val','node_val','addr','tmp1','tmp2','bit0','bit1']})
    offset_regs = [kb.alloc_scratch() for _ in range(16)]
    addr_tmp = [kb.alloc_scratch() for _ in range(32)]
    
    kb.emit("flow", ("pause",))
    
    def emit_hash_interleaved(gdesks):
        gd = [gdesks[hash_perm[0]], gdesks[hash_perm[1]], gdesks[hash_perm[2]], gdesks[hash_perm[3]]]
        for d in gd:
            desk = desks[d]
            kb.emit("valu", ("multiply_add", desk['val'], desk['val'], v_fma_mult[0], v_hash_consts[0]))
            kb.emit("valu", ("^", desk['tmp1'], desk['val'], v_hash_consts[1]))
            kb.emit("valu", (">>", desk['tmp2'], desk['val'], v_hash_shifts[1]))
            kb.emit("valu", ("^", desk['val'], desk['tmp1'], desk['tmp2']))
            kb.emit("valu", ("multiply_add", desk['val'], desk['val'], v_fma_mult[2], v_hash_consts[2]))
            kb.emit("valu", ("+", desk['tmp1'], desk['val'], v_hash_consts[3]))
            kb.emit("valu", ("<<", desk['tmp2'], desk['val'], v_hash_shifts[3]))
            kb.emit("valu", ("^", desk['val'], desk['tmp1'], desk['tmp2']))
            kb.emit("valu", ("multiply_add", desk['val'], desk['val'], v_fma_mult[4], v_hash_consts[4]))
            kb.emit("valu", ("^", desk['tmp1'], desk['val'], v_hash_consts[5]))
            kb.emit("valu", (">>", desk['tmp2'], desk['val'], v_hash_shifts[5]))
            kb.emit("valu", ("^", desk['val'], desk['tmp1'], desk['tmp2']))
    
    def emit_branch_addr(di):
        d = desks[di]
        kb.emit("valu", ("&", d['tmp1'], d['val'], v_one))
        kb.emit("valu", ("multiply_add", d['addr'], d['addr'], v_two, v_1_minus_fp))
        kb.emit("valu", ("+", d['addr'], d['addr'], d['tmp1']))
    
    def emit_fused_0123(gdesks):
        for d in gdesks: kb.emit("valu", ("^", desks[d]['val'], desks[d]['val'], v_tree[0]))
        emit_hash_interleaved(gdesks)
        for d in gdesks: kb.emit("valu", ("&", desks[d]['bit0'], desks[d]['val'], v_one))
        for d in gdesks: kb.emit("flow", ("vselect", desks[d]['node_val'], desks[d]['bit0'], v_tree[2], v_tree[1]))
        for d in gdesks: kb.emit("valu", ("^", desks[d]['val'], desks[d]['val'], desks[d]['node_val']))
        emit_hash_interleaved(gdesks)
        for d in gdesks: kb.emit("valu", ("&", desks[d]['bit1'], desks[d]['val'], v_one))
        for d in gdesks:
            dk = desks[d]
            kb.emit("flow", ("vselect", dk['tmp2'], dk['bit1'], v_tree[4], v_tree[3]))
            kb.emit("flow", ("vselect", dk['node_val'], dk['bit1'], v_tree[6], v_tree[5]))
            kb.emit("flow", ("vselect", dk['node_val'], dk['bit0'], dk['node_val'], dk['tmp2']))
        for d in gdesks: kb.emit("valu", ("^", desks[d]['val'], desks[d]['val'], desks[d]['node_val']))
        emit_hash_interleaved(gdesks)
        for d in gdesks: kb.emit("valu", ("&", desks[d]['idx'], desks[d]['val'], v_one))
        for d in gdesks:
            dk = desks[d]
            kb.emit("flow", ("vselect", dk['tmp2'], dk['idx'], v_tree[8], v_tree[7]))
            kb.emit("flow", ("vselect", dk['node_val'], dk['idx'], v_tree[10], v_tree[9]))
            kb.emit("flow", ("vselect", dk['tmp2'], dk['bit1'], dk['node_val'], dk['tmp2']))
            kb.emit("flow", ("vselect", dk['node_val'], dk['idx'], v_tree[12], v_tree[11]))
            kb.emit("flow", ("vselect", dk['addr'], dk['idx'], v_tree[14], v_tree[13]))
            kb.emit("flow", ("vselect", dk['node_val'], dk['bit1'], dk['addr'], dk['node_val']))
            kb.emit("flow", ("vselect", dk['node_val'], dk['bit0'], dk['node_val'], dk['tmp2']))
        for d in gdesks: kb.emit("valu", ("^", desks[d]['val'], desks[d]['val'], desks[d]['node_val']))
        emit_hash_interleaved(gdesks)
        for d in gdesks:
            dk = desks[d]
            kb.emit("valu", ("&", dk['tmp1'], dk['val'], v_one))
            kb.emit("valu", ("multiply_add", dk['addr'], dk['bit0'], v_two, dk['bit1']))
            kb.emit("valu", ("multiply_add", dk['addr'], dk['addr'], v_two, dk['idx']))
            kb.emit("valu", ("multiply_add", dk['addr'], dk['addr'], v_two, dk['tmp1']))
            kb.emit("valu", ("+", dk['addr'], dk['addr'], v_fp_plus_15))
    
    def emit_fused_111214(gdesks):
        emit_fused_0123(gdesks)  # Same structure
    
    def emit_gather_interleaved(gdesks):
        """Per-desk: loads, XOR, hash, branch"""
        ho = [gdesks[gather_perm[i]] for i in range(4)]
        for d in ho:
            dk = desks[d]
            for lane in range(VLEN): kb.emit("load", ("load", dk['node_val'] + lane, dk['addr'] + lane))
            kb.emit("valu", ("^", dk['val'], dk['val'], dk['node_val']))
            kb.emit("valu", ("multiply_add", dk['val'], dk['val'], v_fma_mult[0], v_hash_consts[0]))
            kb.emit("valu", ("^", dk['tmp1'], dk['val'], v_hash_consts[1]))
            kb.emit("valu", (">>", dk['tmp2'], dk['val'], v_hash_shifts[1]))
            kb.emit("valu", ("^", dk['val'], dk['tmp1'], dk['tmp2']))
            kb.emit("valu", ("multiply_add", dk['val'], dk['val'], v_fma_mult[2], v_hash_consts[2]))
            kb.emit("valu", ("+", dk['tmp1'], dk['val'], v_hash_consts[3]))
            kb.emit("valu", ("<<", dk['tmp2'], dk['val'], v_hash_shifts[3]))
            kb.emit("valu", ("^", dk['val'], dk['tmp1'], dk['tmp2']))
            kb.emit("valu", ("multiply_add", dk['val'], dk['val'], v_fma_mult[4], v_hash_consts[4]))
            kb.emit("valu", ("^", dk['tmp1'], dk['val'], v_hash_consts[5]))
            kb.emit("valu", (">>", dk['tmp2'], dk['val'], v_hash_shifts[5]))
            kb.emit("valu", ("^", dk['val'], dk['tmp1'], dk['tmp2']))
            kb.emit("valu", ("&", dk['tmp1'], dk['val'], v_one))
            kb.emit("valu", ("multiply_add", dk['addr'], dk['addr'], v_two, v_1_minus_fp))
            kb.emit("valu", ("+", dk['addr'], dk['addr'], dk['tmp1']))
    
    def emit_r10(gdesks):
        for d in gdesks:
            dk = desks[d]
            for lane in range(VLEN): kb.emit("load", ("load", dk['node_val'] + lane, dk['addr'] + lane))
        for d in gdesks: kb.emit("valu", ("^", desks[d]['val'], desks[d]['val'], desks[d]['node_val']))
        emit_hash_interleaved(gdesks)
    
    def emit_r15(gdesks):
        for d in gdesks:
            dk = desks[d]
            for lane in range(VLEN): kb.emit("load", ("load", dk['node_val'] + lane, dk['addr'] + lane))
        for d in gdesks: kb.emit("valu", ("^", desks[d]['val'], desks[d]['val'], desks[d]['node_val']))
        emit_hash_interleaved(gdesks)
    
    def emit_tile(ti):
        off = ti * 16 * VLEN
        for d in range(16): kb.emit("load", ("const", offset_regs[d], off + d * VLEN))
        for d in range(16):
            kb.emit("alu", ("+", addr_tmp[d*2], kb.scratch["inp_indices_p"], offset_regs[d]))
            kb.emit("alu", ("+", addr_tmp[d*2+1], kb.scratch["inp_values_p"], offset_regs[d]))
        for d in range(16):
            kb.emit("load", ("vload", desks[d]['idx'], addr_tmp[d*2]))
            kb.emit("load", ("vload", desks[d]['val'], addr_tmp[d*2+1]))
        
        for g in range(4):
            gd = list(range(g*4, (g+1)*4))
            emit_fused_0123(gd)
            for _ in range(4, 10): emit_gather_interleaved(gd)
            emit_r10(gd)
            emit_fused_111214(gd)
            emit_r15(gd)
        
        for d in range(16):
            kb.emit("store", ("vstore", addr_tmp[d*2], desks[d]['idx']))
            kb.emit("store", ("vstore", addr_tmp[d*2+1], desks[d]['val']))
    
    emit_tile(0); emit_tile(1)
    
    phases, cur = [], []
    for e, s in kb.slots:
        if e == "flow" and s == ("pause",): phases.append(cur); cur = []
        else: cur.append((e, s))
    phases.append(cur)
    
    kb.instrs = []
    for i, phase in enumerate(phases):
        if phase: kb.instrs.extend(_schedule_slots(phase))
        if i < len(phases) - 1: kb.instrs.append({"flow": [("pause",)]})
    kb.instrs.append({"flow": [("pause",)]})
    
    return len(kb.instrs)

# Test a selection of permutations for the gather order
# Hash order fixed at [0,3,2,1] (known best)
best_cycles = 99999
best_gp = None
hp = (0, 3, 2, 1)  # hash perm (within fused rounds)

for gp in itertools.permutations(range(4)):
    c = test_perm(gp, hp)
    if c < best_cycles:
        best_cycles = c
        best_gp = gp
        print(f"NEW BEST: gather_perm={gp}, cycles={c}")

print(f"\nBest: gather_perm={best_gp}, hash_perm={hp}, cycles={best_cycles}")

# Now try different hash perms with the best gather perm
print(f"\nSweeping hash perms with gather_perm={best_gp}:")
best_hp = hp
for hp2 in itertools.permutations(range(4)):
    c = test_perm(best_gp, hp2)
    if c < best_cycles:
        best_cycles = c
        best_hp = hp2
        print(f"NEW BEST: hash_perm={hp2}, cycles={c}")

print(f"\nFinal best: gather_perm={best_gp}, hash_perm={best_hp}, cycles={best_cycles}")
