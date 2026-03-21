"""Sweep hash permutation within fused rounds + gather perm=(1,2,3,0)"""
import sys, random, itertools
from collections import defaultdict

sys.path.insert(0, '/home/hestiasadmin/projects/original_performance_takehome')
from problem import *

def _vec_range(base, length=VLEN):
    return range(base, base + length)

def _slot_rw(engine, slot):
    reads, writes = [], []
    if engine == "alu": _op, dest, a1, a2 = slot; reads = [a1, a2]; writes = [dest]
    elif engine == "valu":
        op = slot[0]
        if op == "vbroadcast": reads = [slot[2]]; writes = list(_vec_range(slot[1]))
        elif op == "multiply_add": dest,a,b,c = slot[1],slot[2],slot[3],slot[4]; reads = list(_vec_range(a))+list(_vec_range(b))+list(_vec_range(c)); writes = list(_vec_range(dest))
        else: _op,dest,a1,a2 = slot; reads = list(_vec_range(a1))+list(_vec_range(a2)); writes = list(_vec_range(dest))
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
        elif op == "vselect": reads = list(_vec_range(slot[2]))+list(_vec_range(slot[3]))+list(_vec_range(slot[4])); writes = list(_vec_range(slot[1]))
    return reads, writes

def _schedule_slots(slots):
    cycles, usage = [], []
    ready_time = defaultdict(int); last_write = defaultdict(lambda: -1); last_read = defaultdict(lambda: -1)
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
        cycle = find_cycle(engine, earliest); ensure_cycle(cycle)
        cycles[cycle].setdefault(engine, []).append(slot); usage[cycle][engine] += 1
        for addr in reads:
            if last_read[addr] < cycle: last_read[addr] = cycle
        for addr in writes: last_write[addr] = cycle; ready_time[addr] = cycle + 1
    return [c for c in cycles if c]

def test_config(gather_perm, fused_hash_perm, gather_hash_perm):
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
                addr = self.alloc_scratch(name or f"c_{val}"); self.emit("load", ("const", addr, val)); self.const_map[val] = addr
            return self.const_map[val]
        def scratch_vconst(self, val, name=None):
            if val not in self.vconst_map:
                s = self.scratch_const(val); a = self.alloc_vec(name or f"v_{val}"); self.emit("valu", ("vbroadcast", a, s)); self.vconst_map[val] = a
            return self.vconst_map[val]
    
    kb = KB()
    ts = kb.alloc_scratch("ts"); ta = kb.alloc_scratch("ta")
    for vn, _ in [("n",1),("fp",4),("ip",5),("vp",6)]: kb.alloc_scratch(vn)
    for vn, idx in [("n",1),("fp",4),("ip",5),("vp",6)]:
        kb.emit("load", ("const", ts, idx)); kb.emit("load", ("load", kb.scratch[vn], ts))
    
    v0 = kb.scratch_vconst(0); v1 = kb.scratch_vconst(1); v2 = kb.scratch_vconst(2)
    vnn = kb.alloc_vec(); kb.emit("valu", ("vbroadcast", vnn, kb.scratch["n"]))
    vfp = kb.alloc_vec(); kb.emit("valu", ("vbroadcast", vfp, kb.scratch["fp"]))
    vmfp = kb.alloc_vec(); kb.emit("valu", ("-", vmfp, v1, vfp))
    vfp1 = kb.alloc_vec(); kb.emit("valu", ("+", vfp1, vfp, v1))
    vfp15 = kb.alloc_vec(); kb.emit("valu", ("+", vfp15, vfp, kb.scratch_vconst(15)))
    
    FMA = {0: 4097, 2: 33, 4: 9}
    vhc, vhs, vfm = [], [], {}
    for hi, (o1,v1_,o2,o3,v3) in enumerate(HASH_STAGES):
        vhc.append(kb.scratch_vconst(v1_))
        if hi in FMA: vfm[hi] = kb.scratch_vconst(FMA[hi]); vhs.append(None)
        else: vhs.append(kb.scratch_vconst(v3))
    
    vt = []
    for i in range(15):
        vn_ = kb.alloc_vec(); vt.append(vn_)
        kb.emit("alu", ("+", ta, kb.scratch["fp"], kb.scratch_const(i)))
        kb.emit("load", ("load", ts, ta)); kb.emit("valu", ("vbroadcast", vn_, ts))
    
    D = []
    for d in range(16):
        D.append({k: kb.alloc_vec() for k in ['idx','val','nv','addr','t1','t2','b0','b1']})
    OR = [kb.alloc_scratch() for _ in range(16)]
    AT = [kb.alloc_scratch() for _ in range(32)]
    kb.emit("flow", ("pause",))
    
    def ehash(gd, perm):
        go = [gd[perm[i]] for i in range(4)]
        for d in go:
            dk = D[d]
            kb.emit("valu",("multiply_add",dk['val'],dk['val'],vfm[0],vhc[0]))
            kb.emit("valu",("^",dk['t1'],dk['val'],vhc[1])); kb.emit("valu",(">>",dk['t2'],dk['val'],vhs[1]))
            kb.emit("valu",("^",dk['val'],dk['t1'],dk['t2']))
            kb.emit("valu",("multiply_add",dk['val'],dk['val'],vfm[2],vhc[2]))
            kb.emit("valu",("+",dk['t1'],dk['val'],vhc[3])); kb.emit("valu",("<<",dk['t2'],dk['val'],vhs[3]))
            kb.emit("valu",("^",dk['val'],dk['t1'],dk['t2']))
            kb.emit("valu",("multiply_add",dk['val'],dk['val'],vfm[4],vhc[4]))
            kb.emit("valu",("^",dk['t1'],dk['val'],vhc[5])); kb.emit("valu",(">>",dk['t2'],dk['val'],vhs[5]))
            kb.emit("valu",("^",dk['val'],dk['t1'],dk['t2']))
    
    def edhash(d):
        dk = D[d]
        kb.emit("valu",("multiply_add",dk['val'],dk['val'],vfm[0],vhc[0]))
        kb.emit("valu",("^",dk['t1'],dk['val'],vhc[1])); kb.emit("valu",(">>",dk['t2'],dk['val'],vhs[1]))
        kb.emit("valu",("^",dk['val'],dk['t1'],dk['t2']))
        kb.emit("valu",("multiply_add",dk['val'],dk['val'],vfm[2],vhc[2]))
        kb.emit("valu",("+",dk['t1'],dk['val'],vhc[3])); kb.emit("valu",("<<",dk['t2'],dk['val'],vhs[3]))
        kb.emit("valu",("^",dk['val'],dk['t1'],dk['t2']))
        kb.emit("valu",("multiply_add",dk['val'],dk['val'],vfm[4],vhc[4]))
        kb.emit("valu",("^",dk['t1'],dk['val'],vhc[5])); kb.emit("valu",(">>",dk['t2'],dk['val'],vhs[5]))
        kb.emit("valu",("^",dk['val'],dk['t1'],dk['t2']))
    
    def ebranch(d):
        dk = D[d]
        kb.emit("valu",("&",dk['t1'],dk['val'],v1)); kb.emit("valu",("multiply_add",dk['addr'],dk['addr'],v2,vmfp))
        kb.emit("valu",("+",dk['addr'],dk['addr'],dk['t1']))
    
    def efused(gd):
        for d in gd: kb.emit("valu",("^",D[d]['val'],D[d]['val'],vt[0]))
        ehash(gd, fused_hash_perm)
        for d in gd: kb.emit("valu",("&",D[d]['b0'],D[d]['val'],v1))
        for d in gd: kb.emit("flow",("vselect",D[d]['nv'],D[d]['b0'],vt[2],vt[1]))
        for d in gd: kb.emit("valu",("^",D[d]['val'],D[d]['val'],D[d]['nv']))
        ehash(gd, fused_hash_perm)
        for d in gd: kb.emit("valu",("&",D[d]['b1'],D[d]['val'],v1))
        for d in gd:
            dk=D[d]; kb.emit("flow",("vselect",dk['t2'],dk['b1'],vt[4],vt[3]))
            kb.emit("flow",("vselect",dk['nv'],dk['b1'],vt[6],vt[5]))
            kb.emit("flow",("vselect",dk['nv'],dk['b0'],dk['nv'],dk['t2']))
        for d in gd: kb.emit("valu",("^",D[d]['val'],D[d]['val'],D[d]['nv']))
        ehash(gd, fused_hash_perm)
        for d in gd: kb.emit("valu",("&",D[d]['idx'],D[d]['val'],v1))
        for d in gd:
            dk=D[d]
            kb.emit("flow",("vselect",dk['t2'],dk['idx'],vt[8],vt[7]))
            kb.emit("flow",("vselect",dk['nv'],dk['idx'],vt[10],vt[9]))
            kb.emit("flow",("vselect",dk['t2'],dk['b1'],dk['nv'],dk['t2']))
            kb.emit("flow",("vselect",dk['nv'],dk['idx'],vt[12],vt[11]))
            kb.emit("flow",("vselect",dk['addr'],dk['idx'],vt[14],vt[13]))
            kb.emit("flow",("vselect",dk['nv'],dk['b1'],dk['addr'],dk['nv']))
            kb.emit("flow",("vselect",dk['nv'],dk['b0'],dk['nv'],dk['t2']))
        for d in gd: kb.emit("valu",("^",D[d]['val'],D[d]['val'],D[d]['nv']))
        ehash(gd, fused_hash_perm)
        for d in gd:
            dk=D[d]; kb.emit("valu",("&",dk['t1'],dk['val'],v1))
            kb.emit("valu",("multiply_add",dk['addr'],dk['b0'],v2,dk['b1']))
            kb.emit("valu",("multiply_add",dk['addr'],dk['addr'],v2,dk['idx']))
            kb.emit("valu",("multiply_add",dk['addr'],dk['addr'],v2,dk['t1']))
            kb.emit("valu",("+",dk['addr'],dk['addr'],vfp15))
    
    def egather(gd):
        ho = [gd[gather_perm[i]] for i in range(4)]
        for d in ho:
            dk=D[d]
            for l in range(VLEN): kb.emit("load",("load",dk['nv']+l,dk['addr']+l))
            kb.emit("valu",("^",dk['val'],dk['val'],dk['nv']))
            edhash(d); ebranch(d)
    
    def er10(gd):
        for d in gd:
            dk=D[d]
            for l in range(VLEN): kb.emit("load",("load",dk['nv']+l,dk['addr']+l))
        for d in gd: kb.emit("valu",("^",D[d]['val'],D[d]['val'],D[d]['nv']))
        ehash(gd, fused_hash_perm)
    
    def er15(gd): er10(gd)
    
    for ti in range(2):
        off = ti*16*VLEN
        for d in range(16): kb.emit("load",("const",OR[d],off+d*VLEN))
        for d in range(16):
            kb.emit("alu",("+",AT[d*2],kb.scratch["ip"],OR[d]))
            kb.emit("alu",("+",AT[d*2+1],kb.scratch["vp"],OR[d]))
        for d in range(16):
            kb.emit("load",("vload",D[d]['idx'],AT[d*2]))
            kb.emit("load",("vload",D[d]['val'],AT[d*2+1]))
        for g in range(4):
            gd = list(range(g*4,(g+1)*4))
            efused(gd)
            for _ in range(4,10): egather(gd)
            er10(gd); efused(gd); er15(gd)
        for d in range(16):
            kb.emit("store",("vstore",AT[d*2],D[d]['idx']))
            kb.emit("store",("vstore",AT[d*2+1],D[d]['val']))
    
    phases, cur = [], []
    for e, s in kb.slots:
        if e == "flow" and s == ("pause",): phases.append(cur); cur = []
        else: cur.append((e, s))
    phases.append(cur)
    instrs = []
    for i, phase in enumerate(phases):
        if phase: instrs.extend(_schedule_slots(phase))
        if i < len(phases) - 1: instrs.append({"flow": [("pause",)]})
    instrs.append({"flow": [("pause",)]})
    return len(instrs)

# Test all 24 fused hash perms
best = 99999
best_fhp = None
gp = (1,2,3,0)
ghp = (0,3,2,1)  # gather hash perm (same as original)

for fhp in itertools.permutations(range(4)):
    c = test_config(gp, fhp, ghp)
    if c < best:
        best = c
        best_fhp = fhp
        print(f"NEW BEST: fused_hash_perm={fhp}, cycles={c}")

print(f"\nBest: gather_perm={gp}, fused_hash_perm={best_fhp}, cycles={best}")
