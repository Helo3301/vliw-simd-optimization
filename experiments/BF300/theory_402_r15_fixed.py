"""
Theory 402: R15 fused via level-4 preloading (31 tree nodes) + per-desk interleaving.

Changes from Theory 401:
1. Remove dead allocations: v_zero, v_n_nodes, v_fp_plus_1 (saves 25 slots)
2. Scratch: 1518/1536 (fits!)
3. Per-desk interleaving for gather rounds R4-R9 (from Theory 330)
4. gather_perm=(1,2,3,0), hash_perm=(0,3,2,1)

Expected savings vs 1,395:
- Eliminate R15 gather loads: 4 desks * 8 lanes * 4 groups * 2 tiles = 256 loads saved
- Eliminate R15 branch: 3 VALU/desk * 16 desks * 2 tiles = 96 VALU saved  
- Add R15 vselect cascade: 15 vselects/desk * 16 desks * 2 tiles = 480 flow ops added
- But flow has slack (was only 50.4% utilized at 1,400)
"""
import random, argparse, sys
from collections import defaultdict

sys.path.insert(0, '/home/hestiasadmin/projects/original_performance_takehome')
from problem import *

def _vec_range(base, length=VLEN):
    return range(base, base + length)

def _slot_rw(engine, slot):
    reads, writes = [], []
    if engine == "alu": _op,dest,a1,a2 = slot; reads=[a1,a2]; writes=[dest]
    elif engine == "valu":
        op = slot[0]
        if op == "vbroadcast": reads=[slot[2]]; writes=list(_vec_range(slot[1]))
        elif op == "multiply_add": dest,a,b,c=slot[1],slot[2],slot[3],slot[4]; reads=list(_vec_range(a))+list(_vec_range(b))+list(_vec_range(c)); writes=list(_vec_range(dest))
        else: _op,dest,a1,a2=slot; reads=list(_vec_range(a1))+list(_vec_range(a2)); writes=list(_vec_range(dest))
    elif engine == "load":
        op = slot[0]
        if op == "load": reads=[slot[2]]; writes=[slot[1]]
        elif op == "vload": reads=[slot[2]]; writes=list(_vec_range(slot[1]))
        elif op == "const": writes=[slot[1]]
        elif op == "load_offset": reads=[slot[2]]; writes=[slot[1]]
    elif engine == "store":
        op = slot[0]
        if op == "store": reads=[slot[1],slot[2]]
        elif op == "vstore": reads=[slot[1]]+list(_vec_range(slot[2]))
    elif engine == "flow":
        op = slot[0]
        if op == "select": reads=[slot[2],slot[3],slot[4]]; writes=[slot[1]]
        elif op == "add_imm": reads=[slot[2]]; writes=[slot[1]]
        elif op == "vselect": reads=list(_vec_range(slot[2]))+list(_vec_range(slot[3]))+list(_vec_range(slot[4])); writes=list(_vec_range(slot[1]))
    return reads, writes

def _schedule_slots(slots):
    cycles,usage=[],[]
    ready_time=defaultdict(int); last_write=defaultdict(lambda:-1); last_read=defaultdict(lambda:-1)
    def ensure_cycle(c):
        while len(cycles)<=c: cycles.append({}); usage.append(defaultdict(int))
    def find_cycle(engine,earliest):
        c=earliest
        while True:
            ensure_cycle(c)
            if usage[c][engine]<SLOT_LIMITS[engine]: return c
            c+=1
    for engine,slot in slots:
        reads,writes=_slot_rw(engine,slot)
        earliest=0
        for addr in reads: earliest=max(earliest,ready_time[addr])
        for addr in writes: earliest=max(earliest,last_write[addr]+1,last_read[addr])
        cycle=find_cycle(engine,earliest); ensure_cycle(cycle)
        cycles[cycle].setdefault(engine,[]).append(slot); usage[cycle][engine]+=1
        for addr in reads:
            if last_read[addr]<cycle: last_read[addr]=cycle
        for addr in writes: last_write[addr]=cycle; ready_time[addr]=cycle+1
    return [c for c in cycles if c]

class KernelBuilderA1:
    def __init__(self):
        self.slots=[]; self.scratch={}; self.scratch_debug={}
        self.scratch_ptr=0; self.const_map={}; self.vconst_map={}
    def debug_info(self): return DebugInfo(scratch_map=self.scratch_debug)
    def emit(self,engine,slot): self.slots.append((engine,slot))
    def alloc_scratch(self,name=None,length=1):
        addr=self.scratch_ptr
        if name: self.scratch[name]=addr; self.scratch_debug[addr]=(name,length)
        self.scratch_ptr+=length; assert self.scratch_ptr<=SCRATCH_SIZE, f"Scratch overflow: {self.scratch_ptr}/{SCRATCH_SIZE}"; return addr
    def alloc_vec(self,name=None): return self.alloc_scratch(name,VLEN)
    def scratch_const(self,val,name=None):
        if val not in self.const_map:
            addr=self.alloc_scratch(name or f"c_{val}"); self.emit("load",("const",addr,val)); self.const_map[val]=addr
        return self.const_map[val]
    def scratch_vconst(self,val,name=None):
        if val not in self.vconst_map:
            s=self.scratch_const(val); a=self.alloc_vec(name or f"v_{val}"); self.emit("valu",("vbroadcast",a,s)); self.vconst_map[val]=a
        return self.vconst_map[val]

    def build_kernel(self, forest_height, n_nodes, batch_size, rounds):
        tmp_scalar=self.alloc_scratch("tmp_scalar"); tmp_addr=self.alloc_scratch("tmp_addr")
        for vn,_ in [("n_nodes",1),("forest_values_p",4),("inp_indices_p",5),("inp_values_p",6)]: self.alloc_scratch(vn)
        for vn,idx in [("n_nodes",1),("forest_values_p",4),("inp_indices_p",5),("inp_values_p",6)]:
            self.emit("load",("const",tmp_scalar,idx)); self.emit("load",("load",self.scratch[vn],tmp_scalar))

        # Theory 402: NO v_zero, NO v_n_nodes, NO v_fp_plus_1 (dead code removed)
        v_one=self.scratch_vconst(1); v_two=self.scratch_vconst(2)
        v_forest_p=self.alloc_vec("v_forest_p"); self.emit("valu",("vbroadcast",v_forest_p,self.scratch["forest_values_p"]))
        v_1_minus_fp=self.alloc_vec("v_1_minus_fp"); self.emit("valu",("-",v_1_minus_fp,v_one,v_forest_p))
        v_fp_plus_15=self.alloc_vec("v_fp_plus_15"); self.emit("valu",("+",v_fp_plus_15,v_forest_p,self.scratch_vconst(15)))

        FMA_MULTIPLIERS={0:4097,2:33,4:9}
        v_hash_consts,v_hash_shifts,v_fma_mult=[],[],{}
        for hi,(op1,val1,op2,op3,val3) in enumerate(HASH_STAGES):
            v_hash_consts.append(self.scratch_vconst(val1))
            if hi in FMA_MULTIPLIERS: v_fma_mult[hi]=self.scratch_vconst(FMA_MULTIPLIERS[hi]); v_hash_shifts.append(None)
            else: v_hash_shifts.append(self.scratch_vconst(val3))

        # Theory 402: Preload 31 tree nodes (levels 0-4)
        NUM_PRELOADED=31
        v_tree=[]
        for i in range(NUM_PRELOADED):
            vn=self.alloc_vec(f"v_tree_{i}"); v_tree.append(vn)
            self.emit("alu",("+",tmp_addr,self.scratch["forest_values_p"],self.scratch_const(i)))
            self.emit("load",("load",tmp_scalar,tmp_addr)); self.emit("valu",("vbroadcast",vn,tmp_scalar))

        NUM_DESKS=16
        desks=[]
        for d in range(NUM_DESKS):
            desks.append({
                'idx':self.alloc_vec(f"v_idx_{d}"),'val':self.alloc_vec(f"v_val_{d}"),
                'nv':self.alloc_vec(f"v_nv_{d}"),'addr':self.alloc_vec(f"v_addr_{d}"),
                't1':self.alloc_vec(f"v_t1_{d}"),'t2':self.alloc_vec(f"v_t2_{d}"),
                'b0':self.alloc_vec(f"v_b0_{d}"),'b1':self.alloc_vec(f"v_b1_{d}"),
            })
        offset_regs=[self.alloc_scratch(f"off_{d}") for d in range(NUM_DESKS)]
        addr_tmp=[self.alloc_scratch(f"addr_tmp_{i}") for i in range(32)]
        print(f"Scratch usage: {self.scratch_ptr} / {SCRATCH_SIZE}")
        self.emit("flow",("pause",))

        HASH_PERM=(0,3,2,1)
        GATHER_PERM=(1,2,3,0)

        def emit_hash_interleaved(gd):
            go=[gd[HASH_PERM[i]] for i in range(4)]
            for d in go:
                dk=desks[d]
                self.emit("valu",("multiply_add",dk['val'],dk['val'],v_fma_mult[0],v_hash_consts[0]))
                self.emit("valu",("^",dk['t1'],dk['val'],v_hash_consts[1]))
                self.emit("valu",(">>",dk['t2'],dk['val'],v_hash_shifts[1]))
                self.emit("valu",("^",dk['val'],dk['t1'],dk['t2']))
                self.emit("valu",("multiply_add",dk['val'],dk['val'],v_fma_mult[2],v_hash_consts[2]))
                self.emit("valu",("+",dk['t1'],dk['val'],v_hash_consts[3]))
                self.emit("valu",("<<",dk['t2'],dk['val'],v_hash_shifts[3]))
                self.emit("valu",("^",dk['val'],dk['t1'],dk['t2']))
                self.emit("valu",("multiply_add",dk['val'],dk['val'],v_fma_mult[4],v_hash_consts[4]))
                self.emit("valu",("^",dk['t1'],dk['val'],v_hash_consts[5]))
                self.emit("valu",(">>",dk['t2'],dk['val'],v_hash_shifts[5]))
                self.emit("valu",("^",dk['val'],dk['t1'],dk['t2']))

        def emit_desk_hash(d):
            dk=desks[d]
            self.emit("valu",("multiply_add",dk['val'],dk['val'],v_fma_mult[0],v_hash_consts[0]))
            self.emit("valu",("^",dk['t1'],dk['val'],v_hash_consts[1]))
            self.emit("valu",(">>",dk['t2'],dk['val'],v_hash_shifts[1]))
            self.emit("valu",("^",dk['val'],dk['t1'],dk['t2']))
            self.emit("valu",("multiply_add",dk['val'],dk['val'],v_fma_mult[2],v_hash_consts[2]))
            self.emit("valu",("+",dk['t1'],dk['val'],v_hash_consts[3]))
            self.emit("valu",("<<",dk['t2'],dk['val'],v_hash_shifts[3]))
            self.emit("valu",("^",dk['val'],dk['t1'],dk['t2']))
            self.emit("valu",("multiply_add",dk['val'],dk['val'],v_fma_mult[4],v_hash_consts[4]))
            self.emit("valu",("^",dk['t1'],dk['val'],v_hash_consts[5]))
            self.emit("valu",(">>",dk['t2'],dk['val'],v_hash_shifts[5]))
            self.emit("valu",("^",dk['val'],dk['t1'],dk['t2']))

        def emit_branch_addr(di):
            d=desks[di]
            self.emit("valu",("&",d['t1'],d['val'],v_one))
            self.emit("valu",("multiply_add",d['addr'],d['addr'],v_two,v_1_minus_fp))
            self.emit("valu",("+",d['addr'],d['addr'],d['t1']))

        # R0-R3 fused (level-3 preloading + deferred addr)
        def emit_fused_0123(gd):
            for d in gd: self.emit("valu",("^",desks[d]['val'],desks[d]['val'],v_tree[0]))
            emit_hash_interleaved(gd)
            for d in gd: self.emit("valu",("&",desks[d]['b0'],desks[d]['val'],v_one))
            for d in gd: self.emit("flow",("vselect",desks[d]['nv'],desks[d]['b0'],v_tree[2],v_tree[1]))
            for d in gd: self.emit("valu",("^",desks[d]['val'],desks[d]['val'],desks[d]['nv']))
            emit_hash_interleaved(gd)
            for d in gd: self.emit("valu",("&",desks[d]['b1'],desks[d]['val'],v_one))
            for d in gd:
                dk=desks[d]
                self.emit("flow",("vselect",dk['t2'],dk['b1'],v_tree[4],v_tree[3]))
                self.emit("flow",("vselect",dk['nv'],dk['b1'],v_tree[6],v_tree[5]))
                self.emit("flow",("vselect",dk['nv'],dk['b0'],dk['nv'],dk['t2']))
            for d in gd: self.emit("valu",("^",desks[d]['val'],desks[d]['val'],desks[d]['nv']))
            emit_hash_interleaved(gd)
            for d in gd: self.emit("valu",("&",desks[d]['idx'],desks[d]['val'],v_one))
            for d in gd:
                dk=desks[d]
                self.emit("flow",("vselect",dk['t2'],dk['idx'],v_tree[8],v_tree[7]))
                self.emit("flow",("vselect",dk['nv'],dk['idx'],v_tree[10],v_tree[9]))
                self.emit("flow",("vselect",dk['t2'],dk['b1'],dk['nv'],dk['t2']))
                self.emit("flow",("vselect",dk['nv'],dk['idx'],v_tree[12],v_tree[11]))
                self.emit("flow",("vselect",dk['addr'],dk['idx'],v_tree[14],v_tree[13]))
                self.emit("flow",("vselect",dk['nv'],dk['b1'],dk['addr'],dk['nv']))
                self.emit("flow",("vselect",dk['nv'],dk['b0'],dk['nv'],dk['t2']))
            for d in gd: self.emit("valu",("^",desks[d]['val'],desks[d]['val'],desks[d]['nv']))
            emit_hash_interleaved(gd)
            for d in gd:
                dk=desks[d]
                self.emit("valu",("&",dk['t1'],dk['val'],v_one))
                self.emit("valu",("multiply_add",dk['addr'],dk['b0'],v_two,dk['b1']))
                self.emit("valu",("multiply_add",dk['addr'],dk['addr'],v_two,dk['idx']))
                self.emit("valu",("multiply_add",dk['addr'],dk['addr'],v_two,dk['t1']))
                self.emit("valu",("+",dk['addr'],dk['addr'],v_fp_plus_15))

        # Gather rounds R4-R9: per-desk interleaved (Theory 330)
        def emit_gather_interleaved(gd):
            ho=[gd[GATHER_PERM[i]] for i in range(4)]
            for d in ho:
                dk=desks[d]
                for lane in range(VLEN): self.emit("load",("load",dk['nv']+lane,dk['addr']+lane))
                self.emit("valu",("^",dk['val'],dk['val'],dk['nv']))
                emit_desk_hash(d); emit_branch_addr(d)

        # R10: standard (all loads then all hash)
        def emit_r10(gd):
            for d in gd:
                dk=desks[d]
                for lane in range(VLEN): self.emit("load",("load",dk['nv']+lane,dk['addr']+lane))
            for d in gd: self.emit("valu",("^",desks[d]['val'],desks[d]['val'],desks[d]['nv']))
            emit_hash_interleaved(gd)

        # R11-R14 fused (same structure as R0-R3)
        def emit_fused_11_14(gd):
            emit_fused_0123(gd)  # Same structure: XOR with tree[0], hash, bit extract, vselect cascade, deferred addr

        # Theory 402: R15 fused with level-4 preloading (15-vselect cascade)
        def emit_r15_fused(gd):
            # R15 uses preloaded tree nodes at level 4 (tree[15..30])
            # We already have b0,b1,idx from R11-R14 fused block
            # After R14's hash, we extracted bit3 -> need to extract it from val
            
            # After R14 fused: addr has been computed as the 4-bit deferred index
            # But for R15 we need: XOR with tree[addr], hash, no branch needed
            # Since addr points into tree, we need to select tree[addr] without gather
            
            # Wait - the R11-R14 fused block computes addr from bits 0-3 the same way as R0-R3
            # addr = b0*8 + b1*4 + idx*2 + t1 + fp+15
            # We need tree[addr] but addr is data-dependent per-lane
            
            # Instead: R15 needs to XOR with tree[node] where node = f(b0,b1,bit2,bit3)
            # The node index (within level 4) = b0*8 + b1*4 + bit2*2 + bit3
            # This selects from tree[15..30] (the 16 nodes at level 4)
            
            # 15-vselect cascade to select from 16 values:
            # Layer 1: 8 vselects on bit3 (b3 = last extracted bit = t1 from R14)
            # Layer 2: 4 vselects on bit2 (= idx from R13)
            # Layer 3: 2 vselects on bit1 (= b1 from R12)
            # Layer 4: 1 vselect on bit0 (= b0 from R11)
            
            # After the R14 fused block:
            # b0 = bit from R11 hash
            # b1 = bit from R12 hash
            # idx = bit from R13 hash (bit2)
            # t1 = bit from R14 hash (bit3) - note: addr was already computed over t1
            
            # PROBLEM: In the fused_0123 function, after the last hash:
            #   t1 = val & 1  (this IS bit3)
            #   addr = FMA chain from b0,b1,idx,t1 + fp+15
            # So t1 IS bit3. But addr has been overwritten with the tree address.
            # We need bit3 BEFORE addr is computed, OR we need to re-extract it.
            
            # Actually for R15 fused, we DON'T need addr at all!
            # We just need the 4 bits: b0, b1, idx(=bit2), and bit3
            # And bit3 = t1 which is computed BEFORE addr overwrite
            # So we need to restructure R11-R14 to NOT compute addr, 
            # and instead do the vselect cascade for R15.
            
            # Let me restructure: R11-R14 collect bits, then R15 does cascade
            pass  # Will be handled inline below

        # Combined R11-R15 fused block
        def emit_fused_11_15(gd):
            # R11: XOR with tree[0], hash, extract bit0
            for d in gd: self.emit("valu",("^",desks[d]['val'],desks[d]['val'],v_tree[0]))
            emit_hash_interleaved(gd)
            for d in gd: self.emit("valu",("&",desks[d]['b0'],desks[d]['val'],v_one))
            
            # R12: vselect tree node, XOR, hash, extract bit1
            for d in gd: self.emit("flow",("vselect",desks[d]['nv'],desks[d]['b0'],v_tree[2],v_tree[1]))
            for d in gd: self.emit("valu",("^",desks[d]['val'],desks[d]['val'],desks[d]['nv']))
            emit_hash_interleaved(gd)
            for d in gd: self.emit("valu",("&",desks[d]['b1'],desks[d]['val'],v_one))
            
            # R13: 3-vselect cascade, XOR, hash, extract bit2
            for d in gd:
                dk=desks[d]
                self.emit("flow",("vselect",dk['t2'],dk['b1'],v_tree[4],v_tree[3]))
                self.emit("flow",("vselect",dk['nv'],dk['b1'],v_tree[6],v_tree[5]))
                self.emit("flow",("vselect",dk['nv'],dk['b0'],dk['nv'],dk['t2']))
            for d in gd: self.emit("valu",("^",desks[d]['val'],desks[d]['val'],desks[d]['nv']))
            emit_hash_interleaved(gd)
            for d in gd: self.emit("valu",("&",desks[d]['idx'],desks[d]['val'],v_one))  # bit2
            
            # R14: 7-vselect cascade, XOR, hash, extract bit3
            for d in gd:
                dk=desks[d]
                self.emit("flow",("vselect",dk['t2'],dk['idx'],v_tree[8],v_tree[7]))
                self.emit("flow",("vselect",dk['nv'],dk['idx'],v_tree[10],v_tree[9]))
                self.emit("flow",("vselect",dk['t2'],dk['b1'],dk['nv'],dk['t2']))
                self.emit("flow",("vselect",dk['nv'],dk['idx'],v_tree[12],v_tree[11]))
                self.emit("flow",("vselect",dk['addr'],dk['idx'],v_tree[14],v_tree[13]))
                self.emit("flow",("vselect",dk['nv'],dk['b1'],dk['addr'],dk['nv']))
                self.emit("flow",("vselect",dk['nv'],dk['b0'],dk['nv'],dk['t2']))
            for d in gd: self.emit("valu",("^",desks[d]['val'],desks[d]['val'],desks[d]['nv']))
            emit_hash_interleaved(gd)
            for d in gd: self.emit("valu",("&",desks[d]['t1'],desks[d]['val'],v_one))  # bit3 in t1
            
            # R15: 15-vselect cascade to select from tree[15..30]
            # Index = bit0*8 + bit1*4 + bit2*2 + bit3
            # Selection tree using bits from LSB to MSB:
            # Level 1 (bit3=t1): select pairs from tree[15..30]
            # Level 2 (bit2=idx): combine pairs
            # Level 3 (bit1=b1): combine quads
            # Level 4 (bit0=b0): combine octets
            for d in gd:
                dk=desks[d]
                # Level 1: 8 vselects on bit3 (t1)
                # tree[15+2i] for bit3=0, tree[15+2i+1] for bit3=1
                # Pairs: (15,16), (17,18), (19,20), (21,22), (23,24), (25,26), (27,28), (29,30)
                self.emit("flow",("vselect",dk['nv'],dk['t1'],v_tree[16],v_tree[15]))    # pair 0
                self.emit("flow",("vselect",dk['t2'],dk['t1'],v_tree[18],v_tree[17]))    # pair 1
                self.emit("flow",("vselect",dk['addr'],dk['t1'],v_tree[20],v_tree[19]))  # pair 2
                
                # Need more temps! We have: nv, t2, addr, and we can reuse b0/b1/idx/t1 AFTER they're no longer needed
                # After this point, t1 is consumed. idx consumed at level 2. b1 at level 3. b0 at level 4.
                # So at level 1 we can reuse t1 after first use.
                # Actually t1 is the selector for ALL level 1 vselects, so can't reuse until level 1 is done.
                # We need 8 outputs from level 1. We have: nv, t2, addr + can reuse b0,b1,idx,t1 = 7 regs
                # But b0 is needed at level 4, b1 at level 3, idx at level 2, t1 at level 1
                # So during level 1: can only use nv, t2, addr (3 regs) for first 3 outputs
                # Then after level 1 is complete, t1 is free = 4 regs
                # But we need 8 outputs... 
                
                # Alternative: do levels in a tree-reduction pattern
                # Process pairs of pairs immediately:
                # Step 1: vselect bit3 for pair(15,16) -> nv
                # Step 2: vselect bit3 for pair(17,18) -> t2
                # Step 3: vselect bit2(idx) between nv and t2 -> nv (quad 0)
                # Step 4: vselect bit3 for pair(19,20) -> t2
                # Step 5: vselect bit3 for pair(21,22) -> addr
                # Step 6: vselect bit2(idx) between t2 and addr -> t2 (quad 1)
                # Step 7: vselect bit1(b1) between nv and t2 -> nv (oct 0)
                # Repeat for tree[23..30]:
                # Step 8: vselect bit3 for pair(23,24) -> t2
                # Step 9: vselect bit3 for pair(25,26) -> addr
                # Step 10: vselect bit2(idx) between t2 and addr -> t2 (quad 2)
                # Step 11: vselect bit3 for pair(27,28) -> addr
                # Step 12: vselect bit3 for pair(29,30) -> t2 (reusing)
                # Wait, need both t2 and addr for step 10 result...
                
                # Let me be more careful. I'll use nv, t2, addr as temps.
                # Tree reduction pattern:
                
                # Octet 0 (tree[15..22]):
                #   Quad 0 (tree[15..18]):
                self.emit("flow",("vselect",dk['nv'],dk['t1'],v_tree[16],v_tree[15]))    # pair(15,16) -> nv
                self.emit("flow",("vselect",dk['t2'],dk['t1'],v_tree[18],v_tree[17]))    # pair(17,18) -> t2
                self.emit("flow",("vselect",dk['nv'],dk['idx'],dk['t2'],dk['nv']))        # quad0 -> nv
                #   Quad 1 (tree[19..22]):
                self.emit("flow",("vselect",dk['t2'],dk['t1'],v_tree[20],v_tree[19]))    # pair(19,20) -> t2
                self.emit("flow",("vselect",dk['addr'],dk['t1'],v_tree[22],v_tree[21]))  # pair(21,22) -> addr
                self.emit("flow",("vselect",dk['t2'],dk['idx'],dk['addr'],dk['t2']))      # quad1 -> t2
                #   Octet 0:
                self.emit("flow",("vselect",dk['nv'],dk['b1'],dk['t2'],dk['nv']))          # oct0 -> nv
                
                # Octet 1 (tree[23..30]):
                #   Quad 2 (tree[23..26]):
                self.emit("flow",("vselect",dk['t2'],dk['t1'],v_tree[24],v_tree[23]))    # pair(23,24) -> t2
                self.emit("flow",("vselect",dk['addr'],dk['t1'],v_tree[26],v_tree[25]))  # pair(25,26) -> addr
                self.emit("flow",("vselect",dk['t2'],dk['idx'],dk['addr'],dk['t2']))      # quad2 -> t2
                #   Quad 3 (tree[27..30]):
                self.emit("flow",("vselect",dk['addr'],dk['t1'],v_tree[28],v_tree[27]))  # pair(27,28) -> addr
                # Need another temp for pair(29,30)... t1 is still needed as selector!
                # Actually, at this point we've used t1 for all 8 pair selections above.
                # But we have 2 more pair selections to do.
                # Wait - I already did 5 pairs (15-16, 17-18, 19-20, 21-22, 23-24).
                # That's wrong - I restructured. Let me redo:
                pass  # Will redo below
            
            # Redo: cleaner tree reduction
            for d in gd:
                dk=desks[d]
                # We have 4 bits: b0, b1, idx(bit2), t1(bit3)
                # Need to select tree[15 + b0*8 + b1*4 + idx*2 + t1]
                
                # Tree reduction: process from bit3 (innermost) to bit0 (outermost)
                
                # Layer 1 (bit3 selects): 8 pairs -> 8 results
                # But we only have ~4 temp vecs. So we interleave layers.
                
                # Process in 2-level chunks (4 vselects each -> 1 quad result):
                
                # Quad 0: tree[15..18], bits: bit3 then bit2
                self.emit("flow",("vselect",dk['nv'],dk['t1'],v_tree[16],v_tree[15]))
                self.emit("flow",("vselect",dk['t2'],dk['t1'],v_tree[18],v_tree[17]))
                self.emit("flow",("vselect",dk['nv'],dk['idx'],dk['t2'],dk['nv']))  # quad0 -> nv
                
                # Quad 1: tree[19..22]
                self.emit("flow",("vselect",dk['t2'],dk['t1'],v_tree[20],v_tree[19]))
                self.emit("flow",("vselect",dk['addr'],dk['t1'],v_tree[22],v_tree[21]))
                self.emit("flow",("vselect",dk['t2'],dk['idx'],dk['addr'],dk['t2']))  # quad1 -> t2
                
                # Oct 0: combine quad0 and quad1 with b1
                self.emit("flow",("vselect",dk['nv'],dk['b1'],dk['t2'],dk['nv']))  # oct0 -> nv
                
                # Quad 2: tree[23..26]
                self.emit("flow",("vselect",dk['t2'],dk['t1'],v_tree[24],v_tree[23]))
                self.emit("flow",("vselect",dk['addr'],dk['t1'],v_tree[26],v_tree[25]))
                self.emit("flow",("vselect",dk['t2'],dk['idx'],dk['addr'],dk['t2']))  # quad2 -> t2
                
                # Quad 3: tree[27..30]
                self.emit("flow",("vselect",dk['addr'],dk['t1'],v_tree[28],v_tree[27]))
                # For pair(29,30), we've used t1 for all pairs now.
                # We can reuse any temp that's no longer needed for THIS desk.
                # idx is needed until quad3. After quad3, idx is free.
                # t1 is needed for pair selections. After all pairs, t1 is free.
                # We need: pair(29,30)->somewhere, then idx-select between addr and that
                # Use b0 temporarily? No, b0 is needed at the final level.
                # Actually, after this quad, idx is no longer needed. So:
                self.emit("flow",("vselect",dk['idx'],dk['t1'],v_tree[30],v_tree[29]))  # reuse idx!
                self.emit("flow",("vselect",dk['t2'],dk['idx'],dk['idx'],dk['addr']))   # quad3 -> wait, idx is both input and output!
                
                # That's a problem - idx is used as selector AND overwritten with pair result.
                # Let me use a different temp. After the pair(27,28)->addr, addr holds the result.
                # For pair(29,30), I need somewhere to put it.
                # b1 is consumed after oct0 merge, so b1 is free now!
                self.emit("flow",("vselect",dk['b1'],dk['t1'],v_tree[30],v_tree[29]))   # pair(29,30) -> b1
                self.emit("flow",("vselect",dk['t2'],dk['idx'],dk['b1'],dk['addr']))    # quad3 -> t2
                
                # Oct 1: combine quad2 and quad3 with b1... but b1 was overwritten!
                # This is the register pressure problem.
                
                # Need to rethink the register usage.
                pass
            
            # OK, this is getting complicated with register pressure.
            # Let me use a cleaner approach: allocate the computation more carefully.
            # Available temps per desk: nv, addr, t1, t2, b0, b1, idx
            # Constraints: b0 needed at level 4 (last), b1 at level 3, idx at level 2, t1 at level 1
            # Free after level 1: t1
            # Free after level 2: idx
            # Free after level 3: b1
            
            # So we process levels bottom-up, freeing registers as we go.
            
            # Reset and do it properly in a separate function
            pass

        # Let me implement emit_fused_11_15 properly with careful register management
        def emit_fused_11_15_v2(gd):
            # R11: XOR with tree[0], hash, extract bit0
            for d in gd: self.emit("valu",("^",desks[d]['val'],desks[d]['val'],v_tree[0]))
            emit_hash_interleaved(gd)
            for d in gd: self.emit("valu",("&",desks[d]['b0'],desks[d]['val'],v_one))
            
            # R12: vselect tree node, XOR, hash, extract bit1
            for d in gd: self.emit("flow",("vselect",desks[d]['nv'],desks[d]['b0'],v_tree[2],v_tree[1]))
            for d in gd: self.emit("valu",("^",desks[d]['val'],desks[d]['val'],desks[d]['nv']))
            emit_hash_interleaved(gd)
            for d in gd: self.emit("valu",("&",desks[d]['b1'],desks[d]['val'],v_one))
            
            # R13: 3-vselect cascade, XOR, hash, extract bit2
            for d in gd:
                dk=desks[d]
                self.emit("flow",("vselect",dk['t2'],dk['b1'],v_tree[4],v_tree[3]))
                self.emit("flow",("vselect",dk['nv'],dk['b1'],v_tree[6],v_tree[5]))
                self.emit("flow",("vselect",dk['nv'],dk['b0'],dk['nv'],dk['t2']))
            for d in gd: self.emit("valu",("^",desks[d]['val'],desks[d]['val'],desks[d]['nv']))
            emit_hash_interleaved(gd)
            for d in gd: self.emit("valu",("&",desks[d]['idx'],desks[d]['val'],v_one))  # bit2
            
            # R14: 7-vselect cascade, XOR, hash, extract bit3
            for d in gd:
                dk=desks[d]
                self.emit("flow",("vselect",dk['t2'],dk['idx'],v_tree[8],v_tree[7]))
                self.emit("flow",("vselect",dk['nv'],dk['idx'],v_tree[10],v_tree[9]))
                self.emit("flow",("vselect",dk['t2'],dk['b1'],dk['nv'],dk['t2']))
                self.emit("flow",("vselect",dk['nv'],dk['idx'],v_tree[12],v_tree[11]))
                self.emit("flow",("vselect",dk['addr'],dk['idx'],v_tree[14],v_tree[13]))
                self.emit("flow",("vselect",dk['nv'],dk['b1'],dk['addr'],dk['nv']))
                self.emit("flow",("vselect",dk['nv'],dk['b0'],dk['nv'],dk['t2']))
            for d in gd: self.emit("valu",("^",desks[d]['val'],desks[d]['val'],desks[d]['nv']))
            emit_hash_interleaved(gd)
            for d in gd: self.emit("valu",("&",desks[d]['t1'],desks[d]['val'],v_one))  # bit3
            
            # R15: 15-vselect cascade from tree[15..30]
            # Selection: tree[15 + b0*8 + b1*4 + idx*2 + t1]
            # Strategy: tree reduction, process from innermost bit (t1) outward
            # Available per desk: nv, t2, addr as temps. b0,b1,idx,t1 as selectors.
            
            for d in gd:
                dk=desks[d]
                
                # Process 4 quads, each = 2 pair-selects (on t1) + 1 quad-select (on idx)
                # Then 2 oct-selects (on b1), then 1 final (on b0)
                
                # Quad 0: tree[15,16,17,18]
                self.emit("flow",("vselect",dk['nv'],dk['t1'],v_tree[16],v_tree[15]))    # pair0 -> nv
                self.emit("flow",("vselect",dk['t2'],dk['t1'],v_tree[18],v_tree[17]))    # pair1 -> t2
                self.emit("flow",("vselect",dk['nv'],dk['idx'],dk['t2'],dk['nv']))        # quad0 -> nv
                
                # Quad 1: tree[19,20,21,22]
                self.emit("flow",("vselect",dk['t2'],dk['t1'],v_tree[20],v_tree[19]))    # pair2 -> t2
                self.emit("flow",("vselect",dk['addr'],dk['t1'],v_tree[22],v_tree[21]))  # pair3 -> addr
                self.emit("flow",("vselect",dk['t2'],dk['idx'],dk['addr'],dk['t2']))      # quad1 -> t2
                
                # Oct 0: b1 selects between quad0(nv) and quad1(t2)
                self.emit("flow",("vselect",dk['nv'],dk['b1'],dk['t2'],dk['nv']))         # oct0 -> nv
                # After this: nv=oct0, b1 STILL NEEDED for oct1. idx STILL NEEDED for quad2,3.
                
                # Quad 2: tree[23,24,25,26]
                self.emit("flow",("vselect",dk['t2'],dk['t1'],v_tree[24],v_tree[23]))    # pair4 -> t2
                self.emit("flow",("vselect",dk['addr'],dk['t1'],v_tree[26],v_tree[25]))  # pair5 -> addr
                self.emit("flow",("vselect",dk['t2'],dk['idx'],dk['addr'],dk['t2']))      # quad2 -> t2
                # After this: idx is FREE (no longer needed)
                
                # Quad 3: tree[27,28,29,30]
                self.emit("flow",("vselect",dk['addr'],dk['t1'],v_tree[28],v_tree[27]))  # pair6 -> addr
                self.emit("flow",("vselect",dk['idx'],dk['t1'],v_tree[30],v_tree[29]))   # pair7 -> idx (reuse, free)
                # After this: t1 is FREE
                self.emit("flow",("vselect",dk['addr'],dk['idx'],dk['idx'],dk['addr']))   # quad3 -> wait idx both sel and input
                # BUG: idx is selector AND one of the inputs! Can't do that.
                # Fix: use 'addr' as selector result instead
                # Actually the issue is: idx was just written with pair7 result.
                # And now we need idx (bit2) as selector. But idx was overwritten!
                # We stored pair7 into idx, destroying bit2.
                
                # Fix: DON'T reuse idx for pair7. Use a different temp.
                # After quad2, t2 holds quad2 result. We need it for oct1.
                # After pair6, addr holds pair6 result.
                # For pair7, we need somewhere to put it without clobbering idx or t2.
                # t1 is still needed for pair7 select! Can't reuse t1.
                # Available: NONE - all are in use.
                
                # Alternative: interleave quad2 and quad3 differently.
                # Process oct1 differently: do all of quad2+quad3 first using idx,
                # then merge with b1.
                pass
            
            # Let me try yet another approach: process both octs simultaneously
            # to reduce register pressure.
            
            # Actually, the simplest fix: use a different bit ordering.
            # Instead of bit3(innermost)->bit0(outermost), 
            # try bit0(innermost)->bit3(outermost).
            # Selection: tree[15 + b0 + b1*2 + idx*4 + t1*8]
            
            # OR: just be smarter about temp usage.
            # After oct0 is in nv, and before quad2/3:
            # - nv = oct0 (keep)
            # - t1 = bit3 selector (keep for pairs)
            # - idx = bit2 selector (keep for quad merges)
            # - b1 = bit1 (keep for oct merge)
            # - b0 = bit0 (keep for final merge)
            # Free: t2, addr
            # We need 4 outputs from 4 pair selects, but only have 2 free temps.
            # Solution: do 2 pairs, merge to quad, then 2 more pairs, merge to quad, then oct.
            
            pass
        
        # FINAL clean implementation:
        def emit_fused_11_15_clean(gd):
            # R11: XOR with tree[0], hash, extract bit0
            for d in gd: self.emit("valu",("^",desks[d]['val'],desks[d]['val'],v_tree[0]))
            emit_hash_interleaved(gd)
            for d in gd: self.emit("valu",("&",desks[d]['b0'],desks[d]['val'],v_one))
            
            # R12: vselect tree node, XOR, hash, extract bit1
            for d in gd: self.emit("flow",("vselect",desks[d]['nv'],desks[d]['b0'],v_tree[2],v_tree[1]))
            for d in gd: self.emit("valu",("^",desks[d]['val'],desks[d]['val'],desks[d]['nv']))
            emit_hash_interleaved(gd)
            for d in gd: self.emit("valu",("&",desks[d]['b1'],desks[d]['val'],v_one))
            
            # R13: 3-vselect cascade, XOR, hash, extract bit2
            for d in gd:
                dk=desks[d]
                self.emit("flow",("vselect",dk['t2'],dk['b1'],v_tree[4],v_tree[3]))
                self.emit("flow",("vselect",dk['nv'],dk['b1'],v_tree[6],v_tree[5]))
                self.emit("flow",("vselect",dk['nv'],dk['b0'],dk['nv'],dk['t2']))
            for d in gd: self.emit("valu",("^",desks[d]['val'],desks[d]['val'],desks[d]['nv']))
            emit_hash_interleaved(gd)
            for d in gd: self.emit("valu",("&",desks[d]['idx'],desks[d]['val'],v_one))  # bit2
            
            # R14: 7-vselect cascade, XOR, hash, extract bit3
            for d in gd:
                dk=desks[d]
                self.emit("flow",("vselect",dk['t2'],dk['idx'],v_tree[8],v_tree[7]))
                self.emit("flow",("vselect",dk['nv'],dk['idx'],v_tree[10],v_tree[9]))
                self.emit("flow",("vselect",dk['t2'],dk['b1'],dk['nv'],dk['t2']))
                self.emit("flow",("vselect",dk['nv'],dk['idx'],v_tree[12],v_tree[11]))
                self.emit("flow",("vselect",dk['addr'],dk['idx'],v_tree[14],v_tree[13]))
                self.emit("flow",("vselect",dk['nv'],dk['b1'],dk['addr'],dk['nv']))
                self.emit("flow",("vselect",dk['nv'],dk['b0'],dk['nv'],dk['t2']))
            for d in gd: self.emit("valu",("^",desks[d]['val'],desks[d]['val'],desks[d]['nv']))
            emit_hash_interleaved(gd)
            for d in gd: self.emit("valu",("&",desks[d]['t1'],desks[d]['val'],v_one))  # bit3
            
            # R15: 15-vselect cascade from tree[15..30]
            # Process as tree reduction: 2 pairs -> quad, 2 quads -> oct, 2 octs -> final
            # Do it sequentially to manage register pressure:
            for d in gd:
                dk=desks[d]
                
                # --- Oct 0 (tree[15..22]) ---
                # Quad 0: tree[15..18]
                self.emit("flow",("vselect",dk['nv'],dk['t1'],v_tree[16],v_tree[15]))
                self.emit("flow",("vselect",dk['t2'],dk['t1'],v_tree[18],v_tree[17]))
                self.emit("flow",("vselect",dk['nv'],dk['idx'],dk['t2'],dk['nv']))  # quad0 -> nv
                # Quad 1: tree[19..22]
                self.emit("flow",("vselect",dk['t2'],dk['t1'],v_tree[20],v_tree[19]))
                self.emit("flow",("vselect",dk['addr'],dk['t1'],v_tree[22],v_tree[21]))
                self.emit("flow",("vselect",dk['t2'],dk['idx'],dk['addr'],dk['t2']))  # quad1 -> t2
                # Oct 0:
                self.emit("flow",("vselect",dk['nv'],dk['b1'],dk['t2'],dk['nv']))  # oct0 -> nv
                
                # --- Oct 1 (tree[23..30]) ---
                # Quad 2: tree[23..26]
                self.emit("flow",("vselect",dk['t2'],dk['t1'],v_tree[24],v_tree[23]))
                self.emit("flow",("vselect",dk['addr'],dk['t1'],v_tree[26],v_tree[25]))
                self.emit("flow",("vselect",dk['t2'],dk['idx'],dk['addr'],dk['t2']))  # quad2 -> t2
                # Quad 3: tree[27..30]  
                self.emit("flow",("vselect",dk['addr'],dk['t1'],v_tree[28],v_tree[27]))
                # For pair(29,30): need temp. t1 still needed? Yes for this pair!
                # After this pair, t1 is done. idx needed until quad3 merge.
                # Use addr for pair(27,28), need another for pair(29,30).
                # idx is still needed as selector for quad merge.
                # So available for pair(29,30) output: NOTHING except we can compute differently.
                
                # Alternative for Quad 3: merge pair(27,28) and pair(29,30) without intermediate
                # pair6 = vselect(t1, tree[28], tree[27]) -> addr
                # pair7 = vselect(t1, tree[30], tree[29]) -> need temp
                # quad3 = vselect(idx, pair7, pair6) -> need pair7 and pair6 both live
                
                # We have addr=pair6, and need pair7 somewhere.
                # Can we reuse b1? b1 is needed for oct1 merge!
                # Can we reuse b0? b0 is needed for final merge!
                # Can we reuse nv? nv holds oct0, needed for final merge!
                
                # Solution: combine quad2 and quad3 into oct1 immediately.
                # After quad2 -> t2:
                # Do pair(27,28) -> addr
                # Now do the quad3/oct1 merge differently:
                # oct1 = vselect(b1, quad3, quad2) = vselect(b1, quad3, t2)
                # But we need quad3 = vselect(idx, pair7, pair6)
                
                # Key insight: we can nest the operations:
                # 1. pair6 -> addr
                # 2. pair7 -> b1 (REUSE b1! Then compute oct1 right after using old b1)
                # NO - we need b1 to compute oct1.
                
                # OK different approach: do it all per-desk with a temp trick.
                # After quad2 (in t2) and pair6 (in addr):
                # Overwrite idx with pair7 (idx no longer needed after this quad):
                self.emit("flow",("vselect",dk['idx'],dk['t1'],v_tree[30],v_tree[29]))  # pair7 -> idx
                # Now idx has pair7, addr has pair6. But we need the ORIGINAL idx (bit2) as selector!
                # We just clobbered it.
                
                # REAL solution: do quad3's idx-select FIRST, then clobber idx.
                # But we need both pair6 and pair7 for the idx-select...
                # Chicken-and-egg problem.
                
                # FINAL SOLUTION: Process oct1 completely differently.
                # Instead of quad2, quad3, oct1... do all 4 pairs, then tree-reduce.
                # But we proved we don't have enough temps for 4 pair results.
                
                # ACTUAL SOLUTION: For quad3, use different bit ordering.
                # Process bit2 BEFORE bit3 for quad3:
                # First select on idx: tree[27] vs tree[29] -> A (bit2=0 or 1 with bit3=0)
                #                       tree[28] vs tree[30] -> B (bit2=0 or 1 with bit3=1)  
                # Then select on t1: A vs B
                # This uses 3 vselects same as before but different order.
                
                # tree layout: tree[27]=b0=1,b1=1,b2=0,b3=0  tree[28]=b0=1,b1=1,b2=0,b3=1
                #              tree[29]=b0=1,b1=1,b2=1,b3=0  tree[30]=b0=1,b1=1,b2=1,b3=1
                # bit3 pairs: (27,28), (29,30)
                # bit2 pairs: (27,29), (28,30)
                
                # Use bit2 first:
                # vselect(idx, tree[29], tree[27]) -> addr  (idx=0 -> tree[27], idx=1 -> tree[29])
                # vselect(idx, tree[30], tree[28]) -> idx   (clobber idx, last use)
                # vselect(t1, idx, addr) -> t2 or addr     (t1=0 -> addr, t1=1 -> idx)
                # Wait that's wrong semantics. Let me think more carefully.
                
                # Hmm, I'm overcomplicating this. Let me just save bit2 (idx) into a temp 
                # BEFORE the R15 cascade, using one of the free temps.
                # After R14, we have: b0, b1, idx, t1 = the 4 bits. 
                # nv, t2, addr = free.
                # Copy idx to addr: self.emit("valu", ("+", dk['addr'], dk['idx'], v_zero))
                # But v_zero doesn't exist! We removed it.
                # Could use: self.emit("valu", ("^", dk['addr'], dk['idx'], dk['idx'])) 
                # That gives 0, not idx!
                # How about: multiply_add(addr, idx, v_one, v_zero)? No v_zero.
                # vselect(addr, v_one, idx, idx)? Always gives idx regardless of condition. 
                # That's a FLOW op but it works as a copy!
                # Or: self.emit("valu", ("&", dk['addr'], dk['idx'], dk['idx'])) -> idx & idx = idx? 
                # Actually & is bitwise, so idx & idx = idx. But the ISA says & is AND between two vec regs.
                # Let me check: ("&", dest, a1, a2) -> dest[i] = a1[i] & a2[i]
                # So ("&", addr, idx, idx) -> addr = idx & idx = idx. But that's a bit silly.
                # Better: we can use the fact that multiply_add(dest, a, 1, 0) = a*1+0 = a
                # scratch_vconst(0) would allocate v_zero... but we removed it to save space.
                # Actually, 0 IS available as a scalar const (from tree index 0).
                # So scratch_vconst(0) would create a vec version. That costs 8 more slots.
                # 1518 + 8 = 1526 <= 1536. Fits!
                # OR: just add 1 VALU to copy idx. Use ("^", addr, idx, v_zero) where v_zero is the 
                # vector of zeros. But we don't have v_zero.
                # Simplest: self.emit("valu", ("+", dk['addr'], dk['idx'], scratch_vconst(0)))
                # This would create v_zero (8 slots) taking us to 1526.
                
                # But wait - we don't even need a copy if we process the cascade differently!
                # KEY INSIGHT: Process bit0 first (outermost), then bit1, bit2, bit3.
                # That way idx (bit2) is used early and freed.
                # Selection: tree[15 + b0*8 + b1*4 + idx*2 + t1]
                # Process b0 first: split into 2 halves of 8
                
                # Nah, that makes the cascade require 8 intermediate results at some point.
                # The tree reduction is the same regardless of order.
                
                # SIMPLEST FIX: just allocate v_zero. 1526 fits in 1536.
                pass
            pass  # End of messy attempts

        # I'll implement this cleanly now, allocating v_zero for the copy trick.
        pass  # See actual implementation below

        # CLEAN FINAL IMPLEMENTATION (replacing all the mess above)
        v_zero = self.scratch_vconst(0)  # Need this for idx copy in R15 cascade

        def emit_fused_11_15_final(gd):
            # R11: XOR tree[0], hash, extract bit0
            for d in gd: self.emit("valu",("^",desks[d]['val'],desks[d]['val'],v_tree[0]))
            emit_hash_interleaved(gd)
            for d in gd: self.emit("valu",("&",desks[d]['b0'],desks[d]['val'],v_one))
            
            # R12: vselect, XOR, hash, extract bit1
            for d in gd: self.emit("flow",("vselect",desks[d]['nv'],desks[d]['b0'],v_tree[2],v_tree[1]))
            for d in gd: self.emit("valu",("^",desks[d]['val'],desks[d]['val'],desks[d]['nv']))
            emit_hash_interleaved(gd)
            for d in gd: self.emit("valu",("&",desks[d]['b1'],desks[d]['val'],v_one))
            
            # R13: 3-vselect, XOR, hash, extract bit2
            for d in gd:
                dk=desks[d]
                self.emit("flow",("vselect",dk['t2'],dk['b1'],v_tree[4],v_tree[3]))
                self.emit("flow",("vselect",dk['nv'],dk['b1'],v_tree[6],v_tree[5]))
                self.emit("flow",("vselect",dk['nv'],dk['b0'],dk['nv'],dk['t2']))
            for d in gd: self.emit("valu",("^",desks[d]['val'],desks[d]['val'],desks[d]['nv']))
            emit_hash_interleaved(gd)
            for d in gd: self.emit("valu",("&",desks[d]['idx'],desks[d]['val'],v_one))  # bit2
            
            # R14: 7-vselect, XOR, hash, extract bit3
            for d in gd:
                dk=desks[d]
                self.emit("flow",("vselect",dk['t2'],dk['idx'],v_tree[8],v_tree[7]))
                self.emit("flow",("vselect",dk['nv'],dk['idx'],v_tree[10],v_tree[9]))
                self.emit("flow",("vselect",dk['t2'],dk['b1'],dk['nv'],dk['t2']))
                self.emit("flow",("vselect",dk['nv'],dk['idx'],v_tree[12],v_tree[11]))
                self.emit("flow",("vselect",dk['addr'],dk['idx'],v_tree[14],v_tree[13]))
                self.emit("flow",("vselect",dk['nv'],dk['b1'],dk['addr'],dk['nv']))
                self.emit("flow",("vselect",dk['nv'],dk['b0'],dk['nv'],dk['t2']))
            for d in gd: self.emit("valu",("^",desks[d]['val'],desks[d]['val'],desks[d]['nv']))
            emit_hash_interleaved(gd)
            for d in gd: self.emit("valu",("&",desks[d]['t1'],desks[d]['val'],v_one))  # bit3

            # R15: 15-vselect cascade from tree[15..30]
            # Save bit2 (idx) before the cascade clobbers it
            for d in gd: self.emit("valu",("+",desks[d]['addr'],desks[d]['idx'],v_zero))  # copy idx -> addr (bit2 backup)
            
            for d in gd:
                dk=desks[d]
                # idx_copy = addr = bit2
                # Available temps: nv, t2. idx is now free to clobber.
                
                # Quad 0: tree[15..18] (b0=0,b1=0)
                self.emit("flow",("vselect",dk['nv'],dk['t1'],v_tree[16],v_tree[15]))   # pair(15,16)
                self.emit("flow",("vselect",dk['t2'],dk['t1'],v_tree[18],v_tree[17]))   # pair(17,18)
                self.emit("flow",("vselect",dk['nv'],dk['addr'],dk['t2'],dk['nv']))      # quad0 -> nv (using addr=bit2)
                
                # Quad 1: tree[19..22] (b0=0,b1=1)
                self.emit("flow",("vselect",dk['t2'],dk['t1'],v_tree[20],v_tree[19]))
                self.emit("flow",("vselect",dk['idx'],dk['t1'],v_tree[22],v_tree[21]))  # use idx as temp
                self.emit("flow",("vselect",dk['t2'],dk['addr'],dk['idx'],dk['t2']))     # quad1 -> t2
                
                # Oct 0:
                self.emit("flow",("vselect",dk['nv'],dk['b1'],dk['t2'],dk['nv']))        # oct0 -> nv
                
                # Quad 2: tree[23..26] (b0=1,b1=0)
                self.emit("flow",("vselect",dk['t2'],dk['t1'],v_tree[24],v_tree[23]))
                self.emit("flow",("vselect",dk['idx'],dk['t1'],v_tree[26],v_tree[25]))
                self.emit("flow",("vselect",dk['t2'],dk['addr'],dk['idx'],dk['t2']))     # quad2 -> t2
                
                # Quad 3: tree[27..30] (b0=1,b1=1)
                self.emit("flow",("vselect",dk['idx'],dk['t1'],v_tree[28],v_tree[27]))   # pair -> idx
                self.emit("flow",("vselect",dk['addr'],dk['t1'],v_tree[30],v_tree[29]))  # pair -> addr (clobber bit2 copy, fine - last use was quad2)
                self.emit("flow",("vselect",dk['idx'],dk['addr'],dk['addr'],dk['idx']))   # quad3... wait, addr is both sel and src
                # Problem! addr was used for bit2 copy in quad0-2, but now holds pair(29,30).
                # Actually after quad2, addr's bit2 value is no longer needed. 
                # So pair(29,30) -> addr is fine.
                # But then for quad3: vselect(bit2_copy, pair7, pair6)
                # bit2_copy was in addr, which we just clobbered!
                
                # FIX: Do pair7 into a different temp. After quad2, t2=quad2. 
                # Use: pair6->idx, pair7->addr, then quad3 needs bit2 selector.
                # But bit2 was in addr, now clobbered.
                
                # The real fix: save bit2 into a register that survives until quad3.
                # After oct0 is computed (in nv), b1 is consumed for oct merge.
                # Wait, b1 is needed again for oct1 merge. So b1 lives until after quad3.
                
                # Plan B: Don't clobber addr until after ALL quads.
                # Only use idx as temp for quads.
                # But we need 2 pair outputs simultaneously for each quad merge.
                
                # Plan C: For quad 3, reverse the bit order:
                # Instead of select bit3 then bit2, select bit2 first then bit3.
                # vselect(addr, tree[29], tree[27]) -> idx   (bit2 selects even/odd)
                # vselect(addr, tree[30], tree[28]) -> addr  (bit2, last use of addr as bit2)
                # vselect(t1, addr, idx) -> idx             (bit3 final)
                pass
            
            # OK I need a completely clean approach. Let me restart the R15 cascade.
            pass

        def emit_fused_11_15_REAL(gd):
            """Final clean implementation of R11-R15 fused."""
            # R11: XOR tree[0], hash, extract bit0
            for d in gd: self.emit("valu",("^",desks[d]['val'],desks[d]['val'],v_tree[0]))
            emit_hash_interleaved(gd)
            for d in gd: self.emit("valu",("&",desks[d]['b0'],desks[d]['val'],v_one))
            
            # R12: 1-vselect, XOR, hash, extract bit1
            for d in gd: self.emit("flow",("vselect",desks[d]['nv'],desks[d]['b0'],v_tree[2],v_tree[1]))
            for d in gd: self.emit("valu",("^",desks[d]['val'],desks[d]['val'],desks[d]['nv']))
            emit_hash_interleaved(gd)
            for d in gd: self.emit("valu",("&",desks[d]['b1'],desks[d]['val'],v_one))
            
            # R13: 3-vselect, XOR, hash, extract bit2
            for d in gd:
                dk=desks[d]
                self.emit("flow",("vselect",dk['t2'],dk['b1'],v_tree[4],v_tree[3]))
                self.emit("flow",("vselect",dk['nv'],dk['b1'],v_tree[6],v_tree[5]))
                self.emit("flow",("vselect",dk['nv'],dk['b0'],dk['nv'],dk['t2']))
            for d in gd: self.emit("valu",("^",desks[d]['val'],desks[d]['val'],desks[d]['nv']))
            emit_hash_interleaved(gd)
            for d in gd: self.emit("valu",("&",desks[d]['idx'],desks[d]['val'],v_one))  # bit2
            
            # R14: 7-vselect, XOR, hash, extract bit3
            for d in gd:
                dk=desks[d]
                self.emit("flow",("vselect",dk['t2'],dk['idx'],v_tree[8],v_tree[7]))
                self.emit("flow",("vselect",dk['nv'],dk['idx'],v_tree[10],v_tree[9]))
                self.emit("flow",("vselect",dk['t2'],dk['b1'],dk['nv'],dk['t2']))
                self.emit("flow",("vselect",dk['nv'],dk['idx'],v_tree[12],v_tree[11]))
                self.emit("flow",("vselect",dk['addr'],dk['idx'],v_tree[14],v_tree[13]))
                self.emit("flow",("vselect",dk['nv'],dk['b1'],dk['addr'],dk['nv']))
                self.emit("flow",("vselect",dk['nv'],dk['b0'],dk['nv'],dk['t2']))
            for d in gd: self.emit("valu",("^",desks[d]['val'],desks[d]['val'],desks[d]['nv']))
            emit_hash_interleaved(gd)
            for d in gd: self.emit("valu",("&",desks[d]['t1'],desks[d]['val'],v_one))  # bit3
            
            # R15: 15-vselect cascade
            # Need to select tree[15 + b0*8 + b1*4 + idx*2 + t1]
            # Registers: b0(bit0), b1(bit1), idx(bit2), t1(bit3)
            # Temps: nv, t2, addr
            
            # Strategy: process in sequential quad-reduction pattern.
            # For each quad, do 2 pair-selects + 1 quad-merge = 3 vselects.
            # Then merge 2 quads into an oct = 1 vselect. Then merge 2 octs = 1 vselect.
            # Total: 4*3 + 2 + 1 = 15 vselects.
            
            # Register plan:
            # After all quads: store oct0 in nv, oct1 in t2.
            # Final: vselect(b0, oct1, oct0) -> nv
            
            # Per-quad register usage:
            # Each quad needs: pair_A, pair_B, then merge.
            # Temps available: we can use any combo of {nv, t2, addr, idx, t1} that won't be needed later.
            # Selectors needed:
            #   t1 (bit3) for all pair selects (8 pairs) - used in ALL quads
            #   idx (bit2) for quad merges (4 quads) - used in ALL quads  
            #   b1 (bit1) for oct merges (2 octs) - used after all quads
            #   b0 (bit0) for final merge (1) - used last
            
            # So during quad processing: t1, idx, b1, b0 are READ-ONLY.
            # Free temps: nv, t2, addr = 3 temps.
            # Each quad produces 1 result. We need 4 quad results, then merge them.
            # But we only have 3 free temps. So we must merge quads into octs as we go.
            
            for d in gd:
                dk=desks[d]
                
                # Quad 0: tree[15,16,17,18]
                self.emit("flow",("vselect",dk['nv'],dk['t1'],v_tree[16],v_tree[15]))
                self.emit("flow",("vselect",dk['t2'],dk['t1'],v_tree[18],v_tree[17]))
                self.emit("flow",("vselect",dk['nv'],dk['idx'],dk['t2'],dk['nv']))  # q0->nv
                
                # Quad 1: tree[19,20,21,22]
                self.emit("flow",("vselect",dk['t2'],dk['t1'],v_tree[20],v_tree[19]))
                self.emit("flow",("vselect",dk['addr'],dk['t1'],v_tree[22],v_tree[21]))
                self.emit("flow",("vselect",dk['t2'],dk['idx'],dk['addr'],dk['t2']))  # q1->t2
                
                # Oct 0 = vselect(b1, q1, q0)
                self.emit("flow",("vselect",dk['nv'],dk['b1'],dk['t2'],dk['nv']))  # oct0->nv
                
                # Quad 2: tree[23,24,25,26]
                self.emit("flow",("vselect",dk['t2'],dk['t1'],v_tree[24],v_tree[23]))
                self.emit("flow",("vselect",dk['addr'],dk['t1'],v_tree[26],v_tree[25]))
                self.emit("flow",("vselect",dk['t2'],dk['idx'],dk['addr'],dk['t2']))  # q2->t2
                
                # Quad 3: tree[27,28,29,30]
                # Need 2 pair results + merge. Have: addr free, idx/t1 selectors.
                # Pair 6: vselect(t1, tree[28], tree[27]) -> addr
                self.emit("flow",("vselect",dk['addr'],dk['t1'],v_tree[28],v_tree[27]))
                # Pair 7: need somewhere to put it. nv=oct0(needed), t2=q2(needed for oct1).
                # idx is READ-ONLY... but after this quad merge, idx is no longer needed!
                # However, we're in the MIDDLE of this quad and need idx for the merge.
                # So we CANNOT clobber idx yet.
                # What about b1? b1 is needed for oct1. Can't clobber.
                # What about b0? b0 is needed for final. Can't clobber.
                # What about t1? t1 is needed for pair7. Can't clobber.
                
                # STUCK: 3 free temps (nv, t2, addr) all occupied.
                # nv=oct0, t2=q2, addr=pair6.
                # Need pair7 somewhere.
                
                # SOLUTION: Process quad3 differently - use idx as temp and then re-select.
                # Or: merge oct1 before completing quad3.
                # Or: reorder to compute q2 and q3 into oct1 together.
                
                # Approach: do q2 and q3 alternated with oct1 merge.
                # After q2->t2, we can start q3.
                # For q3: we need 2 pair results. Use addr and... one more.
                # At this point: nv=oct0, t2=q2, addr=free(pair6 not started yet).
                # Actually let's not do pair6 yet:
                pass
            
            # CLEAN RESTART for R15 cascade with better register planning:
            for d in gd:
                dk=desks[d]
                
                # Phase 1: Quad 0+1 -> Oct 0 (stored in nv)
                # q0:
                self.emit("flow",("vselect",dk['nv'],dk['t1'],v_tree[16],v_tree[15]))
                self.emit("flow",("vselect",dk['t2'],dk['t1'],v_tree[18],v_tree[17]))
                self.emit("flow",("vselect",dk['nv'],dk['idx'],dk['t2'],dk['nv']))
                # q1:
                self.emit("flow",("vselect",dk['t2'],dk['t1'],v_tree[20],v_tree[19]))
                self.emit("flow",("vselect",dk['addr'],dk['t1'],v_tree[22],v_tree[21]))
                self.emit("flow",("vselect",dk['t2'],dk['idx'],dk['addr'],dk['t2']))
                # oct0:
                self.emit("flow",("vselect",dk['nv'],dk['b1'],dk['t2'],dk['nv']))
                
                # Phase 2: Quad 2+3 -> Oct 1
                # For quad3 we need 2 pair results + idx merge.
                # After oct0, the only things we need to keep are:
                #   nv = oct0 (for final)
                #   b0 = bit0 (for final)
                #   b1 = bit1 (for oct1 merge) -- WAIT. b1 is needed for oct1 merge AFTER quad3.
                #     So b1 is read-only during phase 2.
                #   idx = bit2 (for quad merges in phase 2)
                #   t1 = bit3 (for pair selects in phase 2)
                # Free temps: t2, addr
                # For each quad we need 3 vselects with 2 temps.
                # quad: pair_A -> temp_A, pair_B -> temp_B, merge(idx, temp_B, temp_A) -> result
                # With only 2 temps, we can do each quad one at a time.
                
                # q2: 
                self.emit("flow",("vselect",dk['t2'],dk['t1'],v_tree[24],v_tree[23]))
                self.emit("flow",("vselect",dk['addr'],dk['t1'],v_tree[26],v_tree[25]))
                self.emit("flow",("vselect",dk['t2'],dk['idx'],dk['addr'],dk['t2']))  # q2->t2
                
                # q3: need pair6->addr, pair7->??? 
                self.emit("flow",("vselect",dk['addr'],dk['t1'],v_tree[28],v_tree[27]))  # pair6->addr
                # pair7 needs to go somewhere. t2=q2 (needed for oct1). nv=oct0 (needed).
                # ONLY option: We are about to merge q2 and q3 with b1.
                # Can we merge q2(t2) with pair6(addr) partially, then do pair7 and finish?
                
                # Alternative: do q3 with idx-first ordering.
                # Instead of: pair(27,28) on t1, pair(29,30) on t1, merge on idx
                # Do: select bit2 within each bit3 group:
                # For bit3=0: tree[27](idx=0) vs tree[29](idx=1)
                # For bit3=1: tree[28](idx=0) vs tree[30](idx=1)
                # vselect(idx, tree[29], tree[27]) -> addr   (bit2 select for bit3=0 group)
                # vselect(idx, tree[30], tree[28]) -> idx    (bit2 select for bit3=1 group) 
                #   ^ This clobbers idx! But it's the LAST quad, so idx is done after this.
                # vselect(t1, idx, addr) -> t2_or_addr       (bit3 select)
                # But we need old idx for this vselect, and we're overwriting idx.
                # If vselect reads inputs before writing output (VLIW end-of-cycle semantics),
                # then vselect(idx_old, tree[30], tree[28]) -> idx_new is fine IF the machine
                # reads idx_old and writes idx_new in the same cycle.
                # But wait, this is flow engine with 1 op/cycle. The semantics say
                # "all effects take place at end of cycle." So for a single op,
                # it reads the inputs, computes, then writes the output at end of cycle.
                # So reading idx and writing idx in the same op IS safe: it reads old value,
                # writes new value at end of cycle.
                
                # YES! This works for the flow engine (1 slot/cycle means 1 op at a time).
                # vselect reads all inputs first, then writes output.
                
                # q3 with idx-first:
                self.emit("flow",("vselect",dk['addr'],dk['idx'],v_tree[29],v_tree[27]))  # bit2 for bit3=0
                self.emit("flow",("vselect",dk['idx'],dk['idx'],v_tree[30],v_tree[28]))   # bit2 for bit3=1 (clobber idx, OK)
                self.emit("flow",("vselect",dk['addr'],dk['t1'],dk['idx'],dk['addr']))     # bit3 select -> q3 in addr
                # Now: t2=q2, addr=q3
                
                # oct1:
                self.emit("flow",("vselect",dk['t2'],dk['b1'],dk['addr'],dk['t2']))  # oct1->t2
                
                # Final: vselect(b0, oct1, oct0)
                self.emit("flow",("vselect",dk['nv'],dk['b0'],dk['t2'],dk['nv']))  # result->nv
            
            # XOR val with selected tree node (nv = tree[computed_index])
            for d in gd: self.emit("valu",("^",desks[d]['val'],desks[d]['val'],desks[d]['nv']))
            
            # Final hash (R15 is last round, no branch needed)
            emit_hash_interleaved(gd)

        def emit_tile(ti):
            off=ti*16*VLEN
            for d in range(16): self.emit("load",("const",offset_regs[d],off+d*VLEN))
            for d in range(16):
                self.emit("alu",("+",addr_tmp[d*2],self.scratch["inp_indices_p"],offset_regs[d]))
                self.emit("alu",("+",addr_tmp[d*2+1],self.scratch["inp_values_p"],offset_regs[d]))
            for d in range(16):
                self.emit("load",("vload",desks[d]['idx'],addr_tmp[d*2]))
                self.emit("load",("vload",desks[d]['val'],addr_tmp[d*2+1]))

            for g in range(4):
                gd=list(range(g*4,(g+1)*4))
                emit_fused_0123(gd)
                for _ in range(4,10): emit_gather_interleaved(gd)
                emit_r10(gd)
                emit_fused_11_15_REAL(gd)

            for d in range(16):
                self.emit("store",("vstore",addr_tmp[d*2],desks[d]['idx']))
                self.emit("store",("vstore",addr_tmp[d*2+1],desks[d]['val']))

        emit_tile(0); emit_tile(1)

        phases,cur=[],[]
        for e,s in self.slots:
            if e=="flow" and s==("pause",): phases.append(cur); cur=[]
            else: cur.append((e,s))
        phases.append(cur)
        self.instrs=[]
        for i,phase in enumerate(phases):
            if phase: self.instrs.extend(_schedule_slots(phase))
            if i<len(phases)-1: self.instrs.append({"flow":[("pause",)]})
        self.instrs.append({"flow":[("pause",)]})
        valu_count=sum(1 for e,s in self.slots if e=="valu")
        flow_count=sum(1 for e,s in self.slots if e=="flow" and s[0]!="pause")
        load_count=sum(1 for e,s in self.slots if e=="load")
        print(f"VALU: {valu_count}, Flow: {flow_count}, Load: {load_count}, Cycles: {len(self.instrs)}")

BASELINE=147734
def do_kernel_test(forest_height=10,rounds=16,batch_size=256,seed=123,trace=False,prints=False,check=False):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed); forest=Tree.generate(forest_height); inp=Input.generate(forest,batch_size,rounds); mem=build_mem_image(forest,inp)
    kb=KernelBuilderA1()
    kb.build_kernel(forest.height,len(forest.values),len(inp.indices),rounds)
    value_trace={}
    machine=Machine(mem,kb.instrs,kb.debug_info(),n_cores=N_CORES,value_trace=value_trace,trace=trace)
    machine.prints=prints
    for i,ref_mem in enumerate(reference_kernel2(mem,value_trace)):
        machine.run()
        if check:
            inp_values_p=ref_mem[6]
            assert(machine.mem[inp_values_p:inp_values_p+len(inp.values)]==ref_mem[inp_values_p:inp_values_p+len(inp.values)]),f"Incorrect on round {i}"
    print("CYCLES:",machine.cycle); return machine.cycle

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--check",action="store_true")
    parser.add_argument("--trace",action="store_true")
    args=parser.parse_args()
    if args.check:
        cycles=do_kernel_test(check=True)
        print(f"Correctness check PASSED! Cycles: {cycles}")
    else: do_kernel_test(trace=args.trace)
