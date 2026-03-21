"""
Theory 401: Fuse R15 into the R11-R14 block using level-4 tree preloading.

Changes from Theory 222 + Theory 330 (best = 1395):
1. Preload 31 tree nodes (levels 0-4, tree[0..30])
2. R11-R15 fused: R15 uses 15-vselect cascade instead of gather
3. R15 still needs XOR + hash (but no branch, no gather loads)
4. Deferred addr NOT needed for R15 (it's the final round)
5. Use per-desk interleaving for gather rounds (GATHER_PERM=(1,2,3,0))

Expected: fewer load-bound cycles, more flow ops but overlapping with VALU.
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
        self.scratch_ptr+=length; assert self.scratch_ptr<=SCRATCH_SIZE; return addr
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

        v_zero=self.scratch_vconst(0); v_one=self.scratch_vconst(1); v_two=self.scratch_vconst(2)
        v_n_nodes=self.alloc_vec(); self.emit("valu",("vbroadcast",v_n_nodes,self.scratch["n_nodes"]))
        v_forest_p=self.alloc_vec(); self.emit("valu",("vbroadcast",v_forest_p,self.scratch["forest_values_p"]))
        v_1_minus_fp=self.alloc_vec(); self.emit("valu",("-",v_1_minus_fp,v_one,v_forest_p))
        v_fp_plus_1=self.alloc_vec(); self.emit("valu",("+",v_fp_plus_1,v_forest_p,v_one))
        v_fp_plus_15=self.alloc_vec(); self.emit("valu",("+",v_fp_plus_15,v_forest_p,self.scratch_vconst(15)))
        # Theory 401: v_fp_plus_31 for 5-round deferred addr
        v_fp_plus_31=self.alloc_vec("v_fp_plus_31"); self.emit("valu",("+",v_fp_plus_31,v_forest_p,self.scratch_vconst(31,"v_31")))

        FMA_MULTIPLIERS={0:4097,2:33,4:9}
        v_hash_consts,v_hash_shifts,v_fma_mult=[],[],{}
        for hi,(op1,val1,op2,op3,val3) in enumerate(HASH_STAGES):
            v_hash_consts.append(self.scratch_vconst(val1))
            if hi in FMA_MULTIPLIERS: v_fma_mult[hi]=self.scratch_vconst(FMA_MULTIPLIERS[hi]); v_hash_shifts.append(None)
            else: v_hash_shifts.append(self.scratch_vconst(val3))

        # Theory 401: Preload 31 tree nodes (levels 0-4)
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
                'node_val':self.alloc_vec(f"v_node_{d}"),'addr':self.alloc_vec(f"v_addr_{d}"),
                'tmp1':self.alloc_vec(f"v_tmp1_{d}"),'tmp2':self.alloc_vec(f"v_tmp2_{d}"),
                'bit0':self.alloc_vec(f"v_bit0_{d}"),'bit1':self.alloc_vec(f"v_bit1_{d}"),
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
                self.emit("valu",("^",dk['tmp1'],dk['val'],v_hash_consts[1]))
                self.emit("valu",(">>",dk['tmp2'],dk['val'],v_hash_shifts[1]))
                self.emit("valu",("^",dk['val'],dk['tmp1'],dk['tmp2']))
                self.emit("valu",("multiply_add",dk['val'],dk['val'],v_fma_mult[2],v_hash_consts[2]))
                self.emit("valu",("+",dk['tmp1'],dk['val'],v_hash_consts[3]))
                self.emit("valu",("<<",dk['tmp2'],dk['val'],v_hash_shifts[3]))
                self.emit("valu",("^",dk['val'],dk['tmp1'],dk['tmp2']))
                self.emit("valu",("multiply_add",dk['val'],dk['val'],v_fma_mult[4],v_hash_consts[4]))
                self.emit("valu",("^",dk['tmp1'],dk['val'],v_hash_consts[5]))
                self.emit("valu",(">>",dk['tmp2'],dk['val'],v_hash_shifts[5]))
                self.emit("valu",("^",dk['val'],dk['tmp1'],dk['tmp2']))

        def emit_desk_hash(d):
            dk=desks[d]
            self.emit("valu",("multiply_add",dk['val'],dk['val'],v_fma_mult[0],v_hash_consts[0]))
            self.emit("valu",("^",dk['tmp1'],dk['val'],v_hash_consts[1]))
            self.emit("valu",(">>",dk['tmp2'],dk['val'],v_hash_shifts[1]))
            self.emit("valu",("^",dk['val'],dk['tmp1'],dk['tmp2']))
            self.emit("valu",("multiply_add",dk['val'],dk['val'],v_fma_mult[2],v_hash_consts[2]))
            self.emit("valu",("+",dk['tmp1'],dk['val'],v_hash_consts[3]))
            self.emit("valu",("<<",dk['tmp2'],dk['val'],v_hash_shifts[3]))
            self.emit("valu",("^",dk['val'],dk['tmp1'],dk['tmp2']))
            self.emit("valu",("multiply_add",dk['val'],dk['val'],v_fma_mult[4],v_hash_consts[4]))
            self.emit("valu",("^",dk['tmp1'],dk['val'],v_hash_consts[5]))
            self.emit("valu",(">>",dk['tmp2'],dk['val'],v_hash_shifts[5]))
            self.emit("valu",("^",dk['val'],dk['tmp1'],dk['tmp2']))

        def emit_branch_addr(di):
            d=desks[di]
            self.emit("valu",("&",d['tmp1'],d['val'],v_one))
            self.emit("valu",("multiply_add",d['addr'],d['addr'],v_two,v_1_minus_fp))
            self.emit("valu",("+",d['addr'],d['addr'],d['tmp1']))

        # Standard R0-R3 fused (same as Theory 222)
        def emit_rounds_0_1_2_3_fused(gd):
            for d in gd: self.emit("valu",("^",desks[d]['val'],desks[d]['val'],v_tree[0]))
            emit_hash_interleaved(gd)
            for d in gd: self.emit("valu",("&",desks[d]['bit0'],desks[d]['val'],v_one))
            for d in gd: self.emit("flow",("vselect",desks[d]['node_val'],desks[d]['bit0'],v_tree[2],v_tree[1]))
            for d in gd: self.emit("valu",("^",desks[d]['val'],desks[d]['val'],desks[d]['node_val']))
            emit_hash_interleaved(gd)
            for d in gd: self.emit("valu",("&",desks[d]['bit1'],desks[d]['val'],v_one))
            for d in gd:
                dk=desks[d]
                self.emit("flow",("vselect",dk['tmp2'],dk['bit1'],v_tree[4],v_tree[3]))
                self.emit("flow",("vselect",dk['node_val'],dk['bit1'],v_tree[6],v_tree[5]))
                self.emit("flow",("vselect",dk['node_val'],dk['bit0'],dk['node_val'],dk['tmp2']))
            for d in gd: self.emit("valu",("^",desks[d]['val'],desks[d]['val'],desks[d]['node_val']))
            emit_hash_interleaved(gd)
            for d in gd: self.emit("valu",("&",desks[d]['idx'],desks[d]['val'],v_one))  # bit2 in idx
            for d in gd:
                dk=desks[d]
                self.emit("flow",("vselect",dk['tmp2'],dk['idx'],v_tree[8],v_tree[7]))
                self.emit("flow",("vselect",dk['node_val'],dk['idx'],v_tree[10],v_tree[9]))
                self.emit("flow",("vselect",dk['tmp2'],dk['bit1'],dk['node_val'],dk['tmp2']))
                self.emit("flow",("vselect",dk['node_val'],dk['idx'],v_tree[12],v_tree[11]))
                self.emit("flow",("vselect",dk['addr'],dk['idx'],v_tree[14],v_tree[13]))
                self.emit("flow",("vselect",dk['node_val'],dk['bit1'],dk['addr'],dk['node_val']))
                self.emit("flow",("vselect",dk['node_val'],dk['bit0'],dk['node_val'],dk['tmp2']))
            for d in gd: self.emit("valu",("^",desks[d]['val'],desks[d]['val'],desks[d]['node_val']))
            emit_hash_interleaved(gd)
            # Deferred 4-bit addr
            for d in gd:
                dk=desks[d]
                self.emit("valu",("&",dk['tmp1'],dk['val'],v_one))  # bit3
                self.emit("valu",("multiply_add",dk['addr'],dk['bit0'],v_two,dk['bit1']))
                self.emit("valu",("multiply_add",dk['addr'],dk['addr'],v_two,dk['idx']))
                self.emit("valu",("multiply_add",dk['addr'],dk['addr'],v_two,dk['tmp1']))
                self.emit("valu",("+",dk['addr'],dk['addr'],v_fp_plus_15))

        # Theory 401: R11-R15 fused (5 rounds)
        def emit_rounds_11_to_15_fused(gd):
            # R11: same as R0
            for d in gd: self.emit("valu",("^",desks[d]['val'],desks[d]['val'],v_tree[0]))
            emit_hash_interleaved(gd)
            for d in gd: self.emit("valu",("&",desks[d]['bit0'],desks[d]['val'],v_one))
            
            # R12: same as R1
            for d in gd: self.emit("flow",("vselect",desks[d]['node_val'],desks[d]['bit0'],v_tree[2],v_tree[1]))
            for d in gd: self.emit("valu",("^",desks[d]['val'],desks[d]['val'],desks[d]['node_val']))
            emit_hash_interleaved(gd)
            for d in gd: self.emit("valu",("&",desks[d]['bit1'],desks[d]['val'],v_one))
            
            # R13: same as R2
            for d in gd:
                dk=desks[d]
                self.emit("flow",("vselect",dk['tmp2'],dk['bit1'],v_tree[4],v_tree[3]))
                self.emit("flow",("vselect",dk['node_val'],dk['bit1'],v_tree[6],v_tree[5]))
                self.emit("flow",("vselect",dk['node_val'],dk['bit0'],dk['node_val'],dk['tmp2']))
            for d in gd: self.emit("valu",("^",desks[d]['val'],desks[d]['val'],desks[d]['node_val']))
            emit_hash_interleaved(gd)
            for d in gd: self.emit("valu",("&",desks[d]['idx'],desks[d]['val'],v_one))  # bit2 in idx
            
            # R14: same as R3 (7-vselect for level 3)
            for d in gd:
                dk=desks[d]
                self.emit("flow",("vselect",dk['tmp2'],dk['idx'],v_tree[8],v_tree[7]))
                self.emit("flow",("vselect",dk['node_val'],dk['idx'],v_tree[10],v_tree[9]))
                self.emit("flow",("vselect",dk['tmp2'],dk['bit1'],dk['node_val'],dk['tmp2']))
                self.emit("flow",("vselect",dk['node_val'],dk['idx'],v_tree[12],v_tree[11]))
                self.emit("flow",("vselect",dk['addr'],dk['idx'],v_tree[14],v_tree[13]))
                self.emit("flow",("vselect",dk['node_val'],dk['bit1'],dk['addr'],dk['node_val']))
                self.emit("flow",("vselect",dk['node_val'],dk['bit0'],dk['node_val'],dk['tmp2']))
            for d in gd: self.emit("valu",("^",desks[d]['val'],desks[d]['val'],desks[d]['node_val']))
            emit_hash_interleaved(gd)
            for d in gd: self.emit("valu",("&",desks[d]['addr'],desks[d]['val'],v_one))  # bit3 in addr
            
            # R15: 15-vselect cascade for level 4 (tree[15..30])
            # Select tree[15 + 8*bit0 + 4*bit1 + 2*bit2 + bit3]
            # where bit2=idx, bit3=addr
            for d in gd:
                dk=desks[d]
                # bit3=0 subtree: tree[15,17,19,21,23,25,27,29]
                # Use bit2(=idx) to select pairs within each bit1 group
                self.emit("flow",("vselect",dk['tmp1'],dk['idx'],v_tree[17],v_tree[15]))  # bit2 sel in (15,17)
                self.emit("flow",("vselect",dk['tmp2'],dk['idx'],v_tree[21],v_tree[19]))  # bit2 sel in (19,21)
                self.emit("flow",("vselect",dk['tmp1'],dk['bit1'],dk['tmp2'],dk['tmp1']))  # bit1 sel
                self.emit("flow",("vselect",dk['tmp2'],dk['idx'],v_tree[25],v_tree[23]))  # bit2 sel in (23,25)
                self.emit("flow",("vselect",dk['node_val'],dk['idx'],v_tree[29],v_tree[27]))  # bit2 sel in (27,29)
                self.emit("flow",("vselect",dk['tmp2'],dk['bit1'],dk['node_val'],dk['tmp2']))  # bit1 sel
                self.emit("flow",("vselect",dk['tmp1'],dk['bit0'],dk['tmp2'],dk['tmp1']))  # bit0 sel -> b3=0 result in tmp1
                
                # bit3=1 subtree: tree[16,18,20,22,24,26,28,30]
                self.emit("flow",("vselect",dk['tmp2'],dk['idx'],v_tree[18],v_tree[16]))
                self.emit("flow",("vselect",dk['node_val'],dk['idx'],v_tree[22],v_tree[20]))
                self.emit("flow",("vselect",dk['tmp2'],dk['bit1'],dk['node_val'],dk['tmp2']))
                self.emit("flow",("vselect",dk['node_val'],dk['idx'],v_tree[26],v_tree[24]))
                # Need another temp for the last pair... use idx temporarily (bit2 no longer needed after)
                self.emit("flow",("vselect",dk['idx'],dk['idx'],v_tree[30],v_tree[28]))
                self.emit("flow",("vselect",dk['node_val'],dk['bit1'],dk['idx'],dk['node_val']))
                self.emit("flow",("vselect",dk['tmp2'],dk['bit0'],dk['node_val'],dk['tmp2']))  # b3=1 result in tmp2
                
                # Final: select between b3=0 (tmp1) and b3=1 (tmp2) using addr (=bit3)
                self.emit("flow",("vselect",dk['node_val'],dk['addr'],dk['tmp2'],dk['tmp1']))

            # XOR with selected node
            for d in gd: self.emit("valu",("^",desks[d]['val'],desks[d]['val'],desks[d]['node_val']))
            # Hash (final round, no branch needed)
            emit_hash_interleaved(gd)
            # NO branch, NO addr computation (final round)

        def emit_gather_interleaved(gd):
            ho=[gd[GATHER_PERM[i]] for i in range(4)]
            for d in ho:
                dk=desks[d]
                for lane in range(VLEN): self.emit("load",("load",dk['node_val']+lane,dk['addr']+lane))
                self.emit("valu",("^",dk['val'],dk['val'],dk['node_val']))
                emit_desk_hash(d); emit_branch_addr(d)

        def emit_r10(gd):
            for d in gd:
                dk=desks[d]
                for lane in range(VLEN): self.emit("load",("load",dk['node_val']+lane,dk['addr']+lane))
            for d in gd: self.emit("valu",("^",desks[d]['val'],desks[d]['val'],desks[d]['node_val']))
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
                emit_rounds_0_1_2_3_fused(gd)
                for _ in range(4,10): emit_gather_interleaved(gd)
                emit_r10(gd)
                # Theory 401: R11-R15 fused (replaces R11-R14 fused + R15 gather)
                emit_rounds_11_to_15_fused(gd)

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
        flow_count=sum(1 for e,s in self.slots if e=="flow")
        load_count=sum(1 for e,s in self.slots if e=="load")
        print(f"Total slots: {len(self.slots)}, VALU: {valu_count}, Flow: {flow_count}, Load: {load_count}, Cycles: {len(self.instrs)}")

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
        print(f"Correctness check {'PASSED' if cycles else 'FAILED'}! Cycles: {cycles}")
    else: do_kernel_test(trace=args.trace)
