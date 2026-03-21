"""
Theory 350: Use jump instructions to create loops for gather rounds.

Instead of emitting 6 identical gather rounds (R4-R9), emit ONE gather round
and jump back to it 5 more times. This reduces code size and potentially
cycle count (since the scheduler can produce a tighter packing for fewer ops).

The challenge: We need a loop counter and conditional jump.
- Use a scalar scratch register as counter
- Use add_imm to increment counter
- Use cond_jump_rel to loop back
- The gather round code is identical for R4-R9 (addr-tracking branch)

Similarly for R10 -> R11-R14 fused -> R15 -> stores.

Key insight: The scheduled instruction count = cycle count.
If we can loop 6 gather rounds into 1 (used 6 times), the code for that 1 round
is scheduled optimally, and we repeat it. The total cycles would be:
  (1 gather round cycles) * 6 instead of (6 gather rounds scheduled flat)

For a single gather round with 4 desks:
- 32 loads + 4 XOR + 48 hash + 12 branch = 96 ops
- VALU: 64 ops -> ceil(64/6) = 11 cycles
- Loads: 32 ops -> 16 cycles
- So about 16 cycles per round, 96 for 6 rounds
- Currently we have (1400 - 27 init - ~700 fused) / 2 tiles / 4 groups ~= 42 cycles per group per gather section
- 42 cycles / 6 rounds = 7 cycles per round (BETTER than theoretical min?)

Wait, that doesn't make sense. Let me think about this more carefully.

Actually, the flat scheduling produces 1400 cycles. The loop approach would produce:
  init + pause + [fused_0123 + 6*gather_body + R10 + fused_1114 + R15 + stores] * groups * tiles

But we can't loop across groups or tiles because different groups use different desk registers.

Let me try: loop only within a group's gather rounds (R4-R9).
Each group's gather body would be scheduled once and executed 6 times via jumps.
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
        elif op == "cond_jump_rel": reads = [slot[1]]
        elif op == "cond_jump": reads = [slot[1]]
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
        loop_counter = self.alloc_scratch("loop_counter")
        
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

        # We can't use the flat scheduler for loops. Instead, we'll build instruction
        # bundles manually for the loop body, and construct the program directly.
        
        # For now, let me just build the program with the scheduler but add jump instructions
        # manually at the right positions.
        
        # Actually, the simplest approach: emit the gather round code once for a group,
        # schedule it, then insert jump-back at the end.
        # But the challenge is: the scheduler operates on the full slot list.
        # We need to schedule different phases separately and stitch them together with jumps.

        # Let's build the program in phases:
        # Phase 0: Init (before pause)
        # Phase 1: Per tile, per group:
        #   - Fused R0-R3 (schedule as block)
        #   - Gather loop body (schedule as block, loop 6 times via jump)
        #   - R10 (schedule as block)
        #   - Fused R11-R14 (schedule as block)
        #   - R15 (schedule as block)
        #   - Stores

        self.program = []  # Build program directly instead of using self.slots
        
        # Helper to schedule a set of slots and append to program
        def schedule_and_append(slots_list):
            if not slots_list:
                return
            scheduled = _schedule_slots(slots_list)
            self.program.extend(scheduled)
        
        # Init phase (everything before pause)
        init_slots = list(self.slots)  # All the init emissions so far
        schedule_and_append(init_slots)
        self.program.append({"flow": [("pause",)]})
        
        HASH_PERM = (0, 3, 2, 1)
        GATHER_PERM = (1, 2, 3, 0)

        def make_hash_slots(gdesks):
            """Return list of (engine, slot) for hash of group desks"""
            slots = []
            gd = [gdesks[HASH_PERM[i]] for i in range(4)]
            for d in gd:
                dk = desks[d]
                slots.append(("valu", ("multiply_add", dk['val'], dk['val'], v_fma_mult[0], v_hash_consts[0])))
                slots.append(("valu", ("^", dk['tmp1'], dk['val'], v_hash_consts[1])))
                slots.append(("valu", (">>", dk['tmp2'], dk['val'], v_hash_shifts[1])))
                slots.append(("valu", ("^", dk['val'], dk['tmp1'], dk['tmp2'])))
                slots.append(("valu", ("multiply_add", dk['val'], dk['val'], v_fma_mult[2], v_hash_consts[2])))
                slots.append(("valu", ("+", dk['tmp1'], dk['val'], v_hash_consts[3])))
                slots.append(("valu", ("<<", dk['tmp2'], dk['val'], v_hash_shifts[3])))
                slots.append(("valu", ("^", dk['val'], dk['tmp1'], dk['tmp2'])))
                slots.append(("valu", ("multiply_add", dk['val'], dk['val'], v_fma_mult[4], v_hash_consts[4])))
                slots.append(("valu", ("^", dk['tmp1'], dk['val'], v_hash_consts[5])))
                slots.append(("valu", (">>", dk['tmp2'], dk['val'], v_hash_shifts[5])))
                slots.append(("valu", ("^", dk['val'], dk['tmp1'], dk['tmp2'])))
            return slots

        def make_gather_body_slots(gdesks):
            """Slots for one gather round with interleaved per-desk pattern"""
            slots = []
            ho = [gdesks[GATHER_PERM[i]] for i in range(4)]
            for d in ho:
                dk = desks[d]
                for lane in range(VLEN):
                    slots.append(("load", ("load", dk['node_val'] + lane, dk['addr'] + lane)))
                slots.append(("valu", ("^", dk['val'], dk['val'], dk['node_val'])))
                # Hash
                slots.append(("valu", ("multiply_add", dk['val'], dk['val'], v_fma_mult[0], v_hash_consts[0])))
                slots.append(("valu", ("^", dk['tmp1'], dk['val'], v_hash_consts[1])))
                slots.append(("valu", (">>", dk['tmp2'], dk['val'], v_hash_shifts[1])))
                slots.append(("valu", ("^", dk['val'], dk['tmp1'], dk['tmp2'])))
                slots.append(("valu", ("multiply_add", dk['val'], dk['val'], v_fma_mult[2], v_hash_consts[2])))
                slots.append(("valu", ("+", dk['tmp1'], dk['val'], v_hash_consts[3])))
                slots.append(("valu", ("<<", dk['tmp2'], dk['val'], v_hash_shifts[3])))
                slots.append(("valu", ("^", dk['val'], dk['tmp1'], dk['tmp2'])))
                slots.append(("valu", ("multiply_add", dk['val'], dk['val'], v_fma_mult[4], v_hash_consts[4])))
                slots.append(("valu", ("^", dk['tmp1'], dk['val'], v_hash_consts[5])))
                slots.append(("valu", (">>", dk['tmp2'], dk['val'], v_hash_shifts[5])))
                slots.append(("valu", ("^", dk['val'], dk['tmp1'], dk['tmp2'])))
                # Branch
                slots.append(("valu", ("&", dk['tmp1'], dk['val'], v_one)))
                slots.append(("valu", ("multiply_add", dk['addr'], dk['addr'], v_two, v_1_minus_fp)))
                slots.append(("valu", ("+", dk['addr'], dk['addr'], dk['tmp1'])))
            return slots

        def make_fused_0123_slots(gdesks):
            slots = []
            for d in gdesks: slots.append(("valu", ("^", desks[d]['val'], desks[d]['val'], v_tree[0])))
            slots.extend(make_hash_slots(gdesks))
            for d in gdesks: slots.append(("valu", ("&", desks[d]['bit0'], desks[d]['val'], v_one)))
            for d in gdesks: slots.append(("flow", ("vselect", desks[d]['node_val'], desks[d]['bit0'], v_tree[2], v_tree[1])))
            for d in gdesks: slots.append(("valu", ("^", desks[d]['val'], desks[d]['val'], desks[d]['node_val'])))
            slots.extend(make_hash_slots(gdesks))
            for d in gdesks: slots.append(("valu", ("&", desks[d]['bit1'], desks[d]['val'], v_one)))
            for d in gdesks:
                dk = desks[d]
                slots.append(("flow", ("vselect", dk['tmp2'], dk['bit1'], v_tree[4], v_tree[3])))
                slots.append(("flow", ("vselect", dk['node_val'], dk['bit1'], v_tree[6], v_tree[5])))
                slots.append(("flow", ("vselect", dk['node_val'], dk['bit0'], dk['node_val'], dk['tmp2'])))
            for d in gdesks: slots.append(("valu", ("^", desks[d]['val'], desks[d]['val'], desks[d]['node_val'])))
            slots.extend(make_hash_slots(gdesks))
            for d in gdesks: slots.append(("valu", ("&", desks[d]['idx'], desks[d]['val'], v_one)))
            for d in gdesks:
                dk = desks[d]
                slots.append(("flow", ("vselect", dk['tmp2'], dk['idx'], v_tree[8], v_tree[7])))
                slots.append(("flow", ("vselect", dk['node_val'], dk['idx'], v_tree[10], v_tree[9])))
                slots.append(("flow", ("vselect", dk['tmp2'], dk['bit1'], dk['node_val'], dk['tmp2'])))
                slots.append(("flow", ("vselect", dk['node_val'], dk['idx'], v_tree[12], v_tree[11])))
                slots.append(("flow", ("vselect", dk['addr'], dk['idx'], v_tree[14], v_tree[13])))
                slots.append(("flow", ("vselect", dk['node_val'], dk['bit1'], dk['addr'], dk['node_val'])))
                slots.append(("flow", ("vselect", dk['node_val'], dk['bit0'], dk['node_val'], dk['tmp2'])))
            for d in gdesks: slots.append(("valu", ("^", desks[d]['val'], desks[d]['val'], desks[d]['node_val'])))
            slots.extend(make_hash_slots(gdesks))
            for d in gdesks:
                dk = desks[d]
                slots.append(("valu", ("&", dk['tmp1'], dk['val'], v_one)))
                slots.append(("valu", ("multiply_add", dk['addr'], dk['bit0'], v_two, dk['bit1'])))
                slots.append(("valu", ("multiply_add", dk['addr'], dk['addr'], v_two, dk['idx'])))
                slots.append(("valu", ("multiply_add", dk['addr'], dk['addr'], v_two, dk['tmp1'])))
                slots.append(("valu", ("+", dk['addr'], dk['addr'], v_fp_plus_15)))
            return slots

        def make_r10_slots(gdesks):
            slots = []
            for d in gdesks:
                dk = desks[d]
                for lane in range(VLEN):
                    slots.append(("load", ("load", dk['node_val'] + lane, dk['addr'] + lane)))
            for d in gdesks: slots.append(("valu", ("^", desks[d]['val'], desks[d]['val'], desks[d]['node_val'])))
            slots.extend(make_hash_slots(gdesks))
            return slots

        def make_r15_slots(gdesks):
            return make_r10_slots(gdesks)  # Same structure

        # Build program for each tile
        for ti in range(2):
            # Tile init
            tile_init_slots = []
            off = ti * 16 * VLEN
            for d in range(16): tile_init_slots.append(("load", ("const", offset_regs[d], off + d * VLEN)))
            for d in range(16):
                tile_init_slots.append(("alu", ("+", addr_tmp[d*2], self.scratch["inp_indices_p"], offset_regs[d])))
                tile_init_slots.append(("alu", ("+", addr_tmp[d*2+1], self.scratch["inp_values_p"], offset_regs[d])))
            for d in range(16):
                tile_init_slots.append(("load", ("vload", desks[d]['idx'], addr_tmp[d*2])))
                tile_init_slots.append(("load", ("vload", desks[d]['val'], addr_tmp[d*2+1])))
            schedule_and_append(tile_init_slots)

            # Per-group processing
            for g in range(4):
                gd = list(range(g*4, (g+1)*4))

                # Fused R0-R3
                schedule_and_append(make_fused_0123_slots(gd))

                # Gather rounds R4-R9: loop 6 times
                # Set counter to 6
                self.program.append({"load": [("const", loop_counter, 6)]})
                loop_start = len(self.program)

                # Schedule one gather body
                gather_body = _schedule_slots(make_gather_body_slots(gd))
                self.program.extend(gather_body)

                # Decrement counter and conditional jump
                self.program.append({"flow": [("add_imm", loop_counter, loop_counter, -1)]})
                # Jump back if counter != 0
                # cond_jump_rel jumps pc += offset if cond != 0
                # We need to jump back to loop_start
                # Current pc after this instruction = len(self.program) + 1 (after appending)
                # But cond_jump sets pc += offset, where offset is relative to CURRENT pc
                # Actually cond_jump_rel: if cond != 0: pc += offset
                # We're at the instruction we're about to append. After execution, pc = current+1.
                # cond_jump_rel modifies pc. If cond_jump_rel is at index X, after execution without jump pc=X+1.
                # With jump: pc = X + 1 + offset. We want pc = loop_start.
                # So offset = loop_start - (X + 1) = loop_start - len(self.program) - 1
                jump_offset = loop_start - len(self.program) - 1
                self.program.append({"flow": [("cond_jump_rel", loop_counter, jump_offset)]})

                # R10
                schedule_and_append(make_r10_slots(gd))

                # Fused R11-R14
                schedule_and_append(make_fused_0123_slots(gd))

                # R15
                schedule_and_append(make_r15_slots(gd))

            # Stores
            store_slots = []
            for d in range(16):
                store_slots.append(("store", ("vstore", addr_tmp[d*2], desks[d]['idx'])))
                store_slots.append(("store", ("vstore", addr_tmp[d*2+1], desks[d]['val'])))
            schedule_and_append(store_slots)

        self.program.append({"flow": [("pause",)]})

        self.instrs = self.program
        valu_count = sum(1 for e, s in self.slots if e == "valu")
        total_cycles = len(self.instrs)
        print(f"Total program instructions: {total_cycles}")
        # Note: actual cycles will be different because of loop repetition
        # The loop body executes 6 times but only appears once in the program


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
