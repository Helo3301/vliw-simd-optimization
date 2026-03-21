"""
Fine-grained sweep: try many small variations of the 1536-cycle kernel
to see if any produce even 1 cycle improvement.
"""
import sys
import os
import random
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from problem import (
    Engine, DebugInfo, SLOT_LIMITS, VLEN, N_CORES, SCRATCH_SIZE,
    Machine, Tree, Input, HASH_STAGES, reference_kernel, build_mem_image, reference_kernel2,
)

def _vec_range(base, length=VLEN):
    return range(base, base + length)

def _slot_rw(engine, slot):
    reads, writes = [], []
    if engine == "alu":
        _op, dest, a1, a2 = slot
        reads = [a1, a2]; writes = [dest]
    elif engine == "valu":
        op = slot[0]
        if op == "vbroadcast":
            reads = [slot[2]]; writes = list(_vec_range(slot[1]))
        elif op == "multiply_add":
            reads = list(_vec_range(slot[2])) + list(_vec_range(slot[3])) + list(_vec_range(slot[4]))
            writes = list(_vec_range(slot[1]))
        else:
            reads = list(_vec_range(slot[2])) + list(_vec_range(slot[3]))
            writes = list(_vec_range(slot[1]))
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
    def ensure(c):
        while len(cycles) <= c: cycles.append({}); usage.append(defaultdict(int))
    def find(eng, earliest):
        c = earliest
        while True:
            ensure(c)
            if usage[c][eng] < SLOT_LIMITS[eng]: return c
            c += 1
    for engine, slot in slots:
        reads, writes = _slot_rw(engine, slot)
        earliest = 0
        for a in reads: earliest = max(earliest, ready_time[a])
        for a in writes: earliest = max(earliest, last_write[a] + 1, last_read[a])
        c = find(engine, earliest)
        ensure(c)
        cycles[c].setdefault(engine, []).append(slot)
        usage[c][engine] += 1
        for a in reads:
            if last_read[a] < c: last_read[a] = c
        for a in writes: last_write[a] = c; ready_time[a] = c + 1
    return [c for c in cycles if c]


def build_and_test(hash_fn, branch_fn=None, check=True):
    """Build kernel with custom hash and branch emission functions."""
    random.seed(123)
    forest = Tree.generate(10)
    inp = Input.generate(forest, 256, 16)
    mem = build_mem_image(forest, inp)

    slots = []
    scratch = {}
    scratch_debug = {}
    scratch_ptr = [0]
    const_map = {}
    vconst_map = {}

    def emit(engine, slot):
        slots.append((engine, slot))

    def alloc_scratch(name=None, length=1):
        addr = scratch_ptr[0]
        if name is not None:
            scratch[name] = addr
            scratch_debug[addr] = (name, length)
        scratch_ptr[0] += length
        assert scratch_ptr[0] <= SCRATCH_SIZE
        return addr

    def alloc_vec(name=None):
        return alloc_scratch(name, VLEN)

    def scratch_const(val, name=None):
        if val not in const_map:
            addr = alloc_scratch(name or f"c_{val}")
            emit("load", ("const", addr, val))
            const_map[val] = addr
        return const_map[val]

    def scratch_vconst(val, name=None):
        if val not in vconst_map:
            scalar = scratch_const(val)
            addr = alloc_vec(name or f"v_{val}")
            emit("valu", ("vbroadcast", addr, scalar))
            vconst_map[val] = addr
        return vconst_map[val]

    tmp_scalar = alloc_scratch("tmp_scalar")
    tmp_addr = alloc_scratch("tmp_addr")
    for var_name, _ in [("n_nodes", 1), ("forest_values_p", 4), ("inp_indices_p", 5), ("inp_values_p", 6)]:
        alloc_scratch(var_name)
    for var_name, idx in [("n_nodes", 1), ("forest_values_p", 4), ("inp_indices_p", 5), ("inp_values_p", 6)]:
        emit("load", ("const", tmp_scalar, idx))
        emit("load", ("load", scratch[var_name], tmp_scalar))

    v_zero = scratch_vconst(0, "v_zero")
    v_one = scratch_vconst(1, "v_one")
    v_two = scratch_vconst(2, "v_two")
    v_three = scratch_vconst(3, "v_three")
    v_n_nodes = alloc_vec("v_n_nodes")
    emit("valu", ("vbroadcast", v_n_nodes, scratch["n_nodes"]))
    v_forest_p = alloc_vec("v_forest_p")
    emit("valu", ("vbroadcast", v_forest_p, scratch["forest_values_p"]))

    FMA_MULTIPLIERS = {0: 4097, 2: 33, 4: 9}
    v_hash_consts, v_hash_shifts, v_fma_mult = [], [], {}
    for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
        v_const = scratch_vconst(val1, f"v_hash_const_{hi}")
        v_hash_consts.append(v_const)
        if hi in FMA_MULTIPLIERS:
            v_fma_mult[hi] = scratch_vconst(FMA_MULTIPLIERS[hi], f"v_fma_mult_{hi}")
            v_hash_shifts.append(None)
        else:
            v_shift = scratch_vconst(val3, f"v_hash_shift_{hi}")
            v_hash_shifts.append(v_shift)

    v_tree = []
    for i in range(7):
        v_node = alloc_vec(f"v_tree_{i}")
        v_tree.append(v_node)
        emit("alu", ("+", tmp_addr, scratch["forest_values_p"], scratch_const(i)))
        emit("load", ("load", tmp_scalar, tmp_addr))
        emit("valu", ("vbroadcast", v_node, tmp_scalar))

    v_diff_1_2 = alloc_vec("v_diff_1_2")
    v_diff_3_4 = alloc_vec("v_diff_3_4")
    v_diff_5_6 = alloc_vec("v_diff_5_6")
    emit("valu", ("-", v_diff_1_2, v_tree[2], v_tree[1]))
    emit("valu", ("-", v_diff_3_4, v_tree[4], v_tree[3]))
    emit("valu", ("-", v_diff_5_6, v_tree[6], v_tree[5]))

    NUM_DESKS = 16
    desks = []
    for d in range(NUM_DESKS):
        desk = {
            'idx': alloc_vec(f"v_idx_{d}"), 'val': alloc_vec(f"v_val_{d}"),
            'node_val': alloc_vec(f"v_node_{d}"), 'addr': alloc_vec(f"v_addr_{d}"),
            'tmp1': alloc_vec(f"v_tmp1_{d}"), 'tmp2': alloc_vec(f"v_tmp2_{d}"),
            'bit0': alloc_vec(f"v_bit0_{d}"),
        }
        desks.append(desk)

    offset_regs = [alloc_scratch(f"off_{d}") for d in range(NUM_DESKS)]
    addr_tmp = [alloc_scratch(f"addr_tmp_{i}") for i in range(32)]

    emit("flow", ("pause",))

    # Closures for hash and branch
    ctx = {
        'emit': emit, 'desks': desks, 'v_fma_mult': v_fma_mult,
        'v_hash_consts': v_hash_consts, 'v_hash_shifts': v_hash_shifts,
        'v_one': v_one, 'v_two': v_two, 'v_three': v_three,
        'v_forest_p': v_forest_p, 'v_tree': v_tree,
        'v_diff_1_2': v_diff_1_2, 'v_diff_3_4': v_diff_3_4, 'v_diff_5_6': v_diff_5_6,
    }

    def default_branch(desk_idx):
        d = desks[desk_idx]
        emit("valu", ("&", d['tmp1'], d['val'], v_one))
        emit("valu", ("multiply_add", d['idx'], d['idx'], v_two, v_one))
        emit("valu", ("+", d['idx'], d['idx'], d['tmp1']))

    if branch_fn is None:
        branch_fn_actual = default_branch
    else:
        branch_fn_actual = lambda di: branch_fn(di, ctx)

    def emit_xor(desk_idx, node_vec):
        emit("valu", ("^", desks[desk_idx]['val'], desks[desk_idx]['val'], node_vec))

    def do_hash(group_desks):
        hash_fn(group_desks, ctx)

    def emit_rounds_0_1_2(gd):
        for d in gd: emit_xor(d, v_tree[0])
        do_hash(gd)
        for d in gd:
            desk = desks[d]
            emit("valu", ("&", desk['bit0'], desk['val'], v_one))
            emit("valu", ("+", desk['idx'], v_one, desk['bit0']))
        for d in gd:
            desk = desks[d]
            emit("valu", ("multiply_add", desk['node_val'], desk['bit0'], v_diff_1_2, v_tree[1]))
        for d in gd:
            emit("valu", ("^", desks[d]['val'], desks[d]['val'], desks[d]['node_val']))
        do_hash(gd)
        for d in gd:
            desk = desks[d]
            emit("valu", ("&", desk['tmp1'], desk['val'], v_one))
            emit("valu", ("multiply_add", desk['idx'], desk['bit0'], v_two, v_three))
            emit("valu", ("+", desk['idx'], desk['idx'], desk['tmp1']))
        for d in gd:
            desk = desks[d]
            emit("valu", ("multiply_add", desk['node_val'], desk['tmp1'], v_diff_3_4, v_tree[3]))
            emit("valu", ("multiply_add", desk['tmp2'], desk['tmp1'], v_diff_5_6, v_tree[5]))
            emit("flow", ("vselect", desk['node_val'], desk['bit0'], desk['tmp2'], desk['node_val']))
        for d in gd:
            emit("valu", ("^", desks[d]['val'], desks[d]['val'], desks[d]['node_val']))
        do_hash(gd)
        for d in gd: default_branch(d)

    def emit_rounds_11_12_13(gd):
        for d in gd: emit_xor(d, v_tree[0])
        do_hash(gd)
        for d in gd:
            desk = desks[d]
            emit("valu", ("&", desk['bit0'], desk['val'], v_one))
            emit("valu", ("+", desk['idx'], v_one, desk['bit0']))
        for d in gd:
            desk = desks[d]
            emit("valu", ("multiply_add", desk['node_val'], desk['bit0'], v_diff_1_2, v_tree[1]))
        for d in gd:
            emit("valu", ("^", desks[d]['val'], desks[d]['val'], desks[d]['node_val']))
        do_hash(gd)
        for d in gd:
            desk = desks[d]
            emit("valu", ("&", desk['tmp1'], desk['val'], v_one))
            emit("valu", ("multiply_add", desk['idx'], desk['bit0'], v_two, v_three))
            emit("valu", ("+", desk['idx'], desk['idx'], desk['tmp1']))
        for d in gd:
            desk = desks[d]
            emit("valu", ("multiply_add", desk['node_val'], desk['tmp1'], v_diff_3_4, v_tree[3]))
            emit("valu", ("multiply_add", desk['tmp2'], desk['tmp1'], v_diff_5_6, v_tree[5]))
            emit("flow", ("vselect", desk['node_val'], desk['bit0'], desk['tmp2'], desk['node_val']))
        for d in gd:
            emit("valu", ("^", desks[d]['val'], desks[d]['val'], desks[d]['node_val']))
        do_hash(gd)
        for d in gd: default_branch(d)

    def emit_gather(gd):
        for d in gd: emit("valu", ("+", desks[d]['addr'], v_forest_p, desks[d]['idx']))
        for d in gd:
            for lane in range(VLEN):
                emit("load", ("load", desks[d]['node_val'] + lane, desks[d]['addr'] + lane))
        for d in gd: emit("valu", ("^", desks[d]['val'], desks[d]['val'], desks[d]['node_val']))
        do_hash(gd)
        for d in gd: default_branch(d)

    def emit_r10(gd):
        for d in gd: emit("valu", ("+", desks[d]['addr'], v_forest_p, desks[d]['idx']))
        for d in gd:
            for lane in range(VLEN):
                emit("load", ("load", desks[d]['node_val'] + lane, desks[d]['addr'] + lane))
        for d in gd: emit("valu", ("^", desks[d]['val'], desks[d]['val'], desks[d]['node_val']))
        do_hash(gd)
        for d in gd: emit("valu", ("^", desks[d]['idx'], desks[d]['idx'], desks[d]['idx']))

    def emit_r15(gd):
        for d in gd: emit("valu", ("+", desks[d]['addr'], v_forest_p, desks[d]['idx']))
        for d in gd:
            for lane in range(VLEN):
                emit("load", ("load", desks[d]['node_val'] + lane, desks[d]['addr'] + lane))
        for d in gd: emit("valu", ("^", desks[d]['val'], desks[d]['val'], desks[d]['node_val']))
        do_hash(gd)

    def emit_tile(tile_idx):
        tile_offset = tile_idx * NUM_DESKS * VLEN
        for d in range(NUM_DESKS): emit("load", ("const", offset_regs[d], tile_offset + d * VLEN))
        for d in range(NUM_DESKS):
            emit("alu", ("+", addr_tmp[d*2], scratch["inp_indices_p"], offset_regs[d]))
            emit("alu", ("+", addr_tmp[d*2+1], scratch["inp_values_p"], offset_regs[d]))
        for d in range(NUM_DESKS):
            emit("load", ("vload", desks[d]['idx'], addr_tmp[d*2]))
            emit("load", ("vload", desks[d]['val'], addr_tmp[d*2+1]))
        groups = [list(range(g*4, (g+1)*4)) for g in range(4)]
        for gd in groups:
            emit_rounds_0_1_2(gd)
            for _ in range(3, 10): emit_gather(gd)
            emit_r10(gd)
            emit_rounds_11_12_13(gd)
            emit_gather(gd)
            emit_r15(gd)
        for d in range(NUM_DESKS):
            emit("store", ("vstore", addr_tmp[d*2], desks[d]['idx']))
            emit("store", ("vstore", addr_tmp[d*2+1], desks[d]['val']))

    emit_tile(0)
    emit_tile(1)

    # Schedule
    phases = []
    current_phase = []
    for engine, slot in slots:
        if engine == "flow" and slot == ("pause",):
            phases.append(current_phase)
            current_phase = []
        else:
            current_phase.append((engine, slot))
    phases.append(current_phase)

    instrs = []
    for i, phase in enumerate(phases):
        if phase: instrs.extend(_schedule_slots(phase))
        if i < len(phases) - 1: instrs.append({"flow": [("pause",)]})
    instrs.append({"flow": [("pause",)]})

    # Run
    value_trace = {}
    machine = Machine(mem, instrs, DebugInfo(scratch_map=scratch_debug), n_cores=N_CORES, value_trace=value_trace, trace=False)
    machine.prints = False
    correct = True
    try:
        for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
            machine.run()
            if check:
                inp_values_p = ref_mem[6]
                assert (machine.mem[inp_values_p : inp_values_p + len(inp.values)]
                        == ref_mem[inp_values_p : inp_values_p + len(inp.values)]), f"Incorrect on round {i}"
    except Exception as e:
        correct = False
    return machine.cycle, correct


# Define various hash emission patterns
def hash_per_desk_interleave(group_desks, ctx):
    """Best known: per-desk hash with interleaved desk order"""
    emit = ctx['emit']
    desks = ctx['desks']
    gd = [group_desks[i] for i in range(0, len(group_desks), 2)] + [group_desks[i] for i in range(1, len(group_desks), 2)]
    for d in gd:
        desk = desks[d]
        emit("valu", ("multiply_add", desk['val'], desk['val'], ctx['v_fma_mult'][0], ctx['v_hash_consts'][0]))
        emit("valu", ("^", desk['tmp1'], desk['val'], ctx['v_hash_consts'][1]))
        emit("valu", (">>", desk['tmp2'], desk['val'], ctx['v_hash_shifts'][1]))
        emit("valu", ("^", desk['val'], desk['tmp1'], desk['tmp2']))
        emit("valu", ("multiply_add", desk['val'], desk['val'], ctx['v_fma_mult'][2], ctx['v_hash_consts'][2]))
        emit("valu", ("+", desk['tmp1'], desk['val'], ctx['v_hash_consts'][3]))
        emit("valu", ("<<", desk['tmp2'], desk['val'], ctx['v_hash_shifts'][3]))
        emit("valu", ("^", desk['val'], desk['tmp1'], desk['tmp2']))
        emit("valu", ("multiply_add", desk['val'], desk['val'], ctx['v_fma_mult'][4], ctx['v_hash_consts'][4]))
        emit("valu", ("^", desk['tmp1'], desk['val'], ctx['v_hash_consts'][5]))
        emit("valu", (">>", desk['tmp2'], desk['val'], ctx['v_hash_shifts'][5]))
        emit("valu", ("^", desk['val'], desk['tmp1'], desk['tmp2']))

def hash_per_desk_stride(group_desks, ctx):
    """Per-desk hash with stride-2 desk order: 0,2,1,3"""
    emit = ctx['emit']
    desks = ctx['desks']
    n = len(group_desks)
    if n == 4:
        gd = [group_desks[0], group_desks[2], group_desks[1], group_desks[3]]
    else:
        gd = list(group_desks)
    for d in gd:
        desk = desks[d]
        emit("valu", ("multiply_add", desk['val'], desk['val'], ctx['v_fma_mult'][0], ctx['v_hash_consts'][0]))
        emit("valu", ("^", desk['tmp1'], desk['val'], ctx['v_hash_consts'][1]))
        emit("valu", (">>", desk['tmp2'], desk['val'], ctx['v_hash_shifts'][1]))
        emit("valu", ("^", desk['val'], desk['tmp1'], desk['tmp2']))
        emit("valu", ("multiply_add", desk['val'], desk['val'], ctx['v_fma_mult'][2], ctx['v_hash_consts'][2]))
        emit("valu", ("+", desk['tmp1'], desk['val'], ctx['v_hash_consts'][3]))
        emit("valu", ("<<", desk['tmp2'], desk['val'], ctx['v_hash_shifts'][3]))
        emit("valu", ("^", desk['val'], desk['tmp1'], desk['tmp2']))
        emit("valu", ("multiply_add", desk['val'], desk['val'], ctx['v_fma_mult'][4], ctx['v_hash_consts'][4]))
        emit("valu", ("^", desk['tmp1'], desk['val'], ctx['v_hash_consts'][5]))
        emit("valu", (">>", desk['tmp2'], desk['val'], ctx['v_hash_shifts'][5]))
        emit("valu", ("^", desk['val'], desk['tmp1'], desk['tmp2']))

def hash_per_desk_3_1_0_2(group_desks, ctx):
    """Per-desk hash with order 3,1,0,2"""
    emit = ctx['emit']
    desks = ctx['desks']
    n = len(group_desks)
    if n == 4:
        gd = [group_desks[3], group_desks[1], group_desks[0], group_desks[2]]
    else:
        gd = list(group_desks)
    for d in gd:
        desk = desks[d]
        emit("valu", ("multiply_add", desk['val'], desk['val'], ctx['v_fma_mult'][0], ctx['v_hash_consts'][0]))
        emit("valu", ("^", desk['tmp1'], desk['val'], ctx['v_hash_consts'][1]))
        emit("valu", (">>", desk['tmp2'], desk['val'], ctx['v_hash_shifts'][1]))
        emit("valu", ("^", desk['val'], desk['tmp1'], desk['tmp2']))
        emit("valu", ("multiply_add", desk['val'], desk['val'], ctx['v_fma_mult'][2], ctx['v_hash_consts'][2]))
        emit("valu", ("+", desk['tmp1'], desk['val'], ctx['v_hash_consts'][3]))
        emit("valu", ("<<", desk['tmp2'], desk['val'], ctx['v_hash_shifts'][3]))
        emit("valu", ("^", desk['val'], desk['tmp1'], desk['tmp2']))
        emit("valu", ("multiply_add", desk['val'], desk['val'], ctx['v_fma_mult'][4], ctx['v_hash_consts'][4]))
        emit("valu", ("^", desk['tmp1'], desk['val'], ctx['v_hash_consts'][5]))
        emit("valu", (">>", desk['tmp2'], desk['val'], ctx['v_hash_shifts'][5]))
        emit("valu", ("^", desk['val'], desk['tmp1'], desk['tmp2']))

def hash_per_desk_forward(group_desks, ctx):
    """Per-desk hash in forward order"""
    emit = ctx['emit']
    desks = ctx['desks']
    for d in group_desks:
        desk = desks[d]
        emit("valu", ("multiply_add", desk['val'], desk['val'], ctx['v_fma_mult'][0], ctx['v_hash_consts'][0]))
        emit("valu", ("^", desk['tmp1'], desk['val'], ctx['v_hash_consts'][1]))
        emit("valu", (">>", desk['tmp2'], desk['val'], ctx['v_hash_shifts'][1]))
        emit("valu", ("^", desk['val'], desk['tmp1'], desk['tmp2']))
        emit("valu", ("multiply_add", desk['val'], desk['val'], ctx['v_fma_mult'][2], ctx['v_hash_consts'][2]))
        emit("valu", ("+", desk['tmp1'], desk['val'], ctx['v_hash_consts'][3]))
        emit("valu", ("<<", desk['tmp2'], desk['val'], ctx['v_hash_shifts'][3]))
        emit("valu", ("^", desk['val'], desk['tmp1'], desk['tmp2']))
        emit("valu", ("multiply_add", desk['val'], desk['val'], ctx['v_fma_mult'][4], ctx['v_hash_consts'][4]))
        emit("valu", ("^", desk['tmp1'], desk['val'], ctx['v_hash_consts'][5]))
        emit("valu", (">>", desk['tmp2'], desk['val'], ctx['v_hash_shifts'][5]))
        emit("valu", ("^", desk['val'], desk['tmp1'], desk['tmp2']))

def hash_per_desk_reverse(group_desks, ctx):
    """Per-desk hash in reverse order"""
    emit = ctx['emit']
    desks = ctx['desks']
    for d in reversed(group_desks):
        desk = desks[d]
        emit("valu", ("multiply_add", desk['val'], desk['val'], ctx['v_fma_mult'][0], ctx['v_hash_consts'][0]))
        emit("valu", ("^", desk['tmp1'], desk['val'], ctx['v_hash_consts'][1]))
        emit("valu", (">>", desk['tmp2'], desk['val'], ctx['v_hash_shifts'][1]))
        emit("valu", ("^", desk['val'], desk['tmp1'], desk['tmp2']))
        emit("valu", ("multiply_add", desk['val'], desk['val'], ctx['v_fma_mult'][2], ctx['v_hash_consts'][2]))
        emit("valu", ("+", desk['tmp1'], desk['val'], ctx['v_hash_consts'][3]))
        emit("valu", ("<<", desk['tmp2'], desk['val'], ctx['v_hash_shifts'][3]))
        emit("valu", ("^", desk['val'], desk['tmp1'], desk['tmp2']))
        emit("valu", ("multiply_add", desk['val'], desk['val'], ctx['v_fma_mult'][4], ctx['v_hash_consts'][4]))
        emit("valu", ("^", desk['tmp1'], desk['val'], ctx['v_hash_consts'][5]))
        emit("valu", (">>", desk['tmp2'], desk['val'], ctx['v_hash_shifts'][5]))
        emit("valu", ("^", desk['val'], desk['tmp1'], desk['tmp2']))


if __name__ == "__main__":
    patterns = {
        "per_desk_interleave": hash_per_desk_interleave,
        "per_desk_stride": hash_per_desk_stride,
        "per_desk_3102": hash_per_desk_3_1_0_2,
        "per_desk_forward": hash_per_desk_forward,
        "per_desk_reverse": hash_per_desk_reverse,
    }

    print(f"{'Pattern':<30} | {'Cycles':>6} | {'vs A1':>5} | OK?")
    print("-" * 55)

    for name, fn in patterns.items():
        try:
            cycles, correct = build_and_test(fn, check=True)
            delta = 1548 - cycles
            status = "YES" if correct else "NO"
            marker = " **WIN**" if delta > 0 and correct else ""
            print(f"{name:<30} | {cycles:>6} | {delta:>+5} | {status:>3}{marker}")
        except Exception as e:
            print(f"{name:<30} | ERROR  | N/A   | ERR: {str(e)[:30]}")
