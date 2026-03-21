"""
C2-3: Cross-Group Tree Node Caching

Key observation: Different groups processing the same round will likely access
overlapping sets of tree nodes. Within a tile:
- 4 groups, each with 4 desks x 8 lanes = 32 elements
- Total 128 elements per tile
- At round 3, they access tree[7..14] = 8 unique nodes
- Each group does up to 32 loads, but total unique nodes = 8!

If we preload ALL 8 nodes at level 3 before round 3 for ALL groups,
then each group can use arithmetic selection instead of gather.

But C2-1 showed 8-way arithmetic selection is expensive.

New idea: What if we preload tree nodes DURING the previous round's hash computation?
- Round 2's hash is happening for group 0
- At the same time, preload tree[7..14] for round 3
- This uses "free" load cycles during VALU-heavy hash

Let me calculate the timing:
- Hash has 18 VALU ops per desk (6 stages x 3 ops)
- For 4 desks: 72 VALU ops
- At 6 VALU/cycle = 12 cycles
- During these 12 cycles, we could do 24 loads (2/cycle)
- We only need 8 loads to preload tree[7..14]

So we CAN preload level 3 during round 2's hash!

The question is: Does this help?
- We save 32 gather loads per group (4x8=32)
- We add 8 broadcast loads (shared) + arithmetic selection ops

Let me implement: Preload tree levels 3 and 4 (15+31 nodes = 46 loads) during
early round hashes, then use arithmetic selection for rounds 3 and 4.

Actually, level 4 has 16 nodes (indices 15-30), which is too many for efficient
arithmetic selection. Let's just focus on level 3 (8 nodes, indices 7-14).

Strategy:
1. During round 0's hash, preload tree[7..14] (8 loads)
2. At round 3, use 8-way arithmetic selection

Wait, but C2-1 already tried this and it was worse. The issue is that
arithmetic selection adds VALU pressure to an already VALU-saturated system.

Let me think differently. What if we could REDUCE the number of rounds
instead of reducing gathers per round?

The kernel does 16 rounds. What if some rounds could be combined?

Actually, the problem states we need 16 rounds for correctness.

What about lazy evaluation? What if we don't compute round N until we know
we need its result?

No, the whole batch needs to be processed correctly.

Let me try a different angle: What if we increase PARALLELISM by processing
MORE elements simultaneously, even at the cost of more scratch space?

Current: 16 desks total, 4 desks per group, 4 groups per tile
What if: 24 desks total, 4 desks per group, 6 groups per tile?

This would process 192 elements at once instead of 128.
But we'd need more scratch space.

Current scratch usage: 1184 / 1536 (from C2-2)
Available: 352 more registers

Each desk needs: 7 vector regs = 56 scalars
8 more desks = 8 * 56 = 448 scalars > 352 available

So we can't add more desks.

What about reducing desk size? What if we used 4-wide vectors instead of 8-wide?

No, VLEN=8 is fixed by the architecture.

Let me try: What if we process groups in a different order to maximize overlap?

Current: Process all rounds for group 0, then all for group 1, etc.
Alternative: Process round 3 for groups 0-3 simultaneously, then round 4, etc.

This would require restructuring the code significantly. The potential benefit
is that loads from different groups could overlap better.

Actually, let me try a simpler optimization: what if we start loading for
the next round BEFORE the current round's hash completes?

For round 3:
1. Compute addr = forest_p + idx
2. Start loads for node_val
3. While loads are in flight, do XOR and start hash
4. Hash needs node_val from step 2, so there's a dependency

The dependency chain is:
idx -> addr -> load node_val -> XOR -> hash -> branch -> new_idx

We can't start the next round's address computation until we have new_idx.
But we COULD start loading tree nodes speculatively if we knew which ones
might be needed.

At round 3, idx is in [7..14]. At round 4, idx will be in [15..30].
We could preload ALL of tree[15..30] during round 3's hash, then select
the right one arithmetically.

16 preloads during hash (12+ cycles available) = feasible
16-way arithmetic selection = expensive (15 vselects or equivalent)

This doesn't seem promising either.

Let me try the simplest possible optimization: just ensure the baseline
(B4-2 code) is optimal. Maybe there's a scheduling improvement.
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
        if slot[0] == "vbroadcast":
            dest, src = slot[1], slot[2]
            reads = [src]
            writes = list(_vec_range(dest))
        elif slot[0] == "multiply_add":
            dest, a, b, c = slot[1], slot[2], slot[3], slot[4]
            reads = list(_vec_range(a)) + list(_vec_range(b)) + list(_vec_range(c))
            writes = list(_vec_range(dest))
        else:
            _op, dest, a1, a2 = slot
            reads = list(_vec_range(a1)) + list(_vec_range(a2))
            writes = list(_vec_range(dest))
    elif engine == "load":
        if slot[0] == "load":
            dest, addr = slot[1], slot[2]
            reads = [addr]
            writes = [dest]
        elif slot[0] == "vload":
            dest, addr = slot[1], slot[2]
            reads = [addr]
            writes = list(_vec_range(dest))
        elif slot[0] == "const":
            dest, _val = slot[1], slot[2]
            writes = [dest]
        elif slot[0] == "load_offset":
            dest, addr, _lane = slot[1], slot[2], slot[3]
            reads = [addr]
            writes = [dest]
        else:
            raise NotImplementedError(f"Unknown load op {slot}")
    elif engine == "store":
        if slot[0] == "store":
            addr, src = slot[1], slot[2]
            reads = [addr, src]
        elif slot[0] == "vstore":
            addr, src = slot[1], slot[2]
            reads = [addr] + list(_vec_range(src))
        else:
            raise NotImplementedError(f"Unknown store op {slot}")
    elif engine == "flow":
        if slot[0] == "select":
            dest, cond, a, b = slot[1], slot[2], slot[3], slot[4]
            reads = [cond, a, b]
            writes = [dest]
        elif slot[0] == "add_imm":
            dest, a, _imm = slot[1], slot[2], slot[3]
            reads = [a]
            writes = [dest]
        elif slot[0] == "vselect":
            dest, cond, a, b = slot[1], slot[2], slot[3], slot[4]
            reads = list(_vec_range(cond)) + list(_vec_range(a)) + list(_vec_range(b))
            writes = list(_vec_range(dest))
        elif slot[0] in ("halt", "pause"):
            pass
        elif slot[0] == "trace_write":
            pass
        elif slot[0] in ("jump", "jump_indirect"):
            pass
        elif slot[0] in ("cond_jump", "cond_jump_rel"):
            pass
        elif slot[0] == "coreid":
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


class KernelBuilderC2_3:
    """
    C2-3: Try preloading level 3 nodes during round 2's hash, then use
    more efficient selection. Focus on minimizing total cycle count.

    Hypothesis: The greedy scheduler might not be finding the optimal
    schedule for the hash + preload overlap. Let's try explicit prefetching.
    """

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

        # Preload tree nodes 0-6 (levels 0-2)
        NUM_PRELOADED = 7
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
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("multiply_add", desk['val'], desk['val'], v_fma_mult[0], v_hash_consts[0]))
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("^", desk['tmp1'], desk['val'], v_hash_consts[1]))
                self.emit("valu", (">>", desk['tmp2'], desk['val'], v_hash_shifts[1]))
                self.emit("valu", ("^", desk['val'], desk['tmp1'], desk['tmp2']))
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("multiply_add", desk['val'], desk['val'], v_fma_mult[2], v_hash_consts[2]))
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("+", desk['tmp1'], desk['val'], v_hash_consts[3]))
                self.emit("valu", ("<<", desk['tmp2'], desk['val'], v_hash_shifts[3]))
                self.emit("valu", ("^", desk['val'], desk['tmp1'], desk['tmp2']))
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("multiply_add", desk['val'], desk['val'], v_fma_mult[4], v_hash_consts[4]))
            for d in group_desks:
                desk = desks[d]
                self.emit("valu", ("^", desk['tmp1'], desk['val'], v_hash_consts[5]))
                self.emit("valu", (">>", desk['tmp2'], desk['val'], v_hash_shifts[5]))
                self.emit("valu", ("^", desk['val'], desk['tmp1'], desk['tmp2']))

        def emit_branch(desk_idx):
            d = desks[desk_idx]
            self.emit("valu", ("&", d['tmp1'], d['val'], v_one))
            self.emit("valu", ("multiply_add", d['idx'], d['idx'], v_two, v_one))
            self.emit("valu", ("+", d['idx'], d['idx'], d['tmp1']))

        def emit_bounds(desk_idx):
            d = desks[desk_idx]
            self.emit("valu", ("<", d['tmp1'], d['idx'], v_n_nodes))
            self.emit("valu", ("*", d['idx'], d['idx'], d['tmp1']))

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

        def emit_round_10_interleaved(group_desks):
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
            for d in group_desks:
                emit_bounds(d)

        def emit_round_15_final_interleaved(group_desks):
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
                emit_rounds_0_1_2_fused(group_desks)

                for _rnd in range(3, 10):
                    emit_gather_round_interleaved(group_desks)

                emit_round_10_interleaved(group_desks)
                emit_rounds_11_12_13_fused(group_desks)
                emit_gather_round_interleaved(group_desks)  # round 14
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

        print(f"Total slots: {len(self.slots)}, Cycles: {len(self.instrs)}")


def do_kernel_test(forest_height: int, rounds: int, batch_size: int, seed: int = 123,
                   trace: bool = False, prints: bool = False, check: bool = False):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilderC2_3()
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
        print(f"Correctness check PASSED! Cycles: {cycles}")
    else:
        do_kernel_test(10, 16, 256, trace=args.trace)
