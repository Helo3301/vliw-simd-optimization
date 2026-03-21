"""
# Experiment H42c: Surgical Broadcast Integration

**ARCHITECTURE:**
- Loop 1: Iterations 1-4 (rounds 0-1) with BROADCAST for round 0
- Loop 2: Iterations 5-32 (rounds 2-15) with exact H38 code

**KEY CHANGE:**
In Loop 1, round 0 uses broadcast (2 cycles) instead of gather (32 cycles).
Round 1 and Loop 2 use normal H38 interleaved gather+hash.

**EXPECTED:**
- Savings: ~120 cycles (30 cycles × 4 iterations)
- Target: < 4,000 cycles (vs H38's 4,062)
"""

from collections import defaultdict
import random
import unittest
import argparse
import sys
import os

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


class KernelBuilderH42c:
    """
    H42c: Surgical integration of broadcast into H38's pipeline.

    Two-loop structure:
    - Loop 1: Rounds 0-1 with broadcast for R0
    - Loop 2: Rounds 2-15 with exact H38 code
    """
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, f"Out of scratch space: {self.scratch_ptr} > {SCRATCH_SIZE}"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        # ============================================================
        # INITIALIZATION (same as H38)
        # ============================================================
        tmp_scalar = self.alloc_scratch("tmp_scalar")
        addr_scalar = self.alloc_scratch("addr_scalar")

        init_vars = [
            "rounds", "n_nodes", "batch_size", "forest_height",
            "forest_values_p", "inp_indices_p", "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp_scalar, i))
            self.add("load", ("load", self.scratch[v], tmp_scalar))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        v_zero = self.alloc_scratch("v_zero", VLEN)
        v_one = self.alloc_scratch("v_one", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)

        self.add("valu", ("vbroadcast", v_zero, zero_const))
        self.add("valu", ("vbroadcast", v_one, one_const))
        self.add("valu", ("vbroadcast", v_two, two_const))
        self.add("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]))

        # Hash constants and FMA multipliers
        FMA_MULTIPLIERS = {0: 4097, 2: 33, 4: 9}

        v_hash_consts = []
        v_hash_shifts = []
        v_fma_mult = {}

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            const_scalar = self.scratch_const(val1)
            shift_scalar = self.scratch_const(val3)
            v_const = self.alloc_scratch(f"v_hash_const_{hi}", VLEN)
            v_shift = self.alloc_scratch(f"v_hash_shift_{hi}", VLEN)
            self.add("valu", ("vbroadcast", v_const, const_scalar))
            self.add("valu", ("vbroadcast", v_shift, shift_scalar))
            v_hash_consts.append(v_const)
            v_hash_shifts.append(v_shift)

            if hi in FMA_MULTIPLIERS:
                mult_scalar = self.scratch_const(FMA_MULTIPLIERS[hi])
                v_mult = self.alloc_scratch(f"v_fma_mult_{hi}", VLEN)
                self.add("valu", ("vbroadcast", v_mult, mult_scalar))
                v_fma_mult[hi] = v_mult

        self.add("flow", ("pause",))

        # 8 DESKS
        NUM_DESKS = 8

        desks = []
        for d in range(NUM_DESKS):
            desk = {
                'idx': self.alloc_scratch(f"v_idx_{d}", VLEN),
                'val': self.alloc_scratch(f"v_val_{d}", VLEN),
                'node_val': self.alloc_scratch(f"v_node_{d}", VLEN),
                'addr': self.alloc_scratch(f"v_addr_{d}", VLEN),
                'tmp1': self.alloc_scratch(f"v_tmp1_{d}", VLEN),
                'tmp2': self.alloc_scratch(f"v_tmp2_{d}", VLEN),
            }
            desks.append(desk)

        addr_tmp = [self.alloc_scratch(f"addr_tmp_{i}") for i in range(16)]
        offset_regs = [self.alloc_scratch(f"off_{d}") for d in range(NUM_DESKS)]

        batch_offset = self.alloc_scratch("batch_offset")
        iter_counter = self.alloc_scratch("iter_counter")

        offset_consts = [self.scratch_const(d * VLEN) for d in range(NUM_DESKS)]

        batch_size_const = self.scratch_const(batch_size)
        desk_stride_const = self.scratch_const(NUM_DESKS * VLEN)

        # For broadcast optimization
        root_node_val = self.alloc_scratch("root_node_val")
        v_root_node = self.alloc_scratch("v_root_node", VLEN)

        # Loop 1: 4 iterations (rounds 0-1)
        loop1_iters_const = self.scratch_const(4)
        # Loop 2: 28 iterations (rounds 2-15, with round fusion = 14 rounds / 2 = 7 × 4 = 28)
        loop2_iters_const = self.scratch_const(28)

        print(f"Scratch usage before main loop: {self.scratch_ptr} / {SCRATCH_SIZE}")

        # ============================================================
        # PROLOGUE: Load tree[0] and broadcast
        # ============================================================
        self.add("load", ("load", root_node_val, self.scratch["forest_values_p"]))
        self.add("valu", ("vbroadcast", v_root_node, root_node_val))

        self.add("load", ("const", batch_offset, 0))
        self.add("load", ("const", iter_counter, 0))

        # ============================================================
        # HELPER FUNCTIONS
        # ============================================================
        def emit_hash_stage(desk_idx, stage):
            d = desks[desk_idx]
            if stage in v_fma_mult:
                return [("multiply_add", d['val'], d['val'], v_fma_mult[stage], v_hash_consts[stage])]
            else:
                if stage == 1:
                    return [("^", d['tmp1'], d['val'], v_hash_consts[stage]), (">>", d['tmp2'], d['val'], v_hash_shifts[stage])]
                elif stage == 3:
                    return [("+", d['tmp1'], d['val'], v_hash_consts[stage]), ("<<", d['tmp2'], d['val'], v_hash_shifts[stage])]
                elif stage == 5:
                    return [("^", d['tmp1'], d['val'], v_hash_consts[stage]), (">>", d['tmp2'], d['val'], v_hash_shifts[stage])]
            return []

        def emit_hash_combine(desk_idx, stage):
            d = desks[desk_idx]
            if stage in [1, 3, 5]:
                return [("^", d['val'], d['tmp1'], d['tmp2'])]
            return []

        def emit_xor_node(desk_idx):
            d = desks[desk_idx]
            return [("^", d['val'], d['val'], d['node_val'])]

        def emit_branch_ops(desk_idx):
            d = desks[desk_idx]
            return [("&", d['tmp1'], d['val'], v_one), ("multiply_add", d['idx'], d['idx'], v_two, v_one)]

        def emit_branch_add(desk_idx):
            d = desks[desk_idx]
            return [("+", d['idx'], d['idx'], d['tmp1'])]

        def emit_bounds_check(desk_idx):
            d = desks[desk_idx]
            return [("<", d['tmp1'], d['idx'], v_n_nodes)]

        def emit_bounds_apply(desk_idx):
            d = desks[desk_idx]
            return [("*", d['idx'], d['idx'], d['tmp1'])]

        # ============================================================
        # LOOP 1: Iterations 1-4 (Rounds 0-1, BROADCAST for R0)
        # ============================================================
        loop1_start = len(self.instrs)

        # PHASE 1: Calculate offsets
        self.instrs.append({
            "alu": [
                ("+", offset_regs[0], batch_offset, offset_consts[0]),
                ("+", offset_regs[1], batch_offset, offset_consts[1]),
                ("+", offset_regs[2], batch_offset, offset_consts[2]),
                ("+", offset_regs[3], batch_offset, offset_consts[3]),
                ("+", offset_regs[4], batch_offset, offset_consts[4]),
                ("+", offset_regs[5], batch_offset, offset_consts[5]),
                ("+", offset_regs[6], batch_offset, offset_consts[6]),
                ("+", offset_regs[7], batch_offset, offset_consts[7]),
            ],
        })

        # PHASE 2: Compute addresses
        self.instrs.append({
            "alu": [
                ("+", addr_tmp[0], self.scratch["inp_indices_p"], offset_regs[0]),
                ("+", addr_tmp[1], self.scratch["inp_values_p"], offset_regs[0]),
                ("+", addr_tmp[2], self.scratch["inp_indices_p"], offset_regs[1]),
                ("+", addr_tmp[3], self.scratch["inp_values_p"], offset_regs[1]),
                ("+", addr_tmp[4], self.scratch["inp_indices_p"], offset_regs[2]),
                ("+", addr_tmp[5], self.scratch["inp_values_p"], offset_regs[2]),
                ("+", addr_tmp[6], self.scratch["inp_indices_p"], offset_regs[3]),
                ("+", addr_tmp[7], self.scratch["inp_values_p"], offset_regs[3]),
            ],
        })
        self.instrs.append({
            "alu": [
                ("+", addr_tmp[8], self.scratch["inp_indices_p"], offset_regs[4]),
                ("+", addr_tmp[9], self.scratch["inp_values_p"], offset_regs[4]),
                ("+", addr_tmp[10], self.scratch["inp_indices_p"], offset_regs[5]),
                ("+", addr_tmp[11], self.scratch["inp_values_p"], offset_regs[5]),
                ("+", addr_tmp[12], self.scratch["inp_indices_p"], offset_regs[6]),
                ("+", addr_tmp[13], self.scratch["inp_values_p"], offset_regs[6]),
                ("+", addr_tmp[14], self.scratch["inp_indices_p"], offset_regs[7]),
                ("+", addr_tmp[15], self.scratch["inp_values_p"], offset_regs[7]),
            ],
        })

        # PHASE 3: Load idx/val for all 8 desks + BROADCAST root to all node_val
        # Interleave loads with broadcast copies (8 vloads + 8 copies)
        self.instrs.append({
            "load": [("vload", desks[0]['idx'], addr_tmp[0]), ("vload", desks[0]['val'], addr_tmp[1])],
            "valu": [("+", desks[0]['node_val'], v_root_node, v_zero), ("+", desks[1]['node_val'], v_root_node, v_zero),
                     ("+", desks[2]['node_val'], v_root_node, v_zero), ("+", desks[3]['node_val'], v_root_node, v_zero),
                     ("+", desks[4]['node_val'], v_root_node, v_zero), ("+", desks[5]['node_val'], v_root_node, v_zero)],
        })
        self.instrs.append({
            "load": [("vload", desks[1]['idx'], addr_tmp[2]), ("vload", desks[1]['val'], addr_tmp[3])],
            "valu": [("+", desks[6]['node_val'], v_root_node, v_zero), ("+", desks[7]['node_val'], v_root_node, v_zero)],
        })
        self.instrs.append({"load": [("vload", desks[2]['idx'], addr_tmp[4]), ("vload", desks[2]['val'], addr_tmp[5])]})
        self.instrs.append({"load": [("vload", desks[3]['idx'], addr_tmp[6]), ("vload", desks[3]['val'], addr_tmp[7])]})
        self.instrs.append({"load": [("vload", desks[4]['idx'], addr_tmp[8]), ("vload", desks[4]['val'], addr_tmp[9])]})
        self.instrs.append({"load": [("vload", desks[5]['idx'], addr_tmp[10]), ("vload", desks[5]['val'], addr_tmp[11])]})
        self.instrs.append({"load": [("vload", desks[6]['idx'], addr_tmp[12]), ("vload", desks[6]['val'], addr_tmp[13])]})
        self.instrs.append({"load": [("vload", desks[7]['idx'], addr_tmp[14]), ("vload", desks[7]['val'], addr_tmp[15])]})

        # PHASE 4: ROUND 0 - Hash only (node_val already set via broadcast)
        # XOR all desks with node_val
        self.instrs.append({"valu": emit_xor_node(0) + emit_xor_node(1) + emit_xor_node(2) + emit_xor_node(3) + emit_xor_node(4) + emit_xor_node(5)})
        self.instrs.append({"valu": emit_xor_node(6) + emit_xor_node(7)})

        # Hash all 8 desks (no interleave with gather needed - just pure compute)
        for stage in range(6):
            ops = []
            for d in range(8):
                ops.extend(emit_hash_stage(d, stage))
            while ops:
                self.instrs.append({"valu": ops[:6]})
                ops = ops[6:]
            if stage in [1, 3, 5]:
                combines = []
                for d in range(8):
                    combines.extend(emit_hash_combine(d, stage))
                while combines:
                    self.instrs.append({"valu": combines[:6]})
                    combines = combines[6:]

        # Branch for all 8 desks
        branch_ops = []
        for d in range(8):
            branch_ops.extend(emit_branch_ops(d))
        while branch_ops:
            self.instrs.append({"valu": branch_ops[:6]})
            branch_ops = branch_ops[6:]

        add_ops = []
        for d in range(8):
            add_ops.extend(emit_branch_add(d))
        while add_ops:
            self.instrs.append({"valu": add_ops[:6]})
            add_ops = add_ops[6:]

        check_ops = []
        for d in range(8):
            check_ops.extend(emit_bounds_check(d))
        while check_ops:
            self.instrs.append({"valu": check_ops[:6]})
            check_ops = check_ops[6:]

        apply_ops = []
        for d in range(8):
            apply_ops.extend(emit_bounds_apply(d))
        while apply_ops:
            self.instrs.append({"valu": apply_ops[:6]})
            apply_ops = apply_ops[6:]

        # PHASE 5: Prepare gather addresses for ROUND 1
        self.instrs.append({
            "valu": [
                ("vbroadcast", desks[0]['addr'], self.scratch["forest_values_p"]),
                ("vbroadcast", desks[1]['addr'], self.scratch["forest_values_p"]),
                ("vbroadcast", desks[2]['addr'], self.scratch["forest_values_p"]),
                ("vbroadcast", desks[3]['addr'], self.scratch["forest_values_p"]),
                ("vbroadcast", desks[4]['addr'], self.scratch["forest_values_p"]),
                ("vbroadcast", desks[5]['addr'], self.scratch["forest_values_p"]),
            ],
        })
        self.instrs.append({
            "valu": [
                ("vbroadcast", desks[6]['addr'], self.scratch["forest_values_p"]),
                ("vbroadcast", desks[7]['addr'], self.scratch["forest_values_p"]),
            ],
        })
        self.instrs.append({
            "valu": [
                ("+", desks[0]['addr'], desks[0]['addr'], desks[0]['idx']),
                ("+", desks[1]['addr'], desks[1]['addr'], desks[1]['idx']),
                ("+", desks[2]['addr'], desks[2]['addr'], desks[2]['idx']),
                ("+", desks[3]['addr'], desks[3]['addr'], desks[3]['idx']),
                ("+", desks[4]['addr'], desks[4]['addr'], desks[4]['idx']),
                ("+", desks[5]['addr'], desks[5]['addr'], desks[5]['idx']),
            ],
        })
        self.instrs.append({
            "valu": [
                ("+", desks[6]['addr'], desks[6]['addr'], desks[6]['idx']),
                ("+", desks[7]['addr'], desks[7]['addr'], desks[7]['idx']),
            ],
        })

        # PHASE 6: ROUND 1 - Interleaved gather + hash (H38 style)
        # Desk 0 gather
        for lane in range(0, VLEN, 2):
            self.instrs.append({"load": [("load", desks[0]['node_val'] + lane, desks[0]['addr'] + lane),
                                          ("load", desks[0]['node_val'] + lane + 1, desks[0]['addr'] + lane + 1)]})

        # Desk 1 gather + desk 0 hash
        self.instrs.append({"load": [("load", desks[1]['node_val'], desks[1]['addr']), ("load", desks[1]['node_val']+1, desks[1]['addr']+1)], "valu": emit_xor_node(0)})
        self.instrs.append({"load": [("load", desks[1]['node_val']+2, desks[1]['addr']+2), ("load", desks[1]['node_val']+3, desks[1]['addr']+3)], "valu": emit_hash_stage(0, 0)})
        self.instrs.append({"load": [("load", desks[1]['node_val']+4, desks[1]['addr']+4), ("load", desks[1]['node_val']+5, desks[1]['addr']+5)], "valu": emit_hash_stage(0, 1)})
        self.instrs.append({"load": [("load", desks[1]['node_val']+6, desks[1]['addr']+6), ("load", desks[1]['node_val']+7, desks[1]['addr']+7)], "valu": emit_hash_combine(0, 1)})

        # Desk 2 gather + desk 0-1 hash
        self.instrs.append({"load": [("load", desks[2]['node_val'], desks[2]['addr']), ("load", desks[2]['node_val']+1, desks[2]['addr']+1)], "valu": emit_hash_stage(0, 2) + emit_xor_node(1)})
        self.instrs.append({"load": [("load", desks[2]['node_val']+2, desks[2]['addr']+2), ("load", desks[2]['node_val']+3, desks[2]['addr']+3)], "valu": emit_hash_stage(0, 3) + emit_hash_stage(1, 0)})
        self.instrs.append({"load": [("load", desks[2]['node_val']+4, desks[2]['addr']+4), ("load", desks[2]['node_val']+5, desks[2]['addr']+5)], "valu": emit_hash_combine(0, 3) + emit_hash_stage(1, 1)})
        self.instrs.append({"load": [("load", desks[2]['node_val']+6, desks[2]['addr']+6), ("load", desks[2]['node_val']+7, desks[2]['addr']+7)], "valu": emit_hash_stage(0, 4) + emit_hash_combine(1, 1)})

        # Desk 3 gather + desk 0-2 hash
        self.instrs.append({"load": [("load", desks[3]['node_val'], desks[3]['addr']), ("load", desks[3]['node_val']+1, desks[3]['addr']+1)], "valu": emit_hash_stage(0, 5) + emit_hash_stage(1, 2) + emit_xor_node(2)})
        self.instrs.append({"load": [("load", desks[3]['node_val']+2, desks[3]['addr']+2), ("load", desks[3]['node_val']+3, desks[3]['addr']+3)], "valu": emit_hash_combine(0, 5) + emit_hash_stage(1, 3) + emit_hash_stage(2, 0)})
        self.instrs.append({"load": [("load", desks[3]['node_val']+4, desks[3]['addr']+4), ("load", desks[3]['node_val']+5, desks[3]['addr']+5)], "valu": emit_branch_ops(0) + emit_hash_combine(1, 3) + emit_hash_stage(2, 1)})
        self.instrs.append({"load": [("load", desks[3]['node_val']+6, desks[3]['addr']+6), ("load", desks[3]['node_val']+7, desks[3]['addr']+7)], "valu": emit_branch_add(0) + emit_hash_stage(1, 4) + emit_hash_combine(2, 1)})

        # Desk 4 gather + desk 0-3 ops
        self.instrs.append({"load": [("load", desks[4]['node_val'], desks[4]['addr']), ("load", desks[4]['node_val']+1, desks[4]['addr']+1)], "valu": emit_bounds_check(0) + emit_hash_stage(1, 5) + emit_hash_stage(2, 2) + emit_xor_node(3)})
        self.instrs.append({"load": [("load", desks[4]['node_val']+2, desks[4]['addr']+2), ("load", desks[4]['node_val']+3, desks[4]['addr']+3)], "valu": emit_bounds_apply(0) + emit_hash_combine(1, 5) + emit_hash_stage(2, 3) + emit_hash_stage(3, 0)})
        self.instrs.append({"load": [("load", desks[4]['node_val']+4, desks[4]['addr']+4), ("load", desks[4]['node_val']+5, desks[4]['addr']+5)], "valu": emit_branch_ops(1) + emit_hash_combine(2, 3) + emit_hash_stage(3, 1)})
        self.instrs.append({"load": [("load", desks[4]['node_val']+6, desks[4]['addr']+6), ("load", desks[4]['node_val']+7, desks[4]['addr']+7)], "valu": emit_branch_add(1) + emit_hash_stage(2, 4) + emit_hash_combine(3, 1)})

        # Desk 5 gather + desk 1-4 ops
        self.instrs.append({"load": [("load", desks[5]['node_val'], desks[5]['addr']), ("load", desks[5]['node_val']+1, desks[5]['addr']+1)], "valu": emit_bounds_check(1) + emit_hash_stage(2, 5) + emit_hash_stage(3, 2) + emit_xor_node(4)})
        self.instrs.append({"load": [("load", desks[5]['node_val']+2, desks[5]['addr']+2), ("load", desks[5]['node_val']+3, desks[5]['addr']+3)], "valu": emit_bounds_apply(1) + emit_hash_combine(2, 5) + emit_hash_stage(3, 3) + emit_hash_stage(4, 0)})
        self.instrs.append({"load": [("load", desks[5]['node_val']+4, desks[5]['addr']+4), ("load", desks[5]['node_val']+5, desks[5]['addr']+5)], "valu": emit_branch_ops(2) + emit_hash_combine(3, 3) + emit_hash_stage(4, 1)})
        self.instrs.append({"load": [("load", desks[5]['node_val']+6, desks[5]['addr']+6), ("load", desks[5]['node_val']+7, desks[5]['addr']+7)], "valu": emit_branch_add(2) + emit_hash_stage(3, 4) + emit_hash_combine(4, 1)})

        # Desk 6 gather + desk 2-5 ops
        self.instrs.append({"load": [("load", desks[6]['node_val'], desks[6]['addr']), ("load", desks[6]['node_val']+1, desks[6]['addr']+1)], "valu": emit_bounds_check(2) + emit_hash_stage(3, 5) + emit_hash_stage(4, 2) + emit_xor_node(5)})
        self.instrs.append({"load": [("load", desks[6]['node_val']+2, desks[6]['addr']+2), ("load", desks[6]['node_val']+3, desks[6]['addr']+3)], "valu": emit_bounds_apply(2) + emit_hash_combine(3, 5) + emit_hash_stage(4, 3) + emit_hash_stage(5, 0)})
        self.instrs.append({"load": [("load", desks[6]['node_val']+4, desks[6]['addr']+4), ("load", desks[6]['node_val']+5, desks[6]['addr']+5)], "valu": emit_branch_ops(3) + emit_hash_combine(4, 3) + emit_hash_stage(5, 1)})
        self.instrs.append({"load": [("load", desks[6]['node_val']+6, desks[6]['addr']+6), ("load", desks[6]['node_val']+7, desks[6]['addr']+7)], "valu": emit_branch_add(3) + emit_hash_stage(4, 4) + emit_hash_combine(5, 1)})

        # Desk 7 gather + desk 3-6 ops
        self.instrs.append({"load": [("load", desks[7]['node_val'], desks[7]['addr']), ("load", desks[7]['node_val']+1, desks[7]['addr']+1)], "valu": emit_bounds_check(3) + emit_hash_stage(4, 5) + emit_hash_stage(5, 2) + emit_xor_node(6)})
        self.instrs.append({"load": [("load", desks[7]['node_val']+2, desks[7]['addr']+2), ("load", desks[7]['node_val']+3, desks[7]['addr']+3)], "valu": emit_bounds_apply(3) + emit_hash_combine(4, 5) + emit_hash_stage(5, 3) + emit_hash_stage(6, 0)})
        self.instrs.append({"load": [("load", desks[7]['node_val']+4, desks[7]['addr']+4), ("load", desks[7]['node_val']+5, desks[7]['addr']+5)], "valu": emit_branch_ops(4) + emit_hash_combine(5, 3) + emit_hash_stage(6, 1)})
        self.instrs.append({"load": [("load", desks[7]['node_val']+6, desks[7]['addr']+6), ("load", desks[7]['node_val']+7, desks[7]['addr']+7)], "valu": emit_branch_add(4) + emit_hash_stage(5, 4) + emit_hash_combine(6, 1)})

        # Finish desks 4-7
        self.instrs.append({"valu": emit_bounds_check(4) + emit_hash_stage(5, 5) + emit_hash_stage(6, 2) + emit_xor_node(7)})
        self.instrs.append({"valu": emit_bounds_apply(4) + emit_hash_combine(5, 5) + emit_hash_stage(6, 3) + emit_hash_stage(7, 0)})
        self.instrs.append({"valu": emit_branch_ops(5) + emit_hash_combine(6, 3) + emit_hash_stage(7, 1)})
        self.instrs.append({"valu": emit_branch_add(5) + emit_hash_stage(6, 4) + emit_hash_combine(7, 1)})
        self.instrs.append({"valu": emit_bounds_check(5) + emit_hash_stage(6, 5) + emit_hash_stage(7, 2)})
        self.instrs.append({"valu": emit_bounds_apply(5) + emit_hash_combine(6, 5) + emit_hash_stage(7, 3)})
        self.instrs.append({"valu": emit_branch_ops(6) + emit_hash_combine(7, 3)})
        self.instrs.append({"valu": emit_branch_add(6) + emit_hash_stage(7, 4)})
        self.instrs.append({"valu": emit_bounds_check(6) + emit_hash_stage(7, 5)})
        self.instrs.append({"valu": emit_bounds_apply(6) + emit_hash_combine(7, 5)})
        self.instrs.append({"valu": emit_branch_ops(7)})
        self.instrs.append({"valu": emit_branch_add(7)})
        self.instrs.append({"valu": emit_bounds_check(7)})
        self.instrs.append({"valu": emit_bounds_apply(7)})

        # PHASE 7: Store all desks
        self.instrs.append({
            "alu": [
                ("+", addr_tmp[0], self.scratch["inp_indices_p"], offset_regs[0]),
                ("+", addr_tmp[1], self.scratch["inp_values_p"], offset_regs[0]),
                ("+", addr_tmp[2], self.scratch["inp_indices_p"], offset_regs[1]),
                ("+", addr_tmp[3], self.scratch["inp_values_p"], offset_regs[1]),
                ("+", addr_tmp[4], self.scratch["inp_indices_p"], offset_regs[2]),
                ("+", addr_tmp[5], self.scratch["inp_values_p"], offset_regs[2]),
                ("+", addr_tmp[6], self.scratch["inp_indices_p"], offset_regs[3]),
                ("+", addr_tmp[7], self.scratch["inp_values_p"], offset_regs[3]),
            ],
        })
        self.instrs.append({
            "alu": [
                ("+", addr_tmp[8], self.scratch["inp_indices_p"], offset_regs[4]),
                ("+", addr_tmp[9], self.scratch["inp_values_p"], offset_regs[4]),
                ("+", addr_tmp[10], self.scratch["inp_indices_p"], offset_regs[5]),
                ("+", addr_tmp[11], self.scratch["inp_values_p"], offset_regs[5]),
                ("+", addr_tmp[12], self.scratch["inp_indices_p"], offset_regs[6]),
                ("+", addr_tmp[13], self.scratch["inp_values_p"], offset_regs[6]),
                ("+", addr_tmp[14], self.scratch["inp_indices_p"], offset_regs[7]),
                ("+", addr_tmp[15], self.scratch["inp_values_p"], offset_regs[7]),
            ],
        })

        for d in range(NUM_DESKS):
            self.instrs.append({"store": [("vstore", addr_tmp[d*2], desks[d]['idx']), ("vstore", addr_tmp[d*2+1], desks[d]['val'])]})

        # Loop 1 control
        self.instrs.append({"alu": [("+", batch_offset, batch_offset, desk_stride_const), ("+", iter_counter, iter_counter, one_const)]})
        self.instrs.append({"alu": [("<", tmp_scalar, iter_counter, loop1_iters_const)]})
        self.add("flow", ("cond_jump", tmp_scalar, loop1_start))

        # ============================================================
        # TRANSITION: Reset for Loop 2
        # ============================================================
        self.add("load", ("const", batch_offset, 0))
        self.add("load", ("const", iter_counter, 0))

        # ============================================================
        # LOOP 2: Iterations 5-32 (Rounds 2-15) - EXACT H38 CODE
        # ============================================================
        loop2_start = len(self.instrs)

        # PHASE 1: Calculate offsets
        self.instrs.append({
            "alu": [
                ("+", offset_regs[0], batch_offset, offset_consts[0]),
                ("+", offset_regs[1], batch_offset, offset_consts[1]),
                ("+", offset_regs[2], batch_offset, offset_consts[2]),
                ("+", offset_regs[3], batch_offset, offset_consts[3]),
                ("+", offset_regs[4], batch_offset, offset_consts[4]),
                ("+", offset_regs[5], batch_offset, offset_consts[5]),
                ("+", offset_regs[6], batch_offset, offset_consts[6]),
                ("+", offset_regs[7], batch_offset, offset_consts[7]),
            ],
        })

        # PHASE 2: Compute addresses
        self.instrs.append({
            "alu": [
                ("+", addr_tmp[0], self.scratch["inp_indices_p"], offset_regs[0]),
                ("+", addr_tmp[1], self.scratch["inp_values_p"], offset_regs[0]),
                ("+", addr_tmp[2], self.scratch["inp_indices_p"], offset_regs[1]),
                ("+", addr_tmp[3], self.scratch["inp_values_p"], offset_regs[1]),
                ("+", addr_tmp[4], self.scratch["inp_indices_p"], offset_regs[2]),
                ("+", addr_tmp[5], self.scratch["inp_values_p"], offset_regs[2]),
                ("+", addr_tmp[6], self.scratch["inp_indices_p"], offset_regs[3]),
                ("+", addr_tmp[7], self.scratch["inp_values_p"], offset_regs[3]),
            ],
        })
        self.instrs.append({
            "alu": [
                ("+", addr_tmp[8], self.scratch["inp_indices_p"], offset_regs[4]),
                ("+", addr_tmp[9], self.scratch["inp_values_p"], offset_regs[4]),
                ("+", addr_tmp[10], self.scratch["inp_indices_p"], offset_regs[5]),
                ("+", addr_tmp[11], self.scratch["inp_values_p"], offset_regs[5]),
                ("+", addr_tmp[12], self.scratch["inp_indices_p"], offset_regs[6]),
                ("+", addr_tmp[13], self.scratch["inp_values_p"], offset_regs[6]),
                ("+", addr_tmp[14], self.scratch["inp_indices_p"], offset_regs[7]),
                ("+", addr_tmp[15], self.scratch["inp_values_p"], offset_regs[7]),
            ],
        })

        # PHASE 3: Load idx/val
        for d in range(0, NUM_DESKS, 2):
            self.instrs.append({"load": [("vload", desks[d]['idx'], addr_tmp[d*2]), ("vload", desks[d]['val'], addr_tmp[d*2+1])]})
            self.instrs.append({"load": [("vload", desks[d+1]['idx'], addr_tmp[(d+1)*2]), ("vload", desks[d+1]['val'], addr_tmp[(d+1)*2+1])]})

        # PHASE 4: Prepare gather addresses
        self.instrs.append({
            "valu": [
                ("vbroadcast", desks[0]['addr'], self.scratch["forest_values_p"]),
                ("vbroadcast", desks[1]['addr'], self.scratch["forest_values_p"]),
                ("vbroadcast", desks[2]['addr'], self.scratch["forest_values_p"]),
                ("vbroadcast", desks[3]['addr'], self.scratch["forest_values_p"]),
                ("vbroadcast", desks[4]['addr'], self.scratch["forest_values_p"]),
                ("vbroadcast", desks[5]['addr'], self.scratch["forest_values_p"]),
            ],
        })
        self.instrs.append({"valu": [("vbroadcast", desks[6]['addr'], self.scratch["forest_values_p"]), ("vbroadcast", desks[7]['addr'], self.scratch["forest_values_p"])]})
        self.instrs.append({
            "valu": [
                ("+", desks[0]['addr'], desks[0]['addr'], desks[0]['idx']),
                ("+", desks[1]['addr'], desks[1]['addr'], desks[1]['idx']),
                ("+", desks[2]['addr'], desks[2]['addr'], desks[2]['idx']),
                ("+", desks[3]['addr'], desks[3]['addr'], desks[3]['idx']),
                ("+", desks[4]['addr'], desks[4]['addr'], desks[4]['idx']),
                ("+", desks[5]['addr'], desks[5]['addr'], desks[5]['idx']),
            ],
        })
        self.instrs.append({"valu": [("+", desks[6]['addr'], desks[6]['addr'], desks[6]['idx']), ("+", desks[7]['addr'], desks[7]['addr'], desks[7]['idx'])]})

        # ============================================================
        # ROUND N: INTERLEAVED GATHER + HASH (H38 style)
        # ============================================================
        # Desk 0 gather
        for lane in range(0, VLEN, 2):
            self.instrs.append({"load": [("load", desks[0]['node_val'] + lane, desks[0]['addr'] + lane), ("load", desks[0]['node_val'] + lane + 1, desks[0]['addr'] + lane + 1)]})

        # Desk 1-7 gather interleaved with hash (exact H38 pattern)
        self.instrs.append({"load": [("load", desks[1]['node_val'], desks[1]['addr']), ("load", desks[1]['node_val']+1, desks[1]['addr']+1)], "valu": emit_xor_node(0)})
        self.instrs.append({"load": [("load", desks[1]['node_val']+2, desks[1]['addr']+2), ("load", desks[1]['node_val']+3, desks[1]['addr']+3)], "valu": emit_hash_stage(0, 0)})
        self.instrs.append({"load": [("load", desks[1]['node_val']+4, desks[1]['addr']+4), ("load", desks[1]['node_val']+5, desks[1]['addr']+5)], "valu": emit_hash_stage(0, 1)})
        self.instrs.append({"load": [("load", desks[1]['node_val']+6, desks[1]['addr']+6), ("load", desks[1]['node_val']+7, desks[1]['addr']+7)], "valu": emit_hash_combine(0, 1)})

        self.instrs.append({"load": [("load", desks[2]['node_val'], desks[2]['addr']), ("load", desks[2]['node_val']+1, desks[2]['addr']+1)], "valu": emit_hash_stage(0, 2) + emit_xor_node(1)})
        self.instrs.append({"load": [("load", desks[2]['node_val']+2, desks[2]['addr']+2), ("load", desks[2]['node_val']+3, desks[2]['addr']+3)], "valu": emit_hash_stage(0, 3) + emit_hash_stage(1, 0)})
        self.instrs.append({"load": [("load", desks[2]['node_val']+4, desks[2]['addr']+4), ("load", desks[2]['node_val']+5, desks[2]['addr']+5)], "valu": emit_hash_combine(0, 3) + emit_hash_stage(1, 1)})
        self.instrs.append({"load": [("load", desks[2]['node_val']+6, desks[2]['addr']+6), ("load", desks[2]['node_val']+7, desks[2]['addr']+7)], "valu": emit_hash_stage(0, 4) + emit_hash_combine(1, 1)})

        self.instrs.append({"load": [("load", desks[3]['node_val'], desks[3]['addr']), ("load", desks[3]['node_val']+1, desks[3]['addr']+1)], "valu": emit_hash_stage(0, 5) + emit_hash_stage(1, 2) + emit_xor_node(2)})
        self.instrs.append({"load": [("load", desks[3]['node_val']+2, desks[3]['addr']+2), ("load", desks[3]['node_val']+3, desks[3]['addr']+3)], "valu": emit_hash_combine(0, 5) + emit_hash_stage(1, 3) + emit_hash_stage(2, 0)})
        self.instrs.append({"load": [("load", desks[3]['node_val']+4, desks[3]['addr']+4), ("load", desks[3]['node_val']+5, desks[3]['addr']+5)], "valu": emit_branch_ops(0) + emit_hash_combine(1, 3) + emit_hash_stage(2, 1)})
        self.instrs.append({"load": [("load", desks[3]['node_val']+6, desks[3]['addr']+6), ("load", desks[3]['node_val']+7, desks[3]['addr']+7)], "valu": emit_branch_add(0) + emit_hash_stage(1, 4) + emit_hash_combine(2, 1)})

        self.instrs.append({"load": [("load", desks[4]['node_val'], desks[4]['addr']), ("load", desks[4]['node_val']+1, desks[4]['addr']+1)], "valu": emit_bounds_check(0) + emit_hash_stage(1, 5) + emit_hash_stage(2, 2) + emit_xor_node(3)})
        self.instrs.append({"load": [("load", desks[4]['node_val']+2, desks[4]['addr']+2), ("load", desks[4]['node_val']+3, desks[4]['addr']+3)], "valu": emit_bounds_apply(0) + emit_hash_combine(1, 5) + emit_hash_stage(2, 3) + emit_hash_stage(3, 0)})
        self.instrs.append({"load": [("load", desks[4]['node_val']+4, desks[4]['addr']+4), ("load", desks[4]['node_val']+5, desks[4]['addr']+5)], "valu": emit_branch_ops(1) + emit_hash_combine(2, 3) + emit_hash_stage(3, 1)})
        self.instrs.append({"load": [("load", desks[4]['node_val']+6, desks[4]['addr']+6), ("load", desks[4]['node_val']+7, desks[4]['addr']+7)], "valu": emit_branch_add(1) + emit_hash_stage(2, 4) + emit_hash_combine(3, 1)})

        self.instrs.append({"load": [("load", desks[5]['node_val'], desks[5]['addr']), ("load", desks[5]['node_val']+1, desks[5]['addr']+1)], "valu": emit_bounds_check(1) + emit_hash_stage(2, 5) + emit_hash_stage(3, 2) + emit_xor_node(4)})
        self.instrs.append({"load": [("load", desks[5]['node_val']+2, desks[5]['addr']+2), ("load", desks[5]['node_val']+3, desks[5]['addr']+3)], "valu": emit_bounds_apply(1) + emit_hash_combine(2, 5) + emit_hash_stage(3, 3) + emit_hash_stage(4, 0)})
        self.instrs.append({"load": [("load", desks[5]['node_val']+4, desks[5]['addr']+4), ("load", desks[5]['node_val']+5, desks[5]['addr']+5)], "valu": emit_branch_ops(2) + emit_hash_combine(3, 3) + emit_hash_stage(4, 1)})
        self.instrs.append({"load": [("load", desks[5]['node_val']+6, desks[5]['addr']+6), ("load", desks[5]['node_val']+7, desks[5]['addr']+7)], "valu": emit_branch_add(2) + emit_hash_stage(3, 4) + emit_hash_combine(4, 1)})

        self.instrs.append({"load": [("load", desks[6]['node_val'], desks[6]['addr']), ("load", desks[6]['node_val']+1, desks[6]['addr']+1)], "valu": emit_bounds_check(2) + emit_hash_stage(3, 5) + emit_hash_stage(4, 2) + emit_xor_node(5)})
        self.instrs.append({"load": [("load", desks[6]['node_val']+2, desks[6]['addr']+2), ("load", desks[6]['node_val']+3, desks[6]['addr']+3)], "valu": emit_bounds_apply(2) + emit_hash_combine(3, 5) + emit_hash_stage(4, 3) + emit_hash_stage(5, 0)})
        self.instrs.append({"load": [("load", desks[6]['node_val']+4, desks[6]['addr']+4), ("load", desks[6]['node_val']+5, desks[6]['addr']+5)], "valu": emit_branch_ops(3) + emit_hash_combine(4, 3) + emit_hash_stage(5, 1)})
        self.instrs.append({"load": [("load", desks[6]['node_val']+6, desks[6]['addr']+6), ("load", desks[6]['node_val']+7, desks[6]['addr']+7)], "valu": emit_branch_add(3) + emit_hash_stage(4, 4) + emit_hash_combine(5, 1)})

        self.instrs.append({"load": [("load", desks[7]['node_val'], desks[7]['addr']), ("load", desks[7]['node_val']+1, desks[7]['addr']+1)], "valu": emit_bounds_check(3) + emit_hash_stage(4, 5) + emit_hash_stage(5, 2) + emit_xor_node(6)})
        self.instrs.append({"load": [("load", desks[7]['node_val']+2, desks[7]['addr']+2), ("load", desks[7]['node_val']+3, desks[7]['addr']+3)], "valu": emit_bounds_apply(3) + emit_hash_combine(4, 5) + emit_hash_stage(5, 3) + emit_hash_stage(6, 0)})
        self.instrs.append({"load": [("load", desks[7]['node_val']+4, desks[7]['addr']+4), ("load", desks[7]['node_val']+5, desks[7]['addr']+5)], "valu": emit_branch_ops(4) + emit_hash_combine(5, 3) + emit_hash_stage(6, 1)})
        self.instrs.append({"load": [("load", desks[7]['node_val']+6, desks[7]['addr']+6), ("load", desks[7]['node_val']+7, desks[7]['addr']+7)], "valu": emit_branch_add(4) + emit_hash_stage(5, 4) + emit_hash_combine(6, 1)})

        # Finish desks 4-7
        self.instrs.append({"valu": emit_bounds_check(4) + emit_hash_stage(5, 5) + emit_hash_stage(6, 2) + emit_xor_node(7)})
        self.instrs.append({"valu": emit_bounds_apply(4) + emit_hash_combine(5, 5) + emit_hash_stage(6, 3) + emit_hash_stage(7, 0)})
        self.instrs.append({"valu": emit_branch_ops(5) + emit_hash_combine(6, 3) + emit_hash_stage(7, 1)})
        self.instrs.append({"valu": emit_branch_add(5) + emit_hash_stage(6, 4) + emit_hash_combine(7, 1)})
        self.instrs.append({"valu": emit_bounds_check(5) + emit_hash_stage(6, 5) + emit_hash_stage(7, 2)})
        self.instrs.append({"valu": emit_bounds_apply(5) + emit_hash_combine(6, 5) + emit_hash_stage(7, 3)})
        self.instrs.append({"valu": emit_branch_ops(6) + emit_hash_combine(7, 3)})
        self.instrs.append({"valu": emit_branch_add(6) + emit_hash_stage(7, 4)})
        self.instrs.append({"valu": emit_bounds_check(6) + emit_hash_stage(7, 5)})
        self.instrs.append({"valu": emit_bounds_apply(6) + emit_hash_combine(7, 5)})
        self.instrs.append({"valu": emit_branch_ops(7)})
        self.instrs.append({"valu": emit_branch_add(7)})
        self.instrs.append({"valu": emit_bounds_check(7)})
        self.instrs.append({"valu": emit_bounds_apply(7)})

        # ============================================================
        # ROUND N+1: Same as Round N (no round fusion in this version for simplicity)
        # ============================================================
        # Prepare gather addresses for round N+1
        self.instrs.append({
            "valu": [
                ("vbroadcast", desks[0]['addr'], self.scratch["forest_values_p"]),
                ("vbroadcast", desks[1]['addr'], self.scratch["forest_values_p"]),
                ("vbroadcast", desks[2]['addr'], self.scratch["forest_values_p"]),
                ("vbroadcast", desks[3]['addr'], self.scratch["forest_values_p"]),
                ("vbroadcast", desks[4]['addr'], self.scratch["forest_values_p"]),
                ("vbroadcast", desks[5]['addr'], self.scratch["forest_values_p"]),
            ],
        })
        self.instrs.append({"valu": [("vbroadcast", desks[6]['addr'], self.scratch["forest_values_p"]), ("vbroadcast", desks[7]['addr'], self.scratch["forest_values_p"])]})
        self.instrs.append({
            "valu": [
                ("+", desks[0]['addr'], desks[0]['addr'], desks[0]['idx']),
                ("+", desks[1]['addr'], desks[1]['addr'], desks[1]['idx']),
                ("+", desks[2]['addr'], desks[2]['addr'], desks[2]['idx']),
                ("+", desks[3]['addr'], desks[3]['addr'], desks[3]['idx']),
                ("+", desks[4]['addr'], desks[4]['addr'], desks[4]['idx']),
                ("+", desks[5]['addr'], desks[5]['addr'], desks[5]['idx']),
            ],
        })
        self.instrs.append({"valu": [("+", desks[6]['addr'], desks[6]['addr'], desks[6]['idx']), ("+", desks[7]['addr'], desks[7]['addr'], desks[7]['idx'])]})

        # Round N+1 gather + hash (same pattern)
        for lane in range(0, VLEN, 2):
            self.instrs.append({"load": [("load", desks[0]['node_val'] + lane, desks[0]['addr'] + lane), ("load", desks[0]['node_val'] + lane + 1, desks[0]['addr'] + lane + 1)]})

        self.instrs.append({"load": [("load", desks[1]['node_val'], desks[1]['addr']), ("load", desks[1]['node_val']+1, desks[1]['addr']+1)], "valu": emit_xor_node(0)})
        self.instrs.append({"load": [("load", desks[1]['node_val']+2, desks[1]['addr']+2), ("load", desks[1]['node_val']+3, desks[1]['addr']+3)], "valu": emit_hash_stage(0, 0)})
        self.instrs.append({"load": [("load", desks[1]['node_val']+4, desks[1]['addr']+4), ("load", desks[1]['node_val']+5, desks[1]['addr']+5)], "valu": emit_hash_stage(0, 1)})
        self.instrs.append({"load": [("load", desks[1]['node_val']+6, desks[1]['addr']+6), ("load", desks[1]['node_val']+7, desks[1]['addr']+7)], "valu": emit_hash_combine(0, 1)})

        self.instrs.append({"load": [("load", desks[2]['node_val'], desks[2]['addr']), ("load", desks[2]['node_val']+1, desks[2]['addr']+1)], "valu": emit_hash_stage(0, 2) + emit_xor_node(1)})
        self.instrs.append({"load": [("load", desks[2]['node_val']+2, desks[2]['addr']+2), ("load", desks[2]['node_val']+3, desks[2]['addr']+3)], "valu": emit_hash_stage(0, 3) + emit_hash_stage(1, 0)})
        self.instrs.append({"load": [("load", desks[2]['node_val']+4, desks[2]['addr']+4), ("load", desks[2]['node_val']+5, desks[2]['addr']+5)], "valu": emit_hash_combine(0, 3) + emit_hash_stage(1, 1)})
        self.instrs.append({"load": [("load", desks[2]['node_val']+6, desks[2]['addr']+6), ("load", desks[2]['node_val']+7, desks[2]['addr']+7)], "valu": emit_hash_stage(0, 4) + emit_hash_combine(1, 1)})

        self.instrs.append({"load": [("load", desks[3]['node_val'], desks[3]['addr']), ("load", desks[3]['node_val']+1, desks[3]['addr']+1)], "valu": emit_hash_stage(0, 5) + emit_hash_stage(1, 2) + emit_xor_node(2)})
        self.instrs.append({"load": [("load", desks[3]['node_val']+2, desks[3]['addr']+2), ("load", desks[3]['node_val']+3, desks[3]['addr']+3)], "valu": emit_hash_combine(0, 5) + emit_hash_stage(1, 3) + emit_hash_stage(2, 0)})
        self.instrs.append({"load": [("load", desks[3]['node_val']+4, desks[3]['addr']+4), ("load", desks[3]['node_val']+5, desks[3]['addr']+5)], "valu": emit_branch_ops(0) + emit_hash_combine(1, 3) + emit_hash_stage(2, 1)})
        self.instrs.append({"load": [("load", desks[3]['node_val']+6, desks[3]['addr']+6), ("load", desks[3]['node_val']+7, desks[3]['addr']+7)], "valu": emit_branch_add(0) + emit_hash_stage(1, 4) + emit_hash_combine(2, 1)})

        self.instrs.append({"load": [("load", desks[4]['node_val'], desks[4]['addr']), ("load", desks[4]['node_val']+1, desks[4]['addr']+1)], "valu": emit_bounds_check(0) + emit_hash_stage(1, 5) + emit_hash_stage(2, 2) + emit_xor_node(3)})
        self.instrs.append({"load": [("load", desks[4]['node_val']+2, desks[4]['addr']+2), ("load", desks[4]['node_val']+3, desks[4]['addr']+3)], "valu": emit_bounds_apply(0) + emit_hash_combine(1, 5) + emit_hash_stage(2, 3) + emit_hash_stage(3, 0)})
        self.instrs.append({"load": [("load", desks[4]['node_val']+4, desks[4]['addr']+4), ("load", desks[4]['node_val']+5, desks[4]['addr']+5)], "valu": emit_branch_ops(1) + emit_hash_combine(2, 3) + emit_hash_stage(3, 1)})
        self.instrs.append({"load": [("load", desks[4]['node_val']+6, desks[4]['addr']+6), ("load", desks[4]['node_val']+7, desks[4]['addr']+7)], "valu": emit_branch_add(1) + emit_hash_stage(2, 4) + emit_hash_combine(3, 1)})

        self.instrs.append({"load": [("load", desks[5]['node_val'], desks[5]['addr']), ("load", desks[5]['node_val']+1, desks[5]['addr']+1)], "valu": emit_bounds_check(1) + emit_hash_stage(2, 5) + emit_hash_stage(3, 2) + emit_xor_node(4)})
        self.instrs.append({"load": [("load", desks[5]['node_val']+2, desks[5]['addr']+2), ("load", desks[5]['node_val']+3, desks[5]['addr']+3)], "valu": emit_bounds_apply(1) + emit_hash_combine(2, 5) + emit_hash_stage(3, 3) + emit_hash_stage(4, 0)})
        self.instrs.append({"load": [("load", desks[5]['node_val']+4, desks[5]['addr']+4), ("load", desks[5]['node_val']+5, desks[5]['addr']+5)], "valu": emit_branch_ops(2) + emit_hash_combine(3, 3) + emit_hash_stage(4, 1)})
        self.instrs.append({"load": [("load", desks[5]['node_val']+6, desks[5]['addr']+6), ("load", desks[5]['node_val']+7, desks[5]['addr']+7)], "valu": emit_branch_add(2) + emit_hash_stage(3, 4) + emit_hash_combine(4, 1)})

        self.instrs.append({"load": [("load", desks[6]['node_val'], desks[6]['addr']), ("load", desks[6]['node_val']+1, desks[6]['addr']+1)], "valu": emit_bounds_check(2) + emit_hash_stage(3, 5) + emit_hash_stage(4, 2) + emit_xor_node(5)})
        self.instrs.append({"load": [("load", desks[6]['node_val']+2, desks[6]['addr']+2), ("load", desks[6]['node_val']+3, desks[6]['addr']+3)], "valu": emit_bounds_apply(2) + emit_hash_combine(3, 5) + emit_hash_stage(4, 3) + emit_hash_stage(5, 0)})
        self.instrs.append({"load": [("load", desks[6]['node_val']+4, desks[6]['addr']+4), ("load", desks[6]['node_val']+5, desks[6]['addr']+5)], "valu": emit_branch_ops(3) + emit_hash_combine(4, 3) + emit_hash_stage(5, 1)})
        self.instrs.append({"load": [("load", desks[6]['node_val']+6, desks[6]['addr']+6), ("load", desks[6]['node_val']+7, desks[6]['addr']+7)], "valu": emit_branch_add(3) + emit_hash_stage(4, 4) + emit_hash_combine(5, 1)})

        self.instrs.append({"load": [("load", desks[7]['node_val'], desks[7]['addr']), ("load", desks[7]['node_val']+1, desks[7]['addr']+1)], "valu": emit_bounds_check(3) + emit_hash_stage(4, 5) + emit_hash_stage(5, 2) + emit_xor_node(6)})
        self.instrs.append({"load": [("load", desks[7]['node_val']+2, desks[7]['addr']+2), ("load", desks[7]['node_val']+3, desks[7]['addr']+3)], "valu": emit_bounds_apply(3) + emit_hash_combine(4, 5) + emit_hash_stage(5, 3) + emit_hash_stage(6, 0)})
        self.instrs.append({"load": [("load", desks[7]['node_val']+4, desks[7]['addr']+4), ("load", desks[7]['node_val']+5, desks[7]['addr']+5)], "valu": emit_branch_ops(4) + emit_hash_combine(5, 3) + emit_hash_stage(6, 1)})
        self.instrs.append({"load": [("load", desks[7]['node_val']+6, desks[7]['addr']+6), ("load", desks[7]['node_val']+7, desks[7]['addr']+7)], "valu": emit_branch_add(4) + emit_hash_stage(5, 4) + emit_hash_combine(6, 1)})

        self.instrs.append({"valu": emit_bounds_check(4) + emit_hash_stage(5, 5) + emit_hash_stage(6, 2) + emit_xor_node(7)})
        self.instrs.append({"valu": emit_bounds_apply(4) + emit_hash_combine(5, 5) + emit_hash_stage(6, 3) + emit_hash_stage(7, 0)})
        self.instrs.append({"valu": emit_branch_ops(5) + emit_hash_combine(6, 3) + emit_hash_stage(7, 1)})
        self.instrs.append({"valu": emit_branch_add(5) + emit_hash_stage(6, 4) + emit_hash_combine(7, 1)})
        self.instrs.append({"valu": emit_bounds_check(5) + emit_hash_stage(6, 5) + emit_hash_stage(7, 2)})
        self.instrs.append({"valu": emit_bounds_apply(5) + emit_hash_combine(6, 5) + emit_hash_stage(7, 3)})
        self.instrs.append({"valu": emit_branch_ops(6) + emit_hash_combine(7, 3)})
        self.instrs.append({"valu": emit_branch_add(6) + emit_hash_stage(7, 4)})
        self.instrs.append({"valu": emit_bounds_check(6) + emit_hash_stage(7, 5)})
        self.instrs.append({"valu": emit_bounds_apply(6) + emit_hash_combine(7, 5)})
        self.instrs.append({"valu": emit_branch_ops(7)})
        self.instrs.append({"valu": emit_branch_add(7)})
        self.instrs.append({"valu": emit_bounds_check(7)})
        self.instrs.append({"valu": emit_bounds_apply(7)})

        # Store
        self.instrs.append({
            "alu": [
                ("+", addr_tmp[0], self.scratch["inp_indices_p"], offset_regs[0]),
                ("+", addr_tmp[1], self.scratch["inp_values_p"], offset_regs[0]),
                ("+", addr_tmp[2], self.scratch["inp_indices_p"], offset_regs[1]),
                ("+", addr_tmp[3], self.scratch["inp_values_p"], offset_regs[1]),
                ("+", addr_tmp[4], self.scratch["inp_indices_p"], offset_regs[2]),
                ("+", addr_tmp[5], self.scratch["inp_values_p"], offset_regs[2]),
                ("+", addr_tmp[6], self.scratch["inp_indices_p"], offset_regs[3]),
                ("+", addr_tmp[7], self.scratch["inp_values_p"], offset_regs[3]),
            ],
        })
        self.instrs.append({
            "alu": [
                ("+", addr_tmp[8], self.scratch["inp_indices_p"], offset_regs[4]),
                ("+", addr_tmp[9], self.scratch["inp_values_p"], offset_regs[4]),
                ("+", addr_tmp[10], self.scratch["inp_indices_p"], offset_regs[5]),
                ("+", addr_tmp[11], self.scratch["inp_values_p"], offset_regs[5]),
                ("+", addr_tmp[12], self.scratch["inp_indices_p"], offset_regs[6]),
                ("+", addr_tmp[13], self.scratch["inp_values_p"], offset_regs[6]),
                ("+", addr_tmp[14], self.scratch["inp_indices_p"], offset_regs[7]),
                ("+", addr_tmp[15], self.scratch["inp_values_p"], offset_regs[7]),
            ],
        })

        for d in range(NUM_DESKS):
            self.instrs.append({"store": [("vstore", addr_tmp[d*2], desks[d]['idx']), ("vstore", addr_tmp[d*2+1], desks[d]['val'])]})

        # Loop 2 control
        self.instrs.append({"alu": [("+", batch_offset, batch_offset, desk_stride_const), ("+", iter_counter, iter_counter, one_const)]})
        self.instrs.append({"alu": [("<", tmp_scalar, batch_offset, batch_size_const), ("<", addr_scalar, iter_counter, loop2_iters_const)]})
        self.instrs.append({"flow": [("select", batch_offset, tmp_scalar, batch_offset, zero_const)]})

        self.add("flow", ("cond_jump", addr_scalar, loop2_start))
        self.instrs.append({"flow": [("pause",)]})


BASELINE = 147734


def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
    check: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilderH42c()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if check:
            assert (
                machine.mem[inp_values_p : inp_values_p + len(inp.values)]
                == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
            ), f"Incorrect result on round {i}"

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    print(f"vs H38 (4062): {4062 / machine.cycle:.2%}")
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
        cycles = do_kernel_test(10, 16, 256, trace=args.trace)
