"""
Fresh Start - Iteration 28: True software pipelining

Timeline (steady state after prolog):
- Cycle N: Load nodes for round R+1 | VALU hash for round R
- Need to track which round each desk is processing

Actually, the issue is that within a single instruction, all operations
happen in parallel. So we need to structure the code to emit combined
instructions.

Let's try: while hashing round R, load nodes for round R+1.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from problem import (
    Machine,
    build_mem_image,
    reference_kernel2,
    Tree,
    Input,
    DebugInfo,
    N_CORES,
    VLEN,
    HASH_STAGES,
    SCRATCH_SIZE,
)

class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch_map = {}
        self.next_addr = 0

    def alloc(self, name, size=1):
        addr = self.next_addr
        self.scratch_map[addr] = (name, size)
        self.next_addr += size
        return addr

    def alloc_vec(self, name):
        return self.alloc(name, VLEN)

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_map)

    def emit(self, instr):
        self.instrs.append(instr)

    def build_kernel(self, forest_height, n_nodes, batch_size, rounds):
        DESK = 16

        # Constants
        c_zero = self.alloc("c_zero")
        c_one = self.alloc("c_one")
        c_vlen = self.alloc("c_vlen")
        c_n_nodes = self.alloc("c_n_nodes")
        c_forest_p = self.alloc("c_forest_p")
        c_indices_p = self.alloc("c_indices_p")
        c_values_p = self.alloc("c_values_p")
        c_desk_offset = self.alloc("c_desk_offset")

        s_tmp = self.alloc("s_tmp")
        s_tree0 = self.alloc("s_tree0")
        s_tree1 = self.alloc("s_tree1")
        s_tree2 = self.alloc("s_tree2")
        s_tree_diff = self.alloc("s_tree_diff")

        # Vector hash constants
        vh_consts = []
        for i in range(len(HASH_STAGES)):
            v_c1 = self.alloc_vec(f"vh_c1_{i}")
            v_c3 = self.alloc_vec(f"vh_c3_{i}")
            vh_consts.append((v_c1, v_c3))

        v_zero = self.alloc_vec("v_zero")
        v_one = self.alloc_vec("v_one")
        v_n_nodes = self.alloc_vec("v_n_nodes")
        v_tree0 = self.alloc_vec("v_tree0")
        v_tree1 = self.alloc_vec("v_tree1")
        v_tree_diff = self.alloc_vec("v_tree_diff")

        # Vectors
        v_idx = [self.alloc_vec(f"v_idx_{i}") for i in range(DESK)]
        v_val = [self.alloc_vec(f"v_val_{i}") for i in range(DESK)]
        v_tmp = [self.alloc_vec(f"v_tmp_{i}") for i in range(DESK)]
        v_tmp2 = [self.alloc_vec(f"v_tmp2_{i}") for i in range(DESK)]
        v_node = [self.alloc_vec(f"v_node_{i}") for i in range(DESK)]  # Preloaded nodes

        # Scalar gather - for current round
        s_addr = [[self.alloc(f"s_addr_{d}_{i}") for i in range(VLEN)] for d in range(DESK)]
        s_node = [[self.alloc(f"s_node_{d}_{i}") for i in range(VLEN)] for d in range(DESK)]

        # Pointers
        idx_p = [self.alloc(f"idx_p_{i}") for i in range(DESK)]
        val_p = [self.alloc(f"val_p_{i}") for i in range(DESK)]

        print(f"Scratch usage: {self.next_addr} / {SCRATCH_SIZE}")

        # === INIT ===
        self.emit({"load": [("const", c_zero, 0), ("const", c_one, 1)]})
        self.emit({"load": [("const", c_vlen, VLEN), ("const", c_desk_offset, DESK * VLEN)]})
        self.emit({"load": [("const", s_tmp, n_nodes)]})
        self.emit({"alu": [("+", c_n_nodes, s_tmp, c_zero)]})

        self.emit({"load": [("const", s_tmp, 4)]})
        self.emit({"load": [("load", c_forest_p, s_tmp)]})
        self.emit({"load": [("const", s_tmp, 5)]})
        self.emit({"load": [("load", c_indices_p, s_tmp)]})
        self.emit({"load": [("const", s_tmp, 6)]})
        self.emit({"load": [("load", c_values_p, s_tmp)]})

        # Preload tree[0], tree[1], tree[2]
        self.emit({"load": [("load", s_tree0, c_forest_p)]})
        self.emit({"alu": [("+", s_tmp, c_forest_p, c_one)]})
        self.emit({"load": [("load", s_tree1, s_tmp)]})
        self.emit({"alu": [("+", s_tmp, s_tmp, c_one)]})
        self.emit({"load": [("load", s_tree2, s_tmp)]})

        self.emit({"alu": [("-", s_tree_diff, s_tree2, s_tree1)]})

        self.emit({"valu": [("vbroadcast", v_tree0, s_tree0)]})
        self.emit({"valu": [("vbroadcast", v_tree1, s_tree1)]})
        self.emit({"valu": [("vbroadcast", v_tree_diff, s_tree_diff)]})

        self.emit({"valu": [("vbroadcast", v_zero, c_zero)]})
        self.emit({"valu": [("vbroadcast", v_one, c_one)]})
        self.emit({"valu": [("vbroadcast", v_n_nodes, c_n_nodes)]})

        for i, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            v_c1, v_c3 = vh_consts[i]
            self.emit({"load": [("const", s_tmp, val1)]})
            self.emit({"valu": [("vbroadcast", v_c1, s_tmp)]})
            self.emit({"load": [("const", s_tmp, val3)]})
            self.emit({"valu": [("vbroadcast", v_c3, s_tmp)]})

        # Setup pointers
        self.emit({"alu": [("+", idx_p[0], c_indices_p, c_zero), ("+", val_p[0], c_values_p, c_zero)]})
        for d in range(1, DESK):
            self.emit({"alu": [("+", idx_p[d], idx_p[d-1], c_vlen), ("+", val_p[d], val_p[d-1], c_vlen)]})

        # Process first half with true pipelining
        self._process_true_pipeline(DESK, v_idx, v_val, v_tmp, v_tmp2, v_node, s_addr, s_node, idx_p, val_p,
                           vh_consts, c_forest_p, c_zero, v_zero, v_one, v_n_nodes, v_tree0, v_tree1, v_tree_diff)

        # Update pointers
        for d in range(DESK):
            self.emit({"alu": [("+", idx_p[d], idx_p[d], c_desk_offset), ("+", val_p[d], val_p[d], c_desk_offset)]})

        # Process second half
        self._process_true_pipeline(DESK, v_idx, v_val, v_tmp, v_tmp2, v_node, s_addr, s_node, idx_p, val_p,
                           vh_consts, c_forest_p, c_zero, v_zero, v_one, v_n_nodes, v_tree0, v_tree1, v_tree_diff)

        self.emit({"flow": [("halt",)]})

    def _gather_nodes(self, DESK, v_idx, v_tmp, s_addr, s_node, c_forest_p, c_zero):
        """Gather tree nodes into v_tmp"""
        for d in range(DESK):
            self.emit({"alu": [("+", s_addr[d][i], c_forest_p, v_idx[d] + i) for i in range(VLEN)]})
        for d in range(DESK):
            for i in range(0, VLEN, 2):
                self.emit({"load": [("load", s_node[d][i], s_addr[d][i]), ("load", s_node[d][i+1], s_addr[d][i+1])]})
        for d in range(DESK):
            self.emit({"alu": [("+", v_tmp[d] + i, s_node[d][i], c_zero) for i in range(VLEN)]})

    def _process_true_pipeline(self, DESK, v_idx, v_val, v_tmp, v_tmp2, v_node, s_addr, s_node, idx_p, val_p,
                       vh_consts, c_forest_p, c_zero, v_zero, v_one, v_n_nodes, v_tree0, v_tree1, v_tree_diff):

        # Load idx/val once
        for d in range(0, DESK, 2):
            self.emit({"load": [("vload", v_idx[d], idx_p[d]), ("vload", v_idx[d+1], idx_p[d+1])]})
        for d in range(0, DESK, 2):
            self.emit({"load": [("vload", v_val[d], val_p[d]), ("vload", v_val[d+1], val_p[d+1])]})

        # === ROUND 0: idx=0 → use tree[0] ===
        for start in range(0, DESK, 6):
            end = min(start + 6, DESK)
            self.emit({"valu": [("^", v_val[d], v_val[d], v_tree0) for d in range(start, end)]})

        self._do_hash(DESK, v_val, v_tmp, v_tmp2, vh_consts)

        # idx = 1 + (val & 1)
        for start in range(0, DESK, 6):
            end = min(start + 6, DESK)
            self.emit({"valu": [("&", v_tmp[d], v_val[d], v_one) for d in range(start, end)]})
        for start in range(0, DESK, 6):
            end = min(start + 6, DESK)
            self.emit({"valu": [("+", v_idx[d], v_one, v_tmp[d]) for d in range(start, end)]})

        # === ROUND 1: idx in {1,2} → arithmetic ===
        for start in range(0, DESK, 6):
            end = min(start + 6, DESK)
            self.emit({"valu": [("-", v_tmp[d], v_idx[d], v_one) for d in range(start, end)]})
        for start in range(0, DESK, 6):
            end = min(start + 6, DESK)
            self.emit({"valu": [("*", v_tmp[d], v_tree_diff, v_tmp[d]) for d in range(start, end)]})
        for start in range(0, DESK, 6):
            end = min(start + 6, DESK)
            self.emit({"valu": [("+", v_tmp[d], v_tree1, v_tmp[d]) for d in range(start, end)]})

        for start in range(0, DESK, 6):
            end = min(start + 6, DESK)
            self.emit({"valu": [("^", v_val[d], v_val[d], v_tmp[d]) for d in range(start, end)]})

        self._do_hash(DESK, v_val, v_tmp, v_tmp2, vh_consts)
        self._do_next_index(DESK, v_idx, v_val, v_tmp, v_one)

        # === ROUNDS 2-15: Pipelined gather + hash ===
        # We'll try to overlap loads with VALU by building combined instructions

        for r in range(2, 16):
            # Compute addresses (ALU)
            for d in range(DESK):
                self.emit({"alu": [("+", s_addr[d][i], c_forest_p, v_idx[d] + i) for i in range(VLEN)]})

            # Build list of all load ops for this round
            load_ops = []
            for d in range(DESK):
                for i in range(0, VLEN, 2):
                    load_ops.append(("load", s_node[d][i], s_addr[d][i]))
                    load_ops.append(("load", s_node[d][i+1], s_addr[d][i+1]))

            # We have 128 load ops total, 2 per cycle = 64 cycles
            # Hash takes ~54 cycles of VALU
            # Let's try to overlap: do loads while doing unrelated VALU work

            # Problem: we can only do unrelated VALU work if we have something to compute
            # We could try: do loads for round R while doing wrap for round R-1
            # But that requires restructuring the loop...

            # For now, just emit loads then hash
            for i in range(0, len(load_ops), 2):
                self.emit({"load": load_ops[i:i+2]})

            # Copy to vectors
            for d in range(DESK):
                self.emit({"alu": [("+", v_tmp[d] + i, s_node[d][i], c_zero) for i in range(VLEN)]})

            # XOR
            for start in range(0, DESK, 6):
                end = min(start + 6, DESK)
                self.emit({"valu": [("^", v_val[d], v_val[d], v_tmp[d]) for d in range(start, end)]})

            self._do_hash(DESK, v_val, v_tmp, v_tmp2, vh_consts)
            self._do_next_index(DESK, v_idx, v_val, v_tmp, v_one)
            self._do_wrap(DESK, v_idx, v_tmp, v_n_nodes)

        # Store once
        for d in range(0, DESK, 2):
            self.emit({"store": [("vstore", idx_p[d], v_idx[d]), ("vstore", idx_p[d+1], v_idx[d+1])]})
        for d in range(0, DESK, 2):
            self.emit({"store": [("vstore", val_p[d], v_val[d]), ("vstore", val_p[d+1], v_val[d+1])]})

    def _do_hash(self, DESK, v_val, v_tmp, v_tmp2, vh_consts):
        for stage_i, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            v_c1, v_c3 = vh_consts[stage_i]
            for start in range(0, DESK, 3):
                end = min(start + 3, DESK)
                self.emit({"valu": [
                    op for d in range(start, end)
                    for op in [(op1, v_tmp[d], v_val[d], v_c1), (op3, v_tmp2[d], v_val[d], v_c3)]
                ]})
            for start in range(0, DESK, 6):
                end = min(start + 6, DESK)
                self.emit({"valu": [(op2, v_val[d], v_tmp[d], v_tmp2[d]) for d in range(start, end)]})

    def _do_next_index(self, DESK, v_idx, v_val, v_tmp, v_one):
        for start in range(0, DESK, 6):
            end = min(start + 6, DESK)
            self.emit({"valu": [("&", v_tmp[d], v_val[d], v_one) for d in range(start, end)]})
        for start in range(0, DESK, 6):
            end = min(start + 6, DESK)
            self.emit({"valu": [("+", v_tmp[d], v_one, v_tmp[d]) for d in range(start, end)]})
        for start in range(0, DESK, 6):
            end = min(start + 6, DESK)
            self.emit({"valu": [("<<", v_idx[d], v_idx[d], v_one) for d in range(start, end)]})
        for start in range(0, DESK, 6):
            end = min(start + 6, DESK)
            self.emit({"valu": [("+", v_idx[d], v_idx[d], v_tmp[d]) for d in range(start, end)]})

    def _do_wrap(self, DESK, v_idx, v_tmp, v_n_nodes):
        for start in range(0, DESK, 6):
            end = min(start + 6, DESK)
            self.emit({"valu": [("<", v_tmp[d], v_idx[d], v_n_nodes) for d in range(start, end)]})
        for start in range(0, DESK, 6):
            end = min(start + 6, DESK)
            self.emit({"valu": [("*", v_idx[d], v_idx[d], v_tmp[d]) for d in range(start, end)]})


def test_kernel():
    forest_height = 10
    rounds = 16
    batch_size = 256

    import random
    random.seed(42)

    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest_height, len(forest.values), batch_size, rounds)

    machine = Machine(mem, kb.instrs, kb.debug_info(), n_cores=N_CORES)
    machine.enable_pause = False
    machine.enable_debug = False
    machine.run()

    ref_mem = list(reference_kernel2(mem))[-1]

    inp_values_p = ref_mem[6]
    inp_indices_p = ref_mem[5]

    print("First 8 indices (machine):", machine.mem[inp_indices_p:inp_indices_p + 8])
    print("First 8 indices (ref):", ref_mem[inp_indices_p:inp_indices_p + 8])

    if machine.mem[inp_values_p:inp_values_p + batch_size] == ref_mem[inp_values_p:inp_values_p + batch_size]:
        print("CORRECT!")
    else:
        print("INCORRECT!")
        for i in range(batch_size):
            if machine.mem[inp_values_p + i] != ref_mem[inp_values_p + i]:
                print(f"First diff at {i}: machine={machine.mem[inp_values_p + i]}, ref={ref_mem[inp_values_p + i]}")
                break
        return

    print(f"CYCLES: {machine.cycle}")
    print(f"Target: 1363")
    print(f"Speedup needed: {machine.cycle / 1363:.1f}x")


if __name__ == "__main__":
    test_kernel()
