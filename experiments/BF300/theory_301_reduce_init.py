"""
Theory 301: Reduce initialization overhead.

Current init: 27 cycles. Includes:
- Fast init vars (4 loads of pointers)
- 10 vconsts (v_zero, v_one, v_two, v_n_nodes, v_forest_p, v_1_minus_fp, v_fp_plus_1, v_fp_plus_15, v_fifteen, hash consts/shifts/fma)
- 15 tree node broadcasts
- 16 desk offset_regs + addr_tmp setup

Ideas:
1. Remove v_zero and v_n_nodes (v_zero unused? v_n_nodes used for bounds checks?)
2. Remove v_fp_plus_1 if we can avoid emit_branch_idx_to_addr
3. Defer tree broadcasts to overlap with other work
"""
# Let me check if v_zero and v_n_nodes are actually used in Theory 222
import sys
sys.path.insert(0, '/home/hestiasadmin/projects/original_performance_takehome')
sys.path.insert(0, '/home/hestiasadmin/projects/original_performance_takehome/experiments/BF300')

from theory_222_hashorder_WIN import KernelBuilderA1
from problem import *
import random

random.seed(123)
forest = Tree.generate(10)
inp = Input.generate(forest, 256, 16)

kb = KernelBuilderA1()
kb.build_kernel(forest.height, len(forest.values), 256, 16)

# Check which scratch addresses are actually read
reads_set = set()
writes_set = set()
for engine, slot in kb.slots:
    from theory_222_hashorder_WIN import _slot_rw
    reads, writes = _slot_rw(engine, slot)
    reads_set.update(reads)
    writes_set.update(writes)

# Check v_zero
v_zero_addr = kb.scratch.get('v_zero')
if v_zero_addr is not None:
    v_zero_range = set(range(v_zero_addr, v_zero_addr + VLEN))
    v_zero_read = v_zero_range & reads_set
    v_zero_write = v_zero_range & writes_set
    print(f"v_zero ({v_zero_addr}): read={bool(v_zero_read)}, written={bool(v_zero_write)}")
    if v_zero_read:
        # Find which ops read v_zero
        for i, (engine, slot) in enumerate(kb.slots):
            reads, _ = _slot_rw(engine, slot)
            if v_zero_range & set(reads):
                print(f"  Read by slot {i}: {engine} {slot}")

# Check v_n_nodes
v_n_nodes_addr = kb.scratch.get('v_n_nodes')
if v_n_nodes_addr is not None:
    v_n_nodes_range = set(range(v_n_nodes_addr, v_n_nodes_addr + VLEN))
    v_n_nodes_read = v_n_nodes_range & reads_set
    print(f"v_n_nodes ({v_n_nodes_addr}): read={bool(v_n_nodes_read)}")
    if v_n_nodes_read:
        count = 0
        for i, (engine, slot) in enumerate(kb.slots):
            reads, _ = _slot_rw(engine, slot)
            if v_n_nodes_range & set(reads):
                count += 1
        print(f"  Read by {count} slots")

# Check v_fp_plus_1
v_fp1_addr = kb.scratch.get('v_fp_plus_1')
if v_fp1_addr is not None:
    v_fp1_range = set(range(v_fp1_addr, v_fp1_addr + VLEN))
    v_fp1_read = v_fp1_range & reads_set
    print(f"v_fp_plus_1 ({v_fp1_addr}): read={bool(v_fp1_read)}")
