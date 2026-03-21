#!/usr/bin/env python3.11
"""
Extract H54 instructions using Python 3.11 and save them for analysis.
"""

import sys
import json
sys.path.insert(0, '/home/hestiasadmin/projects/original_performance_takehome')

from experiments.H54_16desk.perf_takehome_h54 import KernelBuilderH54

kb = KernelBuilderH54()
kb.build_kernel(forest_height=10, n_nodes=1023, batch_size=256, rounds=16)

print(f'Total instructions: {len(kb.instrs)}')

# Find main loop
loop_start = None
loop_end = None
for i, instr in enumerate(kb.instrs):
    if 'flow' in instr:
        for slot in instr['flow']:
            if slot[0] == 'cond_jump':
                loop_end = i + 1
                loop_start = slot[2]
                print(f'Main loop: {loop_start} to {loop_end} = {loop_end - loop_start} cycles')
                break

# Convert instructions to JSON-serializable format
def convert_instr(instr):
    result = {}
    for engine, slots in instr.items():
        result[engine] = [list(slot) if isinstance(slot, tuple) else slot for slot in slots]
    return result

instrs_json = [convert_instr(instr) for instr in kb.instrs]

# Save to file
output = {
    'total_instructions': len(kb.instrs),
    'loop_start': loop_start,
    'loop_end': loop_end,
    'loop_cycles': loop_end - loop_start if loop_start and loop_end else None,
    'instructions': instrs_json,
    'scratch_map': {str(k): list(v) for k, v in kb.scratch_debug.items()},
}

with open('/home/hestiasadmin/projects/original_performance_takehome/experiments/H59_ilp_schedule/h54_instrs.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f'Saved instructions to h54_instrs.json')

# Count operations
from collections import defaultdict
op_counts = defaultdict(int)
for instr in kb.instrs[loop_start:loop_end]:
    for engine, slots in instr.items():
        op_counts[engine] += len(slots)

print("\nOperation counts in main loop:")
for engine in ['alu', 'valu', 'load', 'store', 'flow']:
    print(f"  {engine}: {op_counts.get(engine, 0)}")
