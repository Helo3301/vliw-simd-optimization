"""Debug: Compare kernel output to reference."""
import random
import sys
sys.path.insert(0, ".")

from problem import (
    Tree, Input, build_mem_image, reference_kernel2, Machine, N_CORES
)
from perf_takehome_math1 import KernelBuilder

def compare_outputs():
    random.seed(123)
    forest = Tree.generate(10)
    inp = Input.generate(forest, 256, 16)
    mem = build_mem_image(forest, inp)

    # Get reference result
    ref_mem = mem.copy()
    for ref_result in reference_kernel2(ref_mem, {}):
        pass  # Run to completion

    # Get MATH1 result
    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), 16)

    machine = Machine(
        mem.copy(),
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace={},
        trace=False,
    )
    # Run until completion (run through both pauses)
    machine.run()  # Run setup phase (until first pause)
    machine.run()  # Run main computation (until final pause)

    # Compare
    inp_indices_p = mem[5]
    inp_values_p = mem[6]
    batch_size = mem[2]

    print(f"Comparing {batch_size} elements...")
    print()

    # Check indices
    idx_mismatches = []
    for i in range(batch_size):
        ref_idx = ref_mem[inp_indices_p + i]
        our_idx = machine.mem[inp_indices_p + i]
        if ref_idx != our_idx:
            idx_mismatches.append((i, ref_idx, our_idx))

    if idx_mismatches:
        print(f"INDEX MISMATCHES: {len(idx_mismatches)}/{batch_size}")
        for i, ref, our in idx_mismatches[:10]:
            print(f"  Element {i}: ref={ref}, ours={our}")
        if len(idx_mismatches) > 10:
            print(f"  ... and {len(idx_mismatches) - 10} more")
    else:
        print("All indices match!")
    print()

    # Check values
    val_mismatches = []
    for i in range(batch_size):
        ref_val = ref_mem[inp_values_p + i]
        our_val = machine.mem[inp_values_p + i]
        if ref_val != our_val:
            val_mismatches.append((i, ref_val, our_val))

    if val_mismatches:
        print(f"VALUE MISMATCHES: {len(val_mismatches)}/{batch_size}")
        for i, ref, our in val_mismatches[:10]:
            print(f"  Element {i}: ref={ref}, ours={our}")
        if len(val_mismatches) > 10:
            print(f"  ... and {len(val_mismatches) - 10} more")
    else:
        print("All values match!")

if __name__ == "__main__":
    compare_outputs()
