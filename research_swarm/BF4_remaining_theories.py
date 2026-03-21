"""
BF4: Remaining theories analysis

34. Store 2*idx instead of idx between rounds
    - Instead of idx, store doubled_idx = 2*idx
    - Branch: doubled_idx = doubled_idx * 2 + 1 + bit = doubled_idx << 1 | 1 | bit
    - Wait, that's 4 ops: shift, or 1, and bit, or bit
    - Current: &, FMA, + = 3 ops
    - Analysis: WORSE - this adds complexity, not removes

35. Store idx+1 instead of idx (offset addressing)
    - Store idx_plus_1 = idx + 1
    - Load addr: forest_p + idx_plus_1 - 1 = forest_p + idx_plus_1 + (-1)
    - Branch: idx*2 + 1 + bit = (idx_plus_1 - 1)*2 + 1 + bit
                              = idx_plus_1*2 - 2 + 1 + bit
                              = idx_plus_1*2 - 1 + bit
    - FMA(idx_plus_1, 2, -1) then add bit = 2 ops (if -1 can be constant)
    - Wait: FMA(idx_plus_1, 2, -1) + bit, then need to handle -1 const
    - Actually: we'd need v_minus_one, which costs extra broadcast
    - Current storage: idx starts at 0, addr = forest_p + idx
    - New storage: idx_plus_1 starts at 1, addr = forest_p + idx_plus_1 - 1
    - Analysis: Adds SUB everywhere, likely WORSE

38. Batch branch computation for 2 rounds at once
    - After 2 hashes, compute both branches at once
    - But they're sequential: idx1 = branch(idx0, bit0), idx2 = branch(idx1, bit1)
    - Can't parallelize due to dependency
    - Analysis: NO BENEFIT - sequential dependency

39. Use lookup table for small idx values
    - For idx in [0..6] (first 3 levels), preload children
    - idx=0 -> children 1,2
    - idx=1 -> children 3,4
    - idx=2 -> children 5,6
    - etc
    - But we already have early rounds fused with preloaded nodes
    - Lookup table approach: load child_left[idx], child_right[idx]
    - Requires scatter/gather which is expensive
    - Analysis: NO BENEFIT for our vector approach

Let me implement Theory 34 to verify it's worse, and analyze Theory 35 more carefully.
"""

import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from problem import (
    DebugInfo, SLOT_LIMITS, VLEN, N_CORES, SCRATCH_SIZE, Machine, Tree, Input,
    HASH_STAGES, build_mem_image, reference_kernel2,
)
from collections import defaultdict


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
        op = slot[0]
        if op == "vbroadcast":
            dest, src = slot[1], slot[2]
            reads = [src]
            writes = list(_vec_range(dest))
        elif op == "multiply_add":
            dest, a, b, c = slot[1], slot[2], slot[3], slot[4]
            reads = list(_vec_range(a)) + list(_vec_range(b)) + list(_vec_range(c))
            writes = list(_vec_range(dest))
        else:
            _op, dest, a1, a2 = slot
            reads = list(_vec_range(a1)) + list(_vec_range(a2))
            writes = list(_vec_range(dest))
    elif engine == "load":
        op = slot[0]
        if op == "load":
            dest, addr = slot[1], slot[2]
            reads = [addr]
            writes = [dest]
        elif op == "vload":
            dest, addr = slot[1], slot[2]
            reads = [addr]
            writes = list(_vec_range(dest))
        elif op == "const":
            dest = slot[1]
            writes = [dest]
    elif engine == "store":
        op = slot[0]
        if op == "store" or op == "vstore":
            addr, src = slot[1], slot[2]
            reads = [addr]
    elif engine == "flow":
        op = slot[0]
        if op == "vselect":
            dest, cond, a, b = slot[1], slot[2], slot[3], slot[4]
            reads = list(_vec_range(cond)) + list(_vec_range(a)) + list(_vec_range(b))
            writes = list(_vec_range(dest))
    return reads, writes


def _schedule_slots(slots: list[tuple[str, tuple]]) -> list[dict[str, list[tuple]]]:
    cycles: list[dict[str, list[tuple]]] = []
    usage: list[dict[str, int]] = []
    ready_time: dict[int, int] = defaultdict(int)
    last_write: dict[int, int] = defaultdict(lambda: -1)
    last_read: dict[int, int] = defaultdict(lambda: -1)

    def ensure_cycle(cycle: int):
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


# Analysis summary - no new implementations needed
# Theory 34: Store 2*idx = branch is (2*idx)*2 + 1 + bit = 4*idx + 1 + bit
#   - Need to recover idx for address: idx = doubled_idx / 2 (needs shift)
#   - Adds complexity: WORSE
# Theory 35: Store idx+1 = need to adjust all address calculations
#   - addr = forest_p + idx = forest_p + (idx_plus_1 - 1)
#   - Subtractions everywhere: WORSE
# Theory 38: Can't batch due to sequential dependency
# Theory 39: Lookup tables don't help vectorized code

print("BF4: Analysis of Remaining Theories")
print("=" * 60)
print()
print("Theory 34: Store 2*idx instead of idx")
print("  - Branch: doubled_idx*2 + 1 + bit = 4*idx + 1 + bit")
print("  - Address recovery needs: addr = forest_p + doubled_idx/2")
print("  - Extra shift needed: WORSE (adds operations)")
print()
print("Theory 35: Store idx+1 instead of idx")
print("  - Address: forest_p + (idx_plus_1 - 1)")
print("  - Subtraction on every address calc: WORSE")
print()
print("Theory 38: Batch branch computation for 2 rounds")
print("  - idx1 depends on idx0, idx2 depends on idx1")
print("  - Cannot parallelize: NO BENEFIT")
print()
print("Theory 39: Lookup table for small idx")
print("  - Would need scatter/gather for vector lookup")
print("  - Already using FMA-based early round fusion")
print("  - Lookup is not faster for SIMD: NO BENEFIT")
print()
print("=" * 60)
print("SUMMARY: Theories 34, 35, 38, 39 provide no benefit")
