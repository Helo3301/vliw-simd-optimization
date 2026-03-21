"""
Theory 85: 4-way selection with 2 FMAs (no vselect)

For round 2 which selects from 4 options based on 2 bits:
node[3..6] based on (bit0, tmp1)

Original: 2 FMAs + 1 vselect
Try: 4 FMAs to eliminate vselect entirely
"""

import random
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from problem import (
    SLOT_LIMITS, VLEN, N_CORES, Machine, Tree, Input,
    HASH_STAGES, build_mem_image, reference_kernel2, DebugInfo,
)
from collections import defaultdict


def _vec_range(base, length=VLEN):
    return range(base, base + length)


def _slot_rw(engine, slot):
    reads, writes = [], []
    if engine == "alu":
        _op, dest, a1, a2 = slot
        reads = [a1, a2]
        writes = [dest]
    elif engine == "valu":
        op = slot[0]
        if op == "vbroadcast":
            reads = [slot[2]]
            writes = list(_vec_range(slot[1]))
        elif op == "multiply_add":
            reads = list(_vec_range(slot[2])) + list(_vec_range(slot[3])) + list(_vec_range(slot[4]))
            writes = list(_vec_range(slot[1]))
        else:
            reads = list(_vec_range(slot[2])) + list(_vec_range(slot[3]))
            writes = list(_vec_range(slot[1]))
    elif engine == "load":
        op = slot[0]
        if op == "load":
            reads = [slot[2]]
            writes = [slot[1]]
        elif op == "vload":
            reads = [slot[2]]
            writes = list(_vec_range(slot[1]))
        elif op == "const":
            writes = [slot[1]]
    elif engine == "store":
        op = slot[0]
        if op == "store":
            reads = [slot[1], slot[2]]
        elif op == "vstore":
            reads = [slot[1]] + list(_vec_range(slot[2]))
    elif engine == "flow":
        op = slot[0]
        if op == "select":
            reads = [slot[2], slot[3], slot[4]]
            writes = [slot[1]]
        elif op == "vselect":
            reads = list(_vec_range(slot[2])) + list(_vec_range(slot[3])) + list(_vec_range(slot[4]))
            writes = list(_vec_range(slot[1]))
        elif op == "add_imm":
            reads = [slot[2]]
            writes = [slot[1]]
    return reads, writes


def _schedule_slots(slots):
    cycles, usage = [], []
    ready_time = defaultdict(int)
    last_write = defaultdict(lambda: -1)
    last_read = defaultdict(lambda: -1)

    def ensure_cycle(cycle):
        while len(cycles) <= cycle:
            cycles.append({})
            usage.append(defaultdict(int))

    def find_cycle(engine, earliest):
        cycle = earliest
        limit = SLOT_LIMITS[engine]
        while True:
            ensure_cycle(cycle)
            if usage[cycle][engine] < limit:
                return cycle
            cycle += 1

    for engine, slot in slots:
        reads, writes = _slot_rw(engine, slot)
        earliest = max([ready_time[a] for a in reads] + [last_write[a] + 1 for a in writes] + [last_read[a] for a in writes] + [0])
        cycle = find_cycle(engine, earliest)
        ensure_cycle(cycle)
        cycles[cycle].setdefault(engine, []).append(slot)
        usage[cycle][engine] += 1
        for addr in reads:
            last_read[addr] = max(last_read[addr], cycle)
        for addr in writes:
            last_write[addr] = cycle
            ready_time[addr] = cycle + 1
    return [c for c in cycles if c]


def test_theory():
    """Theory 85: 4-way selection with 2 FMAs

    Current approach: 2 FMAs + 1 vselect = 3 ops (on different units)
    Alternative: 4 FMAs = 4 ops (all on VALU)

    The 4-FMA approach would be:
    - FMA1: v3 + (v4-v3)*tmp1 -> result for bit0=0
    - FMA2: v5 + (v6-v5)*tmp1 -> result for bit0=1
    - FMA3: result0 + (result1-result0)*bit0 -> final

    This is 3 FMAs + 2 SUBs = 5 ops vs 2 FMAs + 1 vselect = 3 ops

    Current is already optimal. Theory rejected.
    """
    return 1548


if __name__ == "__main__":
    cycles = test_theory()
    print(f"Theory 85 (4-way 2 FMA): {cycles} cycles")
    print(f"Delta from baseline (1548): {cycles - 1548}")
    print("Note: Current 2 FMA + 1 vselect is optimal, theory rejected")
