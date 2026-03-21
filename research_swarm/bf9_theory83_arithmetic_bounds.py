"""
Theory 83: Use arithmetic instead of comparison for bounds

Instead of: cmp = idx >= n_nodes; idx = select(cmp, 0, idx)
Try: idx = idx - (idx >= n_nodes) * idx  [one FMA if available]

This converts a comparison+select into arithmetic ops.
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
    """Theory 83: No explicit bounds checks exist in baseline to convert to arithmetic.
    The baseline already optimizes bounds away in R10 (sets idx=0 directly).
    This theory is N/A."""
    return 1548


if __name__ == "__main__":
    cycles = test_theory()
    print(f"Theory 83 (Arithmetic bounds): {cycles} cycles")
    print(f"Delta from baseline (1548): {cycles - 1548}")
    print("Note: Theory N/A - baseline has no explicit bounds checks")
