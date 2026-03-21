"""
H59: Constraint-Based Optimal Schedule Analysis

This script:
1. Extracts operations from the H54 kernel instruction stream
2. Builds dependency graph (def-use chains)
3. Creates a CP-SAT model to find optimal schedule
4. Compares optimal vs actual schedule to identify improvement potential
"""

import sys
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set
import time
import json

from ortools.sat.python import cp_model


# Constants from the problem
SLOT_LIMITS = {
    "alu": 12,
    "valu": 6,
    "load": 2,
    "store": 2,
    "flow": 1,
}

VLEN = 8


@dataclass
class Operation:
    """Represents a single operation in the instruction stream."""
    op_id: int
    cycle: int  # Original cycle in H54
    engine: str  # 'alu', 'valu', 'load', 'store', 'flow'
    slot: tuple  # The operation tuple
    dest: Optional[int]  # Destination scratch address (if any)
    srcs: List[int]  # Source scratch addresses
    latency: int = 1


def parse_slot(engine: str, slot: tuple) -> Tuple[Optional[int], List[int]]:
    """
    Parse a slot tuple to extract destination and source addresses.
    Returns (dest, [srcs])
    """
    if not isinstance(slot, tuple) or len(slot) == 0:
        return None, []

    op = slot[0]

    if engine == 'load':
        if op == 'const':
            # ("const", dest, value) - dest is scratch addr
            return slot[1], []
        elif op == 'load':
            # ("load", dest, src_addr) - scalar load
            return slot[1], [slot[2]]
        elif op == 'vload':
            # ("vload", dest, src_addr) - vector load
            return slot[1], [slot[2]]
        else:
            return None, []

    elif engine == 'store':
        if op == 'vstore':
            # ("vstore", addr, src) - store from src to memory at addr
            return None, [slot[1], slot[2]]
        elif op == 'store':
            return None, [slot[1], slot[2]]
        else:
            return None, []

    elif engine == 'valu':
        if op == 'vbroadcast':
            # ("vbroadcast", dest, scalar_src)
            return slot[1], [slot[2]]
        elif op == 'multiply_add':
            # ("multiply_add", dest, a, b, c) - dest = a * b + c
            return slot[1], [slot[2], slot[3], slot[4]]
        elif op in ['+', '-', '*', '^', '&', '|', '<', '>', '<<', '>>', '<=', '>=']:
            # Binary ops: (op, dest, src1, src2)
            return slot[1], [slot[2], slot[3]]
        else:
            return None, []

    elif engine == 'alu':
        if op in ['+', '-', '*', '^', '&', '|', '<', '>', '<<', '>>', '<=', '>=']:
            # Binary ops: (op, dest, src1, src2)
            return slot[1], [slot[2], slot[3]]
        else:
            return None, []

    elif engine == 'flow':
        if op == 'pause':
            return None, []
        elif op == 'cond_jump':
            # ("cond_jump", cond, target)
            return None, [slot[1]]
        elif op == 'select':
            # ("select", dest, cond, true_val, false_val)
            return slot[1], [slot[2], slot[3], slot[4]]
        elif op == 'vselect':
            return slot[1], [slot[2], slot[3], slot[4]]
        else:
            return None, []

    return None, []


def get_dest_range(dest: Optional[int], engine: str, op_type: str) -> List[int]:
    """Get all destination addresses for an operation."""
    if dest is None or dest == -1:
        return []
    if engine == "valu" or (engine == "load" and op_type == "vload"):
        return list(range(dest, dest + VLEN))
    return [dest]


def get_source_ranges(srcs: List[int], engine: str) -> List[int]:
    """Get all source addresses for an operation (including vector expansion)."""
    addrs = []
    for src in srcs:
        if engine == "valu":
            # Vector sources read VLEN addresses
            addrs.extend(range(src, src + VLEN))
        else:
            addrs.append(src)
    return addrs


def extract_operations(instrs: List[dict], start_cycle: int = 0, end_cycle: int = None) -> Tuple[List[Operation], List[Tuple[int, int]]]:
    """
    Extract operations and dependencies from instruction stream.

    Returns:
        operations: List of Operation objects
        dependencies: List of (src_op_id, dst_op_id) dependency edges
    """
    operations = []
    dependencies = []
    last_def: Dict[int, int] = {}  # scratch_addr -> op_id that last wrote to it

    if end_cycle is None:
        end_cycle = len(instrs)

    for cycle_idx in range(start_cycle, min(end_cycle, len(instrs))):
        instr = instrs[cycle_idx]
        for engine, slots in instr.items():
            if engine == 'debug':
                continue
            for slot in slots:
                op_id = len(operations)
                dest, srcs = parse_slot(engine, slot)
                op_type = slot[0] if slot else None

                op = Operation(
                    op_id=op_id,
                    cycle=cycle_idx - start_cycle,  # Normalize to 0-based
                    engine=engine,
                    slot=slot,
                    dest=dest,
                    srcs=srcs,
                    latency=1
                )
                operations.append(op)

                # Get source address ranges for dependencies
                src_addrs = get_source_ranges(srcs, engine)
                for addr in src_addrs:
                    if addr in last_def:
                        writer_id = last_def[addr]
                        if writer_id != op_id:
                            dependencies.append((writer_id, op_id))

                # Update last_def for destination addresses
                dest_addrs = get_dest_range(dest, engine, op_type)
                for addr in dest_addrs:
                    last_def[addr] = op_id

    # Deduplicate dependencies
    dependencies = list(set(dependencies))

    return operations, dependencies


def build_cpsat_model(
    operations: List[Operation],
    dependencies: List[Tuple[int, int]],
    horizon: int
) -> Tuple[cp_model.CpModel, Dict[int, any], any]:
    """
    Build CP-SAT model for optimal scheduling.

    Returns:
        model: The CP-SAT model
        starts: Dict mapping op_id -> start time variable
        makespan: The makespan variable
    """
    model = cp_model.CpModel()

    # Start time variables
    starts = {}
    ends = {}
    intervals = {}

    for op in operations:
        starts[op.op_id] = model.NewIntVar(0, horizon, f'start_{op.op_id}')
        ends[op.op_id] = model.NewIntVar(0, horizon + 1, f'end_{op.op_id}')
        # end = start + latency
        model.Add(ends[op.op_id] == starts[op.op_id] + op.latency)

        # Create interval for cumulative constraint
        intervals[op.op_id] = model.NewIntervalVar(
            starts[op.op_id],
            op.latency,
            ends[op.op_id],
            f'interval_{op.op_id}'
        )

    # Makespan variable
    makespan = model.NewIntVar(0, horizon + 1, 'makespan')

    # Dependency constraints: dst must start after src completes
    for src_id, dst_id in dependencies:
        model.Add(starts[dst_id] >= ends[src_id])

    # Resource constraints using cumulative
    for engine, limit in SLOT_LIMITS.items():
        engine_intervals = []
        for op in operations:
            if op.engine == engine:
                engine_intervals.append(intervals[op.op_id])

        if engine_intervals:
            demands = [1] * len(engine_intervals)
            model.AddCumulative(engine_intervals, demands, limit)

    # Makespan must be >= all end times
    for op in operations:
        model.Add(makespan >= ends[op.op_id])

    # Objective: minimize makespan
    model.Minimize(makespan)

    return model, starts, makespan


def load_h54_instructions():
    """
    Load the H54 kernel instructions by building from source.
    """
    # We need to create the kernel builder without the problem module
    # by defining a minimal version inline

    class MockDebugInfo:
        def __init__(self, scratch_map=None):
            self.scratch_map = scratch_map or {}

    HASH_STAGES = [
        ("+", 0x7ED55D16, "+", "<<", 12),
        ("^", 0xC761C23C, "^", ">>", 19),
        ("+", 0x165667B1, "+", "<<", 5),
        ("+", 0xD3A2646C, "^", "<<", 9),
        ("+", 0xFD7046C5, "+", "<<", 3),
        ("^", 0xB55A4F09, "^", ">>", 16),
    ]

    SCRATCH_SIZE = 1536
    NUM_DESKS = 16

    class KernelBuilderH54:
        """Minimal version of H54 kernel builder."""
        def __init__(self):
            self.instrs = []
            self.scratch = {}
            self.scratch_debug = {}
            self.scratch_ptr = 0
            self.const_map = {}

        def debug_info(self):
            return MockDebugInfo(scratch_map=self.scratch_debug)

        def add(self, engine, slot):
            self.instrs.append({engine: [slot]})

        def alloc_scratch(self, name=None, length=1):
            addr = self.scratch_ptr
            if name is not None:
                self.scratch[name] = addr
                self.scratch_debug[addr] = (name, length)
            self.scratch_ptr += length
            assert self.scratch_ptr <= SCRATCH_SIZE
            return addr

        def scratch_const(self, val, name=None):
            if val not in self.const_map:
                addr = self.alloc_scratch(name)
                self.add("load", ("const", addr, val))
                self.const_map[val] = addr
            return self.const_map[val]

        def build_kernel(self, forest_height, n_nodes, batch_size, rounds):
            """Build the H54 kernel - simplified version for analysis."""
            # Standard initialization
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

            # Hash constants
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

            # Allocate desk structures
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

            addr_tmp = [self.alloc_scratch(f"addr_tmp_{i}") for i in range(32)]
            offset_regs = [self.alloc_scratch(f"off_{d}") for d in range(NUM_DESKS)]
            batch_offset = self.alloc_scratch("batch_offset")
            iter_counter = self.alloc_scratch("iter_counter")

            offset_consts = []
            for d in range(NUM_DESKS):
                offset_consts.append(self.scratch_const(d * VLEN))

            batch_size_const = self.scratch_const(batch_size)
            desk_stride_const = self.scratch_const(NUM_DESKS * VLEN)
            total_iterations = (batch_size // VLEN) * (rounds // 2) // NUM_DESKS
            total_const = self.scratch_const(total_iterations)

            self.add("load", ("const", batch_offset, 0))
            self.add("load", ("const", iter_counter, 0))

            # === MAIN LOOP ===
            main_loop_start = len(self.instrs)

            # PHASE 1: Calculate offsets
            self.instrs.append({
                "alu": [
                    ("+", offset_regs[i], batch_offset, offset_consts[i])
                    for i in range(12)
                ],
            })
            self.instrs.append({
                "alu": [
                    ("+", offset_regs[i], batch_offset, offset_consts[i])
                    for i in range(12, 16)
                ],
            })

            # Compute load addresses
            for start_d in range(0, 12, 6):
                self.instrs.append({
                    "alu": [
                        item for d in range(start_d, min(start_d + 6, 12))
                        for item in [
                            ("+", addr_tmp[d*2], self.scratch["inp_indices_p"], offset_regs[d]),
                            ("+", addr_tmp[d*2+1], self.scratch["inp_values_p"], offset_regs[d]),
                        ]
                    ][:12],
                })
            self.instrs.append({
                "alu": [
                    ("+", addr_tmp[d*2], self.scratch["inp_indices_p"], offset_regs[d])
                    for d in range(6, 12)
                ] + [
                    ("+", addr_tmp[d*2+1], self.scratch["inp_values_p"], offset_regs[d])
                    for d in range(6, 10)
                ][:12],
            })
            self.instrs.append({
                "alu": [
                    ("+", addr_tmp[d*2], self.scratch["inp_indices_p"], offset_regs[d])
                    for d in range(12, 16)
                ] + [
                    ("+", addr_tmp[d*2+1], self.scratch["inp_values_p"], offset_regs[d])
                    for d in range(12, 16)
                ],
            })

            # PHASE 2: Load idx/val for 16 desks
            for d in range(0, NUM_DESKS, 2):
                self.instrs.append({
                    "load": [
                        ("vload", desks[d]['idx'], addr_tmp[d*2]),
                        ("vload", desks[d]['val'], addr_tmp[d*2+1]),
                    ],
                })
                self.instrs.append({
                    "load": [
                        ("vload", desks[d+1]['idx'], addr_tmp[(d+1)*2]),
                        ("vload", desks[d+1]['val'], addr_tmp[(d+1)*2+1]),
                    ],
                })

            # PHASE 3: Prepare gather addresses
            for start in [0, 6, 12]:
                count = min(6, NUM_DESKS - start)
                self.instrs.append({
                    "valu": [
                        ("vbroadcast", desks[i]['addr'], self.scratch["forest_values_p"])
                        for i in range(start, start + count)
                    ],
                })

            for start in [0, 6, 12]:
                count = min(6, NUM_DESKS - start)
                self.instrs.append({
                    "valu": [
                        ("+", desks[i]['addr'], desks[i]['addr'], desks[i]['idx'])
                        for i in range(start, start + count)
                    ],
                })

            # ROUNDS 1 and 2: Interleaved gather + hash
            # This is a simplified version - actual has complex interleaving
            for round_num in range(2):
                if round_num == 1:
                    # Prepare addresses for round 2
                    for start in [0, 6, 12]:
                        count = min(6, NUM_DESKS - start)
                        self.instrs.append({
                            "valu": [
                                ("vbroadcast", desks[i]['addr'], self.scratch["forest_values_p"])
                                for i in range(start, start + count)
                            ],
                        })
                    for start in [0, 6, 12]:
                        count = min(6, NUM_DESKS - start)
                        self.instrs.append({
                            "valu": [
                                ("+", desks[i]['addr'], desks[i]['addr'], desks[i]['idx'])
                                for i in range(start, start + count)
                            ],
                        })

                # Gather for each desk (8 scalar loads)
                for d in range(NUM_DESKS):
                    for lane in range(0, VLEN, 2):
                        valu_ops = []
                        # Add hash operations from previous desks
                        if d > 0:
                            # XOR node value
                            valu_ops.append(("^", desks[d-1]['val'], desks[d-1]['val'], desks[d-1]['node_val']))
                        if d > 1:
                            # FMA stage 0
                            valu_ops.append(("multiply_add", desks[d-2]['val'], desks[d-2]['val'], v_fma_mult[0], v_hash_consts[0]))

                        instr = {
                            "load": [
                                ("load", desks[d]['node_val'] + lane, desks[d]['addr'] + lane),
                                ("load", desks[d]['node_val'] + lane + 1, desks[d]['addr'] + lane + 1),
                            ],
                        }
                        if valu_ops:
                            instr["valu"] = valu_ops[:6]
                        self.instrs.append(instr)

                # Finish remaining hash operations after gathers
                for stage in range(6):
                    for d in range(NUM_DESKS - 4, NUM_DESKS):
                        if stage in [0, 2, 4]:
                            self.instrs.append({
                                "valu": [
                                    ("multiply_add", desks[d]['val'], desks[d]['val'], v_fma_mult[stage], v_hash_consts[stage])
                                ],
                            })
                        else:
                            self.instrs.append({
                                "valu": [
                                    ("^", desks[d]['tmp1'], desks[d]['val'], v_hash_consts[stage]),
                                    (">>" if stage in [1, 5] else "<<", desks[d]['tmp2'], desks[d]['val'], v_hash_shifts[stage]),
                                ],
                            })
                            self.instrs.append({
                                "valu": [
                                    ("^", desks[d]['val'], desks[d]['tmp1'], desks[d]['tmp2'])
                                ],
                            })

                # Branch calculations
                for d in range(NUM_DESKS):
                    self.instrs.append({
                        "valu": [
                            ("&", desks[d]['tmp1'], desks[d]['val'], v_one),
                            ("multiply_add", desks[d]['idx'], desks[d]['idx'], v_two, v_one),
                        ],
                    })
                    self.instrs.append({
                        "valu": [
                            ("+", desks[d]['idx'], desks[d]['idx'], desks[d]['tmp1']),
                            ("<", desks[d]['tmp1'], desks[d]['idx'], v_n_nodes),
                        ],
                    })
                    self.instrs.append({
                        "valu": [
                            ("*", desks[d]['idx'], desks[d]['idx'], desks[d]['tmp1']),
                        ],
                    })

            # STORE PHASE
            for d in range(0, 12, 6):
                self.instrs.append({
                    "alu": [
                        item for i in range(d, min(d + 6, 12))
                        for item in [
                            ("+", addr_tmp[i*2], self.scratch["inp_indices_p"], offset_regs[i]),
                            ("+", addr_tmp[i*2+1], self.scratch["inp_values_p"], offset_regs[i]),
                        ]
                    ][:12],
                })
            self.instrs.append({
                "alu": [
                    ("+", addr_tmp[d*2], self.scratch["inp_indices_p"], offset_regs[d])
                    for d in range(12, 16)
                ] + [
                    ("+", addr_tmp[d*2+1], self.scratch["inp_values_p"], offset_regs[d])
                    for d in range(12, 16)
                ],
            })

            for d in range(NUM_DESKS):
                self.instrs.append({
                    "store": [
                        ("vstore", addr_tmp[d*2], desks[d]['idx']),
                        ("vstore", addr_tmp[d*2+1], desks[d]['val']),
                    ],
                })

            # Loop control
            self.instrs.append({
                "alu": [
                    ("+", batch_offset, batch_offset, desk_stride_const),
                    ("+", iter_counter, iter_counter, one_const),
                ],
            })
            self.instrs.append({
                "alu": [
                    ("<", tmp_scalar, batch_offset, batch_size_const),
                    ("<", addr_scalar, iter_counter, total_const),
                ],
            })
            self.instrs.append({
                "flow": [
                    ("select", batch_offset, tmp_scalar, batch_offset, zero_const),
                ],
            })
            self.add("flow", ("cond_jump", addr_scalar, main_loop_start))
            self.instrs.append({"flow": [("pause",)]})

    kb = KernelBuilderH54()
    kb.build_kernel(forest_height=10, n_nodes=1023, batch_size=256, rounds=16)
    return kb.instrs


def find_main_loop(instrs: List[dict]) -> Tuple[int, int]:
    """Find the main loop start and end indices."""
    main_loop_start = None
    main_loop_end = None

    for i, instr in enumerate(instrs):
        if 'flow' in instr:
            for slot in instr['flow']:
                if slot[0] == 'cond_jump':
                    main_loop_end = i + 1
                    main_loop_start = slot[2]
                    break
        if main_loop_end:
            break

    return main_loop_start, main_loop_end


def count_operations_by_engine(operations: List[Operation]) -> Dict[str, int]:
    """Count operations by engine type."""
    counts = defaultdict(int)
    for op in operations:
        counts[op.engine] += 1
    return dict(counts)


def analyze_current_schedule(operations: List[Operation], original_cycles: int) -> Dict[str, any]:
    """Analyze the current H54 schedule."""
    counts = count_operations_by_engine(operations)

    # Calculate utilization
    total_slots = {
        engine: original_cycles * limit
        for engine, limit in SLOT_LIMITS.items()
    }
    used_slots = counts
    utilization = {
        engine: used_slots.get(engine, 0) / total_slots[engine] * 100
        for engine in SLOT_LIMITS
    }

    return {
        'counts': counts,
        'actual_makespan': original_cycles,
        'total_slots': total_slots,
        'utilization': utilization,
    }


def compute_theoretical_minimum(operations: List[Operation]) -> int:
    """Compute theoretical minimum cycles based on resource constraints only."""
    counts = count_operations_by_engine(operations)

    mins = {}
    for engine, limit in SLOT_LIMITS.items():
        if engine in counts:
            mins[engine] = (counts[engine] + limit - 1) // limit
        else:
            mins[engine] = 0

    return max(mins.values()) if mins else 0


def solve_optimal_schedule(
    operations: List[Operation],
    dependencies: List[Tuple[int, int]],
    time_limit_seconds: int = 300,
    original_makespan: int = None
) -> Dict[str, any]:
    """Solve for optimal schedule using CP-SAT."""
    if original_makespan:
        horizon = original_makespan + 10
    else:
        horizon = len(operations) * 2

    print(f"Building CP-SAT model with {len(operations)} operations, {len(dependencies)} dependencies")
    print(f"Horizon: {horizon}")

    model, starts, makespan = build_cpsat_model(operations, dependencies, horizon)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.num_search_workers = 8

    print(f"Solving with {time_limit_seconds}s time limit...")
    start_time = time.time()
    status = solver.Solve(model)
    solve_time = time.time() - start_time

    result = {
        'status': solver.StatusName(status),
        'solve_time': solve_time,
        'optimal_makespan': None,
        'schedule': None,
    }

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        result['optimal_makespan'] = solver.Value(makespan)
        result['schedule'] = {
            op.op_id: solver.Value(starts[op.op_id])
            for op in operations
        }

    return result


def analyze_schedule_differences(operations: List[Operation], optimal_schedule: Dict[int, int]):
    """Analyze differences between original and optimal schedules."""
    print("\n[Step 7] Analyzing schedule differences...")

    moved_earlier = []
    moved_later = []
    same = []

    for op in operations:
        diff = optimal_schedule[op.op_id] - op.cycle
        if diff < 0:
            moved_earlier.append((op, diff))
        elif diff > 0:
            moved_later.append((op, diff))
        else:
            same.append(op)

    print(f"\n  Operations moved earlier: {len(moved_earlier)}")
    print(f"  Operations moved later:   {len(moved_later)}")
    print(f"  Operations unchanged:     {len(same)}")

    if moved_earlier:
        print("\n  Top 10 operations moved earlier:")
        moved_earlier.sort(key=lambda x: x[1])
        for op, diff in moved_earlier[:10]:
            print(f"    Op {op.op_id:4d} ({op.engine:5s}): moved {-diff} cycles earlier")

    # Movement by engine
    print("\n  Movement by engine type:")
    engine_movement = defaultdict(list)
    for op, diff in moved_earlier + moved_later:
        engine_movement[op.engine].append(diff)

    for engine in SLOT_LIMITS:
        if engine in engine_movement:
            diffs = engine_movement[engine]
            avg_diff = sum(diffs) / len(diffs)
            print(f"    {engine:6s}: avg movement {avg_diff:+.1f} cycles")


def main():
    """Main analysis routine."""
    print("=" * 70)
    print("H59: Constraint-Based Optimal Schedule Analysis")
    print("=" * 70)

    # Step 1: Load H54 instructions
    print("\n[Step 1] Loading H54 kernel (simplified model)...")
    instrs = load_h54_instructions()
    print(f"  Total instructions (cycles): {len(instrs)}")

    # Step 2: Find main loop
    print("\n[Step 2] Finding main loop boundaries...")
    main_loop_start, main_loop_end = find_main_loop(instrs)
    if main_loop_start is None:
        print("  Could not find main loop!")
        return None

    main_loop_cycles = main_loop_end - main_loop_start
    print(f"  Main loop: cycles {main_loop_start} to {main_loop_end}")
    print(f"  Main loop size: {main_loop_cycles} cycles")

    # Step 3: Extract operations
    print("\n[Step 3] Extracting operations from main loop...")
    operations, dependencies = extract_operations(instrs, main_loop_start, main_loop_end)
    print(f"  Operations: {len(operations)}")
    print(f"  Dependencies: {len(dependencies)}")

    # Step 4: Analyze current schedule
    print("\n[Step 4] Analyzing current schedule...")
    current_analysis = analyze_current_schedule(operations, main_loop_cycles)

    print(f"\n  Operation counts by engine:")
    for engine, count in sorted(current_analysis['counts'].items()):
        print(f"    {engine:6s}: {count:4d} ops")

    print(f"\n  Current makespan: {current_analysis['actual_makespan']} cycles")

    print(f"\n  Resource utilization:")
    for engine in SLOT_LIMITS:
        util = current_analysis['utilization'].get(engine, 0)
        print(f"    {engine:6s}: {util:5.1f}%")

    # Step 5: Compute theoretical minimum
    print("\n[Step 5] Computing theoretical minimum...")
    theoretical_min = compute_theoretical_minimum(operations)
    print(f"  Theoretical minimum (resource-bound): {theoretical_min} cycles")

    # Step 6: Solve optimal schedule
    print("\n[Step 6] Solving for optimal schedule...")
    solve_result = solve_optimal_schedule(
        operations,
        dependencies,
        time_limit_seconds=300,
        original_makespan=current_analysis['actual_makespan']
    )

    print(f"\n  Solver status: {solve_result['status']}")
    print(f"  Solve time: {solve_result['solve_time']:.2f}s")

    results = {
        'current_makespan': current_analysis['actual_makespan'],
        'theoretical_min': theoretical_min,
        'operations': len(operations),
        'dependencies': len(dependencies),
        'main_loop_cycles': main_loop_cycles,
        'counts': current_analysis['counts'],
        'utilization': current_analysis['utilization'],
    }

    if solve_result['optimal_makespan']:
        optimal = solve_result['optimal_makespan']
        current = current_analysis['actual_makespan']

        print(f"\n  Results:")
        print(f"    Current makespan:      {current} cycles")
        print(f"    Optimal makespan:      {optimal} cycles")
        print(f"    Theoretical minimum:   {theoretical_min} cycles")

        improvement = (current - optimal) / current * 100 if current > 0 else 0
        print(f"\n    Improvement potential: {current - optimal} cycles ({improvement:.1f}%)")

        gap_to_theory = (optimal - theoretical_min) / theoretical_min * 100 if theoretical_min > 0 else 0
        print(f"    Gap from theory:       {optimal - theoretical_min} cycles ({gap_to_theory:.1f}%)")

        results['optimal_makespan'] = optimal
        results['improvement_cycles'] = current - optimal
        results['improvement_pct'] = improvement
        results['gap_to_theory_cycles'] = optimal - theoretical_min
        results['gap_to_theory_pct'] = gap_to_theory
        results['solver_status'] = solve_result['status']
        results['solve_time'] = solve_result['solve_time']

        if solve_result['schedule']:
            analyze_schedule_differences(operations, solve_result['schedule'])
    else:
        print("  No feasible solution found within time limit.")
        results['solver_status'] = solve_result['status']
        results['solve_time'] = solve_result['solve_time']

    # Save results
    results_path = "/home/hestiasadmin/projects/original_performance_takehome/experiments/H59_ilp_schedule/results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {results_path}")

    return results


if __name__ == "__main__":
    results = main()
