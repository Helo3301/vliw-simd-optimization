"""
Genetic Algorithm Framework for Kernel Evolution
================================================

Target: Find sub-1,400 cycle solutions
Current best: 1,645 cycles (H140)
Target: 1,363 cycles

Mutation parameters:
- GROUP_SIZE: 1, 2, 3, 4, 6, 8, 16
- TILE_COUNT: 1, 2
- NUM_PRELOADED: 3, 5, 7, 9, 11
- ROUND_ORDER: various orderings
- INTERLEAVE_PATTERN: different desk interleaving strategies
- HASH_INTERLEAVE: 1, 2, 3 (hash stages interleaved across desks)
"""

import random
import copy
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import subprocess

@dataclass
class KernelGenome:
    """Represents the genetic makeup of a kernel variant."""
    group_size: int = 4
    tile_count: int = 2  # 1 or 2 tiles
    num_preloaded: int = 7
    interleave_depth: int = 1  # How many rounds to fuse together
    hash_interleave: int = 1  # Hash stage interleaving factor
    round_fusion_start: int = 0  # Start of fused round range
    round_fusion_end: int = 16  # End of fused round range
    desk_batch_size: int = 4  # Desks processed together
    use_speculative_addr: bool = False  # Speculative address computation
    inline_diffs: bool = True  # Inline diff computations
    skip_final_branch: bool = True  # Skip computing branch in round 15
    parallel_hash_desks: int = 1  # Number of desks to hash in parallel

    def __str__(self):
        return (f"GS{self.group_size}_T{self.tile_count}_P{self.num_preloaded}_"
                f"ID{self.interleave_depth}_HI{self.hash_interleave}_"
                f"DB{self.desk_batch_size}_PH{self.parallel_hash_desks}")

    def mutate(self) -> 'KernelGenome':
        """Create a mutated copy."""
        child = copy.deepcopy(self)

        # Choose which gene to mutate
        gene = random.choice([
            'group_size', 'tile_count', 'num_preloaded',
            'interleave_depth', 'hash_interleave', 'desk_batch_size',
            'parallel_hash_desks', 'skip_final_branch', 'inline_diffs'
        ])

        if gene == 'group_size':
            child.group_size = random.choice([1, 2, 3, 4, 5, 6, 8, 16])
        elif gene == 'tile_count':
            child.tile_count = random.choice([1, 2])
        elif gene == 'num_preloaded':
            child.num_preloaded = random.choice([3, 5, 7, 9, 11, 15])
        elif gene == 'interleave_depth':
            child.interleave_depth = random.choice([1, 2, 4, 8])
        elif gene == 'hash_interleave':
            child.hash_interleave = random.choice([1, 2, 3, 4])
        elif gene == 'desk_batch_size':
            child.desk_batch_size = random.choice([1, 2, 4, 8, 16])
        elif gene == 'parallel_hash_desks':
            child.parallel_hash_desks = random.choice([1, 2, 4])
        elif gene == 'skip_final_branch':
            child.skip_final_branch = not child.skip_final_branch
        elif gene == 'inline_diffs':
            child.inline_diffs = not child.inline_diffs

        return child

    @staticmethod
    def crossover(parent1: 'KernelGenome', parent2: 'KernelGenome') -> 'KernelGenome':
        """Create offspring by crossing two parents."""
        child = KernelGenome()

        # Uniform crossover - each gene from random parent
        child.group_size = random.choice([parent1.group_size, parent2.group_size])
        child.tile_count = random.choice([parent1.tile_count, parent2.tile_count])
        child.num_preloaded = random.choice([parent1.num_preloaded, parent2.num_preloaded])
        child.interleave_depth = random.choice([parent1.interleave_depth, parent2.interleave_depth])
        child.hash_interleave = random.choice([parent1.hash_interleave, parent2.hash_interleave])
        child.desk_batch_size = random.choice([parent1.desk_batch_size, parent2.desk_batch_size])
        child.parallel_hash_desks = random.choice([parent1.parallel_hash_desks, parent2.parallel_hash_desks])
        child.skip_final_branch = random.choice([parent1.skip_final_branch, parent2.skip_final_branch])
        child.inline_diffs = random.choice([parent1.inline_diffs, parent2.inline_diffs])

        return child


@dataclass
class Individual:
    """An individual in the population."""
    genome: KernelGenome
    fitness: float = float('inf')  # Lower is better (cycle count)
    correct: bool = False
    generation: int = 0


class GeneticAlgorithm:
    """Genetic algorithm for kernel optimization."""

    def __init__(self, population_size: int = 20, mutation_rate: float = 0.3):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population: List[Individual] = []
        self.generation = 0
        self.best_ever: Optional[Individual] = None
        self.history: List[Tuple[int, float]] = []  # (generation, best_fitness)

    def initialize_population(self):
        """Create initial population with diverse genomes."""
        # Start with known good configuration (H140)
        baseline = KernelGenome(
            group_size=4, tile_count=2, num_preloaded=7,
            interleave_depth=1, hash_interleave=1, desk_batch_size=4,
            parallel_hash_desks=1, skip_final_branch=True, inline_diffs=True
        )
        self.population.append(Individual(genome=baseline, generation=0))

        # Add some random variants
        variations = [
            KernelGenome(group_size=2, tile_count=2, num_preloaded=7),
            KernelGenome(group_size=8, tile_count=2, num_preloaded=7),
            KernelGenome(group_size=4, tile_count=1, num_preloaded=7),
            KernelGenome(group_size=4, tile_count=2, num_preloaded=3),
            KernelGenome(group_size=4, tile_count=2, num_preloaded=15),
            KernelGenome(group_size=6, tile_count=2, num_preloaded=7),
            KernelGenome(group_size=1, tile_count=2, num_preloaded=7),
            KernelGenome(group_size=16, tile_count=2, num_preloaded=7),
            KernelGenome(group_size=4, tile_count=2, num_preloaded=7, hash_interleave=2),
            KernelGenome(group_size=4, tile_count=2, num_preloaded=7, parallel_hash_desks=2),
        ]

        for v in variations:
            self.population.append(Individual(genome=v, generation=0))

        # Fill rest with random mutations of baseline
        while len(self.population) < self.population_size:
            mutant = baseline.mutate()
            for _ in range(random.randint(1, 3)):  # Multiple mutations
                mutant = mutant.mutate()
            self.population.append(Individual(genome=mutant, generation=0))

    def evaluate_individual(self, ind: Individual, timeout: int = 60) -> float:
        """Evaluate fitness of an individual by generating and running kernel."""
        # Generate kernel code
        kernel_code = generate_kernel_code(ind.genome)

        # Write to temp file
        kernel_path = f"/tmp/ga_kernel_{id(ind)}.py"
        with open(kernel_path, 'w') as f:
            f.write(kernel_code)

        try:
            # Run with correctness check
            result = subprocess.run(
                ['python3.11', kernel_path, '--check'],
                capture_output=True,
                text=True,
                timeout=timeout
            )

            if result.returncode == 0:
                # Parse cycle count from output
                for line in result.stdout.split('\n'):
                    if 'CYCLES:' in line:
                        cycles = int(line.split(':')[1].strip())
                        ind.fitness = cycles
                        ind.correct = True
                        return cycles

            # Failed correctness check
            ind.fitness = float('inf')
            ind.correct = False
            return float('inf')

        except subprocess.TimeoutExpired:
            ind.fitness = float('inf')
            ind.correct = False
            return float('inf')
        except Exception as e:
            print(f"Error evaluating: {e}")
            ind.fitness = float('inf')
            ind.correct = False
            return float('inf')
        finally:
            # Clean up
            if os.path.exists(kernel_path):
                os.remove(kernel_path)

    def select_parents(self) -> Tuple[Individual, Individual]:
        """Tournament selection for parents."""
        tournament_size = 3

        def tournament():
            candidates = random.sample(self.population, min(tournament_size, len(self.population)))
            return min(candidates, key=lambda x: x.fitness)

        return tournament(), tournament()

    def evolve_generation(self):
        """Create next generation through selection, crossover, and mutation."""
        # Evaluate all individuals
        for ind in self.population:
            if ind.fitness == float('inf'):
                self.evaluate_individual(ind)

        # Sort by fitness
        self.population.sort(key=lambda x: x.fitness)

        # Track best
        if self.population[0].correct and (
            self.best_ever is None or self.population[0].fitness < self.best_ever.fitness
        ):
            self.best_ever = copy.deepcopy(self.population[0])
            self.history.append((self.generation, self.best_ever.fitness))

        # Elitism - keep best 2
        new_population = self.population[:2]

        # Generate rest through crossover and mutation
        while len(new_population) < self.population_size:
            parent1, parent2 = self.select_parents()

            # Crossover
            child_genome = KernelGenome.crossover(parent1.genome, parent2.genome)

            # Mutation
            if random.random() < self.mutation_rate:
                child_genome = child_genome.mutate()

            child = Individual(genome=child_genome, generation=self.generation + 1)
            new_population.append(child)

        self.population = new_population
        self.generation += 1

    def run(self, num_generations: int = 100):
        """Run the genetic algorithm for specified generations."""
        print(f"Starting GA with population size {self.population_size}")
        print(f"Target: < 1400 cycles")
        print("=" * 60)

        self.initialize_population()

        for gen in range(num_generations):
            self.evolve_generation()

            best = min(self.population, key=lambda x: x.fitness)
            avg = sum(x.fitness for x in self.population if x.fitness < float('inf')) / max(1, len([x for x in self.population if x.fitness < float('inf')]))

            print(f"Gen {gen+1}: Best={best.fitness:.0f}, Avg={avg:.0f}, "
                  f"Correct={sum(1 for x in self.population if x.correct)}/{len(self.population)}, "
                  f"Genome={best.genome}")

            if best.fitness < 1400:
                print(f"\n{'='*60}")
                print(f"TARGET ACHIEVED! {best.fitness} cycles")
                print(f"Genome: {best.genome}")
                print(f"{'='*60}")
                break

        print(f"\nFinal best: {self.best_ever.fitness if self.best_ever else 'N/A'} cycles")
        return self.best_ever


def generate_kernel_code(genome: KernelGenome) -> str:
    """Generate kernel source code from genome."""
    # This is a template that will be filled in with genome parameters
    template = '''"""
GA-Generated Kernel Variant
Genome: {genome_str}
"""

import random
import unittest
import argparse
import sys
import os
from collections import defaultdict

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
    """Get read and write addresses for a slot."""
    reads: list[int] = []
    writes: list[int] = []

    if engine == "alu":
        _op, dest, a1, a2 = slot
        reads = [a1, a2]
        writes = [dest]
    elif engine == "valu":
        match slot:
            case ("vbroadcast", dest, src):
                reads = [src]
                writes = list(_vec_range(dest))
            case ("multiply_add", dest, a, b, c):
                reads = list(_vec_range(a)) + list(_vec_range(b)) + list(_vec_range(c))
                writes = list(_vec_range(dest))
            case (_op, dest, a1, a2):
                reads = list(_vec_range(a1)) + list(_vec_range(a2))
                writes = list(_vec_range(dest))
            case _:
                raise NotImplementedError(f"Unknown valu op {{slot}}")
    elif engine == "load":
        match slot:
            case ("load", dest, addr):
                reads = [addr]
                writes = [dest]
            case ("vload", dest, addr):
                reads = [addr]
                writes = list(_vec_range(dest))
            case ("const", dest, _val):
                writes = [dest]
            case ("load_offset", dest, addr, _lane):
                reads = [addr]
                writes = [dest]
            case _:
                raise NotImplementedError(f"Unknown load op {{slot}}")
    elif engine == "store":
        match slot:
            case ("store", addr, src):
                reads = [addr, src]
            case ("vstore", addr, src):
                reads = [addr] + list(_vec_range(src))
            case _:
                raise NotImplementedError(f"Unknown store op {{slot}}")
    elif engine == "flow":
        match slot:
            case ("select", dest, cond, a, b):
                reads = [cond, a, b]
                writes = [dest]
            case ("add_imm", dest, a, _imm):
                reads = [a]
                writes = [dest]
            case ("vselect", dest, cond, a, b):
                reads = list(_vec_range(cond)) + list(_vec_range(a)) + list(_vec_range(b))
                writes = list(_vec_range(dest))
            case ("halt",) | ("pause",) | ("trace_write", _) | ("jump", _) | ("jump_indirect", _) | ("cond_jump", _, _) | ("cond_jump_rel", _, _) | ("coreid", _):
                pass
            case _:
                raise NotImplementedError(f"Unknown flow op {{slot}}")

    return reads, writes


def _schedule_slots(slots: list[tuple[str, tuple]]) -> list[dict[str, list[tuple]]]:
    """Automatically schedule operations into VLIW bundles respecting dependencies."""
    cycles: list[dict[str, list[tuple]]] = []
    usage: list[dict[str, int]] = []
    ready_time: dict[int, int] = defaultdict(int)
    last_write: dict[int, int] = defaultdict(lambda: -1)
    last_read: dict[int, int] = defaultdict(lambda: -1)

    def ensure_cycle(cycle: int) -> None:
        while len(cycles) <= cycle:
            cycles.append({{}})
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


class KernelBuilder:
    """GA-Generated kernel with genome parameters."""

    # Genome parameters
    GROUP_SIZE = {group_size}
    TILE_COUNT = {tile_count}
    NUM_PRELOADED = {num_preloaded}
    SKIP_FINAL_BRANCH = {skip_final_branch}

    def __init__(self):
        self.slots: list[tuple[str, tuple]] = []
        self.scratch = {{}}
        self.scratch_debug = {{}}
        self.scratch_ptr = 0
        self.const_map = {{}}
        self.vconst_map = {{}}

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
        assert self.scratch_ptr <= SCRATCH_SIZE, f"Out of scratch space: {{self.scratch_ptr}}"
        return addr

    def alloc_vec(self, name=None):
        return self.alloc_scratch(name, VLEN)

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name or f"c_{{val}}")
            self.emit("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def scratch_vconst(self, val, name=None):
        if val not in self.vconst_map:
            scalar = self.scratch_const(val)
            addr = self.alloc_vec(name or f"v_{{val}}")
            self.emit("valu", ("vbroadcast", addr, scalar))
            self.vconst_map[val] = addr
        return self.vconst_map[val]

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        tmp_scalar = self.alloc_scratch("tmp_scalar")
        tmp_addr = self.alloc_scratch("tmp_addr")

        # Fast init - only load essential header values
        fast_init_vars = [
            ("n_nodes", 1),
            ("forest_values_p", 4),
            ("inp_indices_p", 5),
            ("inp_values_p", 6),
        ]
        for var_name, _ in fast_init_vars:
            self.alloc_scratch(var_name)
        for var_name, idx in fast_init_vars:
            self.emit("load", ("const", tmp_scalar, idx))
            self.emit("load", ("load", self.scratch[var_name], tmp_scalar))

        # Vector constants
        v_zero = self.scratch_vconst(0, "v_zero")
        v_one = self.scratch_vconst(1, "v_one")
        v_two = self.scratch_vconst(2, "v_two")
        v_three = self.scratch_vconst(3, "v_three")
        v_n_nodes = self.alloc_vec("v_n_nodes")
        self.emit("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]))

        v_forest_p = self.alloc_vec("v_forest_p")
        self.emit("valu", ("vbroadcast", v_forest_p, self.scratch["forest_values_p"]))

        # Hash constants
        FMA_MULTIPLIERS = {{0: 4097, 2: 33, 4: 9}}
        v_hash_consts = []
        v_hash_shifts = []
        v_fma_mult = {{}}

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            v_const = self.scratch_vconst(val1, f"v_hash_const_{{hi}}")
            v_hash_consts.append(v_const)
            if hi in FMA_MULTIPLIERS:
                v_fma_mult[hi] = self.scratch_vconst(FMA_MULTIPLIERS[hi], f"v_fma_mult_{{hi}}")
                v_hash_shifts.append(None)
            else:
                v_shift = self.scratch_vconst(val3, f"v_hash_shift_{{hi}}")
                v_hash_shifts.append(v_shift)

        # Preload tree nodes
        v_tree = []
        for i in range(self.NUM_PRELOADED):
            v_node = self.alloc_vec(f"v_tree_{{i}}")
            v_tree.append(v_node)
            self.emit("alu", ("+", tmp_addr, self.scratch["forest_values_p"], self.scratch_const(i)))
            self.emit("load", ("load", tmp_scalar, tmp_addr))
            self.emit("valu", ("vbroadcast", v_node, tmp_scalar))

        # Precompute differences for selection
        v_diff_1_2 = self.alloc_vec("v_diff_1_2")
        v_diff_3_4 = self.alloc_vec("v_diff_3_4")
        v_diff_5_6 = self.alloc_vec("v_diff_5_6")
        self.emit("valu", ("-", v_diff_1_2, v_tree[2], v_tree[1]))
        self.emit("valu", ("-", v_diff_3_4, v_tree[4], v_tree[3]))
        self.emit("valu", ("-", v_diff_5_6, v_tree[6], v_tree[5]))

        # Allocate per-desk vectors
        NUM_DESKS = 16 * self.TILE_COUNT
        desks = []
        for d in range(NUM_DESKS):
            desk = {{
                'idx': self.alloc_vec(f"v_idx_{{d}}"),
                'val': self.alloc_vec(f"v_val_{{d}}"),
                'node_val': self.alloc_vec(f"v_node_{{d}}"),
                'addr': self.alloc_vec(f"v_addr_{{d}}"),
                'tmp1': self.alloc_vec(f"v_tmp1_{{d}}"),
                'tmp2': self.alloc_vec(f"v_tmp2_{{d}}"),
            }}
            desks.append(desk)

        offset_regs = [self.alloc_scratch(f"off_{{d}}") for d in range(NUM_DESKS)]
        addr_tmp = [self.alloc_scratch(f"addr_tmp_{{i}}") for i in range(NUM_DESKS * 2)]

        self.emit("flow", ("pause",))

        # Helper functions
        def emit_hash_stages(desk_idx):
            d = desks[desk_idx]
            for hi in range(6):
                if hi in v_fma_mult:
                    self.emit("valu", ("multiply_add", d['val'], d['val'], v_fma_mult[hi], v_hash_consts[hi]))
                elif hi == 1:
                    self.emit("valu", ("^", d['tmp1'], d['val'], v_hash_consts[hi]))
                    self.emit("valu", (">>", d['tmp2'], d['val'], v_hash_shifts[hi]))
                    self.emit("valu", ("^", d['val'], d['tmp1'], d['tmp2']))
                elif hi == 3:
                    self.emit("valu", ("+", d['tmp1'], d['val'], v_hash_consts[hi]))
                    self.emit("valu", ("<<", d['tmp2'], d['val'], v_hash_shifts[hi]))
                    self.emit("valu", ("^", d['val'], d['tmp1'], d['tmp2']))
                elif hi == 5:
                    self.emit("valu", ("^", d['tmp1'], d['val'], v_hash_consts[hi]))
                    self.emit("valu", (">>", d['tmp2'], d['val'], v_hash_shifts[hi]))
                    self.emit("valu", ("^", d['val'], d['tmp1'], d['tmp2']))

        def emit_branch(desk_idx):
            d = desks[desk_idx]
            self.emit("valu", ("&", d['tmp1'], d['val'], v_one))
            self.emit("valu", ("multiply_add", d['idx'], d['idx'], v_two, v_one))
            self.emit("valu", ("+", d['idx'], d['idx'], d['tmp1']))

        def emit_bounds(desk_idx):
            d = desks[desk_idx]
            self.emit("valu", ("<", d['tmp1'], d['idx'], v_n_nodes))
            self.emit("valu", ("*", d['idx'], d['idx'], d['tmp1']))

        def emit_round_0(desk_idx):
            d = desks[desk_idx]
            self.emit("valu", ("^", d['val'], d['val'], v_tree[0]))
            emit_hash_stages(desk_idx)
            emit_branch(desk_idx)

        def emit_round_1(desk_idx):
            d = desks[desk_idx]
            self.emit("valu", ("-", d['tmp1'], d['idx'], v_one))
            self.emit("valu", ("multiply_add", d['node_val'], d['tmp1'], v_diff_1_2, v_tree[1]))
            self.emit("valu", ("^", d['val'], d['val'], d['node_val']))
            emit_hash_stages(desk_idx)
            emit_branch(desk_idx)

        def emit_round_2(desk_idx):
            d = desks[desk_idx]
            self.emit("valu", ("-", d['tmp1'], d['idx'], v_three))
            self.emit("valu", ("&", d['tmp2'], d['tmp1'], v_one))
            self.emit("valu", (">>", d['addr'], d['tmp1'], v_one))
            self.emit("valu", ("multiply_add", d['node_val'], d['tmp2'], v_diff_3_4, v_tree[3]))
            self.emit("valu", ("multiply_add", d['tmp1'], d['tmp2'], v_diff_5_6, v_tree[5]))
            self.emit("valu", ("-", d['tmp2'], d['tmp1'], d['node_val']))
            self.emit("valu", ("multiply_add", d['node_val'], d['addr'], d['tmp2'], d['node_val']))
            self.emit("valu", ("^", d['val'], d['val'], d['node_val']))
            emit_hash_stages(desk_idx)
            emit_branch(desk_idx)

        def emit_gather_round(desk_idx):
            d = desks[desk_idx]
            self.emit("valu", ("+", d['addr'], v_forest_p, d['idx']))
            for lane in range(VLEN):
                self.emit("load", ("load", d['node_val'] + lane, d['addr'] + lane))
            self.emit("valu", ("^", d['val'], d['val'], d['node_val']))
            emit_hash_stages(desk_idx)
            emit_branch(desk_idx)

        def emit_round_10(desk_idx):
            d = desks[desk_idx]
            self.emit("valu", ("+", d['addr'], v_forest_p, d['idx']))
            for lane in range(VLEN):
                self.emit("load", ("load", d['node_val'] + lane, d['addr'] + lane))
            self.emit("valu", ("^", d['val'], d['val'], d['node_val']))
            emit_hash_stages(desk_idx)
            emit_branch(desk_idx)
            emit_bounds(desk_idx)

        def emit_round_11(desk_idx):
            d = desks[desk_idx]
            self.emit("valu", ("^", d['val'], d['val'], v_tree[0]))
            emit_hash_stages(desk_idx)
            emit_branch(desk_idx)

        def emit_round_12(desk_idx):
            d = desks[desk_idx]
            self.emit("valu", ("-", d['tmp1'], d['idx'], v_one))
            self.emit("valu", ("multiply_add", d['node_val'], d['tmp1'], v_diff_1_2, v_tree[1]))
            self.emit("valu", ("^", d['val'], d['val'], d['node_val']))
            emit_hash_stages(desk_idx)
            emit_branch(desk_idx)

        def emit_round_13(desk_idx):
            d = desks[desk_idx]
            self.emit("valu", ("-", d['tmp1'], d['idx'], v_three))
            self.emit("valu", ("&", d['tmp2'], d['tmp1'], v_one))
            self.emit("valu", (">>", d['addr'], d['tmp1'], v_one))
            self.emit("valu", ("multiply_add", d['node_val'], d['tmp2'], v_diff_3_4, v_tree[3]))
            self.emit("valu", ("multiply_add", d['tmp1'], d['tmp2'], v_diff_5_6, v_tree[5]))
            self.emit("valu", ("-", d['tmp2'], d['tmp1'], d['node_val']))
            self.emit("valu", ("multiply_add", d['node_val'], d['addr'], d['tmp2'], d['node_val']))
            self.emit("valu", ("^", d['val'], d['val'], d['node_val']))
            emit_hash_stages(desk_idx)
            emit_branch(desk_idx)

        def emit_round_15_final(desk_idx):
            d = desks[desk_idx]
            self.emit("valu", ("+", d['addr'], v_forest_p, d['idx']))
            for lane in range(VLEN):
                self.emit("load", ("load", d['node_val'] + lane, d['addr'] + lane))
            self.emit("valu", ("^", d['val'], d['val'], d['node_val']))
            emit_hash_stages(desk_idx)
            if not self.SKIP_FINAL_BRANCH:
                emit_branch(desk_idx)

        def emit_tile(tile_idx):
            tile_offset = tile_idx * 16 * VLEN
            local_desks = list(range(tile_idx * 16, (tile_idx + 1) * 16))

            # Load offsets
            for i, d in enumerate(local_desks):
                self.emit("load", ("const", offset_regs[d], tile_offset + i * VLEN))

            # Compute addresses
            for d in local_desks:
                self.emit("alu", ("+", addr_tmp[d*2], self.scratch["inp_indices_p"], offset_regs[d]))
                self.emit("alu", ("+", addr_tmp[d*2+1], self.scratch["inp_values_p"], offset_regs[d]))

            # Load idx/val
            for d in local_desks:
                self.emit("load", ("vload", desks[d]['idx'], addr_tmp[d*2]))
                self.emit("load", ("vload", desks[d]['val'], addr_tmp[d*2+1]))

            # Process in groups
            GROUP_SIZE = self.GROUP_SIZE
            num_groups = (16 + GROUP_SIZE - 1) // GROUP_SIZE

            for g in range(num_groups):
                group_start = tile_idx * 16 + g * GROUP_SIZE
                group_end = min(group_start + GROUP_SIZE, tile_idx * 16 + 16)
                group_desks = list(range(group_start, group_end))

                for d in group_desks:
                    emit_round_0(d)
                for d in group_desks:
                    emit_round_1(d)
                for d in group_desks:
                    emit_round_2(d)
                for _rnd in range(3, 10):
                    for d in group_desks:
                        emit_gather_round(d)
                for d in group_desks:
                    emit_round_10(d)
                for d in group_desks:
                    emit_round_11(d)
                for d in group_desks:
                    emit_round_12(d)
                for d in group_desks:
                    emit_round_13(d)
                for d in group_desks:
                    emit_gather_round(d)
                for d in group_desks:
                    emit_round_15_final(d)

            # Store results
            for d in local_desks:
                self.emit("store", ("vstore", addr_tmp[d*2], desks[d]['idx']))
                self.emit("store", ("vstore", addr_tmp[d*2+1], desks[d]['val']))

        # Emit tiles
        for t in range(self.TILE_COUNT):
            emit_tile(t)

        # Schedule
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
                self.instrs.append({{"flow": [("pause",)]}})

        self.instrs.append({{"flow": [("pause",)]}})


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
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)

    value_trace = {{}}
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
            ), f"Incorrect result on round {{i}}"

    print("CYCLES: ", machine.cycle)
    return machine.cycle


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()

    if args.check:
        cycles = do_kernel_test(10, 16, 256, check=True)
        print(f"Correctness check PASSED! Cycles: {{cycles}}")
    else:
        do_kernel_test(10, 16, 256, trace=args.trace)
'''

    return template.format(
        genome_str=str(genome),
        group_size=genome.group_size,
        tile_count=genome.tile_count,
        num_preloaded=genome.num_preloaded,
        skip_final_branch=genome.skip_final_branch,
    )


if __name__ == "__main__":
    ga = GeneticAlgorithm(population_size=15, mutation_rate=0.4)
    best = ga.run(num_generations=100)
