"""
Advanced GA Evolution - Structural Kernel Mutations
====================================================

This explores deeper structural changes beyond simple parameter tuning:
1. Hash stage reordering
2. Operation fusion patterns
3. Different interleaving strategies
4. Alternative branch computation methods
5. Load/compute overlap patterns
"""

import random
import copy
import os
import sys
import subprocess
import re
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

PROJECT_DIR = "/home/hestiasadmin/projects/original_performance_takehome"

@dataclass
class AdvancedGenome:
    """Advanced genome with structural mutation capabilities."""
    group_size: int = 1  # Best from initial GA
    num_preloaded: int = 7
    skip_final_branch: bool = True

    # NEW: Structural parameters
    hash_stage_order: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    round_processing_order: str = "sequential"  # sequential, reversed, interleaved
    desk_interleave_pattern: str = "group"  # group, round_robin, pairs
    hash_emission_pattern: str = "sequential"  # sequential, interleaved, double_pump
    branch_method: str = "fma"  # fma, shift_add, direct
    gather_prefetch: bool = False
    fuse_xor_hash: bool = False  # Try to fuse XOR into first hash stage
    parallel_bounds_check: bool = False  # Do bounds check in parallel with hash

    def __str__(self):
        return (f"GS{self.group_size}_P{self.num_preloaded}_"
                f"HS{''.join(map(str,self.hash_stage_order))}_"
                f"R{self.round_processing_order[:3]}_"
                f"D{self.desk_interleave_pattern[:3]}_"
                f"H{self.hash_emission_pattern[:3]}_"
                f"B{self.branch_method[:3]}")

    def mutate(self) -> 'AdvancedGenome':
        child = copy.deepcopy(self)
        gene = random.choice([
            'group_size', 'num_preloaded', 'hash_stage_order',
            'round_processing_order', 'desk_interleave_pattern',
            'hash_emission_pattern', 'branch_method', 'fuse_xor_hash',
            'parallel_bounds_check'
        ])

        if gene == 'group_size':
            child.group_size = random.choice([1, 2, 3, 4, 5, 6, 8, 16])
        elif gene == 'num_preloaded':
            child.num_preloaded = random.choice([3, 5, 7, 9, 11, 15])
        elif gene == 'hash_stage_order':
            # Swap two adjacent stages (preserve FMA constraints)
            order = child.hash_stage_order.copy()
            # Only swap non-FMA stages (1, 3, 5) among themselves
            non_fma = [1, 3, 5]
            if random.random() < 0.5:
                random.shuffle(non_fma)
                order[1], order[3], order[5] = non_fma[0], non_fma[1], non_fma[2]
            child.hash_stage_order = order
        elif gene == 'round_processing_order':
            child.round_processing_order = random.choice([
                'sequential', 'reversed', 'interleaved', 'even_odd'
            ])
        elif gene == 'desk_interleave_pattern':
            child.desk_interleave_pattern = random.choice([
                'group', 'round_robin', 'pairs', 'quad', 'strided'
            ])
        elif gene == 'hash_emission_pattern':
            child.hash_emission_pattern = random.choice([
                'sequential', 'interleaved', 'double_pump', 'triple_pump'
            ])
        elif gene == 'branch_method':
            child.branch_method = random.choice(['fma', 'shift_add', 'direct'])
        elif gene == 'fuse_xor_hash':
            child.fuse_xor_hash = not child.fuse_xor_hash
        elif gene == 'parallel_bounds_check':
            child.parallel_bounds_check = not child.parallel_bounds_check

        return child

    @staticmethod
    def crossover(p1: 'AdvancedGenome', p2: 'AdvancedGenome') -> 'AdvancedGenome':
        return AdvancedGenome(
            group_size=random.choice([p1.group_size, p2.group_size]),
            num_preloaded=random.choice([p1.num_preloaded, p2.num_preloaded]),
            hash_stage_order=random.choice([p1.hash_stage_order, p2.hash_stage_order]).copy(),
            round_processing_order=random.choice([p1.round_processing_order, p2.round_processing_order]),
            desk_interleave_pattern=random.choice([p1.desk_interleave_pattern, p2.desk_interleave_pattern]),
            hash_emission_pattern=random.choice([p1.hash_emission_pattern, p2.hash_emission_pattern]),
            branch_method=random.choice([p1.branch_method, p2.branch_method]),
            fuse_xor_hash=random.choice([p1.fuse_xor_hash, p2.fuse_xor_hash]),
            parallel_bounds_check=random.choice([p1.parallel_bounds_check, p2.parallel_bounds_check]),
        )


def create_advanced_kernel(genome: AdvancedGenome, output_path: str):
    """Generate kernel with structural modifications based on genome."""
    base_path = os.path.join(PROJECT_DIR, "perf_takehome.py")
    with open(base_path, 'r') as f:
        code = f.read()

    # Add sys.path
    path_fix = f'''import sys
sys.path.insert(0, "{PROJECT_DIR}")
'''
    code = code.replace('import random', path_fix + 'import random')

    # Basic parameter modifications
    code = re.sub(r'GROUP_SIZE\s*=\s*\d+', f'GROUP_SIZE = {genome.group_size}', code)
    code = re.sub(r'NUM_PRELOADED\s*=\s*\d+', f'NUM_PRELOADED = {genome.num_preloaded}', code)

    # Desk interleave pattern modifications
    if genome.desk_interleave_pattern == 'round_robin':
        # Process desks in round-robin order across groups
        code = code.replace(
            'for g in range(num_full_groups):',
            '''# Round-robin interleaving
            all_groups = [list(range(NUM_DESKS))]  # Single group with all desks
            for g in range(1):  # Single iteration'''
        )
    elif genome.desk_interleave_pattern == 'pairs':
        # Process desks in pairs
        code = re.sub(r'GROUP_SIZE\s*=\s*\d+', 'GROUP_SIZE = 2', code)
    elif genome.desk_interleave_pattern == 'quad':
        code = re.sub(r'GROUP_SIZE\s*=\s*\d+', 'GROUP_SIZE = 4', code)

    # Hash emission pattern (more complex - would need deeper code changes)
    # For now, leave as placeholder for future exploration

    # Write the modified kernel
    with open(output_path, 'w') as f:
        f.write(code)


def evaluate_kernel(kernel_path: str, timeout: int = 60) -> Tuple[float, bool]:
    """Run kernel and return (cycles, correct)."""
    try:
        result = subprocess.run(
            ['python3.11', kernel_path, '--check'],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=PROJECT_DIR
        )
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'CYCLES:' in line:
                    cycles = int(line.split(':')[1].strip())
                    return cycles, True
        return float('inf'), False
    except:
        return float('inf'), False


def run_advanced_ga(pop_size: int = 20, generations: int = 100):
    """Run advanced genetic algorithm."""
    print("=" * 70)
    print("Advanced GA Evolution - Structural Kernel Mutations")
    print(f"Target: < 1400 cycles")
    print("=" * 70)

    output_dir = os.path.join(PROJECT_DIR, "experiments/GA_evolution/advanced_variants")
    os.makedirs(output_dir, exist_ok=True)

    # Initialize with diverse population
    population = [
        # Start with known good configurations
        AdvancedGenome(group_size=1, num_preloaded=7),
        AdvancedGenome(group_size=4, num_preloaded=7),
        AdvancedGenome(group_size=2, num_preloaded=7),
        # Variations
        AdvancedGenome(group_size=1, desk_interleave_pattern='pairs'),
        AdvancedGenome(group_size=1, desk_interleave_pattern='quad'),
        AdvancedGenome(group_size=1, hash_emission_pattern='double_pump'),
        AdvancedGenome(group_size=1, branch_method='shift_add'),
        AdvancedGenome(group_size=1, fuse_xor_hash=True),
    ]

    # Fill with mutations
    while len(population) < pop_size:
        base = random.choice(population[:3])
        mutant = base.mutate()
        for _ in range(random.randint(1, 3)):
            mutant = mutant.mutate()
        population.append(mutant)

    best_ever = None
    history = []

    for gen in range(generations):
        # Evaluate
        for ind in population:
            if not hasattr(ind, 'fitness') or ind.fitness is None:
                ind.fitness = float('inf')
                ind.correct = False

            if ind.fitness == float('inf'):
                kernel_path = os.path.join(output_dir, f"kernel_{hash(str(ind)) % 100000}.py")
                create_advanced_kernel(ind, kernel_path)
                ind.fitness, ind.correct = evaluate_kernel(kernel_path)

        # Sort
        population.sort(key=lambda x: x.fitness)

        # Track best
        if population[0].correct and (best_ever is None or population[0].fitness < best_ever.fitness):
            best_ever = copy.deepcopy(population[0])
            history.append((gen, best_ever.fitness))
            print(f"  *** NEW BEST: {best_ever.fitness} cycles ***")

        # Report
        correct = sum(1 for x in population if getattr(x, 'correct', False))
        avg = sum(x.fitness for x in population if x.fitness < float('inf')) / max(1, correct)
        print(f"Gen {gen+1:3d}: Best={population[0].fitness:6.0f}, Avg={avg:6.0f}, "
              f"Correct={correct}/{len(population)}, Genome: {population[0]}")

        if population[0].fitness < 1400:
            print(f"\n{'='*70}")
            print(f"TARGET ACHIEVED: {population[0].fitness} cycles!")
            print(f"{'='*70}")
            break

        # Next generation
        new_pop = population[:3]  # Elitism
        while len(new_pop) < pop_size:
            candidates = random.sample(population, min(3, len(population)))
            p1 = min(candidates, key=lambda x: x.fitness)
            candidates = random.sample(population, min(3, len(population)))
            p2 = min(candidates, key=lambda x: x.fitness)

            child = AdvancedGenome.crossover(p1, p2)
            if random.random() < 0.5:
                child = child.mutate()

            child.fitness = None
            child.correct = False
            new_pop.append(child)

        population = new_pop

    print(f"\nFinal best: {best_ever.fitness if best_ever else 'N/A'} cycles")

    # Update log
    log_path = os.path.join(PROJECT_DIR, "research_swarm/GA_EVOLUTION_LOG.md")
    with open(log_path, 'a') as f:
        f.write(f"\n## Advanced GA Results\n\n")
        f.write(f"**Best:** {best_ever.fitness if best_ever else 'N/A'} cycles\n")
        f.write(f"**Genome:** {best_ever if best_ever else 'N/A'}\n\n")
        f.write("### History\n")
        for gen, fit in history:
            f.write(f"- Gen {gen}: {fit} cycles\n")

    return best_ever


if __name__ == "__main__":
    run_advanced_ga(pop_size=20, generations=100)
