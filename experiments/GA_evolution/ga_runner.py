"""
GA Evolution Runner - Direct parameter sweep with kernel variants
=================================================================

Instead of complex code generation, we directly modify parameters in the
existing kernel code and test different configurations.

Parameters to evolve:
- GROUP_SIZE: 1, 2, 3, 4, 5, 6, 8, 16
- TILE_COUNT: 1, 2
- NUM_PRELOADED: 3, 5, 7, 9, 11, 15
"""

import random
import copy
import os
import sys
import time
import subprocess
from dataclasses import dataclass
from typing import List, Tuple, Optional
import re

# Project directory
PROJECT_DIR = "/home/hestiasadmin/projects/original_performance_takehome"

@dataclass
class Genome:
    """Represents kernel parameters."""
    group_size: int = 4
    num_preloaded: int = 7
    skip_final_branch: bool = True
    double_buffer: bool = False

    def __str__(self):
        return f"GS{self.group_size}_P{self.num_preloaded}_SF{int(self.skip_final_branch)}_DB{int(self.double_buffer)}"

    def mutate(self) -> 'Genome':
        child = copy.deepcopy(self)
        gene = random.choice(['group_size', 'num_preloaded', 'skip_final_branch'])

        if gene == 'group_size':
            child.group_size = random.choice([1, 2, 3, 4, 5, 6, 8, 16])
        elif gene == 'num_preloaded':
            child.num_preloaded = random.choice([3, 5, 7, 9, 11, 15])
        elif gene == 'skip_final_branch':
            child.skip_final_branch = not child.skip_final_branch

        return child

    @staticmethod
    def crossover(p1: 'Genome', p2: 'Genome') -> 'Genome':
        return Genome(
            group_size=random.choice([p1.group_size, p2.group_size]),
            num_preloaded=random.choice([p1.num_preloaded, p2.num_preloaded]),
            skip_final_branch=random.choice([p1.skip_final_branch, p2.skip_final_branch]),
        )


@dataclass
class Individual:
    genome: Genome
    fitness: float = float('inf')
    correct: bool = False


def create_kernel_variant(genome: Genome, output_path: str):
    """Create a kernel variant by modifying the base kernel."""
    # Read base kernel
    base_path = os.path.join(PROJECT_DIR, "perf_takehome.py")
    with open(base_path, 'r') as f:
        code = f.read()

    # Add sys.path for imports
    path_fix = f'''import sys
sys.path.insert(0, "{PROJECT_DIR}")
'''
    code = code.replace('import random', path_fix + 'import random')

    # Modify GROUP_SIZE
    code = re.sub(
        r'GROUP_SIZE\s*=\s*\d+',
        f'GROUP_SIZE = {genome.group_size}',
        code
    )

    # Modify NUM_PRELOADED
    code = re.sub(
        r'NUM_PRELOADED\s*=\s*\d+',
        f'NUM_PRELOADED = {genome.num_preloaded}',
        code
    )

    # Write modified kernel
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

    except subprocess.TimeoutExpired:
        return float('inf'), False
    except Exception as e:
        print(f"Error: {e}")
        return float('inf'), False


def run_ga(population_size: int = 10, num_generations: int = 100):
    """Run genetic algorithm."""
    print("=" * 60)
    print("GA Evolution for VLIW Kernel Optimization")
    print(f"Target: < 1400 cycles")
    print(f"Population: {population_size}, Generations: {num_generations}")
    print("=" * 60)

    # Create output directory
    output_dir = os.path.join(PROJECT_DIR, "experiments/GA_evolution/variants")
    os.makedirs(output_dir, exist_ok=True)

    # Initialize population
    population = [
        Individual(Genome(group_size=4, num_preloaded=7, skip_final_branch=True)),  # Baseline
        Individual(Genome(group_size=2, num_preloaded=7, skip_final_branch=True)),
        Individual(Genome(group_size=3, num_preloaded=7, skip_final_branch=True)),
        Individual(Genome(group_size=5, num_preloaded=7, skip_final_branch=True)),
        Individual(Genome(group_size=6, num_preloaded=7, skip_final_branch=True)),
        Individual(Genome(group_size=8, num_preloaded=7, skip_final_branch=True)),
        Individual(Genome(group_size=4, num_preloaded=3, skip_final_branch=True)),
        Individual(Genome(group_size=4, num_preloaded=11, skip_final_branch=True)),
        Individual(Genome(group_size=4, num_preloaded=15, skip_final_branch=True)),
        Individual(Genome(group_size=1, num_preloaded=7, skip_final_branch=True)),
    ]

    # Fill remaining with mutations
    while len(population) < population_size:
        base = random.choice(population[:3])
        mutant = base.genome.mutate()
        population.append(Individual(mutant))

    best_ever = None
    history = []

    for gen in range(num_generations):
        # Evaluate population
        for ind in population:
            if ind.fitness == float('inf'):
                kernel_path = os.path.join(output_dir, f"kernel_{ind.genome}.py")
                create_kernel_variant(ind.genome, kernel_path)
                ind.fitness, ind.correct = evaluate_kernel(kernel_path)

        # Sort by fitness
        population.sort(key=lambda x: x.fitness)

        # Track best
        if population[0].correct and (best_ever is None or population[0].fitness < best_ever.fitness):
            best_ever = copy.deepcopy(population[0])
            history.append((gen, best_ever.fitness))

        # Report
        correct_count = sum(1 for x in population if x.correct)
        avg = sum(x.fitness for x in population if x.fitness < float('inf')) / max(1, correct_count)
        print(f"Gen {gen+1:3d}: Best={population[0].fitness:6.0f}, Avg={avg:6.0f}, "
              f"Correct={correct_count}/{len(population)}, Best genome: {population[0].genome}")

        # Check target
        if population[0].fitness < 1400:
            print(f"\n{'='*60}")
            print(f"TARGET ACHIEVED: {population[0].fitness} cycles!")
            print(f"Genome: {population[0].genome}")
            print(f"{'='*60}")
            break

        # Create next generation (elitism + crossover + mutation)
        new_pop = population[:2]  # Keep top 2

        while len(new_pop) < population_size:
            # Tournament selection
            candidates = random.sample(population, min(3, len(population)))
            p1 = min(candidates, key=lambda x: x.fitness)
            candidates = random.sample(population, min(3, len(population)))
            p2 = min(candidates, key=lambda x: x.fitness)

            # Crossover
            child_genome = Genome.crossover(p1.genome, p2.genome)

            # Mutation
            if random.random() < 0.4:
                child_genome = child_genome.mutate()

            new_pop.append(Individual(child_genome))

        population = new_pop

    print(f"\nFinal best: {best_ever.fitness if best_ever else 'N/A'} cycles")
    if best_ever:
        print(f"Best genome: {best_ever.genome}")

    # Write evolution log
    log_path = os.path.join(PROJECT_DIR, "research_swarm/GA_EVOLUTION_LOG.md")
    with open(log_path, 'w') as f:
        f.write("# GA Evolution Log\n\n")
        f.write(f"**Target:** < 1400 cycles\n")
        f.write(f"**Best achieved:** {best_ever.fitness if best_ever else 'N/A'} cycles\n")
        f.write(f"**Best genome:** {best_ever.genome if best_ever else 'N/A'}\n\n")
        f.write("## Evolution History\n\n")
        f.write("| Generation | Best Fitness |\n")
        f.write("|------------|-------------|\n")
        for gen, fitness in history:
            f.write(f"| {gen} | {fitness} |\n")

    return best_ever


if __name__ == "__main__":
    run_ga(population_size=15, num_generations=100)
