"""
Systematic Parameter Sweep for GA Evolution
============================================

Test all combinations of key parameters systematically.
"""

import sys
sys.path.insert(0, "/home/hestiasadmin/projects/original_performance_takehome")

import os
import subprocess
import re
import itertools
from dataclasses import dataclass

PROJECT_DIR = "/home/hestiasadmin/projects/original_performance_takehome"


def create_variant(group_size, num_preloaded, output_path):
    """Create a kernel variant."""
    base_path = os.path.join(PROJECT_DIR, "perf_takehome.py")
    with open(base_path, 'r') as f:
        code = f.read()

    path_fix = f'''import sys
sys.path.insert(0, "{PROJECT_DIR}")
'''
    code = code.replace('import random', path_fix + 'import random')
    code = re.sub(r'GROUP_SIZE\s*=\s*\d+', f'GROUP_SIZE = {group_size}', code)
    code = re.sub(r'NUM_PRELOADED\s*=\s*\d+', f'NUM_PRELOADED = {num_preloaded}', code)

    with open(output_path, 'w') as f:
        f.write(code)


def test_variant(kernel_path):
    """Test a variant and return cycles."""
    try:
        result = subprocess.run(
            ['python3.11', kernel_path, '--check'],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=PROJECT_DIR
        )
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'CYCLES:' in line:
                    return int(line.split(':')[1].strip()), True
        return float('inf'), False
    except:
        return float('inf'), False


def run_sweep():
    """Run systematic parameter sweep."""
    print("=" * 70)
    print("Systematic Parameter Sweep")
    print("=" * 70)

    output_dir = os.path.join(PROJECT_DIR, "experiments/GA_evolution/sweep_variants")
    os.makedirs(output_dir, exist_ok=True)

    # Parameter space
    group_sizes = [1, 2, 3, 4, 5, 6, 8, 16]
    preload_counts = [3, 5, 7, 9, 11, 15]

    results = []

    for gs, np in itertools.product(group_sizes, preload_counts):
        kernel_path = os.path.join(output_dir, f"kernel_gs{gs}_np{np}.py")
        create_variant(gs, np, kernel_path)
        cycles, correct = test_variant(kernel_path)

        status = f"{cycles:6.0f}" if correct else "FAIL"
        print(f"GROUP_SIZE={gs:2d}, NUM_PRELOADED={np:2d}: {status}")

        if correct:
            results.append((cycles, gs, np))

    # Sort by cycles
    results.sort()

    print("\n" + "=" * 70)
    print("TOP 10 CONFIGURATIONS:")
    print("=" * 70)
    for i, (cycles, gs, np) in enumerate(results[:10]):
        print(f"{i+1}. {cycles} cycles: GROUP_SIZE={gs}, NUM_PRELOADED={np}")

    # Write results
    log_path = os.path.join(PROJECT_DIR, "research_swarm/GA_SWEEP_RESULTS.md")
    with open(log_path, 'w') as f:
        f.write("# Systematic Parameter Sweep Results\n\n")
        f.write("## Full Results\n\n")
        f.write("| Rank | Cycles | GROUP_SIZE | NUM_PRELOADED |\n")
        f.write("|------|--------|------------|---------------|\n")
        for i, (cycles, gs, np) in enumerate(results):
            f.write(f"| {i+1} | {cycles} | {gs} | {np} |\n")

    return results[0] if results else None


if __name__ == "__main__":
    best = run_sweep()
    if best:
        print(f"\nBest: {best[0]} cycles with GROUP_SIZE={best[1]}, NUM_PRELOADED={best[2]}")
