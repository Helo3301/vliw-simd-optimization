# VLIW SIMD Optimization: 112.7x Speedup

**Final result: 1,311 cycles** — from a 147,734-cycle baseline.

This repository contains my optimized solution to [Anthropic's VLIW SIMD performance take-home challenge](https://github.com/anthropics/original_performance_takehome). The kernel performs parallel tree traversal with hash computation on a custom VLIW architecture.

## Results

| Metric | Value |
|--------|-------|
| Final cycles | **1,311** |
| Baseline | 147,734 |
| Speedup | **112.7x** |
| VALU utilization | 98.5% |
| Submission tests | 9/9 passing |

### Benchmark Comparison

| Threshold | Cycles | My Speedup vs. |
|-----------|--------|----------------|
| Naive baseline | 147,734 | 112.7x |
| Opus 4.5 improved harness | 1,363 | 1.04x |

## Key Techniques

### Algorithmic Innovations
- **Level-3 Tree Fusion**: All 256 batch elements start at the same tree root. Preload `tree[0]` through `tree[14]` and use vselect cascades instead of expensive gather operations for rounds 0-3 and 11-14.
- **Address-Tracking Branch**: Track scatter addresses directly (`addr = 2*addr + (1-fp) + bit`) instead of maintaining indices.
- **Deferred Index Computation**: Collect 4 bits across fused rounds and compute the final address once via FMA chain.
- **16-Desk Interleaving**: Pipeline parallelism across 2 tiles of 8 desks each.

### Scheduler Revolution
The biggest improvement (77 cycles, 5.5%) came from replacing the greedy list scheduler:

1. **Dependency graph construction** — Build DAG of ~10,000 ops
2. **Downstream VALU analysis** — Prioritize ops that feed long VALU chains
3. **Priority-based topological sort** — Schedule non-VALU ops that feed VALU before VALU ops themselves
4. **Simulated annealing** — Discover non-obvious reorderings via block reversals

## Files

| File | Description |
|------|-------------|
| `perf_takehome.py` | Optimized kernel (1,311 cycles) |
| `OPTIMIZATION_REPORT.md` | Detailed technical writeup |
| `problem.py` | ISA simulator (from Anthropic, unmodified) |
| `tests/` | Submission tests (from Anthropic, unmodified) |

## Reproduction

```bash
# Verify tests are untouched
git diff origin/main tests/

# Run submission tests (first run takes ~2-6 min due to SA warmup)
python3.11 tests/submission_tests.py

# Quick correctness check
python3.11 perf_takehome.py --check
```

**Note:** Python 3.11+ required (`match` syntax in problem.py).

## Deep Dive

For the full technical writeup with code snippets, tables, and lessons learned, see:
- [OPTIMIZATION_REPORT.md](./OPTIMIZATION_REPORT.md) in this repo
- [Blog post on hestiascreations.com](https://hestiascreations.com/projects/vliw)

## License

The optimized kernel and documentation are my original work. The ISA simulator (`problem.py`) and test suite are from [Anthropic's original repository](https://github.com/anthropics/original_performance_takehome).

---

*Built with [Claude Code](https://claude.ai/code) — January 2026*
