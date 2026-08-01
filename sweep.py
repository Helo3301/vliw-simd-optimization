"""Batch hypothesis search over the kernel's configuration space.

  python3.11 sweep.py [N]        # default 1000 configurations

Every configuration tested is written to sweep_results.json, so successive
batches explore new territory instead of re-running the same points. Screening
is build-only (~0.74s); a configuration is simulated and correctness-checked
only if it beats the incumbent, because the check is the expensive part.

The space is deliberately weighted toward the scheduler. The kernel sits 38
cycles above a bound derived from its own dependency graph, and that residual
is queue-dry windows -- so the priority function's shape is the parameter most
likely to matter, and it is sampled continuously rather than as a fixed menu.
"""

import contextlib
import importlib
import io
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DB = os.path.join(HERE, "sweep_results.json")

# name -> list of values to sample from. All defaults reproduce 1138 cycles.
SPACE = {
    # Per-context spill placement. The global flags were the whole story until
    # ALU_ADD flipped; these let each context choose independently.
    "PTH_ALU_BIT_G":   ["0", "1"],
    "PTH_ALU_BIT_F":   ["0", "1"],
    "PTH_ALU_XOR_G":   ["0", "1"],
    "PTH_ALU_XOR_F":   ["0", "1"],
    "PTH_ALU_ADD_G":   ["0", "1"],
    "PTH_ALU_ADD_F":   ["0", "1"],
    "PTH_ALU_ADD_15":  ["0", "1"],
    "PTH_HASH_LAYOUT": ["0", "1"],
    "PTH_EMIT_CHUNK":  ["1", "2", "3", "4"],
    "PTH_DL_THRESH":   [str(v) for v in (4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 64)],
    "PTH_HASH_ORDER":  [str(v) for v in range(24)],
    "PTH_R10_ORDER":   [str(v) for v in range(-1, 24)],
    "PTH_GROUP_ORDER": [str(v) for v in range(24)],
    "PTH_LANE_ORDER":  ["0", "1", "2"],
    "PTH_PRIO_TL":     [str(v) for v in (-1, 0, 4, 8, 12, 16, 20, 24, 28, 32, 48, 64)],
    "PTH_PRIO_A":      ["0", "1", "2", "3"],
    "PTH_PRIO_B":      ["0", "1", "2", "3"],
}
BASELINE = {k: v[0] for k, v in SPACE.items()}
BASELINE.update({"PTH_ALU_BIT_G": "1", "PTH_ALU_BIT_F": "1", "PTH_ALU_XOR_G": "1",
                 "PTH_ALU_XOR_F": "1", "PTH_ALU_ADD_G": "0", "PTH_ALU_ADD_F": "0",
                 "PTH_ALU_ADD_15": "0", "PTH_HASH_LAYOUT": "0",
                 "PTH_EMIT_CHUNK": "2", "PTH_DL_THRESH": "24",
                 "PTH_R10_ORDER": "-1", "PTH_PRIO_TL": "-1"})


def key(cfg):
    return "|".join(f"{k}={cfg[k]}" for k in sorted(cfg))


def load_db():
    try:
        with open(DB) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def build_cycles(cfg):
    """Build only -- no simulation. Returns cycles, or None if the config is
    infeasible (scratch overflow etc.)."""
    for k, v in cfg.items():
        os.environ[k] = v
    os.environ["PTH_NO_SA"] = "1"
    try:
        import perf_takehome as P
        importlib.reload(P)
        with contextlib.redirect_stdout(io.StringIO()):
            kb = P.KernelBuilder()
            kb.build_kernel(10, 2047, 256, 16)
        return len(kb.instrs)
    except Exception:
        return None


def verify(cfg, seeds=5):
    """Full correctness against the reference on BOTH output arrays."""
    for k, v in cfg.items():
        os.environ[k] = v
    os.environ["PTH_NO_SA"] = "1"
    import perf_takehome as P
    importlib.reload(P)
    from problem import (Tree, Input, build_mem_image, reference_kernel2,
                         Machine, N_CORES)
    with contextlib.redirect_stdout(io.StringIO()):
        kb = P.KernelBuilder()
        kb.build_kernel(10, 2047, 256, 16)
    for seed in range(seeds):
        random.seed(seed)
        forest = Tree.generate(10)
        inp = Input.generate(forest, 256, 16)
        mem = build_mem_image(forest, inp)
        m = Machine(mem, kb.instrs, kb.debug_info(), n_cores=N_CORES)
        m.enable_pause = False
        m.enable_debug = False
        m.run()
        for ref in reference_kernel2(mem):
            pass
        ip, vp, nn = ref[5], ref[6], len(inp.indices)
        if (m.mem[vp:vp + nn] != ref[vp:vp + nn]
                or m.mem[ip:ip + nn] != ref[ip:ip + nn]):
            return False
    return True


def main(n):
    db = load_db()
    rng = random.Random()
    print(f"sweep: {n} configurations, {len(db)} already on record")

    if key(BASELINE) not in db:
        db[key(BASELINE)] = build_cycles(BASELINE)
    best_cycles = min((v for v in db.values() if v), default=None)
    best_key = min((k for k, v in db.items() if v == best_cycles), default=None)
    print(f"incumbent: {best_cycles} cycles")

    t0, tested, skipped, infeasible, improvements = time.time(), 0, 0, 0, []
    while tested + skipped < n:
        cfg = {k: rng.choice(v) for k, v in SPACE.items()}
        kk = key(cfg)
        if kk in db:
            skipped += 1
            continue
        c = build_cycles(cfg)
        db[kk] = c
        tested += 1
        if c is None:
            infeasible += 1
        elif best_cycles is None or c < best_cycles:
            ok = verify(cfg)
            print(f"  {c} cycles  correct={ok}  {kk}", flush=True)
            if ok:
                improvements.append((c, kk))
                best_cycles, best_key = c, kk
        if tested % 100 == 0:
            print(f"  ...{tested} built, {time.time()-t0:.0f}s, best {best_cycles}",
                  flush=True)

    with open(DB, "w") as f:
        json.dump(db, f)

    vals = sorted(v for v in db.values() if v)
    print(f"\nbatch: {tested} built ({infeasible} infeasible), {skipped} already known, "
          f"{time.time()-t0:.0f}s")
    print(f"corpus: {len(db)} configurations")
    print(f"best:   {best_cycles} cycles")
    print(f"        {best_key}")
    if vals:
        print(f"spread: min {vals[0]}  p10 {vals[len(vals)//10]}  median "
              f"{vals[len(vals)//2]}  max {vals[-1]}")
    if improvements:
        print("IMPROVEMENTS (verified on both arrays):")
        for c, k in improvements:
            print(f"  {c}  {k}")
    else:
        print("no verified improvement this batch")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1000)
