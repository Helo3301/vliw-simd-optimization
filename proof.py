"""Machine-checked justification for this kernel's result.

Run:  python3.11 proof.py

Every claim below is labelled with its epistemic status:

  PROVED    machine-checked here, over all inputs, by z3 or by exhaustive
            computation over the actual emitted program
  VERIFIED  checked empirically on sampled inputs -- strong evidence, not proof
  ASSUMED   stated explicitly because the argument rests on it and it is NOT
            established here

The point of the file is that the ASSUMED list is short, visible, and separate
from everything else. Earlier writeups of this work quoted "floors" that were
really conditionals on a chosen op mix, and they moved every time the mix did.
"""

import os
import subprocess
import sys
import random
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import problem
from problem import (
    HASH_STAGES, SLOT_LIMITS, VLEN, N_CORES,
    Tree, Input, build_mem_image, reference_kernel2, Machine, myhash,
)
import perf_takehome as P

M32 = 2 ** 32
results = []


def record(status, claim, detail=""):
    results.append((status, claim, detail))
    print(f"  [{status:8s}] {claim}")
    if detail:
        print(f"             {detail}")


# ---------------------------------------------------------------- claim 1
def claim_algebraic_identities():
    """The two rewrites that let the hash reach 10 VALU ops, over ALL 2^32
    inputs, via z3 -- not sampled."""
    print("\n1. Algebraic rewrites the kernel depends on")
    try:
        import z3
    except ImportError:
        record("FAILED", "z3 unavailable -- identities UNCHECKED",
               "a proof script that passes when its prover is missing is worse "
               "than no proof script; this fails closed")
        return

    a = z3.BitVec("a", 32)
    C2, C3 = HASH_STAGES[2][1], HASH_STAGES[3][1]
    SH3 = HASH_STAGES[3][4]
    C5, SH5 = HASH_STAGES[5][1], HASH_STAGES[5][4]

    # stage 2+3 fusion: s2 = a*33 + C2; stage3 = (s2 + C3) ^ (s2 << 9).
    # Both operands are affine in a, so each is one multiply_add and s2 is
    # never materialised. This is what took the hash from 12 ops to 11.
    s2 = a * 33 + C2
    lhs = (s2 + C3) ^ (s2 << SH3)
    rhs = (a * 33 + ((C2 + C3) % M32)) ^ (a * ((33 << SH3) % M32) + ((C2 << SH3) % M32))
    s = z3.Solver()
    s.add(lhs != rhs)
    ok = s.check() == z3.unsat
    record("PROVED" if ok else "FAILED",
           "stage 2+3 fuse to 3 VALU ops (12 -> 11 per hash)",
           "forall a in [0,2^32): (s2+C3)^(s2<<9) == FMA(a,33,C2+C3) ^ FMA(a,16896,C2<<9)")

    # C5 fold: stage 5 is (s4^C5)^(s4>>16); the next round XORs a node into it.
    # Folding C5 into the node constant leaves w = s4 ^ (s4>>16), 2 ops not 3,
    # and complements the branch bit (C5 bit 0 is 1). NOTE: the shipped kernel
    # runs the 11-op hash. This fold was built, verified, and measured at zero
    # cycles -- VALU is not the binding engine -- so it was not kept. The
    # identity is proved here because the 10-op figure appears in the bounds.
    s4, node = z3.BitVec("s4", 32), z3.BitVec("node", 32)
    lhs2 = ((s4 ^ C5) ^ z3.LShR(s4, SH5)) ^ node
    rhs2 = (s4 ^ z3.LShR(s4, SH5)) ^ (C5 ^ node)
    s = z3.Solver(); s.add(lhs2 != rhs2)
    ok2 = s.check() == z3.unsat
    s = z3.Solver()
    s.add(z3.Extract(0, 0, (s4 ^ C5) ^ z3.LShR(s4, SH5))
          != ~z3.Extract(0, 0, s4 ^ z3.LShR(s4, SH5)))
    ok3 = s.check() == z3.unsat
    record("PROVED" if (ok2 and ok3) else "FAILED",
           "C5 fold would be exact (11 -> 10), though it is not shipped",
           "forall s4,node: ((s4^C5)^(s4>>16))^node == (s4^(s4>>16))^(C5^node); bit0 inverts")


# ---------------------------------------------------------------- claim 2
def claim_stage_minimality():
    """Can hash stage 1 be done in fewer than 3 VALU ops? Exhaustive search
    over all 1- and 2-op straight-line programs on this ISA with ARBITRARY
    32-bit constants, each checked by z3 over all inputs."""
    print("\n2. Per-stage minimality (bounded exhaustive search)")
    try:
        import z3
    except ImportError:
        record("FAILED", "z3 unavailable -- minimality UNCHECKED")
        return

    C1, SH1 = HASH_STAGES[1][1], HASH_STAGES[1][4]

    def target(x):
        return (x ^ C1) ^ z3.LShR(x, SH1)

    def binops():
        return {
            "+":  lambda x, y: x + y,
            "-":  lambda x, y: x - y,
            "*":  lambda x, y: x * y,
            "^":  lambda x, y: x ^ y,
            "&":  lambda x, y: x & y,
            "|":  lambda x, y: x | y,
            "<<": lambda x, y: x << y,
            ">>": lambda x, y: z3.LShR(x, y),
        }

    a = z3.BitVec("a", 32)
    k1, k2, k3 = z3.BitVec("k1", 32), z3.BitVec("k2", 32), z3.BitVec("k3", 32)
    found = None
    pending = []   # a timeout is NOT evidence of non-existence; retry these

    def query(expr, ms=4000, retry=True):
        s = z3.Solver(); s.set("timeout", ms)
        s.add(z3.ForAll([a], expr == target(a)))
        r = s.check()
        if r == z3.unknown and retry:
            pending.append(expr)
        return r == z3.sat

    # 1-op programs: op(a,k) and op(k,a), plus multiply_add(a,k1,k2)
    cands = []
    for nm, f in binops().items():
        cands.append((f"{nm}(a,k)", f(a, k1)))
        cands.append((f"{nm}(k,a)", f(k1, a)))
    cands.append(("fma(a,k1,k2)", a * k1 + k2))

    for nm, expr in cands:
        if query(expr):
            found = ("1 op", nm)
            break

    tested = len(cands)
    if found is None:
        # 2-op programs: t = op1(a,k1) or op1(k1,a) or fma; then combine any two
        # of {a, t, k2}
        firsts = []
        for nm, f in binops().items():
            firsts.append((f"{nm}(a,k1)", f(a, k1)))
            firsts.append((f"{nm}(k1,a)", f(k1, a)))
        firsts.append(("fma(a,k1,k3)", a * k1 + k3))   # arbitrary addend, not just a*k
        for n1, t in firsts:
            pool = [("a", a), ("t", t), ("k2", k2)]
            ops2 = dict(binops())
            for nm, f in ops2.items():
                for xn, x in pool:
                    for yn, y in pool:
                        tested += 1
                        if query(f(x, y)):
                            found = ("2 ops", f"{n1}; {nm}({xn},{yn})")
                            break
                    if found: break
                if found: break
            if not found:
                # multiply_add as the SECOND op, over every operand triple
                for xn, x in pool:
                    for yn, y in pool:
                        for zn, z_ in pool:
                            tested += 1
                            if query(x * y + z_):
                                found = ("2 ops", f"{n1}; fma({xn},{yn},{zn})")
                                break
                        if found: break
                    if found: break
                if found: break
            if found: break

    # second pass: give every timed-out shape a much larger budget before
    # conceding it is unresolved
    unknown = 0
    if found is None and pending:
        for expr in pending:
            s2 = z3.Solver(); s2.set("timeout", 120000)
            s2.add(z3.ForAll([a], expr == target(a)))
            r = s2.check()
            if r == z3.sat:
                found = ("2 ops", "found on retry")
                break
            if r == z3.unknown:
                unknown += 1

    if found:
        record("FAILED", f"stage 1 needs >=3 VALU ops -- found a {found[0]} program",
               found[1])
    elif unknown:
        # A solver timeout is not a proof of non-existence. Say so.
        record("PARTIAL", "hash stage 1 needs >=3 VALU ops",
               f"{tested - unknown}/{tested} candidate shapes refuted over all "
               f"32-bit constants; {unknown} timed out and are UNRESOLVED, so "
               f"this is not a proof")
    else:
        record("PROVED", "hash stage 1 needs >=3 VALU ops",
               f"no 1- or 2-op program over this ISA reproduces (a^C1)^(a>>19) "
               f"for all a; all {tested} candidate shapes refuted by z3, each "
               f"closed over all 32-bit constants")
    record("ASSUMED", "the 10-op hash is globally minimal",
           "minimality is established per stage, over the stage decomposition. "
           "A whole-hash superoptimisation is not attempted.")


# ---------------------------------------------------------------- claim 3
def claim_select_lower_bound():
    print("\n3. Cost of any 16-way node selection")
    record("ARGUED", "selecting 1 of 16 values needs >=15 binary combines",
           "a binary tree with 16 leaves has 15 internal nodes; every op on "
           "this ISA combines at most 2 candidate values (multiply_add takes 3 "
           "inputs but is affine, so with a 0/1 multiplier it still picks "
           "between 2). Bounds EVERY selection scheme, including unthought-of ones.")


# ---------------------------------------------------------------- claim 3b
def claim_hash_bound():
    print("\n3b. Hash work as a lower bound (corrected)")
    ops = 512 * 10
    valu_only = -(-ops // 6)
    best = min(max(x / 6, (ops - x) * 8 / 12) for x in range(0, ops + 1))
    record("ARGUED", f"hash work alone forces >= {round(best)} cycles, not {valu_only}",
           f"{ops} vector-ops. An earlier writeup quoted {valu_only} = {ops}/6, which "
           f"silently assumed VALU placement. The same work runs on the ALU at 8 "
           f"slots/op, so the real bound minimises max(x/6, ({ops}-x)*8/12) = "
           f"{round(best)}. Depends on the 10-op figure, which is only ASSUMED below.")


# ---------------------------------------------------------------- claim 4
def build():
    kb = P.KernelBuilder()
    kb.build_kernel(10, 2047, 256, 16)
    return kb


def claim_schedule_bound(kb):
    """Lower bound on cycles for THIS op set under ANY schedule, from the
    dependency graph alone."""
    print("\n4. Schedule lower bound for this op set (any scheduler)")
    slots = P._strip_dead([(e, s) for e, s in kb.slots
                           if not (e == "flow" and s == ("pause",))])
    n = len(slots)
    rw = [P._slot_rw(e, s) for e, s in slots]

    def compute(raw_only):
        wmap, rmap, est = {}, defaultdict(list), [0] * n
        for i, (reads, writes) in enumerate(rw):
            t = 0
            for x in reads:
                if x in wmap:
                    t = max(t, est[wmap[x]] + 1)          # RAW: a true dependence
            if not raw_only:
                # WAR/WAW exist only because values were given physical scratch
                # addresses before scheduling. Register renaming removes them.
                for x in writes:
                    if x in wmap:
                        t = max(t, est[wmap[x]] + 1)      # WAW
                    for r in rmap.get(x, ()):
                        t = max(t, est[r])                # WAR may share a cycle
            est[i] = t
            for x in reads:
                rmap[x].append(i)
            for x in writes:
                wmap[x] = i; rmap[x] = []
        L = sorted(est[i] for i in range(n) if slots[i][0] == "load")
        N = len(L)
        k = max(range(N), key=lambda j: L[j] + -(-(N - j) // 2))
        core = L[k] + -(-(N - k) // 2)
        g = [i for i, (e, s) in enumerate(slots) if e == "load" and s[0] == "load"]
        tail = max(est) + 1 - max(est[i] for i in g)
        return core + tail, core, tail, k, L[k], N

    phys, core, tail, k, ek, N = compute(False)
    renamed = compute(True)[0]
    achieved = len(kb.instrs)
    record("PROVED", f"no schedule of these {n} ops, AS ALLOCATED, beats {phys} cycles",
           f"max_k(est_k + ceil((N-k)/2)) = {core} at k={k} (est={ek}, N={N} loads) "
           f"+ {tail}-cycle tail. Scope: this op set AND this assignment of values "
           f"to scratch addresses -- not a bound on the problem.")
    record("PROVED", f"with perfect register renaming the same op set bounds at {renamed}",
           f"recomputed with every WAR/WAW edge deleted, which renaming can always "
           f"achieve. The anti-dependency artifact in the bound above is worth "
           f"{phys - renamed} cycles, so scheduling-vs-allocation is NOT where the "
           f"gap to faster published designs lives -- the op set is.")
    record("VERIFIED", f"achieved {achieved} cycles -- {achieved - phys} above the as-allocated bound",
           f"{100 * phys / achieved:.1f}% of the optimum for this op set and allocation")
    return achieved


# ---------------------------------------------------------------- claim 5
def claim_schedule_legal(kb):
    print("\n5. Emitted schedule respects the machine")
    bad = []
    for c, instr in enumerate(kb.instrs):
        for eng, sl in instr.items():
            if eng == "debug":
                continue
            if len(sl) > SLOT_LIMITS[eng]:
                bad.append((c, eng, len(sl)))
    record("PROVED" if not bad else "FAILED",
           "no instruction bundle exceeds its engine's slot limit",
           f"checked {len(kb.instrs)} bundles against {SLOT_LIMITS}")


# ---------------------------------------------------------------- claim 6
def claim_correct(kb, seeds=20):
    print(f"\n6. Correctness against reference_kernel2 ({seeds} seeds, BOTH arrays)")
    bad = 0
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
        if m.mem[vp:vp + nn] != ref[vp:vp + nn] or m.mem[ip:ip + nn] != ref[ip:ip + nn]:
            bad += 1
    record("VERIFIED" if bad == 0 else "FAILED",
           f"{seeds - bad}/{seeds} seeds match on inp_values AND inp_indices",
           "submission_tests.py compares only inp_values; this compares both, "
           "which is how the inherited kernel's 256 wrong indices went unnoticed")


# ---------------------------------------------------------------- claim 7
# SHA-256 of Anthropic's files at upstream commit 5452f74bd977807ac2e74f3d29432b9df6f25197.
# Pinned as a static manifest on purpose: comparing against a git remote makes the
# check depend on how the clone happens to be named, and in a fork whose main has
# absorbed this work it degenerates into self-comparison.
UPSTREAM_COMMIT = "5452f74bd977807ac2e74f3d29432b9df6f25197"
UPSTREAM_SHA256 = {
    "problem.py":                "fadb0f0858e2259f5759077a5544b9906dad3ceee80d37b4f0aa77da730c93c9",
    "tests/submission_tests.py": "11c57cc999da93acb41201191073cd657ddffa87635359b3157c6e177c18ea0a",
    "tests/frozen_problem.py":   "fadb0f0858e2259f5759077a5544b9906dad3ceee80d37b4f0aa77da730c93c9",
}


def claim_integrity():
    import hashlib
    print("\n7. Submission integrity (static manifest, not a remote name)")
    here = os.path.dirname(os.path.abspath(__file__))
    bad = []
    for rel, want in sorted(UPSTREAM_SHA256.items()):
        path = os.path.join(here, rel)
        try:
            got = hashlib.sha256(open(path, "rb").read()).hexdigest()
        except OSError:
            bad.append(f"{rel}: missing"); continue
        if got != want:
            bad.append(f"{rel}: {got[:12]} != {want[:12]}")
    record("PROVED" if not bad else "FAILED",
           f"tests/ and problem.py match upstream {UPSTREAM_COMMIT[:12]} by SHA-256",
           "; ".join(bad) if bad else
           f"{len(UPSTREAM_SHA256)} files pinned by content hash")


# ---------------------------------------------------------------- assumptions
def assumptions():
    print("\n8. What this does NOT establish")
    record("ASSUMED", "gathers cannot be replaced by contiguous vloads",
           "a vload delivers tree[B+j] to lane j, so it serves a lane only "
           "under exact alignment idx_j == B+j. At levels >=5 the indices are "
           "hash-driven and independent across lanes, so a straight-line "
           "program cannot arrange it. NOT proved here -- this is the load "
           "side of the bound and the place a real attack would aim.")
    record("ASSUMED", "no cheaper mechanism exists for delivering a tree value to a lane",
           "four were enumerated and priced (scalar load, vselect cascade, "
           "vbroadcast, scatter-then-vload). Exhaustion over imagination, not proof.")


if __name__ == "__main__":
    print("=" * 78)
    print("Proof obligations for the VLIW SIMD take-home kernel")
    print("=" * 78)
    claim_algebraic_identities()
    claim_stage_minimality()
    claim_select_lower_bound()
    claim_hash_bound()
    kb = build()
    claim_schedule_bound(kb)
    claim_schedule_legal(kb)
    claim_correct(kb)
    claim_integrity()
    assumptions()
    print("\n" + "=" * 78)
    tally = defaultdict(int)
    for st, _, _ in results:
        tally[st] += 1
    print("  ".join(f"{k}: {v}" for k, v in sorted(tally.items())))
    print("=" * 78)
    sys.exit(1 if tally.get("FAILED") else 0)
