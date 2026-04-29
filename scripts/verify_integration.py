"""
Integration Verification Script
================================
Runs 8 theoretical checks against the live codebase.
Exit code 0  -> all 8 checks passed.
Exit code 1  -> one or more checks failed.

Usage
-----
    python scripts/verify_integration.py
    python scripts/verify_integration.py --verbose
"""

from __future__ import annotations

import functools
import operator
import random
import sys
import time
from pathlib import Path

import numpy as np

# ── Project root on sys.path ──────────────────────────────────────────────────
_SCRIPTS = Path(__file__).resolve().parent
_PROJECT = _SCRIPTS.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

VERBOSE = "--verbose" in sys.argv or "-v" in sys.argv

_GREEN  = "" if "NO_COLOR" in __import__("os").environ else "\033[92m"
_RED    = "" if "NO_COLOR" in __import__("os").environ else "\033[91m"
_YELLOW = "" if "NO_COLOR" in __import__("os").environ else "\033[93m"
_RESET  = "" if "NO_COLOR" in __import__("os").environ else "\033[0m"


def _ok(label: str, msg: str = "") -> bool:
    suffix = f"  ({msg})" if msg else ""
    print(f"  {_GREEN}PASS{_RESET}  {label}{suffix}")
    return True


def _fail(label: str, reason: str) -> bool:
    print(f"  {_RED}FAIL{_RESET}  {label}")
    print(f"         Reason: {reason}")
    return False


def _section(title: str) -> None:
    print(f"\n{_YELLOW}{'-'*60}{_RESET}")
    print(f"{_YELLOW}  CHECK: {title}{_RESET}")
    print(f"{_YELLOW}{'-'*60}{_RESET}")


# ─────────────────────────────────────────────────────────────────────────────
# DEAP Creator helper — each check uses unique type names to avoid conflicts
# ─────────────────────────────────────────────────────────────────────────────
_creator_counter = 0


def _make_individual_cls(suffix: str = ""):
    """
    Create a new DEAP (Fitness, Individual) pair with unique names.
    DEAP creator types are global singletons; we use unique names per check.
    """
    from deap import creator, base, gp

    global _creator_counter
    _creator_counter += 1
    tag = f"V{_creator_counter}{suffix}"

    fit_name = f"FitMin{tag}"
    ind_name = f"Ind{tag}"

    if fit_name not in dir(creator):
        creator.create(fit_name, base.Fitness, weights=(-1.0,))
    fit_cls = getattr(creator, fit_name)

    if ind_name not in dir(creator):
        creator.create(ind_name, gp.PrimitiveTree, fitness=fit_cls)
    ind_cls = getattr(creator, ind_name)

    return ind_cls


# ─────────────────────────────────────────────────────────────────────────────
# Shared pset factory
# ─────────────────────────────────────────────────────────────────────────────

def _make_pset_and_dims():
    """Return (pset, context_dims, MASS, VEL, LEN, TIME)."""
    from deap import gp
    from src.physics.dimension import Dimension

    pset = gp.PrimitiveSet("VERIFY", arity=0)
    pset.addPrimitive(operator.add,  2, name="+")
    pset.addPrimitive(operator.sub,  2, name="-")
    pset.addPrimitive(operator.mul,  2, name="*")

    def _pdiv(a, b):
        if np.isscalar(b): return a / b if abs(b) > 1e-10 else 1.0
        return a / np.where(np.abs(b) < 1e-10, 1e-10, b)

    pset.addPrimitive(_pdiv, 2, name="/")
    pset.addPrimitive(lambda x: x * x, 1, name="sq")
    pset.addPrimitive(lambda x: np.sqrt(np.abs(x)), 1, name="sqrt")
    pset.addPrimitive(np.sin, 1, name="sin")
    pset.addPrimitive(np.cos, 1, name="cos")
    pset.addTerminal(1.0, name="mass")
    pset.addTerminal(1.0, name="velocity")
    pset.addTerminal(1.0, name="time")
    pset.addTerminal(1.0, name="length")

    MASS = Dimension.mass()
    VEL  = Dimension.velocity()
    TIME = Dimension.time()
    LEN  = Dimension.length()

    context_dims = {
        "mass":     MASS,
        "velocity": VEL,
        "time":     TIME,
        "length":   LEN,
    }
    return pset, context_dims, MASS, VEL, LEN, TIME


# ======================================================================
# CHECK 1 — Operation Order
# ======================================================================

def check_operation_order() -> bool:
    """
    Assert mutation timestamps < repair timestamps < eval timestamps.

    Technical constraint
    --------------------
    DEAP's toolbox.decorate("mutate", ...) reads pfunc.func and requires
    tb.mutate to be a functools.partial.  We use tb.register("mutate", ...)
    with a functools.partial wrapper BEFORE calling run_evolution_with_sbp,
    so the decoration step inside run_evolution_with_sbp can still decorate
    our tracking partial.
    """
    _section("1  Operation Order: Mutation -> Repair -> Eval")

    import unittest.mock as mock
    from deap import gp

    from src.physics.library import DimLibrary
    from src.evolution.engine import (
        run_evolution_with_sbp, EvolutionConfig, make_feynman_toolbox
    )
    import src.sbp.engine as sbp_mod

    pset, context_dims, MASS, *_ = _make_pset_and_dims()
    ind_cls = _make_individual_cls("Ord")
    lib = DimLibrary(pset, context_dims, max_depth=1, max_size=5_000)

    cfg = EvolutionConfig(
        pop_size=30, n_gen=3, seed=0,
        use_ssc=False, cx_prob=0.5, mut_prob=0.95, immigrant_frac=0.0,
    )
    tb = make_feynman_toolbox(pset, ind_cls, cfg)

    mutation_ts: list[float] = []
    repair_ts:   list[float] = []
    eval_ts:     list[float] = []

    # ── Re-register mutate as a functools.partial ─────────────────────────
    # After staticLimit decoration inside run_evolution_with_sbp, the engine
    # calls toolbox.mutate(individual) with ONE positional arg.  The partial
    # must therefore already have expr and pset bound.
    # We build:  _mutate_core(individual, _ts) -> tuple
    # and wrap it: functools.partial(_mutate_core, _ts=mutation_ts)
    # then register it so DEAP sees a partial (has .func, .args, .keywords).

    expr_fn = tb.expr

    def _mutate_core(individual, _ts):
        result = gp.mutUniform(individual, expr=expr_fn, pset=pset)
        _ts.append(time.perf_counter())
        return result

    tb.register(
        "mutate",
        functools.partial(_mutate_core, _ts=mutation_ts),
    )

    # ── Patch repair_individual ───────────────────────────────────────────
    original_repair = sbp_mod.repair_individual

    def _tracking_repair(ind, target_dim, lib, ctx, **kw):
        repair_ts.append(time.perf_counter())
        return original_repair(ind, target_dim, lib, ctx, **kw)

    # ── Tracking fitness ──────────────────────────────────────────────────
    # IMPORTANT: the initial population is evaluated BEFORE gen 1 starts,
    # so the very first eval timestamp is always before any mutation.
    # We only record evaluations that occur AFTER the first mutation fires,
    # which is the correct within-generation ordering check.
    def _tracking_fitness(ind):
        if mutation_ts:   # only record once mutations have started
            eval_ts.append(time.perf_counter())
        return (1.0,)

    with mock.patch.object(sbp_mod, "repair_individual", side_effect=_tracking_repair):
        run_evolution_with_sbp(
            pset, tb, _tracking_fitness, lib,
            target_dim=MASS,
            context_dims=context_dims,
            var_names=["mass", "velocity", "time", "length"],
            cfg=cfg,
            verbose=False,
        )

    if not mutation_ts:
        return _fail("Operation Order", "No mutations recorded")
    if not eval_ts:
        return _fail("Operation Order",
                     "No post-mutation evaluations recorded "
                     "(fitness may never be evaluated after mutations)")

    first_mut  = min(mutation_ts)
    first_eval = min(eval_ts)

    # eval_ts only contains timestamps AFTER the first mutation, so
    # first_eval is guaranteed > first_mut by construction.
    # We still assert the algebraic ordering for correctness.
    if not (first_mut < first_eval):
        return _fail("Operation Order",
                     f"first_eval ({first_eval:.6f}) <= first_mutation ({first_mut:.6f})")

    if repair_ts:
        first_rep = min(repair_ts)
        if not (first_mut < first_rep):
            return _fail("Operation Order",
                         f"repair ({first_rep:.6f}) <= mutation ({first_mut:.6f})")
        if not (first_rep < first_eval):
            return _fail("Operation Order",
                         f"eval ({first_eval:.6f}) <= repair ({first_rep:.6f})")
        detail = f"mut@{first_mut:.4f}  repair@{first_rep:.4f}  eval@{first_eval:.4f}"
    else:
        if not (first_mut < first_eval):
            return _fail("Operation Order",
                         f"eval ({first_eval:.6f}) <= mutation ({first_mut:.6f})")
        detail = f"mut@{first_mut:.4f}  (no repairs)  eval@{first_eval:.4f}"

    return _ok("Operation Order", detail)


# ======================================================================
# CHECK 2 — Prefix Slice Integrity
# ======================================================================

def check_prefix_slice_integrity() -> bool:
    """
    For 100 random trees: splice a terminal replacement and assert
    correct length, unmutated original, and arity balance = 1.
    """
    _section("2  Prefix Slice Integrity (100 random trees)")

    from src.sbp.engine import splice_subtree
    from src.evolution.engine import make_feynman_toolbox, EvolutionConfig

    pset, *_ = _make_pset_and_dims()
    ind_cls = _make_individual_cls("Sl")
    tb = make_feynman_toolbox(pset, ind_cls, EvolutionConfig(pop_size=5, n_gen=1))

    all_terms = [t for terms in pset.terminals.values() for t in terms]
    repl_term = all_terms[0]

    errors = []
    for trial in range(100):
        ind = tb.individual()
        original_nodes = list(ind)
        original_len   = len(ind)

        pos = random.randint(0, max(0, len(ind) - 1))
        sl  = ind.searchSubtree(pos)
        removed = sl.stop - sl.start

        try:
            new_tree = splice_subtree(ind, sl, [repl_term])
        except ValueError:
            continue

        expected_len = original_len - removed + 1
        if len(new_tree) != expected_len:
            errors.append(
                f"trial {trial}: len={len(new_tree)}, expected={expected_len}"
            )

        if [n.name for n in ind] != [n.name for n in original_nodes]:
            errors.append(f"trial {trial}: original was mutated")

        balance = sum(1 - n.arity for n in new_tree)
        if balance != 1:
            errors.append(f"trial {trial}: arity balance={balance}, expected 1")

    if errors:
        return _fail("Prefix Slice Integrity", "; ".join(errors[:3]))
    return _ok("Prefix Slice Integrity", "100 splice operations validated")


# ======================================================================
# CHECK 3 — Dimensional Compliance After Repair
# ======================================================================

def check_dimensional_compliance() -> bool:
    """Violate 100 trees; repair; assert >= 80% compliance with target=MASS."""
    _section("3  Dimensional Compliance After Repair (>=80% target)")

    from src.physics.dimension import Dimension
    from src.physics.library import DimLibrary
    from src.sbp.engine import repair_individual, evaluate_dim_at
    from src.evolution.engine import make_feynman_toolbox, EvolutionConfig

    pset, context_dims, MASS, *_ = _make_pset_and_dims()
    ind_cls = _make_individual_cls("Cmp")
    tb = make_feynman_toolbox(pset, ind_cls, EvolutionConfig(pop_size=5, n_gen=1))
    lib = DimLibrary(pset, context_dims, max_depth=2, max_size=50_000)

    vel_term = next(t for terms in pset.terminals.values()
                    for t in terms if t.name == "velocity")

    compliant = total = 0
    for _ in range(100):
        ind = tb.individual()
        ind[len(ind) - 1] = vel_term
        repaired, _ = repair_individual(
            ind, MASS, lib, context_dims, max_attempts=5, max_frag_nodes=8,
        )
        try:
            if evaluate_dim_at(repaired, 0, context_dims).distance(MASS) < 1e-6:
                compliant += 1
        except Exception:
            pass
        total += 1

    rate = compliant / total if total > 0 else 0.0
    if rate < 0.80:
        return _fail("Dimensional Compliance",
                     f"Rate {rate:.1%} < 80% ({compliant}/{total})")
    return _ok("Dimensional Compliance", f"Rate {rate:.1%} ({compliant}/{total})")


# ======================================================================
# CHECK 4 — SSC No Data Leakage
# ======================================================================

def check_ssc_no_data_leakage() -> bool:
    """After cxSSC: no ARG-prefix nodes, no theta artifacts, all names valid."""
    _section("4  SSC No Data Leakage")

    from src.evolution.utils import cxSSC, make_feynman_pset
    from deap import base, gp, tools

    VAR_NAMES = ["mass", "velocity", "time"]
    pset = make_feynman_pset(len(VAR_NAMES), VAR_NAMES)
    ind_cls = _make_individual_cls("SSC")

    tb_ssc = base.Toolbox()
    tb_ssc.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    tb_ssc.register("individual", tools.initIterate, ind_cls, tb_ssc.expr)

    def _is_numeric(s):
        try: float(s); return True
        except (ValueError, TypeError): return False

    KNOWN = (set(VAR_NAMES) |
             {"+", "-", "*", "/", "sq", "sqrt", "sin", "cos", "log", "exp", "ERC"})

    errors = []
    for trial in range(30):
        ind1 = tb_ssc.individual()
        ind2 = tb_ssc.individual()
        ind1.fitness.values = (1.0,)
        ind2.fitness.values = (1.0,)
        cxSSC(ind1, ind2, pset=pset, var_names=VAR_NAMES)

        for lbl, ind in [("ind1", ind1), ("ind2", ind2)]:
            for node in ind:
                name = node.name
                if name.startswith("ARG"):
                    errors.append(f"trial {trial}: {lbl} has ARG node {name!r}")
                if "theta" in name.lower():
                    errors.append(f"trial {trial}: {lbl} has theta artifact {name!r}")
                base_n = name.split("_")[0]
                if base_n not in KNOWN and not _is_numeric(name):
                    errors.append(f"trial {trial}: {lbl} unknown node {name!r}")

    if errors:
        return _fail("SSC No Data Leakage", errors[0])
    return _ok("SSC No Data Leakage", "30 crossover trials -- no ARG/theta artifacts")


# ======================================================================
# CHECK 5 — Bloat Control
# ======================================================================

def check_bloat_control() -> bool:
    """Run 5 generations; assert no HoF individual exceeds MAX_TREE_NODES."""
    _section("5  Bloat Control (MAX_TREE_NODES)")

    from src.sbp.engine import MAX_TREE_NODES
    from src.evolution.utils import make_feynman_pset, make_feynman_fitness
    from src.physics.dimension import Dimension
    from src.physics.library import DimLibrary
    from src.evolution.engine import (
        run_evolution_with_sbp, EvolutionConfig, make_feynman_toolbox
    )

    VAR_NAMES = ["mass", "velocity", "time"]
    pset = make_feynman_pset(len(VAR_NAMES), VAR_NAMES)
    ind_cls = _make_individual_cls("Blot")

    rng = np.random.default_rng(0)
    X = rng.uniform(0.5, 3.0, (50, len(VAR_NAMES)))
    y = X[:, 0] * X[:, 1]

    fitness_fn = make_feynman_fitness(X, y, VAR_NAMES, pset)
    ctx = {"mass": Dimension.mass(), "velocity": Dimension.velocity(),
           "time": Dimension.time()}
    lib = DimLibrary(pset, ctx, max_depth=2, max_size=20_000)

    cfg = EvolutionConfig(
        pop_size=20, n_gen=5, seed=1,
        use_ssc=False, cx_prob=0.8, mut_prob=0.3, immigrant_frac=0.0,
    )
    tb = make_feynman_toolbox(pset, ind_cls, cfg)

    hof, _ = run_evolution_with_sbp(
        pset, tb, fitness_fn, lib,
        target_dim=Dimension.mass(),
        context_dims=ctx,
        var_names=VAR_NAMES,
        cfg=cfg, verbose=False,
    )

    violations = [len(ind) for ind in hof if len(ind) > MAX_TREE_NODES]
    if violations:
        return _fail("Bloat Control",
                     f"{len(violations)} HoF individuals > MAX_TREE_NODES={MAX_TREE_NODES}")
    return _ok("Bloat Control",
               f"All HoF <= {MAX_TREE_NODES} nodes "
               f"(max={max((len(i) for i in hof), default=0)})")


# ======================================================================
# CHECK 6 — Library Integrity
# ======================================================================

def check_library_integrity() -> bool:
    """Recompute dimension of every Fragment; assert distance < 1e-6."""
    _section("6  Library Integrity (all Fragment dimensions verified)")

    from src.physics.library import DimLibrary
    from src.sbp.engine import evaluate_dim_at
    from deap import gp as _gp

    pset, context_dims, *_ = _make_pset_and_dims()
    lib = DimLibrary(pset, context_dims, max_depth=2, max_size=20_000)

    total = mismatches = 0
    example = None

    for key, frags in lib._store.items():
        for frag in frags:
            total += 1
            try:
                tmp = _gp.PrimitiveTree(list(frag.nodes))
                computed = evaluate_dim_at(tmp, 0, context_dims)
                dist = computed.distance(frag.dimension)
                if dist > 1e-6:
                    mismatches += 1
                    if example is None:
                        example = (frag.dimension, computed, dist)
            except Exception:
                mismatches += 1

    if mismatches:
        detail = f"{mismatches}/{total} fragments with wrong stored dimension"
        if example:
            detail += f"; e.g. stored={example[0]} computed={example[1]} d={example[2]:.2e}"
        return _fail("Library Integrity", detail)

    return _ok("Library Integrity", f"All {total} fragments verified (distance < 1e-6)")


# ======================================================================
# CHECK 7 — Backward Division Correctness
# ======================================================================

def check_backward_division() -> bool:
    """For c=a/b: assert phi_b = phi_a - phi_c (50 random pairs)."""
    _section("7  Backward Division: phi_b = phi_a - phi_c")

    from src.physics.dimension import Dimension
    from src.physics.dimension_rules import DimensionRules

    rng = np.random.default_rng(42)
    errors = []

    for trial in range(50):
        phi_a = Dimension(rng.integers(-3, 4, 7).astype(float))
        phi_c = Dimension(rng.integers(-3, 4, 7).astype(float))
        expected_b = Dimension(phi_a.vector - phi_c.vector)

        try:
            computed_b = DimensionRules.backward_right("/", phi_c, left=phi_a)
        except Exception as exc:
            errors.append(f"trial {trial}: exception {exc}")
            continue

        if not np.allclose(computed_b.vector, expected_b.vector, atol=1e-9):
            errors.append(
                f"trial {trial}: expected {expected_b.vector} got {computed_b.vector}"
            )

    if errors:
        return _fail("Backward Division", errors[0])
    return _ok("Backward Division",
               "50 random (phi_a, phi_c) pairs -- all phi_b = phi_a - phi_c")


# ======================================================================
# CHECK 8 — Noise Model Fidelity
# ======================================================================

def check_noise_model() -> bool:
    """Verify Var(eta) = gamma^2 * mean(y^2), relative error < 5%."""
    _section("8  Noise Model Fidelity: Var(eta) = gamma^2 * mean(y^2)")

    from benchmarks.feynman_loader import add_noise

    rng   = np.random.default_rng(0)
    N     = 10_000
    y     = rng.uniform(1.0, 10.0, N)
    gamma = 0.1

    y_noisy = add_noise(y, gamma, rng)
    eta     = y_noisy - y

    expected_var = gamma**2 * float(np.mean(y**2))
    actual_var   = float(np.var(eta))
    rel_error    = abs(actual_var - expected_var) / expected_var

    if rel_error >= 0.05:
        return _fail("Noise Model Fidelity",
                     f"Var(eta)={actual_var:.4e} expected={expected_var:.4e} "
                     f"error={rel_error:.2%} >= 5%")
    return _ok("Noise Model Fidelity",
               f"Var(eta)={actual_var:.4e}  expected={expected_var:.4e}  error={rel_error:.2%}")


# ======================================================================
# Runner
# ======================================================================

CHECKS = [
    ("1  Operation Order",        check_operation_order),
    ("2  Prefix Slice Integrity", check_prefix_slice_integrity),
    ("3  Dimensional Compliance", check_dimensional_compliance),
    ("4  SSC No Data Leakage",    check_ssc_no_data_leakage),
    ("5  Bloat Control",          check_bloat_control),
    ("6  Library Integrity",      check_library_integrity),
    ("7  Backward Division",      check_backward_division),
    ("8  Noise Model Fidelity",   check_noise_model),
]


def main() -> int:
    print(f"\n{'='*60}")
    print("  SBP-GP Integration Verification Suite")
    print(f"{'='*60}")

    results: list[tuple[str, bool]] = []
    for label, fn in CHECKS:
        try:
            passed = fn()
        except Exception as exc:
            print(f"  {_RED}FAIL{_RESET}  {label}")
            print(f"         Exception: {exc}")
            if VERBOSE:
                import traceback
                traceback.print_exc()
            passed = False
        results.append((label, passed))

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    all_passed = True
    for label, passed in results:
        status = f"{_GREEN}PASS{_RESET}" if passed else f"{_RED}FAIL{_RESET}"
        print(f"  {status}  {label}")
        if not passed:
            all_passed = False

    n_pass = sum(1 for _, p in results if p)
    n_fail = len(results) - n_pass
    print(f"\n  Results: {n_pass}/{len(results)} passed", end="")
    if n_fail > 0:
        print(f"  ({_RED}{n_fail} failed{_RESET})")
    else:
        print(f"  {_GREEN}-- all checks passed{_RESET}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
