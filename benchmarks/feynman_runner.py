"""
Feynman Benchmark Runner
=========================
Orchestrates large-scale comparative trials of the SBP-GP framework
against a baseline GP (no repair) using the Feynman SR dataset.

Protocol (Reissmann et al. 2025, Section 4.1 / SRBench)
---------------------------------------------------------
For each (equation, noise_level, trial):
  1. Load data with load_feynman_equation (same seed → paired test).
  2. Build pset + toolbox + DimLibrary.
  3. Run baseline GP (use_ssc=False, no SBP repair).
  4. Run SBP-GP (use_ssc=True, SBP repair enabled) with the same seed.
  5. Record R², complexity, solution rate for each run.

Statistical Analysis
---------------------
scipy.stats.wilcoxon paired test on R² and complexity improvements
across all trials.  P-values are printed in the final report.

Usage
-----
    from benchmarks.feynman_runner import run_benchmark
    df = run_benchmark(
        equations=["I.12.1", "I.14.3"],
        noise_levels=[0.0, 0.01, 0.1],
        n_trials=10,
        n_gen=50,
        pop_size=300,
    )
    df.to_csv("results.csv", index=False)
"""

from __future__ import annotations

import copy
import logging
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Ensure project root is on sys.path when runner is invoked directly
_PROJECT = Path(__file__).resolve().parent.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from deap import creator, base, gp, tools

from benchmarks.feynman_loader import load_feynman_equation, FeynmanEquation
from benchmarks.metrics import compute_trial_metrics

from src.physics.library import DimLibrary
from src.evolution.utils import make_feynman_pset, make_feynman_fitness, MAX_HEIGHT
from src.evolution.engine import (
    EvolutionConfig,
    run_evolution_with_sbp,
    make_feynman_toolbox,
)

logger = logging.getLogger(__name__)


# ======================================================================
# Creator registration (idempotent)
# ======================================================================

def _ensure_creators():
    if "FitnessMinBench" not in dir(creator):
        creator.create("FitnessMinBench", base.Fitness, weights=(-1.0,))
    if "IndividualBench" not in dir(creator):
        creator.create("IndividualBench", gp.PrimitiveTree,
                       fitness=creator.FitnessMinBench)


# ======================================================================
# Single-trial runner
# ======================================================================

def _run_single_trial(
    eq: FeynmanEquation,
    cfg: EvolutionConfig,
    seed: int,
    lib_cache: dict,
    verbose: bool = False,
) -> dict:
    """
    Execute one GP trial for a single equation.

    Parameters
    ----------
    eq        : FeynmanEquation  — pre-loaded data + metadata
    cfg       : EvolutionConfig  — fully configured (use_ssc, sbp_attempts
                                   already set by the caller's ablation mode)
    seed      : int              — RNG seed for this trial
    lib_cache : dict             — keyed by filename; caches DimLibrary
    verbose   : bool             — if True, engine prints per-gen stats live

    Returns
    -------
    dict with keys: r2, complexity, solved, fitness, best_found,
                    wall_time_s, gen_rows
    """
    _ensure_creators()

    pset       = make_feynman_pset(len(eq.var_names), eq.var_names)
    fitness_fn = make_feynman_fitness(
        eq.X_train, eq.y_train, eq.var_names, pset,
        target_dim=eq.target_dim,
        context_dims=eq.context_dims,
        lambda_penalty=0.1,
        optimize_constants=True,
    )

    # ── DimLibrary ─────────────────────────────────────────────────────────
    # Skip the expensive DimLibrary build entirely for modes where SBP repair
    # is disabled (sbp_attempts == 0).  The library is only used by
    # repair_individual, which is patched to a no-op in those modes.
    if cfg.sbp_attempts > 0:
        if eq.filename not in lib_cache:
            lib_cache[eq.filename] = DimLibrary(
                pset, eq.context_dims,
                max_depth=2,
                max_size=100_000,
            )
        lib = lib_cache[eq.filename]
    else:
        lib = None   # never accessed — repair_individual is patched out

    trial_cfg      = copy.copy(cfg)
    trial_cfg.seed = seed
    # NOTE: use_ssc and sbp_attempts are already set by the ablation config;
    #       do NOT override them here.

    tb = make_feynman_toolbox(pset, creator.IndividualBench, trial_cfg)

    t0 = time.perf_counter()

    target_dim = eq.target_dim

    # If sbp_attempts == 0 patch repair to a guaranteed no-op.
    if trial_cfg.sbp_attempts == 0:
        import unittest.mock as mock
        from src.sbp import engine as sbp_engine
        noop = lambda ind, *a, **kw: (ind, False)
        with mock.patch.object(sbp_engine, "repair_individual", new=noop):
            hof, eng_log = run_evolution_with_sbp(
                pset, tb, fitness_fn, lib,
                target_dim=target_dim,
                context_dims=eq.context_dims,
                var_names=eq.var_names,
                cfg=trial_cfg,
                verbose=verbose,
            )
    else:
        hof, eng_log = run_evolution_with_sbp(
            pset, tb, fitness_fn, lib,
            target_dim=target_dim,
            context_dims=eq.context_dims,
            var_names=eq.var_names,
            cfg=trial_cfg,
            verbose=verbose,
        )

    wall_time = time.perf_counter() - t0

    # ── Build per-generation rows (post-hoc equation string + r²) ────────────
    from benchmarks.metrics import deap_to_sympy, r2_score
    from src.evolution.utils import eval_tree

    var_ctx  = {name: eq.X_test[:, i] for i, name in enumerate(eq.var_names)}
    gen_rows: list[dict] = []
    for g, best_ind, fit_min, fit_avg in zip(
        eng_log["gen"],
        eng_log["best_ind"],
        eng_log["min"],
        eng_log["avg"],
    ):
        if best_ind is None:
            continue
        # r² on test set (cheap — pure numpy)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y_pred = eval_tree(best_ind, var_ctx)
            if np.isscalar(y_pred):
                y_pred = np.full_like(eq.y_test, float(y_pred))
            else:
                y_pred = np.asarray(y_pred, dtype=float)
            gen_r2 = r2_score(eq.y_test, y_pred)
        except Exception:
            gen_r2 = float("-inf")
        # equation string (no costly sympy.simplify — just structure)
        sym_expr = deap_to_sympy(best_ind, eq.var_names)
        gen_eq   = str(sym_expr) if sym_expr is not None else str(best_ind)
        gen_rows.append({
            "gen":      g,
            "fitness":  fit_min,
            "avg_fit":  fit_avg,
            "r2":       gen_r2,
            "equation": gen_eq,
        })

    best = hof[0] if len(hof) > 0 else None
    if best is None:
        return {
            "r2": float("-inf"), "complexity": 10_000, "solved": False,
            "fitness": float("inf"), "best_found": "", "wall_time_s": wall_time,
            "gen_rows": gen_rows,
        }

    metrics = compute_trial_metrics(
        best, pset, eq.var_names,
        eq.X_test, eq.y_test, eq.formula,
    )
    metrics["wall_time_s"] = wall_time
    metrics["fitness"]   = best.fitness.values[0] if best.fitness.valid else float("inf")
    metrics["gen_rows"]  = gen_rows
    return metrics


# ======================================================================
# Main benchmark orchestrator
# ======================================================================

def run_benchmark(
    equations: list[str],
    noise_levels: list[float] = (0.0, 0.01, 0.1),
    n_trials: int = 10,
    n_gen: int = 50,
    pop_size: int = 300,
    seed_base: int = 0,
    max_rows: int = 10_000,
    sbp_attempts: int = 5,
    sbp_max_frag: int = 10,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run full comparative benchmark: Baseline GP vs. SBP-GP.

    The same seeds are used for both conditions to enable paired
    Wilcoxon statistical testing.

    Parameters
    ----------
    equations : list[str]
        Feynman equation filenames to benchmark (e.g. ["I.12.1"]).
    noise_levels : list[float]
        Noise amplitudes γ to test (Equation 3).
    n_trials : int
        Number of independent trials per (equation, noise, condition).
    n_gen : int
        Generations per trial.
    pop_size : int
        Population size per trial.
    seed_base : int
        Base seed; trial i uses seed_base + i.
    max_rows : int
        Maximum rows loaded per dataset file.
    sbp_attempts, sbp_max_frag : int
        SBP repair hyper-parameters.
    verbose : bool
        Print per-trial progress.

    Returns
    -------
    pd.DataFrame
        One row per (equation, noise_level, trial, condition) with
        columns: equation, noise, trial, condition, r2, complexity,
                 solved, wall_time_s.
    """
    _ensure_creators()

    cfg = EvolutionConfig(
        pop_size=pop_size,
        n_gen=n_gen,
        use_ssc=True,
        cx_prob=0.7,
        mut_prob=0.2,
        sbp_attempts=sbp_attempts,
        sbp_max_frag=sbp_max_frag,
        immigrant_frac=0.05,
    )

    rows = []
    lib_cache: dict = {}

    for eq_name in equations:
        for noise in noise_levels:
            # Load data ONCE per (equation, noise) combination;
            # seed=42 is always the data split seed (independent of trial seed)
            try:
                eq = load_feynman_equation(
                    eq_name,
                    noise_level=noise,
                    seed=42,
                    max_rows=max_rows,
                )
            except Exception as exc:
                logger.warning("Skipping %s (noise=%.2f): %s", eq_name, noise, exc)
                continue

            for trial in range(n_trials):
                trial_seed = seed_base + trial

                for condition, use_sbp in [("baseline", False), ("sbp_gp", True)]:
                    if verbose:
                        print(f"  [{eq_name}] noise={noise:.2f} | trial={trial:02d} | "
                              f"{condition} ... ", end="", flush=True)
                    try:
                        m = _run_single_trial(
                            eq,
                            cfg=EvolutionConfig(
                                pop_size=cfg.pop_size,
                                n_gen=cfg.n_gen,
                                hof_size=cfg.hof_size,
                                cx_prob=cfg.cx_prob,
                                mut_prob=cfg.mut_prob,
                                min_depth=cfg.min_depth,
                                max_depth=cfg.max_depth,
                                tournament_size=cfg.tournament_size,
                                immigrant_frac=cfg.immigrant_frac,
                                use_ssc=use_sbp,
                                sbp_attempts=cfg.sbp_attempts if use_sbp else 0,
                                sbp_max_frag=cfg.sbp_max_frag if use_sbp else 0,
                            ),
                            seed=trial_seed,
                            lib_cache=lib_cache,
                        )
                        row = {
                            "equation":    eq_name,
                            "noise":       noise,
                            "trial":       trial,
                            "condition":   condition,
                            "fitness":     m.get("fitness", float("inf")),
                            "r2":          m["r2"],
                            "complexity":  m["complexity"],
                            "solved":      m["solved"],
                            "best_found":  m.get("best_found", ""),
                            "wall_time_s": m["wall_time_s"],
                        }
                        if verbose:
                            eq_str = m.get('best_found', '')
                            print(f"fit={m.get('fitness', float('inf')):.4e}  r²={m['r2']:.3f}  cplx={m['complexity']}  "
                                  f"solved={m['solved']}  eq={eq_str}")
                    except Exception as exc:
                        logger.warning("Trial failed: %s", exc, exc_info=True)
                        row = {
                            "equation":    eq_name,
                            "noise":       noise,
                            "trial":       trial,
                            "condition":   condition,
                            "fitness":     float("inf"),
                            "r2":          float("-inf"),
                            "complexity":  10_000,
                            "solved":      False,
                            "best_found":  "",
                            "wall_time_s": 0.0,
                        }
                        if verbose:
                            print(f"ERROR: {exc}")

                    rows.append(row)

    df = pd.DataFrame(rows)
    return df


# ======================================================================
# Ablation-aware single-mode experiment runner
# ======================================================================

def run_experiment(
    equations: list[str],
    noise_levels: list[float] = (0.0, 0.01, 0.1),
    n_trials: int = 10,
    seed_base: int = 0,
    max_rows: int = 10_000,
    verbose: bool = True,
    config: EvolutionConfig | None = None,
) -> pd.DataFrame:
    """
    Run a single-mode ablation experiment with a pre-built EvolutionConfig.

    Unlike ``run_benchmark`` (which always compares baseline vs. sbp_gp),
    this function runs only the mode encoded in ``config``.  The
    ``mode`` label is derived from the config fields and stored in the
    DataFrame so results from multiple calls can be concatenated for
    analysis.

    Parameters
    ----------
    equations : list[str]
        Feynman equation filenames to benchmark.
    noise_levels : list[float]
    n_trials : int
    seed_base : int
    max_rows : int
    verbose : bool
    config : EvolutionConfig
        Fully configured object.  If None, defaults to EvolutionConfig().

    Returns
    -------
    pd.DataFrame
        One row per (equation, noise, trial) with columns:
        equation, noise, trial, mode, condition, fitness, r2, complexity,
        solved, best_found, wall_time_s.
    """
    _ensure_creators()

    if config is None:
        config = EvolutionConfig()

    # Derive a human-readable mode label from the config
    if not config.use_ssc and config.sbp_attempts == 0:
        mode_label = "standard"
    elif config.use_ssc and config.sbp_attempts == 0:
        mode_label = "ssc-only"
    else:
        mode_label = "hybrid"

    rows: list[dict] = []
    lib_cache: dict = {}

    for eq_name in equations:
        for noise in noise_levels:
            try:
                eq = load_feynman_equation(
                    eq_name,
                    noise_level=noise,
                    seed=42,
                    max_rows=max_rows,
                )
            except Exception as exc:
                logger.warning("Skipping %s (noise=%.2f): %s", eq_name, noise, exc)
                continue

            for trial in range(n_trials):
                trial_seed = seed_base + trial

                if verbose:
                    print(
                        f"\n  [{eq_name}] noise={noise:.2f} | trial={trial:02d} | "
                        f"mode={mode_label}"
                    )
                    # Header for the real-time gen output that follows
                    print(f"    {'Gen':>4}  {'BestFit':>12}  {'AvgFit':>12}  "
                          f"{'R²':>8}  Best Equation (post-trial)")
                    print(f"    {'---':>4}  {'-------':>12}  {'------':>12}  "
                          f"{'---':>8}  {'----------------------------'}")

                try:
                    m = _run_single_trial(
                        eq, config, trial_seed, lib_cache,
                        verbose=verbose,   # engine prints Gen N | min avg live
                    )
                    gen_rows = m.pop("gen_rows", [])

                    # ── Terminal: print per-gen table if verbose ───────────
                    if verbose and gen_rows:
                        print(f"\n    {'Gen':>4}  {'Fitness':>12}  {'AvgFit':>12}  "
                              f"{'R²':>8}  Best Equation")
                        print(f"    {'---':>4}  {'-------':>12}  {'------':>12}  "
                              f"{'--':>8}  ----------------")
                        for gr in gen_rows:
                            eq_short = (gr['equation'][:60] + '...'
                                        if len(gr['equation']) > 60
                                        else gr['equation'])
                            print(f"    {gr['gen']:>4}  {gr['fitness']:>12.4e}  "
                                  f"{gr['avg_fit']:>12.4e}  "
                                  f"{gr['r2']:>8.4f}  {eq_short}")
                        print()

                    row = {
                        "record_type":  "trial_summary",
                        "equation":     eq_name,
                        "noise":        noise,
                        "trial":        trial,
                        "mode":         mode_label,
                        "condition":    mode_label,
                        "gen":          "",
                        "fitness":      m.get("fitness", float("inf")),
                        "avg_fit":      "",
                        "r2":           m["r2"],
                        "complexity":   m["complexity"],
                        "solved":       m["solved"],
                        "best_found":   m.get("best_found", ""),
                        "wall_time_s":  m["wall_time_s"],
                    }
                    if verbose:
                        eq_str = m.get("best_found", "")
                        print(
                            f"  - TRIAL SUMMARY: "
                            f"fit={m.get('fitness', float('inf')):.4e}  "
                            f"r²={m['r2']:.3f}  cplx={m['complexity']}  "
                            f"solved={m['solved']}  eq={eq_str}"
                        )

                    # emit one gen_log row per generation
                    for gr in gen_rows:
                        rows.append({
                            "record_type":  "gen_log",
                            "equation":     eq_name,
                            "noise":        noise,
                            "trial":        trial,
                            "mode":         mode_label,
                            "condition":    mode_label,
                            "gen":          gr["gen"],
                            "fitness":      gr["fitness"],
                            "avg_fit":      gr["avg_fit"],
                            "r2":           gr["r2"],
                            "complexity":   "",
                            "solved":       "",
                            "best_found":   gr["equation"],
                            "wall_time_s":  "",
                        })

                except Exception as exc:
                    logger.warning("Trial failed: %s", exc, exc_info=True)
                    row = {
                        "record_type":  "trial_summary",
                        "equation":     eq_name,
                        "noise":        noise,
                        "trial":        trial,
                        "mode":         mode_label,
                        "condition":    mode_label,
                        "gen":          "",
                        "fitness":      float("inf"),
                        "avg_fit":      "",
                        "r2":           float("-inf"),
                        "complexity":   10_000,
                        "solved":       False,
                        "best_found":   "",
                        "wall_time_s":  0.0,
                    }
                    if verbose:
                        print(f"ERROR: {exc}")

                rows.append(row)

    return pd.DataFrame(rows)


# ======================================================================
# Statistical report
# ======================================================================

def print_statistical_report(df: pd.DataFrame) -> None:
    """
    Print a Wilcoxon paired-test report on R² and complexity improvements.

    Groups by (equation, noise) and tests paired baseline vs. SBP-GP
    across trials.

    Parameters
    ----------
    df : pd.DataFrame
        Output of run_benchmark().
    """
    from scipy import stats

    print("\n" + "=" * 68)
    print("STATISTICAL REPORT — Wilcoxon Paired Test (SBP-GP vs Baseline)")
    print("=" * 68)
    print(f"{'Equation':<15} {'Noise':>6} | {'ΔR² p-val':>12} {'ΔCplx p-val':>14} {'SR(sbp)':>9} {'SR(base)':>9}")
    print("-" * 68)

    for (eq_name, noise), grp in df.groupby(["equation", "noise"]):
        base_r2  = grp[grp.condition == "baseline" ]["r2"      ].values
        sbp_r2   = grp[grp.condition == "sbp_gp"   ]["r2"      ].values
        base_cx  = grp[grp.condition == "baseline" ]["complexity"].values
        sbp_cx   = grp[grp.condition == "sbp_gp"   ]["complexity"].values
        base_sol = grp[grp.condition == "baseline" ]["solved"  ].mean()
        sbp_sol  = grp[grp.condition == "sbp_gp"   ]["solved"  ].mean()

        try:
            _, p_r2 = stats.wilcoxon(sbp_r2 - base_r2, alternative="greater")
        except Exception:
            p_r2 = float("nan")

        try:
            _, p_cx = stats.wilcoxon(base_cx - sbp_cx, alternative="greater")
        except Exception:
            p_cx = float("nan")

        print(
            f"{eq_name:<15} {noise:>6.2f} | "
            f"{p_r2:>12.4f} {p_cx:>14.4f} "
            f"{sbp_sol:>9.2f} {base_sol:>9.2f}"
        )

    print("=" * 68)
    print("p-values < 0.05 indicate significant improvement by SBP-GP.\n")


# ======================================================================
# CLI entry point
# ======================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Feynman SBP-GP Benchmark")
    parser.add_argument("--equations", nargs="+", default=["I.12.1", "I.14.3"],
                        help="Feynman equation filenames to benchmark")
    parser.add_argument("--noise",     nargs="+", type=float, default=[0.0, 0.01, 0.1])
    parser.add_argument("--trials",    type=int,   default=5)
    parser.add_argument("--n-gen",     type=int,   default=30)
    parser.add_argument("--pop-size",  type=int,   default=200)
    parser.add_argument("--out",       type=str,   default="benchmark_results.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    df = run_benchmark(
        equations    = args.equations,
        noise_levels = args.noise,
        n_trials     = args.trials,
        n_gen        = args.n_gen,
        pop_size     = args.pop_size,
    )
    df.to_csv(args.out, index=False)
    print(f"\nResults saved to {args.out}")
    print_statistical_report(df)
