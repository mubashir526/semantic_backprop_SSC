"""
Feynman Benchmark CLI Entry Point
===================================
Orchestrates SBP-GP experiments against the Feynman SR dataset.

Usage examples
--------------
    # Verify the integration (runs 8 checks, exits 0 on success)
    python scripts/run_feynman.py --verify-only

    # Run the full hybrid mode (default) on one equation
    python scripts/run_feynman.py --equation I.6.2 --trials 5

    # Run ablation: standard GP only (no SSC, no SBP)
    python scripts/run_feynman.py --equation I.6.2 --mode standard

    # Run ablation: SSC enabled but no SBP repair
    python scripts/run_feynman.py --equation I.6.2 --mode ssc-only

    # Run full hybrid (SSC + SBP), save results
    python scripts/run_feynman.py \\
        --equation I.12.1 I.14.3 \\
        --trials 10 \\
        --noise 0.0 0.01 0.1 \\
        --n-gen 100 --pop-size 1000 \\
        --mode hybrid \\
        --out results/run_001.csv
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

# ── Project root on sys.path ──────────────────────────────────────────────────
_SCRIPTS = Path(__file__).resolve().parent
_PROJECT = _SCRIPTS.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ======================================================================
# Ablation hyper-parameters (edit here to tune for complex equations)
# ======================================================================

# Base config defaults — tuned for complex exponential equations (e.g. I.6.2)
_POP_SIZE        = 1000
_N_GEN           = 100
_HOF_SIZE        = 10
_CX_PROB         = 0.7
_MUT_PROB        = 0.3
_MIN_DEPTH       = 1
_MAX_DEPTH       = 5
_TOURNAMENT_SIZE = 5
_IMMIGRANT_FRAC  = 0.1

# Per-mode SBP settings
_MODE_SETTINGS = {
    "standard": dict(use_ssc=False, sbp_attempts=0, sbp_max_frag=0),
    "ssc-only": dict(use_ssc=True,  sbp_attempts=0, sbp_max_frag=0),
    "hybrid":   dict(use_ssc=True,  sbp_attempts=8, sbp_max_frag=7),
}


# ======================================================================
# Argument parser
# ======================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_feynman.py",
        description="SBP-GP Feynman Benchmark Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--equation", nargs="+", default=["I.8.14"],
        metavar="EQ",
        help="One or more Feynman dataset equation filenames.",
    )
    p.add_argument(
        "--trials", type=int, default=1,
        help="Number of independent trials per (equation, noise) pair.",
    )
    p.add_argument(
        "--noise", nargs="+", type=float, default=[0.0, 0.01, 0.1],
        metavar="γ",
        help="Noise levels γ (Equation 3, Reissmann et al. 2025).",
    )
    p.add_argument(
        "--n-gen", type=int, default=_N_GEN,
        help="Number of GP generations per trial.",
    )
    p.add_argument(
        "--pop-size", type=int, default=_POP_SIZE,
        help="Population size per trial.",
    )
    p.add_argument(
        "--mode",
        choices=["standard", "ssc-only", "hybrid"],
        default="hybrid",
        help=(
            "Ablation mode: "
            "'standard' = no SSC, no SBP; "
            "'ssc-only' = SSC on, SBP off; "
            "'hybrid'   = SSC + SBP (full system)."
        ),
    )
    p.add_argument(
        "--verify-only", action="store_true",
        help="Run verify_integration.py and exit; skip benchmark.",
    )
    p.add_argument(
        "--out", type=str, default=None,
        metavar="PATH",
        help=(
            "Output CSV path for results DataFrame. "
            "Defaults to results/feynman_<eq>_<mode>_<timestamp>.csv"
        ),
    )
    p.add_argument(
        "--seed-base", type=int, default=0,
        help="Base RNG seed; trial i uses seed_base + i.",
    )
    p.add_argument(
        "--max-rows", type=int, default=10_000,
        help="Maximum rows loaded per dataset file (large files have millions).",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print per-trial progress.",
    )
    return p


# ======================================================================
# Build EvolutionConfig from parsed args + mode
# ======================================================================

def _build_config(args: argparse.Namespace):
    """
    Construct an EvolutionConfig from CLI arguments and the selected
    ablation mode.  The base hyper-parameters are the module-level
    constants (_POP_SIZE, _N_GEN, …) overridden by any CLI flags.
    The mode determines use_ssc and sbp_attempts.
    """
    from src.evolution.engine import EvolutionConfig

    mode_kw = _MODE_SETTINGS[args.mode]

    cfg = EvolutionConfig(
        pop_size        = args.pop_size,
        n_gen           = args.n_gen,
        hof_size        = _HOF_SIZE,
        cx_prob         = _CX_PROB,
        mut_prob        = _MUT_PROB,
        min_depth       = _MIN_DEPTH,
        max_depth       = _MAX_DEPTH,
        tournament_size = _TOURNAMENT_SIZE,
        immigrant_frac  = _IMMIGRANT_FRAC,
        # Mode-dependent settings
        use_ssc         = mode_kw["use_ssc"],
        sbp_attempts    = mode_kw["sbp_attempts"],
        sbp_max_frag    = mode_kw["sbp_max_frag"],
    )
    return cfg


# ======================================================================
# Verify-only path
# ======================================================================

def _run_verify() -> int:
    """
    Execute verify_integration.py in a subprocess so its sys.exit() is
    captured cleanly.  Returns the exit code.
    """
    verify_script = _SCRIPTS / "verify_integration.py"
    if not verify_script.exists():
        logger.error("verify_integration.py not found at %s", verify_script)
        return 1

    logger.info("Running integration verification …")
    result = subprocess.run(
        [sys.executable, str(verify_script)],
        cwd=str(_PROJECT),
    )
    return result.returncode


# ======================================================================
# Benchmark path
# ======================================================================

def _run_benchmark(args: argparse.Namespace) -> int:
    """
    Execute the full benchmark using benchmarks.feynman_runner.run_experiment.
    The ablation mode controls which algorithmic components are active.

    Returns
    -------
    int  — exit code (0 = success)
    """
    from benchmarks.feynman_runner import run_experiment, print_statistical_report

    # ── Build EvolutionConfig from mode + CLI overrides ────────────────────
    cfg = _build_config(args)

    # ── Resolve output path ────────────────────────────────────────────────
    if args.out:
        out_path = Path(args.out)
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        results_dir = _PROJECT / "results"
        results_dir.mkdir(exist_ok=True)
        eq_tag   = "_".join(args.equation[:2]).replace(".", "")
        mode_tag = args.mode.replace("-", "_")
        out_path = results_dir / f"feynman_{eq_tag}_{mode_tag}_{ts}.csv"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Equations  : %s", args.equation)
    logger.info("Noise levels: %s", args.noise)
    logger.info("Trials     : %d", args.trials)
    logger.info("Mode       : %s  (use_ssc=%s  sbp_attempts=%d)",
                args.mode, cfg.use_ssc, cfg.sbp_attempts)
    logger.info("pop_size=%d  n_gen=%d  cx=%.2f  mut=%.2f  tourn=%d  imm=%.2f",
                cfg.pop_size, cfg.n_gen, cfg.cx_prob, cfg.mut_prob,
                cfg.tournament_size, cfg.immigrant_frac)
    logger.info("Output     : %s", out_path)

    t0 = time.perf_counter()

    try:
        df = run_experiment(
            equations    = args.equation,
            noise_levels = args.noise,
            n_trials     = args.trials,
            seed_base    = args.seed_base,
            max_rows     = args.max_rows,
            verbose      = args.verbose,
            config       = cfg,
        )
    except Exception as exc:
        logger.error("Benchmark failed: %s", exc, exc_info=True)
        return 1

    wall = time.perf_counter() - t0

    # ── Save results ───────────────────────────────────────────────────────
    df.to_csv(out_path, index=False)
    logger.info("Results saved → %s  (%.1f s)", out_path, wall)

    # ── Print summary table ────────────────────────────────────────────────
    _print_summary_table(df)

    # ── Statistical report: only meaningful if multiple conditions present ─
    if "record_type" in df.columns:
        summary_df = df[df["record_type"] == "trial_summary"]
    else:
        summary_df = df

    conditions = summary_df["condition"].unique() if "condition" in summary_df.columns else []
    if "sbp_gp" in conditions and "baseline" in conditions:
        print_statistical_report(summary_df)

    return 0


# ======================================================================
# Summary table printer
# ======================================================================

def _print_summary_table(df) -> None:
    print("\n" + "=" * 78)
    print("  RESULTS SUMMARY")
    print("=" * 78)
    print(f"{'Equation':<15} {'Noise':>6} {'Mode':>10} {'Cond':>10} | "
          f"{'Fit mean':>10} {'R² mean':>9} {'R² std':>8} {'SR':>6} {'Cplx':>6}")
    print("-" * 78)

    # Filter to only trial summaries (ignore per-gen log rows)
    if "record_type" in df.columns:
        summary_df = df[df["record_type"] == "trial_summary"]
    else:
        summary_df = df

    group_cols = ["equation", "noise", "mode", "condition"]
    # Gracefully handle DataFrames that lack the 'mode' column
    group_cols = [c for c in group_cols if c in summary_df.columns]

    for keys, grp in summary_df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        eq    = keys[0]
        noise = keys[1] if len(keys) > 1 else ""
        mode  = keys[2] if len(keys) > 2 else ""
        cond  = keys[3] if len(keys) > 3 else ""

        noise_str = f"{noise:6.2f}" if isinstance(noise, (int, float)) else f"{noise:>6}"
        fit_m  = grp["fitness"].mean() if "fitness" in grp.columns else float("nan")
        r2_m   = grp["r2"].mean()
        r2_s   = grp["r2"].std()
        sr     = grp["solved"].mean()
        cplx   = grp["complexity"].mean()
        print(f"{eq:<15} {noise_str} {mode:>10} {cond:>10} | "
              f"{fit_m:>10.4e} {r2_m:>9.3f} {r2_s:>8.3f} {sr:>6.2f} {cplx:>6.0f}")

    print("=" * 78)


# ======================================================================
# Entry point
# ======================================================================

def main() -> int:
    parser = _build_parser()
    args   = parser.parse_args()

    if args.verbose:
        # We rely on the verbose flags passed down to feynman_runner.py instead
        # of setting the root logger to DEBUG, which floods the terminal.
        pass

    # ── --verify-only ─────────────────────────────────────────────────────
    if args.verify_only:
        code = _run_verify()
        if code == 0:
            print("\n  ✓  All integration checks passed.")
        else:
            print("\n  ✗  One or more integration checks FAILED.")
        return code

    # ── Full benchmark ────────────────────────────────────────────────────
    return _run_benchmark(args)


if __name__ == "__main__":
    sys.exit(main())
