"""
Unit tests for benchmarks/feynman_loader.py and benchmarks/metrics.py.
Run with: pytest benchmarks/tests/test_feynman_loader.py -v

All tests use lightweight in-memory fixtures — no real dataset files
are needed, so the suite runs in < 5 seconds.
"""
from __future__ import annotations

import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import pandas as pd

# ── Resolve project root for imports ──────────────────────────────────────────
_HERE = Path(__file__).resolve().parent          # benchmarks/tests/
_PROJECT = _HERE.parent.parent                   # integrated_sbp_gp/
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from benchmarks.feynman_loader import add_noise, load_feynman_equation, FeynmanEquation
from benchmarks.metrics import (
    r2_score,
    expression_complexity,
    is_symbolic_solution,
    compute_trial_metrics,
)


# ======================================================================
# Minimal fixture data
# ======================================================================

# One equation: F = m * g  (I.14.3 style: 3 variables, last col = target)
# We create a tiny temporary dataset file and matching CSVs.

@pytest.fixture(scope="module")
def tmp_dataset(tmp_path_factory):
    """
    Create a minimal self-contained Feynman-compatible dataset on disk.

    Equation: F = m * g * z   (I.14.3)
    Variables: m, g, z (all in range [1,5])
    Target: F = m*g*z
    """
    tmp = tmp_path_factory.mktemp("feynman_data")
    data_dir = tmp / "Feynman_with_units"
    data_dir.mkdir()

    # ── Generate data ─────────────────────────────────────────────────────
    rng = np.random.default_rng(0)
    N   = 200
    m   = rng.uniform(1, 5, N)
    g   = rng.uniform(1, 5, N)
    z   = rng.uniform(1, 5, N)
    F   = m * g * z

    data = np.column_stack([m, g, z, F])
    eq_file = data_dir / "I.14.3"
    np.savetxt(eq_file, data)

    # ── Minimal FeynmanEquations.csv ─────────────────────────────────────
    eq_csv = tmp / "FeynmanEquations.csv"
    eq_csv.write_text(
        "Filename,Number,Output,Formula,# variables,"
        "v1_name,v1_low,v1_high,"
        "v2_name,v2_low,v2_high,"
        "v3_name,v3_low,v3_high\n"
        "I.14.3,15,U,m*g*z,3,"
        "m,1,5,"
        "g,1,5,"
        "z,1,5\n"
    )

    # ── Minimal units.csv ─────────────────────────────────────────────────
    units_csv = tmp / "units.csv"
    units_csv.write_text(
        "Variable,Units,m,s,kg,T,V\n"
        "m,Mass,0,0,1,0,0\n"
        "g,Acceleration,1,-2,0,0,0\n"
        "z,Length,1,0,0,0,0\n"
        "U,Energy,2,-2,1,0,0\n"
    )

    return {
        "data_dir":     data_dir,
        "equations_csv": eq_csv,
        "units_csv":    units_csv,
        "N":            N,
        "true_F":       F,
        "m": m, "g": g, "z": z,
    }


@pytest.fixture(scope="module")
def loaded_eq(tmp_dataset):
    """Load the test equation with no noise."""
    return load_feynman_equation(
        "I.14.3",
        noise_level=0.0,
        seed=42,
        dataset_dir=tmp_dataset["data_dir"],
        equations_csv=tmp_dataset["equations_csv"],
        units_csv=tmp_dataset["units_csv"],
    )


# ======================================================================
# TASK 1 Tests — feynman_loader.py
# ======================================================================

class TestFeynmanLoader:

    def test_returns_feynman_equation(self, loaded_eq):
        assert isinstance(loaded_eq, FeynmanEquation)

    def test_correct_variable_count(self, loaded_eq):
        assert len(loaded_eq.var_names) == 3
        assert loaded_eq.var_names == ["m", "g", "z"]

    def test_formula_correct(self, loaded_eq):
        assert loaded_eq.formula == "m*g*z"

    def test_no_row_overlap_in_splits(self, loaded_eq, tmp_dataset):
        """
        Reproduction Check: train and test must share no rows.
        """
        # Identify rows by their exact values
        X_train = loaded_eq.X_train
        X_test  = loaded_eq.X_test

        # Convert rows to sets of tuples for overlap detection
        train_set = {tuple(row) for row in X_train}
        test_set  = {tuple(row) for row in X_test}
        overlap   = train_set & test_set
        assert len(overlap) == 0, f"Train/test overlap: {len(overlap)} rows shared"

    def test_split_fractions(self, loaded_eq, tmp_dataset):
        """75/25 split should be approximately maintained."""
        N = tmp_dataset["N"]
        n_train = len(loaded_eq.X_train)
        n_test  = len(loaded_eq.X_test)
        assert n_train + n_test == N
        # 75% ± 2 rows (integer rounding)
        assert abs(n_train - int(0.75 * N)) <= 2

    def test_reproduction_same_seed_gives_identical_splits(self, tmp_dataset):
        """
        Reproduction Check: two calls with the same seed must return
        bit-for-bit identical X_train / y_train arrays.
        """
        common_kwargs = dict(
            noise_level=0.0,
            seed=7,
            dataset_dir=tmp_dataset["data_dir"],
            equations_csv=tmp_dataset["equations_csv"],
            units_csv=tmp_dataset["units_csv"],
        )
        eq1 = load_feynman_equation("I.14.3", **common_kwargs)
        eq2 = load_feynman_equation("I.14.3", **common_kwargs)

        np.testing.assert_array_equal(eq1.X_train, eq2.X_train,
                                      err_msg="X_train differs across seeds")
        np.testing.assert_array_equal(eq1.y_train, eq2.y_train,
                                      err_msg="y_train differs across seeds")

    def test_different_seeds_give_different_splits(self, tmp_dataset):
        """Different seeds must produce different permutations."""
        kw = dict(
            noise_level=0.0,
            dataset_dir=tmp_dataset["data_dir"],
            equations_csv=tmp_dataset["equations_csv"],
            units_csv=tmp_dataset["units_csv"],
        )
        eq_a = load_feynman_equation("I.14.3", seed=1, **kw)
        eq_b = load_feynman_equation("I.14.3", seed=99, **kw)
        # Very unlikely to be equal unless dataset is trivially small
        assert not np.array_equal(eq_a.X_train, eq_b.X_train), \
            "Different seeds produced identical splits"

    def test_context_dims_populated(self, loaded_eq):
        """context_dims must have an entry for every variable."""
        from src.physics.dimension import Dimension
        for name in loaded_eq.var_names:
            assert name in loaded_eq.context_dims
            assert isinstance(loaded_eq.context_dims[name], Dimension)

    def test_target_dim_is_dimension(self, loaded_eq):
        from src.physics.dimension import Dimension
        assert isinstance(loaded_eq.target_dim, Dimension)

    def test_noiseless_y_matches_ground_truth(self, loaded_eq, tmp_dataset):
        """With gamma=0 the returned y values must equal F = m*g*z exactly."""
        X_all = np.vstack([loaded_eq.X_train, loaded_eq.X_test])
        y_all = np.concatenate([loaded_eq.y_train, loaded_eq.y_test])
        F_pred = X_all[:, 0] * X_all[:, 1] * X_all[:, 2]
        # Sort both by first column to align them
        idx_eq = np.argsort(X_all[:, 0])
        np.testing.assert_allclose(y_all[idx_eq], F_pred[idx_eq], rtol=1e-9)


class TestNoiseModel:
    """
    Verify the RMS-scaled noise model (Equation 3).

    η ~ N(0, γ · sqrt(mean(y²)))
    Var(η) = γ² · mean(y²)
    """

    def test_rms_noise_variance_within_5pct(self):
        """
        Core requirement: injected noise variance ≈ γ² · mean(y²).
        Tolerance: 5 % (tight but achievable with N=10_000).
        """
        rng   = np.random.default_rng(0)
        N     = 10_000
        y     = rng.uniform(1.0, 10.0, N)      # positive values, known RMS
        gamma = 0.1

        y_noisy = add_noise(y, gamma, rng)
        eta     = y_noisy - y                   # isolate the noise component

        expected_var = gamma**2 * float(np.mean(y**2))
        actual_var   = float(np.var(eta))

        rel_error = abs(actual_var - expected_var) / expected_var
        assert rel_error < 0.05, (
            f"Noise variance {actual_var:.6f} deviates from expected "
            f"{expected_var:.6f} by {rel_error:.2%} (limit 5%)"
        )

    def test_noise_uses_rms_not_std(self):
        """
        Explicitly confirm: std(y) ≠ rms(y) for a non-zero-mean signal,
        and the loader uses rms (sqrt of mean of squares), not std.
        """
        rng   = np.random.default_rng(1)
        N     = 10_000
        y     = np.full(N, 5.0) + rng.normal(0, 0.1, N)   # mean ≈ 5, std ≈ 0.1
        gamma = 0.05

        rms    = float(np.sqrt(np.mean(y**2)))
        std    = float(np.std(y))
        assert abs(rms - std) > 0.5, (
            "Test setup error: rms and std should differ significantly for this signal"
        )

        # Inject noise using the loader's add_noise
        y_noisy   = add_noise(y, gamma, rng)
        eta       = y_noisy - y
        actual_var = float(np.var(eta))

        # Should match RMS-based formula
        expected_rms_var = gamma**2 * float(np.mean(y**2))
        expected_std_var = gamma**2 * std**2

        err_rms = abs(actual_var - expected_rms_var) / expected_rms_var
        err_std = abs(actual_var - expected_std_var) / expected_std_var

        assert err_rms < err_std, (
            "Noise variance is closer to the std-based formula than the "
            "RMS-based formula — loader is using the wrong denominator."
        )

    def test_zero_noise_unchanged(self):
        """gamma=0 must return an exact copy of y."""
        rng = np.random.default_rng(2)
        y   = rng.uniform(1, 10, 100)
        y_n = add_noise(y, 0.0, rng)
        np.testing.assert_array_equal(y, y_n)

    def test_noisy_split_leaves_test_clean(self, tmp_dataset):
        """
        Test labels must NOT be corrupted — noise is applied to train only.
        """
        eq_noisy = load_feynman_equation(
            "I.14.3",
            noise_level=0.1,
            seed=42,
            dataset_dir=tmp_dataset["data_dir"],
            equations_csv=tmp_dataset["equations_csv"],
            units_csv=tmp_dataset["units_csv"],
        )
        eq_clean = load_feynman_equation(
            "I.14.3",
            noise_level=0.0,
            seed=42,
            dataset_dir=tmp_dataset["data_dir"],
            equations_csv=tmp_dataset["equations_csv"],
            units_csv=tmp_dataset["units_csv"],
        )
        # Test sets should be identical (same seed → same rows, no noise on test)
        np.testing.assert_array_equal(
            eq_noisy.X_test, eq_clean.X_test,
            err_msg="X_test differs between noisy and clean loads"
        )
        np.testing.assert_array_equal(
            eq_noisy.y_test, eq_clean.y_test,
            err_msg="y_test is corrupted by noise (should be clean)"
        )


# ======================================================================
# TASK 2 Tests — metrics.py
# ======================================================================

class TestR2Score:

    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert r2_score(y, y) == pytest.approx(1.0)

    def test_constant_prediction(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.full_like(y_true, np.mean(y_true))
        assert r2_score(y_true, y_pred) == pytest.approx(0.0)

    def test_bad_prediction(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = -y_true
        assert r2_score(y_true, y_pred) < 0.0

    def test_inf_input_returns_neg_inf(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, np.inf, 3.0])
        assert r2_score(y_true, y_pred) == float("-inf")


class TestExpressionComplexity:
    """
    Simplification Logic: sympy.simplify() must reduce redundant terms.
    """

    def test_single_symbol(self):
        """A single variable x should have complexity 1."""
        c = expression_complexity("x", ["x"])
        assert c == 1

    def test_redundant_addition_simplifies(self):
        """
        x + x - x = x.
        Raw tree has 4 nodes; simplified has 1.
        """
        c = expression_complexity("x + x - x", ["x"])
        assert c == 1, (
            f"Expected complexity 1 (x + x - x = x), got {c}. "
            "Check that sympy.simplify() is applied before counting."
        )

    def test_constant_expression(self):
        """2 + 3 = 5, a single number → complexity 1."""
        c = expression_complexity("2 + 3", [])
        assert c == 1

    def test_complex_tree_greater_than_simple(self):
        """A multi-term expression must score higher than a leaf."""
        c_simple  = expression_complexity("x", ["x"])
        c_complex = expression_complexity("x * y + z / x", ["x", "y", "z"])
        assert c_complex > c_simple

    def test_unparseable_expression_returns_penalty(self):
        """A malformed expression must return 10_000 (max penalty)."""
        c = expression_complexity("((((not valid python", ["x"])
        assert c == 10_000

    def test_multiply_by_one_simplifies(self):
        """1 * x = x → complexity 1."""
        c = expression_complexity("1 * x", ["x"])
        assert c == 1

    def test_zero_addition_simplifies(self):
        """x + 0 = x → complexity 1."""
        c = expression_complexity("x + 0", ["x"])
        assert c == 1


class TestSymbolicSolutionRate:
    """
    Equation 6: a trial is Correct if f - f̂ or f / f̂ simplifies to const.
    """

    def test_identical_expressions_correct(self):
        assert is_symbolic_solution("x*y", "x*y", ["x", "y"]) is True

    def test_constant_multiple_correct(self):
        """2*x*y is proportional to x*y → ratio is constant."""
        assert is_symbolic_solution("2*x*y", "x*y", ["x", "y"]) is True

    def test_constant_additive_correct(self):
        """x*y + 3 differs from x*y by a constant."""
        assert is_symbolic_solution("x*y + 3", "x*y", ["x", "y"]) is True

    def test_wrong_expression_incorrect(self):
        """x*y vs x*z: neither difference nor ratio is constant."""
        assert is_symbolic_solution("x*z", "x*y", ["x", "y", "z"]) is False

    def test_unparseable_returns_false(self):
        assert is_symbolic_solution("{{{{", "x*y", ["x", "y"]) is False

    def test_quadratic_vs_linear_incorrect(self):
        """x**2 vs x: ratio = x is not constant."""
        assert is_symbolic_solution("x**2", "x", ["x"]) is False


class TestComputeTrialMetrics:
    """
    End-to-end test of compute_trial_metrics using a DEAP individual.
    """

    @pytest.fixture(scope="class")
    def setup_env(self):
        import operator
        import functools
        import random as _random
        from deap import gp, creator, base, tools

        var_names = ["m", "g", "z"]
        pset = gp.PrimitiveSet("BENCH_TEST", arity=0)
        pset.addPrimitive(operator.mul, 2, name="*")
        pset.addPrimitive(operator.add, 2, name="+")
        pset.addTerminal(1.0, name="m")
        pset.addTerminal(1.0, name="g")
        pset.addTerminal(1.0, name="z")

        if "FitnessMinMT" not in dir(creator):
            creator.create("FitnessMinMT", base.Fitness, weights=(-1.0,))
        if "IndividualMT" not in dir(creator):
            creator.create("IndividualMT", gp.PrimitiveTree,
                           fitness=creator.FitnessMinMT)

        # Ground truth: *(*(m g) z)
        tree = gp.PrimitiveTree.from_string("*(*(m g) z)", pset)
        ind  = creator.IndividualMT(tree)

        rng = np.random.default_rng(0)
        m   = rng.uniform(1, 5, 50)
        g   = rng.uniform(1, 5, 50)
        z   = rng.uniform(1, 5, 50)
        X_test = np.column_stack([m, g, z])
        y_test = m * g * z

        return pset, var_names, X_test, y_test, ind

    def test_perfect_ind_high_r2(self, setup_env):
        pset, var_names, X_test, y_test, ind = setup_env
        metrics = compute_trial_metrics(
            ind, pset, var_names, X_test, y_test, "m*g*z"
        )
        assert metrics["r2"] > 0.99, f"R² should be ~1.0 for ground truth, got {metrics['r2']}"

    def test_solved_flag_true_for_correct(self, setup_env):
        pset, var_names, X_test, y_test, ind = setup_env
        metrics = compute_trial_metrics(
            ind, pset, var_names, X_test, y_test, "m*g*z"
        )
        assert metrics["solved"] is True

    def test_complexity_positive(self, setup_env):
        pset, var_names, X_test, y_test, ind = setup_env
        metrics = compute_trial_metrics(
            ind, pset, var_names, X_test, y_test, "m*g*z"
        )
        assert metrics["complexity"] >= 1
