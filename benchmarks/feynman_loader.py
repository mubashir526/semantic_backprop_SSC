"""
Feynman Dataset Loader
=======================
Loads, splits, and optionally noise-corrupts Feynman Symbolic Regression
benchmark equations.

Noise Model (Reissmann et al. 2025, Equation 3)
-----------------------------------------------
    ỹ_i = y_i + η,    η ~ N(0, γ · RMS(y))

where RMS(y) = sqrt( mean(y²) )

CRITICAL: Noise is scaled by the *RMS* of target values, NOT the standard
deviation.  Do not substitute np.std(y) here.

Data Splitting
--------------
Reproducible 75/25 train-test split using a caller-supplied (or default)
integer seed.  Row indices are shuffled deterministically so that two
calls with the same seed return identical splits with no row overlap.

Dimension Loading
-----------------
Physical dimensions are parsed from dataset/units.csv using the
[m, s, kg, T, V] → [kg, m, s, K, A, mol, cd] conversion from the
reference DataLoader (backprop_gep/data_loader.py).
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# ── Project root resolution ───────────────────────────────────────────────────
# Resolve paths relative to the package root so callers don't need to set CWD.
_HERE        = Path(__file__).resolve().parent            # benchmarks/
_PROJECT     = _HERE.parent                               # integrated_sbp_gp/
_DATASET_DIR = _PROJECT / "dataset" / "Feynman_with_units"
_EQUATIONS_CSV = _PROJECT / "dataset" / "FeynmanEquations.csv"
_UNITS_CSV     = _PROJECT / "dataset" / "units.csv"


# ======================================================================
# Return type
# ======================================================================

@dataclass
class FeynmanEquation:
    """
    All metadata and data splits for one Feynman benchmark equation.

    Attributes
    ----------
    filename : str
        The Feynman dataset filename (e.g. ``"I.10.7"``).
    formula : str
        Ground-truth formula string (from FeynmanEquations.csv).
    var_names : list[str]
        Input variable names in column order.
    target_name : str
        Name of the target (output) variable.
    context_dims : dict[str, Dimension]
        Maps variable name → 7-element SI Dimension.
    target_dim : Dimension
        Physical dimension of the target.
    X_train : np.ndarray, shape (N_train, n_vars)
    y_train : np.ndarray, shape (N_train,)
    X_test  : np.ndarray, shape (N_test, n_vars)
    y_test  : np.ndarray, shape (N_test,)
    noise_level : float
        γ used (0.0 = noiseless).
    """
    filename:     str
    formula:      str
    var_names:    list[str]
    target_name:  str
    context_dims: dict       # dict[str, Dimension]
    target_dim:   object     # Dimension
    X_train:      np.ndarray
    y_train:      np.ndarray
    X_test:       np.ndarray
    y_test:       np.ndarray
    noise_level:  float = 0.0


# ======================================================================
# Dimension parsing  (mirrors reference backprop_gep/data_loader.py)
# ======================================================================

def _parse_units_csv(units_csv: Path) -> dict:
    """
    Parse units.csv ([m, s, kg, T, V] basis) and return a dict:
        variable_name → Dimension (7-element SI vector)

    Conversion to [kg, m, s, K, A, mol, cd]:
        V (Voltage) = kg·m²·s⁻³·A⁻¹
    """
    from src.physics.dimension import Dimension

    df = pd.read_csv(units_csv)
    units_dict: dict = {}

    for _, row in df.iterrows():
        var_name = row.get("Variable", None)
        if pd.isna(var_name) or str(var_name).strip() == "":
            continue

        def _f(col):
            v = row.get(col, 0.0)
            return float(v) if not pd.isna(v) else 0.0

        m  = _f("m")
        s  = _f("s")
        kg = _f("kg")
        T  = _f("T")
        V  = _f("V")

        vec = np.zeros(7)
        vec[0] = kg + V * 1        # kg
        vec[1] = m  + V * 2        # m
        vec[2] = s  + V * -3       # s
        vec[3] = T                  # K  (temperature)
        vec[4] = V * -1             # A  (current)
        vec[5] = 0.0                # mol
        vec[6] = 0.0                # cd

        units_dict[str(var_name).strip()] = Dimension(vec)

    return units_dict


# Module-level cache so the CSV is only parsed once per process
_UNITS_CACHE: dict | None = None

def _get_units(units_csv: Path = _UNITS_CSV) -> dict:
    global _UNITS_CACHE
    if _UNITS_CACHE is None:
        _UNITS_CACHE = _parse_units_csv(units_csv)
    return _UNITS_CACHE


# ======================================================================
# Noise injection — Equation 3
# ======================================================================

def add_noise(
    y: np.ndarray,
    gamma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Apply Equation 3 noise model from Reissmann et al. (2025):

        ỹ_i = y_i + η,   η ~ N(0, γ · sqrt(mean(y²)))

    Parameters
    ----------
    y     : np.ndarray, shape (N,)
    gamma : float, noise amplitude (0 = no noise)
    rng   : np.random.Generator  (for reproducibility)

    Returns
    -------
    np.ndarray  — noisy target values
    """
    if gamma == 0.0:
        return y.copy()

    rms = float(np.sqrt(np.mean(y ** 2)))   # NOT np.std(y)
    sigma = gamma * rms
    eta = rng.normal(0.0, sigma, size=y.shape)
    return y + eta


# ======================================================================
# Main public API
# ======================================================================

def load_feynman_equation(
    filename: str,
    noise_level: float = 0.0,
    train_frac: float = 0.75,
    seed: int = 42,
    max_rows: int = 10_000,
    dataset_dir: Path | str | None = None,
    equations_csv: Path | str | None = None,
    units_csv: Path | str | None = None,
) -> FeynmanEquation:
    """
    Load one Feynman benchmark equation and return a FeynmanEquation.

    Parameters
    ----------
    filename : str
        Feynman dataset file name (e.g. ``"I.10.7"``), which must exist
        inside ``dataset/Feynman_with_units/``.
    noise_level : float
        γ in Equation 3 (0.0 = noiseless).
    train_frac : float
        Fraction of rows assigned to the training set (default 0.75).
    seed : int
        RNG seed for reproducible shuffle and noise injection.
    max_rows : int
        Cap on the number of rows loaded (large files can be > 10 M rows).
    dataset_dir, equations_csv, units_csv : Path or None
        Override default file paths (useful for testing with fixtures).

    Returns
    -------
    FeynmanEquation
    """
    from src.physics.dimension import Dimension

    # ── Resolve paths ─────────────────────────────────────────────────────
    ddir   = Path(dataset_dir)   if dataset_dir   else _DATASET_DIR
    eqcsv  = Path(equations_csv) if equations_csv else _EQUATIONS_CSV
    ucsv   = Path(units_csv)     if units_csv     else _UNITS_CSV

    # ── Load metadata ──────────────────────────────────────────────────────
    meta_df = pd.read_csv(eqcsv)
    matches = meta_df[meta_df["Filename"] == filename]
    if len(matches) == 0:
        raise ValueError(f"No metadata for equation '{filename}' in {eqcsv}")
    row = matches.iloc[0]

    n_vars    = int(row["# variables"])
    formula   = str(row["Formula"])
    var_names = [str(row[f"v{i}_name"]) for i in range(1, n_vars + 1)]
    target_name = str(row["Output"])

    # ── Load dimension map ─────────────────────────────────────────────────
    units_dict = _get_units(ucsv)

    def _dim(name: str) -> "Dimension":
        return units_dict.get(name, Dimension.dimensionless())

    context_dims = {name: _dim(name) for name in var_names}
    target_dim   = _dim(target_name)

    # ── Load raw data ──────────────────────────────────────────────────────
    file_path = ddir / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    # Large files: load only the first `max_rows` rows
    data = np.loadtxt(file_path, max_rows=max_rows)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] < n_vars + 1:
        raise ValueError(
            f"Data file '{filename}' has {data.shape[1]} columns, "
            f"expected at least {n_vars + 1}."
        )

    X = data[:, :n_vars].astype(float)
    y = data[:, n_vars].astype(float)

    # ── Reproducible 75/25 train-test split ───────────────────────────────
    rng = np.random.default_rng(seed)
    indices = np.arange(len(y))
    rng.shuffle(indices)
    split  = int(train_frac * len(indices))
    train_idx = indices[:split]
    test_idx  = indices[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    # ── Noise injection (Equation 3) ───────────────────────────────────────
    # Use a separate RNG seeded from the main seed so noise is reproducible
    # but independent of the split shuffle.
    noise_rng = np.random.default_rng(seed + 1)
    y_train   = add_noise(y_train, noise_level, noise_rng)
    # Test labels are kept clean (evaluate on ground truth)

    return FeynmanEquation(
        filename     = filename,
        formula      = formula,
        var_names    = var_names,
        target_name  = target_name,
        context_dims = context_dims,
        target_dim   = target_dim,
        X_train      = X_train,
        y_train      = y_train,
        X_test       = X_test,
        y_test       = y_test,
        noise_level  = noise_level,
    )


# ======================================================================
# Convenience: list available equations
# ======================================================================

def list_equations(equations_csv: Path | str | None = None) -> list[str]:
    """Return a sorted list of available Feynman equation filenames."""
    eqcsv = Path(equations_csv) if equations_csv else _EQUATIONS_CSV
    df    = pd.read_csv(eqcsv)
    return sorted(df["Filename"].dropna().tolist())
