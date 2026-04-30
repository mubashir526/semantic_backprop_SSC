"""
Evolutionary Utilities
=======================
PSet factory, Feynman fitness function, and Semantic Similarity
Crossover (SSC) for the integrated DEAP-based GP framework.

Architecture Notes
------------------
- PSet uses arity=0 with addTerminal(1.0, name=var_name) for each
  variable.  This gives terminal nodes whose .name attribute directly
  matches context_dims keys, enabling evaluate_dim_at lookups without
  any translation layer.

- Operator names MUST match DimensionRules.KNOWN_OPERATORS exactly
  ('+', '-', '*', '/', 'sq', 'sqrt', 'sin', 'cos', 'log', 'exp').
  These names cannot be compiled via gp.compile because Python treats
  '+' etc. as syntax.  We therefore use a custom eval_tree() function
  for numerical evaluation (avoids gp.compile entirely).

- ERCs are absent from context_dims → evaluate_dim_at falls back to
  Dimension.dimensionless() — correct per Section 3.3.

- No param_incl or scipy.curve_fit on the Feynman path.  Variables are
  physical constants, not free parameters.

- SSC evaluates the raw symbolic tree only — no constant fitting.
"""

from __future__ import annotations

import copy
import math
import operator
import random
import warnings
from typing import Any

import numpy as np

from deap import gp, tools
from scipy.optimize import minimize
from src.sbp.engine import MAX_TREE_NODES
from src.physics.dimension import Dimension
from src.physics.dimension_rules import DimensionalViolation
from src.sbp.engine import evaluate_dim_at

# ======================================================================
# Constants
# ======================================================================

# SSC window (Cohen et al. / gp_alg.py reference values)
SSC_ALPHA: float = 1e-4      # lower SSD: subtrees must differ by ≥ this
SSC_BETA:  float = 0.4       # upper SSD: subtrees must differ by ≤ this
SSC_MAX_TRIALS: int = 12     # attempts before falling back to cxOnePoint
SSC_N_POINTS: int = 20       # random sample points for semantic evaluation

# Maximum tree height allowed after any variation operator
MAX_HEIGHT: int = 17

# Dispatch map from operator name → binary/unary Python function
# Used by eval_tree() for numerical evaluation without gp.compile.
_BINARY_OPS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": lambda a, b: (
        a / b if (np.isscalar(b) and abs(b) > 1e-10)
        else a / np.where(np.abs(b) < 1e-10, 1e-10, b)
        if not np.isscalar(b)
        else 1.0
    ),
}

_UNARY_OPS_FN = {
    "sq":   lambda x: x * x,
    "sqrt": lambda x: np.sqrt(np.abs(x)) if not np.isscalar(x) else math.sqrt(abs(x)),
    "sin":  lambda x: np.sin(x)  if not np.isscalar(x) else math.sin(x),
    "cos":  lambda x: np.cos(x)  if not np.isscalar(x) else math.cos(x),
    "log":  lambda x: np.log(np.abs(x) + 1e-10) if not np.isscalar(x) else math.log(abs(x) + 1e-10),
    "exp":  lambda x: np.exp(np.clip(x, -100, 100)) if not np.isscalar(x) else math.exp(max(-100.0, min(100.0, x))),
}


# ======================================================================
# 1. Custom tree evaluator (replaces gp.compile)
# ======================================================================

def eval_tree(
    tree: "gp.PrimitiveTree",
    context: dict[str, Any],
) -> Any:
    """
    Recursively evaluate a PrimitiveTree using ``context`` for variable
    lookups.  Returns a scalar or numpy array.

    This is used instead of ``gp.compile`` because operator names like
    '*', '+', '/' conflict with Python syntax when placed in a lambda
    expression by DEAP's compiler.

    Parameters
    ----------
    tree : gp.PrimitiveTree
        The individual to evaluate.
    context : dict[str, Any]
        Maps terminal names to their numerical values (scalars or arrays).
        ERCs / unknown names fall back to their stored float value.

    Returns
    -------
    scalar or np.ndarray
    """
    def _eval(pos: int):
        node = tree[pos]
        arity = node.arity

        if arity == 0:
            # Terminal: variable or ERC
            name = node.name
            if name in context:
                return context[name]
            # ERC or constant — node.value holds the numeric value
            try:
                return float(node.value)
            except (AttributeError, TypeError, ValueError):
                try:
                    return float(name)
                except (TypeError, ValueError):
                    return 1.0

        if arity == 1:
            child = _eval(pos + 1)
            fn = _UNARY_OPS_FN.get(node.name)
            if fn is None:
                raise ValueError(f"Unknown unary op: {node.name!r}")
            return fn(child)

        # Binary
        left  = _eval(pos + 1)
        right_pos = tree.searchSubtree(pos + 1).stop
        right = _eval(right_pos)
        fn = _BINARY_OPS.get(node.name)
        if fn is None:
            raise ValueError(f"Unknown binary op: {node.name!r}")
        return fn(left, right)

    return _eval(0)


# ======================================================================
# 2. Protected numeric operators (for gp operator functions)
# ======================================================================

def _protected_div(a, b):
    if np.isscalar(b):
        return a / b if abs(b) > 1e-10 else 1.0
    return a / np.where(np.abs(b) < 1e-10, 1e-10, b)

def _safe_sq(x):    return x * x
def _safe_sin(x):   return np.sin(x)   if not np.isscalar(x) else math.sin(x)
def _safe_cos(x):   return np.cos(x)   if not np.isscalar(x) else math.cos(x)

def _protected_log(x):
    return np.log(np.abs(x) + 1e-10) if not np.isscalar(x) else math.log(abs(x) + 1e-10)

def _protected_exp(x):
    return (np.exp(np.clip(x, -100.0, 100.0)) if not np.isscalar(x)
            else math.exp(max(-100.0, min(100.0, x))))

def _protected_sqrt(x):
    return np.sqrt(np.abs(x)) if not np.isscalar(x) else math.sqrt(abs(x))


# ======================================================================
# 3. PSet Factory
# ======================================================================

def make_feynman_pset(
    n_vars: int,
    var_names: list[str],
    erc_range: tuple[float, float] = (-5.0, 5.0),
) -> gp.PrimitiveSet:
    """
    Build a DEAP PrimitiveSet for Feynman symbolic regression.

    Architecture
    ------------
    - arity=0: variables are added as named terminals via addTerminal.
      This means terminal.name == var_name (e.g. 'mass'), which is what
      evaluate_dim_at uses for context_dims lookup.
    - Operator names match DimensionRules.KNOWN_OPERATORS exactly.
    - ERCs use functools.partial to avoid the lambda pickling warning.

    Parameters
    ----------
    n_vars : int
        Number of input variables.
    var_names : list[str]
        Variable names.  MUST match context_dims / feynman_dims.json keys.
    erc_range : (float, float)
        Range for Ephemeral Random Constants (dimensionless).

    Returns
    -------
    gp.PrimitiveSet  (arity=0)
    """
    import functools

    pset = gp.PrimitiveSet("FEYNMAN", arity=0)

    # ── Binary operators ─────────────────────────────────────────────────
    pset.addPrimitive(operator.add,    2, name="+")
    pset.addPrimitive(operator.sub,    2, name="-")
    pset.addPrimitive(operator.mul,    2, name="*")
    pset.addPrimitive(_protected_div,  2, name="/")

    # ── Unary operators ──────────────────────────────────────────────────
    pset.addPrimitive(_safe_sq,        1, name="sq")
    pset.addPrimitive(_protected_sqrt, 1, name="sqrt")
    pset.addPrimitive(_safe_sin,       1, name="sin")
    pset.addPrimitive(_safe_cos,       1, name="cos")
    pset.addPrimitive(_protected_log,  1, name="log")
    pset.addPrimitive(_protected_exp,  1, name="exp")

    # ── Named terminals — one per variable ───────────────────────────────
    # addTerminal with name= gives terminal.name == var_name.
    # This is the mechanism that makes evaluate_dim_at work.
    for name in var_names:
        pset.addTerminal(1.0, name=name)

    # ── Ephemeral Random Constant (dimensionless) ─────────────────────────
    # Use functools.partial instead of lambda to avoid pickling warning.
    lo, hi = erc_range
    pset.addEphemeralConstant(
        "ERC",
        functools.partial(random.uniform, lo, hi),
    )

    return pset


# ======================================================================
# 4. Feynman Fitness Function Factory
# ======================================================================

def make_feynman_fitness(
    X_train: np.ndarray,
    y_train: np.ndarray,
    var_names: list[str],
    pset: gp.PrimitiveSet,
    target_dim: Dimension | None = None,
    context_dims: dict[str, Dimension] | None = None,
    lambda_penalty: float = 0.1,
    optimize_constants: bool = True,
) -> callable:
    """
    Return a fitness callable: ``fitness(individual) -> (mse,)``.

    Uses eval_tree() instead of gp.compile() to avoid Python syntax
    conflicts with operator names like '*' and '+'.

    Parameters
    ----------
    X_train : np.ndarray, shape (N, n_vars)
    y_train : np.ndarray, shape (N,)
    var_names : list[str]
        Variable names in column order (match pset terminal names).
    pset : gp.PrimitiveSet
        Unused — kept for API symmetry and future use.
    target_dim: Dimension
        Target dimension for penalty calculation.
    context_dims: dict[str, Dimension]
        Terminal dimension mapping for penalty calculation.
    lambda_penalty: float
        Penalty multiplier for dimensional deviations.
    optimize_constants: bool
        Whether to run CG optimization on ERC nodes.

    Returns
    -------
    callable  →  ``(float,)``
    """
    # Build variable context: name → column array (fixed for the run)
    var_context: dict[str, np.ndarray] = {
        name: X_train[:, i] for i, name in enumerate(var_names)
    }

    def _evaluate(individual) -> tuple[float]:
        try:
            # 1. Dimensional Penalty (Fail Fast)
            dim_dist = 0.0
            if target_dim is not None and context_dims is not None:
                try:
                    evaluated_dim = evaluate_dim_at(individual, 0, context_dims)
                    dim_dist = evaluated_dim.distance(target_dim)
                except DimensionalViolation:
                    return (1e18,)

            # 2. Constant Optimization via CG
            const_nodes = [node for node in individual if node.arity == 0 and node.name not in var_context]
            if optimize_constants and const_nodes:
                x0 = []
                for n in const_nodes:
                    try:
                        x0.append(float(n.value))
                    except (AttributeError, TypeError, ValueError):
                        try:
                            x0.append(float(n.name))
                        except (TypeError, ValueError):
                            x0.append(1.0)
                x0 = np.array(x0, dtype=float)

                def objective(params):
                    for node, p in zip(const_nodes, params):
                        node.value = p
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            y_p = eval_tree(individual, var_context)
                        if np.isscalar(y_p):
                            y_p = np.full_like(y_train, float(y_p), dtype=float)
                        else:
                            y_p = np.asarray(y_p, dtype=float)
                        if not np.all(np.isfinite(y_p)):
                            return 1e18
                        mse_obj = float(np.mean((y_train - y_p) ** 2))
                        return mse_obj if np.isfinite(mse_obj) else 1e18
                    except Exception:
                        return 1e18

                try:
                    res = minimize(
                        objective,
                        x0,
                        method='CG',
                        options={'maxiter': 5, 'gtol': 1e-6, 'disp': False}
                    )
                    best_params = res.x
                except Exception:
                    best_params = x0

                # Write optimized constants back to tree
                for node, p in zip(const_nodes, best_params):
                    node.value = p

            # 3. Final MSE Evaluation
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y_pred = eval_tree(individual, var_context)

            if np.isscalar(y_pred):
                y_pred = np.full_like(y_train, float(y_pred), dtype=float)
            else:
                y_pred = np.asarray(y_pred, dtype=float)

            if not np.all(np.isfinite(y_pred)):
                return (1e18,)

            mse = float(np.mean((y_train - y_pred) ** 2))
            if not np.isfinite(mse):
                return (1e18,)

            fitness = mse + lambda_penalty * dim_dist
            return (fitness,)

        except Exception:
            return (1e18,)

    return _evaluate


# ======================================================================
# 5. Semantic Similarity Crossover (cxSSC)
# ======================================================================

def _eval_subtree_semantics(
    subtree_nodes: list,
    pset: gp.PrimitiveSet,
    sample_context: dict[str, np.ndarray],
) -> np.ndarray | None:
    """
    Numerically evaluate a subtree prefix-list on ``sample_context``.

    Uses eval_tree() on a temporary PrimitiveTree — NO constant fitting,
    NO gp.compile, NO X_train access.  This is the data-leakage guard.

    Returns a 1-D float array, or None on error / non-finite output.
    """
    try:
        tmp_tree = gp.PrimitiveTree(subtree_nodes)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vals = eval_tree(tmp_tree, sample_context)

        if np.isscalar(vals):
            n = next(iter(sample_context.values())).shape[0]
            vals = np.full(n, float(vals))
        else:
            vals = np.asarray(vals, dtype=float)

        if not np.all(np.isfinite(vals)):
            return None
        return vals
    except Exception:
        return None


def cxSSC(
    ind1: Any,
    ind2: Any,
    pset: gp.PrimitiveSet,
    var_names: list[str],
    sample_ranges: dict[str, tuple[float, float]] | None = None,
    alpha: float = SSC_ALPHA,
    beta:  float = SSC_BETA,
    max_trials: int = SSC_MAX_TRIALS,
    n_points: int = SSC_N_POINTS,
) -> tuple[Any, Any]:
    """
    Semantic Similarity-based Crossover (Cohen et al.).

    Finds crossover pairs whose subtree outputs differ in (alpha, beta)
    on randomly sampled evaluation points, then swaps those subtrees.
    Falls back to ``gp.cxOnePoint`` if no pair found within max_trials.

    Data-Leakage Guard
    ------------------
    Semantics are evaluated on *random* sample points generated internally.
    X_train is never accessed here.  No constant fitting occurs.

    Height Limit
    ------------
    If an offspring height > MAX_HEIGHT after a swap, that offspring is
    reverted to its original parent.

    Parameters
    ----------
    ind1, ind2 : PrimitiveTree
        Individuals modified in-place (DEAP convention).
    pset : gp.PrimitiveSet
        Used to construct temporary trees for evaluation.
    var_names : list[str]
        Variable terminal names for the sample context.
    sample_ranges : dict or None
        Maps variable names to (lo, hi).  Defaults to (-3.0, 3.0).
    alpha, beta : float
        SSD acceptance window.
    max_trials : int
    n_points : int

    Returns
    -------
    (ind1, ind2)
    """
    if len(ind1) < 2 or len(ind2) < 2:
        return gp.cxOnePoint(ind1, ind2)

    rng = np.random.default_rng()
    if sample_ranges is None:
        sample_ranges = {n: (-3.0, 3.0) for n in var_names}

    sample_context: dict[str, np.ndarray] = {
        name: rng.uniform(*sample_ranges.get(name, (-3.0, 3.0)), size=n_points)
        for name in var_names
    }

    # Save originals for height-limit reversion
    orig1 = copy.copy(ind1)
    orig2 = copy.copy(ind2)

    for _ in range(max_trials):
        idx1 = random.randint(1, len(ind1) - 1)
        idx2 = random.randint(1, len(ind2) - 1)

        sl1 = ind1.searchSubtree(idx1)
        sl2 = ind2.searchSubtree(idx2)

        sub1_nodes = list(ind1)[sl1.start:sl1.stop]
        sub2_nodes = list(ind2)[sl2.start:sl2.stop]

        # ── Semantic evaluation (no X_train, no curve_fit) ────────────────
        sem1 = _eval_subtree_semantics(sub1_nodes, pset, sample_context)
        sem2 = _eval_subtree_semantics(sub2_nodes, pset, sample_context)

        if sem1 is None or sem2 is None:
            continue

        ssd = float(np.mean(np.abs(sem1 - sem2)))
        if not (alpha < ssd < beta):
            continue

        # ── Swap ──────────────────────────────────────────────────────────
        nodes1 = list(ind1)
        nodes2 = list(ind2)

        new_nodes1 = nodes1[:sl1.start] + sub2_nodes + nodes1[sl1.stop:]
        new_nodes2 = nodes2[:sl2.start] + sub1_nodes + nodes2[sl2.stop:]

        if not new_nodes1 or not new_nodes2:
            continue

        # Use explicit integer slice to avoid DEAP's slice.__start is None check
        ind1[0:len(ind1)] = new_nodes1
        ind2[0:len(ind2)] = new_nodes2

        # ── Length-limit reversion ────────────────────────────────────────
        if len(ind1) > MAX_TREE_NODES:
            ind1[0:len(ind1)] = list(orig1)
        if len(ind2) > MAX_TREE_NODES:
            ind2[0:len(ind2)] = list(orig2)

        # ── Fitness invalidation ──────────────────────────────────────────
        if list(ind1) != list(orig1) and ind1.fitness.valid:
            del ind1.fitness.values
        if list(ind2) != list(orig2) and ind2.fitness.valid:
            del ind2.fitness.values

        return ind1, ind2

    # ── Fallback ──────────────────────────────────────────────────────────
    return gp.cxOnePoint(ind1, ind2)
