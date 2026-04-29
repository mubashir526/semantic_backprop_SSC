"""
Benchmark Metrics
==================
Evaluation metrics for Feynman symbolic regression following
Reissmann et al. (2025) Sections 4.1 and Equation 6.

Three metrics are implemented:

1. r2_score(y_true, y_pred)
   Standard coefficient of determination on held-out test data.

2. expression_complexity(expr_str, var_names)
   Node count AFTER sympy.simplify() — prevents bloated trees from
   receiving unfair complexity penalties.

3. symbolic_solution_rate(found_expr, true_expr, var_names, timeout)
   A trial is "Correct" (returns True) if:
       f - f̂  simplifies to a constant, OR
       f / f̂  simplifies to a constant.
   Both checks use a 5-second SymPy timeout.

Note on DEAP tree strings
-------------------------
DEAP's str(individual) produces prefix notation: ``*(*(m, g), z)``.
Because ``*``, ``+``, ``-``, ``/`` are Python syntax tokens, SymPy's
``parse_expr`` cannot handle them directly.  Use ``deap_to_sympy()``
to convert a DEAP PrimitiveTree to a SymPy expression before calling
``is_symbolic_solution`` or ``expression_complexity``.
"""

from __future__ import annotations

import signal
import warnings
from typing import Any

import numpy as np


# ======================================================================
# 1. R² Score
# ======================================================================

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Coefficient of determination R² on test data.

        R² = 1 - SS_res / SS_tot

    Returns -inf if SS_tot == 0 (degenerate case) or if y_pred contains
    non-finite values.

    Parameters
    ----------
    y_true : np.ndarray, shape (N,)
    y_pred : np.ndarray, shape (N,)

    Returns
    -------
    float in (-∞, 1]
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if not np.all(np.isfinite(y_pred)):
        return float("-inf")

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))

    if ss_tot < 1e-30:
        return float("-inf")

    return 1.0 - ss_res / ss_tot


# ======================================================================
# 2. Expression Complexity
# ======================================================================

def expression_complexity(
    expr_str: str,
    var_names: list[str] | None = None,
    timeout: int = 5,
) -> int:
    """
    Count the total nodes (terminals + primitives) in a symbolic
    expression AFTER simplification via sympy.simplify().

    Simplification prevents bloated GP trees (e.g. x + x - x which
    equals x but has 4 nodes) from artificially inflating complexity.

    Parameters
    ----------
    expr_str : str
        A SymPy-parseable expression string.
    var_names : list[str] or None
        Variable names to declare as SymPy symbols.  If None they are
        inferred automatically by SymPy.
    timeout : int
        Seconds allowed for simplification (default 5).

    Returns
    -------
    int
        Number of nodes in the simplified expression tree.
        Returns a large value (10_000) if simplification times out or
        raises an exception.
    """
    import sympy

    def _count_nodes(expr) -> int:
        """Recursively count all nodes in a SymPy expression tree."""
        if not expr.args:
            return 1   # leaf: Symbol, Number, etc.
        return 1 + sum(_count_nodes(a) for a in expr.args)

    # ── Parse ─────────────────────────────────────────────────────────────
    try:
        if var_names:
            local_dict = {name: sympy.Symbol(name) for name in var_names}
        else:
            local_dict = {}

        parsed = sympy.parse_expr(
            expr_str,
            local_dict=local_dict,
            transformations="all",
        )
    except Exception:
        return 10_000  # unparseable → maximum penalty

    # ── Simplify with timeout ──────────────────────────────────────────────
    simplified = _simplify_with_timeout(parsed, timeout)
    return _count_nodes(simplified)


def _simplify_with_timeout(expr, timeout: int):
    """
    Call sympy.simplify(expr) with a hard wall-clock timeout.

    On POSIX systems uses SIGALRM.  On Windows (no SIGALRM) falls back
    to returning the un-simplified expression after logging a warning.
    """
    import sympy

    try:
        # ── POSIX: SIGALRM timeout ─────────────────────────────────────
        _handler_available = hasattr(signal, "SIGALRM")

        if _handler_available:
            def _handler(signum, frame):
                raise TimeoutError("SymPy simplification timed out")

            old_handler = signal.signal(signal.SIGALRM, _handler)
            signal.alarm(timeout)
            try:
                result = sympy.simplify(expr)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            return result

        else:
            # ── Windows: threading-based timeout ──────────────────────
            import threading

            result_holder = [expr]   # default: return un-simplified
            exception_holder = [None]

            def _worker():
                try:
                    result_holder[0] = sympy.simplify(expr)
                except Exception as e:
                    exception_holder[0] = e

            t = threading.Thread(target=_worker, daemon=True)
            t.start()
            t.join(timeout=timeout)
            if t.is_alive():
                warnings.warn(
                    "SymPy simplification exceeded timeout — returning "
                    "un-simplified expression.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                return expr
            if exception_holder[0] is not None:
                return expr
            return result_holder[0]

    except Exception:
        return expr


# ======================================================================
# 3. Symbolic Solution Rate  (Equation 6)
# ======================================================================

def is_symbolic_solution(
    found_expr_str: str,
    true_expr_str: str,
    var_names: list[str] | None = None,
    timeout: int = 5,
) -> bool:
    """
    Determine whether ``found_expr_str`` is symbolically equivalent to
    ``true_expr_str`` per Equation 6 of Reissmann et al. (2025).

    A trial is "Correct" if EITHER:
        (a) true_expr - found_expr  simplifies to a numeric constant, OR
        (b) true_expr / found_expr  simplifies to a numeric constant.

    Both checks are performed with a ``timeout``-second SymPy budget.

    Parameters
    ----------
    found_expr_str : str
        The expression found by GP.
    true_expr_str : str
        The ground-truth formula.
    var_names : list[str] or None
        Variable names used when building the SymPy symbol table.
    timeout : int
        Per-check timeout in seconds.

    Returns
    -------
    bool
        True if the expression satisfies either correctness criterion.
    """
    import sympy

    if var_names:
        local_dict = {name: sympy.Symbol(name) for name in var_names}
    else:
        local_dict = {}

    # ── Parse both expressions ─────────────────────────────────────────────
    try:
        f_true  = sympy.parse_expr(true_expr_str,  local_dict=local_dict, transformations="all")
        f_found = sympy.parse_expr(found_expr_str, local_dict=local_dict, transformations="all")
    except Exception:
        return False

    # ── Check (a): difference is constant ────────────────────────────────
    try:
        diff        = f_true - f_found
        diff_simple = _simplify_with_timeout(diff, timeout)
        if diff_simple.is_number:
            return True
    except Exception:
        pass

    # ── Check (b): ratio is constant ─────────────────────────────────────
    try:
        ratio        = f_true / f_found
        ratio_simple = _simplify_with_timeout(ratio, timeout)
        if ratio_simple.is_number:
            return True
    except Exception:
        pass

    return False


# ======================================================================
# 4. Composite trial metrics
# ======================================================================

# ======================================================================
# 5. DEAP → SymPy expression converter
# ======================================================================

# Map from DEAP operator name → SymPy function name / infix string
_OP_MAP = {
    "+":    "Add",
    "-":    "(lambda a,b: a-b)",
    "*":    "Mul",
    "/":    "(lambda a,b: a/b)",
    "sq":   "(lambda x: x**2)",
    "sqrt": "sqrt",
    "sin":  "sin",
    "cos":  "cos",
    "log":  "log",
    "exp":  "exp",
}


def deap_to_sympy(individual, var_names: list[str] | None = None):
    """
    Convert a DEAP PrimitiveTree to a SymPy expression.

    DEAP's str() uses operator-first prefix notation that conflicts with
    Python syntax (e.g. ``*(a,b)``).  This function directly walks the
    flat node list and builds a SymPy expression tree.

    Parameters
    ----------
    individual : deap.gp.PrimitiveTree
    var_names : list[str] or None
        If provided, these names are declared as SymPy Symbols.

    Returns
    -------
    sympy.Expr or None on error.
    """
    import sympy

    local_syms = {}
    if var_names:
        for name in var_names:
            local_syms[name] = sympy.Symbol(name)

    # Safe namespace for eval
    _ns = {
        "Add": sympy.Add,
        "Mul": sympy.Mul,
        "sqrt": sympy.sqrt,
        "sin":  sympy.sin,
        "cos":  sympy.cos,
        "log":  sympy.log,
        "exp":  sympy.exp,
    }
    _ns.update(local_syms)

    def _build(pos: int):
        node  = individual[pos]
        arity = node.arity

        if arity == 0:
            name = node.name
            if name in local_syms:
                return local_syms[name], pos + 1
            # ERC / numeric constant
            try:
                return sympy.Float(float(node.value)), pos + 1
            except (AttributeError, TypeError, ValueError):
                try:
                    return sympy.Float(float(name)), pos + 1
                except (TypeError, ValueError):
                    return sympy.Integer(1), pos + 1

        if arity == 1:
            child, next_pos = _build(pos + 1)
            op = node.name
            if op == "sq":
                return child ** 2, next_pos
            elif op == "sqrt":
                return sympy.sqrt(child), next_pos
            elif op == "sin":
                return sympy.sin(child), next_pos
            elif op == "cos":
                return sympy.cos(child), next_pos
            elif op == "log":
                return sympy.log(child), next_pos
            elif op == "exp":
                return sympy.exp(child), next_pos
            else:
                return child, next_pos

        # Binary
        left,  mid_pos  = _build(pos + 1)
        right_pos_start = individual.searchSubtree(pos + 1).stop
        right, next_pos = _build(right_pos_start)
        op = node.name
        if op == "+":
            return left + right, next_pos
        elif op == "-":
            return left - right, next_pos
        elif op == "*":
            return left * right, next_pos
        elif op == "/":
            return left / right, next_pos
        else:
            return left + right, next_pos

    try:
        expr, _ = _build(0)
        return expr
    except Exception:
        return None


def compute_trial_metrics(
    individual,
    pset,
    var_names: list[str],
    X_test: np.ndarray,
    y_test: np.ndarray,
    true_formula: str,
    sympy_timeout: int = 5,
) -> dict[str, Any]:
    """
    Compute all three metrics for a single GP individual.

    Parameters
    ----------
    individual : deap.gp.PrimitiveTree
    pset : deap.gp.PrimitiveSet
    var_names : list[str]
    X_test : np.ndarray
    y_test : np.ndarray
    true_formula : str
    sympy_timeout : int

    Returns
    -------
    dict with keys: 'r2', 'complexity', 'solved'
    """
    from src.evolution.utils import eval_tree

    # ── Numerical predictions ──────────────────────────────────────────────
    var_context = {name: X_test[:, i] for i, name in enumerate(var_names)}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_pred = eval_tree(individual, var_context)

        if np.isscalar(y_pred):
            y_pred = np.full_like(y_test, float(y_pred))
        else:
            y_pred = np.asarray(y_pred, dtype=float)
    except Exception:
        y_pred = np.full_like(y_test, np.nan)

    # ── R² ─────────────────────────────────────────────────────────────────
    r2 = r2_score(y_test, y_pred)

    # ── Convert DEAP tree to SymPy expression ──────────────────────────────
    # str(individual) produces DEAP prefix notation (*(a,b)) which cannot be
    # parsed by SymPy directly because '*' is a Python syntax token.
    # deap_to_sympy() walks the flat node list and builds a SymPy tree.
    sympy_expr = deap_to_sympy(individual, var_names)

    if sympy_expr is not None:
        import sympy
        found_str = str(sympy_expr)
    else:
        found_str = str(individual)   # fallback (may fail SymPy parsing)

    # ── Complexity ─────────────────────────────────────────────────────────
    # Pass the SymPy-compatible string so simplify() can process it.
    complexity = expression_complexity(found_str, var_names, timeout=sympy_timeout)

    # ── Symbolic solution rate ─────────────────────────────────────────────
    solved = is_symbolic_solution(
        found_str, true_formula, var_names, timeout=sympy_timeout
    )

    return {"r2": r2, "complexity": complexity, "solved": solved, "best_found": found_str}
