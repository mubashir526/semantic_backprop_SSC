"""
Dimension Inference Rules
==========================
Implements the forward and backward dimension inference rules from
Table 1, Algorithm 2, and Algorithm 3 of Reissmann et al. (2025).

This module is the *dispatcher* layer: given an operator name and
dimension contexts it computes the resulting or required dimension.

All methods are static — DimensionRules holds no state.
No DEAP or GEP chromosome imports are used here.

Operator semantics
------------------
Binary operators  : '+', '-', '*', '/'
Unary operators   : 'sq' (square, n=2), 'sqrt', 'sin', 'cos', 'log', 'exp'

Three-case split for '*' and '/'
---------------------------------
When both children are unknown (neither left nor right dimension is
provided), the target dimension is split evenly between them:
    Case 3 (*) : φ_a = φ_c / 2,  φ_b = φ_c − φ_a
    Case 3 (/) : φ_a = φ_c / 2,  φ_b = φ_a − φ_c  (so that φ_a/φ_b = φ_c)

This is Algorithm 2/3 logic from the paper.
"""

from __future__ import annotations
import numpy as np
from src.physics.dimension import Dimension, N_SI_DIMS


# ====================================================================== #
#  Custom exception
# ====================================================================== #

class DimensionalViolation(Exception):
    """
    Raised when a dimensional constraint is violated:
      - '+' or '-' applied to operands whose dimension vectors differ.
      - Transcendental function (sin, cos, log, exp) applied to a
        dimensioned (non-dimensionless) quantity.
    """


# ====================================================================== #
#  DimensionRules dispatcher
# ====================================================================== #

class DimensionRules:
    """
    Static-method dispatcher for Table 1 (Reissmann et al. 2025).

    All forward rules compute the output dimension from input dimensions.
    All backward rules compute the required input dimension from the
    output (target) dimension and whichever sibling dimension is known.
    """

    KNOWN_OPERATORS = frozenset({'+', '-', '*', '/', 'sq', 'sqrt',
                                  'sin', 'cos', 'log', 'exp'})

    # ------------------------------------------------------------------ #
    #  Forward rules  (Table 1 columns "Operation" → "Dimension of c")
    # ------------------------------------------------------------------ #

    @staticmethod
    def forward(op: str, left: Dimension, right: Dimension | None = None) -> Dimension:
        """
        Compute the output dimension of ``op(left, right)``.

        Parameters
        ----------
        op    : operator name, must be in KNOWN_OPERATORS.
        left  : dimension of the left (or sole) operand.
        right : dimension of the right operand (binary operators only).

        Returns
        -------
        Dimension — output dimension.

        Raises
        ------
        DimensionalViolation
            If '+' or '-' operands have mismatched dimensions, or if a
            transcendental function receives a dimensioned argument.
        ValueError
            If op is unknown or required operands are missing.
        """
        if op not in DimensionRules.KNOWN_OPERATORS:
            raise ValueError(f"Unknown operator '{op}'. "
                             f"Expected one of {DimensionRules.KNOWN_OPERATORS}.")

        # ── Binary: addition / subtraction ──────────────────────────────
        if op in ('+', '-'):
            if right is None:
                raise ValueError(f"Operator '{op}' requires a right operand.")
            if left.distance(right) > 1e-9:
                raise DimensionalViolation(
                    f"Dimensional mismatch for '{op}': "
                    f"left={left}, right={right}, "
                    f"distance={left.distance(right):.2e} > 1e-9."
                )
            # Both operands must share the same dimension; result inherits it.
            return Dimension(left.vector.copy())

        # ── Binary: multiplication ───────────────────────────────────────
        if op == '*':
            if right is None:
                raise ValueError("Operator '*' requires a right operand.")
            # φ_c = φ_a + φ_b  (Table 1 row 3)
            return Dimension(left.vector + right.vector)

        # ── Binary: division ─────────────────────────────────────────────
        if op == '/':
            if right is None:
                raise ValueError("Operator '/' requires a right operand.")
            # φ_c = φ_a − φ_b  (Table 1 row 4)
            return Dimension(left.vector - right.vector)

        # ── Unary: square  (fixed n=2) ───────────────────────────────────
        if op == 'sq':
            # φ_c = 2 · φ_a  (Table 1 row 5, n=2)
            return Dimension(left.vector * 2.0)

        # ── Unary: square root ───────────────────────────────────────────
        if op == 'sqrt':
            # φ_c = φ_a / 2  (Table 1 row 6)
            return Dimension(left.vector / 2.0)

        # ── Unary: transcendental (sin, cos, log, exp) ───────────────────
        if op in ('sin', 'cos', 'log', 'exp'):
            if not left.is_dimensionless():
                raise DimensionalViolation(
                    f"Transcendental function '{op}' requires a dimensionless "
                    f"argument, got {left}."
                )
            # Result is always dimensionless
            return Dimension.dimensionless()

        # Should never reach here given KNOWN_OPERATORS guard at the top.
        raise ValueError(f"Unhandled operator '{op}'.")

    # ------------------------------------------------------------------ #
    #  Backward rules — left child
    # ------------------------------------------------------------------ #

    @staticmethod
    def backward_left(op: str,
                      target: Dimension,
                      right: Dimension | None = None) -> Dimension:
        """
        Compute the dimension the LEFT child must have so that
        ``op(left, right) == target``.

        For binary operators, ``right`` must be provided (the current/known
        dimension of the right operand).  For unary operators ``right`` is
        ignored.

        Algorithm 2/3 three-case logic for '*' and '/':
          Case 1 — right is known → compute left.
          Case 2 — right is None  → split target evenly (Case 3).
        """
        if op not in DimensionRules.KNOWN_OPERATORS:
            raise ValueError(f"Unknown operator '{op}'.")

        # ── '+' and '-' ──────────────────────────────────────────────────
        if op in ('+', '-'):
            # φ_a = φ_c  (backward rule: both children must equal target)
            return Dimension(target.vector.copy())

        # ── '*'  (Algorithm 2) ───────────────────────────────────────────
        if op == '*':
            if right is not None:
                # Case 1: right is known → φ_a = φ_c − φ_b
                return Dimension(target.vector - right.vector)
            else:
                # Case 3: split → φ_a = φ_c / 2
                return Dimension(target.vector / 2.0)

        # ── '/'  (Algorithm 3) ───────────────────────────────────────────
        if op == '/':
            if right is not None:
                # Case 1: right (divisor) is known → φ_a = φ_c + φ_b
                # Because φ_c = φ_a − φ_b  ⟹  φ_a = φ_c + φ_b
                return Dimension(target.vector + right.vector)
            else:
                # Case 3: split → φ_a = φ_c / 2
                return Dimension(target.vector / 2.0)

        # ── 'sq' ─────────────────────────────────────────────────────────
        if op == 'sq':
            # φ_a = φ_c / 2
            return Dimension(target.vector / 2.0)

        # ── 'sqrt' ───────────────────────────────────────────────────────
        if op == 'sqrt':
            # φ_a = 2 · φ_c
            return Dimension(target.vector * 2.0)

        # ── Transcendental ───────────────────────────────────────────────
        if op in ('sin', 'cos', 'log', 'exp'):
            # Input must be dimensionless regardless of target
            return Dimension.dimensionless()

        raise ValueError(f"Unhandled operator '{op}'.")

    # ------------------------------------------------------------------ #
    #  Backward rules — right child
    # ------------------------------------------------------------------ #

    @staticmethod
    def backward_right(op: str,
                       target: Dimension,
                       left: Dimension | None = None) -> Dimension:
        """
        Compute the dimension the RIGHT child must have so that
        ``op(left, right) == target``.

        Unary operators have no right child — calling this for unary ops
        raises ValueError.

        Algorithm 2/3 three-case logic for '*' and '/':
          Case 2 — left is known → compute right.
          Case 3 — left is None  → split target evenly.

        CRITICAL derivation for division:
            φ_c = φ_a − φ_b  →  φ_b = φ_a − φ_c
        Any other formula is mathematically incorrect.
        """
        if op not in DimensionRules.KNOWN_OPERATORS:
            raise ValueError(f"Unknown operator '{op}'.")

        # ── '+' and '-' ──────────────────────────────────────────────────
        if op in ('+', '-'):
            # φ_b = φ_c  (both children must equal target)
            return Dimension(target.vector.copy())

        # ── '*'  (Algorithm 2) ───────────────────────────────────────────
        if op == '*':
            if left is not None:
                # Case 2: left is known → φ_b = φ_c − φ_a
                return Dimension(target.vector - left.vector)
            else:
                # Case 3: split → φ_b = φ_c − (φ_c/2) = φ_c/2
                half = target.vector / 2.0
                return Dimension(target.vector - half)

        # ── '/'  (Algorithm 3) ───────────────────────────────────────────
        if op == '/':
            if left is not None:
                # Case 2: left (dividend) is known → φ_b = φ_a − φ_c
                # CRITICAL: φ_c = φ_a − φ_b  ⟹  φ_b = φ_a − φ_c
                return Dimension(left.vector - target.vector)
            else:
                # Case 3: split
                # Choose φ_a = φ_c / 2, then φ_b = φ_a − φ_c = −φ_c/2
                half_a = target.vector / 2.0
                return Dimension(half_a - target.vector)

        # ── Unary operators have no right child ──────────────────────────
        if op in ('sq', 'sqrt', 'sin', 'cos', 'log', 'exp'):
            raise ValueError(
                f"Operator '{op}' is unary and has no right child. "
                "Use backward_left() instead."
            )

        raise ValueError(f"Unhandled operator '{op}'.")

    # ------------------------------------------------------------------ #
    #  Convenience wrappers matching reference naming convention
    # ------------------------------------------------------------------ #
    # These are thin aliases so any code ported from backprop_gep that
    # calls the verbose form (forward_add, backward_mul, etc.) continues
    # to work without modification.

    @staticmethod
    def forward_add(dim_a: Dimension, dim_b: Dimension) -> Dimension:
        return DimensionRules.forward('+', dim_a, dim_b)

    @staticmethod
    def forward_sub(dim_a: Dimension, dim_b: Dimension) -> Dimension:
        return DimensionRules.forward('-', dim_a, dim_b)

    @staticmethod
    def forward_mul(dim_a: Dimension, dim_b: Dimension) -> Dimension:
        return DimensionRules.forward('*', dim_a, dim_b)

    @staticmethod
    def forward_div(dim_a: Dimension, dim_b: Dimension) -> Dimension:
        return DimensionRules.forward('/', dim_a, dim_b)

    @staticmethod
    def forward_sq(dim_a: Dimension) -> Dimension:
        return DimensionRules.forward('sq', dim_a)

    @staticmethod
    def forward_root(dim_a: Dimension) -> Dimension:
        return DimensionRules.forward('sqrt', dim_a)

    @staticmethod
    def backward_add(dim_c: Dimension):
        """Returns (φ_a, φ_b) — both equal to dim_c."""
        return (Dimension(dim_c.vector.copy()),
                Dimension(dim_c.vector.copy()))

    @staticmethod
    def backward_sub(dim_c: Dimension):
        """Returns (φ_a, φ_b) — both equal to dim_c."""
        return (Dimension(dim_c.vector.copy()),
                Dimension(dim_c.vector.copy()))

    @staticmethod
    def backward_mul(dim_c: Dimension,
                     dim_a: Dimension | None = None,
                     dim_b: Dimension | None = None) -> Dimension:
        """
        Algorithm 2 backward multiplication (reference-compatible signature).
        Provide exactly one of dim_a or dim_b.
        """
        if dim_b is not None and dim_a is None:
            return DimensionRules.backward_left('*', dim_c, right=dim_b)
        elif dim_a is not None and dim_b is None:
            return DimensionRules.backward_right('*', dim_c, left=dim_a)
        else:
            raise ValueError("Provide exactly one of dim_a or dim_b.")

    @staticmethod
    def backward_div(dim_c: Dimension,
                     dim_a: Dimension | None = None,
                     dim_b: Dimension | None = None) -> Dimension:
        """
        Algorithm 3 backward division (reference-compatible signature).

        - If dim_b (divisor) is known:  φ_a = φ_c + φ_b
        - If dim_a (dividend) is known: φ_b = φ_a − φ_c  ← CRITICAL

        Provide exactly one of dim_a or dim_b.
        """
        if dim_b is not None and dim_a is None:
            return DimensionRules.backward_left('/', dim_c, right=dim_b)
        elif dim_a is not None and dim_b is None:
            return DimensionRules.backward_right('/', dim_c, left=dim_a)
        else:
            raise ValueError("Provide exactly one of dim_a or dim_b.")

    @staticmethod
    def backward_sq(dim_c: Dimension) -> Dimension:
        return DimensionRules.backward_left('sq', dim_c)

    @staticmethod
    def backward_root(dim_c: Dimension) -> Dimension:
        return DimensionRules.backward_left('sqrt', dim_c)

    @staticmethod
    def backward_trig(dim_c: Dimension) -> Dimension:
        """sin/cos backward: input must be dimensionless."""
        return Dimension.dimensionless()

    @staticmethod
    def backward_log_exp(dim_c: Dimension) -> Dimension:
        """log/exp backward: input must be dimensionless."""
        return Dimension.dimensionless()
