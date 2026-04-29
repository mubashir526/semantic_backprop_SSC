"""
Physical Dimension Vector Algebra
==================================
Implements the 7-component SI dimension vector representation and the
distance metric from Equation 1 of Reissmann et al. (2025).

Vector index order (MUST match Table 3 of Reissmann et al.):
    [0] Mass        (kg)
    [1] Length      (m)
    [2] Time        (s)
    [3] Temperature (K)
    [4] Current     (A)
    [5] Amount      (mol)
    [6] Luminosity  (cd)

Distance metric (Equation 1):
    d_φ(φ_a, φ_b) = (1/7) * Σ( (φ_a - φ_b)² )

This module is pure physics/math — no DEAP or GEP chromosome imports.
"""

import numpy as np

# Number of SI base dimensions
N_SI_DIMS: int = 7

# Human-readable labels for debug / repr purposes
SI_LABELS = ("kg", "m", "s", "K", "A", "mol", "cd")


class Dimension:
    """
    Immutable 7-component SI dimension vector.

    Arithmetic dunder methods return NEW Dimension objects (no mutation).
    The vector layout matches Table 3 of Reissmann et al. (2025):
        index 0 → kg, 1 → m, 2 → s, 3 → K, 4 → A, 5 → mol, 6 → cd
    """

    __slots__ = ("vector",)

    # ------------------------------------------------------------------ #
    #  Construction
    # ------------------------------------------------------------------ #

    def __init__(self, vector):
        """
        Parameters
        ----------
        vector : array-like of length 7
            SI exponent values in the order [kg, m, s, K, A, mol, cd].
        """
        if len(vector) != N_SI_DIMS:
            raise ValueError(
                f"Dimension vector must have exactly {N_SI_DIMS} elements "
                f"([kg, m, s, K, A, mol, cd]), got {len(vector)}."
            )
        self.vector = np.array(vector, dtype=float)

    @staticmethod
    def dimensionless() -> "Dimension":
        """Return the dimensionless unit: φ = [0, 0, 0, 0, 0, 0, 0]."""
        return Dimension(np.zeros(N_SI_DIMS))

    # ------------------------------------------------------------------ #
    #  Dimension Algebra  (Table 1 forward rules, returning new objects)
    # ------------------------------------------------------------------ #

    def __add__(self, other: "Dimension") -> "Dimension":
        """
        Multiplication forward rule (Table 1, row 3):
            φ_{a·b} = φ_a + φ_b   (pointwise vector addition)

        NOTE: This is the *dimensional* add used in the forward rule for
        the product operator (c = a * b), NOT arithmetic addition of
        quantities.  For quantity addition/subtraction the dimensions must
        be checked for equality by DimensionRules.
        """
        return Dimension(self.vector + other.vector)

    def __sub__(self, other: "Dimension") -> "Dimension":
        """
        Division forward rule (Table 1, row 4):
            φ_{a/b} = φ_a − φ_b   (pointwise vector subtraction)
        """
        return Dimension(self.vector - other.vector)

    def __mul__(self, scalar: float) -> "Dimension":
        """
        Power rule (Table 1, row 5):
            φ_{a^n} = n * φ_a   (scalar multiplication)
        """
        return Dimension(self.vector * scalar)

    def __rmul__(self, scalar: float) -> "Dimension":
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> "Dimension":
        """
        Root rule (Table 1, row 6):
            φ_{√a} = φ_a / 2   (scalar division)
        """
        return Dimension(self.vector / scalar)

    # ------------------------------------------------------------------ #
    #  Distance Metric  (Equation 1, Reissmann et al.)
    # ------------------------------------------------------------------ #

    def distance(self, other: "Dimension") -> float:
        """
        Equation 1:  d_φ(φ_a, φ_b) = (1/7) · Σ( (φ_a − φ_b)² )

        This is the *mean squared* element-wise difference, NOT the
        Euclidean L2-norm.  Returns 0.0 for identical dimensions.
        """
        diff = self.vector - other.vector
        return float(np.sum(diff ** 2) / N_SI_DIMS)

    # ------------------------------------------------------------------ #
    #  Utilities
    # ------------------------------------------------------------------ #

    def is_dimensionless(self) -> bool:
        """Return True iff all exponents are (approximately) zero."""
        return bool(np.allclose(self.vector, 0.0))

    def __eq__(self, other) -> bool:
        if not isinstance(other, Dimension):
            return NotImplemented
        return bool(np.allclose(self.vector, other.vector))

    def __hash__(self) -> int:
        """
        Hash based on rounded vector so that numerically equivalent
        Dimension objects map to the same dictionary key.
        Rounding to 6 decimal places matches the spec requirement.
        """
        return hash(tuple(np.round(self.vector, 6).tolist()))

    def __repr__(self) -> str:
        parts = ", ".join(
            f"{lbl}^{int(e)}" if e == int(e) else f"{lbl}^{e}"
            for lbl, e in zip(SI_LABELS, self.vector)
            if e != 0.0
        )
        return f"Dimension([{', '.join(f'{v:g}' for v in self.vector)}])" \
               + (f"  # {parts}" if parts else "  # dimensionless")

    # ------------------------------------------------------------------ #
    #  Convenience constructors for common physical quantities
    # ------------------------------------------------------------------ #

    # The constants below are taken directly from the reference
    # backprop_imple/dimension.py to guarantee identical SI vectors.

    @staticmethod
    def mass() -> "Dimension":
        """kg  →  [1, 0, 0, 0, 0, 0, 0]"""
        return Dimension([1, 0, 0, 0, 0, 0, 0])

    @staticmethod
    def length() -> "Dimension":
        """m  →  [0, 1, 0, 0, 0, 0, 0]"""
        return Dimension([0, 1, 0, 0, 0, 0, 0])

    @staticmethod
    def time() -> "Dimension":
        """s  →  [0, 0, 1, 0, 0, 0, 0]"""
        return Dimension([0, 0, 1, 0, 0, 0, 0])

    @staticmethod
    def velocity() -> "Dimension":
        """m/s  →  [0, 1, -1, 0, 0, 0, 0]"""
        return Dimension([0, 1, -1, 0, 0, 0, 0])

    @staticmethod
    def acceleration() -> "Dimension":
        """m/s²  →  [0, 1, -2, 0, 0, 0, 0]"""
        return Dimension([0, 1, -2, 0, 0, 0, 0])

    @staticmethod
    def force() -> "Dimension":
        """N = kg·m/s²  →  [1, 1, -2, 0, 0, 0, 0]"""
        return Dimension([1, 1, -2, 0, 0, 0, 0])

    @staticmethod
    def energy() -> "Dimension":
        """J = kg·m²/s²  →  [1, 2, -2, 0, 0, 0, 0]"""
        return Dimension([1, 2, -2, 0, 0, 0, 0])
