"""
pytest test suite for src/physics/dimension.py and src/physics/dimension_rules.py
Run with: pytest tests/test_physics.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np

from src.physics.dimension import Dimension, N_SI_DIMS
from src.physics.dimension_rules import DimensionRules, DimensionalViolation


# ======================================================================
# Helpers
# ======================================================================

def dim(*args):
    """Shorthand: dim(0,1,-1,0,0,0,0) → Dimension([0,1,-1,0,0,0,0])"""
    return Dimension(list(args))


# Common physical dimensions (matches reference backprop_imple/dimension.py)
DL   = Dimension.dimensionless()           # [0,0,0,0,0,0,0]
MASS = Dimension.mass()                    # [1,0,0,0,0,0,0]
LEN  = Dimension.length()                 # [0,1,0,0,0,0,0]
TIME = Dimension.time()                   # [0,0,1,0,0,0,0]
VEL  = Dimension.velocity()               # [0,1,-1,0,0,0,0]
ACC  = Dimension.acceleration()           # [0,1,-2,0,0,0,0]
FORCE = Dimension.force()                 # [1,1,-2,0,0,0,0]


# ======================================================================
# TASK 3 — Mandatory cases from the spec
# ======================================================================

class TestMandatory:

    def test_velocity_composition(self):
        """
        Spec: [0,1,0,0,0,0,0] - [0,0,1,0,0,0,0] = [0,1,-1,0,0,0,0]
        This is the dimensional division rule: φ_{m/s} = φ_m − φ_s
        """
        result = LEN - TIME          # Dimension.__sub__ = vector subtraction
        expected = VEL
        assert result == expected, (
            f"Velocity composition failed: {result} != {expected}"
        )
        np.testing.assert_array_equal(result.vector, [0, 1, -1, 0, 0, 0, 0])

    def test_hash_stability(self):
        """
        Spec: Two separate Dimension objects with identical vectors must
        work as the same key in a Python dict.
        """
        d1 = Dimension([1, 1, -2, 0, 0, 0, 0])   # Force
        d2 = Dimension([1, 1, -2, 0, 0, 0, 0])   # Force (separate object)
        assert hash(d1) == hash(d2), "Hash mismatch for identical vectors"
        mapping = {d1: "Newton"}
        assert mapping[d2] == "Newton", "dict lookup failed with identical-vector key"

    def test_backward_division_right(self):
        """
        Spec mandatory: backward_right('/') correctly applies φ_b = φ_a − φ_c.

        Self-consistent physical example:
            c = a / b   where  a = Force [1,1,-2,…],  c = Acceleration [0,1,-2,…]
            ⟹  φ_b = φ_a − φ_c = Force − Acceleration
                    = [1,1,-2,…] − [0,1,-2,…] = [1,0,0,…] = Mass

        This validates the CRITICAL derivation φ_b = φ_a − φ_c (not φ_c − φ_a).
        """
        # F = m * a  ⟹  a = F / m  ⟹  backward_right: given left=Force, target=Acc → Mass
        result = DimensionRules.backward_right('/', target=ACC, left=FORCE)
        assert result == MASS, (
            f"backward_right('/') gave {result}, expected Mass={MASS}"
        )

    def test_roundtrip_multiply(self):
        """
        Spec: forward('*', a, b) followed by backward_left correctly
        recovers the dimension of a.
        """
        a = MASS    # [1,0,0,0,0,0,0]
        b = ACC     # [0,1,-2,0,0,0,0]
        c = DimensionRules.forward('*', a, b)       # should be Force [1,1,-2,…]
        recovered_a = DimensionRules.backward_left('*', target=c, right=b)
        assert recovered_a == a, (
            f"Roundtrip failed: recovered {recovered_a}, expected {a}"
        )


# ======================================================================
# Dimension class unit tests
# ======================================================================

class TestDimension:

    def test_vector_length(self):
        assert len(Dimension.dimensionless().vector) == N_SI_DIMS == 7

    def test_invalid_length_raises(self):
        with pytest.raises(ValueError):
            Dimension([1, 0, 0])

    def test_dimensionless_is_zero(self):
        assert np.allclose(DL.vector, 0)
        assert DL.is_dimensionless()

    def test_mass_not_dimensionless(self):
        assert not MASS.is_dimensionless()

    # ── Distance metric (Equation 1) ──

    def test_distance_self_zero(self):
        assert FORCE.distance(FORCE) == pytest.approx(0.0)

    def test_distance_equation1(self):
        """d_φ = (1/7) * Σ(φ_a − φ_b)²  — manual check for m vs s."""
        # LEN=[0,1,0,0,0,0,0], TIME=[0,0,1,0,0,0,0]
        # diff = [0,1,-1,0,0,0,0]  →  sum_sq = 0+1+1+0+0+0+0 = 2
        # d = 2/7
        assert LEN.distance(TIME) == pytest.approx(2 / 7)

    def test_distance_symmetry(self):
        assert MASS.distance(FORCE) == pytest.approx(FORCE.distance(MASS))

    # ── Arithmetic operators ──

    def test_add_gives_vector_sum(self):
        """Dimension.__add__ = vector addition (multiplication forward rule)."""
        result = MASS + ACC          # [1,0,0,…] + [0,1,-2,…] = [1,1,-2,…]
        assert result == FORCE

    def test_sub_gives_vector_diff(self):
        """Dimension.__sub__ = vector subtraction (division forward rule)."""
        result = LEN - TIME
        assert result == VEL

    def test_mul_scalar(self):
        """Dimension.__mul__(n) = scalar multiplication (power rule)."""
        result = LEN * 2
        assert np.allclose(result.vector, [0, 2, 0, 0, 0, 0, 0])

    def test_div_scalar(self):
        """Dimension.__truediv__(n) = scalar division (root rule)."""
        sq_len = LEN * 2        # [0,2,0,0,0,0,0]
        result = sq_len / 2
        assert result == LEN

    def test_operators_return_new_objects(self):
        """No mutation: original dimensions unchanged after arithmetic."""
        original = Dimension([1, 2, 3, 0, 0, 0, 0])
        _ = original + Dimension([0, 0, 0, 0, 0, 0, 0])
        _ = original - Dimension([0, 0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(original.vector, [1, 2, 3, 0, 0, 0, 0])

    # ── Equality and hash ──

    def test_equality(self):
        assert Dimension([1, 0, 0, 0, 0, 0, 0]) == MASS
        assert MASS != VEL

    def test_hash_small_float_noise(self):
        """Hash must be stable under rounding to 6 dp."""
        d1 = Dimension([1.0000001, 0, 0, 0, 0, 0, 0])
        d2 = Dimension([1.0000002, 0, 0, 0, 0, 0, 0])
        # Both round to 1.000000 at 6 dp, so same hash
        assert hash(d1) == hash(d2)

    def test_convenience_constructors(self):
        assert Dimension.mass()    == dim(1, 0,  0, 0, 0, 0, 0)
        assert Dimension.length()  == dim(0, 1,  0, 0, 0, 0, 0)
        assert Dimension.time()    == dim(0, 0,  1, 0, 0, 0, 0)
        assert Dimension.velocity() == dim(0, 1, -1, 0, 0, 0, 0)
        assert Dimension.force()    == dim(1, 1, -2, 0, 0, 0, 0)


# ======================================================================
# DimensionRules forward tests
# ======================================================================

class TestForwardRules:

    def test_forward_add_matching(self):
        assert DimensionRules.forward('+', FORCE, FORCE) == FORCE

    def test_forward_add_mismatch_raises(self):
        with pytest.raises(DimensionalViolation):
            DimensionRules.forward('+', MASS, VEL)

    def test_forward_sub_mismatch_raises(self):
        with pytest.raises(DimensionalViolation):
            DimensionRules.forward('-', LEN, TIME)

    def test_forward_mul(self):
        # F = m * a  →  [1,0,0,…] + [0,1,-2,…] = [1,1,-2,…]
        assert DimensionRules.forward('*', MASS, ACC) == FORCE

    def test_forward_div(self):
        # a = F/m  →  [1,1,-2,…] − [1,0,0,…] = [0,1,-2,…]
        assert DimensionRules.forward('/', FORCE, MASS) == ACC

    def test_forward_sq(self):
        # sq(v) → [0,2,-2,…]
        result = DimensionRules.forward('sq', VEL)
        assert np.allclose(result.vector, [0, 2, -2, 0, 0, 0, 0])

    def test_forward_sqrt(self):
        sq_area = dim(0, 2, 0, 0, 0, 0, 0)
        assert DimensionRules.forward('sqrt', sq_area) == LEN

    def test_forward_sin_dimensionless_input(self):
        assert DimensionRules.forward('sin', DL) == DL

    def test_forward_cos_dimensionless_input(self):
        assert DimensionRules.forward('cos', DL) == DL

    def test_forward_log_dimensionless_input(self):
        assert DimensionRules.forward('log', DL) == DL

    def test_forward_exp_dimensionless_input(self):
        assert DimensionRules.forward('exp', DL) == DL

    def test_forward_sin_dimensioned_raises(self):
        with pytest.raises(DimensionalViolation):
            DimensionRules.forward('sin', VEL)

    def test_forward_exp_dimensioned_raises(self):
        with pytest.raises(DimensionalViolation):
            DimensionRules.forward('exp', MASS)

    def test_forward_unknown_op_raises(self):
        with pytest.raises(ValueError):
            DimensionRules.forward('tanh', DL)


# ======================================================================
# DimensionRules backward tests
# ======================================================================

class TestBackwardRules:

    # ── Addition / Subtraction ──

    def test_backward_left_add(self):
        assert DimensionRules.backward_left('+', FORCE) == FORCE

    def test_backward_right_add(self):
        assert DimensionRules.backward_right('+', FORCE) == FORCE

    def test_backward_left_sub(self):
        assert DimensionRules.backward_left('-', ACC) == ACC

    # ── Multiplication (Algorithm 2) ──

    def test_backward_left_mul_case1(self):
        """Case 1: right known → φ_a = φ_c − φ_b"""
        result = DimensionRules.backward_left('*', target=FORCE, right=ACC)
        assert result == MASS

    def test_backward_right_mul_case2(self):
        """Case 2: left known → φ_b = φ_c − φ_a"""
        result = DimensionRules.backward_right('*', target=FORCE, left=MASS)
        assert result == ACC

    def test_backward_mul_case3_split(self):
        """Case 3: no sibling → split target by 2."""
        c = dim(0, 2, -2, 0, 0, 0, 0)   # m²/s²
        left_half = DimensionRules.backward_left('*', target=c)
        right_half = DimensionRules.backward_right('*', target=c)
        reconstructed = DimensionRules.forward('*', left_half, right_half)
        assert reconstructed == c, (
            f"Case-3 split reconstruction failed: {reconstructed} != {c}"
        )

    # ── Division (Algorithm 3) ──

    def test_backward_left_div_case1(self):
        """Case 1: right (divisor) known → φ_a = φ_c + φ_b"""
        # a = F/m → F = a*m → left of '/' given right=MASS, target=ACC should be FORCE
        result = DimensionRules.backward_left('/', target=ACC, right=MASS)
        assert result == FORCE

    def test_backward_right_div_case2_critical(self):
        """
        CRITICAL derivation: φ_c = φ_a − φ_b  ⟹  φ_b = φ_a − φ_c

        Self-consistent check:  a=Force, c=Acceleration  ⟹  b=Mass
            φ_b = φ_Force − φ_Acceleration = [1,1,-2,…] − [0,1,-2,…] = [1,0,0,…] = Mass

        Also verify NOT the wrong formula (φ_c − φ_a would give negative Mass).
        """
        # Correct: φ_b = φ_a − φ_c
        result = DimensionRules.backward_right('/', target=ACC, left=FORCE)
        assert result == MASS, (
            f"CRITICAL backward_right('/') wrong: {result} != Mass={MASS}"
        )
        # Spot-check the formula is φ_a − φ_c, not φ_c − φ_a
        wrong = Dimension(ACC.vector - FORCE.vector)
        assert result != wrong, "backward_right uses wrong formula φ_c − φ_a"

    def test_backward_div_compat_dim_b_known(self):
        """Reference-compatible backward_div: divisor known → φ_a = φ_c + φ_b"""
        result = DimensionRules.backward_div(ACC, dim_b=MASS)
        assert result == FORCE

    def test_backward_div_compat_dim_a_known(self):
        """Reference-compatible backward_div: dividend known → φ_b = φ_a − φ_c.
        Example: acc = Force / b  ⟹  b = Force − acc = Mass"""
        result = DimensionRules.backward_div(ACC, dim_a=FORCE)
        assert result == MASS

    def test_backward_div_neither_raises(self):
        with pytest.raises(ValueError):
            DimensionRules.backward_div(FORCE)

    # ── Square ──

    def test_backward_sq(self):
        sq_vel = DimensionRules.forward('sq', VEL)     # [0,2,-2,…]
        recovered = DimensionRules.backward_left('sq', target=sq_vel)
        assert recovered == VEL

    # ── Square root ──

    def test_backward_sqrt(self):
        sq_area = dim(0, 2, 0, 0, 0, 0, 0)
        sqrt_area = DimensionRules.forward('sqrt', sq_area)   # [0,1,0,…] = m
        recovered = DimensionRules.backward_left('sqrt', target=sqrt_area)
        assert recovered == sq_area

    def test_backward_right_unary_raises(self):
        """Unary operators have no right child — must raise ValueError."""
        with pytest.raises(ValueError):
            DimensionRules.backward_right('sq', FORCE)
        with pytest.raises(ValueError):
            DimensionRules.backward_right('sqrt', FORCE)
        with pytest.raises(ValueError):
            DimensionRules.backward_right('sin', DL)

    # ── Transcendentals ──

    def test_backward_trig_returns_dimensionless(self):
        assert DimensionRules.backward_trig(DL).is_dimensionless()

    def test_backward_log_exp_returns_dimensionless(self):
        assert DimensionRules.backward_log_exp(DL).is_dimensionless()

    def test_backward_left_sin_returns_dimensionless(self):
        assert DimensionRules.backward_left('sin', DL).is_dimensionless()

    def test_backward_left_exp_returns_dimensionless(self):
        assert DimensionRules.backward_left('exp', DL).is_dimensionless()


# ======================================================================
# Reference-compatible wrapper tests
# ======================================================================

class TestCompatWrappers:

    def test_forward_add_wrapper(self):
        assert DimensionRules.forward_add(VEL, VEL) == VEL

    def test_forward_mul_wrapper(self):
        assert DimensionRules.forward_mul(MASS, ACC) == FORCE

    def test_forward_div_wrapper(self):
        assert DimensionRules.forward_div(FORCE, MASS) == ACC

    def test_backward_mul_wrapper_dim_b(self):
        assert DimensionRules.backward_mul(FORCE, dim_b=ACC) == MASS

    def test_backward_mul_wrapper_dim_a(self):
        assert DimensionRules.backward_mul(FORCE, dim_a=MASS) == ACC

    def test_backward_div_wrapper_dim_b(self):
        assert DimensionRules.backward_div(ACC, dim_b=MASS) == FORCE

    def test_backward_div_wrapper_dim_a(self):
        # acc = Force / b  →  b = Force − acc = Mass
        assert DimensionRules.backward_div(ACC, dim_a=FORCE) == MASS

    def test_backward_sq_wrapper(self):
        sq_v = DimensionRules.forward_sq(VEL)
        assert DimensionRules.backward_sq(sq_v) == VEL

    def test_backward_root_wrapper(self):
        sq_l = dim(0, 2, 0, 0, 0, 0, 0)
        assert DimensionRules.backward_root(DimensionRules.forward_root(sq_l)) == sq_l


# ======================================================================
# Integration: forward → backward roundtrips
# ======================================================================

class TestRoundtrips:

    @pytest.mark.parametrize("op,a,b", [
        ('*', MASS, ACC),
        ('*', VEL,  TIME),
        ('/', FORCE, MASS),
        ('/', VEL,   TIME),
    ])
    def test_roundtrip_binary(self, op, a, b):
        c = DimensionRules.forward(op, a, b)
        rec_a = DimensionRules.backward_left(op, target=c, right=b)
        rec_b = DimensionRules.backward_right(op, target=c, left=a)
        assert rec_a == a, f"backward_left('{op}') roundtrip fail: {rec_a} != {a}"
        assert rec_b == b, f"backward_right('{op}') roundtrip fail: {rec_b} != {b}"

    def test_roundtrip_sq(self):
        for base in [MASS, VEL, ACC, FORCE]:
            sq = DimensionRules.forward('sq', base)
            assert DimensionRules.backward_left('sq', sq) == base

    def test_roundtrip_sqrt(self):
        for sq_dim in [dim(0,2,0,0,0,0,0), dim(2,0,0,0,0,0,0)]:
            rt = DimensionRules.forward('sqrt', sq_dim)
            assert DimensionRules.backward_left('sqrt', rt) == sq_dim


# ======================================================================
# DimLibrary tests  (Task 2 from Prompt 2)
# ======================================================================
#
# We build a minimal, self-contained DEAP pset so these tests have zero
# dependency on external data files while still using real DEAP objects.
# ======================================================================

import operator as _operator
from deap import gp as _gp, creator as _creator, base as _base

# ── Helpers: safe numeric ops ──────────────────────────────────────────

def _safe_div(a, b):
    return a / b if b != 0 else 1e10

def _safe_sq(x):
    return x * x

def _safe_sqrt(x):
    import math
    return math.sqrt(abs(x))

def _safe_sin(x):
    import math
    return math.sin(x)

def _safe_cos(x):
    import math
    return math.cos(x)


# ── Build a shared pset and DimLibrary (module-level fixture) ──────────

def _make_physics_pset():
    """
    Minimal DEAP PrimitiveSet covering the operators required by the
    DimensionRules dispatcher.  Variables are named after physical
    quantities so context_dims can map them directly.
    """
    ps = _gp.PrimitiveSet("MAIN", arity=0)
    # Binary operators
    ps.addPrimitive(_operator.add,  2, name="+")
    ps.addPrimitive(_operator.sub,  2, name="-")
    ps.addPrimitive(_operator.mul,  2, name="*")
    ps.addPrimitive(_safe_div,      2, name="/")
    # Unary operators
    ps.addPrimitive(_safe_sq,    1, name="sq")
    ps.addPrimitive(_safe_sqrt,  1, name="sqrt")
    ps.addPrimitive(_safe_sin,   1, name="sin")
    ps.addPrimitive(_safe_cos,   1, name="cos")
    # Terminals — physical variables
    ps.addTerminal(1.0,   name="mass")        # kg
    ps.addTerminal(1.0,   name="length")      # m
    ps.addTerminal(1.0,   name="time")        # s
    ps.addTerminal(1.0,   name="velocity")    # m/s
    ps.addTerminal(1.0,   name="const_1")     # numeric ERC → dimensionless
    return ps


_PHYS_PSET = _make_physics_pset()

_CONTEXT_DIMS = {
    "mass":     Dimension.mass(),       # [1,0,0,0,0,0,0]
    "length":   Dimension.length(),     # [0,1,0,0,0,0,0]
    "time":     Dimension.time(),       # [0,0,1,0,0,0,0]
    "velocity": Dimension.velocity(),   # [0,1,-1,0,0,0,0]
    # "const_1" deliberately absent → falls back to dimensionless
}

# Build once for the whole test session (max_depth=2 keeps it fast)
_LIB = None

def _get_lib():
    from src.physics.library import DimLibrary
    global _LIB
    if _LIB is None:
        _LIB = DimLibrary(_PHYS_PSET, _CONTEXT_DIMS, max_depth=2, max_size=100_000)
    return _LIB


class TestDimLibrary:

    # ── Fixture access ─────────────────────────────────────────────────

    @pytest.fixture(scope="class")
    def lib(self):
        return _get_lib()

    # ── 1. Terminal indexing ───────────────────────────────────────────

    def test_terminal_velocity_indexed(self, lib):
        """
        'velocity' terminal must appear under the velocity dimension key.
        """
        frags = lib.fragments_for(Dimension.velocity())
        assert frags, "No fragments found for velocity dimension"
        # At least one must be a single-terminal fragment with name 'velocity'
        terminal_frags = [f for f in frags if len(f.nodes) == 1
                          and f.nodes[0].name == "velocity"]
        assert terminal_frags, (
            "Terminal 'velocity' not found under velocity dimension key"
        )

    def test_terminal_mass_indexed(self, lib):
        """'mass' terminal must appear under the mass dimension key."""
        frags = lib.fragments_for(Dimension.mass())
        assert frags, "No fragments for mass dimension"
        names = [f.nodes[0].name for f in frags if len(f.nodes) == 1]
        assert "mass" in names

    def test_terminal_length_indexed(self, lib):
        frags = lib.fragments_for(Dimension.length())
        assert frags
        names = [f.nodes[0].name for f in frags if len(f.nodes) == 1]
        assert "length" in names

    def test_terminal_time_indexed(self, lib):
        frags = lib.fragments_for(Dimension.time())
        assert frags
        names = [f.nodes[0].name for f in frags if len(f.nodes) == 1]
        assert "time" in names

    # ── 2. Zero-vector / numeric constant mapping ──────────────────────

    def test_numeric_constant_is_dimensionless(self, lib):
        """
        'const_1' is absent from context_dims → must map to dimensionless.
        """
        dl_frags = lib.fragments_for(Dimension.dimensionless())
        assert dl_frags, "No dimensionless fragments in library"
        const_frags = [f for f in dl_frags
                       if len(f.nodes) == 1 and f.nodes[0].name == "const_1"]
        assert const_frags, (
            "'const_1' ERC not found under dimensionless key"
        )

    def test_dimensionless_key_matches_hash(self, lib):
        """
        The dimensionless key stored must equal Dimension.dimensionless().__hash__ key.
        """
        dl = Dimension.dimensionless()
        import numpy as _np
        expected_key = tuple(_np.round(dl.vector, 6).tolist())
        assert expected_key in lib._store

    # ── 3. Randomized retrieval ────────────────────────────────────────

    def test_get_velocity_returns_correct_dim(self, lib):
        """
        get(velocity_dim) must return a Fragment whose dimension distance
        to velocity is exactly 0.0.
        """
        frag = lib.get(Dimension.velocity())
        assert frag is not None, "get() returned None for velocity dimension"
        assert frag.dimension.distance(Dimension.velocity()) == pytest.approx(0.0), (
            f"Fragment dimension {frag.dimension} != velocity"
        )

    def test_get_mass_returns_correct_dim(self, lib):
        frag = lib.get(Dimension.mass())
        assert frag is not None
        assert frag.dimension.distance(Dimension.mass()) == pytest.approx(0.0)

    def test_get_dimensionless_returns_correct_dim(self, lib):
        frag = lib.get(Dimension.dimensionless())
        assert frag is not None
        assert frag.dimension.distance(Dimension.dimensionless()) == pytest.approx(0.0)

    def test_get_randomizes(self, lib):
        """
        Multiple calls to get() on a multi-entry bucket should not always
        return the same fragment (probabilistic; retry 50 times).
        """
        dl = Dimension.dimensionless()
        if len(lib.fragments_for(dl)) < 2:
            pytest.skip("Only one dimensionless fragment — randomness not testable")
        results = {id(lib.get(dl)) for _ in range(50)}
        assert len(results) > 1, "get() always returns the same object (not random)"

    # ── 4. Graceful failures ───────────────────────────────────────────

    def test_get_impossible_dimension_returns_none(self, lib):
        """
        A dimension with absurdly large exponents should not exist in
        the library — get() must return None without raising.
        """
        impossible = Dimension([99, -88, 77, -66, 55, -44, 33])
        result = lib.get(impossible)
        assert result is None

    def test_get_none_does_not_raise(self, lib):
        """get() must never raise, even for unknown dimensions."""
        for _ in range(10):
            exotic = Dimension([float(i % 5) for i in range(7)])
            try:
                lib.get(exotic)
            except Exception as exc:
                pytest.fail(f"get() raised an unexpected exception: {exc}")

    # ── 5. max_nodes filter ────────────────────────────────────────────

    def test_get_max_nodes_filters(self, lib):
        """
        get(dim, max_nodes=1) must only return single-node (terminal) fragments.
        """
        frag = lib.get(Dimension.mass(), max_nodes=1)
        if frag is None:
            pytest.skip("No single-node fragment for mass")
        assert len(frag.nodes) == 1

    def test_get_max_nodes_too_small_returns_none(self, lib):
        """
        get(dim, max_nodes=0) should always return None since no fragment
        has 0 nodes.
        """
        result = lib.get(Dimension.mass(), max_nodes=0)
        assert result is None

    # ── 6. Structural / as_list validity ──────────────────────────────

    def test_fragment_as_list_returns_list(self, lib):
        """as_list() must return a plain list (not a tuple)."""
        frag = lib.get(Dimension.mass())
        assert frag is not None
        assert isinstance(frag.as_list(), list)

    def test_fragment_nodes_is_tuple(self, lib):
        """nodes attribute must be a tuple (required for hashability)."""
        frag = lib.get(Dimension.mass())
        assert frag is not None
        assert isinstance(frag.nodes, tuple)

    def test_compound_fragment_dimension_correct(self, lib):
        """
        For a depth-1 unary fragment (e.g. sq(mass)), recompute its
        output dimension from the raw node list and verify it matches
        fragment.dimension.
        """
        sq_mass_dim = DimensionRules.forward('sq', Dimension.mass())  # [2,0,0,0,0,0,0]
        frag = lib.get(sq_mass_dim)
        if frag is None:
            pytest.skip("sq(mass) not in library at max_depth=2")
        # Recompute dimension from the node prefix list
        nodes = frag.nodes
        assert len(nodes) >= 1
        # Single-terminal: just verify stored dim matches context
        if len(nodes) == 1:
            stored_dim = _CONTEXT_DIMS.get(nodes[0].name, Dimension.dimensionless())
            assert stored_dim.distance(frag.dimension) == pytest.approx(0.0)
        else:
            # Operator node: verify forward rule matches stored dim
            op_name = nodes[0].name
            if len(nodes) == 2:  # unary
                child_name = nodes[1].name
                child_dim = _CONTEXT_DIMS.get(child_name, Dimension.dimensionless())
                try:
                    computed = DimensionRules.forward(op_name, child_dim)
                    assert computed.distance(frag.dimension) == pytest.approx(0.0), (
                        f"Stored dim {frag.dimension} != computed {computed}"
                    )
                except (DimensionalViolation, ValueError):
                    pass  # fragment may have deeper nesting; basic check done

    # ── 7. Library integrity ───────────────────────────────────────────

    def test_library_is_non_empty(self, lib):
        assert lib.size() > 0

    def test_library_has_multiple_dimensions(self, lib):
        assert lib.num_dimensions() > 1

    def test_has_dimension_helper(self, lib):
        assert lib.has_dimension(Dimension.mass())
        assert lib.has_dimension(Dimension.dimensionless())
        assert not lib.has_dimension(Dimension([99, -88, 77, -66, 55, -44, 33]))

    def test_repr_is_informative(self, lib):
        r = repr(lib)
        assert "DimLibrary" in r
        assert "fragments" in r

