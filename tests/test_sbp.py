"""
Test suite for src/sbp/engine.py
Run with: pytest tests/test_sbp.py -v

All tests use a self-contained DEAP pset and DimLibrary so there are
no dependencies on external data files.
"""
from __future__ import annotations

import sys
import os
import copy
import operator

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── DEAP imports ──────────────────────────────────────────────────────────────
from deap import gp, creator, base, tools

# ── Project imports ───────────────────────────────────────────────────────────
from src.physics.dimension import Dimension
from src.physics.dimension_rules import DimensionRules, DimensionalViolation
from src.physics.library import DimLibrary, Fragment
from src.sbp.engine import (
    splice_subtree,
    evaluate_dim_at,
    target_dim_for_left,
    target_dim_for_right,
    repair_individual,
    MAX_TREE_NODES,
    _EPSILON,
)


# ======================================================================
# Shared DEAP pset and library (built once per session)
# ======================================================================

def _safe_div(a, b):
    return a / b if b != 0.0 else 1e10

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


# Build a pset whose terminal names correspond to physical quantities
_PSET = gp.PrimitiveSet("MAIN", arity=0)
_PSET.addPrimitive(operator.add,  2, name="+")
_PSET.addPrimitive(operator.sub,  2, name="-")
_PSET.addPrimitive(operator.mul,  2, name="*")
_PSET.addPrimitive(_safe_div,     2, name="/")
_PSET.addPrimitive(_safe_sq,      1, name="sq")
_PSET.addPrimitive(_safe_sqrt,    1, name="sqrt")
_PSET.addPrimitive(_safe_sin,     1, name="sin")
_PSET.addPrimitive(_safe_cos,     1, name="cos")
_PSET.addTerminal(1.0, name="mass")        # kg
_PSET.addTerminal(1.0, name="length")      # m
_PSET.addTerminal(1.0, name="time")        # s
_PSET.addTerminal(1.0, name="velocity")    # m/s
_PSET.addTerminal(1.0, name="const_1")     # dimensionless ERC

# Register DEAP Individual type (guard against duplicate registration)
if "FitnessMin_SBP" not in dir(creator):
    creator.create("FitnessMin_SBP", base.Fitness, weights=(-1.0,))
if "Individual_SBP" not in dir(creator):
    creator.create("Individual_SBP", gp.PrimitiveTree,
                   fitness=creator.FitnessMin_SBP)

_CONTEXT_DIMS: dict[str, Dimension] = {
    "mass":     Dimension.mass(),       # [1,0,0,0,0,0,0]
    "length":   Dimension.length(),     # [0,1,0,0,0,0,0]
    "time":     Dimension.time(),       # [0,0,1,0,0,0,0]
    "velocity": Dimension.velocity(),   # [0,1,-1,0,0,0,0]
    # "const_1" absent → dimensionless fallback
}

_LIB: DimLibrary | None = None

def get_lib() -> DimLibrary:
    global _LIB
    if _LIB is None:
        _LIB = DimLibrary(_PSET, _CONTEXT_DIMS, max_depth=2, max_size=100_000)
    return _LIB


# ── Helper: build an Individual from a string expression ──────────────────────

def make_ind(expr_str: str) -> "creator.Individual_SBP":
    """
    Parse a prefix-notation expression string into a PrimitiveTree.
    Example: make_ind("+(mass velocity)")
    """
    tree = gp.PrimitiveTree.from_string(expr_str, _PSET)
    ind  = creator.Individual_SBP(tree)
    return ind

def make_ind_nodes(nodes: list) -> "creator.Individual_SBP":
    """Build an Individual directly from a node list."""
    ind = creator.Individual_SBP(nodes)
    # Stamp a valid fitness so we can check invalidation later
    ind.fitness.values = (99.0,)
    return ind


# ── Physical dimension shortcuts ──────────────────────────────────────────────

MASS  = Dimension.mass()
LEN   = Dimension.length()
TIME  = Dimension.time()
VEL   = Dimension.velocity()
DL    = Dimension.dimensionless()
FORCE = Dimension.force()


# ======================================================================
# Helper: look up a DEAP node by name from pset
# ======================================================================

def _prim(name: str):
    """Retrieve a Primitive object by name from _PSET."""
    for prims in _PSET.primitives.values():
        for p in prims:
            if p.name == name:
                return p
    raise KeyError(f"Primitive '{name}' not found in pset")

def _term(name: str):
    """Retrieve a Terminal object by name from _PSET."""
    for terms in _PSET.terminals.values():
        for t in terms:
            if t.name == name:
                return t
    raise KeyError(f"Terminal '{name}' not found in pset")


# ======================================================================
# TASK 2 — Test Suite
# ======================================================================

class TestSpliceSubtree:
    """
    Prefix Integrity: splice_subtree returns a valid DEAP list and does
    NOT mutate the original.
    """

    def test_splice_terminal_in_place(self):
        """Replace a leaf terminal with another terminal."""
        # Tree: +(mass velocity) = [+, mass, velocity]  (3 nodes)
        ind   = make_ind("+(mass velocity)")
        # Replace pos=1 (mass) with [length]
        sl    = ind.searchSubtree(1)
        repl  = [_term("length")]
        new_t = splice_subtree(ind, sl, repl)

        assert new_t[1].name == "length", "Splice did not replace terminal"
        assert ind[1].name   == "mass",   "Original tree was mutated"

    def test_splice_subtree_with_operator(self):
        """Replace a terminal with a unary fragment [sq, mass]."""
        ind  = make_ind("+(mass velocity)")
        sl   = ind.searchSubtree(1)          # [mass]
        repl = [_prim("sq"), _term("mass")]
        new_t = splice_subtree(ind, sl, repl)

        assert new_t[1].name == "sq",   "Expected sq at pos 1"
        assert new_t[2].name == "mass", "Expected mass at pos 2"
        assert len(new_t) == 4,         "Tree length should be 4"

    def test_splice_does_not_mutate_original(self):
        """The original individual must be bit-for-bit unchanged."""
        ind     = make_ind("*(mass velocity)")
        before  = [n.name for n in ind]
        sl      = ind.searchSubtree(2)
        repl    = [_term("length")]
        _       = splice_subtree(ind, sl, repl)
        after   = [n.name for n in ind]
        assert before == after, "splice_subtree mutated the original tree"

    def test_splice_returns_new_object(self):
        """Result must be a different object from the input."""
        ind   = make_ind("+(mass velocity)")
        sl    = ind.searchSubtree(1)
        repl  = [_term("length")]
        new_t = splice_subtree(ind, sl, repl)
        assert new_t is not ind

    def test_splice_bloat_guard_raises(self):
        """Splice that would exceed MAX_TREE_NODES must raise ValueError."""
        ind  = make_ind("+(mass velocity)")
        sl   = ind.searchSubtree(1)
        # Build a replacement list longer than MAX_TREE_NODES
        big_repl = [_term("mass")] * (MAX_TREE_NODES + 5)
        with pytest.raises(ValueError, match="limit is"):
            splice_subtree(ind, sl, big_repl)

    def test_splice_invalidates_fitness(self):
        """Result tree must have fitness.valid == False after splice."""
        ind = make_ind_nodes([_prim("+"), _term("mass"), _term("velocity")])
        assert ind.fitness.valid, "Precondition: fitness should be valid"
        sl    = ind.searchSubtree(1)
        repl  = [_term("length")]
        new_t = splice_subtree(ind, sl, repl)
        assert not new_t.fitness.valid, "Fitness should be invalidated after splice"


class TestEvaluateDimAt:
    """
    Dimensional evaluation including the boundary / right-child logic.
    """

    def test_single_terminal_mass(self):
        ind = make_ind("mass")
        assert evaluate_dim_at(ind, 0, _CONTEXT_DIMS) == MASS

    def test_single_terminal_erc(self):
        """const_1 absent from context → dimensionless."""
        ind = make_ind("const_1")
        assert evaluate_dim_at(ind, 0, _CONTEXT_DIMS) == DL

    def test_unary_sq_mass(self):
        """sq(mass) → [2,0,0,0,0,0,0]"""
        ind    = make_ind("sq(mass)")
        result = evaluate_dim_at(ind, 0, _CONTEXT_DIMS)
        assert result == Dimension([2, 0, 0, 0, 0, 0, 0])

    def test_binary_mul_mass_acc(self):
        """
        *(mass sq(velocity)) should give kg·m²/s²  [1,2,-2,0,0,0,0]

        Tree prefix: [*, mass, sq, velocity]  — 4 nodes
          pos 0 = *   (arity 2)
          pos 1 = mass (arity 0) — left child
          pos 2 = sq   (arity 1) — right child
          pos 3 = velocity (arity 0)
        """
        ind    = make_ind("*(mass sq(velocity))")
        result = evaluate_dim_at(ind, 0, _CONTEXT_DIMS)
        # mass [1,0,0,…] + sq(velocity)[0,2,-2,…] = [1,2,-2,…]
        assert result == Dimension([1, 2, -2, 0, 0, 0, 0])

    def test_right_child_boundary(self):
        """
        Boundary Logic: evaluate_dim_at must use searchSubtree to skip
        the left subtree and find the right child correctly.

        Tree: /(sq(mass) velocity)
        Prefix: [/, sq, mass, velocity]  (4 nodes)
          pos 0 = /         arity 2
          pos 1 = sq        arity 1  ← left child
          pos 2 = mass      arity 0
          pos 3 = velocity  arity 0  ← right child (left subtree was [sq,mass]=2 nodes)

        right_pos = searchSubtree(1).stop = 3  ✓
        Result dim: sq(mass)/velocity = [2,0,…] − [0,1,-1,…] = [2,-1,1,…]
        """
        ind    = make_ind("/(sq(mass) velocity)")
        result = evaluate_dim_at(ind, 0, _CONTEXT_DIMS)
        expected = Dimension([2, -1, 1, 0, 0, 0, 0])
        assert result == expected, (
            f"Right-child boundary test failed: {result} != {expected}"
        )

    def test_right_child_position_directly(self):
        """
        Explicitly verify that the right child of a binary op at pos=0
        is found at searchSubtree(left_pos).stop, not at pos+2.

        Tree: +(sq(mass) velocity) — 4 nodes
        left subtree: [sq, mass] occupies pos 1..2
        right child is at pos 3, NOT pos 2.
        """
        ind = make_ind("+(sq(mass) velocity)")
        # Left subtree slice
        left_sl  = ind.searchSubtree(1)      # slice(1, 3)
        right_pos = left_sl.stop             # 3
        assert ind[right_pos].name == "velocity", (
            f"Right child should be 'velocity' at pos {right_pos}, "
            f"got '{ind[right_pos].name}'"
        )

    def test_nested_binary(self):
        """*(*(mass velocity) time) → MASS + VEL + TIME vectors."""
        ind    = make_ind("*(*(mass velocity) time)")
        result = evaluate_dim_at(ind, 0, _CONTEXT_DIMS)
        # mass*velocity = [1,1,-1,…], then *time = [1,1,0,…]
        expected = Dimension([1, 1, 0, 0, 0, 0, 0])
        assert result == expected


class TestTargetDimForChildren:
    """Verify backward rules are correctly dispatched."""

    def test_target_left_mul_uses_right(self):
        """
        *(mass velocity): target_dim_for_left at pos=0 with target=Force
        should give Force - velocity_dim = [1,0,1,…].
        """
        # *(mass velocity) — left=mass, right=velocity, op=*
        ind    = make_ind("*(mass velocity)")
        target = FORCE   # [1,1,-2,…]
        tl     = target_dim_for_left(ind, 0, target, _CONTEXT_DIMS)
        # backward_left('*', FORCE, right=VEL) = FORCE_vec - VEL_vec
        expected = DimensionRules.backward_left('*', target, right=VEL)
        assert tl == expected

    def test_target_right_div(self):
        """
        /(mass velocity): target_dim_for_right at pos=0 with target=DL
        should give left - target = mass - DL = mass.
        """
        ind    = make_ind("/(mass velocity)")
        target = DL
        tr     = target_dim_for_right(ind, 0, target, _CONTEXT_DIMS)
        expected = DimensionRules.backward_right('/', target, left=MASS)
        assert tr == expected

    def test_target_left_add(self):
        """For +, target_left = target_dim itself."""
        ind    = make_ind("+(mass mass)")
        target = MASS
        tl     = target_dim_for_left(ind, 0, target, _CONTEXT_DIMS)
        assert tl == MASS

    def test_target_right_unary_is_none(self):
        """Unary operators have no right child → returns None."""
        ind    = make_ind("sq(mass)")
        result = target_dim_for_right(ind, 0, MASS, _CONTEXT_DIMS)
        assert result is None

    def test_target_left_sqrt(self):
        """sqrt backward: target_left = 2 * target."""
        ind    = make_ind("sqrt(mass)")
        target = Dimension([0, 1, 0, 0, 0, 0, 0])  # length (sqrt of area)
        tl     = target_dim_for_left(ind, 0, target, _CONTEXT_DIMS)
        assert tl == Dimension([0, 2, 0, 0, 0, 0, 0])


class TestRepairIndividual:
    """
    End-to-end repair tests.
    """

    def test_repair_violation_mass_plus_velocity(self):
        """
        Repair Success: +(mass velocity) has a DimensionalViolation.
        After repair_individual the result should be dimensionally compliant
        with target=MASS.
        """
        lib = get_lib()
        ind = make_ind("+(mass velocity)")

        # Precondition: tree currently violates the forward '+' rule
        with pytest.raises(DimensionalViolation):
            evaluate_dim_at(ind, 0, _CONTEXT_DIMS)

        repaired, modified = repair_individual(
            ind, MASS, lib, _CONTEXT_DIMS, max_attempts=5
        )
        assert modified, "repair_individual should have modified the tree"
        # The repaired tree must now be compliant
        result_dim = evaluate_dim_at(repaired, 0, _CONTEXT_DIMS)
        assert result_dim.distance(MASS) <= _EPSILON, (
            f"Repaired tree still violates target: {result_dim} vs {MASS}"
        )

    def test_repair_already_compliant_returns_false(self):
        """A tree that already satisfies target_dim must not be modified."""
        lib = get_lib()
        ind = make_ind("mass")
        repaired, modified = repair_individual(
            ind, MASS, lib, _CONTEXT_DIMS, max_attempts=5
        )
        assert not modified, "Compliant tree should not be modified"
        assert repaired is ind or [n.name for n in repaired] == [n.name for n in ind]

    def test_repair_returns_new_object_when_modified(self):
        """When modified=True the returned object must be different from input."""
        lib = get_lib()
        ind = make_ind("+(mass velocity)")
        repaired, modified = repair_individual(
            ind, MASS, lib, _CONTEXT_DIMS, max_attempts=5
        )
        if modified:
            assert repaired is not ind, "Modified result must be a new object"

    def test_stale_fitness_check(self):
        """
        Stale Fitness: any tree returned with modified=True must have
        fitness.valid == False.
        """
        lib = get_lib()
        # Stamp valid fitness on the input
        ind = make_ind_nodes([_prim("+"), _term("mass"), _term("velocity")])
        assert ind.fitness.valid

        repaired, modified = repair_individual(
            ind, MASS, lib, _CONTEXT_DIMS, max_attempts=5
        )
        if modified:
            assert not repaired.fitness.valid, (
                "Modified tree must have invalidated fitness"
            )

    def test_repair_velocity_target(self):
        """Repair a dimensionless tree towards target=velocity."""
        lib = get_lib()
        ind = make_ind("const_1")    # dimensionless

        repaired, modified = repair_individual(
            ind, VEL, lib, _CONTEXT_DIMS, max_attempts=5
        )
        if modified:
            result_dim = evaluate_dim_at(repaired, 0, _CONTEXT_DIMS)
            assert result_dim.distance(VEL) <= _EPSILON


class TestBloatGuard:
    """
    Verify that repairs which would exceed MAX_TREE_NODES are rejected.
    """

    def test_bloat_splice_rejected(self):
        """
        splice_subtree must raise ValueError when new tree would exceed
        MAX_TREE_NODES.
        """
        ind      = make_ind("+(mass velocity)")
        sl       = ind.searchSubtree(1)
        too_many = [_term("mass")] * (MAX_TREE_NODES + 10)
        with pytest.raises(ValueError):
            splice_subtree(ind, sl, too_many)

    def test_repair_respects_max_frag_nodes(self):
        """
        When max_frag_nodes=1, repair may only use single-node fragments.
        The repair either succeeds with a terminal replacement or makes
        no change — it must never crash.
        """
        lib = get_lib()
        ind = make_ind("+(mass velocity)")
        # Should not raise regardless of outcome
        repaired, modified = repair_individual(
            ind, MASS, lib, _CONTEXT_DIMS,
            max_attempts=5, max_frag_nodes=1,
        )
        # If modified, tree must be valid DEAP (has a length)
        assert len(repaired) >= 1

    def test_max_tree_nodes_constant_is_positive(self):
        assert MAX_TREE_NODES > 0


class TestEdgeCases:
    """Additional boundary and robustness tests."""

    def test_single_node_tree(self):
        """A single terminal tree evaluates correctly."""
        ind    = make_ind("mass")
        result = evaluate_dim_at(ind, 0, _CONTEXT_DIMS)
        assert result == MASS

    def test_deep_unary_chain(self):
        """sq(sq(mass)) → dimension [4,0,0,0,0,0,0]."""
        ind    = make_ind("sq(sq(mass))")
        result = evaluate_dim_at(ind, 0, _CONTEXT_DIMS)
        assert result == Dimension([4, 0, 0, 0, 0, 0, 0])

    def test_splice_preserves_length_single_for_single(self):
        """Replacing one terminal with another keeps length constant."""
        ind   = make_ind("+(mass velocity)")
        sl    = ind.searchSubtree(2)   # velocity
        repl  = [_term("time")]
        new_t = splice_subtree(ind, sl, repl)
        assert len(new_t) == len(ind)

    def test_evaluate_dim_at_sub_position(self):
        """evaluate_dim_at should work on interior subtrees."""
        # +(sq(mass) velocity): pos=1 is sq(mass)
        ind    = make_ind("+(sq(mass) velocity)")
        result = evaluate_dim_at(ind, 1, _CONTEXT_DIMS)
        assert result == Dimension([2, 0, 0, 0, 0, 0, 0])

    def test_repair_does_not_mutate_input(self):
        """repair_individual must never modify the input individual."""
        lib    = get_lib()
        ind    = make_ind("+(mass velocity)")
        before = [n.name for n in ind]
        repair_individual(ind, MASS, lib, _CONTEXT_DIMS)
        after  = [n.name for n in ind]
        assert before == after, "repair_individual mutated the input individual"
