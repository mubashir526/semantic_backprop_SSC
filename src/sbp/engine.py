"""
Semantic Backpropagation Engine
=================================
Adapts Algorithm 1 from Reissmann et al. (2025) to work natively with
DEAP PrimitiveTree objects.

Key differences from the GEP reference (backprop_gep/backpropagation.py):

  - DEAP trees are flat prefix lists, not linked-node structures.
    All subtree boundaries are obtained via tree.searchSubtree(pos),
    which returns a slice object [start:stop).

  - Right-child position for arity-2 nodes is:
        right_pos = tree.searchSubtree(pos + 1).stop
    This is the ONLY correct way to skip the entire left subtree.

  - Splices produce a NEW PrimitiveTree via copy + slice assignment
    and immediately invalidate fitness (del new_tree.fitness.values).

  - The first successful modification returns immediately because any
    splice shifts all subsequent positions, making stored indices stale.

  - Max-size bloat guard: repairs that would grow the tree beyond
    MAX_TREE_NODES are rejected before the splice is applied.

Critical order constraint (from Section 3.3 of the paper):
    repair_individual() MUST be called AFTER variation operators
    (crossover / mutation) but BEFORE constant injection and fitness
    evaluation.
"""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING

from src.physics.dimension import Dimension
from src.physics.dimension_rules import DimensionRules, DimensionalViolation
from src.physics.library import DimLibrary, Fragment

if TYPE_CHECKING:
    from deap.gp import PrimitiveTree

logger = logging.getLogger(__name__)

# Hard upper bound on tree size after any repair splice
MAX_TREE_NODES: int = 50

# Compliance tolerance: dimension distance ≤ ε is considered "correct"
_EPSILON: float = 1e-9

# Operator names that are unary (no right child)
_UNARY_OPS = frozenset({"sq", "sqrt", "sin", "cos", "log", "exp"})


# ======================================================================
# 1. Integrated Tree Utilities
# ======================================================================

def splice_subtree(
    tree: "PrimitiveTree",
    sl: slice,
    replacement_nodes: list,
) -> "PrimitiveTree":
    """
    Return a NEW PrimitiveTree with the subtree at ``sl`` replaced by
    ``replacement_nodes``.

    The original tree is never mutated.  The returned tree has its
    fitness invalidated (del new_tree.fitness.values) so it will be
    re-evaluated on the next fitness pass.

    Parameters
    ----------
    tree : PrimitiveTree
        The parent individual to splice.
    sl : slice
        The slice [start:stop) returned by tree.searchSubtree(pos).
    replacement_nodes : list
        The replacement prefix-list of DEAP Primitive/Terminal objects
        (e.g. Fragment.as_list()).

    Returns
    -------
    PrimitiveTree
        New tree with the splice applied.

    Raises
    ------
    ValueError
        If the splice would grow the tree beyond MAX_TREE_NODES.
    """
    original_len = len(tree)
    removed_len  = sl.stop - sl.start
    added_len    = len(replacement_nodes)
    new_len      = original_len - removed_len + added_len

    if new_len > MAX_TREE_NODES:
        raise ValueError(
            f"Splice rejected: resulting tree would have {new_len} nodes "
            f"(limit is {MAX_TREE_NODES})."
        )

    new_tree = copy.copy(tree)
    # Slice assignment on the underlying list
    new_tree[sl] = replacement_nodes

    # Invalidate stale fitness
    if new_tree.fitness.valid:
        del new_tree.fitness.values

    return new_tree


def evaluate_dim_at(
    tree: "PrimitiveTree",
    pos: int,
    context_dims: dict[str, Dimension],
) -> Dimension:
    """
    Recursively compute the physical dimension of the subtree rooted at
    position ``pos`` in the flat DEAP prefix list.

    Right-child invariant:
        For arity-2 nodes, the right child position is:
            right_pos = tree.searchSubtree(pos + 1).stop
        This skips the entire left subtree — do NOT use pos + 2.

    Parameters
    ----------
    tree : PrimitiveTree
        The individual to evaluate.
    pos : int
        0-indexed position of the root node of the subtree.
    context_dims : dict[str, Dimension]
        Maps terminal names to their physical dimension.
        Terminals absent from this dict are dimensionless (ERCs/constants).

    Returns
    -------
    Dimension
        The output dimension of the subtree.

    Raises
    ------
    DimensionalViolation
        If a dimensional constraint is violated (e.g. adding mismatched dims).
    ValueError
        If an unknown operator name is encountered.
    """
    node = tree[pos]
    arity = node.arity

    # ── Terminal ──────────────────────────────────────────────────────────
    if arity == 0:
        return context_dims.get(node.name, Dimension.dimensionless())

    # ── Unary ─────────────────────────────────────────────────────────────
    if arity == 1:
        left_dim = evaluate_dim_at(tree, pos + 1, context_dims)
        return DimensionRules.forward(node.name, left_dim)

    # ── Binary ────────────────────────────────────────────────────────────
    # right_pos MUST be computed by skipping the full left subtree
    left_dim  = evaluate_dim_at(tree, pos + 1, context_dims)
    right_pos = tree.searchSubtree(pos + 1).stop
    right_dim = evaluate_dim_at(tree, right_pos, context_dims)
    return DimensionRules.forward(node.name, left_dim, right_dim)


# ======================================================================
# 2. Target Dimension Calculation
# ======================================================================

def target_dim_for_left(
    tree: "PrimitiveTree",
    pos: int,
    target_dim: Dimension,
    context_dims: dict[str, Dimension],
) -> Dimension | None:
    """
    Compute the dimension the LEFT child of node at ``pos`` must satisfy
    so that the node as a whole achieves ``target_dim``.

    For binary operators the *current* right child dimension is evaluated
    and used as the "known sibling" in Algorithm 2/3.

    Returns None if the target cannot be determined (e.g. evaluation error).
    """
    node = tree[pos]
    op   = node.name

    # ── Unary: no sibling needed ──────────────────────────────────────────
    if op in _UNARY_OPS:
        try:
            return DimensionRules.backward_left(op, target_dim)
        except Exception:
            return None

    # ── Binary: need the right child's current dimension ──────────────────
    if node.arity < 2:
        return None

    right_pos = tree.searchSubtree(pos + 1).stop
    try:
        right_dim = evaluate_dim_at(tree, right_pos, context_dims)
    except Exception:
        right_dim = None

    try:
        return DimensionRules.backward_left(op, target_dim, right=right_dim)
    except Exception:
        return None


def target_dim_for_right(
    tree: "PrimitiveTree",
    pos: int,
    target_dim: Dimension,
    context_dims: dict[str, Dimension],
) -> Dimension | None:
    """
    Compute the dimension the RIGHT child of node at ``pos`` must satisfy
    so that the node as a whole achieves ``target_dim``.

    The *current* left child dimension is evaluated and used as the
    "known sibling" in Algorithm 2/3.

    Returns None for unary operators (no right child) or on error.
    """
    node = tree[pos]
    op   = node.name

    if op in _UNARY_OPS or node.arity < 2:
        return None

    try:
        left_dim = evaluate_dim_at(tree, pos + 1, context_dims)
    except Exception:
        left_dim = None

    try:
        return DimensionRules.backward_right(op, target_dim, left=left_dim)
    except Exception:
        return None


# ======================================================================
# 3. Algorithm 1 — Internal Recursive Propagation
# ======================================================================

def _propagate(
    tree: "PrimitiveTree",
    pos: int,
    target_dim: Dimension,
    lib: DimLibrary,
    context_dims: dict[str, Dimension],
    max_frag_nodes: int,
) -> tuple["PrimitiveTree", bool]:
    """
    Recursively attempt to repair the subtree at ``pos`` so it evaluates
    to ``target_dim``.

    Implements Algorithm 1, Lines 1-25 of Reissmann et al. (2025).

    Invariant
    ---------
    Returns ``(new_tree, True)`` after the FIRST successful modification.
    Any splice invalidates all subsequent positional indices, so further
    traversal must restart from pos=0 in the outer loop.

    Parameters
    ----------
    tree : PrimitiveTree
        Current individual (never mutated in-place).
    pos : int
        Position of the subtree root in the flat prefix list.
    target_dim : Dimension
        The required output dimension of the subtree.
    lib : DimLibrary
        The precomputed fragment store.
    context_dims : dict[str, Dimension]
        Terminal→Dimension mapping.
    max_frag_nodes : int
        Maximum fragment size for library lookups (bloat control).

    Returns
    -------
    (PrimitiveTree, bool)
        The (possibly modified) tree and a flag indicating whether a
        modification was made.
    """
    # ── Line 3: check compliance ──────────────────────────────────────────
    try:
        current_dim = evaluate_dim_at(tree, pos, context_dims)
        if current_dim.distance(target_dim) <= _EPSILON:
            return tree, False   # already satisfies target
    except Exception:
        pass   # evaluation failed → definitely needs correction

    # ── Terminal: cannot propagate further, must use library ──────────────
    node = tree[pos]
    if node.arity == 0:
        current_sl = tree.searchSubtree(pos)
        frag = lib.get(target_dim, max_nodes=max_frag_nodes)
        if frag is not None:
            try:
                new_tree = splice_subtree(tree, current_sl, frag.as_list())
                logger.debug(
                    "Library splice at terminal pos=%d: replaced %d nodes with %d-node fragment "
                    "(dim=%s).", pos, current_sl.stop - current_sl.start, len(frag), frag.dimension,
                )
                return new_tree, True
            except ValueError:
                pass
        return tree, False

    # ── Left child ────────────────────────────────────────────────────────
    left_pos     = pos + 1
    target_left  = target_dim_for_left(tree, pos, target_dim, context_dims)

    if target_left is not None:
        left_deviates = True
        try:
            left_dim      = evaluate_dim_at(tree, left_pos, context_dims)
            left_deviates = left_dim.distance(target_left) > _EPSILON
        except Exception:
            pass

        if left_deviates:
            # Library replacement attempt on the left subtree
            left_sl   = tree.searchSubtree(left_pos)
            left_frag = lib.get(target_left, max_nodes=max_frag_nodes)
            if left_frag is not None:
                try:
                    new_tree = splice_subtree(tree, left_sl, left_frag.as_list())
                    return new_tree, True
                except ValueError:
                    pass

            # Recursive descent into left child
            new_tree, changed = _propagate(
                tree, left_pos, target_left, lib, context_dims, max_frag_nodes
            )
            if changed:
                return new_tree, True

    # ── Right child (binary operators only) ───────────────────────────────
    if node.arity < 2:
        return tree, False

    right_pos    = tree.searchSubtree(left_pos).stop
    target_right = target_dim_for_right(tree, pos, target_dim, context_dims)

    if target_right is not None:
        right_deviates = True
        try:
            right_dim      = evaluate_dim_at(tree, right_pos, context_dims)
            right_deviates = right_dim.distance(target_right) > _EPSILON
        except Exception:
            pass

        if right_deviates:
            # Library replacement attempt on the right subtree
            right_sl   = tree.searchSubtree(right_pos)
            right_frag = lib.get(target_right, max_nodes=max_frag_nodes)
            if right_frag is not None:
                try:
                    new_tree = splice_subtree(tree, right_sl, right_frag.as_list())
                    return new_tree, True
                except ValueError:
                    pass

            # Recursive descent into right child
            new_tree, changed = _propagate(
                tree, right_pos, target_right, lib, context_dims, max_frag_nodes
            )
            if changed:
                return new_tree, True

    # ── Fallback: If propagation down the tree failed or wasn't applicable,
    # try replacing the entire current subtree from the library.
    current_sl    = tree.searchSubtree(pos)
    current_nodes = current_sl.stop - current_sl.start
    frag = lib.get(target_dim, max_nodes=max_frag_nodes)
    if frag is not None:
        try:
            new_tree = splice_subtree(tree, current_sl, frag.as_list())
            logger.debug(
                "Fallback library splice at pos=%d: replaced %d nodes with %d-node fragment "
                "(dim=%s).", pos, current_nodes, len(frag), frag.dimension,
            )
            return new_tree, True
        except ValueError:
            pass

    return tree, False


# ======================================================================
# 4. Public Entry Point
# ======================================================================

def repair_individual(
    individual: "PrimitiveTree",
    target_dim: Dimension,
    lib: DimLibrary,
    context_dims: dict[str, Dimension],
    max_attempts: int = 5,
    max_frag_nodes: int = 10,
) -> tuple["PrimitiveTree", bool]:
    """
    Outer repair loop — Algorithm 1, Lines 29-36.

    Calls ``_propagate`` from pos=0 up to ``max_attempts`` times,
    restarting after each successful splice (because splice indices shift).
    Stops early if the tree becomes dimensionally compliant.

    Critical Order Constraint
    -------------------------
    Call this AFTER crossover/mutation and BEFORE constant injection
    (param_incl) or fitness evaluation.

    Parameters
    ----------
    individual : PrimitiveTree
        The individual to repair (not mutated in-place).
    target_dim : Dimension
        The required output dimension.
    lib : DimLibrary
        The precomputed fragment store.
    context_dims : dict[str, Dimension]
        Terminal→Dimension mapping.
    max_attempts : int
        Maximum number of propagation passes (default 5).
    max_frag_nodes : int
        Maximum library fragment size allowed during repair (default 10).

    Returns
    -------
    (PrimitiveTree, bool)
        The repaired individual (may be the original if no change was
        needed) and a flag ``modified`` indicating whether any splice
        occurred.
    """
    # Fast exit if already compliant
    try:
        root_dim = evaluate_dim_at(individual, 0, context_dims)
        if root_dim.distance(target_dim) <= _EPSILON:
            return individual, False
    except Exception:
        pass

    current = individual
    modified = False

    for attempt in range(max_attempts):
        # Check compliance at the start of each attempt
        try:
            root_dim = evaluate_dim_at(current, 0, context_dims)
            if root_dim.distance(target_dim) <= _EPSILON:
                break   # compliant — stop early
        except Exception:
            pass

        current, changed = _propagate(
            current, 0, target_dim, lib, context_dims, max_frag_nodes
        )
        if changed:
            modified = True
        else:
            break   # propagation found nothing — further attempts futile

    return current, modified
