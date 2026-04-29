"""
Semantic Fragment Library
==========================
Implements the precomputed store of dimensionally-valid DEAP subtrees
described in Appendix C.2 of Reissmann et al. (2025).

The library maps physical dimension keys → lists of Fragment objects,
where each Fragment is a valid prefix-list of DEAP Primitive/Terminal
nodes that can be directly spliced into a PrimitiveTree during SBP repair.

Build Algorithm
---------------
Phase 0 — Terminals:
    Each terminal in pset is assigned a dimension from context_dims
    (keyed by terminal.name).  Numeric constants / ERCs fall back to
    the dimensionless zero vector.

Phase 1..max_depth — Grow:
    Prepend each Primitive to all existing fragment lists whose
    arity-sized tuple of child-fragment dimensions is valid under
    DimensionRules.forward().  DimensionalViolation causes silent
    discard, not a crash.

Capping:
    - Global cap  : stops building once total fragments ≥ max_size.
    - Per-key cap : stores ≤ 500 fragments per dimension key.

Key Format (CRITICAL — must match Dimension.__hash__):
    tuple(np.round(dim.vector, 6).tolist())

No DEAP imports at the module level — DEAP objects are accepted as
opaque objects with `.name` and `.arity` attributes.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.physics.dimension import Dimension
from src.physics.dimension_rules import DimensionRules, DimensionalViolation

logger = logging.getLogger(__name__)

# Maximum fragments stored per dimension key (diversity cap)
_MAX_PER_KEY: int = 500

# Operators whose output is dimensionless AND whose input must be dimensionless.
# Nesting these is semantically redundant (e.g. sin(cos(x))), so we skip.
_TRANSCENDENTAL_NAMES = frozenset({"sin", "cos", "exp", "log"})


# ======================================================================
# Fragment dataclass
# ======================================================================

@dataclass(frozen=True)
class Fragment:
    """
    A single valid DEAP prefix sub-list that can be spliced into a
    PrimitiveTree.

    Attributes
    ----------
    nodes : tuple
        Immutable tuple of DEAP Primitive or Terminal objects in
        prefix (pre-order) order.  Must be a tuple for hashability.
    dimension : Dimension
        The output dimension produced when this fragment is evaluated.
    depth : int
        Tree depth of the fragment (terminal = 0, unary-of-terminal = 1, …).
    """
    nodes: tuple          # immutable — required for hashability
    dimension: Dimension
    depth: int = field(default=0, compare=False)

    def as_list(self) -> list:
        """Return the node sequence as a plain list ready for DEAP splicing."""
        return list(self.nodes)

    def __len__(self) -> int:
        return len(self.nodes)


# ======================================================================
# DimLibrary
# ======================================================================

class DimLibrary:
    """
    Precomputed store of dimensionally-valid DEAP subtrees.

    Parameters
    ----------
    pset : deap.gp.PrimitiveSet
        The DEAP primitive set defining operators and terminals.
    context_dims : dict[str, Dimension]
        Maps terminal names to their physical Dimension.
        Terminals absent from this dict are treated as dimensionless
        (suitable for ERCs and numeric constants).
    max_depth : int
        Maximum tree depth to enumerate (default 3).
        Depth 0 = terminals only; depth d = operator + depth-(d-1) children.
    max_size : int
        Global fragment cap across all keys (default 100_000).
    """

    def __init__(
        self,
        pset,
        context_dims: dict[str, Dimension],
        max_depth: int = 3,
        max_size: int = 100_000,
    ):
        self._store: dict[tuple, list[Fragment]] = {}
        self._total: int = 0
        self._build(pset, context_dims, max_depth, max_size)

    # ------------------------------------------------------------------ #
    #  Key helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _key(dim: Dimension) -> tuple:
        """
        Compute the dictionary key for a Dimension.
        MUST exactly match Dimension.__hash__ rounding (6 decimal places).
        """
        return tuple(np.round(dim.vector, 6).tolist())

    def _insert(self, frag: Fragment) -> bool:
        """
        Insert a fragment into the store, respecting both caps.
        Returns True if inserted, False if any cap was hit.
        """
        if not hasattr(self, "_warned_keys"):
            self._warned_keys = set()
            
        k = self._key(frag.dimension)
        bucket = self._store.setdefault(k, [])
        if len(bucket) >= _MAX_PER_KEY:
            if k not in self._warned_keys:
                logger.debug(
                    "Per-key cap (%d) reached for dimension key %s — "
                    "further fragments with this dimension will be skipped.",
                    _MAX_PER_KEY, k,
                )
                self._warned_keys.add(k)
            return False
        bucket.append(frag)
        self._total += 1
        return True

    # ------------------------------------------------------------------ #
    #  Build algorithm
    # ------------------------------------------------------------------ #

    def _build(self, pset, context_dims: dict[str, Dimension],
               max_depth: int, max_size: int) -> None:
        """
        Enumerate fragments depth-by-depth up to max_depth.

        We keep a list-of-lists indexed by depth:
            by_depth[d] = list of (Fragment, Dimension) at depth d.

        At depth 0 we populate terminals.
        At depth d > 0 we combine operators with child fragments from
        depths 0..d-1 (the arity children must jointly cover depth d-1).
        """
        # Collect primitives and terminals from pset —————————————————————
        # pset.primitives is dict[ret_type → list[Primitive]]
        # pset.terminals  is dict[ret_type → list[Terminal]]
        primitives: list[Any] = []
        for prims in pset.primitives.values():
            primitives.extend(prims)

        terminals_deap: list[Any] = []
        for terms in pset.terminals.values():
            terminals_deap.extend(terms)

        # by_depth[d] holds all Fragment objects created at depth d
        by_depth: list[list[Fragment]] = [[] for _ in range(max_depth + 1)]

        # ── Phase 0: Terminals ───────────────────────────────────────────
        for term in terminals_deap:
            # ERC terminals are stored in pset as MetaEphemeral *classes*
            # (not instances).  Their class-level .arity is a property
            # descriptor, which causes DEAP's PrimitiveTree.__setitem__ to
            # crash with "unsupported operand type(s) for -: 'property' and
            # 'int'" when the fragment is spliced back into a tree.
            # Calling term() instantiates the ERC, giving a concrete object
            # whose .arity is an integer 0.
            from deap.gp import MetaEphemeral as _MetaEphemeral
            node = term() if isinstance(term, _MetaEphemeral) else term

            name = node.name
            dim = context_dims.get(name, Dimension.dimensionless())
            frag = Fragment(
                nodes=(node,),
                dimension=dim,
                depth=0,
            )
            if self._total >= max_size:
                logger.info("Global cap reached during terminal phase.")
                return
            self._insert(frag)
            by_depth[0].append(frag)

        # ── Phases 1..max_depth: Grow by prepending operators ─────────────
        for d in range(1, max_depth + 1):
            if self._total >= max_size:
                break

            new_at_depth: list[Fragment] = []

            for prim in primitives:
                if self._total >= max_size:
                    break
                arity = prim.arity

                if arity == 1:
                    # Child can be any fragment at depth 0..d-1
                    candidates = []
                    for dd in range(d):
                        candidates.extend(by_depth[dd])
                    for child_frag in candidates:
                        if self._total >= max_size:
                            break
                        # Anti-redundancy: skip transcendental(transcendental(…))
                        if (prim.name in _TRANSCENDENTAL_NAMES
                                and child_frag.nodes[0].name
                                    in _TRANSCENDENTAL_NAMES):
                            continue
                        try:
                            out_dim = DimensionRules.forward(
                                prim.name, child_frag.dimension
                            )
                        except (DimensionalViolation, ValueError):
                            continue
                        new_frag = Fragment(
                            nodes=(prim,) + child_frag.nodes,
                            dimension=out_dim,
                            depth=d,
                        )
                        if self._insert(new_frag):
                            new_at_depth.append(new_frag)

                elif arity == 2:
                    # Left child at depth dl, right child at depth dr,
                    # with max(dl, dr) == d-1  so the new node is depth d.
                    # We iterate pairs where at least one child is at d-1.
                    left_pool: list[Fragment] = []
                    for dd in range(d):
                        left_pool.extend(by_depth[dd])

                    right_pool: list[Fragment] = left_pool  # same universe

                    for left_frag in left_pool:
                        if self._total >= max_size:
                            break
                        for right_frag in right_pool:
                            if self._total >= max_size:
                                break
                            # Ensure combined depth reaches exactly d
                            if max(left_frag.depth, right_frag.depth) != d - 1:
                                continue
                            try:
                                out_dim = DimensionRules.forward(
                                    prim.name,
                                    left_frag.dimension,
                                    right_frag.dimension,
                                )
                            except (DimensionalViolation, ValueError):
                                continue
                            new_frag = Fragment(
                                nodes=(prim,) + left_frag.nodes + right_frag.nodes,
                                dimension=out_dim,
                                depth=d,
                            )
                            if self._insert(new_frag):
                                new_at_depth.append(new_frag)

            by_depth[d].extend(new_at_depth)

    # ------------------------------------------------------------------ #
    #  Retrieval
    # ------------------------------------------------------------------ #

    def get(
        self,
        target_dim: Dimension,
        max_nodes: int | None = None,
    ) -> Fragment | None:
        """
        Return a uniformly random Fragment whose dimension equals target_dim.

        Parameters
        ----------
        target_dim : Dimension
            The required output dimension.
        max_nodes : int or None
            If given, only consider fragments with len(fragment.nodes) ≤ max_nodes.
            (Used by SBP to respect head-length constraints.)

        Returns
        -------
        Fragment or None
            None if no matching fragment exists or all are too large.
        """
        k = self._key(target_dim)
        candidates = self._store.get(k, [])

        if max_nodes is not None:
            candidates = [f for f in candidates if len(f.nodes) <= max_nodes]

        if not candidates:
            return None

        return random.choice(candidates)

    # ------------------------------------------------------------------ #
    #  Introspection
    # ------------------------------------------------------------------ #

    def size(self) -> int:
        """Total number of fragments stored across all keys."""
        return self._total

    def num_dimensions(self) -> int:
        """Number of distinct dimension keys in the store."""
        return len(self._store)

    def keys(self):
        """Iterate over all dimension keys (as tuples)."""
        return self._store.keys()

    def has_dimension(self, dim: Dimension) -> bool:
        """Return True if at least one fragment exists for this dimension."""
        return self._key(dim) in self._store

    def fragments_for(self, dim: Dimension) -> list[Fragment]:
        """Return all fragments stored for a given dimension (read-only copy)."""
        return list(self._store.get(self._key(dim), []))

    def __repr__(self) -> str:
        return (
            f"DimLibrary({self._total} fragments "
            f"across {self.num_dimensions()} dimensions)"
        )
