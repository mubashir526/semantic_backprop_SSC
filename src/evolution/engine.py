"""
Evolutionary Engine
====================
Runs the DEAP generation loop with the mandatory Reissmann et al. operation order:

    Selection → Crossover → Mutation → SBP Repair → Fitness Evaluation

The SBP repair step is applied ONLY to individuals with invalid fitness
(those that were modified by variation operators).  Elites that carry
valid fitness are never repaired.

This ordering guarantee is critical: repairing before evaluation ensures
the fitness landscape is dimensionally consistent.  Repairing after would
evaluate invalid expressions, wasting evaluation budget.

Public API
----------
    run_evolution_with_sbp(...)  →  (hall_of_fame, log_dict)
"""

from __future__ import annotations

import copy
import logging
import random
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from deap import algorithms, base, creator, gp, tools

from src.physics.dimension import Dimension
from src.physics.library import DimLibrary
from src.sbp.engine import repair_individual, MAX_TREE_NODES
from src.evolution.utils import cxSSC

logger = logging.getLogger(__name__)


# ======================================================================
# Runtime configuration dataclass
# ======================================================================

@dataclass
class EvolutionConfig:
    """All hyper-parameters for a single evolution run."""

    # Population / generation budget
    pop_size:     int   = 300
    n_gen:        int   = 50
    hof_size:     int   = 10

    # Operator probabilities
    cx_prob:      float = 0.7
    mut_prob:     float = 0.2

    # SSC vs standard crossover
    use_ssc:      bool  = True

    # DEAP tree generation parameters
    min_depth:    int   = 1
    max_depth:    int   = 4

    # SBP parameters
    sbp_attempts:    int = 5
    sbp_max_frag:    int = 10

    # Tournament size
    tournament_size: int = 3

    # Immigration fraction each generation (diversity injection)
    immigrant_frac:  float = 0.05

    # Random seed (None = non-deterministic)
    seed: int | None = None


# ======================================================================
# Statistics helper
# ======================================================================

def _build_stats() -> tools.Statistics:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min",  lambda v: float(np.min(v)))
    stats.register("avg",  lambda v: float(np.mean(v)))
    stats.register("max",  lambda v: float(np.max(v)))
    return stats


# ======================================================================
# Main evolution loop
# ======================================================================

def run_evolution_with_sbp(
    pset: gp.PrimitiveSet,
    toolbox: base.Toolbox,
    fitness_fn: Callable,
    lib: DimLibrary,
    target_dim: Dimension,
    context_dims: dict[str, Dimension],
    var_names: list[str],
    cfg: EvolutionConfig | None = None,
    sample_ranges: dict[str, tuple[float, float]] | None = None,
    verbose: bool = True,
) -> tuple[tools.HallOfFame, dict]:
    """
    Run the GP evolutionary loop with integrated SBP repair.

    MANDATORY OPERATION ORDER (enforced each generation):
        1. Selection      — tournament selection of offspring pool
        2. Crossover      — cxSSC or standard; del fitness immediately
        3. Mutation       — mutUniform; del fitness immediately
        4. SBP Repair     — repair_individual on INVALID-fitness only
        5. Fitness Eval   — evaluate INVALID-fitness only

    Parameters
    ----------
    pset : gp.PrimitiveSet
        The primitive set (built by make_feynman_pset).
    toolbox : base.Toolbox
        DEAP toolbox with at least ``individual``, ``population``,
        ``mutate``, and ``select`` registered.
        ``mate`` will be overridden by this function to respect use_ssc.
    fitness_fn : Callable
        ``fitness_fn(ind) -> (float,)``  (from make_feynman_fitness).
    lib : DimLibrary
        Precomputed fragment library for SBP repair.
    target_dim : Dimension
        Required output dimension of evolved expressions.
    context_dims : dict[str, Dimension]
        Terminal → Dimension map for evaluate_dim_at.
    var_names : list[str]
        Variable names in pset argument order (needed by cxSSC).
    cfg : EvolutionConfig or None
        Hyper-parameters; defaults to EvolutionConfig().
    sample_ranges : dict or None
        For cxSSC semantic evaluation; see utils.cxSSC docstring.
    verbose : bool
        Print per-generation statistics.

    Returns
    -------
    (HallOfFame, log_dict)
        HallOfFame contains the top ``cfg.hof_size`` individuals.
        log_dict has lists: 'gen', 'min', 'avg', 'max'.
    """
    if cfg is None:
        cfg = EvolutionConfig()

    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

    # ── Register mate according to use_ssc flag ───────────────────────────
    if cfg.use_ssc:
        toolbox.register(
            "mate", cxSSC,
            pset=pset,
            var_names=var_names,
            sample_ranges=sample_ranges,
        )
    else:
        toolbox.register("mate", gp.cxOnePoint)

    # Enforce strict NODE length limit on crossover and mutation
    length_limit = gp.staticLimit(
        key=lambda ind: len(ind),
        max_value=MAX_TREE_NODES,
    )
    toolbox.decorate("mate",   length_limit)
    toolbox.decorate("mutate", length_limit)

    # ── Initialise population ─────────────────────────────────────────────
    pop  = toolbox.population(n=cfg.pop_size)
    hof  = tools.HallOfFame(cfg.hof_size)
    stats = _build_stats()

    log = {"gen": [], "min": [], "avg": [], "max": [], "best_ind": []}

    # ── Evaluate initial population ───────────────────────────────────────
    for ind in pop:
        if not ind.fitness.valid:
            ind.fitness.values = fitness_fn(ind)
    hof.update(pop)
    log["best_ind"].append(copy.deepcopy(hof[0]) if len(hof) > 0 else None)

    if verbose:
        rec = stats.compile(pop)
        logger.info("Gen   0 | min=%.4e  avg=%.4e", rec["min"], rec["avg"])
        print(f"Gen   0 | min={rec['min']:.4e}  avg={rec['avg']:.4e}")
    _append_log(log, 0, pop, stats)

    # ══════════════════════════════════════════════════════════════════════
    # Generation loop
    # ══════════════════════════════════════════════════════════════════════
    for gen in range(1, cfg.n_gen + 1):

        # ── STEP 1: Selection ─────────────────────────────────────────────
        offspring = toolbox.select(pop, len(pop))
        offspring = [copy.deepcopy(ind) for ind in offspring]

        # ── STEP 2: Crossover ─────────────────────────────────────────────
        for i in range(0, len(offspring) - 1, 2):
            if random.random() < cfg.cx_prob:
                offspring[i], offspring[i + 1] = toolbox.mate(
                    offspring[i], offspring[i + 1]
                )
                # Immediately invalidate to mark as "needs repair + eval"
                if offspring[i].fitness.valid:
                    del offspring[i].fitness.values
                if offspring[i + 1].fitness.valid:
                    del offspring[i + 1].fitness.values

        # ── STEP 3: Mutation ──────────────────────────────────────────────
        for i in range(len(offspring)):
            if random.random() < cfg.mut_prob:
                (offspring[i],) = toolbox.mutate(offspring[i])
                # Immediately invalidate
                if offspring[i].fitness.valid:
                    del offspring[i].fitness.values

        # ── STEP 4: SBP Repair ────────────────────────────────────────────
        # CRITICAL: repair ONLY individuals with invalid fitness.
        # Elites (valid fitness) are never touched.
        for i, ind in enumerate(offspring):
            if not ind.fitness.valid:
                repaired, modified = repair_individual(
                    ind,
                    target_dim,
                    lib,
                    context_dims,
                    max_attempts=cfg.sbp_attempts,
                    max_frag_nodes=cfg.sbp_max_frag,
                )
                if modified:
                    offspring[i] = repaired
                    # repair_individual already invalidates fitness on the
                    # returned tree, but be explicit here
                    if offspring[i].fitness.valid:
                        del offspring[i].fitness.values

        # ── STEP 5: Fitness Evaluation ────────────────────────────────────
        # Evaluate ONLY individuals with invalid fitness (post-repair).
        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = fitness_fn(ind)

        # ── Immigration (diversity injection) ─────────────────────────────
        n_immigrants = max(1, int(cfg.pop_size * cfg.immigrant_frac))
        immigrants   = toolbox.population(n=n_immigrants)
        for imm in immigrants:
            imm.fitness.values = fitness_fn(imm)
        offspring.extend(immigrants)

        # ── Next generation ───────────────────────────────────────────────
        pop = toolbox.select(offspring, cfg.pop_size)
        hof.update(pop)
        log["best_ind"].append(copy.deepcopy(hof[0]) if len(hof) > 0 else None)

        # ── Logging ───────────────────────────────────────────────────────
        _append_log(log, gen, pop, stats)
        if verbose:
            rec = log
            g = rec["gen"][-1]
            print(
                f"Gen {g:3d} | min={rec['min'][-1]:.4e}"
                f"  avg={rec['avg'][-1]:.4e}",
                flush=True
            )

        # ── Early stopping ────────────────────────────────────────────────
        if log["min"][-1] < 1e-10:
            if verbose:
                print("Early stop: solution found.")
            break

    return hof, log


# ======================================================================
# Private helpers
# ======================================================================

def _append_log(log: dict, gen: int, pop, stats: tools.Statistics) -> None:
    rec = stats.compile(pop)
    log["gen"].append(gen)
    log["min"].append(rec["min"])
    log["avg"].append(rec["avg"])
    log["max"].append(rec["max"])


# ======================================================================
# Convenience: build a standard DEAP toolbox for Feynman tasks
# ======================================================================

def make_feynman_toolbox(
    pset: gp.PrimitiveSet,
    individual_cls,
    cfg: EvolutionConfig | None = None,
) -> base.Toolbox:
    """
    Build a DEAP Toolbox pre-configured for Feynman symbolic regression.

    The caller must still register ``evaluate`` before passing to
    ``run_evolution_with_sbp``.

    Parameters
    ----------
    pset : gp.PrimitiveSet
    individual_cls : DEAP creator class
        Usually ``creator.Individual`` (must have been created beforehand).
    cfg : EvolutionConfig or None

    Returns
    -------
    base.Toolbox
    """
    if cfg is None:
        cfg = EvolutionConfig()

    tb = base.Toolbox()
    tb.register(
        "expr", gp.genHalfAndHalf,
        pset=pset, min_=cfg.min_depth, max_=cfg.max_depth,
    )
    tb.register("individual", tools.initIterate, individual_cls, tb.expr)
    tb.register("population", tools.initRepeat, list, tb.individual)
    tb.register(
        "select", tools.selTournament,
        tournsize=cfg.tournament_size,
    )
    tb.register(
        "mutate", gp.mutUniform,
        expr=tb.expr, pset=pset,
    )
    return tb
