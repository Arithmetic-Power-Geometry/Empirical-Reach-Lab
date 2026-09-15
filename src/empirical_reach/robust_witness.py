"""Finite-sample cost of witnessing incompatible possible worlds.

Copyright (C) 2026 Mohammad Amir Khusru Akhtar
Licensed under the Apache License, Version 2.0.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log
from typing import Sequence

import numpy as np

from .core import Experiment
from .witness import CostedExperiment, incompatible_pairs


def bhattacharyya_coefficient(p: np.ndarray, q: np.ndarray) -> float:
    """BC(P,Q) in [0,1]; smaller means easier discrimination."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return float(np.sum(np.sqrt(p * q)))


def samples_for_pair(experiment: Experiment, pair: tuple[int, int], error: float = 0.05) -> int | None:
    """Sufficient repeat count from the Bhattacharyya bound for equal-prior testing.

    Pe(n) <= 0.5 * BC(P,Q)^n. Returns None if the experiment cannot separate the pair.
    """
    if not 0 < error < 0.5:
        raise ValueError("error must lie in (0,0.5)")
    i, j = pair
    bc = bhattacharyya_coefficient(experiment.channel[i], experiment.channel[j])
    if np.isclose(bc, 1.0):
        return None
    if np.isclose(bc, 0.0):
        return 1
    return max(1, ceil(log(2.0 * error) / log(bc)))


@dataclass(frozen=True)
class RobustWitnessResult:
    experiment: Experiment
    repeats: int
    cost_per_run: float
    total_cost: float
    witnessed_pairs: tuple[tuple[int, int], ...]


def robust_witness_cost(
    current: Sequence[Experiment],
    candidate: CostedExperiment,
    outcome: Sequence[float],
    error: float = 0.05,
    require_all: bool = True,
) -> RobustWitnessResult | None:
    """Cost to statistically witness hidden consequential incompatibilities."""
    pairs = incompatible_pairs(current, outcome)
    if not pairs:
        return None
    ns = [(pair, samples_for_pair(candidate.experiment, pair, error)) for pair in pairs]
    separable = [(p, n) for p, n in ns if n is not None]
    if require_all and len(separable) != len(pairs):
        return None
    if not separable:
        return None
    repeats = max(n for _, n in separable)
    return RobustWitnessResult(
        experiment=candidate.experiment,
        repeats=repeats,
        cost_per_run=candidate.cost,
        total_cost=float(repeats * candidate.cost),
        witnessed_pairs=tuple(p for p, _ in separable),
    )


def cheapest_robust_witness(
    current: Sequence[Experiment],
    candidates: Sequence[CostedExperiment],
    outcome: Sequence[float],
    error: float = 0.05,
    require_all: bool = True,
) -> RobustWitnessResult | None:
    results = [
        robust_witness_cost(current, c, outcome, error=error, require_all=require_all)
        for c in candidates
    ]
    feasible = [r for r in results if r is not None]
    if not feasible:
        return None
    return min(feasible, key=lambda r: (r.total_cost, r.repeats, r.experiment.name))
