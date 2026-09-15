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


def bhattacharyya_coefficient(p: Sequence[float], q: Sequence[float]) -> float:
    """BC(P,Q)=sum_y sqrt(P_y Q_y), in [0,1]."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return float(np.sum(np.sqrt(p * q)))


def repetitions_for_pair(
    experiment: Experiment,
    pair: tuple[int, int],
    max_error: float = 0.05,
) -> int | None:
    """Sufficient iid repetitions for equal-prior Bayes error <= max_error.

    Uses P_e(n) <= 0.5 * BC(P,Q)^n. Returns None when the channel gives
    identical response laws and therefore cannot distinguish the pair.
    """
    if not 0 < max_error < 0.5:
        raise ValueError("max_error must lie in (0,0.5)")
    i, j = pair
    bc = bhattacharyya_coefficient(experiment.channel[i], experiment.channel[j])
    if bc >= 1.0 - 1e-15:
        return None
    if bc <= 1e-15:
        return 1
    return max(1, ceil(log(2.0 * max_error) / log(bc)))


def repetitions_for_all_incompatible_pairs(
    current: Sequence[Experiment],
    experiment: Experiment,
    outcome: Sequence[float],
    max_error: float = 0.05,
) -> int | None:
    """Repetitions sufficient for every currently hidden consequential pair."""
    pairs = incompatible_pairs(current, outcome)
    if not pairs:
        return 0
    needs = [repetitions_for_pair(experiment, p, max_error) for p in pairs]
    if any(n is None for n in needs):
        return None
    return max(int(n) for n in needs if n is not None)


@dataclass(frozen=True)
class StatisticalWitness:
    candidate: CostedExperiment
    repetitions: int
    setup_cost: float
    per_trial_cost: float
    total_cost: float


def cheapest_statistical_witness(
    current: Sequence[Experiment],
    candidates: Sequence[CostedExperiment],
    outcome: Sequence[float],
    max_error: float = 0.05,
    setup_costs: dict[str, float] | None = None,
) -> StatisticalWitness | None:
    """Cheapest reliable witness after accounting for repetitions.

    `CostedExperiment.cost` is interpreted as per-trial cost. Optional setup
    costs model fixed instrument/protocol overhead.
    """
    setup_costs = setup_costs or {}
    feasible: list[StatisticalWitness] = []
    for c in candidates:
        n = repetitions_for_all_incompatible_pairs(
            current, c.experiment, outcome, max_error=max_error
        )
        if n is None:
            continue
        setup = float(setup_costs.get(c.experiment.name, 0.0))
        total = setup + n * c.cost
        feasible.append(StatisticalWitness(c, n, setup, c.cost, total))
    if not feasible:
        return None
    return min(feasible, key=lambda r: (r.total_cost, r.repetitions, r.candidate.experiment.name))
