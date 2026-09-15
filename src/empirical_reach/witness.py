"""Minimum-cost experiments that witness unresolved incompatible worlds.

Copyright (C) 2026 Mohammad Amir Khusru Akhtar
Licensed under the Apache License, Version 2.0.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .core import Experiment, empirical_partition, reach_gain


@dataclass(frozen=True)
class CostedExperiment:
    experiment: Experiment
    cost: float

    def __post_init__(self) -> None:
        if self.cost < 0:
            raise ValueError("experiment cost must be non-negative")


def incompatible_pairs(
    current: Sequence[Experiment], outcome: Sequence[float], tol: float = 1e-12
) -> list[tuple[int, int]]:
    """Pairs still observationally equivalent but consequentially incompatible."""
    f = np.asarray(outcome, dtype=float)
    pairs: list[tuple[int, int]] = []
    for block in empirical_partition(current, len(f)):
        for a_pos, i in enumerate(block):
            for j in block[a_pos + 1:]:
                if abs(float(f[i] - f[j])) > tol:
                    pairs.append((i, j))
    return pairs


def splits_pair(experiment: Experiment, pair: tuple[int, int], tol: float = 1e-12) -> bool:
    """Whether an experiment gives different response laws to the two worlds."""
    i, j = pair
    return not np.allclose(experiment.channel[i], experiment.channel[j], atol=tol, rtol=0.0)


def witnessed_pairs(
    current: Sequence[Experiment], candidate: Experiment, outcome: Sequence[float]
) -> list[tuple[int, int]]:
    """Currently hidden consequential pairs separated by candidate."""
    return [p for p in incompatible_pairs(current, outcome) if splits_pair(candidate, p)]


def cheapest_witness(
    current: Sequence[Experiment],
    candidates: Sequence[CostedExperiment],
    outcome: Sequence[float],
) -> CostedExperiment | None:
    """Cheapest candidate that separates at least one incompatible possible-world pair.

    Ties are broken by larger reach gain, then by experiment name for determinism.
    """
    feasible = [
        c for c in candidates if witnessed_pairs(current, c.experiment, outcome)
    ]
    if not feasible:
        return None
    return min(
        feasible,
        key=lambda c: (
            c.cost,
            -reach_gain(current, c.experiment, outcome),
            c.experiment.name,
        ),
    )


def cheapest_reach_reducer(
    current: Sequence[Experiment],
    candidates: Sequence[CostedExperiment],
    outcome: Sequence[float],
    min_gain: float = 0.0,
) -> CostedExperiment | None:
    """Cheapest experiment giving strictly more than `min_gain` reach reduction."""
    feasible = [
        c for c in candidates
        if reach_gain(current, c.experiment, outcome) > min_gain + 1e-12
    ]
    if not feasible:
        return None
    return min(feasible, key=lambda c: (c.cost, c.experiment.name))
