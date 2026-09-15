"""Exact minimum-cost search for experimental programs covering hidden incompatibilities.

Copyright (C) 2026 Mohammad Amir Khusru Akhtar
Licensed under the Apache License, Version 2.0.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .genesis import Primitive
from .witness import incompatible_pairs, splits_pair


@dataclass(frozen=True)
class CoverSolution:
    names: tuple[str, ...]
    trial_cost: float
    covered_pairs: int
    total_pairs: int
    nodes_visited: int
    nodes_pruned: int


def minimum_cost_pair_cover(current, primitives: Sequence[Primitive], outcome) -> CoverSolution | None:
    """Provably cheapest primitive subset that splits every incompatible pair.

    This solves the deterministic one-trial structural problem by branch-and-bound.
    It is a weighted set-cover instance over consequentially incompatible pairs.
    """
    pairs = incompatible_pairs(current, outcome)
    if not pairs:
        return CoverSolution((), 0.0, 0, 0, 1, 0)

    full = (1 << len(pairs)) - 1
    items = []
    for p in primitives:
        mask = 0
        for k, pair in enumerate(pairs):
            if splits_pair(p.experiment, pair):
                mask |= 1 << k
        if mask:
            items.append((p, mask))

    union = 0
    for _, mask in items:
        union |= mask
    if union != full:
        return None

    # Coverage-per-cost ordering finds a good incumbent early.
    items.sort(key=lambda pm: (-pm[1].bit_count() / max(pm[0].per_trial_cost, 1e-15), pm[0].per_trial_cost))

    best_cost = float("inf")
    best_names: tuple[str, ...] | None = None
    visited = 0
    pruned = 0

    # Suffix unions prove infeasibility of branches cheaply.
    suffix = [0] * (len(items) + 1)
    for i in range(len(items) - 1, -1, -1):
        suffix[i] = suffix[i + 1] | items[i][1]

    def dfs(i: int, mask: int, cost: float, names: tuple[str, ...]):
        nonlocal best_cost, best_names, visited, pruned
        visited += 1
        if cost >= best_cost - 1e-15:
            pruned += 1
            return
        if mask == full:
            best_cost, best_names = cost, names
            return
        if i == len(items) or (mask | suffix[i]) != full:
            pruned += 1
            return

        primitive, pmask = items[i]
        # Include only if it adds a new distinction.
        if (mask | pmask) != mask:
            dfs(i + 1, mask | pmask, cost + primitive.per_trial_cost, names + (primitive.experiment.name,))
        dfs(i + 1, mask, cost, names)

    dfs(0, 0, 0.0, ())
    if best_names is None:
        return None
    return CoverSolution(best_names, best_cost, len(pairs), len(pairs), visited, pruned)
