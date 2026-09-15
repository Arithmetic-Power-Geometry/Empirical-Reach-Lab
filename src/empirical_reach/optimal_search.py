"""Exact branch-and-bound search for the cheapest witness program.

Copyright (C) 2026 Mohammad Amir Khusru Akhtar
Licensed under the Apache License, Version 2.0.

The search objective is additive program cost. A program is feasible iff its
joint response law separates every currently observationally equivalent pair
whose consequential outcomes disagree. Because adding probes can only add
pair coverage and can only increase non-negative cost, branch-and-bound is
exact for this finite grammar objective.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Sequence

from .core import Experiment, product_experiment
from .genesis import Primitive
from .witness import incompatible_pairs, splits_pair


@dataclass(frozen=True)
class ExactSearchResult:
    names: tuple[str, ...]
    experiment: Experiment
    cost: float
    expanded_nodes: int
    pruned_nodes: int
    incompatible_pair_count: int


def _coverage(exp: Experiment, pairs: Sequence[tuple[int, int]]) -> frozenset[int]:
    return frozenset(k for k, pair in enumerate(pairs) if splits_pair(exp, pair))


def exact_cheapest_witness_program(
    current: Sequence[Experiment],
    primitives: Sequence[Primitive],
    outcome,
    max_length: int | None = None,
) -> ExactSearchResult | None:
    """Provably minimize additive cost over subsets of a finite probe grammar.

    This is the deterministic/infinite-sample closure-breaking objective. It
    deliberately separates the combinatorial optimum from the finite-sample
    evidence-cost objective implemented in statistical_witness.py.
    """
    pairs = incompatible_pairs(current, outcome)
    if not pairs:
        return ExactSearchResult((), product_experiment(current, "already_sufficient"), 0.0, 0, 0, 0)
    target = frozenset(range(len(pairs)))
    items = []
    for p in primitives:
        cov = _coverage(p.experiment, pairs)
        if cov:
            items.append((p, cov))
    if not items:
        return None

    # Cheapest high-coverage probes first yields an incumbent early.
    items.sort(key=lambda pc: (pc[0].per_trial_cost / len(pc[1]), pc[0].per_trial_cost, pc[0].experiment.name))
    n = len(items)
    max_length = n if max_length is None else min(max_length, n)

    suffix_union = [frozenset() for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        suffix_union[i] = suffix_union[i + 1] | items[i][1]

    best_cost = inf
    best_indices: tuple[int, ...] | None = None
    expanded = 0
    pruned = 0

    def dfs(i: int, chosen: tuple[int, ...], covered: frozenset[int], cost: float):
        nonlocal best_cost, best_indices, expanded, pruned
        expanded += 1
        if covered == target:
            if cost < best_cost:
                best_cost, best_indices = cost, chosen
            return
        if i >= n or len(chosen) >= max_length or cost >= best_cost:
            pruned += 1
            return
        if (covered | suffix_union[i]) != target:
            pruned += 1
            return

        # Admissible lower bound: at least the cheapest remaining probe that
        # adds any still-uncovered pair must be paid for.
        missing = target - covered
        adding_costs = [items[j][0].per_trial_cost for j in range(i, n) if items[j][1] & missing]
        if not adding_costs or cost + min(adding_costs) >= best_cost:
            pruned += 1
            return

        p, cov = items[i]
        dfs(i + 1, chosen + (i,), covered | cov, cost + p.per_trial_cost)
        dfs(i + 1, chosen, covered, cost)

    dfs(0, (), frozenset(), 0.0)
    if best_indices is None:
        return None
    chosen_primitives = [items[i][0] for i in best_indices]
    exps = [p.experiment for p in chosen_primitives]
    joint = exps[0] if len(exps) == 1 else product_experiment(exps, "+".join(p.experiment.name for p in chosen_primitives))
    return ExactSearchResult(
        tuple(p.experiment.name for p in chosen_primitives), joint, best_cost,
        expanded, pruned, len(pairs)
    )
