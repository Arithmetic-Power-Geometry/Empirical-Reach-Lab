"""Search a grammar of primitive probes for the cheapest reliable witness.

Copyright (C) 2026 Mohammad Amir Khusru Akhtar
Licensed under the Apache License, Version 2.0.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

from .core import Experiment, product_experiment
from .statistical_witness import repetitions_for_all_incompatible_pairs


@dataclass(frozen=True)
class Primitive:
    experiment: Experiment
    per_trial_cost: float


@dataclass(frozen=True)
class ProgramWitness:
    names: tuple[str, ...]
    experiment: Experiment
    repetitions: int
    program_trial_cost: float
    total_cost: float


def synthesize_programs(primitives: Sequence[Primitive], max_length: int = 3):
    """Enumerate non-repeating parallel probe programs from a primitive grammar."""
    if max_length < 1:
        raise ValueError("max_length must be >=1")
    for length in range(1, min(max_length, len(primitives)) + 1):
        for combo in combinations(primitives, length):
            names = tuple(p.experiment.name for p in combo)
            exp = combo[0].experiment if length == 1 else product_experiment(
                [p.experiment for p in combo], name="+".join(names)
            )
            yield names, exp, sum(p.per_trial_cost for p in combo)


def cheapest_generated_witness(
    current: Sequence[Experiment],
    primitives: Sequence[Primitive],
    outcome,
    max_error: float = 0.05,
    max_length: int = 3,
) -> ProgramWitness | None:
    """Find cheapest reliable witness without supplying candidate experiments.

    Candidates are generated from the primitive grammar itself. Total evidence
    cost is repetitions times the summed per-trial cost of the program.
    """
    best = None
    for names, exp, trial_cost in synthesize_programs(primitives, max_length):
        n = repetitions_for_all_incompatible_pairs(current, exp, outcome, max_error)
        if n is None:
            continue
        result = ProgramWitness(names, exp, n, trial_cost, n * trial_cost)
        if best is None or (result.total_cost, len(names), names) < (
            best.total_cost, len(best.names), best.names
        ):
            best = result
    return best
