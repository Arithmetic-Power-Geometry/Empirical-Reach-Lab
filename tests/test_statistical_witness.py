"""Finite-sample witness tests.

Copyright (C) 2026 Mohammad Amir Khusru Akhtar
Licensed under the Apache License, Version 2.0.
"""
import numpy as np

from empirical_reach.core import Experiment
from empirical_reach.worlds import make_hidden_bit_world
from empirical_reach.witness import CostedExperiment
from empirical_reach.statistical_witness import (
    cheapest_statistical_witness,
    repetitions_for_all_incompatible_pairs,
)


def z_channel(p0: float, p1: float, name: str) -> Experiment:
    # Binary output: P(Y=1|z=0)=p0, P(Y=1|z=1)=p1.
    return Experiment(name, np.array([
        [1-p0, p0], [1-p1, p1], [1-p0, p0], [1-p1, p1]
    ]))


def test_redundant_x_channel_is_statistically_impossible_witness():
    _, current, _, outcome = make_hidden_bit_world()
    assert repetitions_for_all_incompatible_pairs(current, current[0], outcome) is None


def test_weak_channel_requires_far_more_samples_than_strong_channel():
    _, current, _, outcome = make_hidden_bit_world()
    weak = z_channel(0.49, 0.51, "weak_51_49")
    strong = z_channel(0.10, 0.90, "strong_90_10")
    n_weak = repetitions_for_all_incompatible_pairs(current, weak, outcome, max_error=0.05)
    n_strong = repetitions_for_all_incompatible_pairs(current, strong, outcome, max_error=0.05)
    assert n_weak is not None and n_strong is not None
    assert n_weak > 1000
    assert n_strong < 10
    assert n_weak > n_strong


def test_cheapest_per_trial_need_not_be_cheapest_reliable_experiment():
    _, current, _, outcome = make_hidden_bit_world()
    weak = CostedExperiment(z_channel(0.49, 0.51, "weak_cheap"), 0.01)
    medium = CostedExperiment(z_channel(0.30, 0.70, "medium"), 0.50)
    strong = CostedExperiment(z_channel(0.10, 0.90, "strong"), 1.00)

    result = cheapest_statistical_witness(
        current, [weak, medium, strong], outcome, max_error=0.05
    )
    assert result is not None
    # The test deliberately lets total evidence cost, not unit price, decide.
    totals = {}
    for c in [weak, medium, strong]:
        n = repetitions_for_all_incompatible_pairs(current, c.experiment, outcome, 0.05)
        totals[c.experiment.name] = float("inf") if n is None else n * c.cost
    assert result.total_cost == min(totals.values())
    assert result.candidate.experiment.name == min(totals, key=totals.get)


def test_error_requirement_monotonically_increases_repetition_need():
    _, current, _, outcome = make_hidden_bit_world()
    e = z_channel(0.30, 0.70, "medium")
    n10 = repetitions_for_all_incompatible_pairs(current, e, outcome, 0.10)
    n05 = repetitions_for_all_incompatible_pairs(current, e, outcome, 0.05)
    n01 = repetitions_for_all_incompatible_pairs(current, e, outcome, 0.01)
    assert n10 <= n05 <= n01
