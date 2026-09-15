"""Finite-sample robust witness tests.

Copyright (C) 2026 Mohammad Amir Khusru Akhtar
Licensed under the Apache License, Version 2.0.
"""
import numpy as np

from empirical_reach.core import Experiment
from empirical_reach.robust_witness import cheapest_robust_witness, samples_for_pair
from empirical_reach.worlds import make_hidden_bit_world
from empirical_reach.witness import CostedExperiment


def z_channel(p0: float, p1: float, name: str) -> Experiment:
    return Experiment(name, np.array([
        [1-p0, p0], [1-p1, p1], [1-p0, p0], [1-p1, p1]
    ]))


def test_weak_channel_requires_more_repeats_than_strong_channel():
    weak = z_channel(0.49, 0.51, "weak")
    strong = z_channel(0.10, 0.90, "strong")
    assert samples_for_pair(weak, (0, 1), error=0.05) > samples_for_pair(strong, (0, 1), error=0.05)


def test_cheapest_per_run_need_not_be_cheapest_total_witness():
    _, current, _, outcome = make_hidden_bit_world()
    weak = CostedExperiment(z_channel(0.49, 0.51, "cheap_weak"), 1.0)
    medium = CostedExperiment(z_channel(0.30, 0.70, "medium"), 4.0)
    strong = CostedExperiment(z_channel(0.05, 0.95, "strong"), 8.0)

    chosen = cheapest_robust_witness(current, [weak, medium, strong], outcome, error=0.05)
    assert chosen is not None
    # This assertion deliberately tests total evidence cost, not sticker price.
    totals = {}
    for c in [weak, medium, strong]:
        r = cheapest_robust_witness(current, [c], outcome, error=0.05)
        totals[c.experiment.name] = r.total_cost if r else float("inf")
    assert chosen.total_cost == min(totals.values())
    assert chosen.experiment.name != "cheap_weak"


def test_visible_only_experiment_has_infinite_witness_cost():
    _, current, _, outcome = make_hidden_bit_world()
    chosen = cheapest_robust_witness(
        current, [CostedExperiment(current[0], 0.01)], outcome, error=0.05
    )
    assert chosen is None
