"""Tests for minimum-cost incompatible-world witnesses.

Copyright (C) 2026 Mohammad Amir Khusru Akhtar
Licensed under the Apache License, Version 2.0.
"""
import numpy as np

from empirical_reach.core import Experiment, reach_gain
from empirical_reach.worlds import make_hidden_bit_world
from empirical_reach.witness import (
    CostedExperiment,
    cheapest_reach_reducer,
    cheapest_witness,
    incompatible_pairs,
    witnessed_pairs,
)


def test_current_knowledge_leaves_two_incompatible_twin_pairs():
    _, current, _, outcome = make_hidden_bit_world()
    assert incompatible_pairs(current, outcome) == [(0, 1), (2, 3)]


def test_existing_x_probe_cannot_witness_hidden_incompatibility():
    _, current, _, outcome = make_hidden_bit_world()
    assert witnessed_pairs(current, current[0], outcome) == []


def test_cheapest_witness_is_not_cheapest_experiment_overall():
    _, current, reveal_z, outcome = make_hidden_bit_world()

    redundant_x = CostedExperiment(current[0], 1.0)
    expensive_z = CostedExperiment(reveal_z, 9.0)

    weak_z_channel = np.array([
        [0.70, 0.30],
        [0.30, 0.70],
        [0.70, 0.30],
        [0.30, 0.70],
    ])
    weak_z = CostedExperiment(Experiment("weak_z_interaction", weak_z_channel), 5.0)

    candidates = [redundant_x, expensive_z, weak_z]
    chosen = cheapest_witness(current, candidates, outcome)

    assert chosen is not None
    assert chosen.experiment.name == "weak_z_interaction"
    assert chosen.cost == 5.0
    assert witnessed_pairs(current, chosen.experiment, outcome) == [(0, 1), (2, 3)]


def test_reach_reducer_requires_partition_refinement_not_just_distribution_change():
    _, current, reveal_z, outcome = make_hidden_bit_world()

    # Any nonzero z sensitivity changes row distributions and splits the hidden twins.
    weak_z = Experiment("weak_z", np.array([
        [0.51, 0.49], [0.49, 0.51], [0.51, 0.49], [0.49, 0.51]
    ]))
    assert reach_gain(current, weak_z, outcome) == 1.0
    assert reach_gain(current, reveal_z, outcome) == 1.0

    chosen = cheapest_reach_reducer(
        current,
        [CostedExperiment(reveal_z, 9.0), CostedExperiment(weak_z, 5.0)],
        outcome,
    )
    assert chosen is not None
    assert chosen.experiment.name == "weak_z"


def test_no_witness_when_all_candidates_are_functions_of_visible_coordinate():
    _, current, _, outcome = make_hidden_bit_world()
    candidates = [CostedExperiment(e, float(i + 1)) for i, e in enumerate(current)]
    assert cheapest_witness(current, candidates, outcome) is None
