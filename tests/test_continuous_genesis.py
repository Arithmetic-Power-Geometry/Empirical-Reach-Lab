"""Continuous experiment-genesis tests.

Copyright (C) 2026 Mohammad Amir Khusru Akhtar
Licensed under the Apache License, Version 2.0.
"""
from math import pi

from empirical_reach.continuous_genesis import generated_probe, search_continuous_probe
from empirical_reach.statistical_witness import repetitions_for_all_incompatible_pairs
from empirical_reach.worlds import make_hidden_bit_world


def test_visible_direction_cannot_break_hidden_twins():
    _, current, _, outcome = make_hidden_bit_world()
    x_only = generated_probe(0.0, 1.0, "x_only_generated")
    assert repetitions_for_all_incompatible_pairs(current, x_only, outcome, .05) is None


def test_continuous_search_synthesizes_hidden_sensitive_probe():
    _, current, _, outcome = make_hidden_bit_world()
    best = search_continuous_probe(current, outcome, .05, angle_steps=31, strength_steps=30)
    assert best is not None
    assert best.angle > 0.0
    assert best.repetitions >= 1
    assert best.total_cost > 0.0


def test_pure_hidden_direction_is_feasible_without_being_in_candidate_list():
    _, current, _, outcome = make_hidden_bit_world()
    z_probe = generated_probe(pi/2, .8, "synthesized_z_direction")
    n = repetitions_for_all_incompatible_pairs(current, z_probe, outcome, .05)
    assert n is not None
