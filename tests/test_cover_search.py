"""Tests for exact minimum-cost incompatibility cover.

Copyright (C) 2026 Mohammad Amir Khusru Akhtar
Licensed under the Apache License, Version 2.0.
"""
import numpy as np

from empirical_reach.core import Experiment
from empirical_reach.cover_search import minimum_cost_pair_cover
from empirical_reach.genesis import Primitive
from empirical_reach.worlds import make_hidden_bit_world


def exp(values, name):
    return Experiment(name, np.array([[1-v, v] for v in values], dtype=float))


def test_exact_solver_finds_compositional_optimum():
    _, current, _, outcome = make_hidden_bit_world()
    primitives = [
        Primitive(exp([0.1, 0.9, 0.5, 0.5], "a"), 0.20),
        Primitive(exp([0.5, 0.5, 0.1, 0.9], "b"), 0.20),
        Primitive(exp([0.1, 0.9, 0.1, 0.9], "global"), 1.00),
    ]
    sol = minimum_cost_pair_cover(current, primitives, outcome)
    assert sol is not None
    assert set(sol.names) == {"a", "b"}
    assert abs(sol.trial_cost - 0.40) < 1e-12


def test_exact_solver_returns_none_if_grammar_cannot_cover_all_pairs():
    _, current, _, outcome = make_hidden_bit_world()
    primitives = [Primitive(exp([0.1, 0.9, 0.5, 0.5], "a"), 0.20)]
    assert minimum_cost_pair_cover(current, primitives, outcome) is None


def test_distractors_do_not_change_optimum():
    _, current, _, outcome = make_hidden_bit_world()
    primitives = [
        Primitive(exp([0.1, 0.9, 0.5, 0.5], "a"), 0.20),
        Primitive(exp([0.5, 0.5, 0.1, 0.9], "b"), 0.20),
    ]
    # Add 100 cheap-looking but empirically irrelevant x-only distractors.
    for i in range(100):
        primitives.append(Primitive(current[i % len(current)], 0.001 + i * 1e-6))
    sol = minimum_cost_pair_cover(current, primitives, outcome)
    assert sol is not None
    assert set(sol.names) == {"a", "b"}
    assert abs(sol.trial_cost - 0.40) < 1e-12
