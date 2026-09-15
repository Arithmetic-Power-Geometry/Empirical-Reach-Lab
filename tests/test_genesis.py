"""Experiment genesis tests.

Copyright (C) 2026 Mohammad Amir Khusru Akhtar
Licensed under the Apache License, Version 2.0.
"""
import numpy as np

from empirical_reach.core import Experiment
from empirical_reach.genesis import Primitive, cheapest_generated_witness
from empirical_reach.worlds import make_hidden_bit_world


def binary_feature(values, name):
    return Experiment(name, np.array([[1-v, v] for v in values], dtype=float))


def test_search_returns_none_when_grammar_is_empirically_closed_to_hidden_bit():
    _, current, _, outcome = make_hidden_bit_world()
    grammar = [
        Primitive(current[0], 0.1),
        Primitive(current[1], 0.1),
        Primitive(current[2], 0.1),
    ]
    assert cheapest_generated_witness(current, grammar, outcome, max_length=3) is None


def test_composition_can_be_cheapest_generated_witness():
    _, current, _, outcome = make_hidden_bit_world()

    # Neither primitive alone separates both consequential twin pairs.
    # a separates worlds 0/1 but not 2/3; b does the reverse.
    a = binary_feature([0.1, 0.9, 0.5, 0.5], "interaction_a")
    b = binary_feature([0.5, 0.5, 0.1, 0.9], "interaction_b")
    expensive_global = binary_feature([0.1, 0.9, 0.1, 0.9], "global_probe")

    grammar = [
        Primitive(current[0], 0.01),  # distractor: visible x only
        Primitive(a, 0.20),
        Primitive(b, 0.20),
        Primitive(expensive_global, 2.00),
    ]

    best = cheapest_generated_witness(
        current, grammar, outcome, max_error=0.05, max_length=2
    )
    assert best is not None
    assert best.names == ("interaction_a", "interaction_b")
    assert best.total_cost < 2.00 * best.repetitions or best.repetitions <= 5


def test_search_does_not_need_hidden_variable_name():
    _, current, _, outcome = make_hidden_bit_world()
    # Channels are anonymous response laws; no primitive is named z.
    p = binary_feature([0.2, 0.8, 0.2, 0.8], "interaction_17")
    best = cheapest_generated_witness(
        current, [Primitive(current[0], 0.001), Primitive(p, 0.3)],
        outcome, max_error=0.05, max_length=1
    )
    assert best is not None
    assert best.names == ("interaction_17",)
