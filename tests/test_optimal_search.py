"""Exact search validation.

Copyright (C) 2026 Mohammad Amir Khusru Akhtar
Licensed under the Apache License, Version 2.0.
"""
import itertools
import numpy as np

from empirical_reach.core import Experiment, product_experiment
from empirical_reach.genesis import Primitive
from empirical_reach.optimal_search import exact_cheapest_witness_program
from empirical_reach.worlds import make_hidden_bit_world
from empirical_reach.witness import incompatible_pairs, splits_pair


def channel(v, name):
    return Experiment(name, np.array([[1-x, x] for x in v], dtype=float))


def brute_force(current, grammar, outcome):
    pairs = incompatible_pairs(current, outcome)
    best = None
    for r in range(1, len(grammar)+1):
        for combo in itertools.combinations(grammar, r):
            exps = [p.experiment for p in combo]
            joint = exps[0] if len(exps)==1 else product_experiment(exps)
            if all(splits_pair(joint, pair) for pair in pairs):
                cost = sum(p.per_trial_cost for p in combo)
                candidate = (cost, tuple(sorted(p.experiment.name for p in combo)))
                if best is None or candidate < best:
                    best = candidate
    return best


def test_branch_and_bound_matches_exhaustive_optimum():
    _, current, _, outcome = make_hidden_bit_world()
    grammar = [
        Primitive(channel([.1,.9,.5,.5], "a"), .2),
        Primitive(channel([.5,.5,.1,.9], "b"), .2),
        Primitive(channel([.1,.9,.1,.9], "global"), 1.0),
        Primitive(current[0], .001),
    ]
    exact = exact_cheapest_witness_program(current, grammar, outcome)
    brute = brute_force(current, grammar, outcome)
    assert exact is not None and brute is not None
    assert abs(exact.cost - brute[0]) < 1e-12
    assert set(exact.names) == set(brute[1])
    assert exact.cost == .4


def test_impossible_grammar_returns_none():
    _, current, _, outcome = make_hidden_bit_world()
    grammar = [Primitive(e, .1) for e in current]
    assert exact_cheapest_witness_program(current, grammar, outcome) is None


def test_large_distractor_grammar_still_finds_hidden_cheap_pair():
    _, current, _, outcome = make_hidden_bit_world()
    grammar = []
    # 40 cheap-looking distractors that are functions only of visible x.
    for i in range(40):
        p0 = .55 + (i % 10) * .02
        grammar.append(Primitive(channel([1-p0,1-p0,p0,p0], f"x_distractor_{i:02d}"), .01 + i*.001))
    grammar += [
        Primitive(channel([.1,.9,.5,.5], "hidden_a"), .17),
        Primitive(channel([.5,.5,.1,.9], "hidden_b"), .19),
        Primitive(channel([.1,.9,.1,.9], "expensive_global"), 2.0),
    ]
    result = exact_cheapest_witness_program(current, grammar, outcome, max_length=3)
    assert result is not None
    assert set(result.names) == {"hidden_a", "hidden_b"}
    assert abs(result.cost - .36) < 1e-12
