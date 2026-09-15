"""Benchmark exact cheapest-witness search with many distractor probes.

Copyright (C) 2026 Mohammad Amir Khusru Akhtar
Licensed under the Apache License, Version 2.0.
"""
import numpy as np

from empirical_reach.core import Experiment
from empirical_reach.genesis import Primitive
from empirical_reach.optimal_search import exact_cheapest_witness_program
from empirical_reach.worlds import make_hidden_bit_world


def channel(v, name):
    return Experiment(name, np.array([[1-x, x] for x in v], dtype=float))


_, current, _, outcome = make_hidden_bit_world()
for distractors in (10, 25, 50, 100, 250):
    grammar = []
    for i in range(distractors):
        p = .55 + (i % 10) * .02
        grammar.append(Primitive(channel([1-p,1-p,p,p], f"x_{i:03d}"), .005 + i*.0001))
    grammar += [
        Primitive(channel([.1,.9,.5,.5], "hidden_a"), .17),
        Primitive(channel([.5,.5,.1,.9], "hidden_b"), .19),
        Primitive(channel([.1,.9,.1,.9], "global"), 2.0),
    ]
    result = exact_cheapest_witness_program(current, grammar, outcome, max_length=3)
    print({
        "primitives": len(grammar),
        "naive_subsets_up_to_3": sum(__import__('math').comb(len(grammar), k) for k in (1,2,3)),
        "best": None if result is None else result.names,
        "cost": None if result is None else result.cost,
        "expanded": None if result is None else result.expanded_nodes,
        "pruned": None if result is None else result.pruned_nodes,
    })
