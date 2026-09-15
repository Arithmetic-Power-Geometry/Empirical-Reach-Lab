"""Generate, rather than hand-supply, the cheapest reliable witness program.

Copyright (C) 2026 Mohammad Amir Khusru Akhtar
Licensed under the Apache License, Version 2.0.
"""
import numpy as np

from empirical_reach.core import Experiment
from empirical_reach.genesis import Primitive, cheapest_generated_witness
from empirical_reach.worlds import make_hidden_bit_world


def channel(values, name):
    return Experiment(name, np.array([[1-v, v] for v in values], dtype=float))


_, current, _, outcome = make_hidden_bit_world()

grammar = [
    Primitive(current[0], 0.01),
    Primitive(current[1], 0.01),
    Primitive(channel([0.1, 0.9, 0.5, 0.5], "interaction_a"), 0.20),
    Primitive(channel([0.5, 0.5, 0.1, 0.9], "interaction_b"), 0.20),
    Primitive(channel([0.1, 0.9, 0.1, 0.9], "global_probe"), 2.00),
]

for depth in (1, 2, 3):
    best = cheapest_generated_witness(
        current, grammar, outcome, max_error=0.05, max_length=depth
    )
    print("max_length", depth, "=>", None if best is None else {
        "program": best.names,
        "repetitions": best.repetitions,
        "trial_cost": best.program_trial_cost,
        "total_cost": best.total_cost,
    })
