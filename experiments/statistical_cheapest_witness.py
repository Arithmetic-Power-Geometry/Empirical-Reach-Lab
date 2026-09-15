"""Compare unit experiment cost with reliable evidence cost.

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


def z_channel(p0, p1, name):
    return Experiment(name, np.array([
        [1-p0, p0], [1-p1, p1], [1-p0, p0], [1-p1, p1]
    ]))


_, current, _, outcome = make_hidden_bit_world()
candidates = [
    CostedExperiment(current[0], 0.001),                 # cheap but blind
    CostedExperiment(z_channel(0.49, 0.51, "weak"), 0.01),
    CostedExperiment(z_channel(0.30, 0.70, "medium"), 0.50),
    CostedExperiment(z_channel(0.10, 0.90, "strong"), 1.00),
]

for error in (0.10, 0.05, 0.01):
    print(f"\nTarget Bayes error <= {error}")
    for c in candidates:
        n = repetitions_for_all_incompatible_pairs(current, c.experiment, outcome, error)
        total = None if n is None else n * c.cost
        print(c.experiment.name, {"per_trial": c.cost, "repetitions": n, "total": total})
    best = cheapest_statistical_witness(current, candidates, outcome, error)
    print("BEST", None if best is None else {
        "experiment": best.candidate.experiment.name,
        "repetitions": best.repetitions,
        "total_cost": best.total_cost,
    })
