"""Compare sticker-price and finite-sample witness cost.

Copyright (C) 2026 Mohammad Amir Khusru Akhtar
Licensed under the Apache License, Version 2.0.
"""
import numpy as np

from empirical_reach.core import Experiment
from empirical_reach.robust_witness import cheapest_robust_witness, robust_witness_cost
from empirical_reach.worlds import make_hidden_bit_world
from empirical_reach.witness import CostedExperiment, incompatible_pairs


def z_channel(p0, p1, name):
    return Experiment(name, np.array([
        [1-p0, p0], [1-p1, p1], [1-p0, p0], [1-p1, p1]
    ]))


_, current, _, outcome = make_hidden_bit_world()
candidates = [
    CostedExperiment(current[0], 0.01),          # cheap but blind
    CostedExperiment(z_channel(0.49, 0.51, "cheap_weak"), 1.0),
    CostedExperiment(z_channel(0.30, 0.70, "medium"), 4.0),
    CostedExperiment(z_channel(0.05, 0.95, "strong"), 8.0),
]

print("Hidden incompatible pairs:", incompatible_pairs(current, outcome))
print("Target pairwise Bayes error <= 0.05 via Bhattacharyya bound")
for c in candidates:
    r = robust_witness_cost(current, c, outcome, error=0.05)
    print(c.experiment.name, None if r is None else {
        "cost_per_run": r.cost_per_run,
        "repeats": r.repeats,
        "total_cost": r.total_cost,
    })

best = cheapest_robust_witness(current, candidates, outcome, error=0.05)
print("BEST:", None if best is None else {
    "experiment": best.experiment.name,
    "repeats": best.repeats,
    "total_cost": best.total_cost,
})
