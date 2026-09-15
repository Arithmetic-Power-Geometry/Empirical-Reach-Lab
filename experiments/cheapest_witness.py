"""Reproduce the minimum-cost incompatible-world witness result.

Copyright (C) 2026 Mohammad Amir Khusru Akhtar
Licensed under the Apache License, Version 2.0.
"""
import numpy as np

from empirical_reach.core import Experiment, reach_gain
from empirical_reach.worlds import make_hidden_bit_world
from empirical_reach.witness import CostedExperiment, cheapest_witness, incompatible_pairs, witnessed_pairs


_, current, reveal_z, outcome = make_hidden_bit_world()

candidates = [
    CostedExperiment(current[1], 1.0),
    CostedExperiment(current[0], 2.0),
    CostedExperiment(current[2], 3.0),
    CostedExperiment(Experiment("weak_z_interaction", np.array([
        [0.70, 0.30], [0.30, 0.70], [0.70, 0.30], [0.30, 0.70]
    ])), 5.0),
    CostedExperiment(reveal_z, 9.0),
]

print("Currently hidden consequentially incompatible pairs:", incompatible_pairs(current, outcome))
print("\nCandidate audit:")
for c in candidates:
    print({
        "experiment": c.experiment.name,
        "cost": c.cost,
        "witnessed_pairs": witnessed_pairs(current, c.experiment, outcome),
        "reach_gain": reach_gain(current, c.experiment, outcome),
    })

chosen = cheapest_witness(current, candidates, outcome)
print("\nCheapest witness:", None if chosen is None else {
    "experiment": chosen.experiment.name,
    "cost": chosen.cost,
    "witnessed_pairs": witnessed_pairs(current, chosen.experiment, outcome),
})
