"""Reproduce continuous cheapest-probe synthesis.

Copyright (C) 2026 Mohammad Amir Khusru Akhtar
Licensed under the Apache License, Version 2.0.
"""
from empirical_reach.continuous_genesis import search_continuous_probe
from empirical_reach.worlds import make_hidden_bit_world

_, current, _, outcome = make_hidden_bit_world()
for eps in (.10, .05, .01):
    best = search_continuous_probe(current, outcome, eps, angle_steps=181, strength_steps=100)
    print({
        "max_error": eps,
        "angle_radians": None if best is None else best.angle,
        "strength": None if best is None else best.strength,
        "repetitions": None if best is None else best.repetitions,
        "per_trial_cost": None if best is None else best.per_trial_cost,
        "total_cost": None if best is None else best.total_cost,
    })
