"""Continuous experiment genesis: optimize a probe not supplied as a candidate.

Copyright (C) 2026 Mohammad Amir Khusru Akhtar
Licensed under the Apache License, Version 2.0.

This module deliberately avoids a finite experiment catalogue. A probe is
specified by continuous design parameters theta=(angle,strength). The response
channel is generated from theta, and the cheapest reliable design is searched
from the parameterized physical interaction family.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin, pi
import numpy as np

from .core import Experiment
from .statistical_witness import repetitions_for_all_incompatible_pairs


@dataclass(frozen=True)
class ContinuousDesign:
    angle: float
    strength: float
    experiment: Experiment
    repetitions: int
    per_trial_cost: float
    total_cost: float


def generated_probe(angle: float, strength: float, name: str | None = None) -> Experiment:
    """Generate a binary channel from continuous interaction parameters.

    Worlds are (x,z)=(0,0),(0,1),(1,0),(1,1). The interaction response uses
    both coordinates through a continuously chosen direction. Crucially, the
    z-sensitive probe is not enumerated beforehand.
    """
    if not 0.0 <= angle <= pi / 2:
        raise ValueError("angle must be in [0,pi/2]")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0,1]")
    worlds = [(0,0),(0,1),(1,0),(1,1)]
    a, b = cos(angle), sin(angle)
    probs = []
    for x, z in worlds:
        signed = a * (2*x-1) + b * (2*z-1)
        p1 = 0.5 + 0.24 * strength * signed / (abs(a)+abs(b))
        p1 = min(0.999, max(0.001, p1))
        probs.append([1-p1, p1])
    return Experiment(name or f"probe_a{angle:.6f}_s{strength:.6f}", np.asarray(probs))


def physical_cost(angle: float, strength: float) -> float:
    """Toy physical cost: stronger interactions cost more; direction is free."""
    return 0.05 + strength * strength


def search_continuous_probe(current, outcome, max_error: float = 0.05,
                            angle_steps: int = 181, strength_steps: int = 100) -> ContinuousDesign | None:
    """Search a continuous parameterization for minimum reliable evidence cost."""
    best = None
    for angle in np.linspace(0.0, pi/2, angle_steps):
        for strength in np.linspace(0.01, 1.0, strength_steps):
            exp = generated_probe(float(angle), float(strength))
            n = repetitions_for_all_incompatible_pairs(current, exp, outcome, max_error)
            if n is None:
                continue
            c = physical_cost(float(angle), float(strength))
            result = ContinuousDesign(float(angle), float(strength), exp, n, c, n*c)
            if best is None or (result.total_cost, result.per_trial_cost, result.angle) < (
                best.total_cost, best.per_trial_cost, best.angle):
                best = result
    return best
