"""Synthetic worlds for closure and empirical-reach tests.

Copyright (C) 2026 Mohammad Amir Khusru Akhtar
Licensed under the Apache License, Version 2.0.
"""
from __future__ import annotations

import numpy as np

from .core import Experiment


def make_hidden_bit_world(noise: float = 0.05):
    """Return a 4-world benchmark with visible bit x and hidden consequential bit z.

    Worlds are (x,z) in lexicographic order:
      0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)

    Existing experiments reveal only x (possibly through different garblings).
    The candidate new interaction reveals z. The outcome is F=z.
    """
    if not 0 <= noise < 0.5:
        raise ValueError("noise must lie in [0, 0.5)")

    worlds = [(0, 0), (0, 1), (1, 0), (1, 1)]

    exact_x = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ])

    noisy_x = np.array([
        [1-noise, noise],
        [1-noise, noise],
        [noise, 1-noise],
        [noise, 1-noise],
    ])

    reveal_z = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    # A second in-alphabet channel: still only a function of x, so it cannot split z.
    ternary_x = np.array([
        [0.75, 0.25, 0.0],
        [0.75, 0.25, 0.0],
        [0.0, 0.25, 0.75],
        [0.0, 0.25, 0.75],
    ])

    current = [
        Experiment("exact_x", exact_x),
        Experiment("noisy_x", noisy_x),
        Experiment("ternary_x", ternary_x),
    ]
    candidate = Experiment("new_interaction_z", reveal_z)
    outcome = np.array([z for _, z in worlds], dtype=float)
    return worlds, current, candidate, outcome
