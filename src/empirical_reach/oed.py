"""Simple optimal-experimental-design baselines.

Copyright (C) 2026 Mohammad Amir Khusru Akhtar
Licensed under the Apache License, Version 2.0.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .core import Experiment, mutual_information


def model_information_gain(
    experiment: Experiment,
    model_labels: Sequence[int],
    prior_world: Sequence[float] | None = None,
) -> float:
    """I(M;Y) where M is a supplied model-class label on world states.

    This baseline intentionally represents the conventional case: the design
    objective can only reward distinctions present in the supplied model
    family. If the consequential coordinate is absent from `model_labels`,
    no amount of optimizing this objective can explicitly target it.
    """
    labels = np.asarray(model_labels)
    n = len(labels)
    if experiment.channel.shape[0] != n:
        raise ValueError("model_labels length must match channel world count")
    p_w = np.full(n, 1.0 / n) if prior_world is None else np.asarray(prior_world, dtype=float)
    p_w = p_w / p_w.sum()

    unique = np.unique(labels)
    p_m = np.array([p_w[labels == m].sum() for m in unique])
    cond = []
    for m in unique:
        idx = labels == m
        weights = p_w[idx] / p_w[idx].sum()
        cond.append(weights @ experiment.channel[idx])
    cond = np.asarray(cond)
    return mutual_information(cond, p_m)


def best_experiment_by_model_ig(
    experiments: Sequence[Experiment],
    model_labels: Sequence[int],
) -> tuple[Experiment, float]:
    scored = [(e, model_information_gain(e, model_labels)) for e in experiments]
    return max(scored, key=lambda x: x[1])
