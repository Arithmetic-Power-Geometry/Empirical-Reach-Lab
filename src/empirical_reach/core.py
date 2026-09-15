"""Core finite-state objects for Empirical Reach Lab.

Copyright (C) 2026 Mohammad Amir Khusru Akhtar
Licensed under the Apache License, Version 2.0.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Experiment:
    """A finite stochastic experiment/channel P(y|w)."""
    name: str
    channel: np.ndarray

    def __post_init__(self) -> None:
        c = np.asarray(self.channel, dtype=float)
        if c.ndim != 2:
            raise ValueError("channel must be a 2D array")
        if np.any(c < -1e-12):
            raise ValueError("channel probabilities must be non-negative")
        if not np.allclose(c.sum(axis=1), 1.0):
            raise ValueError("each world-state row must sum to one")
        object.__setattr__(self, "channel", c)


def _signature(experiments: Sequence[Experiment], w: int, decimals: int = 12) -> tuple:
    """Hashable signature of all response laws available for world w."""
    return tuple(
        tuple(np.round(e.channel[w], decimals=decimals).tolist())
        for e in experiments
    )


def empirical_partition(experiments: Sequence[Experiment], n_worlds: int) -> list[tuple[int, ...]]:
    """Partition worlds by equality of all current output distributions."""
    groups: dict[tuple, list[int]] = {}
    for w in range(n_worlds):
        groups.setdefault(_signature(experiments, w), []).append(w)
    return [tuple(v) for v in groups.values()]


def unresolved_diameter(experiments: Sequence[Experiment], outcome: Sequence[float]) -> float:
    """Maximum consequential variation hidden inside an empirical class."""
    f = np.asarray(outcome, dtype=float)
    part = empirical_partition(experiments, len(f))
    diam = 0.0
    for block in part:
        vals = f[list(block)]
        diam = max(diam, float(vals.max() - vals.min()))
    return diam


def reach_gain(experiments: Sequence[Experiment], candidate: Experiment, outcome: Sequence[float]) -> float:
    before = unresolved_diameter(experiments, outcome)
    after = unresolved_diameter([*experiments, candidate], outcome)
    return before - after


def mutual_information(channel: np.ndarray, prior: Sequence[float] | None = None) -> float:
    """I(W;Y) in bits for a finite channel."""
    c = np.asarray(channel, dtype=float)
    n = c.shape[0]
    p_w = np.full(n, 1.0 / n) if prior is None else np.asarray(prior, dtype=float)
    p_w = p_w / p_w.sum()
    p_y = p_w @ c
    out = 0.0
    for w in range(n):
        for y in range(c.shape[1]):
            joint = p_w[w] * c[w, y]
            if joint > 0 and p_y[y] > 0:
                out += joint * np.log2(c[w, y] / p_y[y])
    return float(out)


def blackwell_dominates(a: Experiment, b: Experiment, tol: float = 1e-9) -> bool:
    """Diagnostic: whether b is representable as a row-stochastic garbling of a."""
    A, B = a.channel, b.channel
    m, _ = A.shape
    if B.shape[0] != m:
        return False
    G, *_ = np.linalg.lstsq(A, B, rcond=None)
    if np.max(np.abs(A @ G - B)) > tol:
        return False
    if np.min(G) < -tol:
        return False
    if np.max(np.abs(G.sum(axis=1) - 1.0)) > tol:
        return False
    return True


def product_experiment(experiments: Sequence[Experiment], name: str | None = None) -> Experiment:
    """Parallel composition of conditionally independent experiments."""
    if not experiments:
        raise ValueError("need at least one experiment")
    n = experiments[0].channel.shape[0]
    if any(e.channel.shape[0] != n for e in experiments):
        raise ValueError("world counts must match")
    output_sizes = [e.channel.shape[1] for e in experiments]
    rows = []
    for w in range(n):
        probs = []
        for ys in product(*[range(s) for s in output_sizes]):
            p = 1.0
            for e, y in zip(experiments, ys):
                p *= e.channel[w, y]
            probs.append(p)
        rows.append(probs)
    return Experiment(name or "x".join(e.name for e in experiments), np.asarray(rows))
