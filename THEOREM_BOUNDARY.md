# Empirical-Reach theorem boundary

Copyright (C) 2026 Mohammad Amir Khusru Akhtar  
Licensed under the Apache License, Version 2.0.

## Problem
Let `K` be current empirical knowledge, `W(K)` its possible worlds, `T` the current experimental repertoire, and `F` a consequential outcome. Define

`w ~_T w'` iff every experiment constructible from `T` has the same response law in `w` and `w'`.

A pair is **consequentially discordant** when `w ~_T w'` but `F(w) != F(w')`.

The target is not ordinary minimum-cost test selection. It is to detect that the current empirical repertoire is insufficient and, where a physically admissible interaction family is available, synthesize the least-cost interaction/program that breaks the relevant equivalence.

## Proposition 1 — Closure obstruction
If `w ~_T w'`, every adaptive policy whose actions are restricted to experiments in `Cl(T)` induces the same transcript distribution in `w` and `w'`. Hence no amount of repetition, computation, adaptivity, or decision rule restricted to `Cl(T)` can distinguish the pair.

### Proof sketch
By definition every allowed experiment has identical conditional response law in the two worlds. Induct on transcript length. The base transcript is identical. If transcript distributions agree through step `t`, the adaptive policy chooses the same distribution over next experiments conditional on each transcript; each chosen experiment has the same response law in both worlds, so the extended transcript distributions agree. Therefore every terminal decision statistic also has the same distribution. QED.

## Proposition 2 — Consequential insufficiency certificate
If there exist `w,w'` with `w ~_T w'` and `F(w) != F(w')`, then `T` is insufficient for universally correct prediction/decision of `F` over `W(K)`.

This is a certificate of missing empirical reach; it does not identify a unique missing variable or physical mechanism.

## Proposition 3 — Strict reach expansion
If an admissible new experiment `e*` has different response laws for at least one consequentially discordant pair, then

`~_{T union {e*}}` is a strict refinement of `~_T`

on the relevant possible-world set. Thus `e*` expands empirical reach even when it carries no information about a supplied model label used by a conventional OED objective.

## Proposition 4 — Finite supplied-candidate reduction
For a finite supplied set of candidate experiments, minimizing additive cost while separating all discordant pairs reduces to weighted test/set cover. This combinatorial optimization is not claimed as novel.

## Proposition 5 — Reliable evidence cost
For noisy response laws `P_e(.|w)` and `P_e(.|w')`, physical unit cost alone is insufficient. A sufficient iid repetition count at target equal-prior Bayes error `epsilon` follows from

`P_error(n) <= 0.5 * BC(P,Q)^n`,

so evidence cost is `setup(e) + n_e(epsilon) * trial_cost(e)`. A weak cheap probe can therefore be more expensive scientifically than a strong costly probe.

## Critical limitation / no-go boundary
Discordant observations alone cannot uniquely synthesize an unknown physical interaction. Without assumptions, prior data, a simulator, a learned surrogate, physical laws, or an admissible interaction grammar linking intervention parameters to possible response laws, infinitely many unseen mechanisms are observationally compatible with the same current data.

Therefore the strongest defensible formulation is:

> infer an empirical-reach deficiency from consequential discordance, then synthesize the minimum-cost closure-breaking experiment **relative to an explicitly stated admissible interaction model/grammar**.

A manuscript must not claim creation of a physically valid probe from literally no response model or physical assumptions.

## Manuscript-ready core claim
The potentially distinctive contribution is the separation of two questions:
1. Which experiment is optimal inside the current empirical alphabet?
2. Does the current empirical alphabet itself collapse worlds that matter, and what is the minimum-cost admissible expansion that resolves that collapse?

The first has extensive prior art. The second is the Empirical-Reach target and must be positioned against observability, active sensing, model discrimination, causal intervention design, sensor placement, and automated experiment/instrument design.
