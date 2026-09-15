# Empirical Reach Lab

A reproducible research laboratory for testing whether autonomous scientific systems can detect limits of a fixed experimental repertoire and discover new information-bearing interactions that expand empirical reach.

Copyright (C) 2026 Mohammad Amir Khusru Akhtar

Licensed under the Apache License, Version 2.0.

## Research question

A scientific agent may execute, optimize, and compose a very large number of experiments while remaining inside the closure of a fixed experimental alphabet. The central question tested here is:

> Can a consequential distinction remain invisible to every experiment constructible from the current alphabet, yet become visible after adding one genuinely new interaction channel?

This repository does **not** assume that such a result is a new theory. It is designed to test the mathematical separation and compare it against classical information structures, Blackwell comparison, active sensing, and optimal experimental design.

## Core definitions

Let `W` be a finite set of world states and let an experiment be a stochastic channel `E(y|w)`.

For a repertoire `T`, two states are empirically equivalent when every experiment in the current closure induces the same output distribution:

`w1 ~_T w2  iff  E(.|w1) = E(.|w2) for every E in Cl(T)`.

For a consequential outcome `F(w)`, define the unresolved empirical diameter

`Delta_T(F) = max d(F(w1), F(w2))` over pairs `w1 ~_T w2`.

A newly introduced experiment expands empirical reach when it refines this equivalence relation and strictly reduces `Delta_T(F)`.

## What is tested

1. **Closure obstruction** — no adaptive policy using only experiments that ignore a latent consequential coordinate can recover that coordinate.
2. **Blackwell baseline** — channels that differ only by garbling can improve quality without creating a qualitatively new distinction.
3. **Reach expansion** — a new interaction channel can split an equivalence class that all existing experiments preserve.
4. **OED failure case** — an information-gain objective over a supplied hypothesis family can score current experiments highly while remaining blind to an excluded consequential coordinate.
5. **Discordant twins** — pairs identical under all current measurements but different in outcome certify repertoire insufficiency for that outcome.

## Quick start

```bash
python -m pip install -e .[test]
pytest -q
python experiments/run_all.py
```

Generated results are written to `results/summary.json`.

## Repository structure

- `src/empirical_reach/core.py` — finite-channel definitions, partitions, Blackwell checks, empirical reach metrics.
- `src/empirical_reach/worlds.py` — synthetic worlds designed to expose fixed-alphabet limitations.
- `src/empirical_reach/oed.py` — simple information-gain baseline over a supplied model family.
- `experiments/run_all.py` — one-command reproducible evaluation.
- `tests/` — theorem-like computational checks.
- `RESULTS.md` — interpretation and falsification criteria.
- `.github/workflows/tests.yml` — CI.

## Falsification criteria

This project should **not** claim a breakthrough if any of the following occurs:

- the proposed reach metric is just a re-expression of a standard Blackwell quantity with no new consequence;
- the best separation disappears when adaptive policies are allowed;
- the proposed `new interaction` is actually constructible from the original experiment closure;
- a standard active sensing or OED baseline already identifies the same missing distinction without expanding its primitive observation alphabet;
- the construction depends on encoding the desired answer into the new sensor by hand rather than deriving a general search principle.

## Current status

The initial codebase establishes only the minimal finite-state separation needed for further research. It is a **test bed**, not a novelty claim.
