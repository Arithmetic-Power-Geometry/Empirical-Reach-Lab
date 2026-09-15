# Prior-Art Boundary: Cheapest Witness Search

Copyright (C) 2026 Mohammad Amir Khusru Akhtar
Licensed under the Apache License, Version 2.0.

## Result of the adversarial check

The finite deterministic problem

> choose a minimum-cost subset of supplied tests so that every required pair of possible worlds is separated

is a weighted/partial form of the classical **Test Cover / Minimum Test Collection** problem. Therefore the repository MUST NOT claim novelty for minimum-cost test-subset selection, set-cover-style branch-and-bound, or pair-separation alone.

## Exact reduction

Let U be the set of consequentially incompatible world pairs that remain equivalent under the current experimental repertoire. For every supplied candidate experiment e define

    S_e = { pair in U : e gives different response laws to the pair }.

Then finding a minimum additive-cost supplied experiment set E such that every pair is witnessed is exactly

    minimize sum_{e in E} c_e
    subject to union_{e in E} S_e = U.

This is weighted set cover over pair-universe U, and when U contains all pairs it is the classical test-cover formulation.

## What survives

The repository remains useful as a falsification laboratory. The potentially new target is NOT selection from a supplied test list. It is the stronger problem in which:

1. current experiments induce consequentially discordant equivalence classes;
2. no supplied experiment is assumed to solve the problem;
3. the system must synthesize a new physically realizable probing operation from an interaction/process grammar or continuous physical design space;
4. finite-sample reliability and total evidence cost are included;
5. the synthesized operation must create empirical distinguishability unavailable in the current experimental closure;
6. ideally the missing latent variable is not named or supplied.

This boundary must be preserved in any manuscript.

## Claim status

- Cheapest subset of supplied tests: **not novel**.
- Branch-and-bound for supplied tests: **not novel by itself**.
- Finite-sample cost-aware discrimination: substantial prior art exists in experimental design/sequential testing; do not claim broadly.
- Consequential-pair restriction: useful formulation, novelty unverified.
- Synthesis of a previously unspecified physical probe that expands the empirical alphabet without naming the missing variable: **surviving candidate**, still requiring dedicated prior-art validation and a nontrivial synthesis benchmark.
