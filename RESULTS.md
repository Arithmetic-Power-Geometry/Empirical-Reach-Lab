# Initial Results and Interpretation

Copyright (C) 2026 Mohammad Amir Khusru Akhtar

This file records the claims the current benchmark can and cannot support.

## Minimal benchmark

Worlds are pairs `(x,z)` with four states. The current experimental alphabet contains only channels that depend on `x`; the consequential outcome is `F=z`.

Therefore every experiment in the current family preserves the equivalence classes `{(0,0),(0,1)}` and `{(1,0),(1,1)}`. Any finite product/composition of these conditionally independent `x`-only experiments remains `z`-blind.

A candidate new interaction directly couples to `z` and refines the empirical partition into four singleton states.

## Computational statements checked by tests

1. **Current repertoire insufficiency**
   - unresolved outcome diameter is `1.0`;
   - discordant twins exist inside each current empirical equivalence class.

2. **Composition does not escape closure**
   - joint composition of all current experiments remains unable to distinguish the hidden consequential bit.

3. **Blackwell comparison is not the same as reach expansion in this toy world**
   - the exact `x` channel Blackwell-dominates a noisy `x` channel;
   - both leave the same `z`-relevant empirical collapse intact.

4. **Supplied-model OED can be optimal and still miss the consequential coordinate**
   - when model identity is defined by `x`, the exact `x` experiment achieves 1 bit of model information gain;
   - the new `z` interaction has 0 bits of information about those supplied model labels;
   - nevertheless the new interaction removes the unresolved outcome diameter completely.

## What this does NOT prove

This benchmark does not yet establish a new scientific theory. It proves only a separation inside a deliberately constructed finite example.

It does not yet show that:

- the empirical-reach quantity is outside standard Blackwell experiment theory in full generality;
- active/adaptive sensing cannot recover the same distinction when richer actions are permitted;
- a practical algorithm can synthesize the new interaction without being handed it;
- the idea scales to continuous states, noisy outcomes, real laboratory systems, or unknown state spaces;
- the concept is novel relative to statistical deficiency, experiment comparison, observability, bisimulation, or sufficient-statistic theory.

## Strongest current finding

The benchmark demonstrates a clean distinction between:

- **optimizing information inside a supplied experimental/model alphabet**, and
- **strictly refining the empirical partition by adding a new primitive interaction**.

This is the result that should be stress-tested against prior art next.

## Next decisive tests

1. Replace equality-based partitions with approximate/statistical distinguishability.
2. Compare empirical reach to Blackwell sufficiency and Le Cam deficiency on finite channels.
3. Add adaptive policies and prove/test closure preservation.
4. Build a search space of candidate interactions not labeled by the hidden variable.
5. Test whether outcome-discordance can rank a useful new interaction above OED baselines.
6. Add noisy and many-state benchmarks where the useful interaction must be inferred rather than directly specified.
