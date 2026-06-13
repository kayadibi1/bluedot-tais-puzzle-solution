# Changelog

All notable changes to this solution are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## [1.0.0] - 2026-06-13

Initial solution release for BlueDot Technical AI Safety Puzzle 1.

### Task 1 - identify the non-linear feature
- Found `country` (index 5) is the feature not linearly represented at layer L (post-ReLU hidden 2).
- Linear-probe AUC at L: `country` 0.49 (chance) vs 0.996-1.000 for the other seven features.
- Collapse is localized to L: `country` is linearly readable at every other layer (emb/h0/h1/h3 ~0.99) and only fails at hidden 2; an MLP probe recovers it at 0.99, confirming the information is present.
- Added `_precompute.py` (caches all layer activations) and `make_figures.py` (reproduces the numbers + Figures 1-3).

### Task 2 - characterize the representation
- `country` is a `food`-gated, sign-flipping direction: the readout axis reverses with `food` (cosine -0.996 across food halves); conditioning on `food` restores linear separability (0.52 -> 0.98, class means 0.01 -> 0.85).
- Weight-level trace: the network reads it with a two-bank ReLU multiplexer (a food-absent bank and a food-present bank); ablating the food/gate axis collapses the `country` logit to chance.
- Ruled out label-XOR, pure radial shell, and multi-directional-linear accounts.
- Added 15 analysis scripts (`a1`-`a10`, `r1`-`r5`) and their reports under `results/`.

### Task 3 - a weirder representation
- Trained a model that re-encodes `country` as a frequency-4 ring (`task3_train.py`, `task3_probe.py`).
- Every polynomial probe below degree 4 fails (linear/quadratic/cubic ~0.50-0.55); only a degree-4 / Fourier readout recovers it (0.98). This strictly beats the original code, which fell to a degree-2 probe.
- Added the trained ring model (`cache/task3/ring_model.pt`) and Figure 4.

### Docs
- Added write-ups: `SUBMISSION_tasks1-2.md`, `SUBMISSION_clean.txt`, `SUBMISSION.html`.
- README now opens with a solution summary and reproduce steps.
