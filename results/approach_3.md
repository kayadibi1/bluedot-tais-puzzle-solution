# Approach #3 — The "Single Direction" Test

**Layer L = post-ReLU hidden 2 (`h2`, 64-dim).** N_train=7000, N_test=1500, all 8
features ~50% prevalence (so AUC=0.5 is true chance).

## Method
For each feature, three regimes on `h2`:
1. **BEST 1-D direction** — Fisher LDA (= diff-of-class-means in a whitened space),
   projected to a single unit vector → test ROC-AUC + best-threshold balanced accuracy.
2. **FULL LINEAR** — 64-d `LogisticRegression` test AUC.
3. **NONLINEAR ceiling** — small `MLPClassifier` (64,32) test AUC.

The README claim ("a single direction describes the feature") predicts, for the 7
linear features: `1D ≈ fullLinear ≈ nonlinear ≈ ceiling`. The odd feature F should
collapse on (1). We then distinguish two failure modes:
- **linear-but-multidirectional**: 1D LOW, fullLinear HIGH.
- **genuinely non-linear**: 1D LOW **and** fullLinear LOW; only nonlinear recovers it.

## Main table (test set, ROC-AUC)

| idx | feature    | 1D_AUC | 1D_balacc | fullLinear_AUC | nonlinear_AUC | (full − 1D) | (nl − full) |
|----:|------------|-------:|----------:|---------------:|--------------:|------------:|------------:|
| 0   | number     | 0.9970 | 0.975 | 0.9973 | 0.9961 | +0.0003 | −0.0012 |
| 1   | question   | 1.0000 | 1.000 | 1.0000 | 1.0000 | +0.0000 | +0.0000 |
| 2   | color      | 0.9974 | 0.977 | 0.9975 | 0.9961 | +0.0000 | −0.0013 |
| 3   | food       | 0.9948 | 0.983 | 0.9960 | 0.9956 | +0.0012 | −0.0005 |
| 4   | sentiment  | 0.9957 | 0.982 | 0.9955 | 0.9953 | −0.0002 | −0.0002 |
| **5** | **country** | **0.5158** | **0.569** | **0.4869** | **0.9898** | **−0.0289** | **+0.5029** |
| 6   | person     | 1.0000 | 1.000 | 1.0000 | 0.9999 | −0.0000 | −0.0001 |
| 7   | body_part  | 0.9984 | 0.983 | 0.9986 | 0.9982 | +0.0002 | −0.0004 |

For the 7 linear features, all three regimes agree to within ±0.002 — a single
direction genuinely suffices. **Country is the lone outlier.**

## F = country (index 5) — confidence: very high

Decisive numbers:
- Best single direction: **AUC 0.5158** (≈ chance), balanced-acc 0.569.
- Full 64-d linear: **AUC 0.4869** (≈ chance — *worse* than the 1-D, i.e. genuinely no
  signed linear signal at any rank).
- Nonlinear MLP: **AUC 0.9898** (matches the ~0.96–1.00 model accuracy claimed in README).
- Gap (nonlinear − fullLinear) = **+0.5029**, by far the largest of all 8 features
  (next largest is essentially 0).

## Failure mode: GENUINELY NON-LINEAR (quadratic / curved), not multi-directional-linear

Evidence ruling out "linear-but-multidirectional":
- Linear logistic on **top-k PCA dims** is at chance for *every* k:
  1d→0.490, 2d→0.499, 4d→0.500, 8d→0.479, 16d→0.487, 32d→0.487, **64d→0.487**.
  No number of *signed* linear directions helps — the class means coincide in all of `h2`.

Evidence it IS non-linear (second-order / quadratic):
- Adding **squared PCA features** (quadratic in `h2`) on the same subspace:
  top-8 PCs 0.479→**0.907**, top-16 0.487→**0.942** (gain ≈ +0.45).
- Full pairwise **quadratic map** on just the top-6 PCs: 0.495→**0.903**.
- Logistic on **|PCA top-16|** (sign-folded magnitudes): **0.911** — confirms the signal
  is sign-symmetric (lives in magnitude, not in signed coordinate).
- Squared projections onto the top diff-**covariance** eigen-axes give a clean ladder:
  m=1→0.738, m=2→0.769, m=12→0.791 — the discriminative signal is carried by the
  **covariance (2nd-moment) structure**, spread across ≳12 quadratic axes, with **zero**
  contribution from the 1st-moment (mean) shift.

## Geometry characterization
- **No mean separation**: along the best logistic axis, d′ ≈ 0.10; country+ mean and
  country− mean essentially coincide. The classes differ in their *shape*, not location.
- **Variance / covariance is the carrier**: country− is a **diffuse, high-variance** cloud
  (per-class total variance 18.1) while country+ is **more compact** (7.7); naive radius
  from the neg-centroid therefore *anti*-separates (AUC 0.21). So it is not a simple
  isotropic radial fold — it is a structured quadratic surface (different covariance
  ellipsoids), which a logistic on cross-/squared-terms or any MLP carves out easily.
- **Multi-lobe**: KMeans on country+ shows multiple antipodal lobes, each only weakly
  1-D separable (k=2 lobes ≈ 0.71–0.74 individually), consistent with the model spreading
  "country-ness" across several gated/quadratic units rather than one read-out direction.
- **Cheap to decode nonlinearly**: even an 8-unit MLP hits 0.977 and a 16-unit MLP 0.990,
  so the model is paying only a small nonlinear price — the feature is fully present at
  `h2`, just *not as a single linear direction*.

### How many linear directions reach ceiling?
**None** — signed linear separability is at chance at all 64 ranks. The feature only
becomes decodable once you allow **quadratic** features; ~12–16 *quadratic* axes recover
most of it (≈0.79 from squared diff-cov axes; ≈0.94 with the full quadratic map). This is
the teaser for the geometry: country is encoded in the **second-order (covariance) /
sign-folded** structure of `h2`, distributed over a dozen-ish directions, not a line.

## Caveats
- Whitening/LDA used a 1e-6 ridge (h2 covariance is near-singular: only ~17% of units
  active). Conclusions are robust — the linear collapse reproduces identically with plain
  standardized logistic at full rank (0.487).
- "Genuinely non-linear" here means *not linearly decodable by any signed direction at
  layer L*; downstream `h3` could still linearize it. The claim is specifically about `h2`.
- The MLP ceiling (0.99) and the quadratic probe (0.94) both confirm the signal is fully
  present at `h2`; the gap between them is just probe capacity, not missing information.

## Artifacts
- `a3_single_direction.py` — main 3-regime table.
- `a3_geometry.py` — PCA-rank, quadratic, |.|-fold, cluster, MLP-ceiling sweeps.
- `a3_confirm_fold.py` — radius/variance + diff-covariance quadratic-axis ladder.
- `results/approach3_raw.npy` — raw numbers.
