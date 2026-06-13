# Approach #7 — Radial / Shell (distance-based) hypothesis

**Layer L = `h2`, post-ReLU hidden-2, 64-dim** (83% sparse, all activations >= 0).
Script: `a7_radial.py`. Raw log: `results/approach_7_log.txt`, numbers: `results/approach_7_data.json`.

## Result in one line
**F = `country` (index 5).** It is the only feature that is *not* linearly readable
at L, and its representation is a **concentric radial / shell structure**: the two
classes share (essentially) the same center but differ in **scale** — `country=1`
is a **tight core**, `country=0` a **diffuse surrounding shell**. No hyperplane
separates them; a radial / Mahalanobis coordinate does.

## Step 1 — Linear-probe AUC sweep (find F)
Standardized h2, L2 logistic, test AUC:

| idx | feature   | linear AUC |
|-----|-----------|-----------:|
| 0 | number    | 0.9973 |
| 1 | question  | 1.0000 |
| 2 | color     | 0.9975 |
| 3 | food      | 0.9960 |
| 4 | sentiment | 0.9955 |
| **5** | **country** | **0.4902** |
| 6 | person    | 1.0000 |
| 7 | body_part | 0.9986 |

`country` is the lone outlier at chance (0.49). Every other feature is ~0.995–1.0.
The full model still classifies country at ~0.96–1.0, so the information is present
at L but **not in any linear direction** — consistent with a non-linear (radial) code.

## Step 2 — Radial coordinate test (country)
Distance from the global mean `mu` (test set):

| coordinate | class0 (country=0) | class1 (country=1) | 1-D AUC |
|---|---:|---:|---:|
| `\|\|x - mu_global\|\|` | 1.583 ± 0.519 | **1.048 ± 0.368** | **0.819** |
| `\|\|x - mu_class0\|\|`  | 1.583 | 1.048 | 0.819 |
| `\|\|x - mu_class1\|\|`  | 1.583 | 1.048 | 0.819 |
| `\|\|x\|\|` (from origin) | 4.982 | 4.832 | 0.541 |

A single scalar — distance from the cloud center — gives **AUC 0.819** (best radius
threshold ≈ 74% accuracy), versus **0.49** for the best linear direction. The result
is identical whichever center you use (global / class-0 / class-1) because the centers
coincide (next section). Distance from the *origin* does **not** work (0.54): the
structure is radial about the **data center**, not about 0.

## Step 3 — Quadratic / radial-augmented probe, and QDA vs LDA
| model | country AUC |
|---|---:|
| linear logistic | 0.4902 |
| logistic + per-dim squares + `\|\|x\|\|^2` | **0.9408** |
| LDA | 0.5161 |
| **QDA** | **0.9899** |
| Mahalanobis (per-class whitened radius, `d0 - d1`) | **0.9946** |

- Adding **quadratic / radial features lifts AUC from 0.49 → 0.94**.
- **QDA − LDA gap = +0.474** (0.516 → 0.990). QDA models per-class covariance; the
  fact that it recovers the feature almost perfectly while LDA cannot is the textbook
  signature of a **covariance-difference (shell/elliptic)** code.
- A per-class Mahalanobis classifier reaches **0.995** — the feature is, to good
  approximation, "which class's ellipsoid am I inside."

## Step 4 — Class-covariance comparison (concentric, different scale)
| quantity | country=0 | country=1 |
|---|---:|---:|
| cov trace | 2.844 | **1.280** |
| cov logdet | −534.1 | −541.6 |
| mean radius from own center | 1.686 | **1.131** |

- **Trace ratio class1/class0 = 0.450** — class1 has < half the total variance.
  Per-dimension, mean var ratio class0/class1 = **2.22**.
- **Centers coincide:** `||mu1 − mu0|| = 0.013` against `||mu_global|| = 4.80`,
  cosine(mu0, mu1) = **0.99999785**. The classes are **concentric**.
- **Bhattacharyya decomposition:** mean-term = 0.002, covariance-term = 0.765 →
  **cov/mean ratio = 454**. Almost 100% of the class separation is carried by the
  *covariance (scale)* difference, essentially none by a mean shift. This is the
  defining numeric fingerprint of a shell vs core, not a hyperplane.

So: **`country=1` = compact core ellipsoid; `country=0` = same-centered, ~2.2× more
diffuse shell.**

## Step 5 — Contrast with a known-linear feature (`question`)
Same battery on `question` (linear AUC 1.0):
- radial 1-D AUC = 0.60 (weak), **QDA − LDA gap = +0.000**, cov/mean Bhat ratio = **0.01**
  (separation is 99% mean-shift), `||mu1 − mu0|| = 0.586` (well-separated centers).

Every linear feature has its separation in the **mean-shift** term and gains nothing
from QDA/quadratic features. The radial signature is **specific to `country`**.

## Verdict
**F = `country` (idx 5), high confidence.** At layer L it is represented as a
**concentric radial / shell geometry**:
- the two classes share the same center (||Δmu|| ≈ 0.01, negligible),
- they differ in **scale**: `country=1` forms a **tight core** (trace 1.28),
  `country=0` a **diffuse shell** (trace 2.84, ~2.2× per-dim variance),
- a **radial coordinate** (distance from center) separates them at AUC **0.82**,
  and a per-class **Mahalanobis / QDA** decision at **0.99** — versus **0.49** for
  any linear probe.

Mechanistically: the downstream layers can read country by an *inside-the-core vs
outside* (norm / quadratic) test, which a single ReLU-MLP layer can implement but a
linear probe cannot — hence the "non-linear feature" at L.

## Caveats
1. The radius gap, while highly significant in AUC, is not perfectly separable in 1-D
   (best single-threshold acc ≈ 74%); the full ~0.99 separation needs the per-class
   *anisotropic* covariance (Mahalanobis/QDA), i.e. it is an **ellipsoidal shell**,
   not a perfectly isotropic spherical one. The "radius" is the dominant but not the
   sole axis.
2. QDA/Mahalanobis required mild covariance regularization (reg 0.05 / shrinkage 0.1)
   because h2 is 83% sparse and within-class covariances are near-rank-deficient; the
   conclusion (huge QDA>LDA gap, scale-driven Bhattacharyya) is robust to the reg level.
3. Analysis is at h2 only; I did not trace which upstream directions build the core vs
   shell, nor confirm the downstream MLP literally uses a norm test (only that the
   information is geometrically radial). Other agents' direction/causal analyses would
   complement this.
4. `country=1` being the *core* (not the shell) is the empirical finding here; it is the
   minority-variance class, plausibly because country mentions cluster onto a small set
   of country tokens, collapsing them into a compact region.

Path: `results/approach_7.md`
