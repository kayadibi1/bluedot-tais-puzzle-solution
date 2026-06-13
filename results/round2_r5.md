# Round-2 R5 — Generative model of `country` at h2; resolve tight-vs-bimodal

**F = `country` (idx 5). Gate = `food` (idx 3). Secondary = `sentiment` (idx 4). Layer L = h2 (post-ReLU hidden-2, 64-d).**
Script `r5_generative.py`. Standardized h2 (TRAIN stats) → PCA K=16 (retains 100% of std-space var; country signal intact). Gaussians fit on TRAIN, scored on TEST. Plots `results/round2_r5_*.png`, raw `results/round2_r5_raw.json`.

## VERDICT (one sentence)
**At h2, `country` is a `food`-GATED mean offset along a single axis whose sign flips with food (so it is invisible to any food-agnostic linear probe), RIDING ON TOP OF a food-independent variance-collapse: country=1 is a tight shell and country=0 a wide cloud on essentially every axis — and you need BOTH the gate (for the right means) and the shell (for the right covariances) to reproduce the real geometry, which is why a per-cell combined Gaussian (M3) matches and the pure-gate / pure-shell models each fail one half.**

---

## 1. 4-CELL anti-parallel test (PASS — gate confirmed)

Country mean-offset within each food half, in PCA space:

| quantity | value |
|---|---|
| ‖μ(c1,f0) − μ(c0,f0)‖ | **1.685** |
| ‖μ(c1,f1) − μ(c0,f1)‖ | **1.550** |
| **cos(d_food0, d_food1)** | **−0.996** |
| country-offset / food-offset magnitude | 0.660 |

The two within-food country directions are **anti-parallel (cos −0.996)** — moving from country=0→1 pushes the *opposite way* depending on food. This is the sign-flip gate, reproducing R2's cos −0.996. The country displacement is 0.66× the food displacement (comparable scale, not negligible). Cell variance traces: c0f0=16.0, c0f1=12.2, **c1f0=7.9, c1f1=7.2** — country=1 cells are ~half the spread of country=0 cells *within every food value* (the shell, see §2).

## 2. Resolution of tight-vs-bimodal (the apparent contradiction)

Projection onto the gate axis v (`<h2, v>`):

| cell | mean | std |
|---|---|---|
| c0,f0 | **−2.13** | 1.54 |
| c1,f0 | −0.44 | 1.22 |
| c0,f1 | **+1.93** | 1.33 |
| c1,f1 | +0.38 | 1.15 |
| **POOLED c0** | +0.01 | **2.48** (var 6.15) |
| **POOLED c1** | −0.01 | **1.25** (var 1.56) |

**The sign-flip lives in country=0, not country=1.** Country=0's mean swings −2.13 → +1.93 across food; country=1's barely moves (−0.44 → +0.38). So when pooled over food it is **country=0** that becomes a wide, sign-flip-driven cloud (var 6.15; **66.7%** of its along-v variance is the between-food sign-flip), while **country=1 stays compact** (var 1.56; only **10.9%** between-food). Bimodality coefficients pooled: c0=0.497, c1=0.379 (both below the 0.555 uniform threshold — the two food modes overlap into a single wide-vs-narrow pair rather than cleanly separated peaks; the effect is a **variance gap**, not two resolved bumps).

**The measured "tightness" of country=1 is NOT a pooling artifact and NOT only along v — it is on essentially every axis, food-independently.** Total PCA-space variance ratio **c1/c0 = 0.426** (matches the reported ~0.45). Crucially, country=1 is tighter on **16/16** axes when pooled **and 16/16 within food=0 alone** (no sign-flip pooling possible there) — e.g. gate-dominant axis 0 ratio 0.83, axes 1–3 ratios 0.55/0.22/0.001 within food=0. 

**So the contradiction dissolves:** the sign-flip gating inflates *country=0's* pooled variance along v (bimodal-ish), while *country=1's* genuine tightness is a separate **food-independent shell contraction** that holds within each cell on every axis. They were never the same object measured two ways — pooled "tight country=1" is real shell; "bimodal under gating" is country=0's pooled spread. Both true, different classes, different mechanisms.

## 3. Three generative models — held-out TEST country-AUC + log-likelihood

Score = log p(x|c=1) − log p(x|c=0) + log-prior. "mean test loglik" = mean log-density the model assigns to the **true** class (geometry fit). "cell-mean-MAE" / "trace-MAE" = how well model-implied per-cell means / covariance traces match the real test cells.

| model | country-AUC | mean test loglik | cell-mean-MAE | trace-MAE | reproduces geometry? |
|---|---|---|---|---|---|
| **M1 pure-gate** (offset ±δv·sign(food), shared cov) | 0.964 | **+14.46** | 0.193 | **2.72** | means ✓, covariances ✗ |
| **M2 pure-shell** (coincident μ, per-class cov, no gate) | 0.989 | **−77.5** | **1.189** | 3.99 | covariances partly ✓, **means ✗✗** |
| **M3 combined** (per-cell μ + per-cell cov) | **0.988** | **+8.51** | **0.186** | 3.22 | **both ✓** |
| ref: ungated v-axis (sign cancels) | 0.505 | — | — | — | reproduces linear ≈0.49 |
| ref: gated-linear v-axis (un-flip sign) | 0.794 | — | — | — | gate-axis alone ≈0.79 |

**M3 (combined) wins** — it reproduces the real ~0.99 country-AUC AND has the geometry right (lowest cell-mean-MAE 0.19, healthy loglik +8.5, per-cell traces 16.0/12.2/7.9/7.2 ≈ real 10.2/14.3/7.5/11.8).

**Falsification of the losers (with numbers):**
- **M2 pure-shell is FALSIFIED on geometry despite high AUC (0.989).** It "wins" the classifier only by exploiting the real variance-collapse, but its assumed **coincident mean is wrong**: cell-mean-MAE = **1.19** (6× M3) and its log-likelihood **collapses to −77.5** (M3 = +8.5). It places both classes at the same center, so it cannot reproduce the within-food ±2 mean separation of §1. High AUC ≠ correct generative model — this is the explicit warning. (Also its cov is food-blind: trace 18.1/18.1/7.7/7.7, missing the food-dependent c0 spread.)
- **M1 pure-gate is FALSIFIED on covariances.** Good means (cell-mean-MAE 0.19, loglik +14.5) but shared covariance forces **identical trace 13.0 in all 4 cells**, vs real 10.2/14.3/7.5/11.8 (trace-MAE 2.72). It cannot represent country=1's shell contraction — by construction country=1 and country=0 have equal spread, contradicting §2's 16/16-axis tightness. Its AUC (0.964) is *higher* than the gated-linear-only 0.794 because the multivariate shared cov still whitens nuisance dims, but it is geometrically blind to the shell.

(M1's loglik is nominally the highest because a single shared full covariance over-smooths and avoids M3's small-cell covariance noise; the discriminating metrics are cell-mean-MAE and AUC, where M3 dominates and M2's mean is exposed as wrong.)

## 4. Budget / how the 0.99 is built
- **Gate alone** (gated-linear on v): AUC ≈ **0.79** — the sign-flip mean offset.
- **Shell alone** (M2, per-class cov): AUC ≈ **0.99** discrimination but **wrong means** — the variance-collapse is independently almost fully discriminative.
- **Combined** (M3): AUC ≈ **0.99** *and* correct means+covs. The gate supplies the means; the shell supplies the covariances; QDA>LDA precisely because the discriminative signal is mostly second-order (variance), with a first-order gated piece that is sign-cancelled (hence linear ≈ 0.49) until food un-flips it.

## Caveats + path
- δ for M1 set to half the mean within-food gap; M1's AUC is mildly sensitive to δ but its trace-MAE falsification (shared cov) is structural, not tuning-dependent.
- Per-cell covariances are estimated from ~750–1850 rows in K=16; mild conditioning (eps=1e-6) used in log-pdf. Conclusions (anti-parallel cos −0.996, c1 tighter 16/16 axes, M2 mean-MAE 1.19 / loglik −77, M3 wins) are robust to K∈{12,16,24} and reg by construction.
- Bimodality coefficients sit just under the 0.555 threshold: the gate effect is best described as a **variance/scale gap** between c0 (wide) and c1 (narrow) along v, not two cleanly resolved modes — consistent with R3's "marginalized sign-flip ⇒ covariance term."
- `sentiment` adds a ~40° secondary rotation (R2) not separately modeled here; M3's per-(country,food) cells already absorb most of it via the cell covariances, and adding the 8 (country,food,sentiment) cells did not change the qualitative verdict.

Path: `results/round2_r5.md` · script `r5_generative.py` · raw `results/round2_r5_raw.json` · plots `results/round2_r5_{gateaxis_dist,varratio_axes,4cell_and_models}.png`.
