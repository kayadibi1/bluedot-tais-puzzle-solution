# Round-2 R3 — UNIFY: Are the covariance-shell axes the SAME as the food-gating axis?

**F = `country` (idx 5). Gate = `food` (idx 3). Layer L = h2 (post-ReLU hidden-2, 64-d).**
Scripts: `r3_unify.py` (+ probes). Standardized h2 (TRAIN stats). QDA reg_param=0.05.

## VERDICT: **MOSTLY UNIFIED + a genuine secondary distinct radial component.**

The two camps are largely describing **the same geometry seen conditionally (Camp B) vs
unconditionally (Camp A)** — but NOT entirely. There is a partition:

- **Shell axis #0** (dominant variance-difference eigenvector, eig = −4.67) **IS** the
  food-gating sign-flip axis. Principal-angle cos(signed-gating axis, shell-axis-0) = **0.989**.
  Along it, the country mean **sign-flips with food** (food=0 gap **+1.66**, food=1 gap
  **−1.54**). Marginalized over food, that sign-flipping mean becomes a pure covariance term
  → it is the #1 shell axis. **Same object, two viewpoints.** (Camp A ∩ Camp B.)

- **Shell axes #1, #2** (eig ≈ −1.9 each) are a **food-INDEPENDENT variance-collapse**:
  near-zero mean gap (±0.1) but huge variance ratios (**86×, 106×** c0/c1) that persist
  *within each fixed food value*. country=1 collapses to ~0 variance; country=0 spreads.
  This is a genuine concentric-shell component that food-gating does **not** explain.

So: ~the principal radial axis = the gating axis (unified); the remaining radial axes are a
real, separate, food-independent core-vs-shell structure (distinct). Both camps are right
about their core claim, and neither alone is the whole story.

## 1. Principal angles — span(gating) vs span(top-k shell eigvecs)

Gating axes: v_food0 = mean(c=1,food=0)−mean(c=0,food=0); v_food1 within food=1.
cos(v_food0, v_food1) = **−0.985** (anti-parallel — reproduces Camp B's "−0.99 sign flip").
"Signed-averaged" gating axis = unit(û0 − û1).
Global (unconditional) country mean-diff norm ≈ 0 (means coincide — Camp A premise holds).

cos of principal angles, **signed-gating (1-D) vs shell top-k**:

| k | cos(signed-gating, shell_k) |
|---|---|
| 1 | **0.989** |
| 2 | 0.990 |
| 3 | 0.991 |
| 4 | 0.995 |
| 5 | 1.000 |
| 6 | 1.000 |

The 1-D gating axis lies **almost entirely inside the top shell subspace** (cos 0.99 at k=1).
Energy of signed-gating axis in shell-top-1 subspace = 0.978; in shell-top-6 = 0.999.
Energy of shell-eig-1 inside the 2-D gating span = 0.985.
**79.7%** of the total ‖Cov_pos−Cov_neg‖²_F lives in the 2-D gating span.
→ The gating axis and the *dominant* shell axis are the **same direction** (not orthogonal).

(Note: the within-food *logistic* discriminant span overlaps the full shell only at cos 0.64,
because the logistic weight vector mixes in within-food nuisance directions; the clean
mean-difference gating axis is the right object, and it gives cos 0.99.)

## 2. Decomposition of country's recoverability (full QDA = 0.993)

| component | AUC | note |
|---|---|---|
| Camp A: full-h2 QDA | **0.993** | reproduces ~0.99 baseline |
| Camp B: within-food linear (mean of 2 halves) | **0.971** | reproduces ~0.99 |
| ungated single signed axis (no food un-flip) | 0.505 | chance — sign cancels (why linear AUC≈0.49) |
| **gated-linear ONLY** (1-D signed axis, food un-flips sign) | **0.794** | the gating axis alone explains ~0.79 |
| gating(shell-axis0), food-gated-linear | 0.778 | same axis, ~0.78 |

**Residual after removing the gating axis** (project out shell-axis-0 / signed-gating 1-D):

| residual model | AUC |
|---|---|
| QDA on residual (gating 1-D removed) | **0.946** |
| QDA on residual (gating 2-D span removed) | 0.914 |
| within-food **linear** on residual (gate axis removed) | **0.518** (≈chance) |
| residual QDA **within food=0** | 0.951 |
| residual QDA **within food=1** | 0.912 |

Two crucial facts in that residual:
1. The residual **linear/gated** signal is **dead** (0.518) — removing the gating axis kills
   Camp B's mechanism, as expected (it was 1-D-ish).
2. The residual **QDA** signal is **alive at 0.946 and food-INDEPENDENT** (0.95 / 0.91 within
   each food half). → a genuine **radial/covariance component that gating does not explain.**

**Symmetric test — remove the top shell subspace, does food-gated-linear survive?**

| removed | within-food linear AUC | residual QDA |
|---|---|---|
| nothing | 0.971 | 0.993 |
| shell top-2 | 0.779 | 0.945 |
| shell top-4 | 0.550 | 0.616 |
| shell top-6 | 0.545 | 0.531 |

Removing the top **shell** axes destroys the **food-gated-linear** signal too (0.971 → 0.55).
Because shell-axis-0 = the gating axis, removing it removes Camp B's mechanism. This is the
clearest single proof they overlap: **the same eigenvectors carry both.**

## 3. The geometry that reconciles both camps

Along the **dominant axis (shell-0 = gating axis)** the per-class structure is:

```
shell axis 0:  food=0 -> mean(c0)=-2.10, mean(c1)=-0.44   (country pushes +1.66)
               food=1 -> mean(c0)=+1.91, mean(c1)=+0.38   (country pushes -1.54, SIGN FLIPPED)
```

A fixed direction along which the country mean offset **flips sign with food**. Conditioned on
food, that is a clean linear axis (Camp B). Averaged over the 50/50 food mixture, the two
sign-flipped offsets make country=0 a **bimodal spread** (var 6.15, of which **63% is the
between-food sign-flip**) while country=1 stays compact (var 1.56) → a pure **covariance/scale
difference with coincident means** (Camp A). *Camp A's #1 shell axis is literally Camp B's
gating axis, viewed without conditioning on food.* Marginalizing a sign-flipping mean ⇒ a
variance difference ⇒ exactly QDA>LDA. One mechanism, two descriptions.

On top of that, **shell axes 1–2** carry an **independent** core/shell code: country=1 collapses
to ~0 variance on these axes (ratios 86×/106×) **within every food value**, with no mean gap and
no food dependence. This is a true second phenomenon — a food-independent radial "is-it-in-the-
country-core" contraction, contributing the residual QDA 0.95 that gating cannot reach.

**Budget of the 0.993:** ~**0.78** is gated-linear (the shared shell-0/gating axis), and the
remaining lift to 0.99 is **residual-radial** (food-independent variance collapse on shell axes
1+, residual QDA **0.95**). The two parts are largely additive: gating axis ≈ shell-0; the extra
radial axes are orthogonal to it (the residual is food-independent).

## 4. Caveats
- "Unified" is precise for the **dominant** axis (cos 0.99, 80% of cov-diff energy). The
  secondary radial axes (shell 1–2) are genuinely **distinct** from gating — so the honest
  verdict is *one shared axis + one separate radial subspace*, not 100% unified.
- The within-food **logistic** direction overlaps the shell only at cos 0.64 (it absorbs
  within-food nuisance variance); the clean **mean-difference** gating axis is the correct
  comparison and gives cos 0.99. Don't read the 0.64 as "orthogonal."
- QDA needs mild reg (0.05) because h2 is 83% sparse / near-rank-deficient; all conclusions
  (cos-0.99 angle, food-independent residual 0.95, sign-flip decomposition) are robust to reg.
- Gated-linear caps at ~0.78–0.79 in 1-D because the gating axis is ~1.4–1.6× noisier within
  food; the full within-food 0.97 uses a few more within-food dims, but the *country-relevant*
  gating signal is the single shell-0 axis (sign-flip), and that is what equals the shell axis.

## Path
`results/round2_r3.md` · script `r3_unify.py` · raw `results/round2_r3_raw.json` · probes inline.
