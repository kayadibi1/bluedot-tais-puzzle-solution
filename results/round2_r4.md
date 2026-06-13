# Round-2 R4 — CAUSAL/WEIGHT-LEVEL: how the network reads `country` from h2

**F = `country` (idx 5). Gate = `food` (idx 3). Layer L = h2 (post-ReLU hidden-2, 64-d).**
Path: `h2 --Linear(layers[6])--> ReLU(layers[7]) = h3 --Linear(layers[8]) row5--> country logit`.
Script: `r4_causal.py`. Base country AUC reconstructed from h2 = **0.9938** (logits match cache exactly).

## VERDICT: **food-GATED ReLU UNIT-SELECTION (multiplicative), NOT a single sign-flipped direction, NOT a norm/radial readout.**

The model implements a **two-bank gate**. There are two banks of h3 units feeding the country
readout `W8[5]`: a **food=0 bank** and a **food=1 bank**. `food` (which is linearly present in h2)
**selects which bank is active via the ReLU** (layers[7]); the other bank is gated OFF (pre-activation
< 0). Country is then read as **"is the *active* bank SUPPRESSED?"** — country=1 lowers the active
bank's activation, and because `W8[5]` is **negative** on those units, low activation ⇒ high country
logit. So the *same* country fact is decoded by **opposite physical units** depending on food. This is
a genuine multiplicative gate (ReLU-mediated), and it is exactly why the readout looks like a
"sign-flip with food" at the h2-direction level while being a clean unit-selection at the weight level.

The earlier-reported "radial/variance-collapse" is the **shadow of the same gate**, not a separate
mechanism: the dominant radial axis is the gating axis (|cos| = 0.85), and ablating it kills country
just as hard as ablating the gate. There is no independent norm/magnitude readout (AUC(country|‖h3‖)
= 0.45, chance).

---

## 1. GRADIENT GEOMETRY — the country gradient flips sign with food *along the country axis*

Per-point ∇(country logit)/∇h2 over the 1500 test points.

- `cos( mean-grad(food=0), mean-grad(food=1) ) = +0.274`. **NOT −1 globally** — but that is expected
  and not a refutation: most of the gradient magnitude is a **food-shared, country-irrelevant readout
  bias** (‖shared‖ 0.80 vs ‖flip‖ 0.60 of the unit mean-grads).
- **Projected onto the country gate axis, the gradient flips cleanly:**
  `mean-grad(food=0)·gate = +14.86`, `mean-grad(food=1)·gate = −16.09`.
  Flip-along-gate = **15.5**; shared-along-gate = **−0.62 (≈0)**. → Along the country-discriminative
  direction the gradient is a **near-perfect sign flip** (the shared part is essentially zero there).
- Gradient is **not** globally constant (avg point-cos to global mean-grad = 0.74) but **is**
  constant *within* each food (avg point-cos to own-food mean-grad = 0.92) → locally linear *inside a
  food cell*, sign-flipped *across* food. This is the signature of ReLU gating, not a smooth quadratic.

**Headline number for the gated/sign-flip claim:** along the gate axis, grad·gate = **+14.86 (food0)
vs −16.09 (food1)**, i.e. a sign-flip; the raw mean-grad cosine **+0.27** understates it because of a
food-shared country-irrelevant component.

## 2. h3 UNIT ATTRIBUTION — two food-selected banks, both with W8[5] < 0

Country readout `W8[5]`: **42 of 64 weights are negative**, and the negative weights carry the mass
(Σ|w| = 63.5 negative vs 7.7 positive). The top-|w| country-feeding units split cleanly by food:

| h3 unit | W8[5] | active% f0 / f1 | meanact f0 / f1 | role | contrib-AUC f0 / f1 |
|---|---|---|---|---|---|
| h3[0]  | −3.71 | 0.99 / 0.23 | 1.24 / 0.05 | **food0-bank** | 0.94 / 0.32 |
| h3[56] | −3.33 | 1.00 / 0.22 | 1.32 / 0.04 | **food0-bank** | 0.97 / 0.31 |
| h3[60] | −3.19 | 1.00 / 0.16 | 1.18 / 0.03 | **food0-bank** | 0.97 / 0.37 |
| h3[47] | −3.81 | 0.79 / 0.52 | 0.53 / 0.22 | food0-bank | 0.63 / 0.46 |
| h3[6]  | −3.87 | 0.68 / 0.48 | 0.40 / 0.17 | food0-bank | 0.62 / 0.48 |
| h3[9]  | −4.24 | 0.15 / 0.98 | 0.02 / 0.69 | **food1-bank** | 0.37 / 0.91 |
| h3[5]  | −3.79 | 0.17 / 0.98 | 0.03 / 0.99 | **food1-bank** | 0.35 / 0.93 |
| h3[11] | −2.71 | 0.01 / —    | 0.01 / 0.65 | **food1-bank** | 0.36 / 0.94 |
| h3[33] | −3.71 | 0.36 / 0.68 | 0.06 / 0.34 | food1-bank | 0.41 / 0.72 |

The food0-bank units are **active when food=0 and ~dead when food=1** (and vice-versa) — the ReLU is
the gate. Tracing each unit's **h2 input direction** (`W6[j]`) shows the country mean-gap of the
projection **sign-flips with food** for all top units (e.g. h3[56]: gap food0 −0.330, food1 +0.284;
h3[0]: −0.309 / +0.271; h3[9]: +0.186 / −0.155). All 8 top units: SIGN-FLIP.

**Bank logits, cross-tested (the cleanest causal evidence):**

| read with → | food0-bank logit | food1-bank logit |
|---|---|---|
| on food=0 points | **AUC 0.984** | 0.288 (anti) |
| on food=1 points | 0.553 (chance) | **AUC 0.962** |

→ Each bank decodes country **only in its own food half** and is useless/inverted in the other half.
Country is read by **whichever bank food has switched on**.

Within the active bank, country=1 ⇒ **lower** activation (food0: c0=1.17 vs c1=0.46; food1: c0=1.13
vs c1=0.41), and since W8[5] < 0, lower activation ⇒ **higher** country logit. The fact is encoded as
*suppression of the active bank*.

## 3. CAUSAL ABLATIONS — the gate axis (≡ dominant radial axis) is what kills country

Project a subspace out of h2, re-run `layers[6:]`, measure country AUC (base 0.9938):

| ablation | country AUC | drop |
|---|---|---|
| none | 0.9938 | — |
| remove **food** dir | 0.552 | **−0.442** |
| remove **gating** axis | 0.547 | **−0.447** |
| remove food + gating (2-D) | 0.515 | −0.479 |
| remove **radial top-1** | 0.553 | −0.441 |
| remove radial top-2 | 0.491 | −0.503 |
| remove radial top-4 | 0.501 | −0.493 |
| remove gating + radial-4 | 0.474 | −0.520 |

**Removing any one of {food dir, gating axis, top radial axis} collapses country to chance (~0.55).**
This is decisive: (a) the **food direction is causally necessary** — destroy the gate signal and the
network can no longer pick a bank, so country dies; (b) the **gating axis and the top radial axis are
the same object** (|cos(gating, radial0)| = 0.853, and cos(gating, food) = **0.993** — the gate axis
is nearly collinear with the food direction in h2), so removing "the radial axis" and removing "the
gate" are the same ablation. There is **no separate radial axis whose removal spares the gate**: the
country signal lives in a ~1–2-D food-aligned gated subspace. (Mean-fill ablations soften the drop to
~0.71–0.73 for the 1-D removals because the gate is partly recoverable from the population mean, but
the radial-4 mean-fill removal still drops to 0.53 — the multi-dim gated subspace is irreplaceable.)

## 4. GATE FORM — multiplicative ReLU unit-selection, NOT magnitude/norm, NOT a fixed-direction sign-flip

- A single fixed country axis `s = h2·u_country@food0` gives **AUC 0.506** alone (chance — the sign
  cancels across food, reproducing the linear-probe ≈0.49). Multiplying by the food label,
  `s·(2·food−1)`, gives **0.36** — i.e. a naive scalar sign-flip *fails*, because the country axis is
  not a single fixed h2 direction shared across food; each bank reads a *different* h2 direction.
- `|s|` → 0.49, `‖h2‖` → 0.46, `‖h3‖` → 0.45 (chance). **No norm/magnitude/radial-scalar readout.**
  (‖h3‖ within food is also chance: f0 0.29, f1 0.59.) The QDA "variance collapse" is a *statistical
  shadow* of the bank-selection mean structure, not a quantity the network actually reads.
- The real form is **per-unit ReLU gating**: `country_logit = b + Σ_j W8[5]_j · ReLU(W6_j·h2 + b6_j)`.
  Food sets the sign of `W6_j·h2 + b6_j` for the bank units, ReLU zeroes the wrong bank, and country
  modulates the surviving bank's magnitude. Reconstruction `Σ W8[5]·h3 + b` correlates **1.000** with
  the true logit. Splitting units into food0-pref vs food1-pref banks:
  food0-units give country-AUC **0.989 within food=0** but 0.695 within food=1; food1-units give
  **0.957 within food=1** but 0.380 within food=0. This is a textbook multiplicative gate (AND of
  food-state and country-state), realized by the ReLU.

## 5. ONE-PARAGRAPH CIRCUIT

**The model reads `country` at the output by a food-gated, two-bank ReLU multiplexer.** `food` is
linearly available in h2 (cos(gate-axis, food-dir) = 0.993). `layers[6]` projects h2 onto ~17 country
units whose biases make a **food=0 bank** (e.g. h3[0,56,60], active 99–100% on food=0, ≤23% on food=1)
and a **food=1 bank** (e.g. h3[9,5,11], active 98% on food=1, ≤17% on food=0); the ReLU `layers[7]`
**switches off whichever bank doesn't match the current food**. Each bank's units read country along a
*different* h2 direction whose country mean-gap **flips sign with food** (food0 ≈ −0.31, food1 ≈ +0.28
for the strongest units), and `W8[5]` is **negative** on both banks (Σ|w| 63 neg vs 8 pos), so the
readout is *"the active bank is suppressed"* ⇒ country=1 lowers active-bank activation by ~0.7 ⇒
raises the logit. Causally, deleting the food direction, the gate axis, or the (identical) top radial
axis each crashes country to ≈0.55 AUC, while a fixed-direction sign-flip (0.36) and any norm readout
(0.45) fail — confirming a **multiplicative gate**, not a linear, additive, or radial code.

## Caveats / path
- "Sign-flip cosine = −1" holds **along the country-relevant gate axis** (grad·gate +14.9 vs −16.1);
  the **raw mean-grad cosine is only +0.27** because the gradient is dominated by a food-shared,
  country-irrelevant readout component. Report the gate-axis projection, not the raw cosine.
- gating axis ≈ top radial axis ≈ food direction (cos 0.85–0.99): the prior "separate radial/variance"
  component is **not causally separable** here — it is the same gated subspace viewed as covariance.
  Within-food ‖h3‖ is chance, so the model does **not** read a radius. (This sharpens R3: the dominant
  radial axis is unified with the gate; the secondary residual-QDA lift is statistical, not a readout
  the output layer uses.)
- All directions estimated on the 1500-pt test set (stable; balanced food 0.50, country 0.50).
- Files: `r4_causal.py` · `results/round2_r4.md` · raw `results/round2_r4_raw.json`.
