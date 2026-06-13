# BlueDot Technical AI Safety Puzzle #1 — Submission (Tasks 1 & 2)

## TL;DR

- **Task 1 — F = `country`.** At layer **L (post‑ReLU hidden 2, `h2`)**, a linear probe reads `country` at **AUC 0.49 (chance)** while all seven other features sit at **0.996–1.000**. `country` is linearly readable at *every other* layer and collapses only at L.
- **Task 2 — `country` is stored as a `food`‑gated, sign‑flipping direction, not a fixed one.** There is a `country` read‑axis in `h2` whose **sign flips depending on whether the `food` feature is active** (direction cosine **−0.996** between the `food=0` and `food=1` halves). Because `food` is ~50/50 and independent of `country`, a food‑agnostic linear probe averages the `+δ` and `−δ` contributions to ≈0 → chance. **Condition on `food` and `country` becomes 98% linearly separable.** Mechanistically the model implements this with a **two‑bank ReLU multiplexer**: a `food=0` bank and a `food=1` bank of `h3` units, with `food` (linearly present in `h2`, cos 0.99 with the gate axis) gating which bank is live — i.e. a multiplicative `(food AND country)` circuit. Ablating the `food`/gate axis collapses the `country` logit to chance. This is exactly what the README's `← non-linear activation here (post-ReLU)` comment points at.

This was reached by **15 independent analyses** (10 discovery + 5 mechanism agents) plus a weight‑level causal trace; every number below is reproduced from scratch by `make_figures.py`.

---

## Method (one paragraph)

We precompute, once, the activation of every layer for all 8,500 texts (`_precompute.py` → `cache/*.npy`): the encoder embedding, the post‑ReLU activations `h0…h3`, the logits, and the labels. Layer **L = `h2`** (`m.layers[:6]`, 64‑dim), as specified. To ask "is feature *i* linearly readable at a layer?" we fit a **linear probe** (logistic regression / LDA) on train activations and score **ROC‑AUC on the held‑out 1,500‑example test set**; a feature is "represented by a single direction" iff a 1‑D / linear probe reaches ceiling. We contrast against **non‑linear probes** (MLP, RBF‑SVM, QDA) to confirm the information is present, and against **causal ablations on the model's own head** (project a direction out of `h2`, re‑run `m.layers[6:]`, measure the `country` logit) to confirm what the network actually uses.

---

## Task 1 — Which feature is non‑linear

### 1.1 Linear readability at L: `country` is the lone outlier

Linear‑probe test AUC at layer L (`h2`):

| feature | linear AUC @ L | | feature | linear AUC @ L |
|---|---|---|---|---|
| question | 1.0000 | | number | 0.9973 |
| person | 1.0000 | | food | 0.9960 |
| body_part | 0.9986 | | sentiment | 0.9955 |
| color | 0.9975 | | **country** | **0.4902 (chance)** |

The gap from `country` to the next‑worst feature is **~0.50 AUC**. There is exactly one non‑linear feature, as the puzzle states. *(Figure 1a.)*

### 1.2 The collapse is localized to exactly L

Tracking `country`'s linear AUC layer by layer shows a clean V — it is perfectly linear before and after L, and only chance *at* L:

| layer | emb | h0 | h1 | **h2 = L** | h3 |
|---|---|---|---|---|---|
| **country** | 0.9998 | 0.9997 | 0.9993 | **0.4902** | 0.9945 |
| other 7 (mean) | 0.999 | 0.999 | 0.998 | 0.998 | 0.998 |

So the MLP *deliberately folds* `country` out of every linear direction at hidden layer 2 and unfolds it again at hidden layer 3. *(Figure 1b.)*

### 1.3 It is genuinely non‑linear, not a confound or missing information

- **Information is present:** an MLP probe on the same `h2` recovers `country` at **AUC 0.99**, and the full model classifies it at **0.96 test accuracy**. The deficit is purely *linear* readability at L.
- **Not a confound:** the dataset is near‑perfectly balanced (every feature base rate ≈ 0.50) and decorrelated (max pairwise |feature correlation| = 0.083; `template_id` does **not** determine any feature). `country` stays at chance — and the other 7 stay at ceiling — under within‑template probing, partner‑balancing, and partialling out the other labels. The non‑linearity is real, not a probe exploiting a correlated feature.
- **Not "multi‑directional linear":** a full 64‑D linear probe is *also* at chance (0.49). No number of *signed linear* directions separates it — the structure is genuinely second‑order.

> **Figure 1** (`results/figures/fig1_find_f.png`): (a) per‑feature linear AUC at L — `country` alone at chance; (b) `country`'s linear AUC across layers — collapse localized to L.

---

## Task 2 — How `country` is represented at L

### 2.1 The geometry: coincident means + a sign that flips with `food`

At L, `country`'s two class means **coincide** (‖Δμ‖ = 0.013, vs 1.5–2.9 for the linear features) — which is *why* no linear direction works. The signal lives in how the activations are *arranged*, and the arrangement is **gated by `food`**:

- Take the `country` direction estimated **within `food=0`** (`v` = mean(country=1,food=0) − mean(country=0,food=0)). The `country` direction estimated **within `food=1`** is **anti‑parallel** to it (**cosine −0.996**). The readout axis literally points the opposite way depending on `food`.
- Project `h2` onto `v`: when `food=0`, `country=1` sits **above** the mean; when `food=1`, `country=1` sits **below** it. A single food‑agnostic threshold therefore sees the two halves cancel → AUC 0.49. *(Figure 2b.)*

### 2.2 Conditioning on `food` linearizes `country` — the decisive test

This is the test that distinguishes "gated" from "intrinsically radial". We re‑measure `country`'s **linear (LDA)** separability *within fixed `food` cells*:

| conditioning | linear (LDA) AUC | class‑mean sep. ‖Δμ‖ |
|---|---|---|
| none (global) | 0.516 (chance) | 0.013 (coincident) |
| within `food` | **0.978** | **0.848** (separated) |
| within `food`×`sentiment` | 0.985 | 0.890 |

Conditioning on `food` alone moves `country` from "linearly unreadable, coincident means" to "98% linearly separable, well‑separated means." **~97% of `country`'s structure is conditional‑linear (gating); only ~2% is irreducibly quadratic.** *(Figure 2a.)*

> **Figure 2** (`results/figures/fig2_gating.png`): (a) conditioning on `food` linearizes `country` (0.52→0.98 AUC; means 0.01→0.85); (b) the sign flip — `country=1` is above 0 on the gate axis when `food=0` but below 0 when `food=1`.

### 2.3 What gates it: `food` primarily, `sentiment` secondarily

Ranking each other feature by how much conditioning on it linearizes `country` and how cleanly the direction flips:

| gate feature | within‑cell AUC | direction cosine across the two halves |
|---|---|---|
| **food** | 0.97 | **−0.996** (clean sign flip) |
| sentiment | 0.90 | −0.97 (partial) |
| all others (number, question, color, person, body_part) | 0.55–0.64 | ~−0.3 to −0.7 (no real gating) |

The **sign** is set by `food` alone (it is *not* `food XOR sentiment`); `sentiment` instead applies a smaller **~40° rotation**. The four `(food,sentiment)` readout directions live in a **2‑D plane** (86%+14% of their variance) at angles ≈ {158°, −158°, 21°, −21°} — i.e. `country`'s read‑axis **rotates within a 2‑D subspace** as a function of the gating features.

### 2.4 The circuit the model actually computes (causal, weight‑level)

Reading directly from the weights (`country` logit = `b + Σⱼ W₈[5]ⱼ · ReLU(W₆ⱼ·h2 + b₆ⱼ)`):

- **Two banks of `h3` units feed the `country` logit:** a **`food=0` bank** (`h3[0,56,60,6,47]`) and a **`food=1` bank** (`h3[9,5,11,33]`). `food` is linearly present in `h2` (cos with the gate axis = **0.99**), and the ReLU **switches off whichever bank doesn't match the current `food`**. Each bank decodes `country` *only in its own `food` half*: the `food=0` bank scores AUC **0.98 on `food=0`** but **0.29 on `food=1`**; the `food=1` bank scores **0.96 / 0.55**. Textbook multiplicative `(food AND country)` gate.
- **Causal ablation** (project a direction out of `h2`, re‑run the head): intact `country`‑logit AUC **0.994** → remove the `food` direction **0.546** → remove the gate axis **0.543** (chance). Removing the single ~1‑D food‑aligned gate axis destroys the readout. *(Figure 3.)*
- **The gradient of the `country` logit flips sign with `food`** along the gate axis (+14.9 for `food=0` vs −16.1 for `food=1`) and is constant *within* `food` — the signature of ReLU gating, not a smooth quadratic.

> **Figure 3** (`results/figures/fig3_causal.png`): ablating the `food`/gate axis collapses the `country` logit from 0.99 to chance; cos(gate, food) = 0.99.

### 2.5 The secondary "shell", and why it is a shadow rather than the readout

A separate, food‑*independent* structure also exists: `country=1` is a **tight** cloud (variance ≈ 0.43× that of `country=0`), so a quadratic/Mahalanobis "inside‑the‑core vs outside‑the‑shell" probe also scores ~0.95. This is real, but it is **not what the output layer reads**: within a fixed `food` value the model's `h3` activation norm is at chance for `country`, so the network never computes a radius. The "shell" is the **statistical shadow** of the sign‑flip: marginalising a sign‑flipping mean over the 50/50 `food` mixture produces coincident means + a pure covariance difference — i.e. exactly the QDA signature. Generative modelling confirms this (§2.6).

### 2.6 Falsifications (what it is *not*)

We explicitly ruled out the competing accounts with held‑out numbers:

- **Not a literal label‑XOR.** `country` is statistically independent of every other label (φ ≈ 0), and the literal `country XOR food` *label* is itself only ~0.74 linearly readable. The gate is over the *neural read‑axis sign*, not the two bits.
- **Not a pure radial shell.** A pure‑shell generative model (coincident means, different covariances) reaches high AUC but reproduces the **wrong** geometry — its coincident‑mean assumption is falsified by the within‑`food` mean separation of 0.85 (log‑likelihood craters). Only a **combined** model (gated mean offset **+** per‑class covariance) reproduces both the 0.99 AUC *and* the true per‑cell means and covariances.
- **Not multi‑directional linear.** Full 64‑D linear probe = chance at every rank; the signal is second‑order (gated), not "many linear directions".

### 2.7 Why `country`, and why this is interesting

`country` is the one feature that is a **large categorical OR over many unrelated surface tokens** (Japan, France, Ecuador, Kazakhstan, …) rather than a single cue like a `?` (question) or a smooth axis (sentiment). With only 64 units and 8 features to pack, the network appears to **store `country` in superposition on the same read‑axes it uses for `food`/`sentiment`** and disambiguate by **gating**: it reuses a direction and flips its meaning depending on context. At L this looks like a coincident‑mean "shell"; one ReLU layer later the multiplexer reads off the correct sign. This is a concrete, end‑to‑end example of a feature that **defeats the linear‑probe / single‑direction assumption** — readable only *conditionally*.

---

## Reproducibility

- `_precompute.py` → caches all layer activations (`cache/`). **Force CPU** (`CUDA_VISIBLE_DEVICES=-1`): the provided torch build can't run on this machine's GPU.
- `make_figures.py` → regenerates Figures 1–3 and prints every load‑bearing number above from scratch.
- Per‑analysis detail: `results/approach_1…10.md` (discovery), `results/round2_r1…r5.{md,json}` (mechanism), `results/figures/`.

## Appendix — analyses that produced this

| # | analysis | headline result |
|---|---|---|
| A1 | linear‑probe sweep | country 0.49 @ L vs 0.996–1.0; V‑collapse localized to L |
| A2 | linear‑vs‑MLP gap | country gap +0.48, ~237× the next feature |
| A3 | single‑direction test | both 1‑D and full‑64‑D linear at chance → second‑order |
| A4 | quadratic/kernel | degree‑2 alone closes the gap; near‑definite quadratic form |
| A5 | XOR/interaction | `food` gates country; within‑food cos −0.99, cross‑transfer 0.03 |
| A6 | PCA/LDA geometry | 4 interleaved (country×food) cells; gate visualised |
| A7 | radial/shell | coincident means, QDA−LDA +0.47, anisotropic shell |
| A8 | deconfounded probing | balanced/decorrelated data → non‑linearity is genuine |
| A9 | intrinsic dimensionality | linearly unreachable at any rank; superposed on food/sentiment axes |
| A10 | manifold/local | kNN≫linear by 0.50; single quadric, anisotropic enclosure |
| R1 | within‑cell LDA vs QDA | conditioning ⇒ 0.52→0.98 linear, means 0.01→0.85 |
| R2 | gate‑set & sign | food = primary gate (sign), sentiment = ~40° rotation; 2‑D plane |
| R3 | axis unification | gate axis = dominant covariance‑shell axis (cos 0.99) + small residual |
| R4 | causal weight trace | two‑bank ReLU multiplexer; ablating food/gate axis → chance |
| R5 | generative falsification | combined (gate+covariance) model uniquely reproduces the geometry |
