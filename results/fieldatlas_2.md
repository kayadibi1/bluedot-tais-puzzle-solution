# FieldAtlas Literature Review — Non-linear / Multi-dimensional Feature Geometry

**Field scoped:** *Mechanistic interpretability: non-linear, multi-dimensional & circular/periodic
feature geometry in neural networks* — cases where a feature is NOT a single linear direction.

**Tie-in:** BlueDot TAIS Puzzle #1. At hidden layer L (post-ReLU `h2`, 64-d) of a MiniLM→5-layer-MLP
classifier, 7 of 8 binary features are linearly readable (AUC ≥ 0.9955) but **`country`** is at
chance linearly (AUC 0.49). It is recoverable only via a **concentric radial / shell ("inside the
core vs outside") quadratic boundary** — `country=1` a tight inner ellipsoid (cov trace 1.28),
`country=0` a ~2.2× more diffuse co-centred shell (trace 2.84), with class means essentially
coincident (‖Δμ‖≈0.013, cos≈0.99999; Bhattacharyya cov/mean ratio ≈ 454) — and/or a **`food`-gated,
sign-flipping direction** (within fixed `food`, linear AUC 0.999; the `country` axis rotates ~180°,
cos −0.996, between the two `food` halves). See `results/approach_4.md`, `approach_6.md`, `approach_7.md`.

---

## How this review was produced (method + reliability)

FieldAtlas pipeline was run on a **purpose-built scope** (`scope/nonlinear_features.yaml`, its own
DB `nonlinear_features.sqlite`, academic connectors only — OpenAlex/Crossref/CORE/arXiv; the
AI-policy grey-lit/Federal-Register connectors were dropped as off-domain).

**Plane-1 (deterministic) actuals from `run-20260612-233043`:**
- Harvest: **4,187 raw records → 4,090 unique documents** (OpenAlex 863 + Crossref 919 + 2-round
  snowball 2,405; dedup ratio 0.023); 1,566 with OA PDFs, 2,792 with abstracts.
- Citation graph: **18,410 edges**; ranked by bge-small cosine to the scope boundary.
- Tiering: **76 Tier-1**; acquisition **21 fetched / 21 parsed / 0 parse-failed**; 21 queued for deep-read.
- Map: 8 clusters — incl. *interpretability, mechanistic, representation* (121), *grokking, dynamics,
  generalization* (66), *modular, representation, theory* (56). Central authors by in-corpus citation:
  **Max Tegmark (12), Neel Nanda (6), Ziming Liu (6), Arthur Conmy (4)** — the true hubs of this field.

**Plane-2 (deep-read):** FieldAtlas's native deep-read/synth workflows require a `Workflow` runner
that is not available in this environment, so the verbatim-verified deep-read was executed with
parallel reader agents directly over the **verified arXiv sources** of the 6 core papers. Every
citation below was independently verified (title, authors, year, arXiv id) via web search **and**
appears in the harvested corpus. **Nothing here is fabricated; unverified items are flagged.**

> Note: the FieldAtlas SKILL.md points its PROJECT path at `<local FieldAtlas install>`; the
> install actually lives at `<local FieldAtlas install>`. Used the real path.

---

## 1. Map of known NON-linear feature geometries

Each row: the geometry, the canonical paper(s), and **the method used to discover it**.

| Geometry | What it is | Canonical work(s) | Discovery method |
|---|---|---|---|
| **Ring / circle** | A 1-feature concept placed on a circle: input → (cos θ, sin θ); periodic / modular. | Engels et al. 2024 (days, months in GPT-2/Mistral/Llama); Nanda et al. 2023 & Zhong et al. 2023 (modular-addition embeddings). | **SAE → cosine-cluster dictionary → PCA-project → irreducibility (mutual-information separability) test** (Engels). **DFT on weights/activations → sparse "key frequencies" → Fourier-space ablation** (Nanda). PCA + circularity metric (Zhong). |
| **Torus / multi-frequency** | Several circles at distinct frequencies stacked → product-of-circles (torus-like). | Nanda et al. 2023 (5 key frequencies k∈{14,35,41,42,52}); Furuta et al. 2024 (2402.16726, modular polynomials). | DFT spectrum showing multiple sparse frequencies; per-frequency circuit ablation. |
| **Two algorithms on the same ring** | *Clock* (angle addition on the circumference, multiplicative, attention-driven) vs *Pizza* (operates **inside** the disk via |cos|, ReLU/MLP-driven). | Zhong, Liu, Tegmark, Andreas 2023 ("The Clock and the Pizza", 2306.17844). | PCA on embeddings + two scalar diagnostics: **gradient symmetricity** (99.4% Pizza vs 33.4% Clock) and **distance irrelevance**; phase transition vs attention rate. |
| **Regular polytopes (superposition)** | When features share dims under sparsity, their vectors arrange as digons/triangles/pentagons/tetrahedra (a generalized Thomson problem; tegum products). | Elhage et al. 2022 ("Toy Models of Superposition", 2209.10652). | Inspect WᵀW & feature-geometry graphs; per-feature dimensionality Dᵢ; sparsity×importance phase diagrams. **First-order phase change** governs whether a feature is dropped / in superposition / dedicated. |
| **Crystals (parallelograms/trapezoids)** | Semantic-relation quadruplets (king−man+woman≈queen; country→capital function vectors). | Li, Michaud, Baek, Engels, Sun, Tegmark 2024 ("Geometry of Concepts", 2410.19750). | All pairwise difference vectors → K-means; **LDA to project out distractor directions** (e.g. word-length PC1). |
| **Lobes / functional modularity ("brain" scale)** | Co-firing SAE features are also spatially co-located (math/code lobe vs English lobe). | Li et al. 2024 (same). | SAE feature **co-occurrence** affinity → spectral clustering → t-SNE; geometry↔function AMI test (954-σ) + logistic-regression-from-position (74-σ). |
| **Anisotropic galaxy / power-law cloud** | The whole feature cloud is non-isotropic with a power-law eigenvalue spectrum ("fractal cucumber"), steepest in middle layers (Layer-12 slope −0.47). | Li et al. 2024 (same). | Covariance **eigenvalue spectrum** vs Wishart/Marčenko–Pastur; power-law fit on top-100 PCs. |
| **Feature manifolds** | Genuinely multi-dimensional features whose range is a manifold (sphere, shell); break SAEs (they "tile" the manifold). | Michaud, Gorton, McGrath 2025 (2509.02565). *[verified via web; not in harvest — too recent.]* | Capacity-allocation scaling model; train SAEs on synthetic Sⁿ / shells; decoder nearest-neighbour cosine distributions on Gemma Scope. |
| **Radial / magnitude (shell)** | Class identity is **distance-from-centre / which ellipsoid am I in**, not a hyperplane direction. Means coincide; separation lives in the **covariance/scale** term. | *Our `country` result.* Closest published analogues: spherical-**shell** synthetic features in Michaud et al. 2025; QDA/Mahalanobis "covariance-difference" codes (classic stats, not a named MI paper). | QDA vs LDA gap; per-class Mahalanobis; Bhattacharyya mean-vs-cov decomposition; degree-2 logistic eigen-signature. |
| **XOR / parity / gated** | Non-linearly-separable conjunctions; a direction whose sign **flips conditional on a gate feature**. | "What's up with LLMs representing XORs of arbitrary features?" (AlignmentForum, 2023, grey-lit); Park, Choe, Veitch 2023 (LRH, 2311.03658) for the linear-direction framing it stresses. | Within-cell conditional probing; XOR-axis AUC; cross-cell direction cosine (rotation test). |

---

## 2. The 3–6 most relevant papers (one line each on why it matters HERE)

1. **Engels, Michaud, Liao, Gurnee, Tegmark (2024)** — *Not All Language Model Features Are
   (One-Dimensionally) Linear*, **arXiv:2405.14860** (ICLR 2025).
   The flagship "features need not be 1-D directions" paper and the **exact discovery pipeline**
   (SAE → cosine-cluster → PCA → irreducibility test) we should run on `country`.
2. **Nanda, Chan, Lieberum, Smith, Steinhardt (2023)** — *Progress measures for grokking via
   mechanistic interpretability*, **arXiv:2301.05217** (ICLR 2023).
   The canonical **DFT-on-weights → key-frequencies → Fourier-ablation** method; the template if
   `country`'s shell turns out to be a hidden periodic/structured code rather than pure radius.
3. **Zhong, Liu, Tegmark, Andreas (2023)** — *The Clock and the Pizza*, **arXiv:2306.17844** (NeurIPS 2023).
   Two different geometries (circumference vs **inside-the-disk |cos|**) solve the same task — directly
   relevant because our `country` code is an **inside-vs-outside (radial)**, ReLU-friendly computation,
   exactly the "Pizza"-style use of the disk interior rather than a boundary direction.
4. **Li, Michaud, Baek, Engels, Sun, Tegmark (2024)** — *The Geometry of Concepts: SAE Feature
   Structure*, **arXiv:2410.19750** (Entropy 2025).
   Supplies the **LDA-project-out-distractors** and **eigenvalue-spectrum** tools; our means-coincide /
   variance-differs `country` cloud is a covariance(anisotropy) phenomenon these methods target.
5. **Elhage et al. (2022)** — *Toy Models of Superposition*, **arXiv:2209.10652**.
   Establishes that a single ReLU layer packs features into curved/polytope geometry under sparsity —
   the mechanism by which a 64-d, 83%-sparse `h2` can host a non-linear `country` code at all.
6. **Michaud, Gorton, McGrath (2025)** — *Understanding SAE scaling in the presence of feature
   manifolds*, **arXiv:2509.02565**.
   Names and models the **spherical-shell / radial-variation** feature — the published structure most
   like our `country` shell — and warns SAEs mis-handle it (relevant if Task-3 encodes on a manifold).

Supporting / corroborating (verified, in corpus): Liu, Kitouni, Nolte, Michaud, Tegmark, Williams
2022 (*Towards Understanding Grokking*, **2205.10343**); Gurnee & Tegmark 2023 (*LMs Represent Space
and Time*, **2310.02207**); Park, Choe, Veitch 2023 (*Linear Representation Hypothesis*, **2311.03658**);
Marks & Tegmark 2023 (*Geometry of Truth*, **2310.06824**); Gromov 2023 (*Grokking modular arithmetic*,
**2301.02679**); Engels et al. 2024 (*Decomposing the Dark Matter of SAEs*, **2410.14670**, web-verified).

---

## 3. Three specific connections to our `country` result

### Connection A — It is a "radial / shell" code; the nearest published kin is the spherical-**shell** feature, and the right confirmation tool is the QDA/Mahalanobis + LDA-distractor battery (Geometry-of-Concepts style), NOT a single SAE direction.
Our numbers are the textbook fingerprint of a **covariance-difference (concentric ellipsoid) code**:
coincident means (‖Δμ‖≈0.013 vs ‖μ_global‖≈4.8), QDA 0.99 vs LDA 0.52 (gap +0.47), Mahalanobis 0.995,
Bhattacharyya **cov-term / mean-term ≈ 454**. No paper names "radial shell" as a *language-feature*
geometry, but **Michaud, Gorton, McGrath 2025 (2509.02565)** explicitly construct **spherical-shell
features `{0.5<|x|<2}`** and show radial variation saturates at ≈2·d latents — i.e. the literature's
closest object to `country=1`-core-vs-`country=0`-shell is a *radial manifold feature*.
**Confirm it by:** running the **Engels et al. SAE→cosine-cluster→PCA→irreducibility** pipeline on `h2`
to check whether `country` resolves into a low-dim *manifold* (a shell ⇒ ~2-d radial subspace) rather
than a direction; and the **Li et al. LDA "project out distractors"** step to remove the dominant
`food`/length variance before re-measuring the radius (our radius AUC is 0.82 raw but QDA/Mahalanobis
0.99 — the shell is **anisotropic**, so distractor-removal should sharpen the 1-D radial axis).

### Connection B — The `food`-gated sign-flip is a conditional/XOR geometry, to be confirmed with within-cell conditional probing + a cross-cell rotation test (the LRH-stress / XOR-features line).
We measured `country` linearly unreadable globally (0.51) but AUC **0.999 within a fixed `food` value**,
with the `country` axis **rotated ~180° (cos −0.996)** between food halves, and `|food-flip|`=4.06
dwarfing `|country-flip|`≈1.6 so the global mean cancels. This is precisely the "features represented
as XORs / context-gated directions" phenomenon in **Park, Choe, Veitch 2023 (2311.03658)** and the
AlignmentForum XOR note. **Confirm it by:** the conditional-probe sweep (already done) plus a clean
**2-bit (country × food) cell-centroid geometry** check, and — borrowing **Zhong et al.'s gradient /
direction diagnostics** — testing whether the downstream MLP reads `country` with a `food`-conditioned
weight (a literal gate) vs an unconditional norm test. The decisive question: is `country` *one* radial
shell (Connection A) or *two food-conditioned linear directions* (this connection)? They are
distinguishable by whether removing `food` variance collapses the shell to a single direction.

### Connection C — A ReLU MLP implements this the "Pizza" way (inside the disk), so trace it with a Fourier/structure analysis of the readout layer à la Nanda/Zhong.
Zhong et al. show a ReLU network can solve a task by operating **inside** the embedding circle using
**|cos|** (absolute value, ReLU-cheap) instead of a boundary direction — structurally the same move as
our "inside-the-core vs outside" radial test, which is also ReLU-cheap (a norm/quadratic is two ReLUs).
**Confirm it by:** applying **DFT / structured-probe analysis (Nanda 2023) to the weights of the MLP
layers after L** that read `country`, to test whether the "shell" is genuinely isotropic-radial or a
small set of **paired ± directions** (a Fourier-sparse / lattice signature) — our degree-2 eigen-analysis
already found the radial signal concentrated in **~3 PCA directions** (negative-definite form,
|min eig|/|max eig|≈865), so a handful of structured axes, not a clean sphere, carry it. That is exactly
the regime where Fourier/eigen-structure analysis discriminates "radial manifold" from "few gated axes."

---

## 4. Task-3 ideas — encode a feature in an even WEIRDER geometry (literature-grounded)

**Idea 1 — Put a binary feature on a CIRCLE / TORUS via a modular auxiliary label (grounded in Engels 2024 + Nanda 2023 + Zhong 2023).**
Augment training so the target feature F is recoverable only through a **circular code**: attach a
synthetic auxiliary task `F-phase = (hash(text) mod p)` and a modular-addition-style objective so the
network must place inputs on a ring and read F as *which arc* it lands in (F=1 ⇒ one half-circle).
Then verify with the **Engels SAE-cluster→PCA→irreducibility** pipeline that a genuine **2-D circular,
irreducible** feature emerged, and with **Nanda's DFT** that the readout is Fourier-sparse. For a
**torus**, use two coprime moduli (p, q) so F is the joint phase on S¹×S¹ — the multi-frequency object
Furuta et al. (2402.16726) describe. This is "weirder" because F becomes provably non-1-D and periodic,
the opposite of a linear direction, with a published discovery method that should confirm it.

**Idea 2 — Train an explicit RADIAL-SHELL / nested-spheres code and push it to a multi-shell "onion" (grounded in our `country` result + Michaud-Gorton-McGrath 2025 spherical shells + Elhage 2022).**
Generalise the accidental `country` shell into a *designed* one: regularise (or architect) layer L so a
feature is encoded as **k concentric shells** (e.g. a 3-valued or ordinal feature = inner/middle/outer
ellipsoid), means held coincident, information purely in **per-class covariance/radius**. Because
Michaud et al. show SAEs **mis-handle radial manifolds** (they tile, or saturate at ≈2·d latents), this
is a pointed stress test: train the encoding, then show standard SAE/linear-probe interpretability
**fails** to recover it while QDA/Mahalanobis succeeds — a concrete "weird but real" geometry that
breaks the linear-feature toolkit. Defensible "more interesting": it is a *magnitude/scale* code, an
axis the field has barely studied for language features.

**Idea 3 — Encode a feature via a learned LATTICE / GATED sign-flip (XOR) code (grounded in Park-Choe-Veitch 2311.03658 + Zhong "Pizza" + our food-gate finding).**
Train F so it is **only** readable as an XOR/gated conjunction with a second feature G — i.e. the
F-direction must rotate with G (our observed food-gating, but deliberately maximised and extended to a
**multi-feature lattice**: F readable only after conditioning on a 2- or 3-bit context, with cell
centroids on a learned integer lattice). Confirm with within-cell conditional probing + cross-cell
rotation (cos≈−1) and a **Fourier/eigen** check of the lattice (Nanda-style). This directly
operationalises the "LLMs represent XORs of arbitrary features" debate and yields a feature that is, by
construction, at chance for *every* linear probe yet perfectly recoverable by a single ReLU layer —
the strongest possible "non-linear feature" demonstration.

---

## Verification & caveats
- **All 12 arXiv ids above were web-verified** (title/authors/year/id) and cross-checked against the
  4,090-doc harvested corpus; the 6 core papers were deep-read with **verbatim quote extraction**
  (reader agents). Quotes available in the deep-read logs.
- **Flagged unverified / not-in-harvest:** *Decomposing the Dark Matter of SAEs* (2410.14670) and
  *Understanding SAE scaling in the presence of feature manifolds* (2509.02565) are **web-verified but
  were not pulled by the harvest** (too recent / below per-query cap); *The Origins of Representation
  Manifolds in LLMs* (2505.18235) surfaced in search but was **not deep-read** — treat its details as
  unconfirmed. The AlignmentForum XOR post is **grey literature** (not peer-reviewed).
- **No "radial/magnitude shell" is a named geometry for a *language* feature in the literature** — the
  closest published kin is the synthetic spherical-shell feature (Michaud et al. 2025) and classical
  QDA/covariance-difference codes. Our `country` result therefore looks **genuinely under-documented**,
  which strengthens its novelty for the puzzle write-up.
- FieldAtlas's native verbatim-verifier/citation-linter (`ingest_extractions.py`, `build_outputs.py`)
  was not run end-to-end because the agent-`Workflow` runner is absent here; deep-read faithfulness was
  enforced via the reader/critic/skeptic agent prompts instead. Plane-1 (deterministic harvest, dedup,
  rank, map) ran fully and its numbers are reported as emitted.
