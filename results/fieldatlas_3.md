# FieldAtlas slice — Probing classifiers & the validity of linear probes

**Field slice:** Probing classifiers and the validity of linear probes for reading features
from neural activations — what linear probes can/cannot detect, non-linear (MLP) probing,
amnesic probing / INLP / concept erasure, conditional & XOR-encoded features that are only
readable given another variable, feature entanglement/absorption, and causal vs correlational
probing.

**Why this slice:** the BlueDot puzzle's concrete result is a *probing-methodology* artifact.
At hidden layer L (`h2`, post-ReLU, 64-d) a **linear probe reads `country` at AUC 0.49
(chance)** while every other of the 8 features reads at AUC ≥ 0.9955, yet **non-linear probes
recover `country`** (RBF-SVM 0.965, kNN 0.933, QDA 0.990, Mahalanobis 0.995, and the full model
0.994). Two structural accounts were advanced: a **radial/“shell” (variance-coded)** geometry
(country=1 = compact core, country=0 = ~2.2× diffuse shell, concentric means ‖Δμ‖≈0.01,
Bhattacharyya cov/mean ratio ≈ 454) and a **`food`-GATED / conditional sign-flipping** direction
(country linearly readable ONLY within a fixed `food` value: within-cell AUC 0.999; the country
axis is **sign-flipped, cos ≈ −0.99**, between the two food halves; global linear AUC ≈ 0.51).
Round-2 unification showed these are **largely the same object** — the dominant variance-shell
eigenvector ≈ the food-gating sign-flip axis (principal-angle cos 0.989) — **plus** a genuine
secondary food-independent radial variance-collapse component (86×/106× variance ratios).

---

## Pipeline / provenance note (read this for the reliability accounting)

The FieldAtlas skill was invoked (`Skill: fieldatlas`) and its deterministic **Plane-1**
pipeline was run on a **dedicated scope + DB** for this field
(`scope/probing_classifiers.yaml`, `data/probing_classifiers.sqlite`) per the skill's
"different field" protocol — academic connectors only (`openalex,crossref,core`), since the
grey-lit / Federal-Register feeds are AI-policy-curated and contribute nothing here. The live
API smoke test passed (OpenAlex 200, CORE 200, arXiv 200) and the harvest ingested a real
corpus (**~4,916 documents** deduped into the DB).

**What did NOT run as the skill intends, and why:** the FieldAtlas **deep-read** and
**synthesis/idea** stages are agentic workflows (`fieldatlas/workflows/*.workflow.js`) that must
be launched through a **Workflow runtime tool** (`pipeline`/`agent`/`phase`/`parallel`
primitives). That tool is **not exposed in this environment**, so those two stages could not be
driven natively. Per the task's explicit fallback clause, the *map + citations + synthesis*
deliverable below was produced by **rigorous web research with verbatim source verification**
(every arXiv id, title, author list, year, and venue was fetched and checked against the actual
abstract page). The Plane-1 corpus grounding is real; the synthesis is human/agent-verified
rather than verbatim-span-verified through the FieldAtlas verifier. **Flagged as such for
honesty** — no FieldAtlas verification numbers (spans verbatim/caught, citation-lint) are
claimed for the synthesis stage because that stage's runtime was unavailable.

All citations below are **VERIFIED** (fetched live). None are fabricated.

---

## 1. Concise map of the field

Probing classifiers = train a (usually simple) classifier on a model's frozen internal
activations to test whether a property is *decodable* from them. The field's central, hard-won
lesson is that **decodability ≠ use ≠ presence**, and that the *probe family* you choose
silently defines what "the representation contains" means.

**(a) The probe + the foundational caveat.** Alain & Bengio introduced the **linear classifier
probe** and showed linear separability rises monotonically with depth. But a probe that succeeds
tells you the info is *linearly present*; a probe that *fails* (like our `country` at L) does
**not** by itself prove absence — only that it is not in a single linear direction.

**(b) Probe capacity & the selectivity problem (the crux for "MLP beats linear").** A more
powerful probe always decodes ≥ as much, so high accuracy can reflect **probe learning** rather
than **representation structure**. Two camps:
- **Hewitt & Liang (control tasks / selectivity):** pair every probe with a *control task*
  (random labels); trust a probe only if it has high *selectivity* (real-task acc − control-task
  acc). Favors **low-capacity (linear)** probes precisely because a high-capacity MLP can fit
  random labels, inflating apparent "encoding."
- **Pimentel et al. (information-theoretic):** probing estimates **mutual information**; since
  MI is invariant to a bijective probe, you should use the *highest-performing* probe you can.
  **Voita & Titov (MDL probing):** report **description length**, not accuracy — it captures both
  *quality* and *the amount of effort/structure* needed to extract a property, and is more stable.

This camp split is exactly the lens for our finding: an MLP probe recovering `country` at 0.99
where a linear probe is at chance is the **textbook MLP-vs-linear gap**, and the literature says
you must *control* it (control tasks / MDL) before concluding "the feature is there but
non-linear" vs "the probe just memorized."

**(c) Conditional / baseline-relative probing.** **Hewitt, Ethayarajh, Liang & Manning —
conditional probing** extends V-information to measure information in a representation *beyond a
baseline*, by **explicitly conditioning on the baseline variable**. This is the direct
methodological formalization of "readable only AFTER conditioning on another variable" — i.e.
our `food`-gating.

**(d) Causal / amnesic probing & concept erasure (correlational → causal).**
- **INLP — Ravfogel et al., "Null It Out":** iteratively train linear classifiers and project
  activations onto their null space to **linearly erase** a concept (guarding protected
  attributes).
- **Amnesic probing — Elazar, Ravfogel, Jacovi & Goldberg:** *remove* a property (via INLP) and
  measure the **causal** effect on the model's behavior — decodability alone doesn't tell you the
  model *uses* the property; amnesic counterfactuals do.
- **LEACE — Belrose et al.:** closed-form, **provably perfect linear erasure** ("concept
  scrubbing" across all layers) — the modern guarantee that *no* linear probe can read a concept
  after erasure.
- **Marks & Tegmark, "Geometry of Truth":** the gold-standard pairing of linear probes with
  **causal interventions** (patching the probe direction flips model behavior) — the bar for
  claiming a probe direction is *used*, not merely *correlated*.

**(e) When features are NOT one linear direction (our case).**
- **XOR / conditional features:** documented that transformer MLPs **compute XORs of arbitrary
  features**, so a property can be present as a *boolean function of other features* that a linear
  probe cannot read but a non-linear probe can. The cleanest published instance:
- **Mallen, Belrose et al., "Eliciting Latent Knowledge from Quirky Language Models":** models
  fine-tuned to err **iff the keyword "Bob" is present**; truth is recoverable by probes
  **conditionally on context** in middle layers. This is a near-exact analog of `country` being
  readable **gated on `food`**.
- **Engels et al., "Not All Language Model Features Are (One-Dimensionally) Linear":** features
  can be **irreducibly multi-dimensional** (e.g. circular day/month features) — direct support for
  our **radial/shell (multi-dimensional, variance-coded)** account.
- **Feature absorption (Chanin et al.) / sparse probing (Gurnee et al.):** SAE latents and neurons
  **entangle/absorb** features (a "starts-with-D" feature absorbed into a "dogs" latent), and many
  features live in **superposition** — readability of one feature is entangled with others, which
  is precisely the *food/country entanglement* we observe.

**Established vs contested.** *Established:* probe choice changes conclusions; decodability ≠
causal use; erasure (INLP/LEACE) + amnesic/causal intervention are the way to upgrade
correlational probes; some features are non-linear / multi-dimensional / conditional. *Contested:*
the **selectivity vs. information-theoretic** debate (low-capacity control-task probes
[Hewitt & Liang] vs. highest-capacity MI/MDL probes [Pimentel; Voita & Titov]) — unresolved, and
it is exactly the debate our MLP-beats-linear result sits inside.

---

## 2. The most relevant papers (one line each on why it matters HERE)

All VERIFIED (live-fetched title/authors/year/venue/arXiv id).

1. **Alain & Bengio (2016), "Understanding intermediate layers using linear classifier probes,"
   arXiv:1610.01644.** — Defines the linear probe; establishes that a *linear* probe failing ≠
   feature absent — the premise of the whole puzzle.
2. **Hewitt & Liang (2019), "Designing and Interpreting Probes with Control Tasks," EMNLP,
   arXiv:1909.03368.** — Control tasks + **selectivity**: the control we MUST run before trusting
   that our MLP's 0.99 reflects representation, not probe memorization.
3. **Pimentel, Valvoda, Hall Maudslay, Zmigrod, Williams & Cotterell (2020),
   "Information-Theoretic Probing for Linguistic Structure," ACL, arXiv:2004.03061.** — Argues for
   the *highest-capacity* probe (MI view) — the counter-position that *licenses* using our MLP probe.
4. **Voita & Titov (2020), "Information-Theoretic Probing with Minimum Description Length," EMNLP,
   arXiv:2003.12298.** — **MDL** as the stable metric capturing *how hard* a feature is to extract
   — the right way to quantify "linear-hard, non-linear-easy" for `country`.
5. **Hewitt, Ethayarajh, Liang & Manning (2021), "Conditional probing: measuring usable
   information beyond a baseline," EMNLP, arXiv:2109.09234.** — Formalizes reading a property
   **conditioned on a baseline variable** — the exact methodology for our `food`-gated `country`.
6. **Ravfogel, Elazar, Gonen, Twiton & Goldberg (2020), "Null It Out: Guarding Protected
   Attributes by Iterative Nullspace Projection," ACL, arXiv:2004.07667.** — **INLP**: linearly
   erase a concept — a control to test whether `country`'s linear signal is *truly* absent at L.
7. **Elazar, Ravfogel, Jacovi & Goldberg (2020), "Amnesic Probing: Behavioral Explanation with
   Amnesic Counterfactuals," TACL, arXiv:2006.00995.** — Decodability → **causal use**: removing
   `country` and measuring behavioral impact is how we'd show the model *uses* the shell/gate code.
8. **Belinkov (2021), "Probing Classifiers: Promises, Shortcomings, and Advances," Computational
   Linguistics, arXiv:2102.12452.** — The canonical survey; the methodological checklist we should
   audit our claim against.
9. **Belrose, Schneider-Joseph, Ravfogel, Cotterell, Raff & Biderman (2023), "LEACE: Perfect
   linear concept erasure in closed form," NeurIPS, arXiv:2306.03819.** — Provable linear erasure
   ("concept scrubbing"): the strongest control for "no linear direction carries `country` at L."
10. **Marks & Tegmark (2023), "The Geometry of Truth: Emergent Linear Structure in LLM
    Representations of True/False Datasets," arXiv:2310.06824.** — Probe + **causal intervention**
    template: the bar for claiming our shell/gate direction is *used*, not just correlated.
11. **Mallen, Belrose, Brumley, Kharchenko et al. (2023), "Eliciting Latent Knowledge from Quirky
    Language Models," arXiv:2312.01037.** — Feature recoverable **conditional on a context keyword
    ("Bob")** — the closest published analog of `country` being readable **gated on `food`**.
12. **Engels, Liao, et al. (2024), "Not All Language Model Features Are One-Dimensionally Linear,"
    arXiv:2405.14860.** — **Irreducibly multi-dimensional** features (circular day/month) — direct
    support for our **radial/shell multi-dimensional** account.
13. **Gurnee, Nanda, Pauly, Harvey, Troitskii & Bertsimas (2023), "Finding Neurons in a Haystack:
    Case Studies with Sparse Probing," arXiv:2305.01610.** — **Superposition / sparse probing**:
    features entangled across neurons — frames why `food` and `country` are entangled at L.
14. **Chanin, Wilken-Smith, Dulka, Bhatnagar, Golechha & Bloom (2024), "A is for Absorption:
    Studying Feature Splitting and Absorption in Sparse Autoencoders," arXiv:2409.14507.** —
    **Feature absorption**: one feature's readability absorbed into another — the SAE-era statement
    of the `food`↔`country` entanglement we see.

---

## 3. Three specific connections to our result

### (a) Is "linear-probe-at-chance but MLP-recovers" exactly a documented probing failure mode? **YES.**
This is the canonical scenario at the heart of the **probe-capacity / selectivity** literature.
Alain & Bengio's framing already implies a linear-probe failure is *not* evidence of absence;
Hewitt & Liang, Pimentel et al., and Voita & Titov exist *because* a higher-capacity probe can
decode what a linear one cannot, and the open question is whether that reflects the
*representation* or the *probe*. Our numbers are a clean instance: linear AUC 0.49 → RBF 0.965 /
kNN 0.933 / QDA 0.990 / Mahalanobis 0.995, with the full model at 0.994 — the information is
present, just not in one linear direction. **Caveat the literature forces:** the MLP-beats-linear
gap *alone* is consistent with both "non-linear representation" AND "high-capacity probe
overfitting/memorizing" — which is why the controls in (c) are mandatory before we conclude.
The **layer-localization** (country linear AUC: emb 0.9998 → h1 0.9994 → **h2 0.5040** → h3 0.9945)
is strong extra evidence it is a *real, transient non-linear bottleneck* at L, not probe
artifact — a model overfit would not vanish one layer later.

### (b) Does the `food`-gated / conditional reading match known conditional-/XOR-feature results? **YES, very tightly.**
- **Conditional probing (Hewitt et al. 2021)** is the *exact* methodology: information about
  `country` is present **conditional on `food`** but not marginally. Our within-`food` linear AUC
  0.999 vs global 0.51 is a textbook "usable-information-beyond-a-baseline" result with `food` as
  the baseline.
- **Quirky models (Mallen, Belrose et al. 2023)** is the closest *empirical* analog: a feature
  (truth) readable **gated on a context bit** ("Bob"). Our `food` gate plays the role of "Bob."
- **XOR features:** our measured `country XOR food` linear AUC = 0.7566 (vs country alone 0.51,
  food alone 1.0), and the **sign-flip (cos −0.99)** of the country axis across food halves, is
  exactly the "boolean function of features that a linear probe can't read but a non-linear one
  can" phenomenon documented for transformer MLPs. **Nuance (matches the literature's nuance):**
  ours is **not a pure 2-bit XOR** — it is a *gated/rotated* direction (the country offset flips
  sign with food), which is why `country XOR food` reads 0.76 not ~0.5; the conjunction cells are
  partially linearly arranged. That is consistent with **conditional/multi-dimensional** features
  (Engels et al.) rather than a clean parity bit.
- The **radial/shell** account maps onto **Engels et al.** (irreducibly multi-dimensional /
  variance-coded features) and the **entanglement** of `food`+`country` maps onto **superposition
  (Gurnee et al.)** and **feature absorption (Chanin et al.)**.

### (c) What controls does the literature say we MUST run to trust the conclusion?
1. **Control tasks / selectivity (Hewitt & Liang 2019).** Re-fit the **MLP probe on random
   labels** for `country` at L; report selectivity (real − control). If the MLP also fits the
   control task well, its 0.99 is partly capacity, not representation. *This is the single most
   important missing control.*
2. **MDL / information-theoretic probing (Voita & Titov 2020; Pimentel et al. 2020).** Report
   **description length** (online/variational coding) for linear vs MLP probes of `country` across
   layers — quantifies "linear-hard, non-linear-easy" in a probe-capacity-robust way, and should
   show a sharp MDL spike at L.
3. **Amnesic / causal probing (Elazar et al. 2020) + erasure (INLP — Ravfogel et al. 2020; LEACE —
   Belrose et al. 2023).** Use **LEACE/INLP** to erase the (gated) `country` direction at L and
   measure the **causal** effect on the model's `country` *logit* (and confirm other 7 logits are
   untouched). Decodability ≠ use; only erasure-then-behavior shows the model *uses* the shell/gate
   code. **Critically for the gated case:** erase the *food-conditional* country direction (the
   sign-flip axis), since a single global linear erasure will (by construction) do nothing — the
   global direction is already at chance.
4. **Causal intervention / patching (Marks & Tegmark 2023).** Patch *along the shell/gating axis*
   (and along the secondary food-independent radial axes) and verify the `country` prediction flips
   — distinguishing the *used* direction from merely-present variance.
5. **Conditional-probing baseline (Hewitt et al. 2021).** Report `country` V-information
   **conditioned on `food`** vs marginal, formalizing the gate.
6. **Random-direction / shuffled-label sanity** on the QDA/Mahalanobis recovery to rule out that
   the *quadratic* probe is exploiting nuisance covariance (the puzzle already noted QDA needs mild
   reg due to 83% sparsity — selectivity controls on QDA matter too).

---

## 4. Two–three concrete ideas for Task 3 (puzzle part 3: "train a weirder representation"), grounded in this literature

The puzzle's open task is to **train a model that encodes a feature in a *more interesting* /
probing-resistant way**. Each idea is grounded in specific verified papers and is a *deliberate*
construction of a documented hard case.

**Idea A — A genuine k-bit XOR / parity gate (grounded in: XOR-feature work; Mallen/Belrose 2023
quirky models; Hewitt et al. 2021 conditional probing).**
Train the head with an auxiliary loss that forces `country` at layer L to be encoded as the
**parity of 2–3 other features** (e.g. `country_readout = food ⊕ sentiment ⊕ color`), so that a
linear probe AND any probe conditioned on a *single* other feature both read chance — readability
appears **only** after conditioning on the full k-1-bit context. Success metric: linear AUC ≈ 0.5,
MLP AUC ≈ 1.0, and **within-single-feature** conditional AUC ≈ 0.5 for all gates (strictly harder
than ours, where conditioning on `food` alone suffices). This makes the "conditional probing"
baseline have to condition on a *set*, and directly demonstrates the XOR-of-features phenomenon as
a controlled artifact. Defense of "more interesting": ours is a 1-gate sign-flip (cos −0.99);
this is an irreducible k-bit parity — no single conditioning variable unlocks it.

**Idea B — An erasure-resistant / "absorbed" encoding (grounded in: INLP Ravfogel 2020; LEACE
Belrose 2023; feature absorption Chanin 2024; superposition Gurnee 2023).**
Train so `country` is carried by a direction that is **provably hard to linearly erase without
destroying another feature** — i.e. deliberately **absorb** `country` into `food`'s subspace
(shared low-rank code) so that applying LEACE/INLP for `country` either fails (no linear direction
to null) or **collaterally damages `food`**. Success metric: LEACE/INLP targeting `country` leaves
`country` logit ≥ X recoverable by MLP, OR erasing it drops `food` accuracy by > Y. This turns
"feature absorption" from an SAE-era bug into a controlled adversarial encoding and directly
stress-tests the erasure controls from §3.

**Idea C — A multi-dimensional manifold code: circular / toroidal `country` (grounded in: Engels
et al. 2024 multi-dimensional features; our own radial/shell finding).**
Instead of a 1-D direction, train `country` (or a constructed cyclic feature like
"month"/"compass-direction") onto an **irreducible 2-D circular manifold** at L, so that *no*
linear probe and *no axis-aligned quadratic* reads it, but a probe that knows the manifold
(angle on the learned circle) reads it at ceiling. Combine with our **shell** trick (encode a
second feature in the *radius*, the gated feature in the *angle*) to get a **polar code**:
linear AUC ≈ 0.5 for both, radius-probe reads feature-1, angle-probe reads feature-2, and they
are only jointly decodable. Success metric: reproduce Engels-style "irreducible
multi-dimensionality" (can't be decomposed into independent/non-co-occurring 1-D features) as a
*deliberately trained* structure, strictly weirder than the puzzle's accidental ellipsoidal shell.

---

### Verification status (honesty ledger)
- **All 14 citations: VERIFIED** — title, authors, year, venue, and arXiv id fetched live from the
  arXiv abstract pages / search and cross-checked. No fabricated references.
- **FieldAtlas Plane-1 (harvest/dedup): RAN** on a dedicated scope+DB; ~4,916 real documents
  ingested; live connectors confirmed (OpenAlex/CORE/arXiv all HTTP 200).
- **FieldAtlas deep-read + synthesis stages: NOT run via the FieldAtlas Workflow runtime** (that
  tool is not available in this environment). The synthesis above is therefore **web-research-grade
  with manual source verification**, NOT FieldAtlas verbatim-span-verified — flagged explicitly.
  No span-verification or citation-lint numbers are claimed for the synthesis.
- **One unverified-by-primary-source item, flagged:** the specific empirical claim that
  "transformer MLPs compute XORs of arbitrary features in early layers" is sourced from a
  LessWrong/AlignmentForum analysis post (Mechanistic-interpretability community), not a
  peer-reviewed paper; it is corroborated by the peer-reviewed quirky-models (2312.01037) and
  conditional-probing (2109.09234) results, which are the citations relied on above.
