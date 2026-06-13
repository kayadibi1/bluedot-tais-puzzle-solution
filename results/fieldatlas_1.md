# FieldAtlas #1 — Linear Representation Hypothesis & Superposition, tied to the `country` result

**Pipeline:** FieldAtlas (project at `C:/Users/Sidar/Desktop/Projects/lit review`), new scope
`scope/linear_rep_superposition.yaml`, dedicated DB `linear_rep_superposition.sqlite`.
**Run:** `run-20260612-233049`. Academic sources only (openalex, crossref, core, arxiv) — this is a
technical ML sub-field with no policy/grey-lit register.

**Reliability numbers (from `artifacts/RUN_REPORT.md`):**
- Gathered: 4636 raw records -> **4543 unique docs** after dedup; 1727 with OA PDF; 17,965 in-corpus citation edges.
- Tiered: Tier-1 (deep-read) **67**, Tier-2 495, Tier-3 (abstract map) 3974.
- Read (F1): **28 full texts parsed**; **4 deep-read extractions verified, 0 rejected**; **20/20 evidence spans string-matched verbatim into source, 0 failed**.
- Citations (F2): report 31 citations, **0 unresolved**; **3/3 research ideas grounded, 0 flagged**.
- NOT-READ honesty: 35 Tier-1 docs were paywalled/no-OA and contribute no deep claims.

Every claim below tagged `[verified-span]` was string-matched verbatim into the paper's parsed full
text by the deterministic verifier (`fieldatlas/verify.py`). External metadata (authors/venue/arXiv id)
was independently web-verified; nothing here is fabricated. Two key papers are flagged as NOT in the
harvested corpus but web-verified.

---

## A. Concise map of the sub-field

**The default claim — Linear Representation Hypothesis (LRH).** High-level concepts are represented
as directions in activation space. Park, Choe & Veitch formalize three *distinct* notions the looser
literature conflates: **subspace** (concept = 1-D subspace; word2vec analogies), **measurement**
("the probability of a concept value can be measured with a linear probe" `[verified-span]`), and
**intervention** ("the value a concept takes on can be changed ... by adding a suitable steering
vector" `[verified-span]`). They add a **causal inner product** under which "concepts that can vary
freely of each other are represented as orthogonal vectors" `[verified-span]`, validated on LLaMA-2.
This is the implicit null hypothesis your `country` feature violates: it is unreadable by the
*measurement* (linear-probe) notion.

**Why features are NOT axis-aligned — Superposition.** Elhage et al. (Toy Models) show small ReLU
nets pack more features than dimensions as overlapping near-orthogonal directions: "when features are
sparse, superposition allows compression beyond what a linear model would do, at the cost of
interference that requires nonlinear filtering" `[verified-span]`. Three load-bearing results:
(i) which encoding a feature gets is **governed by a phase change** — not-learned / superposed /
dedicated-dimension `[verified-span]`; (ii) superposition "organizes features into geometric
structures such as digons, triangles, pentagons, and tetrahedrons" `[verified-span]` (uniform
polytopes); (iii) **computation can happen in superposition** — the canonical case is absolute value,
"a very simple way to compute it with ReLU neurons" `[verified-span]` (abs(x)=ReLU(x)+ReLU(-x)).
Anti-correlated features "prefer to have them interfere, especially with negative interference"
`[verified-span]`; correlated features prefer orthogonality or collapse to their principal component.

**Empirical signature in real LLMs — Sparse probing.** Gurnee et al.: "Many early layer neurons are
in superposition, where features are represented as sparse linear combinations of polysemantic
neurons" `[verified-span]`; k-sparse probing "is an effective methodology to locate such neurons
(even in superposition), but requires careful use and follow-up analysis" `[verified-span]`.
Higher-level features (e.g. `is_python_code`) tend to be monosemantic in middle layers `[verified-span]`.

**The standard fix — dictionary learning / sparse autoencoders (SAEs).** If features are an
overcomplete near-orthogonal basis in superposition, recover them with sparse dictionaries: SAEs "find
highly interpretable features" (Cunningham/Sharkey et al. 2309.08600), scaled in Scaling-Monosemanticity
(Anthropic), refined by gated SAEs, transcoders, and sparse feature circuits.

**The contested frontier — features that are NOT one direction.** The **Polytope Lens** (Black,
Sharkey et al., Conjecture) argues the features-as-directions view is incomplete: under nonlinearity,
"scaling directions in one layer can change the direction (and hence the features represented) in
later layers" `[verified-span]`, so it needs a region "within which directions represent the correct
feature and outside of which they don't" `[verified-span]`. They recast piecewise-linear nets as a
partition of activation space into convex **polytope regions** `[verified-span]`, each with its own
affine map — i.e. a feature can be a *region*, not a ray. Interference: "the activation of one feature
coactivates all feature directions sharing a non-zero dot product with it" `[verified-span]`.
The published canonical example of a genuinely non-linear feature is **Engels et al. "Not All Language
Model Features Are (One-Dimensionally) Linear"** — irreducible *multi-dimensional* features (circular
day-of-week/month used for modular arithmetic). [FLAG: NOT in harvested corpus; web-verified below.]

**Open debates:** directions vs. polytope-regions as the primitive; how many features are *genuinely*
multi-dimensional vs. an artifact of the wrong basis; lack of a clean theory of non-uniform / correlated
superposition (Elhage et al. explicitly concede this `[verified-span]`).

---

## B. Most relevant papers (each: why it matters HERE)

All citations web-verified.

1. **Toy Models of Superposition** — Elhage, Hume, Olsson, Schiefer, ... Wattenberg, Olah (Anthropic /
   Transformer Circuits Thread, Sept 2022). arXiv:2209.10652 · https://arxiv.org/abs/2209.10652 ·
   https://transformer-circuits.pub/2022/toy_model/
   *Why here:* supplies the exact mechanism for `country` — interference requiring **nonlinear
   filtering**, the **phase change** that decides linear-vs-entangled, and **computation in
   superposition** (abs = ReLU(x)+ReLU(-x)), the structural twin of a radial/sign-flip read.

2. **The Linear Representation Hypothesis and the Geometry of LLMs** — Park, Choe, Veitch (ICML 2024).
   arXiv:2311.03658 · https://arxiv.org/abs/2311.03658
   *Why here:* defines the **measurement = linear-probe** notion of linear representation that the
   `country` feature uniquely fails, while the other 7 features satisfy it.

3. **Finding Neurons in a Haystack: Case Studies with Sparse Probing** — Gurnee, Nanda, Pauly, Harvey,
   Troitskii, Bertsimas (May 2023). arXiv:2305.01610 · https://arxiv.org/abs/2305.01610
   *Why here:* the real-LLM evidence that superposed features need **k-sparse / non-trivial probes**,
   not a single direction — exactly the move from a failed linear probe to the food-gated / shell read.

4. **Interpreting Neural Networks through the Polytope Lens** — Black, Sharkey, Grinsztajn, Winsor,
   Braun, ... Leahy (Conjecture, Nov 2022). arXiv:2211.12312 · https://arxiv.org/abs/2211.12312
   *Why here:* the clearest argument that a feature can need a **region / "distribution of validity"**
   instead of a direction — the literature's closest prediction of a Mahalanobis/radial-shell boundary.

5. **Not All Language Model Features Are (One-Dimensionally) Linear** — Engels, Liao, Michaud, Gurnee,
   Tegmark (ICLR 2025). arXiv:2405.14860 · https://arxiv.org/abs/2405.14860 · code:
   https://github.com/JoshEngels/MultiDimensionalFeatures
   [FLAG: NOT in the harvested corpus — the harvest mis-keyed this title to a different paper; metadata
   web-verified, not from the corpus DB.]
   *Why here:* the single most direct published analog — features that *no single direction* can read,
   found via SAEs; gives a ready-made "irreducibility" test for `country`.

6. **Towards Monosemanticity: Decomposing Language Models with Dictionary Learning** — Bricken et al.
   (Anthropic, Oct 2023). https://transformer-circuits.pub/2023/monosemantic-features
   [FLAG: present in corpus only at abstract level / not deep-read; web-verified.]
   *Why here:* the canonical SAE result — the standard tool you would point at the `country` feature in Task 3.

Supporting corpus papers (in-corpus, abstract-level, lint-verified ids): Polysemanticity and Capacity
(2210.01892), Scaling/evaluating SAEs (2406.04093), Gated SAEs (2404.16014), Transcoders (2406.11944),
Sparse Feature Circuits (2403.19647), Geometry of Concepts SAE structure (10.3390/e27040344), Task
structure & nonlinearity (2401.13558), Open Problems in Mech Interp (2501.16496).

---

## C. Three connections from this literature to the `country` result

The `country` feature at layer L has **coincident class means** (linear probe fails), is recoverable
by a **quadratic/Mahalanobis "radial shell"** boundary, OR by a **food-gated sign-flipping direction**.

**Connection 1 — Radial shell = "distribution of validity" / polytope region (mechanism: nonlinear,
region-bounded feature).** The Polytope Lens predicts exactly this: directions are not scale-invariant
under nonlinearity, so a feature may only be valid "within which directions represent the correct
feature and outside of which they don't" `[verified-span]`, and piecewise-linear nets carve activation
space "into a number of convex shapes called polytopes" `[verified-span]`. A radial/Mahalanobis shell
is a smooth instance of a validity region: `country` is not a ray but a *shell* in activation space.
The literature predicts such representations whenever the directions frame is pushed through ReLU/GELU.

**Connection 2 — Coincident means via interference / anti-correlated superposition (mechanism:
superposition + negative interference).** Toy Models shows superposition stores features at the cost of
"interference that requires nonlinear filtering" `[verified-span]`, and that anti-correlated features
"prefer to have them interfere, especially with negative interference" `[verified-span]`. If `country`
is superposed against the other 7 features (or anti-correlated with one), the projection of its two
classes onto any single linear axis is pushed to **coincide** (means collapse), leaving the signal only
in second-order / off-axis structure — precisely a vanishing first moment with surviving second moment
(Mahalanobis-readable). This is the literature's account of *why* a linear probe would see equal means.

**Connection 3 — Food-gated sign-flip = computation in superposition / conditional feature (mechanism:
conditional/gated feature, computation-in-superposition).** Toy Models' headline non-storage result is
that models *compute* in superposition, the example being abs(x)=ReLU(x)+ReLU(-x) `[verified-span]` — a
**sign-folding** nonlinearity. A direction whose sign-meaning flips conditioned on the `food` feature is
the same construction: `country` is read by `sign(food) * <w, h>`, i.e. a feature *gated* by another,
implemented through the ReLU the way abs is. Engels et al. (web-verified, not in corpus) give the
published existence proof that such irreducible/multi-dimensional features really occur in trained LLMs.
Gurnee et al. supply the recovery tool: when one direction fails, a **k-sparse / conditional probe**
locates the feature in superposition `[verified-span]`.

**Verdict:** Yes — the literature predicts and names this. The `country` representation is a textbook
intersection of (a) superposition-with-interference collapsing class means, (b) a polytope/validity
*region* (the shell) rather than a direction, and (c) a *conditional/gated feature computed in
superposition* (the food-gated sign flip). None of this is anomalous to the field; it is exactly its
contested frontier.

---

## D. Concrete ideas for Task 3 ("an even weirder representation of a feature")

All three were grounded against the deep-read papers and passed the citation linter (`ideas.md`, 3 accepted / 0 flagged).

**Idea 1 — Engineer a phase transition between linear and radial-shell coding by tuning sparsity +
food-correlation.** Toy Models' phase change `[verified-span]` and correlated/anti-correlated theory
`[verified-span]` say the encoding *snaps* between regimes. Sweep the `country` base rate (sparsity) and
its correlation with `food` in synthetic data; at each setting measure (a) linear-probe AUC, (b)
radial/Mahalanobis AUC, (c) food-gated sign-flip AUC. Expected: a sharp boundary where the feature jumps
from linear-direction to coincident-means/shell — turning the puzzle's one observation into a *phase
diagram*. (Grounds: 2209.10652, 2305.01610. Feasibility: high — the puzzle's own MLP harness.)

**Idea 2 — Train an SAE on hidden layer L and test whether `country` is irreducibly multi-dimensional.**
Following the polytope/sparse-probing line, train a (gated) SAE on layer L and apply the Engels-style
irreducibility test: do the 7 linear features each map to one SAE latent while `country` only appears as a
multi-dimensional / food-gated / shell latent? A clean within-model demo that SAEs recover the non-linear
feature the linear probe misses. (Grounds: 2211.12312, 2305.01610.)

**Idea 3 — Force computation-in-superposition: make the target feature an XOR/product gated by another.**
Toy Models proves abs-style computation in superposition `[verified-span]`. Construct a feature whose
readout is the **XOR** of two others (non-linearly separable, mean-coincident by construction) or a
**product gated** by a third, forcing a curved/region boundary. Verify no linear probe and no single
radial shell suffices, but a 2-layer/kernel probe does — a feature *strictly weirder* than `country`,
with a documented mechanism. (Grounds: 2209.10652, 2211.12312.)

---

## E. Things I could not verify / flags

- **Workflow tool absent.** FieldAtlas's agent-swarm deep-read/synth workflows require a `Workflow`/Task
  runner not present in this environment. I drove the deterministic Python layer directly and acted as the
  reader/synthesis agent myself, then ran the *deterministic guards* that matter: span verifier
  (`ingest_extractions.py`, 20/20 spans verbatim) and citation linter (`build_outputs.py`, 0 unresolved).
  So the reliability property (no fake reading, no fabricated cites) is preserved, but the multi-agent
  critic/skeptic cross-check phase did not run.
- **Skill base path was wrong.** SKILL.md hardcodes `C:/Users/Sidar/Desktop/lit review`; the project is
  actually at `C:/Users/Sidar/Desktop/Projects/lit review`. Used the real path.
- **Two key papers NOT in the harvested corpus:** Engels et al. "Not All LM Features Are Linear"
  (arXiv:2405.14860) and the Olah et al. "Zoom In: Circuits" thread — web-verified, not corpus-grounded.
  The corpus *did* mis-key the Engels title onto a different paper (a metadata collision), which is why
  the citation linter, run against true canonical ids, is load-bearing.
- **SAE "Find Highly Interpretable Features" (2309.08600) and Towards Monosemanticity** are in-corpus at
  abstract level only (no OA PDF at harvest time), so they are cited but NOT deep-read/span-verified.
- **Firecrawl out of credits;** external verification used the built-in WebSearch instead.

**FieldAtlas artifacts (full versions):** `C:/Users/Sidar/Desktop/Projects/lit review/artifacts/`
(`report.md`, `trends.md`, `ideas.md`, `map.md`, `RUN_REPORT.md`, `report.html`).
