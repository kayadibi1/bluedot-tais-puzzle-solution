# Approach #10 - Curved-manifold / local-structure / periodic analysis

Layer L = h2 (post-ReLU hidden layer 2, 64-dim). Train N=7000, Test N=1500.

## 0. Linear-probe AUC sweep on h2 (confirm F = the low outlier)

| idx | feature | linear_AUC |
|---|---|---|
| 0 | number | 0.9973 |
| 1 | question | 1.0000 |
| 2 | color | 0.9975 |
| 3 | food | 0.9960 |
| 4 | sentiment | 0.9955 |
| 5 | country | 0.4869 |
| 6 | person | 1.0000 |
| 7 | body_part | 0.9986 |

**F = `country` (index 5)**, linear AUC=0.4869; next-worst (`sentiment`)=0.9955; gap=0.5086.

Linear control feature = `food` (index 3), linear AUC=0.9960.

## 1. LOCAL vs GLOBAL: kNN (small k) vs linear probe

If kNN AUC >> linear AUC, F's structure is local/curved (not a half-space).

| feature | linear_AUC | kNN k=1 | kNN k=3 | kNN k=5 | kNN k=9 | kNN k=15 | kNN k=31 |
|---|---|---|---|---|---|---|---|
| country | 0.4869 | 0.9602 | 0.9686 | 0.9725 | 0.9773 | 0.9855 | 0.9892 |
| food | 0.9960 | 0.9733 | 0.9857 | 0.9879 | 0.9904 | 0.9911 | 0.9888 |

**F kNN-vs-linear gap = +0.5023** (best kNN AUC 0.9892 vs linear 0.4869).
Control `food` kNN-vs-linear gap = -0.0049 (kNN 0.9911 vs linear 0.9960).
Ratio F-gap / control-gap = 102.0x.

## 2. Manifold embedding of h2 colored by F's label

Running Isomap (n=2500) ...
Running UMAP (n=2500) ...
Saved results\approach_10_embeddings.png

## 3. Periodic / angular structure test

- Radial separability (|r| in F-subspace): AUC = 0.5557 (>0.5 => shell/dome; positive class at one radius band).
- Per-sector F+ rate (24 angular sectors), range [0.41, 0.57], std=0.042 (high std + alternation => periodic).
- Dominant angular frequency of F+ rate = 4 cycle(s) over 2pi (freq=1 => single arc/half; >=2 => alternating sectors / multi-lobe).
- Angular+radial logistic probe (cos/sin k=1..3 in F-subspace): test AUC = 0.5741 vs raw-linear 0.4869.
  Angular-vs-linear gap = +0.0872 (in just a 2-D curved subspace).

Saved results\approach_10_angular.png

## 4. Intrinsic dimensionality of F's class manifolds

- PCA participation-ratio dim: F+ cloud = 2.46, F- cloud = 4.30, all-h2 = 3.93 (of 64).
- PCs to 90% var: F+ cloud needs 4, F- cloud needs 6, all-h2 needs 7.
- Local participation-ratio dim around F+ points (30-NN): median=3.95, mean=3.92 (low => points lie on a thin local manifold/filament).

## Summary characterization of F's geometry at Layer L

- F = `country` (index 5); linear AUC 0.4869 is the unique low outlier (next-worst 0.9955).
- kNN recovers F: best kNN AUC 0.9892 => kNN-vs-linear gap +0.5023; control `food` gap -0.0049. Local structure carries F, a global half-space does not.
- Radial AUC 0.5557; angular+radial 2-D probe AUC 0.5741; dominant angular frequency 4.
- Intrinsic dim: F+ PR-dim 2.46, local PR-dim ~3.95.

**Shape verdict (refined by deep dive + enclosure tests below):**
F=country is an **ellipsoidal shell / enclosed blob**: the country-POSITIVE points
form one compact cluster, with the country-NEGATIVE points wrapped around them as a
surrounding shell. A hyperplane cannot cut an interior blob out of an enclosing
shell (hence linear AUC ~ chance 0.49), but a single quadric (ellipsoid) does.

## Deep-dive: precise shape (see approach_10_deep.md, a10_manifold_deep.py)

- 1-hidden MLP(32) on h2 -> F test AUC **0.9931** (F is decodable, not linearly).
- In the MLP's own discriminative 2-D plane: linear 0.508, **quadratic 0.992**,
  kNN(15) 0.979. Quadratic >= kNN => a SINGLE smooth curved surface, not many
  interleaved filaments. The plot (approach_10_shape.png) shows positives as a
  central blob enclosed by negatives fanning into outer lobes.
- Class centroids nearly coincide: ||muP-muN|| = 0.079, nearest-centroid AUC 0.507
  => concentric/enclosed, NOT two displaced blobs.
- Single FULL quadratic (all cross terms, PCA-12) AUC **0.9923** ~ kNN-full 0.9855
  => kNN gives no advantage over one quadric => ONE fold/quadric, not filaments.
- Local same-label purity (16-NN): F=0.914 vs control food=0.892 (high) -- locally
  clustered yet linearly at chance => globally enclosed, locally pure.
- F is NOT a literal XOR/parity of the other 7 features: logistic on other-7
  linear scores 0.479, +pairwise products only 0.552 (gain +0.073).

## Enclosure quantification (a10_enclosure.py)

- 2-D MLP-plane, radius from positive-blob center: AUC **0.986**; median radius
  pos=0.504 vs neg=1.861; **99.97% of negatives lie outside the 95th-pct positive
  radius** -- near-total enclosure.
- 64-D Mahalanobis-to-F-ellipsoid AUC **0.991** (Maha^2 pos=7.4 vs neg=35.3).
- 64-D plain Euclidean radius AUC only 0.790 => the shell is ANISOTROPIC
  (ellipsoidal, not spherical); that anisotropy is why axis-aligned quadratic
  (0.942) < full quadratic with cross terms (0.992).

**FINAL GEOMETRY:** F=country at Layer L is an **anisotropic ellipsoidal shell**:
country-positive inputs collapse to a tight interior ellipsoidal cluster while
country-negatives surround them as an enclosing shell. Separable by one quadric /
Mahalanobis radius (AUC ~0.99), invisible to any hyperplane (AUC ~0.49). The
earlier "negative-definite dome / positive INSIDE shell" (approach #4) is the same
object, here pinned down quantitatively as an enclosure (99.97% of negatives
outside the positive blob) rather than a ring or interleaved filaments.

Plots: approach_10_embeddings.png, approach_10_angular.png, approach_10_shape.png.
