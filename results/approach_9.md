# Approach 9 -- Intrinsic Dimensionality / Minimal Subspace per Feature

**Layer L = h2 (post-ReLU hidden 2, 64-d).** Probes on standardized h2; AUC on held-out test (1500). Ceiling = small-MLP nonlinear probe.

## VERDICT: F = **country** (index 5)

- 1-direction linear AUC = **0.4869**
- Full 64-d linear AUC = **0.4869**
- Nonlinear ceiling AUC = **0.9898**
- Nonlinear-minus-fulllinear gap = **+0.5029**
- Deficit type: **TRUE NONLINEARITY -- variance/covariance-coded (class MEANS coincide, so NO linear direction works at any dim; a low-dim NONLINEAR probe on the covariance-difference axes recovers F)**

## Per-feature table: #directions to linear ceiling

| idx | feature | 1-dir AUC | full-lin AUC | nonlin AUC | k_greedy->ceil | k_pca->ceil | nl-full gap | L1 #nz |
|----|---------|-----------|--------------|------------|------|------|------|------|
| 0 | number | 0.9973 | 0.9973 | 0.9961 | 1 | 8 | -0.0012 | 6 |
| 1 | question | 1.0000 | 1.0000 | 1.0000 | 1 | 8 | +0.0000 | 7 |
| 2 | color | 0.9975 | 0.9975 | 0.9961 | 1 | 16 | -0.0013 | 7 |
| 3 | food | 0.9960 | 0.9960 | 0.9956 | 1 | 16 | -0.0005 | 8 |
| 4 | sentiment | 0.9955 | 0.9955 | 0.9953 | 1 | 16 | -0.0002 | 7 |
| 5 | country | 0.4869 | 0.4869 | 0.9898 | None | None | +0.5029 | 8 |
| 6 | person | 1.0000 | 1.0000 | 0.9999 | 1 | 8 | -0.0001 | 7 |
| 7 | body_part | 0.9986 | 0.9986 | 0.9982 | 1 | 8 | -0.0004 | 6 |

## Greedy-deflation AUC curves (AUC vs #directions)

- **number**: 1:0.997, 2:0.997, 3:0.997, 4:0.997, 5:0.997, 6:0.997, 7:0.997, 8:0.997
- **question**: 1:1.000, 2:1.000, 3:1.000, 4:1.000, 5:1.000, 6:1.000, 7:1.000, 8:1.000
- **color**: 1:0.997, 2:0.997, 3:0.997, 4:0.997, 5:0.997, 6:0.997, 7:0.997, 8:0.997
- **food**: 1:0.996, 2:0.996, 3:0.996, 4:0.996, 5:0.996, 6:0.996, 7:0.996, 8:0.996
- **sentiment**: 1:0.995, 2:0.995, 3:0.995, 4:0.995, 5:0.995, 6:0.995, 7:0.995, 8:0.995
- **country**: 1:0.487, 2:0.487, 3:0.487, 4:0.486, 5:0.487, 6:0.490, 7:0.490, 8:0.489
- **person**: 1:1.000, 2:1.000, 3:1.000, 4:1.000, 5:1.000, 6:1.000, 7:1.000, 8:1.000
- **body_part**: 1:0.999, 2:0.999, 3:0.999, 4:0.999, 5:0.999, 6:0.999, 7:0.999, 8:0.999

## PCA-restricted probe AUC (top-k PCA dims, k=[1, 2, 3, 4, 5, 8, 16, 32, 64])

- **number**: 1:0.739, 2:0.746, 3:0.814, 4:0.813, 5:0.830, 8:0.997, 16:0.997, 32:0.997, 64:0.997
- **question**: 1:0.525, 2:0.556, 3:0.729, 4:0.729, 5:0.927, 8:1.000, 16:1.000, 32:1.000, 64:1.000
- **color**: 1:0.651, 2:0.714, 3:0.714, 4:0.784, 5:0.803, 8:0.906, 16:0.997, 32:0.997, 64:0.997
- **food**: 1:0.671, 2:0.911, 3:0.912, 4:0.912, 5:0.913, 8:0.984, 16:0.996, 32:0.996, 64:0.996
- **sentiment**: 1:0.529, 2:0.683, 3:0.724, 4:0.739, 5:0.900, 8:0.967, 16:0.995, 32:0.995, 64:0.995
- **country**: 1:0.490, 2:0.499, 3:0.495, 4:0.500, 5:0.504, 8:0.479, 16:0.487, 32:0.487, 64:0.487
- **person**: 1:0.762, 2:0.869, 3:0.951, 4:0.952, 5:0.953, 8:1.000, 16:1.000, 32:1.000, 64:1.000
- **body_part**: 1:0.742, 2:0.757, 3:0.959, 4:0.950, 5:0.971, 8:0.998, 16:0.999, 32:0.999, 64:0.999

## F=country deep dive (geometry of the nonlinear carrier)

### Why linear fails: class MEANS coincide for F only

||mu_pos - mu_neg|| in standardized h2 (small => no linear direction can separate the classes):

| feature | mean-sep |
|---|---|
| number | 2.427 |
| question | 1.639 |
| color | 1.922 |
| food | 2.468 |
| sentiment | 1.554 |
| country | 0.079  <== F |
| person | 2.919 |
| body_part | 2.672 |

### F's carrier = covariance-difference axes (2nd-moment coding)

Top covariance-difference eigenvalues (eig of Cov_pos - Cov_neg): [-4.669, -1.989, -1.937, -1.063, -0.442, -0.183, -0.093, -0.032].
Negative eigenvalues => F-positive examples have LOWER variance / collapse toward a cluster along these axes (variance/multi-cluster code).

### Linear vs nonlinear probe inside top-m covariance-diff axes

| m (cov-diff axes) | linear AUC | nonlinear AUC | nl-lin |
|---|---|---|---|
| 1 | 0.4929 | 0.7342 | +0.2413 |
| 2 | 0.5062 | 0.8641 | +0.3579 |
| 3 | 0.5093 | 0.9182 | +0.4089 |
| 4 | 0.4856 | 0.9624 | +0.4767 |
| 6 | 0.4987 | 0.9823 | +0.4836 |
| 8 | 0.4999 | 0.9858 | +0.4860 |
| 12 | 0.4721 | 0.9905 | +0.5184 |
| 16 | 0.4869 | 0.9896 | +0.5027 |

- Linear NEVER reaches ceiling in any cov-diff subspace (reaches at m=None); nonlinear reaches nl-ceiling at m=8, and AUC>=0.95 already by m=4.
- **F carrier subspace dimension ~4** (smallest #cov-diff axes with nonlinear AUC>=0.95); |eigvals| of basis axes: [4.669, 1.989, 1.937, 1.063].

### Overlap of F's carrier subspace with the 7 linear read directions

|proj| in [0,1]; ~0 => F's nonlinear subspace is nearly ORTHOGONAL to each linear feature's single direction.

| feature | |proj of its 1-D dir onto F-subspace| | |cos(F_axis1, other_dir1)| |
|---|---|---|
| number | 0.122 | 0.031 |
| question | 0.062 | 0.020 |
| color | 0.048 | 0.005 |
| food | 0.651 | 0.626 |
| sentiment | 0.838 | 0.105 |
| person | 0.084 | 0.015 |
| body_part | 0.096 | 0.045 |

### DECISIVE shared-subspace test: is F superposed on the 7 read dirs?

- F nonlinear AUC using ONLY the 7 linear-readout dims = **0.9894** (full-space ceiling 0.9898).
- F nonlinear AUC after REMOVING those 7 dims (orthogonal complement) = **0.9222**.
- Conclusion: F is **SUPERPOSED / FOLDED ON the shared 7-dim linear-readout subspace**. Country is not stored in a private block; its value is encoded non-linearly (variance/cluster geometry) within the SAME subspace the model uses to linearly read the other features (note the strong overlap with food/sentiment directions above).

## Method notes / caveats

- 'k_greedy->ceiling' = smallest #greedy-deflation (linear) directions whose restricted logistic probe is within 0.005 AUC of the nonlinear ceiling. For the 7 linear features this is 1; for F it is never reached.
- For F the linear probe is at chance at EVERY dim because the class means coincide; the signal is in the COVARIANCE, so F's carrier is found via eigenvectors of (Cov_pos - Cov_neg), not logistic directions.
- 'Carrier dimension ~k' uses an AUC>=0.95 knee; the full nonlinear ceiling (~0.99) needs a few more axes (diminishing returns). The exact integer is soft -- the robust claim is 'a handful (~4-6), not 1'.
- PCA ordering is unsupervised; cov-diff ordering is supervised and is the right basis for a mean-coinciding/variance-coded feature.
- AUC ceilings near 1.0 compress the 7 linear features together; the decisive discriminator is F's 1-dir AND full-64d linear AUC ~0.49 (chance) with a +0.50 nonlinear gap.
