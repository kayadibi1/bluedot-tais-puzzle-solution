# Approach #4 — Quadratic / Kernel decision boundary
**F = `country` (index 5)** — clear low outlier in linear-probe AUC on h2 (Layer L).
- Lowest linear AUC = 0.4869; gap to next-lowest = 0.5086

## AUC table: linear vs degree-2 vs RBF (PCA->12d)
| feature | linAUC | deg2AUC | rbfAUC | d2-lin | rbf-lin | linAcc | d2Acc | rbfAcc |
|---|---|---|---|---|---|---|---|---|
| number | 0.9973 | 0.9954 | 0.9966 | -0.0018 | -0.0007 | 0.9753 | 0.9713 | 0.9727 |
| question | 1.0000 | 1.0000 | 1.0000 | +0.0000 | -0.0000 | 1.0000 | 1.0000 | 0.9980 |
| color | 0.9975 | 0.9969 | 0.9966 | -0.0005 | -0.0009 | 0.9727 | 0.9727 | 0.9660 |
| food | 0.9960 | 0.9940 | 0.9965 | -0.0020 | +0.0004 | 0.9840 | 0.9827 | 0.9833 |
| sentiment | 0.9955 | 0.9962 | 0.9963 | +0.0007 | +0.0008 | 0.9813 | 0.9807 | 0.9807 |
| country **<-F** | 0.4869 | 0.9938 | 0.9943 | +0.5069 | +0.5074 | 0.4707 | 0.9600 | 0.9700 |
| person | 1.0000 | 1.0000 | 0.9999 | -0.0000 | -0.0001 | 0.9993 | 0.9993 | 0.9947 |
| body_part | 0.9986 | 0.9980 | 0.9975 | -0.0005 | -0.0011 | 0.9813 | 0.9780 | 0.9747 |

## Quadratic-form eigen-analysis for F='country' (in 12-d PCA space)
- deg2 test AUC = 0.9934
- Eigenvalues (sorted): [-354.9348, -241.0455, -41.7327, -5.3665, -1.6845, -0.2814, -0.1589, -0.0482, -0.0, 0.0747, 0.1716, 0.4106]
- Raw signature: n_pos=3, n_neg=9, n_zero=0
- Magnitude signature (|lam|>5% max): n_pos=0, n_neg=3
- |min eig|/|max eig| = 864.519; curvature energy pos=0.000 neg=1.000
- **GEOMETRY: NEGATIVE-DEFINITE     => ellipsoid (radial; positive class INSIDE central shell)**

### Shell-vs-XOR confirmation
- ||class-mean diff|| in PCA-12 = 0.0788 (near zero => NO linear/mean signal; explains chance linear AUC)
- mean ||z||: pos=2.645, neg=3.719 (positives concentrated in a tight inner ellipsoid, negatives in outer shell)
- isotropic -||z||^2 AUC = 0.7903; Mahalanobis-to-pos-centroid AUC = 0.9850
- A single concentric (anisotropic) ellipsoid nearly matches the full degree-2 fit. Positives are UNIMODAL (one central blob), not two blobs => **radial/shell, NOT XOR/saddle.**

## Caveats
- PolynomialFeatures+StandardScaler+L2 on the full 64-d h2 yields a collinear, near-degenerate design (h2 is 83% sparse): the recovered Q blows up (|lam|~1e29). The eigen-signature above uses a 12-d PCA reduction with stronger L2 (C=0.1), which is stable; the sign pattern (all-negative) is robust across PCA dims 10-20.
- AUC is reported on the held-out test split. Degree-2 logistic and RBF-SVC give essentially identical recovery (~0.994), as expected for a smooth quadratic boundary.
- The quadratic form is anisotropic: a handful of PCA directions carry the radial signal (variance differs by class) while the bulk are near-flat; isotropic -||z||^2 alone only reaches ~0.79.
