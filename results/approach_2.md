# Approach #2 - Linear vs Non-linear Probe Gap at Layer L (h2)

Layer L = post-ReLU of hidden layer 2, 64-dim. Train N=7000, Test N=1500.

Linear probe = LogisticRegressionCV (StandardScaler, C tuned over 1e-3..1e3, 5-fold ROC-AUC).

Non-linear = best of {MLP(64), MLP(128,64), RBF-SVC, GradientBoosting}. Scale-sensitive probes share the SAME StandardScaler as the linear probe; GBoost uses raw features (scale-invariant). Metric: TEST ROC-AUC.


## Ranked table (by gap, descending)

| rank | idx | feature | linear_AUC | linear_bAcc | nonlinear_AUC | nonlinear_bAcc | best_NL | GAP |
|---|---|---|---|---|---|---|---|---|
| 1 | 5 | country  <-- F | 0.5088 | 0.4965 | 0.9933 | 0.9608 | SVC-RBF | +0.4844 |
| 2 | 4 | sentiment | 0.9953 | 0.9804 | 0.9973 | 0.9811 | SVC-RBF | +0.0020 |
| 3 | 3 | food | 0.9950 | 0.9827 | 0.9963 | 0.9853 | SVC-RBF | +0.0013 |
| 4 | 0 | number | 0.9968 | 0.9766 | 0.9975 | 0.9772 | SVC-RBF | +0.0007 |
| 5 | 2 | color | 0.9968 | 0.9679 | 0.9973 | 0.9756 | SVC-RBF | +0.0005 |
| 6 | 6 | person | 1.0000 | 0.9980 | 1.0000 | 0.9993 | SVC-RBF | +0.0000 |
| 7 | 1 | question | 1.0000 | 1.0000 | 1.0000 | 1.0000 | MLP(64) | +0.0000 |
| 8 | 7 | body_part | 0.9985 | 0.9786 | 0.9984 | 0.9786 | MLP(128,64) | -0.0001 |

## Per-feature non-linear breakdown (all NL probe AUCs)

| feature | linear_AUC | MLP(64) | MLP(128,64) | SVC-RBF | GBoost |
|---|---|---|---|---|---|
| number | 0.9968 | 0.9970 | 0.9968 | 0.9975 | 0.9967 |
| question | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| color | 0.9968 | 0.9970 | 0.9967 | 0.9973 | 0.9958 |
| food | 0.9950 | 0.9948 | 0.9942 | 0.9963 | 0.9943 |
| sentiment | 0.9953 | 0.9947 | 0.9951 | 0.9973 | 0.9949 |
| country | 0.5088 | 0.9867 | 0.9878 | 0.9933 | 0.9824 |
| person | 1.0000 | 0.9998 | 1.0000 | 1.0000 | 0.9998 |
| body_part | 0.9985 | 0.9977 | 0.9984 | 0.9981 | 0.9978 |

## Conclusion

- **F = `country` (index 5)** is the non-linear feature.
- Decisive gap: F gap = **+0.4844** (linear AUC 0.5088 -> nonlinear AUC 0.9933, via SVC-RBF).
- Next-largest gap: `sentiment` at +0.0020. Separation F-vs-rest = **+0.4824** (237.2x).
- The other 7 features: gaps in [-0.0001, +0.0020], mean +0.0006 (~0 = both probes at ceiling).
- F's linear AUC (0.5088) is the only one well below ceiling, confirming linear-unreadability at L while remaining nonlinearly recoverable (nonlinear AUC 0.9933).

## Linear-probe robustness (rule out probe-fitting artifact)

`country` stays at chance across every linear setup tried (TEST ROC-AUC):

| setup | AUC |
|---|---|
| LogReg scaled C=0.01 | 0.4784 |
| LogReg scaled C=0.1  | 0.4755 |
| LogReg scaled C=1    | 0.4902 |
| LogReg scaled C=10   | 0.5066 |
| LogReg scaled C=100  | 0.5102 |
| LogReg scaled C=1000 | 0.5104 |
| LogReg RAW features  | 0.4726 |
| LDA                  | 0.5161 |

No linear model exceeds ~0.52. The +0.4844 gap is a genuine non-linearity, not under-regularization.

## Geometry of F (`country`)

Class-centroid separation in h2, per feature (the diagnostic of linear separability):

| feature | mean-diff norm | mean-diff / within-class std | best-linear-direction AUC |
|---|---|---|---|
| number    | 1.1168 | 17.67 | 0.8509 |
| question  | 0.5858 |  8.77 | 0.7128 |
| color     | 0.8216 | 12.53 | 0.7727 |
| food      | 1.3008 | 20.95 | 0.8764 |
| sentiment | 0.6166 |  9.23 | 0.8884 |
| **country** | **0.0134** | **0.20** | **0.4919** |
| person    | 1.2349 | 20.12 | 0.8565 |
| body_part | 0.8769 | 13.79 | 0.7974 |

**Signature:** for `country` the positive and negative class means are essentially
COINCIDENT (mean-diff norm 0.0134, ~85x smaller than any other feature; ratio to
within-class scale 0.20 vs 8.8-21 for the rest). A projection onto the mean-difference
direction gives AUC 0.49 = chance. Coincident class means with non-linear recoverability
is the classic geometry of a **multimodal / XOR-like (entangled)** encoding: the
country=1 points are NOT a single half-space but several disjoint clusters (plausibly one
per country value) interleaved with country=0 points, so the two classes have the same
center of mass and no linear hyperplane separates them, yet an RBF/MLP carving local
regions recovers them at AUC 0.99. (The downstream Linear64->8 head reads F non-linearly
because it sits AFTER two more ReLU layers, h3/logits, which untangle it.)