"""
Confirm the FOLDED / antipodal structure of country at h2.
Hypothesis: there exists a low-dim subspace where country- clusters near the origin
and country+ lies far from origin in BOTH directions (so |.| / radius separates classes,
but no single signed direction does).
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import json, numpy as np
from numpy.linalg import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

C = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
ROOT = os.path.dirname(os.path.abspath(__file__))
Xtr = np.load(C + r"\h2_train.npy").astype(np.float64)
Xte = np.load(C + r"\h2_test.npy").astype(np.float64)
Ytr = np.load(C + r"\labels_train.npy"); Yte = np.load(C + r"\labels_test.npy")
ytr, yte = Ytr[:, 5], Yte[:, 5]
sc = StandardScaler().fit(Xtr); Xtr_s = sc.transform(Xtr); Xte_s = sc.transform(Xte)

# RADIUS test: does squared distance-from-(neg-centroid) separate classes? (pure isotropic fold)
mu_neg = Xtr_s[ytr == 0].mean(0)
r2_tr = ((Xtr_s - mu_neg) ** 2).sum(1); r2_te = ((Xte_s - mu_neg) ** 2).sum(1)
print(f"squared-radius-from-neg-centroid, single feature -> AUC {roc_auc_score(yte, r2_te):.4f}")
print(f"  mean r2: country+ {r2_tr[ytr==1].mean():.2f}  country- {r2_tr[ytr==0].mean():.2f}")

# variance per class (the fold signature: pos has MUCH larger spread)
print(f"  total var: country+ {Xtr_s[ytr==1].var(0).sum():.2f}  country- {Xtr_s[ytr==0].var(0).sum():.2f}")

# Best single SIGNED direction (LDA) AUC and its |.|-folded AUC
lda = LinearDiscriminantAnalysis(n_components=1).fit(Xtr_s, ytr)
ax_tr = lda.transform(Xtr_s)[:, 0]; ax_te = lda.transform(Xte_s)[:, 0]
print(f"\nLDA signed axis AUC {roc_auc_score(yte, ax_te):.4f}  |.|-folded AUC "
      f"{roc_auc_score(yte, np.abs(ax_te - np.median(ax_tr[ytr==0]))):.4f}")

# How many quadratic directions: eigen-decomp of (Cov_pos - Cov_neg) — top eigenvectors are
# the discriminative QUADRATIC axes. Build features from projections onto top-m, squared.
Cp = np.cov(Xtr_s[ytr == 1], rowvar=False)
Cn = np.cov(Xtr_s[ytr == 0], rowvar=False)
ev, V = np.linalg.eigh(Cp - Cn)            # ascending
order = np.argsort(-np.abs(ev))            # by |eigenvalue|
print("\nQuadratic-axis ladder (logistic on squared projections onto top-m diff-cov eigvecs):")
for m in [1, 2, 3, 4, 6, 8, 12, 16]:
    idx = order[:m]
    Ptr = (Xtr_s @ V[:, idx]) ** 2
    Pte = (Xte_s @ V[:, idx]) ** 2
    clf = LogisticRegression(max_iter=4000).fit(Ptr, ytr)
    a = roc_auc_score(yte, clf.decision_function(Pte))
    print(f"  m={m:2d} squared-axes -> AUC {a:.4f}")
print("DONE")
