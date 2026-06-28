"""
Approach #3 -- geometry deep-dive for F = country (idx 5).
Goal: characterize HOW country is encoded at h2, and how many LINEAR dirs reach ceiling.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import json
import numpy as np
from numpy.linalg import norm, eigh
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

C = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
ROOT = os.path.dirname(os.path.abspath(__file__))
feats = json.load(open(os.path.join(ROOT, "feature_names.json")))
Xtr = np.load(C + r"\h2_train.npy").astype(np.float64)
Xte = np.load(C + r"\h2_test.npy").astype(np.float64)
Ytr = np.load(C + r"\labels_train.npy"); Yte = np.load(C + r"\labels_test.npy")
kF = 5
ytr, yte = Ytr[:, kF], Yte[:, kF]

scaler = StandardScaler().fit(Xtr)
Xtr_s = scaler.transform(Xtr); Xte_s = scaler.transform(Xte)

print("=== 1) Variance / norm: are country activations 'sparse / gated'? ===")
# how many h2 units are active (post-ReLU > 0) on average, country pos vs neg?
Xtr_raw = np.load(C + r"\h2_train.npy")
act_pos = (Xtr_raw[ytr == 1] > 0).mean()
act_neg = (Xtr_raw[ytr == 0] > 0).mean()
print(f"mean fraction of active (>0) h2 units: country+ {act_pos:.3f}  country- {act_neg:.3f}")
print(f"mean L2 norm of h2: country+ {np.linalg.norm(Xtr_raw[ytr==1],axis=1).mean():.3f} "
      f"country- {np.linalg.norm(Xtr_raw[ytr==0],axis=1).mean():.3f}")

print("\n=== 2) How many LINEAR directions to reach ceiling (top-k PCA subspace logistic) ===")
# Use PCA components on standardized train; fit logistic on first-k PCs, measure test AUC.
pca = PCA(n_components=64).fit(Xtr_s)
Ztr = pca.transform(Xtr_s); Zte = pca.transform(Xte_s)
for kdim in [1, 2, 3, 4, 5, 8, 12, 16, 24, 32, 48, 64]:
    clf = LogisticRegression(max_iter=4000, C=1.0).fit(Ztr[:, :kdim], ytr)
    a = roc_auc_score(yte, clf.decision_function(Zte[:, :kdim]))
    print(f"  PCA top-{kdim:2d} dims -> linear AUC {a:.4f}")

print("\n=== 3) Quadratic probe: is country a QUADRATIC (XOR-like) function of h2? ===")
# Add per-PC squared terms on a modest PCA subspace; linear-in-features = quadratic-in-h2.
for kdim in [8, 16, 24]:
    Ztr_k = Ztr[:, :kdim]; Zte_k = Zte[:, :kdim]
    Qtr = np.column_stack([Ztr_k, Ztr_k**2])
    Qte = np.column_stack([Zte_k, Zte_k**2])
    clf = LogisticRegression(max_iter=5000, C=1.0).fit(Qtr, ytr)
    a = roc_auc_score(yte, clf.decision_function(Qte))
    aL = roc_auc_score(yte, LogisticRegression(max_iter=5000).fit(Ztr_k, ytr).decision_function(Zte_k))
    print(f"  PCA top-{kdim:2d}: linear AUC {aL:.4f}  ->  +squares (quadratic) AUC {a:.4f}  (gain {a-aL:+.4f})")

print("\n=== 4) Full quadratic feature map on small subspace ===")
# full pairwise products on top-6 PCs (21 quad + 6 lin) to detect XOR pairs
from itertools import combinations_with_replacement
kdim = 6
Ztr_k = Ztr[:, :kdim]; Zte_k = Zte[:, :kdim]
def quad_map(Z):
    cols = [Z]
    pairs = []
    for i, j in combinations_with_replacement(range(Z.shape[1]), 2):
        pairs.append((Z[:, i] * Z[:, j])[:, None])
    return np.column_stack([Z] + pairs)
Qtr = quad_map(Ztr_k); Qte = quad_map(Zte_k)
clf = LogisticRegression(max_iter=8000, C=1.0).fit(Qtr, ytr)
a = roc_auc_score(yte, clf.decision_function(Qte))
print(f"  full quadratic map on top-{kdim} PCs: AUC {a:.4f}  (linear-only was "
      f"{roc_auc_score(yte, LogisticRegression(max_iter=5000).fit(Ztr_k,ytr).decision_function(Zte_k)):.4f})")

print("\n=== 5) Cluster structure: do country+ examples form MULTIPLE lobes? ===")
# KMeans on country+ in h2, then check linear separability of each lobe-vs-neg.
Xpos = Xtr_s[ytr == 1]; Xneg = Xtr_s[ytr == 0]
for ncl in [2, 3, 4]:
    km = KMeans(n_clusters=ncl, n_init=10, random_state=0).fit(Xpos)
    # for each cluster, diff-of-means dir vs negatives; measure that single dir's AUC for that lobe
    aucs = []
    for c in range(ncl):
        mu1 = Xpos[km.labels_ == c].mean(0); mu0 = Xneg.mean(0)
        d = mu1 - mu0; d = d / (norm(d) + 1e-12)
        # test: among test positives, which are in this lobe? assign by nearest train centroid
        # simpler: this lobe vs all negatives -> 1D separability on TRAIN positives-of-lobe
        s_pos = Xpos[km.labels_ == c] @ d; s_neg = Xneg @ d
        y = np.r_[np.ones(len(s_pos)), np.zeros(len(s_neg))]
        s = np.r_[s_pos, s_neg]
        aucs.append(roc_auc_score(y, s))
    sizes = [int((km.labels_ == c).sum()) for c in range(ncl)]
    print(f"  k={ncl}: per-lobe (lobe+ vs all neg) 1D AUC = "
          f"{['%.3f'%x for x in aucs]}  sizes={sizes}")

print("\n=== 6) Are the lobes ANTIPODAL? (sign-folding / |projection| structure) ===")
# Fit MLP, take its first-layer to find the country-relevant direction(s); test if |proj| separates.
# Practical proxy: find direction(s) where country+ has BIMODAL distribution straddling country-.
# Use logistic on absolute value of top PCs.
for kdim in [8, 16, 24]:
    Atr = np.abs(Ztr[:, :kdim]); Ate = np.abs(Zte[:, :kdim])
    clf = LogisticRegression(max_iter=5000).fit(Atr, ytr)
    a = roc_auc_score(yte, clf.decision_function(Ate))
    print(f"  logistic on |PCA top-{kdim}| -> AUC {a:.4f}")

print("\n=== 7) Reference: MLP ceiling + how shallow nonlinearity suffices ===")
for hl in [(8,), (16,), (32,), (64, 32)]:
    clf = MLPClassifier(hidden_layer_sizes=hl, max_iter=800, alpha=1e-4,
                        random_state=0, early_stopping=True, n_iter_no_change=25).fit(Xtr_s, ytr)
    a = roc_auc_score(yte, clf.predict_proba(Xte_s)[:, 1])
    print(f"  MLP hidden={hl}: AUC {a:.4f}")
print("DONE")
