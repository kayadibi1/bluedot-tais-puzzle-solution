"""
Approach #3: THE "SINGLE DIRECTION" TEST
=========================================
README claim: at Layer L (post-ReLU hidden 2, h2, 64-d), 7 of 8 features are read
by a SINGLE DIRECTION; ONE feature F is not.

For each feature we quantify three regimes on h2:
  (1) BEST 1-D direction  : Fisher LDA / whitened diff-of-means -> 1-D ROC-AUC + bal-acc.
  (2) FULL LINEAR         : 64-d LogisticRegression AUC.
  (3) NONLINEAR ceiling   : small MLPClassifier AUC.

Diagnostics:
  - 7 linear features:  1D ~= fullLinear ~= nonlinear ~= ceiling.
  - F: does (1) collapse?
       * "linear-but-multidirectional":  1D LOW, fullLinear HIGH.
       * "genuinely non-linear":          1D LOW AND fullLinear LOW, only nonlinear recovers.
  - For F: how many linear directions needed to reach ceiling (incremental LDA-on-residual /
    top-k logistic subspace).
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import json
import numpy as np
from numpy.linalg import pinv, eigh, norm
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

C = r"C:\Users\Sidar\Desktop\puzzle\bluedot-tais-puzzle\cache"
ROOT = r"C:\Users\Sidar\Desktop\puzzle\bluedot-tais-puzzle"
feats = json.load(open(os.path.join(ROOT, "feature_names.json")))

Xtr = np.load(C + r"\h2_train.npy").astype(np.float64)
Xte = np.load(C + r"\h2_test.npy").astype(np.float64)
Ytr = np.load(C + r"\labels_train.npy")
Yte = np.load(C + r"\labels_test.npy")

N, D = Xtr.shape
print(f"h2 train {Xtr.shape}, test {Xte.shape}; {len(feats)} features")

# --- whitening on TRAIN (for fair 1-D LDA-style direction) ---
scaler = StandardScaler().fit(Xtr)
Xtr_s = scaler.transform(Xtr)
Xte_s = scaler.transform(Xte)

# Whitening transform (ZCA-ish via covariance eigendecomp on standardized train)
cov = np.cov(Xtr_s, rowvar=False)
cov += 1e-6 * np.eye(D)  # ridge for invertibility
evals, evecs = eigh(cov)
evals = np.clip(evals, 1e-8, None)
W = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T  # whitening matrix
Xtr_w = Xtr_s @ W
Xte_w = Xte_s @ W


def best_threshold_balacc(scores, y):
    """Sweep thresholds, return best balanced accuracy."""
    order = np.argsort(scores)
    s_sorted = scores[order]
    # candidate thresholds = midpoints
    cand = (s_sorted[:-1] + s_sorted[1:]) / 2.0
    # subsample candidates if huge
    if len(cand) > 2000:
        cand = cand[np.linspace(0, len(cand) - 1, 2000).astype(int)]
    best = 0.0
    for t in cand:
        ba = balanced_accuracy_score(y, (scores > t).astype(int))
        if ba > best:
            best = ba
    return best


def one_d_direction_auc(Xtr_w, ytr, Xte_w, yte):
    """Diff-of-class-means direction in WHITENED space == Fisher LDA direction.
    Project test onto unit direction, measure 1-D AUC + best-threshold bal-acc."""
    mu1 = Xtr_w[ytr == 1].mean(0)
    mu0 = Xtr_w[ytr == 0].mean(0)
    d = mu1 - mu0
    d = d / (norm(d) + 1e-12)
    s_te = Xte_w @ d
    auc = roc_auc_score(yte, s_te)
    ba = best_threshold_balacc(s_te, yte)
    return auc, ba, d


def full_linear_auc(Xtr, ytr, Xte, yte):
    clf = LogisticRegression(max_iter=3000, C=1.0)
    clf.fit(Xtr, ytr)
    p = clf.decision_function(Xte)
    return roc_auc_score(yte, p), clf


def nonlinear_auc(Xtr, ytr, Xte, yte):
    clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=600,
                        alpha=1e-4, random_state=0, early_stopping=True,
                        n_iter_no_change=20)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    return roc_auc_score(yte, p)


# ----- main table -----
rows = []
one_d_dirs = {}
for k, name in enumerate(feats):
    ytr, yte = Ytr[:, k], Yte[:, k]
    auc1d, ba1d, d = one_d_direction_auc(Xtr_w, ytr, Xte_w, yte)
    aucfull, _ = full_linear_auc(Xtr_s, ytr, Xte_s, yte)
    aucnl = nonlinear_auc(Xtr_s, ytr, Xte_s, yte)
    one_d_dirs[k] = d
    rows.append(dict(idx=k, name=name, auc1d=auc1d, ba1d=ba1d,
                     aucfull=aucfull, aucnl=aucnl,
                     gap_full_1d=aucfull - auc1d,
                     gap_nl_full=aucnl - aucfull))
    print(f"[{k}] {name:10s}  1D={auc1d:.4f} (ba {ba1d:.3f})  full={aucfull:.4f}  nl={aucnl:.4f}"
          f"  (full-1D)={aucfull-auc1d:+.4f}  (nl-full)={aucnl-aucfull:+.4f}")

# Identify F = feature with the largest collapse of the 1-D direction.
# Primary signal: lowest 1D AUC and/or largest (full-1D) gap.
worst = min(rows, key=lambda r: r["auc1d"])
print(f"\nLowest 1-D AUC feature: [{worst['idx']}] {worst['name']}  (1D AUC={worst['auc1d']:.4f})")
big_gap = max(rows, key=lambda r: r["gap_full_1d"])
print(f"Largest (full-1D) gap:  [{big_gap['idx']}] {big_gap['name']}  gap={big_gap['gap_full_1d']:+.4f}")

F = worst  # candidate

# ===== Deep-dive on F: how many LINEAR directions needed? =====
kF = F["idx"]
ytr, yte = Ytr[:, kF], Yte[:, kF]
print(f"\n==== DEEP DIVE on F=[{kF}] {F['name']} ====")

# (A) Greedy LDA-on-residual: add directions one at a time in whitened space,
#     each = diff-of-means of the residual after projecting out previous dirs.
def greedy_lda_subspace(Xtr_w, ytr, Xte_w, yte, max_k=8):
    Rtr = Xtr_w.copy()
    Rte = Xte_w.copy()
    dirs = []
    aucs = []
    for j in range(max_k):
        mu1 = Rtr[ytr == 1].mean(0)
        mu0 = Rtr[ytr == 0].mean(0)
        d = mu1 - mu0
        nd = norm(d)
        if nd < 1e-9:
            break
        d = d / nd
        dirs.append(d)
        # logistic regression on the accumulated subspace projections (test AUC)
        Ptr = Xtr_w @ np.array(dirs).T  # (N, j+1)
        Pte = Xte_w @ np.array(dirs).T
        clf = LogisticRegression(max_iter=2000)
        clf.fit(Ptr, ytr)
        a = roc_auc_score(yte, clf.decision_function(Pte))
        aucs.append(a)
        # deflate residual by removing component along d
        Rtr = Rtr - np.outer(Rtr @ d, d)
        Rte = Rte - np.outer(Rte @ d, d)
    return aucs

greedy = greedy_lda_subspace(Xtr_w, ytr, Xte_w, yte, max_k=8)
print("Greedy LDA-residual subspace test AUC by #dirs:",
      ", ".join(f"{i+1}:{a:.4f}" for i, a in enumerate(greedy)))

# (B) Sanity: a strong linear feature for comparison (the best 1-D feature)
best_lin = max(rows, key=lambda r: r["auc1d"])
print(f"(for contrast) best linear feature = [{best_lin['idx']}] {best_lin['name']} "
      f"1D AUC={best_lin['auc1d']:.4f}")

# (C) Folded / two-cluster geometry test for F:
#     If F is XOR-like / multi-cluster (e.g. positives form 2 lobes), the class means
#     can nearly coincide while a 2-D quadratic separates them. Test: does adding the
#     squared-projection feature help a linear classifier (i.e. quadratic in 1-D)?
mu1 = Xtr_w[ytr == 1].mean(0); mu0 = Xtr_w[ytr == 0].mean(0)
dF = mu1 - mu0; dF = dF / (norm(dF) + 1e-12)
# top LDA dir from logistic instead (more informative axis)
clf_full = LogisticRegression(max_iter=3000).fit(Xtr_s, ytr)
wlin = clf_full.coef_[0]; wlin = wlin / norm(wlin)
proj_tr = Xtr_s @ wlin; proj_te = Xte_s @ wlin
# quadratic features of the single logistic axis
Qtr = np.column_stack([proj_tr, proj_tr**2])
Qte = np.column_stack([proj_te, proj_te**2])
clfq = LogisticRegression(max_iter=3000).fit(Qtr, ytr)
auc_quad1 = roc_auc_score(yte, clfq.decision_function(Qte))
auc_lin1 = roc_auc_score(yte, proj_te)
print(f"1-D logistic-axis: linear AUC={auc_lin1:.4f}  vs  +square term AUC={auc_quad1:.4f} "
      f"(delta={auc_quad1-auc_lin1:+.4f})")

# (D) Means overlap diagnostic: |mu1-mu0| / pooled std along best logistic axis
s1 = proj_tr[ytr == 1]; s0 = proj_tr[ytr == 0]
sep = abs(s1.mean() - s0.mean()) / (0.5 * (s1.std() + s0.std()) + 1e-12)
print(f"Class-mean separation along logistic axis (d'): {sep:.3f}")
print(f"  pos mean={s1.mean():+.3f} std={s1.std():.3f} | neg mean={s0.mean():+.3f} std={s0.std():.3f}")

# Save results
results = dict(
    rows=rows,
    F=dict(idx=kF, name=F["name"]),
    greedy_lda_auc=greedy,
    quad_test=dict(lin1d=auc_lin1, quad1d=auc_quad1, dprime_logistic=sep),
    best_linear_feature=dict(idx=best_lin["idx"], name=best_lin["name"], auc1d=best_lin["auc1d"]),
)
np.save(os.path.join(ROOT, "results", "approach3_raw.npy"), results, allow_pickle=True)
print("\nDONE.")
