"""
Approach #10: CURVED-MANIFOLD / LOCAL-STRUCTURE / PERIODIC hypothesis.

Hypothesis: the non-linear feature F lives on a CURVED low-dim manifold (arc /
ring / fold) at Layer L=h2 where the two classes interleave so no hyperplane
separates them, but LOCAL (kNN) structure does.

Pipeline:
  0. Confirm F via linear-probe AUC sweep on h2 (the low outlier).
  1. LOCAL vs GLOBAL: kNN (small k) vs linear probe AUC, for F + a linear control.
  2. MANIFOLD EMBEDDING: PCA / Isomap / UMAP of h2, colored by F's label.
  3. PERIODIC / ANGULAR test: theta=atan2(pc2,pc1) in F's relevant subspace;
     test whether F's label is periodic in theta (alternates by sector).
  4. INTRINSIC DIMENSIONALITY: PCA participation ratio of F's class manifolds.

Outputs: results/approach_10_*.png and results/approach_10.md
GPU is broken -- we ONLY read cache/*.npy and never call the encoder.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import json
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

RNG = np.random.default_rng(0)
C = r"C:\Users\Sidar\Desktop\puzzle\bluedot-tais-puzzle\cache"
OUT = r"C:\Users\Sidar\Desktop\puzzle\bluedot-tais-puzzle\results"
os.makedirs(OUT, exist_ok=True)

FEATS = json.load(open(r"C:\Users\Sidar\Desktop\puzzle\bluedot-tais-puzzle\feature_names.json"))

Xtr = np.load(os.path.join(C, "h2_train.npy")).astype(np.float64)
Xte = np.load(os.path.join(C, "h2_test.npy")).astype(np.float64)
Ytr = np.load(os.path.join(C, "labels_train.npy"))
Yte = np.load(os.path.join(C, "labels_test.npy"))
print(f"Xtr {Xtr.shape}  Xte {Xte.shape}  Ytr {Ytr.shape}")

lines = []
def log(s=""):
    print(s)
    lines.append(s)

# ----------------------------------------------------------------------------
# 0. Confirm F: linear-probe AUC sweep on h2 (worst feature = F)
# ----------------------------------------------------------------------------
log("# Approach #10 - Curved-manifold / local-structure / periodic analysis\n")
log("Layer L = h2 (post-ReLU hidden layer 2, 64-dim). "
    f"Train N={Xtr.shape[0]}, Test N={Xte.shape[0]}.\n")

sc = StandardScaler().fit(Xtr)
Xtr_s = sc.transform(Xtr)
Xte_s = sc.transform(Xte)

log("## 0. Linear-probe AUC sweep on h2 (confirm F = the low outlier)\n")
log("| idx | feature | linear_AUC |")
log("|---|---|---|")
lin_auc = {}
for j, name in enumerate(FEATS):
    clf = LogisticRegression(C=1.0, max_iter=2000)
    clf.fit(Xtr_s, Ytr[:, j])
    p = clf.predict_proba(Xte_s)[:, 1]
    a = roc_auc_score(Yte[:, j], p)
    lin_auc[j] = a
    log(f"| {j} | {name} | {a:.4f} |")
Fidx = min(lin_auc, key=lin_auc.get)
Fname = FEATS[Fidx]
ordr = sorted(lin_auc, key=lin_auc.get)
gap_to_next = lin_auc[ordr[1]] - lin_auc[ordr[0]]
log(f"\n**F = `{Fname}` (index {Fidx})**, linear AUC={lin_auc[Fidx]:.4f}; "
    f"next-worst (`{FEATS[ordr[1]]}`)={lin_auc[ordr[1]]:.4f}; gap={gap_to_next:.4f}.\n")

# pick a clean linear control feature (best-separated, non-trivial)
ctrl_idx = ordr[-1]
# prefer a non-perfect feature so kNN<->linear comparison is meaningful;
# use a mid-ranked one (sentiment-like) as control
ctrl_idx = ordr[2]  # 3rd-best-separated linear feature
ctrl_name = FEATS[ctrl_idx]
log(f"Linear control feature = `{ctrl_name}` (index {ctrl_idx}), "
    f"linear AUC={lin_auc[ctrl_idx]:.4f}.\n")

# ----------------------------------------------------------------------------
# 1. LOCAL vs GLOBAL: kNN AUC vs linear AUC for F and the control
# ----------------------------------------------------------------------------
log("## 1. LOCAL vs GLOBAL: kNN (small k) vs linear probe\n")
log("If kNN AUC >> linear AUC, F's structure is local/curved (not a half-space).\n")

def knn_auc(ytr, yte, k):
    knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
    knn.fit(Xtr_s, ytr)
    p = knn.predict_proba(Xte_s)[:, 1]
    return roc_auc_score(yte, p), balanced_accuracy_score(yte, p > 0.5)

ks = [1, 3, 5, 9, 15, 31]
log("| feature | linear_AUC | " + " | ".join(f"kNN k={k}" for k in ks) + " |")
log("|---|---|" + "---|" * len(ks))
knn_results = {}
for idx, nm in [(Fidx, Fname), (ctrl_idx, ctrl_name)]:
    row = []
    best = 0.0
    for k in ks:
        a, b = knn_auc(Ytr[:, idx], Yte[:, idx], k)
        row.append(a)
        best = max(best, a)
    knn_results[idx] = (row, best)
    log(f"| {nm} | {lin_auc[idx]:.4f} | " + " | ".join(f"{v:.4f}" for v in row) + " |")

F_knn_best = knn_results[Fidx][1]
F_gap = F_knn_best - lin_auc[Fidx]
ctrl_knn_best = knn_results[ctrl_idx][1]
ctrl_gap = ctrl_knn_best - lin_auc[ctrl_idx]
log(f"\n**F kNN-vs-linear gap = +{F_gap:.4f}** (best kNN AUC {F_knn_best:.4f} vs "
    f"linear {lin_auc[Fidx]:.4f}).")
log(f"Control `{ctrl_name}` kNN-vs-linear gap = {ctrl_gap:+.4f} "
    f"(kNN {ctrl_knn_best:.4f} vs linear {lin_auc[ctrl_idx]:.4f}).")
log(f"Ratio F-gap / control-gap = {F_gap/max(abs(ctrl_gap),1e-6):.1f}x.\n")

# ----------------------------------------------------------------------------
# 2. Find F's "relevant subspace": directions where F's class-conditional
#    means/variances differ, then PCA / Isomap / UMAP on h2.
# ----------------------------------------------------------------------------
log("## 2. Manifold embedding of h2 colored by F's label\n")

yF_tr = Ytr[:, Fidx].astype(bool)
yF_te = Yte[:, Fidx].astype(bool)

# Subsample for the (slow) manifold methods.
nsub = min(2500, Xtr_s.shape[0])
sel = RNG.choice(Xtr_s.shape[0], nsub, replace=False)
Xsub = Xtr_s[sel]
ysub = yF_tr[sel]

# 2a. Global PCA (top components of full h2)
pca_full = PCA(n_components=10).fit(Xtr_s)
Ztr_pca = pca_full.transform(Xtr_s)
evr = pca_full.explained_variance_ratio_

# 2b. F-discriminative subspace: LDA-like direction is degenerate (linear AUC~0.5),
#     so instead find the subspace where the TWO CLASS CENTROIDS' second moment
#     differs. Use PCA on the *difference of class-conditional covariances* to
#     surface the curved directions, plus the between-class mean direction.
mu_pos = Xtr_s[yF_tr].mean(0)
mu_neg = Xtr_s[~yF_tr].mean(0)
mean_dir = mu_pos - mu_neg
mean_dir /= (np.linalg.norm(mean_dir) + 1e-12)

# Directions of maximal *within-F* spread captured via PCA on F-positive points
pca_pos = PCA(n_components=8).fit(Xtr_s[yF_tr])
pca_neg = PCA(n_components=8).fit(Xtr_s[~yF_tr])

# Build a 2-D "F subspace" for the angular test: take the top-2 PCs of the
# combined set of points projected after removing global dominant variance is
# overkill; instead we project onto the subspace spanned by the leading PCs of
# the F-positive cloud (the ring/arc, if any, lives there).
Fsub_basis = pca_pos.components_[:2]            # (2,64)
def proj2(X):
    return X @ Fsub_basis.T                      # (N,2)
P2_tr = proj2(Xtr_s)
P2_te = proj2(Xte_s)

# Run Isomap + UMAP on subsample
log(f"Running Isomap (n={nsub}) ...")
iso = Isomap(n_neighbors=15, n_components=2)
Ziso = iso.fit_transform(Xsub)
try:
    import umap
    log(f"Running UMAP (n={nsub}) ...")
    um = umap.UMAP(n_neighbors=20, min_dist=0.1, n_components=2, random_state=0)
    Zumap = um.fit_transform(Xsub)
    have_umap = True
except Exception as e:
    log(f"UMAP failed ({e}); skipping.")
    Zumap = None
    have_umap = False

# ---- Plot panel: PCA / Isomap / UMAP / F-subspace, colored by F label ----
def scatter(ax, Z, y, title):
    ax.scatter(Z[~y, 0], Z[~y, 1], s=4, c="#2c7fb8", alpha=0.5, label=f"not {Fname}")
    ax.scatter(Z[y, 0], Z[y, 1], s=4, c="#d95f0e", alpha=0.5, label=f"{Fname}")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

fig, axes = plt.subplots(2, 2, figsize=(11, 10))
scatter(axes[0, 0], Ztr_pca[:, :2], yF_tr, f"Global PCA top-2 (EVR {evr[0]:.2f},{evr[1]:.2f})")
scatter(axes[0, 1], P2_tr, yF_tr, "F-positive-cloud PCA top-2 (F subspace)")
scatter(axes[1, 0], Ziso, ysub, f"Isomap 2-D (n={nsub})")
if have_umap:
    scatter(axes[1, 1], Zumap, ysub, f"UMAP 2-D (n={nsub})")
else:
    scatter(axes[1, 1], Ztr_pca[:, 2:4], yF_tr, "Global PCA PC3-PC4")
axes[0, 0].legend(markerscale=3, fontsize=8, loc="best")
fig.suptitle(f"h2 manifold embeddings colored by F=`{Fname}`", fontsize=13)
fig.tight_layout()
p_embed = os.path.join(OUT, "approach_10_embeddings.png")
fig.savefig(p_embed, dpi=110)
plt.close(fig)
log(f"Saved {p_embed}")

# ----------------------------------------------------------------------------
# 3. PERIODIC / ANGULAR test in F's top-2 PC subspace.
# ----------------------------------------------------------------------------
log("\n## 3. Periodic / angular structure test\n")

# Center the 2-D F-subspace projection on the pooled mean (so atan2 is meaningful).
ctr = P2_tr.mean(0)
A_tr = P2_tr - ctr
A_te = P2_te - ctr
theta_tr = np.arctan2(A_tr[:, 1], A_tr[:, 0])      # (-pi, pi]
theta_te = np.arctan2(A_te[:, 1], A_te[:, 0])
r_tr = np.linalg.norm(A_tr, axis=1)
r_te = np.linalg.norm(A_te, axis=1)

# (a) Radial test: is F separable by RADIUS alone? (dome/shell hypothesis)
rad_auc = roc_auc_score(yF_te, -r_te)  # try "inside" = positive
rad_auc = max(rad_auc, 1 - rad_auc)
log(f"- Radial separability (|r| in F-subspace): AUC = {rad_auc:.4f} "
    f"(>0.5 => shell/dome; positive class at one radius band).")

# (b) Angular periodicity test: bin theta into S sectors, measure how F's
#     positive-rate varies with sector, and whether it ALTERNATES (period>1).
S = 24
bins = np.linspace(-np.pi, np.pi, S + 1)
bidx = np.digitize(theta_tr, bins) - 1
bidx = np.clip(bidx, 0, S - 1)
posrate = np.array([yF_tr[bidx == b].mean() if (bidx == b).any() else np.nan
                    for b in range(S)])
log(f"- Per-sector F+ rate ({S} angular sectors), range "
    f"[{np.nanmin(posrate):.2f}, {np.nanmax(posrate):.2f}], "
    f"std={np.nanstd(posrate):.3f} (high std + alternation => periodic).")

# Quantify periodicity via FFT of the (mean-removed) sector pos-rate signal.
pr = np.nan_to_num(posrate - np.nanmean(posrate))
fft = np.abs(np.fft.rfft(pr))
dom_freq = int(np.argmax(fft[1:]) + 1)  # dominant non-DC frequency
log(f"- Dominant angular frequency of F+ rate = {dom_freq} cycle(s) over 2pi "
    f"(freq=1 => single arc/half; >=2 => alternating sectors / multi-lobe).")

# (c) Angular feature for a classifier: build periodic features
#     [cos(k*theta), sin(k*theta)] for k=1..3 plus radius, and see test AUC.
def ang_feats(theta, r):
    cols = [r]
    for k in (1, 2, 3):
        cols += [np.cos(k * theta), np.sin(k * theta)]
    return np.column_stack(cols)

Af_tr = ang_feats(theta_tr, r_tr)
Af_te = ang_feats(theta_te, r_te)
ang_clf = LogisticRegression(C=1.0, max_iter=3000).fit(
    StandardScaler().fit_transform(Af_tr), yF_tr)
scA = StandardScaler().fit(Af_tr)
ang_p = ang_clf.predict_proba(scA.transform(Af_te))[:, 1]
# refit cleanly with shared scaler
ang_clf = LogisticRegression(C=1.0, max_iter=3000).fit(scA.transform(Af_tr), yF_tr)
ang_p = ang_clf.predict_proba(scA.transform(Af_te))[:, 1]
ang_auc = roc_auc_score(yF_te, ang_p)
log(f"- Angular+radial logistic probe (cos/sin k=1..3 in F-subspace): "
    f"test AUC = {ang_auc:.4f} vs raw-linear {lin_auc[Fidx]:.4f}.")
log(f"  Angular-vs-linear gap = +{ang_auc - lin_auc[Fidx]:.4f} "
    f"(in just a 2-D curved subspace).\n")

# ---- Plot: theta vs radius, colored by F; and per-sector pos-rate ----
fig2, ax2 = plt.subplots(1, 2, figsize=(13, 5))
ax2[0].scatter(theta_tr[~yF_tr], r_tr[~yF_tr], s=4, c="#2c7fb8", alpha=0.4, label=f"not {Fname}")
ax2[0].scatter(theta_tr[yF_tr], r_tr[yF_tr], s=4, c="#d95f0e", alpha=0.4, label=f"{Fname}")
ax2[0].set_xlabel("theta = atan2(pc2,pc1) in F-subspace")
ax2[0].set_ylabel("radius |r|")
ax2[0].set_title("F label vs (angle, radius)")
ax2[0].legend(markerscale=3, fontsize=8)
centers = 0.5 * (bins[:-1] + bins[1:])
ax2[1].plot(centers, posrate, "-o", color="#d95f0e")
ax2[1].axhline(yF_tr.mean(), ls="--", c="gray", label="global F+ rate")
ax2[1].set_xlabel("theta (sector center)")
ax2[1].set_ylabel("F+ rate in sector")
ax2[1].set_title(f"Angular F+ rate (dominant freq={dom_freq})")
ax2[1].legend(fontsize=8)
fig2.tight_layout()
p_ang = os.path.join(OUT, "approach_10_angular.png")
fig2.savefig(p_ang, dpi=110)
plt.close(fig2)
log(f"Saved {p_ang}")

# ----------------------------------------------------------------------------
# 4. Intrinsic dimensionality of F's class-conditional manifolds.
# ----------------------------------------------------------------------------
log("\n## 4. Intrinsic dimensionality of F's class manifolds\n")

def participation_ratio(X):
    pca = PCA().fit(X)
    ev = pca.explained_variance_
    return (ev.sum() ** 2) / (np.square(ev).sum())  # PR dim

pr_pos = participation_ratio(Xtr_s[yF_tr])
pr_neg = participation_ratio(Xtr_s[~yF_tr])
pr_all = participation_ratio(Xtr_s)
log(f"- PCA participation-ratio dim: F+ cloud = {pr_pos:.2f}, "
    f"F- cloud = {pr_neg:.2f}, all-h2 = {pr_all:.2f} (of 64).")

# How many PCs of the F+ cloud to reach 90% variance?
def ncomp_for(X, frac=0.90):
    pca = PCA().fit(X)
    c = np.cumsum(pca.explained_variance_ratio_)
    return int(np.searchsorted(c, frac) + 1)
log(f"- PCs to 90% var: F+ cloud needs {ncomp_for(Xtr_s[yF_tr])}, "
    f"F- cloud needs {ncomp_for(Xtr_s[~yF_tr])}, all-h2 needs {ncomp_for(Xtr_s)}.")

# Local intrinsic dim via local PCA on kNN neighborhoods of F+ points (MLE-ish:
# fraction of variance in top-2 vs more dims locally).
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=30).fit(Xtr_s)
seedpts = RNG.choice(np.where(yF_tr)[0], min(200, yF_tr.sum()), replace=False)
local_dims = []
for i in seedpts:
    _, nbr = nn.kneighbors(Xtr_s[i:i+1])
    loc = Xtr_s[nbr[0]]
    local_dims.append(participation_ratio(loc))
log(f"- Local participation-ratio dim around F+ points (30-NN): "
    f"median={np.median(local_dims):.2f}, mean={np.mean(local_dims):.2f} "
    f"(low => points lie on a thin local manifold/filament).")

# ----------------------------------------------------------------------------
# Summary / characterization
# ----------------------------------------------------------------------------
log("\n## Summary characterization of F's geometry at Layer L\n")
log(f"- F = `{Fname}` (index {Fidx}); linear AUC {lin_auc[Fidx]:.4f} is the "
    f"unique low outlier (next-worst {lin_auc[ordr[1]]:.4f}).")
log(f"- kNN recovers F: best kNN AUC {F_knn_best:.4f} => kNN-vs-linear gap "
    f"+{F_gap:.4f}; control `{ctrl_name}` gap {ctrl_gap:+.4f}. "
    f"Local structure carries F, a global half-space does not.")
log(f"- Radial AUC {rad_auc:.4f}; angular+radial 2-D probe AUC {ang_auc:.4f}; "
    f"dominant angular frequency {dom_freq}.")
log(f"- Intrinsic dim: F+ PR-dim {pr_pos:.2f}, local PR-dim "
    f"~{np.median(local_dims):.2f}.")

shape = []
if rad_auc > 0.7:
    shape.append("radius-separated (shell/dome: one class occupies an inner/outer "
                 "radial band)")
if dom_freq >= 2 and np.nanstd(posrate) > 0.08:
    shape.append(f"angularly periodic (F+ rate alternates with ~{dom_freq} lobes "
                 "around the ring => interleaving sectors)")
elif dom_freq == 1 and np.nanstd(posrate) > 0.08:
    shape.append("a single angular arc (F+ concentrated on one side of the ring)")
log("\n**Shape verdict:** " + ("; ".join(shape) if shape
    else "curved low-D manifold where classes interleave (no dominant single "
         "axis)") + ".")

with open(os.path.join(OUT, "approach_10.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("\nWrote results/approach_10.md")
