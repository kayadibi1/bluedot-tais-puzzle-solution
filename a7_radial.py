"""
Approach #7: RADIAL / SHELL (distance-based) hypothesis.

Question: at Layer L (= h2, post-ReLU hidden-2, 64-dim), seven of eight features
are linearly readable, one feature F is not. Hypothesis: F is encoded by DISTANCE
from a center (a radial / shell structure) -- one class forms a core ball, the
other a surrounding shell, so no hyperplane separates them but a radial coordinate
(norm / Mahalanobis distance) does.

Steps:
  1. Confirm F via a linear-probe AUC sweep across all 8 features on h2 (find the
     low outlier).
  2. Radial coordinate test: distribution of ||x - mu|| per class (mu = global mean,
     mu = each class mean). AUC of the 1-D radial coordinate.
  3. Quadratic / radial probe: augment x with ||x||^2 and per-dim squares, fit
     logistic; compare AUC to plain linear. Fit QDA vs LDA -- a big QDA>LDA gap
     signals covariance-difference (shell/elliptic) structure.
  4. Class covariance comparison: trace/det of each class covariance, Mahalanobis &
     Bhattacharyya structure -- are the two classes concentric with different scale?
  5. Contrast with a known-linear feature to show the radial signature is specific.
"""
import os, json
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
from numpy.linalg import slogdet, pinv
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis as LDA,
    QuadraticDiscriminantAnalysis as QDA,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

np.random.seed(0)
C = r"C:\Users\Sidar\Desktop\puzzle\bluedot-tais-puzzle\cache"
RES = r"C:\Users\Sidar\Desktop\puzzle\bluedot-tais-puzzle\results"
os.makedirs(RES, exist_ok=True)

Xtr = np.load(os.path.join(C, "h2_train.npy"))
Xte = np.load(os.path.join(C, "h2_test.npy"))
Ytr = np.load(os.path.join(C, "labels_train.npy"))
Yte = np.load(os.path.join(C, "labels_test.npy"))
feats = json.load(open(os.path.join(os.path.dirname(C), "feature_names.json")))

out = []
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    out.append(s)

log("=" * 70)
log("APPROACH #7: RADIAL / SHELL HYPOTHESIS  (Layer L = h2, post-ReLU hidden-2)")
log("=" * 70)
log(f"Xtr {Xtr.shape}  Xte {Xte.shape}  features={feats}")
log(f"h2 sparsity (frac zeros) train={ (Xtr==0).mean():.3f}")

# ---------------------------------------------------------------------------
# STEP 1: linear-probe AUC sweep -> find F (the low outlier)
# ---------------------------------------------------------------------------
log("\n" + "-" * 70)
log("STEP 1: linear-probe AUC sweep across all 8 features (find F)")
log("-" * 70)
sc = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

lin_auc = {}
for j in range(8):
    ytr, yte = Ytr[:, j], Yte[:, j]
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xtr_s, ytr)
    p = clf.predict_proba(Xte_s)[:, 1]
    a = roc_auc_score(yte, p)
    lin_auc[j] = a
    log(f"  feat {j} {feats[j]:<10} linear-probe AUC = {a:.4f}  (pos rate {yte.mean():.2f})")

F = min(lin_auc, key=lin_auc.get)
log(f"\n  >> Lowest linear AUC = feature {F} '{feats[F]}'  (AUC {lin_auc[F]:.4f}) -> candidate F")
# pick a clearly-linear contrast feature (highest AUC)
LINREF = max(lin_auc, key=lin_auc.get)
log(f"  >> Contrast (clearly-linear) feature = {LINREF} '{feats[LINREF]}' (AUC {lin_auc[LINREF]:.4f})")

# ---------------------------------------------------------------------------
# helper: 1-D AUC of a scalar coordinate (orient so higher->class1)
# ---------------------------------------------------------------------------
def auc1d(score, y):
    a = roc_auc_score(y, score)
    return max(a, 1 - a)

# ---------------------------------------------------------------------------
# STEP 2: radial coordinate test for a feature
# ---------------------------------------------------------------------------
def radial_analysis(j, tag):
    log("\n" + "-" * 70)
    log(f"STEP 2/3/4 for feature {j} '{feats[j]}'  [{tag}]")
    log("-" * 70)
    ytr, yte = Ytr[:, j], Yte[:, j]
    res = {"feat": feats[j], "idx": int(j)}

    # --- radial coordinates ---
    mu_glob = Xtr.mean(0)
    mu0 = Xtr[ytr == 0].mean(0)   # class-0 center
    mu1 = Xtr[ytr == 1].mean(0)   # class-1 center

    def norm_to(mu):
        return np.linalg.norm(Xte - mu, axis=1)
    r_glob = norm_to(mu_glob)
    r_mu0 = norm_to(mu0)
    r_mu1 = norm_to(mu1)
    # also plain L2 norm of the raw activation (radius from origin)
    r_origin = np.linalg.norm(Xte, axis=1)

    # mean radius per class (test)
    for name, r in [("||x-mu_global||", r_glob), ("||x-mu_class0||", r_mu0),
                    ("||x-mu_class1||", r_mu1), ("||x|| (origin)", r_origin)]:
        m0, m1 = r[yte == 0].mean(), r[yte == 1].mean()
        s0, s1 = r[yte == 0].std(), r[yte == 1].std()
        a = auc1d(r, yte)
        log(f"  radial {name:<18} class0 r={m0:6.3f}+-{s0:5.3f}  "
            f"class1 r={m1:6.3f}+-{s1:5.3f}  | 1-D AUC={a:.4f}")
    res["radial_auc_global"] = float(auc1d(r_glob, yte))
    res["radial_auc_origin"] = float(auc1d(r_origin, yte))
    res["radial_auc_best"] = float(max(auc1d(r_glob, yte), auc1d(r_mu0, yte),
                                       auc1d(r_mu1, yte), auc1d(r_origin, yte)))
    # which class is larger radius (from global mean)
    res["class1_larger_radius"] = bool(r_glob[yte == 1].mean() > r_glob[yte == 0].mean())

    # --- linear probe AUC (baseline) ---
    lin = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr_s, ytr)
    a_lin = roc_auc_score(yte, lin.predict_proba(Xte_s)[:, 1])
    res["linear_auc"] = float(a_lin)

    # --- quadratic/radial-augmented logistic ---
    # augment standardized x with per-dim squares + ||x||^2
    def aug(Xs):
        n2 = (Xs ** 2).sum(1, keepdims=True)
        return np.hstack([Xs, Xs ** 2, n2])
    Xtr_a, Xte_a = aug(Xtr_s), aug(Xte_s)
    quad = LogisticRegression(max_iter=5000, C=1.0).fit(Xtr_a, ytr)
    a_quad = roc_auc_score(yte, quad.predict_proba(Xte_a)[:, 1])
    res["quad_aug_auc"] = float(a_quad)
    log(f"  linear-probe AUC = {a_lin:.4f}   quad/radial-augmented AUC = {a_quad:.4f}"
        f"   (gain {a_quad - a_lin:+.4f})")

    # --- LDA vs QDA ---
    lda = LDA().fit(Xtr, ytr)
    a_lda = roc_auc_score(yte, lda.predict_proba(Xte)[:, 1])
    # QDA needs full-rank within-class cov; h2 is sparse -> add reg
    qda = QDA(reg_param=0.05).fit(Xtr, ytr)
    a_qda = roc_auc_score(yte, qda.predict_proba(Xte)[:, 1])
    res["lda_auc"] = float(a_lda)
    res["qda_auc"] = float(a_qda)
    res["qda_minus_lda"] = float(a_qda - a_lda)
    log(f"  LDA AUC = {a_lda:.4f}   QDA AUC = {a_qda:.4f}   (QDA-LDA gap {a_qda - a_lda:+.4f})")

    # --- class covariance comparison (concentric / scale difference?) ---
    X0, X1 = Xtr[ytr == 0], Xtr[ytr == 1]
    c0 = np.cov(X0, rowvar=False)
    c1 = np.cov(X1, rowvar=False)
    tr0, tr1 = np.trace(c0), np.trace(c1)
    # regularized logdet for stability
    eps = 1e-4 * np.eye(c0.shape[0])
    _, ld0 = slogdet(c0 + eps)
    _, ld1 = slogdet(c1 + eps)
    # center distance vs spread
    dmu = np.linalg.norm(mu1 - mu0)
    log(f"  class0 cov: trace={tr0:.3f}  logdet={ld0:.2f}   "
        f"mean-radius(from own mean)={np.sqrt(np.mean(((X0-mu0)**2).sum(1))):.3f}")
    log(f"  class1 cov: trace={tr1:.3f}  logdet={ld1:.2f}   "
        f"mean-radius(from own mean)={np.sqrt(np.mean(((X1-mu1)**2).sum(1))):.3f}")
    log(f"  trace ratio class1/class0 = {tr1/tr0:.3f}   "
        f"||mu1-mu0|| = {dmu:.3f}")
    res["trace0"] = float(tr0); res["trace1"] = float(tr1)
    res["trace_ratio_1_0"] = float(tr1 / tr0)
    res["logdet0"] = float(ld0); res["logdet1"] = float(ld1)
    res["center_dist"] = float(dmu)

    # --- Mahalanobis / Bhattacharyya structure ---
    # Bhattacharyya distance between two Gaussians
    cm = 0.5 * (c0 + c1) + eps
    _, ldm = slogdet(cm)
    diff = (mu1 - mu0)
    maha = diff @ pinv(cm) @ diff
    bhat = 0.125 * maha + 0.5 * (ldm - 0.5 * (ld0 + ld1))
    log(f"  Bhattacharyya: mean-term(0.125*Maha)={0.125*maha:.3f}  "
        f"cov-term={0.5*(ldm-0.5*(ld0+ld1)):.3f}   total D_B={bhat:.3f}")
    res["bhat_mean_term"] = float(0.125 * maha)
    res["bhat_cov_term"] = float(0.5 * (ldm - 0.5 * (ld0 + ld1)))
    res["bhat_total"] = float(bhat)
    # ratio: if cov-term >> mean-term, separation is mostly covariance(shell)-driven
    res["cov_over_mean_term"] = float((0.5*(ldm-0.5*(ld0+ld1))) / max(0.125*maha, 1e-9))
    log(f"  >> covariance-term / mean-term ratio = {res['cov_over_mean_term']:.2f}  "
        f"(>>1 => separation is scale/shell-driven, not a hyperplane shift)")
    return res

resF = radial_analysis(F, "CANDIDATE F")
resR = radial_analysis(LINREF, "LINEAR CONTRAST")

# also run for ALL features to make the radial-AUC table complete
log("\n" + "-" * 70)
log("STEP 5: radial-coordinate (||x-mu_global||) 1-D AUC for ALL features")
log("-" * 70)
mu_glob = Xtr.mean(0)
r_glob_te = np.linalg.norm(Xte - mu_glob, axis=1)
r_origin_te = np.linalg.norm(Xte, axis=1)
radial_table = {}
for j in range(8):
    yte = Yte[:, j]
    a_glob = auc1d(r_glob_te, yte)
    a_orig = auc1d(r_origin_te, yte)
    radial_table[j] = (a_glob, a_orig)
    flag = "  <-- F" if j == F else ""
    log(f"  feat {j} {feats[j]:<10} linAUC={lin_auc[j]:.4f}  "
        f"radial(||x-mu||)AUC={a_glob:.4f}  norm(||x||)AUC={a_orig:.4f}{flag}")

# ---------------------------------------------------------------------------
# VERDICT
# ---------------------------------------------------------------------------
log("\n" + "=" * 70)
log("VERDICT")
log("=" * 70)
is_radial = (resF["radial_auc_best"] > 0.7) and (resF["cov_over_mean_term"] > 1.0)
log(f"F = {feats[F]} (idx {F})")
log(f"  linear AUC        = {resF['linear_auc']:.4f}")
log(f"  radial 1-D AUC    = {resF['radial_auc_best']:.4f}  (best of global/origin/class centers)")
log(f"  quad-aug AUC      = {resF['quad_aug_auc']:.4f}")
log(f"  QDA-LDA gap       = {resF['qda_minus_lda']:+.4f}  (LDA {resF['lda_auc']:.4f} -> QDA {resF['qda_auc']:.4f})")
log(f"  trace ratio c1/c0 = {resF['trace_ratio_1_0']:.3f}")
log(f"  cov/mean Bhat     = {resF['cov_over_mean_term']:.2f}")
core = "class1 (label=1)" if not resF["class1_larger_radius"] else "class0 (label=0)"
shell = "class0 (label=0)" if not resF["class1_larger_radius"] else "class1 (label=1)"
log(f"  larger-radius (shell) class = {'class1' if resF['class1_larger_radius'] else 'class0'}")
log(f"  => CORE (tight, small radius): {core};  SHELL (large radius): {shell}")
log(f"\n  Contrast feature '{feats[LINREF]}': radial AUC={resR['radial_auc_best']:.4f}, "
    f"QDA-LDA gap={resR['qda_minus_lda']:+.4f}, cov/mean={resR['cov_over_mean_term']:.2f}")

# Save JSON of key numbers
summary = {"F": feats[F], "F_idx": int(F), "linear_auc_sweep": {feats[j]: float(lin_auc[j]) for j in range(8)},
           "candidate_F": resF, "linear_contrast": resR,
           "radial_table": {feats[j]: {"lin": float(lin_auc[j]),
                                       "radial_global": float(radial_table[j][0]),
                                       "norm_origin": float(radial_table[j][1])} for j in range(8)}}
json.dump(summary, open(os.path.join(RES, "approach_7_data.json"), "w"), indent=2)
open(os.path.join(RES, "approach_7_log.txt"), "w").write("\n".join(out))
log(f"\nSaved results/approach_7_data.json and results/approach_7_log.txt")
print("DONE")
