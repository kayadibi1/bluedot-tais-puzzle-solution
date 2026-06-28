"""
R3 — UNIFY: Are Camp-A (radial shell / covariance-difference) axes the SAME directions
as Camp-B (food-gated sign-flip) axis?  Or orthogonal => two phenomena?

F = country (idx 5). Gate = food (idx 3). Layer L = h2 (post-ReLU hidden-2, 64-d).

Plan:
 1. Build GATING axes (within-food mean diffs v_food0, v_food1; signed-averaged dir)
    and SHELL axes (top eigenvectors of Cov_pos - Cov_neg for country).
 2. Principal angles between span(gating) and span(top-k shell eigvecs), k=1..6.
 3. Decompose country recoverability:
      - gated-linear-only AUC (project onto gating axis, un-flip sign with food).
      - residual QDA AUC after projecting OUT the gating subspace.
      - symmetric: food-gated-linear AUC after projecting OUT top shell subspace.
 4. Synthesize: unified (same geometry conditional vs unconditional) or distinct.
"""
import os, json
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
from numpy.linalg import eigh, svd, norm
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

np.random.seed(0)
C   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RES, exist_ok=True)

Xtr = np.load(os.path.join(C, "h2_train.npy")).astype(np.float64)
Xte = np.load(os.path.join(C, "h2_test.npy")).astype(np.float64)
Ytr = np.load(os.path.join(C, "labels_train.npy"))
Yte = np.load(os.path.join(C, "labels_test.npy"))

I_COUNTRY, I_FOOD = 5, 3
ctr = Ytr[:, I_COUNTRY].astype(int); cte = Yte[:, I_COUNTRY].astype(int)
ftr = Ytr[:, I_FOOD].astype(int);    fte = Yte[:, I_FOOD].astype(int)

# Standardize using TRAIN stats (consistent with prior probes); keeps geometry sane.
scaler = StandardScaler().fit(Xtr)
Ztr = scaler.transform(Xtr)
Zte = scaler.transform(Xte)

log = {}

# ---------------------------------------------------------------------------
# 1a. GATING axes: within-food country mean-difference directions
# ---------------------------------------------------------------------------
def mean_diff(Z, c, mask):
    pos = Z[mask & (c == 1)].mean(0)
    neg = Z[mask & (c == 0)].mean(0)
    return pos - neg

v_food0 = mean_diff(Ztr, ctr, ftr == 0)        # country dir inside food=0
v_food1 = mean_diff(Ztr, ctr, ftr == 1)        # country dir inside food=1
u0 = v_food0 / norm(v_food0)
u1 = v_food1 / norm(v_food1)
cos_gate = float(u0 @ u1)
log["v_food0_norm"] = float(norm(v_food0))
log["v_food1_norm"] = float(norm(v_food1))
log["cos_vfood0_vfood1"] = cos_gate

# "signed-averaged" country direction: flip food=1 dir to align, then average.
# This is the single axis Camp B uses (un-flip the sign per food half).
sign = 1.0 if cos_gate >= 0 else -1.0
v_signed = (u0 + sign * u1)
v_signed = v_signed / norm(v_signed)

# Unconditional (global) country mean-diff — should be ~0 (means coincide).
v_global = mean_diff(Ztr, ctr, np.ones(len(ctr), bool))
log["v_global_country_meandiff_norm"] = float(norm(v_global))

# Gating subspace: the 2-D span of the two within-food directions (they are nearly
# anti-parallel so effectively ~1-D, but keep both to be fair to Camp B).
G = np.stack([u0, u1], axis=1)                 # (64, 2)
Qg, _ = np.linalg.qr(G)                        # orthonormal basis of gating span
# effective rank of gating span
sg = svd(G, compute_uv=False)
log["gating_span_singvals"] = sg.tolist()

# ---------------------------------------------------------------------------
# 1b. SHELL axes: top eigenvectors of (Cov_pos - Cov_neg) for country
# ---------------------------------------------------------------------------
Zp = Ztr[ctr == 1]; Zn = Ztr[ctr == 0]
Cp = np.cov(Zp, rowvar=False)
Cn = np.cov(Zn, rowvar=False)
Dcov = Cp - Cn                                  # variance-difference matrix
w, V = eigh(Dcov)                               # ascending eigenvalues
order = np.argsort(np.abs(w))[::-1]             # by |eigenvalue| (variance gap magnitude)
w = w[order]; V = V[:, order]
log["shell_top_eigs_signed"] = w[:8].tolist()   # sign: + => pos-class MORE variance here
# country=1 is the tight core (less variance) => Cp - Cn should be NEGATIVE on shell axes
shell_vecs = V                                  # columns = shell eigenvectors (by |w|)

# ---------------------------------------------------------------------------
# 2. PRINCIPAL ANGLES between span(gating) and span(top-k shell eigvecs)
# ---------------------------------------------------------------------------
def principal_cosines(A, B):
    """cosines of principal angles between column-spaces of A and B (orthonormalized)."""
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    s = svd(Qa.T @ Qb, compute_uv=False)
    return np.clip(s, 0, 1)

# (i) gating axis (1-D signed) vs top-k shell subspace
print("Principal angles: span(gating) vs span(top-k shell eigvecs)")
pa_table = []
for k in range(1, 7):
    Sk = shell_vecs[:, :k]
    # cos between the SIGNED 1-D gating axis and the shell-k subspace
    cos1d = principal_cosines(v_signed[:, None], Sk)[0]
    # cos principal angles between the 2-D gating span and shell-k subspace
    cos2d = principal_cosines(Qg, Sk)
    pa_table.append((k, float(cos1d), [float(x) for x in cos2d]))
    print(f"  k={k}: cos(signed-gating, shell_k)={cos1d:.4f} | "
          f"cos PAs(2D-gating, shell_k)={np.round(cos2d,4).tolist()}")
log["principal_angles"] = pa_table

# Also: how much of each shell eigenvector lies in the gating span, and vice versa.
# Projection energy of top shell eigvec onto gating span:
for k in range(1, 4):
    s_k = shell_vecs[:, k-1]
    proj = Qg @ (Qg.T @ s_k)
    log[f"shell_eig{k}_energy_in_gating_span"] = float(norm(proj)**2)
# Projection energy of signed gating axis onto top-k shell subspace:
for k in [1, 2, 3, 6]:
    Sk = shell_vecs[:, :k]
    Qs, _ = np.linalg.qr(Sk)
    proj = Qs @ (Qs.T @ v_signed)
    log[f"gating_energy_in_shell_top{k}"] = float(norm(proj)**2)

# ---------------------------------------------------------------------------
# 3a. Gated-linear-only AUC: project h2 onto gating axis, un-flip with food.
#     score = (Z @ v_signed) * sign_per_food.  We orient so country=1 -> high.
# ---------------------------------------------------------------------------
def gated_linear_score(Z, f):
    raw = Z @ v_signed
    # within food=0 the country dir is +u0~+v_signed; within food=1 it's u1 (anti).
    # un-flip: multiply food=1 samples by sign of (u1 . v_signed)
    s_food1 = np.sign(u1 @ v_signed) if (u1 @ v_signed) != 0 else 1.0
    out = np.where(f == 0, raw, raw * s_food1)
    return out

# Orient using train, evaluate AUC on test.
g_tr = gated_linear_score(Ztr, ftr)
if roc_auc_score(ctr, g_tr) < 0.5:
    orient = -1.0
else:
    orient = 1.0
g_te = orient * gated_linear_score(Zte, fte)
auc_gated_linear = roc_auc_score(cte, g_te)
log["AUC_gated_linear_only"] = float(auc_gated_linear)
print(f"\nGated-linear-only AUC (1-D, food un-flip) = {auc_gated_linear:.4f}")

# Sanity: ungated single linear axis (v_signed without un-flip) — should be ~chance.
auc_ungated = roc_auc_score(cte, Zte @ v_signed)
auc_ungated = max(auc_ungated, 1 - auc_ungated)
log["AUC_ungated_signed_axis"] = float(auc_ungated)
print(f"Ungated single signed axis AUC (no un-flip)  = {auc_ungated:.4f}")

# Full QDA baseline on standardized h2 (reproduce Camp A ~0.99).
def qda_auc(Ztr_, ctr_, Zte_, cte_, reg=0.05):
    q = QDA(reg_param=reg).fit(Ztr_, ctr_)
    p = q.predict_proba(Zte_)[:, 1]
    return roc_auc_score(cte_, p)

auc_qda_full = qda_auc(Ztr, ctr, Zte, cte)
log["AUC_QDA_full_h2"] = float(auc_qda_full)
print(f"QDA full-h2 AUC (Camp A baseline)            = {auc_qda_full:.4f}")

# ---------------------------------------------------------------------------
# 3b. Residual after PROJECTING OUT the gating subspace: is shell signal left?
#     Remove the 2-D gating span (Qg) from Z, then QDA on residual.
# ---------------------------------------------------------------------------
def project_out(Z, Qbasis):
    return Z - (Z @ Qbasis) @ Qbasis.T

# Remove gating span (2-D). Also remove just the signed 1-D axis for comparison.
Qsigned, _ = np.linalg.qr(v_signed[:, None])
Ztr_res_g1 = project_out(Ztr, Qsigned); Zte_res_g1 = project_out(Zte, Qsigned)
Ztr_res_g2 = project_out(Ztr, Qg);      Zte_res_g2 = project_out(Zte, Qg)

auc_res_qda_g1 = qda_auc(Ztr_res_g1, ctr, Zte_res_g1, cte)
auc_res_qda_g2 = qda_auc(Ztr_res_g2, ctr, Zte_res_g2, cte)
log["AUC_QDA_residual_after_remove_gating1D"] = float(auc_res_qda_g1)
log["AUC_QDA_residual_after_remove_gating2D"] = float(auc_res_qda_g2)
print(f"\nQDA on residual after removing gating 1-D axis = {auc_res_qda_g1:.4f}")
print(f"QDA on residual after removing gating 2-D span = {auc_res_qda_g2:.4f}")

# Within-food linear AUC on residual (does gated-linear survive removing its own axis?)
def within_food_linear_auc(Ztr_, Zte_):
    aucs = []
    for fv in (0, 1):
        mtr = ftr == fv; mte = fte == fv
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(Ztr_[mtr], ctr[mtr])
        p = clf.predict_proba(Zte_[mte])[:, 1]
        aucs.append(roc_auc_score(cte[mte], p))
    return float(np.mean(aucs)), aucs

wf_full, _   = within_food_linear_auc(Ztr, Zte)
wf_res_g2, _ = within_food_linear_auc(Ztr_res_g2, Zte_res_g2)
log["AUC_within_food_linear_full"] = wf_full
log["AUC_within_food_linear_residual_after_gating2D"] = wf_res_g2
print(f"Within-food linear AUC (full)                = {wf_full:.4f}")
print(f"Within-food linear AUC (residual, gate axis removed) = {wf_res_g2:.4f}")

# ---------------------------------------------------------------------------
# 3c. Symmetric: remove top shell subspace, does food-gated-linear survive?
# ---------------------------------------------------------------------------
for k in [2, 4, 6]:
    Sk = shell_vecs[:, :k]
    Qs, _ = np.linalg.qr(Sk)
    Ztr_rs = project_out(Ztr, Qs); Zte_rs = project_out(Zte, Qs)
    # gated-linear on residual: rebuild within-food dirs in residual, score
    wf_rs, _ = within_food_linear_auc(Ztr_rs, Zte_rs)
    # also QDA on residual (should drop if shell axes carried the signal)
    auc_qda_rs = qda_auc(Ztr_rs, ctr, Zte_rs, cte)
    log[f"AUC_within_food_linear_residual_after_shell{k}"] = wf_rs
    log[f"AUC_QDA_residual_after_shell{k}"] = float(auc_qda_rs)
    print(f"After removing shell top-{k}: within-food linear AUC={wf_rs:.4f} | "
          f"residual QDA AUC={auc_qda_rs:.4f}")

# ---------------------------------------------------------------------------
# 4. The reconciliation arithmetic: a sign-flipping mean along a fixed axis,
#    marginalized over food, produces a PURE covariance (variance) difference.
#    Test: along the signed gating axis, compare per-class variance pos vs neg.
# ---------------------------------------------------------------------------
proj_axis = Ztr @ v_signed
var_pos = proj_axis[ctr == 1].var()
var_neg = proj_axis[ctr == 0].var()
log["var_along_gating_axis_country1"] = float(var_pos)
log["var_along_gating_axis_country0"] = float(var_neg)
# decompose neg-class variance into within-food + between-food(=sign-flip mean) parts
def var_decomp(proj, c, f, cls):
    x = proj[c == cls]
    ff = f[c == cls]
    overall = x.var()
    within = np.mean([x[ff == v].var() * (ff == v).mean() for v in (0, 1)])
    # between-food mean spread
    gm = x.mean()
    between = np.sum([(ff == v).mean() * (x[ff == v].mean() - gm)**2 for v in (0, 1)])
    return overall, within, between
ov0, wi0, be0 = var_decomp(proj_axis, ctr, ftr, 0)
ov1, wi1, be1 = var_decomp(proj_axis, ctr, ftr, 1)
log["country0_var_total/within/betweenfood"] = [float(ov0), float(wi0), float(be0)]
log["country1_var_total/within/betweenfood"] = [float(ov1), float(wi1), float(be1)]
print(f"\nAlong signed gating axis: var(country=1)={var_pos:.3f}  var(country=0)={var_neg:.3f}")
print(f"  country=0 var: total={ov0:.3f} within-food={wi0:.3f} between-food(signflip)={be0:.3f}")
print(f"  country=1 var: total={ov1:.3f} within-food={wi1:.3f} between-food(signflip)={be1:.3f}")

# How much of the shell variance-difference lives in the gating span?
# total |Cov_pos - Cov_neg| energy vs energy captured by gating-span projection.
total_energy = np.sum(w**2)                       # = ||Dcov||_F^2 (eigvals)
Dproj = Qg.T @ Dcov @ Qg                          # gating-span block of Dcov
gating_block_energy = np.sum(Dproj**2)
log["Dcov_total_fro2"] = float(total_energy)
log["Dcov_energy_in_gating_span"] = float(gating_block_energy)
log["Dcov_frac_energy_in_gating_span"] = float(gating_block_energy / total_energy)
print(f"\n||Cov_pos-Cov_neg||_F^2 total = {total_energy:.3f}; "
      f"in gating 2-D span = {gating_block_energy:.3f} "
      f"({100*gating_block_energy/total_energy:.2f}%)")

with open(os.path.join(RES, "round2_r3_raw.json"), "w") as fout:
    json.dump(log, fout, indent=2)
print("\nSaved results/round2_r3_raw.json")
