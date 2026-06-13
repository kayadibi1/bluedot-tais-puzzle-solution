"""
R5 (relaunch) — Build an explicit GENERATIVE model of `country`'s h2 geometry and
falsify whichever hypothesis fails.

F = country (idx 5). Gate = food (idx 3). Secondary = sentiment (idx 4).
Layer L = h2 (post-ReLU hidden-2, 64-d). GPU broken -> CPU/.npy only.

The apparent contradiction to reconcile:
  - Under SIGN-FLIP gating (country mean offset flips sign with food, cos -0.996),
    country=1 POOLED over food should look BIMODAL / higher-variance along the gate axis.
  - Yet country=1 was MEASURED as TIGHTER (variance ratio ~0.45).
Resolve by explicit generative modeling and quantify gate vs shell vs combined.

TASKS:
 1) 4-CELL GEOMETRY in reduced PCA space: per (country,food) mean & cov.
    Anti-parallel test: is [mu(c1,f0)-mu(c0,f0)] anti-parallel to [mu(c1,f1)-mu(c0,f1)]?
 2) RESOLVE tight-vs-bimodal: distribution of <h2,v> along gate axis v, country=1 vs 0,
    split by food and pooled. Is c=1 bimodal pooled? Is the measured tightness on OTHER axes?
 3) THREE generative models (M1 pure-gate, M2 pure-shell, M3 combined); compare held-out
    country-AUC + log-likelihood; reproduce real per-cell means/covs. Falsify losers.
 4) One-sentence verdict.
"""
import os, json
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
from numpy.linalg import norm, slogdet, inv, eigh
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(0)
C   = r"C:\Users\Sidar\Desktop\puzzle\bluedot-tais-puzzle\cache"
RES = r"C:\Users\Sidar\Desktop\puzzle\bluedot-tais-puzzle\results"
os.makedirs(RES, exist_ok=True)

Xtr = np.load(os.path.join(C, "h2_train.npy")).astype(np.float64)
Xte = np.load(os.path.join(C, "h2_test.npy")).astype(np.float64)
Ytr = np.load(os.path.join(C, "labels_train.npy"))
Yte = np.load(os.path.join(C, "labels_test.npy"))

I_COUNTRY, I_FOOD, I_SENT = 5, 3, 4
ctr = Ytr[:, I_COUNTRY].astype(int); cte = Yte[:, I_COUNTRY].astype(int)
ftr = Ytr[:, I_FOOD].astype(int);    fte = Yte[:, I_FOOD].astype(int)
str_ = Ytr[:, I_SENT].astype(int);   ste = Yte[:, I_SENT].astype(int)

# Standardize using TRAIN stats (consistent with R1/R2/R3).
scaler = StandardScaler().fit(Xtr)
Ztr = scaler.transform(Xtr)
Zte = scaler.transform(Xte)

# Reduced PCA space: fit on TRAIN (all rows). Keep enough dims to retain country signal.
K = 16
pca = PCA(n_components=K, svd_solver="full").fit(Ztr)
Ptr = pca.transform(Ztr)   # (Ntr, K)
Pte = pca.transform(Zte)   # (Nte, K)
evr = pca.explained_variance_ratio_

log = {"K_pca": K, "pca_evr_top": evr[:8].tolist(), "pca_evr_cumsum_K": float(evr.sum())}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def cell_mask(c, f, cv, fv):
    return (c == cv) & (f == fv)

def gauss_logpdf(X, mu, Sig, eps=1e-6):
    """log N(X; mu, Sig) per row. Sig regularized."""
    d = X.shape[1]
    S = Sig + eps * np.eye(d)
    sign, logdet = slogdet(S)
    Si = inv(S)
    diff = X - mu
    quad = np.einsum("ni,ij,nj->n", diff, Si, diff)
    return -0.5 * (d * np.log(2 * np.pi) + logdet + quad)

def class_auc_from_loglik(ll1, ll0_pos, ll0_neg):
    """score = log p(x|c=1) - log p(x|c=0); AUC vs true country labels handled by caller."""
    pass

# ===========================================================================
# TASK 1 — 4-CELL GEOMETRY (means + covariances) in reduced PCA space
# ===========================================================================
cells = {}
for cv in (0, 1):
    for fv in (0, 1):
        m = cell_mask(ctr, ftr, cv, fv)
        Xc = Ptr[m]
        cells[(cv, fv)] = {
            "n": int(m.sum()),
            "mu": Xc.mean(0),
            "cov": np.cov(Xc, rowvar=False),
            "trace": float(np.trace(np.cov(Xc, rowvar=False))),
        }

# Country offset vectors within each food half
d_f0 = cells[(1, 0)]["mu"] - cells[(0, 0)]["mu"]   # country dir at food=0
d_f1 = cells[(1, 1)]["mu"] - cells[(0, 1)]["mu"]   # country dir at food=1
cos_anti = float(d_f0 @ d_f1 / (norm(d_f0) * norm(d_f1)))
# Food offset (within country=0): how big is food displacement vs country displacement
food_off_c0 = cells[(0, 1)]["mu"] - cells[(0, 0)]["mu"]
food_off_c1 = cells[(1, 1)]["mu"] - cells[(1, 0)]["mu"]

log["task1"] = {
    "cell_n": {f"c{cv}f{fv}": cells[(cv, fv)]["n"] for cv in (0, 1) for fv in (0, 1)},
    "d_food0_norm": float(norm(d_f0)),
    "d_food1_norm": float(norm(d_f1)),
    "cos_d_food0_d_food1": cos_anti,
    "food_offset_c0_norm": float(norm(food_off_c0)),
    "food_offset_c1_norm": float(norm(food_off_c1)),
    "country_offset_mean_norm": float(0.5 * (norm(d_f0) + norm(d_f1))),
    "country_vs_food_ratio": float(0.5 * (norm(d_f0) + norm(d_f1)) / (0.5 * (norm(food_off_c0) + norm(food_off_c1)))),
    "cell_trace": {f"c{cv}f{fv}": cells[(cv, fv)]["trace"] for cv in (0, 1) for fv in (0, 1)},
}

# Gate axis v = signed-average of within-food country directions (un-flip food=1).
u0 = d_f0 / norm(d_f0)
u1 = d_f1 / norm(d_f1)
s = 1.0 if (u0 @ u1) >= 0 else -1.0
v_gate = u0 + s * u1
v_gate = v_gate / norm(v_gate)          # unit gate axis in PCA space (K-dim)
log["task1"]["gate_axis_built_from"] = "signed-avg within-food country mean-diff (food=1 sign-flipped to align)"

# ===========================================================================
# TASK 2 — RESOLVE tight-vs-bimodal
# Project onto gate axis v; look at <P, v> per (country,food) and POOLED.
# Then ask: where IS country=1 tight? (variance per PCA axis, c1 vs c0).
# ===========================================================================
g_tr = Ptr @ v_gate     # scalar gate projection per train sample

def stats(arr):
    return float(np.mean(arr)), float(np.std(arr)), float(np.var(arr))

proj = {}
for cv in (0, 1):
    for fv in (0, 1):
        m = cell_mask(ctr, ftr, cv, fv)
        mu_, sd_, var_ = stats(g_tr[m])
        proj[f"c{cv}f{fv}"] = {"mean": mu_, "std": sd_, "var": var_, "n": int(m.sum())}

# Pooled over food
pooled = {}
for cv in (0, 1):
    m = (ctr == cv)
    mu_, sd_, var_ = stats(g_tr[m])
    pooled[f"c{cv}"] = {"mean": mu_, "std": sd_, "var": var_, "n": int(m.sum())}

# Bimodality test for pooled distributions along v: dip-style via bimodality coefficient
# BC = (skew^2 + 1) / kurtosis ; BC > 0.555 suggests bimodal. Also report between-food
# component of variance (the sign-flip) vs within-cell.
from scipy.stats import skew, kurtosis
def bimod_coef(a):
    n = len(a)
    g = skew(a); k = kurtosis(a, fisher=True)  # excess kurtosis
    bc = (g**2 + 1.0) / (k + 3.0 * (n - 1)**2 / ((n - 2) * (n - 3)))
    return float(bc), float(g), float(k)

bc_c1, sk_c1, ku_c1 = bimod_coef(g_tr[ctr == 1])
bc_c0, sk_c0, ku_c0 = bimod_coef(g_tr[ctr == 0])

# Decompose pooled variance along v into between-food (sign-flip) + within-food
def decomp_var(arr, food):
    overall = np.var(arr)
    mu0 = arr[food == 0].mean(); mu1 = arr[food == 1].mean()
    p0 = (food == 0).mean(); p1 = (food == 1).mean()
    gmean = arr.mean()
    between = p0 * (mu0 - gmean)**2 + p1 * (mu1 - gmean)**2
    within = p0 * np.var(arr[food == 0]) + p1 * np.var(arr[food == 1])
    return float(overall), float(between), float(within), float(between / overall)

ov_c0, bw_c0, wi_c0, frac_c0 = decomp_var(g_tr[ctr == 1] if False else g_tr[ctr == 0], ftr[ctr == 0])
ov_c1, bw_c1, wi_c1, frac_c1 = decomp_var(g_tr[ctr == 1], ftr[ctr == 1])

# Where is country=1 tight? Variance ratio c1/c0 per PCA axis (pooled and within-food).
var_c1 = Ptr[ctr == 1].var(0)
var_c0 = Ptr[ctr == 0].var(0)
ratio_pooled = var_c1 / var_c0     # <1 means c1 tighter on that axis

# within food=0 only (removes the sign-flip pooling artifact)
m_f0 = (ftr == 0)
var_c1_f0 = Ptr[(ctr == 1) & m_f0].var(0)
var_c0_f0 = Ptr[(ctr == 0) & m_f0].var(0)
ratio_f0 = var_c1_f0 / var_c0_f0

# gate-axis index proxy: which PCA axis has largest |v_gate| component
gate_dom_axis = int(np.argmax(np.abs(v_gate)))

# total variance (overall scalar) c1 vs c0 in full PCA space
tot_var_c1 = float(Ptr[ctr == 1].var(0).sum())
tot_var_c0 = float(Ptr[ctr == 0].var(0).sum())

log["task2"] = {
    "proj_per_cell": proj,
    "proj_pooled": pooled,
    "gate_axis_dominant_pca_index": gate_dom_axis,
    "bimodality_coef_c1_pooled": bc_c1, "skew_c1": sk_c1, "exkurt_c1": ku_c1,
    "bimodality_coef_c0_pooled": bc_c0, "skew_c0": sk_c0, "exkurt_c0": ku_c0,
    "bc_threshold_uniform_0.555": 0.555,
    "var_decomp_along_v": {
        "c0": {"overall": ov_c0, "between_food": bw_c0, "within_food": wi_c0, "frac_between": frac_c0},
        "c1": {"overall": ov_c1, "between_food": bw_c1, "within_food": wi_c1, "frac_between": frac_c1},
    },
    "total_var_pca_c1": tot_var_c1, "total_var_pca_c0": tot_var_c0,
    "total_var_ratio_c1_over_c0": float(tot_var_c1 / tot_var_c0),
    "var_ratio_pooled_per_axis_top8": ratio_pooled[:8].tolist(),
    "var_ratio_withinfood0_per_axis_top8": ratio_f0[:8].tolist(),
    "n_axes_c1_tighter_pooled": int((ratio_pooled < 1).sum()),
    "n_axes_c1_tighter_withinfood0": int((ratio_f0 < 1).sum()),
}

# ===========================================================================
# TASK 3 — THREE GENERATIVE MODELS. Fit on TRAIN, score on TEST.
# Score = log p(x | c=1) - log p(x | c=0). country-AUC + mean test log-lik.
# ===========================================================================
# Common pieces fit on TRAIN
mu_all = Ptr.mean(0)
def cov_of(mask):
    return np.cov(Ptr[mask], rowvar=False)

# delta along gate axis: half the within-food country gap magnitude (avg)
delta = 0.5 * (norm(d_f0) + norm(d_f1)) / 2.0   # offset magnitude each side of food-mean
# More principled: per-food, c1 mean and c0 mean along v
gv_tr = g_tr  # already P@v
# food sign convention: along v, sign of (mean c1 - mean c0) within each food
sgn_f0 = np.sign(proj["c1f0"]["mean"] - proj["c0f0"]["mean"])
sgn_f1 = np.sign(proj["c1f1"]["mean"] - proj["c0f1"]["mean"])

def model_loglik(model, X, c, f):
    """Return per-row (ll1, ll0) for given model on test rows X with food f."""
    ll1 = np.full(len(X), -np.inf)
    ll0 = np.full(len(X), -np.inf)
    if model == "M1":
        # pure gate: country shifts mean by +delta*v*sign(food-half), shared cov Sig0.
        Sig = M1["Sig"]
        for fv in (0, 1):
            sel = (f == fv)
            mu1 = M1["mu_f"][fv] + M1["sgn"][fv] * M1["delta"] * v_gate
            mu0 = M1["mu_f"][fv] - M1["sgn"][fv] * M1["delta"] * v_gate
            ll1[sel] = gauss_logpdf(X[sel], mu1, Sig)
            ll0[sel] = gauss_logpdf(X[sel], mu0, Sig)
    elif model == "M2":
        # pure shell: same mean mu, c1 ~ N(mu, Sig_tight), c0 ~ N(mu, Sig_wide). No gate.
        ll1 = gauss_logpdf(X, M2["mu"], M2["Sig1"])
        ll0 = gauss_logpdf(X, M2["mu"], M2["Sig0"])
    elif model == "M3":
        # combined: per (country,food) cell full Gaussian (gate offset + per-class cov + sentiment-free).
        for fv in (0, 1):
            sel = (f == fv)
            ll1[sel] = gauss_logpdf(X[sel], M3[(1, fv)]["mu"], M3[(1, fv)]["cov"])
            ll0[sel] = gauss_logpdf(X[sel], M3[(0, fv)]["mu"], M3[(0, fv)]["cov"])
    return ll1, ll0

# ---- Fit M1: pure gate (equal covariances) ----
Sig_shared = cov_of(np.ones(len(ctr), bool))    # global pooled cov
mu_f = {fv: Ptr[ftr == fv].mean(0) for fv in (0, 1)}   # food-conditional mean (country-agnostic)
M1 = {"Sig": Sig_shared, "mu_f": mu_f, "delta": delta,
      "sgn": {0: float(sgn_f0), 1: float(sgn_f1)}}

# ---- Fit M2: pure shell (no gate, coincident mean, per-class cov) ----
M2 = {"mu": mu_all, "Sig1": cov_of(ctr == 1), "Sig0": cov_of(ctr == 0)}

# ---- Fit M3: combined per-cell Gaussian ----
M3 = {}
for cv in (0, 1):
    for fv in (0, 1):
        m = cell_mask(ctr, ftr, cv, fv)
        M3[(cv, fv)] = {"mu": Ptr[m].mean(0), "cov": cov_of(m)}

# Priors p(c=1), p(c=0) (use train base rate; AUC is threshold-free so prior only shifts)
p1 = (ctr == 1).mean(); p0 = 1 - p1
logprior = np.log(p1) - np.log(p0)

results = {}
for model in ("M1", "M2", "M3"):
    ll1, ll0 = model_loglik(model, Pte, cte, fte)
    score = (ll1 - ll0) + logprior
    auc = roc_auc_score(cte, score)
    # held-out total data log-likelihood under the generative model:
    # for each row use its TRUE class density (how well the model fits the real data)
    ll_true = np.where(cte == 1, ll1, ll0)
    mean_ll = float(np.mean(ll_true))
    results[model] = {"country_auc": float(auc), "mean_test_loglik": mean_ll}

# Reference: full QDA / per-cell oracle on test, and linear-on-v AUC
# (a) ungated single axis (sign cancels) -> ~chance, reproduces linear ~0.49
auc_ungated = roc_auc_score(cte, Pte @ v_gate)
# (b) gated-linear: un-flip sign with food
gated_lin = (Pte @ v_gate) * np.where(fte == 0, sgn_f0, sgn_f1)
auc_gated_lin = roc_auc_score(cte, gated_lin)

results["ref_ungated_v_axis"] = {"country_auc": float(auc_ungated)}
results["ref_gated_linear_v_axis"] = {"country_auc": float(auc_gated_lin)}

# Per-cell mean/cov reproduction error: compare model-implied cell means/cov-traces vs real TEST cells
def real_test_cell(cv, fv):
    m = cell_mask(cte, fte, cv, fv)
    Xc = Pte[m]
    return Xc.mean(0), np.cov(Xc, rowvar=False)

cell_fit = {}
for model in ("M1", "M2", "M3"):
    mean_err = []
    trace_real = []; trace_model = []
    for cv in (0, 1):
        for fv in (0, 1):
            rmu, rcov = real_test_cell(cv, fv)
            if model == "M1":
                mmu = M1["mu_f"][fv] + (1 if cv == 1 else -1) * M1["sgn"][fv] * M1["delta"] * v_gate
                mcov = M1["Sig"]
            elif model == "M2":
                mmu = M2["mu"]
                mcov = M2["Sig1"] if cv == 1 else M2["Sig0"]
            else:
                mmu = M3[(cv, fv)]["mu"]; mcov = M3[(cv, fv)]["cov"]
            mean_err.append(float(norm(mmu - rmu)))
            trace_real.append(float(np.trace(rcov)))
            trace_model.append(float(np.trace(mcov)))
    cell_fit[model] = {
        "mean_mae_over_cells": float(np.mean(mean_err)),
        "trace_real": trace_real, "trace_model": trace_model,
        "trace_mae": float(np.mean(np.abs(np.array(trace_real) - np.array(trace_model)))),
    }

log["task3"] = {"models": results, "cell_fit": cell_fit, "delta_gate": float(delta),
                "sgn_f0": float(sgn_f0), "sgn_f1": float(sgn_f1)}

# ===========================================================================
# PLOTS
# ===========================================================================
# Plot A: distribution along gate axis v — c1 vs c0, split by food and pooled.
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
bins = np.linspace(g_tr.min(), g_tr.max(), 60)
for ax, fsel, title in [
    (axes[0], (ftr == 0), "food=0"),
    (axes[1], (ftr == 1), "food=1"),
    (axes[2], np.ones(len(ftr), bool), "POOLED over food"),
]:
    ax.hist(g_tr[fsel & (ctr == 0)], bins=bins, alpha=0.5, density=True, label="country=0", color="C0")
    ax.hist(g_tr[fsel & (ctr == 1)], bins=bins, alpha=0.5, density=True, label="country=1", color="C3")
    ax.set_title(f"<h2, v_gate>  ({title})")
    ax.set_xlabel("projection on gate axis v"); ax.legend()
axes[2].annotate(f"c1 BC={bc_c1:.3f}\nc0 BC={bc_c0:.3f}\n(>0.555 ~ bimodal)",
                 xy=(0.02, 0.7), xycoords="axes fraction", fontsize=9,
                 bbox=dict(boxstyle="round", fc="w"))
fig.suptitle("TASK 2: gate-axis projection — sign-flip makes POOLED country=0 bimodal, not country=1")
fig.tight_layout()
fig.savefig(os.path.join(RES, "round2_r5_gateaxis_dist.png"), dpi=110)
plt.close(fig)

# Plot B: variance ratio c1/c0 per PCA axis (pooled vs within-food0) — where is c1 tight?
fig, ax = plt.subplots(figsize=(9, 4.2))
idx = np.arange(K)
ax.bar(idx - 0.2, ratio_pooled, width=0.4, label="pooled c1/c0", color="C0")
ax.bar(idx + 0.2, ratio_f0, width=0.4, label="within food=0 c1/c0", color="C2")
ax.axhline(1.0, color="k", lw=0.8, ls="--")
ax.axvline(gate_dom_axis, color="C3", lw=1.2, ls=":", label=f"gate-dom axis ({gate_dom_axis})")
ax.set_yscale("log")
ax.set_xlabel("PCA axis"); ax.set_ylabel("variance ratio c1/c0 (log)")
ax.set_title("TASK 2: country=1 is TIGHTER (<1) on shell axes; on the gate axis pooled c0 is wide (sign-flip)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(RES, "round2_r5_varratio_axes.png"), dpi=110)
plt.close(fig)

# Plot C: 4-cell means in the (gate axis v, top shell axis) plane + AUC bars
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
# shell axis = top eigvec of (cov_c0 - cov_c1) in PCA space (variance-collapse dir)
covdiff = cov_of(ctr == 0) - cov_of(ctr == 1)
ew, ev = eigh(covdiff)
shell_ax = ev[:, -1]                     # largest c0-minus-c1 variance direction
# orthogonalize shell vs gate for a clean 2D plot
shell_perp = shell_ax - (shell_ax @ v_gate) * v_gate
shell_perp = shell_perp / norm(shell_perp)
gx = Ptr @ v_gate; sy = Ptr @ shell_perp
axc = axes[0]
colors = {(0,0):"#1f77b4",(0,1):"#17becf",(1,0):"#d62728",(1,1):"#ff7f0e"}
for cv in (0,1):
    for fv in (0,1):
        m = cell_mask(ctr, ftr, cv, fv)
        axc.scatter(gx[m][:400], sy[m][:400], s=4, alpha=0.25, color=colors[(cv,fv)],
                    label=f"c{cv}f{fv}")
for cv in (0,1):
    for fv in (0,1):
        m = cell_mask(ctr, ftr, cv, fv)
        axc.scatter(gx[m].mean(), sy[m].mean(), s=200, marker="*",
                    edgecolor="k", color=colors[(cv,fv)], zorder=5)
axc.set_xlabel("gate axis v"); axc.set_ylabel("shell axis (c0-c1 var collapse, ⊥v)")
axc.set_title("4-cell geometry: gate axis sign-flips (c1 stars swap sides); shell axis c1 collapses to 0")
axc.legend(fontsize=7, markerscale=2)

axb = axes[1]
mnames = ["M1\npure-gate", "M2\npure-shell", "M3\ncombined"]
aucs = [results["M1"]["country_auc"], results["M2"]["country_auc"], results["M3"]["country_auc"]]
bars = axb.bar(mnames, aucs, color=["#ff7f0e", "#2ca02c", "#9467bd"])
axb.axhline(0.99, color="k", ls="--", lw=1, label="real QDA ~0.99")
axb.axhline(auc_gated_lin, color="C1", ls=":", lw=1, label=f"gated-linear {auc_gated_lin:.2f}")
axb.set_ylim(0.4, 1.02); axb.set_ylabel("held-out country AUC")
for b, a in zip(bars, aucs):
    axb.text(b.get_x()+b.get_width()/2, a+0.005, f"{a:.3f}", ha="center", fontsize=9)
axb.set_title("Generative model held-out country-AUC")
axb.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(RES, "round2_r5_4cell_and_models.png"), dpi=110)
plt.close(fig)

# ===========================================================================
# SAVE LOG
# ===========================================================================
with open(os.path.join(RES, "round2_r5_raw.json"), "w") as fh:
    json.dump(log, fh, indent=2, default=float)

# ---- console summary ----
print("="*70)
print("TASK 1 — 4-CELL ANTI-PARALLEL TEST")
print(f"  ||d(c1-c0|food0)|| = {norm(d_f0):.3f}   ||d(c1-c0|food1)|| = {norm(d_f1):.3f}")
print(f"  cos(d_food0, d_food1) = {cos_anti:.3f}   (anti-parallel if ~ -1)")
print(f"  country offset / food offset magnitude = {log['task1']['country_vs_food_ratio']:.3f}")
print(f"  cell traces: " + ", ".join(f"c{cv}f{fv}={cells[(cv,fv)]['trace']:.2f}" for cv in (0,1) for fv in (0,1)))
print("="*70)
print("TASK 2 — TIGHT vs BIMODAL (along gate axis v)")
for k in ("c0f0","c1f0","c0f1","c1f1"):
    print(f"  {k}: mean={proj[k]['mean']:+.3f} std={proj[k]['std']:.3f}")
print(f"  POOLED  c0: mean={pooled['c0']['mean']:+.3f} std={pooled['c0']['std']:.3f} var={pooled['c0']['var']:.3f}")
print(f"  POOLED  c1: mean={pooled['c1']['mean']:+.3f} std={pooled['c1']['std']:.3f} var={pooled['c1']['var']:.3f}")
print(f"  bimodality coef pooled  c0={bc_c0:.3f}  c1={bc_c1:.3f}  (>0.555 ~ bimodal)")
print(f"  var-along-v frac from between-food sign-flip:  c0={frac_c0:.3f}  c1={frac_c1:.3f}")
print(f"  TOTAL PCA var ratio c1/c0 = {tot_var_c1/tot_var_c0:.3f}  (c1 tighter overall)")
print(f"  #axes c1 tighter (pooled)={int((ratio_pooled<1).sum())}/{K}   (within food=0)={int((ratio_f0<1).sum())}/{K}")
print("="*70)
print("TASK 3 — GENERATIVE MODELS (held-out TEST)")
for m in ("M1","M2","M3"):
    print(f"  {m}: country-AUC={results[m]['country_auc']:.3f}   mean test loglik={results[m]['mean_test_loglik']:.2f}   "
          f"cell-mean-MAE={cell_fit[m]['mean_mae_over_cells']:.3f}  trace-MAE={cell_fit[m]['trace_mae']:.3f}")
print(f"  ref ungated v-axis AUC = {auc_ungated:.3f}   gated-linear v-axis AUC = {auc_gated_lin:.3f}")
print("="*70)
print("DONE. raw -> results/round2_r5_raw.json")
