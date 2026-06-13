"""
R4 CAUSAL: confirm at the WEIGHT level that the network reads `country` (idx5) from
h2 via food(idx3)-gating (multiplicative / sign-flip), and quantify any radial/magnitude
contribution.

Path from h2:  h2 --Linear(layers[6])--> ReLU(layers[7]) --> h3 --Linear(layers[8]) row5--> country logit.

Run: CUDA_VISIBLE_DEVICES=-1 python r4_causal.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import json
import numpy as np
import torch, torch.nn as nn
from sklearn.metrics import roc_auc_score

torch.set_grad_enabled(True)
np.set_printoptions(suppress=True, precision=4, linewidth=140)

ROOT = r"C:\Users\Sidar\Desktop\puzzle\bluedot-tais-puzzle"
C = ROOT + r"\cache"
NAMES = ["number", "question", "color", "food", "sentiment", "country", "person", "body_part"]
COUNTRY, FOOD, SENT = 5, 3, 4

# ---------------- load model ----------------
class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(384, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 8))
    def forward(self, x): return self.layers(x)

m = Head(); m.load_state_dict(torch.load(ROOT + r"\model.pt", map_location="cpu", weights_only=False)); m.eval()

h2_np = np.load(C + r"\h2_test.npy").astype(np.float32)
Y = np.load(C + r"\labels_test.npy")
yC = Y[:, COUNTRY]; yF = Y[:, FOOD]; yS = Y[:, SENT]
h2 = torch.tensor(h2_np)

W6 = m.layers[6].weight.detach().numpy(); b6 = m.layers[6].bias.detach().numpy()  # (64,64)
W8 = m.layers[8].weight.detach().numpy(); b8 = m.layers[8].bias.detach().numpy()  # (8,64)
w_country = W8[COUNTRY]  # (64,) row5 readout over h3
b_country = b8[COUNTRY]

def head(x_t):
    """x_t: torch tensor (N,64) at h2 -> country logit numpy (N,)"""
    with torch.no_grad():
        return m.layers[6:](x_t).numpy()[:, COUNTRY]

base_logit = head(h2)
base_auc = roc_auc_score(yC, base_logit)
OUT = {"base_country_auc_from_h2": round(float(base_auc), 4)}
print(f"[base] country AUC from h2 = {base_auc:.4f}")

# =====================================================================
# TASK 1: GRADIENT GEOMETRY
#   grad of country logit wrt h2 at each test point. Sign-flips with food?
# =====================================================================
print("\n[1] GRADIENT GEOMETRY")
h2g = h2.clone().requires_grad_(True)
logit = m.layers[6:](h2g)[:, COUNTRY]      # (N,)
g = torch.autograd.grad(logit.sum(), h2g)[0].numpy()  # (N,64) per-point grad (sum -> each row is own grad)

gn = np.linalg.norm(g, axis=1)
g0 = g[yF == 0]; g1 = g[yF == 1]
mg0 = g0.mean(0); mg1 = g1.mean(0)
cos_food = float(mg0 @ mg1 / (np.linalg.norm(mg0) * np.linalg.norm(mg1)))

# overall mean grad vs per-food mean grad, and "constancy" of grad direction
mg_all = g.mean(0)
# average cosine of each point's grad to the global mean grad (constancy)
unit_g = g / (gn[:, None] + 1e-12)
cos_to_meanall = unit_g @ (mg_all / np.linalg.norm(mg_all))
# average cosine of each point's grad to its OWN-food mean grad
u0 = mg0 / np.linalg.norm(mg0); u1 = mg1 / np.linalg.norm(mg1)
cos_within = np.where(yF == 0, unit_g @ u0, unit_g @ u1)

OUT["grad"] = {
    "cos_meanGrad_food0_vs_food1": round(cos_food, 4),
    "mean_grad_norm": round(float(gn.mean()), 4),
    "mean_grad_norm_food0": round(float(gn[yF == 0].mean()), 4),
    "mean_grad_norm_food1": round(float(gn[yF == 1].mean()), 4),
    "norm_of_global_mean_grad": round(float(np.linalg.norm(mg_all)), 4),
    "norm_of_food0_mean_grad": round(float(np.linalg.norm(mg0)), 4),
    "norm_of_food1_mean_grad": round(float(np.linalg.norm(mg1)), 4),
    "avg_point_cos_to_global_meangrad": round(float(cos_to_meanall.mean()), 4),
    "avg_point_cos_to_ownfood_meangrad": round(float(cos_within.mean()), 4),
}
print(f"  cos( mean-grad(food=0), mean-grad(food=1) ) = {cos_food:+.4f}   (expect ~ -1 if gated)")
print(f"  mean |grad|: all={gn.mean():.3f} food0={gn[yF==0].mean():.3f} food1={gn[yF==1].mean():.3f}")
print(f"  |global mean grad|={np.linalg.norm(mg_all):.3f}  |food0 mean|={np.linalg.norm(mg0):.3f} |food1 mean|={np.linalg.norm(mg1):.3f}")
print(f"  avg point-grad cos to GLOBAL mean grad = {cos_to_meanall.mean():+.3f}  (low => not globally constant)")
print(f"  avg point-grad cos to OWN-FOOD mean grad = {cos_within.mean():+.3f}  (high => constant *within* food)")

# =====================================================================
# TASK 2: h3 UNIT ATTRIBUTION
#   layers[8] row5 weights over the 64 h3 units. Top-|w|. Trace each h3 unit
#   back through layers[6]+ReLU to its h2 input direction (= W6 row).
#   Identify a +country@food=0 unit vs +country@food=1 unit (sign-flip pair).
# =====================================================================
print("\n[2] h3 UNIT ATTRIBUTION (country readout = W8[5] over h3)")
# h3 activations
with torch.no_grad():
    h3 = m.layers[6:8](h2).numpy()   # (N,64) post-ReLU hidden3
# Each h3 unit j: preact = W6[j] . h2 + b6[j], then ReLU. Its CONTRIBUTION to country logit
# at a point = w_country[j] * h3[:,j]. Attribution = mean over points where unit is active.
contrib = h3 * w_country[None, :]       # (N,64) per-unit contribution to country logit
mean_contrib = contrib.mean(0)
# How does each unit's contribution correlate with country, split by food?
order = np.argsort(-np.abs(w_country))
top = order[:12]

# For each top unit, characterize: w_country sign, fraction active, and the
# food-split of its activation, and corr of (w_country*h3) with country label within each food.
def split_stats(j):
    a = h3[:, j]
    act = a > 0
    # mean activation by (food,country)
    grid = {}
    for f in (0, 1):
        for c in (0, 1):
            mask = (yF == f) & (yC == c)
            grid[(f, c)] = float(a[mask].mean())
    # does this unit's contribution help separate country POSITIVELY within each food?
    cj = contrib[:, j]
    def auc_safe(mask):
        yy = yC[mask]
        if len(np.unique(yy)) < 2: return np.nan
        return roc_auc_score(yy, cj[mask])
    return dict(j=int(j), w=float(w_country[j]), frac_active=float(act.mean()),
                act_f0c0=grid[(0,0)], act_f0c1=grid[(0,1)],
                act_f1c0=grid[(1,0)], act_f1c1=grid[(1,1)],
                auc_food0=auc_safe(yF==0), auc_food1=auc_safe(yF==1))

unit_rows = [split_stats(j) for j in top]
OUT["h3_units"] = unit_rows
print("  unit  w_country  frac_act |  act f0c0/f0c1   f1c0/f1c1 |  contribAUC food0/food1")
for r in unit_rows:
    print(f"  h3[{r['j']:2d}] {r['w']:+.3f}   {r['frac_active']:.2f}  | "
          f"{r['act_f0c0']:5.2f}/{r['act_f0c1']:5.2f}  {r['act_f1c0']:5.2f}/{r['act_f1c1']:5.2f} | "
          f"{r['auc_food0']:.3f}/{r['auc_food1']:.3f}")

# Trace top units to their h2 input direction (W6 row) and check that direction's
# country-separating sign within each food (diff of mean projection).
print("\n  -- top units' h2 input directions (W6 row) and food-conditioned country sign --")
trace_rows = []
for r in unit_rows[:8]:
    j = r["j"]
    v = W6[j]; vu = v / np.linalg.norm(v)
    proj = h2_np @ vu     # projection of h2 onto this unit's input axis
    # country mean gap of the projection, within each food
    def gap(f):
        mm = yF == f
        return float(proj[mm & (yC == 1)].mean() - proj[mm & (yC == 0)].mean())
    g0_, g1_ = gap(0), gap(1)
    trace_rows.append(dict(j=int(j), w=float(w_country[j]),
                           proj_gap_food0=round(g0_, 3), proj_gap_food1=round(g1_, 3),
                           gap_signflip=bool(np.sign(g0_) != np.sign(g1_))))
    print(f"  h3[{j:2d}] w={w_country[j]:+.3f}  proj country-gap food0={g0_:+.3f} food1={g1_:+.3f}  "
          f"{'SIGN-FLIP' if np.sign(g0_)!=np.sign(g1_) else 'same-sign'}")
OUT["h3_trace"] = trace_rows

# =====================================================================
# Build axes for ablation: food dir, gating axis, radial axes (in RAW h2 coords,
# but for ablation we operate directly on h2 by removing rank-1 / rank-k subspaces).
# Use TRAIN-free, simple constructions from test (directions are stable).
# =====================================================================
def unit(v):
    n = np.linalg.norm(v); return v / n if n > 0 else v

# (a) FOOD direction in h2: diff-of-means of food (raw h2)
food_dir = unit(h2_np[yF == 1].mean(0) - h2_np[yF == 0].mean(0))

# (b) GATING / shell axis: signed country axis. within-food country diff-of-means,
#     sign-aligned then averaged -> the sign-flip axis.
def dom(mask_extra):
    mm = mask_extra
    return h2_np[mm & (yC == 1)].mean(0) - h2_np[mm & (yC == 0)].mean(0)
c_f0 = dom(yF == 0); c_f1 = dom(yF == 1)
# they are anti-parallel; signed-average = unit(u0 - u1)
u_f0 = unit(c_f0); u_f1 = unit(c_f1)
gating_axis = unit(u_f0 - u_f1)
cos_cf = float(u_f0 @ u_f1)

# (c) radial / variance-difference axes: eigenvectors of (Cov_c0 - Cov_c1) WITHIN food
#     (food-independent shell). Pool within-food-centered data.
Xc = h2_np.copy().astype(np.float64)
# center within each food to remove the gating mean structure
for f in (0, 1):
    mm = yF == f
    Xc[mm] -= Xc[mm].mean(0)
# now per-country covariance of the food-centered data
C0 = np.cov(Xc[yC == 0].T); C1 = np.cov(Xc[yC == 1].T)
Cd = C0 - C1
evals, evecs = np.linalg.eigh(Cd)         # ascending
# largest |eig| variance-difference axes (country0 spread vs country1 collapse => large +eig)
idx_rad = np.argsort(-np.abs(evals))
radial_axes = evecs[:, idx_rad[:4]].T      # (4,64) top-4 radial axes
# make sure gating axis is excluded from radial set numerically: report overlaps
OUT["axes"] = {
    "cos_country_axis_food0_vs_food1": round(cos_cf, 4),
    "cos_gating_vs_food": round(float(gating_axis @ food_dir), 4),
    "cos_gating_vs_radial_top1": round(float(abs(gating_axis @ radial_axes[0])), 4),
    "radial_top_eigs": [round(float(evals[i]), 3) for i in idx_rad[:6]],
}
print("\n  axis overlaps: cos(country@f0, country@f1) =", round(cos_cf, 4),
      "| cos(gating,food)=", round(float(gating_axis @ food_dir), 4),
      "| |cos(gating,radial0)|=", round(float(abs(gating_axis @ radial_axes[0])), 4))

# =====================================================================
# TASK 3: CAUSAL ABLATIONS
#   Remove a subspace from h2 (project out), run head, measure country AUC drop.
# =====================================================================
print("\n[3] CAUSAL ABLATIONS (project subspace out of h2, re-run head, measure country AUC)")
def project_out(X, dirs):
    """dirs: list/array of (k,64) orthonormalized basis. Remove its span from X."""
    B = np.atleast_2d(np.array(dirs))
    # orthonormalize
    Q, _ = np.linalg.qr(B.T)   # (64,k)
    P = Q @ Q.T
    return X - X @ P.T          # remove projection

def ablate_auc(dirs, mean_fill=None):
    Xab = project_out(h2_np, dirs)
    if mean_fill is not None:
        # add back the mean component along removed dirs (ablate to dataset mean, not 0)
        B = np.atleast_2d(np.array(dirs)); Q, _ = np.linalg.qr(B.T)
        mu = h2_np.mean(0)
        Xab = Xab + (mu @ Q) @ Q.T
    logit = head(torch.tensor(Xab.astype(np.float32)))
    return roc_auc_score(yC, logit)

ablations = {
    "none": base_auc,
    "remove_food_dir": ablate_auc([food_dir]),
    "remove_gating_axis": ablate_auc([gating_axis]),
    "remove_food+gating": ablate_auc([food_dir, gating_axis]),
    "remove_radial_top1": ablate_auc([radial_axes[0]]),
    "remove_radial_top2": ablate_auc(radial_axes[:2]),
    "remove_radial_top4": ablate_auc(radial_axes[:4]),
    "remove_gating+radial4": ablate_auc(np.vstack([gating_axis, radial_axes[:4]])),
    "remove_food+gating+radial4": ablate_auc(np.vstack([food_dir, gating_axis, radial_axes[:4]])),
}
# also mean-fill variants for the key ones (ablate to mean instead of 0)
ablations_meanfill = {
    "remove_food_dir(meanfill)": ablate_auc([food_dir], mean_fill=True),
    "remove_gating_axis(meanfill)": ablate_auc([gating_axis], mean_fill=True),
    "remove_radial_top4(meanfill)": ablate_auc(radial_axes[:4], mean_fill=True),
}
OUT["ablations"] = {k: round(float(v), 4) for k, v in ablations.items()}
OUT["ablations"].update({k: round(float(v), 4) for k, v in ablations_meanfill.items()})
for k, v in ablations.items():
    print(f"  {k:34s} country AUC = {v:.4f}   (drop {base_auc - v:+.4f})")
for k, v in ablations_meanfill.items():
    print(f"  {k:34s} country AUC = {v:.4f}   (drop {base_auc - v:+.4f})")

# =====================================================================
# TASK 4: GATE FORM -- is the country logit ~ (country-axis proj) gated by food sign?
#   Test the ReLU unit set: does the active set of the country-feeding h3 units
#   split by food? i.e. unit A active when food=0, unit B active when food=1.
# =====================================================================
print("\n[4] GATE FORM: do country-feeding ReLU units split by food? (multiplicative gate vs norm)")
# Use the two strongest OPPOSITE-sign-gap units from the trace to test the pair hypothesis.
# Build s = projection on the canonical country axis (food=0 within-food diff of means).
s = h2_np @ u_f0        # canonical country score (positive=country@food0)
# multiplicative model: country logit ~ s * pm(food)
pm = lambda v: 2.0 * v - 1.0
def auc_of(score): return roc_auc_score(yC, score)
gate_tests = {
    "s_alone": auc_of(s),
    "s_times_pm_food": auc_of(s * pm(yF)),
    "abs_s": auc_of(np.abs(s)),
    "neg_abs_s": auc_of(-np.abs(s)),
}
OUT["gate_form"] = {k: round(float(v), 4) for k, v in gate_tests.items()}
for k, v in gate_tests.items():
    print(f"  {k:20s} AUC={v:.4f}")

# Per-h3-unit: among the top-|w| country units, classify each as a food-gated unit:
# a unit is "food-0 gated" if it is much more active (and country-discriminative) when food=0.
print("\n  -- food-conditioned activation of top country-feeding units (gate evidence) --")
gate_unit_rows = []
for r in unit_rows[:10]:
    j = r["j"]
    a = h3[:, j]
    fa0 = float((a[yF == 0] > 0).mean()); fa1 = float((a[yF == 1] > 0).mean())
    ma0 = float(a[yF == 0].mean()); ma1 = float(a[yF == 1].mean())
    # contribution sign to country: w_country[j]; country-gap of activation within active food
    gate_unit_rows.append(dict(j=int(j), w=float(w_country[j]),
                               fracactive_food0=round(fa0,3), fracactive_food1=round(fa1,3),
                               meanact_food0=round(ma0,3), meanact_food1=round(ma1,3)))
    print(f"  h3[{j:2d}] w={w_country[j]:+.3f}  active%% f0={fa0:.2f} f1={fa1:.2f}  "
          f"meanact f0={ma0:.3f} f1={ma1:.3f}  "
          f"{'-> FOOD0-GATED' if ma0>1.4*max(ma1,1e-6) else ('-> FOOD1-GATED' if ma1>1.4*max(ma0,1e-6) else '-> ungated')}")
OUT["gate_units"] = gate_unit_rows

# Decompose the country logit reconstruction: country_logit ~= b_country + sum_j w_country[j]*h3[:,j]
# Group the contribution by food-gated unit sets to see additive structure.
recon = h3 @ w_country + b_country
print(f"\n  recon check: corr(recon, base_logit)={np.corrcoef(recon, base_logit)[0,1]:.4f}")
# split units into those whose mean activation is higher for food0 vs food1
mean_a = h3.mean(0)
hi0 = np.array([h3[yF==0,j].mean() > h3[yF==1,j].mean() for j in range(64)])
contrib_food0units = h3[:, hi0] @ w_country[hi0]
contrib_food1units = h3[:, ~hi0] @ w_country[~hi0]
OUT["gate_form"]["recon_corr"] = round(float(np.corrcoef(recon, base_logit)[0,1]), 4)
OUT["gate_form"]["auc_food0unit_contrib"] = round(float(roc_auc_score(yC, contrib_food0units)), 4)
OUT["gate_form"]["auc_food1unit_contrib"] = round(float(roc_auc_score(yC, contrib_food1units)), 4)
# crucial: within food=0, the food0-units carry country; within food=1, the food1-units carry it
for f in (0,1):
    mm = yF==f
    a0 = roc_auc_score(yC[mm], contrib_food0units[mm])
    a1 = roc_auc_score(yC[mm], contrib_food1units[mm])
    OUT["gate_form"][f"food{f}: food0units_auc"] = round(float(a0),4)
    OUT["gate_form"][f"food{f}: food1units_auc"] = round(float(a1),4)
    print(f"  within food={f}: food0-units country-AUC={a0:.3f}  food1-units country-AUC={a1:.3f}")

# =====================================================================
# Save
# =====================================================================
with open(ROOT + r"\results\round2_r4_raw.json", "w") as f:
    json.dump(OUT, f, indent=2, default=str)
print("\nSaved results/round2_r4_raw.json")
