"""
R2 GATE analysis: which features gate `country` (idx5) at layer L=h2, and is the
gating a clean sign-flip (XOR) or a general rotation?

Run: CUDA_VISIBLE_DEVICES=-1 python r2_gate.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import json
import itertools
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

RNG = np.random.RandomState(0)
NAMES = ["number", "question", "color", "food", "sentiment", "country", "person", "body_part"]
COUNTRY = 5
OTHERS = [0, 1, 2, 3, 4, 6, 7]

# ---------- load ----------
h2_tr = np.load("cache/h2_train.npy");  lab_tr = np.load("cache/labels_train.npy")
h2_te = np.load("cache/h2_test.npy");   lab_te = np.load("cache/labels_test.npy")
y_tr = lab_tr[:, COUNTRY];              y_te = lab_te[:, COUNTRY]

# Standardize using TRAIN stats (shared scaler so directions live in same coords)
scaler = StandardScaler().fit(h2_tr)
X_tr = scaler.transform(h2_tr)
X_te = scaler.transform(h2_te)

out = {}  # results dict for the md

def diff_of_means(X, y):
    """Diff-of-means direction (country=1 minus country=0), unit-normalized."""
    mu1 = X[y == 1].mean(0)
    mu0 = X[y == 0].mean(0)
    d = mu1 - mu0
    n = np.linalg.norm(d)
    return d / n if n > 0 else d

def logreg_dir(X, y):
    """Logistic-regression weight direction, unit-normalized."""
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X, y)
    w = clf.coef_[0]
    n = np.linalg.norm(w)
    return (w / n if n > 0 else w), clf

def cell_auc(Xtr, ytr, Xte, yte):
    """Train logreg on a cell's train half, eval AUC on its test half."""
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return np.nan
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, ytr)
    p = clf.decision_function(Xte)
    return roc_auc_score(yte, p)

# =====================================================================
# PART 0: global linear AUC of country (sanity)
# =====================================================================
clf_g = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)
global_auc = roc_auc_score(y_te, clf_g.decision_function(X_te))
out["global_country_auc"] = round(float(global_auc), 4)
print(f"[0] GLOBAL country linear AUC (test) = {global_auc:.4f}")

# =====================================================================
# PART 1: single-gate search
#   For each other feature g:
#     - within-g=0 and within-g=1 train/eval country AUC
#     - cosine between country diff-of-means dir in g=0 vs g=1
# =====================================================================
print("\n[1] SINGLE-GATE SEARCH")
single = []
for g in OTHERS:
    rows = {}
    aucs = {}
    dirs_dm = {}
    dirs_lr = {}
    for v in [0, 1]:
        mtr = lab_tr[:, g] == v
        mte = lab_te[:, g] == v
        Xtr, ytr = X_tr[mtr], y_tr[mtr]
        Xte, yte = X_te[mte], y_te[mte]
        aucs[v] = cell_auc(Xtr, ytr, Xte, yte)
        dirs_dm[v] = diff_of_means(Xtr, ytr)
        dirs_lr[v], _ = logreg_dir(Xtr, ytr)
        rows[v] = (int(mtr.sum()), int(mte.sum()))
    cos_dm = float(dirs_dm[0] @ dirs_dm[1])
    cos_lr = float(dirs_lr[0] @ dirs_lr[1])
    mean_auc = np.nanmean([aucs[0], aucs[1]])
    single.append(dict(g=g, name=NAMES[g], auc0=aucs[0], auc1=aucs[1],
                       mean_auc=mean_auc, cos_dm=cos_dm, cos_lr=cos_lr,
                       n0=rows[0], n1=rows[1]))
    print(f"  gate={NAMES[g]:10s} AUC0={aucs[0]:.3f} AUC1={aucs[1]:.3f} "
          f"mean={mean_auc:.3f}  cos(dm)={cos_dm:+.3f} cos(lr)={cos_lr:+.3f}")

# gating strength = mean within-cell AUC; a TRUE gate also has cos near -1
single_sorted = sorted(single, key=lambda r: -r["mean_auc"])
out["single_gate"] = single_sorted

# =====================================================================
# PART 2: pair / triple gate search -- find minimal set that
#   (a) pushes within-cell AUC -> ~1.0  AND
#   (b) makes the country direction CONSISTENT within each joint cell.
# =====================================================================
def joint_cell_analysis(gates):
    """gates: tuple of feature idxs. Returns per-cell AUCs + dir matrix."""
    combos = list(itertools.product([0, 1], repeat=len(gates)))
    cells = {}
    dirs = {}
    for combo in combos:
        mtr = np.ones(len(lab_tr), bool)
        mte = np.ones(len(lab_te), bool)
        for gi, v in zip(gates, combo):
            mtr &= lab_tr[:, gi] == v
            mte &= lab_te[:, gi] == v
        Xtr, ytr = X_tr[mtr], y_tr[mtr]
        Xte, yte = X_te[mte], y_te[mte]
        a = cell_auc(Xtr, ytr, Xte, yte)
        cells[combo] = dict(ntr=int(mtr.sum()), nte=int(mte.sum()), auc=a)
        # direction from the (train+test pooled for a stable dir estimate)
        Xall = np.vstack([Xtr, Xte]); yall = np.concatenate([ytr, yte])
        dirs[combo] = diff_of_means(Xall, yall)
    return cells, dirs

def cos_matrix(dirs):
    keys = list(dirs.keys())
    M = np.array([[dirs[a] @ dirs[b] for b in keys] for a in keys])
    return keys, M

print("\n[2] PAIR / TRIPLE GATE SEARCH (target: high within-cell AUC + consistent dir)")
pair_results = []
# try all single, pairs, and the food/sentiment triples
candidate_sets = [(g,) for g in OTHERS] + list(itertools.combinations(OTHERS, 2))
# prioritize food/sentiment combos for triples
candidate_sets += [(3, 4, g) for g in [0, 1, 2, 6, 7]]
for gates in candidate_sets:
    cells, dirs = joint_cell_analysis(gates)
    aucs = [c["auc"] for c in cells.values() if not np.isnan(c["auc"])]
    min_auc = float(np.min(aucs)) if aucs else np.nan
    mean_auc = float(np.mean(aucs)) if aucs else np.nan
    min_ntr = min(c["ntr"] for c in cells.values())
    pair_results.append(dict(gates=gates, names=[NAMES[g] for g in gates],
                             mean_auc=mean_auc, min_auc=min_auc, min_ntr=min_ntr,
                             ncells=len(cells)))

pair_results.sort(key=lambda r: -r["min_auc"])
out["set_search"] = pair_results
print("  Top sets by MIN within-cell AUC:")
for r in pair_results[:12]:
    print(f"   {'x'.join(r['names']):28s} mean_auc={r['mean_auc']:.3f} "
          f"min_auc={r['min_auc']:.3f} ncells={r['ncells']} min_ntr={r['min_ntr']}")

# =====================================================================
# PART 3: SIGN-FLIP vs ROTATION
#   Full cosine matrix between per-cell country directions for the best
#   minimal gate set. Pure XOR/sign-flip => off-diagonals are exactly +-1.
# =====================================================================
print("\n[3] SIGN-FLIP vs ROTATION  (cosine matrix between per-cell country dirs)")

# food alone (the prior-claimed gate)
cells_f, dirs_f = joint_cell_analysis((3,))
keys_f, M_f = cos_matrix(dirs_f)
print("  -- gate = food alone --")
print("   cells:", keys_f)
print("   cos matrix:\n", np.round(M_f, 3))
out["cos_food"] = dict(keys=[str(k) for k in keys_f], M=M_f.round(4).tolist())

# food x sentiment (4 cells)
cells_fs, dirs_fs = joint_cell_analysis((3, 4))
keys_fs, M_fs = cos_matrix(dirs_fs)
print("  -- gate = food x sentiment (4 cells) --")
print("   cells (food,sent):", keys_fs)
print("   cos matrix:\n", np.round(M_fs, 3))
out["cos_food_sent"] = dict(keys=[str(k) for k in keys_fs], M=M_fs.round(4).tolist())

# Quantify: how close are off-diagonal cosines to +-1 (sign flip) vs spread (rotation)?
def offdiag_stats(M):
    iu = np.triu_indices_from(M, k=1)
    vals = M[iu]
    return dict(vals=vals.round(3).tolist(),
                mean_abs=float(np.mean(np.abs(vals))),
                min=float(vals.min()), max=float(vals.max()),
                frac_near_pm1=float(np.mean(np.abs(np.abs(vals) - 1) < 0.1)))

out["offdiag_food"] = offdiag_stats(M_f)
out["offdiag_food_sent"] = offdiag_stats(M_fs)
print("  food off-diag stats:", out["offdiag_food"])
print("  food_sent off-diag stats:", out["offdiag_food_sent"])

# =====================================================================
# PART 4: IS THE GATE MULTIPLICATIVE?
#   Hypothesis: country_signal ~ <h2, v_country> * sign(food)  (XOR/product).
#   Build v_country as the diff-of-means within food=0 (a canonical country axis),
#   then test:
#     s = X @ v_country
#     (a) s alone               -> AUC for country  (should be poor / flipped)
#     (b) s * food_label        -> AUC               (multiplicative w/ true gate)
#     (c) s * sign(food proj)   -> AUC               (multiplicative w/ readout gate)
#     (d) additive baseline [s, food] logistic       -> AUC
# =====================================================================
print("\n[4] MULTIPLICATIVE-GATE TEST")
# canonical country axis from food=0 train cell
m_f0 = lab_tr[:, 3] == 0
v_country = diff_of_means(X_tr[m_f0], y_tr[m_f0])  # unit
# also a food readout axis (food is linear): logreg weight
food_dir, _ = logreg_dir(X_tr, lab_tr[:, 3])

s_tr = X_tr @ v_country
s_te = X_te @ v_country
food_proj_tr = X_tr @ food_dir
food_proj_te = X_te @ food_dir
food_lab_tr = lab_tr[:, 3].astype(float)
food_lab_te = lab_te[:, 3].astype(float)

def auc_of(score, y):
    return roc_auc_score(y, score)

# map {0,1} food to {-1,+1} for a genuine sign multiply
def pm(v):  # to +-1
    return 2 * v - 1

res4 = {}
res4["s_alone"] = auc_of(s_te, y_te)
res4["s_times_foodlabel_pm"] = auc_of(s_te * pm(food_lab_te), y_te)
res4["s_times_sign_foodproj"] = auc_of(s_te * np.sign(food_proj_te), y_te)
# additive baseline: 2-feature logreg on [s, food_proj]
add_clf = LogisticRegression(max_iter=2000).fit(
    np.c_[s_tr, food_proj_tr, food_lab_tr], y_tr)
res4["additive_s_food"] = auc_of(
    add_clf.decision_function(np.c_[s_te, food_proj_te, food_lab_te]), y_te)
# multiplicative as an explicit interaction feature fed to logreg
inter_tr = np.c_[s_tr, food_proj_tr, s_tr * pm(food_lab_tr)]
inter_te = np.c_[s_te, food_proj_te, s_te * pm(food_lab_te)]
mult_clf = LogisticRegression(max_iter=2000).fit(inter_tr, y_tr)
res4["logreg_with_interaction"] = auc_of(mult_clf.decision_function(inter_te), y_te)

# Now with food x sentiment: does sign also flip with sentiment? test product of both signs
sent_lab_te = lab_te[:, 4].astype(float)
res4["s_times_food_x_sent_signs"] = auc_of(
    s_te * pm(food_lab_te) * pm(sent_lab_te), y_te)
res4["s_times_foodlabel_only_vs_sent"] = res4["s_times_foodlabel_pm"]

for k, v in res4.items():
    print(f"  {k:32s} AUC={v:.4f}")
out["multiplicative"] = {k: round(float(v), 4) for k, v in res4.items()}

# =====================================================================
# PART 5: variance accounting -- gating-explained vs residual
#   Compare global country AUC, single-gate(food) mean cell AUC,
#   food x sentiment min cell AUC, and a "fully conditioned on all 7" cell AUC.
# =====================================================================
print("\n[5] VARIANCE / GATING ACCOUNTING")
# fully conditioning on all 7 others (micro-cells); only where cell has both classes & enough n
def full_condition_auc(min_n=30):
    # group test by full 7-tuple of others; pool train dirs are not used --
    # instead measure: within each micro-cell is country still separable?
    # Use a single global axis fit WITHIN each cell via diff-of-means on test? too small.
    # Instead: train one logreg per (food,sent,color,...) cell is infeasible (too many).
    # Approx: report food x sentiment x color (8 cells) min AUC.
    pass

cells_fsc, _ = joint_cell_analysis((3, 4, 2))
aucs_fsc = [c["auc"] for c in cells_fsc.values() if not np.isnan(c["auc"])]
out["food_sent_color_minauc"] = round(float(np.min(aucs_fsc)), 4)
out["food_sent_color_meanauc"] = round(float(np.mean(aucs_fsc)), 4)
print(f"  food x sentiment x color: mean cell AUC={np.mean(aucs_fsc):.3f} "
      f"min={np.min(aucs_fsc):.3f} (8 cells)")

# How much does adding sentiment to food help the WORST cell?
food_cells = [c["auc"] for c in cells_f.values()]
fs_cells = [c["auc"] for c in cells_fs.values()]
out["food_min_auc"] = round(float(np.min(food_cells)), 4)
out["food_sent_min_auc"] = round(float(np.min(fs_cells)), 4)
print(f"  food alone:        cell AUCs={np.round(food_cells,3)}  min={np.min(food_cells):.3f}")
print(f"  food x sentiment:  cell AUCs={np.round(fs_cells,3)}  min={np.min(fs_cells):.3f}")

# =====================================================================
# PART 6: Per-cell country directions across food x sent: are they a
#   sign-flip lattice? Project each cell dir onto cell(0,0) dir to see the
#   pattern of signs. Also test if sentiment-flip and food-flip commute.
# =====================================================================
print("\n[6] GATING SYMMETRY GROUP")
base = dirs_fs[(0, 0)]
for combo, d in dirs_fs.items():
    print(f"   cell food={combo[0]} sent={combo[1]}: cos to (0,0) = {base @ d:+.3f}")
# Build the implied sign table
sign_table = {c: float(np.sign(base @ d)) for c, d in dirs_fs.items()}
out["sign_table_food_sent"] = {str(k): v for k, v in sign_table.items()}

# Test XOR structure: is sign == (-1)^(food XOR sent)?  or (-1)^food only? etc.
def xor_pred(c, use_food, use_sent):
    e = 0
    if use_food: e ^= c[0]
    if use_sent: e ^= c[1]
    return (-1.0) ** e
for label, (uf, us) in [("food_only", (1, 0)), ("sent_only", (0, 1)),
                         ("food_XOR_sent", (1, 1))]:
    match = all(np.sign(base @ dirs_fs[c]) == xor_pred(c, uf, us) for c in dirs_fs)
    # need to align overall sign; allow global flip
    preds = np.array([xor_pred(c, uf, us) for c in dirs_fs])
    obs = np.array([np.sign(base @ dirs_fs[c]) for c in dirs_fs])
    match = bool(np.all(obs == preds) or np.all(obs == -preds))
    print(f"   sign pattern == {label}?  {match}")
    out[f"sign_is_{label}"] = match

# ---------- save ----------
with open("results/round2_r2.json", "w") as f:
    json.dump(out, f, indent=2, default=str)
print("\nSaved results/round2_r2.json")
