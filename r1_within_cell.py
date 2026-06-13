"""
R1 DECISIVE TEST: does conditioning LINEARIZE country, or does the radial shell
persist WITHIN cells?

For country (feature idx 5) at Layer L = h2 (post-ReLU hidden 2, 64-d):
  LDA (linear, class-mean based) vs QDA (quadratic, covariance based)

If conditioning on `food` (and more) makes country's class MEANS separate
(LDA AUC -> ~0.99, ||dmu_within|| large) => GATING / superposition (B).
If QDA still >> LDA WITHIN every cell (means still coincide within cells)
=> intrinsic RADIAL SHELL (A), not explained by gating.

We sweep conditioning granularity: none -> food -> food x sentiment -> full
conjunction of the other 7 labels, tracking LDA AUC, QDA AUC, ||dmu||, gap.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import json
import numpy as np
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis as LDA,
    QuadraticDiscriminantAnalysis as QDA,
)
from sklearn.metrics import roc_auc_score

C = r"C:/Users/Sidar/Desktop/puzzle/bluedot-tais-puzzle/cache"
NAMES = ["number", "question", "color", "food", "sentiment",
         "country", "person", "body_part"]
COUNTRY = 5
FOOD = 3
SENT = 4
QDA_REG = 0.01          # shrinkage so per-cell covariances stay invertible
MIN_CELL = 40           # min samples (per split) for a cell to be scored
RNG = np.random.default_rng(0)

Xtr = np.load(C + "/h2_train.npy")
Xte = np.load(C + "/h2_test.npy")
Ytr = np.load(C + "/labels_train.npy")
Yte = np.load(C + "/labels_test.npy")


def safe_auc(ytrue, score):
    if len(np.unique(ytrue)) < 2:
        return np.nan
    return roc_auc_score(ytrue, score)


def fit_eval(Xtr_, ytr_, Xte_, yte_):
    """Train LDA & QDA on train slice, eval AUC on test slice.
    Returns (lda_auc, qda_auc, dmu_train, n_train_pos, n_train_neg)."""
    res = dict(lda=np.nan, qda=np.nan, dmu=np.nan,
               ntr1=int((ytr_ == 1).sum()), ntr0=int((ytr_ == 0).sum()),
               nte1=int((yte_ == 1).sum()), nte0=int((yte_ == 0).sum()))
    # need both classes in train AND test
    if res["ntr1"] < 2 or res["ntr0"] < 2 or res["nte1"] < 1 or res["nte0"] < 1:
        return res
    m1 = Xtr_[ytr_ == 1].mean(0)
    m0 = Xtr_[ytr_ == 0].mean(0)
    res["dmu"] = float(np.linalg.norm(m1 - m0))
    try:
        l = LDA().fit(Xtr_, ytr_)
        res["lda"] = safe_auc(yte_, l.predict_proba(Xte_)[:, 1])
    except Exception:
        pass
    try:
        q = QDA(reg_param=QDA_REG).fit(Xtr_, ytr_)
        res["qda"] = safe_auc(yte_, q.predict_proba(Xte_)[:, 1])
    except Exception:
        pass
    return res


def pooled_within(cells):
    """Given a list of per-cell result dicts (each already fit/eval'd within
    its own cell on its own splits), produce sample-weighted summary over cells
    that were scorable. Weight by # test samples in the cell."""
    rows = [c for c in cells if not np.isnan(c["lda"]) and not np.isnan(c["qda"])]
    if not rows:
        return None
    w = np.array([c["nte1"] + c["nte0"] for c in rows], float)
    lda = np.array([c["lda"] for c in rows])
    qda = np.array([c["qda"] for c in rows])
    dmu = np.array([c["dmu"] for c in rows])
    cov_te = sum(c["nte1"] + c["nte0"] for c in rows)
    cov_tr = sum(c["ntr1"] + c["ntr0"] for c in rows)
    return dict(
        n_cells_scored=len(rows),
        n_cells_total=len(cells),
        lda=float(np.average(lda, weights=w)),
        qda=float(np.average(qda, weights=w)),
        dmu=float(np.average(dmu, weights=w)),
        gap=float(np.average(qda - lda, weights=w)),
        lda_unw=float(lda.mean()),
        qda_unw=float(qda.mean()),
        coverage_test=cov_te,
        coverage_train=cov_tr,
        frac_test_covered=cov_te / len(Xte),
    )


def cells_by_key(key_idx):
    """key_idx: list of feature indices to condition on (the 'cell' identity).
    For each distinct combination of those feature values, fit/eval country
    WITHIN that cell using that cell's own train (from Xtr) and test (from Xte)
    slices. Returns list of per-cell result dicts (with the cell key)."""
    yctr = Ytr[:, COUNTRY]
    ycte = Yte[:, COUNTRY]
    if key_idx:
        Ktr = Ytr[:, key_idx]
        Kte = Yte[:, key_idx]
        keys = np.unique(np.vstack([Ktr, Kte]), axis=0)
    else:
        keys = np.zeros((1, 0), int)
    out = []
    for k in keys:
        if key_idx:
            mtr = np.all(Ktr == k, axis=1)
            mte = np.all(Kte == k, axis=1)
        else:
            mtr = np.ones(len(Xtr), bool)
            mte = np.ones(len(Xte), bool)
        r = fit_eval(Xtr[mtr], yctr[mtr], Xte[mte], ycte[mte])
        r["key"] = k.tolist()
        out.append(r)
    return out


# ---------------------------------------------------------------------------
# 1) GLOBAL (no conditioning)
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 1: GLOBAL (no conditioning)")
g = cells_by_key([])[0]
print(f"  country  LDA={g['lda']:.4f}  QDA={g['qda']:.4f}  "
      f"||dmu||={g['dmu']:.4f}  gap={g['qda']-g['lda']:.4f}")
# also report global variance asymmetry (shell signature)
y = Ytr[:, COUNTRY]
vr = np.trace(np.cov(Xtr[y == 1].T)) / np.trace(np.cov(Xtr[y == 0].T))
print(f"  variance ratio trace(cov|c=1)/trace(cov|c=0) = {vr:.3f}  "
      f"(<1 => country=1 is the tight inner cluster)")

# ---------------------------------------------------------------------------
# 2) WITHIN food (food=0, food=1 separately)
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 2: WITHIN food (condition on food)")
food_cells = cells_by_key([FOOD])
for c in food_cells:
    print(f"  food={c['key']}  LDA={c['lda']:.4f}  QDA={c['qda']:.4f}  "
          f"||dmu||={c['dmu']:.4f}  gap={c['qda']-c['lda']:.4f}  "
          f"(ntr={c['ntr0']+c['ntr1']}, nte={c['nte0']+c['nte1']})")
food_sum = pooled_within(food_cells)

# ---------------------------------------------------------------------------
# 3a) WITHIN food x sentiment (4 cells)
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 3a: WITHIN food x sentiment (4 cells)")
fs_cells = cells_by_key([FOOD, SENT])
for c in fs_cells:
    print(f"  food,sent={c['key']}  LDA={c['lda']:.4f}  QDA={c['qda']:.4f}  "
          f"||dmu||={c['dmu']:.4f}  gap={c['qda']-c['lda']:.4f}  "
          f"(ntr={c['ntr0']+c['ntr1']}, nte={c['nte0']+c['nte1']})")
fs_sum = pooled_within(fs_cells)

# ---------------------------------------------------------------------------
# 3b) WITHIN FULL conjunction of the other 7 labels (2^7 = 128 cells).
#     Cells are tiny (~55 train / ~12 test each), so per-cell test AUC is not
#     estimable. Instead we POOL predictions across cells and compute ONE AUC
#     over all pooled test points. Two complementary estimators:
#       (i)  holdout: fit on cell's train slice, predict its test slice,
#            concatenate (y,score) across cells -> single pooled AUC.
#       (ii) cross-fitted on TRAIN only: 5-fold within each cell (more data,
#            tighter estimate). Posteriors from LDA/QDA are calibrated so
#            concatenating them across cells is valid for a pooled ROC.
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 3b: WITHIN FULL conjunction of other 7 labels (pooled, <=128 cells)")
from sklearn.model_selection import StratifiedKFold
other7 = [i for i in range(8) if i != COUNTRY]


def pooled_predictions(key_idx, mode="holdout"):
    """Pool per-cell posteriors into one ROC.
    mode='holdout': train on Xtr cell, predict Xte cell.
    mode='cv'     : 5-fold CV within each cell's TRAIN data (uses Xtr only).
    Returns dict with pooled lda/qda AUC, mean ||dmu||, coverage, n_cells."""
    Ktr = Ytr[:, key_idx]
    Kte = Yte[:, key_idx]
    keys = np.unique(np.vstack([Ktr, Kte]), axis=0)
    yL, yQ, sL, sQ = [], [], [], []        # pooled labels/scores
    dmus, ncells, cov = [], 0, 0
    for k in keys:
        mtr = np.all(Ktr == k, axis=1)
        Xc, yc = Xtr[mtr], Ytr[mtr, COUNTRY]
        if mode == "holdout":
            mte = np.all(Kte == k, axis=1)
            Xev, yev = Xte[mte], Yte[mte, COUNTRY]
            if (yc == 1).sum() < 2 or (yc == 0).sum() < 2:
                continue
            if len(yev) == 0:
                continue
            dmus.append(np.linalg.norm(Xc[yc == 1].mean(0) - Xc[yc == 0].mean(0)))
            try:
                pl = LDA().fit(Xc, yc).predict_proba(Xev)[:, 1]
                pq = QDA(reg_param=QDA_REG).fit(Xc, yc).predict_proba(Xev)[:, 1]
            except Exception:
                continue
            yL.extend(yev); sL.extend(pl); yQ.extend(yev); sQ.extend(pq)
            ncells += 1; cov += len(yev)
        else:  # cv within cell's train data
            if (yc == 1).sum() < 3 or (yc == 0).sum() < 3:
                continue
            dmus.append(np.linalg.norm(Xc[yc == 1].mean(0) - Xc[yc == 0].mean(0)))
            nsp = min(5, (yc == 1).sum(), (yc == 0).sum())
            skf = StratifiedKFold(n_splits=nsp, shuffle=True, random_state=0)
            for tri, tei in skf.split(Xc, yc):
                if len(np.unique(yc[tri])) < 2:
                    continue
                try:
                    pl = LDA().fit(Xc[tri], yc[tri]).predict_proba(Xc[tei])[:, 1]
                    pq = QDA(reg_param=QDA_REG).fit(Xc[tri], yc[tri]).predict_proba(Xc[tei])[:, 1]
                except Exception:
                    continue
                yL.extend(yc[tei]); sL.extend(pl)
                yQ.extend(yc[tei]); sQ.extend(pq)
            ncells += 1; cov += len(yc)
    if not yL:
        return None
    la = safe_auc(np.array(yL), np.array(sL))
    qa = safe_auc(np.array(yQ), np.array(sQ))
    n_tot = len(Xte) if mode == "holdout" else len(Xtr)
    return dict(lda=float(la), qda=float(qa), dmu=float(np.mean(dmus)),
                gap=float(qa - la), n_cells_scored=ncells, n_cells_total=len(keys),
                coverage_test=cov, frac_test_covered=cov / n_tot, mode=mode)


full_sum = pooled_predictions(other7, mode="holdout")
full_cv = pooled_predictions(other7, mode="cv")
print(f"  holdout-pooled : LDA={full_sum['lda']:.4f} QDA={full_sum['qda']:.4f} "
      f"||dmu||={full_sum['dmu']:.4f} gap={full_sum['gap']:.4f} "
      f"cells={full_sum['n_cells_scored']}/{full_sum['n_cells_total']} "
      f"covTest={full_sum['coverage_test']}")
print(f"  cv-pooled(train): LDA={full_cv['lda']:.4f} QDA={full_cv['qda']:.4f} "
      f"||dmu||={full_cv['dmu']:.4f} gap={full_cv['gap']:.4f} "
      f"cells={full_cv['n_cells_scored']}/{full_cv['n_cells_total']} "
      f"covTrain={full_cv['coverage_test']}")
robust_sum = full_cv  # use cv-pooled as the robust full estimate

# ---- Independent confirmation: mean-residualized linear probe ----
# Subtract each cell's class-agnostic mean (the cell-mean of x), then run a
# single GLOBAL linear probe on residuals. If GATING (B), removing per-cell
# offsets that encode the food-sign-flip should let ONE linear direction read
# country -> high AUC. If intrinsic shell (A), residualizing means changes
# little (means already coincide within cells) -> still chance.
print("  -- residualized global linear probe (subtract per-cell mean) --")
from sklearn.linear_model import LogisticRegression
for kidx, kname in [([FOOD], "food"), ([FOOD, SENT], "food x sent"), (other7, "full other7")]:
    Ktr = Ytr[:, kidx]; Kte = Yte[:, kidx]
    Xr_tr = Xtr.copy().astype(float); Xr_te = Xte.copy().astype(float)
    keys = np.unique(np.vstack([Ktr, Kte]), axis=0)
    for k in keys:
        mtr = np.all(Ktr == k, axis=1); mte = np.all(Kte == k, axis=1)
        if mtr.sum() == 0:
            continue
        cm = Xtr[mtr].mean(0)          # cell mean from TRAIN only (no leakage)
        Xr_tr[mtr] -= cm; Xr_te[mte] -= cm
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xr_tr, Ytr[:, COUNTRY])
    auc = safe_auc(Yte[:, COUNTRY], clf.predict_proba(Xr_te)[:, 1])
    print(f"     resid by {kname:<12}: global linear AUC = {auc:.4f}")

# ---------------------------------------------------------------------------
# DECISIVE QUANTIFICATION: conditional-linear vs irreducibly-quadratic
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 4: conditional-linear vs irreducibly-quadratic decomposition")


def linear_fraction(lda_auc, qda_auc):
    """Fraction of country's separability (above chance) achieved by the
    LINEAR (mean-based) model, relative to the quadratic ceiling.
       (LDA-0.5) / (QDA-0.5)."""
    num = max(lda_auc - 0.5, 0.0)
    den = max(qda_auc - 0.5, 1e-9)
    return num / den


summaries = {
    "none": dict(lda=g["lda"], qda=g["qda"], dmu=g["dmu"],
                 gap=g["qda"] - g["lda"], n_cells_scored=1, n_cells_total=1,
                 coverage_test=len(Xte), frac_test_covered=1.0),
    "food": food_sum,
    "food_x_sentiment": fs_sum,
    "full_other7": full_sum,
}
if robust_sum:
    summaries["full_other7_cv"] = robust_sum

for name, s in summaries.items():
    if s is None:
        continue
    s["linear_fraction"] = linear_fraction(s["lda"], s["qda"])
    s["quadratic_fraction"] = 1.0 - s["linear_fraction"]

print(f"{'conditioning':<22}{'LDA':>7}{'QDA':>7}{'||dmu||':>9}"
      f"{'gap':>7}{'lin%':>7}{'quad%':>7}{'cells':>8}{'covTe':>7}")
order = ["none", "food", "food_x_sentiment", "full_other7", "full_other7_cv"]
for name in order:
    s = summaries.get(name)
    if s is None:
        continue
    cells = f"{s.get('n_cells_scored','-')}/{s.get('n_cells_total','-')}"
    print(f"{name:<22}{s['lda']:>7.3f}{s['qda']:>7.3f}{s['dmu']:>9.4f}"
          f"{s['gap']:>7.3f}{s['linear_fraction']*100:>6.0f}%"
          f"{s['quadratic_fraction']*100:>6.0f}%{cells:>8}"
          f"{s['frac_test_covered']*100:>6.0f}%")

# ---------------------------------------------------------------------------
# VERDICT logic
# ---------------------------------------------------------------------------
print("=" * 70)
print("VERDICT")
# Does conditioning on food linearize country?
food_lin = food_sum["linear_fraction"]
food_dmu = food_sum["dmu"]
food_gap = food_sum["gap"]
gating = (food_sum["lda"] > 0.9) and (food_dmu > 5 * g["dmu"])
intrinsic = (food_sum["gap"] > 0.25) and (food_sum["lda"] < 0.7)
print(f"  food-conditioned: LDA {g['lda']:.3f}->{food_sum['lda']:.3f}, "
      f"||dmu|| {g['dmu']:.4f}->{food_dmu:.4f}, gap {g['qda']-g['lda']:.3f}->{food_gap:.3f}")
if gating and not intrinsic:
    print("  => GATING (B): conditioning on food LINEARIZES country.")
elif intrinsic and not gating:
    print("  => INTRINSIC SHELL (A): QDA>>LDA persists within cells.")
else:
    print("  => MIXED / see numbers; full-conjunction result is decisive.")

# Save machine-readable
out = dict(global_=g, food=food_sum, food_x_sentiment=fs_sum,
           full_other7=full_sum, full_other7_minN=robust_sum,
           variance_ratio=float(vr),
           per_cell={
               "food": [{k: (v if not isinstance(v, float) or np.isfinite(v) else None)
                         for k, v in c.items()} for c in food_cells],
               "food_x_sentiment": [{k: (v if not isinstance(v, float) or np.isfinite(v) else None)
                                     for k, v in c.items()} for c in fs_cells],
           },
           summaries={k: v for k, v in summaries.items() if v is not None})
with open(r"C:/Users/Sidar/Desktop/puzzle/bluedot-tais-puzzle/results/round2_r1.json", "w") as f:
    json.dump(out, f, indent=2, default=lambda o: None if (isinstance(o, float) and not np.isfinite(o)) else o)
print("saved results/round2_r1.json")
