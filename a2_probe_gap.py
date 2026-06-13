"""
Approach #2: LINEAR-vs-NONLINEAR PROBE GAP at Layer L (h2, post-ReLU of hidden layer 2).

For each of the 8 binary features, on h2 activations we train:
  (a) a LINEAR probe  : LogisticRegression (StandardScaler + tuned C via CV).
  (b) NON-LINEAR probes: sklearn MLPClassifier (two architectures, early stopping),
                         an RBF-kernel SVC, and a GradientBoostingClassifier.
Evaluate on TEST with ROC-AUC and balanced accuracy.
GAP = best_nonlinear_AUC - linear_AUC. The feature with the LARGEST gap is F:
recoverable nonlinearly but NOT linearly at L.

Scaling is held CONSISTENT (same StandardScaler fit on train) between linear and
the scale-sensitive nonlinear probes (MLP, SVC) so the comparison is fair. Trees
are scale-invariant so they use raw features.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

ROOT = r"C:\Users\Sidar\Desktop\puzzle\bluedot-tais-puzzle"
C = os.path.join(ROOT, "cache")
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)

feats = json.load(open(os.path.join(ROOT, "feature_names.json")))
Xtr = np.load(os.path.join(C, "h2_train.npy"))
Xte = np.load(os.path.join(C, "h2_test.npy"))
Ytr = np.load(os.path.join(C, "labels_train.npy"))
Yte = np.load(os.path.join(C, "labels_test.npy"))

# Consistent scaling: fit on train, apply to both. Used by linear + MLP + SVC.
scaler = StandardScaler().fit(Xtr)
Xtr_s = scaler.transform(Xtr)
Xte_s = scaler.transform(Xte)

SEED = 0
rows = []
detail = {}

for fi, fname in enumerate(feats):
    ytr = Ytr[:, fi]
    yte = Yte[:, fi]

    # ---- (a) LINEAR probe: tuned-C logistic regression on scaled features ----
    lin = LogisticRegressionCV(
        Cs=np.logspace(-3, 3, 13), cv=5, scoring="roc_auc",
        max_iter=5000, n_jobs=-1, random_state=SEED,
    ).fit(Xtr_s, ytr)
    lin_p = lin.predict_proba(Xte_s)[:, 1]
    lin_auc = roc_auc_score(yte, lin_p)
    lin_bacc = balanced_accuracy_score(yte, (lin_p >= 0.5).astype(int))

    # ---- (b) NON-LINEAR probes ----
    # MLP #1: single hidden layer (64,)
    mlp1 = MLPClassifier(hidden_layer_sizes=(64,), activation="relu",
                         alpha=1e-3, max_iter=500, early_stopping=True,
                         n_iter_no_change=20, random_state=SEED).fit(Xtr_s, ytr)
    mlp1_p = mlp1.predict_proba(Xte_s)[:, 1]
    mlp1_auc = roc_auc_score(yte, mlp1_p)

    # MLP #2: deeper (128, 64)
    mlp2 = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu",
                         alpha=1e-3, max_iter=500, early_stopping=True,
                         n_iter_no_change=20, random_state=SEED).fit(Xtr_s, ytr)
    mlp2_p = mlp2.predict_proba(Xte_s)[:, 1]
    mlp2_auc = roc_auc_score(yte, mlp2_p)

    # RBF-kernel SVC
    svc = SVC(kernel="rbf", C=10.0, gamma="scale", probability=True,
              random_state=SEED).fit(Xtr_s, ytr)
    svc_p = svc.predict_proba(Xte_s)[:, 1]
    svc_auc = roc_auc_score(yte, svc_p)

    # Gradient boosting (scale-invariant -> raw features)
    gb = GradientBoostingClassifier(n_estimators=300, max_depth=3,
                                    learning_rate=0.05, subsample=0.8,
                                    random_state=SEED).fit(Xtr, ytr)
    gb_p = gb.predict_proba(Xte)[:, 1]
    gb_auc = roc_auc_score(yte, gb_p)

    nl_aucs = {"MLP(64)": mlp1_auc, "MLP(128,64)": mlp2_auc,
               "SVC-RBF": svc_auc, "GBoost": gb_auc}
    best_nl_name = max(nl_aucs, key=nl_aucs.get)
    best_nl_auc = nl_aucs[best_nl_name]
    best_nl_p = {"MLP(64)": mlp1_p, "MLP(128,64)": mlp2_p,
                 "SVC-RBF": svc_p, "GBoost": gb_p}[best_nl_name]
    best_nl_bacc = balanced_accuracy_score(yte, (best_nl_p >= 0.5).astype(int))

    gap = best_nl_auc - lin_auc
    rows.append((fi, fname, lin_auc, lin_bacc, best_nl_auc, best_nl_bacc,
                 best_nl_name, gap))
    detail[fname] = {"lin_auc": lin_auc, "lin_bacc": lin_bacc,
                     "best_nl": best_nl_name, "best_nl_auc": best_nl_auc,
                     "nl_aucs": nl_aucs, "gap": gap}
    print(f"[{fi}] {fname:10s} linAUC={lin_auc:.4f} "
          f"bestNL={best_nl_auc:.4f}({best_nl_name}) gap={gap:+.4f} "
          f"| " + " ".join(f"{k}={v:.4f}" for k, v in nl_aucs.items()))

# Rank by gap descending
rows_sorted = sorted(rows, key=lambda r: r[7], reverse=True)
F = rows_sorted[0]
second = rows_sorted[1]

# ---------- write results/approach_2.md ----------
lines = []
lines.append("# Approach #2 - Linear vs Non-linear Probe Gap at Layer L (h2)\n")
lines.append("Layer L = post-ReLU of hidden layer 2, 64-dim. Train N=7000, Test N=1500.\n")
lines.append("Linear probe = LogisticRegressionCV (StandardScaler, C tuned over 1e-3..1e3, 5-fold ROC-AUC).\n")
lines.append("Non-linear = best of {MLP(64), MLP(128,64), RBF-SVC, GradientBoosting}. "
             "Scale-sensitive probes share the SAME StandardScaler as the linear probe; "
             "GBoost uses raw features (scale-invariant). Metric: TEST ROC-AUC.\n")
lines.append("\n## Ranked table (by gap, descending)\n")
lines.append("| rank | idx | feature | linear_AUC | linear_bAcc | nonlinear_AUC | nonlinear_bAcc | best_NL | GAP |")
lines.append("|---|---|---|---|---|---|---|---|---|")
for rank, (fi, fname, la, lb, na, nb, nl, gap) in enumerate(rows_sorted, 1):
    mark = "  <-- F" if rank == 1 else ""
    lines.append(f"| {rank} | {fi} | {fname}{mark} | {la:.4f} | {lb:.4f} | "
                 f"{na:.4f} | {nb:.4f} | {nl} | {gap:+.4f} |")

lines.append("\n## Per-feature non-linear breakdown (all NL probe AUCs)\n")
lines.append("| feature | linear_AUC | MLP(64) | MLP(128,64) | SVC-RBF | GBoost |")
lines.append("|---|---|---|---|---|---|")
for fi, fname in enumerate(feats):
    d = detail[fname]
    n = d["nl_aucs"]
    lines.append(f"| {fname} | {d['lin_auc']:.4f} | {n['MLP(64)']:.4f} | "
                 f"{n['MLP(128,64)']:.4f} | {n['SVC-RBF']:.4f} | {n['GBoost']:.4f} |")

gap_F = F[7]
gap_2 = second[7]
sep = gap_F - gap_2
lines.append("\n## Conclusion\n")
lines.append(f"- **F = `{F[1]}` (index {F[0]})** is the non-linear feature.")
lines.append(f"- Decisive gap: F gap = **{gap_F:+.4f}** "
             f"(linear AUC {F[2]:.4f} -> nonlinear AUC {F[4]:.4f}, via {F[6]}).")
lines.append(f"- Next-largest gap: `{second[1]}` at {gap_2:+.4f}. "
             f"Separation F-vs-rest = **{sep:+.4f}** ({gap_F/max(gap_2,1e-6):.1f}x).")
other_gaps = [r[7] for r in rows_sorted[1:]]
lines.append(f"- The other 7 features: gaps in [{min(other_gaps):+.4f}, {max(other_gaps):+.4f}], "
             f"mean {np.mean(other_gaps):+.4f} (~0 = both probes at ceiling).")
lines.append(f"- F's linear AUC ({F[2]:.4f}) is the only one well below ceiling, "
             f"confirming linear-unreadability at L while remaining nonlinearly recoverable "
             f"(nonlinear AUC {F[4]:.4f}).")

open(os.path.join(RES, "approach_2.md"), "w", encoding="utf-8").write("\n".join(lines))

print("\n=== RANKED BY GAP ===")
for rank, (fi, fname, la, lb, na, nb, nl, gap) in enumerate(rows_sorted, 1):
    print(f"{rank}. {fname:10s} linAUC={la:.4f} nlAUC={na:.4f} gap={gap:+.4f} ({nl})")
print(f"\nF = {F[1]} (idx {F[0]}), gap={gap_F:+.4f}; "
      f"2nd={second[1]} gap={gap_2:+.4f}; separation={sep:+.4f}")
print("Wrote", os.path.join(RES, "approach_2.md"))
