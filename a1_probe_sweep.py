"""Approach #1: Linear probe sweep at Layer L (h2) and across all layers.

For each of the 8 binary features, fit a linear logistic-regression probe on the
TRAIN activations (StandardScaled) and evaluate on TEST. Report base rate, test
ROC-AUC, balanced accuracy, and raw accuracy. Rank features by Layer-L linear AUC
to identify the non-linearly-represented feature F.

Also: repeat the sweep at every layer (emb, h0, h1, h2, h3) to trace how F's linear
AUC evolves, and sanity-check the full-model logits accuracy (should be ~ceiling).

No encoder / torch needed -- everything is read from cache/ .npy.
"""
import os, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score

ROOT = r"C:\Users\Sidar\Desktop\puzzle\bluedot-tais-puzzle"
C = os.path.join(ROOT, "cache")
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

feats = json.load(open(os.path.join(ROOT, "feature_names.json")))
Ytr = np.load(os.path.join(C, "labels_train.npy"))
Yte = np.load(os.path.join(C, "labels_test.npy"))

LAYERS = ["emb", "h0", "h1", "h2", "h3"]
C_GRID = [0.5, 1.0, 5.0]  # a couple of regularization strengths


def load_layer(name):
    return (np.load(os.path.join(C, f"{name}_train.npy")),
            np.load(os.path.join(C, f"{name}_test.npy")))


def fit_probe(Xtr, ytr, Xte, yte):
    """Fit linear logistic regression, pick best C by train-CV-ish (use train AUC
    as tie-break is leaky; instead use a small held-out from train). We use a
    simple, clean rule: pick C maximizing 5-fold CV AUC on TRAIN, then refit on
    full train and evaluate on TEST. Returns metrics dict for best C."""
    from sklearn.model_selection import cross_val_score
    best_c, best_cv = None, -1.0
    for c in C_GRID:
        pipe = make_pipeline(StandardScaler(),
                             LogisticRegression(C=c, max_iter=5000, solver="lbfgs"))
        try:
            cv = cross_val_score(pipe, Xtr, ytr, cv=5, scoring="roc_auc").mean()
        except Exception:
            cv = -1.0
        if cv > best_cv:
            best_cv, best_c = cv, c
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(C=best_c, max_iter=5000, solver="lbfgs"))
    pipe.fit(Xtr, ytr)
    score = pipe.decision_function(Xte)
    pred = (score > 0).astype(int)
    return {
        "C": best_c,
        "cv_auc": best_cv,
        "auc": roc_auc_score(yte, score),
        "bal_acc": balanced_accuracy_score(yte, pred),
        "acc": accuracy_score(yte, pred),
    }


def sweep_layer(name):
    Xtr, Xte = load_layer(name)
    out = {}
    for fi, fname in enumerate(feats):
        ytr, yte = Ytr[:, fi], Yte[:, fi]
        # skip degenerate (single-class) -- not expected here
        if len(np.unique(ytr)) < 2:
            out[fname] = {"C": None, "cv_auc": np.nan, "auc": np.nan,
                          "bal_acc": np.nan, "acc": np.nan, "base_rate": ytr.mean()}
            continue
        m = fit_probe(Xtr, ytr, Xte, yte)
        m["base_rate"] = yte.mean()
        out[fname] = m
    return out


def main():
    print(f"train N={Ytr.shape[0]}, test N={Yte.shape[0]}, features={feats}")

    # --- Sanity: full-model accuracy from logits cache ---
    lg_te = np.load(os.path.join(C, "logits_test.npy"))
    full_pred = (lg_te > 0).astype(int)
    full_acc = (full_pred == Yte).mean(axis=0)
    # AUC of the model's own logits per feature (recoverability ceiling)
    full_auc = np.array([roc_auc_score(Yte[:, i], lg_te[:, i]) for i in range(8)])

    # --- Sweep every layer ---
    results = {}
    for ln in LAYERS:
        print(f"sweeping layer {ln} ...")
        results[ln] = sweep_layer(ln)

    # ---------- Build report ----------
    lines = []
    lines.append("# Approach 1 - Linear Probe Sweep\n")
    lines.append("Linear logistic-regression probes (StandardScaler + LogisticRegression, "
                 "best C of {0.5,1,5} chosen by 5-fold train-CV AUC) fit on TRAIN "
                 "activations, evaluated on held-out TEST.\n")
    lines.append(f"- train N = {Ytr.shape[0]}, test N = {Yte.shape[0]}")
    lines.append(f"- Layer L = h2 (post-ReLU of hidden layer 2), 64-dim\n")

    # Sanity table
    lines.append("## Sanity: full-model recoverability (from logits cache, TEST)\n")
    lines.append("| feat | name | base_rate | full_acc | full_logit_AUC |")
    lines.append("|---|---|---|---|---|")
    for i, fn in enumerate(feats):
        lines.append(f"| {i} | {fn} | {Yte[:,i].mean():.3f} | {full_acc[i]:.4f} | {full_auc[i]:.4f} |")
    lines.append(f"\nAll features near ceiling for the full model "
                 f"(min acc {full_acc.min():.4f}, min AUC {full_auc.min():.4f}).\n")

    # Layer-L ranking table
    L = results["h2"]
    rank = sorted(feats, key=lambda f: L[f]["auc"])  # ascending: worst first
    lines.append("## Layer L (h2) linear-probe ranking (ascending by AUC = worst first)\n")
    lines.append("| rank | feat | name | base_rate | test_AUC | test_bal_acc | test_acc | bestC |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r, fn in enumerate(rank):
        i = feats.index(fn)
        m = L[fn]
        lines.append(f"| {r+1} | {i} | {fn} | {m['base_rate']:.3f} | {m['auc']:.4f} | "
                     f"{m['bal_acc']:.4f} | {m['acc']:.4f} | {m['C']} |")

    worst = rank[0]
    second = rank[1]
    gap = L[second]["auc"] - L[worst]["auc"]
    lines.append(f"\n**F = `{worst}` (feature index {feats.index(worst)})** is the clear "
                 f"linear-readability outlier at Layer L.")
    lines.append(f"- F test AUC = {L[worst]['auc']:.4f}; next-worst (`{second}`) = "
                 f"{L[second]['auc']:.4f}; **gap = {gap:.4f}**.")
    lines.append(f"- The other 7 features sit at AUC {min(L[f]['auc'] for f in feats if f!=worst):.4f} "
                 f"- {max(L[f]['auc'] for f in feats):.4f} (near ceiling).\n")

    # Layer-by-layer evolution of F vs others
    lines.append("## Layer-by-layer evolution of linear AUC\n")
    lines.append("| name | " + " | ".join(LAYERS) + " |")
    lines.append("|---|" + "|".join(["---"]*len(LAYERS)) + "|")
    for fn in feats:
        row = [fn] + [f"{results[ln][fn]['auc']:.4f}" for ln in LAYERS]
        mark = "  <-- F" if fn == worst else ""
        lines.append("| " + " | ".join(row) + " |" + mark)

    lines.append(f"\nF (`{worst}`) linear AUC trajectory across layers:")
    traj = " -> ".join(f"{ln}:{results[ln][worst]['auc']:.4f}" for ln in LAYERS)
    lines.append(f"  {traj}\n")

    # mean AUC of the other 7 per layer for contrast
    lines.append("Mean linear AUC of the other 7 features per layer (for contrast):")
    others = [f for f in feats if f != worst]
    contrast = " -> ".join(
        f"{ln}:{np.mean([results[ln][f]['auc'] for f in others]):.4f}" for ln in LAYERS)
    lines.append(f"  {contrast}\n")

    # Conclusion
    lines.append("## Conclusion\n")
    lines.append(f"At Layer L (h2), **`{worst}`** is non-linearly represented: a single "
                 f"linear direction reads it at only AUC {L[worst]['auc']:.4f} "
                 f"(bal_acc {L[worst]['bal_acc']:.4f}), while all 7 other features are "
                 f"linearly readable at AUC >= "
                 f"{min(L[f]['auc'] for f in feats if f!=worst):.4f}. "
                 f"The gap to the next-worst feature is {gap:.4f}. "
                 f"The full model still recovers `{worst}` at "
                 f"acc {full_acc[feats.index(worst)]:.4f} / logit-AUC "
                 f"{full_auc[feats.index(worst)]:.4f}, confirming the deficit is "
                 f"specifically linear readability at L, not absence of the feature.\n")

    report = "\n".join(lines)
    with open(os.path.join(RESULTS, "approach_1.md"), "w", encoding="utf-8") as f:
        f.write(report)
    print("\n" + report)
    print("\n[saved] results/approach_1.md")

    # also dump raw numbers as json for downstream agents
    dump = {ln: {fn: {k: (None if (isinstance(v, float) and np.isnan(v)) else
                          (float(v) if isinstance(v, (np.floating, float, int)) else v))
                      for k, v in results[ln][fn].items()} for fn in feats}
            for ln in LAYERS}
    dump["_full_model"] = {feats[i]: {"acc": float(full_acc[i]), "logit_auc": float(full_auc[i]),
                                      "base_rate": float(Yte[:, i].mean())} for i in range(8)}
    dump["_F"] = worst
    with open(os.path.join(RESULTS, "approach_1.json"), "w", encoding="utf-8") as f:
        json.dump(dump, f, indent=2)


if __name__ == "__main__":
    main()
