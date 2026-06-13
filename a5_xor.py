"""Approach #5: XOR / interaction structure of feature F at Layer L (h2).

F is confirmed (approach 1) to be `country` (idx 5): linear AUC ~0.504 at h2
(chance) while all 7 others are >=0.995. A near-exactly-chance linear AUC is the
signature of an XOR/parity-like encoding: F is linearly separable only CONDITIONAL
on another feature/region, not globally.

This script:
 1) CONDITIONAL LINEAR PROBES: for each other feature g, split by g==0 / g==1 and
    fit a separate linear probe for F within each subset. Build a matrix of F's
    conditional linear AUC. If F becomes linear once conditioned on some g, that's
    an F-by-g interaction.
 2) LITERAL XOR TEST: does F's label == XOR/AND/OR of two binarized linear
    directions? Find best linear directions for each other feature, binarize, and
    test agreement of F with pairwise XOR/AND/OR. Also test whether adding an
    explicit interaction (product) term rescues a linear probe of F.
 3) DATA EXAMINATION: pull example texts for the four (F,g) cells.
 4) Geometry: 2-component representation test -- does an XOR-style 2-blob-per-class
    structure exist? Check via class-conditional cluster analysis and a small MLP
    vs linear probe gap.

Everything read from cache/ .npy. No encoder / torch-forward needed. (torch only
used optionally; we set CUDA off if imported, but we avoid it entirely here.)
"""
import os, json
import numpy as np
from itertools import combinations
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score
from sklearn.model_selection import cross_val_score

ROOT = r"C:\Users\Sidar\Desktop\puzzle\bluedot-tais-puzzle"
C = os.path.join(ROOT, "cache")
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)
RNG = np.random.RandomState(0)

feats = json.load(open(os.path.join(ROOT, "feature_names.json")))
Xtr = np.load(os.path.join(C, "h2_train.npy"))
Xte = np.load(os.path.join(C, "h2_test.npy"))
Ytr = np.load(os.path.join(C, "labels_train.npy"))
Yte = np.load(os.path.join(C, "labels_test.npy"))
texts_tr = json.load(open(os.path.join(C, "texts_train.json")))
texts_te = json.load(open(os.path.join(C, "texts_test.json")))

F = feats.index("country")  # confirmed outlier; re-verified below
NF = len(feats)


def lin_auc(Xa, ya, Xb, yb, Cc=1.0):
    """Linear logistic probe; train on (Xa,ya), eval AUC on (Xb,yb)."""
    if len(np.unique(ya)) < 2 or len(np.unique(yb)) < 2:
        return np.nan, np.nan
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(C=Cc, max_iter=5000, solver="lbfgs"))
    pipe.fit(Xa, ya)
    s = pipe.decision_function(Xb)
    return (roc_auc_score(yb, s),
            balanced_accuracy_score(yb, (s > 0).astype(int)))


def direction_score(fi):
    """Linear logistic direction for feature fi: return projected scores for
    train and test (decision_function), fitted on train."""
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs"))
    pipe.fit(Xtr, Ytr[:, fi])
    return pipe.decision_function(Xtr), pipe.decision_function(Xte)


def main():
    out = {}
    print(f"F = {feats[F]} (idx {F})")

    # ---- 0. Re-verify global linear AUC for F and others ----
    glob = {}
    for fi, fn in enumerate(feats):
        a, b = lin_auc(Xtr, Ytr[:, fi], Xte, Yte[:, fi])
        glob[fn] = {"auc": a, "bal_acc": b}
    out["global_linear"] = glob
    print("global F linear AUC =", round(glob[feats[F]]["auc"], 4))

    # ================================================================
    # 1. CONDITIONAL LINEAR PROBES: F | g==v
    # ================================================================
    cond = {}
    yF_tr, yF_te = Ytr[:, F], Yte[:, F]
    for gi, gn in enumerate(feats):
        if gi == F:
            continue
        cond[gn] = {}
        for v in (0, 1):
            mtr = Ytr[:, gi] == v
            mte = Yte[:, gi] == v
            a, ba = lin_auc(Xtr[mtr], yF_tr[mtr], Xte[mte], yF_te[mte])
            cond[gn][v] = {
                "auc": a, "bal_acc": ba,
                "n_tr": int(mtr.sum()), "n_te": int(mte.sum()),
                "F_rate_tr": float(yF_tr[mtr].mean()) if mtr.sum() else np.nan,
            }
    out["conditional"] = cond

    # Identify best conditioning feature: the g for which BOTH subsets are most
    # linear (min over v of conditional AUC is high) and which beats global.
    cond_summary = []
    for gn, d in cond.items():
        aucs = [d[v]["auc"] for v in (0, 1) if not np.isnan(d[v]["auc"])]
        if not aucs:
            continue
        cond_summary.append((gn, min(aucs), np.mean(aucs), d[0]["auc"], d[1]["auc"]))
    cond_summary.sort(key=lambda r: -r[1])  # by worst-subset AUC desc
    out["cond_summary"] = cond_summary

    # ================================================================
    # 2. LITERAL XOR / AND / OR TEST against pairs of binarized directions
    # ================================================================
    # Binarized linear-direction predictions for each feature (on train; choose
    # threshold 0 from the logistic, then evaluate agreement with F label on test).
    dir_tr, dir_te = {}, {}
    for fi in range(NF):
        s_tr, s_te = direction_score(fi)
        dir_tr[fi] = (s_tr > 0).astype(int)
        dir_te[fi] = (s_te > 0).astype(int)

    # also use the *true* labels of the other features as ideal binarized dirs,
    # to test the cleanest combinatorial hypothesis F == g1 (op) g2.
    def agree(pred, y):
        return float((pred == y).mean())

    pair_results = []
    others = [i for i in range(NF) if i != F]
    for a, b in combinations(others, 2):
        for op_name, op in (("XOR", lambda x, y: x ^ y),
                            ("AND", lambda x, y: x & y),
                            ("OR", lambda x, y: x | y),
                            ("XNOR", lambda x, y: 1 - (x ^ y))):
            # using TRUE labels of a,b
            p_true = op(Yte[:, a], Yte[:, b])
            acc_true = agree(p_true, yF_te)
            # using binarized linear directions of a,b
            p_dir = op(dir_te[a], dir_te[b])
            acc_dir = agree(p_dir, yF_te)
            pair_results.append((feats[a], feats[b], op_name, acc_true, acc_dir))
    pair_results.sort(key=lambda r: -max(r[3], r[4]))
    out["pair_xor"] = pair_results[:25]

    # Single-feature equality baseline (is F just == some g or its negation?)
    single = []
    for g in others:
        single.append((feats[g], agree(Yte[:, g], yF_te),
                       agree(1 - Yte[:, g], yF_te)))
    single.sort(key=lambda r: -max(r[1], r[2]))
    out["single_equality"] = single

    # ================================================================
    # 2b. Does adding an interaction (product) term rescue a linear probe?
    #     Build features [proj_a, proj_b, proj_a*proj_b] for top candidate pair
    #     and for ALL pairs of projected scores, logistic-regress F on them.
    # ================================================================
    # projected (continuous) scores
    proj_tr = {fi: direction_score(fi) for fi in range(NF)}
    PT = {fi: proj_tr[fi][0] for fi in range(NF)}
    PE = {fi: proj_tr[fi][1] for fi in range(NF)}

    rescue = []
    for a, b in combinations(others, 2):
        # linear-only on the two projections
        Za_tr = np.column_stack([PT[a], PT[b]])
        Za_te = np.column_stack([PE[a], PE[b]])
        auc_lin, _ = lin_auc(Za_tr, yF_tr, Za_te, yF_te)
        # with interaction term
        Zi_tr = np.column_stack([PT[a], PT[b], PT[a] * PT[b]])
        Zi_te = np.column_stack([PE[a], PE[b], PE[a] * PE[b]])
        auc_int, _ = lin_auc(Zi_tr, yF_tr, Zi_te, yF_te)
        rescue.append((feats[a], feats[b], float(auc_lin), float(auc_int),
                       float(auc_int - auc_lin)))
    rescue.sort(key=lambda r: -r[3])
    out["interaction_rescue"] = rescue[:25]

    # ================================================================
    # 3. NONLINEAR CEILING: small MLP probe on h2 for F (how recoverable is it
    #    with a nonlinear reader?) + quadratic-feature linear probe on full h2.
    # ================================================================
    mlp = make_pipeline(StandardScaler(),
                        MLPClassifier(hidden_layer_sizes=(64,), max_iter=2000,
                                      alpha=1e-3, random_state=0))
    mlp.fit(Xtr, yF_tr)
    s = mlp.predict_proba(Xte)[:, 1]
    out["mlp_F_auc"] = float(roc_auc_score(yF_te, s))
    out["mlp_F_acc"] = float(accuracy_score(yF_te, (s > 0.5).astype(int)))

    # ================================================================
    # 4. GEOMETRY: is country represented as 2 blobs per class (XOR layout)?
    #    Project h2 onto the top conditioning feature's direction and F's best
    #    within-subset directions, and characterize.
    #    Also: LDA-style -- find direction maximizing F separation *within* each
    #    g-subset; check if those two within-subset F-directions differ (sign flip
    #    => XOR geometry: same axis reads opposite depending on g).
    # ================================================================
    geom = {}
    # pick the best conditioning g = highest worst-subset conditional AUC
    if cond_summary:
        bestg_name = cond_summary[0][0]
        bestg = feats.index(bestg_name)
        geom["best_conditioning_g"] = bestg_name
        # within-subset F directions (weight vectors in standardized h2 space)
        wdirs = {}
        for v in (0, 1):
            m = Ytr[:, bestg] == v
            sc = StandardScaler().fit(Xtr[m])
            lr = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(Xtr[m]),
                                                              yF_tr[m])
            w = lr.coef_[0] / np.linalg.norm(lr.coef_[0])
            wdirs[v] = w
        cos = float(np.dot(wdirs[0], wdirs[1]))
        geom["within_subset_F_dir_cosine"] = cos  # ~ -1 => XOR sign flip; ~+1 => same
        # global F direction
        scg = StandardScaler().fit(Xtr)
        lrg = LogisticRegression(C=1.0, max_iter=5000).fit(scg.transform(Xtr), yF_tr)
        wg = lrg.coef_[0] / np.linalg.norm(lrg.coef_[0])
        geom["cos_global_vs_sub0"] = float(np.dot(wg, wdirs[0]))
        geom["cos_global_vs_sub1"] = float(np.dot(wg, wdirs[1]))
    out["geometry"] = geom

    # 4b. Cluster test: within country==1, are there 2 clusters aligned with g?
    #     Compare mean h2 of the 4 cells.
    cells = {}
    for fv in (0, 1):
        for gv in (0, 1):
            if cond_summary:
                m = (Ytr[:, F] == fv) & (Ytr[:, bestg] == gv)
                cells[(fv, gv)] = Xtr[m].mean(0) if m.sum() else None
    if cond_summary and all(c is not None for c in cells.values()):
        # distance structure between the 4 cell-centroids
        def d(p, q):
            return float(np.linalg.norm(cells[p] - cells[q]))
        geom["cell_centroid_dists"] = {
            "F0g0_F1g0": d((0, 0), (1, 0)),  # F flip, g=0
            "F0g1_F1g1": d((0, 1), (1, 1)),  # F flip, g=1
            "F0g0_F0g1": d((0, 0), (0, 1)),  # g flip, F=0
            "F1g0_F1g1": d((1, 0), (1, 1)),  # g flip, F=1
            "F0g0_F1g1": d((0, 0), (1, 1)),  # diagonal
            "F0g1_F1g0": d((0, 1), (1, 0)),  # anti-diagonal
        }

    # ================================================================
    # 5. DATA: example texts for the four (F, bestg) cells
    # ================================================================
    examples = {}
    if cond_summary:
        for fv in (0, 1):
            for gv in (0, 1):
                m = np.where((Yte[:, F] == fv) & (Yte[:, bestg] == gv))[0]
                idx = m[:6].tolist()
                examples[f"F={fv},{bestg_name}={gv}"] = [texts_te[i] for i in idx]
    out["examples"] = examples

    # ---------------- write JSON + markdown ----------------
    def _san(o):
        if isinstance(o, dict):
            return {k: _san(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_san(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if (isinstance(o, float) and np.isnan(o)) else float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        return o
    json.dump(_san(out), open(os.path.join(RESULTS, "approach_5.json"), "w"),
              indent=2)

    write_md(out)
    # console summary
    print("\nCONDITIONAL AUC (worst-subset desc):")
    for gn, worst, mean, a0, a1 in cond_summary:
        print(f"  cond on {gn:10s}: g=0 AUC={a0:.3f}  g=1 AUC={a1:.3f}  worst={worst:.3f}")
    print("\nTOP literal pair combos (op on TRUE labels / on linear dirs):")
    for a, b, op, at, ad in out["pair_xor"][:6]:
        print(f"  {a}+{b} {op}: acc_true={at:.3f} acc_dir={ad:.3f}")
    print("\nINTERACTION RESCUE (proj a,b -> +product):")
    for a, b, al, ai, dl in out["interaction_rescue"][:6]:
        print(f"  {a}+{b}: lin={al:.3f} +int={ai:.3f} (gain {dl:+.3f})")
    print(f"\nMLP probe F AUC={out['mlp_F_auc']:.3f}  acc={out['mlp_F_acc']:.3f}")
    if "within_subset_F_dir_cosine" in geom:
        print(f"within-subset F-direction cosine = {geom['within_subset_F_dir_cosine']:.3f}")
    print("\n[saved] results/approach_5.md / .json")


def write_md(out):
    L = []
    F_name = feats[F]
    L.append("# Approach 5 - XOR / Interaction Structure of F at Layer L (h2)\n")
    L.append(f"**F = `{F_name}` (index {F})**, confirmed: global linear probe AUC "
             f"= {out['global_linear'][F_name]['auc']:.4f} "
             f"(bal_acc {out['global_linear'][F_name]['bal_acc']:.4f}) at h2, "
             f"i.e. essentially chance, while all 7 other features are linearly "
             f"readable at AUC >= "
             f"{min(out['global_linear'][f]['auc'] for f in feats if f!=F_name):.4f}.\n")
    L.append("A linear AUC pinned at ~0.50 (not merely degraded to ~0.8) is the "
             "tell-tale signature of a **parity/XOR-like** encoding: no single "
             "direction carries net signal, because the feature's sign is *flipped* "
             "depending on another variable.\n")

    # Conditional matrix
    L.append("## 1. Conditional linear probes:  AUC of  F | g==v\n")
    L.append("For each other feature g, split data by g and fit a separate linear "
             "probe for F within each half.\n")
    L.append("| conditioning g | F-AUC (g=0) | F-AUC (g=1) | worst-of-two | n_te(g=0) | n_te(g=1) |")
    L.append("|---|---|---|---|---|---|")
    cs = {r[0]: r for r in out["cond_summary"]}
    for gn in [f for f in feats if f != F_name]:
        d = out["conditional"][gn]
        a0, a1 = d[0]["auc"], d[1]["auc"]
        worst = cs[gn][1] if gn in cs else min(a0, a1)
        L.append(f"| {gn} | {a0:.3f} | {a1:.3f} | **{worst:.3f}** | "
                 f"{d[0]['n_te']} | {d[1]['n_te']} |")
    top = out["cond_summary"][0]
    rescued = top[1] > 0.9
    L.append("")
    if rescued:
        L.append(f"**Conditioning on `{top[0]}` RESCUES linearity**: within each "
                 f"`{top[0]}` subset, F is linearly separable at AUC "
                 f">= {top[1]:.3f} (g=0:{top[3]:.3f}, g=1:{top[4]:.3f}) -- vs "
                 f"~0.50 globally. This is a clean F-by-`{top[0]}` interaction.\n")
    else:
        L.append(f"Best conditioning feature `{top[0]}` raises worst-subset AUC to "
                 f"{top[1]:.3f} (from ~0.50 global) -- partial but not full rescue.\n")

    # Literal XOR
    L.append("## 2. Literal XOR / AND / OR test\n")
    L.append("Agreement of F's label with op(g1,g2). `acc_true` uses the *true* "
             "labels of g1,g2; `acc_dir` uses binarized linear-probe directions.\n")
    L.append("| g1 | g2 | op | acc(true labels) | acc(linear dirs) |")
    L.append("|---|---|---|---|---|")
    for a, b, op, at, ad in out["pair_xor"][:12]:
        L.append(f"| {a} | {b} | {op} | {at:.3f} | {ad:.3f} |")
    best_pair = out["pair_xor"][0]
    L.append("")
    L.append(f"Best combinatorial match: **`{best_pair[0]}` {best_pair[2]} "
             f"`{best_pair[1]}`** == F at acc {max(best_pair[3],best_pair[4]):.3f}.\n")

    L.append("### Single-feature equality baseline (is F just == some g?)\n")
    L.append("| g | acc(F==g) | acc(F==~g) |")
    L.append("|---|---|---|")
    for gn, e, ne in out["single_equality"][:5]:
        L.append(f"| {gn} | {e:.3f} | {ne:.3f} |")
    L.append("")

    # Interaction rescue
    L.append("## 2b. Does an explicit interaction (product) term rescue the probe?\n")
    L.append("Logistic probe of F on [proj_g1, proj_g2] (linear) vs adding "
             "[proj_g1 * proj_g2] (interaction). proj = continuous linear-direction "
             "score for that feature.\n")
    L.append("| g1 | g2 | AUC linear | AUC +interaction | gain |")
    L.append("|---|---|---|---|---|")
    for a, b, al, ai, dl in out["interaction_rescue"][:10]:
        L.append(f"| {a} | {b} | {al:.3f} | {ai:.3f} | {dl:+.3f} |")
    L.append("")

    # MLP ceiling + geometry
    L.append("## 3. Nonlinear reader ceiling\n")
    L.append(f"- A 1-hidden-layer MLP probe (64 units) on raw h2 reads F at "
             f"AUC {out['mlp_F_auc']:.3f} / acc {out['mlp_F_acc']:.3f} "
             f"-- F is fully present, just not in a single linear direction.\n")

    g = out.get("geometry", {})
    if "within_subset_F_dir_cosine" in g:
        L.append("## 4. Geometry at L\n")
        cos = g["within_subset_F_dir_cosine"]
        L.append(f"- Best conditioning feature: **`{g['best_conditioning_g']}`**.")
        L.append(f"- Within-subset F-direction cosine "
                 f"(dir learned in g=0 vs g=1) = **{cos:.3f}**.")
        if cos < -0.3:
            L.append("  A strongly *negative* cosine means the same h2 axis reads "
                     "country with OPPOSITE sign depending on g -- exactly the XOR "
                     "geometry (the global probe averages the two to ~0).")
        elif cos > 0.5:
            L.append("  A positive cosine means the F-axis is roughly shared across "
                     "g-subsets; the global probe failure then comes from a g-"
                     "dependent *offset/threshold* rather than a sign flip.")
        else:
            L.append("  An intermediate cosine indicates the F-readout axis rotates "
                     "with g (a curved/interaction boundary), partially XOR-like.")
        L.append(f"- cos(global F-dir, g=0 sub dir) = {g.get('cos_global_vs_sub0',float('nan')):.3f}; "
                 f"cos(global, g=1 sub dir) = {g.get('cos_global_vs_sub1',float('nan')):.3f}.")
        if "cell_centroid_dists" in g:
            d = g["cell_centroid_dists"]
            L.append("\n4-cell centroid distances in h2 (F x g):")
            L.append("| pair | distance |")
            L.append("|---|---|")
            for k, v in d.items():
                L.append(f"| {k} | {v:.3f} |")
        L.append("")

    # Examples
    L.append("## 5. Example texts per (F, conditioning-g) cell\n")
    for k, exs in out.get("examples", {}).items():
        L.append(f"**{k}**:")
        for e in exs:
            L.append(f"  - {e}")
        L.append("")

    # Verdict
    L.append("## Verdict\n")
    top = out["cond_summary"][0]
    L.append(f"- **F = `{F_name}`** (high confidence).")
    if top[1] > 0.9:
        L.append(f"- **Interaction confirmed**: F is linearly separable *conditional* "
                 f"on `{top[0]}` (within-subset AUC >= {top[1]:.3f}) but ~chance "
                 f"globally. Partner feature/region = **`{top[0]}`**.")
    g2 = out.get("geometry", {})
    if g2.get("within_subset_F_dir_cosine", 0) < -0.3:
        L.append(f"- The within-subset F directions have cosine "
                 f"{g2['within_subset_F_dir_cosine']:.3f} (sign-flipped) -> "
                 f"**XOR/parity geometry**: country is encoded along an axis whose "
                 f"meaning inverts with `{top[0]}`.")
    best_pair = out["pair_xor"][0]
    if max(best_pair[3], best_pair[4]) > 0.85:
        L.append(f"- Literal combinatorial fit: F ~= `{best_pair[0]}` "
                 f"{best_pair[2]} `{best_pair[1]}` (acc {max(best_pair[3],best_pair[4]):.3f}).")
    L.append("\n### Caveats")
    L.append("- Conditional-probe subsets are smaller -> AUCs are slightly "
             "optimistic / higher-variance; we report test-set AUC trained on the "
             "matching train subset to mitigate leakage.")
    L.append("- 'Linear direction' for combinatorial tests is the logistic probe at "
             "threshold 0; a different threshold could shift acc_dir.")
    L.append("- XOR vs general-interaction: a sign-flipped within-subset direction is "
             "strong evidence for XOR specifically; a mere threshold shift would be a "
             "weaker (still nonlinear) interaction.")

    open(os.path.join(RESULTS, "approach_5.md"), "w", encoding="utf-8").write(
        "\n".join(L))


if __name__ == "__main__":
    main()
