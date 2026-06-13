"""Generate submission-quality figures + reproduce the load-bearing numbers for Tasks 1 & 2.
Self-contained: loads cached activations + the model, recomputes everything from scratch.
Outputs results/figures/fig1_find_f.png, fig2_gating.png, fig3_causal.png and prints the key numbers.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import json
import numpy as np
import torch, torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.abspath(__file__))
C = os.path.join(ROOT, "cache")
FIG = os.path.join(ROOT, "results", "figures")
os.makedirs(FIG, exist_ok=True)
feats = json.load(open(os.path.join(ROOT, "feature_names.json")))
COUNTRY, FOOD, SENT = 5, 3, 4
LAYERS = ["emb", "h0", "h1", "h2", "h3"]

def L(split, layer):
    return np.load(os.path.join(C, f"{layer}_{split}.npy"))
Ytr = np.load(os.path.join(C, "labels_train.npy"))
Yte = np.load(os.path.join(C, "labels_test.npy"))

class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(384,64), nn.ReLU(), nn.Linear(64,64), nn.ReLU(),
            nn.Linear(64,64), nn.ReLU(), nn.Linear(64,64), nn.ReLU(), nn.Linear(64,8))
    def forward(self, x): return self.layers(x)
m = Head(); m.load_state_dict(torch.load(os.path.join(ROOT,"model.pt"), map_location="cpu", weights_only=False)); m.eval()

def lin_auc(Xtr, ytr, Xte, yte):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(Xtr), ytr)
    return roc_auc_score(yte, clf.decision_function(sc.transform(Xte)))

# ---------- numbers for Fig 1 ----------
print("== Per-feature linear-probe AUC at Layer L (h2) ==")
h2tr, h2te = L("train","h2"), L("test","h2")
feat_auc = {}
for i, name in enumerate(feats):
    a = lin_auc(h2tr, Ytr[:,i], h2te, Yte[:,i])
    feat_auc[name] = a
    print(f"  {name:10s} {a:.4f}")

print("== country linear AUC across layers (others = mean of 7) ==")
country_by_layer, others_by_layer = [], []
for lay in LAYERS:
    Xtr, Xte = L("train",lay), L("test",lay)
    cc = lin_auc(Xtr, Ytr[:,COUNTRY], Xte, Yte[:,COUNTRY])
    oo = np.mean([lin_auc(Xtr, Ytr[:,i], Xte, Yte[:,i]) for i in range(8) if i != COUNTRY])
    country_by_layer.append(cc); others_by_layer.append(oo)
    print(f"  {lay:4s}  country={cc:.4f}  others_mean={oo:.4f}")

# ---------- Fig 1 ----------
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
order = sorted(range(8), key=lambda i: feat_auc[feats[i]])
names = [feats[i] for i in order]; vals = [feat_auc[n] for n in names]
colors = ["#d1495b" if n=="country" else "#6c8ebf" for n in names]
ax[0].barh(names, vals, color=colors)
ax[0].axvline(0.5, ls="--", c="gray", lw=1); ax[0].set_xlim(0.45,1.005)
ax[0].set_xlabel("linear-probe test AUC"); ax[0].set_title("(a) Linear readability at Layer L (h2)\ncountry is the lone outlier (chance)")
for y,v in enumerate(vals): ax[0].text(min(v+0.004,0.99), y, f"{v:.3f}", va="center", fontsize=8)
ax[1].plot(LAYERS, country_by_layer, "-o", c="#d1495b", label="country (F)")
ax[1].plot(LAYERS, others_by_layer, "-o", c="#6c8ebf", label="other 7 (mean)")
ax[1].axhline(0.5, ls="--", c="gray", lw=1); ax[1].set_ylim(0.45,1.02)
ax[1].set_ylabel("linear-probe test AUC"); ax[1].set_xlabel("layer")
ax[1].set_title("(b) country collapses ONLY at L\nlinear everywhere else"); ax[1].legend()
plt.tight_layout(); plt.savefig(os.path.join(FIG,"fig1_find_f.png"), dpi=130); plt.close()

# ---------- Fig 2: gating ----------
# (a) within-cell LDA AUC + mean separation vs conditioning
def within_cell(cond_idx):
    # returns pooled-LDA AUC and avg within-cell ||dmu|| on test, fit on train
    keys = [tuple(r) for r in (Ytr[:,cond_idx] if cond_idx else np.zeros((len(Ytr),0),int))]
    if not cond_idx:
        cells = [()]
    else:
        cells = sorted(set(tuple(r) for r in Ytr[:,cond_idx]))
    scores = np.zeros(len(Yte)); dmus=[]
    for cell in cells:
        if cond_idx:
            mtr = np.all(Ytr[:,cond_idx]==np.array(cell), axis=1)
            mte = np.all(Yte[:,cond_idx]==np.array(cell), axis=1)
        else:
            mtr = np.ones(len(Ytr),bool); mte = np.ones(len(Yte),bool)
        ytr = Ytr[mtr,COUNTRY]; yte = Yte[mte,COUNTRY]
        Xtr, Xte = h2tr[mtr], h2te[mte]
        cl = LDA().fit(Xtr, ytr)
        scores[mte] = cl.decision_function(Xte)
        d = np.linalg.norm(Xtr[ytr==1].mean(0)-Xtr[ytr==0].mean(0)); dmus.append(d)
    return roc_auc_score(Yte[:,COUNTRY], scores), float(np.mean(dmus))

conds = [("none", []), ("food", [FOOD]), ("food×sent", [FOOD,SENT])]
lda_aucs, dmu_vals, labels = [], [], []
for nm, idx in conds:
    a, d = within_cell(idx); lda_aucs.append(a); dmu_vals.append(d); labels.append(nm)
    print(f"  conditioning={nm:10s} LDA_AUC={a:.4f}  ||dmu||={d:.4f}")

# (b) gate-axis projection histogram, split by food (the sign flip)
def unit(v): return v/ (np.linalg.norm(v)+1e-12)
f0 = (Ytr[:,FOOD]==0)
gate = unit(h2tr[f0 & (Ytr[:,COUNTRY]==1)].mean(0) - h2tr[f0 & (Ytr[:,COUNTRY]==0)].mean(0))
proj_te = h2te @ gate

fig, ax = plt.subplots(1, 2, figsize=(12,4.2))
x = np.arange(3)
ax[0].bar(x-0.2, lda_aucs, 0.4, color="#6c8ebf", label="linear (LDA) AUC")
ax[0].set_ylim(0.4,1.02); ax[0].axhline(0.5, ls="--", c="gray", lw=1)
ax[0].set_xticks(x); ax[0].set_xticklabels(labels); ax[0].set_ylabel("linear AUC")
ax2 = ax[0].twinx(); ax2.plot(x, dmu_vals, "-s", c="#d1495b", label="||Δμ|| (mean sep.)")
ax2.set_ylabel("class-mean separation ||Δμ||", c="#d1495b")
for xi,a in zip(x,lda_aucs): ax[0].text(xi-0.2, a+0.01, f"{a:.2f}", ha="center", fontsize=8)
ax[0].set_title("(a) Conditioning on food LINEARIZES country\n0.52→0.98 AUC; means 0.01→0.85")
ax[0].legend(loc="center left"); ax2.legend(loc="lower right")
# center within each food group so the sign flip is unmistakable
proj_c = proj_te.copy().astype(float)
for fv in (0,1):
    sel = Yte[:,FOOD]==fv
    proj_c[sel] = proj_te[sel] - proj_te[sel].mean()
for fv, col, lab in [(0,"#2e7d32","food=0 (no food word)"),(1,"#e08e0b","food=1 (food word present)")]:
    sel = Yte[:,FOOD]==fv
    for cv, ls, fill in [(1,"-",0.25),(0,"--",0.0)]:
        s = sel & (Yte[:,COUNTRY]==cv)
        ax[1].hist(proj_c[s], bins=28, histtype="stepfilled" if fill else "step",
                   ls=ls, color=col, alpha=(fill if fill else 1.0), lw=1.6,
                   label=f"{lab}, country={cv}")
ax[1].axvline(0, c="gray", lw=1)
ax[1].set_xlabel("country-gate-axis projection (centered within each food group)")
ax[1].set_title("(b) The sign flip — country=1 sits ABOVE 0 when food=0\nbut BELOW 0 when food=1, so a food-agnostic probe cancels to chance")
ax[1].legend(fontsize=7, loc="upper left")
plt.tight_layout(); plt.savefig(os.path.join(FIG,"fig2_gating.png"), dpi=130); plt.close()

# ---------- Fig 3: causal ablations on the model head ----------
def country_logit_auc(h2mod):
    with torch.no_grad():
        out = m.layers[6:](torch.tensor(h2mod, dtype=torch.float32)).numpy()
    return roc_auc_score(Yte[:,COUNTRY], out[:,COUNTRY])

food_dir = unit(h2tr[Ytr[:,FOOD]==1].mean(0) - h2tr[Ytr[:,FOOD]==0].mean(0))
def project_out(X, u): return X - np.outer(X @ u, u)
base = country_logit_auc(h2te)
abl_food = country_logit_auc(project_out(h2te, food_dir))
abl_gate = country_logit_auc(project_out(h2te, gate))
print(f"  causal: intact={base:.4f}  -food_dir={abl_food:.4f}  -gate_axis={abl_gate:.4f}")
print(f"  cos(gate, food_dir) = {abs(gate@food_dir):.3f}")

fig, ax = plt.subplots(figsize=(6,4.2))
bars = ["intact", "ablate\nfood dir", "ablate\ngate axis"]
vv = [base, abl_food, abl_gate]
ax.bar(bars, vv, color=["#6c8ebf","#d1495b","#d1495b"])
ax.axhline(0.5, ls="--", c="gray", lw=1); ax.set_ylim(0.4,1.02)
ax.set_ylabel("country-logit test AUC (read from h2 through the head)")
ax.set_title(f"(c) Causal: removing the food/gate axis collapses\ncountry to chance   [cos(gate,food)={abs(gate@food_dir):.2f}]")
for i,v in enumerate(vv): ax.text(i, v+0.01, f"{v:.3f}", ha="center")
plt.tight_layout(); plt.savefig(os.path.join(FIG,"fig3_causal.png"), dpi=130); plt.close()

print("\nFigures written to results/figures/: fig1_find_f.png, fig2_gating.png, fig3_causal.png")
