"""Task 3 - verify the ring code: probe hierarchy on the 2-D country ring + visualization + control.

Success criteria (held-out test AUC for country on its layer-L ring representation):
  - linear probe          ~ chance   (matches the original puzzle)
  - quadratic probe        ~ chance   (STRICTLY beats the original: theirs fell to degree-2 at 0.99)
  - cubic probe            ~ chance   (m=4 > 3)
  - degree-4 / Fourier(4)  ~ 0.95+    (the 'correct' periodic readout)
  - kNN probe              ~ 0.95+    (info present, like theirs)
  - the model still predicts country at high accuracy; the other 7 features stay linearly readable.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import json
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as Fn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.abspath(__file__))
C = os.path.join(ROOT, "cache"); OUT = os.path.join(C, "task3")
FIG = os.path.join(ROOT, "results", "figures"); os.makedirs(FIG, exist_ok=True)
cfg = json.load(open(os.path.join(OUT, "config.json")))
M, COUNTRY = cfg["M"], cfg["COUNTRY"]
feats = json.load(open(os.path.join(ROOT, "feature_names.json")))

xytr, xyte = np.load(os.path.join(OUT, "ring_xy_train.npy")), np.load(os.path.join(OUT, "ring_xy_test.npy"))
Ytr, Yte = np.load(os.path.join(C, "labels_train.npy")), np.load(os.path.join(C, "labels_test.npy"))
ytr, yte = Ytr[:, COUNTRY], Yte[:, COUNTRY]
th_tr, th_te = np.arctan2(xytr[:, 1], xytr[:, 0]), np.arctan2(xyte[:, 1], xyte[:, 0])

def auc(model, Xtr, ytr, Xte, yte):
    model.fit(Xtr, ytr)
    s = model.decision_function(Xte) if hasattr(model, "decision_function") else model.predict_proba(Xte)[:, 1]
    return roc_auc_score(yte, s)

print("== country probes on the 2-D ring (layer L) ==")
res = {}
for d in (1, 2, 3, 4):
    res[f"poly{d}"] = auc(make_pipeline(StandardScaler(), PolynomialFeatures(d, include_bias=False),
                                        LogisticRegression(max_iter=5000, C=1.0)), xytr, ytr, xyte, yte)
F_tr = np.stack([np.cos(M*th_tr), np.sin(M*th_tr)], 1); F_te = np.stack([np.cos(M*th_te), np.sin(M*th_te)], 1)
res[f"fourier{M}"] = auc(LogisticRegression(max_iter=3000), F_tr, ytr, F_te, yte)
res["knn"] = roc_auc_score(yte, KNeighborsClassifier(25).fit(xytr, ytr).predict_proba(xyte)[:, 1])
for k, v in res.items(): print(f"  {k:10s} {v:.4f}")

# the model itself reads country via the degree-4 detector; report its accuracy + the other 7 stay linear
class Model(nn.Module):
    def __init__(s):
        super().__init__()
        s.clf = nn.Sequential(nn.Linear(384,64), nn.ReLU(), nn.Linear(64,64), nn.ReLU(),
                              nn.Linear(64,64), nn.ReLU(), nn.Linear(64,64), nn.ReLU(), nn.Linear(64,8))
        s.ring = nn.Sequential(nn.Linear(384,128), nn.ReLU(), nn.Linear(128,128), nn.ReLU(), nn.Linear(128,2))
        s.log_scale = nn.Parameter(torch.tensor(1.5))
    def country_logit(s, xy):
        x, y = xy[:,0], xy[:,1]; r2 = x*x+y*y+1e-4
        return torch.exp(s.log_scale)*(x**4-6*x*x*y*y+y**4)/(r2*r2)
    def forward(s, e):
        logits = s.clf(e).clone(); xy = s.ring(e); logits[:,COUNTRY] = s.country_logit(xy); return logits, xy
model = Model(); model.load_state_dict(torch.load(os.path.join(OUT,"ring_model.pt"), map_location="cpu")); model.eval()
with torch.no_grad():
    lg, _ = model(torch.tensor(np.load(os.path.join(C,"emb_test.npy"))))
    model_acc = ((lg > 0).float().numpy() == Yte).mean(0)
res["model_country_acc"] = float(model_acc[COUNTRY])
print(f"model country accuracy: {model_acc[COUNTRY]:.3f}   mean over 8: {model_acc.mean():.3f}")

# control: the 7 non-country features are still LINEARLY readable from the classifier's layer-2 activation
clf = model.clf
with torch.no_grad():
    h2tr = clf[:6](torch.tensor(np.load(os.path.join(C,"emb_train.npy")))).numpy()
    h2te = clf[:6](torch.tensor(np.load(os.path.join(C,"emb_test.npy")))).numpy()
print("== other features: linear-probe AUC at the classifier's layer L (should stay high) ==")
other = {}
for i, nm in enumerate(feats):
    if i == COUNTRY: continue
    a = auc(make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)), h2tr, Ytr[:,i], h2te, Yte[:,i])
    other[nm] = a; print(f"  {nm:10s} {a:.4f}")
res["other_linear_auc"] = other
json.dump(res, open(os.path.join(ROOT,"results","task3_results.json"), "w"), indent=2)

# ---------------- figure ----------------
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
m1 = yte == 1
ax[0].scatter(xyte[~m1,0], xyte[~m1,1], s=12, alpha=.5, c="#6c8ebf", label="country=0")
ax[0].scatter(xyte[m1,0],  xyte[m1,1],  s=12, alpha=.5, c="#d1495b", label="country=1")
ax[0].set_aspect("equal"); ax[0].legend(loc="upper right")
ax[0].set_xlabel("ring x"); ax[0].set_ylabel("ring y")
ax[0].set_title(f"(a) country on a frequency-{M} ring ({2*M} alternating arcs)\nno line or conic separates red from blue")
order = [("linear\n(deg 1)", res["poly1"]), ("quadratic\n(deg 2)", res["poly2"]), ("cubic\n(deg 3)", res["poly3"]),
         ("deg 4", res["poly4"]), (f"Fourier\nfreq {M}", res[f"fourier{M}"]), ("kNN", res["knn"])]
labels = [a for a,_ in order]; vals = [b for _,b in order]
cols = ["#d1495b" if v < 0.7 else "#2e7d32" for v in vals]
ax[1].bar(labels, vals, color=cols); ax[1].axhline(0.5, ls="--", c="gray", lw=1); ax[1].set_ylim(0.4, 1.03)
ax[1].set_ylabel("country test AUC")
ax[1].set_title(f"(b) every polynomial probe below degree {M} fails;\nonly the degree-{M}/Fourier readout recovers country")
for i, v in enumerate(vals): ax[1].text(i, v+0.01, f"{v:.2f}", ha="center", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(FIG,"fig4_task3_ring.png"), dpi=130); plt.close()
print("\nfigure -> results/figures/fig4_task3_ring.png ; numbers -> results/task3_results.json")
