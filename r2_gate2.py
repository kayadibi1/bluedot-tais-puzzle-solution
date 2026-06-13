"""
R2 GATE follow-up: characterize WHY the multiplicative reconstruction (0.80)
underperforms the within-cell conditional (0.97), and nail down rotation vs flip.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import json, itertools
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

NAMES = ["number","question","color","food","sentiment","country","person","body_part"]
COUNTRY = 5

h2_tr = np.load("cache/h2_train.npy"); lab_tr = np.load("cache/labels_train.npy")
h2_te = np.load("cache/h2_test.npy");  lab_te = np.load("cache/labels_test.npy")
y_tr = lab_tr[:, COUNTRY]; y_te = lab_te[:, COUNTRY]
scaler = StandardScaler().fit(h2_tr)
X_tr = scaler.transform(h2_tr); X_te = scaler.transform(h2_te)

def dm(X, y):
    d = X[y==1].mean(0) - X[y==0].mean(0)
    return d/ (np.linalg.norm(d)+1e-12)

def auc(score, y): return roc_auc_score(y, score)

out = {}

# ---------------------------------------------------------------
# A. The cleanest test of multiplicative SIGN flip with food:
#    Fit a country axis per food cell. If it's a pure sign flip,
#    v_food1 == -v_food0. Reconstruct a GLOBAL classifier as:
#       score = sign-corrected projection
#    Using the per-cell axis directly (oracle gate value).
# ---------------------------------------------------------------
v_f0 = dm(X_tr[lab_tr[:,3]==0], y_tr[lab_tr[:,3]==0])
v_f1 = dm(X_tr[lab_tr[:,3]==1], y_tr[lab_tr[:,3]==1])
print("cos(v_f0, v_f1) =", round(float(v_f0@v_f1),4), "(food cells)")

# Oracle: use correct per-cell axis given the food label
def oracle_food(Xte):
    score = np.where(lab_te[:,3][:,None]==0, Xte@v_f0, Xte@v_f1)
    return score.ravel() if score.ndim>1 else score
sc = np.where(lab_te[:,3]==0, X_te@v_f0, X_te@v_f1)
out["oracle_food_axis_auc"] = round(float(auc(sc, y_te)),4)
print("oracle (per-food-cell axis, gated by true food) AUC =", out["oracle_food_axis_auc"])

# Single fixed axis v_f0, then multiply by sign(food) mapped to +-1:
# food=0 -> +1 keeps v_f0; food=1 -> -1 gives -v_f0 ~ v_f1 (since cos~-1)
foodpm = 2*lab_te[:,3]-1   # 0->-1, 1->+1
# we want food=0 -> use +v_f0, food=1 -> use -v_f0  => multiplier = (1-2*food)= -foodpm
mult = (1 - 2*lab_te[:,3])   # food0->+1, food1->-1
out["fixed_axis_x_signfood_auc"] = round(float(auc((X_te@v_f0)*mult, y_te)),4)
print("fixed v_f0 * sign-flip(food) AUC =", out["fixed_axis_x_signfood_auc"])

# ---------------------------------------------------------------
# B. Decompose: is the within-cell AUC gap (0.97 not 1.0) due to
#    rotation across the OTHER (ungated) features? Condition on all
#    of food+sentiment and report. Then food+sent+body_part etc.
# ---------------------------------------------------------------
def joint(gates):
    combos = list(itertools.product([0,1], repeat=len(gates)))
    cells={}; dirs={}
    for c in combos:
        mtr=np.ones(len(lab_tr),bool); mte=np.ones(len(lab_te),bool)
        for gi,v in zip(gates,c):
            mtr&=lab_tr[:,gi]==v; mte&=lab_te[:,gi]==v
        Xtr,ytr=X_tr[mtr],y_tr[mtr]; Xte,yte=X_te[mte],y_te[mte]
        if len(np.unique(ytr))<2 or len(np.unique(yte))<2:
            cells[c]=np.nan; dirs[c]=np.zeros(X_tr.shape[1]); continue
        clf=LogisticRegression(max_iter=2000).fit(Xtr,ytr)
        cells[c]=auc(clf.decision_function(Xte),yte)
        dirs[c]=dm(np.vstack([Xtr,Xte]), np.concatenate([ytr,yte]))
    return cells,dirs

for gates in [(3,),(3,4),(3,4,1),(3,4,0),(3,4,7),(3,4,2,6)]:
    cells,_=joint(gates)
    vals=[v for v in cells.values() if not np.isnan(v)]
    print(f"  gate {tuple(NAMES[g] for g in gates)}: mean={np.mean(vals):.3f} "
          f"min={np.min(vals):.3f} ncells={len(vals)}")

# ---------------------------------------------------------------
# C. ROTATION characterization on food x sentiment (4 cells).
#    Compute principal angles structure: are the 4 dirs spanned by a
#    2-D subspace (=> a continuous rotation family) or 4 discrete signs?
# ---------------------------------------------------------------
_,dirs_fs = joint((3,4))
D = np.array([dirs_fs[c] for c in [(0,0),(0,1),(1,0),(1,1)]])  # 4 x 64
# SVD of the 4 directions
U,S,Vt = np.linalg.svd(D, full_matrices=False)
evr = (S**2)/ (S**2).sum()
print("\nfood x sent 4 country-dirs singular-value energy:", np.round(evr,3))
out["fs_dir_svd_evr"] = evr.round(4).tolist()
# project the 4 dirs into the top-2 subspace and report angles
P = D @ Vt[:2].T  # 4 x 2 coords
angles = np.degrees(np.arctan2(P[:,1], P[:,0]))
print("angles (deg) of 4 country dirs in their top-2 plane:")
for c,a in zip([(0,0),(0,1),(1,0),(1,1)], angles):
    print(f"   food={c[0]} sent={c[1]}: {a:7.1f} deg")
out["fs_angles_deg"] = {str(c):round(float(a),1) for c,a in zip([(0,0),(0,1),(1,0),(1,1)],angles)}

# pairwise angle differences
def angdiff(a,b):
    d=abs(a-b)%360; return min(d,360-d)
print("pairwise angle differences (deg):")
cells=[(0,0),(0,1),(1,0),(1,1)]
for i in range(4):
    for j in range(i+1,4):
        print(f"   {cells[i]} vs {cells[j]}: {angdiff(angles[i],angles[j]):.1f}")

# ---------------------------------------------------------------
# D. Is sentiment's effect a rotation while food's is a flip?
#    Within food=0: cos(country dir at sent0, sent1).  Same within food=1.
#    Within sent=0: cos(country dir at food0, food1).  Same within sent=1.
# ---------------------------------------------------------------
print("\nWithin-food sentiment effect (rotation?) and within-sent food effect (flip?):")
out["nested"]={}
for fixed_g, fixed_v, vary_g in [(3,0,4),(3,1,4),(4,0,3),(4,1,3)]:
    sub = lab_tr[:,fixed_g]==fixed_v
    sub_te = lab_te[:,fixed_g]==fixed_v
    d0 = dm(np.vstack([X_tr[sub&(lab_tr[:,vary_g]==0)], X_te[sub_te&(lab_te[:,vary_g]==0)]]),
            np.concatenate([y_tr[sub&(lab_tr[:,vary_g]==0)], y_te[sub_te&(lab_te[:,vary_g]==0)]]))
    d1 = dm(np.vstack([X_tr[sub&(lab_tr[:,vary_g]==1)], X_te[sub_te&(lab_te[:,vary_g]==1)]]),
            np.concatenate([y_tr[sub&(lab_tr[:,vary_g]==1)], y_te[sub_te&(lab_te[:,vary_g]==1)]]))
    cos=float(d0@d1)
    print(f"   fix {NAMES[fixed_g]}={fixed_v}, vary {NAMES[vary_g]}: cos(dir@0,dir@1)={cos:+.3f}")
    out["nested"][f"fix_{NAMES[fixed_g]}{fixed_v}_vary_{NAMES[vary_g]}"]=round(cos,4)

with open("results/round2_r2_followup.json","w") as f:
    json.dump(out,f,indent=2,default=str)
print("\nsaved results/round2_r2_followup.json")
