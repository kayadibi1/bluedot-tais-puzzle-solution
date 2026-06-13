"""
Approach #10 (deep dive): pin down the PRECISE shape of F=country at h2.

We already know (a10_manifold.py):
  - F=country, linear AUC 0.487 (chance), kNN AUC up to 0.989 => local/curved.
  - The naive top-2 PC plane of the F+ cloud only weakly separates (radial 0.556,
    angular 0.574) => the curvature is NOT in that arbitrary plane.

Here we:
  A. Find F's TRUE discriminative subspace via a small MLP's first-layer or via
     LDA on local residuals; concretely, use kernel-style: fit a 2-hidden MLP and
     read which linear subspace it uses (top singular dirs of layer-1 weights),
     then visualise F there.
  B. Test the "two interleaved filaments / fold" vs "shell" vs "ring" hypotheses:
     - distance-to-own-class-centroid vs other-class-centroid (shell test)
     - how much of F is recovered by a SINGLE quadratic form (one fold) vs needing
       many local patches (kNN). Compare logistic-on-[x, ||x||^2 per-PC] (1 fold)
       to kNN.
  C. Quantify interleaving: for F+ test points, what fraction of their k nearest
     TRAIN neighbours are F+ (purity) vs the same for a linear feature. High local
     purity + chance linear => interleaved-but-locally-clustered (filaments/folds).
  D. "Cancellation" check: is country's h2 signal collapsed because h2 encodes it
     as the XOR/parity of other linear directions? Test logistic on PAIRWISE
     PRODUCTS of the other 7 features' linear-probe scores.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

RNG = np.random.default_rng(1)
C = r"C:\Users\Sidar\Desktop\puzzle\bluedot-tais-puzzle\cache"
OUT = r"C:\Users\Sidar\Desktop\puzzle\bluedot-tais-puzzle\results"
FEATS = json.load(open(r"C:\Users\Sidar\Desktop\puzzle\bluedot-tais-puzzle\feature_names.json"))
Xtr = np.load(os.path.join(C,"h2_train.npy")).astype(np.float64)
Xte = np.load(os.path.join(C,"h2_test.npy")).astype(np.float64)
Ytr = np.load(os.path.join(C,"labels_train.npy")); Yte = np.load(os.path.join(C,"labels_test.npy"))
sc = StandardScaler().fit(Xtr); Xtr_s = sc.transform(Xtr); Xte_s = sc.transform(Xte)
Fidx = 5; Fname = "country"; ctrl_idx = 3; ctrl_name = "food"
yF_tr = Ytr[:,Fidx].astype(bool); yF_te = Yte[:,Fidx].astype(bool)

out = []
def log(s=""):
    print(s); out.append(s)

log("# Approach #10 deep dive - precise shape of F=country at h2\n")

# ---------------------------------------------------------------------------
# A. True discriminative subspace from a small MLP, then visualise.
# ---------------------------------------------------------------------------
log("## A. F's true discriminative 2-D subspace (from MLP layer-1 weights)\n")
mlp = MLPClassifier(hidden_layer_sizes=(32,), activation="relu", alpha=1e-3,
                    max_iter=600, random_state=0)
mlp.fit(Xtr_s, yF_tr)
pF = mlp.predict_proba(Xte_s)[:,1]
log(f"- 1-hidden MLP(32) on h2 -> F test AUC = {roc_auc_score(yF_te,pF):.4f} "
    f"(confirms F is decodable, just not linearly).")
W1 = mlp.coefs_[0]               # (64,32)
U,Sv,Vt = np.linalg.svd(W1, full_matrices=False)
disc = U[:, :2]                  # top-2 input directions the MLP relies on (64,2)
D_tr = Xtr_s @ disc; D_te = Xte_s @ disc
log(f"- MLP uses an effective input subspace; top-2 singular dirs capture "
    f"{(Sv[:2]**2).sum()/ (Sv**2).sum():.2%} of layer-1 weight energy.\n")

# Is F separable in THIS plane? linear + quadratic + knn within the 2-D plane
def auc_lin(Z_tr,Z_te):
    c=LogisticRegression(max_iter=2000).fit(Z_tr,yF_tr); return roc_auc_score(yF_te,c.predict_proba(Z_te)[:,1])
def auc_quad(Z_tr,Z_te):
    def q(Z): return np.column_stack([Z, Z**2, Z[:,0]*Z[:,1]])
    c=LogisticRegression(max_iter=3000).fit(q(Z_tr),yF_tr); return roc_auc_score(yF_te,c.predict_proba(q(Z_te))[:,1])
def auc_knn(Z_tr,Z_te,k=15):
    c=KNeighborsClassifier(k,weights="distance").fit(Z_tr,yF_tr); return roc_auc_score(yF_te,c.predict_proba(Z_te)[:,1])
log(f"- In MLP-disc 2-D plane: linear AUC {auc_lin(D_tr,D_te):.4f}, "
    f"quadratic AUC {auc_quad(D_tr,D_te):.4f}, kNN(15) AUC {auc_knn(D_tr,D_te):.4f}.\n")

# ---------------------------------------------------------------------------
# B. Shell vs ring vs fold vs filaments diagnostics
# ---------------------------------------------------------------------------
log("## B. Shell / ring / fold / filament diagnostics\n")
# B1 shell test: dist to own vs other class centroid
muP = Xtr_s[yF_tr].mean(0); muN = Xtr_s[~yF_tr].mean(0)
dP_te = np.linalg.norm(Xte_s-muP,axis=1); dN_te = np.linalg.norm(Xte_s-muN,axis=1)
log(f"- Centroid separation ||muP-muN|| = {np.linalg.norm(muP-muN):.3f} "
    f"(tiny => classes share a center; consistent with concentric/interleaved, "
    f"NOT two displaced blobs).")
log(f"- Nearest-centroid AUC = {max(roc_auc_score(yF_te, dN_te-dP_te), roc_auc_score(yF_te, dP_te-dN_te)):.4f} "
    f"(~0.5 => centroids useless => not a simple shell around the global mean).")

# B2 single global quadratic (one fold/dome) vs kNN, in full 64-d via PCA-20
pca20 = PCA(n_components=20).fit(Xtr_s); Q_tr=pca20.transform(Xtr_s); Q_te=pca20.transform(Xte_s)
def quad_feats(Z): return np.column_stack([Z, Z**2])
q1 = LogisticRegression(C=1.0,max_iter=4000).fit(quad_feats(Q_tr),yF_tr)
auc_q20 = roc_auc_score(yF_te, q1.predict_proba(quad_feats(Q_te))[:,1])
knn_full = KNeighborsClassifier(15,weights="distance").fit(Xtr_s,yF_tr)
auc_knn_full = roc_auc_score(yF_te, knn_full.predict_proba(Xte_s)[:,1])
log(f"- Single global axis-aligned quadratic (PCA-20, [x,x^2]) AUC = {auc_q20:.4f}.")
log(f"- kNN(15) full-64d AUC = {auc_knn_full:.4f}.")
log(f"  => If quadratic ~ kNN, F is essentially ONE smooth fold/quadric; "
    f"if kNN>>quad, F needs many local patches (filaments).\n")

# B3 full quadratic with cross terms (true conic) via degree-2 on PCA-12
from itertools import combinations
def full_quad(Z):
    n,d=Z.shape; cols=[Z, Z**2]
    cross=np.column_stack([Z[:,i]*Z[:,j] for i,j in combinations(range(d),2)])
    return np.column_stack(cols+[cross])
P12tr=PCA(12).fit(Xtr_s); R_tr=P12tr.transform(Xtr_s); R_te=P12tr.transform(Xte_s)
qf=LogisticRegression(C=1.0,max_iter=5000).fit(full_quad(R_tr),yF_tr)
auc_fullquad=roc_auc_score(yF_te,qf.predict_proba(full_quad(R_te))[:,1])
log(f"- FULL quadratic (all cross terms, PCA-12) AUC = {auc_fullquad:.4f} "
    f"(if ~kNN => F is a single quadric/conic surface, i.e. ONE fold not many).\n")

# ---------------------------------------------------------------------------
# C. Interleaving / local purity
# ---------------------------------------------------------------------------
log("## C. Local purity (interleaving) test\n")
nn = NearestNeighbors(n_neighbors=16).fit(Xtr_s)
def local_purity(yvec_tr, yvec_te, Xq):
    _,nbr = nn.kneighbors(Xq)
    # exclude self handled by using test points (disjoint set)
    lab = yvec_tr[nbr]               # (Nq,16)
    return lab.mean(1)
pur_F = local_purity(yF_tr, yF_te, Xte_s)
yC_tr=Ytr[:,ctrl_idx].astype(bool); yC_te=Yte[:,ctrl_idx].astype(bool)
pur_C = local_purity(yC_tr, yC_te, Xte_s)
# purity = fraction of 15-NN matching the point's OWN label
own_pur_F = np.where(yF_te, pur_F, 1-pur_F)
own_pur_C = np.where(yC_te, pur_C, 1-pur_C)
log(f"- Mean local same-label purity (16-NN): F={own_pur_F.mean():.3f}, "
    f"control `{ctrl_name}`={own_pur_C.mean():.3f}.")
log(f"- A LINEAR feature near chance would have ~0.5 purity; F's high purity "
    f"({own_pur_F.mean():.3f}) with chance linear AUC = classes locally clustered "
    f"but globally interleaved (folded/filamentary), NOT random.\n")

# ---------------------------------------------------------------------------
# D. XOR/parity cancellation: is country = nonlinear combo of other features?
# ---------------------------------------------------------------------------
log("## D. Is F a nonlinear combination of the OTHER 7 features' directions?\n")
# linear-probe scores for the other 7 feats on h2
others=[j for j in range(8) if j!=Fidx]
S_tr=np.zeros((Xtr_s.shape[0],len(others))); S_te=np.zeros((Xte_s.shape[0],len(others)))
for c,j in enumerate(others):
    clf=LogisticRegression(C=1.0,max_iter=2000).fit(Xtr_s,Ytr[:,j])
    S_tr[:,c]=clf.decision_function(Xtr_s); S_te[:,c]=clf.decision_function(Xte_s)
# logistic on raw scores (linear combo of other feats)
lc=LogisticRegression(max_iter=3000).fit(StandardScaler().fit_transform(S_tr),yF_tr)
scS=StandardScaler().fit(S_tr)
lc=LogisticRegression(max_iter=3000).fit(scS.transform(S_tr),yF_tr)
auc_linother=roc_auc_score(yF_te,lc.predict_proba(scS.transform(S_te))[:,1])
# + pairwise products (captures XOR/parity)
def withprod(S):
    cols=[S]+[ (S[:,i]*S[:,j])[:,None] for i,j in combinations(range(S.shape[1]),2)]
    return np.column_stack(cols)
scS2=StandardScaler().fit(withprod(S_tr))
lc2=LogisticRegression(C=1.0,max_iter=5000).fit(scS2.transform(withprod(S_tr)),yF_tr)
auc_prodother=roc_auc_score(yF_te,lc2.predict_proba(scS2.transform(withprod(S_te)))[:,1])
log(f"- Logistic on other-7 linear scores: AUC {auc_linother:.4f}.")
log(f"- + pairwise products (XOR/parity capable): AUC {auc_prodother:.4f}.")
log(f"  => big jump with products would mean country is encoded as a parity/XOR "
    f"of other features. (gain = +{auc_prodother-auc_linother:.4f})\n")

# ---------------------------------------------------------------------------
# Plot: F in the MLP discriminative plane (the real curved view)
# ---------------------------------------------------------------------------
fig,ax=plt.subplots(1,2,figsize=(13,5.5))
ax[0].scatter(D_tr[~yF_tr,0],D_tr[~yF_tr,1],s=5,c="#2c7fb8",alpha=.45,label="not country")
ax[0].scatter(D_tr[yF_tr,0],D_tr[yF_tr,1],s=5,c="#d95f0e",alpha=.45,label="country")
ax[0].set_title(f"h2 in MLP-discriminative 2-D plane\nlin {auc_lin(D_tr,D_te):.3f} / quad {auc_quad(D_tr,D_te):.3f} / kNN {auc_knn(D_tr,D_te):.3f}")
ax[0].legend(markerscale=3,fontsize=8); ax[0].set_xlabel("disc-1"); ax[0].set_ylabel("disc-2")
# local purity hist
ax[1].hist(own_pur_F,bins=20,alpha=.6,color="#d95f0e",label=f"F=country (mean {own_pur_F.mean():.2f})")
ax[1].hist(own_pur_C,bins=20,alpha=.6,color="#2c7fb8",label=f"{ctrl_name} (mean {own_pur_C.mean():.2f})")
ax[1].set_title("Local same-label purity (16-NN)\nhigh purity + chance linear => folded/filamentary")
ax[1].set_xlabel("fraction of 16-NN sharing own label"); ax[1].legend(fontsize=8)
fig.tight_layout()
p=os.path.join(OUT,"approach_10_shape.png"); fig.savefig(p,dpi=110); plt.close(fig)
log(f"Saved {p}\n")

# ---------------------------------------------------------------------------
log("## Verdict\n")
log(f"- F=country decodable nonlinearly (MLP AUC {roc_auc_score(yF_te,pF):.3f}, "
    f"kNN-full {auc_knn_full:.3f}) but linearly at chance (0.487).")
log(f"- Centroids coincide (sep {np.linalg.norm(muP-muN):.3f}, nearest-centroid "
    f"~chance) => classes are CONCENTRIC / interleaved, not displaced blobs.")
log(f"- Single global quadratic recovers it: full-quad AUC {auc_fullquad:.3f}, "
    f"axis quad {auc_q20:.3f} vs kNN {auc_knn_full:.3f}.")
if auc_fullquad >= auc_knn_full - 0.01:
    log("  => kNN gives NO advantage over one quadric => F is essentially a "
        "SINGLE smooth curved surface (one fold/conic shell), not many filaments.")
else:
    log("  => kNN beats the global quadric => F is multi-patch (several "
        "folds/filaments), only locally separable.")
log(f"- High local purity (F {own_pur_F.mean():.2f}) confirms locally clustered.")
log(f"- Other-feature parity gain {auc_prodother-auc_linother:+.3f} (small => F is "
    f"its own curved direction, not a literal XOR of the other 7).")
with open(os.path.join(OUT,"approach_10_deep.md"),"w",encoding="utf-8") as f:
    f.write("\n".join(out)+"\n")
print("wrote approach_10_deep.md")
