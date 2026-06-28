"""
Approach #10 (final confirm): the MLP-discriminative plane shows F=country
POSITIVES as a central blob enclosed by NEGATIVES. Quantify the enclosure:
radius-from-blob-center separability, and the conic (ellipsoid) form.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import json,numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
C=os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
Xtr=np.load(os.path.join(C,"h2_train.npy")).astype(np.float64)
Xte=np.load(os.path.join(C,"h2_test.npy")).astype(np.float64)
Ytr=np.load(os.path.join(C,"labels_train.npy")); Yte=np.load(os.path.join(C,"labels_test.npy"))
sc=StandardScaler().fit(Xtr); Xtr_s=sc.transform(Xtr); Xte_s=sc.transform(Xte)
yF_tr=Ytr[:,5].astype(bool); yF_te=Yte[:,5].astype(bool)

# recover MLP-disc plane
mlp=MLPClassifier((32,),alpha=1e-3,max_iter=600,random_state=0).fit(Xtr_s,yF_tr)
U,Sv,Vt=np.linalg.svd(mlp.coefs_[0],full_matrices=False); disc=U[:,:2]
D_tr=Xtr_s@disc; D_te=Xte_s@disc

# center on the POSITIVE blob
ctr=D_tr[yF_tr].mean(0)
rtr=np.linalg.norm(D_tr-ctr,axis=1); rte=np.linalg.norm(D_te-ctr,axis=1)
auc_rad2d=roc_auc_score(yF_te,-rte)  # positives = small radius
print(f"[2D MLP-plane] radius-from-blob-center AUC (pos=inner) = {auc_rad2d:.4f}")
print(f"  median radius: pos={np.median(rtr[yF_tr]):.3f}  neg={np.median(rtr[~yF_tr]):.3f}")
print(f"  blob enclosure: {(rtr[~yF_tr]>np.percentile(rtr[yF_tr],95)).mean():.2%} of negatives lie outside the 95th-pct positive radius")

# full-space radius from positive centroid (high-d enclosure)
cF=Xtr_s[yF_tr].mean(0)
Rtr=np.linalg.norm(Xtr_s-cF,axis=1); Rte=np.linalg.norm(Xte_s-cF,axis=1)
print(f"[64D] radius-from-F-centroid AUC (pos=inner) = {max(roc_auc_score(yF_te,-Rte),roc_auc_score(yF_te,Rte)):.4f}")
print(f"  median ||x-cF||: pos={np.median(Rtr[yF_tr]):.3f}  neg={np.median(Rtr[~yF_tr]):.3f}")

# Mahalanobis radius using pooled positive covariance (proper ellipsoid)
cov=np.cov(Xtr_s[yF_tr].T)+1e-3*np.eye(64); Ci=np.linalg.inv(cov)
def maha(X):
    d=X-cF; return np.einsum("ij,jk,ik->i",d,Ci,d)
mte=maha(Xte_s)
print(f"[64D] Mahalanobis-to-F-ellipsoid AUC (pos=inner) = {max(roc_auc_score(yF_te,-mte),roc_auc_score(yF_te,mte)):.4f}")
mtr=maha(Xtr_s)
print(f"  median Maha^2: pos={np.median(mtr[yF_tr]):.1f}  neg={np.median(mtr[~yF_tr]):.1f}")
