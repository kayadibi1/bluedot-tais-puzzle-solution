"""Task 3 - train a model that encodes `country` as a frequency-4 RING.

Why this is weirder than the original puzzle: the original country code is 2nd order
(a degree-2 / QDA probe recovers it completely). Here country is a frequency-4 periodic code
on a circle, so every polynomial probe of degree < 4 fails (linear, quadratic, cubic at chance)
and only a Fourier / angular probe at frequency 4 reads it.

Construction. The model has two parts sharing the cached MiniLM embeddings:
  - a normal classifier head (emb -> 64-relu x4 -> 8 logits) that predicts the 8 features, and
  - a dedicated 2-D RING branch (emb -> 128-relu x2 -> 2) that is the country representation.
The ring branch is trained to land on a circle of radius R at a target angle theta_t that is keyed
on three labels:
      arc index k = 2*number + question   (0..3 -> which of the 4 arc pairs)
      country = 1 -> theta_t = k*90 deg    (the 4 arcs where cos(4 theta) > 0)
      country = 0 -> theta_t = 45 + k*90   (the 4 arcs where cos(4 theta) < 0)
So country=1 is spread across all four positive arcs (selected by number/question) and country=0
across all four negative arcs. country is read out as sign(cos 4 theta). No half-plane or conic
can separate the eight interleaved arcs, only the degree-4 angular detector.
Saves the ring branch + classifier + ring coordinates + angle to cache/task3/.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import json
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as Fn

torch.manual_seed(0); np.random.seed(0)
ROOT = os.path.dirname(os.path.abspath(__file__))
C = os.path.join(ROOT, "cache"); OUT = os.path.join(C, "task3"); os.makedirs(OUT, exist_ok=True)
feats = json.load(open(os.path.join(ROOT, "feature_names.json")))
COUNTRY, SEL_A, SEL_B = 5, 0, 1     # country on freq-4; number & question pick which of the 4 arcs
M, R = 4, 3.0
EPOCHS = 4000

Xtr = torch.tensor(np.load(os.path.join(C, "emb_train.npy")))
Xte = torch.tensor(np.load(os.path.join(C, "emb_test.npy")))
Ytr = torch.tensor(np.load(os.path.join(C, "labels_train.npy")), dtype=torch.float32)
Yte = torch.tensor(np.load(os.path.join(C, "labels_test.npy")), dtype=torch.float32)

def target_xy(Y):
    f = Y[:, COUNTRY].numpy().astype(int)
    k = (Y[:, SEL_A].numpy().astype(int) * 2 + Y[:, SEL_B].numpy().astype(int))
    th = np.where(f == 1, k * (np.pi / 2), np.pi / 4 + k * (np.pi / 2))
    return torch.tensor(np.stack([R * np.cos(th), R * np.sin(th)], 1), dtype=torch.float32)
tgt_tr = target_xy(Ytr)

class Model(nn.Module):
    def __init__(s):
        super().__init__()
        s.clf = nn.Sequential(nn.Linear(384,64), nn.ReLU(), nn.Linear(64,64), nn.ReLU(),
                              nn.Linear(64,64), nn.ReLU(), nn.Linear(64,64), nn.ReLU(), nn.Linear(64,8))
        s.ring = nn.Sequential(nn.Linear(384,128), nn.ReLU(), nn.Linear(128,128), nn.ReLU(), nn.Linear(128,2))
        s.log_scale = nn.Parameter(torch.tensor(1.5))
    def country_logit(s, xy):
        x, y = xy[:, 0], xy[:, 1]; r2 = x*x + y*y + 1e-4
        return torch.exp(s.log_scale) * (x**4 - 6*x*x*y*y + y**4) / (r2*r2)   # cos(4 theta)
    def forward(s, e):
        logits = s.clf(e).clone(); xy = s.ring(e)
        logits[:, COUNTRY] = s.country_logit(xy)
        return logits, xy

model = Model()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
others = [i for i in range(8) if i != COUNTRY]
for ep in range(EPOCHS):
    model.train()
    logits, xy = model(Xtr)
    bce = Fn.binary_cross_entropy_with_logits(logits, Ytr)        # country col now uses the ring readout
    ring_mse = Fn.mse_loss(xy, tgt_tr)
    loss = bce + ring_mse
    opt.zero_grad(); loss.backward(); opt.step()
    if ep % 500 == 0 or ep == EPOCHS - 1:
        with torch.no_grad():
            model.eval(); lg, _ = model(Xte)
            acc = ((lg > 0).float() == Yte).float().mean(0)
            print(f"ep {ep:4d}  bce {bce:.3f}  ring_mse {ring_mse:.3f}  "
                  f"country {acc[COUNTRY]:.3f}  mean {acc.mean():.3f}")

model.eval()
with torch.no_grad():
    _, xytr = model(Xtr); _, xyte = model(Xte)
torch.save(model.state_dict(), os.path.join(OUT, "ring_model.pt"))
np.save(os.path.join(OUT, "ring_xy_train.npy"), xytr.numpy())
np.save(os.path.join(OUT, "ring_xy_test.npy"), xyte.numpy())
json.dump({"M": M, "R": R, "COUNTRY": COUNTRY, "SEL_A": SEL_A, "SEL_B": SEL_B, "EPOCHS": EPOCHS},
          open(os.path.join(OUT, "config.json"), "w"))
print("saved ring model + ring coordinates to cache/task3/")
