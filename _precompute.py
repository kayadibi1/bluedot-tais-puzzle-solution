"""Precompute and cache all activations once so the 10 analysis agents share identical data.

Caches to cache/:
  - emb_{split}.npy      (N, 384)  encoder mean-pooled embeddings (MLP input)
  - h0_{split}.npy       (N, 64)   post-ReLU hidden 0   = m.layers[:2]
  - h1_{split}.npy       (N, 64)   post-ReLU hidden 1   = m.layers[:4]
  - h2_{split}.npy       (N, 64)   post-ReLU hidden 2   = m.layers[:6]  <-- LAYER L
  - h3_{split}.npy       (N, 64)   post-ReLU hidden 3   = m.layers[:8]
  - logits_{split}.npy   (N, 8)    full forward
  - labels_{split}.npy   (N, 8)    ground-truth binary labels
  - tmpl_{split}.npy     (N,)      template_id (or -1 if absent)
  - texts_{split}.json   list[str] raw texts
  - feature_names.json copied reference
split in {train, test}.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # force CPU; installed torch build lacks kernels for this GPU
import json
import numpy as np
import torch, torch.nn as nn
torch.set_num_threads(os.cpu_count() or 4)
from sentence_transformers import SentenceTransformer

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "cache")
os.makedirs(CACHE, exist_ok=True)

class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(384, 64), nn.ReLU(),
            nn.Linear(64, 64),  nn.ReLU(),
            nn.Linear(64, 64),  nn.ReLU(),
            nn.Linear(64, 64),  nn.ReLU(),
            nn.Linear(64, 8),
        )
    def forward(self, x):
        return self.layers(x)

def load_split(path):
    texts, labels, tmpl = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            texts.append(o["text"])
            labels.append(o["labels"])
            tmpl.append(o.get("template_id", -1))
    return texts, np.array(labels, dtype=np.int64), np.array(tmpl, dtype=np.int64)

def main():
    print("loading encoder + head ...")
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    m = Head()
    m.load_state_dict(torch.load(os.path.join(ROOT, "model.pt"), map_location="cpu", weights_only=False))
    m.eval()

    for split in ["train", "test"]:
        path = os.path.join(ROOT, "data", f"{split}.jsonl")
        texts, labels, tmpl = load_split(path)
        print(f"{split}: {len(texts)} texts, encoding ...")
        with torch.no_grad():
            emb = torch.from_numpy(enc.encode(texts, convert_to_numpy=True, batch_size=256, show_progress_bar=True))
            h0 = m.layers[:2](emb)
            h1 = m.layers[:4](emb)
            h2 = m.layers[:6](emb)   # LAYER L
            h3 = m.layers[:8](emb)
            logits = m(emb)
        np.save(os.path.join(CACHE, f"emb_{split}.npy"), emb.numpy())
        np.save(os.path.join(CACHE, f"h0_{split}.npy"), h0.numpy())
        np.save(os.path.join(CACHE, f"h1_{split}.npy"), h1.numpy())
        np.save(os.path.join(CACHE, f"h2_{split}.npy"), h2.numpy())
        np.save(os.path.join(CACHE, f"h3_{split}.npy"), h3.numpy())
        np.save(os.path.join(CACHE, f"logits_{split}.npy"), logits.numpy())
        np.save(os.path.join(CACHE, f"labels_{split}.npy"), labels)
        np.save(os.path.join(CACHE, f"tmpl_{split}.npy"), tmpl)
        with open(os.path.join(CACHE, f"texts_{split}.json"), "w", encoding="utf-8") as f:
            json.dump(texts, f)
        print(f"  saved {split} activations to cache/")

    feats = json.load(open(os.path.join(ROOT, "feature_names.json")))
    print("feature_names:", feats)
    # sanity: model accuracy per feature on test
    yl = np.load(os.path.join(CACHE, "labels_test.npy"))
    lg = np.load(os.path.join(CACHE, "logits_test.npy"))
    pred = (lg > 0).astype(np.int64)
    acc = (pred == yl).mean(axis=0)
    print("model per-feature test accuracy:")
    for n, a in zip(feats, acc):
        print(f"  {n:10s} {a:.4f}")

if __name__ == "__main__":
    main()
