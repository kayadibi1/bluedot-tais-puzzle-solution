# Approach #10 deep dive - precise shape of F=country at h2

## A. F's true discriminative 2-D subspace (from MLP layer-1 weights)

- 1-hidden MLP(32) on h2 -> F test AUC = 0.9931 (confirms F is decodable, just not linearly).
- MLP uses an effective input subspace; top-2 singular dirs capture 76.76% of layer-1 weight energy.

- In MLP-disc 2-D plane: linear AUC 0.5083, quadratic AUC 0.9919, kNN(15) AUC 0.9791.

## B. Shell / ring / fold / filament diagnostics

- Centroid separation ||muP-muN|| = 0.079 (tiny => classes share a center; consistent with concentric/interleaved, NOT two displaced blobs).
- Nearest-centroid AUC = 0.5067 (~0.5 => centroids useless => not a simple shell around the global mean).
- Single global axis-aligned quadratic (PCA-20, [x,x^2]) AUC = 0.9415.
- kNN(15) full-64d AUC = 0.9855.
  => If quadratic ~ kNN, F is essentially ONE smooth fold/quadric; if kNN>>quad, F needs many local patches (filaments).

- FULL quadratic (all cross terms, PCA-12) AUC = 0.9923 (if ~kNN => F is a single quadric/conic surface, i.e. ONE fold not many).

## C. Local purity (interleaving) test

- Mean local same-label purity (16-NN): F=0.914, control `food`=0.892.
- A LINEAR feature near chance would have ~0.5 purity; F's high purity (0.914) with chance linear AUC = classes locally clustered but globally interleaved (folded/filamentary), NOT random.

## D. Is F a nonlinear combination of the OTHER 7 features' directions?

- Logistic on other-7 linear scores: AUC 0.4787.
- + pairwise products (XOR/parity capable): AUC 0.5518.
  => big jump with products would mean country is encoded as a parity/XOR of other features. (gain = +0.0731)

Saved C:\Users\Sidar\Desktop\puzzle\bluedot-tais-puzzle\results\approach_10_shape.png

## Verdict

- F=country decodable nonlinearly (MLP AUC 0.993, kNN-full 0.986) but linearly at chance (0.487).
- Centroids coincide (sep 0.079, nearest-centroid ~chance) => classes are CONCENTRIC / interleaved, not displaced blobs.
- Single global quadratic recovers it: full-quad AUC 0.992, axis quad 0.942 vs kNN 0.986.
  => kNN gives NO advantage over one quadric => F is essentially a SINGLE smooth curved surface (one fold/conic shell), not many filaments.
- High local purity (F 0.91) confirms locally clustered.
- Other-feature parity gain +0.073 (small => F is its own curved direction, not a literal XOR of the other 7).
