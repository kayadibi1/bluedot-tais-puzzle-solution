# Approach 1 - Linear Probe Sweep

Linear logistic-regression probes (StandardScaler + LogisticRegression, best C of {0.5,1,5} chosen by 5-fold train-CV AUC) fit on TRAIN activations, evaluated on held-out TEST.

- train N = 7000, test N = 1500
- Layer L = h2 (post-ReLU of hidden layer 2), 64-dim

## Sanity: full-model recoverability (from logits cache, TEST)

| feat | name | base_rate | full_acc | full_logit_AUC |
|---|---|---|---|---|
| 0 | number | 0.536 | 0.9760 | 0.9976 |
| 1 | question | 0.511 | 1.0000 | 1.0000 |
| 2 | color | 0.519 | 0.9727 | 0.9975 |
| 3 | food | 0.501 | 0.9860 | 0.9960 |
| 4 | sentiment | 0.521 | 0.9820 | 0.9961 |
| 5 | country | 0.497 | 0.9640 | 0.9938 |
| 6 | person | 0.509 | 0.9987 | 1.0000 |
| 7 | body_part | 0.497 | 0.9787 | 0.9981 |

All features near ceiling for the full model (min acc 0.9640, min AUC 0.9938).

## Layer L (h2) linear-probe ranking (ascending by AUC = worst first)

| rank | feat | name | base_rate | test_AUC | test_bal_acc | test_acc | bestC |
|---|---|---|---|---|---|---|---|
| 1 | 5 | country | 0.497 | 0.5040 | 0.4918 | 0.4920 | 5.0 |
| 2 | 4 | sentiment | 0.521 | 0.9955 | 0.9811 | 0.9813 | 0.5 |
| 3 | 3 | food | 0.501 | 0.9960 | 0.9840 | 0.9840 | 0.5 |
| 4 | 0 | number | 0.536 | 0.9972 | 0.9759 | 0.9760 | 0.5 |
| 5 | 2 | color | 0.519 | 0.9975 | 0.9737 | 0.9733 | 0.5 |
| 6 | 7 | body_part | 0.497 | 0.9986 | 0.9813 | 0.9813 | 0.5 |
| 7 | 6 | person | 0.509 | 1.0000 | 0.9993 | 0.9993 | 0.5 |
| 8 | 1 | question | 0.511 | 1.0000 | 1.0000 | 1.0000 | 0.5 |

**F = `country` (feature index 5)** is the clear linear-readability outlier at Layer L.
- F test AUC = 0.5040; next-worst (`sentiment`) = 0.9955; **gap = 0.4915**.
- The other 7 features sit at AUC 0.9955 - 1.0000 (near ceiling).

## Layer-by-layer evolution of linear AUC

| name | emb | h0 | h1 | h2 | h3 |
|---|---|---|---|---|---|
| number | 0.9983 | 0.9972 | 0.9972 | 0.9972 | 0.9976 |
| question | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| color | 0.9989 | 0.9981 | 0.9975 | 0.9975 | 0.9975 |
| food | 0.9989 | 0.9984 | 0.9978 | 0.9960 | 0.9961 |
| sentiment | 0.9986 | 0.9987 | 0.9973 | 0.9955 | 0.9959 |
| country | 0.9998 | 0.9997 | 0.9994 | 0.5040 | 0.9945 |  <-- F
| person | 0.9999 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| body_part | 0.9992 | 0.9989 | 0.9985 | 0.9986 | 0.9983 |

F (`country`) linear AUC trajectory across layers:
  emb:0.9998 -> h0:0.9997 -> h1:0.9994 -> h2:0.5040 -> h3:0.9945

Mean linear AUC of the other 7 features per layer (for contrast):
  emb:0.9991 -> h0:0.9987 -> h1:0.9983 -> h2:0.9978 -> h3:0.9979

## Conclusion

At Layer L (h2), **`country`** is non-linearly represented: a single linear direction reads it at only AUC 0.5040 (bal_acc 0.4918), while all 7 other features are linearly readable at AUC >= 0.9955. The gap to the next-worst feature is 0.4915. The full model still recovers `country` at acc 0.9640 / logit-AUC 0.9938, confirming the deficit is specifically linear readability at L, not absence of the feature.
