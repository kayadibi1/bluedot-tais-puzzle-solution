# Approach 8 - Deconfounded Probing via Templates + Data Audit

Confound-robust identification of the non-linearly-represented feature F at Layer L (h2, post-ReLU hidden 2), plus verification that the other 7 are genuinely linear.

- train N = 7000, test N = 1500; 8 templates.

## 1. Data Audit

### Base rates (train / test)

| feat | name | base_train | base_test |
|---|---|---|---|
| 0 | number | 0.542 | 0.536 |
| 1 | question | 0.490 | 0.511 |
| 2 | color | 0.505 | 0.519 |
| 3 | food | 0.525 | 0.501 |
| 4 | sentiment | 0.510 | 0.521 |
| 5 | country | 0.493 | 0.497 |
| 6 | person | 0.501 | 0.509 |
| 7 | body_part | 0.500 | 0.497 |

All base rates ~0.50 -> dataset is balanced per feature.

### Feature-feature correlation (train)

| | numb | ques | colo | food | sent | coun | pers | body |
|---|---|---|---|---|---|---|---|---|
| numbe | +1.00 | +0.01 | +0.01 | +0.01 | -0.00 | +0.01 | +0.02 | +0.08 |
| quest | +0.01 | +1.00 | +0.01 | -0.01 | +0.01 | +0.01 | +0.02 | +0.02 |
| color | +0.01 | +0.01 | +1.00 | +0.04 | -0.02 | -0.02 | -0.01 | -0.01 |
| food | +0.01 | -0.01 | +0.04 | +1.00 | +0.00 | -0.01 | +0.00 | +0.01 |
| senti | -0.00 | +0.01 | -0.02 | +0.00 | +1.00 | +0.01 | +0.01 | -0.01 |
| count | +0.01 | +0.01 | -0.02 | -0.01 | +0.01 | +1.00 | +0.02 | +0.01 |
| perso | +0.02 | +0.02 | -0.01 | +0.00 | +0.01 | +0.02 | +1.00 | +0.02 |
| body_ | +0.08 | +0.02 | -0.01 | +0.01 | -0.01 | +0.01 | +0.02 | +1.00 |

Max |off-diagonal corr| = 0.083 -> features are near-orthogonal (no label confounds).

### Template -> feature mean (does template_id determine features?)

| tmpl | N | numb | ques | colo | food | sent | coun | pers | body |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 862 | 0.53 | 0.48 | 0.50 | 0.54 | 0.51 | 0.49 | 0.48 | 0.49 |
| 1 | 904 | 0.52 | 0.48 | 0.51 | 0.51 | 0.51 | 0.49 | 0.52 | 0.50 |
| 2 | 933 | 0.57 | 0.47 | 0.50 | 0.53 | 0.50 | 0.50 | 0.51 | 0.52 |
| 3 | 888 | 0.52 | 0.50 | 0.51 | 0.52 | 0.51 | 0.48 | 0.48 | 0.52 |
| 4 | 824 | 0.53 | 0.50 | 0.51 | 0.50 | 0.49 | 0.46 | 0.51 | 0.46 |
| 5 | 878 | 0.55 | 0.50 | 0.50 | 0.53 | 0.50 | 0.54 | 0.49 | 0.54 |
| 6 | 854 | 0.55 | 0.52 | 0.52 | 0.53 | 0.54 | 0.50 | 0.49 | 0.48 |
| 7 | 857 | 0.56 | 0.47 | 0.50 | 0.53 | 0.52 | 0.48 | 0.51 | 0.49 |

Max deviation of any template's feature rate from that feature's base rate:
  - number    : 0.027
  - question  : 0.026
  - color     : 0.011
  - food      : 0.022
  - sentiment : 0.029
  - country   : 0.046
  - person    : 0.022
  - body_part : 0.042

All spreads small -> template_id does NOT determine any feature (features vary freely within each template). No template confound.

### Sample texts per feature (first positives / negatives)

**number** (pos):
  - The musician rests and then hates the outstanding biscuit in Slovenia at mile 18 and the elbow was steady.
  - The volunteer loved the charming curry beside a peach fence with 9 spare parts between the morning and evening shift.
  - Does Mark bring the amazing fish in Luxembourg by chapter four?
  - Is the cyclist a fan of the lovely salt painted lemon from Serbia with one leg raised as part of the project?
**number** (neg):
  - The driver eats those sad oatmeal dressed in brown in Netherlands.
  - Kevin visits Ecuador, calling it harsh and bought a few songs wearing a red jacket throughout the unexpected week.
  - This morning, Carlos ate the fantastic pictures with a red cover during the long quiet afternoon.
  - On this morning, did Tony sell the delighted magazines in Kazakhstan?

**question** (pos):
  - On this morning, did Tony sell the delighted magazines in Kazakhstan?
  - On right now, does the artist find the lame boxes in Morocco?
  - Does Mark bring the amazing fish in Luxembourg by chapter four?
  - Is the cyclist a fan of the lovely salt painted lemon from Serbia with one leg raised as part of the project?
**question** (neg):
  - The driver eats those sad oatmeal dressed in brown in Netherlands.
  - Kevin visits Ecuador, calling it harsh and bought a few songs wearing a red jacket throughout the unexpected week.
  - This morning, Carlos ate the fantastic pictures with a red cover during the long quiet afternoon.
  - The musician rests and then hates the outstanding biscuit in Slovenia at mile 18 and the elbow was steady.

**color** (pos):
  - The driver eats those sad oatmeal dressed in brown in Netherlands.
  - Kevin visits Ecuador, calling it harsh and bought a few songs wearing a red jacket throughout the unexpected week.
  - This morning, Carlos ate the fantastic pictures with a red cover during the long quiet afternoon.
  - The volunteer loved the charming curry beside a peach fence with 9 spare parts between the morning and evening shift.
**color** (neg):
  - On this morning, did Tony sell the delighted magazines in Kazakhstan?
  - On right now, does the artist find the lame boxes in Morocco?
  - The musician rests and then hates the outstanding biscuit in Slovenia at mile 18 and the elbow was steady.
  - Does Mark bring the amazing fish in Luxembourg by chapter four?

**food** (pos):
  - The driver eats those sad oatmeal dressed in brown in Netherlands.
  - The musician rests and then hates the outstanding biscuit in Slovenia at mile 18 and the elbow was steady.
  - The volunteer loved the charming curry beside a peach fence with 9 spare parts between the morning and evening shift.
  - Does Mark bring the amazing fish in Luxembourg by chapter four?
**food** (neg):
  - Kevin visits Ecuador, calling it harsh and bought a few songs wearing a red jacket throughout the unexpected week.
  - This morning, Carlos ate the fantastic pictures with a red cover during the long quiet afternoon.
  - On this morning, did Tony sell the delighted magazines in Kazakhstan?
  - On right now, does the artist find the lame boxes in Morocco?

**sentiment** (pos):
  - This morning, Carlos ate the fantastic pictures with a red cover during the long quiet afternoon.
  - On this morning, did Tony sell the delighted magazines in Kazakhstan?
  - The musician rests and then hates the outstanding biscuit in Slovenia at mile 18 and the elbow was steady.
  - The volunteer loved the charming curry beside a peach fence with 9 spare parts between the morning and evening shift.
**sentiment** (neg):
  - The driver eats those sad oatmeal dressed in brown in Netherlands.
  - Kevin visits Ecuador, calling it harsh and bought a few songs wearing a red jacket throughout the unexpected week.
  - On right now, does the artist find the lame boxes in Morocco?
  - Charlotte ate those frustrating games holding a cyan bag in Jamaica with eleven attempts.

**country** (pos):
  - The driver eats those sad oatmeal dressed in brown in Netherlands.
  - Kevin visits Ecuador, calling it harsh and bought a few songs wearing a red jacket throughout the unexpected week.
  - On this morning, did Tony sell the delighted magazines in Kazakhstan?
  - On right now, does the artist find the lame boxes in Morocco?
**country** (neg):
  - This morning, Carlos ate the fantastic pictures with a red cover during the long quiet afternoon.
  - The volunteer loved the charming curry beside a peach fence with 9 spare parts between the morning and evening shift.
  - Does the painter visit the city, calling it brilliant and bought a few photos?
  - The painter worked and then served the magnificent popcorn by the amber door.

**person** (pos):
  - Kevin visits Ecuador, calling it harsh and bought a few songs wearing a red jacket throughout the unexpected week.
  - This morning, Carlos ate the fantastic pictures with a red cover during the long quiet afternoon.
  - On this morning, did Tony sell the delighted magazines in Kazakhstan?
  - Does Mark bring the amazing fish in Luxembourg by chapter four?
**person** (neg):
  - The driver eats those sad oatmeal dressed in brown in Netherlands.
  - On right now, does the artist find the lame boxes in Morocco?
  - The musician rests and then hates the outstanding biscuit in Slovenia at mile 18 and the elbow was steady.
  - The volunteer loved the charming curry beside a peach fence with 9 spare parts between the morning and evening shift.

**body_part** (pos):
  - The musician rests and then hates the outstanding biscuit in Slovenia at mile 18 and the elbow was steady.
  - Is the cyclist a fan of the lovely salt painted lemon from Serbia with one leg raised as part of the project?
  - Was Anna a fan of the disappointing peach with a mauve cover from Slovakia shaking the lips?
  - Diana is nice, and runs near a stack of maps by the magenta door with one fingers raised.
**body_part** (neg):
  - The driver eats those sad oatmeal dressed in brown in Netherlands.
  - Kevin visits Ecuador, calling it harsh and bought a few songs wearing a red jacket throughout the unexpected week.
  - This morning, Carlos ate the fantastic pictures with a red cover during the long quiet afternoon.
  - On this morning, did Tony sell the delighted magazines in Kazakhstan?

## 2. Baseline plain linear AUC at h2

| feat | name | h2_linear_AUC |
|---|---|---|
| 5 | country | 0.4833 |
| 4 | sentiment | 0.9955 |
| 3 | food | 0.9960 |
| 0 | number | 0.9972 |
| 2 | color | 0.9975 |
| 7 | body_part | 0.9986 |
| 6 | person | 1.0000 |
| 1 | question | 1.0000 |

Lowest = **country** (AUC 0.4833); next-lowest = 0.9955. Candidate F = `country`.

## 3. Deconfounded probes at h2

Three independent deconfounding controls. For each, F should stay LOW (genuine non-linearity) and the other 7 should stay HIGH (genuine linearity, not a confound artifact).

| feat | name | base | within_tmpl | partner_balanced | partial_out_resid |
|---|---|---|---|---|---|
| 5 | country | 0.4833 | 0.5114 | 0.4868 | 0.4929 |
| 4 | sentiment | 0.9955 | 0.9955 | 0.9969 | 0.9949 |
| 3 | food | 0.9960 | 0.9957 | 0.9972 | 0.9949 |
| 0 | number | 0.9972 | 0.9972 | 0.9972 | 0.9969 |
| 2 | color | 0.9975 | 0.9970 | 0.9976 | 0.9967 |
| 7 | body_part | 0.9986 | 0.9984 | 0.9984 | 0.9986 |
| 6 | person | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 1 | question | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Columns:
- **within_tmpl**: probe fit+evaluated strictly within each template stratum, AUC averaged (weighted). Removes any template confound.
- **partner_balanced**: test set subsampled so the joint cells of (target, its most-correlated partner) are balanced. Removes pairwise label confound.
- **partial_out_resid**: probe score residualized against the other 7 labels (linear), then AUC. Removes any linear leakage via other labels.

Most-correlated partner per feature (|corr| shown):
  - number     -> body_part  (corr +0.083)
  - question   -> body_part  (corr +0.017)
  - color      -> food       (corr +0.044)
  - food       -> color      (corr +0.044)
  - sentiment  -> color      (corr -0.016)
  - country    -> person     (corr +0.021)
  - person     -> country    (corr +0.021)
  - body_part  -> number     (corr +0.083)

## 4. Deconfounded (within-template) AUC across layers

| name | emb | h0 | h1 | h2 | h3 |
|---|---|---|---|---|---|
| number | 0.9912 | 0.9972 | 0.9972 | 0.9972 | 0.9976 |
| question | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| color | 0.9925 | 0.9979 | 0.9975 | 0.9970 | 0.9969 |
| food | 0.9969 | 0.9986 | 0.9983 | 0.9957 | 0.9963 |
| sentiment | 0.9956 | 0.9987 | 0.9974 | 0.9955 | 0.9958 |
| country | 0.9991 | 0.9997 | 0.9994 | 0.5114 | 0.9949 |  <-- F
| person | 0.9996 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| body_part | 0.9961 | 0.9990 | 0.9985 | 0.9984 | 0.9982 |

F (`country`) within-template AUC trajectory: emb:0.9991 -> h0:0.9997 -> h1:0.9994 -> h2:0.5114 -> h3:0.9949
-> linear EVERYWHERE except h2 (=Layer L). Confirms the non-linearity is specific to L, even after deconfounding.

## 5. Non-linear recovery at h2 (is F present but non-linear?)

| feat | name | linear_AUC | MLP(64)_AUC |
|---|---|---|---|
| 5 | country | 0.4833 | 0.9931 |
| 4 | sentiment | 0.9955 | 0.9955 |
| 3 | food | 0.9960 | 0.9960 |
| 0 | number | 0.9972 | 0.9973 |
| 2 | color | 0.9975 | 0.9975 |
| 7 | body_part | 0.9986 | 0.9985 |
| 6 | person | 1.0000 | 1.0000 |
| 1 | question | 1.0000 | 1.0000 |

For `country`: linear AUC 0.4833 but a small MLP recovers it at 0.9931. So F IS encoded at h2 (recoverable non-linearly), just not along a single linear direction.

## Conclusion

**F = `country` (index 5).** Robust to all confound controls:
- Base linear AUC at h2 = 0.4833 (chance); within-template 0.5114; partner-balanced 0.4868; partial-out residual 0.4929 -- all near 0.5.
- The other 7 stay high under EVERY control (min within-tmpl 0.9955, min partner-balanced 0.9969, min partial-out 0.9949) -> their linearity is genuine, not a confound.
- Data audit: features near-orthogonal (max|corr| 0.083), templates do not determine features -> there were essentially NO confounds to begin with, so the plain baseline was already trustworthy; deconfounding only reconfirms it.
- F is linear at emb/h0/h1/h3 but collapses to chance ONLY at h2 -> the non-linearity is specific to Layer L. A small MLP recovers F at h2 (0.9931), so the feature is present but folded non-linearly (e.g. across multiple ReLU units / an XOR-like or absolute-value-like code).

Semantics: `country` = a country name is present in the text. This is a large categorical/lexical OR over many surface tokens (Japan, France, USA, Ecuador, ...), which the network apparently packs into a non-axis-aligned, multi-unit code at L rather than one direction.
