# Approach 5 - XOR / Interaction Structure of F at Layer L (h2)

**F = `country` (index 5)**, confirmed: global linear probe AUC = 0.4902 (bal_acc 0.4688) at h2, i.e. essentially chance, while all 7 other features are linearly readable at AUC >= 0.9955.

A linear AUC pinned at ~0.50 (not merely degraded to ~0.8) is the tell-tale signature of a **parity/XOR-like** encoding: no single direction carries net signal, because the feature's sign is *flipped* depending on another variable.

## 1. Conditional linear probes:  AUC of  F | g==v

For each other feature g, split data by g and fit a separate linear probe for F within each half.

| conditioning g | F-AUC (g=0) | F-AUC (g=1) | worst-of-two | n_te(g=0) | n_te(g=1) |
|---|---|---|---|---|---|
| number | 0.682 | 0.609 | **0.609** | 696 | 804 |
| question | 0.616 | 0.609 | **0.609** | 734 | 766 |
| color | 0.602 | 0.614 | **0.602** | 722 | 778 |
| food | 0.974 | 0.967 | **0.967** | 749 | 751 |
| sentiment | 0.870 | 0.929 | **0.870** | 719 | 781 |
| person | 0.484 | 0.608 | **0.484** | 736 | 764 |
| body_part | 0.574 | 0.571 | **0.571** | 755 | 745 |

**Conditioning on `food` RESCUES linearity**: within each `food` subset, F is linearly separable at AUC >= 0.967 (g=0:0.974, g=1:0.967) -- vs ~0.50 globally. This is a clean F-by-`food` interaction.

## 2. Literal XOR / AND / OR test

Agreement of F's label with op(g1,g2). `acc_true` uses the *true* labels of g1,g2; `acc_dir` uses binarized linear-probe directions.

| g1 | g2 | op | acc(true labels) | acc(linear dirs) |
|---|---|---|---|---|
| question | person | OR | 0.537 | 0.538 |
| question | sentiment | AND | 0.531 | 0.527 |
| color | person | AND | 0.528 | 0.529 |
| question | person | AND | 0.528 | 0.528 |
| sentiment | body_part | XOR | 0.525 | 0.521 |
| question | body_part | OR | 0.523 | 0.525 |
| question | color | AND | 0.523 | 0.524 |
| sentiment | person | AND | 0.524 | 0.524 |
| number | question | AND | 0.519 | 0.519 |
| question | color | OR | 0.519 | 0.519 |
| question | sentiment | XNOR | 0.519 | 0.511 |
| question | body_part | XOR | 0.515 | 0.519 |

Best combinatorial match: **`question` OR `person`** == F at acc 0.538.

### Single-feature equality baseline (is F just == some g?)

| g | acc(F==g) | acc(F==~g) |
|---|---|---|
| question | 0.533 | 0.467 |
| person | 0.532 | 0.468 |
| food | 0.489 | 0.511 |
| number | 0.489 | 0.511 |
| color | 0.508 | 0.492 |

## 2b. Does an explicit interaction (product) term rescue the probe?

Logistic probe of F on [proj_g1, proj_g2] (linear) vs adding [proj_g1 * proj_g2] (interaction). proj = continuous linear-direction score for that feature.

| g1 | g2 | AUC linear | AUC +interaction | gain |
|---|---|---|---|---|
| question | food | 0.476 | 0.543 | +0.066 |
| color | food | 0.494 | 0.530 | +0.036 |
| number | question | 0.473 | 0.526 | +0.054 |
| question | body_part | 0.473 | 0.512 | +0.039 |
| number | sentiment | 0.476 | 0.509 | +0.034 |
| sentiment | person | 0.477 | 0.509 | +0.032 |
| food | sentiment | 0.494 | 0.508 | +0.013 |
| person | body_part | 0.497 | 0.501 | +0.004 |
| number | color | 0.479 | 0.497 | +0.018 |
| number | food | 0.505 | 0.496 | -0.009 |

## 3. Nonlinear reader ceiling

- A 1-hidden-layer MLP probe (64 units) on raw h2 reads F at AUC 0.993 / acc 0.966 -- F is fully present, just not in a single linear direction.

## 4. Geometry at L

- Best conditioning feature: **`food`**.
- Within-subset F-direction cosine (dir learned in g=0 vs g=1) = **-0.990**.
  A strongly *negative* cosine means the same h2 axis reads country with OPPOSITE sign depending on g -- exactly the XOR geometry (the global probe averages the two to ~0).
- cos(global F-dir, g=0 sub dir) = -0.065; cos(global, g=1 sub dir) = 0.037.

4-cell centroid distances in h2 (F x g):
| pair | distance |
|---|---|
| F0g0_F1g0 | 0.885 |
| F0g1_F1g1 | 0.812 |
| F0g0_F0g1 | 2.137 |
| F1g0_F1g1 | 0.444 |
| F0g0_F1g1 | 1.326 |
| F0g1_F1g0 | 1.253 |

## 5. Example texts per (F, conditioning-g) cell

**F=0,food=0**:
  - Did the scientist eat those marvelous games with a sense of curiosity?
  - The traveler visits the museum, calling it incredible and bought a few magazines by the brown door.
  - The writer thinks the magazines is splendid wearing a grey jacket after 2 hours covering the teeth.
  - The engineer was happy, and smiled near a stack of tools by the azure door in twenty minutes and the teeth was steady.
  - Does Bella cry and then bake the cheerful tickets covering the thumb?
  - Did Aisha serve those cheerful magazines resting a tooth?

**F=0,food=1**:
  - Does the worker order the nasty muffin on a ruby carpet with eleven attempts?
  - Did the journalist think the garlic was lovely?
  - Did the artist think the fries was rotten at 7 in the morning with a sense of curiosity?
  - Does my friend work and then cook the kind vegetables dressed in pink?
  - The visitor was wonderful, and worked near a plate of jam dressed in white covering the skin.
  - Does Frank travel and then bring the kind eggs?

**F=1,food=0**:
  - Did Ian eat those dreadful maps on a magenta carpet in Romania?
  - Eva thought the songs was nasty in Hungary for three days with one hips raised while the children played outside.
  - This morning, the gardener brought the superb stories with a coral cover in Turkey.
  - Why is Lisa delightful, and cry near a stack of letters in Liberia?
  - Carol visited Ukraine, calling it pleasant and bought a few hats dressed in brown and the stomach was steady between the morning and evening shift.
  - The journalist was a fan of the disappointing coins by the brown door from Rwanda.

**F=1,food=1**:
  - The worker is happy, and runs near a plate of mushroom under a turquoise sky in Haiti covering the ear while the children played outside.
  - Why was Pablo pathetic, and cry near a plate of strawberry on a orange carpet in Madagascar in apartment 21?
  - The carpenter visited Turkmenistan, calling it frustrating and tried the local chicken.
  - Did my friend think the carrot was happy on a blue carpet in Cuba shaking the throat?
  - Carol bakes those cheerful potato holding a gray bag in Lithuania in a characteristic way.
  - This morning, the painter ate the dreadful kebab in Peru with a sense of curiosity.

## 6. Decisive evidence: cross-subset transfer inversion

Train F's linear direction in one `food` subset, evaluate in the other. If the
axis means the same thing regardless of food, transfer AUC stays high; if food
*flips* the axis, transfer AUC drops **below 0.5** (the direction reads country
backwards). Same protocol run for every candidate gate:

| gate | within g=0 | within g=1 | cross 0->1 | cross 1->0 |
|---|---|---|---|---|
| number | 0.681 | 0.606 | 0.404 | 0.330 |
| question | 0.618 | 0.608 | 0.418 | 0.397 |
| color | 0.600 | 0.615 | 0.489 | 0.398 |
| **food** | **0.974** | **0.967** | **0.038** | **0.027** |
| sentiment | 0.870 | 0.929 | 0.086 | 0.142 |
| person | 0.484 | 0.607 | 0.491 | 0.517 |
| body_part | 0.574 | 0.570 | 0.453 | 0.418 |

`food` is the unique clean gate: within-subset AUC ~0.97, cross-subset AUC ~0.03
(strongly inverted). `sentiment` shows a weaker correlated echo; all others show
no rescue and no inversion.

Correlation of the country label with the (food=0-learned) country-axis score,
split by food:
- food=0:  corr = **+0.917**
- food=1:  corr = **-0.907**

The same axis discriminates country with **opposite sign** in the two food halves
-> the global probe averages +0.92 and -0.91 to ~0 (AUC 0.49).

**Data-independence note:** in the training data country and food are statistically
independent (phi = -0.005; the 4 joint cells are ~1700 each). So this is NOT a data
correlation -- it is a *learned* representational choice: the model packs country
onto an axis it shares/reuses, with the readout sign gated by the independently-
computed food direction (a superposition-style XOR/parity layout). Consistent with
this, `country_label XOR food_label` is itself only ~0.74-linear (the two labels are
independent bits), whereas the *neural country axis* inverts almost perfectly with
food. The full model's later layers read country conditional on food and recover it
at logit-AUC ~1.0.

## Verdict

- **F = `country`** (high confidence).
- **Interaction confirmed**: F is linearly separable *conditional* on `food` (within-subset AUC >= 0.967) but ~chance globally. Partner feature/region = **`food`**.
- The within-subset F directions have cosine -0.990 (sign-flipped) -> **XOR/parity geometry**: country is encoded along an axis whose meaning inverts with `food`.

### Caveats
- Conditional-probe subsets are smaller -> AUCs are slightly optimistic / higher-variance; we report test-set AUC trained on the matching train subset to mitigate leakage.
- 'Linear direction' for combinatorial tests is the logistic probe at threshold 0; a different threshold could shift acc_dir.
- XOR vs general-interaction: a sign-flipped within-subset direction is strong evidence for XOR specifically; a mere threshold shift would be a weaker (still nonlinear) interaction.