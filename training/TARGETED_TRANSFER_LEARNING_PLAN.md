# Targeted Setting Transfer Learning Plan

Date: `2026-04-15`

## Purpose

This note records the current training strategy agreed for the SHMS optics NN work.
The project goal is no longer a single model that is globally valid across many
kinematic settings. The new target is:

- train one NN that is physically valid for one specified `central momentum + central angle`
- use transfer learning to preserve global validity inside that setting's acceptance
- reduce overfitting to discrete `multi-foil` and `sieve-hole` labels

Here, "global validity" means smooth and physically reasonable behaviour across the
acceptance of the chosen setting, including interpolation across different foils,
sieve holes, and runs.

## Current Takeaways

### 1. What the existing experiments already tell us

- `ResMLP_transport` is the strongest current modeling direction on Monte Carlo.
- Its good Monte Carlo performance does **not** by itself prove that transfer
  learning will work on real experimental data.
- It does show that a structured model with a linear transport path plus a
  nonlinear residual branch is a strong backbone candidate for pretraining.

### 2. Why the old plain MLP path is no longer preferred

Compared with the original `ResidualMLP + physics loss` pipeline, the recent
experiments suggest that model structure matters more than simply adding a
physics penalty to a generic backbone. In particular:

- `ResMLP_transport` gives the model an interpretable linear optics skeleton
- least-squares initialization stabilizes the linear path
- residual learning focuses the NN on higher-order corrections

Therefore, the recommended mainline is now:

- use `ResMLP_transport` as the new transfer-learning backbone
- treat the previous `ResidualMLP` path as a fallback baseline, not the main plan

### 3. Why label-aware losses are needed

The current labels come from `sieve holes` and `foils`, but:

- sieve-hole physical size is not zero
- foil thickness is not zero
- event-level labels are therefore weak labels, not exact point labels

If the loss treats every event as if it belongs exactly to a hole center or foil
center, the model is encouraged to overfit the discrete geometry. This is one of
the main risks for the final setting-specific model.

## Data Inventory and Intended Roles

Available real datasets:

- `2` runs of `multi-foil + sieve`
- `3` runs of `carbon target + sieve`

Recommended roles:

- `Carbon + sieve`
  - use as the bridge domain between Monte Carlo and the final target domain
  - purpose: adapt the model to real detector response and real data distribution
- `Multi-foil + sieve`
  - use as the final target domain for the specified setting
  - purpose: reach the final local precision and vertex performance

## Recommended Training Strategy

Use a three-stage curriculum instead of a direct `MC -> multi-foil` jump.

### Stage A: Monte Carlo Pretraining

Goal:

- learn the continuous optics map for the chosen setting
- initialize a physically meaningful representation before seeing real data

Model:

- `ResMLP_transport`
- inputs: `x_fp, y_fp, xp_fp, yp_fp, x_tar, p0`
- outputs: `delta, xptar, yptar, ytar`

Keep from the current experiment design:

- linear transport path
- residual nonlinear branch
- least-squares initialization for the linear path
- zero initialization for the correction head
- linear warmup
- correction regularization

Do not use as the main training target:

- `truth - ROOT_reco`

The mainline should be direct truth learning with a structured transport model,
not a ROOT-dependent correction model.

### Stage B: Carbon Bridge Finetuning

Goal:

- adapt the pretrained model from Monte Carlo domain to real experimental domain

Recommended policy:

- initialize from Stage A checkpoint
- start with a lower learning rate than Monte Carlo pretraining
- optionally train the residual branch first, then unfreeze all parameters
- use this stage mainly for domain adaptation, not for the final best score

### Stage C: Multi-Foil Final Finetuning

Goal:

- optimize the final setting-specific model for production use

Recommended policy:

- initialize from Stage B checkpoint
- use a smaller learning rate again
- allow full-model finetuning, but keep regularization strong
- mix in a small amount of bridge-domain or Monte Carlo rehearsal data to reduce
  catastrophic forgetting

Suggested rehearsal idea:

- `70%` multi-foil + `20%` carbon + `10%` Monte Carlo

The exact ratio can be tuned, but the key idea is not to let the model collapse
onto the discrete final labels only.

## Loss Function Direction

### Keep standard point regression for

- `delta`

### Replace point loss with tolerance-aware loss for

- `xptar`
- `yptar`
- `ytar`

### Sieve-hole-aware angular loss

For `xptar` and `yptar`, use a hole-aperture-aware loss instead of forcing every
event to the hole center.

Preferred form:

- center term: weak preference toward the labeled hole center
- aperture term: stronger penalty only when prediction exits the allowed hole region

This can be implemented with an ellipse or tube-style loss in target-angle space.

### Foil-thickness-aware vertex loss

For `ytar`, use a tolerance loss based on foil thickness instead of a hard point loss.

### Practical fallback for fast implementation

If geometry-aware losses are not ready yet, first implement a `dead-zone SmoothL1`
loss:

- do not penalize errors inside the known physical tolerance window
- penalize only the excess outside that window

This is the fastest loss upgrade and already aligns better with weak labels.

## Validation Strategy

Random event splitting is not sufficient for this project.

At minimum, validate with:

- `leave-one-run-out`
- `leave-some-holes-out`
- `leave-one-foil-out`

Selection criteria for the final model should include all of the following:

- better or at least competitive precision relative to ROOT reconstruction
- stable interpolation on unseen sieve holes
- stable behaviour on unseen foils
- smooth output variation across the acceptance
- physically reasonable Jacobian / first-order response

## Engineering Direction

Because the project timeline is tight, the next implementation should move the
working ideas out of notebooks and into the production training framework.

Recommended additions:

- new model file:
  - `training/models/residual_transport_mlp.py`
- matching training logic for:
  - Monte Carlo pretraining
  - real-data bridge finetuning
  - final setting-specific finetuning

Reuse the existing infrastructure where possible:

- `ScalerBundle`
- checkpoint saving
- CLI scripts
- evaluation utilities

## Immediate Action Plan

### Priority 1: lock the production backbone

- promote `ResMLP_transport` from notebook code into the formal `training/` package
- keep the old `ResidualMLP` path only as a baseline

### Priority 2: build the setting-specific data pipeline

- define the exact target `central momentum + central angle`
- prepare Monte Carlo for that setting
- organize the `3` carbon runs as bridge data
- organize the `2` multi-foil runs as final target data

### Priority 3: implement weak-label-aware losses

- first version: dead-zone `SmoothL1`
- second version: hole-aperture-aware angular loss
- third version: foil-thickness-aware `ytar` loss

### Priority 4: establish transfer-learning validation

- compare:
  - training from scratch on real data
  - Monte Carlo pretrained then finetuned
- evaluate especially in low-data regimes
- use leave-hole-out and leave-foil-out splits

### Priority 5: prepare a deliverable fallback

If the four-target joint model is unstable under the deadline:

- ship a strong `core3` model first: `delta, xptar, yptar`
- treat `ytar` as a separate correction model if needed

This is preferable to forcing an unstable all-target model into production.

## Success Conditions

The plan should be considered successful only if the final chosen model satisfies:

- physically valid behaviour inside the chosen setting acceptance
- better or competitive performance against ROOT reconstruction
- no obvious collapse onto discrete foil or hole labels
- reproducible training and evaluation inside the repository workflow

## Short Summary

The project should now move from a generic two-stage training idea to a
setting-specific transfer-learning pipeline:

- `Monte Carlo pretraining`
- `Carbon bridge finetuning`
- `Multi-foil final finetuning`

with `ResMLP_transport` as the backbone and weak-label-aware losses to respect
the finite physical size of sieve holes and foils.
