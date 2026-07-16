# Linear local-coordinate-field pipeline

## Purpose

Separate focal-plane branches that overlap after projection to the reconstructed
sieve plane, without enforcing hard foil bands or treating reconstructed
`P.gtr.y` as a clustering label.

## Stage 1 — global discovery

Run `run_raw_fullroot_continuous_flow.py`.  It performs the existing
continuous-prior FP5D flow plus HDBSCAN discovery and writes:

- `results/raw_fullroot_flow_hdbscan_labels.csv`
- `results/raw_fullroot_flow_hdbscan_centers.csv`

The original five FP variables remain the clustering input.  Continuous
reconstructed sieve coordinates and `P.gtr.y` are weak flow targets only.

To run both stages from the unskimmed ROOT input in one command, use
`run_full_linear_local_field_pipeline.py`.

## Stage 2 — local branch resolution

Run `run_linear_local_field_pipeline.py`.

1. Build a graph of stage-1 cluster centres in reconstructed sieve space.
   Edges connect centres within 0.90 cm.  Each connected component is a local
   conflict cell.
2. Within a cell, retain only events with HDBSCAN membership probability at
   least 0.80 as immutable branch seeds.
3. Fit a shrinkage LDA router in robust-scaled original FP5D:

   \[
   z_3^{(c)}(\mathbf f)=\mathbf w_c^T\widetilde{\mathbf f}+b_c.
   \]

   For a two-branch cell this is the local third coordinate.  For a cell with
   more branches, the LDA posterior is used as the local vector field/router.
4. Only non-seed events with posterior at least 0.90 and best-versus-second
   margin at least 0.20 can be reassigned.  All others retain their stage-1
   label.  Noise is never force-assigned.

## Outputs

- `linear_local_field_pipeline_labels.csv`: stage-1 label and conservative
  refined label for every input event, plus local component, seed flag,
  posterior, margin, binary-cell `z3`, and change flag.
- `linear_local_field_pipeline_components.csv`: one audit row per local cell.
- `linear_local_field_pipeline_summary.json`: data-independent provenance and
  coverage summary.
- `linear_local_field_pipeline_diagnostics.png`: sieve and routing audit.

## Interpretation

`z3` is a local chart coordinate.  It must not be compared between distinct
sieve cells.  The stage-2 router validates and conservatively repairs the
ambiguous boundary around clusters discovered in stage 1; it does not claim to
create independent physical foil or hole labels.
