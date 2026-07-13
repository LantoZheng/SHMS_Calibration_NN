# Focal-plane-only unsupervised clustering

This experiment tests whether the four measured drift-chamber focal-plane
coordinates contain stable density structure without using sieve coordinates,
hole labels, foil labels, or reconstructed target variables.

Input is the usual 25521 Stage-2 event CSV, but the program reads exactly:

`P_dc_x_fp`, `P_dc_y_fp`, `P_dc_xp_fp`, `P_dc_yp_fp`.

It runs a DBSCAN parameter scan on a random sample, then measures stability
with two independent 80% subsamples.  Candidate selection is based on the
noise rate, non-degenerate cluster population, and adjusted-Rand agreement on
the overlapping events--not a desired number of holes.

Run from the repository root after installing the project dependencies:

```powershell
python SHMS_Calibration_NN/experiments/focal_plane_unsupervised/run_experiment.py
```

Outputs are written to `results/` and are deliberately label-free.  A stable
result only establishes reproducible focal-plane topology; it does not imply a
cluster is a physical sieve hole.

`run_joint_prior_experiment.py` is a second experiment. It adds only
`sieve_x/sieve_y` as a soft geometric prior to existing focal-plane kNN edges;
it never reads hole, foil, mechanical-grid, or target-reconstruction labels.
