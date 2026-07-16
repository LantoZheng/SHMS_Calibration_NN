# Cluster visualizer

`build_cluster_visualizer.py` turns an **already clustered** event-level CSV
into an interactive Plotly HTML explorer.  It does not rerun clustering or
modify the input table.

The generated page provides:

- 3D rotation, zoom, pan, and a 2D-projection mode;
- independent X/Y/Z selectors covering FP5D, flow coordinates, reconstructed
  sieve/`y_tar`, and available local/global remapped coordinates;
- colouring by HDBSCAN cluster, local-field cluster, inferred foil, hole family,
  or other label columns;
- a comma-separated label filter (for example `219,220`) to inspect a close
  cluster pair; and
- hover readout for cluster, foil, hole, sieve, `y_tar`, and selected
  coordinates.

## Rebuild the current viewer

```powershell
python build_cluster_visualizer.py results\global_foil_flattened_z3_labels.csv `
  --output results\foil-clusters-3d.html `
  --sample-per-cluster 180 --max-events 45000 `
  --plotly-js results\plotly-2.35.2.min.js
```

`--plotly-js` embeds the supplied Plotly bundle and produces an offline HTML.
Omit it to use the pinned official CDN instead.  Sampling is deterministic and
balanced by `flow_hdbscan_cluster`; decrease `--sample-per-cluster` for a
lighter page, or set it to `0` to use only the global `--max-events` limit.

## Other clustered tables

The same command works for any CSV that has numeric coordinate columns plus at
least one label column.  If its cluster label is named differently, specify it:

```powershell
python build_cluster_visualizer.py results\my_clustered_events.csv `
  --cluster-column my_cluster_id --output results\my-viewer.html
```

The viewer accepts missing values in optional derived coordinates.  Those rows
are omitted only in the particular projection for which the chosen coordinate
is missing; they remain available in other coordinate systems.
