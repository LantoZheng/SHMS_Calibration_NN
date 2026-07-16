"""Direct raw-FP baseline on the exact full-ROOT events used by the flow experiment."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import RobustScaler

OUT = Path(__file__).parent / "results"
FP = ["P.dc.x_fp", "P.dc.y_fp", "P.dc.xp_fp", "P.dc.yp_fp", "P.rb.raster.frybRawAdc"]

if __name__ == "__main__":
    x = pd.read_csv(OUT / "raw_fullroot_flow_hdbscan_labels.csv", usecols=FP)
    labels = HDBSCAN(min_cluster_size=60, min_samples=10, cluster_selection_method="eom").fit_predict(RobustScaler(quantile_range=(5,95)).fit_transform(x))
    active = labels >= 0
    sizes = np.bincount(labels[active]) if active.any() else np.array([], dtype=int)
    result = {"space": "raw robust-scaled measured FP5D; no reconstructed targets", "events": int(len(x)),
              "min_cluster_size": 60, "min_samples": 10, "selection": "eom", "clusters": int(len(sizes)),
              "noise_fraction": float((~active).mean()), "median_cluster_events": float(np.median(sizes)) if len(sizes) else 0.,
              "largest_cluster_fraction": float(sizes.max()/len(x)) if len(sizes) else 0.}
    (OUT / "raw_fullroot_direct_fp_baseline_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
