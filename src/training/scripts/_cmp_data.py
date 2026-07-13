"""Quick comparison of GUI-labeled vs NN-relabeled training data."""
import pandas as pd, numpy as np, os

label_cols = ["weak_hole_xptar_center", "weak_hole_yptar_center", "foil_ytar_center",
              "weak_hole_xptar_tol", "weak_hole_yptar_tol", "foil_ytar_tol"]

# Load GUI data
gui_path = "/Users/zhengxiaoyang/Desktop/AI_ML R-SIDIS/stage2_soc_gui_labeled.csv"
print(f"Loading GUI: {gui_path}")
gui = pd.read_csv(gui_path)
print(f"GUI: {len(gui)} events")

for c in label_cols:
    if c in gui.columns:
        d = gui[c].dropna()
        print(f"  {c}: mean={d.mean():.6f}, std={d.std():.6f}")

# Check V3 training data
v3_path = "/Users/zhengxiaoyang/Desktop/AI_ML R-SIDIS/SHMS_Calibration_NN/outputs/iterative_gui_v3_preserved/dataset/iter1/stage2_nnrelabel.csv"
print(f"\nV3 iter1 path: {v3_path}")
print(f"  exists: {os.path.exists(v3_path)}")
if os.path.exists(v3_path):
    v3 = pd.read_csv(v3_path)
    print(f"V3 iter1: {len(v3)} events")
    for c in label_cols:
        if c in v3.columns:
            d = v3[c].dropna()
            print(f"  {c}: mean={d.mean():.6f}, std={d.std():.6f}")

# Compare labels: same events?
gui_ids = gui[["P_dc_x_fp", "P_dc_y_fp", "foil_position"]].dropna()
if os.path.exists(v3_path):
    v3 = pd.read_csv(v3_path)
    v3_ids = v3[["P_dc_x_fp", "P_dc_y_fp", "foil_position"]].dropna()
    print(f"\nGUI unique (x_fp, y_fp, foil) pairs: {len(gui_ids.drop_duplicates())}")
    print(f"V3 unique (x_fp, y_fp, foil) pairs: {len(v3_ids.drop_duplicates())}")
