import numpy as np
import pandas as pd
import torch

from training.data.preprocessing import ScalerBundle
from training.data.stage2_root_dataset import Stage2RootDataset
from training.losses import Stage2WeakLabelLoss


def _make_stage2_df(n: int = 32) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "P_dc_x_fp": rng.normal(0.0, 1.0, n),
            "P_dc_y_fp": rng.normal(0.0, 1.0, n),
            "P_dc_xp_fp": rng.normal(0.0, 0.02, n),
            "P_dc_yp_fp": rng.normal(0.0, 0.02, n),
            "P.rb.raster.fr_ybpm_tar": rng.normal(0.0, 0.1, n),
            "P_ngcer_npeSum": np.full(n, 5.0),
            "P_hgcer_npeSum": np.full(n, 1.0),
            "P_cal_etottracknorm": np.full(n, 1.0),
            "P_gtr_dp": rng.normal(0.0, 1.0, n),
            "P_gtr_th": rng.normal(0.0, 0.01, n),
            "P_gtr_ph": rng.normal(0.0, 0.01, n),
            "P_react_z": rng.normal(0.0, 2.0, n),
            "weak_hole_xptar_center": rng.normal(0.0, 0.01, n),
            "weak_hole_xptar_tol": np.full(n, 0.002),
            "weak_hole_yptar_center": rng.normal(0.0, 0.01, n),
            "weak_hole_yptar_tol": np.full(n, 0.002),
            "weak_foil_ytar_center": rng.normal(0.0, 2.0, n),
            "weak_foil_ytar_tol": np.full(n, 0.5),
            "foil_position": rng.integers(0, 3, n),
            "hole_id": rng.integers(0, 20, n),
            "weak_label_weight": np.full(n, 1.0),
        }
    )


def _make_scaler_bundle(df: pd.DataFrame) -> ScalerBundle:
    X = df[["P_dc_x_fp", "P_dc_y_fp", "P_dc_xp_fp", "P_dc_yp_fp", "P.rb.raster.fr_ybpm_tar"]].to_numpy()
    Y = df[["P_gtr_dp", "weak_hole_xptar_center", "weak_hole_yptar_center", "weak_foil_ytar_center"]].to_numpy()
    bundle = ScalerBundle(
        input_features=["x_fp", "y_fp", "xp_fp", "yp_fp", "fry"],
        target_features=["delta", "xptar", "yptar", "ytar"],
    )
    bundle.fit(X, Y)
    return bundle


def test_stage2_root_dataset_from_dataframe():
    df = _make_stage2_df(24)
    scaler = _make_scaler_bundle(df)
    ds = Stage2RootDataset(
        df,
        scaler_bundle=scaler,
        feature_schema=["x_fp", "y_fp", "xp_fp", "yp_fp", "fry"],
        branch_map={
            "x_fp": "P_dc_x_fp",
            "y_fp": "P_dc_y_fp",
            "xp_fp": "P_dc_xp_fp",
            "yp_fp": "P_dc_yp_fp",
        },
        direct_fry_branch="P.rb.raster.fr_ybpm_tar",
        label_map={
            "delta": {"enabled": True, "center_col": "P_gtr_dp", "tolerance": 0.0},
            "xptar": {"enabled": True, "center_col": "weak_hole_xptar_center", "tol_col": "weak_hole_xptar_tol"},
            "yptar": {"enabled": True, "center_col": "weak_hole_yptar_center", "tol_col": "weak_hole_yptar_tol"},
            "ytar": {"enabled": True, "center_col": "weak_foil_ytar_center", "tol_col": "weak_foil_ytar_tol"},
        },
        metadata_cols={"foil_position": "foil_position", "hole_id": "hole_id"},
        weight_col="weak_label_weight",
    )
    assert len(ds) == 24
    assert ds.X.shape[1] == 5
    sample = ds[0]
    assert "inputs" in sample and "targets" in sample and "tolerances" in sample
    assert sample["targets"]["xptar"].shape == (1,)


def test_stage2_weak_label_loss_deadzone_behavior():
    loss_fn = Stage2WeakLabelLoss(use_huber=False)
    batch = {
        "targets": {
            "delta": torch.tensor([[0.0], [0.0]]),
            "xptar": torch.tensor([[1.0], [1.0]]),
            "yptar": torch.tensor([[2.0], [2.0]]),
            "ytar": torch.tensor([[3.0], [3.0]]),
        },
        "tolerances": {
            "delta": torch.tensor([[0.0], [0.0]]),
            "xptar": torch.tensor([[0.2], [0.2]]),
            "yptar": torch.tensor([[0.1], [0.1]]),
            "ytar": torch.tensor([[0.5], [0.5]]),
        },
        "target_mask": {
            "delta": torch.tensor([[1.0], [1.0]]),
            "xptar": torch.tensor([[1.0], [1.0]]),
            "yptar": torch.tensor([[1.0], [1.0]]),
            "ytar": torch.tensor([[1.0], [1.0]]),
        },
        "weight": torch.tensor([1.0, 1.0]),
    }
    preds = {
        "delta": torch.tensor([[0.0], [0.0]]),
        "xptar": torch.tensor([[1.1], [0.85]]),  # both within tolerance
        "yptar": torch.tensor([[2.3], [2.0]]),   # first outside tolerance
        "ytar": torch.tensor([[3.0], [3.4]]),    # both within tolerance
    }
    loss = loss_fn(preds, batch)
    assert loss.item() > 0.0
    metrics = loss_fn.compute_metrics(preds, batch)
    assert metrics["xptar_within_tol"] == 1.0
    assert metrics["ytar_within_tol"] == 1.0
    assert metrics["yptar_within_tol"] < 1.0
