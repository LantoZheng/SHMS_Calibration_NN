"""Unit tests for SieveDataset (no real data required)."""

import pytest
import numpy as np
import pandas as pd
import torch

from training.data.sieve_dataset import SieveDataset


def _make_df(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "P_dc_x_fp": rng.uniform(-20, 20, n),
            "P_dc_y_fp": rng.uniform(-10, 10, n),
            "P_dc_xp_fp": rng.uniform(-0.06, 0.06, n),
            "P_dc_yp_fp": rng.uniform(-0.04, 0.04, n),
            "P_react_x": rng.normal(0, 0.1, n),
            "P_set": rng.uniform(3.5, 6.0, n),
            "I_mag": rng.uniform(0.8, 1.2, n),
            "P_gtr_dp": rng.uniform(-8, 8, n),
            "P_gtr_th": rng.uniform(-0.05, 0.05, n),
            "P_gtr_ph": rng.uniform(-0.03, 0.03, n),
            "P_react_z": rng.uniform(-5, 5, n),
            "cluster_weight": rng.uniform(0.5, 1.0, n),
        }
    )


def test_from_dataframe():
    df = _make_df(40)
    ds = SieveDataset(df)
    assert len(ds) == 40


def test_from_dict():
    df = _make_df(20)
    ds = SieveDataset(df.to_dict(orient="list"))
    assert len(ds) == 20


def test_getitem_keys():
    ds = SieveDataset(_make_df(10))
    item = ds[0]
    assert "inputs" in item
    assert "targets" in item
    assert set(item["targets"].keys()) == {"delta", "xptar", "yptar", "ytar"}


def test_target_shapes():
    ds = SieveDataset(_make_df(10))
    item = ds[0]
    for k, v in item["targets"].items():
        assert v.shape == (1,), f"{k} expected shape (1,), got {v.shape}"


def test_input_dim_with_xtar_and_p0():
    df = _make_df(10)
    ds = SieveDataset(df, p0_value=4.4, x_tar_col="P_react_x")
    # 4 DC + 1 x_tar + 1 p0 = 6 features
    assert ds.X.shape[1] == 6


def test_input_dim_without_xtar():
    df = _make_df(10)
    ds = SieveDataset(df, p0_value=None, x_tar_col=None)
    # Only 4 DC features
    assert ds.X.shape[1] == 4


def test_include_pset_imag_adds_features():
    df = _make_df(12)
    ds = SieveDataset(df, include_pset_imag=True)
    # 4 DC + 1 x_tar + 2 operating params = 7
    assert ds.X.shape[1] == 7


def test_pset_imag_dropped_when_disabled():
    df = _make_df(8)
    ds = SieveDataset(
        df,
        input_cols=[
            "P_dc_x_fp",
            "P_dc_y_fp",
            "P_dc_xp_fp",
            "P_dc_yp_fp",
            "P_set",
            "I_mag",
        ],
        x_tar_col=None,
        include_pset_imag=False,
    )
    assert ds.X.shape[1] == 4


def test_weight_col():
    df = _make_df(10)
    ds = SieveDataset(df, weight_col="cluster_weight")
    item = ds[0]
    assert "weight" in item
    assert isinstance(item["weight"], torch.Tensor)


def test_from_csv(tmp_path):
    df = _make_df(30)
    csv_path = str(tmp_path / "sieve.csv")
    df.to_csv(csv_path, index=False)
    ds = SieveDataset(csv_path)
    assert len(ds) == 30


def test_with_scaler():
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    df = _make_df(50)
    # Fit scalers on raw X/Y
    X_raw = df[["P_dc_x_fp", "P_dc_y_fp", "P_dc_xp_fp", "P_dc_yp_fp"]].values
    Y_raw = df[["P_gtr_dp", "P_gtr_th", "P_gtr_ph", "P_react_z"]].values
    scaler_X = StandardScaler().fit(X_raw)
    scaler_Y = StandardScaler().fit(Y_raw)

    ds = SieveDataset(df, x_tar_col=None, p0_value=None, scaler_X=scaler_X, scaler_Y=scaler_Y)
    assert len(ds) == 50
    # Scaled data should have approximately zero mean
    assert abs(ds.X.mean().item()) < 1.0


def test_scaler_mismatch_when_adding_pset_imag():
    from sklearn.preprocessing import StandardScaler

    df = _make_df(15)
    base_X = df[["P_dc_x_fp", "P_dc_y_fp", "P_dc_xp_fp", "P_dc_yp_fp"]].values
    scaler_X = StandardScaler().fit(base_X)

    with pytest.raises(ValueError):
        SieveDataset(
            df,
            x_tar_col=None,
            p0_value=None,
            scaler_X=scaler_X,
            include_pset_imag=True,
        )
