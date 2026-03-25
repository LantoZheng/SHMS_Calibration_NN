"""
SieveDataset — PyTorch Dataset wrapping pre-labeled sieve/foil data.

The dataset accepts data already processed by SHMS_Optics_calibration_tools
(cluster-labeled sieve-hole / multi-foil events) in one of three formats:
    * pandas DataFrame
    * path to a CSV or Parquet file
    * dict {column_name: array}

Default column naming follows hcana branch conventions used in the
existing models/ calibration reports.

Input columns (hcana names)
----------------------------
P_dc_x_fp   → x_fp  (focal-plane x, cm)
P_dc_y_fp   → y_fp  (focal-plane y, cm)
P_dc_xp_fp  → xp_fp (focal-plane x′, rad)
P_dc_yp_fp  → yp_fp (focal-plane y′, rad)
[P_react_x] → x_tar (optional, appended after DC columns)
[p0 column] → central momentum (optional constant, appended last)
[P_set]     → magnet momentum setpoint (optional)
[I_mag]     → magnet current (optional)

Target columns (hcana / reconstructed names)
----------------------------------------------
P_gtr_dp    → delta (momentum offset, %)
P_gtr_th    → xptar (in-plane angle at target, rad)
P_gtr_ph    → yptar (out-of-plane angle at target, rad)
P_react_z   → ytar  (reaction vertex z, cm)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from training.data.preprocessing import ScalerBundle, add_p0_feature, add_xtar_feature

_DEFAULT_INPUT_COLS: List[str] = [
    "P_dc_x_fp",
    "P_dc_y_fp",
    "P_dc_xp_fp",
    "P_dc_yp_fp",
]
_DEFAULT_TARGET_COLS: List[str] = [
    "P_gtr_dp",
    "P_gtr_th",
    "P_gtr_ph",
    "P_react_z",
]
_TARGET_KEYS: List[str] = ["delta", "xptar", "yptar", "ytar"]


class SieveDataset(Dataset):
    """
    Dataset for sieve-hole / foil calibration data (pre-labeled).

    Parameters
    ----------
    data_source   : pd.DataFrame, CSV/Parquet path (str), or dict.
    input_cols    : input column names (DC focal-plane variables).
    target_cols   : target column names (reconstructed optics quantities).
    p0_value      : central momentum (GeV/c); appended as a constant column.
    x_tar_col     : column name for target-plane x (appended after DC cols).
                    Pass None to omit.
    weight_col    : optional column name for per-sample weights
                    (e.g., cluster confidence).
    scaler_X      : pre-fitted scaler for inputs (should come from pre-training).
    scaler_Y      : pre-fitted scaler for targets.
    include_pset_imag : if True, append P_set and I_mag columns to the inputs.
                        If False (default), they are dropped even if present in
                        the dataframe or input_cols.
    """

    def __init__(
        self,
        data_source: Union[pd.DataFrame, str, dict],
        input_cols: Optional[List[str]] = None,
        target_cols: Optional[List[str]] = None,
        p0_value: Optional[float] = None,
        x_tar_col: Optional[str] = "P_react_x",
        weight_col: Optional[str] = None,
        scaler_X: Optional[object] = None,
        scaler_Y: Optional[object] = None,
        include_pset_imag: bool = False,
    ) -> None:
        df = self._load(data_source)

        operating_cols = ["P_set", "I_mag"]
        base_input_cols = [c for c in (input_cols or _DEFAULT_INPUT_COLS) if c not in operating_cols]

        self.input_cols = base_input_cols
        self.target_cols = list(target_cols or _DEFAULT_TARGET_COLS)
        self.p0_value = p0_value
        self.x_tar_col = x_tar_col
        self.weight_col = weight_col
        self.include_pset_imag = include_pset_imag
        self._operating_cols = operating_cols if include_pset_imag else []

        # Build raw input array
        X_raw = df[self.input_cols].to_numpy(dtype=np.float32)

        if include_pset_imag:
            missing = [c for c in operating_cols if c not in df.columns]
            if missing:
                raise KeyError(
                    f"include_pset_imag=True but missing columns: {missing}"
                )
            ops = df[operating_cols].to_numpy(dtype=np.float32)
            X_raw = np.concatenate([X_raw, ops], axis=1)

        if x_tar_col and x_tar_col in df.columns:
            x_tar = df[x_tar_col].to_numpy(dtype=np.float32)
            X_raw = add_xtar_feature(X_raw, x_tar)

        if p0_value is not None:
            X_raw = add_p0_feature(X_raw, float(p0_value))

        if (
            scaler_X is not None
            and hasattr(scaler_X, "n_features_in_")
            and scaler_X.n_features_in_ != X_raw.shape[1]
        ):
            raise ValueError(
                "scaler_X expects "
                f"{getattr(scaler_X, 'n_features_in_', 'unknown')} features "
                f"but input data has {X_raw.shape[1]}. "
                "Check include_pset_imag and input column configuration to "
                "match the scaler fitted during pre-training."
            )

        # Build raw target array
        Y_raw = df[self.target_cols].to_numpy(dtype=np.float32)

        # Apply scalers (fine-tuning MUST reuse pre-training scalers)
        X = scaler_X.transform(X_raw) if scaler_X is not None else X_raw
        Y = scaler_Y.transform(Y_raw) if scaler_Y is not None else Y_raw

        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

        # Per-sample weights (optional)
        if weight_col and weight_col in df.columns:
            w = df[weight_col].to_numpy(dtype=np.float32)
            self.weights = torch.tensor(w, dtype=torch.float32)
        else:
            self.weights = None

    @staticmethod
    def _load(data_source: Union[pd.DataFrame, str, dict]) -> pd.DataFrame:
        if isinstance(data_source, pd.DataFrame):
            return data_source.copy()
        if isinstance(data_source, dict):
            return pd.DataFrame(data_source)
        if isinstance(data_source, str):
            path = data_source
            if path.endswith(".parquet"):
                return pd.read_parquet(path)
            return pd.read_csv(path)
        raise TypeError(
            f"data_source must be a DataFrame, str, or dict; got {type(data_source)}"
        )

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        y = self.Y[idx]
        item: Dict[str, object] = {
            "inputs": self.X[idx],
            "targets": {
                "delta": y[0:1],
                "xptar": y[1:2],
                "yptar": y[2:3],
                "ytar": y[3:4],
            },
        }
        if self.weights is not None:
            item["weight"] = self.weights[idx]
        return item
