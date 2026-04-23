"""
Stage-2 dataset for full ROOT weak-label fine-tuning.

This dataset is designed for the upcoming SHMS `3foils + sieve` full ROOT
production file. It assumes that the weak-label construction stage has already
written geometric supervision back into the ROOT tree (or an equivalent
CSV/Parquet export) using columns such as:

- hole-angle centres / tolerances for xptar and yptar
- foil-centre / tolerance for ytar
- foil / hole metadata for validation splits

The implementation is intentionally config-driven so that exact branch names can
be fixed once the full ROOT file arrives.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from training.data.preprocessing import ScalerBundle, resolve_feature_schema


_TARGET_KEYS = ["delta", "xptar", "yptar", "ytar"]
_BASE_FEATURES = ["x_fp", "y_fp", "xp_fp", "yp_fp"]
_DEFAULT_BRANCH_MAP = {
    "x_fp": "P_dc_x_fp",
    "y_fp": "P_dc_y_fp",
    "xp_fp": "P_dc_xp_fp",
    "yp_fp": "P_dc_yp_fp",
}
_DEFAULT_LABEL_MAP = {
    "delta": {
        "enabled": True,
        "center_col": "P_gtr_dp",
        "tolerance": 0.0,
        "reference_branch": "P_gtr_dp",
    },
    "xptar": {
        "enabled": True,
        "center_col": "weak_hole_xptar_center",
        "tol_col": "weak_hole_xptar_tol",
        "reference_branch": "P_gtr_th",
    },
    "yptar": {
        "enabled": True,
        "center_col": "weak_hole_yptar_center",
        "tol_col": "weak_hole_yptar_tol",
        "reference_branch": "P_gtr_ph",
    },
    "ytar": {
        "enabled": True,
        "center_col": "weak_foil_ytar_center",
        "tol_col": "weak_foil_ytar_tol",
        "reference_branch": "P_react_z",
    },
}
_DEFAULT_METADATA_COLS = {
    "run_id": "run_id",
    "foil_position": "foil_position",
    "hole_id": "hole_id",
    "hole_row": "hole_row",
    "hole_col": "hole_col",
}
_DEFAULT_FRY_PROXY_BRANCHES = ["P_rb_raster_fryaRawAdc", "P_rb_raster_frybRawAdc"]
_DEFAULT_DIRECT_FRY_BRANCH = "P.rb.raster.fr_ybpm_tar"


@dataclass
class Stage2DatasetSummary:
    raw_events: int
    kept_events: int
    cutflow: Dict[str, int]


class Stage2RootDataset(Dataset):
    """Stage-2 weak-label dataset that can read ROOT, CSV, Parquet, or DataFrame."""

    def __init__(
        self,
        data_source: Union[str, Path, pd.DataFrame, dict],
        *,
        tree_name: str = "T",
        scaler_bundle: Optional[ScalerBundle] = None,
        feature_schema: Optional[Iterable[str]] = None,
        branch_map: Optional[Dict[str, str]] = None,
        label_map: Optional[Dict[str, dict]] = None,
        metadata_cols: Optional[Dict[str, str]] = None,
        weight_col: Optional[str] = None,
        fry_mode: str = "direct_or_proxy",
        direct_fry_branch: Optional[str] = None,
        fry_proxy_branches: Optional[Iterable[str]] = None,
        cuts: Optional[Dict[str, object]] = None,
        max_events: Optional[int] = None,
    ) -> None:
        self.scaler_bundle = scaler_bundle
        self.feature_names = resolve_feature_schema(
            feature_schema,
            include_fry="fry" in list(feature_schema or []),
            include_xtar=False,
            include_p0=False,
        )
        self.branch_map = {**_DEFAULT_BRANCH_MAP, **dict(branch_map or {})}
        self.label_map = {
            key: {**_DEFAULT_LABEL_MAP.get(key, {}), **dict((label_map or {}).get(key, {}))}
            for key in _TARGET_KEYS
        }
        self.metadata_cols = {**_DEFAULT_METADATA_COLS, **dict(metadata_cols or {})}
        self.weight_col = weight_col or "weak_label_weight"
        self.fry_mode = str(fry_mode)
        self.direct_fry_branch = direct_fry_branch or _DEFAULT_DIRECT_FRY_BRANCH
        self.fry_proxy_branches = list(fry_proxy_branches or _DEFAULT_FRY_PROXY_BRANCHES)
        self.cuts = dict(cuts or {})

        requested_columns = self._collect_required_columns()
        df = self._load_dataframe(data_source, tree_name=tree_name, columns=requested_columns)
        if max_events is not None:
            df = df.iloc[: max_events].copy()

        self.raw_df = df.copy()
        self.summary = self._apply_cuts(df)
        self.df = df.reset_index(drop=True)

        X_raw = self._build_input_matrix(self.df)
        target_centres, target_tolerances, target_masks = self._build_target_tensors(self.df)

        if scaler_bundle is not None:
            X_scaled = scaler_bundle.transform_X(X_raw).astype(np.float32)
            centre_matrix = np.column_stack([target_centres[k] for k in _TARGET_KEYS]).astype(np.float32)
            centre_scaled = scaler_bundle.transform_Y(centre_matrix).astype(np.float32)
            tol_scaled = np.column_stack(
                [
                    target_tolerances[k].astype(np.float32) / float(scaler_bundle.scaler_Y.scale_[i])
                    for i, k in enumerate(_TARGET_KEYS)
                ]
            )
        else:
            X_scaled = X_raw.astype(np.float32)
            centre_scaled = np.column_stack([target_centres[k] for k in _TARGET_KEYS]).astype(np.float32)
            tol_scaled = np.column_stack([target_tolerances[k] for k in _TARGET_KEYS]).astype(np.float32)

        self.X = torch.tensor(X_scaled, dtype=torch.float32)
        self.targets = {
            k: torch.tensor(centre_scaled[:, i : i + 1], dtype=torch.float32)
            for i, k in enumerate(_TARGET_KEYS)
        }
        self.tolerances = {
            k: torch.tensor(tol_scaled[:, i : i + 1], dtype=torch.float32)
            for i, k in enumerate(_TARGET_KEYS)
        }
        self.target_mask = {
            k: torch.tensor(target_masks[k].astype(np.float32).reshape(-1, 1), dtype=torch.float32)
            for k in _TARGET_KEYS
        }

        if weight_col and weight_col in self.df.columns:
            self.weights = torch.tensor(self.df[weight_col].to_numpy(dtype=np.float32), dtype=torch.float32)
        else:
            self.weights = None

        self.metadata = self._build_metadata_frame(self.df)

    def _collect_required_columns(self) -> list[str]:
        columns = set()
        for feat in _BASE_FEATURES:
            branch = self.branch_map.get(feat, f"P_dc_{feat}" if feat.endswith("_fp") else None)
            if branch is None:
                raise KeyError(f"Missing branch_map entry for feature '{feat}'")
            columns.add(branch)

        if "fry" in self.feature_names:
            if self.direct_fry_branch:
                columns.add(self.direct_fry_branch)
            columns.update(self.fry_proxy_branches)

        for cfg in self.label_map.values():
            center_col = cfg.get("center_col")
            tol_col = cfg.get("tol_col")
            if center_col:
                columns.add(center_col)
            if tol_col:
                columns.add(tol_col)

        for col in self.metadata_cols.values():
            if col:
                columns.add(col)

        if self.weight_col:
            columns.add(self.weight_col)

        if bool(self.cuts.get("use_pid", True)):
            for name in ["P_ngcer_npeSum", "P_hgcer_npeSum", "P_cal_etottracknorm"]:
                columns.add(name)
            aero_branch = self.cuts.get("aero_branch", "P_aero_npeSum")
            columns.add(aero_branch)

        if bool(self.cuts.get("use_quality", True)):
            for name in [
                self.label_map.get("delta", {}).get("reference_branch", "P_gtr_dp"),
                self.label_map.get("xptar", {}).get("reference_branch", "P_gtr_th"),
                self.label_map.get("yptar", {}).get("reference_branch", "P_gtr_ph"),
                self.label_map.get("ytar", {}).get("reference_branch", "P_react_z"),
            ]:
                if name:
                    columns.add(name)

        return sorted(columns)

    @staticmethod
    def _load_dataframe(
        data_source: Union[str, Path, pd.DataFrame, dict],
        *,
        tree_name: str,
        columns: list[str],
    ) -> pd.DataFrame:
        if isinstance(data_source, pd.DataFrame):
            return data_source.copy()
        if isinstance(data_source, dict):
            return pd.DataFrame(data_source)

        path = Path(data_source)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix == ".parquet":
            return pd.read_parquet(path)
        if suffix == ".root":
            import uproot

            with uproot.open(path) as f:
                tree = f[tree_name]
                return tree.arrays(columns, library="pd")

        raise TypeError(f"Unsupported data source: {data_source}")

    def _apply_cuts(self, df: pd.DataFrame) -> Stage2DatasetSummary:
        raw_events = len(df)
        cutflow = {"raw": raw_events}

        finite_cols = list(df.columns)
        finite = np.isfinite(df[finite_cols]).all(axis=1)
        df.drop(index=df.index[~finite], inplace=True)
        cutflow["finite"] = int(len(df))

        if bool(self.cuts.get("use_pid", True)):
            ngcer_min = float(self.cuts.get("ngcer_min", 2.0))
            hgcer_min = float(self.cuts.get("hgcer_min", 0.5))
            cal_min = float(self.cuts.get("cal_etot_min", 0.6))
            cal_max = float(self.cuts.get("cal_etot_max", 1.8))
            aero_min = self.cuts.get("aero_min", None)
            aero_branch = str(self.cuts.get("aero_branch", "P_aero_npeSum"))

            pid_mask = (
                (df["P_ngcer_npeSum"] >= ngcer_min)
                & (df["P_hgcer_npeSum"] >= hgcer_min)
                & df["P_cal_etottracknorm"].between(cal_min, cal_max)
            )
            if aero_min is not None and aero_branch in df.columns:
                pid_mask &= df[aero_branch] >= float(aero_min)
            df.drop(index=df.index[~pid_mask], inplace=True)
        cutflow["pid"] = int(len(df))

        if bool(self.cuts.get("use_quality", True)):
            dp_branch = self.label_map.get("delta", {}).get("reference_branch", "P_gtr_dp")
            th_branch = self.label_map.get("xptar", {}).get("reference_branch", "P_gtr_th")
            ph_branch = self.label_map.get("yptar", {}).get("reference_branch", "P_gtr_ph")
            z_branch = self.label_map.get("ytar", {}).get("reference_branch", "P_react_z")
            quality_mask = (
                df[dp_branch].between(float(self.cuts.get("dp_min", -25.0)), float(self.cuts.get("dp_max", 22.0)))
                & (df[th_branch].abs() <= float(self.cuts.get("xptar_abs_max", 0.08)))
                & (df[ph_branch].abs() <= float(self.cuts.get("yptar_abs_max", 0.06)))
                & (df[z_branch].abs() <= float(self.cuts.get("ytar_abs_max", 120.0)))
            )
            df.drop(index=df.index[~quality_mask], inplace=True)
        cutflow["quality"] = int(len(df))

        return Stage2DatasetSummary(raw_events=raw_events, kept_events=len(df), cutflow=cutflow)

    def _build_input_matrix(self, df: pd.DataFrame) -> np.ndarray:
        feature_arrays = []
        for feature in _BASE_FEATURES:
            branch = self.branch_map[feature]
            feature_arrays.append(df[branch].to_numpy(dtype=np.float32).reshape(-1, 1))

        if "fry" in self.feature_names:
            feature_arrays.append(self._build_fry_feature(df).reshape(-1, 1))

        return np.concatenate(feature_arrays, axis=1)

    def _build_fry_feature(self, df: pd.DataFrame) -> np.ndarray:
        if self.direct_fry_branch and self.direct_fry_branch in df.columns:
            return df[self.direct_fry_branch].to_numpy(dtype=np.float32)

        if self.fry_mode in {"direct_or_proxy", "adc-normalized"} and len(self.fry_proxy_branches) >= 2:
            if self.scaler_bundle is None:
                raise RuntimeError("fry proxy mode requires scaler_bundle for MC-aligned scaling.")
            avg_adc = df[self.fry_proxy_branches].mean(axis=1).to_numpy(dtype=np.float64)
            adc_mean = float(np.mean(avg_adc))
            adc_std = float(np.std(avg_adc))
            mc_fry_mean = float(self.scaler_bundle.scaler_X.mean_[4])
            mc_fry_scale = float(self.scaler_bundle.scaler_X.scale_[4])
            if adc_std < 1e-12:
                return np.full(len(df), mc_fry_mean, dtype=np.float32)
            z = (avg_adc - adc_mean) / adc_std
            return (mc_fry_mean + z * mc_fry_scale).astype(np.float32)

        if self.fry_mode == "mc-mean":
            if self.scaler_bundle is None:
                raise RuntimeError("mc-mean fry mode requires scaler_bundle.")
            mc_fry_mean = float(self.scaler_bundle.scaler_X.mean_[4])
            return np.full(len(df), mc_fry_mean, dtype=np.float32)

        raise KeyError(
            "Feature schema requests 'fry' but neither a direct fry branch nor a valid proxy configuration was available."
        )

    def _build_target_tensors(
        self,
        df: pd.DataFrame,
    ) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        centres: Dict[str, np.ndarray] = {}
        tolerances: Dict[str, np.ndarray] = {}
        masks: Dict[str, np.ndarray] = {}

        for key in _TARGET_KEYS:
            cfg = dict(self.label_map.get(key, {}))
            center_col = cfg.get("center_col")
            tol_col = cfg.get("tol_col")
            tol_value = float(cfg.get("tolerance", 0.0))
            enabled = bool(cfg.get("enabled", center_col is not None))

            if enabled and center_col and center_col in df.columns:
                centre = df[center_col].to_numpy(dtype=np.float32)
                mask = np.isfinite(centre)
            else:
                centre = np.zeros(len(df), dtype=np.float32)
                mask = np.zeros(len(df), dtype=bool)

            if enabled and tol_col and tol_col in df.columns:
                tol = df[tol_col].to_numpy(dtype=np.float32)
                tol = np.where(np.isfinite(tol), tol, tol_value).astype(np.float32)
            else:
                tol = np.full(len(df), tol_value, dtype=np.float32)

            centres[key] = centre
            tolerances[key] = tol
            masks[key] = mask

        return centres, tolerances, masks

    def _build_metadata_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        metadata = pd.DataFrame(index=df.index)
        for key, col in self.metadata_cols.items():
            if col in df.columns:
                metadata[key] = df[col]
            else:
                metadata[key] = -1
        return metadata.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        item: Dict[str, object] = {
            "inputs": self.X[idx],
            "targets": {k: v[idx] for k, v in self.targets.items()},
            "tolerances": {k: v[idx] for k, v in self.tolerances.items()},
            "target_mask": {k: v[idx] for k, v in self.target_mask.items()},
            "metadata": {
                key: torch.tensor(self.metadata.iloc[idx][key])
                if np.issubdtype(type(self.metadata.iloc[idx][key]), np.number)
                else self.metadata.iloc[idx][key]
                for key in self.metadata.columns
            },
        }
        if self.weights is not None:
            item["weight"] = self.weights[idx]
        return item
