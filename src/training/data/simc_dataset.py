"""
SIMCDataset — PyTorch Dataset wrapping SIMC ROOT ntuples.

SIMC branch → internal feature mapping
--------------------------------------
The loader accepts either the legacy `hs*` naming used by older ntuples or the
`ps*` naming produced by the current single-arm MC workflow.

Input aliases
~~~~~~~~~~~~~
hsxfp / psxfp     → x_fp   (focal-plane x, cm)
hsyfp / psyfp     → y_fp   (focal-plane y, cm)
hsxpfp / psxpfp   → xp_fp  (focal-plane x′, rad)
hsypfp / psypfp   → yp_fp  (focal-plane y′, rad)
fry / hsfry / psfry → fry  (beam raster y, cm; optional)

Target aliases
~~~~~~~~~~~~~~
hsdeltai / psdeltai   → delta  (thrown δ = (p−p0)/p0, %)
hsxptari / psxptari   → xptar  (thrown θ at target, rad)
hsyptari / psyptari   → yptar  (thrown φ at target, rad)
hsztari / psztari     → ytar   (analysis convention: z_react used as y_tar proxy, cm)

x_tar is synthesised as Gaussian noise (raster model) with σ = x_tar_sigma_cm.
p0 is appended as a constant column (units: GeV/c).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from training.data.preprocessing import add_p0_feature, add_xtar_feature, resolve_feature_schema

# Default canonical branch keys ───────────────────────────────────────────────
_DEFAULT_INPUT_BRANCHES: List[str] = ["x_fp", "y_fp", "xp_fp", "yp_fp"]
_DEFAULT_TARGET_BRANCHES: List[str] = ["delta", "xptar", "yptar", "ytar"]

_BRANCH_ALIASES: Dict[str, List[str]] = {
    "x_fp": ["hsxfp", "psxfp", "pfxfp"],
    "y_fp": ["hsyfp", "psyfp", "pfyfp"],
    "xp_fp": ["hsxpfp", "psxpfp", "pfxpfp"],
    "yp_fp": ["hsypfp", "psypfp", "pfypfp"],
    "delta": ["hsdeltai", "psdeltai", "pfdeltai"],
    "xptar": ["hsxptari", "psxptari", "pfxptari"],
    "yptar": ["hsyptari", "psyptari", "pfyptari"],
    "ytar": ["hsztari", "psztari", "pfztari"],
}


def _resolve_branch_alias(
    branch_key: str,
    available_branches: List[str],
) -> str:
    """Resolve a canonical branch key or explicit branch name to an actual branch."""
    if branch_key in available_branches:
        return branch_key

    candidates = _BRANCH_ALIASES.get(branch_key, [branch_key])
    for candidate in candidates:
        if candidate in available_branches:
            return candidate

    raise KeyError(
        f"Could not resolve branch '{branch_key}'. Checked: {', '.join(candidates)}"
    )


def _resolve_branch_group(
    branch_keys: List[str],
    available_branches: List[str],
) -> Dict[str, str]:
    """Resolve a list of canonical branch keys to actual branch names."""
    return {branch_key: _resolve_branch_alias(branch_key, available_branches) for branch_key in branch_keys}


def _resolve_fry_branch(available_branches: List[str], requested_branch: Optional[str]) -> str:
    candidates = [requested_branch] if requested_branch else ["fry", "hsfry", "psfry"]
    for candidate in candidates:
        if candidate and candidate in available_branches:
            return candidate
    raise KeyError(
        "Requested fry feature, but no fry branch was found. Checked: "
        + ", ".join([c for c in candidates if c])
    )


class SIMCDataset(Dataset):
    """
    Load SIMC ROOT ntuples and expose them as a PyTorch Dataset.

    Parameters
    ----------
    root_file_paths   : list of ROOT file paths (glob patterns are NOT expanded
                        here — pass already-expanded paths).
    tree_name         : ROOT tree name inside each file (SIMC default: 'h10').
    input_branches    : list of branch names for model inputs.
    target_branches   : list of branch names for model targets.
    p0_value          : central momentum (GeV/c); appended only when the
                        feature schema includes `p0`.
    max_events        : optional cap on total number of events loaded.
    cuts              : optional boolean-array cut expression string; evaluated
                        via ``pandas.DataFrame.eval`` on the raw branch data.
    scaler_X          : pre-fitted StandardScaler for inputs (optional).
    scaler_Y          : pre-fitted StandardScaler for targets (optional).
    fit_scalers       : if True *and* scaler_X/scaler_Y are None, fit new
                        scalers on the loaded data.
    feature_schema    : ordered input feature schema. Defaults to the 4-D
                        focal-plane set, with optional legacy additions.
    include_fry       : legacy helper to include fry when feature_schema is not
                        explicitly provided.
    fry_branch        : explicit fry branch name; defaults to auto-detect.
    x_tar_sigma_cm    : σ (cm) for the Gaussian raster model used to synthesise
                        x_tar when requested in the feature schema.
    rng_seed          : seed for the x_tar RNG (reproducibility).
    """

    def __init__(
        self,
        root_file_paths: List[str],
        tree_name: str = "h10",
        input_branches: Optional[List[str]] = None,
        target_branches: Optional[List[str]] = None,
        p0_value: Optional[float] = None,
        max_events: Optional[int] = None,
        cuts: Optional[str] = None,
        scaler_X: Optional[object] = None,
        scaler_Y: Optional[object] = None,
        fit_scalers: bool = False,
        feature_schema: Optional[List[str]] = None,
        include_fry: bool = False,
        fry_branch: Optional[str] = None,
        x_tar_sigma_cm: float = 0.1,
        rng_seed: int = 42,
    ) -> None:
        try:
            import uproot  # lazy import — not needed at import time
        except ImportError as exc:
            raise ImportError(
                "uproot is required for SIMCDataset. "
                "Install it with: pip install uproot"
            ) from exc

        import pandas as pd

        self.input_branches = list(input_branches or _DEFAULT_INPUT_BRANCHES)
        self.target_branches = list(target_branches or _DEFAULT_TARGET_BRANCHES)
        self.p0_value = p0_value
        self.include_fry = include_fry
        self.fry_branch = fry_branch
        self.x_tar_sigma_cm = x_tar_sigma_cm
        self.feature_names = resolve_feature_schema(
            feature_schema,
            include_fry=include_fry,
            include_xtar=False,
            include_p0=(p0_value is not None),
        )

        frames: list = []
        for fpath in root_file_paths:
            with uproot.open(fpath) as root_file:
                tree = root_file[tree_name]
                available = list(tree.keys())
                resolved_inputs = _resolve_branch_group(self.input_branches, available)
                resolved_targets = _resolve_branch_group(self.target_branches, available)
                resolved_fry_branch = None
                if "fry" in self.feature_names:
                    resolved_fry_branch = _resolve_fry_branch(available, fry_branch)
                wanted = list(dict.fromkeys([
                    *resolved_inputs.values(),
                    *resolved_targets.values(),
                    *([resolved_fry_branch] if resolved_fry_branch else []),
                ]))
                arr = tree.arrays(wanted, library="pd")
                rename_map = {
                    **{actual: canonical for canonical, actual in resolved_inputs.items()},
                    **{actual: canonical for canonical, actual in resolved_targets.items()},
                }
                if resolved_fry_branch is not None:
                    rename_map[resolved_fry_branch] = "fry"
                arr = arr.rename(columns=rename_map)
            frames.append(arr)

        df = pd.concat(frames, ignore_index=True)

        # Apply optional cuts
        if cuts:
            mask = df.eval(cuts)
            df = df.loc[mask].reset_index(drop=True)

        if max_events is not None:
            df = df.iloc[:max_events].reset_index(drop=True)

        rng = np.random.default_rng(rng_seed)

        # Build raw input array according to the requested feature schema
        X_raw = df[_DEFAULT_INPUT_BRANCHES].to_numpy(dtype=np.float32)
        assembled_names = ["x_fp", "y_fp", "xp_fp", "yp_fp"]
        if "fry" in self.feature_names:
            if "fry" not in df.columns:
                raise KeyError("Requested fry feature, but no resolved fry column is present in the dataset")
            fry = df["fry"].to_numpy(dtype=np.float32).reshape(-1, 1)
            X_raw = np.concatenate([X_raw, fry], axis=1)
            assembled_names.append("fry")
        if "x_tar" in self.feature_names:
            x_tar = rng.normal(0.0, x_tar_sigma_cm, size=(len(df),)).astype(np.float32)
            X_raw = add_xtar_feature(X_raw, x_tar)
            assembled_names.append("x_tar")

        if "p0" in self.feature_names:
            if p0_value is None:
                raise ValueError("Feature schema requests 'p0' but no p0_value was provided.")
            X_raw = add_p0_feature(X_raw, float(p0_value))
            assembled_names.append("p0")

        if assembled_names != self.feature_names:
            raise RuntimeError(
                f"Resolved feature assembly order {assembled_names} does not match requested schema {self.feature_names}."
            )

        # Build raw target array
        Y_raw = df[_DEFAULT_TARGET_BRANCHES].to_numpy(dtype=np.float32)

        # Scalers
        self._scaler_X = scaler_X
        self._scaler_Y = scaler_Y

        if fit_scalers and scaler_X is None and scaler_Y is None:
            from sklearn.preprocessing import StandardScaler

            self._scaler_X = StandardScaler().fit(X_raw)
            self._scaler_Y = StandardScaler().fit(Y_raw)

        X = self._scaler_X.transform(X_raw) if self._scaler_X is not None else X_raw
        Y = self._scaler_Y.transform(Y_raw) if self._scaler_Y is not None else Y_raw

        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x = self.X[idx]
        y = self.Y[idx]
        return {
            "inputs": x,
            "targets": {
                "delta": y[0:1],
                "xptar": y[1:2],
                "yptar": y[2:3],
                "ytar": y[3:4],
            },
        }

    @property
    def scaler_X(self):
        return self._scaler_X

    @property
    def scaler_Y(self):
        return self._scaler_Y
