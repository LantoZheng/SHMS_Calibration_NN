"""
SIMCDataset — PyTorch Dataset wrapping SIMC ROOT ntuples.

SIMC branch → physical meaning mapping
---------------------------------------
hsxfp   → x_fp   (focal-plane x, cm)
hsyfp   → y_fp   (focal-plane y, cm)
hsxpfp  → xp_fp  (focal-plane x′, rad)
hsypfp  → yp_fp  (focal-plane y′, rad)
hsdeltai→ delta  (thrown δ = (p−p0)/p0, %)
hsxptari→ xptar  (thrown θ at target, rad)
hsyptari→ yptar  (thrown φ at target, rad)
hsztari → ytar   (thrown z_react → used directly as y_tar proxy, cm)

x_tar is synthesised as Gaussian noise (raster model) with σ = x_tar_sigma_cm.
p0 is appended as a constant column (units: GeV/c).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from training.data.preprocessing import ScalerBundle, add_p0_feature, add_xtar_feature

# Default SIMC branch names ──────────────────────────────────────────────────
_DEFAULT_INPUT_BRANCHES: List[str] = ["hsxfp", "hsyfp", "hsxpfp", "hsypfp"]
_DEFAULT_TARGET_BRANCHES: List[str] = ["hsdeltai", "hsxptari", "hsyptari", "hsztari"]


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
    p0_value          : central momentum (GeV/c); appended as a constant column.
    max_events        : optional cap on total number of events loaded.
    cuts              : optional boolean-array cut expression string; evaluated
                        via ``pandas.DataFrame.eval`` on the raw branch data.
    scaler_X          : pre-fitted StandardScaler for inputs (optional).
    scaler_Y          : pre-fitted StandardScaler for targets (optional).
    fit_scalers       : if True *and* scaler_X/scaler_Y are None, fit new
                        scalers on the loaded data.
    x_tar_sigma_cm    : σ (cm) for the Gaussian raster model used to synthesise
                        x_tar (default: 0.1 cm).
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
        self.x_tar_sigma_cm = x_tar_sigma_cm

        all_branches = list(set(self.input_branches + self.target_branches))

        frames: list = []
        for fpath in root_file_paths:
            with uproot.open(fpath) as root_file:
                tree = root_file[tree_name]
                # Only request branches that exist in this tree
                available = set(tree.keys())
                wanted = [b for b in all_branches if b in available]
                arr = tree.arrays(wanted, library="pd")
            frames.append(arr)

        df = pd.concat(frames, ignore_index=True)

        # Apply optional cuts
        if cuts:
            mask = df.eval(cuts)
            df = df.loc[mask].reset_index(drop=True)

        if max_events is not None:
            df = df.iloc[:max_events].reset_index(drop=True)

        rng = np.random.default_rng(rng_seed)

        # Build raw input array + synthesised x_tar
        X_raw = df[self.input_branches].to_numpy(dtype=np.float32)
        x_tar = rng.normal(0.0, x_tar_sigma_cm, size=(len(df),)).astype(np.float32)
        X_raw = add_xtar_feature(X_raw, x_tar)

        if p0_value is not None:
            X_raw = add_p0_feature(X_raw, float(p0_value))

        # Build raw target array
        Y_raw = df[self.target_branches].to_numpy(dtype=np.float32)

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
