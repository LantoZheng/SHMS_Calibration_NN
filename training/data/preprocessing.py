"""
ScalerBundle and feature-engineering utilities for the SHMS optics pipeline.

ScalerBundle wraps a pair of StandardScalers (for X and Y) together with
feature-name metadata so that the same normalisation is reproducibly
applied across pre-training and fine-tuning stages.

Utility functions
-----------------
add_p0_feature(X, p0)          — append a constant p0 column to X
add_xtar_feature(X, x_tar)     — append an x_tar column to X
"""

from __future__ import annotations

import json
import os
from typing import List

import numpy as np
from sklearn.preprocessing import StandardScaler


class ScalerBundle:
    """
    Paired X / Y StandardScalers with reproducible save / load.

    Parameters
    ----------
    input_features  : list[str]  Names of the input columns (for metadata).
    target_features : list[str]  Names of the target columns (for metadata).
    """

    def __init__(
        self,
        input_features: List[str],
        target_features: List[str],
    ) -> None:
        self.input_features = list(input_features)
        self.target_features = list(target_features)
        self.scaler_X = StandardScaler()
        self.scaler_Y = StandardScaler()
        self._fitted = False

    # ── Fit / transform ──────────────────────────────────────────────────

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "ScalerBundle":
        """Fit both scalers on training data. Returns self."""
        self.scaler_X.fit(X)
        self.scaler_Y.fit(Y)
        self._fitted = True
        return self

    def set_fitted_scalers(
        self, scaler_X: StandardScaler, scaler_Y: StandardScaler
    ) -> "ScalerBundle":
        """
        Attach externally-fitted scalers and mark this bundle as fitted.

        Useful when scalers are fitted by another component (e.g. SIMCDataset)
        but should still be persisted via ScalerBundle.save().
        """
        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y
        self._fitted = True
        return self

    def transform_X(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.scaler_X.transform(X)

    def transform_Y(self, Y: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.scaler_Y.transform(Y)

    def inverse_transform_Y(self, Y_scaled: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.scaler_Y.inverse_transform(Y_scaled)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("ScalerBundle has not been fitted yet.")

    # ── Persist ──────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """
        Save the bundle to *path* (JSON with embedded numpy arrays).

        The JSON file contains mean/scale arrays for both scalers plus
        feature-name metadata. No external numpy files are written.
        """
        self._check_fitted()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        payload = {
            "input_features": self.input_features,
            "target_features": self.target_features,
            "scaler_X_mean": self.scaler_X.mean_.tolist(),
            "scaler_X_scale": self.scaler_X.scale_.tolist(),
            "scaler_Y_mean": self.scaler_Y.mean_.tolist(),
            "scaler_Y_scale": self.scaler_Y.scale_.tolist(),
        }
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "ScalerBundle":
        """Load a previously saved ScalerBundle from *path*."""
        with open(path) as fh:
            payload = json.load(fh)

        bundle = cls(
            input_features=payload["input_features"],
            target_features=payload["target_features"],
        )

        def _restore(scaler: StandardScaler, mean: list, scale: list) -> None:
            scaler.mean_ = np.array(mean, dtype=np.float64)
            scaler.scale_ = np.array(scale, dtype=np.float64)
            scaler.var_ = scaler.scale_ ** 2
            scaler.n_features_in_ = len(mean)
            scaler.n_samples_seen_ = 0  # not stored, set to sentinel

        _restore(bundle.scaler_X, payload["scaler_X_mean"], payload["scaler_X_scale"])
        _restore(bundle.scaler_Y, payload["scaler_Y_mean"], payload["scaler_Y_scale"])
        bundle._fitted = True
        return bundle


# ── Feature-engineering utilities ────────────────────────────────────────────


def add_p0_feature(X: np.ndarray, p0: float) -> np.ndarray:
    """
    Append a constant p0 column (GeV/c) to the feature matrix X.

    Parameters
    ----------
    X  : (N, M) array
    p0 : scalar central momentum value

    Returns
    -------
    (N, M+1) array with p0 appended as the last column.
    """
    col = np.full((X.shape[0], 1), float(p0), dtype=X.dtype)
    return np.concatenate([X, col], axis=1)


def add_xtar_feature(X: np.ndarray, x_tar: np.ndarray) -> np.ndarray:
    """
    Append an x_tar column to the feature matrix X.

    Parameters
    ----------
    X     : (N, M) array
    x_tar : (N,) or (N, 1) array of target x-plane values

    Returns
    -------
    (N, M+1) array.
    """
    col = np.asarray(x_tar, dtype=X.dtype).reshape(-1, 1)
    return np.concatenate([X, col], axis=1)
