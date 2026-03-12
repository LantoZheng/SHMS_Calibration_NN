"""
OpticsEvaluator — Comprehensive evaluation and comparison utilities.

Evaluates a trained ResidualMLP against the polynomial baseline stored in
the existing models/poly_coeffs_*.json format.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

_TARGET_KEYS = ["delta", "xptar", "yptar", "ytar"]
# Maps internal target key → hcana column name in poly_coeffs JSON
_TARGET_POLY_KEY = {
    "delta": "P_gtr_dp",
    "xptar": "P_gtr_th",
    "yptar": "P_gtr_ph",
    "ytar": "P_react_z",
}


class OpticsEvaluator:
    """
    Evaluate and compare SHMS optics reconstruction models.

    Parameters
    ----------
    model         : trained ResidualMLP.
    scaler_bundle : ScalerBundle used during training (for inverse transforms).
    device        : 'cpu' or 'cuda'.
    """

    def __init__(self, model, scaler_bundle, device: str = "cpu") -> None:
        self.model = model
        self.scaler_bundle = scaler_bundle
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    # ── Primary evaluation ───────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, dataset: Dataset, batch_size: int = 1024) -> dict:
        """
        Return per-target MSE, RMSE, MAE, R² in **physical units**
        (after inverse-transforming predictions and targets).

        Returns
        -------
        dict  Keys: 'delta_mse', 'delta_rmse', 'delta_mae', 'delta_r2', ...
        """
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        preds_all: Dict[str, list] = {k: [] for k in _TARGET_KEYS}
        tgts_all: Dict[str, list] = {k: [] for k in _TARGET_KEYS}

        for batch in loader:
            inputs = batch["inputs"].to(self.device)
            preds = self.model(inputs)
            for k in _TARGET_KEYS:
                if k in preds:
                    preds_all[k].extend(preds[k].squeeze().cpu().numpy().tolist())
                if k in batch["targets"]:
                    tgts_all[k].extend(
                        batch["targets"][k].squeeze().numpy().tolist()
                    )

        results: dict = {}
        for i, k in enumerate(_TARGET_KEYS):
            if not preds_all[k]:
                continue
            p_arr = np.array(preds_all[k], dtype=np.float64)
            t_arr = np.array(tgts_all[k], dtype=np.float64)

            n_targets = len(_TARGET_KEYS)
            p_arr = self._inverse_transform_column(p_arr, i, n_targets, is_target=True)
            t_arr = self._inverse_transform_column(t_arr, i, n_targets, is_target=True)

            residuals = p_arr - t_arr
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((t_arr - t_arr.mean()) ** 2)
            results[f"{k}_mse"] = float(np.mean(residuals ** 2))
            results[f"{k}_rmse"] = float(np.sqrt(np.mean(residuals ** 2)))
            results[f"{k}_mae"] = float(np.mean(np.abs(residuals)))
            results[f"{k}_r2"] = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

        return results

    # ── Polynomial baseline comparison ───────────────────────────────────

    def compare_with_polynomial(
        self, dataset: Dataset, poly_coeffs_json: str
    ):
        """
        Compare NN predictions with the degree-3 polynomial baseline.

        The JSON uses 4 input features (DC focal-plane, no x_tar/p0):
            P_dc_x_fp, P_dc_y_fp, P_dc_xp_fp, P_dc_yp_fp

        Returns
        -------
        pd.DataFrame  Columns: target, nn_rmse, poly_rmse, nn_r2, poly_r2
        """
        import pandas as pd
        from sklearn.preprocessing import PolynomialFeatures, StandardScaler

        with open(poly_coeffs_json) as fh:
            poly_data = json.load(fh)

        degree = poly_data.get("polynomial_degree", 3)
        scaler_mean = np.array(poly_data["scaler_X_mean"])
        scaler_scale = np.array(poly_data["scaler_X_scale"])

        poly_sc = StandardScaler()
        poly_sc.mean_ = scaler_mean
        poly_sc.scale_ = scaler_scale
        poly_sc.var_ = scaler_scale ** 2
        poly_sc.n_features_in_ = len(scaler_mean)
        poly_sc.n_samples_seen_ = 0

        poly_tf = PolynomialFeatures(degree=degree, include_bias=True)

        # Gather raw (unscaled) inputs and targets from the dataset
        # Assumption: dataset X has ≥4 columns (first 4 = DC focal-plane vars)
        loader = DataLoader(dataset, batch_size=2048, shuffle=False)
        X_raw_list: list = []
        tgts_all: Dict[str, list] = {k: [] for k in _TARGET_KEYS}
        nn_preds_all: Dict[str, list] = {k: [] for k in _TARGET_KEYS}

        with torch.no_grad():
            for batch in loader:
                inputs = batch["inputs"].to(self.device)
                preds = self.model(inputs)
                X_raw_list.append(batch["inputs"].numpy())
                for k in _TARGET_KEYS:
                    if k in preds:
                        nn_preds_all[k].extend(
                            preds[k].squeeze().cpu().numpy().tolist()
                        )
                    if k in batch["targets"]:
                        tgts_all[k].extend(
                            batch["targets"][k].squeeze().numpy().tolist()
                        )

        X_all = np.concatenate(X_raw_list, axis=0)
        # Use only first 4 columns for polynomial (DC focal-plane features)
        X_dc4 = X_all[:, :4]

        # Inverse-transform if scaler_bundle is set (X was normalised)
        if self.scaler_bundle is not None:
            try:
                n_feats = self.scaler_bundle.scaler_X.n_features_in_
                X_full_zero = np.zeros((len(X_all), n_feats))
                X_full_zero[:, :4] = X_dc4
                X_full_it = self.scaler_bundle.scaler_X.inverse_transform(X_full_zero)
                X_dc4 = X_full_it[:, :4]
            except Exception:
                pass

        # Normalise with polynomial scaler
        X_dc4_sc = poly_sc.transform(X_dc4)
        X_poly = poly_tf.fit_transform(X_dc4_sc)

        rows = []
        n_t = len(_TARGET_KEYS)
        for k in _TARGET_KEYS:
            poly_key = _TARGET_POLY_KEY.get(k, k)
            if poly_key not in poly_data.get("coefficients", {}):
                continue
            coeffs_info = poly_data["coefficients"][poly_key]
            coef_vec = np.array(coeffs_info["coefficients"])
            intercept = coeffs_info.get("intercept", 0.0)

            # Polynomial raw prediction (predicts directly in physical units)
            poly_pred = X_poly @ coef_vec + intercept

            i = _TARGET_KEYS.index(k)
            t_arr = self._inverse_transform_column(
                np.array(tgts_all[k]), i, n_t, is_target=True
            )
            nn_arr = self._inverse_transform_column(
                np.array(nn_preds_all[k]), i, n_t, is_target=True
            )

            def rmse(a, b):
                return float(np.sqrt(np.mean((a - b) ** 2)))

            def r2(pred, true):
                ss_res = np.sum((pred - true) ** 2)
                ss_tot = np.sum((true - true.mean()) ** 2)
                return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

            rows.append(
                {
                    "target": k,
                    "nn_rmse": rmse(nn_arr, t_arr),
                    "poly_rmse": rmse(poly_pred, t_arr),
                    "nn_r2": r2(nn_arr, t_arr),
                    "poly_r2": r2(poly_pred, t_arr),
                }
            )

        return pd.DataFrame(rows)

    # ── Visualisation ────────────────────────────────────────────────────

    def plot_vertex_resolution(
        self,
        dataset: Dataset,
        target: str = "ytar",
        nbins: int = 100,
        save_path: Optional[str] = None,
    ) -> None:
        """Histogram of residuals for *target* with Gaussian fit overlay."""
        import matplotlib.pyplot as plt

        residuals = self._get_residuals(dataset, target)
        sigma = self.compute_resolution_sigma(dataset, target)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(residuals, bins=nbins, density=True, alpha=0.7, label="Residuals")
        x_range = np.linspace(residuals.min(), residuals.max(), 300)
        gauss = (
            1 / (sigma * np.sqrt(2 * np.pi))
            * np.exp(-0.5 * ((x_range - residuals.mean()) / sigma) ** 2)
        )
        ax.plot(x_range, gauss, "r-", lw=2, label=f"Gaussian fit σ={sigma:.4f}")
        ax.set_xlabel(f"Residual ({target})")
        ax.set_ylabel("Density")
        ax.set_title(f"Vertex resolution — {target}")
        ax.legend()
        fig.tight_layout()
        self._save_figure(fig, save_path)
        plt.show()

    def plot_sieve_reconstruction(
        self,
        dataset: Dataset,
        save_path: Optional[str] = None,
    ) -> None:
        """2-D scatter of reconstructed (yptar, xptar) showing sieve-hole sharpness."""
        import matplotlib.pyplot as plt

        yp = self._get_predictions(dataset, "yptar")
        xp = self._get_predictions(dataset, "xptar")

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(yp, xp, s=0.5, alpha=0.3, rasterized=True)
        ax.set_xlabel("yptar (rad)")
        ax.set_ylabel("xptar (rad)")
        ax.set_title("Sieve-hole reconstruction (NN)")
        fig.tight_layout()
        self._save_figure(fig, save_path)
        plt.show()

    def compute_resolution_sigma(self, dataset: Dataset, target: str) -> float:
        """Fit Gaussian to residual distribution and return σ."""
        from scipy.optimize import curve_fit

        residuals = self._get_residuals(dataset, target)

        def _gauss(x, mu, sigma, amp):
            return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

        counts, edges = np.histogram(residuals, bins=100, density=True)
        centres = 0.5 * (edges[:-1] + edges[1:])
        p0 = [residuals.mean(), residuals.std(), counts.max()]
        try:
            popt, _ = curve_fit(_gauss, centres, counts, p0=p0, maxfev=5000)
            return float(abs(popt[1]))
        except Exception:
            return float(residuals.std())

    # ── Internal helpers ─────────────────────────────────────────────────

    def _inverse_transform_column(
        self, arr: np.ndarray, column_index: int, n_cols: int, is_target: bool = True
    ) -> np.ndarray:
        """
        Inverse-transform a single column from a scaled array.

        Builds a zero-padded full-width array, inverse-transforms it, and
        returns the requested column.

        Parameters
        ----------
        arr          : 1-D array of scaled values.
        column_index : column index within the scaler.
        n_cols       : total number of columns in the scaler.
        is_target    : True → use scaler_Y; False → use scaler_X.
        """
        if self.scaler_bundle is None:
            return arr
        try:
            full = np.zeros((len(arr), n_cols))
            full[:, column_index] = arr
            scaler = (
                self.scaler_bundle.scaler_Y if is_target else self.scaler_bundle.scaler_X
            )
            full_it = scaler.inverse_transform(full)
            return full_it[:, column_index]
        except Exception:
            return arr

    def _save_figure(self, fig, save_path: Optional[str]) -> None:
        """Save *fig* to *save_path* if provided, creating parent dirs."""
        if save_path:
            import os
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=150)

    @torch.no_grad()
    def _get_predictions(self, dataset: Dataset, target: str) -> np.ndarray:
        loader = DataLoader(dataset, batch_size=1024, shuffle=False)
        preds: list = []
        for batch in loader:
            inputs = batch["inputs"].to(self.device)
            out = self.model(inputs)
            if target in out:
                preds.extend(out[target].squeeze().cpu().numpy().tolist())
        return np.array(preds)

    @torch.no_grad()
    def _get_residuals(self, dataset: Dataset, target: str) -> np.ndarray:
        loader = DataLoader(dataset, batch_size=1024, shuffle=False)
        residuals: list = []
        for batch in loader:
            inputs = batch["inputs"].to(self.device)
            out = self.model(inputs)
            if target in out and target in batch["targets"]:
                diff = (
                    out[target].squeeze().cpu().numpy()
                    - batch["targets"][target].squeeze().numpy()
                )
                residuals.extend(diff.tolist())
        return np.array(residuals)
