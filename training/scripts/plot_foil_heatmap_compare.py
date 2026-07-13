#!/usr/bin/env python3
"""3x3 sieve-plane heatmap comparison: GUI v2 vs V3 Baseline vs HCANA truth.

Projection to sieve plane:
  - HCANA:   full SHMS formula (delta-corrected, via project_to_sieve)
  - NN models: pure  s = angle * 253  (no delta / x_tar / y_tar correction)

Plot style follows outputs/cluster_mechanical_overlay/overlay.png:
  - 200-bin 2D histogram, YlOrRd colormap, log-scale color axis
  - Gray chessboard grid lines at mechanical hole design coordinates
  - Black "+" markers at mechanical hole design centers
  - Per-row (per-foil) shared color scale + colorbar
"""
from __future__ import annotations

import os, sys, time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]          # .../SHMS_Calibration_NN
_PROJECT_ROOT = _REPO.parent                          # .../AI_ML R-SIDIS
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_PROJECT_ROOT))

from training.data.stage2_root_dataset import Stage2RootDataset
from training.data.preprocessing import ScalerBundle
from training.models import build_model_from_config
from SHMS_Optics_calibration_tools import project_to_sieve

_TARGET_KEYS = ["delta", "xptar", "yptar", "ytar"]
_MODELS = {
    "GUI v2": str(_REPO / "checkpoints/stage2_gui_v2/best_finetune.pth"),
    "V3 Baseline": str(_REPO / "checkpoints/stage2_transport_fullroot_25521_mainline_centerout_nnrelabel_v3/best_finetune.pth"),
}
_SCALER = str(_REPO / "checkpoints/pretrain_25521_fry_cuda_5d_notebooklike/scaler_bundle.json")
_CSV = str(_PROJECT_ROOT / "stage2_soc_gui_labeled.csv")
_OUT = _REPO / "plots/model_comparison/foil_heatmap_3x3.png"

FOILS = [0, 1, 2]
BINS = 200
LIM = (-20.0, 20.0)
SIEVE_D = 253.0


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    mcfg = ckpt.get("config", {}).get("model", {})
    model = build_model_from_config(mcfg, input_dim=mcfg.get("input_dim", 5))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


@torch.no_grad()
def predict_physical(model, dataset, scaler: ScalerBundle, device: torch.device, batch_size=4096) -> np.ndarray:
    """Run inference and inverse-transform to physical units: [delta, xptar, yptar, ytar]."""
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    blocks = []
    for batch in loader:
        out = model(batch["inputs"].to(device))
        blocks.append(torch.cat([out[k] for k in _TARGET_KEYS], dim=1).cpu().numpy())
    scaled = np.concatenate(blocks, axis=0).astype(np.float64)
    return scaler.inverse_transform_Y(scaled)


def build_hole_design(df, foil_val: int) -> tuple[np.ndarray, np.ndarray]:
    """Median mechanical hole design centers (nominal x_tar=y_tar=delta=0 projection)."""
    sub = df.loc[df["foil_position"] == foil_val, ["hole_row", "hole_col", "weak_hole_xptar_center", "weak_hole_yptar_center"]].dropna()
    if sub.empty:
        return np.array([]), np.array([])
    grouped = sub.groupby(["hole_row", "hole_col"], as_index=False)[["weak_hole_xptar_center", "weak_hole_yptar_center"]].median()
    mech_x = grouped["weak_hole_xptar_center"].to_numpy(dtype=np.float64) * SIEVE_D
    mech_y = grouped["weak_hole_yptar_center"].to_numpy(dtype=np.float64) * SIEVE_D
    return mech_x, mech_y


def draw_chessboard(ax, mech_x: np.ndarray, mech_y: np.ndarray):
    if len(mech_x) == 0:
        return
    for gx in np.unique(np.round(mech_x, 4)):
        ax.axvline(gx, color="#444444", linewidth=0.6, alpha=0.45, zorder=3)
    for gy in np.unique(np.round(mech_y, 4)):
        ax.axhline(gy, color="#444444", linewidth=0.6, alpha=0.45, zorder=3)
    ax.plot(mech_x, mech_y, "+", color="black", markersize=5, markeredgewidth=0.8, zorder=5, label="Mech. hole center")


def plot_panel(ax, x: np.ndarray, y: np.ndarray, mech_x, mech_y, vmax: float, title: str):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    h, xedges, yedges = np.histogram2d(x, y, bins=BINS, range=[LIM, LIM])
    im = ax.imshow(
        h.T, origin="lower", extent=[LIM[0], LIM[1], LIM[0], LIM[1]],
        cmap="YlOrRd", norm=LogNorm(vmin=1, vmax=max(vmax, 1.0)),
        aspect="auto", interpolation="nearest",
    )
    draw_chessboard(ax, mech_x, mech_y)
    ax.set_xlim(*LIM)
    ax.set_ylim(*LIM)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("sieve_x [cm]", fontsize=8)
    ax.grid(False)
    return im


def main():
    os.makedirs(_OUT.parent, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    scaler = ScalerBundle.load(_SCALER)
    ds = Stage2RootDataset(
        data_source=_CSV, scaler_bundle=scaler, tree_name="T",
        feature_schema=["x_fp", "y_fp", "xp_fp", "yp_fp", "fry"],
        cuts={"use_pid": False, "use_quality": False},
        max_events=None,
    )
    df = ds.df.reset_index(drop=True)
    foil = df["foil_position"].to_numpy(dtype=int)

    # --- HCANA truth: full formula projection (reads x_tar/y_tar/delta from df) ---
    hcana_sx, hcana_sy = project_to_sieve(df)

    # --- NN model predictions: pure  s = angle * 253  (no x_tar/y_tar offset) ---
    sieve_by_model: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, ckpt in _MODELS.items():
        print(f"Running inference: {name}")
        model = load_model(ckpt, device)
        t0 = time.time()
        phys = predict_physical(model, ds, scaler, device)
        xptar_p, yptar_p = phys[:, 1], phys[:, 2]
        sx = xptar_p * SIEVE_D
        sy = yptar_p * SIEVE_D
        sieve_by_model[name] = (sx, sy)
        print(f"  {time.time() - t0:.1f}s")

    col_order = ["GUI v2", "V3 Baseline", "HCANA"]
    fig, axes = plt.subplots(3, 3, figsize=(15, 15.5))
    fig.subplots_adjust(hspace=0.55, wspace=0.3, top=0.92, bottom=0.05)

    for row_idx, foil_val in enumerate(FOILS):
        mask = foil == foil_val
        n_events = int(mask.sum())
        mech_x, mech_y = build_hole_design(df, foil_val)
        foil_ytar = df.loc[mask, "foil_ytar_center"].median() if "foil_ytar_center" in df.columns else float("nan")

        # Per-row shared color scale
        row_data = {
            "GUI v2": sieve_by_model["GUI v2"],
            "V3 Baseline": sieve_by_model["V3 Baseline"],
            "HCANA": (hcana_sx, hcana_sy),
        }
        row_vmax = 1.0
        for sx, sy in row_data.values():
            xm, ym = sx[mask], sy[mask]
            fm = np.isfinite(xm) & np.isfinite(ym)
            h, _, _ = np.histogram2d(xm[fm], ym[fm], bins=BINS, range=[LIM, LIM])
            row_vmax = max(row_vmax, h.max())

        im = None
        for col_idx, col_name in enumerate(col_order):
            sx, sy = row_data[col_name]
            im = plot_panel(axes[row_idx, col_idx], sx[mask], sy[mask], mech_x, mech_y, row_vmax, col_name)
            if col_idx == 0:
                axes[row_idx, col_idx].set_ylabel("sieve_y [cm]", fontsize=8)

        # Row banner placed above the middle column, using its axes position
        mid_pos = axes[row_idx, 1].get_position()
        fig.text(
            0.5, mid_pos.y1 + 0.025,
            f"Foil {foil_val}  (ytar={foil_ytar:.0f} cm, n={n_events})",
            ha="center", va="bottom", fontsize=12, fontweight="bold",
        )

        cbar = fig.colorbar(im, ax=axes[row_idx, :].tolist(), fraction=0.02, pad=0.01)
        cbar.set_label("events / bin (log)", fontsize=8)

    fig.suptitle(
        "Sieve-Plane Heatmap: GUI v2 vs V3 Baseline vs HCANA (NN: \u03b8\u00d7253 linear, HCANA: full \u03b4-corrected)",
        fontsize=13, fontweight="bold", y=0.99,
    )
    fig.savefig(_OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {_OUT}")


if __name__ == "__main__":
    main()

