import numpy as np
import pandas as pd
import torch
from types import SimpleNamespace

from training.data.preprocessing import ScalerBundle
from training.data.stage2_root_dataset import Stage2RootDataset
from training.losses import Stage2WeakLabelLoss
from training.scripts.build_stage2_labels_from_25521_fullroot import (
    assign_events_to_mechanical_holes,
    build_observed_mechanical_hole_design_from_event_grid,
    build_cluster_to_mechanical_hole_map,
    build_full_candidate_mechanical_hole_design_from_clusters,
    build_mechanical_hole_design_from_grid,
    compute_equal_hole_total_weights,
    extract_cluster_centers,
    infer_lattice_origin_cm,
    mm_to_target_angle,
)
from training.trainers.stage2_transport import Stage2TransportTrainer


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
    assert isinstance(sample["metadata"]["foil_position"], torch.Tensor)
    assert isinstance(sample["metadata"]["hole_id"], torch.Tensor)


def test_stage2_root_dataset_supports_strong_foil_label():
    df = _make_stage2_df(16)
    df["foil_ytar_center"] = np.where(df["foil_position"] == 0, 10.0, np.where(df["foil_position"] == 1, 0.0, -10.0))
    scaler = _make_scaler_bundle(df.assign(weak_foil_ytar_center=df["foil_ytar_center"]))
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
            "ytar": {"enabled": True, "center_col": "foil_ytar_center", "tol_col": None, "tolerance": 0.0},
        },
        metadata_cols={"foil_position": "foil_position", "hole_id": "hole_id"},
        weight_col="weak_label_weight",
    )
    sample = ds[0]
    assert torch.allclose(sample["tolerances"]["ytar"], torch.tensor([0.0]))


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


def test_stage2_weak_label_loss_hole_separation_prefers_correct_mechanical_hole():
    loss_fn = Stage2WeakLabelLoss(
        use_huber=False,
        hole_separation_weight=1.0,
        hole_separation_temperature=0.1,
        hole_center_bank={
            0: {
                "keys": [[0, 0], [0, 1]],
                "centers": [[0.0, 0.0], [1.0, 0.0]],
            }
        },
    )
    batch = {
        "targets": {
            "xptar": torch.tensor([[0.0], [1.0]]),
            "yptar": torch.tensor([[0.0], [0.0]]),
        },
        "tolerances": {
            "xptar": torch.tensor([[0.0], [0.0]]),
            "yptar": torch.tensor([[0.0], [0.0]]),
        },
        "target_mask": {
            "xptar": torch.tensor([[1.0], [1.0]]),
            "yptar": torch.tensor([[1.0], [1.0]]),
        },
        "metadata": {
            "foil_position": torch.tensor([0, 0]),
            "hole_row": torch.tensor([0, 0]),
            "hole_col": torch.tensor([0, 1]),
        },
        "weight": torch.tensor([1.0, 1.0]),
    }
    preds_good = {
        "xptar": torch.tensor([[0.02], [0.98]]),
        "yptar": torch.tensor([[0.00], [0.01]]),
    }
    preds_bad = {
        "xptar": torch.tensor([[0.95], [0.05]]),
        "yptar": torch.tensor([[0.00], [0.01]]),
    }

    loss_good = loss_fn(preds_good, batch)
    loss_bad = loss_fn(preds_bad, batch)
    assert loss_good.item() < loss_bad.item()


def test_stage2_weak_label_loss_sieve_plane_prefers_smaller_radial_sieve_error():
    loss_fn = Stage2WeakLabelLoss(
        use_huber=False,
        target_weights={"delta": 0.0, "xptar": 0.0, "yptar": 0.0, "ytar": 0.0},
        sieve_plane_weight=1.0,
        sieve_plane_huber_delta_cm=0.3,
        sieve_distance_cm=253.0,
        target_scales={"xptar": 1.0 / 253.0, "yptar": 1.0 / 253.0},
    )
    batch = {
        "targets": {
            "delta": torch.tensor([[0.0], [0.0]]),
            "xptar": torch.tensor([[0.0], [0.0]]),
            "yptar": torch.tensor([[0.0], [0.0]]),
            "ytar": torch.tensor([[0.0], [0.0]]),
        },
        "tolerances": {
            "delta": torch.tensor([[0.0], [0.0]]),
            "xptar": torch.tensor([[0.0], [0.0]]),
            "yptar": torch.tensor([[0.0], [0.0]]),
            "ytar": torch.tensor([[0.0], [0.0]]),
        },
        "target_mask": {
            "delta": torch.tensor([[0.0], [0.0]]),
            "xptar": torch.tensor([[1.0], [1.0]]),
            "yptar": torch.tensor([[1.0], [1.0]]),
            "ytar": torch.tensor([[0.0], [0.0]]),
        },
        "weight": torch.tensor([1.0, 1.0]),
    }
    preds_good = {
        "delta": torch.tensor([[0.0], [0.0]]),
        "xptar": torch.tensor([[0.10 / 253.0], [0.00 / 253.0]]),
        "yptar": torch.tensor([[0.00 / 253.0], [0.10 / 253.0]]),
        "ytar": torch.tensor([[0.0], [0.0]]),
    }
    preds_bad = {
        "delta": torch.tensor([[0.0], [0.0]]),
        "xptar": torch.tensor([[0.40 / 253.0], [0.00 / 253.0]]),
        "yptar": torch.tensor([[0.00 / 253.0], [0.40 / 253.0]]),
        "ytar": torch.tensor([[0.0], [0.0]]),
    }

    loss_good = loss_fn(preds_good, batch)
    loss_bad = loss_fn(preds_bad, batch)
    assert loss_good.item() < loss_bad.item()


def test_build_mechanical_hole_design_from_grid_uses_spacing_mm():
    grid_index = pd.DataFrame(
        {
            "foil_position": [0, 0, 1],
            "row": [-1, 0, 1],
            "col": [2, 0, -1],
        }
    )
    args = SimpleNamespace(
        hole_x_spacing_mm=25.0,
        hole_y_spacing_mm=16.4,
        hole_tolerance_mm=3.0,
        sieve_distance_cm=253.0,
        hole_origin_xptar=0.0,
        hole_origin_yptar=0.0,
    )

    design = build_mechanical_hole_design_from_grid(grid_index, args)

    x_step = mm_to_target_angle(25.0, 253.0)
    y_step = mm_to_target_angle(16.4, 253.0)
    tol = mm_to_target_angle(3.0, 253.0)

    row0 = design.iloc[0]
    assert row0["foil_position"] == 0
    assert row0["hole_row"] == -1
    assert row0["hole_col"] == 2
    assert np.isclose(row0["weak_hole_xptar_center"], 2.0 * x_step)
    assert np.isclose(row0["weak_hole_yptar_center"], -1.0 * y_step)
    assert np.isclose(row0["weak_hole_xptar_tol"], tol)
    assert np.isclose(row0["weak_hole_yptar_tol"], tol)


def test_infer_lattice_origin_cm_recovers_small_offset():
    values = pd.Series([-5.18, -2.69, -0.19, 2.34, 4.82])
    origin = infer_lattice_origin_cm(values, spacing_cm=2.5, initial_origin_cm=0.0)
    assert np.isclose(origin, -0.18, atol=0.08)


def test_build_cluster_to_mechanical_hole_map_matches_nearest_spacing_grid():
    df_clusters = pd.DataFrame(
        {
            "cluster": [10, 11, 12],
            "cluster_center_x": [-5.22, -0.12, 2.61],
            "cluster_center_y": [-1.58, 0.03, 1.71],
            "foil_position": [0, 0, 0],
        }
    )
    clustering_results = {
        0: {
            "df": df_clusters.assign(is_noise=False),
            "params": {},
            "n_clusters": 3,
            "n_clusters_before_post": 3,
            "postprocess_report": {},
        }
    }
    args = SimpleNamespace(
        hole_x_spacing_mm=25.0,
        hole_y_spacing_mm=16.4,
        hole_tolerance_mm=3.0,
        sieve_distance_cm=253.0,
        hole_origin_xptar=0.0,
        hole_origin_yptar=0.0,
        cluster_hole_assignment_mode="nearest",
        cluster_hole_occupancy_penalty_cm=0.0,
    )

    design, meta = build_full_candidate_mechanical_hole_design_from_clusters(clustering_results, args)
    assignments, match_summary = build_cluster_to_mechanical_hole_map(clustering_results, design, args)

    assert not assignments.empty
    assert match_summary["per_foil"]["0"]["max_match_distance_cm"] < 0.4
    matched = assignments.sort_values("cluster").reset_index(drop=True)
    assert matched.loc[0, "hole_col"] == -2
    assert matched.loc[1, "hole_col"] == 0
    assert matched.loc[2, "hole_col"] == 1
    assert matched.loc[0, "hole_row"] == -1
    assert matched.loc[1, "hole_row"] == 0
    assert matched.loc[2, "hole_row"] == 1


def test_build_cluster_to_mechanical_hole_map_penalizes_reusing_occupied_hole():
    df_clusters = pd.DataFrame(
        {
            "cluster": [20, 21],
            "cluster_center_x": [0.10, 1.15],
            "cluster_center_y": [0.00, 0.00],
            "foil_position": [0, 0],
        }
    )
    clustering_results = {
        0: {
            "df": df_clusters.assign(is_noise=False),
            "params": {},
            "n_clusters": 2,
            "n_clusters_before_post": 2,
            "postprocess_report": {},
        }
    }
    design = pd.DataFrame(
        {
            "foil_position": [0, 0],
            "hole_row": [0, 0],
            "hole_col": [0, 1],
            "candidate_sieve_x_cm": [0.0, 2.5],
            "candidate_sieve_y_cm": [0.0, 0.0],
            "weak_hole_xptar_center": [0.0 / 253.0, 2.5 / 253.0],
            "weak_hole_yptar_center": [0.0, 0.0],
            "weak_hole_xptar_tol": [3.0 / 253.0 / 10.0, 3.0 / 253.0 / 10.0],
            "weak_hole_yptar_tol": [3.0 / 253.0 / 10.0, 3.0 / 253.0 / 10.0],
        }
    )
    args = SimpleNamespace(
        hole_x_spacing_mm=25.0,
        hole_y_spacing_mm=16.4,
        hole_tolerance_mm=3.0,
        sieve_distance_cm=253.0,
        hole_origin_xptar=0.0,
        hole_origin_yptar=0.0,
        cluster_hole_assignment_mode="center_out_penalized",
        cluster_hole_occupancy_penalty_cm=0.35,
    )

    assignments, match_summary = build_cluster_to_mechanical_hole_map(clustering_results, design, args)

    matched = assignments.sort_values("cluster").reset_index(drop=True)
    assert matched.loc[0, "hole_col"] == 0
    assert matched.loc[1, "hole_col"] == 1
    assert matched.loc[1, "hole_occupancy_before_assignment"] == 0
    assert matched.loc[1, "match_distance_cm"] > matched.loc[1, "nearest_match_distance_cm"]
    assert match_summary["duplicate_mechanical_holes"] == 0
    assert match_summary["per_foil"]["0"]["reassigned_from_nearest_count"] == 1


def test_direct_grid_builds_observed_design_and_filters_sparse_holes():
    rng = np.random.default_rng(7)
    dense_a = pd.DataFrame(
        {
            "foil_position": 0,
            "sieve_x": rng.normal(-2.5, 0.10, 20),
            "sieve_y": rng.normal(-1.64, 0.08, 20),
        }
    )
    dense_b = pd.DataFrame(
        {
            "foil_position": 0,
            "sieve_x": rng.normal(0.0, 0.10, 18),
            "sieve_y": rng.normal(0.0, 0.08, 18),
        }
    )
    sparse = pd.DataFrame(
        {
            "foil_position": 0,
            "sieve_x": rng.normal(5.0, 0.05, 3),
            "sieve_y": rng.normal(4.92, 0.05, 3),
        }
    )
    df = pd.concat([dense_a, dense_b, sparse], ignore_index=True)
    args = SimpleNamespace(
        hole_x_spacing_mm=25.0,
        hole_y_spacing_mm=16.4,
        hole_tolerance_mm=3.0,
        sieve_distance_cm=253.0,
        hole_origin_xptar=0.0,
        hole_origin_yptar=0.0,
        direct_grid_min_hole_population=5,
    )

    design, meta = build_observed_mechanical_hole_design_from_event_grid(df, args)
    assigned, filtered_design, summary = assign_events_to_mechanical_holes(df, design, args)

    kept_holes = set(zip(filtered_design["hole_row"], filtered_design["hole_col"]))
    assert kept_holes == {(-1, -1), (0, 0)}
    assert meta["min_population"] == 5
    assert len(filtered_design) == 2
    assert len(assigned) == 38
    assert assigned["cluster"].nunique() == 2
    assert summary["per_foil"]["0"]["n_holes"] == 2


def test_direct_grid_caps_holes_per_foil_by_population_then_match_quality():
    rows = []
    hole_specs = [
        (-1, -1, -2.5, -1.64, 20),
        (0, 0, 0.0, 0.0, 18),
        (1, 1, 2.5, 1.64, 12),
    ]
    for hole_row, hole_col, x0, y0, n in hole_specs:
        for idx in range(n):
            rows.append(
                {
                    "foil_position": 0,
                    "sieve_x": x0 + 0.01 * idx,
                    "sieve_y": y0 + 0.005 * idx,
                }
            )
    df = pd.DataFrame(rows)
    args = SimpleNamespace(
        hole_x_spacing_mm=25.0,
        hole_y_spacing_mm=16.4,
        hole_tolerance_mm=3.0,
        sieve_distance_cm=253.0,
        hole_origin_xptar=0.0,
        hole_origin_yptar=0.0,
        direct_grid_min_hole_population=5,
        direct_grid_max_holes_per_foil=2,
        direct_grid_max_match_distance_cm=None,
    )

    design, _ = build_observed_mechanical_hole_design_from_event_grid(df, args)
    assigned, filtered_design, summary = assign_events_to_mechanical_holes(df, design, args)

    kept_holes = set(zip(filtered_design["hole_row"], filtered_design["hole_col"]))
    assert kept_holes == {(-1, -1), (0, 0)}
    assert assigned["cluster"].nunique() == 2
    assert summary["max_holes_per_foil"] == 2
    assert summary["per_foil_hole_cap"]["0"]["holes_before_cap"] == 3
    assert summary["per_foil_hole_cap"]["0"]["holes_after_cap"] == 2


def test_equal_hole_total_weights_make_each_hole_sum_equal():
    populations = pd.Series(([20.0] * 20) + ([5.0] * 5), dtype=float)
    weights = compute_equal_hole_total_weights(populations)
    assert np.allclose(weights[:20], np.full(20, 0.05))
    assert np.allclose(weights[20:], np.full(5, 0.2))

    hole_a_total = float(weights[:20].sum())
    hole_b_total = float(weights[20:].sum())
    assert np.isclose(hole_a_total, 1.0)
    assert np.isclose(hole_a_total, hole_b_total)


def test_stage2_weak_label_loss_hole_separation_accepts_negative_hole_indices():
    loss_fn = Stage2WeakLabelLoss(
        use_huber=False,
        hole_separation_weight=1.0,
        hole_separation_temperature=0.1,
        hole_center_bank={
            0: {
                "keys": [[-1, -1], [0, 0]],
                "centers": [[0.0, 0.0], [1.0, 0.0]],
            }
        },
    )
    batch = {
        "targets": {
            "xptar": torch.tensor([[0.0], [1.0]]),
            "yptar": torch.tensor([[0.0], [0.0]]),
        },
        "tolerances": {
            "xptar": torch.tensor([[0.0], [0.0]]),
            "yptar": torch.tensor([[0.0], [0.0]]),
        },
        "target_mask": {
            "xptar": torch.tensor([[1.0], [1.0]]),
            "yptar": torch.tensor([[1.0], [1.0]]),
        },
        "metadata": {
            "foil_position": torch.tensor([0, 0]),
            "hole_row": torch.tensor([-1, 0]),
            "hole_col": torch.tensor([-1, 0]),
        },
        "weight": torch.tensor([1.0, 1.0]),
    }
    preds_good = {
        "xptar": torch.tensor([[0.02], [0.98]]),
        "yptar": torch.tensor([[0.00], [0.01]]),
    }
    preds_bad = {
        "xptar": torch.tensor([[0.95], [0.05]]),
        "yptar": torch.tensor([[0.00], [0.01]]),
    }

    loss_good = loss_fn(preds_good, batch)
    loss_bad = loss_fn(preds_bad, batch)
    assert loss_good.item() < loss_bad.item()


def test_stage2_trainer_loader_kwargs_enable_prefetch_only_with_workers():
    kwargs = Stage2TransportTrainer._build_loader_kwargs(
        batch_size=2048,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    assert kwargs["num_workers"] == 4
    assert kwargs["persistent_workers"] is True
    assert kwargs["prefetch_factor"] == 4

    kwargs_no_workers = Stage2TransportTrainer._build_loader_kwargs(
        batch_size=2048,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=True,
        prefetch_factor=4,
    )
    assert kwargs_no_workers["num_workers"] == 0
    assert "persistent_workers" not in kwargs_no_workers
    assert "prefetch_factor" not in kwargs_no_workers
