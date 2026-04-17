from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.style.use('seaborn-v0_8-whitegrid')

ROOT = Path(r"c:\Users\Lanto\Desktop\AI_ML R-SIDIS\SHMS_Calibration_NN")
OUT = ROOT / "training" / "Logbook"
OLD_METRICS = ROOT / "checkpoints" / "pretrain_25521_fry_cuda" / "eval_transport_latest" / "metrics_payload.json"
NEW_METRICS = ROOT / "checkpoints" / "pretrain_25521_fry_cuda_5d_notebooklike" / "eval_transport_final" / "metrics_payload.json"
OLD_HISTORY = ROOT / "checkpoints" / "pretrain_25521_fry_cuda" / "training_history_pretrain.json"

EXPERIMENT = {
    "best_val_loss": 0.002586,
    "nn_rmse": [0.278672, 0.002024, 0.001486, 1.653929],
    "root_rmse": [0.316723, 0.0023, 0.001544, 2.006163],
    "best_epoch": 55,
    "label": "Notebook experiment",
}
TARGETS = ["delta", "xptar", "yptar", "ytar"]


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def add_box(ax, xy, width, height, text, fc="#EAF2FF", ec="#4C72B0", fontsize=11, radius=0.03):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        linewidth=1.5,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=fontsize)
    return patch


def add_arrow(ax, p1, p2, color="#4C72B0"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=15, linewidth=1.5, color=color))


def plot_rush_strategy(old_payload, new_payload):
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_box(ax, (0.05, 0.72), 0.22, 0.16, "Goal\nRecover experiment-like performance\nunder schedule pressure", fc="#FFF4E6", ec="#DD8452")
    add_box(ax, (0.37, 0.72), 0.24, 0.16, "Keep 5D input only\n[x_fp, y_fp, xp_fp, yp_fp, fry]", fc="#EAF7EA", ec="#55A868")
    add_box(ax, (0.72, 0.72), 0.22, 0.16, "Remove physics penalty\nand restore notebook-style\nweighting + hyperparameters", fc="#F4ECFF", ec="#8172B3")

    add_box(ax, (0.07, 0.40), 0.22, 0.16, "Old formal pretraining\nphysics-informed + 5D\nbest val loss = %.4f" % old_payload["checkpoint_val_loss"], fc="#FCEBEC", ec="#C44E52")
    add_box(ax, (0.39, 0.40), 0.22, 0.16, "New sprint training\n5D + notebook-like config\nbest val loss = %.4f" % new_payload["checkpoint_val_loss"], fc="#EAF2FF", ec="#4C72B0")
    add_box(ax, (0.71, 0.40), 0.22, 0.16, "Immediate evaluation\ncompare vs ROOT reco\nand previous attempts", fc="#EAF7EA", ec="#55A868")

    add_box(ax, (0.24, 0.10), 0.52, 0.16,
            "Outcome: delta/xptar/yptar/ytar all improved over old formal pretraining;\n"
            "xptar and yptar now exceed the original notebook experiment on current 25521 MC.",
            fc="#FFF9DB", ec="#CCB974", fontsize=12)

    add_arrow(ax, (0.27, 0.80), (0.37, 0.80))
    add_arrow(ax, (0.61, 0.80), (0.72, 0.80))
    add_arrow(ax, (0.16, 0.72), (0.18, 0.56))
    add_arrow(ax, (0.50, 0.72), (0.50, 0.56))
    add_arrow(ax, (0.83, 0.72), (0.82, 0.56))
    add_arrow(ax, (0.29, 0.48), (0.39, 0.48))
    add_arrow(ax, (0.61, 0.48), (0.71, 0.48))
    add_arrow(ax, (0.50, 0.40), (0.50, 0.26))

    ax.set_title("Rush-mode training strategy and current overall idea", fontsize=16, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "01_rush_training_strategy.png", dpi=180)
    plt.close(fig)


def plot_network_architecture():
    fig, ax = plt.subplots(figsize=(15, 8.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_box(ax, (0.04, 0.40), 0.14, 0.18, "5D Input\n[x_fp, y_fp, xp_fp, yp_fp, fry]", fc="#FFF4E6", ec="#DD8452")

    add_box(ax, (0.24, 0.22), 0.18, 0.18, "Linear path\nLinear(5 → 4)\nleast-squares init", fc="#EAF7EA", ec="#55A868")
    add_box(ax, (0.24, 0.62), 0.18, 0.16, "Residual input\nLinear(5 → 192) + SiLU", fc="#EAF2FF", ec="#4C72B0")
    add_box(ax, (0.48, 0.62), 0.18, 0.16, "4 × ResidualBlock\n[Linear → SiLU →\nDropout(0.1) → Linear + skip]", fc="#EAF2FF", ec="#4C72B0")
    add_box(ax, (0.72, 0.62), 0.18, 0.16, "Correction head\nLinear(192 → 64) → SiLU\n→ Linear(64 → 4)", fc="#F4ECFF", ec="#8172B3")

    add_box(ax, (0.72, 0.28), 0.18, 0.15, "+  Element-wise sum\nlinear output + correction", fc="#FFF9DB", ec="#CCB974")

    output_x = [0.66, 0.75, 0.84, 0.93]
    labels = ["delta", "xptar", "yptar", "ytar"]
    for x, label in zip(output_x, labels):
        add_box(ax, (x - 0.04, 0.11), 0.08, 0.075, label, fc="#F7F7F7", ec="#8C8C8C", fontsize=11, radius=0.02)

    ax.text(0.13, 0.63, "branch split", fontsize=10, color="#555555")
    ax.text(0.33, 0.16, "stable first-order transport prior", ha="center", fontsize=10, color="#2F6B3B")
    ax.text(0.81, 0.55, "learn high-order nonlinear correction", ha="center", fontsize=10, color="#5C4A90")

    add_arrow(ax, (0.18, 0.49), (0.24, 0.31))
    add_arrow(ax, (0.18, 0.49), (0.24, 0.70))
    add_arrow(ax, (0.42, 0.70), (0.48, 0.70))
    add_arrow(ax, (0.66, 0.70), (0.72, 0.70))
    add_arrow(ax, (0.42, 0.31), (0.72, 0.355))
    add_arrow(ax, (0.81, 0.62), (0.81, 0.43))
    add_arrow(ax, (0.81, 0.28), (0.81, 0.20))

    for x in output_x:
        add_arrow(ax, (0.81, 0.20), (x, 0.20))
        add_arrow(ax, (x, 0.20), (x, 0.185))

    ax.plot([0.66, 0.93], [0.20, 0.20], color="#4C72B0", linewidth=1.5, alpha=0.8)

    ax.text(0.50, 0.02,
            "Training behavior: first 10 epochs freeze correction branch to stabilise linear transport;\n"
            "after warmup, unfreeze correction branch and optimise the full transport-residual decomposition.",
            ha="center", va="center", fontsize=11)
    ax.set_title("Detailed neural network structure (current 5D transport-residual model)", fontsize=16, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "02_network_architecture.png", dpi=180)
    plt.close(fig)


def plot_pretrain_finetune_comparison():
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis("off")

    rows = [
        ["Data source", "25521 MC ROOT / large-statistics synthetic supervision", "Sieve / foil / labeled experimental data"],
        ["Input philosophy", "Same focal-plane representation; current sprint version keeps 5D with fry", "Same focal-plane core; may use run-specific labeled features and p0 from runtime"],
        ["Objective", "Learn robust optics prior and nonlinear transport correction", "Adapt prior to real detector/domain effects and labeled calibration targets"],
        ["Loss style", "Pure data loss in current sprint (no physics penalty) + target weighting", "Physics weight can be reintroduced gently during supervised adaptation"],
        ["Shared points", "Same transport-residual backbone, same scaler discipline, same output targets", "Same output targets delta/xptar/yptar/ytar and checkpoint handoff"],
        ["Main difference", "Prior learning on synthetic truth", "Domain adaptation to experiment and calibration-specific labels"],
    ]

    table = ax.table(
        cellText=rows,
        colLabels=["Aspect", "Pretraining", "Post-training / Fine-tuning"],
        cellLoc="left",
        colLoc="center",
        loc="center",
        colColours=["#DDEAF7", "#EAF7EA", "#FCEBEC"],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.0)
    ax.set_title("Pretraining vs post-training: same backbone, different supervision role", fontsize=16, weight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(OUT / "03_pretrain_vs_finetune_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_posttraining_global_adaptation():
    fig, ax = plt.subplots(figsize=(14.5, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_box(
        ax,
        (0.04, 0.67),
        0.20,
        0.18,
        "Limited labeled data\nfoil positions + sieve-hole IDs\nsmall but high-value supervision",
        fc="#FFF4E6",
        ec="#DD8452",
    )
    add_box(
        ax,
        (0.29, 0.67),
        0.22,
        0.18,
        "Reuse pretraining prior\nload best_pretrain checkpoint\nreuse Stage-1 scaler bundle",
        fc="#EAF7EA",
        ec="#55A868",
    )
    add_box(
        ax,
        (0.56, 0.67),
        0.18,
        0.18,
        "Head-only adaptation first\nfreeze backbone\nsmall LR = 1e-5",
        fc="#EAF2FF",
        ec="#4C72B0",
    )
    add_box(
        ax,
        (0.78, 0.67),
        0.17,
        0.18,
        "Optional global unfreeze\nafter epoch 100\nvery small LR = 1e-6",
        fc="#F4ECFF",
        ec="#8172B3",
    )

    add_box(
        ax,
        (0.10, 0.38),
        0.20,
        0.14,
        "Why this can still tune globally?\nEach labeled event constrains the same shared network\nparameters used for all kinematics.",
        fc="#F7F7F7",
        ec="#8C8C8C",
        fontsize=11,
    )
    add_box(
        ax,
        (0.40, 0.36),
        0.24,
        0.18,
        "Not memorising local holes\nThe model sees focal-plane coordinates and learns\na smooth correction function over the full phase space.",
        fc="#EAF2FF",
        ec="#4C72B0",
        fontsize=11,
    )
    add_box(
        ax,
        (0.71, 0.36),
        0.22,
        0.18,
        "Regularised global update\nphysics-aware loss can remain small but present\nReduceLROnPlateau + early stopping prevent overfit.",
        fc="#FFF9DB",
        ec="#CCB974",
        fontsize=11,
    )

    add_box(
        ax,
        (0.22, 0.09),
        0.56,
        0.16,
        "Core message: limited foil / sievehole labels do not only fix a few local points;\n"
        "they calibrate the shared transport-residual map, so the post-training correction is propagated\n"
        "to the whole acceptance as a globally consistent adjustment.",
        fc="#FCEBEC",
        ec="#C44E52",
        fontsize=12,
    )

    add_arrow(ax, (0.24, 0.76), (0.29, 0.76))
    add_arrow(ax, (0.51, 0.76), (0.56, 0.76))
    add_arrow(ax, (0.74, 0.76), (0.78, 0.76))

    add_arrow(ax, (0.15, 0.67), (0.20, 0.52))
    add_arrow(ax, (0.40, 0.67), (0.49, 0.54))
    add_arrow(ax, (0.65, 0.67), (0.79, 0.54))

    add_arrow(ax, (0.30, 0.45), (0.40, 0.45))
    add_arrow(ax, (0.64, 0.45), (0.71, 0.45))
    add_arrow(ax, (0.52, 0.36), (0.52, 0.25))

    ax.set_title(
        "PostTraining idea: use sparse foil/sievehole supervision to drive global fine-tuning",
        fontsize=16,
        weight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUT / "03b_posttraining_global_adaptation.png", dpi=180)
    plt.close(fig)


def plot_attempts_comparison(old_payload, new_payload):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    x = np.arange(len(TARGETS))
    width = 0.2
    notebook = np.array(EXPERIMENT["nn_rmse"])
    old_formal = np.array(old_payload["metrics"]["nn_rmse"])
    new_formal = np.array(new_payload["metrics"]["nn_rmse"])
    root_current = np.array(new_payload["metrics"]["root_rmse"])

    axes[0].bar(x - 1.5 * width, notebook, width, label="Notebook experiment")
    axes[0].bar(x - 0.5 * width, old_formal, width, label="Old formal 5D")
    axes[0].bar(x + 0.5 * width, new_formal, width, label="New 5D notebook-like")
    axes[0].bar(x + 1.5 * width, root_current, width, label="ROOT reco (25521)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(TARGETS)
    axes[0].set_title("RMSE comparison across attempts")
    axes[0].set_ylabel("RMSE")
    axes[0].legend(fontsize=9)

    groups = ["old formal", "new 5D notebook-like"]
    best_vals = [old_payload["checkpoint_val_loss"], new_payload["checkpoint_val_loss"]]
    best_epochs = [old_payload["checkpoint_epoch"], new_payload["checkpoint_epoch"]]
    axes[1].bar(groups, best_vals, color=["#C44E52", "#4C72B0"])
    for i, (val, epoch) in enumerate(zip(best_vals, best_epochs)):
        axes[1].text(i, val + max(best_vals) * 0.02, f"epoch {epoch}\n{val:.4f}", ha="center", va="bottom", fontsize=10)
    axes[1].axhline(EXPERIMENT["best_val_loss"], color="#55A868", linestyle="--", label="Notebook experiment best val loss")
    axes[1].set_title("Best validation loss comparison")
    axes[1].set_ylabel("best val loss")
    axes[1].legend(fontsize=9)

    fig.suptitle("Current results versus previous attempts", fontsize=16, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "04_attempts_comparison.png", dpi=180)
    plt.close(fig)


def plot_old_new_training_curve(old_history):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(old_history["val_loss"], label="Old formal 5D val loss", color="#C44E52", linewidth=2)
    ax.plot(old_history["train_loss"], label="Old formal 5D train loss", color="#C44E52", linestyle="--", alpha=0.7)
    ax.axhline(0.01013091469389598, color="#4C72B0", linestyle="--", label="New 5D notebook-like final best val")
    ax.axhline(EXPERIMENT["best_val_loss"], color="#55A868", linestyle=":", label="Notebook experiment best val")
    ax.set_title("Training progress context: old formal curve vs new best milestone")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "05_training_progress_context.png", dpi=180)
    plt.close(fig)


def plot_summary_radar(old_payload, new_payload):
    labels = TARGETS
    old_vals = np.array(old_payload["metrics"]["rmse_improvement_vs_linear_pct"], dtype=float)
    new_vals = np.array(new_payload["metrics"]["rmse_improvement_vs_linear_pct"], dtype=float)
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])
    old_vals = np.concatenate([old_vals, [old_vals[0]]])
    new_vals = np.concatenate([new_vals, [new_vals[0]]])

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, old_vals, label="Old formal 5D", linewidth=2, color="#C44E52")
    ax.fill(angles, old_vals, alpha=0.15, color="#C44E52")
    ax.plot(angles, new_vals, label="New 5D notebook-like", linewidth=2, color="#4C72B0")
    ax.fill(angles, new_vals, alpha=0.15, color="#4C72B0")
    ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels)
    ax.set_title("Improvement vs internal linear path (%)", pad=20, fontsize=15, weight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.15))
    fig.tight_layout()
    fig.savefig(OUT / "06_improvement_radar.png", dpi=180)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    old_payload = load_json(OLD_METRICS)
    new_payload = load_json(NEW_METRICS)
    old_history = load_json(OLD_HISTORY)

    plot_rush_strategy(old_payload, new_payload)
    plot_network_architecture()
    plot_pretrain_finetune_comparison()
    plot_posttraining_global_adaptation()
    plot_attempts_comparison(old_payload, new_payload)
    plot_old_new_training_curve(old_history)
    plot_summary_radar(old_payload, new_payload)

    summary = {
        "old_formal_best_val": old_payload["checkpoint_val_loss"],
        "new_notebooklike_best_val": new_payload["checkpoint_val_loss"],
        "notebook_experiment_best_val": EXPERIMENT["best_val_loss"],
        "old_formal_rmse": old_payload["metrics"]["nn_rmse"],
        "new_notebooklike_rmse": new_payload["metrics"]["nn_rmse"],
        "notebook_experiment_rmse": EXPERIMENT["nn_rmse"],
    }
    with open(OUT / "logbook_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
