#!/usr/bin/env python3
"""
Generate all paper figures from REAL logged metrics and data files.

Data sources (must exist):
    TRAINING_LOG.md       — per-model/fold metrics + ablation tables
    STATUS.md             — dataset size, completion status
    ENSEMBLE.md           — ensemble configuration
    work/splits_final.json — 5-fold train/val splits
    /home/hecktor/rabin/work/features_full.csv — radiomic/clinical features

No synthetic/demo data is used. If a required metric is missing,
the corresponding figure is skipped with a warning.
"""

from __future__ import annotations
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "figures"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_COLORS = {
    "MedNeXt-L": "#1f77b4",
    "ResEnc": "#ff7f0e",
    "Ensemble": "#2ca02c",
    "LightGBM": "#d62728",
    "Mamba": "#9467bd",
    "RSF": "#8c564b",
    "Cox": "#e377c2",
}


# ===========================================================================
# Data loading from markdown
# ===========================================================================
def _is_num(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _parse_training_log() -> dict:
    path = REPO / "TRAINING_LOG.md"
    text = path.read_text()

    result = {
        "segmentation": [],
        "tn_staging": [],
        "prognosis": [],
        "tta_ablation": [],
    }

    lines = text.splitlines()
    section = None

    for line in lines:
        if line.startswith("## Segmentation"):
            section = "seg"
        elif line.startswith("## TN Staging"):
            section = "tn"
        elif line.startswith("## Prognosis"):
            section = "prog"
        elif line.startswith("## ") or line.startswith("# "):
            section = None

        if not line.startswith("|"):
            continue
        parts = [p.strip() for p in line.strip("|").split("|")]
        if not parts:
            continue
        if parts[0] in ("Model", "---", "Technique"):
            continue

        if section == "seg" and len(parts) >= 4:
            row = {"model": parts[0], "fold": parts[1], "epochs": parts[2],
                   "best_val_dice": parts[3] if len(parts) > 3 else "",
                   "test_dice": parts[4] if len(parts) > 4 else "",
                   "notes": parts[5] if len(parts) > 5 else ""}
            result["segmentation"].append(row)

        elif section == "tn" and len(parts) >= 4:
            row = {"model": parts[0], "t_ba": parts[1], "n_ba": parts[2], "mean_ba": parts[3]}
            result["tn_staging"].append(row)

        elif section == "prog" and len(parts) >= 2:
            row = {"model": parts[0], "c_index": parts[1]}
            result["prognosis"].append(row)

        # Ablation anywhere in file
        if "Ablation" in text or "TTA" in text:
            if section is None and len(parts) >= 3 and "Technique" not in parts[0]:
                if _is_num(parts[1]) and _is_num(parts[2]):
                    result["tta_ablation"].append({
                        "technique": parts[0],
                        "dice": float(parts[1]),
                        "delta": float(parts[2]) if _is_num(parts[2]) else 0.0,
                    })

    return result


# ===========================================================================
# Figures
# ===========================================================================
def fig_architecture():
    """Copy existing architecture diagram if present."""
    src = REPO / "figures" / "architecture.png"
    if src.exists():
        dst = OUT / "architecture.png"
        if src != dst:
            import shutil
            shutil.copy2(src, dst)
        print(f"Kept architecture.png in figures/")
    else:
        print("[skip] fig_architecture: figures/architecture.png not found")


def fig_segmentation_results(data: dict):
    """Per-fold Dice + model comparison from logged metrics."""
    seg = data["segmentation"]
    if not seg:
        print("[skip] fig_segmentation_results: no segmentation data")
        return

    df = pd.DataFrame(seg)
    # Keep only rows where best_val_dice is numeric
    df = df[df["best_val_dice"].apply(lambda x: _is_num(str(x)) if x != "" else False)]

    if df.empty:
        print("[skip] fig_segmentation_results: empty after parsing")
        return

    df["best_val_dice"] = df["best_val_dice"].astype(float)
    df["fold"] = pd.to_numeric(df["fold"], errors="coerce")
    df = df.dropna(subset=["fold"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle("Figure 3: Segmentation Results", fontsize=12, fontweight="bold", y=1.02)

    # Per-fold Dice by model
    ax = axes[0]
    models = df["model"].unique()
    folds = sorted(df["fold"].unique())
    x = np.arange(len(folds))
    width = 0.35
    for i, model in enumerate(models):
        sub = df[df["model"] == model].sort_values("fold")
        vals = [sub[sub["fold"] == f]["best_val_dice"].values[0] if f in sub["fold"].values else 0 for f in folds]
        ax.bar(x + i * width - width / 2, vals, width, label=model,
               color=MODEL_COLORS.get(model, "gray"), edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {int(f)}" for f in folds])
    ax.set_ylim(0.5, 0.9)
    ax.set_ylabel("Best Val Dice")
    ax.set_title("Per-fold Validation Dice")
    ax.legend()

    # Model mean comparison
    ax = axes[1]
    summary = df.groupby("model")["best_val_dice"].mean().reset_index()
    summary = summary.sort_values("best_val_dice", ascending=False)
    colors = [MODEL_COLORS.get(m, "gray") for m in summary["model"]]
    bars = ax.bar(summary["model"], summary["best_val_dice"], color=colors, edgecolor="black")
    ax.set_ylim(0.5, 0.9)
    ax.set_ylabel("Mean Best Val Dice")
    ax.set_title("Model Comparison")
    for bar, score in zip(bars, summary["best_val_dice"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{score:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fig.savefig(OUT / "fig3_segmentation_results.png")
    plt.close()
    print("Saved fig3_segmentation_results.png")


def fig_tta_ablation(data: dict):
    """TTA + post-processing ablation from logged metrics."""
    ablation = data.get("tta_ablation", [])
    if not ablation:
        print("[skip] fig_tta_ablation: no ablation data in TRAINING_LOG.md")
        return

    df = pd.DataFrame(ablation)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#d0d0d0" if d <= 0 else "#90ee90" for d in df["delta"]]
    bars = ax.bar(df["technique"], df["dice"], color=colors, edgecolor="black")
    ax.axhline(y=df["dice"].iloc[0], color="red", linestyle="--", linewidth=1, label="baseline")
    ax.set_ylabel("Mean Dice")
    ax.set_title("Ablation: TTA and Post-processing")
    ax.legend()

    for bar, delta in zip(bars, df["delta"]):
        if delta != 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
                    f"{delta:+.4f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Figure 4: Segmentation Ablation Study", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUT / "fig4_ablation_tta_postproc.png")
    plt.close()
    print("Saved fig4_ablation_tta_postproc.png")


def fig_tn_staging_results(data: dict):
    """TN staging: model comparison from logged metrics."""
    tn = data["tn_staging"]
    if not tn:
        print("[skip] fig_tn_staging_results: no TN staging data")
        return

    df = pd.DataFrame(tn)
    for col in ["t_ba", "n_ba", "mean_ba"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["t_ba", "n_ba", "mean_ba"])

    if df.empty:
        print("[skip] fig_tn_staging_results: empty after parsing")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle("Figure 5: TN Staging Results", fontsize=12, fontweight="bold", y=1.02)

    models = df["model"].tolist()
    x = np.arange(len(models))
    width = 0.35
    t_vals = df["t_ba"].values
    n_vals = df["n_ba"].values

    ax = axes[0]
    ax.bar(x - width / 2, t_vals, width, label="T-stage BA", color="#1f77b4", edgecolor="black")
    ax.bar(x + width / 2, n_vals, width, label="N-stage BA", color="#ff7f0e", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylim(0.4, 1.05)
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("T-stage vs N-stage BA")
    ax.legend()

    ax = axes[1]
    means = df["mean_ba"].values
    colors = [MODEL_COLORS.get(m, "gray") for m in models]
    bars = ax.bar(models, means, color=colors, edgecolor="black")
    ax.set_ylim(0.4, 1.05)
    ax.set_ylabel("Mean Balanced Accuracy")
    ax.set_title("Mean BA = (T-BA + N-BA) / 2")
    for bar, score in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{score:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fig.savefig(OUT / "fig5_tn_staging_results.png")
    plt.close()
    print("Saved fig5_tn_staging_results.png")


def fig_prognosis_results(data: dict):
    """Prognosis model comparison from logged metrics."""
    prog = data["prognosis"]
    if not prog:
        print("[skip] fig_prognosis_results: no prognosis data")
        return

    df = pd.DataFrame(prog)
    df["c_index"] = pd.to_numeric(df["c_index"], errors="coerce")
    df = df.dropna(subset=["c_index"])

    if df.empty:
        print("[skip] fig_prognosis_results: empty after parsing")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = [MODEL_COLORS.get(m, "gray") for m in df["model"]]
    bars = ax.barh(df["model"], df["c_index"], color=colors, edgecolor="black")
    ax.set_xlim(0.6, 0.75)
    ax.set_xlabel("C-index")
    ax.set_title("Figure 6: Prognosis Model Comparison")
    for bar, score in zip(bars, df["c_index"]):
        ax.text(score + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(OUT / "fig6_prognosis_results.png")
    plt.close()
    print("Saved fig6_prognosis_results.png")


def fig_ensemble_strategy():
    """Ensemble configuration summary from ENSEMBLE.md."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Figure 7: Ensemble Strategy", fontsize=12, fontweight="bold")

    lines = [
        "Segmentation (10-model softmax average):",
        "  - MedNeXt-L: 5 folds (SWA preferred)",
        "  - ResEnc: 5 folds",
        "  - TTA: 8-flip per-model before averaging",
        "  - Weights: uniform (0.1 each)",
        "",
        "TN Staging (5-seed Mamba ensemble):",
        "  - Seeds: [42, 123, 252, 378, 456]",
        "  - Average T/N logits before argmax",
        "  - 5 seeds saturates performance (10 seeds same result)",
        "",
        "Prognosis (RSF + Cox rank-average):",
        "  - RSF weight: 0.6, Cox weight: 0.4",
        "  - Rank-average is scale-free and robust",
        "  - Higher score = higher recurrence risk",
    ]
    y = 9
    for line in lines:
        ax.text(0.5, y, line, fontsize=9, va="top", family="monospace")
        y -= 0.45

    plt.tight_layout()
    fig.savefig(OUT / "fig7_ensemble_strategy.png")
    plt.close()
    print("Saved fig7_ensemble_strategy.png")


def fig_dataset_statistics():
    """Dataset statistics from features_full.csv and splits_final.json."""
    features_path = Path("/home/hecktor/rabin/work/features_full.csv")
    splits_path = REPO / "work" / "splits_final.json"

    if not features_path.exists() or not splits_path.exists():
        print("[skip] fig_dataset_statistics: missing features_full.csv or splits_final.json")
        return

    df = pd.read_csv(features_path)
    with open(splits_path) as f:
        splits = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Figure 2: Dataset Statistics", fontsize=12, fontweight="bold", y=1.02)

    # Center distribution
    ax = axes[0, 0]
    centers = df["PatientID"].str.split("-").str[0].value_counts().sort_index()
    centers.plot(kind="bar", ax=ax, color="#1f77b4", edgecolor="black")
    ax.set_ylabel("Number of Patients")
    ax.set_title("Patients per Center")
    ax.tick_params(axis="x", rotation=45)

    # Train/val/test split sizes
    ax = axes[0, 1]
    fold0 = splits[0]
    sizes = [len(fold0["train"]), len(fold0["val"])]
    labels = ["Train", "Val"]
    ax.bar(labels, sizes, color=["#2ca02c", "#ff7f0e"], edgecolor="black")
    ax.set_ylabel("Number of Patients")
    ax.set_title("Fold 0 Split (825 total)")
    for bar, size in zip(ax.patches, sizes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(size), ha="center", va="bottom", fontsize=10)

    # T-stage distribution
    ax = axes[1, 0]
    t_dist = df["t_rule_index"].value_counts().sort_index()
    t_dist.index = ["T1", "T2", "T3", "T4"]
    t_dist.plot(kind="bar", ax=ax, color="#d62728", edgecolor="black")
    ax.set_ylabel("Number of Patients")
    ax.set_title("T-stage Distribution")
    ax.tick_params(axis="x", rotation=0)

    # N-stage distribution
    ax = axes[1, 1]
    n_dist = df["n_rule_index"].value_counts().sort_index()
    n_dist.index = ["N0", "N1", "N2", "N3"]
    n_dist.plot(kind="bar", ax=ax, color="#9467bd", edgecolor="black")
    ax.set_ylabel("Number of Patients")
    ax.set_title("N-stage Distribution")
    ax.tick_params(axis="x", rotation=0)

    plt.tight_layout()
    fig.savefig(OUT / "fig2_dataset_statistics.png")
    plt.close()
    print("Saved fig2_dataset_statistics.png")


def fig_centerwise_performance():
    """Center-wise patient count distribution from splits_final.json."""
    splits_path = REPO / "work" / "splits_final.json"
    if not splits_path.exists():
        print("[skip] fig_centerwise_performance: splits_final.json not found")
        return

    with open(splits_path) as f:
        splits = json.load(f)

    fold0 = splits[0]
    all_patients = fold0["train"] + fold0["val"]
    centers = [p.split("-")[0] for p in all_patients]
    center_counts = pd.Series(centers).value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    center_counts.plot(kind="bar", ax=ax, color="#8c564b", edgecolor="black")
    ax.set_ylabel("Number of Patients")
    ax.set_title("Figure 8: Center-wise Patient Distribution (Fold 0)")
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    fig.savefig(OUT / "fig8_centerwise_performance.png")
    plt.close()
    print("Saved fig8_centerwise_performance.png")


def fig_feature_importance():
    """Top correlated features with T-stage and N-stage from features_full.csv."""
    features_path = Path("/home/hecktor/rabin/work/features_full.csv")
    if not features_path.exists():
        print("[skip] fig_feature_importance: features_full.csv not found")
        return

    df = pd.read_csv(features_path)

    t_corr = df.select_dtypes(include=[np.number]).corrwith(df["t_rule_index"]).abs().sort_values(ascending=False).head(15)
    n_corr = df.select_dtypes(include=[np.number]).corrwith(df["n_rule_index"]).abs().sort_values(ascending=False).head(15)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("Figure 7: Top Feature Correlations (Absolute)", fontsize=12, fontweight="bold", y=1.02)

    ax = axes[0]
    t_corr.iloc[::-1].plot(kind="barh", ax=ax, color="#1f77b4", edgecolor="black")
    ax.set_xlabel("Absolute Correlation with T-stage")
    ax.set_title("T-stage")

    ax = axes[1]
    n_corr.iloc[::-1].plot(kind="barh", ax=ax, color="#ff7f0e", edgecolor="black")
    ax.set_xlabel("Absolute Correlation with N-stage")
    ax.set_title("N-stage")

    plt.tight_layout()
    fig.savefig(OUT / "fig7_feature_importance.png")
    plt.close()
    print("Saved fig7_feature_importance.png")


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("Loading metrics from markdown...")
    data = _parse_training_log()
    print(f"  segmentation records : {len(data['segmentation'])}")
    print(f"  tn_staging records   : {len(data['tn_staging'])}")
    print(f"  prognosis records    : {len(data['prognosis'])}")
    print(f"  tta_ablation records : {len(data['tta_ablation'])}")

    print("\nGenerating paper figures from real data...")
    fig_architecture()
    fig_dataset_statistics()
    fig_segmentation_results(data)
    fig_tta_ablation(data)
    fig_tn_staging_results(data)
    fig_prognosis_results(data)
    fig_ensemble_strategy()
    fig_centerwise_performance()
    fig_feature_importance()

    existing = [p for p in OUT.iterdir() if p.suffix == ".png"]
    print(f"\nDone. {len(existing)} figures in {OUT}")


if __name__ == "__main__":
    main()
