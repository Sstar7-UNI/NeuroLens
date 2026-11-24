from __future__ import annotations

import argparse
import csv
import math
import shutil
import warnings
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from neurolens.data_management.DatasetManager import DatasetManager
from neurolens.data_management.SplitType import SplitType
from neurolens.evaluation.Testbench import ExperimentSpec, Testbench, _NUMPY_NATIVE_OPS


CLASS_LABELS = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
CLASS_KEYS = ["glioma", "meningioma", "no_tumor", "pituitary"]
VERIFY_STAGES = {"3_verify", "4_verify", "5_verify"}
BASELINE_NAME = "baseline_aug8"


def _read_csv(path: Path) -> List[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _to_float(val, default: float = np.nan) -> float:
    if val is None or val == "":
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _to_int(val, default: int = 0) -> int:
    if val is None or val == "":
        return default
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default


def _safe_nanmean(vals: List[float]) -> float:
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return float(np.nanmean(arr))


def _safe_nanstd(vals: List[float]) -> float:
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return float(np.nanstd(arr))


def _slot_of(row: dict) -> str:
    stage = row.get("stage", "")
    name = row.get("experiment_name", "")
    if name == BASELINE_NAME or stage == "1_aug":
        return "Baseline"
    if stage in {"2_combos", "3_combos"}:
        return "2_combos"
    if name.startswith("X"):
        return "X_denoise"
    if name.startswith("Y"):
        return "Y_harmonize"
    if name.startswith("Z"):
        return "Z_contrast"
    if name.startswith("W"):
        return "W_channels"
    return stage or "Other"


def plot_01_training_dynamics(epoch_rows: List[dict], out_dir: Path) -> bool:
    if not epoch_rows:
        return False

    grouped: Dict[str, Dict[int, List[dict]]] = {}
    for r in epoch_rows:
        stage = r.get("stage", "")
        if stage in VERIFY_STAGES:
            continue
        label = f"{stage} | {r.get('experiment_name', '')}"
        ep = _to_int(r.get("epoch"), 0)
        grouped.setdefault(label, {}).setdefault(ep, []).append(r)

    if not grouped:
        for r in epoch_rows:
            label = f"{r.get('stage', '')} | {r.get('experiment_name', '')}"
            ep = _to_int(r.get("epoch"), 0)
            grouped.setdefault(label, {}).setdefault(ep, []).append(r)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11.5), dpi=150)
    ax_loss, ax_f1, ax_min, ax_ema = axes.flat

    cmap = plt.get_cmap("tab20")
    for idx, (label, ep_map) in enumerate(grouped.items()):
        epochs = sorted(ep_map.keys())
        val_loss = [_safe_nanmean([_to_float(x.get("val_loss")) for x in ep_map[e]]) for e in epochs]
        val_f1 = [_safe_nanmean([_to_float(x.get("val_macro_f1")) for x in ep_map[e]]) for e in epochs]
        min_f1 = [
            _safe_nanmean([_to_float(x.get("val_min_class_f1")) for x in ep_map[e]]) for e in epochs
        ]
        ema_f1 = [
            _safe_nanmean([_to_float(x.get("val_ema_balanced_f1")) for x in ep_map[e]])
            for e in epochs
        ]
        is_base = BASELINE_NAME in label
        color = "black" if is_base else cmap(idx % 20)
        ls = "--" if is_base else "-"
        lw = 2.2 if is_base else 1.5
        ax_loss.plot(epochs, val_loss, label=label, color=color, linestyle=ls, linewidth=lw)
        ax_f1.plot(epochs, val_f1, label=label, color=color, linestyle=ls, linewidth=lw)
        ax_min.plot(epochs, min_f1, label=label, color=color, linestyle=ls, linewidth=lw)
        ax_ema.plot(epochs, ema_f1, label=label, color=color, linestyle=ls, linewidth=lw)

    ax_loss.set_title("Validation Loss Trajectory")
    ax_loss.set_ylabel("Cross-Entropy Loss")
    ax_f1.set_title("Validation Macro F1")
    ax_f1.set_ylabel("Macro F1")
    ax_min.set_title("Validation Weakest-Class F1")
    ax_min.set_ylabel("Min Class F1")
    ax_ema.set_title("Selection Criterion: Val EMA Balanced F1 (f1_ema)")
    ax_ema.set_ylabel("f1_ema")

    for ax in axes.flat:
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)

    handles, labels = ax_ema.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(4, max(1, len(labels))),
            fontsize=7.8,
            frameon=True,
        )
    fig.suptitle("NeuroLens Training Dynamics Across All Screening Runs", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0.18, 1, 0.95])
    fig.savefig(out_dir / "01_training_dynamics.png")
    plt.close(fig)

    # 01c: Slot-by-slot f1_ema trajectories vs baseline so every curve is readable
    slot_groups = [
        ("Slot X: Edge-Preserving Denoisers", lambda r: _slot_of(r) == "X_denoise"),
        ("Slot Y: MRI Intensity Harmonization", lambda r: _slot_of(r) == "Y_harmonize"),
        ("Slots Z & W: Contrast & Multi-Channel Stack", lambda r: _slot_of(r) in {"Z_contrast", "W_channels"}),
        ("Stage 2: Cross-Slot Combinations", lambda r: _slot_of(r) == "2_combos"),
    ]
    base_ep_map: Dict[int, List[dict]] = {}
    for r in epoch_rows:
        if r.get("stage", "") not in VERIFY_STAGES and r.get("experiment_name") == BASELINE_NAME:
            base_ep_map.setdefault(_to_int(r.get("epoch"), 0), []).append(r)

    fig_s, axes_s = plt.subplots(2, 2, figsize=(15, 10), dpi=150)
    s_cmap = plt.get_cmap("tab10")
    for ax, (panel_title, pred) in zip(axes_s.flat, slot_groups):
        if base_ep_map:
            b_eps = sorted(base_ep_map.keys())
            b_ema = [
                _safe_nanmean([_to_float(x.get("val_ema_balanced_f1")) for x in base_ep_map[e]])
                for e in b_eps
            ]
            ax.plot(b_eps, b_ema, label=BASELINE_NAME, color="black", linestyle="--", linewidth=2.0)

        sub_grouped: Dict[str, Dict[int, List[dict]]] = {}
        for r in epoch_rows:
            if r.get("stage", "") in VERIFY_STAGES:
                continue
            if pred(r):
                sub_grouped.setdefault(r.get("experiment_name", ""), {}).setdefault(
                    _to_int(r.get("epoch"), 0), []
                ).append(r)

        for idx, (ename, ep_map) in enumerate(sub_grouped.items()):
            eps = sorted(ep_map.keys())
            ema = [
                _safe_nanmean([_to_float(x.get("val_ema_balanced_f1")) for x in ep_map[e]])
                for e in eps
            ]
            ax.plot(eps, ema, label=ename, color=s_cmap(idx % 10), linewidth=1.8)

        ax.set_title(panel_title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Val f1_ema")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)

    fig_s.suptitle("Training Trajectories (f1_ema) Broken Down by Filter Slot & Stage 2 Combos", fontsize=13)
    fig_s.tight_layout(rect=[0, 0, 1, 0.95])
    fig_s.savefig(out_dir / "01c_training_dynamics_by_slot.png")
    plt.close(fig_s)

    # 01b: Multi-seed verification dynamics (all 4 validation metrics)
    verify_grouped: Dict[str, Dict[int, List[dict]]] = {}
    for r in epoch_rows:
        if r.get("stage", "") in VERIFY_STAGES:
            label = r.get("experiment_name", "")
            ep = _to_int(r.get("epoch"), 0)
            verify_grouped.setdefault(label, {}).setdefault(ep, []).append(r)

    if verify_grouped:
        fig_v, axes_v = plt.subplots(2, 2, figsize=(15, 9.5), dpi=150)
        v_cmap = plt.get_cmap("tab10")
        panels = [
            (axes_v[0, 0], "val_loss", "Val Loss (Mean ± SD)", "Cross-Entropy Loss"),
            (axes_v[0, 1], "val_macro_f1", "Val Macro F1 (Mean ± SD)", "Macro F1"),
            (axes_v[1, 0], "val_min_class_f1", "Val Weakest-Class F1 (Mean ± SD)", "Min Class F1"),
            (axes_v[1, 1], "val_ema_balanced_f1", "Val f1_ema (Mean ± SD)", "f1_ema"),
        ]

        for idx, (label, ep_map) in enumerate(verify_grouped.items()):
            epochs = sorted(ep_map.keys())
            color = v_cmap(idx % 10)
            for ax, key, title, ylabel in panels:
                means = np.array(
                    [_safe_nanmean([_to_float(x.get(key)) for x in ep_map[e]]) for e in epochs]
                )
                stds = np.array(
                    [_safe_nanstd([_to_float(x.get(key)) for x in ep_map[e]]) for e in epochs]
                )
                ax.plot(epochs, means, label=label, color=color, linewidth=1.8)
                ax.fill_between(epochs, means - stds, means + stds, color=color, alpha=0.15)

        for ax, _, title, ylabel in panels:
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Epoch")
            ax.grid(True, alpha=0.3)
        axes_v[1, 1].legend(loc="lower right", fontsize=8.5)
        fig_v.suptitle("Stage 3 Multi-Seed Verification Training Trajectories", fontsize=13)
        fig_v.tight_layout(rect=[0, 0, 1, 0.95])
        fig_v.savefig(out_dir / "01b_verify_training_dynamics.png")
        plt.close(fig_v)

    return True


def plot_02_tournament_progression(metric_rows: List[dict], out_dir: Path) -> bool:
    screen_rows = [r for r in metric_rows if r.get("stage") not in VERIFY_STAGES]
    if not screen_rows:
        return False

    labels = [f"{r['stage']}\n{r['experiment_name']}" for r in screen_rows]
    ema_scores = [_to_float(r.get("best_val_ema_balanced_f1"), 0.0) for r in screen_rows]
    min_scores = [_to_float(r.get("best_val_min_class_f1"), 0.0) for r in screen_rows]

    stage_colors = {
        "Baseline": "#4C78A8",
        "X_denoise": "#F58518",
        "Y_harmonize": "#E45756",
        "Z_contrast": "#72B7B2",
        "W_channels": "#B279A2",
        "2_combos": "#54A24B",
    }
    colors = [stage_colors.get(_slot_of(r), "#72B7B2") for r in screen_rows]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.9), 6.2), dpi=150)
    bars = ax.bar(x - 0.18, ema_scores, width=0.36, color=colors, label="f1_ema (Selection Metric)")
    ax.bar(
        x + 0.18,
        min_scores,
        width=0.36,
        color="#9D755D",
        alpha=0.82,
        label="Weakest-Class F1",
    )

    base_rows = [
        r for r in screen_rows if r.get("stage") == "1_aug" or r.get("experiment_name") == BASELINE_NAME
    ]
    if base_rows:
        base_ema = max(_to_float(r.get("best_val_ema_balanced_f1"), 0.0) for r in base_rows)
        ax.axhline(
            base_ema,
            color="#4C78A8",
            linestyle="--",
            linewidth=1.2,
            alpha=0.85,
            label=f"Baseline ({base_ema:.4f})",
        )

    for bar, val in zip(bars, ema_scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            val + 0.003,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=7.8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=32, ha="right", fontsize=8.2)
    ax.set_ylabel("Validation Score")
    all_scores = [v for v in (min_scores + ema_scores) if math.isfinite(v)]
    y_lo = max(0.0, min(all_scores) - 0.06) if all_scores else 0.0
    y_hi = min(1.03, max(all_scores) + 0.05) if all_scores else 1.0
    ax.set_ylim(y_lo, y_hi)
    ax.set_title("Testbench Progression Across All Filter Slots & Cross-Slot Combinations")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out_dir / "02_stage_tournament_progression.png")
    plt.close(fig)
    return True


def plot_03_filter_slot_ablation(metric_rows: List[dict], out_dir: Path) -> bool:
    aug_rows = [r for r in metric_rows if r.get("stage") == "1_aug"]
    filter_rows = [r for r in metric_rows if r.get("stage") == "2_filters"]
    if not aug_rows and not filter_rows:
        v2_filters = [r for r in metric_rows if r.get("stage") == "1_filters"]
        aug_rows = [r for r in v2_filters if r.get("experiment_name") == BASELINE_NAME]
        filter_rows = [r for r in v2_filters if r.get("experiment_name") != BASELINE_NAME]
    if not aug_rows or not filter_rows:
        return False

    aug_star = max(aug_rows, key=lambda r: _to_float(r.get("best_val_ema_balanced_f1"), -1.0))
    ref_ema = _to_float(aug_star.get("best_val_ema_balanced_f1"), 0.0)
    ref_min = _to_float(aug_star.get("best_val_min_class_f1"), 0.0)

    names = [r["experiment_name"] for r in filter_rows]
    delta_ema = [_to_float(r.get("best_val_ema_balanced_f1"), 0.0) - ref_ema for r in filter_rows]
    delta_min = [_to_float(r.get("best_val_min_class_f1"), 0.0) - ref_min for r in filter_rows]

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.05), 5.6), dpi=150)
    bar_colors = ["#54A24B" if de > 0 else "#E45756" for de in delta_ema]
    bars = ax.bar(x - 0.18, delta_ema, width=0.36, color=bar_colors, label="Δ f1_ema vs Baseline")
    ax.bar(x + 0.18, delta_min, width=0.36, color="#4C78A8", label="Δ Weakest-Class F1 vs Baseline")

    for bar, de in zip(bars, delta_ema):
        y_pos = de + 0.0012 if de >= 0 else de - 0.0025
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            y_pos,
            f"{de:+.4f}",
            ha="center",
            va="bottom" if de >= 0 else "top",
            fontsize=8,
        )

    # Visual separators between Slots X, Y, Z, W
    slots = [_slot_of(r) for r in filter_rows]
    for i in range(1, len(slots)):
        if slots[i] != slots[i - 1]:
            ax.axvline(i - 0.5, color="gray", linestyle=":", linewidth=1.0, alpha=0.6)

    ax.axhline(0.0, color="black", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=22, ha="right")
    ax.set_ylabel(f"Score Delta vs {aug_star['experiment_name']} ({ref_ema:.4f})")
    ax.set_title("Stage 1 Orthogonal Filter Slot Screening (X: Denoise, Y: Harmonize, Z: Contrast, W: Stack)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out_dir / "03_filter_slot_ablation.png")
    plt.close(fig)

    # 03b: Stage 2 Cross-Slot Combinations vs Baseline & Best Single Filter
    combo_rows = [r for r in metric_rows if r.get("stage") in {"2_combos", "3_combos"}]
    if combo_rows:
        best_single_ema = max(_to_float(r.get("best_val_ema_balanced_f1"), 0.0) for r in filter_rows)
        c_names = [r["experiment_name"] for r in combo_rows]
        c_d_base = [_to_float(r.get("best_val_ema_balanced_f1"), 0.0) - ref_ema for r in combo_rows]
        c_d_single = [
            _to_float(r.get("best_val_ema_balanced_f1"), 0.0) - best_single_ema for r in combo_rows
        ]
        c_d_min = [_to_float(r.get("best_val_min_class_f1"), 0.0) - ref_min for r in combo_rows]

        cx = np.arange(len(c_names))
        fig_c, ax_c = plt.subplots(figsize=(max(10, len(c_names) * 1.45), 5.6), dpi=150)
        b1 = ax_c.bar(cx - 0.24, c_d_base, width=0.24, color="#54A24B", label="Δ f1_ema vs Baseline")
        ax_c.bar(
            cx,
            c_d_single,
            width=0.24,
            color="#F58518",
            label=f"Δ f1_ema vs Best Single Filter ({best_single_ema:.4f})",
        )
        ax_c.bar(
            cx + 0.24,
            c_d_min,
            width=0.24,
            color="#4C78A8",
            label="Δ Weakest-Class F1 vs Baseline",
        )
        for bar, val in zip(b1, c_d_base):
            ax_c.text(
                bar.get_x() + bar.get_width() / 2.0,
                val + 0.0012 if val >= 0 else val - 0.0025,
                f"{val:+.4f}",
                ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=8,
            )
        ax_c.axhline(0.0, color="black", linewidth=0.9)
        ax_c.set_xticks(cx)
        ax_c.set_xticklabels(c_names, rotation=20, ha="right", fontsize=8.5)
        ax_c.set_ylabel("Validation Score Delta")
        ax_c.set_title("Stage 2 Cross-Slot Combinations: Gains Over Baseline & Synergy Over Best Single Filter")
        ax_c.grid(True, axis="y", alpha=0.3)
        ax_c.legend(fontsize=8.5)
        fig_c.tight_layout()
        fig_c.savefig(out_dir / "03b_combo_synergy_ablation.png")
        plt.close(fig_c)

    return True


def plot_04_confusion_matrices(metric_rows: List[dict], out_dir: Path) -> bool:
    verify_rows = [
        r for r in metric_rows if r.get("stage") in VERIFY_STAGES and _to_int(r.get("test_total")) > 0
    ]
    if not verify_rows:
        return False

    by_cfg: Dict[str, np.ndarray] = {}
    counts_by_cfg: Dict[str, int] = {}
    for r in verify_rows:
        name = r["experiment_name"]
        cm = np.array(
            [[_to_int(r.get(f"cm_{row_i}{col_i}"), 0) for col_i in range(4)] for row_i in range(4)],
            dtype=np.float64,
        )
        by_cfg[name] = by_cfg.get(name, np.zeros((4, 4), dtype=np.float64)) + cm
        counts_by_cfg[name] = counts_by_cfg.get(name, 0) + 1

    n_cfgs = len(by_cfg)
    if n_cfgs == 4:
        nrows, ncols = 2, 2
        figsize = (10.5, 9.2)
    else:
        nrows, ncols = 1, n_cfgs
        figsize = (5.0 * n_cfgs, 4.8)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=150, constrained_layout=True)
    axes_flat = [axes] if n_cfgs == 1 else list(np.asarray(axes).flat)

    im = None
    for ax, (name, cm_sum) in zip(axes_flat, by_cfg.items()):
        row_sums = np.maximum(cm_sum.sum(axis=1, keepdims=True), 1.0)
        cm_norm = cm_sum / row_sums
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
        for i in range(4):
            for j in range(4):
                pct = 100.0 * cm_norm[i, j]
                cnt = int(round(cm_sum[i, j]))
                txt_color = "white" if cm_norm[i, j] > 0.55 else "black"
                ax.text(
                    j,
                    i,
                    f"{pct:.1f}%\n({cnt})",
                    ha="center",
                    va="center",
                    color=txt_color,
                    fontsize=8.5,
                )
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(CLASS_LABELS, rotation=25, ha="right", fontsize=8.5)
        ax.set_yticklabels(CLASS_LABELS, fontsize=8.5)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{name} (n={counts_by_cfg[name]} seed(s))", fontsize=10)

    for extra_ax in axes_flat[n_cfgs:]:
        extra_ax.set_visible(False)

    if im is not None:
        fig.colorbar(im, ax=axes_flat[:n_cfgs], shrink=0.82, label="Row-Normalized Recall")
    fig.suptitle("Held-Out Test Confusion Matrices (Aggregated Across Seeds)", fontsize=13)
    fig.savefig(out_dir / "04_test_confusion_matrices.png")
    plt.close(fig)
    return True


def plot_05_per_class_metrics(metric_rows: List[dict], out_dir: Path) -> bool:
    verify_rows = [
        r for r in metric_rows if r.get("stage") in VERIFY_STAGES and _to_int(r.get("test_total")) > 0
    ]
    if not verify_rows:
        return False

    grouped: Dict[str, List[dict]] = {}
    for r in verify_rows:
        grouped.setdefault(r["experiment_name"], []).append(r)

    cfg_names = list(grouped.keys())
    x = np.arange(len(CLASS_LABELS))
    width = 0.8 / max(1, len(cfg_names))

    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5), dpi=150)
    metric_specs = [
        ("f1", "Per-Class Test F1 Score"),
        ("sensitivity", "Per-Class Sensitivity (Recall)"),
        ("precision", "Per-Class Precision"),
        ("specificity", "Per-Class Specificity"),
    ]

    all_plotted_vals: List[float] = []
    for ax, (suffix, title) in zip(axes.flat, metric_specs):
        for idx, cname in enumerate(cfg_names):
            rows = grouped[cname]
            means = [
                _safe_nanmean([_to_float(r.get(f"{ck}_{suffix}")) for r in rows]) for ck in CLASS_KEYS
            ]
            stds = [
                _safe_nanstd([_to_float(r.get(f"{ck}_{suffix}")) for r in rows]) for ck in CLASS_KEYS
            ]
            all_plotted_vals.extend([m for m in means if math.isfinite(m)])
            offset = (idx - (len(cfg_names) - 1) / 2.0) * width
            ax.bar(x + offset, means, width=width * 0.92, yerr=stds, capsize=3, label=cname)

        ax.set_xticks(x)
        ax.set_xticklabels(CLASS_LABELS, rotation=12)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)

    y_lo = max(0.0, min(all_plotted_vals) - 0.06) if all_plotted_vals else 0.0
    for ax in axes.flat:
        ax.set_ylim(y_lo, 1.02)

    axes[1, 1].legend(loc="lower right", fontsize=8.5)
    fig.suptitle("4-Class Diagnostic Breakdown on Held-Out Test Set (Mean ± SD Across Seeds)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "05_per_class_metrics.png")
    plt.close(fig)
    return True


def plot_06_statistical_verification(
    metric_rows: List[dict], stats_rows: List[dict], out_dir: Path
) -> bool:
    verify_rows = [
        r for r in metric_rows if r.get("stage") in VERIFY_STAGES and _to_int(r.get("test_total")) > 0
    ]
    if not verify_rows and not stats_rows:
        return False

    grouped: Dict[str, List[dict]] = {}
    for r in verify_rows:
        grouped.setdefault(r["experiment_name"], []).append(r)

    names = [r["config_name"] for r in stats_rows] if stats_rows else list(grouped.keys())
    f1_means, f1_stds, min_means, mcc_means, mcc_stds, pvals = [], [], [], [], [], []
    ci_lows, ci_highs, deltas = [], [], []
    stats_by_name = {r["config_name"]: r for r in stats_rows}

    for n in names:
        if n in stats_by_name:
            sr = stats_by_name[n]
            f1_means.append(_to_float(sr.get("test_f1_mean"), 0.0))
            f1_stds.append(_to_float(sr.get("test_f1_std"), 0.0))
            min_means.append(_to_float(sr.get("test_min_class_f1_mean"), 0.0))
            mcc_means.append(_to_float(sr.get("test_mcc_mean"), 0.0))
            mcc_stds.append(_to_float(sr.get("test_mcc_std"), 0.0))
            pvals.append(_to_float(sr.get("paired_t_pval"), 1.0))
            deltas.append(_to_float(sr.get("delta_f1_vs_baseline"), 0.0))
            ci_lows.append(_to_float(sr.get("ci95_low"), 0.0))
            ci_highs.append(_to_float(sr.get("ci95_high"), 0.0))
        elif n in grouped:
            rows = grouped[n]
            f1_means.append(_safe_nanmean([_to_float(r.get("test_macro_f1")) for r in rows]))
            f1_stds.append(_safe_nanstd([_to_float(r.get("test_macro_f1")) for r in rows]))
            min_means.append(_safe_nanmean([_to_float(r.get("test_min_class_f1")) for r in rows]))
            mcc_means.append(_safe_nanmean([_to_float(r.get("test_mcc")) for r in rows]))
            mcc_stds.append(_safe_nanstd([_to_float(r.get("test_mcc")) for r in rows]))
            pvals.append(1.0)
            deltas.append(0.0)
            ci_lows.append(0.0)
            ci_highs.append(0.0)

    if not f1_means:
        return False

    x = np.arange(len(names))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.6), dpi=150)

    bars = ax1.bar(
        x - 0.18,
        f1_means,
        yerr=f1_stds,
        capsize=4,
        width=0.36,
        color="#4C78A8",
        label="Test Macro F1 (±SD)",
    )
    ax1.bar(
        x + 0.18,
        min_means,
        width=0.36,
        color="#54A24B",
        label="Weakest-Class F1",
    )
    for idx, (bar, pv) in enumerate(zip(bars, pvals)):
        top_y = f1_means[idx] + f1_stds[idx] + 0.012
        label_txt = f"{f1_means[idx]:.3f}" if idx == 0 else f"{f1_means[idx]:.3f}\n(p={pv:.3f})"
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            min(top_y, 0.99),
            label_txt,
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=18, ha="right")
    y1_lo = max(0.0, min(min_means + f1_means) - 0.08)
    y1_hi = max(1.05, max(m + s for m, s in zip(f1_means, f1_stds)) + 0.08)
    ax1.set_ylim(y1_lo, y1_hi)
    ax1.set_ylabel("Held-Out Test Score")
    ax1.set_title("Multi-Seed Paired Verification (Macro F1 & Weakest Class)")
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.legend(loc="lower right", fontsize=8.5)

    ax2.bar(
        x,
        mcc_means,
        yerr=mcc_stds,
        capsize=4,
        width=0.46,
        color="#F58518",
        label="Matthews Correlation Coeff (±SD)",
    )
    for i, (val, sd) in enumerate(zip(mcc_means, mcc_stds)):
        ax2.text(
            i,
            min(val + sd + 0.012, 0.99),
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=18, ha="right")
    y2_lo = max(0.0, min(mcc_means) - 0.08)
    y2_hi = max(1.03, max(m + s for m, s in zip(mcc_means, mcc_stds)) + 0.08)
    ax2.set_ylim(y2_lo, y2_hi)
    ax2.set_ylabel("MCC")
    ax2.set_title("Multi-Seed Matthews Correlation Coefficient")
    ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Stage 3 Statistical Verification Summary", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_dir / "06_statistical_verification.png")
    plt.close(fig)

    # 06b: Comprehensive Test Metrics (Accuracy, Kappa, ROC-AUC), 95% CI, and Compute Cost
    fig2, axes2 = plt.subplots(1, 3, figsize=(17, 5.4), dpi=150)
    ax_ci, ax_metrics, ax_cost = axes2.flat

    # Panel 1: 95% CI of Delta F1 vs Baseline
    y_pos = np.arange(len(names))
    ax_ci.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    for i, (d, lo, hi, pv) in enumerate(zip(deltas, ci_lows, ci_highs, pvals)):
        color = "#4C78A8" if i == 0 else ("#54A24B" if d > 0 else "#E45756")
        ax_ci.errorbar(
            d,
            i,
            xerr=[[max(0.0, d - lo)], [max(0.0, hi - d)]],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=4,
            markersize=7,
            linewidth=1.8,
        )
        txt = "Baseline (0.0000)" if i == 0 else f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}] (p={pv:.3f})"
        ax_ci.text(d, i - 0.22, txt, ha="center", va="bottom", fontsize=7.8)
    ax_ci.set_yticks(y_pos)
    ax_ci.set_yticklabels(names, fontsize=8.5)
    ax_ci.invert_yaxis()
    ax_ci.set_xlabel("Δ Test Macro F1 vs Baseline (95% CI)")
    ax_ci.set_title("Paired Effect Size & 95% Confidence Interval")
    ax_ci.grid(True, axis="x", alpha=0.3)

    # Panel 2: Accuracy, Cohen's Kappa, Macro ROC-AUC
    acc_means = [_safe_nanmean([_to_float(r.get("test_accuracy")) for r in grouped.get(n, [])]) for n in names]
    kap_means = [_safe_nanmean([_to_float(r.get("test_cohen_kappa")) for r in grouped.get(n, [])]) for n in names]
    auc_means = [_safe_nanmean([_to_float(r.get("test_roc_auc_macro")) for r in grouped.get(n, [])]) for n in names]
    ax_metrics.bar(x - 0.25, acc_means, width=0.25, color="#4C78A8", label="Test Accuracy")
    ax_metrics.bar(x, kap_means, width=0.25, color="#F58518", label="Cohen's Kappa")
    ax_metrics.bar(x + 0.25, auc_means, width=0.25, color="#54A24B", label="Macro ROC-AUC")
    ax_metrics.set_xticks(x)
    ax_metrics.set_xticklabels(names, rotation=20, ha="right", fontsize=8.5)
    valid_m = [v for v in (acc_means + kap_means + auc_means) if math.isfinite(v)]
    ax_metrics.set_ylim(max(0.0, min(valid_m) - 0.06) if valid_m else 0.0, 1.02)
    ax_metrics.set_ylabel("Score")
    ax_metrics.set_title("Accuracy, Cohen's Kappa & Macro ROC-AUC")
    ax_metrics.grid(True, axis="y", alpha=0.3)
    ax_metrics.legend(loc="lower right", fontsize=8)

    # Panel 3: Train Time (s) & Inference Time (ms/img)
    tr_times = [_safe_nanmean([_to_float(r.get("train_time_s")) for r in grouped.get(n, [])]) for n in names]
    inf_times = [
        _safe_nanmean([_to_float(r.get("inference_time_ms_per_img")) for r in grouped.get(n, [])])
        for n in names
    ]
    ax_cost_r = ax_cost.twinx()
    p1 = ax_cost.bar(x - 0.18, tr_times, width=0.35, color="#72B7B2", label="Train Time (s)")
    p2 = ax_cost_r.bar(x + 0.18, inf_times, width=0.35, color="#B279A2", label="Inference (ms/img)")
    ax_cost.set_xticks(x)
    ax_cost.set_xticklabels(names, rotation=20, ha="right", fontsize=8.5)
    ax_cost.set_ylabel("Mean Train Time (s)", color="#2B6CB0")
    ax_cost_r.set_ylabel("Inference Latency (ms/img)", color="#805AD5")
    ax_cost.set_title("Computational Cost: Train Time & Inference Speed")
    ax_cost.grid(True, axis="y", alpha=0.3)
    lines = [p1, p2]
    ax_cost.legend(lines, [l.get_label() for l in lines], loc="upper left", fontsize=8)

    fig2.suptitle("Stage 3 Effect Sizes, Global Agreement Metrics & Compute Cost", fontsize=13)
    fig2.tight_layout(rect=[0, 0, 1, 0.94])
    fig2.savefig(out_dir / "06b_test_comprehensive_and_cost.png")
    plt.close(fig2)

    return True


def _render_spec_grid(
    tb: Testbench,
    chosen: list,
    train_fit_pool: list,
    specs_with_channels: list[tuple[str, ExperimentSpec, int]],
    title: str,
    out_path: Path,
) -> None:
    n_rows = len(chosen)
    n_cols = len(specs_with_channels)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.25 * n_cols, 2.35 * n_rows), dpi=150)
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    fitted_nyul_cache: dict[str, object] = {}

    def _run_pipeline_on_record(rec, spec: ExperimentSpec, ops: list) -> np.ndarray:
        img = tb._decode_raw_image(str(rec.imagePath), spec.input_size)
        for op in ops:
            if hasattr(op, "fit"):
                cache_key = spec.denoise_method
                if cache_key not in fitted_nyul_cache:
                    fit_imgs = []
                    pre_ops = tb.build_common_prefix_ops(spec)
                    if spec.denoise_method != "none":
                        from neurolens.evaluation.Testbench import FastDenoiseFilter
                        pre_ops.append(FastDenoiseFilter(method=spec.denoise_method))
                    for f_rec in train_fit_pool:
                        f_im = tb._decode_raw_image(str(f_rec.imagePath), spec.input_size)
                        for p_op in pre_ops:
                            inp_f = (
                                f_im
                                if isinstance(p_op, _NUMPY_NATIVE_OPS)
                                else tf.convert_to_tensor(f_im, dtype=tf.float32)
                            )
                            res_f = p_op.apply(inp_f)
                            f_im = np.asarray(
                                res_f.numpy() if tf.is_tensor(res_f) else res_f, dtype=np.float32
                            )
                            if f_im.ndim == 2:
                                f_im = f_im[..., np.newaxis]
                        fit_imgs.append(f_im[..., 0])
                    op.fit(fit_imgs)
                    fitted_nyul_cache[cache_key] = op
                op = fitted_nyul_cache[cache_key]

            inp = (
                img
                if isinstance(op, _NUMPY_NATIVE_OPS)
                else tf.convert_to_tensor(img, dtype=tf.float32)
            )
            res = op.apply(inp)
            img = np.asarray(res.numpy() if tf.is_tensor(res) else res, dtype=np.float32)
            if img.ndim == 2:
                img = img[..., np.newaxis]
        return tb._attach_feature_channels(img, spec.feature_channels)

    for col_idx, (col_title, spec, ch_idx) in enumerate(specs_with_channels):
        ops = tb.build_pipeline(spec)
        for row_idx, rec in enumerate(chosen):
            img = _run_pipeline_on_record(rec, spec, ops)
            use_ch = min(ch_idx, img.shape[-1] - 1)
            disp = img[..., use_ch]
            ax = axes[row_idx, col_idx]
            ax.imshow(disp, cmap="magma" if use_ch > 0 else "gray")
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(col_title, fontsize=8.0)
            if col_idx == 0:
                ax.set_ylabel(rec.label.name, fontsize=9.0)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path)
    plt.close(fig)


def plot_07_config_image_effects(
    dataset_path: str, metric_rows: List[dict], out_dir: Path
) -> bool:
    ds_dir = Path(dataset_path)
    if not ds_dir.exists():
        return False

    manager = DatasetManager(str(ds_dir))
    manifest = manager.loadDataset()
    samples = manifest.getSamples()
    if not samples:
        return False

    by_label = {}
    for rec in samples:
        if rec.label not in by_label:
            by_label[rec.label] = rec
        if len(by_label) == 4:
            break
    chosen = list(by_label.values())
    train_fit_pool = [r for r in samples if r.split != SplitType.TEST][:40]

    tmp_work_dir = out_dir / "_tmp_preview_work"
    tb = Testbench(
        dataset_path=str(ds_dir),
        output_csv=str(out_dir / "_tmp_preview.csv"),
        work_dir=str(tmp_work_dir),
        verbose=0,
    )

    # 07: Every single Stage 1 operator tested in Testbench
    stage1_preview_specs: list[tuple[str, ExperimentSpec, int]] = [
        (
            "Raw Input",
            ExperimentSpec(
                stage="p",
                name="raw",
                crop_to_brain=False,
                n4_bias=False,
                denoise_method="none",
                contrast_method="none",
            ),
            0,
        ),
        ("Baseline\n(Med+CLAHE)", ExperimentSpec(stage="p", name="base"), 0),
        ("X1: SUSAN", ExperimentSpec(stage="p", name="x1", denoise_method="SUSAN"), 0),
        ("X2: GIF", ExperimentSpec(stage="p", name="x2", denoise_method="GIF"), 0),
        ("X3: Wavelet", ExperimentSpec(stage="p", name="x3", denoise_method="wavelet"), 0),
        ("Y1: Pct Clip", ExperimentSpec(stage="p", name="y1", intensity_norm="percentile_clip"), 0),
        ("Y2: WhiteStripe", ExperimentSpec(stage="p", name="y2", intensity_norm="WhiteStripe"), 0),
        ("Y3: Nyul", ExperimentSpec(stage="p", name="y3", intensity_norm="Nyul"), 0),
        (
            "Z1: TopHat+BotHat",
            ExperimentSpec(
                stage="p",
                name="z1",
                edge_filter="TopHatBottomHat",
                contrast_method="CLAHE_1.0_8",
            ),
            0,
        ),
        ("W1: +LBP Ch", ExperimentSpec(stage="p", name="w1", feature_channels="lbp"), 1),
        ("W2: +LoG (σ=1.0)", ExperimentSpec(stage="p", name="w2a", feature_channels="log"), 1),
        ("W2: +LoG (σ=2.5)", ExperimentSpec(stage="p", name="w2b", feature_channels="log"), 2),
    ]

    _render_spec_grid(
        tb,
        chosen,
        train_fit_pool,
        stage1_preview_specs,
        "All Stage 1 MRI Preprocessing & Multi-Channel Operators Across Tumor Classes",
        out_dir / "07_config_image_effects.png",
    )

    # 07b: Dynamic visual comparison of all Stage 2 combos from CSV
    combo_rows = [r for r in metric_rows if r.get("stage") in {"2_combos", "3_combos"}]
    if combo_rows:
        combo_preview_specs: list[tuple[str, ExperimentSpec, int]] = [
            ("Baseline\n(Med+CLAHE)", ExperimentSpec(stage="p", name="base"), 0)
        ]
        for r in combo_rows:
            spec = ExperimentSpec(
                stage="p",
                name=r["experiment_name"],
                crop_to_brain=bool(_to_int(r.get("crop_to_brain"), 1)),
                n4_bias=bool(_to_int(r.get("n4_bias"), 1)),
                input_size=_to_int(r.get("input_size"), 128),
                denoise_method=r.get("denoise_method", "median"),
                intensity_norm=r.get("intensity_norm", "none"),
                contrast_method=r.get("contrast_method", "CLAHE_2.0_8"),
                edge_filter=r.get("edge_filter", "none"),
                feature_channels=r.get("feature_channels", "none"),
            )
            short_title = r["experiment_name"].replace("+", "\n+")
            combo_preview_specs.append((f"{short_title}\n[Ch0: MRI]", spec, 0))
            if spec.num_channels() > 1:
                combo_preview_specs.append((f"{short_title}\n[Extra Ch]", spec, 1))

        _render_spec_grid(
            tb,
            chosen,
            train_fit_pool,
            combo_preview_specs,
            "Stage 2 Cross-Slot Filter Combinations Visual Effects",
            out_dir / "07b_stage2_combo_image_effects.png",
        )

    for tmp_p in (
        out_dir / "_tmp_preview.csv",
        out_dir / "_tmp_preview_epochs.csv",
    ):
        if tmp_p.exists():
            tmp_p.unlink()
    shutil.rmtree(tmp_work_dir, ignore_errors=True)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate NeuroLens Testbench Visualizations")
    parser.add_argument(
        "--metrics", "--input", dest="metrics", default="results/testbench_v2_metrics.csv"
    )
    parser.add_argument("--epochs", default=None)
    parser.add_argument("--stats", default=None)
    parser.add_argument("--dataset", default="../data/images")
    parser.add_argument("--output-dir", default="results/plots")
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    epochs_path = (
        Path(args.epochs)
        if args.epochs
        else metrics_path.with_name(metrics_path.name.replace("_metrics", "_epochs"))
    )
    stats_path = (
        Path(args.stats)
        if args.stats
        else metrics_path.with_name("statistical_comparison.csv")
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_rows = _read_csv(metrics_path)
    epoch_rows = _read_csv(epochs_path)
    stats_rows = _read_csv(stats_path)

    print(
        f"Loaded {len(metric_rows)} metric row(s), {len(epoch_rows)} epoch row(s), "
        f"{len(stats_rows)} statistical comparison row(s)."
    )

    jobs = [
        ("01_training_dynamics (+01b, +01c)", lambda: plot_01_training_dynamics(epoch_rows, out_dir)),
        ("02_stage_tournament_progression", lambda: plot_02_tournament_progression(metric_rows, out_dir)),
        ("03_filter_slot_ablation (+03b)", lambda: plot_03_filter_slot_ablation(metric_rows, out_dir)),
        ("04_test_confusion_matrices", lambda: plot_04_confusion_matrices(metric_rows, out_dir)),
        ("05_per_class_metrics", lambda: plot_05_per_class_metrics(metric_rows, out_dir)),
        (
            "06_statistical_verification (+06b)",
            lambda: plot_06_statistical_verification(metric_rows, stats_rows, out_dir),
        ),
        (
            "07_config_image_effects (+07b)",
            lambda: plot_07_config_image_effects(args.dataset, metric_rows, out_dir),
        ),
    ]

    generated = 0
    for name, fn in jobs:
        ok = fn()
        status = "OK" if ok else "SKIPPED (no matching rows)"
        print(f"  [{status}] {name}")
        if ok:
            generated += 1

    print(f"Saved {generated} chart set(s) to '{out_dir}'.")


if __name__ == "__main__":
    main()