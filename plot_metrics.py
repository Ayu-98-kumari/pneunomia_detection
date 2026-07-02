"""
plot_metrics.py — generate 4 metric plots from all trained models.

Outputs (saved to plots/ by default):
  confusion_matrices.png  — one confusion matrix per model (2 x 3 grid)
  roc_curves.png          — ROC curves for all models overlaid on one chart
  model_comparison.png    — grouped bar chart (accuracy + macro F1)
  threshold_sweep.png     — precision / recall / F1 / accuracy vs threshold
                            for the distilled model

Usage:
    python plot_metrics.py
    python plot_metrics.py --output_dir results/plots
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.models import get_model
from src.utils import get_device, load_config, set_seed

# ── colour palette ──────────────────────────────────────────────────────────
BG      = "#111827"
PANEL   = "#1f2937"
COLOURS = ["#60a5fa", "#34d399", "#f87171", "#fbbf24", "#a78bfa", "#fb923c"]
GRAY    = "gray"

MODEL_CONFIGS = [
    ("configs/deeper_cnn.yaml",        "DeeperCNN (standalone)"),
    ("configs/resnet18.yaml",          "ResNet-18 (teacher)"),
    ("configs/distillation_best.yaml", "Distilled CNN (student)"),
]

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


# ── helpers ─────────────────────────────────────────────────────────────────

def _build_eval_transform(config):
    is_distill = "student_model" in config
    num_ch = 1 if is_distill else config["num_channels"]
    t = [
        transforms.Grayscale(num_output_channels=num_ch),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
    if not is_distill and config.get("normalize", False):
        t.append(transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ))
    return transforms.Compose(t)


def _load_model(config, device):
    checkpoint_path = config["checkpoint_path"]
    if not os.path.exists(checkpoint_path):
        return None, None

    is_distill = "student_model" in config
    model_cfg = {"model": config["student_model"]} if is_distill else config
    model = get_model(model_cfg, pretrained=False)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        threshold = ckpt.get("threshold", config.get("threshold", 0.5))
    else:
        model.load_state_dict(ckpt)
        threshold = config.get("threshold", 0.5)

    model = model.to(device).eval()
    return model, threshold


def _run_inference(model, config, device):
    """Returns (true_labels, pred_probs) as numpy arrays."""
    eval_tf = _build_eval_transform(config)
    test_ds = datasets.ImageFolder(root=config["data_dir"] + "/test", transform=eval_tf)
    loader  = DataLoader(test_ds, batch_size=32, shuffle=False)

    all_labels, all_probs = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            logits = model(imgs.to(device))
            probs  = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)

    return np.array(all_labels), np.array(all_probs)


def _dark_axes(ax, title=""):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color(GRAY)
    if title:
        ax.set_title(title, color="white", fontweight="bold", fontsize=11)


# ── 1. confusion matrices ────────────────────────────────────────────────────

def plot_confusion_matrices(results, output_path):
    n = len(results)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 4))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Confusion Matrices — All Models", fontsize=14,
                 color="white", fontweight="bold", y=1.01)

    flat = np.array(axes).flatten()

    for i, ax in enumerate(flat):
        if i >= n:
            ax.set_visible(False)
            continue

        name, true_y, pred_p, thr = results[i]
        pred_y = (pred_p >= thr).astype(int)
        cm = confusion_matrix(true_y, pred_y)

        im = ax.imshow(cm, cmap="Blues", aspect="auto")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(CLASS_NAMES, color="white", fontsize=9)
        ax.set_yticklabels(CLASS_NAMES, color="white", fontsize=9,
                           rotation=90, va="center")
        ax.set_xlabel("Predicted", color="white", fontsize=9)
        ax.set_ylabel("True", color="white", fontsize=9)
        _dark_axes(ax, f"{name}\n(threshold={thr})")

        for r in range(2):
            for c in range(2):
                val = cm[r, c]
                txt_color = "white" if val > cm.max() / 2 else "black"
                ax.text(c, r, str(val), ha="center", va="center",
                        color=txt_color, fontsize=16, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved → {output_path}")
    plt.close(fig)


# ── 2. ROC curves ────────────────────────────────────────────────────────────

def plot_roc_curves(results, output_path):
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor(BG)
    _dark_axes(ax, "ROC Curves — All Models")

    for i, (name, true_y, pred_p, _) in enumerate(results):
        fpr, tpr, _ = roc_curve(true_y, pred_p)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=COLOURS[i], lw=2,
                label=f"{name}  (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "w--", lw=1, alpha=0.4)
    ax.set_xlabel("False Positive Rate", color="white")
    ax.set_ylabel("True Positive Rate", color="white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(facecolor=PANEL, labelcolor="white", edgecolor=GRAY, fontsize=9)
    ax.yaxis.grid(True, color=GRAY, alpha=0.25)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved → {output_path}")
    plt.close(fig)


# ── 3. model comparison bar chart ────────────────────────────────────────────

def _best_threshold(true_y, pred_p):
    """Return the threshold that maximises macro F1 on the given predictions."""
    best_thr, best_f1 = 0.5, 0.0
    for t in np.arange(0.05, 0.96, 0.025):
        pred_y = (pred_p >= t).astype(int)
        f1 = f1_score(true_y, pred_y, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, t
    return round(float(best_thr), 3)


def plot_model_comparison(results, output_path):
    names, accs, f1s, best_thrs = [], [], [], []
    for name, true_y, pred_p, thr in results:
        pred_y = (pred_p >= thr).astype(int)
        names.append(name)
        accs.append(accuracy_score(true_y, pred_y))
        f1s.append(f1_score(true_y, pred_y, average="macro"))
        best_thrs.append(thr)
        print(f"  {name:<20} threshold={thr}  "
              f"acc={accs[-1]:.3f}  f1={f1s[-1]:.3f}")

    x     = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(BG)
    _dark_axes(ax, "Model Comparison — Accuracy and Macro F1\n"
                   "(each model evaluated at its best threshold)")

    b1 = ax.bar(x - width / 2, accs, width, label="Accuracy",  color="#60a5fa", alpha=0.9)
    b2 = ax.bar(x + width / 2, f1s,  width, label="Macro F1", color="#34d399", alpha=0.9)

    for bar in (*b1, *b2):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", color="white", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{n}\n(thr={t})" for n, t in zip(names, best_thrs)],
        rotation=0, ha="center", color="white", fontsize=8.5,
    )
    ax.set_ylabel("Score", color="white")
    ax.set_ylim(0, 1.12)
    ax.yaxis.grid(True, color=GRAY, alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(facecolor=PANEL, labelcolor="white", edgecolor=GRAY)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved → {output_path}")
    plt.close(fig)


# ── 4. threshold sweep (distilled model) ────────────────────────────────────

def plot_threshold_sweep(true_y, pred_p, best_threshold, output_path):
    thresholds = np.arange(0.05, 1.0, 0.025)
    precisions, recalls, f1s, accs = [], [], [], []

    for t in thresholds:
        pred_y = (pred_p >= t).astype(int)
        precisions.append(precision_score(true_y, pred_y, zero_division=0))
        recalls.append(recall_score(true_y, pred_y, zero_division=0))
        f1s.append(f1_score(true_y, pred_y, average="macro", zero_division=0))
        accs.append(accuracy_score(true_y, pred_y))

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(BG)
    _dark_axes(ax, "Distilled CNN — Metrics vs Decision Threshold")

    ax.plot(thresholds, precisions, color="#60a5fa", lw=2,   label="Precision")
    ax.plot(thresholds, recalls,    color="#f87171", lw=2,   label="Recall")
    ax.plot(thresholds, f1s,        color="#34d399", lw=2.5, label="Macro F1", linestyle="--")
    ax.plot(thresholds, accs,       color="#fbbf24", lw=2,   label="Accuracy")

    ax.axvline(x=best_threshold, color="white", linestyle=":",
               alpha=0.7, label=f"Best threshold ({best_threshold})")

    ax.set_xlabel("Threshold", color="white")
    ax.set_ylabel("Score", color="white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.yaxis.grid(True, color=GRAY, alpha=0.25)
    ax.xaxis.grid(True, color=GRAY, alpha=0.15)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(facecolor=PANEL, labelcolor="white", edgecolor=GRAY)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved → {output_path}")
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate metric plots for all trained models"
    )
    parser.add_argument("--output_dir", default="plots",
                        help="Directory to save plots (default: plots/)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = get_device()
    set_seed(args.seed)

    results       = []   # (name, true_labels, pred_probs, threshold)
    distill_entry = None

    for config_path, model_name in MODEL_CONFIGS:
        print(f"\n[{model_name}]  {config_path}")
        if not os.path.exists(config_path):
            print("  Config not found — skipping.")
            continue

        config = load_config(config_path)
        model, threshold = _load_model(config, device)
        if model is None:
            print(f"  Checkpoint not found ({config['checkpoint_path']}) — skipping.")
            continue

        print(f"  Threshold: {threshold}  |  Running inference...")
        true_y, pred_p = _run_inference(model, config, device)
        results.append((model_name, true_y, pred_p, threshold))
        print(f"  Done — {len(true_y)} test images.")

        if "distillation" in config_path:
            distill_entry = (true_y, pred_p, threshold)

    if not results:
        print("\nNo checkpoints found. Train the models first, then re-run.")
        return

    print(f"\nGenerating plots for {len(results)} model(s)...\n")

    plot_confusion_matrices(
        results,
        os.path.join(args.output_dir, "confusion_matrices.png"),
    )
    plot_roc_curves(
        results,
        os.path.join(args.output_dir, "roc_curves.png"),
    )
    plot_model_comparison(
        results,
        os.path.join(args.output_dir, "model_comparison.png"),
    )

    if distill_entry is not None:
        true_y, pred_p, best_thr = distill_entry
        plot_threshold_sweep(
            true_y, pred_p, best_thr,
            os.path.join(args.output_dir, "threshold_sweep.png"),
        )
    else:
        print("Distilled model checkpoint not found — threshold_sweep.png skipped.")

    print(f"\nDone. All plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
