"""
compare_models.py — side-by-side comparison of the three key models.

Outputs (saved to plots/ by default):
  comparison_predictions.png  — 3 models × 8 random X-rays (rows=models, cols=images)
  comparison_metrics.png      — confusion matrices, ROC curves, bar chart

Usage:
    python compare_models.py
    python compare_models.py --num_images 8 --seed 42 --output_dir plots
"""

import argparse
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score, auc, confusion_matrix, f1_score, roc_curve,
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.models import get_model
from src.utils import get_device, load_config, set_seed


# ── constants ────────────────────────────────────────────────────────────────

BG      = "#111827"
PANEL   = "#1f2937"
CORRECT = "#22c55e"
WRONG   = "#ef4444"
COLOURS = ["#60a5fa", "#a78bfa", "#34d399"]   # DeeperCNN, ResNet-18, Distilled
GRAY    = "gray"
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

MODEL_DEFS = [
    {
        "name":        "DeeperCNN\n(standalone)",
        "short":       "DeeperCNN",
        "config_path": "configs/deeper_cnn.yaml",
    },
    {
        "name":        "ResNet-18\n(teacher)",
        "short":       "ResNet-18",
        "config_path": "configs/resnet18.yaml",
    },
    {
        "name":        "Distilled CNN\n(student)",
        "short":       "Distilled CNN",
        "config_path": "configs/distillation_best.yaml",
    },
]


# ── helpers ──────────────────────────────────────────────────────────────────

def _build_transform(config):
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


def _load_entry(defn, device):
    config    = load_config(defn["config_path"])
    ckpt_path = config["checkpoint_path"]
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    is_distill = "student_model" in config
    model_cfg  = {"model": config["student_model"]} if is_distill else config
    model      = get_model(model_cfg, pretrained=False)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        threshold = ckpt.get("threshold", config.get("threshold", 0.5))
    else:
        model.load_state_dict(ckpt)
        threshold = config.get("threshold", 0.5)

    return {
        **defn,
        "model":     model.to(device).eval(),
        "transform": _build_transform(config),
        "threshold": threshold,
        "data_dir":  config["data_dir"],
    }


def _predict(model, transform, img_rgb, device):
    tensor = transform(img_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        return torch.sigmoid(model(tensor)).item()


def _full_inference(model, transform, data_dir, device):
    ds     = datasets.ImageFolder(root=data_dir + "/test", transform=transform)
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    labels, probs = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            p = torch.sigmoid(model(imgs.to(device))).squeeze(1).cpu().numpy()
            labels.extend(lbls.numpy())
            probs.extend(p)
    return np.array(labels), np.array(probs)


def _style_ax(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_color(GRAY)


# ── Figure 1: prediction grid ─────────────────────────────────────────────────
# Layout: rows = models (3), cols = images (num_images)
# Each cell: X-ray image + prediction + confidence + coloured border
# True label shown as column header on the first row only

def plot_prediction_grid(entries, num_images, device, output_path):
    data_dir = entries[0]["data_dir"]
    raw_test = datasets.ImageFolder(root=data_dir + "/test")
    indices  = random.sample(range(len(raw_test)), num_images)

    imgs_display, imgs_model, true_labels = [], [], []
    for idx in indices:
        path, label = raw_test.samples[idx]
        imgs_display.append(Image.open(path).convert("L"))
        imgs_model.append(Image.open(path).convert("RGB"))
        true_labels.append(label)

    n_rows = len(entries)
    n_cols = num_images

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 2.8, n_rows * 4.0),
                              gridspec_kw={"hspace": 0.55, "wspace": 0.12})
    fig.patch.set_facecolor(BG)
    # reserve left margin so row labels don't touch the first column's border
    fig.subplots_adjust(left=0.09)
    fig.suptitle(
        "Chest X-Ray Predictions — DeeperCNN vs ResNet-18 vs Distilled CNN",
        color="white", fontweight="bold", fontsize=12, y=0.99,
    )

    for row, entry in enumerate(entries):
        for col in range(n_cols):
            ax         = axes[row, col]
            true_label = true_labels[col]
            prob       = _predict(entry["model"], entry["transform"],
                                  imgs_model[col], device)
            pred       = 1 if prob >= entry["threshold"] else 0
            ok         = pred == true_label

            ax.imshow(imgs_display[col], cmap="bone")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor(BG)

            color  = CORRECT if ok else WRONG
            symbol = "✓" if ok else "✗"

            # show true label as column header on first row only
            header = f"True: {CLASS_NAMES[true_label]}\n" if row == 0 else ""
            ax.set_title(
                f"{header}{symbol} {CLASS_NAMES[pred]}\n({prob:.2f})",
                color=color, fontsize=7.5, fontweight="bold", pad=3,
            )

            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_edgecolor(color)
                sp.set_linewidth(2.5)

    # row labels: use fig.text so they sit in the reserved left margin,
    # completely clear of the first column's spine border
    for row, entry in enumerate(entries):
        pos = axes[row, 0].get_position()
        y_mid = (pos.y0 + pos.y1) / 2
        fig.text(
            0.01, y_mid,
            f"{entry['short']}\n(thr={entry['threshold']})",
            color=COLOURS[row], fontsize=8.5, fontweight="bold",
            ha="left", va="center", rotation=90,
        )

    correct_p = mpatches.Patch(color=CORRECT, label="Correct prediction")
    wrong_p   = mpatches.Patch(color=WRONG,   label="Incorrect prediction")
    fig.legend(
        handles=[correct_p, wrong_p], loc="lower center", ncol=2,
        fontsize=9, facecolor=BG, edgecolor=GRAY, labelcolor="white",
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"Saved → {output_path}")
    plt.close(fig)


# ── Figure 2: metrics ─────────────────────────────────────────────────────────
# Layout: top row = 3 confusion matrices, bottom left = ROC curves, bottom right = bar chart

def plot_metrics_figure(entries, device, output_path):
    data_dir = entries[0]["data_dir"]
    print("Running full test-set inference for metrics...")
    results = []
    for entry in entries:
        true_y, pred_p = _full_inference(
            entry["model"], entry["transform"], data_dir, device
        )
        results.append({**entry, "true_y": true_y, "pred_p": pred_p})
        print(f"  {entry['short']:<15} threshold={entry['threshold']}  "
              f"images={len(true_y)}")

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor(BG)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # top row: one confusion matrix per model
    for col, r in enumerate(results):
        ax     = fig.add_subplot(gs[0, col])
        pred_y = (r["pred_p"] >= r["threshold"]).astype(int)
        cm     = confusion_matrix(r["true_y"], pred_y)

        ax.imshow(cm, cmap="Blues", aspect="auto")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(CLASS_NAMES, color="white", fontsize=8)
        ax.set_yticklabels(CLASS_NAMES, color="white", fontsize=8,
                           rotation=90, va="center")
        ax.set_xlabel("Predicted", color="white", fontsize=8)
        ax.set_ylabel("True",      color="white", fontsize=8)
        _style_ax(ax)
        ax.set_title(f"{r['short']}  (thr={r['threshold']})",
                     color=COLOURS[col], fontweight="bold", fontsize=10)

        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                ax.text(j, i, str(val), ha="center", va="center",
                        color="white" if val > cm.max() / 2 else "black",
                        fontsize=14, fontweight="bold")

    # bottom left: ROC curves (spans 2 columns)
    ax_roc = fig.add_subplot(gs[1, :2])
    _style_ax(ax_roc)
    ax_roc.set_title("ROC Curves", color="white", fontweight="bold", fontsize=11)

    for i, r in enumerate(results):
        fpr, tpr, _ = roc_curve(r["true_y"], r["pred_p"])
        roc_auc     = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, color=COLOURS[i], lw=2,
                    label=f"{r['short']}  (AUC = {roc_auc:.3f})")

    ax_roc.plot([0, 1], [0, 1], "w--", lw=1, alpha=0.4)
    ax_roc.set_xlabel("False Positive Rate", color="white")
    ax_roc.set_ylabel("True Positive Rate",  color="white")
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1.02)
    ax_roc.yaxis.grid(True, color=GRAY, alpha=0.25)
    ax_roc.legend(facecolor=PANEL, labelcolor="white", edgecolor=GRAY, fontsize=9)

    # bottom right: bar chart
    ax_bar = fig.add_subplot(gs[1, 2])
    _style_ax(ax_bar)
    ax_bar.set_title("Accuracy & Macro F1\n(at best threshold)",
                     color="white", fontweight="bold", fontsize=10)

    names = [r["short"] for r in results]
    accs  = [accuracy_score(r["true_y"],
                            (r["pred_p"] >= r["threshold"]).astype(int))
             for r in results]
    f1s   = [f1_score(r["true_y"],
                      (r["pred_p"] >= r["threshold"]).astype(int),
                      average="macro")
             for r in results]

    x     = np.arange(len(names))
    width = 0.35
    b1 = ax_bar.bar(x - width / 2, accs, width,
                    color="#60a5fa", alpha=0.9, label="Accuracy")
    b2 = ax_bar.bar(x + width / 2, f1s,  width,
                    color="#34d399", alpha=0.9, label="Macro F1")

    for bar in (*b1, *b2):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom", color="white", fontsize=7.5,
        )

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(names, color="white", fontsize=8,
                            rotation=15, ha="right")
    ax_bar.set_ylabel("Score", color="white")
    ax_bar.set_ylim(0, 1.12)
    ax_bar.yaxis.grid(True, color=GRAY, alpha=0.25)
    ax_bar.set_axisbelow(True)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.legend(facecolor=PANEL, labelcolor="white", edgecolor=GRAY, fontsize=8)

    fig.suptitle(
        "Model Comparison — DeeperCNN vs ResNet-18 vs Distilled CNN",
        color="white", fontweight="bold", fontsize=13, y=1.01,
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"Saved → {output_path}")
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Side-by-side comparison of DeeperCNN, ResNet-18 and Distilled CNN"
    )
    parser.add_argument("--num_images", type=int, default=6,
                        help="Number of X-ray images to show per model row (default: 6)")
    parser.add_argument("--output_dir", default="plots")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = get_device()
    set_seed(args.seed)

    print("Loading models...")
    entries = [_load_entry(defn, device) for defn in MODEL_DEFS]
    for e in entries:
        print(f"  {e['short']:<15} threshold={e['threshold']}")
    print()

    print(f"Generating prediction grid "
          f"({len(entries)} models × {args.num_images} images)...")
    plot_prediction_grid(
        entries, args.num_images, device,
        os.path.join(args.output_dir, "comparison_predictions.png"),
    )

    print("\nGenerating metrics figure...")
    plot_metrics_figure(
        entries, device,
        os.path.join(args.output_dir, "comparison_metrics.png"),
    )

    print(f"\nDone. Both plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
