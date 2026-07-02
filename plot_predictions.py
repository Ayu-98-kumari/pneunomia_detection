import argparse
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from PIL import Image
from torchvision import datasets, transforms

from src.utils import load_config, get_device, set_seed
from src.models import get_model


CLASS_LABELS = ["NORMAL", "PNEUMONIA"]


def build_eval_transform(config):
    if "student_model" in config:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    t = [
        transforms.Grayscale(num_output_channels=config["num_channels"]),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
    if config.get("normalize", False):
        t.append(transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ))
    return transforms.Compose(t)


def load_model(config, checkpoint_path, device):
    is_distill = "student_model" in config
    model_cfg = {"model": config["student_model"]} if is_distill else config
    model = get_model(model_cfg, pretrained=False)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        threshold = ckpt.get("threshold", config.get("threshold", 0.5))
        print(f"Metadata checkpoint — alpha: {ckpt.get('alpha')}, threshold: {threshold}")
    else:
        model.load_state_dict(ckpt)
        threshold = config.get("threshold", 0.5)

    model = model.to(device)
    model.eval()
    return model, threshold


def run_predictions(model, dataset, threshold, device):
    correct, incorrect = [], []
    with torch.no_grad():
        for idx in range(len(dataset)):
            img_tensor, true_label = dataset[idx]
            logit = model(img_tensor.unsqueeze(0).to(device))
            prob = torch.sigmoid(logit).item()
            pred_label = 1 if prob >= threshold else 0
            entry = {
                "idx": idx,
                "true": CLASS_LABELS[true_label],
                "pred": CLASS_LABELS[pred_label],
                "prob": prob,
                "correct": true_label == pred_label,
            }
            if entry["correct"]:
                correct.append(entry)
            else:
                incorrect.append(entry)
    return correct, incorrect


def plot_grid(sampled, raw_test, model_label, threshold, n_correct, n_incorrect, output_path):
    cols = 4
    rows = (len(sampled) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.4))
    fig.patch.set_facecolor("#111827")

    fig.suptitle(
        f"Pneumonia Detection  —  {model_label.upper()}\n"
        f"Threshold: {threshold}   |   "
        f"Correct: {n_correct}   |   Incorrect: {n_incorrect}",
        fontsize=11, fontweight="bold", color="white", y=1.02,
    )

    flat_axes = np.array(axes).flatten()

    for i, ax in enumerate(flat_axes):
        if i >= len(sampled):
            ax.set_visible(False)
            continue

        r = sampled[i]
        img_path = raw_test.samples[r["idx"]][0]
        img = Image.open(img_path).convert("L")

        ax.imshow(img, cmap="bone")
        ax.set_xticks([])
        ax.set_yticks([])

        color = "#22c55e" if r["correct"] else "#ef4444"
        symbol = "✓" if r["correct"] else "✗"

        ax.set_title(
            f"{symbol}  True: {r['true']}\n    Pred: {r['pred']}  ({r['prob']:.2f})",
            fontsize=8.5, color=color, pad=5, fontweight="bold",
        )

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(color)
            spine.set_linewidth(2.5)

        ax.set_facecolor("#111827")

    legend_handles = [
        mpatches.Patch(color="#22c55e", label="Correct prediction"),
        mpatches.Patch(color="#ef4444", label="Incorrect prediction"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        fontsize=9,
        facecolor="#111827",
        edgecolor="#374151",
        labelcolor="white",
        bbox_to_anchor=(0.5, -0.03),
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved → {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot chest X-ray predictions for any trained model"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint path (overrides config)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Decision threshold (overrides checkpoint/config)")
    parser.add_argument("--num_images", type=int, default=16,
                        help="Total images to show (default 16, should be multiple of 4)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output PNG path (default: plots/predictions_<config>.png)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()
    set_seed(args.seed)

    checkpoint_path = args.checkpoint or config["checkpoint_path"]
    model, threshold = load_model(config, checkpoint_path, device)
    if args.threshold is not None:
        threshold = args.threshold

    is_distill = "student_model" in config
    model_label = config.get("student_model") if is_distill else config.get("model", "model")

    print(f"\nModel:      {model_label}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Threshold:  {threshold}")
    print(f"Device:     {device}\n")

    eval_transform = build_eval_transform(config)
    data_dir = config["data_dir"]

    test_dataset = datasets.ImageFolder(root=data_dir + "/test", transform=eval_transform)
    raw_test = datasets.ImageFolder(root=data_dir + "/test")

    print(f"Test images: {len(test_dataset)}  |  Classes: {raw_test.classes}")
    print("Running inference...")

    correct, incorrect = run_predictions(model, test_dataset, threshold, device)
    print(f"Correct: {len(correct)}  |  Incorrect: {len(incorrect)}\n")

    n = args.num_images
    n_incorrect = min(len(incorrect), n // 2)
    n_correct = min(len(correct), n - n_incorrect)

    sampled = random.sample(correct, n_correct) + random.sample(incorrect, n_incorrect)
    random.shuffle(sampled)

    config_name = os.path.splitext(os.path.basename(args.config))[0]
    output_path = args.output or f"plots/predictions_{config_name}.png"

    plot_grid(
        sampled, raw_test, model_label, threshold,
        len(correct), len(incorrect), output_path,
    )


if __name__ == "__main__":
    main()
