"""
save_with_metadata.py — one-time script to embed threshold metadata into
existing plain-state-dict checkpoints for ResNet-18 and DeeperCNN.

Run once after training:
    python save_with_metadata.py

After this, all three key model checkpoints carry the same metadata format
as the distilled model checkpoint (model_state_dict + threshold + class_names).
"""

import os
import torch

CHECKPOINTS = [
    {
        "path": "checkpoints/pneumonia_resnet18.pth",
        "model": "resnet18",
        "threshold": 0.95,
        "class_names": ["NORMAL", "PNEUMONIA"],
    },
    {
        "path": "checkpoints/pneumonia_deeper_cnn.pth",
        "model": "deeper",
        "threshold": 0.9,
        "class_names": ["NORMAL", "PNEUMONIA"],
    },
]


def main():
    for entry in CHECKPOINTS:
        path = entry["path"]

        if not os.path.exists(path):
            print(f"Skipping {path} — file not found.")
            continue

        print(f"Processing {path} ...")
        ckpt = torch.load(path, map_location="cpu")

        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            ckpt["threshold"]   = entry["threshold"]
            ckpt["class_names"] = entry["class_names"]
            print(f"  Already metadata checkpoint — updated threshold to {entry['threshold']}")
        else:
            ckpt = {
                "model_state_dict": ckpt,
                "model":            entry["model"],
                "threshold":        entry["threshold"],
                "class_names":      entry["class_names"],
            }
            print(f"  Wrapped plain state dict — threshold={entry['threshold']}")

        torch.save(ckpt, path)
        print(f"  Saved → {path}\n")

    print("Done. Run plot_metrics.py or evaluate.py to verify.")


if __name__ == "__main__":
    main()
