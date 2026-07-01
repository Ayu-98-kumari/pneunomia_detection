import argparse

import torch

from src.utils import load_config, get_device, set_seed
from src.dataset import get_distillation_dataloaders
from src.models import get_model
from src.distill import evaluate_distill


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained distilled student model on the test set"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to distillation_best.yaml"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to checkpoint (overrides config)"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()
    set_seed(config["seed"])

    checkpoint_path = args.checkpoint or config["checkpoint_path"]

    print(f"Config:        {args.config}")
    print(f"Student model: {config['student_model']}")
    print(f"Checkpoint:    {checkpoint_path}")
    print(f"Device:        {device}")
    print()

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # handle both plain state dicts and metadata-rich distillation checkpoints
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        print(f"Metadata checkpoint detected:")
        print(f"  Teacher:   {checkpoint.get('teacher_model')}")
        print(f"  Alpha:     {checkpoint.get('alpha')}")
        print(f"  Threshold: {checkpoint.get('threshold')}")
        print(f"  Classes:   {checkpoint.get('class_names')}")
    else:
        state_dict = checkpoint
        print(f"Plain state dict checkpoint detected.")
        print(f"  Alpha:     {config.get('alpha')}")
        print(f"  Threshold: {config.get('threshold')}")

    print()

    student_config = {"model": config["student_model"]}
    model = get_model(student_config, pretrained=False)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print("Student model loaded successfully.")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
    print()

    _, _, test_loader = get_distillation_dataloaders(config)

    evaluate_distill(model, test_loader, config["thresholds"], device)


if __name__ == "__main__":
    main()
