import argparse
import os

import torch

from src.utils import load_config, get_device, set_seed
from src.dataset import get_distillation_dataloaders
from src.models import get_model
from src.distill import load_frozen_teacher, train_distill, evaluate_distill


def main():
    parser = argparse.ArgumentParser(
        description="Train best distilled student (DeeperCNN) from frozen ResNet-18 teacher"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to distillation_best.yaml"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()
    set_seed(config["seed"])

    print(f"Config:            {args.config}")
    print(f"Student model:     {config['student_model']}")
    print(f"Teacher:           {config['teacher_checkpoint']}")
    print(f"Alpha:             {config['alpha']}")
    print(f"Threshold:         {config['threshold']}")
    print(f"Device:            {device}")
    print()

    train_loader, val_loader, test_loader = get_distillation_dataloaders(config)

    teacher = load_frozen_teacher(config, device)
    print()

    student_config = {"model": config["student_model"]}
    student = get_model(student_config, pretrained=False).to(device)

    total_params = sum(p.numel() for p in student.parameters())
    print(f"Student parameters: {total_params:,}")
    print()

    checkpoint_path = config["checkpoint_path"]
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    train_distill(
        student, teacher, train_loader, val_loader, config, device, checkpoint_path
    )

    print("\nLoading best student for evaluation...")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    student.load_state_dict(state_dict)
    student = student.to(device)

    evaluate_distill(student, test_loader, config["thresholds"], device)

    print("\nSaving final checkpoint with metadata...")
    final_checkpoint = {
        "model_state_dict": student.state_dict(),
        "model_name": "DeeperPneumoniaCNN",
        "teacher_model": "ResNet-18",
        "alpha": config["alpha"],
        "threshold": config["threshold"],
        "saved_based_on": "validation_accuracy",
        "input_type": "1-channel grayscale, 224x224",
        "class_names": ["NORMAL", "PNEUMONIA"],
    }

    final_path = checkpoint_path.replace(".pth", "_with_metadata.pth")
    torch.save(final_checkpoint, final_path)
    print(f"Final checkpoint with metadata saved to: {final_path}")


if __name__ == "__main__":
    main()
