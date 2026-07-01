import argparse
import os
import copy

import torch

from src.utils import load_config, get_device, set_seed
from src.dataset import get_distillation_dataloaders
from src.models import get_model
from src.distill import load_frozen_teacher, train_distill, evaluate_distill


def main():
    parser = argparse.ArgumentParser(
        description="Alpha sweep: train distilled student across multiple alpha values for comparison"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to distillation_sweep.yaml"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()
    set_seed(config["seed"])

    alphas = config["alphas"]
    checkpoint_dir = config["checkpoint_dir"]
    thresholds = config["thresholds"]

    print(f"Config:        {args.config}")
    print(f"Student model: {config['student_model']}")
    print(f"Teacher:       {config['teacher_checkpoint']}")
    print(f"Alpha sweep:   {alphas}")
    print(f"Device:        {device}")
    print()

    train_loader, val_loader, test_loader = get_distillation_dataloaders(config)

    teacher = load_frozen_teacher(config, device)
    print()

    sweep_results = []

    for alpha in alphas:
        print("\n" + "=" * 70)
        print(f"ALPHA = {alpha}")
        print("=" * 70)

        run_config = copy.deepcopy(config)
        run_config["alpha"] = alpha

        student_config = {"model": config["student_model"]}
        student = get_model(student_config, pretrained=False).to(device)

        alpha_str = str(alpha).replace(".", "")
        checkpoint_path = os.path.join(
            checkpoint_dir, f"pneumonia_distilled_sweep_alpha{alpha_str}.pth"
        )

        best_val_acc = train_distill(
            student, teacher, train_loader, val_loader,
            run_config, device, checkpoint_path
        )

        print(f"\nLoading best student (alpha={alpha}) for evaluation...")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        student.load_state_dict(state_dict)
        student = student.to(device)

        evaluate_distill(student, test_loader, thresholds, device)

        sweep_results.append({
            "alpha": alpha,
            "best_val_acc": best_val_acc,
            "checkpoint": checkpoint_path
        })

    print("\n" + "=" * 70)
    print("SWEEP SUMMARY")
    print("=" * 70)
    for result in sweep_results:
        print(
            f"Alpha: {result['alpha']} "
            f"| Best Val Acc: {result['best_val_acc']:.4f} "
            f"| Checkpoint: {result['checkpoint']}"
        )


if __name__ == "__main__":
    main()
