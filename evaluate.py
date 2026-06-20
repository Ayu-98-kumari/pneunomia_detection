import argparse

import torch

from src.utils import load_config, get_device, set_seed
from src.dataset import get_dataloaders
from src.models import get_model
from src.evaluate import evaluate_model


def main():
    parser = argparse.ArgumentParser(description="Evaluate pneumonia detection model")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config YAML file"
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

    print(f"Config: {args.config}")
    print(f"Model: {config['model']}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print()

    _, _, test_loader, _ = get_dataloaders(config)

    model = get_model(config, pretrained=False)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print("Model loaded successfully.\n")

    evaluate_model(model, test_loader, config, device)


if __name__ == "__main__":
    main()
