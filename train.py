import argparse

from src.utils import load_config, get_device, set_seed
from src.dataset import get_dataloaders
from src.models import get_model
from src.train import train_model


def main():
    parser = argparse.ArgumentParser(description="Train pneumonia detection model")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config YAML file"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()
    set_seed(config["seed"])

    print(f"Config: {args.config}")
    print(f"Model: {config['model']}")
    print(f"Device: {device}")
    print()

    train_loader, val_loader, _, _ = get_dataloaders(config)

    model = get_model(config, pretrained=True).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    train_model(model, train_loader, val_loader, config, device)


if __name__ == "__main__":
    main()
