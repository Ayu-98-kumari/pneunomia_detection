import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report


def run_inference(model, test_loader, device):
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            outputs = model(images)

            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).long().squeeze(1)

            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy().squeeze())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def print_metrics(labels, preds, class_names):
    cm = confusion_matrix(labels, preds)

    print("Confusion Matrix:")
    print(cm)
    print()
    print("Classification Report:")
    print(classification_report(labels, preds, target_names=class_names))


def threshold_analysis(labels, probs, thresholds, class_names):
    print("\n" + "=" * 60)
    print("THRESHOLD ANALYSIS")
    print("=" * 60)

    for threshold in thresholds:
        print(f"\nThreshold: {threshold}")
        print("-" * 40)

        preds = (probs >= threshold).astype(int)

        cm = confusion_matrix(labels, preds)
        print("Confusion Matrix:")
        print(cm)
        print()
        print(classification_report(
            labels, preds, target_names=class_names, zero_division=0
        ))


def evaluate_model(model, test_loader, config, device):
    class_names = ["NORMAL", "PNEUMONIA"]
    thresholds = config["thresholds"]

    print("Running evaluation on test set...")
    labels, preds, probs = run_inference(model, test_loader, device)

    print("\n" + "=" * 60)
    print("TEST RESULTS (threshold=0.5)")
    print("=" * 60 + "\n")
    print_metrics(labels, preds, class_names)

    threshold_analysis(labels, probs, thresholds, class_names)
