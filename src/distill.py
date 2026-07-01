import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report

from src.models import get_model


def load_frozen_teacher(config, device):
    teacher_config = {"model": "resnet18"}
    teacher = get_model(teacher_config, pretrained=False)

    state_dict = torch.load(config["teacher_checkpoint"], map_location="cpu")
    teacher.load_state_dict(state_dict)

    teacher = teacher.to(device)
    teacher.eval()

    for param in teacher.parameters():
        param.requires_grad = False

    frozen = not any(p.requires_grad for p in teacher.parameters())
    print(f"Teacher loaded from: {config['teacher_checkpoint']}")
    print(f"Teacher frozen: {frozen}")

    return teacher


def train_one_epoch_distill(student, teacher, train_loader, optimizer, alpha, device):
    student.train()
    teacher.eval()

    hard_criterion = nn.BCEWithLogitsLoss()
    soft_criterion = nn.BCEWithLogitsLoss()

    running_total_loss = 0.0
    running_soft_loss = 0.0
    running_hard_loss = 0.0

    all_labels = []
    all_preds = []

    for batch_idx, (student_images, teacher_images, labels) in enumerate(train_loader):
        student_images = student_images.to(device)
        teacher_images = teacher_images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()

        with torch.no_grad():
            teacher_logits = teacher(teacher_images)
            teacher_probs = torch.sigmoid(teacher_logits).detach()

        student_logits = student(student_images)

        soft_loss = soft_criterion(student_logits, teacher_probs)
        hard_loss = hard_criterion(student_logits, labels)
        total_loss = alpha * soft_loss + (1 - alpha) * hard_loss

        total_loss.backward()
        optimizer.step()

        running_total_loss += total_loss.item()
        running_soft_loss += soft_loss.item()
        running_hard_loss += hard_loss.item()

        probs = torch.sigmoid(student_logits)
        preds = (probs >= 0.5).long()

        all_labels.extend(labels.cpu().long().numpy().squeeze())
        all_preds.extend(preds.cpu().numpy().squeeze())

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
            current_loss = running_total_loss / (batch_idx + 1)
            current_acc = accuracy_score(all_labels, all_preds)
            current_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
            print(
                f"  Batch {batch_idx+1}/{len(train_loader)} "
                f"| Total Loss: {current_loss:.4f} "
                f"| Soft: {running_soft_loss/(batch_idx+1):.4f} "
                f"| Hard: {running_hard_loss/(batch_idx+1):.4f} "
                f"| Acc: {current_acc:.4f} "
                f"| Macro F1: {current_f1:.4f}"
            )

    epoch_loss = running_total_loss / len(train_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return epoch_loss, epoch_acc, epoch_f1


def validate_distill(student, val_loader, device):
    student.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for student_images, _, labels in val_loader:
            student_images = student_images.to(device)

            student_logits = student(student_images)

            probs = torch.sigmoid(student_logits)
            preds = (probs >= 0.5).long().squeeze(1)

            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())

    val_acc = accuracy_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    val_precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    val_recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)

    return val_acc, val_f1, val_precision, val_recall


def train_distill(student, teacher, train_loader, val_loader, config, device, checkpoint_path):
    num_epochs = config["num_epochs"]
    lr = config["lr"]
    alpha = config["alpha"]

    optimizer = optim.Adam(student.parameters(), lr=lr)

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs} | Alpha: {alpha}")
        print("-" * 60)

        train_loss, train_acc, train_f1 = train_one_epoch_distill(
            student, teacher, train_loader, optimizer, alpha, device
        )

        val_acc, val_f1, val_precision, val_recall = validate_distill(
            student, val_loader, device
        )

        print(
            f"\n  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train Macro F1: {train_f1:.4f}"
        )
        print(
            f"  Val Acc: {val_acc:.4f} | Val Precision: {val_precision:.4f} "
            f"| Val Recall: {val_recall:.4f} | Val Macro F1: {val_f1:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(student.state_dict(), checkpoint_path)
            print(f"  -> Best model saved (Val Acc: {best_val_acc:.4f})")

    print(f"\nTraining finished. Best Val Acc: {best_val_acc:.4f}")
    return best_val_acc


def evaluate_distill(model, test_loader, thresholds, device):
    model.eval()

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for student_images, _, labels in test_loader:
            student_images = student_images.to(device)

            outputs = model(student_images)
            probs = torch.sigmoid(outputs)

            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy().squeeze())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    for threshold in thresholds:
        preds = (all_probs >= threshold).astype(int)

        cm = confusion_matrix(all_labels, preds)

        print(f"\nThreshold: {threshold}")
        print("Confusion Matrix:")
        print(cm)
        print(classification_report(
            all_labels, preds,
            target_names=["NORMAL", "PNEUMONIA"],
            zero_division=0
        ))

    return all_labels, all_probs
