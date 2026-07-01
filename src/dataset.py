import random
from collections import Counter

import torch
from PIL import Image
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler


def get_transforms(config):
    num_channels = config["num_channels"]
    augment = config["augmentation"]
    normalize = config["normalize"]

    train_transform_list = [
        transforms.Grayscale(num_output_channels=num_channels),
        transforms.Resize((224, 224)),
    ]

    if augment:
        train_transform_list.append(transforms.RandomRotation(degrees=10))
        train_transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

    train_transform_list.append(transforms.ToTensor())

    if normalize:
        train_transform_list.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        )

    eval_transform_list = [
        transforms.Grayscale(num_output_channels=num_channels),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]

    if normalize:
        eval_transform_list.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        )

    train_transform = transforms.Compose(train_transform_list)
    eval_transform = transforms.Compose(eval_transform_list)

    return train_transform, eval_transform


def create_weighted_sampler(dataset):
    labels = []
    for _, label in dataset:
        labels.append(label)

    class_counts = Counter(labels)
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}

    sample_weights = [class_weights[label] for label in labels]
    sample_weights = torch.DoubleTensor(sample_weights)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    return sampler


def get_dataloaders(config):
    data_dir = config["data_dir"]
    batch_size = config["batch_size"]
    seed = config["seed"]
    use_weighted_sampling = config["weighted_sampling"]

    train_transform, eval_transform = get_transforms(config)

    train_full_augmented = datasets.ImageFolder(
        root=data_dir + "/train",
        transform=train_transform,
    )

    train_full_plain = datasets.ImageFolder(
        root=data_dir + "/train",
        transform=eval_transform,
    )

    test_dataset = datasets.ImageFolder(
        root=data_dir + "/test",
        transform=eval_transform,
    )

    train_size = int(0.8 * len(train_full_augmented))
    val_size = len(train_full_augmented) - train_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, _ = random_split(
        train_full_augmented, [train_size, val_size], generator=generator
    )

    generator = torch.Generator().manual_seed(seed)
    _, val_dataset = random_split(
        train_full_plain, [train_size, val_size], generator=generator
    )

    sampler = None
    shuffle_train = True

    if use_weighted_sampling:
        sampler = create_weighted_sampler(train_dataset)
        shuffle_train = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        sampler=sampler,
    )

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    class_names = train_full_augmented.classes

    print(f"Classes: {class_names}")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    return train_loader, val_loader, test_loader, class_names


# ── Distillation dataset ──────────────────────────────────────────────────────

_STUDENT_TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

_TEACHER_TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class DualTransformDataset(Dataset):
    """Returns (student_image, teacher_image, label) for distillation."""

    def __init__(self, base_dataset, indices, augment=False):
        self.base_dataset = base_dataset
        self.indices = indices
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image_path, label = self.base_dataset.samples[self.indices[idx]]
        image = Image.open(image_path).convert("RGB")

        if self.augment:
            angle = random.uniform(-10, 10)
            image = TF.rotate(image, angle)
            if random.random() < 0.5:
                image = TF.hflip(image)

        return _STUDENT_TRANSFORM(image), _TEACHER_TRANSFORM(image), label


def get_distillation_dataloaders(config):
    data_dir = config["data_dir"]
    batch_size = config["batch_size"]
    seed = config["seed"]

    raw_train = datasets.ImageFolder(root=data_dir + "/train")
    raw_test = datasets.ImageFolder(root=data_dir + "/test")

    train_size = int(0.8 * len(raw_train))
    val_size = len(raw_train) - train_size

    all_indices = torch.randperm(
        len(raw_train), generator=torch.Generator().manual_seed(seed)
    ).tolist()

    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size:]
    test_indices = list(range(len(raw_test)))

    train_dataset = DualTransformDataset(raw_train, train_indices, augment=True)
    val_dataset = DualTransformDataset(raw_train, val_indices, augment=False)
    test_dataset = DualTransformDataset(raw_test, test_indices, augment=False)

    train_labels = [raw_train.samples[i][1] for i in train_indices]
    class_counts = Counter(train_labels)
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = torch.DoubleTensor([class_weights[l] for l in train_labels])

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Classes: {raw_train.classes}")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    print(f"Class counts in train: {dict(class_counts)}")

    return train_loader, val_loader, test_loader
