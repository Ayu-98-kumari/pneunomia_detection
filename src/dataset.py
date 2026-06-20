from collections import Counter

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler


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
