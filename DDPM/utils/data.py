"""
Data loading utilities for DDPM training.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple


def get_cifar10_transforms(image_size: int = 32) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get transforms for CIFAR-10 dataset.

    Args:
        image_size: Target image size

    Returns:
        Tuple of (train_transforms, test_transforms)
    """
    train_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # Scale to [-1, 1]
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    return train_transforms, test_transforms


def get_cifar10_dataloader(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
    image_size: int = 32,
    train: bool = True,
    shuffle: bool = True,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Get CIFAR-10 dataloader.

    Args:
        data_dir: Directory to store/load data
        batch_size: Batch size
        num_workers: Number of data loading workers
        image_size: Target image size
        train: Whether to load training set
        shuffle: Whether to shuffle data
        pin_memory: Whether to pin memory for faster GPU transfer

    Returns:
        DataLoader instance
    """
    train_transforms, test_transforms = get_cifar10_transforms(image_size)
    transform = train_transforms if train else test_transforms

    dataset = datasets.CIFAR10(
        root=data_dir,
        train=train,
        download=True,
        transform=transform,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=train,  # Drop last incomplete batch during training
    )

    return dataloader


def unnormalize(x: torch.Tensor) -> torch.Tensor:
    """
    Unnormalize images from [-1, 1] to [0, 1].

    Args:
        x: Images in range [-1, 1]

    Returns:
        Images in range [0, 1]
    """
    return (x + 1) / 2


def normalize(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize images from [0, 1] to [-1, 1].

    Args:
        x: Images in range [0, 1]

    Returns:
        Images in range [-1, 1]
    """
    return x * 2 - 1
