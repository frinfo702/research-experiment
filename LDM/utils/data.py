"""
Data loading utilities for DDPM training.
Supports CelebA-HQ, LSUN-Churches, and CIFAR-10.
Auto-downloads datasets if not present.
"""

import os
import subprocess
import zipfile
from pathlib import Path
from typing import Tuple, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image


def get_transforms(image_size: int, train: bool = True) -> transforms.Compose:
    """
    Get transforms for image datasets.

    Args:
        image_size: Target image size
        train: Whether this is for training (enables augmentation)

    Returns:
        Compose transform
    """
    transform_list = [
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
    ]
    if train:
        transform_list.append(transforms.RandomHorizontalFlip())
    transform_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # Scale to [-1, 1]
        ]
    )
    return transforms.Compose(transform_list)


class ImageFolderDataset(Dataset):
    """
    Generic image folder dataset.
    Expects images directly in data_dir or in subdirectories.
    """

    def __init__(
        self,
        data_dir: str,
        transform: Optional[transforms.Compose] = None,
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp"),
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.extensions = extensions
        self.image_paths = self._find_images()

    def _find_images(self):
        """Find all image files in the directory."""
        image_paths = []
        for ext in self.extensions:
            image_paths.extend(self.data_dir.rglob(f"*{ext}"))
            image_paths.extend(self.data_dir.rglob(f"*{ext.upper()}"))
        return sorted(image_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        # Return dummy label for compatibility
        return image, 0


def _check_gdown_installed():
    """Check if gdown is installed, install if not."""
    try:
        import gdown

        return True
    except ImportError:
        print("gdown not found. Installing...")
        subprocess.check_call(["pip", "install", "-q", "gdown"])
        return True


def download_celebahq(data_dir: str) -> Path:
    """
    Download CelebA-HQ dataset if not present.

    Args:
        data_dir: Directory to download to (e.g., ./data/celebahq)

    Returns:
        Path to the data directory
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    existing_images = list(data_path.glob("*.jpg")) + list(data_path.glob("*.png"))
    existing_images += list(data_path.rglob("*.jpg")) + list(data_path.rglob("*.png"))
    if len(existing_images) > 100:
        print(f"CelebA-HQ already exists: {len(existing_images)} images in {data_path}")
        return data_path

    print("=" * 60)
    print("CelebA-HQ dataset not found. Downloading...")
    print("=" * 60)

    _check_gdown_installed()
    import gdown

    # CelebA-HQ 256x256 from public Google Drive
    file_id = "1badu11NqxGf6qM3PTTooQDJvQbejgbTv"
    url = f"https://drive.google.com/uc?id={file_id}"
    output_zip = data_path / "celebahq.zip"

    try:
        print(f"Downloading from Google Drive (file_id: {file_id})...")
        gdown.download(url, str(output_zip), quiet=False)

        if output_zip.exists():
            print("Extracting...")
            with zipfile.ZipFile(output_zip, "r") as z:
                z.extractall(data_path)
            output_zip.unlink()
            print(f"CelebA-HQ downloaded and extracted to {data_path}")
        else:
            raise FileNotFoundError("Download failed")

    except Exception as e:
        print(f"\nAutomatic download failed: {e}")
        print("\n" + "=" * 60)
        print("MANUAL DOWNLOAD INSTRUCTIONS:")
        print("=" * 60)
        print("1. Download CelebA-HQ from one of these sources:")
        print("   - Kaggle: https://www.kaggle.com/datasets/lamsimon/celebahq")
        print("   - GitHub: https://github.com/tkarras/progressive_growing_of_gans")
        print(f"2. Extract images to: {data_path}")
        print("3. Re-run the training script")
        print("=" * 60)
        raise RuntimeError(f"Please download CelebA-HQ manually to {data_path}")

    return data_path


def download_lsun_churches(data_dir: str) -> Path:
    """
    Download LSUN Churches dataset if not present.

    Args:
        data_dir: Directory to download to (e.g., ./data/lsun_churches)

    Returns:
        Path to the data directory
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    existing_images = (
        list(data_path.rglob("*.jpg"))
        + list(data_path.rglob("*.webp"))
        + list(data_path.rglob("*.png"))
    )
    if len(existing_images) > 100:
        print(
            f"LSUN Churches already exists: {len(existing_images)} images in {data_path}"
        )
        return data_path

    print("=" * 60)
    print("LSUN Churches dataset not found.")
    print("=" * 60)
    print("\nLSUN is very large (~2GB+). Manual download recommended.")
    print("\nMANUAL DOWNLOAD INSTRUCTIONS:")
    print("=" * 60)
    print("Option 1 - From LSUN official:")
    print(
        '  python3 -c "from torchvision.datasets import LSUN; '
        "LSUN('./data', classes=['church_outdoor_train'], download=True)\""
    )
    print("\nOption 2 - From Kaggle:")
    print("  https://www.kaggle.com/datasets/jhoward/lsun_church")
    print(f"\nExtract images to: {data_path}")
    print("=" * 60)

    # Try using torchvision LSUN if available
    try:
        print("\nAttempting automatic download via torchvision...")
        from torchvision.datasets import LSUN

        lsun_root = data_path.parent / "lsun_raw"
        lsun_root.mkdir(parents=True, exist_ok=True)

        # This may take a while
        lsun_dataset = LSUN(
            str(lsun_root), classes=["church_outdoor_train"], download=True
        )

        # Extract images to our target directory
        print(f"Extracting {len(lsun_dataset)} images...")
        for i, (img, _) in enumerate(lsun_dataset):
            if i >= 30000:  # Limit to 30k images
                break
            img.save(data_path / f"church_{i:06d}.jpg")
            if (i + 1) % 1000 == 0:
                print(f"  Extracted {i + 1} images...")

        print(f"LSUN Churches downloaded to {data_path}")
        return data_path

    except Exception as e:
        print(f"\nAutomatic download failed: {e}")
        raise RuntimeError(f"Please download LSUN Churches manually to {data_path}")


def get_celebahq_dataloader(
    data_dir: str = "./data/celebahq",
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 128,
    train: bool = True,
    shuffle: bool = True,
    pin_memory: bool = True,
    auto_download: bool = True,
) -> DataLoader:
    """
    Get CelebA-HQ dataloader. Auto-downloads if not present.

    Args:
        data_dir: Directory containing CelebA-HQ images
        batch_size: Batch size
        num_workers: Number of data loading workers
        image_size: Target image size (128 recommended)
        train: Whether to load training set
        shuffle: Whether to shuffle data
        pin_memory: Whether to pin memory for faster GPU transfer
        auto_download: Whether to auto-download if not present

    Returns:
        DataLoader instance
    """
    data_path = Path(data_dir)

    # Check if data exists, download if not
    if auto_download:
        existing = list(data_path.rglob("*.jpg")) + list(data_path.rglob("*.png"))
        if len(existing) < 100:
            download_celebahq(data_dir)

    transform = get_transforms(image_size, train=train)
    dataset = ImageFolderDataset(data_dir, transform=transform)

    if len(dataset) == 0:
        raise ValueError(
            f"No images found in {data_dir}. "
            "Please download CelebA-HQ and place images in this directory."
        )

    print(f"Loaded CelebA-HQ: {len(dataset)} images from {data_dir}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=train,
    )

    return dataloader


def get_lsun_churches_dataloader(
    data_dir: str = "./data/lsun_churches",
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 128,
    train: bool = True,
    shuffle: bool = True,
    pin_memory: bool = True,
    auto_download: bool = True,
) -> DataLoader:
    """
    Get LSUN Churches dataloader. Auto-downloads if not present.

    Args:
        data_dir: Directory containing LSUN Churches images
        batch_size: Batch size
        num_workers: Number of data loading workers
        image_size: Target image size (128 recommended)
        train: Whether to load training set
        shuffle: Whether to shuffle data
        pin_memory: Whether to pin memory for faster GPU transfer
        auto_download: Whether to auto-download if not present

    Returns:
        DataLoader instance
    """
    data_path = Path(data_dir)

    # Check if data exists, download if not
    if auto_download:
        existing = (
            list(data_path.rglob("*.jpg"))
            + list(data_path.rglob("*.webp"))
            + list(data_path.rglob("*.png"))
        )
        if len(existing) < 100:
            download_lsun_churches(data_dir)

    transform = get_transforms(image_size, train=train)
    dataset = ImageFolderDataset(data_dir, transform=transform)

    if len(dataset) == 0:
        raise ValueError(
            f"No images found in {data_dir}. "
            "Please download LSUN Churches and place images in this directory."
        )

    print(f"Loaded LSUN Churches: {len(dataset)} images from {data_dir}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=train,
    )

    return dataloader


def get_cifar10_transforms(
    image_size: int = 32,
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get transforms for CIFAR-10 dataset (legacy support).

    Args:
        image_size: Target image size

    Returns:
        Tuple of (train_transforms, test_transforms)
    """
    train_transforms = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    test_transforms = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

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
    Get CIFAR-10 dataloader. Auto-downloads via torchvision.

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
        download=True,  # Auto-download
        transform=transform,
    )

    print(f"Loaded CIFAR-10: {len(dataset)} images")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=train,
    )

    return dataloader


def get_dataloader(
    dataset: str,
    data_dir: str,
    batch_size: int,
    num_workers: int,
    image_size: int,
    train: bool = True,
    shuffle: bool = True,
    pin_memory: bool = True,
    auto_download: bool = True,
) -> DataLoader:
    """
    Get dataloader for specified dataset. Auto-downloads if not present.

    Args:
        dataset: Dataset name ("celebahq", "lsun_churches", or "cifar10")
        data_dir: Data directory
        batch_size: Batch size
        num_workers: Number of workers
        image_size: Target image size
        train: Training mode
        shuffle: Shuffle data
        pin_memory: Pin memory
        auto_download: Whether to auto-download if not present

    Returns:
        DataLoader instance
    """
    if dataset == "celeba_hq_256":
        return get_celebahq_dataloader(
            data_dir=os.path.join(data_dir, "celeba_hq_256"),
            batch_size=batch_size,
            num_workers=num_workers,
            image_size=image_size,
            train=train,
            shuffle=shuffle,
            pin_memory=pin_memory,
            auto_download=auto_download,
        )
    elif dataset == "lsun_churches":
        return get_lsun_churches_dataloader(
            data_dir=os.path.join(data_dir, "lsun_churches"),
            batch_size=batch_size,
            num_workers=num_workers,
            image_size=image_size,
            train=train,
            shuffle=shuffle,
            pin_memory=pin_memory,
            auto_download=auto_download,
        )
    elif dataset == "cifar10":
        return get_cifar10_dataloader(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            image_size=image_size,
            train=train,
            shuffle=shuffle,
            pin_memory=pin_memory,
        )
    else:
        raise ValueError(
            f"Unknown dataset: {dataset}. Supported: celeba_hq_256, lsun_churches, cifar10"
        )


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
