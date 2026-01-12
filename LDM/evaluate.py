"""
Evaluation script for DDPM.
Computes FID score between generated samples and real data.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import CIFAR10
from torchvision.models import Inception_V3_Weights
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from config import Config, get_config
from models import create_diffusion, create_model
from utils import unnormalize


class InceptionV3Features(nn.Module):
    """
    Inception V3 model for feature extraction.
    Uses the pool3 layer (2048-dim) as in the original FID paper.
    """

    def __init__(self):
        super().__init__()

        # Load pretrained Inception V3
        inception = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)

        # Extract layers up to pool3
        self.blocks = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )

        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from images.

        Args:
            x: Images [B, 3, H, W] in range [0, 1]

        Returns:
            Features [B, 2048]
        """
        # Resize to 299x299 (Inception input size)
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)

        # Normalize using ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        # Extract features
        features = self.blocks(x)
        features = features.view(features.size(0), -1)

        return features


def compute_statistics(
    features: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and covariance statistics of features.

    Args:
        features: Feature matrix [N, D]

    Returns:
        Tuple of (mean [D], covariance [D, D])
    """
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def compute_fid(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    Compute Frechet Inception Distance.

    FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))

    Args:
        mu1: Mean of first distribution
        sigma1: Covariance of first distribution
        mu2: Mean of second distribution
        sigma2: Covariance of second distribution
        eps: Small value for numerical stability

    Returns:
        FID score
    """
    diff = mu1 - mu2

    # Compute sqrt(sigma1 * sigma2)
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    # Handle numerical errors
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Handle imaginary components
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError("Imaginary component in sqrtm")
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)

    return float(fid)


@torch.no_grad()
def extract_features_from_dataloader(
    dataloader: DataLoader,
    model: InceptionV3Features,
    device: torch.device,
    max_samples: Optional[int] = None,
) -> np.ndarray:
    """
    Extract features from a dataloader.

    Args:
        dataloader: DataLoader with images
        model: Inception model
        device: Device to use
        max_samples: Maximum number of samples to process

    Returns:
        Feature matrix [N, 2048]
    """
    features_list = []
    total = 0

    for batch in tqdm(dataloader, desc="Extracting features"):
        if isinstance(batch, (list, tuple)):
            images = batch[0]
        else:
            images = batch

        # Assume images are in [0, 1] or normalize if in [-1, 1]
        if images.min() < 0:
            images = unnormalize(images)

        images = images.to(device)
        features = model(images)
        features_list.append(features.cpu().numpy())

        total += images.shape[0]
        if max_samples is not None and total >= max_samples:
            break

    features = np.concatenate(features_list, axis=0)
    if max_samples is not None:
        features = features[:max_samples]

    return features


@torch.no_grad()
def extract_features_from_tensor(
    images: torch.Tensor,
    model: InceptionV3Features,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Extract features from a tensor of images.

    Args:
        images: Images tensor [N, C, H, W]
        model: Inception model
        device: Device to use
        batch_size: Batch size for processing

    Returns:
        Feature matrix [N, 2048]
    """
    features_list = []
    num_batches = (len(images) + batch_size - 1) // batch_size

    for i in tqdm(range(num_batches), desc="Extracting features"):
        start = i * batch_size
        end = min(start + batch_size, len(images))
        batch = images[start:end].to(device)

        features = model(batch)
        features_list.append(features.cpu().numpy())

    return np.concatenate(features_list, axis=0)


def compute_cifar10_stats(
    data_dir: str,
    device: torch.device,
    batch_size: int = 256,
    save_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute FID statistics for CIFAR-10 training set.

    Args:
        data_dir: Directory for CIFAR-10 data
        device: Device to use
        batch_size: Batch size
        save_path: Optional path to save statistics

    Returns:
        Tuple of (mean, covariance)
    """
    # Check if stats already exist
    if save_path and Path(save_path).exists():
        print(f"Loading precomputed stats from {save_path}")
        stats = np.load(save_path)
        return stats["mu"], stats["sigma"]

    print("Computing CIFAR-10 statistics...")

    # Load CIFAR-10
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    dataset = CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    # Initialize Inception model
    inception = InceptionV3Features().to(device)
    inception.eval()

    # Extract features
    features = extract_features_from_dataloader(dataloader, inception, device)

    # Compute statistics
    mu, sigma = compute_statistics(features)

    # Save if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(save_path, mu=mu, sigma=sigma)
        print(f"Saved statistics to {save_path}")

    return mu, sigma


def _amp_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if device.type == "mps":
        return torch.autocast(device_type="mps", dtype=torch.float16)
    return torch.autocast(device_type="cpu", enabled=False)


def _sampling_config(use_ddim: bool, ddim_steps: int, ddim_eta: float) -> Tuple[bool, int, float]:
    if not use_ddim:
        return False, 0, 0.0
    if ddim_steps < 2:
        raise ValueError("ddim-steps must be >= 2")
    if ddim_eta < 0:
        raise ValueError("ddim-eta must be >= 0")
    return True, ddim_steps, ddim_eta


@torch.no_grad()
def evaluate_model(
    checkpoint_path: str,
    config: Config,
    device: torch.device,
    num_samples: int = 50000,
    batch_size: int = 256,
    data_dir: str = "./data",
    stats_path: Optional[str] = None,
    use_ema: bool = True,
    use_ddim: bool = False,
    ddim_steps: int = 50,
    ddim_eta: float = 0.0,
    use_autocast: bool = True,
    use_channels_last: bool = True,
) -> dict:
    """
    Evaluate a trained model by computing FID score.

    Args:
        checkpoint_path: Path to model checkpoint
        config: Model configuration
        device: Device to use
        num_samples: Number of samples to generate
        batch_size: Batch size for generation and feature extraction
        data_dir: Directory for CIFAR-10 data
        stats_path: Optional path to precomputed CIFAR-10 stats
        use_ema: Whether to use EMA weights
        use_ddim: Whether to use DDIM sampling
        ddim_steps: Number of DDIM steps
        ddim_eta: DDIM eta (0.0 = deterministic)
        use_autocast: Whether to enable torch.autocast mixed precision
        use_channels_last: Whether to use channels_last memory format

    Returns:
        Dictionary with evaluation results
    """
    print(f"Evaluating model: {checkpoint_path}")
    print(f"Generating {num_samples} samples...")

    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = create_model(config).to(device)
    if use_channels_last and device.type in {"cuda", "mps"}:
        model = model.to(memory_format=torch.channels_last)
    if use_ema and "ema_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["ema_state_dict"])
        print("Using EMA weights")
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Using model weights")

    model.eval()
    diffusion = create_diffusion(model, config).to(device)

    # Initialize Inception model
    inception = InceptionV3Features().to(device)
    if use_channels_last and device.type in {"cuda", "mps"}:
        inception = inception.to(memory_format=torch.channels_last)
    inception.eval()

    # Generate samples and extract features
    print("Generating samples and extracting features...")
    gen_features_list = []
    num_batches = (num_samples + batch_size - 1) // batch_size

    use_ddim, ddim_steps, ddim_eta = _sampling_config(use_ddim, ddim_steps, ddim_eta)

    for i in tqdm(range(num_batches), desc="Generating"):
        current_batch = min(batch_size, num_samples - i * batch_size)

        # Generate samples
        with _amp_context(device) if use_autocast else torch.autocast("cpu", enabled=False):
            if use_ddim:
                samples = diffusion.sample_ddim(
                    batch_size=current_batch,
                    image_size=config.model.image_size,
                    num_steps=ddim_steps,
                    eta=ddim_eta,
                    progress=False,
                    memory_format=torch.channels_last if use_channels_last else None,
                )
            else:
                samples = diffusion.sample(
                    batch_size=current_batch,
                    image_size=config.model.image_size,
                    progress=False,
                    memory_format=torch.channels_last if use_channels_last else None,
                )

        # Unnormalize from [-1, 1] to [0, 1]
        samples = unnormalize(samples.float())
        samples = torch.clamp(samples, 0, 1)

        # Extract features
        features = inception(samples)
        gen_features_list.append(features.cpu().numpy())

    gen_features = np.concatenate(gen_features_list, axis=0)[:num_samples]

    # Compute generated statistics
    mu_gen, sigma_gen = compute_statistics(gen_features)

    # Get real statistics
    if stats_path is None:
        stats_path = Path(data_dir) / "cifar10_fid_stats.npz"
    mu_real, sigma_real = compute_cifar10_stats(
        data_dir, device, batch_size, str(stats_path)
    )

    # Compute FID
    fid = compute_fid(mu_real, sigma_real, mu_gen, sigma_gen)

    results = {
        "fid": fid,
        "num_samples": num_samples,
        "checkpoint": checkpoint_path,
        "timesteps": config.diffusion.timesteps,
        "beta_schedule": config.diffusion.beta_schedule,
    }

    print(f"\nResults:")
    print(f"  FID: {fid:.4f}")
    print(f"  Samples: {num_samples}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate DDPM model")

    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50000,
        help="Number of samples for FID calculation",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")

    # Model config
    parser.add_argument(
        "--timesteps", type=int, default=1000, help="Number of diffusion timesteps"
    )
    parser.add_argument(
        "--beta-schedule",
        type=str,
        default="linear",
        choices=["linear", "cosine", "quadratic"],
        help="Beta schedule type",
    )

    # Paths
    parser.add_argument("--data-dir", type=str, default="./data", help="Data directory")
    parser.add_argument(
        "--stats-path",
        type=str,
        default=None,
        help="Path to precomputed CIFAR-10 stats",
    )

    # Options
    parser.add_argument(
        "--no-ema", action="store_true", help="Use model weights instead of EMA"
    )
    parser.add_argument("--ddim", action="store_true", help="Use DDIM sampling")
    parser.add_argument("--ddim-steps", type=int, default=50, help="Number of DDIM steps")
    parser.add_argument("--ddim-eta", type=float, default=0.0, help="DDIM eta (0.0 = deterministic)")
    parser.add_argument("--no-autocast", action="store_true", help="Disable torch.autocast")
    parser.add_argument("--no-channels-last", action="store_true", help="Disable channels_last")

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "mps", "cpu"],
        help="Device to use",
    )

    args = parser.parse_args()

    # Setup device
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Create config
    config = get_config(
        timesteps=args.timesteps,
        beta_schedule=args.beta_schedule,
    )

    # Evaluate
    results = evaluate_model(
        checkpoint_path=args.checkpoint,
        config=config,
        device=device,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        stats_path=args.stats_path,
        use_ema=not args.no_ema,
        use_ddim=args.ddim,
        ddim_steps=args.ddim_steps,
        ddim_eta=args.ddim_eta,
        use_autocast=not args.no_autocast,
        use_channels_last=not args.no_channels_last,
    )

    print(f"\nFinal FID: {results['fid']:.4f}")


if __name__ == "__main__":
    main()
