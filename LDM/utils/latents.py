"""
Latent preparation utilities for LDM training.
Supports CelebA-HQ, LSUN-Churches, and CIFAR-10.
"""

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

from .data import get_dataloader


def _default_latent_cache_path(
    cache_dir: str,
    dataset: str,
    image_size: int,
    vae_model_id: str,
    downsample_factor: int,
) -> Path:
    safe_id = vae_model_id.replace("/", "_")
    name = f"{dataset}_{image_size}_f{downsample_factor}_{safe_id}_latents.pt"
    return Path(cache_dir) / name


@torch.no_grad()
def prepare_latents(
    *,
    vae,
    config,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    force_recompute: bool = False,
    pin_memory: bool = True,
) -> Tuple[TensorDataset, Path]:
    """
    Encode the full dataset into latents and cache to disk.

    Returns:
        TensorDataset of latents (and labels) and the cache path.
    """
    cache_dir = Path(config.vae.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_path = _default_latent_cache_path(
        cache_dir=str(cache_dir),
        dataset=config.data.dataset,
        image_size=config.data.image_size,
        vae_model_id=config.vae.model_id,
        downsample_factor=config.vae.downsample_factor,
    )

    if cache_path.exists() and not force_recompute:
        payload = torch.load(cache_path, map_location="cpu")
        latents = payload["latents"]
        labels = payload.get("labels")
        if labels is None:
            dataset = TensorDataset(latents)
        else:
            dataset = TensorDataset(latents, labels)
        print(f"Loaded cached latents from {cache_path}")
        print(f"  Latent shape: {latents.shape}")
        return dataset, cache_path

    # Use unified dataloader
    dataloader = get_dataloader(
        dataset=config.data.dataset,
        data_dir=config.data.data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=config.data.image_size,
        train=True,
        shuffle=False,
        pin_memory=pin_memory,
    )

    vae.eval()
    latents_list = []
    labels_list = []

    scaling = config.vae.latent_scaling_factor
    use_fp16 = config.vae.use_fp16

    print(f"Encoding latents for {config.data.dataset} at {config.data.image_size}px...")
    print(f"  VAE: {config.vae.model_id}")
    print(f"  Downsample factor: {config.vae.downsample_factor}")
    print(f"  Expected latent size: {config.data.image_size // config.vae.downsample_factor}x{config.data.image_size // config.vae.downsample_factor}")

    for batch in dataloader:
        images, labels = batch
        images = images.to(device)
        if use_fp16:
            images = images.half()

        latent_dist = vae.encode(images).latent_dist
        latents = latent_dist.sample() * scaling

        latents_list.append(latents.cpu())
        labels_list.append(labels.cpu())

    latents = torch.cat(latents_list, dim=0)
    labels = torch.cat(labels_list, dim=0)

    print(f"Encoded {len(latents)} samples")
    print(f"  Latent shape: {latents.shape}")

    payload = {"latents": latents, "labels": labels}
    torch.save(payload, cache_path)
    print(f"Saved latents to {cache_path}")

    dataset = TensorDataset(latents, labels)
    return dataset, cache_path
