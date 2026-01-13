"""
Prepare and cache VAE latents for LDM training.
"""

import argparse
from pathlib import Path

import torch
from diffusers import AutoencoderKL

from config import get_config
from utils import prepare_latents


def get_device(device: str) -> torch.device:
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_vae(config, device: torch.device) -> AutoencoderKL:
    dtype = torch.float16 if (config.vae.use_fp16 and device.type == "cuda") else torch.float32
    vae = AutoencoderKL.from_pretrained(
        config.vae.model_id,
        subfolder=config.vae.subfolder,
    ).to(device, dtype=dtype)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    return vae


def main():
    parser = argparse.ArgumentParser(description="Prepare latents for LDM training")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Data directory")
    parser.add_argument("--dataset", type=str, default="celeba_hq_256",
                        choices=["celeba_hq_256", "lsun_churches", "cifar10"],
                        help="Dataset to use")
    parser.add_argument("--image-size", type=int, default=128,
                        help="Input image size (must be divisible by downsample factor)")
    parser.add_argument("--downsample-factor", type=int, default=4,
                        help="VAE downsample factor (4 for 128px->32x32 latent)")
    parser.add_argument("--vae-model-id", type=str, default="stabilityai/sd-vae-ft-mse",
                        help="Hugging Face VAE model id")
    parser.add_argument("--vae-subfolder", type=str, default=None,
                        help="Optional VAE subfolder (e.g., 'vae')")
    parser.add_argument("--vae-cache-dir", type=str, default="./data/latents",
                        help="Directory for cached latents")
    parser.add_argument("--vae-fp16", action="store_true",
                        help="Use FP16 VAE encoding on CUDA")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for encoding")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of dataloader workers")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "mps", "cpu"],
                        help="Device to use")
    parser.add_argument("--force", action="store_true",
                        help="Recompute and overwrite cache")

    args = parser.parse_args()

    config = get_config()
    config.data.data_dir = args.data_dir
    config.data.dataset = args.dataset
    config.data.image_size = args.image_size
    config.vae.model_id = args.vae_model_id
    config.vae.subfolder = args.vae_subfolder
    config.vae.cache_dir = args.vae_cache_dir
    config.vae.use_fp16 = args.vae_fp16
    config.vae.downsample_factor = args.downsample_factor

    if config.data.image_size % config.vae.downsample_factor != 0:
        raise ValueError(f"image_size ({config.data.image_size}) must be divisible by VAE downsample_factor ({config.vae.downsample_factor}).")

    device = get_device(args.device)
    if device.type != "cuda":
        config.vae.use_fp16 = False

    vae = load_vae(config, device)
    dataset, cache_path = prepare_latents(
        vae=vae,
        config=config,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        force_recompute=args.force,
    )

    print(f"Latents saved to: {cache_path}")
    print(f"Latent tensor shape: {dataset.tensors[0].shape}")


if __name__ == "__main__":
    main()
