"""
Sampling script for LDM.
Generate images from trained latent diffusion models.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional

import torch
import torchvision
from tqdm import tqdm
from diffusers import AutoencoderKL

sys.path.insert(0, str(Path(__file__).parent))

from config import Config, get_config
from models import create_model, create_diffusion
from utils import unnormalize


def load_model_from_checkpoint(
    checkpoint_path: str,
    config: Config,
    device: torch.device,
    use_ema: bool = True,
):
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint
        config: Model configuration
        device: Device to load to
        use_ema: Whether to use EMA weights

    Returns:
        Diffusion model ready for sampling
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = create_model(config).to(device)

    if use_ema and "ema_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["ema_state_dict"])
        print("Loaded EMA weights")
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Loaded model weights")

    model.eval()
    diffusion = create_diffusion(model, config).to(device)

    return diffusion


def load_vae(config: Config, device: torch.device) -> AutoencoderKL:
    dtype = torch.float16 if (config.vae.use_fp16 and device.type == "cuda") else torch.float32
    vae = AutoencoderKL.from_pretrained(
        config.vae.model_id,
        subfolder=config.vae.subfolder,
    ).to(device, dtype=dtype)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    return vae


@torch.no_grad()
def decode_latents(vae, latents: torch.Tensor, scaling: float) -> torch.Tensor:
    latents = latents / scaling
    images = vae.decode(latents).sample
    images = unnormalize(images).clamp(0, 1)
    return images


@torch.no_grad()
def generate_samples(
    diffusion,
    vae,
    config: Config,
    num_samples: int,
    batch_size: int,
    output_dir: Path,
    device: torch.device,
    save_individual: bool = False,
    prefix: str = "sample",
):
    """
    Generate samples from diffusion model.

    Args:
        diffusion: Diffusion model
        num_samples: Total number of samples to generate
        batch_size: Batch size for generation
        image_size: Image size
        output_dir: Directory to save samples
        device: Device to use
        save_individual: Whether to save individual images
        prefix: Prefix for saved files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_samples = []
    num_batches = (num_samples + batch_size - 1) // batch_size

    for i in tqdm(range(num_batches), desc="Generating samples"):
        current_batch_size = min(batch_size, num_samples - i * batch_size)

        latents = diffusion.sample(
            batch_size=current_batch_size,
            image_size=config.model.image_size,
            channels=config.model.out_channels,
            progress=False,
        )

        samples = decode_latents(vae, latents, config.vae.latent_scaling_factor)

        all_samples.append(samples.cpu())

        # Save individual images if requested
        if save_individual:
            for j, sample in enumerate(samples):
                idx = i * batch_size + j
                save_path = output_dir / f"{prefix}_{idx:05d}.png"
                torchvision.utils.save_image(sample, save_path)

    # Concatenate all samples
    all_samples = torch.cat(all_samples, dim=0)[:num_samples]

    # Save grid of first 64 samples
    grid_samples = all_samples[:64]
    grid = torchvision.utils.make_grid(grid_samples, nrow=8, padding=2)
    grid_path = output_dir / f"{prefix}_grid.png"
    torchvision.utils.save_image(grid, grid_path)

    # Save all samples as a single tensor file (for FID calculation)
    tensor_path = output_dir / f"{prefix}_all.pt"
    torch.save(all_samples, tensor_path)

    print(f"Saved {num_samples} samples to {output_dir}")
    print(f"Grid saved to {grid_path}")
    print(f"Tensor saved to {tensor_path}")

    return all_samples


@torch.no_grad()
def generate_interpolation(
    diffusion,
    vae,
    config: Config,
    num_steps: int = 10,
    output_dir: Path = Path("./samples"),
    device: torch.device = torch.device("cpu"),
):
    """
    Generate interpolation between two random noise samples.

    Args:
        diffusion: Diffusion model
        num_steps: Number of interpolation steps
        image_size: Image size
        output_dir: Directory to save results
        device: Device to use
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate two random noise vectors
    latent_size = config.model.image_size
    z1 = torch.randn(1, config.model.in_channels, latent_size, latent_size, device=device)
    z2 = torch.randn(1, config.model.in_channels, latent_size, latent_size, device=device)

    # Interpolate
    alphas = torch.linspace(0, 1, num_steps)
    samples = []

    for alpha in tqdm(alphas, desc="Interpolating"):
        # Spherical interpolation
        z = slerp(alpha.item(), z1, z2)

        # Denoise from this starting point
        x = z.clone()
        for t in reversed(range(diffusion.timesteps)):
            t_batch = torch.full((1,), t, device=device, dtype=torch.long)
            x = diffusion.p_sample(x, t_batch)

        decoded = decode_latents(vae, x, config.vae.latent_scaling_factor)
        samples.append(decoded.cpu())

    # Save as grid
    samples = torch.cat(samples, dim=0)
    grid = torchvision.utils.make_grid(samples, nrow=num_steps, padding=2)
    save_path = output_dir / "interpolation.png"
    torchvision.utils.save_image(grid, save_path)

    print(f"Interpolation saved to {save_path}")

    return samples


def slerp(alpha: float, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """
    Spherical linear interpolation.

    Args:
        alpha: Interpolation factor (0 = z1, 1 = z2)
        z1: First vector
        z2: Second vector

    Returns:
        Interpolated vector
    """
    z1_flat = z1.flatten()
    z2_flat = z2.flatten()

    # Normalize
    z1_norm = z1_flat / z1_flat.norm()
    z2_norm = z2_flat / z2_flat.norm()

    # Compute angle
    omega = torch.acos((z1_norm * z2_norm).sum().clamp(-1, 1))

    if omega.abs() < 1e-10:
        return (1 - alpha) * z1 + alpha * z2

    # Slerp
    sin_omega = torch.sin(omega)
    result = (
        torch.sin((1 - alpha) * omega) / sin_omega * z1_flat
        + torch.sin(alpha * omega) / sin_omega * z2_flat
    )

    return result.reshape(z1.shape)


@torch.no_grad()
def generate_denoising_process(
    diffusion,
    vae,
    config: Config,
    num_images: int = 4,
    num_vis_steps: int = 10,
    output_dir: Path = Path("./samples"),
    device: torch.device = torch.device("cpu"),
):
    """
    Visualize the denoising process.

    Args:
        diffusion: Diffusion model
        image_size: Image size
        num_images: Number of images to generate
        num_vis_steps: Number of steps to visualize
        output_dir: Directory to save results
        device: Device to use
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate samples with intermediates
    samples, intermediates = diffusion.sample(
        batch_size=num_images,
        image_size=config.model.image_size,
        channels=config.model.out_channels,
        return_intermediates=True,
        progress=True,
    )

    # Select steps to visualize
    total_steps = len(intermediates)
    vis_indices = [int(i * (total_steps - 1) / (num_vis_steps - 1)) for i in range(num_vis_steps)]

    # Create visualization grid
    vis_samples = []
    for idx in vis_indices:
        decoded = decode_latents(vae, intermediates[idx], config.vae.latent_scaling_factor)
        vis_samples.append(decoded)

    # Stack: [num_vis_steps, num_images, C, H, W]
    vis_samples = torch.stack(vis_samples, dim=0)

    # Rearrange for grid: show each image's denoising process as a row
    rows = []
    for i in range(num_images):
        row = vis_samples[:, i]  # [num_vis_steps, C, H, W]
        rows.append(row)

    grid_samples = torch.cat(rows, dim=0)  # [num_images * num_vis_steps, C, H, W]
    grid = torchvision.utils.make_grid(grid_samples, nrow=num_vis_steps, padding=2)

    save_path = output_dir / "denoising_process.png"
    torchvision.utils.save_image(grid, save_path)

    print(f"Denoising process visualization saved to {save_path}")

    return vis_samples


def main():
    parser = argparse.ArgumentParser(description="Generate samples from LDM")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--output-dir", type=str, default="./generated",
                        help="Output directory for samples")
    parser.add_argument("--num-samples", type=int, default=64,
                        help="Number of samples to generate")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for generation")

    # Model config
    parser.add_argument("--timesteps", type=int, default=1000,
                        help="Number of diffusion timesteps")
    parser.add_argument("--beta-schedule", type=str, default="linear",
                        choices=["linear", "cosine", "quadratic"],
                        help="Beta schedule type")
    parser.add_argument("--image-size", type=int, default=128,
                        help="Input image size")
    parser.add_argument("--downsample-factor", type=int, default=4,
                        help="VAE downsample factor (4 for 128px->32x32 latent)")
    parser.add_argument("--vae-model-id", type=str, default="stabilityai/sd-vae-ft-mse",
                        help="Hugging Face VAE model id")
    parser.add_argument("--vae-subfolder", type=str, default=None,
                        help="Optional VAE subfolder (e.g., 'vae' for SD checkpoints)")
    parser.add_argument("--vae-fp16", action="store_true",
                        help="Use FP16 VAE decoding on CUDA")

    # Options
    parser.add_argument("--no-ema", action="store_true",
                        help="Use model weights instead of EMA")
    parser.add_argument("--save-individual", action="store_true",
                        help="Save individual images")
    parser.add_argument("--interpolation", action="store_true",
                        help="Generate interpolation visualization")
    parser.add_argument("--denoising-vis", action="store_true",
                        help="Generate denoising process visualization")

    # Device
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "mps", "cpu"],
                        help="Device to use")

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
    config.data.image_size = args.image_size
    config.vae.model_id = args.vae_model_id
    config.vae.subfolder = args.vae_subfolder
    config.vae.use_fp16 = args.vae_fp16
    config.vae.downsample_factor = args.downsample_factor
    config.device = args.device
    config.model.image_size = config.data.image_size // config.vae.downsample_factor
    config.model.in_channels = config.vae.latent_channels
    config.model.out_channels = config.vae.latent_channels

    if config.device != "cuda":
        config.vae.use_fp16 = False
    if config.data.image_size % config.vae.downsample_factor != 0:
        raise ValueError(f"image_size ({config.data.image_size}) must be divisible by VAE downsample_factor ({config.vae.downsample_factor}).")

    # Load model
    diffusion = load_model_from_checkpoint(
        args.checkpoint,
        config,
        device,
        use_ema=not args.no_ema,
    )
    vae = load_vae(config, device)

    output_dir = Path(args.output_dir)

    # Generate samples
    if args.interpolation:
        generate_interpolation(
            diffusion,
            vae,
            config,
            output_dir=output_dir,
            device=device,
        )
    elif args.denoising_vis:
        generate_denoising_process(
            diffusion,
            vae,
            config,
            output_dir=output_dir,
            device=device,
        )
    else:
        generate_samples(
            diffusion,
            vae,
            config,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            output_dir=output_dir,
            device=device,
            save_individual=args.save_individual,
        )


if __name__ == "__main__":
    main()
