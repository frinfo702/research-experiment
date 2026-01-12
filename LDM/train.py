"""
Training script for LDM (Latent Diffusion Model).
"""

import os
import sys
import argparse
import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
import torchvision
import wandb
from dotenv import load_dotenv

from diffusers import AutoencoderKL

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import Config, get_config
from models import create_model, create_diffusion, EMA
from utils import (
    unnormalize,
    setup_logger,
    MetricsLogger,
    CheckpointManager,
    prepare_latents,
)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(config: Config) -> torch.device:
    """Get appropriate device."""
    if config.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif config.device == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def save_samples(
    diffusion,
    vae,
    config: Config,
    step: int,
    sample_dir: Path,
    num_samples: int = 64,
):
    """Generate and save decoded sample images."""
    diffusion.eval()
    with torch.no_grad():
        latents = diffusion.sample(
            batch_size=num_samples,
            image_size=config.model.image_size,
            channels=config.model.out_channels,
            progress=True,
        )

        latents = latents / config.vae.latent_scaling_factor
        decoded = vae.decode(latents).sample
    diffusion.train()

    # Unnormalize and save
    decoded = unnormalize(decoded)
    decoded = torch.clamp(decoded, 0, 1)

    # Save grid
    grid = torchvision.utils.make_grid(decoded, nrow=8, padding=2)
    save_path = sample_dir / f"samples_{step:08d}.png"
    torchvision.utils.save_image(grid, save_path)

    return decoded, grid


def load_vae(config: Config, device: torch.device) -> AutoencoderKL:
    """Load and freeze the pretrained VAE."""
    dtype = (
        torch.float16
        if (config.vae.use_fp16 and device.type == "cuda")
        else torch.float32
    )
    vae = AutoencoderKL.from_pretrained(
        config.vae.model_id,
        subfolder=config.vae.subfolder,
    ).to(device, dtype=dtype)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    return vae


def setup_wandb(config: Config, args) -> Optional[wandb.sdk.wandb_run.Run]:
    """Initialize Weights & Biases logging."""
    if args.wandb_mode == "disabled":
        return None

    load_dotenv()
    wandb_key = args.wandb_key or os.getenv("WANDB_API_KEY")
    if wandb_key:
        wandb.login(key=wandb_key)

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config={
            "exp_name": config.exp_name,
            "timesteps": config.diffusion.timesteps,
            "beta_schedule": config.diffusion.beta_schedule,
            "batch_size": config.training.batch_size,
            "learning_rate": config.training.learning_rate,
            "image_size": config.data.image_size,
            "latent_size": config.model.image_size,
            "vae_model_id": config.vae.model_id,
        },
        mode=args.wandb_mode,
        name=config.exp_name,
    )
    return run


def train(
    config: Config,
    args,
    resume_from: Optional[str] = None,
):
    """
    Main training function.

    Args:
        config: Training configuration
        resume_from: Optional path to checkpoint to resume from
    """
    # Setup
    device = get_device(config)
    set_seed(config.seed)

    # Create directories
    output_dir = Path(config.output_dir) / config.exp_name
    checkpoint_dir = output_dir / "checkpoints"
    sample_dir = output_dir / "samples"
    log_dir = output_dir / "logs"

    for d in [checkpoint_dir, sample_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logger("train", str(log_dir))
    metrics_logger = MetricsLogger(str(log_dir), config.exp_name)
    checkpoint_manager = CheckpointManager(str(checkpoint_dir))

    logger.info(f"Training LDM with config: {config.exp_name}")
    logger.info(f"Device: {device}")
    logger.info(f"Timesteps: {config.diffusion.timesteps}")
    logger.info(f"Beta schedule: {config.diffusion.beta_schedule}")

    # Create model and diffusion
    model = create_model(config).to(device)
    diffusion = create_diffusion(model, config).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")

    # EMA
    ema = EMA(model, decay=config.training.ema_decay)
    ema.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.0,
    )

    # Learning rate scheduler with warmup
    def lr_lambda(step):
        if step < config.training.warmup_steps:
            return step / config.training.warmup_steps
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if device.type != "cuda":
        config.training.mixed_precision = False
        if device.type == "mps":
            config.training.num_workers = 0

    # Mixed precision
    scaler = (
        GradScaler()
        if config.training.mixed_precision and device.type == "cuda"
        else None
    )

    # Load VAE and prepare latents
    vae = load_vae(config, device)

    latent_batch_size = config.training.batch_size
    if device.type in {"mps", "cpu"}:
        latent_batch_size = min(latent_batch_size, 16)
        logger.info(
            f"Latent encode batch size set to {latent_batch_size} for {device.type}"
        )

    latents_dataset, cache_path = prepare_latents(
        vae=vae,
        config=config,
        device=device,
        batch_size=latent_batch_size,
        num_workers=config.training.num_workers,
        force_recompute=args.force_latents,
        pin_memory=(device.type == "cuda"),
    )
    logger.info(f"Latents cache: {cache_path}")

    dataloader = torch.utils.data.DataLoader(
        latents_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    # Resume from checkpoint
    start_step = 0
    if resume_from:
        checkpoint_info = checkpoint_manager.load(
            resume_from, model, optimizer, ema, scheduler, device=str(device)
        )
        start_step = checkpoint_info["step"]
        logger.info(f"Resumed from step {start_step}")

    # W&B setup
    wandb_run = setup_wandb(config, args)

    # Training loop
    model.train()
    step = start_step
    epoch = 0
    running_loss = 0.0

    pbar = tqdm(total=config.training.total_steps - start_step, desc="Training")

    while step < config.training.total_steps:
        epoch += 1

        for batch in dataloader:
            if step >= config.training.total_steps:
                break

            latents = batch[0].to(device)

            # Forward pass with optional mixed precision
            optimizer.zero_grad()

            if scaler is not None:
                with autocast():
                    loss = diffusion(latents)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = diffusion(latents)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
                optimizer.step()

            scheduler.step()
            ema.update(model)

            step += 1
            running_loss += loss.item()

            # Logging
            if step % config.training.log_every == 0:
                avg_loss = running_loss / config.training.log_every
                lr = scheduler.get_last_lr()[0]

                metrics_logger.log(
                    step,
                    {
                        "loss": avg_loss,
                        "lr": lr,
                    },
                )

                if wandb_run is not None:
                    wandb.log({"loss": avg_loss, "lr": lr, "step": step})

                pbar.set_postfix(
                    {
                        "loss": f"{avg_loss:.4f}",
                        "lr": f"{lr:.6f}",
                    }
                )

                running_loss = 0.0

            # Save samples
            if step % config.training.sample_every == 0:
                logger.info(f"Generating samples at step {step}")
                # Use EMA model for sampling
                ema_diffusion = create_diffusion(ema.shadow, config).to(device)
                samples, grid = save_samples(
                    ema_diffusion,
                    vae,
                    config,
                    step,
                    sample_dir,
                    num_samples=64,
                )
                if wandb_run is not None:
                    wandb.log({"samples": wandb.Image(grid), "step": step})

            # Save checkpoint
            if step % config.training.save_every == 0:
                logger.info(f"Saving checkpoint at step {step}")
                checkpoint_manager.save(
                    step=step,
                    model=model,
                    optimizer=optimizer,
                    ema=ema,
                    scheduler=scheduler,
                    metrics={"loss": loss.item()},
                )

            pbar.update(1)

    pbar.close()

    # Final save
    logger.info("Training complete. Saving final checkpoint...")
    checkpoint_manager.save(
        step=step,
        model=model,
        optimizer=optimizer,
        ema=ema,
        scheduler=scheduler,
        metrics={"loss": loss.item()},
    )

    # Save final samples
    ema_diffusion = create_diffusion(ema.shadow, config).to(device)
    save_samples(
        ema_diffusion,
        vae,
        config,
        step,
        sample_dir,
        num_samples=64,
    )

    # Save summary
    metrics_logger.save_summary(
        {
            "exp_name": config.exp_name,
            "total_steps": step,
            "timesteps": config.diffusion.timesteps,
            "beta_schedule": config.diffusion.beta_schedule,
            "final_loss": loss.item(),
        }
    )

    logger.info("Done!")

    if wandb_run is not None:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Train LDM (Latent Diffusion Model)")

    # Config overrides
    parser.add_argument(
        "--timesteps", type=int, default=1000, help="Number of diffusion timesteps"
    )
    parser.add_argument(
        "--beta-schedule",
        type=str,
        default="cosine",
        choices=["linear", "cosine", "quadratic"],
        help="Beta schedule type",
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Training batch size"
    )
    parser.add_argument(
        "--total-steps", type=int, default=800000, help="Total training steps"
    )
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--exp-name", type=str, default=None, help="Experiment name")

    # Paths
    parser.add_argument(
        "--output-dir", type=str, default="./outputs", help="Output directory"
    )
    parser.add_argument("--data-dir", type=str, default="./data", help="Data directory")
    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="Input image size (must be divisible by downsample factor)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="celeba_hq_256",
        choices=["celeba_hq_256", "lsun_churches", "cifar10"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--downsample-factor",
        type=int,
        default=4,
        help="VAE downsample factor (4 for 128px->32x32 latent)",
    )
    parser.add_argument(
        "--vae-model-id",
        type=str,
        default="stabilityai/sd-vae-ft-mse",
        help="Hugging Face VAE model id",
    )
    parser.add_argument(
        "--vae-subfolder",
        type=str,
        default=None,
        help="Optional VAE subfolder (e.g., 'vae' for SD checkpoints)",
    )
    parser.add_argument(
        "--vae-cache-dir",
        type=str,
        default="./data/latents",
        help="Directory for cached latents",
    )
    parser.add_argument(
        "--vae-fp16", action="store_true", help="Use FP16 VAE encoding on CUDA"
    )
    parser.add_argument(
        "--force-latents",
        action="store_true",
        help="Recompute and overwrite latent cache",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )

    # Model overrides
    parser.add_argument(
        "--model-channels",
        type=int,
        default=None,
        help="Base channel count for U-Net",
    )
    parser.add_argument(
        "--channel-mult",
        type=str,
        default=None,
        help="Comma-separated channel multipliers (e.g., '1,2,3,4')",
    )
    parser.add_argument(
        "--num-res-blocks",
        type=int,
        default=None,
        help="Number of residual blocks per resolution",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=None,
        help="Number of attention heads",
    )
    parser.add_argument(
        "--attention-resolutions",
        type=str,
        default=None,
        help="Comma-separated attention resolutions (e.g., '16,8')",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Dropout rate",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "mps", "cpu"],
        help="Device to use",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # W&B
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="ldm-celeba_hq_256",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Weights & Biases entity (team/user)",
    )
    parser.add_argument(
        "--wandb-key", type=str, default=None, help="Weights & Biases API key"
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="Weights & Biases mode",
    )

    # Convenience flags
    parser.add_argument(
        "--debug", action="store_true", help="Debug mode (fewer steps, more logging)"
    )

    args = parser.parse_args()

    # Create config
    config = get_config(
        timesteps=args.timesteps,
        beta_schedule=args.beta_schedule,
        exp_name=args.exp_name,
    )

    # Apply overrides
    config.training.batch_size = args.batch_size
    config.training.total_steps = args.total_steps
    config.training.learning_rate = args.lr
    config.output_dir = args.output_dir
    config.data.data_dir = args.data_dir
    config.data.image_size = args.image_size
    config.data.dataset = args.dataset
    config.device = args.device
    config.seed = args.seed
    config.vae.model_id = args.vae_model_id
    config.vae.subfolder = args.vae_subfolder
    config.vae.cache_dir = args.vae_cache_dir
    config.vae.use_fp16 = args.vae_fp16
    config.vae.downsample_factor = args.downsample_factor

    # Ensure latent dimensions match the updated config
    config.model.in_channels = config.vae.latent_channels
    config.model.out_channels = config.vae.latent_channels
    config.model.image_size = config.data.image_size // config.vae.downsample_factor

    if args.model_channels is not None:
        config.model.model_channels = args.model_channels
    if args.channel_mult is not None:
        config.model.channel_mult = tuple(
            int(x.strip()) for x in args.channel_mult.split(",") if x.strip()
        )
    if args.num_res_blocks is not None:
        config.model.num_res_blocks = args.num_res_blocks
    if args.num_heads is not None:
        config.model.num_heads = args.num_heads
    if args.attention_resolutions is not None:
        config.model.attention_resolutions = tuple(
            int(x.strip()) for x in args.attention_resolutions.split(",") if x.strip()
        )
    if args.dropout is not None:
        config.model.dropout = args.dropout

    if config.data.image_size % config.vae.downsample_factor != 0:
        raise ValueError(
            f"image_size ({config.data.image_size}) must be divisible by VAE downsample_factor ({config.vae.downsample_factor})."
        )

    if config.device != "cuda":
        config.vae.use_fp16 = False

    # Debug mode
    if args.debug:
        config.training.total_steps = 1000
        config.training.log_every = 10
        config.training.sample_every = 100
        config.training.save_every = 500

    # Train
    train(config, args, resume_from=args.resume)


if __name__ == "__main__":
    main()
