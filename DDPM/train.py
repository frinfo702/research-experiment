"""
Training script for DDPM.
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

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import Config, get_config
from models import create_model, create_diffusion, EMA
from utils import (
    get_cifar10_dataloader,
    unnormalize,
    setup_logger,
    MetricsLogger,
    CheckpointManager,
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
    step: int,
    sample_dir: Path,
    num_samples: int = 64,
    image_size: int = 32,
    device: torch.device = torch.device("cpu"),
):
    """Generate and save sample images."""
    import torchvision

    diffusion.eval()
    with torch.no_grad():
        samples = diffusion.sample(
            batch_size=num_samples,
            image_size=image_size,
            progress=True,
        )
    diffusion.train()

    # Unnormalize and save
    samples = unnormalize(samples)
    samples = torch.clamp(samples, 0, 1)

    # Save grid
    grid = torchvision.utils.make_grid(samples, nrow=8, padding=2)
    save_path = sample_dir / f"samples_{step:08d}.png"
    torchvision.utils.save_image(grid, save_path)

    return samples


def train(
    config: Config,
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

    logger.info(f"Training DDPM with config: {config.exp_name}")
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

    # Mixed precision
    scaler = GradScaler() if config.training.mixed_precision and device.type == "cuda" else None

    # Data
    dataloader = get_cifar10_dataloader(
        data_dir=config.data_dir,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        image_size=config.model.image_size,
        train=True,
    )

    # Resume from checkpoint
    start_step = 0
    if resume_from:
        checkpoint_info = checkpoint_manager.load(
            resume_from, model, optimizer, ema, scheduler, device=str(device)
        )
        start_step = checkpoint_info["step"]
        logger.info(f"Resumed from step {start_step}")

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

            images = batch[0].to(device)

            # Forward pass with optional mixed precision
            optimizer.zero_grad()

            if scaler is not None:
                with autocast():
                    loss = diffusion(images)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = diffusion(images)
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

                metrics_logger.log(step, {
                    "loss": avg_loss,
                    "lr": lr,
                })

                pbar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "lr": f"{lr:.6f}",
                })

                running_loss = 0.0

            # Save samples
            if step % config.training.sample_every == 0:
                logger.info(f"Generating samples at step {step}")
                # Use EMA model for sampling
                ema_diffusion = create_diffusion(ema.shadow, config).to(device)
                save_samples(
                    ema_diffusion,
                    step,
                    sample_dir,
                    num_samples=64,
                    image_size=config.model.image_size,
                    device=device,
                )

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
        step,
        sample_dir,
        num_samples=64,
        image_size=config.model.image_size,
        device=device,
    )

    # Save summary
    metrics_logger.save_summary({
        "exp_name": config.exp_name,
        "total_steps": step,
        "timesteps": config.diffusion.timesteps,
        "beta_schedule": config.diffusion.beta_schedule,
        "final_loss": loss.item(),
    })

    logger.info("Done!")


def main():
    parser = argparse.ArgumentParser(description="Train DDPM")

    # Config overrides
    parser.add_argument("--timesteps", type=int, default=1000,
                        help="Number of diffusion timesteps")
    parser.add_argument("--beta-schedule", type=str, default="linear",
                        choices=["linear", "cosine", "quadratic"],
                        help="Beta schedule type")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Training batch size")
    parser.add_argument("--total-steps", type=int, default=800000,
                        help="Total training steps")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--exp-name", type=str, default=None,
                        help="Experiment name")

    # Paths
    parser.add_argument("--output-dir", type=str, default="./outputs",
                        help="Output directory")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Data directory")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")

    # Device
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "mps", "cpu"],
                        help="Device to use")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # Convenience flags
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode (fewer steps, more logging)")

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
    config.data_dir = args.data_dir
    config.device = args.device
    config.seed = args.seed

    # Debug mode
    if args.debug:
        config.training.total_steps = 1000
        config.training.log_every = 10
        config.training.sample_every = 100
        config.training.save_every = 500

    # Train
    train(config, resume_from=args.resume)


if __name__ == "__main__":
    main()
