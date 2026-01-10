"""
Default configuration for DDPM training and evaluation.
Based on Ho et al. 2020 "Denoising Diffusion Probabilistic Models"
"""

from dataclasses import dataclass, field
from typing import Literal, Optional
from pathlib import Path


@dataclass
class ModelConfig:
    """U-Net model configuration."""
    image_size: int = 32
    in_channels: int = 3
    out_channels: int = 3
    model_channels: int = 128
    channel_mult: tuple = (1, 2, 2, 2)
    num_res_blocks: int = 2
    attention_resolutions: tuple = (16,)  # Apply attention at 16x16 resolution
    dropout: float = 0.1
    num_heads: int = 4
    use_scale_shift_norm: bool = True


@dataclass
class DiffusionConfig:
    """Diffusion process configuration."""
    timesteps: int = 1000
    beta_schedule: Literal["linear", "cosine", "quadratic"] = "linear"
    beta_start: float = 1e-4
    beta_end: float = 0.02
    # For cosine schedule
    s: float = 0.008  # Small offset to prevent beta from being too small


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 128
    learning_rate: float = 2e-4
    total_steps: int = 800_000
    warmup_steps: int = 5000
    grad_clip: float = 1.0
    ema_decay: float = 0.9999
    save_every: int = 10000
    sample_every: int = 10000
    log_every: int = 100
    num_workers: int = 4
    mixed_precision: bool = True


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    num_samples: int = 50000  # For FID calculation
    batch_size: int = 256
    fid_stats_path: Optional[str] = None  # Path to precomputed stats


@dataclass
class Config:
    """Main configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # Paths
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"
    sample_dir: str = "./samples"
    log_dir: str = "./logs"

    # Dataset
    dataset: str = "cifar10"
    data_dir: str = "./data"

    # Device
    device: str = "cuda"
    seed: int = 42

    # Experiment name
    exp_name: str = "ddpm_cifar10_baseline"


def get_config(
    timesteps: int = 1000,
    beta_schedule: str = "linear",
    exp_name: Optional[str] = None,
) -> Config:
    """Get configuration with optional overrides."""
    config = Config()
    config.diffusion.timesteps = timesteps
    config.diffusion.beta_schedule = beta_schedule

    if exp_name is None:
        exp_name = f"ddpm_cifar10_T{timesteps}_{beta_schedule}"
    config.exp_name = exp_name

    return config


# Ablation configurations
ABLATION_TIMESTEPS = [100, 500, 1000, 2000]
ABLATION_SCHEDULES = ["linear", "cosine", "quadratic"]
