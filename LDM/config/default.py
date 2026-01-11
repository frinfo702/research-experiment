"""
Default configuration for LDM training and evaluation.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class ModelConfig:
    """U-Net model configuration."""

    image_size: int = 16  # 128 / 8 = 16 latent size
    in_channels: int = 4
    out_channels: int = 4
    model_channels: int = 128  # Larger for 16x16 latent
    channel_mult: tuple = (1, 2, 2, 4)  # More levels for 16x16
    num_res_blocks: int = 2
    attention_resolutions: tuple = (16, 8)  # Attention at 16x16 and 8x8
    dropout: float = 0.1
    num_heads: int = 8
    use_scale_shift_norm: bool = True


@dataclass
class DiffusionConfig:
    """Diffusion process configuration."""

    timesteps: int = 1000
    beta_schedule: Literal["linear", "cosine", "quadratic"] = "cosine"
    beta_start: float = 1e-4
    beta_end: float = 0.02
    # For cosine schedule
    s: float = 0.008  # Small offset to prevent beta from being too small


@dataclass
class VAEConfig:
    """VAE configuration for latent diffusion."""

    model_id: str = "stabilityai/sd-vae-ft-mse"
    subfolder: Optional[str] = None
    downsample_factor: int = 8  # default value of stabilityai/sd-vae-ft-mse
    latent_channels: int = 4
    latent_scaling_factor: float = 0.18215
    cache_dir: str = "./data/latents"
    use_fp16: bool = True


@dataclass
class DataConfig:
    """Dataset configuration."""

    dataset: str = "celeba_hq_256"  # "celeba_hq_256" or "lsun_churches"
    data_dir: str = "./data"
    image_size: int = 128  # 128px input, 32x32 latent with f=4


@dataclass
class TrainingConfig:
    """Training configuration."""

    batch_size: int = 64
    learning_rate: float = 2e-4
    total_steps: int = 100_000
    warmup_steps: int = 5000
    grad_clip: float = 1.0
    ema_decay: float = 0.9999
    save_every: int = 10000
    sample_every: int = 5000
    log_every: int = 100
    num_workers: int = 4
    mixed_precision: bool = False  # to avoid noises and collapses on mps


@dataclass
class EvalConfig:
    """Evaluation configuration."""

    num_samples: int = 2000  # For FID calculation
    batch_size: int = 32
    fid_stats_path: Optional[str] = None  # Path to precomputed stats


@dataclass
class Config:
    """Main configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    vae: VAEConfig = field(default_factory=VAEConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # Paths
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"
    sample_dir: str = "./samples"
    log_dir: str = "./logs"

    # Device
    device: str = "mps"  # on Mac
    seed: int = 42

    # Experiment name
    exp_name: str = "ldm_celebahq_baseline"


def get_config(
    timesteps: int = 1000,
    beta_schedule: str = "cosine",
    exp_name: Optional[str] = None,
) -> Config:
    """Get configuration with optional overrides."""
    config = Config()
    config.diffusion.timesteps = timesteps
    config.diffusion.beta_schedule = beta_schedule

    # Ensure model input channels and resolution match latent space
    config.model.in_channels = config.vae.latent_channels
    config.model.out_channels = config.vae.latent_channels
    config.model.image_size = config.data.image_size // config.vae.downsample_factor

    if exp_name is None:
        exp_name = f"ldm_{config.data.dataset}_T{timesteps}_{beta_schedule}"
    config.exp_name = exp_name

    return config


# Ablation configurations
ABLATION_TIMESTEPS = [100, 500, 1000, 2000]
ABLATION_SCHEDULES = ["linear", "cosine", "quadratic"]

# Supported datasets
SUPPORTED_DATASETS = ["celeba_hq_256", "lsun_churches", "cifar10"]
