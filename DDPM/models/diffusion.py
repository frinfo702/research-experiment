"""
Diffusion process implementation for DDPM.
Includes forward process (adding noise) and reverse process (denoising).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Literal
from tqdm import tqdm


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """
    Linear beta schedule as proposed in Ho et al. 2020.

    Args:
        timesteps: Number of diffusion steps
        beta_start: Starting beta value
        beta_end: Ending beta value

    Returns:
        Beta values for each timestep [T]
    """
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine beta schedule as proposed in Nichol & Dhariwal 2021.

    This schedule provides better sample quality at the end of the diffusion process.

    Args:
        timesteps: Number of diffusion steps
        s: Small offset to prevent beta from being too small at t=0

    Returns:
        Beta values for each timestep [T]
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


def quadratic_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """
    Quadratic beta schedule.

    Args:
        timesteps: Number of diffusion steps
        beta_start: Starting beta value
        beta_end: Ending beta value

    Returns:
        Beta values for each timestep [T]
    """
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def get_beta_schedule(
    schedule_type: Literal["linear", "cosine", "quadratic"],
    timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    s: float = 0.008,
) -> torch.Tensor:
    """Get beta schedule by name."""
    if schedule_type == "linear":
        return linear_beta_schedule(timesteps, beta_start, beta_end)
    elif schedule_type == "cosine":
        return cosine_beta_schedule(timesteps, s)
    elif schedule_type == "quadratic":
        return quadratic_beta_schedule(timesteps, beta_start, beta_end)
    else:
        raise ValueError(f"Unknown beta schedule: {schedule_type}")


class GaussianDiffusion(nn.Module):
    """
    Gaussian Diffusion Process.

    Implements the forward and reverse diffusion processes from DDPM.
    The forward process adds Gaussian noise to data according to a schedule.
    The reverse process learns to denoise by predicting the noise.
    """

    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 1000,
        beta_schedule: Literal["linear", "cosine", "quadratic"] = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        s: float = 0.008,
        loss_type: Literal["l1", "l2", "huber"] = "l2",
    ):
        """
        Initialize diffusion process.

        Args:
            model: Denoising model (U-Net)
            timesteps: Number of diffusion steps
            beta_schedule: Type of beta schedule
            beta_start: Starting beta (for linear/quadratic)
            beta_end: Ending beta (for linear/quadratic)
            s: Offset for cosine schedule
            loss_type: Type of loss function
        """
        super().__init__()

        self.model = model
        self.timesteps = timesteps
        self.loss_type = loss_type

        # Get beta schedule
        betas = get_beta_schedule(beta_schedule, timesteps, beta_start, beta_end, s)

        # Precompute diffusion parameters
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Register buffers (will be moved to device with model)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # Calculations for diffusion q(x_t | x_0)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped",
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer("posterior_mean_coef1",
                             betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer("posterior_mean_coef2",
                             (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
        """Extract values from a at indices t and reshape for broadcasting."""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion process: q(x_t | x_0).

        Args:
            x_0: Clean images [B, C, H, W]
            t: Timesteps [B]
            noise: Optional pre-generated noise

        Returns:
            Tuple of (noisy images, noise)
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )

        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t, noise

    def predict_start_from_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Predict x_0 from x_t and predicted noise."""
        sqrt_recip_alphas_cumprod_t = self._extract(
            self.sqrt_recip_alphas_cumprod, t, x_t.shape
        )
        sqrt_recipm1_alphas_cumprod_t = self._extract(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
        )
        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * noise

    def q_posterior(
        self,
        x_0: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute posterior q(x_{t-1} | x_t, x_0).

        Returns:
            Tuple of (posterior mean, posterior log variance)
        """
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_0
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_variance = self._extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_log_variance

    def p_mean_variance(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        clip_denoised: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute mean and variance for p(x_{t-1} | x_t).

        Args:
            x_t: Noisy images at timestep t
            t: Current timesteps
            clip_denoised: Whether to clip predicted x_0 to [-1, 1]

        Returns:
            Tuple of (model mean, posterior variance, posterior log variance)
        """
        # Predict noise
        predicted_noise = self.model(x_t, t)

        # Predict x_0
        x_0_pred = self.predict_start_from_noise(x_t, t, predicted_noise)

        if clip_denoised:
            x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)

        # Get posterior
        model_mean, posterior_log_variance = self.q_posterior(x_0_pred, x_t, t)
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)

        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        """
        Sample x_{t-1} from p(x_{t-1} | x_t).

        Args:
            x_t: Noisy images at timestep t
            t: Current timesteps (same value for all samples in batch)
            clip_denoised: Whether to clip predicted x_0

        Returns:
            Denoised images x_{t-1}
        """
        model_mean, _, model_log_variance = self.p_mean_variance(
            x_t, t, clip_denoised
        )

        noise = torch.randn_like(x_t)
        # No noise at t=0
        nonzero_mask = (t != 0).float().reshape(-1, *((1,) * (len(x_t.shape) - 1)))

        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise

    @torch.no_grad()
    def p_sample_loop(
        self,
        shape: tuple,
        return_intermediates: bool = False,
        clip_denoised: bool = True,
        progress: bool = True,
    ) -> torch.Tensor:
        """
        Full reverse diffusion sampling loop.

        Args:
            shape: Shape of samples to generate (B, C, H, W)
            return_intermediates: Whether to return intermediate samples
            clip_denoised: Whether to clip during denoising
            progress: Whether to show progress bar

        Returns:
            Generated samples (and optionally intermediates)
        """
        device = self.betas.device
        batch_size = shape[0]

        # Start from pure noise
        x = torch.randn(shape, device=device)

        intermediates = [x] if return_intermediates else None

        # Reverse diffusion
        timesteps = list(reversed(range(self.timesteps)))
        if progress:
            timesteps = tqdm(timesteps, desc="Sampling", leave=False)

        for t in timesteps:
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x = self.p_sample(x, t_batch, clip_denoised)

            if return_intermediates:
                intermediates.append(x)

        if return_intermediates:
            return x, intermediates
        return x

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        image_size: int,
        channels: int = 3,
        return_intermediates: bool = False,
        clip_denoised: bool = True,
        progress: bool = True,
    ) -> torch.Tensor:
        """
        Generate samples.

        Args:
            batch_size: Number of samples to generate
            image_size: Size of images
            channels: Number of channels
            return_intermediates: Whether to return intermediate samples
            clip_denoised: Whether to clip during denoising
            progress: Whether to show progress bar

        Returns:
            Generated samples [B, C, H, W] in range [-1, 1]
        """
        return self.p_sample_loop(
            shape=(batch_size, channels, image_size, image_size),
            return_intermediates=return_intermediates,
            clip_denoised=clip_denoised,
            progress=progress,
        )

    def compute_loss(
        self,
        x_0: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute training loss.

        Args:
            x_0: Clean images [B, C, H, W]
            t: Optional timesteps (sampled uniformly if not provided)
            noise: Optional noise (sampled if not provided)

        Returns:
            Loss value
        """
        batch_size = x_0.shape[0]
        device = x_0.device

        # Sample timesteps uniformly
        if t is None:
            t = torch.randint(0, self.timesteps, (batch_size,), device=device, dtype=torch.long)

        # Sample noise
        if noise is None:
            noise = torch.randn_like(x_0)

        # Forward diffusion
        x_t, _ = self.q_sample(x_0, t, noise)

        # Predict noise
        predicted_noise = self.model(x_t, t)

        # Compute loss
        if self.loss_type == "l1":
            loss = F.l1_loss(predicted_noise, noise)
        elif self.loss_type == "l2":
            loss = F.mse_loss(predicted_noise, noise)
        elif self.loss_type == "huber":
            loss = F.smooth_l1_loss(predicted_noise, noise)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return loss

    def forward(self, x_0: torch.Tensor) -> torch.Tensor:
        """Forward pass computes the training loss."""
        return self.compute_loss(x_0)


def create_diffusion(model: nn.Module, config) -> GaussianDiffusion:
    """Create diffusion process from config."""
    return GaussianDiffusion(
        model=model,
        timesteps=config.diffusion.timesteps,
        beta_schedule=config.diffusion.beta_schedule,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        s=config.diffusion.s,
    )
