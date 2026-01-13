"""
U-Net architecture for DDPM.
Based on the architecture from Ho et al. 2020, with modifications from
Nichol & Dhariwal 2021 (Improved DDPM).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def exists(x):
    return x is not None


def default(val, d):
    return val if exists(val) else d


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for timestep encoding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Swish(nn.Module):
    """Swish activation function."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    """GroupNorm with float32 computation for stability."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)


def normalization(channels: int, num_groups: int = 32) -> nn.Module:
    """Create a normalization layer."""
    return GroupNorm32(min(num_groups, channels), channels)


class ResidualBlock(nn.Module):
    """
    Residual block with timestep conditioning.
    Uses GroupNorm and optional scale-shift conditioning.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.1,
        use_scale_shift_norm: bool = True,
    ):
        super().__init__()
        self.use_scale_shift_norm = use_scale_shift_norm

        self.norm1 = normalization(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.norm2 = normalization(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        # Time embedding projection
        time_out_dim = out_channels * 2 if use_scale_shift_norm else out_channels
        self.time_mlp = nn.Sequential(
            Swish(),
            nn.Linear(time_emb_dim, time_out_dim),
        )

        self.dropout = nn.Dropout(dropout)

        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # Add time embedding
        time_emb = self.time_mlp(time_emb)
        time_emb = rearrange(time_emb, "b c -> b c 1 1")

        if self.use_scale_shift_norm:
            scale, shift = time_emb.chunk(2, dim=1)
            h = self.norm2(h) * (1 + scale) + shift
        else:
            h = h + time_emb
            h = self.norm2(h)

        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Multi-head self-attention block."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.norm = normalization(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

        self.scale = self.head_dim**-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        # Normalize
        x_norm = self.norm(x)

        # QKV projection
        qkv = self.qkv(x_norm)
        qkv = rearrange(qkv, "b (three heads d) h w -> three b heads (h w) d",
                        three=3, heads=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        attn = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        attn = attn.softmax(dim=-1)

        # Aggregate
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = rearrange(out, "b heads (h w) d -> b (heads d) h w", h=h, w=w)

        # Project
        out = self.proj(out)

        return x + out


class Downsample(nn.Module):
    """Downsample by factor of 2."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Upsample by factor of 2."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net model for DDPM.

    Architecture:
    - Encoder with residual blocks and downsampling
    - Middle block with attention
    - Decoder with residual blocks, attention, and upsampling
    - Skip connections between encoder and decoder
    """

    def __init__(
        self,
        image_size: int = 32,
        in_channels: int = 3,
        out_channels: int = 3,
        model_channels: int = 128,
        channel_mult: tuple = (1, 2, 2, 2),
        num_res_blocks: int = 2,
        attention_resolutions: tuple = (16,),
        dropout: float = 0.1,
        num_heads: int = 4,
        use_scale_shift_norm: bool = True,
    ):
        super().__init__()

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels

        # Time embedding
        time_emb_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels, time_emb_dim),
            Swish(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Initial projection
        self.init_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        # Build encoder
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        channels = [model_channels]
        ch = model_channels
        resolution = image_size

        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult

            for _ in range(num_res_blocks):
                layers = [
                    ResidualBlock(
                        ch, out_ch, time_emb_dim, dropout, use_scale_shift_norm
                    )
                ]
                ch = out_ch

                if resolution in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads))

                self.downs.append(nn.ModuleList(layers))
                channels.append(ch)

            # Downsample (except last level)
            if level != len(channel_mult) - 1:
                self.downs.append(nn.ModuleList([Downsample(ch)]))
                channels.append(ch)
                resolution //= 2

        # Middle block
        self.mid = nn.ModuleList([
            ResidualBlock(ch, ch, time_emb_dim, dropout, use_scale_shift_norm),
            AttentionBlock(ch, num_heads),
            ResidualBlock(ch, ch, time_emb_dim, dropout, use_scale_shift_norm),
        ])

        # Build decoder
        for level, mult in enumerate(reversed(channel_mult)):
            out_ch = model_channels * mult

            for i in range(num_res_blocks + 1):
                skip_ch = channels.pop()
                layers = [
                    ResidualBlock(
                        ch + skip_ch, out_ch, time_emb_dim, dropout, use_scale_shift_norm
                    )
                ]
                ch = out_ch

                if resolution in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads))

                # Upsample (except last block of each level, and not on last level)
                if level != len(channel_mult) - 1 and i == num_res_blocks:
                    layers.append(Upsample(ch))
                    resolution *= 2

                self.ups.append(nn.ModuleList(layers))

        # Final projection
        self.final_norm = normalization(ch)
        self.final_conv = nn.Conv2d(ch, out_channels, 3, padding=1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with proper scaling."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Zero initialize the final conv for better training stability
        nn.init.zeros_(self.final_conv.weight)
        nn.init.zeros_(self.final_conv.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input images [B, C, H, W]
            t: Timesteps [B]

        Returns:
            Predicted noise [B, C, H, W]
        """
        # Time embedding
        t_emb = self.time_embed(t)

        # Initial conv
        h = self.init_conv(x)
        hs = [h]

        # Encoder
        for layers in self.downs:
            for layer in layers:
                if isinstance(layer, ResidualBlock):
                    h = layer(h, t_emb)
                elif isinstance(layer, AttentionBlock):
                    h = layer(h)
                else:  # Downsample
                    h = layer(h)
            hs.append(h)

        # Middle
        for layer in self.mid:
            if isinstance(layer, ResidualBlock):
                h = layer(h, t_emb)
            else:
                h = layer(h)

        # Decoder
        for layers in self.ups:
            h = torch.cat([h, hs.pop()], dim=1)
            for layer in layers:
                if isinstance(layer, ResidualBlock):
                    h = layer(h, t_emb)
                elif isinstance(layer, AttentionBlock):
                    h = layer(h)
                else:  # Upsample
                    h = layer(h)

        # Final
        h = self.final_norm(h)
        h = F.silu(h)
        h = self.final_conv(h)

        return h


def create_model(config) -> UNet:
    """Create U-Net model from config."""
    return UNet(
        image_size=config.model.image_size,
        in_channels=config.model.in_channels,
        out_channels=config.model.out_channels,
        model_channels=config.model.model_channels,
        channel_mult=config.model.channel_mult,
        num_res_blocks=config.model.num_res_blocks,
        attention_resolutions=config.model.attention_resolutions,
        dropout=config.model.dropout,
        num_heads=config.model.num_heads,
        use_scale_shift_norm=config.model.use_scale_shift_norm,
    )
