# %%
# 出力ディレクトリ
import os

os.environ["DIFFUSERS_NO_ADVISORY_WARNINGS"] = "1"
OUTPUT_DIR = "./outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory: {OUTPUT_DIR}")

# %%
import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import wandb
from diffusers import AutoencoderKL
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from tqdm import tqdm

# ローカル（M4 Mac）固定
DEVICE = torch.device("mps")
print("Device:", DEVICE)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class ModelConfig:
    image_size: int = 32  # 128 / 4 = 32 latent size
    in_channels: int = 4
    out_channels: int = 4
    model_channels: int = 128  # Larger for 32x32 latent
    channel_mult: tuple = (1, 2, 3, 4)  # More levels for 32x32
    num_res_blocks: int = 2
    attention_resolutions: tuple = (16, 8)  # Attention at 16x16 and 8x8
    dropout: float = 0.1
    num_heads: int = 8
    use_scale_shift_norm: bool = True


@dataclass
class DiffusionConfig:
    timesteps: int = 1000
    beta_schedule: Literal["linear", "cosine", "quadratic"] = "cosine"
    beta_start: float = 1e-4
    beta_end: float = 0.02
    s: float = 0.008


@dataclass
class VAEConfig:
    model_id: str = "stabilityai/sd-vae-ft-mse"
    subfolder: Optional[str] = None
    downsample_factor: int = 8  # f=8 for 256px -> 32x32 latent
    latent_channels: int = 4
    latent_scaling_factor: float = 0.18215
    cache_dir: str = "./data/latents"
    use_fp16: bool = False


@dataclass
class DataConfig:
    dataset: str = "celebahq"
    data_dir: str = "./data"
    image_size: int = 256
    use_drive_cache: bool = True
    drive_cache_dir: str = "./data/drive_cache"


@dataclass
class TrainingConfig:
    batch_size: int = 8
    learning_rate: float = 2e-4
    total_steps: int = 100_000
    warmup_steps: int = 5000
    grad_clip: float = 1.0
    ema_decay: float = 0.9999
    save_every: int = 5000
    sample_every: int = 5000
    log_every: int = 100
    num_workers: int = 0
    mixed_precision: bool = False


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    vae: VAEConfig = field(default_factory=VAEConfig)
    data: DataConfig = field(default_factory=DataConfig)
    output_dir: str = "./outputs"
    exp_name: str = "ldm_celebahq_mps"
    seed: int = 42


config = Config()

config.model.image_size = config.data.image_size // config.vae.downsample_factor
config.model.in_channels = config.vae.latent_channels
config.model.out_channels = config.vae.latent_channels

set_seed(config.seed)
print("Config:")
print(f"  Dataset: {config.data.dataset}")
print(f"  Image size: {config.data.image_size}px")
print(f"  Downsample factor: {config.vae.downsample_factor}")
print(f"  Latent size: {config.model.image_size}x{config.model.image_size}")

# %%
# Directory Config
# ローカル実行では data/ と outputs/ を使用します
DATA_ROOT = Path("./data")
OUTPUT_ROOT = Path("./outputs")

DATA_ROOT.mkdir(parents=True, exist_ok=True)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
config.data.data_dir = str(DATA_ROOT)
config.output_dir = str(OUTPUT_ROOT)
OUTPUT_DIR = config.output_dir
print("Data dir:", config.data.data_dir)
print("Output dir:", config.output_dir)

# %%
# Encoding Latensts
import itertools
from pathlib import Path

import torch
from diffusers import AutoencoderKL
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from tqdm import tqdm


def _iter_images(path: Path, extensions=(".jpg", ".jpeg", ".png", ".webp")):
    for ext in extensions:
        yield from path.rglob(f"*{ext}")
        yield from path.rglob(f"*{ext.upper()}")


def download_celebahq():
    # ローカルのデータのみ使用する
    local_candidates = [
        Path(config.data.data_dir) / "celeba_hq_256",
        Path("./data") / "celeba_hq_256",
    ]

    for cand in local_candidates:
        # Check for at least one image file
        for _ in _iter_images(cand):
            print(f"CelebA-HQ found locally: {cand}")
            return cand

    local_dir = local_candidates[0]
    local_dir.mkdir(parents=True, exist_ok=True)

    msg = f"CelebA-HQ が見つかりません。以下に画像を配置してください: {local_dir}"
    raise FileNotFoundError(msg)


def download_lsun_churches():
    # Download LSUN Churches dataset (subset for training).
    data_dir = Path(config.data.data_dir) / "lsun_churches"
    data_dir.mkdir(parents=True, exist_ok=True)

    existing_images = (
        list(data_dir.glob("*.jpg"))
        + list(data_dir.glob("*.webp"))
        + list(data_dir.glob("*.png"))
    )
    if len(existing_images) > 1000:
        print(f"LSUN Churches already downloaded: {len(existing_images)} images")
        return data_dir

    print("Downloading LSUN Churches dataset...")
    print("Note: Full LSUN is very large. Using a smaller subset for demo.")
    # You can replace this with your own LSUN source
    print("Please manually download LSUN Churches and place images in:", data_dir)
    print('Or switch to CelebA-HQ by changing config.data.dataset to "celebahq"')

    return data_dir


# Generic ImageFolder dataset
class ImageFolderDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        transform=None,
        extensions=(".jpg", ".jpeg", ".png", ".webp"),
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.extensions = extensions
        self.image_paths = self._find_images()
        print(f"Found {len(self.image_paths)} images in {data_dir}")

    def _find_images(self):
        image_paths = []
        for ext in self.extensions:
            image_paths.extend(self.data_dir.rglob(f"*{ext}"))
            image_paths.extend(self.data_dir.rglob(f"*{ext.upper()}"))
        return sorted(image_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, 0  # Dummy label


def get_dataloader(
    dataset_name: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    train: bool = True,
):
    # Get dataloader for CelebA-HQ or LSUN Churches.
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip()
            if train
            else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    if dataset_name == "celebahq":
        data_dir = download_celebahq()
    elif dataset_name == "lsun_churches":
        data_dir = download_lsun_churches()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataset = ImageFolderDataset(str(data_dir), transform=transform)
    if len(dataset) == 0:
        raise ValueError(f"No images found in {data_dir}")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        drop_last=train,
    )


def load_vae():
    vae = AutoencoderKL.from_pretrained(
        config.vae.model_id, subfolder=config.vae.subfolder
    )
    vae = vae.to(DEVICE, dtype=torch.float32).eval()
    for p in vae.parameters():
        p.requires_grad = False
    return vae


@torch.no_grad()
def prepare_latents(vae, force_recompute: bool = False):
    cache_dir = Path(config.vae.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_id = config.vae.model_id.replace("/", "_")
    cache_path = (
        cache_dir
        / f"{config.data.dataset}_{config.data.image_size}_f{config.vae.downsample_factor}_{safe_id}_latents.pt"
    )

    if cache_path.exists() and not force_recompute:
        payload = torch.load(cache_path, map_location="cpu")
        dataset = TensorDataset(payload["latents"], payload["labels"])
        print(f"Loaded cached latents: {cache_path}")
        print(f"  Shape: {payload['latents'].shape}")
        return dataset

    loader = get_dataloader(
        config.data.dataset,
        config.data.image_size,
        config.training.batch_size,
        config.training.num_workers,
        train=True,
    )

    print(
        f"Encoding latents for {config.data.dataset} at {config.data.image_size}px..."
    )
    print(f"  VAE: {config.vae.model_id}")
    print(f"  Downsample factor: {config.vae.downsample_factor}")
    print(
        f"  Expected latent size: {config.model.image_size}x{config.model.image_size}"
    )

    latents_list, labels_list = [], []
    for images, labels in tqdm(loader, desc="Encoding latents"):
        images = images.to(DEVICE)
        scale = getattr(vae.config, "scaling_factor", config.vae.latent_scaling_factor)
        dist = vae.encode(images).latent_dist
        latents = dist.sample() * scale
        latents_list.append(latents.cpu())
        labels_list.append(labels.cpu())

    latents = torch.cat(latents_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    torch.save({"latents": latents, "labels": labels}, cache_path)
    print(f"Saved latents to: {cache_path}")
    print(f"  Shape: {latents.shape}")
    return TensorDataset(latents, labels)


vae = load_vae()
# Sync config with actual VAE settings (fix downsample mismatch + scaling)
config.vae.latent_scaling_factor = getattr(
    vae.config, "scaling_factor", config.vae.latent_scaling_factor
)
config.vae.downsample_factor = 2 ** (len(vae.config.block_out_channels) - 1)
config.model.image_size = config.data.image_size // config.vae.downsample_factor
config.model.in_channels = vae.config.latent_channels
config.model.out_channels = vae.config.latent_channels

print(f"VAE scaling_factor: {config.vae.latent_scaling_factor}")
print(f"VAE downsample_factor: {config.vae.downsample_factor}")
print(f"Latent size: {config.model.image_size}x{config.model.image_size}")
latent_dataset = prepare_latents(vae, force_recompute=False)
latent_loader = DataLoader(
    latent_dataset,
    batch_size=config.training.batch_size,
    shuffle=True,
    num_workers=config.training.num_workers,
    drop_last=True,
)
print(f"Latent batches: {len(latent_loader)}")

# %%
# UNet (Latent Space)


class SinusoidalPositionEmbeddings(nn.Module):
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
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)


def normalization(channels: int, num_groups: int = 32) -> nn.Module:
    return GroupNorm32(min(num_groups, channels), channels)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        time_emb_dim,
        dropout=0.1,
        use_scale_shift_norm=True,
    ):
        super().__init__()
        self.use_scale_shift_norm = use_scale_shift_norm
        self.norm1 = normalization(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = normalization(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        time_out_dim = out_channels * 2 if use_scale_shift_norm else out_channels
        self.time_mlp = nn.Sequential(Swish(), nn.Linear(time_emb_dim, time_out_dim))
        self.dropout = nn.Dropout(dropout)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, time_emb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        time_emb = self.time_mlp(time_emb)
        time_emb = time_emb[:, :, None, None]
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
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = normalization(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = self.head_dim**-0.5

    def forward(self, x):
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm)
        qkv = qkv.reshape(b, 3, self.num_heads, self.head_dim, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        q = q.permute(0, 1, 3, 2)
        k = k.permute(0, 1, 3, 2)
        v = v.permute(0, 1, 3, 2)
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).reshape(b, c, h, w)
        out = self.proj(out)
        return x + out


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.image_size = cfg.image_size
        time_emb_dim = cfg.model_channels * 4
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(cfg.model_channels),
            nn.Linear(cfg.model_channels, time_emb_dim),
            Swish(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.init_conv = nn.Conv2d(cfg.in_channels, cfg.model_channels, 3, padding=1)

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        channels = [cfg.model_channels]
        ch = cfg.model_channels
        resolution = cfg.image_size

        for level, mult in enumerate(cfg.channel_mult):
            out_ch = cfg.model_channels * mult
            for _ in range(cfg.num_res_blocks):
                layers = [
                    ResidualBlock(
                        ch, out_ch, time_emb_dim, cfg.dropout, cfg.use_scale_shift_norm
                    )
                ]
                ch = out_ch
                if resolution in cfg.attention_resolutions:
                    layers.append(AttentionBlock(ch, cfg.num_heads))
                self.downs.append(nn.ModuleList(layers))
                channels.append(ch)
            if level != len(cfg.channel_mult) - 1:
                self.downs.append(nn.ModuleList([Downsample(ch)]))
                channels.append(ch)
                resolution //= 2

        self.mid = nn.ModuleList(
            [
                ResidualBlock(
                    ch, ch, time_emb_dim, cfg.dropout, cfg.use_scale_shift_norm
                ),
                AttentionBlock(ch, cfg.num_heads),
                ResidualBlock(
                    ch, ch, time_emb_dim, cfg.dropout, cfg.use_scale_shift_norm
                ),
            ]
        )

        for level, mult in enumerate(reversed(cfg.channel_mult)):
            out_ch = cfg.model_channels * mult
            for i in range(cfg.num_res_blocks + 1):
                skip_ch = channels.pop()
                layers = [
                    ResidualBlock(
                        ch + skip_ch,
                        out_ch,
                        time_emb_dim,
                        cfg.dropout,
                        cfg.use_scale_shift_norm,
                    )
                ]
                ch = out_ch
                if resolution in cfg.attention_resolutions:
                    layers.append(AttentionBlock(ch, cfg.num_heads))
                if level != len(cfg.channel_mult) - 1 and i == cfg.num_res_blocks:
                    layers.append(Upsample(ch))
                    resolution *= 2
                self.ups.append(nn.ModuleList(layers))

        self.final_norm = normalization(ch)
        self.final_conv = nn.Conv2d(ch, cfg.out_channels, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_embed(t)
        h = self.init_conv(x)
        hs = [h]
        for layers in self.downs:
            for layer in layers:
                if isinstance(layer, ResidualBlock):
                    h = layer(h, t_emb)
                else:
                    h = layer(h)
            hs.append(h)
        for layer in self.mid:
            if isinstance(layer, ResidualBlock):
                h = layer(h, t_emb)
            else:
                h = layer(h)
        for layers in self.ups:
            h = torch.cat([h, hs.pop()], dim=1)
            for layer in layers:
                if isinstance(layer, ResidualBlock):
                    h = layer(h, t_emb)
                else:
                    h = layer(h)
        h = self.final_norm(h)
        h = F.silu(h)
        return self.final_conv(h)


# %%
# Diffusion + EMA


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


class GaussianDiffusion(nn.Module):
    def __init__(self, model, cfg: DiffusionConfig):
        super().__init__()
        self.model = model
        self.timesteps = cfg.timesteps
        betas = cosine_beta_schedule(cfg.timesteps, cfg.s)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t, noise

    def predict_start_from_noise(self, x_t, t, noise):
        sqrt_recip = self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1 = self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return sqrt_recip * x_t - sqrt_recipm1 * noise

    def q_posterior(self, x_0, x_t, t):
        mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_0
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        log_var = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return mean, log_var

    def p_mean_variance(self, x_t, t):
        predicted_noise = self.model(x_t, t)
        x_0_pred = self.predict_start_from_noise(x_t, t, predicted_noise)
        x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)
        model_mean, model_log_variance = self.q_posterior(x_0_pred, x_t, t)
        return model_mean, model_log_variance

    @torch.no_grad()
    def p_sample(self, x_t, t):
        model_mean, model_log_variance = self.p_mean_variance(x_t, t)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().reshape(-1, *((1,) * (len(x_t.shape) - 1)))
        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise

    @torch.no_grad()
    def sample(self, batch_size, image_size, channels, progress=True):
        device = self.betas.device
        x = torch.randn((batch_size, channels, image_size, image_size), device=device)
        timesteps = list(reversed(range(self.timesteps)))
        if progress:
            timesteps = tqdm(timesteps, desc="Sampling", leave=False)
        for t in timesteps:
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x = self.p_sample(x, t_batch)
        return x

    def compute_loss(self, x_0):
        b = x_0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=x_0.device, dtype=torch.long)
        noise = torch.randn_like(x_0)
        x_t, _ = self.q_sample(x_0, t, noise)
        predicted_noise = self.model(x_t, t)
        return F.mse_loss(predicted_noise, noise)


# %%
# EMA


class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.shadow[name] = (
                1.0 - self.decay
            ) * param.data + self.decay * self.shadow[name]

    def apply_to(self, model):
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state):
        if not state:
            return
        self.decay = state.get("decay", self.decay)
        self.shadow = state.get("shadow", self.shadow)


# %%
# Training
model = UNet(config.model).to(DEVICE)
diffusion = GaussianDiffusion(model, config.diffusion).to(DEVICE)
ema = EMA(model, decay=config.training.ema_decay)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)
scaler = None

warmup_steps = max(0, config.training.warmup_steps)
total_steps = max(1, config.training.total_steps)
if warmup_steps > 0:
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-6,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_steps - warmup_steps),
        eta_min=0.0,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps],
    )
else:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=0.0,
    )

# Count parameters
num_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {num_params:,}")

OUTPUT_DIR = config.output_dir
CHECKPOINT_DIR = Path(config.output_dir) / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def decode_latents(latents):
    scale = getattr(vae.config, "scaling_factor", config.vae.latent_scaling_factor)
    latents = latents / scale
    latents = latents.to(dtype=vae.dtype)
    images = vae.decode(latents).sample
    images = (images + 1) / 2
    return images.clamp(0, 1)


@torch.no_grad()
def vae_sanity_check(n: int = 8, out_path: str = None):
    # VAE-only reconstruction to verify normalization and scaling
    loader = get_dataloader(
        config.data.dataset,
        config.data.image_size,
        batch_size=n,
        num_workers=config.training.num_workers,
        train=False,
    )
    images, _ = next(iter(loader))
    images = images.to(DEVICE)

    dist = vae.encode(images).latent_dist
    latents = dist.sample()
    if latents.shape[-1] != config.model.image_size:
        print(
            f"WARNING: latent size {latents.shape[-1]} != expected {config.model.image_size}"
        )

    scale = getattr(vae.config, "scaling_factor", config.vae.latent_scaling_factor)
    latents_scaled = latents * scale
    recon = decode_latents(latents_scaled)

    images_vis = (images + 1) / 2
    images_vis = images_vis.clamp(0, 1)

    grid = torch.cat([images_vis, recon], dim=0)
    grid = torchvision.utils.make_grid(grid, nrow=n, padding=2)
    if out_path is None:
        out_path = str(Path(OUTPUT_DIR) / "vae_recon.png")
    torchvision.utils.save_image(grid, out_path)
    print(f"Saved VAE sanity check: {out_path}")


def log_samples(step):
    model.eval()
    with torch.no_grad():
        latents = diffusion.sample(
            64, config.model.image_size, config.model.out_channels, progress=False
        )
        images = decode_latents(latents)
    grid = torchvision.utils.make_grid(images, nrow=8, padding=2)
    save_path = Path(OUTPUT_DIR) / f"samples_{step:08d}.png"
    torchvision.utils.save_image(grid, save_path)
    if wandb.run is not None:
        wandb.log({"samples": wandb.Image(grid), "step": step})
    model.train()


def _checkpoint_state(step, running_loss):
    return {
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler,
        "step": step,
        "running_loss": running_loss,
    }


def save_checkpoint(step, running_loss):
    state = _checkpoint_state(step, running_loss)
    ckpt_path = CHECKPOINT_DIR / f"ckpt_{step:08d}.pt"
    torch.save(state, ckpt_path)
    torch.save(state, CHECKPOINT_DIR / "latest.pt")
    print(f"Saved checkpoint: {ckpt_path}")


def find_latest_checkpoint():
    latest = CHECKPOINT_DIR / "latest.pt"
    if latest.exists():
        return latest
    candidates = sorted(CHECKPOINT_DIR.glob("ckpt_*.pt"))
    return candidates[-1] if candidates else None


def load_checkpoint(path):
    state = torch.load(path, map_location=DEVICE)
    step = int(state.get("step", 0))
    running_loss = float(state.get("running_loss", 0.0))
    model.load_state_dict(state["model"])
    ema.load_state_dict(state.get("ema", {}))
    optimizer.load_state_dict(state["optimizer"])
    if state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    else:
        scheduler.last_epoch = step - 1
    print(f"Resumed from checkpoint: {path} (step={step})")
    return step, running_loss


# WANDB_API_KEY が環境変数にあればログイン
WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
if WANDB_API_KEY:
    wandb.login(key=WANDB_API_KEY)

wandb.init(
    dir="./wandb_runs",
    project=f"ldm-{config.data.dataset}",
    name=config.exp_name,
    config={
        "dataset": config.data.dataset,
        "timesteps": config.diffusion.timesteps,
        "beta_schedule": config.diffusion.beta_schedule,
        "batch_size": config.training.batch_size,
        "lr": config.training.learning_rate,
        "image_size": config.data.image_size,
        "latent_size": config.model.image_size,
        "downsample_factor": config.vae.downsample_factor,
        "model_params": num_params,
    },
)

step = 0
running_loss = 0.0
latest_ckpt = find_latest_checkpoint()
if latest_ckpt is not None:
    step, running_loss = load_checkpoint(latest_ckpt)

pbar = tqdm(total=config.training.total_steps, initial=step, desc="Training")

while step < config.training.total_steps:
    for batch in latent_loader:
        if step >= config.training.total_steps:
            break
        latents = batch[0].to(DEVICE)
        optimizer.zero_grad()
        loss = diffusion.compute_loss(latents)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
        optimizer.step()
        scheduler.step()
        ema.update(model)
        step += 1
        running_loss += loss.item()
        if step % config.training.log_every == 0:
            avg_loss = running_loss / config.training.log_every
            lr = scheduler.get_last_lr()[0]
            wandb.log({"loss": avg_loss, "lr": lr, "step": step})
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{lr:.6f}"})
            running_loss = 0.0
        if step % config.training.sample_every == 0:
            log_samples(step)
        if step % config.training.save_every == 0:
            save_checkpoint(step, running_loss)
        pbar.update(1)

pbar.close()
save_checkpoint(step, running_loss)
wandb.finish()
print("Training done.")
