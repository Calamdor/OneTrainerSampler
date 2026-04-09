"""TAESD tiny autoencoder preview decoder + Latent2RGB.

Used to decode latents into RGB preview images during diffusion steps.

  - TAESD (image): for Chroma/Flux — simple conv+upsample stack, auto-downloaded
  - Latent2RGB: for Wan 2.2 — instant matrix multiply, no model needed

Model weights are auto-downloaded to the ``vae/`` folder on first use.
"""
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# ---------------------------------------------------------------------------
# Download URLs — small (~5 MB each)
# ---------------------------------------------------------------------------

_DOWNLOAD_URLS = {
    "chroma": "https://github.com/madebyollin/taesd/raw/main/taef1_decoder.pth",
}

_MODEL_DIR = Path(__file__).resolve().parents[2] / "vae"

# ---------------------------------------------------------------------------
# Wan 2.1/2.2-14B Latent2RGB factors (from ComfyUI latent_formats.py Wan21)
# Fast matrix multiply: 16 latent channels → 3 RGB. No model needed.
# ---------------------------------------------------------------------------

_WAN_RGB_FACTORS = torch.tensor([
    [-0.1299, -0.1692,  0.2932],
    [ 0.0671,  0.0406,  0.0442],
    [ 0.3568,  0.2548,  0.1747],
    [ 0.0372,  0.2344,  0.1420],
    [ 0.0313,  0.0189, -0.0328],
    [ 0.0296, -0.0956, -0.0665],
    [-0.3477, -0.4059, -0.2925],
    [ 0.0166,  0.1902,  0.1975],
    [-0.0412,  0.0267, -0.1364],
    [-0.1293,  0.0740,  0.1636],
    [ 0.0680,  0.3019,  0.1128],
    [ 0.0032,  0.0581,  0.0639],
    [-0.1251,  0.0927,  0.1699],
    [ 0.0060, -0.0633,  0.0005],
    [ 0.3477,  0.2275,  0.2950],
    [ 0.1984,  0.0913,  0.1861],
]).T  # (3, 16) for F.linear

_WAN_RGB_BIAS = torch.tensor([-0.1835, -0.0868, -0.3360])

# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

def _conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class _Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3


class _Block(nn.Module):
    """Residual block: 3×conv + skip."""
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(
            _conv(n_in, n_out), nn.ReLU(),
            _conv(n_out, n_out), nn.ReLU(),
            _conv(n_out, n_out),
        )
        self.skip = (nn.Conv2d(n_in, n_out, 1, bias=False)
                     if n_in != n_out else nn.Identity())
        self.fuse = nn.ReLU()

    def forward(self, x):
        return self.fuse(self.conv(x) + self.skip(x))


# ---------------------------------------------------------------------------
# TAESD image decoder (Chroma / Flux)
# ---------------------------------------------------------------------------

def _build_image_decoder(latent_channels: int = 16) -> nn.Sequential:
    return nn.Sequential(
        _Clamp(), _conv(latent_channels, 64), nn.ReLU(),
        _Block(64, 64), _Block(64, 64), _Block(64, 64),
        nn.Upsample(scale_factor=2), _conv(64, 64, bias=False),
        _Block(64, 64), _Block(64, 64), _Block(64, 64),
        nn.Upsample(scale_factor=2), _conv(64, 64, bias=False),
        _Block(64, 64), _Block(64, 64), _Block(64, 64),
        nn.Upsample(scale_factor=2), _conv(64, 64, bias=False),
        _Block(64, 64), _conv(64, 3),
    )


class TAESDDecoder(nn.Module):
    """Tiny AutoEncoder decoder for image latents (Flux/Chroma)."""

    def __init__(self, latent_channels: int = 16):
        super().__init__()
        self.taesd_decoder = _build_image_decoder(latent_channels)
        self.vae_scale = nn.Parameter(torch.tensor(1.0))
        self.vae_shift = nn.Parameter(torch.tensor(0.0))

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode latent (B, C, H, W) → RGB (B, 3, H*8, W*8) in [-1, 1]."""
        out = self.taesd_decoder((x - self.vae_shift) * self.vae_scale)
        return out.sub(0.5).mul(2)


# ---------------------------------------------------------------------------
# TAEHV video decoder building blocks
# ---------------------------------------------------------------------------

class _MemBlock(nn.Module):
    """Temporal memory block: concat current + previous frame, then conv+skip."""
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(
            _conv(n_in * 2, n_out), nn.ReLU(),
            _conv(n_out, n_out), nn.ReLU(),
            _conv(n_out, n_out),
        )
        self.skip = (nn.Conv2d(n_in, n_out, 1, bias=False)
                     if n_in != n_out else nn.Identity())
        self.act = nn.ReLU()

    def forward(self, x, past):
        return self.act(self.conv(torch.cat([x, past], 1)) + self.skip(x))


class _TGrow(nn.Module):
    """Temporal growth: expand time dimension by stride."""
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f, n_f * stride, 1, bias=False)

    def forward(self, x):
        _NT, C, H, W = x.shape
        x = self.conv(x)
        return x.reshape(-1, C, H, W)


def _build_video_decoder(latent_channels: int = 16,
                         time_upscale=(False, True, True)) -> nn.Sequential:
    """Build TAEHV decoder for Wan 2.1 / Wan 2.2 14B (16-channel latent)."""
    nf = [256, 128, 64, 64]
    act = nn.ReLU()
    return nn.Sequential(
        _Clamp(), _conv(latent_channels, nf[0]), act,
        _MemBlock(nf[0], nf[0]), _MemBlock(nf[0], nf[0]), _MemBlock(nf[0], nf[0]),
        nn.Upsample(scale_factor=2),
        _TGrow(nf[0], 2 if time_upscale[0] else 1),
        _conv(nf[0], nf[1], bias=False),
        _MemBlock(nf[1], nf[1]), _MemBlock(nf[1], nf[1]), _MemBlock(nf[1], nf[1]),
        nn.Upsample(scale_factor=2),
        _TGrow(nf[1], 2 if time_upscale[1] else 1),
        _conv(nf[1], nf[2], bias=False),
        _MemBlock(nf[2], nf[2]), _MemBlock(nf[2], nf[2]), _MemBlock(nf[2], nf[2]),
        nn.Upsample(scale_factor=2),
        _TGrow(nf[2], 2 if time_upscale[2] else 1),
        _conv(nf[2], nf[3], bias=False),
        nn.ReLU(), _conv(nf[3], 3),
    )


def _apply_with_memblocks(model: nn.Sequential, x: torch.Tensor) -> torch.Tensor:
    """Run video decoder with temporal memory (parallel mode)."""
    B, T, C, H, W = x.shape
    x = x.reshape(B * T, C, H, W)
    for block in model:
        if isinstance(block, _MemBlock):
            BT, C2, H2, W2 = x.shape
            T2 = BT // B
            xr = x.reshape(B, T2, C2, H2, W2)
            mem = F.pad(xr, (0, 0, 0, 0, 0, 0, 1, 0), value=0)[:, :T2]
            mem = mem.reshape(x.shape)
            x = block(x, mem)
        elif isinstance(block, _TGrow):
            x = block(x)
        else:
            x = block(x)
    BT, C2, H2, W2 = x.shape
    return x.view(B, BT // B, C2, H2, W2)


class TAEHVDecoder(nn.Module):
    """Tiny AutoEncoder decoder for video latents (Wan 2.2 T2V-A14B)."""

    def __init__(self, latent_channels: int = 16):
        super().__init__()
        self.latent_channels = latent_channels
        self.decoder = _build_video_decoder(latent_channels)
        # Wan 2.1/14B: time_upscale=(False, True, True) → t_upscale=4
        self.t_upscale = 4
        self.frames_to_trim = self.t_upscale - 1  # 3

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode video latent (B, C, T, H, W) → RGB (B, 3, T_out, H*8, W*8).

        For single-frame preview, pass T=1. Output T_out=1 after trimming.
        """
        if x.ndim == 4:
            x = x.unsqueeze(2)  # (B, C, H, W) → (B, C, 1, H, W)
        # (B, C, T, H, W) → (B, T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        x = _apply_with_memblocks(self.decoder, x)
        # (B, T_out, 3, H, W) → trim first frames → (B, 3, T_out, H, W)
        x = x[:, self.frames_to_trim:]
        return x.permute(0, 2, 1, 3, 4)


# ---------------------------------------------------------------------------
# Model loading and caching
# ---------------------------------------------------------------------------

_decoder_cache: dict[str, nn.Module] = {}


def _ensure_model_file(model_type: str) -> Path:
    """Download model file if not present. Returns path."""
    url = _DOWNLOAD_URLS[model_type]
    filename = url.rsplit("/", 1)[-1]
    path = _MODEL_DIR / filename
    if path.exists():
        return path
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[preview] Downloading TAESD decoder: {filename} …")
    try:
        torch.hub.download_url_to_file(url, str(path), progress=True)
        print(f"[preview] Saved to {path}")
    except Exception as exc:
        print(f"[preview] Download failed: {exc}")
        raise
    return path


def get_decoder(model_type: str, device: torch.device) -> nn.Module:
    """Load (and cache) the TAESD/TAEHV decoder for the given model type."""
    if model_type in _decoder_cache:
        dec = _decoder_cache[model_type]
        if next(dec.parameters()).device == device:
            return dec
        dec.to(device)
        return dec

    path = _ensure_model_file(model_type)
    sd = torch.load(str(path), map_location="cpu", weights_only=True)

    if model_type == "chroma":
        dec = TAESDDecoder(latent_channels=16)
        # TAESD image files contain only decoder weights (no prefix)
        dec.taesd_decoder.load_state_dict(sd)
    else:
        raise ValueError(f"Unknown model_type for TAESD: {model_type}")

    dec.eval()
    dec.to(device)
    _decoder_cache[model_type] = dec
    return dec


def unload_decoders() -> None:
    """Free all cached TAESD decoders from GPU."""
    for dec in _decoder_cache.values():
        dec.cpu()
    _decoder_cache.clear()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Chroma latent unpacking
# ---------------------------------------------------------------------------

def unpack_chroma_latent(packed: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Unpack Chroma's packed latent (B, H*W/4, 64) → (B, 16, H, W).

    Args:
        packed: (B, tokens, 64) packed latent from the diffusion loop.
        h: latent height (image_height // 8).
        w: latent width (image_width // 8).
    """
    B = packed.shape[0]
    # (B, h/2 * w/2, 64) → (B, h/2, w/2, 16, 2, 2) → (B, 16, h, w)
    x = packed.view(B, h // 2, w // 2, 16, 2, 2)
    x = x.permute(0, 3, 1, 4, 2, 5)
    return x.reshape(B, 16, h, w)


# ---------------------------------------------------------------------------
# Main preview entry point
# ---------------------------------------------------------------------------

@torch.no_grad()
def decode_preview(
    latent: torch.Tensor,
    model_type: str,
    device: torch.device,
    unpack_hw: tuple[int, int] | None = None,
    vae_scale: float | None = None,
    vae_shift: float | None = None,
    wan_mean: torch.Tensor | None = None,
    wan_std: torch.Tensor | None = None,
) -> Image.Image:
    """Decode a predicted-x0 latent tensor to a preview PIL Image.

    The latent is in diffusion space and must be denormalized to VAE space
    before the TAESD decoder can produce meaningful output.

    Args:
        latent: predicted x0 from the diffusion loop.
            Chroma: packed (B, tokens, 64) — pass unpack_hw=(H//8, W//8).
            Wan: (B, C, T, H, W) video latent.
        model_type: "chroma" or "wan".
        device: GPU device for decode.
        unpack_hw: (H//8, W//8) for Chroma latent unpacking.
        vae_scale: Chroma VAE scaling_factor (default 0.3611).
        vae_shift: Chroma VAE shift_factor (default 0.1159).
        wan_mean: Wan VAE latents_mean tensor (C,) or (1,C,1,1,1).
        wan_std: Wan VAE latents_std tensor (C,) or (1,C,1,1,1).

    Returns:
        PIL.Image.Image in RGB mode.
    """
    if model_type == "chroma":
        dec = get_decoder(model_type, device)
        if unpack_hw is not None:
            x = unpack_chroma_latent(latent[:1], *unpack_hw)
        else:
            x = latent[:1]
        x = x.to(device=device, dtype=torch.float32)
        # Denormalize: diffusion space → VAE space
        # ChromaSampler does: vae_latent = (diff_latent / scaling_factor) + shift_factor
        _s = vae_scale if vae_scale is not None else 0.3611
        _sh = vae_shift if vae_shift is not None else 0.1159
        x = (x / _s) + _sh
        rgb = dec.decode(x)           # (1, 3, H*8, W*8) in [-1, 1]

    elif model_type == "wan":
        # Wan uses Latent2RGB (fast matrix multiply, no model needed).
        # This is what ComfyUI defaults to for Wan previews.
        # Extract a single frame from the 5D video latent.
        if latent.ndim == 5:
            T = latent.shape[2]
            frame_idx = T // 2
            x = latent[:1, :, frame_idx, :, :]   # (1, 16, H, W)
        else:
            x = latent[:1]
        x = x.to(device=device, dtype=torch.float32)
        # Latent2RGB: 16 channels → 3 RGB via learned linear transform.
        # Input is raw diffusion-space latent (the factors are trained on
        # this space — no denormalization needed).
        factors = _WAN_RGB_FACTORS.to(device=device, dtype=torch.float32)
        bias = _WAN_RGB_BIAS.to(device=device, dtype=torch.float32)
        # x is (1, 16, H, W) → movedim to (H, W, 16) → linear → (H, W, 3)
        rgb = F.linear(x[0].movedim(0, -1), factors, bias)
        # Output is in ~[-1, 1], convert to [0, 1]
        rgb = (rgb + 1.0).div(2.0).clamp(0, 1)
        rgb = rgb.mul(255).byte().cpu().numpy()
        img = Image.fromarray(rgb, "RGB")
        # Latent2RGB output is at latent resolution (image/8).
        # Upscale to approximate output size for a useful preview.
        out_w, out_h = img.size[0] * 8, img.size[1] * 8
        return img.resize((out_w, out_h), Image.LANCZOS)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Convert to [0, 255] uint8.
    # TAESD (Chroma) outputs [-1, 1]; TAEHV (Wan) outputs ~[0, 1].
    rgb = rgb[0]
    if model_type == "chroma":
        rgb = rgb.clamp(-1, 1).add(1).div(2)   # [-1, 1] → [0, 1]
    else:
        rgb = rgb.clamp(0, 1)                   # [0, 1] already
    rgb = rgb.mul(255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(rgb, "RGB")
