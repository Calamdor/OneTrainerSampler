"""
GGUF compile-friendly LoRA forward functions.

Three variants for different GGUF quantization types:
  - plain: dequant → merge → F.linear (no requantization)
  - int8:  dequant → merge → int8 quantize-matmul-dequantize
  - fp8:   dequant → merge → fp8 quantize-matmul-dequantize

All share the same setup via _prepare_weight_and_bias().
"""
import torch
import torch.nn.functional as F

_GGUF_FP8_MAX = 448.0  # torch.finfo(torch.float8_e4m3fn).max


def is_gguf_module(module: torch.nn.Module) -> bool:
    """Check if module uses GGUF quantized weights (has quant_type on weight)."""
    w = getattr(module, "weight", None)
    return w is not None and hasattr(w, "quant_type")


def _gguf_dequant_weight(weight):
    """Dequantize a GGUF weight tensor.

    Uses only standard torch ops — fully traceable by dynamo with
    fullgraph=True (matching OneTrainer's LinearGGUFA8.forward).
    """
    from diffusers.quantizers.gguf.utils import dequantize_gguf_tensor
    return dequantize_gguf_tensor(weight.detach())


def _prepare_weight_and_bias(self):
    """Shared setup: dequant → dtype/device → merge LoRA → get bias."""
    w = _gguf_dequant_weight(self.weight)
    dt = self._gguf_compile_dt
    w = w.to(device=self._gguf_compile_dev, dtype=dt)
    if self._lora_d is not None:
        w = w + self._lora_u @ self._lora_d
    bias = self.bias
    if bias is not None:
        bias = bias.to(dtype=dt)
    return w, bias, dt


def gguf_compile_forward_plain(self, x):
    """Plain GGUF: dequant → merge → F.linear (no requantization)."""
    w, bias, _ = _prepare_weight_and_bias(self)
    x_2d = x.reshape(-1, x.shape[-1])
    y = F.linear(x_2d, w, bias)
    return y.reshape(x.shape[:-1] + (y.shape[-1],))


def gguf_compile_forward_int8(self, x):
    """GGUF_A8I: dequant → merge → int8 quantize-matmul-dequantize."""
    w, bias, dt = _prepare_weight_and_bias(self)
    x_2d = x.reshape(-1, x.shape[-1])

    x_absmax = x_2d.abs().amax(dim=-1, keepdim=True).clamp_(min=1e-7)
    x_scale = x_absmax.mul_(1.0 / 127.0)
    x_8 = x_2d.div(x_scale).round_().to(torch.int8)

    w_absmax = w.abs().amax(dim=-1, keepdim=True).clamp_(min=1e-7)
    w_scale = w_absmax.mul_(1.0 / 127.0)
    w_8 = w.div(w_scale).round_().to(torch.int8)

    y = torch._int_mm(x_8, w_8.t())
    y = y.float().mul_(w_scale.t()).mul_(x_scale).to(dtype=dt)
    if bias is not None:
        y = y.add_(bias)
    return y.reshape(x.shape[:-1] + (y.shape[-1],))


def gguf_compile_forward_fp8(self, x):
    """GGUF_A8F: dequant → merge → fp8 quantize-matmul-dequantize."""
    try:
        w, bias, dt = _prepare_weight_and_bias(self)
        x_2d = x.reshape(-1, x.shape[-1])

        if x_2d.shape[0] <= 16:
            y = F.linear(x_2d, w, bias)
            return y.reshape(x.shape[:-1] + (y.shape[-1],))

        x_scale = x_2d.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / _GGUF_FP8_MAX
        x_fp8 = (x_2d / x_scale).clamp(-_GGUF_FP8_MAX, _GGUF_FP8_MAX).to(torch.float8_e4m3fn)

        w_scale = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / _GGUF_FP8_MAX
        w_fp8 = (w / w_scale).clamp(-_GGUF_FP8_MAX, _GGUF_FP8_MAX).to(torch.float8_e4m3fn)

        one = torch.ones(1, device=self._gguf_compile_dev, dtype=torch.float32)
        y = torch._scaled_mm(x_fp8, w_fp8.t(), scale_a=one, scale_b=one, out_dtype=torch.float32)
        y = y.mul(w_scale.t()).mul(x_scale).to(dtype=dt)

        if bias is not None:
            y = y.add_(bias)
        return y.reshape(x.shape[:-1] + (y.shape[-1],))
    except Exception as _e:
        d_shape = getattr(self._lora_d, 'shape', None)
        u_shape = getattr(self._lora_u, 'shape', None)
        raise RuntimeError(
            f"gguf_compile_forward_fp8 failed: weight_byte_shape={self.weight.shape} "
            f"x_in={x.shape} lora_d={d_shape} lora_u={u_shape}: {_e}"
        ) from _e


def select_gguf_forward(module: torch.nn.Module):
    """Select the right GGUF forward variant based on module._dtype."""
    _mod_dtype = getattr(module, '_dtype', None)
    if _mod_dtype == torch.float8_e4m3fn:
        return gguf_compile_forward_fp8
    elif _mod_dtype == torch.int8:
        return gguf_compile_forward_int8
    return gguf_compile_forward_plain


class GGUFCompilePatch:
    """Handle for undoing a GGUF compile-friendly forward patch."""
    __slots__ = ("_module", "_orig_forward")

    def __init__(self, module: torch.nn.Module, orig_forward):
        self._module = module
        self._orig_forward = orig_forward

    def remove(self) -> None:
        self._module.forward = self._orig_forward
        for attr in ('_lora_factors', '_lora_d', '_lora_u',
                     '_gguf_compile_dt', '_gguf_compile_dev',
                     '_gguf_compile_is_fp8'):
            try:
                delattr(self._module, attr)
            except AttributeError:
                pass


class FactorRef:
    """Lightweight no-op handle for additional LoRAs on an already-patched module."""
    __slots__ = ()

    def remove(self) -> None:
        pass
