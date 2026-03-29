"""
Weight-merge LoRA paths — bake LoRA deltas directly into model weights.

Two strategies:
  - _WeightMerge: for float weights (BF16/FP16/FP32). Direct in-place add.
  - _QuantizedWeightMerge: for quantized weights (W8A8I/W8A8F/FP8).
    Batched dequant→merge-all→requant in one GPU round-trip.

Both provide .remove() for undoing the merge.
"""
import torch


class _WeightMerge:
    """Undoable in-place LoRA weight merge for float layers.

    Stores low-rank factors (d, u, scale) rather than the materialized delta.
    The delta is recomputed on .remove() — negligible cost vs. the RAM saved.
    """
    __slots__ = ("_module", "_d", "_u", "_scale")

    def __init__(self, module: torch.nn.Module,
                 d: torch.Tensor, u: torch.Tensor, scale: float):
        self._module = module
        self._d      = d      # [rank, in_dim], float32, CPU
        self._u      = u      # [out_dim, rank], float32, CPU
        self._scale  = scale

    def remove(self) -> None:
        w = getattr(self._module, "weight", None)
        if w is not None:
            delta = (self._scale * (self._u @ self._d)).to(device=w.device, dtype=w.dtype)
            w.data.sub_(delta)


class _QuantizedWeightMerge:
    """Undoable LoRA weight merge for quantized modules (W8A8I, W8A8F, FP8).

    Accumulates LoRA deltas per module, then applies them in ONE
    dequant→merge-all→requant pass via flush().  On remove(), reversed.
    """
    __slots__ = ("_module", "_deltas", "_device")

    def __init__(self, module: torch.nn.Module, device: torch.device):
        self._module = module
        self._deltas = []
        self._device = device

    def add(self, d: torch.Tensor, u: torch.Tensor, scale: float) -> None:
        self._deltas.append((d, u, scale))

    def flush(self) -> None:
        self._apply_all(1.0)

    def _apply_all(self, sign: float) -> None:
        module = self._module
        w = module.weight.detach()
        orig_device = w.device

        s = getattr(module, 'scale', None)
        if s is not None:
            w_float = w.to(device=self._device, dtype=torch.float32) * s.to(
                device=self._device, dtype=torch.float32)
        else:
            w_float = w.to(device=self._device, dtype=torch.float32)

        for d, u, sc in self._deltas:
            delta = (sign * sc * (u @ d)).to(device=self._device, dtype=torch.float32)
            w_float.add_(delta)

        _dtype = getattr(module, '_dtype', w.dtype)
        if _dtype == torch.int8:
            from modules.util.quantization_util import quantize_int8_tensorwise
            w_q, new_scale = quantize_int8_tensorwise(w_float)
        elif _dtype == torch.float8_e4m3fn:
            from modules.util.quantization_util import quantize_fp8_tensorwise
            w_q, new_scale = quantize_fp8_tensorwise(w_float)
        else:
            fp8_max = torch.finfo(torch.float8_e4m3fn).max
            abs_max = w_float.abs().max()
            new_scale = torch.clamp(abs_max, min=1e-12) / fp8_max
            w_q = w_float.div_(new_scale).to(dtype=torch.float8_e4m3fn)

        module.weight.data = w_q.to(device=orig_device)
        if s is not None:
            s.copy_(new_scale.to(device=s.device))

    def remove(self) -> None:
        self._apply_all(-1.0)


def can_merge(module: torch.nn.Module) -> bool:
    """True if the module's weight is a plain float tensor (BF16/FP16/FP32)."""
    w = getattr(module, "weight", None)
    return (
        w is not None
        and isinstance(w, torch.Tensor)
        and w.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and w.is_leaf
    )


def can_merge_quantized(module: torch.nn.Module) -> bool:
    """True for quantized modules that support dequant→merge→requantize.

    Excludes SVDQuant (has svd_up/svd_down) — SVD correction layers mean
    the LoRA delta can't simply be merged into the quantized residual.
    """
    if not hasattr(module, 'scale'):
        return False
    if hasattr(module, 'svd_up') or hasattr(module, 'svd_down'):
        return False
    w = getattr(module, "weight", None)
    if w is None:
        return False
    return w.dtype in (torch.int8, torch.float8_e4m3fn)
