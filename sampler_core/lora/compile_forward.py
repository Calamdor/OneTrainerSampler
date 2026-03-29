"""
Compile-friendly LoRA forward for non-GGUF quantized modules.

Uses module attributes (not closure variables) so torch.compile sees
stable guards.  LoRA factors stored on CPU, moved to GPU per-transformer
via move_lora_factors_to_device() or per-block via offload_quantized.
"""
import torch
import torch.nn.functional as F


def quantized_compile_forward(self, x):
    """Forward: original quantized forward + LoRA output-addition."""
    y = self._orig_forward_for_lora(x)
    if self._lora_d is not None:
        y = y + F.linear(F.linear(x, self._lora_d), self._lora_u)
    return y


def rebuild_merged_lora(module: torch.nn.Module) -> None:
    """Merge all LoRA factors into a single (D, U) pair via concatenation.

    D_merged = cat([d_1, d_2, …], dim=0)           # [sum(ranks), in_dim]
    U_merged = cat([u_1*s_1, u_2*s_2, …], dim=1)   # [out_dim, sum(ranks)]

    Stored on CPU; moved to GPU by the offload_quantized patch or
    move_lora_factors_to_device().
    """
    factors = getattr(module, '_lora_factors', [])
    if not factors:
        module._lora_d = None
        module._lora_u = None
        return
    ds = [dv for dv, _, _ in factors]
    us = [uv * s for _, uv, s in factors]
    module._lora_d = torch.cat(ds, dim=0).cpu()
    module._lora_u = torch.cat(us, dim=1).cpu()


def move_lora_factors_to_device(transformer: torch.nn.Module, device: torch.device) -> None:
    """Bulk-move all LoRA merged factors in a transformer to the given device."""
    for module in transformer.modules():
        d = getattr(module, '_lora_d', None)
        if d is not None and d.device != device:
            module._lora_d = module._lora_d.to(device, non_blocking=True)
            module._lora_u = module._lora_u.to(device, non_blocking=True)


class QuantizedCompilePatch:
    """Handle for undoing a quantized compile-friendly forward patch."""
    __slots__ = ("_module", "_orig_forward")

    def __init__(self, module: torch.nn.Module, orig_forward):
        self._module = module
        self._orig_forward = orig_forward

    def remove(self) -> None:
        self._module.forward = self._orig_forward
        for attr in ('_lora_factors', '_lora_d', '_lora_u',
                     '_orig_forward_for_lora'):
            try:
                delattr(self._module, attr)
            except AttributeError:
                pass
