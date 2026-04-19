"""
GGUF detection utilities and lightweight LoRA handle for sampler backends.

The actual LoRA forward for GGUF modules uses quantized_compile_forward
(from compile_forward.py), which calls the original GGUF forward unchanged
and adds LoRA as a separate additive term — matching OT's LoRAModule pattern.
"""
import torch


def is_gguf_module(module: torch.nn.Module) -> bool:
    """Check if module uses GGUF quantized weights (has quant_type on weight)."""
    w = getattr(module, "weight", None)
    return w is not None and hasattr(w, "quant_type")


class FactorRef:
    """Lightweight no-op handle for additional LoRAs on an already-patched module."""
    __slots__ = ()

    def remove(self) -> None:
        pass
