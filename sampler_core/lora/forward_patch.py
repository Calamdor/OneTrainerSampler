"""
Legacy closure-based forward patch — fallback when compile is off.

Replaces module.forward with a wrapper closure that calls the original
forward and adds the LoRA delta.  Works but closures have per-session
Python identity, breaking persistent compile cache.
"""
import torch
import torch.nn.functional as F

_PLAIN_FLOAT = (torch.float16, torch.bfloat16, torch.float32)


class ForwardPatch:
    """Undoable forward-method patch for quantized linear layers.

    When multiple LoRAs target the same layer, patches are stacked.
    Removal MUST happen in reverse application order.
    """
    __slots__ = ("_module", "_orig_forward")

    def __init__(self, module: torch.nn.Module, orig_forward):
        self._module       = module
        self._orig_forward = orig_forward

    def remove(self) -> None:
        self._module.forward = self._orig_forward


def make_forward_patch(orig_fwd, down, up, scale, module, hint_device):
    """Create a closure-based forward patch and apply it to the module."""
    w = getattr(module, "weight", None)
    if w is not None and w.dtype in _PLAIN_FLOAT:
        dev = w.device
    elif hint_device is not None:
        dev = hint_device
    else:
        dev = torch.device("cpu")

    dt = getattr(module, "compute_dtype", None)
    if dt is None:
        dt = (w.dtype if (w is not None and w.dtype in _PLAIN_FLOAT)
              else torch.bfloat16)

    dv = down.to(device=dev, dtype=dt)
    uv = up.to(device=dev, dtype=dt)

    def patched(x):
        return orig_fwd(x) + F.linear(F.linear(x, dv), uv) * scale

    module.forward = patched
    return ForwardPatch(module, orig_fwd)


def log_first_patch(module, hint_device, on_log):
    """Log device/dtype for the first forward-patched module."""
    w = getattr(module, "weight", None)
    dev = (w.device if (w is not None and w.dtype in _PLAIN_FLOAT)
           else hint_device or "cpu")
    dt = getattr(module, "compute_dtype", None) or (
        w.dtype if (w is not None and w.dtype in _PLAIN_FLOAT)
        else torch.bfloat16)
    on_log(f"[LoRA] patch device={dev} dtype={dt}")
