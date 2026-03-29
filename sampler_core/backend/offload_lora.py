"""
Shared offload_quantized monkey-patch for LoRA factor device management.

When offloading is active, the conductor calls offload_quantized() per
module to move weights between CPU/GPU.  This patch extends it to also
move LoRA merged factors (_lora_d/_lora_u) so they follow weights
through the conductor — correct for any offload percentage.
"""
from sampler_core.lora.compile_forward import move_lora_factors_to_device


def setup_offload_lora_patch(has_lora: bool, model_has_conductor: bool,
                              transformer=None, train_device=None):
    """Install the offload_quantized monkey-patch.

    Args:
        has_lora: True if LoRA hooks are active
        model_has_conductor: True if offload conductor is set up
        transformer: for no-offload mode, bulk-move factors here
        train_device: GPU device

    Returns:
        cleanup callable (call in finally block), or None if no patch needed.
    """
    if not has_lora:
        return None

    from modules.util import quantization_util as _qu
    import modules.util.LayerOffloadConductor as _loc

    _orig = _qu.offload_quantized

    def _patched(module, device, non_blocking=False, allocator=None):
        _orig(module, device, non_blocking, allocator)
        d = getattr(module, '_lora_d', None)
        if d is not None:
            module._lora_d = module._lora_d.to(device, non_blocking=non_blocking)
            module._lora_u = module._lora_u.to(device, non_blocking=non_blocking)

    _qu.offload_quantized = _patched
    _loc.offload_quantized = _patched

    # No-offload mode: bulk-move all factors to GPU once.
    if not model_has_conductor and transformer is not None and train_device is not None:
        move_lora_factors_to_device(transformer, train_device)

    def _cleanup():
        _qu.offload_quantized = _orig
        _loc.offload_quantized = _orig

    return _cleanup
