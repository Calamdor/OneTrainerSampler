"""
Simple user-controlled layer offload config.

All samplers use the same pattern: checkbox + percentage spinbox.
The OffloadConfig duck-types LayerOffloadConductor's constructor.
Each model backend calls its own OT enable_checkpointing_for_*
function — that call is NOT shared here since it's model-specific.
"""
import sampler_core  # noqa: F401

from modules.util.enum.GradientCheckpointingMethod import GradientCheckpointingMethod


class OffloadConfig:
    """Minimal duck-type satisfying LayerOffloadConductor's interface."""
    def __init__(self, train_device, temp_device, layer_offload_fraction: float,
                 use_compile: bool = False):
        self.train_device              = str(train_device)
        self.temp_device               = str(temp_device)
        self.layer_offload_fraction    = float(layer_offload_fraction)
        self.gradient_checkpointing    = GradientCheckpointingMethod.CPU_OFFLOADED
        self.enable_activation_offloading = False
        self.enable_async_offloading   = True
        self.compile                   = use_compile
