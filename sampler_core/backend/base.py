"""
Abstract base class for all sampler backends.

Concrete model backends (ChromaBackend, WanBackend, …) inherit this and
implement the three abstract methods.  unload_model, apply_loras,
remove_loras, and cancel are fully shared.
"""
import gc
import os
import threading
from abc import ABC, abstractmethod

import torch


class Cancelled(Exception):
    """Raised inside sample() when _cancel_event is set between steps."""
    pass


class BaseSamplerBackend(ABC):
    def __init__(self):
        self.model = None
        self.lora_hooks: list = []
        self.train_device = torch.device("cuda")
        self.temp_device  = torch.device("cpu")
        self._cancel_event = threading.Event()
        # Signature of the currently applied LoRA set: list of (path, weight).
        # None means nothing is applied or the state is unknown.
        self._applied_lora_sig: list[tuple[str, float]] | None = None

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def load_model(self, cfg: dict, on_status) -> None:
        """Load model using parameters from cfg.  Call on_status(str) for progress."""

    @abstractmethod
    def sample(self, cfg: dict, on_progress, on_done, on_error) -> None:
        """
        Run inference synchronously (called from a background thread).
        on_progress(step, total), on_done(path_or_None), on_error(msg).
        """

    @abstractmethod
    def _inject_lora(self, state_dict: dict, weight: float, entry: dict) -> list:
        """
        Inject one LoRA into the model.
        Returns a list of hook handles (call handle.remove() to undo).
        `entry` is the full LoRA config dict (path, weight, enabled, model extras).
        """

    # ------------------------------------------------------------------
    # Shared implementations
    # ------------------------------------------------------------------

    def loras_current(self, loras: list[dict]) -> bool:
        """
        Return True if `loras` exactly matches what is already applied.

        Compares only enabled entries with valid paths against the recorded
        signature.  Weight differences smaller than 1e-6 are ignored.
        Callers can skip remove_loras() + apply_loras() when this is True.
        """
        requested = [
            (e["path"], float(e.get("weight", 1.0)))
            for e in loras
            if e.get("enabled", True) and os.path.isfile(e.get("path", ""))
        ]
        if self._applied_lora_sig is None:
            return not requested      # nothing applied ↔ nothing requested
        if len(self._applied_lora_sig) != len(requested):
            return False
        return all(
            a[0] == b[0] and abs(a[1] - b[1]) < 1e-6
            for a, b in zip(self._applied_lora_sig, requested)
        )

    def unload_model(self, on_status=None) -> None:
        from modules.util.torch_util import torch_gc
        self.remove_loras()
        if self.model is not None:
            # Move all components back to CPU in case sampling was interrupted
            # and left something on GPU.
            for _fn in ("transformer_to", "text_encoder_to", "vae_to"):
                try:
                    getattr(self.model, _fn)(self.temp_device)
                except Exception:
                    pass
            torch_gc()  # free VRAM before dropping CPU tensors

            # torch.compile caches compiled graphs that hold tensor references;
            # reset clears those so the underlying storage can be freed.
            try:
                import torch._dynamo
                torch._dynamo.reset()
            except Exception:
                pass

            # Explicitly delete large sub-components by name so their reference
            # counts drop immediately rather than waiting for cyclic GC.
            model = self.model
            self.model = None
            for _attr in ("transformer", "transformer_2",
                          "text_encoder", "vae",
                          "transformer_offload_conductor",
                          "transformer_2_offload_conductor"):
                try:
                    delattr(model, _attr)
                except AttributeError:
                    pass
            del model
            gc.collect()
            gc.collect()  # second pass catches cycles broken by the first
            torch_gc()

        if on_status:
            on_status("Unloaded")

    def apply_loras(self, loras: list[dict], on_status) -> None:
        def _status(msg: str) -> None:
            print(msg)
            on_status(msg)

        if self.model is None:
            _status("Error: model not loaded")
            return

        self.remove_loras()
        total_hooks = 0
        applied = 0
        skipped = 0

        for entry in loras:
            name = os.path.basename(entry.get("path", ""))
            if not entry.get("enabled", True):
                skipped += 1
                continue
            path = entry.get("path", "")
            if not os.path.isfile(path):
                _status(f"[LoRA] NOT FOUND: {name}")
                skipped += 1
                continue
            try:
                _status(f"[LoRA] Loading {name} …")
                from safetensors.torch import load_file
                state_dict = load_file(path)
            except Exception as exc:
                _status(f"[LoRA] ERROR loading {name}: {exc}")
                skipped += 1
                continue

            weight = float(entry.get("weight", 1.0))
            try:
                self._on_log = _status   # let _inject_lora implementations pick this up
                hooks = self._inject_lora(state_dict, weight, entry)
            except Exception as exc:
                import traceback as _tb
                _status(f"[LoRA] ERROR injecting {name}: {exc}")
                _tb.print_exc()
                skipped += 1
                continue
            self.lora_hooks.extend(hooks)
            total_hooks += len(hooks)
            applied += 1
            if len(hooks) == 0:
                _status(f"[LoRA] {name} — 0 hooks applied (key format mismatch?)")
            else:
                _status(f"[LoRA] Applied {name}  w={weight}  hooks={len(hooks)}")

        _status(
            f"LoRAs done — {applied} applied, {skipped} skipped"
            f"  ({total_hooks} total hooks)"
        )
        # Record what is now applied so the queue can skip redundant reloads.
        self._applied_lora_sig = [
            (e.get("path", ""), float(e.get("weight", 1.0)))
            for e in loras
            if e.get("enabled", True) and os.path.isfile(e.get("path", ""))
        ]

    def remove_loras(self) -> None:
        for h in self.lora_hooks:
            h.remove()
        self.lora_hooks.clear()
        self._applied_lora_sig = None

    def cancel(self) -> None:
        self._cancel_event.set()
