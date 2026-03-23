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
    # Subclasses declare which cfg keys constitute model identity.
    # If any of these differ from the last successful load_model call,
    # model_needs_reload() returns True and the queue worker reloads.
    MODEL_IDENTITY_KEYS: tuple[str, ...] = ()

    def __init__(self):
        self.model = None
        self.lora_hooks: list = []
        self.train_device = torch.device("cuda")
        self.temp_device  = torch.device("cpu")
        self._cancel_event = threading.Event()
        # Signature of the currently applied LoRA set: list of (path, weight).
        # None means nothing is applied or the state is unknown.
        self._applied_lora_sig: list[tuple[str, float]] | None = None
        # cfg snapshot from the last successful load_model(); None if nothing loaded.
        self._loaded_cfg: dict | None = None

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def model_needs_reload(self, cfg: dict) -> bool:
        """Return True if cfg specifies a different model than what is loaded."""
        if self.model is None or self._loaded_cfg is None:
            return True
        return any(cfg.get(k) != self._loaded_cfg.get(k)
                   for k in self.MODEL_IDENTITY_KEYS)

    @abstractmethod
    def load_model(self, cfg: dict, on_status) -> None:
        """Load model using parameters from cfg.  Call on_status(str) for progress.
        Implementations must set self._loaded_cfg = cfg on successful completion."""

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

            # Sync any pending async CUDA transfers (offload conductor uses
            # background streams; unsynchronized streams hold tensor references).
            try:
                torch.cuda.synchronize()
            except Exception:
                pass

            # Break the conductor ↔ OffloadCheckpointLayer reference cycle
            # explicitly before GC.  Each OffloadCheckpointLayer stores a back-
            # reference to its conductor, and the conductor's __layers list holds
            # every layer — forming a cycle gc.collect() would eventually break,
            # but clearing it here lets refcounting free everything immediately.
            model = self.model
            self.model = None
            self._loaded_cfg = None
            for _cond_attr in ("transformer_offload_conductor",
                               "transformer_2_offload_conductor"):
                conductor = getattr(model, _cond_attr, None)
                if conductor is not None:
                    # Clear the internal layer list via Python name-mangling path.
                    for _inner in ("_LayerOffloadConductor__layers",
                                   "_LayerOffloadConductor__layer_device_map",
                                   "_LayerOffloadConductor__activations_map"):
                        try:
                            inner = getattr(conductor, _inner)
                            if isinstance(inner, list):
                                inner.clear()
                            elif isinstance(inner, dict):
                                inner.clear()
                        except AttributeError:
                            pass
                    try:
                        delattr(model, _cond_attr)
                    except AttributeError:
                        pass

            for _attr in ("transformer", "transformer_2", "text_encoder", "vae"):
                try:
                    delattr(model, _attr)
                except AttributeError:
                    pass
            del model
            gc.collect()
            gc.collect()  # second pass catches any remaining cycles
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
                _status(f"[LoRA] {name} — 0 handles returned (key format mismatch?)")
            else:
                _status(f"[LoRA] Applied {name}  weight={weight}  handles={len(hooks)}")

        _status(
            f"LoRAs done — {applied} applied, {skipped} skipped"
            f"  ({total_hooks} total handles, {len(self.lora_hooks)} in active list)"
        )
        # Record what is now applied so the queue can skip redundant reloads.
        self._applied_lora_sig = [
            (e.get("path", ""), float(e.get("weight", 1.0)))
            for e in loras
            if e.get("enabled", True) and os.path.isfile(e.get("path", ""))
        ]

    def remove_loras(self) -> None:
        # Reverse order is required for _ForwardPatch correctness: each patch
        # wraps the previous one, so the last-applied patch must be unwound
        # first to restore the original forward chain.  _WeightMerge handles
        # are order-independent (delta subtraction commutes), so reversed() is
        # safe for both types.
        for h in reversed(self.lora_hooks):
            h.remove()
        self.lora_hooks.clear()
        self._applied_lora_sig = None
        gc.collect()
        torch.cuda.empty_cache()

    def cancel(self) -> None:
        self._cancel_event.set()
