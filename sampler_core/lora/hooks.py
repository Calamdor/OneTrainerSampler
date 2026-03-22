"""
Generic LoRA hook injection engine.

apply_lora_hooks() is the single entry point.  Each model provides a
`key_translator` closure that maps a stripped LoRA key to (target, module_path).

For non-quantized linear layers (BF16/FP16/FP32 weight), the LoRA delta is
merged directly into the weight tensor — zero per-step overhead.  Quantized
layers (NF4, W4A16, GGUF, …) fall back to forward hooks with tensor caching.

Both paths return objects with a `.remove()` method so BaseSamplerBackend's
`remove_loras()` works identically for both.
"""
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Weight-merge path
# ---------------------------------------------------------------------------

class _WeightMerge:
    """
    Undoable in-place LoRA weight merge.

    Implements the same .remove() interface as torch hook handles so that
    BaseSamplerBackend.remove_loras() can treat merges and hooks uniformly.

    The delta is stored on CPU (float32) to avoid holding a second copy of the
    model's weights on the GPU.  On .remove() it is moved to the module's
    current device/dtype before subtracting — handles layer-offload correctly.
    """
    __slots__ = ("_module", "_delta_cpu")

    def __init__(self, module: torch.nn.Module, delta_cpu: torch.Tensor):
        self._module   = module
        self._delta_cpu = delta_cpu  # [out_dim, in_dim], float32, CPU

    def remove(self) -> None:
        w = getattr(self._module, "weight", None)
        if w is not None:
            w.data.sub_(self._delta_cpu.to(device=w.device, dtype=w.dtype))


def _can_merge(module: torch.nn.Module) -> bool:
    """
    Return True if the module's weight can be modified in-place.
    Float-dtype nn.Parameter → True.
    Quantized / custom / missing weight → False (fall back to hook).
    """
    w = getattr(module, "weight", None)
    return (
        w is not None
        and isinstance(w, torch.Tensor)
        and w.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and w.is_leaf
    )


# ---------------------------------------------------------------------------
# Module lookup helper
# ---------------------------------------------------------------------------

def get_module_by_dotpath(root: torch.nn.Module, path: str) -> torch.nn.Module:
    """
    Walk `root` via dot-separated `path`, transparently unwrapping OT
    OffloadCheckpointLayer wrappers (.checkpoint attribute).

    torch.compile OptimizedModule wrappers do not need special handling here:
    OptimizedModule.__getattr__ transparently delegates to _orig_mod, so
    attribute traversal through a compiled block works naturally.
    """
    m = root
    for part in path.split("."):
        m = getattr(m, part)
        if hasattr(m, "checkpoint") and isinstance(m.checkpoint, torch.nn.Module):
            m = m.checkpoint
    return m


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def apply_lora_hooks(
        transformer: torch.nn.Module | None,
        text_encoder: torch.nn.Module | None,
        state_dict: dict,
        weight: float,
        key_translator,
        on_log=None,
) -> list:
    """
    Inject a LoRA into `transformer` and/or `text_encoder`.

    For each target module:
      - If the weight is a plain float tensor (BF16/FP16/FP32): merge the
        LoRA delta directly into the weight (fast path, zero per-step cost).
      - Otherwise: install a forward hook that computes the LoRA on the fly
        with cached device tensors (quantized / custom layer fallback).

    key_translator(stripped_key: str) -> (target: str, module_path: str) | None
      target is 'transformer' or 'text_encoder'.
      Return None to skip the key.

    Returns a list of _WeightMerge / hook-handle objects; call .remove() on
    each to undo.  BaseSamplerBackend.remove_loras() does this automatically.
    """
    # Group keys into {(target, mod_path): {down, up, alpha}}
    loras: dict[tuple[str, str], dict] = {}
    for key, tensor in state_dict.items():
        for suf, slot in [
            (".lora_A.weight", "down"),
            (".lora_down.weight", "down"),
            (".lora_B.weight", "up"),
            (".lora_up.weight", "up"),
            (".alpha", "alpha"),
        ]:
            if not key.endswith(suf):
                continue
            stripped = key[: -len(suf)]
            result = key_translator(stripped)
            if result is None:
                break
            target, mod_path = result
            entry = loras.setdefault((target, mod_path), {})
            entry[slot] = tensor.item() if slot == "alpha" else tensor
            break

    handles = []
    fail_count = 0
    merge_count = 0
    hook_count = 0

    for (target, mod_path), lora in loras.items():
        if "down" not in lora or "up" not in lora:
            continue
        root = transformer if target == "transformer" else text_encoder
        if root is None:
            continue
        try:
            module = get_module_by_dotpath(root, mod_path)
        except AttributeError:
            fail_count += 1
            continue

        rank  = lora["down"].shape[0]
        scale = (lora.get("alpha", rank) / rank) * weight
        d = lora["down"].float().cpu()  # [rank, in_dim]
        u = lora["up"].float().cpu()    # [out_dim, rank]

        # Dimension guard: skip if the LoRA output dim doesn't match the module.
        # This catches LoRAs trained on a different model variant (different hidden
        # dim / mlp ratio) — applying them would crash torch.compile shape tracing.
        mod_out_dim = getattr(getattr(module, "weight", None), "shape", (None,))[0]
        if mod_out_dim is not None and u.shape[0] != mod_out_dim:
            fail_count += 1
            continue

        if _can_merge(module):
            # ---- Fast path: merge delta directly into weights ----------------
            # delta = scale * up @ down  →  shape [out_dim, in_dim]
            # Computed once here on CPU; moved to the module's device/dtype.
            delta_cpu = (scale * (u @ d))  # float32, CPU
            w = module.weight
            w.data.add_(delta_cpu.to(device=w.device, dtype=w.dtype))
            handles.append(_WeightMerge(module, delta_cpu))
            merge_count += 1
        else:
            # ---- Fallback: forward hook with cached GPU tensors ---------------
            # Cache is keyed by (device, dtype) so it handles layer-offload
            # device changes without stale tensor references.
            def _make_hook(down, up, s):
                _cache: dict = {}
                def hook(mod, inp, out):
                    x = inp[0] if isinstance(inp, (tuple, list)) else inp
                    key = (out.device, out.dtype)
                    if key not in _cache:
                        _cache[key] = (
                            down.to(device=out.device, dtype=out.dtype),
                            up.to(device=out.device, dtype=out.dtype),
                        )
                    dv, uv = _cache[key]
                    return out + F.linear(F.linear(x, dv), uv) * s
                return hook

            handles.append(module.register_forward_hook(_make_hook(d, u, scale)))
            hook_count += 1

    # ---- Logging -------------------------------------------------------
    # on_log is already _status (prints + GUI) when called from apply_loras.
    # Fall back to plain print when called standalone.
    _log = on_log if on_log is not None else print
    total_parsed = len(loras)
    ok_count = merge_count + hook_count
    if total_parsed == 0:
        _log("[LoRA] no LoRA entries parsed — check key format")
        for k in list(state_dict.keys())[:8]:
            _log(f"  key: {k}")
    elif ok_count == 0:
        _log(f"[LoRA] 0 layers applied — {fail_count} failed "
             f"({total_parsed} entries parsed)")
        for k in list(state_dict.keys())[:8]:
            _log(f"  key: {k}")
        # Show the translated module paths being looked up (first 4)
        for (tgt, mod_path), _ in list(loras.items())[:4]:
            _log(f"  lookup: {tgt}:{mod_path}")
        # Show model modules relevant to attention / blocks
        if transformer is not None:
            relevant = [
                n for n, _ in transformer.named_modules()
                if any(x in n for x in ("attn", "to_q", "to_k", "block", "ff."))
            ]
            for n in relevant[:8]:
                _log(f"  model: {n}")
    else:
        parts = []
        if merge_count:
            parts.append(f"{merge_count} merged")
        if hook_count:
            parts.append(f"{hook_count} hooked")
        skipped = total_parsed - ok_count
        detail = ", ".join(parts)
        if skipped:
            _log(f"[LoRA] {ok_count}/{total_parsed} layers applied "
                 f"({detail}, {skipped} skipped — dim mismatch or missing module)")
        else:
            _log(f"[LoRA] {ok_count}/{total_parsed} layers applied ({detail})")

    return handles
