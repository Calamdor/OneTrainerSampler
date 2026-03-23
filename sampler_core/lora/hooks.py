"""
Generic LoRA hook injection engine.

apply_lora_hooks() is the single entry point.  Each model provides a
`key_translator` closure that maps a stripped LoRA key to (target, module_path).

For non-quantized linear layers (BF16/FP16/FP32 weight), the LoRA delta is
merged directly into the weight tensor — zero per-step overhead.  Quantized
layers (NF4, W4A16, GGUF, …) use forward method patching: the module's
.forward is replaced with a wrapper that calls the original forward and adds
the LoRA delta.

Method patching is used instead of register_forward_hook because:
  - torch.compile(fullgraph=True) on a parent block bypasses post-forward
    hooks on child modules (hooks fire outside the traced graph).
  - Method patching is traceable: dynamo inlines `patched` and compiles the
    LoRA delta (F.linear × 2 + scale) as part of the block graph, fusing it
    with surrounding ops.  Only the base-layer kernel (GGUF/NF4) is an
    opaque graph break via torch.compiler.disable(orig_fwd).
  - Guard stability: `_disabled_orig` has a stable Python identity for the
    lifetime of one LoRA apply; dynamo never traverses into orig_fwd's own
    closure chain.  One retrace on first LoRA apply; zero thereafter.

IMPORTANT — removal order: _ForwardPatch.remove() must be called in reverse
application order when multiple LoRAs target the same layer.  BaseSamplerBackend
.remove_loras() iterates reversed(self.lora_hooks) to guarantee this.

Both paths return objects with a .remove() method so BaseSamplerBackend's
remove_loras() works identically for both.
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

    Stores the low-rank factors (d, u, scale) rather than the materialized
    delta to avoid holding a full [out_dim, in_dim] float32 matrix per layer.
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
# Forward patch path
# ---------------------------------------------------------------------------

class _ForwardPatch:
    """
    Undoable forward-method patch for quantized linear layers.

    Replaces module.forward with a wrapper that calls the original forward
    and adds the LoRA delta.  Restores the original method on .remove().

    IMPORTANT: when multiple LoRAs target the same layer, patches are stacked
    (each new patch wraps the previous).  Removal MUST happen in reverse
    application order — BaseSamplerBackend.remove_loras() uses reversed() to
    guarantee this.
    """
    __slots__ = ("_module", "_orig_forward")

    def __init__(self, module: torch.nn.Module, orig_forward):
        self._module       = module
        self._orig_forward = orig_forward

    def remove(self) -> None:
        self._module.forward = self._orig_forward


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
        hint_device: torch.device | None = None,
) -> list:
    """
    Inject a LoRA into `transformer` and/or `text_encoder`.

    For each target module:
      - If the weight is a plain float tensor (BF16/FP16/FP32): merge the
        LoRA delta directly into the weight (fast path, zero per-step cost).
      - Otherwise: patch module.forward with a wrapper that calls the original
        forward and adds the LoRA delta (compile-safe; one retrace on apply/remove).

    key_translator(stripped_key: str) -> (target: str, module_path: str) | None
      target is 'transformer' or 'text_encoder'.
      Return None to skip the key.

    Returns a list of _WeightMerge / _ForwardPatch objects; call .remove() on
    each to undo.  BaseSamplerBackend.remove_loras() does this automatically
    in reversed order (required for correct _ForwardPatch unwinding).
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
            # Compute delta on the fly and apply; store only the small factors.
            w = module.weight
            w.data.add_((scale * (u @ d)).to(device=w.device, dtype=w.dtype))
            handles.append(_WeightMerge(module, d, u, scale))
            merge_count += 1
        else:
            # ---- Fallback: forward method patch (compile-safe) ----------------
            # Replaces module.forward with a wrapper that calls the original
            # forward and adds the LoRA delta from the same input tensor x.
            # dv/uv are pre-moved to the module's device+dtype at injection
            # time — the patched forward is unconditional and contains no
            # mutable closed-over state, so torch.compile can trace it cleanly.
            #
            # Why not register_forward_hook:
            #   torch.compile(fullgraph=True) on the parent transformer block
            #   inlines child module calls and silently bypasses post-forward
            #   hooks.  Method patching is visible to the compiler and allows
            #   the delta computation to be fused into the compiled graph after
            #   one retrace.  dynamic=True blocks also benefit: graph breaks
            #   disappear, replaced by a single guard-based retrace on apply/remove.
            def _make_patch(orig_fwd, down, up, s, mod, hint_dev):
                # Resolve device and dtype at injection time so the patched
                # forward contains no Python-level conditionals or mutable
                # closed-over state.  Any Python dict/conditional inside a
                # compiled function causes torch.compile to re-guard on every
                # call, triggering a recompile loop when the dict mutates on
                # the first call (cache miss).
                #
                # Device: use the weight's device if it is a float (i.e. a
                #   real compute device).  GGUF weights live on CPU even when
                #   inference runs on CUDA — in that case fall back to
                #   hint_dev (self.train_device passed from the backend).
                # Dtype:  LinearGGUFA8 stores compute_dtype (set by
                #   _ot_quantize_layers).  Other quantized layers fall back
                #   to bfloat16 (safe for all Chroma/Wan dtypes).
                w   = getattr(mod, "weight", None)
                if w is not None and w.dtype.is_floating_point:
                    dev = w.device          # float weight → real compute device
                elif hint_dev is not None:
                    dev = hint_dev          # GGUF/quantized → use backend hint
                else:
                    dev = torch.device("cpu")
                dt  = getattr(mod, "compute_dtype", None)
                if dt is None:
                    dt = (w.dtype if (w is not None and w.dtype.is_floating_point)
                          else torch.bfloat16)
                dv = down.to(device=dev, dtype=dt)
                uv = up.to(device=dev, dtype=dt)
                # Disable compilation on patched so dynamo never traverses
                # orig_fwd's closure chain to build identity guards.  Without
                # this, dynamo guards on inner cell contents of orig_fwd (which
                # is itself a closure for GGUF/quantised layers); those cells
                # change identity on every LoRA re-apply, accumulating to the
                # 256-recompile limit and then falling back to full eager.
                #
                # Note: disabling only orig_fwd (not patched) does NOT help —
                # dynamo propagates the disable status upward through any closure
                # that captures a disabled callable, ending up treating patched
                # as disabled anyway.  Disabling patched directly is simpler and
                # produces the same graph-break structure.
                #
                # Guard stability: the compiled block guards on patched's object
                # identity (stable for the lifetime of this apply); loras_current()
                # prevents redundant remove+reapply so the guard only fails once
                # per intentional LoRA change.
                def patched(x):
                    return orig_fwd(x) + F.linear(F.linear(x, dv), uv) * s
                return torch.compiler.disable(patched)

            orig = module.forward
            patch = _make_patch(orig, d, u, scale, module, hint_device)
            module.forward = patch
            handles.append(_ForwardPatch(module, orig))
            hook_count += 1
            if hook_count == 1 and on_log:
                # Log the resolved device+dtype once (all layers use the same).
                w2   = getattr(module, "weight", None)
                dev2 = (w2.device if (w2 is not None and w2.dtype.is_floating_point)
                        else hint_device or "cpu")
                dt2  = getattr(module, "compute_dtype", None) or (
                    w2.dtype if (w2 is not None and w2.dtype.is_floating_point)
                    else torch.bfloat16)
                on_log(f"[LoRA] patch device={dev2} dtype={dt2}")

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
        skipped = total_parsed - ok_count
        parts = []
        if merge_count:
            parts.append(f"{merge_count} weight-merged")
        if hook_count:
            parts.append(f"{hook_count} forward-patched")
        if skipped:
            parts.append(f"{skipped} skipped (dim mismatch or missing module)")
        if fail_count:
            parts.append(f"{fail_count} path lookup failed")
        _log(f"[LoRA] {ok_count}/{total_parsed} layers applied — " + ", ".join(parts))

    return handles
