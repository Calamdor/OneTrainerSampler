"""
Generic LoRA hook injection engine.

apply_lora_hooks() is the single entry point.  Each model provides a
`key_translator` closure that maps a stripped LoRA key to (target, module_path).

For non-quantized linear layers (BF16/FP16/FP32 weight), the LoRA delta is
merged directly into the weight tensor — zero per-step overhead.  Quantized
layers (NF4, W4A16, GGUF, …) use one of two strategies:

  1. GGUF compile-friendly path (compile_friendly=True):
     Inspired by ComfyUI-GGUF (city96): dequantize weight on-the-fly,
     merge LoRA onto the dequantized float, call F.linear.  LoRA factors
     stored as module attributes (not closure variables) so torch.compile
     guards are stable and persistent disk cache works across sessions.
     The dequant function uses only standard torch ops, so dynamo traces
     through it cleanly with fullgraph=True (matching OneTrainer).

  2. Forward-patch path (compile_friendly=False, legacy):
     Replaces module.forward with a wrapper closure.  Works but closures
     have per-session Python identity, breaking persistent compile cache.

Method patching is used instead of register_forward_hook because:
  - torch.compile(fullgraph=True) on a parent block bypasses post-forward
    hooks on child modules (hooks fire outside the traced graph).
  - Method patching is traceable by dynamo.

IMPORTANT — removal order: handles must be removed in reverse application
order when multiple LoRAs target the same layer.  BaseSamplerBackend
.remove_loras() iterates reversed(self.lora_hooks) to guarantee this.

All paths return objects with a .remove() method so BaseSamplerBackend's
remove_loras() works identically for all.
"""
import torch
import torch.nn.functional as F

_GGUF_FP8_MAX = 448.0  # torch.finfo(torch.float8_e4m3fn).max


# ---------------------------------------------------------------------------
# Weight-merge path (float layers — zero per-step cost)
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
# GGUF compile-friendly path (ComfyUI-style dequant → merge → F.linear)
# ---------------------------------------------------------------------------

def _is_gguf_module(module: torch.nn.Module) -> bool:
    """Check if module uses GGUF quantized weights (has quant_type on weight)."""
    w = getattr(module, "weight", None)
    return w is not None and hasattr(w, "quant_type")


def _gguf_dequant_weight(weight):
    """Dequantize a GGUF weight tensor.

    NOT wrapped in torch.compiler.disable — dequantize_gguf_tensor uses
    only standard torch ops (view, reshape, bitwise, arithmetic) that
    dynamo traces through cleanly.  This matches OneTrainer's
    LinearGGUFA8.forward() which also calls dequantize_gguf_tensor
    without a compiler disable, allowing fullgraph=True compilation.
    """
    from diffusers.quantizers.gguf.utils import dequantize_gguf_tensor
    return dequantize_gguf_tensor(weight.detach())


def _gguf_compile_forward_int8(self, x):
    """Compile-friendly forward for GGUF_A8I modules (int8 activation requant).

    Bound to modules as an unbound function via types.MethodType so it
    appears as a regular method — no closure variables for dynamo to guard on.

    Flow:
      1. Dequantize GGUF weight → float (graph break — one per linear)
      2. Merge all LoRA factors onto dequantized weight (compiled, pure torch)
      3. Int8 quantize-matmul-dequantize (compiled, same speed as original A8I)

    The int8 path (quantize merged weight + input to int8, _int_mm, rescale)
    matches LinearGGUFIntA8RequantFunction but uses only standard torch ops —
    no custom autograd Function, so torch.compile traces through cleanly.
    """
    w = _gguf_dequant_weight(self.weight)
    dt = self._gguf_compile_dt
    w = w.to(device=self._gguf_compile_dev, dtype=dt)

    # Compiled: merge pre-concatenated LoRA delta onto dequantized weight.
    # Factors live on CPU; moved to compute device on demand (ComfyUI-style).
    if self._lora_d is not None:
        dev = w.device
        w = w + self._lora_u.to(dev, non_blocking=True) @ self._lora_d.to(dev, non_blocking=True)

    # Compiled: int8 quantized matmul (same as original A8I path)
    # All ops are standard torch — _int_mm is a built-in ATen op.
    bias = self.bias
    if bias is not None:
        bias = bias.to(dtype=dt)

    x_2d = x.reshape(-1, x.shape[-1])

    # Quantize input to int8 (per-row scaling)
    x_absmax = x_2d.abs().amax(dim=-1, keepdim=True).clamp_(min=1e-7)
    x_scale = x_absmax.mul_(1.0 / 127.0)
    x_8 = x_2d.div(x_scale).round_().to(torch.int8)

    # Quantize merged weight to int8 (per-row scaling)
    w_absmax = w.abs().amax(dim=-1, keepdim=True).clamp_(min=1e-7)
    w_scale = w_absmax.mul_(1.0 / 127.0)
    w_8 = w.div(w_scale).round_().to(torch.int8)

    # Int8 matmul → rescale to compute dtype
    y = torch._int_mm(x_8, w_8.t())
    y = y.float().mul_(w_scale.t()).mul_(x_scale).to(dtype=dt)
    if bias is not None:
        y = y.add_(bias)

    return y.reshape(x.shape[:-1] + (y.shape[-1],))


def _gguf_compile_forward_fp8(self, x):
    """Compile-friendly forward for GGUF_A8F modules (fp8 activation requant).

    Bound to modules as an unbound function via types.MethodType so it
    appears as a regular method — no closure variables for dynamo to guard on.

    Flow:
      1. Dequantize GGUF weight → float (graph break — one per linear)
      2. Merge all LoRA factors onto dequantized weight (compiled, pure torch)
      3. FP8 axiswise quantize input + merged weight, torch._scaled_mm, rescale

    Matches LinearGGUFFpA8RequantFunction.forward logic exactly. Falls back to
    F.linear for small batches (<=16 rows) — same as LinearGGUFA8.forward.

    Requires a CUDA GPU with SM >= 89 (Ada Lovelace / RTX 40xx or newer)
    for native fp8 matmul support.
    """
    try:
        w = _gguf_dequant_weight(self.weight)           # graph break
        dt = self._gguf_compile_dt
        w = w.to(device=self._gguf_compile_dev, dtype=dt)

        # Compiled: merge pre-concatenated LoRA delta onto dequantized weight.
        if self._lora_d is not None:
            dev = w.device
            w = w + self._lora_u.to(dev, non_blocking=True) @ self._lora_d.to(dev, non_blocking=True)

        bias = self.bias
        if bias is not None:
            bias = bias.to(dtype=dt)

        x_2d = x.reshape(-1, x.shape[-1])

        # Small-batch fallback: skip fp8 quantization (matches LinearGGUFA8.forward)
        if x_2d.shape[0] <= 16:
            y = torch.nn.functional.linear(x_2d, w, bias)
            return y.reshape(x.shape[:-1] + (y.shape[-1],))

        # FP8 axiswise quantization of input
        x_scale = x_2d.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / _GGUF_FP8_MAX
        x_fp8 = (x_2d / x_scale).clamp(-_GGUF_FP8_MAX, _GGUF_FP8_MAX).to(torch.float8_e4m3fn)

        # FP8 axiswise quantization of merged weight
        w_scale = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / _GGUF_FP8_MAX
        w_fp8 = (w / w_scale).clamp(-_GGUF_FP8_MAX, _GGUF_FP8_MAX).to(torch.float8_e4m3fn)

        # Scaled fp8 matmul → float32, then rescale to compute dtype
        one = torch.ones(1, device=self._gguf_compile_dev, dtype=torch.float32)
        y = torch._scaled_mm(x_fp8, w_fp8.t(), scale_a=one, scale_b=one, out_dtype=torch.float32)
        y = y.mul(w_scale.t()).mul(x_scale).to(dtype=dt)

        if bias is not None:
            y = y.add_(bias)

        return y.reshape(x.shape[:-1] + (y.shape[-1],))
    except Exception as _e:
        d_shape = getattr(self._lora_d, 'shape', None)
        u_shape = getattr(self._lora_u, 'shape', None)
        raise RuntimeError(
            f"_gguf_compile_forward_fp8 failed: weight_byte_shape={self.weight.shape} "
            f"x_in={x.shape} lora_d={d_shape} lora_u={u_shape}: {_e}"
        ) from _e


class _GGUFCompilePatch:
    """Compile-friendly LoRA patch for GGUF modules.

    Replaces forward with dequant → merge → int8 or fp8 matmul depending on
    the module's _dtype attribute.  LoRA factors stored as module attributes
    (_lora_factors) — NOT closure variables — so torch.compile guards are
    based on stable module structure, and persistent disk cache works across
    app restarts.

    One handle is created per module (first LoRA).  Subsequent LoRAs on the
    same module get a lightweight _GGUFFactorRef handle.  remove() on the
    primary handle restores the original forward and cleans up all factors.
    """
    __slots__ = ("_module", "_orig_forward")

    def __init__(self, module: torch.nn.Module, orig_forward):
        self._module = module
        self._orig_forward = orig_forward

    def remove(self) -> None:
        self._module.forward = self._orig_forward
        for attr in ('_lora_factors', '_lora_d', '_lora_u',
                     '_gguf_compile_dt', '_gguf_compile_dev',
                     '_gguf_compile_is_fp8'):
            try:
                delattr(self._module, attr)
            except AttributeError:
                pass


class _GGUFFactorRef:
    """Lightweight handle for additional LoRAs on an already-patched GGUF module.

    The primary _GGUFCompilePatch handle manages forward restoration.
    This ref exists so that BaseSamplerBackend.apply_loras() gets a non-empty
    handle list per LoRA (preventing the '0 handles' warning).
    remove() is a no-op — cleanup is handled by _GGUFCompilePatch.remove().
    """
    __slots__ = ()

    def remove(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Quantized compile-friendly path (non-GGUF quantized modules)
# ---------------------------------------------------------------------------

def _quantized_compile_forward(self, x):
    """Compile-friendly forward for non-GGUF quantized modules.

    Calls the original quantized forward, then adds LoRA delta via
    output-space addition.  Uses pre-merged factors (_lora_d / _lora_u)
    so the cost is ONE matmul pair regardless of how many LoRAs are
    stacked.  Factors live on CPU and are moved to the compute device
    on demand (ComfyUI-style), so only the active layer's LoRA
    consumes VRAM.
    """
    y = self._orig_forward_for_lora(x)
    if self._lora_d is not None:
        dev = x.device
        y = y + F.linear(F.linear(x, self._lora_d.to(dev, non_blocking=True)),
                         self._lora_u.to(dev, non_blocking=True))
    return y


def _rebuild_merged_lora(module: torch.nn.Module) -> None:
    """Merge all LoRA factors into a single (D, U) pair via concatenation.

    Given N LoRAs with factors (d_i, u_i, s_i):
      D_merged = cat([d_1, d_2, …], dim=0)           # [sum(ranks), in_dim]
      U_merged = cat([u_1*s_1, u_2*s_2, …], dim=1)   # [out_dim, sum(ranks)]

    Then U_merged @ D_merged = sum(u_i @ d_i * s_i) in one matmul.

    Merged tensors are stored on CPU in pinned memory (ComfyUI-style).
    The forward functions move them to the compute device on demand
    with non_blocking=True so only the active layer's LoRA factors
    consume VRAM, and the transfer overlaps with GPU computation.
    """
    factors = getattr(module, '_lora_factors', [])
    if not factors:
        module._lora_d = None
        module._lora_u = None
        return
    ds = [dv for dv, _, _ in factors]
    us = [uv * s for _, uv, s in factors]
    d_merged = torch.cat(ds, dim=0).cpu()
    u_merged = torch.cat(us, dim=1).cpu()
    # Pinned memory enables fast DMA transfers (~2x vs unpinned).
    try:
        module._lora_d = d_merged.pin_memory()
        module._lora_u = u_merged.pin_memory()
    except RuntimeError:
        # pin_memory can fail if CUDA isn't available or limit exceeded
        module._lora_d = d_merged
        module._lora_u = u_merged


class _QuantizedCompilePatch:
    """Compile-friendly LoRA patch for non-GGUF quantized modules.

    Same pattern as _GGUFCompilePatch: module attributes + shared forward.
    remove() restores the original forward and cleans up all factors.
    """
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


# ---------------------------------------------------------------------------
# Forward patch path (legacy — for non-compiled quantized modules)
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

def get_module_by_dotpath(root: torch.nn.Module, path: str, debug_log=None) -> torch.nn.Module:
    """
    Walk `root` via dot-separated `path`, transparently unwrapping OT
    OffloadCheckpointLayer wrappers (.checkpoint attribute).

    torch.compile OptimizedModule wrappers do not need special handling here:
    OptimizedModule.__getattr__ transparently delegates to _orig_mod, so
    attribute traversal through a compiled block works naturally.
    
    If debug_log is provided, logs the traversal path for debugging.
    """
    m = root
    traversal_log: list[str] = []
    if debug_log:
        traversal_log.append(f"START: {type(root).__name__}")
    for part in path.split("."):
        m = getattr(m, part)
        if hasattr(m, "checkpoint") and isinstance(m.checkpoint, torch.nn.Module):
            if debug_log:
                traversal_log.append(f"  ↓ UNWRAP checkpoint: {type(m).__name__} → {type(m.checkpoint).__name__}")
            m = m.checkpoint
        else:
            if debug_log:
                traversal_log.append(f"  ↓ {part}: {type(m).__name__}")
    
    if debug_log and len(traversal_log) > 1:
        debug_log(f"[MODULE-LOOKUP] {path}\n  " + "\n  ".join(traversal_log))
    
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
        compile_friendly: bool = False,
) -> list:
    """
    Inject a LoRA into `transformer` and/or `text_encoder`.

    For each target module:
      - If the weight is a plain float tensor (BF16/FP16/FP32): merge the
        LoRA delta directly into the weight (fast path, zero per-step cost).
      - If compile_friendly=True and module is GGUF: use dequant → merge →
        F.linear path (ComfyUI-style, no graph breaks from LoRA).
      - Otherwise: patch module.forward with a closure wrapper.

    key_translator(stripped_key: str) -> (target: str, module_path: str) | None
      target is 'transformer' or 'text_encoder'.
      Return None to skip the key.

    Returns a list of handle objects; call .remove() on each to undo.
    BaseSamplerBackend.remove_loras() does this in reversed order.
    """
    import types

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
    qcomp_count = 0
    gguf_count = 0

    for (target, mod_path), lora in loras.items():
        if "down" not in lora or "up" not in lora:
            continue
        root = transformer if target == "transformer" else text_encoder
        if root is None:
            continue
        try:
            module = get_module_by_dotpath(root, mod_path)
        except AttributeError:
            if on_log:
                on_log(f"[MODULE-LOOKUP-FAIL] {target}:{mod_path} — AttributeError")
            fail_count += 1
            continue

        rank  = lora["down"].shape[0]
        scale = (lora.get("alpha", rank) / rank) * weight
        d = lora["down"].float().cpu()  # [rank, in_dim]
        u = lora["up"].float().cpu()    # [out_dim, rank]

        # Dimension guard: skip if the LoRA output dim doesn't match the module.
        # Prefer out_features (logical dim) over weight.shape[0] (may be GGUF packed).
        mod_out_dim = getattr(module, "out_features", None)
        if mod_out_dim is None:
            mod_out_dim = getattr(getattr(module, "weight", None), "shape", (None,))[0]
        if mod_out_dim is not None and u.shape[0] != mod_out_dim:
            if on_log:
                on_log(f"[LoRA-DIM-FAIL] {target}:{mod_path} — LoRA up dim={u.shape[0]}, "
                       f"module weight dim={mod_out_dim} — SKIPPED")
            fail_count += 1
            continue

        if _can_merge(module):
            # ---- Fast path: merge delta directly into weights ----------------
            w = module.weight
            w.data.add_((scale * (u @ d)).to(device=w.device, dtype=w.dtype))
            handles.append(_WeightMerge(module, d, u, scale))
            merge_count += 1

        elif compile_friendly and _is_gguf_module(module):
            # ---- GGUF compile-friendly path ----------------------------------
            # Dequant → merge LoRA → int8/fp8 matmul.  LoRA factors stored as
            # module attributes on CPU (ComfyUI-style); moved to compute device
            # on demand in the forward so only the active layer's factors use VRAM.
            dev = hint_device or torch.device("cuda")
            dt = getattr(module, "compute_dtype", None) or torch.bfloat16

            if not hasattr(module, '_lora_factors'):
                orig_fwd = module.forward
                module._lora_factors = []
                module._gguf_compile_dt = dt
                module._gguf_compile_dev = dev
                is_fp8 = getattr(module, '_dtype', None) == torch.float8_e4m3fn
                module._gguf_compile_is_fp8 = is_fp8
                forward_fn = _gguf_compile_forward_fp8 if is_fp8 else _gguf_compile_forward_int8
                module.forward = types.MethodType(forward_fn, module)
                handles.append(_GGUFCompilePatch(module, orig_fwd))
            else:
                handles.append(_GGUFFactorRef())

            # Store on CPU; forward moves to GPU on demand.
            dv = d.to(dtype=dt)
            uv = u.to(dtype=dt)
            module._lora_factors.append((dv, uv, scale))
            _rebuild_merged_lora(module)
            gguf_count += 1

            if gguf_count == 1 and on_log:
                on_log(f"[LoRA] GGUF compile-friendly: dtype={dt} (factors on CPU)")

        elif compile_friendly:
            # ---- Quantized compile-friendly path (non-GGUF) ------------------
            # LoRA factors as module attributes + shared forward.  Factors on
            # CPU, moved to compute device on demand (ComfyUI-style).
            dt = getattr(module, "compute_dtype", None) or torch.bfloat16

            if not hasattr(module, '_lora_factors'):
                orig_fwd = module.forward
                module._lora_factors = []
                module._orig_forward_for_lora = orig_fwd
                module.forward = types.MethodType(_quantized_compile_forward, module)
                handles.append(_QuantizedCompilePatch(module, orig_fwd))
            else:
                handles.append(_GGUFFactorRef())

            # Store on CPU; forward moves to GPU on demand.
            dv = d.to(dtype=dt)
            uv = u.to(dtype=dt)
            module._lora_factors.append((dv, uv, scale))
            _rebuild_merged_lora(module)
            qcomp_count += 1

            if qcomp_count == 1 and on_log:
                on_log(f"[LoRA] compile-friendly: dtype={dt} (factors on CPU)")

        else:
            # ---- Fallback: forward method patch (closure-based) ----------------
            # Used when compile_friendly=False (no torch.compile).
            def _make_patch(orig_fwd, down, up, s, mod, hint_dev):
                w   = getattr(mod, "weight", None)
                _PLAIN_FLOAT = (torch.float16, torch.bfloat16, torch.float32)
                if w is not None and w.dtype in _PLAIN_FLOAT:
                    dev = w.device
                elif hint_dev is not None:
                    dev = hint_dev
                else:
                    dev = torch.device("cpu")
                dt  = getattr(mod, "compute_dtype", None)
                if dt is None:
                    dt = (w.dtype if (w is not None and w.dtype in _PLAIN_FLOAT)
                          else torch.bfloat16)
                dv = down.to(device=dev, dtype=dt)
                uv = up.to(device=dev, dtype=dt)
                def patched(x):
                    return orig_fwd(x) + F.linear(F.linear(x, dv), uv) * s
                return patched

            orig = module.forward
            patch = _make_patch(orig, d, u, scale, module, hint_device)
            module.forward = patch
            handles.append(_ForwardPatch(module, orig))
            hook_count += 1
            if hook_count == 1 and on_log:
                w2   = getattr(module, "weight", None)
                _PLAIN = (torch.float16, torch.bfloat16, torch.float32)
                dev2 = (w2.device if (w2 is not None and w2.dtype in _PLAIN)
                        else hint_device or "cpu")
                dt2  = getattr(module, "compute_dtype", None) or (
                    w2.dtype if (w2 is not None and w2.dtype in _PLAIN)
                    else torch.bfloat16)
                on_log(f"[LoRA] patch device={dev2} dtype={dt2}")

    # ---- Logging -------------------------------------------------------
    _log = on_log if on_log is not None else print
    total_parsed = len(loras)
    ok_count = merge_count + hook_count + qcomp_count + gguf_count
    if total_parsed == 0:
        _log("[LoRA] no LoRA entries parsed — check key format")
        for k in list(state_dict.keys())[:8]:
            _log(f"  key: {k}")
    elif ok_count == 0:
        _log(f"[LoRA] 0 layers applied — {fail_count} failed "
             f"({total_parsed} entries parsed)")
        for k in list(state_dict.keys())[:8]:
            _log(f"  key: {k}")
        for (tgt, mod_path), _ in list(loras.items())[:4]:
            _log(f"  lookup: {tgt}:{mod_path}")
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
        if gguf_count:
            parts.append(f"{gguf_count} gguf-compile")
        if qcomp_count:
            parts.append(f"{qcomp_count} compile-friendly")
        if hook_count:
            parts.append(f"{hook_count} forward-patched")
        if skipped:
            parts.append(f"{skipped} skipped (dim mismatch or missing module)")
        if fail_count:
            parts.append(f"{fail_count} path lookup failed")
        _log(f"[LoRA] {ok_count}/{total_parsed} layers applied — " + ", ".join(parts))

    return handles
