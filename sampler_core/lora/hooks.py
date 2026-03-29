"""
Generic LoRA hook injection engine.

apply_lora_hooks() is the single entry point.  Dispatches to one of
five injection paths based on module type and compile mode:

  1. Float weight merge (BF16/FP16/FP32) — direct in-place add
  2. Quantized weight merge (W8A8I/W8A8F/FP8) — batched dequant→merge→requant
  3. GGUF compile-friendly — dequant+merge per-forward
  4. Quantized compile-friendly — output-addition with module attributes
  5. Forward-patch closure fallback (compile off)

Implementation details for each path live in their own modules:
  - sampler_core.lora.merge          (paths 1-2)
  - sampler_core.lora.gguf_forward   (path 3)
  - sampler_core.lora.compile_forward (path 4)
  - sampler_core.lora.forward_patch  (path 5)

All paths return objects with a .remove() method.  Removal MUST happen
in reverse application order (BaseSamplerBackend.remove_loras uses
reversed()).
"""
import torch

from sampler_core.lora.merge import (
    _WeightMerge, _QuantizedWeightMerge, can_merge, can_merge_quantized,
)
from sampler_core.lora.gguf_forward import (
    is_gguf_module, GGUFCompilePatch, FactorRef, select_gguf_forward,
)
from sampler_core.lora.compile_forward import (
    quantized_compile_forward, QuantizedCompilePatch,
    rebuild_merged_lora, move_lora_factors_to_device,
)
from sampler_core.lora.forward_patch import (
    make_forward_patch, log_first_patch,
)


# ---------------------------------------------------------------------------
# Module lookup helper
# ---------------------------------------------------------------------------

def get_module_by_dotpath(root: torch.nn.Module, path: str) -> torch.nn.Module:
    """Walk `root` via dot-separated `path`, unwrapping OffloadCheckpointLayer."""
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
        compile_friendly: bool = False,
) -> list:
    """Inject a LoRA into `transformer` and/or `text_encoder`.

    Returns a list of handle objects; call .remove() on each to undo.
    """
    import types

    # ---- Parse state_dict keys -----------------------------------------
    loras: dict[tuple[str, str], dict] = {}
    for key, tensor in state_dict.items():
        for suf, slot in [
            (".lora_A.weight", "down"), (".lora_down.weight", "down"),
            (".lora_B.weight", "up"),   (".lora_up.weight", "up"),
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

    # ---- Dispatch each LoRA entry to the right path --------------------
    handles = []
    counts = {"merge": 0, "qmerge": 0, "gguf": 0, "qcomp": 0, "hook": 0, "fail": 0}
    _qmerge_pending: dict[int, _QuantizedWeightMerge] = {}

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
            counts["fail"] += 1
            continue

        rank  = lora["down"].shape[0]
        scale = (lora.get("alpha", rank) / rank) * weight
        d = lora["down"].float().cpu()
        u = lora["up"].float().cpu()

        # Dimension guard
        mod_out_dim = getattr(module, "out_features", None)
        if mod_out_dim is None:
            mod_out_dim = getattr(getattr(module, "weight", None), "shape", (None,))[0]
        if mod_out_dim is not None and u.shape[0] != mod_out_dim:
            if on_log:
                on_log(f"[LoRA-DIM-FAIL] {target}:{mod_path} — LoRA up dim={u.shape[0]}, "
                       f"module weight dim={mod_out_dim} — SKIPPED")
            counts["fail"] += 1
            continue

        # ---- Path 1: float weight merge --------------------------------
        if can_merge(module):
            w = module.weight
            w.data.add_((scale * (u @ d)).to(device=w.device, dtype=w.dtype))
            handles.append(_WeightMerge(module, d, u, scale))
            counts["merge"] += 1

        # ---- Path 2: quantized weight merge (batched) ------------------
        elif can_merge_quantized(module):
            dev = hint_device or torch.device("cuda")
            _qm_key = id(module)
            if _qm_key not in _qmerge_pending:
                qm = _QuantizedWeightMerge(module, dev)
                _qmerge_pending[_qm_key] = qm
                handles.append(qm)
            _qmerge_pending[_qm_key].add(d, u, scale)
            counts["qmerge"] += 1

        # ---- Path 3: GGUF compile-friendly -----------------------------
        elif compile_friendly and is_gguf_module(module):
            dev = hint_device or torch.device("cuda")
            dt = getattr(module, "compute_dtype", None) or torch.bfloat16
            if not hasattr(module, '_lora_factors'):
                orig_fwd = module.forward
                module._lora_factors = []
                module._gguf_compile_dt = dt
                module._gguf_compile_dev = dev
                module.forward = types.MethodType(select_gguf_forward(module), module)
                handles.append(GGUFCompilePatch(module, orig_fwd))
            else:
                handles.append(FactorRef())
            dv = d.to(dtype=dt)
            uv = u.to(dtype=dt)
            module._lora_factors.append((dv, uv, scale))
            rebuild_merged_lora(module)
            counts["gguf"] += 1
            if counts["gguf"] == 1 and on_log:
                on_log(f"[LoRA] GGUF compile-friendly: dtype={dt} (factors on CPU)")

        # ---- Path 4: quantized compile-friendly (non-GGUF) -------------
        elif compile_friendly:
            dt = getattr(module, "compute_dtype", None) or torch.bfloat16
            if not hasattr(module, '_lora_factors'):
                orig_fwd = module.forward
                module._lora_factors = []
                module._orig_forward_for_lora = orig_fwd
                module.forward = types.MethodType(quantized_compile_forward, module)
                handles.append(QuantizedCompilePatch(module, orig_fwd))
            else:
                handles.append(FactorRef())
            dv = d.to(dtype=dt)
            uv = u.to(dtype=dt)
            module._lora_factors.append((dv, uv, scale))
            rebuild_merged_lora(module)
            counts["qcomp"] += 1
            if counts["qcomp"] == 1 and on_log:
                on_log(f"[LoRA] compile-friendly: dtype={dt} (factors on CPU)")

        # ---- Path 5: fallback closure-based forward patch --------------
        else:
            handles.append(make_forward_patch(
                module.forward, d, u, scale, module, hint_device))
            counts["hook"] += 1
            if counts["hook"] == 1 and on_log:
                log_first_patch(module, hint_device, on_log)

    # ---- Flush batched quantized merges --------------------------------
    if _qmerge_pending:
        if on_log:
            on_log(f"[LoRA] Flushing {len(_qmerge_pending)} quantized weight merges …")
        for qm in _qmerge_pending.values():
            qm.flush()
        _qmerge_pending.clear()

    # ---- Summary log ---------------------------------------------------
    _log = on_log if on_log is not None else print
    total_parsed = len(loras)
    ok = counts["merge"] + counts["qmerge"] + counts["gguf"] + counts["qcomp"] + counts["hook"]
    if total_parsed == 0:
        _log("[LoRA] no LoRA entries parsed — check key format")
        for k in list(state_dict.keys())[:8]:
            _log(f"  key: {k}")
    elif ok == 0:
        _log(f"[LoRA] 0 layers applied — {counts['fail']} failed ({total_parsed} entries parsed)")
        for k in list(state_dict.keys())[:8]:
            _log(f"  key: {k}")
        for (tgt, mp), _ in list(loras.items())[:4]:
            _log(f"  lookup: {tgt}:{mp}")
        if transformer is not None:
            relevant = [n for n, _ in transformer.named_modules()
                        if any(x in n for x in ("attn", "to_q", "to_k", "block", "ff."))]
            for n in relevant[:8]:
                _log(f"  model: {n}")
    else:
        skipped = total_parsed - ok
        parts = []
        if counts["merge"]:
            parts.append(f"{counts['merge']} weight-merged")
        if counts["qmerge"]:
            parts.append(f"{counts['qmerge']} quantized-merged")
        if counts["gguf"]:
            parts.append(f"{counts['gguf']} gguf-compile")
        if counts["qcomp"]:
            parts.append(f"{counts['qcomp']} compile-friendly")
        if counts["hook"]:
            parts.append(f"{counts['hook']} forward-patched")
        if skipped:
            parts.append(f"{skipped} skipped")
        if counts["fail"]:
            parts.append(f"{counts['fail']} lookup failed")
        _log(f"[LoRA] {ok}/{total_parsed} layers applied — " + ", ".join(parts))

    return handles
