"""Chroma LoRA key translation.

make_chroma_translator() returns a key_translator closure compatible with
apply_lora_hooks() in sampler_core.lora.hooks.

Accepts:
  - OT format          lora_transformer.* / lora_te.*
  - OT us-sep format   lora_transformer_* / lora_te_*
  - External dot       transformer.* / t5.* / text_encoder.*
  - lora_unet_ format  kohya / ai-toolkit native Flux/Chroma

For lora_unet_ format, call expand_lora_unet_fused() on the state dict
BEFORE calling apply_lora_hooks().  Native Flux stores img_attn.qkv /
txt_attn.qkv / single-block linear1 as fused projections; the diffusers
model has separate to_q / to_k / to_v layers.  The expander splits the
fused LoRA up-matrix into equal slices and emits synthetic _0/_1/_2/_3
entries that the translator then maps normally.
"""
import re

from sampler_core.lora.key_util import build_us_map

# External format → diffusers ChromaTransformer2DModel path
# double_blocks → transformer_blocks
_CHROMA_DOUBLE_SUB_MAP = {
    "img_attn.qkv.0":  "attn.to_q",
    "img_attn.qkv.1":  "attn.to_k",
    "img_attn.qkv.2":  "attn.to_v",
    "txt_attn.qkv.0":  "attn.add_q_proj",
    "txt_attn.qkv.1":  "attn.add_k_proj",
    "txt_attn.qkv.2":  "attn.add_v_proj",
    "img_attn.proj":   "attn.to_out.0",
    "img_mlp.0":       "ff.net.0.proj",
    "img_mlp.2":       "ff.net.2",
    "txt_attn.proj":   "attn.to_add_out",
    "txt_mlp.0":       "ff_context.net.0.proj",
    "txt_mlp.2":       "ff_context.net.2",
}

# single_blocks → single_transformer_blocks
_CHROMA_SINGLE_SUB_MAP = {
    "linear1.0": "attn.to_q",
    "linear1.1": "attn.to_k",
    "linear1.2": "attn.to_v",
    "linear1.3": "proj_mlp",
    "linear2":   "proj_out",
}

# distilled_guidance_layer
_CHROMA_DGL_SUB_MAP = {
    "in_layer":  "linear_1",
    "out_layer": "linear_2",
}

# top-level transformer renames
_CHROMA_TOP_MAP = {
    "txt_in":             "context_embedder",
    "img_in.proj":        "x_embedder",
    "final_layer.linear": "proj_out",
}


def _translate_external_chroma_transformer_path(path: str) -> str | None:
    """
    Translate an external LoRA path (relative to the transformer root) to the
    diffusers ChromaTransformer2DModel attribute path.
    Returns None if the path cannot be translated.
    """
    m = re.match(r"^double_blocks\.(\d+)\.(.+)$", path)
    if m:
        mapped = _CHROMA_DOUBLE_SUB_MAP.get(m.group(2))
        return f"transformer_blocks.{m.group(1)}.{mapped}" if mapped else None

    m = re.match(r"^single_blocks\.(\d+)\.(.+)$", path)
    if m:
        mapped = _CHROMA_SINGLE_SUB_MAP.get(m.group(2))
        return f"single_transformer_blocks.{m.group(1)}.{mapped}" if mapped else None

    m = re.match(r"^distilled_guidance_layer\.layers\.(\d+)\.(.+)$", path)
    if m:
        mapped = _CHROMA_DGL_SUB_MAP.get(m.group(2))
        return f"distilled_guidance_layer.layers.{m.group(1)}.{mapped}" if mapped else None

    for src, dst in _CHROMA_TOP_MAP.items():
        if path == src or path.startswith(src + "."):
            return path.replace(src, dst, 1)

    return None


# ── lora_unet_ format (kohya / ai-toolkit for native Flux/Chroma) ─────────────
# Sub-path (after stripping block prefix) in underscore format → external dot path.
# Fused qkv_0/1/2 entries are produced by expand_lora_unet_fused(); they are
# not present in the original LoRA file.

_DOUBLE_BLOCK_US_TO_EXT = {
    "img_attn_proj":  "img_attn.proj",
    "img_attn_qkv_0": "img_attn.qkv.0",   # expanded from fused img_attn_qkv
    "img_attn_qkv_1": "img_attn.qkv.1",
    "img_attn_qkv_2": "img_attn.qkv.2",
    "img_mlp_0":      "img_mlp.0",
    "img_mlp_2":      "img_mlp.2",
    "txt_attn_proj":  "txt_attn.proj",
    "txt_attn_qkv_0": "txt_attn.qkv.0",   # expanded from fused txt_attn_qkv
    "txt_attn_qkv_1": "txt_attn.qkv.1",
    "txt_attn_qkv_2": "txt_attn.qkv.2",
    "txt_mlp_0":      "txt_mlp.0",
    "txt_mlp_2":      "txt_mlp.2",
}

_SINGLE_BLOCK_US_TO_EXT = {
    "linear1_0": "linear1.0",   # expanded from fused linear1
    "linear1_1": "linear1.1",
    "linear1_2": "linear1.2",
    "linear1_3": "linear1.3",
    "linear2":   "linear2",
}

_TOP_LEVEL_US_TO_EXT = {
    "txt_in":             "txt_in",
    "img_in_proj":        "img_in.proj",
    "final_layer_linear": "final_layer.linear",
}

# Fused base-key suffix → number of equal output slices.
# expand_lora_unet_fused() uses these to split up-matrices.
_FUSED_SPLITS = {
    "_img_attn_qkv": 3,   # → Q, K, V
    "_txt_attn_qkv": 3,   # → add_Q, add_K, add_V
    "_linear1":      4,   # → to_q, to_k, to_v, proj_mlp
}

# diffusion_model.* dot-format fused suffixes.
# Values are the number of slices. expand_diffusion_model_fused() handles
# non-uniform splits (linear1 Q/K/V/proj_mlp have different output dims).
_DM_FUSED_DOT = {
    ".img_attn.qkv": 3,   # → attn.to_q, attn.to_k, attn.to_v
    ".txt_attn.qkv": 3,   # → attn.add_q_proj, attn.add_k_proj, attn.add_v_proj
    ".linear1":      4,   # → attn.to_q, attn.to_k, attn.to_v, proj_mlp
}


def expand_lora_unet_fused(state_dict: dict) -> dict:
    """
    Expand fused projections in a lora_unet_* state dict into separate slices.

    Native Flux/Chroma stores img_attn.qkv, txt_attn.qkv, and single-block
    linear1 as single fused linear layers whose output is the concatenation of
    Q/K/V (and proj_mlp for linear1).  Diffusers splits these into separate
    modules (to_q, to_k, to_v, …).

    This function:
      - Finds fused base keys that end with _img_attn_qkv, _txt_attn_qkv, _linear1
      - Splits the lora_up (B) matrix along dim 0 into N equal parts
      - Emits synthetic base_0 / base_1 / … entries (shared down/A matrix)
      - Removes the original fused entry

    Only touches keys with the lora_unet_ prefix; everything else is unchanged.
    Returns a new dict (original is not modified).
    """
    _ALL_LORA_SUFF  = (".lora_A.weight", ".lora_down.weight",
                       ".lora_B.weight", ".lora_up.weight", ".alpha")

    # Collect fused base keys that have at least one LoRA tensor present.
    fused: dict[str, int] = {}
    for key in state_dict:
        if not key.startswith("lora_unet_"):
            continue
        for lsuf in _ALL_LORA_SUFF:
            if key.endswith(lsuf):
                base = key[: -len(lsuf)]
                for fused_suf, n in _FUSED_SPLITS.items():
                    if base.endswith(fused_suf):
                        fused[base] = n
                break

    if not fused:
        return state_dict

    out = dict(state_dict)
    for base, n in fused.items():
        # Accept either A/B or down/up naming.
        _da = out.get(base + ".lora_A.weight")
        down = _da if _da is not None else out.get(base + ".lora_down.weight")
        _ub = out.get(base + ".lora_B.weight")
        up   = _ub if _ub is not None else out.get(base + ".lora_up.weight")
        alpha = out.get(base + ".alpha")

        if down is None or up is None:
            continue
        if up.shape[0] % n != 0:
            continue  # can't split evenly — leave untouched

        use_ab = (base + ".lora_A.weight") in out
        part   = up.shape[0] // n

        for i in range(n):
            nb = f"{base}_{i}"
            if use_ab:
                out[nb + ".lora_A.weight"] = down
                out[nb + ".lora_B.weight"] = up[i * part: (i + 1) * part]
            else:
                out[nb + ".lora_down.weight"] = down
                out[nb + ".lora_up.weight"]   = up[i * part: (i + 1) * part]
            if alpha is not None:
                out[nb + ".alpha"] = alpha

        for suf in _ALL_LORA_SUFF:
            out.pop(base + suf, None)

    return out


def expand_diffusion_model_fused(state_dict: dict) -> dict:
    """
    Expand fused projections in a diffusion_model.* state dict into separate slices.

    Some community Chroma LoRA trainers (e.g. the native ComfyUI trainer) write
    keys under a ``diffusion_model.`` prefix using native Chroma path names:
      diffusion_model.double_blocks.N.img_attn.qkv   — fused Q/K/V (3 equal slices)
      diffusion_model.double_blocks.N.txt_attn.qkv   — fused add_Q/add_K/add_V
      diffusion_model.single_blocks.N.linear1         — fused Q/K/V/proj_mlp

    For double-block QKV the up (B) matrix divides evenly into 3 equal parts.

    For single-block linear1 the split is NON-uniform: Q, K, V each have
    hidden_size output channels and proj_mlp has (total − 3×hidden_size) channels.
    ``hidden_size`` is inferred from the lora_A (down) input dimension.

    Emits synthetic ``base.0 / base.1 / …`` entries (shared A matrix, sliced B).
    Only touches diffusion_model.* keys; everything else is unchanged.
    Returns a new dict (original is not modified).
    """
    _ALL_LORA_SUFF = (".lora_A.weight", ".lora_down.weight",
                      ".lora_B.weight", ".lora_up.weight", ".alpha")

    fused: dict[str, int] = {}
    for key in state_dict:
        if not key.startswith("diffusion_model."):
            continue
        for lsuf in _ALL_LORA_SUFF:
            if key.endswith(lsuf):
                base = key[: -len(lsuf)]
                for dot_suf, n in _DM_FUSED_DOT.items():
                    if base.endswith(dot_suf):
                        fused[base] = n
                break

    if not fused:
        return state_dict

    out = dict(state_dict)
    for base, n in fused.items():
        _da = out.get(base + ".lora_A.weight")
        down = _da if _da is not None else out.get(base + ".lora_down.weight")
        _ub = out.get(base + ".lora_B.weight")
        up   = _ub if _ub is not None else out.get(base + ".lora_up.weight")
        alpha = out.get(base + ".alpha")

        if down is None or up is None:
            continue

        total = up.shape[0]

        # Determine per-slice output sizes.
        # linear1: hidden_size inferred from lora_A input dim (down.shape[1]).
        #   Slices are [hs, hs, hs, total − 3*hs] for Q, K, V, proj_mlp.
        # QKV (double-block img/txt): equal 3-way split.
        if base.endswith(".linear1"):
            hs = down.shape[1]
            mlp_dim = total - 3 * hs
            if mlp_dim <= 0:
                continue  # malformed tensor
            slice_sizes = [hs, hs, hs, mlp_dim]
        else:
            if total % n != 0:
                continue
            part = total // n
            slice_sizes = [part] * n

        use_ab = (base + ".lora_A.weight") in out
        offset = 0
        for i, sz in enumerate(slice_sizes):
            nb = f"{base}.{i}"
            if use_ab:
                out[nb + ".lora_A.weight"] = down
                out[nb + ".lora_B.weight"] = up[offset: offset + sz]
            else:
                out[nb + ".lora_down.weight"] = down
                out[nb + ".lora_up.weight"]   = up[offset: offset + sz]
            if alpha is not None:
                out[nb + ".alpha"] = alpha
            offset += sz

        for suf in _ALL_LORA_SUFF:
            out.pop(base + suf, None)

    return out


def make_chroma_translator(transformer, text_encoder):
    """
    Return a key_translator closure for use with apply_lora_hooks().

    key_translator(stripped_key) -> (target, module_path)
      target is 'transformer' or 'text_encoder'.
    """
    _tr_us = build_us_map(transformer) if transformer is not None else {}
    _te_us = build_us_map(text_encoder) if text_encoder is not None else {}

    def key_translator(stripped: str):
        # OT dot-separated
        if stripped.startswith("lora_transformer."):
            return ("transformer", stripped[len("lora_transformer."):])
        if stripped.startswith("lora_te."):
            return ("text_encoder", stripped[len("lora_te."):])
        # OT underscore-separated
        if stripped.startswith("lora_transformer_"):
            us = stripped[len("lora_transformer_"):]
            return ("transformer", _tr_us.get(us, us))
        if stripped.startswith("lora_te_"):
            us = stripped[len("lora_te_"):]
            return ("text_encoder", _te_us.get(us, us))
        # lora_unet_ (kohya / ai-toolkit native Flux/Chroma format)
        if stripped.startswith("lora_unet_"):
            us = stripped[len("lora_unet_"):]
            m = re.match(r"^double_blocks_(\d+)_(.+)$", us)
            if m:
                ext_sub = _DOUBLE_BLOCK_US_TO_EXT.get(m.group(2))
                if ext_sub is None:
                    return None
                translated = _translate_external_chroma_transformer_path(
                    f"double_blocks.{m.group(1)}.{ext_sub}")
                return ("transformer", translated) if translated else None
            m = re.match(r"^single_blocks_(\d+)_(.+)$", us)
            if m:
                ext_sub = _SINGLE_BLOCK_US_TO_EXT.get(m.group(2))
                if ext_sub is None:
                    return None
                translated = _translate_external_chroma_transformer_path(
                    f"single_blocks.{m.group(1)}.{ext_sub}")
                return ("transformer", translated) if translated else None
            m = re.match(r"^distilled_guidance_layer_layers_(\d+)_(.+)$", us)
            if m:
                sub_map = {"in_layer": "in_layer", "out_layer": "out_layer"}
                ext_sub = sub_map.get(m.group(2))
                if ext_sub is None:
                    return None
                translated = _translate_external_chroma_transformer_path(
                    f"distilled_guidance_layer.layers.{m.group(1)}.{ext_sub}")
                return ("transformer", translated) if translated else None
            # Top-level transformer keys (txt_in, img_in_proj, final_layer_linear)
            ext_sub = _TOP_LEVEL_US_TO_EXT.get(us)
            if ext_sub is not None:
                translated = _translate_external_chroma_transformer_path(ext_sub)
                return ("transformer", translated if translated else ext_sub)
            return None

        # diffusion_model.* — native Chroma format (some community trainers)
        # After expand_diffusion_model_fused(), fused QKV/linear1 keys have already
        # been split into base.0/base.1/… so they arrive here with numeric suffixes
        # that _translate_external_chroma_transformer_path handles correctly.
        if stripped.startswith("diffusion_model."):
            raw = stripped[len("diffusion_model."):]
            translated = _translate_external_chroma_transformer_path(raw)
            return ("transformer", translated if translated is not None else raw)

        # External: transformer.*
        if stripped.startswith("transformer."):
            raw = stripped[len("transformer."):]
            translated = _translate_external_chroma_transformer_path(raw)
            return ("transformer", translated if translated is not None else raw)
        # External: t5.* or text_encoder.*
        if stripped.startswith("t5."):
            return ("text_encoder", stripped[len("t5."):])
        if stripped.startswith("text_encoder."):
            return ("text_encoder", stripped[len("text_encoder."):])
        # Bare path — assume transformer
        return ("transformer", stripped)

    return key_translator
