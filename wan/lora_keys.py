"""Wan2.2 LoRA key translation.

Handles musubi-tuner / OT format LoRA keys for WanTransformer3DModel.
Expert routing (HIGH = transformer, LOW = transformer_2) lives here.
"""

# Musubi-tuner / original Wan2.2 LoRA module names → diffusers WanTransformer3DModel paths.
_WAN_LORA_PATH_MAP = [
    (".self_attn.q",  ".attn1.to_q"),
    (".self_attn.k",  ".attn1.to_k"),
    (".self_attn.v",  ".attn1.to_v"),
    (".self_attn.o",  ".attn1.to_out.0"),
    (".cross_attn.q", ".attn2.to_q"),
    (".cross_attn.k", ".attn2.to_k"),
    (".cross_attn.v", ".attn2.to_v"),
    (".cross_attn.o", ".attn2.to_out.0"),
    (".ffn.0",        ".ffn.net.0.proj"),
    (".ffn.2",        ".ffn.net.2"),
]


def _translate_wan_lora_path(path: str) -> str:
    """Translate musubi-tuner module path to diffusers WanTransformer3DModel path."""
    for src, dst in _WAN_LORA_PATH_MAP:
        if path.endswith(src):
            return path[: -len(src)] + dst
    return path


def _detect_expert_from_filename(path: str) -> str:
    """Heuristic: infer expert target (HIGH/LOW/BOTH) from LoRA filename."""
    name = path.lower()
    if "highnoise" in name or "high" in name:
        return "HIGH"
    if "lownoise" in name or "low" in name:
        return "LOW"
    return "BOTH"


def make_wan_translator(transformer):  # noqa: ARG001 — kept for API symmetry with Chroma
    """Return a key_translator closure for apply_lora_hooks().

    Handles musubi-tuner format:
      diffusion_model.blocks.N.self_attn.q.*

    Also handles OT dot format:
      lora_transformer.blocks.N.attn1.to_q.*
      lora_transformer_2.blocks.N.attn1.to_q.*

    Wan LoRAs only target diffusion blocks — no text_encoder targeting.
    Returns ("transformer", translated_path) or None if the key is unrecognised.
    """
    _PREFIXES = (
        "diffusion_model.",
        "lora_transformer_2.",
        "lora_transformer.",
    )

    def key_translator(stripped_key: str):
        # Strip known LoRA prefixes
        path = stripped_key
        for prefix in _PREFIXES:
            if path.startswith(prefix):
                path = path[len(prefix):]
                break

        # Apply musubi-tuner → diffusers path map
        path = _translate_wan_lora_path(path)

        return ("transformer", path)

    return key_translator
