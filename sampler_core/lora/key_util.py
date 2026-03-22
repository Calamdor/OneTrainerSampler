"""Generic LoRA key utilities shared across all sampler models."""
import torch


def build_us_map(root: torch.nn.Module) -> dict[str, str]:
    """
    Build a reverse lookup: underscore-separated path → dot-separated path.

    Some LoRA exporters replace dots with underscores in key names
    (e.g. lora_transformer_single_transformer_blocks_0_attn_to_k).
    This map lets us recover the real module path.

    Handles torch.compile and layer-offload layouts:

    1. Whole-model compile: root itself is OptimizedModule.
       named_modules() yields "_orig_mod.transformer_blocks.0.attn.to_q".
       Register bare path (strip leading "_orig_mod.") under bare underscore key.

    2. Per-block compile with _orig_mod as registered submodule:
       named_modules() yields "transformer_blocks.0._orig_mod.attn.to_q".
       Register the path with "_orig_mod" stripped out under bare underscore key.
       OptimizedModule.__getattr__ transparently delegates to _orig_mod, so
       "transformer_blocks.0.attn.to_q" traverses correctly at runtime.

    3. Per-block compile with _orig_mod stored as a plain __dict__ attribute
       (torch.compile uses self.__dict__["_orig_mod"] = mod to bypass
       nn.Module.__setattr__, so _orig_mod is NOT in _modules and
       named_modules() does NOT recurse into it).
       For each OptimizedModule found, manually iterate its _orig_mod's
       named_modules() and register the combined paths.  get_module_by_dotpath
       reaches the inner module via OptimizedModule.__getattr__ delegation, so
       the real accessible path is just "parent.sub" without "_orig_mod".

    4. Layer offload — OT OffloadCheckpointLayer wraps each block via
       self.checkpoint = orig_block.  named_modules() yields paths like
       "transformer_blocks.0.checkpoint.attn.to_k".  Register the stripped
       path "transformer_blocks.0.attn.to_k" as well so LoRA key lookups
       succeed.  get_module_by_dotpath() auto-unwraps .checkpoint during
       traversal, so the bare path resolves correctly at apply time.
    """
    result: dict[str, str] = {}
    _PREFIX  = "_orig_mod."
    _MID     = "._orig_mod."
    _opt_cls = getattr(getattr(torch, "_dynamo", None), "OptimizedModule", None)

    _CHKPT_MID = ".checkpoint."

    for name, mod in root.named_modules():
        if not name:
            continue
        result[name.replace(".", "_")] = name
        # Case 1: whole-model compile — path starts with "_orig_mod."
        if name.startswith(_PREFIX):
            bare = name[len(_PREFIX):]
            result.setdefault(bare.replace(".", "_"), bare)
        # Case 2: per-block compile — "_orig_mod" embedded mid-path
        if _MID in name:
            bare = name.replace(_MID, ".")
            result.setdefault(bare.replace(".", "_"), bare)
        # Case 3: per-block compile — _orig_mod not in _modules, so
        # named_modules() stops at this OptimizedModule and doesn't descend.
        # Manually index the inner module's sub-paths.
        if _opt_cls is not None and isinstance(mod, _opt_cls) and hasattr(mod, "_orig_mod"):
            # Compute the real base path (strip any _orig_mod. artifacts).
            if _MID in name:
                base = name.replace(_MID, ".")
            elif name.startswith(_PREFIX):
                base = name[len(_PREFIX):]
            else:
                base = name
            for sub_name, _ in mod._orig_mod.named_modules():
                if not sub_name:
                    continue
                real = f"{base}.{sub_name}"
                result.setdefault(real.replace(".", "_"), real)
        # Case 4: layer offload — OT OffloadCheckpointLayer wraps each block as
        # self.checkpoint = orig_block.  named_modules() yields paths like
        # "transformer_blocks.0.checkpoint.attn.to_k".  LoRA keys are trained
        # against the unwrapped block, so we also register the path without
        # ".checkpoint." so lookups succeed.  get_module_by_dotpath() already
        # auto-unwraps .checkpoint at each traversal step, so the bare path
        # "transformer_blocks.0.attn.to_k" works correctly at apply time.
        if _CHKPT_MID in name:
            bare = name.replace(_CHKPT_MID, ".")
            result.setdefault(bare.replace(".", "_"), bare)
    return result
