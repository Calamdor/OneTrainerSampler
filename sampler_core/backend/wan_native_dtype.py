"""Monkey-patch diffusers WanTransformerBlock to use native-dtype arithmetic.

The upstream diffusers implementation upcasts modulation, norms, and residual
connections to float32 inside every transformer block.  The original Wan model
(and Kijai's ComfyUI WanVideoWrapper) runs these operations in the model's
native dtype (bf16).  Because the weights were trained/evaluated in bf16,
the float32 intermediates produce a visible warm/red colour shift that
accumulates through 40 blocks.

Call ``patch_wan_transformer(model)`` before sampling and
``unpatch_wan_transformer(model)`` in the finally block.
"""
from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Replacement block forward — native-dtype modulation & residuals
# ---------------------------------------------------------------------------

def _block_forward_native(self, hidden_states, encoder_hidden_states, temb, rotary_emb):
    """WanTransformerBlock.forward with bf16-native arithmetic."""
    if temb.ndim == 4:
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table.unsqueeze(0) + temb
        ).chunk(6, dim=2)
        shift_msa = shift_msa.squeeze(2)
        scale_msa = scale_msa.squeeze(2)
        gate_msa = gate_msa.squeeze(2)
        c_shift_msa = c_shift_msa.squeeze(2)
        c_scale_msa = c_scale_msa.squeeze(2)
        c_gate_msa = c_gate_msa.squeeze(2)
    else:
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table + temb
        ).chunk(6, dim=1)

    # 1. Self-attention
    norm_hidden_states = (
        self.norm1(hidden_states) * (1 + scale_msa) + shift_msa
    ).type_as(hidden_states)
    attn_output = self.attn1(norm_hidden_states, None, None, rotary_emb)
    hidden_states = hidden_states + attn_output * gate_msa

    # 2. Cross-attention
    norm_hidden_states = self.norm2(hidden_states).type_as(hidden_states)
    attn_output = self.attn2(norm_hidden_states, encoder_hidden_states, None, None)
    hidden_states = hidden_states + attn_output

    # 3. Feed-forward
    norm_hidden_states = (
        self.norm3(hidden_states) * (1 + c_scale_msa) + c_shift_msa
    ).type_as(hidden_states)
    ff_output = self.ffn(norm_hidden_states)
    hidden_states = hidden_states + ff_output * c_gate_msa

    return hidden_states


# ---------------------------------------------------------------------------
# Replacement transformer forward — float32 patch embed + native-dtype output
# ---------------------------------------------------------------------------

def _transformer_forward_native(self, hidden_states, timestep, encoder_hidden_states,
                                encoder_hidden_states_image=None, return_dict=True,
                                attention_kwargs=None):
    """WanTransformer3DModel.forward with float32 patch embed and native output norm."""
    from diffusers.models.modeling_outputs import Transformer2DModelOutput

    batch_size, num_channels, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = self.config.patch_size
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p_h
    post_patch_width = width // p_w

    rotary_emb = self.rope(hidden_states)

    # Patch embedding in float32 then cast back — matches original Wan impl
    _hs_dtype = hidden_states.dtype
    hidden_states = self.patch_embedding(hidden_states.float()).to(_hs_dtype)
    hidden_states = hidden_states.flatten(2).transpose(1, 2)

    if timestep.ndim == 2:
        ts_seq_len = timestep.shape[1]
        timestep = timestep.flatten()
    else:
        ts_seq_len = None

    temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = (
        self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image,
            timestep_seq_len=ts_seq_len,
        )
    )
    if ts_seq_len is not None:
        timestep_proj = timestep_proj.unflatten(2, (6, -1))
    else:
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

    if encoder_hidden_states_image is not None:
        encoder_hidden_states = torch.concat(
            [encoder_hidden_states_image, encoder_hidden_states], dim=1)

    if torch.is_grad_enabled() and self.gradient_checkpointing:
        for block in self.blocks:
            hidden_states = self._gradient_checkpointing_func(
                block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)
    else:
        for block in self.blocks:
            hidden_states = block(
                hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

    # Output norm — native dtype (no float32 upcast)
    if temb.ndim == 3:
        shift, scale = (
            self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)
        ).chunk(2, dim=2)
        shift = shift.squeeze(2)
        scale = scale.squeeze(2)
    else:
        shift, scale = (
            self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)
        ).chunk(2, dim=1)

    shift = shift.to(hidden_states.device)
    scale = scale.to(hidden_states.device)

    hidden_states = (
        self.norm_out(hidden_states) * (1 + scale) + shift
    ).type_as(hidden_states)
    hidden_states = self.proj_out(hidden_states)

    hidden_states = hidden_states.reshape(
        batch_size, post_patch_num_frames, post_patch_height, post_patch_width,
        p_t, p_h, p_w, -1,
    )
    hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
    output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_ORIG_ATTR = "_orig_forward_pre_native_patch"


def _unwrap_block(block):
    """Unwrap OffloadCheckpointLayer to get the real WanTransformerBlock."""
    inner = getattr(block, "checkpoint", None)
    if inner is not None and hasattr(inner, "scale_shift_table"):
        return inner
    return block


def patch_wan_transformer(model) -> None:
    """Patch both transformer experts in *model* for native-dtype arithmetic.

    Safe to call multiple times — skips already-patched transformers.
    When offload is enabled, blocks are wrapped in OffloadCheckpointLayer —
    we patch the inner block so the wrapper's ``self.checkpoint(*args)``
    call invokes our native-dtype forward.
    """
    import types

    for attr in ("transformer", "transformer_2"):
        tr = getattr(model, attr, None)
        if tr is None:
            continue
        if hasattr(tr, _ORIG_ATTR):
            continue  # already patched

        # Save originals
        setattr(tr, _ORIG_ATTR, tr.forward)
        tr.forward = types.MethodType(_transformer_forward_native, tr)

        # Patch each block (unwrap offload wrappers if present)
        for entry in tr.blocks:
            block = _unwrap_block(entry)
            if not hasattr(block, _ORIG_ATTR):
                setattr(block, _ORIG_ATTR, block.forward)
                block.forward = types.MethodType(_block_forward_native, block)

    print("[wan_native_dtype] patched transformer blocks for native-dtype arithmetic")


def unpatch_wan_transformer(model) -> None:
    """Restore the original diffusers forward methods."""
    for attr in ("transformer", "transformer_2"):
        tr = getattr(model, attr, None)
        if tr is None:
            continue
        orig = getattr(tr, _ORIG_ATTR, None)
        if orig is not None:
            tr.forward = orig
            delattr(tr, _ORIG_ATTR)

        for entry in tr.blocks:
            block = _unwrap_block(entry)
            orig = getattr(block, _ORIG_ATTR, None)
            if orig is not None:
                block.forward = orig
                delattr(block, _ORIG_ATTR)
