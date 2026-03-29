"""
Compiled SageAttention wrapper for fullgraph=True torch.compile.

Diffusers calls sageattn() directly — a triton kernel that dynamo can't
trace, causing a graph break inside compiled blocks.  This module wraps
sageattn in a @torch.library.custom_op with a register_fake handler so
dynamo can trace through it without breaking the graph.

Inspired by Kijai/ComfyUI-WanVideoWrapper's attention.py which uses the
same pattern: custom_op + register_fake for compile compatibility.

Usage:
    from sampler_core.util.sage_compile import patch_sage_attention
    patch_sage_attention()   # call once before sampling
    unpatch_sage_attention() # restore original after sampling
"""
import torch

_patched = False
_orig_fn = None


def patch_sage_attention() -> None:
    """Replace diffusers' sage dispatch with a custom_op-wrapped version."""
    global _patched, _orig_fn
    if _patched:
        return

    try:
        from sageattention import sageattn
    except ImportError:
        return  # sage not installed — nothing to patch

    # Register custom op (idempotent — safe to call multiple times)
    if not hasattr(torch.ops, "ots") or not hasattr(torch.ops.ots, "sageattn"):
        @torch.library.custom_op("ots::sageattn", mutates_args=())
        def _sageattn_op(
            q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
            tensor_layout: str = "NHD",
            is_causal: bool = False,
            sm_scale: float = 0.0,
        ) -> torch.Tensor:
            kwargs = dict(tensor_layout=tensor_layout, is_causal=is_causal)
            if sm_scale != 0.0:
                kwargs["sm_scale"] = sm_scale
            if not (q.dtype == k.dtype == v.dtype):
                return sageattn(q, k.to(q.dtype), v.to(q.dtype), **kwargs)
            elif q.dtype == torch.float32:
                return sageattn(
                    q.to(torch.bfloat16), k.to(torch.bfloat16),
                    v.to(torch.bfloat16), **kwargs
                ).to(torch.float32)
            return sageattn(q, k, v, **kwargs)

        @_sageattn_op.register_fake
        def _sageattn_fake(q, k, v, tensor_layout="NHD", is_causal=False, sm_scale=0.0):
            return torch.empty_like(q)

    # Monkey-patch diffusers' sage forward op
    try:
        import diffusers.models.attention_dispatch as _ad
        _orig_fn = getattr(_ad, "_sage_attention_forward_op", None)
        if _orig_fn is not None:
            def _compiled_sage_forward_op(
                query, key, value, attn_mask=None, dropout_p=0.0,
                is_causal=False, scale=None, return_lse=False,
            ):
                out = torch.ops.ots.sageattn(
                    query, key, value,
                    tensor_layout="NHD",
                    is_causal=is_causal,
                    sm_scale=float(scale) if scale is not None else 0.0,
                )
                if return_lse:
                    return out, None
                return out

            _ad._sage_attention_forward_op = _compiled_sage_forward_op
            _patched = True
    except (ImportError, AttributeError):
        pass  # diffusers version doesn't have this dispatch path


def unpatch_sage_attention() -> None:
    """Restore the original diffusers sage dispatch."""
    global _patched, _orig_fn
    if _patched and _orig_fn is not None:
        try:
            import diffusers.models.attention_dispatch as _ad
            _ad._sage_attention_forward_op = _orig_fn
        except (ImportError, AttributeError):
            pass
    _patched = False
    _orig_fn = None
