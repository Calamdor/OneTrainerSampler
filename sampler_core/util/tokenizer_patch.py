"""
Tokenizer wrapper that removes the hardcoded 512-token truncation used in
OT's training paths.

T5 uses relative positional attention and handles sequences longer than 512
tokens without modification.  The limit in OT's encode_text methods reflects
training convention, not a model constraint.  The only real limit is VRAM.

Usage:
    from sampler_core.util.tokenizer_patch import patch_tokenizer_no_truncate
    patch_tokenizer_no_truncate(model)   # call once after model load

This replaces model.tokenizer in-place with a wrapper that:
  - Changes padding='max_length'  →  padding='longest'
  - Changes truncation=True       →  truncation=False
  - Drops any explicit max_length kwarg passed to __call__
  - Forwards all attribute access to the real tokenizer transparently

OT's training code is unaffected — the patch is instance-level only.
"""


class _NoTruncateTokenizer:
    """
    Transparent proxy around a HuggingFace tokenizer that disables truncation
    and switches to 'longest' padding so prompts longer than 512 tokens pass
    through to T5 intact.
    """
    def __init__(self, tokenizer):
        # Store under a mangled name to avoid colliding with forwarded attrs
        object.__setattr__(self, "_tok", tokenizer)

    def __call__(self, text, **kwargs):
        tok = object.__getattribute__(self, "_tok")
        # Remove the hardcoded training limit
        kwargs.pop("max_length", None)
        # Don't truncate — T5 handles longer sequences fine
        kwargs["truncation"] = False
        # Pad only to the actual sequence length, not to a fixed 512
        if kwargs.get("padding") == "max_length":
            kwargs["padding"] = "longest"
        return tok(text, **kwargs)

    def __getattr__(self, name):
        tok = object.__getattribute__(self, "_tok")
        return getattr(tok, name)

    def __setattr__(self, name, value):
        tok = object.__getattribute__(self, "_tok")
        setattr(tok, name, value)


def patch_tokenizer_no_truncate(model) -> None:
    """
    Replace model.tokenizer with a _NoTruncateTokenizer wrapper.
    Safe to call multiple times — won't double-wrap.
    """
    tok = getattr(model, "tokenizer", None)
    if tok is None or isinstance(tok, _NoTruncateTokenizer):
        return
    model.tokenizer = _NoTruncateTokenizer(tok)
