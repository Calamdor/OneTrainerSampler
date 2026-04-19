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
  - Probes actual token length and sets max_length = max(512, actual_len)
  - Keeps padding='max_length' so short prompts still pad to 512 (the Wan
    transformer goes out-of-distribution on sub-512 sequences and produces
    garbage output)
  - Disables truncation so prompts longer than 512 survive intact
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
        # Probe actual token length so short prompts still get 512-padded
        # (the Wan transformer was trained with 512-token zero-padded inputs —
        # shorter sequences go out-of-distribution and produce garbage output).
        probe = tok(text, truncation=False, padding=False,
                    add_special_tokens=kwargs.get("add_special_tokens", True))
        ids = probe["input_ids"]
        if ids and isinstance(ids[0], list):
            actual_len = max((len(x) for x in ids), default=0)
        else:
            actual_len = len(ids)
        kwargs["max_length"] = max(512, actual_len)
        kwargs["truncation"] = False
        kwargs["padding"]    = "max_length"
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
