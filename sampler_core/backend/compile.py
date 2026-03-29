"""
Shared torch.compile utilities for sampler backends.

Both Chroma and Wan backends use identical compile logic — block
iteration differs but the inner operations are the same.
"""
import torch


def strip_premature_compile(block_lists) -> None:
    """Remove OT's premature fullgraph=True compile from inner blocks.

    OT's create_checkpoint(compile=True) calls orig_module.compile(fullgraph=True)
    which is too early (before LoRA).  Strip it so we can recompile after LoRA.
    """
    for blk_list in block_lists:
        for block in blk_list:
            target = _get_compilable(block)
            if target._compiled_call_impl is not None:
                del target._compiled_call_impl


def _get_compilable(block):
    """Return the module that should be compiled for a given block.

    Offload path: the inner .checkpoint module (not the wrapper).
    No-offload path: the block itself.
    """
    if hasattr(block, 'checkpoint') and isinstance(block.checkpoint, torch.nn.Module):
        return block.checkpoint
    return block


def ensure_blocks_compiled(block_lists) -> None:
    """Compile transformer blocks in-place with fullgraph=True.

    Must be called AFTER LoRA injection so dynamo traces include patched
    forwards.  Uses module.compile() (in-place) to preserve block type
    for _kwargs_to_args signature introspection.
    """
    for blk_list in block_lists:
        for block in blk_list:
            target = _get_compilable(block)
            if target._compiled_call_impl is not None:
                continue
            target.compile(fullgraph=True)
