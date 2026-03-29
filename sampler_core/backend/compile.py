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
        for blk in blk_list:
            if hasattr(blk, 'checkpoint') and \
                    blk.checkpoint._compiled_call_impl is not None:
                del blk.checkpoint._compiled_call_impl


def ensure_blocks_compiled(block_lists) -> None:
    """Compile transformer blocks in-place with fullgraph=True.

    Must be called AFTER LoRA injection so dynamo traces include patched
    forwards.  Uses module.compile() (in-place) to preserve block type
    for _kwargs_to_args signature introspection.
    """
    for blk_list in block_lists:
        for i in range(len(blk_list)):
            block = blk_list[i]
            # Offload path: compile the INNER block, not the wrapper.
            if hasattr(block, 'checkpoint') and \
                    isinstance(block.checkpoint, torch.nn.Module):
                if block.checkpoint._compiled_call_impl is not None:
                    continue
                block.checkpoint.compile(fullgraph=True)
                continue
            # No-offload path: compile the block directly.
            if block._compiled_call_impl is not None:
                continue
            block.compile(fullgraph=True)
