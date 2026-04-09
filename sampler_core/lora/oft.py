"""OFT (Orthogonal Fine-Tuning) inference support.

Computes block-diagonal orthogonal rotation matrices from saved
OFT parameters and applies them via weight merge or forward patch.

OFT rotates the input to a linear layer: y = Linear(R @ x).
For float weights this is equivalent to W' = W @ R^T, which can be
pre-merged at zero per-forward cost.  For quantised/GGUF weights a
forward patch rotates the input on each call instead.

Math:  Cayley–Neumann (5 terms) on skew-symmetric block matrices,
matching OneTrainer's training implementation.
"""
import math
import types

import torch


# ---------------------------------------------------------------------------
# Core: compute rotation matrices from saved parameters
# ---------------------------------------------------------------------------

def compute_oft_rotation(oft_R_weight: torch.Tensor,
                         scale: float = 1.0) -> torch.Tensor:
    """Build orthogonal rotation blocks from OFT skew-symmetric params.

    Args:
        oft_R_weight: (r, n_elements) – upper-triangle elements per block.
        scale: strength multiplier (0 → identity, 1 → full effect).

    Returns:
        (r, block_size, block_size) orthogonal rotation matrices.
    """
    r, n_elements = oft_R_weight.shape
    block_size = int((1 + math.sqrt(1 + 8 * n_elements)) / 2)

    params = oft_R_weight.float() * scale

    # Skew-symmetric matrices from upper-triangle elements
    rows, cols = torch.triu_indices(block_size, block_size, 1)
    Q = torch.zeros(r, block_size, block_size, dtype=torch.float32)
    Q[:, rows, cols] = params
    Q = Q - Q.transpose(-2, -1)

    # Cayley–Neumann 5-term: R = I + 2Q + 2Q² + 2Q³ + Q⁴
    R = torch.eye(block_size, dtype=torch.float32).unsqueeze(0).expand(r, -1, -1).clone()
    R.add_(Q, alpha=2.0)
    Q2 = torch.bmm(Q, Q)
    R.add_(Q2, alpha=2.0)
    Q3 = torch.bmm(Q2, Q)
    R.add_(Q3, alpha=2.0)
    Q4 = torch.bmm(Q3, Q)
    R.add_(Q4)

    return R


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rotate_weight(module: torch.nn.Module, R: torch.Tensor,
                   inverse: bool = False) -> None:
    """Rotate a Linear module's weight in-place using block-diagonal R."""
    w = module.weight
    Rd = R.to(device=w.device, dtype=w.dtype)
    r, bs, _ = Rd.shape
    wr = w.data.reshape(w.shape[0], r, bs)
    if inverse:
        # Undo: W_orig = W_rotated @ R  (since R is orthogonal, R^{-1} = R^T)
        w.data.copy_(torch.einsum("oij,ijk->oik", wr, Rd).reshape(w.shape))
    else:
        # Apply: W_new = W @ R^T  ≡  einsum("oik,ijk->oij", ...)
        w.data.copy_(torch.einsum("oik,ijk->oij", wr, Rd).reshape(w.shape))


# ---------------------------------------------------------------------------
# Path 1: float weight merge  (zero per-forward cost)
# ---------------------------------------------------------------------------

class _OFTWeightMerge:
    """Handle for undoing an OFT rotation on a float-weight module."""
    __slots__ = ("module", "R")

    def __init__(self, module: torch.nn.Module, R: torch.Tensor):
        self.module = module
        self.R = R                  # (r, bs, bs) CPU float32

    def remove(self):
        _rotate_weight(self.module, self.R, inverse=True)


# ---------------------------------------------------------------------------
# Path 2: compile-friendly forward patch  (module attributes)
# ---------------------------------------------------------------------------

def _oft_compile_forward(self, x, *args, **kwargs):
    """Patched forward that rotates input using module-stored R."""
    Rv = self._oft_R
    if Rv.device != x.device or Rv.dtype != x.dtype:
        Rv = Rv.to(device=x.device, dtype=x.dtype)
        self._oft_R = Rv
    r  = self._oft_r
    bs = self._oft_bs
    bdims = x.shape[:-1]
    xr = x.reshape(*bdims, r, bs)
    x_rot = torch.einsum("...rk,rkc->...rc", xr, Rv)
    return self._orig_forward_for_oft(x_rot.reshape(x.shape), *args, **kwargs)


class _OFTCompilePatch:
    """Handle for undoing compile-friendly OFT forward patch."""
    __slots__ = ("module", "orig_forward")

    def __init__(self, module: torch.nn.Module, orig_forward):
        self.module = module
        self.orig_forward = orig_forward

    def remove(self):
        self.module.forward = self.orig_forward
        for attr in ("_oft_R", "_oft_r", "_oft_bs", "_orig_forward_for_oft"):
            if hasattr(self.module, attr):
                delattr(self.module, attr)


# ---------------------------------------------------------------------------
# Path 3: closure forward patch  (compile off)
# ---------------------------------------------------------------------------

class _OFTForwardPatch:
    """Handle for undoing closure-based OFT forward patch."""
    __slots__ = ("module", "orig_forward")

    def __init__(self, module: torch.nn.Module, orig_forward):
        self.module = module
        self.orig_forward = orig_forward

    def remove(self):
        self.module.forward = self.orig_forward


def make_oft_forward_patch(module, R, hint_device):
    """Replace module.forward with a closure that rotates input via R."""
    orig_forward = module.forward
    r, bs, _ = R.shape
    R_ref = [R]

    def patched_forward(x, *args, **kwargs):
        Rv = R_ref[0]
        if Rv.device != x.device or Rv.dtype != x.dtype:
            Rv = R_ref[0].to(device=x.device, dtype=x.dtype)
            R_ref[0] = Rv
        bdims = x.shape[:-1]
        xr = x.reshape(*bdims, r, bs)
        x_rot = torch.einsum("...rk,rkc->...rc", xr, Rv)
        return orig_forward(x_rot.reshape(x.shape), *args, **kwargs)

    module.forward = patched_forward
    return _OFTForwardPatch(module, orig_forward)
