"""Shared hidden-state helpers for steering hooks.

Generation-time hooks see slightly different tensor ranks depending on whether
Transformers is in the prompt prefill path or the cached decoding path.  These
helpers keep Phase 3/4 steering code rank- and dtype-safe.
"""
from __future__ import annotations

import torch


def first_hidden(output):
    """Return the hidden-state tensor from a module hook output."""
    return output[0] if isinstance(output, tuple) else output


def replace_first_hidden(output, h: torch.Tensor):
    """Replace the hidden-state tensor while preserving tuple outputs."""
    return (h,) + output[1:] if isinstance(output, tuple) else h


def boundary_state(h: torch.Tensor, boundary_idx: int) -> torch.Tensor | None:
    """Return the boundary hidden state for [B, S, D] or [S, D] tensors."""
    if h.dim() == 3:
        if boundary_idx >= h.shape[1]:
            return None
        return h[:, boundary_idx, :]
    if h.dim() == 2:
        if boundary_idx >= h.shape[0]:
            return None
        return h[boundary_idx, :]
    return None


def write_boundary_state(
    h: torch.Tensor,
    boundary_idx: int,
    h_new: torch.Tensor,
) -> torch.Tensor:
    """Write a boundary state back, preserving original dtype and device."""
    h_out = h.clone()
    h_new = h_new.to(dtype=h_out.dtype, device=h_out.device)
    if h.dim() == 3:
        h_out[:, boundary_idx, :] = h_new
    elif h.dim() == 2:
        h_out[boundary_idx, :] = h_new
    return h_out


def last_token_state(output) -> torch.Tensor | None:
    """Return the last-token hidden state for [B, S, D] or [S, D] outputs."""
    h = first_hidden(output)
    if h.dim() == 3:
        return h[:, -1, :]
    if h.dim() == 2:
        return h[-1:, :]
    return None
