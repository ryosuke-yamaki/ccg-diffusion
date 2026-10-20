"""Sample-local latent updates for batched inference-time guidance.

Attend-and-Excite and SynGen both define their latent update for one image at a
time.  A physical image batch must therefore sum the *per-sample* scalar losses
instead of averaging them: the gradient slice for every image then has exactly
the same scale as its batch-size-one counterpart.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch


def sample_local_latent_update(
    latents: torch.Tensor,
    losses: Sequence[torch.Tensor],
    step_size: float,
    *,
    active_mask: Optional[Sequence[bool]] = None,
) -> torch.Tensor:
    """Apply one official-style gradient update without batch averaging.

    ``losses[i]`` must be a scalar whose graph reaches ``latents[i]``.  Summing
    active losses preserves each sample's batch-size-one gradient; taking their
    mean would incorrectly divide every update by the physical batch size.
    The result intentionally remains attached to the graph, matching the
    official A&E and SynGen ``_update_latent`` helpers.
    """
    if latents.ndim < 1:
        raise ValueError("latents must have a batch dimension")
    if latents.dtype != torch.float32:
        raise TypeError("sample-local guidance requires FP32 master latents")
    if not bool(torch.isfinite(latents).all().item()):
        raise FloatingPointError("master latents contain NaN or Inf")
    if len(losses) != latents.shape[0]:
        raise ValueError(
            f"expected {latents.shape[0]} per-sample losses, got {len(losses)}"
        )
    if not isinstance(step_size, (int, float)) or isinstance(step_size, bool):
        raise TypeError("step_size must be a real number")
    if step_size < 0:
        raise ValueError("step_size must be non-negative")

    mask = tuple(True for _ in losses) if active_mask is None else tuple(active_mask)
    if len(mask) != len(losses):
        raise ValueError("active_mask must have one entry per sample")
    if any(not isinstance(active, bool) for active in mask):
        raise TypeError("active_mask entries must be bool")

    active_losses = []
    for index, (loss, active) in enumerate(zip(losses, mask)):
        if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
            raise ValueError(f"losses[{index}] must be a scalar tensor")
        if active:
            if not loss.requires_grad:
                raise ValueError(f"active losses[{index}] is detached from autograd")
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError(f"active losses[{index}] contains NaN or Inf")
            active_losses.append(loss)

    if not active_losses or step_size == 0:
        return latents

    total_loss = torch.stack(active_losses).sum()
    gradient = torch.autograd.grad(
        total_loss.requires_grad_(True),
        [latents],
        retain_graph=True,
    )[0]
    if not bool(torch.isfinite(gradient).all().item()):
        raise FloatingPointError("sample-local guidance gradient contains NaN or Inf")
    candidate = latents - float(step_size) * gradient
    if not bool(torch.isfinite(candidate).all().item()):
        raise FloatingPointError("sample-local guidance candidate contains NaN or Inf")
    return candidate
