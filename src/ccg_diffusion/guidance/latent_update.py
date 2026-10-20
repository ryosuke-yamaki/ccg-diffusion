"""Predicated Diffusion latent update with a per-image relative-L2 cap (paper Eq. 1, Table 8).

One batched backward pass gives the gradient of the summed per-image losses
with respect to the batch of latents; each image is then updated independently
with ``x <- x - eta * grad`` where the step ``eta * grad`` is scaled down so
that ``||step|| <= delta * ||x||`` (delta = 0.02 by default, 0.06 on boosted
images). A candidate that is non-finite or exceeds ``max_latent_abs`` is
retried with halved steps (backtracking). ``GenerationDiagnostics`` is the
per-image ledger of these updates; ``as_dict()`` is the recorded
``generation_diagnostics`` object. Used by SD1.4 and Sana.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch


@dataclass
class GenerationDiagnostics:
    """Auditable per-image counters of the guided denoising loop (field names are recorded)."""

    guidance_updates_attempted: int = 0
    guidance_updates_applied: int = 0
    outer_updates_attempted: int = 0
    refinement_updates_attempted: int = 0
    zero_loss_skips: int = 0
    finite_guard_nonfinite_loss: int = 0
    finite_guard_nonfinite_gradient: int = 0
    finite_guard_nonfinite_candidate: int = 0
    finite_guard_nonfinite_noise_pred: int = 0
    finite_guard_nonfinite_scheduler_output: int = 0
    latent_bound_violations: int = 0
    backtracking_attempts: int = 0
    backtracking_updates: int = 0
    latent_update_clips: int = 0
    max_raw_gradient_l2: float = 0.0
    max_proposed_update_ratio: float = 0.0
    max_applied_update_ratio: float = 0.0
    min_update_clip_factor: float = 1.0
    max_latent_abs: float = 0.0
    max_pre_update_latent_abs: float = 0.0
    max_post_scheduler_latent_abs: float = 0.0
    pd_loss_evaluations: int = 0
    first_pd_loss: float | None = None
    last_pd_loss: float | None = None
    min_pd_loss: float | None = None
    max_pd_loss: float | None = None
    effective_batch_size: int = 1
    valid_batch_size: int = 1
    batch_padding_count: int = 0
    generation_kernel: str | None = None
    strict_finite_applied: bool = False
    guided_scheduler_input_scaling: bool = False
    batch_wall_time_ms: float | None = None
    amortized_wall_time_ms: float | None = None
    peak_cuda_memory_allocated_bytes: int | None = None
    peak_cuda_memory_reserved_bytes: int | None = None
    phase_timings_ms: dict[str, float] = field(default_factory=dict)
    initial_latent_sha256: str | None = None
    final_latent_sha256: str | None = None
    generation_batch_id: str | None = None
    attention_block_scope: str | None = None
    captured_attention_processors: list[str] = field(default_factory=list)
    captured_attention_processor_count: int = 0
    pd_loss_guard_counts: dict[str, int] = field(
        default_factory=lambda: {
            "constant_attention_maps": 0,
            "exist_log_term_clamps": 0,
        }
    )
    # Optional ledgers of the Spatial Relation Loss, the route and the Count
    # Loss; ``None`` ledgers are omitted from ``as_dict``.
    relation_guidance: dict[str, Any] | None = None
    semantic_guidance: dict[str, Any] | None = None
    count_guidance: dict[str, Any] | None = None
    _pd_loss_sum: float = field(default=0.0, repr=False)
    _finite_pd_loss_evaluations: int = field(default=0, repr=False)

    def as_dict(self) -> dict[str, Any]:
        payload = {
            key: value
            for key, value in self.__dict__.items()
            if not key.startswith("_")
        }
        payload["mean_pd_loss"] = (
            self._pd_loss_sum / self._finite_pd_loss_evaluations
            if self._finite_pd_loss_evaluations
            else None
        )
        if payload.get("relation_guidance") is None:
            payload.pop("relation_guidance", None)
        if payload.get("semantic_guidance") is None:
            payload.pop("semantic_guidance", None)
        if payload.get("count_guidance") is None:
            payload.pop("count_guidance", None)
        return payload


def record_pd_loss(
    diagnostics: GenerationDiagnostics | None,
    loss: torch.Tensor,
) -> None:
    """Add one scalar PD loss to the per-image audit summary."""

    if diagnostics is None:
        return
    diagnostics.pd_loss_evaluations += 1
    value = float(loss.detach().float().item())
    if not math.isfinite(value):
        return
    diagnostics._finite_pd_loss_evaluations += 1
    diagnostics._pd_loss_sum += value
    if diagnostics.first_pd_loss is None:
        diagnostics.first_pd_loss = value
        diagnostics.min_pd_loss = value
        diagnostics.max_pd_loss = value
    diagnostics.last_pd_loss = value
    diagnostics.min_pd_loss = min(float(diagnostics.min_pd_loss), value)
    diagnostics.max_pd_loss = max(float(diagnostics.max_pd_loss), value)


def finite_and_max_abs_by_track(
    tensor: torch.Tensor,
) -> tuple[list[bool], list[float]]:
    """Return finite flags and absolute maxima with the first axis as tracks."""

    flattened = tensor.detach().float().reshape(tensor.shape[0], -1)
    finite = torch.isfinite(flattened).all(dim=1).cpu().tolist()
    maxima = flattened.abs().amax(dim=1).cpu().tolist()
    return [bool(value) for value in finite], [float(value) for value in maxima]


def _record_batched_pre_update_latent_abs(
    diagnostics: Sequence[GenerationDiagnostics],
    latents: torch.Tensor,
) -> None:
    """Record each image's finite latent magnitude before a batched PD update."""

    finite, maxima = finite_and_max_abs_by_track(latents)
    for record, is_finite, maximum in zip(diagnostics, finite, maxima):
        if is_finite:
            record.max_pre_update_latent_abs = max(
                record.max_pre_update_latent_abs,
                maximum,
            )


def apply_latent_gradient(
    latents: torch.Tensor,
    grad_cond: torch.Tensor,
    step_size: float,
    *,
    diagnostics: GenerationDiagnostics | None,
    strict_finite: bool,
    max_update_ratio: float | None,
    max_latent_abs: float | None,
    allow_backtracking: bool,
) -> torch.Tensor:
    """Apply one already-computed gradient under the stable update contract."""

    if not torch.isfinite(grad_cond).all():
        if diagnostics is not None:
            diagnostics.finite_guard_nonfinite_gradient += 1
        if strict_finite:
            raise FloatingPointError("non-finite Predicated Diffusion gradient")
        return latents.detach()

    base = latents.detach()
    base_max_abs = float(base.float().abs().max().item())
    proposed_update = (step_size * grad_cond).detach()
    raw_gradient_l2 = float(torch.linalg.vector_norm(grad_cond.float()).item())
    latent_l2 = float(torch.linalg.vector_norm(base.float()).item())
    proposed_l2 = float(torch.linalg.vector_norm(proposed_update.float()).item())
    denominator = max(latent_l2, 1e-12)
    proposed_ratio = proposed_l2 / denominator
    clip_factor = 1.0
    if max_update_ratio is not None and proposed_ratio > max_update_ratio:
        clip_factor = max_update_ratio / proposed_ratio
        proposed_update = proposed_update * clip_factor
        if diagnostics is not None:
            diagnostics.latent_update_clips += 1

    if diagnostics is not None:
        diagnostics.max_raw_gradient_l2 = max(
            diagnostics.max_raw_gradient_l2, raw_gradient_l2
        )
        diagnostics.max_proposed_update_ratio = max(
            diagnostics.max_proposed_update_ratio, proposed_ratio
        )
        diagnostics.min_update_clip_factor = min(
            diagnostics.min_update_clip_factor, clip_factor
        )

    for attempt, multiplier in enumerate((1.0, 0.5, 0.25, 0.125, 0.0625)):
        updated = (base - multiplier * proposed_update).detach()
        candidate_is_finite = bool(torch.isfinite(updated).all().item())
        candidate_max_abs = float(updated.float().abs().max().item())
        within_abs_bound = (
            max_latent_abs is None or candidate_max_abs <= max_latent_abs
        )
        if candidate_is_finite and within_abs_bound:
            if diagnostics is not None:
                diagnostics.guidance_updates_applied += 1
                diagnostics.backtracking_attempts += attempt
                if attempt:
                    diagnostics.backtracking_updates += 1
                applied_l2 = float(
                    torch.linalg.vector_norm((updated - base).float()).item()
                )
                diagnostics.max_applied_update_ratio = max(
                    diagnostics.max_applied_update_ratio,
                    applied_l2 / denominator,
                )
                diagnostics.max_latent_abs = max(
                    diagnostics.max_latent_abs, candidate_max_abs
                )
            if strict_finite and attempt and not allow_backtracking:
                raise FloatingPointError(
                    "Predicated Diffusion latent update required backtracking"
                )
            return updated
        if diagnostics is not None:
            if not candidate_is_finite:
                diagnostics.finite_guard_nonfinite_candidate += 1
            elif not within_abs_bound:
                diagnostics.latent_bound_violations += 1
    if strict_finite:
        raise FloatingPointError(
            "all Predicated Diffusion latent update candidates violated "
            "finite/bound constraints: "
            f"base_max_abs={base_max_abs:.6g}, max_latent_abs={max_latent_abs}"
        )
    return base


def update_latents_batched(
    latents: torch.Tensor,
    losses: Sequence[torch.Tensor],
    step_size: float,
    *,
    diagnostics: Sequence[GenerationDiagnostics],
    strict_finite: bool,
    max_update_ratio: float | None,
    max_latent_abs: float | None,
    step_size_multipliers: Sequence[float] | None = None,
    per_job_max_update_ratios: Sequence[float] | None = None,
) -> torch.Tensor:
    """Apply independent per-image updates from one batched backward pass.

    ``step_size_multipliers`` scale the step of each image (4.0 on boosted
    images) and ``per_job_max_update_ratios`` give each image's cap delta.
    Images whose loss is exactly zero keep their latents.
    """

    if latents.shape[0] != len(losses) or len(losses) != len(diagnostics):
        raise ValueError("latents, losses, and diagnostics must have equal batch size")

    for values, name in (
        (step_size_multipliers, "step_size_multipliers"),
        (per_job_max_update_ratios, "per_job_max_update_ratios"),
    ):
        if values is not None and (
            len(values) != len(losses)
            or any(isinstance(value, bool) or not math.isfinite(value) or value <= 0.0 for value in values)
        ):
            raise ValueError(f"{name} must contain one finite positive value per image")

    _record_batched_pre_update_latent_abs(diagnostics, latents)
    active_losses: list[torch.Tensor] = []
    active_indices: list[int] = []
    for index, (loss, record) in enumerate(zip(losses, diagnostics)):
        record.guidance_updates_attempted += 1
        if not loss.requires_grad:
            record.finite_guard_nonfinite_loss += 1
            if strict_finite:
                raise RuntimeError(
                    f"Predicated Diffusion loss does not require gradients at batch index {index}"
                )
            continue
        if not torch.isfinite(loss.detach()).all():
            record.finite_guard_nonfinite_loss += 1
            if strict_finite:
                raise FloatingPointError(
                    f"non-finite Predicated Diffusion loss at batch index {index}"
                )
            continue
        if loss.detach().item() == 0.0:
            record.zero_loss_skips += 1
            continue
        active_losses.append(loss)
        active_indices.append(index)

    if not active_losses:
        return latents.detach()

    gradient = torch.autograd.grad(torch.stack(active_losses).sum(), [latents])[0]
    updated_tracks: list[torch.Tensor] = []
    active_set = set(active_indices)
    for index, record in enumerate(diagnostics):
        if index not in active_set:
            updated_tracks.append(latents[index].detach())
            continue
        track_step_size = step_size if step_size_multipliers is None else step_size * step_size_multipliers[index]
        track_cap = max_update_ratio if per_job_max_update_ratios is None else per_job_max_update_ratios[index]
        if record.semantic_guidance is not None:
            record.semantic_guidance["update_step_size_trajectory"].append(float(track_step_size))
            record.semantic_guidance["update_cap_trajectory"].append(track_cap)
        updated_tracks.append(
            apply_latent_gradient(
                latents[index],
                gradient[index],
                track_step_size,
                diagnostics=record,
                strict_finite=strict_finite,
                max_update_ratio=track_cap,
                max_latent_abs=max_latent_abs,
                allow_backtracking=True,
            )
        )
    return torch.stack(updated_tracks)
