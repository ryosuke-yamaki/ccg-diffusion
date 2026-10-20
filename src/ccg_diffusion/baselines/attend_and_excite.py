"""Attend-and-Excite (Chefer et al., 2023): noun targets, smoothed-max objective and refinement.

The attention objective, Gaussian smoothing, step-size schedule and
threshold-driven refinement follow the official implementation at commit
``163efdfd341bf3590df3c0c2b582935fbc8e8343``. Only the subject selection is
automated: every literal spaCy ``token.pos_ == "NOUN"`` occurrence (the rule of
SynGen's ``dvmp.py``) is aligned to the SD1.4 CLIP token axis. For a noun that
spans several CLIP subtokens, the subtoken with the highest 10-seed mean of the
step-0 smoothed maximum is selected (``occurrence_id``, ``select_mean_winner``;
the GPU probe lives in ``baselines.sd14_generation``).

This module does not own a diffusion pipeline: the SD1.4 baseline runner feeds
per-image attention maps to these functions and applies
``sample_local_latent_update``.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Iterable, Mapping, Protocol, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from ccg_diffusion.provenance import canonical_json


SOURCE_REPOSITORY = "yuval-alaluf/Attend-and-Excite"
SOURCE_COMMIT = "163efdfd341bf3590df3c0c2b582935fbc8e8343"
AUTOMATION_SOURCE_COMMIT = "50b8eab175dc9b31ce9c7c4b2eba28c817fc440e"
ATTENTION_RESOLUTION = 16
INTERVENTION_STEPS = 25
MAX_REFINEMENT_STEPS = 20
SCALE_FACTOR = 20.0
SCALE_RANGE = (1.0, 0.5)
THRESHOLDS: Mapping[int, float] = {0: 0.05, 10: 0.5, 20: 0.8}
CANDIDATE_SELECTION_PROTOCOL = "final_lexical_subtoken_candidate_v1"
ENSEMBLE_SELECTION_PROTOCOL = (
    "first_timestep_seed42_51_mean_smoothed_max_v1"
)
SELECTION_SEEDS = tuple(range(42, 52))


class AlignmentError(ValueError):
    """A spaCy noun occurrence cannot be mapped to the CLIP input exactly."""


class SubtokenSelectionError(ValueError):
    """A subtoken-selection candidate, probe or selected artifact is invalid."""


class _SpacyToken(Protocol):
    text: str
    pos_: str
    idx: int


class _Tokenizer(Protocol):
    model_max_length: int

    def __call__(self, text: str, **kwargs: Any) -> Any:
        ...


@dataclass(frozen=True)
class NounTarget:
    """One literal spaCy ``NOUN`` occurrence and its CLIP alignment."""

    text: str
    char_start: int
    char_end: int
    clip_indices: tuple[int, ...]
    selected_clip_index: int

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("noun target text must be non-empty")
        if self.char_start < 0 or self.char_end <= self.char_start:
            raise ValueError("noun target character span is invalid")
        if not self.clip_indices:
            raise ValueError("noun target must contain at least one CLIP index")
        if any(isinstance(index, bool) or index <= 0 for index in self.clip_indices):
            raise ValueError("noun target CLIP indices must be positive integers")
        if tuple(sorted(set(self.clip_indices))) != self.clip_indices:
            raise ValueError("noun target CLIP indices must be unique and increasing")
        if self.selected_clip_index not in self.clip_indices:
            raise ValueError("selected A&E subtoken must belong to the noun occurrence")


@dataclass(frozen=True)
class AttendAndExcitePlan:
    """Validated prompt-local input for an Attend-and-Excite generation."""

    prompt: str
    targets: tuple[NounTarget, ...]
    status: str
    source_commit: str = SOURCE_COMMIT
    automation_source_commit: str = AUTOMATION_SOURCE_COMMIT
    selection_protocol: str = CANDIDATE_SELECTION_PROTOCOL

    @property
    def selected_clip_indices(self) -> tuple[int, ...]:
        return tuple(target.selected_clip_index for target in self.targets)


def _input_ids(encoded: Any) -> list[int]:
    if isinstance(encoded, Mapping):
        ids = encoded.get("input_ids")
    else:
        ids = getattr(encoded, "input_ids", None)
    if ids is None:
        raise TypeError("tokenizer output does not contain input_ids")
    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().tolist()
    if ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise ValueError("expected tokenizer output for exactly one prompt")
        ids = ids[0]
    if not isinstance(ids, (list, tuple)):
        raise TypeError("tokenizer input_ids must be a sequence")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in ids):
        raise TypeError("tokenizer input_ids must contain integers")
    return list(ids)


def _tokenize_without_specials(tokenizer: _Tokenizer, text: str) -> list[int]:
    return _input_ids(
        tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
        )
    )


def _align_character_span(
    prompt: str,
    *,
    char_start: int,
    char_end: int,
    tokenizer: _Tokenizer,
) -> tuple[int, ...]:
    """Map one exact character span to BOS-inclusive CLIP token indices."""
    if not 0 <= char_start < char_end <= len(prompt):
        raise AlignmentError("noun character span is outside the prompt")

    full_ids = _tokenize_without_specials(tokenizer, prompt)
    prefix_ids = _tokenize_without_specials(tokenizer, prompt[:char_start])
    through_ids = _tokenize_without_specials(tokenizer, prompt[:char_end])
    prefix_count = len(prefix_ids)
    through_count = len(through_ids)
    if through_count <= prefix_count:
        raise AlignmentError("noun occurrence produced no lexical CLIP token")
    if full_ids[prefix_count:through_count] != through_ids[prefix_count:through_count]:
        raise AlignmentError(
            "prefix tokenization does not match the full-prompt CLIP tokenization"
        )

    # CLIP inserts one BOS token before the lexical sequence.
    indices = tuple(range(prefix_count + 1, through_count + 1))
    max_length = getattr(tokenizer, "model_max_length", 77)
    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 3:
        raise ValueError("tokenizer.model_max_length must be an integer >= 3")
    if indices[-1] >= max_length - 1:
        raise AlignmentError("noun occurrence lies outside the CLIP lexical window")
    return indices


def extract_noun_targets(
    prompt: str,
    *,
    doc: Iterable[_SpacyToken],
    tokenizer: _Tokenizer,
) -> AttendAndExcitePlan:
    """Build literal-``NOUN`` A&E targets for one prompt.

    ``doc`` must be the spaCy 3.6.1 ``en_core_web_trf`` parse of ``prompt`` in
    the experiment runner.  ``PROPN`` is deliberately excluded, matching the
    literal subject export in SynGen's official ``dvmp.py``.
    """
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")

    targets = []
    previous_start = -1
    for token in doc:
        if token.pos_ != "NOUN":
            continue
        start = int(token.idx)
        end = start + len(token.text)
        if start < previous_start:
            raise ValueError("spaCy noun occurrences must be in prompt order")
        if prompt[start:end] != token.text:
            raise AlignmentError(
                f"spaCy span {start}:{end} does not equal noun text {token.text!r}"
            )
        indices = _align_character_span(
            prompt,
            char_start=start,
            char_end=end,
            tokenizer=tokenizer,
        )
        targets.append(
            NounTarget(
                text=token.text,
                char_start=start,
                char_end=end,
                clip_indices=indices,
                selected_clip_index=indices[-1],
            )
        )
        previous_start = start

    plan = AttendAndExcitePlan(
        prompt=prompt,
        targets=tuple(targets),
        status="ok" if targets else "no_noun_target",
    )
    validate_plan(plan, max_token_count=tokenizer.model_max_length)
    return plan


def validate_plan(
    plan: AttendAndExcitePlan,
    *,
    max_token_count: int,
) -> None:
    """Fail closed on malformed or cross-method A&E preprocessing state."""
    if plan.source_commit != SOURCE_COMMIT:
        raise ValueError("unexpected Attend-and-Excite source commit")
    if plan.automation_source_commit != AUTOMATION_SOURCE_COMMIT:
        raise ValueError("unexpected A&E noun-automation source commit")
    if plan.selection_protocol not in {
        CANDIDATE_SELECTION_PROTOCOL,
        ENSEMBLE_SELECTION_PROTOCOL,
    }:
        raise ValueError("unexpected A&E subtoken-selection protocol")
    if plan.status not in {"ok", "no_noun_target"}:
        raise ValueError(f"invalid A&E sidecar status: {plan.status!r}")
    if (plan.status == "ok") != bool(plan.targets):
        raise ValueError("A&E sidecar status and targets disagree")
    if isinstance(max_token_count, bool) or not isinstance(max_token_count, int):
        raise TypeError("max_token_count must be an integer")
    for target in plan.targets:
        if target.char_end > len(plan.prompt):
            raise ValueError("A&E noun span exceeds the prompt")
        if plan.prompt[target.char_start:target.char_end] != target.text:
            raise ValueError("A&E noun span no longer matches the prompt")
        if any(index >= max_token_count - 1 for index in target.clip_indices):
            raise ValueError("A&E target index reaches EOS/padding")
        if (
            plan.selection_protocol == CANDIDATE_SELECTION_PROTOCOL
            and target.selected_clip_index != target.clip_indices[-1]
        ):
            raise ValueError("candidate A&E sidecar must select the final lexical subtoken")


def _official_gaussian_kernel(
    *,
    kernel_size: int,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(kernel_size, bool) or not isinstance(kernel_size, int):
        raise TypeError("kernel_size must be an integer")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    # Preserve the official GaussianSmoothing exponent exactly.  It uses
    # ((x - mean) / (2 * sigma)) ** 2 rather than the more usual
    # (x - mean) ** 2 / (2 * sigma ** 2).
    axis = torch.arange(kernel_size, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    mean = (kernel_size - 1) / 2
    normalizer = 1 / (sigma * math.sqrt(2 * math.pi))
    kernel_y = normalizer * torch.exp(-((yy - mean) / (2 * sigma)) ** 2)
    kernel_x = normalizer * torch.exp(-((xx - mean) / (2 * sigma)) ** 2)
    kernel = kernel_y * kernel_x
    kernel = kernel / kernel.sum()
    return kernel.to(dtype=dtype).reshape(1, 1, kernel_size, kernel_size)


def max_attention(
    attention_maps: torch.Tensor,
    indices_to_alter: Sequence[int],
    *,
    smooth_attentions: bool = True,
    sigma: float = 0.5,
    kernel_size: int = 3,
) -> tuple[torch.Tensor, ...]:
    """Return official A&E max activations for one sample.

    ``attention_maps`` is ``[height, width, CLIP tokens]`` and indices are
    BOS-inclusive raw CLIP positions.  SD1.4 uses the official ``1:-1`` token
    window, including the padded EOS columns between the first EOS and the
    final slot.
    """
    if attention_maps.ndim != 3:
        raise ValueError("attention_maps must have shape [height, width, tokens]")
    if tuple(attention_maps.shape[:2]) != (
        ATTENTION_RESOLUTION,
        ATTENTION_RESOLUTION,
    ):
        raise ValueError("A&E attention maps must be exactly 16x16")
    if attention_maps.shape[-1] < 3:
        raise ValueError("attention_maps must include BOS, lexical tokens, and EOS")
    if not bool(torch.isfinite(attention_maps.float()).all().item()):
        raise FloatingPointError("A&E attention maps contain NaN or Inf")
    if not indices_to_alter:
        raise ValueError("A&E loss requires at least one target index")
    if any(
        isinstance(index, bool)
        or not isinstance(index, int)
        or not 1 <= index < attention_maps.shape[-1] - 1
        for index in indices_to_alter
    ):
        raise ValueError("A&E target index is outside the SD1.4 lexical window")

    # Model cross-attention arrives in BF16 under the shared SD1.4 profile;
    # the official objective is reduced in FP32 for stable softmax/max/grad.
    attention_for_text = attention_maps.to(dtype=torch.float32)[:, :, 1:-1] * 100
    attention_for_text = torch.softmax(attention_for_text, dim=-1)
    maxima = []
    for raw_index in indices_to_alter:
        image = attention_for_text[:, :, raw_index - 1]
        if smooth_attentions:
            kernel = _official_gaussian_kernel(
                kernel_size=kernel_size,
                sigma=sigma,
                device=image.device,
                dtype=image.dtype,
            )
            pad = kernel_size // 2
            padded = F.pad(image[None, None], (pad, pad, pad, pad), mode="reflect")
            image = F.conv2d(padded, kernel).squeeze(0).squeeze(0)
        maxima.append(image.max())
    return tuple(maxima)


def sample_loss(
    attention_maps: torch.Tensor,
    indices_to_alter: Sequence[int],
    *,
    smooth_attentions: bool = True,
    sigma: float = 0.5,
    kernel_size: int = 3,
) -> torch.Tensor:
    """Compute ``max_i(max(0, 1 - max_attention_i))`` for one sample."""
    maxima = max_attention(
        attention_maps,
        indices_to_alter,
        smooth_attentions=smooth_attentions,
        sigma=sigma,
        kernel_size=kernel_size,
    )
    # Use scalar comparisons instead of ``torch.stack(...).max()``.  Python's
    # ``max`` in the official code keeps the first value on a tie, whereas a
    # tensor reduction may distribute gradients among tied maxima.
    losses = []
    for maximum in maxima:
        candidate = 1.0 - maximum
        losses.append(
            candidate
            if bool(candidate.detach().item() > 0)
            else torch.zeros((), dtype=maximum.dtype, device=maximum.device)
        )
    worst = losses[0]
    for candidate in losses[1:]:
        if bool((candidate > worst).detach().item()):
            worst = candidate
    return worst


def batch_losses(
    attention_maps: torch.Tensor,
    target_indices: Sequence[Sequence[int]],
    *,
    smooth_attentions: bool = True,
    sigma: float = 0.5,
    kernel_size: int = 3,
) -> tuple[torch.Tensor, ...]:
    """Compute independent A&E losses for a physical image batch."""
    if attention_maps.ndim != 4:
        raise ValueError("attention_maps must have shape [batch, height, width, tokens]")
    if len(target_indices) != attention_maps.shape[0]:
        raise ValueError("target_indices must have one entry per sample")
    return tuple(
        sample_loss(
            sample_maps,
            sample_indices,
            smooth_attentions=smooth_attentions,
            sigma=sigma,
            kernel_size=kernel_size,
        )
        for sample_maps, sample_indices in zip(attention_maps, target_indices)
    )


def step_size(step_index: int, *, num_inference_steps: int = 50) -> float:
    """Return official ``20 * sqrt(linspace(1.0, 0.5)[step])``."""
    if isinstance(step_index, bool) or not isinstance(step_index, int):
        raise TypeError("step_index must be an integer")
    if isinstance(num_inference_steps, bool) or not isinstance(num_inference_steps, int):
        raise TypeError("num_inference_steps must be an integer")
    if num_inference_steps < 1 or not 0 <= step_index < num_inference_steps:
        raise ValueError("step_index must fit num_inference_steps")
    scale_range = np.linspace(
        SCALE_RANGE[0],
        SCALE_RANGE[1],
        num_inference_steps,
    )
    return float(SCALE_FACTOR * np.sqrt(scale_range[step_index]))


def refinement_mask(
    losses: Sequence[torch.Tensor],
    *,
    threshold: float,
    iteration_counts: Sequence[int],
    max_refinement_steps: int = MAX_REFINEMENT_STEPS,
) -> tuple[bool, ...]:
    """Select samples that need another official threshold refinement."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between zero and one")
    if len(losses) != len(iteration_counts):
        raise ValueError("iteration_counts must have one entry per loss")
    if max_refinement_steps < 1:
        raise ValueError("max_refinement_steps must be positive")

    target_loss = max(0.0, 1.0 - threshold)
    mask = []
    for index, (loss, count) in enumerate(zip(losses, iteration_counts)):
        if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
            raise ValueError(f"losses[{index}] must be a scalar tensor")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("refinement iteration counts must be non-negative integers")
        mask.append(bool(loss.detach().item() > target_loss and count < max_refinement_steps))
    return tuple(mask)


def occurrence_id(
    prompt: str,
    *,
    occurrence_index: int,
    text: str,
    character_span: Sequence[int],
    candidate_token_indices: Sequence[int],
) -> str:
    """Return the stable identity shared by the selection evidence and all ten seed rows."""

    identity = {
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "occurrence_index": occurrence_index,
        "text": text,
        "character_span": list(character_span),
        "candidate_token_indices": list(candidate_token_indices),
    }
    digest = hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()
    return f"ae_noun_{digest}"


def no_target_id(prompt: str) -> str:
    """Return a stable identity for one stock-fallback prompt population."""

    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return f"ae_no_target_{digest}"


def select_mean_winner(
    candidate_token_indices: Sequence[int],
    candidate_means: Sequence[float],
) -> int:
    """Select the maximum mean, resolving exact ties by the smallest token index."""

    if not candidate_token_indices or len(candidate_token_indices) != len(candidate_means):
        raise SubtokenSelectionError("candidate indices/means must align")
    if tuple(sorted(set(candidate_token_indices))) != tuple(candidate_token_indices):
        raise SubtokenSelectionError("candidate indices must be unique and increasing")
    if any(not isinstance(value, (int, float)) for value in candidate_means):
        raise SubtokenSelectionError("candidate means must be numeric")
    position = min(
        range(len(candidate_token_indices)),
        key=lambda index: (-float(candidate_means[index]), candidate_token_indices[index]),
    )
    return int(candidate_token_indices[position])
