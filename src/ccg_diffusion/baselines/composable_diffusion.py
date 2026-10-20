"""Composable Diffusion (Liu et al., 2022): component prompts and score composition.

The prompt is split on the word ``and`` (case-insensitive, word boundary) into
component prompts; the denoiser predictions are combined with the official
formula::

    eps_u + sum_i w_i * (eps_i - eps_u)

with weight ``w_i = 7.5`` for every component. A prompt without ``and`` has one
component and reduces to ordinary classifier-free guidance. Components are
padded along a component axis so that one physical batch of four images can
have different component counts.

Reference source:
``energy-based-model/Compositional-Visual-Generation-with-Composable-Diffusion-Models-PyTorch``
at commit ``77af82bbc73bbc987ca238bb9a3b0902ea0e12d0``.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

import torch


SOURCE_REPOSITORY = "energy-based-model/Compositional-Visual-Generation-with-Composable-Diffusion-Models-PyTorch"
SOURCE_COMMIT = "77af82bbc73bbc987ca238bb9a3b0902ea0e12d0"
COMPONENT_WEIGHT = 7.5

_AND_PATTERN = re.compile(r"\band\b", flags=re.IGNORECASE)


@dataclass(frozen=True)
class ComposablePlan:
    """Deterministic component decomposition for one input prompt."""

    prompt: str
    components: tuple[str, ...]
    weights: tuple[float, ...]
    source_commit: str = SOURCE_COMMIT

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if not self.components:
            raise ValueError("at least one component is required")
        if len(self.components) != len(self.weights):
            raise ValueError("components and weights must have equal lengths")
        if any(not isinstance(component, str) or not component for component in self.components):
            raise ValueError("components must be non-empty strings")
        if any(not isinstance(weight, (int, float)) for weight in self.weights):
            raise TypeError("weights must be numeric")
        if any(not torch.isfinite(torch.tensor(float(weight))) for weight in self.weights):
            raise ValueError("weights must be finite")
        if self.source_commit != SOURCE_COMMIT:
            raise ValueError("unexpected Composable Diffusion source commit")


@dataclass(frozen=True)
class ComponentBatch:
    """Padded component texts, mask, and weights for an image batch.

    ``texts_by_slot`` is component-major: each outer item contains one text
    for every image in the batch.  Padded entries use the empty string so a
    runner can encode every slot as one ordinary image batch; ``component_mask``
    makes those predictions inert during composition.
    """

    texts_by_slot: tuple[tuple[str, ...], ...]
    component_mask: torch.Tensor
    weights: torch.Tensor

    def __post_init__(self) -> None:
        if self.component_mask.ndim != 2:
            raise ValueError("component_mask must have shape [batch, components]")
        if self.component_mask.dtype != torch.bool:
            raise TypeError("component_mask must be boolean")
        if self.weights.shape != self.component_mask.shape:
            raise ValueError("weights must match component_mask shape")
        if not self.weights.is_floating_point():
            raise TypeError("weights must be floating point")
        if not bool(torch.isfinite(self.weights).all().item()):
            raise ValueError("weights must be finite")
        batch_size, component_count = self.component_mask.shape
        if batch_size == 0 or component_count == 0:
            raise ValueError("a non-empty image and component batch is required")
        if len(self.texts_by_slot) != component_count:
            raise ValueError("texts_by_slot must match the padded component count")
        if any(len(slot) != batch_size for slot in self.texts_by_slot):
            raise ValueError("every component slot must contain one text per image")
        if not bool(self.component_mask.any(dim=1).all().item()):
            raise ValueError("every image must have at least one active component")
        for slot_index, slot in enumerate(self.texts_by_slot):
            for batch_index, text in enumerate(slot):
                is_active = bool(self.component_mask[batch_index, slot_index].item())
                if is_active and not text:
                    raise ValueError("active component texts must be non-empty")
                if not is_active and text:
                    raise ValueError("padded component texts must be empty strings")

    @property
    def batch_size(self) -> int:
        return int(self.component_mask.shape[0])

    @property
    def component_count(self) -> int:
        return int(self.component_mask.shape[1])


def split_prompt(
    prompt: str,
    *,
    component_weight: float = COMPONENT_WEIGHT,
) -> ComposablePlan:
    """Split ``prompt`` by the frozen literal ``and`` automation rule.

    Only leading/trailing whitespace on each resulting component is removed.
    Case, punctuation, articles, and word order remain unchanged.  A prompt
    without ``and`` is represented by one component and therefore reduces to
    ordinary classifier-free guidance under the official composition formula.
    """

    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    if not isinstance(component_weight, (int, float)):
        raise TypeError("component_weight must be numeric")
    weight = float(component_weight)
    if not torch.isfinite(torch.tensor(weight)):
        raise ValueError("component_weight must be finite")

    components = tuple(part.strip() for part in _AND_PATTERN.split(prompt))
    if not components or any(not component for component in components):
        raise ValueError("literal 'and' splitting produced an empty component")
    return ComposablePlan(
        prompt=prompt,
        components=components,
        weights=(weight,) * len(components),
    )


def pack_plans(
    plans: Sequence[ComposablePlan],
    *,
    device: torch.device | str | None = None,
) -> ComponentBatch:
    """Pad variable component counts without changing the image batch axis."""

    if not plans:
        raise ValueError("plans must not be empty")
    component_count = max(len(plan.components) for plan in plans)
    batch_size = len(plans)
    mask = torch.zeros((batch_size, component_count), dtype=torch.bool, device=device)
    weights = torch.zeros((batch_size, component_count), dtype=torch.float32, device=device)
    slots: list[list[str]] = [[""] * batch_size for _ in range(component_count)]

    for batch_index, plan in enumerate(plans):
        if plan.source_commit != SOURCE_COMMIT:
            raise ValueError("all plans must use the pinned source commit")
        for slot_index, (component, weight) in enumerate(zip(plan.components, plan.weights)):
            slots[slot_index][batch_index] = component
            mask[batch_index, slot_index] = True
            weights[batch_index, slot_index] = float(weight)

    return ComponentBatch(
        texts_by_slot=tuple(tuple(slot) for slot in slots),
        component_mask=mask,
        weights=weights,
    )


def compose_noise_predictions(
    unconditional: torch.Tensor,
    conditional: torch.Tensor,
    component_mask: torch.Tensor,
    weights: torch.Tensor,
    *,
    stable_fp32: bool = True,
) -> torch.Tensor:
    """Apply the official Composable Diffusion score formula per sample.

    Parameters
    ----------
    unconditional:
        Unconditional noise prediction with shape ``[batch, ...]``.
    conditional:
        Component predictions with shape ``[batch, components, ...]``.
    component_mask / weights:
        Padded component metadata with shape ``[batch, components]``.
    stable_fp32:
        The experiment profile keeps the subtraction and ordered component
        reduction in FP32.  ``False`` is available for official-environment
        batch-1 parity checks and preserves the prediction dtype.
    """

    if unconditional.ndim < 2:
        raise ValueError("unconditional must have shape [batch, ...]")
    if conditional.ndim != unconditional.ndim + 1:
        raise ValueError("conditional must have shape [batch, components, ...]")
    if conditional.shape[0] != unconditional.shape[0]:
        raise ValueError("conditional and unconditional batch sizes differ")
    if tuple(conditional.shape[2:]) != tuple(unconditional.shape[1:]):
        raise ValueError("conditional prediction dimensions do not match unconditional")
    expected_meta_shape = tuple(conditional.shape[:2])
    if tuple(component_mask.shape) != expected_meta_shape:
        raise ValueError("component_mask must match [batch, components]")
    if tuple(weights.shape) != expected_meta_shape:
        raise ValueError("weights must match [batch, components]")
    if component_mask.dtype != torch.bool:
        raise TypeError("component_mask must be boolean")
    if not weights.is_floating_point():
        raise TypeError("weights must be floating point")
    if component_mask.device != unconditional.device or weights.device != unconditional.device:
        raise ValueError("predictions, mask, and weights must be on the same device")
    if conditional.device != unconditional.device:
        raise ValueError("conditional and unconditional must be on the same device")
    if not bool(component_mask.any(dim=1).all().item()):
        raise ValueError("every sample must have at least one active component")
    if not bool(torch.isfinite(unconditional).all().item()):
        raise FloatingPointError("unconditional prediction contains NaN or Inf")
    if not bool(torch.isfinite(conditional).all().item()):
        raise FloatingPointError("conditional prediction contains NaN or Inf")
    if not bool(torch.isfinite(weights).all().item()):
        raise FloatingPointError("component weights contain NaN or Inf")

    work_dtype = torch.float32 if stable_fp32 else unconditional.dtype
    unconditional_work = unconditional.to(dtype=work_dtype)
    conditional_work = conditional.to(dtype=work_dtype)
    weights_work = weights.to(dtype=work_dtype)
    result = unconditional_work.clone()
    broadcast_tail = (1,) * (unconditional.ndim - 1)

    # Preserve the official implementation's left-to-right component loop.
    for slot_index in range(conditional.shape[1]):
        active = component_mask[:, slot_index].reshape(-1, *broadcast_tail)
        weight = weights_work[:, slot_index].reshape(-1, *broadcast_tail)
        delta = conditional_work[:, slot_index] - unconditional_work
        result = result + torch.where(active, weight * delta, torch.zeros_like(delta))

    if not bool(torch.isfinite(result).all().item()):
        raise FloatingPointError("composed prediction contains NaN or Inf")
    return result
