"""Region Assignment Loss (paper Eq. 5-6).

Two word groups (noun plus modifiers of each noun phrase) are assigned to the
left and right halves of the image. With pixel x-coordinates in ``[0, 1]`` and
the soft half-plane mask ``M_L(x) = sigmoid((x_b - x) / kappa)``
(x_b = 0.5, kappa = 0.08; ``M_R = 1 - M_L``), each word's mass fraction in its
assigned half is ``f = sum(A * M) / sum(A)`` over its lexical-piece mean map
``A``, and the loss is the mean of ``-log f`` over all words. Callers add
``REGION_WEIGHT * region_assignment_loss(...)`` (lambda_region = 0.25).

``PrefixRegionLoss`` is the per-image loss override of the SD1.4 kernel for a
prompt with a Positional Prefix: it evaluates the Predicated Diffusion loss on
the remapped guidance targets of the prefixed text and adds the weighted
Region Assignment Loss. ``in_source_coordinates`` gives the same word groups on
the unprefixed prompt (Positional Prefix removed, Region Assignment Loss kept).
``RegionGroups`` carries the groups on Sana's Gemma token axis.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Mapping, Protocol, Sequence

import torch

from ccg_diffusion.guidance.positional_prefix import PrefixLayoutPlan
from ccg_diffusion.guidance.predicated_diffusion_loss import WholeWordSpans, compute_pd_loss
from ccg_diffusion.provenance import canonical_json, sha256_text
from ccg_diffusion.targets.pd_indices import PDIndices

REGION_WEIGHT = .25  # lambda_region
BOUNDARY = 0.5  # x_b (the literal .5 is written in the loss expression)
BOUNDARY_WIDTH = 0.08  # kappa (the literal .08 is written in the loss expression)


class RegionLayout(Protocol):
    """Word groups assigned to the left (first) and right (second) halves of the image."""
    @property
    def active(self) -> bool: ...

    @property
    def relation(self) -> str | None: ...

    @property
    def groups(self) -> Sequence[Sequence[Sequence[int]]]: ...


@dataclass(frozen=True)
class RegionGroups:
    """Region Assignment Loss word groups on an arbitrary token axis (Sana)."""

    active: bool
    relation: str | None
    groups: tuple[tuple[tuple[int, ...], ...], ...]


def region_assignment_loss(attention: torch.Tensor, layout: RegionLayout) -> torch.Tensor:
    """Mean ``-log`` mass fraction of each grouped word in its assigned half (unweighted, FP32)."""
    if attention.ndim != 3 or attention.shape[0] != attention.shape[1]:
        raise ValueError("prefix regional loss expects square [H,W,tokens] attention")
    work = attention.float()
    if not layout.active:
        return work.sum() * 0
    if len(layout.groups) != 2 or layout.relation != "left":
        raise ValueError("invalid prefix regional plan")
    axis = (torch.arange(work.shape[0], device=work.device, dtype=work.dtype) + .5) / work.shape[0]
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    coordinate = x
    sign = -1
    first = torch.sigmoid(sign * (coordinate - .5) / .08)
    losses = []
    for words, mask in zip(layout.groups, (first, 1 - first), strict=True):
        if not words:
            raise ValueError("prefix regional group contains no bound word")
        for pieces in words:
            mass = work[..., list(pieces)].mean(-1)
            fraction = (mass * mask).sum() / mass.sum().clamp_min(1e-8)
            losses.append(-fraction.clamp_min(1e-8).log())
    return torch.stack(losses).mean()


def in_source_coordinates(plan: PrefixLayoutPlan, program: Mapping[str, Any]) -> PrefixLayoutPlan:
    """Region-only plan: the same word groups and sides, located on the unaltered source prompt.

    The prefix mapping is inverted so each grouped piece refers to its source position; the PD
    program is the original one and whole-word spans keep their source pieces.
    """
    if not plan.active or sha256_text(canonical_json(program)) != plan.program_sha256:
        raise ValueError("region-only plan requires the active prefix plan of this source program")
    inverse = {target: source for source, target in plan.mapping}
    if len(inverse) != len(plan.mapping) or any(p not in inverse for g in plan.groups for w in g for p in w):
        raise ValueError("prefix groups are not covered by the source mapping")
    groups = tuple(tuple(tuple(inverse[p] for p in word) for word in group) for group in plan.groups)
    return replace(plan, mapping=tuple((source, source) for source, _ in plan.mapping),
                   mapped_program_json=canonical_json(dict(program)), groups=groups,
                   conditioning_token_count=plan.original_token_count, phantom_anchor=None)


class PrefixRegionLoss:
    """Per-image loss override: L_PD on the prefix coordinates plus ``region_weight`` x Region Assignment Loss.

    The kernel calls it with the image's original guidance targets; it checks
    them against the plan, remaps the whole-word spans, and evaluates the loss
    on the remapped targets.
    """

    def __init__(self, plan: PrefixLayoutPlan, *, region_weight: float) -> None:
        if (not plan.active or plan.phantom_anchor is not None or isinstance(region_weight, bool)
                or not math.isfinite(region_weight) or region_weight < 0):
            raise ValueError("invalid prefix PD configuration")
        self.plan, self.region_weight = plan, region_weight
        mapped = plan.mapped_program
        self.indices = PDIndices(**mapped)
        self.executed_program_sha256 = sha256_text(canonical_json(mapped))
        self.evaluations = 0
        self.geometry_sum = 0.0

    def __call__(self, attention: torch.Tensor, indices: PDIndices, **pd_kwargs: Any) -> torch.Tensor:
        original = {field: getattr(indices, field) for field in ("exist", "corr", "leak", "possession")}
        if sha256_text(canonical_json(original)) != self.plan.program_sha256:
            raise ValueError("prefix PD received a different original program")
        spans = pd_kwargs.get("pair_token_spans")
        if spans is not None:
            if spans.text_token_count != self.plan.original_token_count:
                raise ValueError("prefix pair spans refer to a different source text window")
            mapping = dict(self.plan.mapping)
            used = {i for pair in [*self.indices.corr, *self.indices.leak] for i in pair}
            pd_kwargs["pair_token_spans"] = WholeWordSpans.from_mapping(
                {mapping[a]: tuple(mapping[p] for p in pieces) for a, pieces in spans.as_mapping().items()
                 if mapping[a] in used}, text_token_count=self.plan.conditioning_token_count)
        loss = compute_pd_loss(attention, self.indices, **pd_kwargs)
        if self.region_weight:
            geometry = region_assignment_loss(attention, self.plan)
            self.geometry_sum += float(geometry.detach())
            loss = loss + self.region_weight * geometry
        self.evaluations += 1
        return loss
