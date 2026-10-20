"""Spatial Relation Loss (paper Eq. 7).

For each closed-set 2D relation between a subject X and an object Y, the
attention map of each endpoint token is min-max normalized and turned into a
centroid ``(cx, cy)`` in ``[0, 1]`` (``cy`` grows downward, as in the image and
the official metric). Per relation the loss is a placement hinge plus a
separation term:

* directional labels (left/right/top/bottom): ``relu(m - gap)`` on the signed
  centre offset along the requested axis, plus ``relu(|off-axis gap| - |gap|)``
  so that the requested axis dominates;
* proximity labels (next to/near/on side of): ``relu(min(dx, dy) - r)``;
* separation: ``mean(a_X * a_Y)`` over the normalized maps.

The loss is the mean over relations. Callers add
``SPATIAL_RELATION_LOSS_WEIGHT * compute_spatial_relation_loss(...)`` to the
Predicated Diffusion loss of the image (lambda_spatial = 6, m = 0.25,
r = 150/512; Table 8).
"""
from __future__ import annotations

import math
from types import MappingProxyType
from typing import Dict, Mapping, Sequence

import torch

from ccg_diffusion.guidance.spatial_relations import (
    DIRECTIONAL_LABELS,
    PROXIMITY_LABELS,
    RelationConstraint,
)

SPATIAL_RELATION_LOSS_WEIGHT = 6.0  # lambda_spatial
# Required signed centre offset of the directional labels, in units of the image edge.
MARGIN = 0.25
# Proximity target of the non-directional labels: the metric's 150 px / 512 px threshold.
PROXIMITY_MAX = 150.0 / 512.0
SEPARATION_WEIGHT = 1.0
OFF_AXIS_WEIGHT = 1.0
# Recorded in guidance records and hashed into the routing policy (KEEP).
SPATIAL_RELATION_LOSS_PARAMETERS: Mapping[str, float | str] = MappingProxyType({
    "weight": 6.0,
    "margin": 0.25,
    "proximity_max": 150.0 / 512.0,
    "separation_weight": 1.0,
    "geometry_variant": "metric_aligned_v1",
    "off_axis_weight": 1.0,
})


def _minmax(attention: torch.Tensor) -> torch.Tensor:
    """Rescale a 2-D map to ``[0, 1]``; a constant map maps to all-zeros."""

    maximum = attention.max()
    minimum = attention.min()
    span = maximum - minimum
    if bool((span <= 0).item()):
        return torch.zeros_like(attention)
    return (attention - minimum) / span


def _centroid(
    normalised: torch.Tensor,
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the attention-weighted centre ``(cx, cy)`` of a ``[0, 1]`` map."""

    mass = normalised.sum()
    weights = normalised / (mass + eps)
    return (weights * x_grid).sum(), (weights * y_grid).sum()


def _pixel_grids(
    attention_res: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``[res, res]`` pixel-centre coordinates in ``[0, 1]``.

    Index 0 of the first axis is the top row, so ``y`` grows downward, the
    same orientation the decoded image and the official metric use.
    """

    centres = (torch.arange(attention_res, device=device, dtype=dtype) + 0.5)
    centres = centres / float(attention_res)
    y_grid = centres.view(-1, 1).expand(attention_res, attention_res)
    x_grid = centres.view(1, -1).expand(attention_res, attention_res)
    return x_grid, y_grid


def compute_spatial_relation_loss(
    attention_maps: torch.Tensor,
    relations: Sequence[RelationConstraint],
    *,
    attention_res: int = 16,
    margin: float = MARGIN,
    proximity_max: float = PROXIMITY_MAX,
    separation_weight: float = SEPARATION_WEIGHT,
    off_axis_weight: float = OFF_AXIS_WEIGHT,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return the mean per-relation placement + separation loss as an FP32 scalar (unweighted).

    ``attention_maps`` is ``[res, res, num_tokens]`` (rows = image y growing
    downward, columns = image x). Relation endpoints index the full token axis.
    An empty ``relations`` returns a zero scalar tied to ``attention_maps``.
    """

    if attention_maps.ndim != 3:
        raise ValueError(
            f"attention_maps must be [res, res, num_tokens], got {tuple(attention_maps.shape)}"
        )
    if attention_maps.shape[0] != attention_res or attention_maps.shape[1] != attention_res:
        raise ValueError(
            f"attention_maps grid {tuple(attention_maps.shape[:2])} does not match "
            f"attention_res={attention_res}"
        )
    numeric_args = {
        "margin": margin,
        "proximity_max": proximity_max,
        "separation_weight": separation_weight,
        "off_axis_weight": off_axis_weight,
        "eps": eps,
    }
    if any(not math.isfinite(float(value)) for value in numeric_args.values()):
        raise ValueError("relation loss numeric arguments must be finite")
    if (
        margin < 0.0
        or proximity_max < 0.0
        or separation_weight < 0.0
        or off_axis_weight < 0.0
        or eps <= 0.0
    ):
        raise ValueError(
            "margin, proximity_max, separation_weight, and off_axis_weight "
            "must be >= 0 and eps must be > 0"
        )

    work_maps = attention_maps.float()
    if not relations:
        return work_maps.sum() * 0.0

    num_tokens = work_maps.shape[-1]
    for relation in relations:
        for index in (relation.subject_clip_index, relation.object_clip_index):
            if index >= num_tokens:
                raise IndexError(
                    f"relation endpoint {index} is outside the {num_tokens}-token axis"
                )

    x_grid, y_grid = _pixel_grids(
        attention_res, device=work_maps.device, dtype=work_maps.dtype
    )
    normalised: Dict[int, torch.Tensor] = {}

    def _norm(index: int) -> torch.Tensor:
        cached = normalised.get(index)
        if cached is None:
            cached = _minmax(work_maps[:, :, index])
            normalised[index] = cached
        return cached

    loss = torch.zeros((), device=work_maps.device, dtype=work_maps.dtype)
    for relation in relations:
        subject = _norm(relation.subject_clip_index)
        obj = _norm(relation.object_clip_index)
        cx_subject, cy_subject = _centroid(subject, x_grid, y_grid, eps=eps)
        cx_object, cy_object = _centroid(obj, x_grid, y_grid, eps=eps)

        if relation.label in DIRECTIONAL_LABELS:
            if relation.label == "left":
                gap = cx_object - cx_subject
                off_axis_gap = torch.abs(cy_object - cy_subject)
            elif relation.label == "right":
                gap = cx_subject - cx_object
                off_axis_gap = torch.abs(cy_object - cy_subject)
            elif relation.label == "top":
                gap = cy_object - cy_subject
                off_axis_gap = torch.abs(cx_object - cx_subject)
            else:  # bottom
                gap = cy_subject - cy_object
                off_axis_gap = torch.abs(cx_object - cx_subject)
            placement = torch.relu(margin - gap)
            # The official directional branch requires the requested axis
            # to dominate the orthogonal axis in addition to the sign.
            placement = placement + off_axis_weight * torch.relu(
                off_axis_gap - torch.abs(gap)
            )
        elif relation.label in PROXIMITY_LABELS:
            dx = torch.abs(cx_subject - cx_object)
            dy = torch.abs(cy_subject - cy_object)
            # The metric grants full proximity credit when either centre
            # axis is within 150 px, so the hinge operates on min(dx, dy).
            distance = torch.minimum(dx, dy)
            placement = torch.relu(distance - proximity_max)
        else:  # pragma: no cover - RelationConstraint validates the label set
            raise ValueError(f"unknown relation label: {relation.label!r}")

        overlap = (subject * obj).mean()
        loss = loss + placement + separation_weight * overlap

    return loss / len(relations)
