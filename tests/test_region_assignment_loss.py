"""Region Assignment Loss, the prefix loss override and the region-only plan."""
from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from conftest import WordTokenizer, phrase_plan_of, prefix_source, screening_stubs
from ccg_diffusion.guidance.positional_prefix import build_prefix_plan
from ccg_diffusion.guidance.predicated_diffusion_loss import WholeWordSpans, compute_pd_loss
from ccg_diffusion.guidance.region_assignment_loss import (
    BOUNDARY,
    BOUNDARY_WIDTH,
    REGION_WEIGHT,
    PrefixRegionLoss,
    RegionGroups,
    in_source_coordinates,
    region_assignment_loss,
)
from ccg_diffusion.targets.pd_indices import PDIndices


def _plan():
    args = prefix_source()
    depth, clause = screening_stubs(args)
    plan = build_prefix_plan(args["prompt"], args["alignment"], args["program"], phrase_plan_of(args), WordTokenizer(),
                             depth=depth, clause=clause)
    return args, plan


def _closed_form(attention: torch.Tensor, groups) -> torch.Tensor:
    """M_L(x) = sigmoid((x_b - x) / kappa), M_R = 1 - M_L; mean of -log(mass fraction in the assigned half)."""
    res = attention.shape[0]
    x = (torch.arange(res, dtype=torch.float64) + 0.5) / res
    left = torch.sigmoid((BOUNDARY - x) / BOUNDARY_WIDTH).expand(res, res)
    terms = []
    for words, mask in zip(groups, (left, 1 - left)):
        for pieces in words:
            mass = attention.double()[..., list(pieces)].mean(-1)
            terms.append(-torch.log((mass * mask).sum() / mass.sum()))
    return torch.stack(terms).mean()


def test_region_assignment_loss_closed_form() -> None:
    attention = torch.rand(16, 16, 12, generator=torch.Generator().manual_seed(3))
    groups = (((3,), (2,)), ((7,), (5, 6)))
    loss = region_assignment_loss(attention, RegionGroups(True, "left", groups))
    assert loss.dtype == torch.float32
    assert abs(float(loss) - float(_closed_form(attention, groups))) < 1e-7
    assert (REGION_WEIGHT, BOUNDARY, BOUNDARY_WIDTH) == (0.25, 0.5, 0.08)


def test_gradient_moves_each_group_toward_its_side() -> None:
    _, plan = _plan()
    attention = torch.ones(16, 16, 77, dtype=torch.bfloat16, requires_grad=True)
    region_assignment_loss(attention, plan).backward()
    used = set()
    for words, side in zip(plan.groups, (-1, 1)):
        for pieces in words:
            used.update(pieces)
            left = attention.grad[:, :8, list(pieces)].float().mean()
            right = attention.grad[:, 8:, list(pieces)].float().mean()
            assert (left - right) * side > 0
    assert not attention.grad[..., sorted(set(range(77)) - used)].count_nonzero()


def test_inactive_layout_and_unsupported_relation() -> None:
    attention = torch.rand(16, 16, 12)
    assert float(region_assignment_loss(attention, RegionGroups(False, None, ()))) == 0.0
    with pytest.raises(ValueError, match="invalid prefix regional plan"):
        region_assignment_loss(attention, RegionGroups(True, "top", (((1,),), ((2,),))))
    with pytest.raises(ValueError, match="square"):
        region_assignment_loss(torch.rand(16, 8, 12), RegionGroups(True, "left", (((1,),), ((2,),))))


def test_prefix_region_loss_remaps_targets_and_spans() -> None:
    args, plan = _plan()
    mapping = dict(plan.mapping)
    indices = PDIndices(**args["program"])
    spans = WholeWordSpans.from_mapping({i: (i,) for i in (2, 3, 6, 7)}, text_token_count=plan.original_token_count)
    attention = torch.rand(16, 16, 77, generator=torch.Generator().manual_seed(5))
    override = PrefixRegionLoss(plan, region_weight=REGION_WEIGHT)
    loss = override(attention, indices, attention_res=16, pair_token_spans=spans)
    remapped = WholeWordSpans.from_mapping({mapping[i]: (mapping[i],) for i in (2, 3, 6, 7)},
                                           text_token_count=plan.conditioning_token_count)
    expected = (compute_pd_loss(attention, PDIndices(**plan.mapped_program), attention_res=16,
                                pair_token_spans=remapped)
                + REGION_WEIGHT * region_assignment_loss(attention, plan))
    assert torch.equal(loss, expected)
    assert override.evaluations == 1 and override.geometry_sum > 0
    assert torch.equal(PrefixRegionLoss(plan, region_weight=0.0)(attention, indices),
                       compute_pd_loss(attention, PDIndices(**plan.mapped_program)))
    with pytest.raises(ValueError, match="different original"):
        override(attention, replace(indices, exist=[1]))
    with pytest.raises(ValueError, match="invalid prefix PD"):
        PrefixRegionLoss(replace(plan, phantom_anchor=4), region_weight=REGION_WEIGHT)


def test_region_only_plan_keeps_groups_and_sides_on_the_source_prompt() -> None:
    args, plan = _plan()
    region_only = in_source_coordinates(plan, args["program"])
    assert region_only.groups == (((3,), (2,)), ((7,), (6,))) and region_only.relation == plan.relation
    assert region_only.mapped_program == args["program"]
    assert dict(region_only.mapping) == {i: i for i, _ in plan.mapping}
    loss = PrefixRegionLoss(region_only, region_weight=REGION_WEIGHT)
    assert loss.indices == PDIndices(**args["program"]) and loss.executed_program_sha256 == plan.program_sha256
    with pytest.raises(ValueError, match="active prefix plan"):
        in_source_coordinates(plan, {**args["program"], "exist": [3]})
