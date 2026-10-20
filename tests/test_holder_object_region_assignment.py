"""Holder-Object Region Assignment, the TAB guidance strength and the TAB objective."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from ccg_diffusion.guidance.predicated_diffusion_loss import SanaWholeWordSpans
from ccg_diffusion.guidance.region_assignment_loss import RegionGroups
from ccg_diffusion.guidance.spatial_relations import RelationConstraint
from ccg_diffusion.methods import TabAblation
from ccg_diffusion.pp_attachment.objective import TabObjective, guidance_strength, tab_objective
from ccg_diffusion.pp_attachment.sana import PLAN_SCHEMA, SanaPPAttachmentLoss
from ccg_diffusion.targets.pd_indices import PDIndices

FIXTURES = Path(__file__).parent / "fixtures"


def _fixture_rows() -> dict[tuple[str, str], dict]:
    return {(r["tab_id"], r["reading"]): r for r in json.loads((FIXTURES / "tab_pp_frozen_jobs.json").read_text())}


def _maps(tokens: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(16, 16, tokens, generator=generator).softmax(-1)


@pytest.mark.requires_clip_tokenizer
def test_region_groups_follow_the_selected_interpretation() -> None:
    from ccg_diffusion.pp_attachment.compile import compile_interpretation
    from ccg_diffusion.pp_attachment.jobs import load_frozen_bundle
    from ccg_diffusion.targets.tokenizer import load_sd14_tokenizer

    bundle, tokenizer = load_frozen_bundle(), load_sd14_tokenizer()
    np_interpretation = bundle.interpretations["tab-pp-037", "np_attach"]
    vp_interpretation = bundle.interpretations["tab-pp-037", "vp_attach"]
    rows = {"np_attach": compile_interpretation(np_interpretation, vp_interpretation, tokenizer, TabAblation()),
            "vp_attach": compile_interpretation(vp_interpretation, np_interpretation, tokenizer, TabAblation())}
    placement = ("arm", "run_id", "condition", "image_id", "effective_sha256", "original_position",
                 "physical_batch_index", "physical_batch_position")
    for reading, other in (("np_attach", "vp_attach"), ("vp_attach", "np_attach")):
        row = rows[reading]
        selected = bundle.interpretations["tab-pp-037", reading].paraphrase
        opposite = bundle.interpretations["tab-pp-037", other].paraphrase
        clip = {int(k): v for k, v in selected["source_to_conditioning_clip"].items()}
        holder, alternative = clip[selected["holder_source_index"]], clip[opposite["holder_source_index"]]
        obj = clip[selected["object_source_index"]]
        groups = row["transfer_region"]["plan"]["groups"]
        selected_group = next(group for group in groups if [holder] in group)
        assert [obj] in selected_group and [alternative] not in selected_group
        # Source order: the girl (source 2) precedes the chair (source 5), so the girl's group is on the left.
        left, right = groups
        assert [clip[2]] in left and [clip[5]] in right
        ablation = row["transfer_trace"]["ablation"]
        assert row["pd_indices"]["possession"] == [] and ablation["removed_possession"] == [[holder, obj]]
        assert row["transfer_region"]["weight"] == 0.25 and row["transfer_trace"]["layout_reason"] == "active"
        assert row["loss_config"]["scale_factor"] == 80.0 and row["loss_config"]["max_update_ratio"] == 0.06
        frozen = _fixture_rows()["tab-pp-037", reading]
        assert row == {k: v for k, v in frozen.items() if k not in placement}


def test_guidance_strength_is_ten_or_eighty_on_the_kernel_base_scale() -> None:
    rows = _fixture_rows()
    default = guidance_strength(rows["tab-pp-000", "np_attach"])
    boosted = guidance_strength(rows["tab-pp-037", "np_attach"])
    assert (default.effective_profile, 20.0 * default.scale_multiplier, default.max_update_ratio) == ("B0", 10.0, .02)
    assert (boosted.effective_profile, 20.0 * boosted.scale_multiplier, boosted.max_update_ratio) == ("P2", 80.0, .06)
    assert default.active_pair_spans is None and boosted.active_pair_spans is None  # no whole-word pooling on SD1.4
    with pytest.raises(ValueError, match="dose and cap"):
        guidance_strength({**rows["tab-pp-000", "np_attach"],
                           "loss_config": {"scale_factor": 80.0, "max_update_ratio": 0.02}})


def test_region_gradient_moves_the_object_toward_its_assigned_side() -> None:
    from ccg_diffusion.guidance.region_assignment_loss import region_assignment_loss

    row = _fixture_rows()["tab-pp-037", "np_attach"]
    objective = TabObjective(row)
    left, right = row["transfer_region"]["plan"]["groups"]
    obj = 8  # "bag" is grouped with its holder "chair" on the right half (image columns 8-15)
    assert [obj] in right
    maps = _maps(len(row["token_ids"])).requires_grad_(True)
    region_assignment_loss(maps, objective.region).backward()
    assert (maps.grad[:, 8:, obj] < 0).all() and (maps.grad[:, :8, obj] > 0).all()
    objective(maps.detach(), PDIndices(**row["pd_indices"]), attention_res=16)
    assert objective.diagnostics()["evaluations"] == 1 and objective.diagnostics()["mean_region_loss"] > 0


def test_objective_adds_six_spatial_and_a_quarter_region() -> None:
    row = _fixture_rows()["tab-pp-000", "np_attach"]
    maps = _maps(len(row["token_ids"]), seed=1)
    indices = PDIndices(**row["pd_indices"])
    objective = TabObjective(row)
    total, spatial, region = tab_objective(maps, indices, objective.relations, objective.region, attention_res=16)
    base, _, _ = tab_objective(maps, indices, (), None, attention_res=16)
    assert torch.allclose(total, base + 6.0 * spatial + .25 * region)
    assert spatial > 0 and region > 0
    with pytest.raises(ValueError, match="program changed"):
        objective(maps, PDIndices(exist=[1]), attention_res=16)


def _sana_plan(scale_factor: float) -> dict:
    return {"schema": PLAN_SCHEMA, "pd_indices": {"exist": [2, 5, 7], "corr": [[4, 5]], "leak": [[4, 2]],
                                                  "possession": []},
            "loss_config": {"scale_factor": scale_factor, "max_update_ratio": .06 if scale_factor == 80 else .02},
            "token_spans": {"text_token_count": 9, "spans": [{"anchor": 2, "lexical_piece_indices": [1, 2]},
                                                             {"anchor": 4, "lexical_piece_indices": [4]},
                                                             {"anchor": 5, "lexical_piece_indices": [5]}]},
            "text_token_count": 9, "relations": [{"label": "next_to", "subject_clip_index": 2,
                                                  "object_clip_index": 7}],
            "region_groups": [[[1, 2]], [[4], [5]]]}


@pytest.mark.parametrize("scale_factor", [10.0, 80.0])
def test_sana_objective_equals_the_shared_tab_objective(scale_factor: float) -> None:
    plan = _sana_plan(scale_factor)
    loss = SanaPPAttachmentLoss(plan, batch_size=2)
    assert loss.scale_multiplier * 20.0 == scale_factor and loss.max_update_ratio == plan["loss_config"][
        "max_update_ratio"]
    maps = _maps(9, seed=3).requires_grad_(True)
    value = loss(maps, 0)
    (gradient,) = torch.autograd.grad(value, maps)
    reference_maps = maps.detach().clone().requires_grad_(True)
    spans = SanaWholeWordSpans.from_mapping({2: (1, 2), 4: (4,), 5: (5,)}, text_token_count=9)
    expected, _, _ = tab_objective(
        reference_maps, PDIndices(**plan["pd_indices"]), (RelationConstraint("next_to", 2, 7),),
        RegionGroups(True, "left", (((1, 2),), ((4,), (5,)))), attention_res=16, special_layout="leading_bos",
        pair_token_spans=spans if scale_factor == 80.0 else None)
    (expected_gradient,) = torch.autograd.grad(expected, reference_maps)
    assert torch.equal(value, expected) and torch.equal(gradient, expected_gradient)
    record = loss.audit_record()
    assert record["route"] == ("P2" if scale_factor == 80.0 else "B0")
    assert record["update_diagnostics"]["generation_kernel"] == "sana_tab_fp32_master"
    assert record["loss_trace"][0]["step"] == 0
    with pytest.raises(ValueError, match="token count"):
        loss(_maps(10), 1)
