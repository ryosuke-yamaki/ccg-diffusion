"""Count Loss: count constraints from the derivation, frozen targets and the density loss."""
from __future__ import annotations

from types import MappingProxyType
from typing import Any

import pytest
import torch

from conftest import marked_dependencies, synthetic_sentence
from ccg_diffusion.guidance.count_loss import (
    ACTIVE_OUTER_STEPS,
    COUNT_WEIGHT,
    TAU,
    CountConstraint,
    CountDecision,
    CountGuidance,
    CountState,
    NounForms,
    augment_count_losses,
    count_divergence,
    distribution,
    extract_count_constraints,
    fresh_count_states,
    make_target,
    numeric_definition,
    numeric_definition_sha256,
)
from ccg_diffusion.targets.pd_indices import PDIndices

FORMS = NounForms(frozenset({"dog", "cat", "mouse", "apple"}), (("s", ""),), MappingProxyType({"mice": ("mouse",)}))


def test_constants_and_noun_forms() -> None:
    assert (TAU, COUNT_WEIGHT, ACTIVE_OUTER_STEPS) == (0.3, 0.1, tuple(range(10)))
    assert numeric_definition()["smoothing"] == {"kernel_size": 3, "sigma": .5, "padding": "replicate"}
    assert FORMS.candidates("dogs") == {"dog"} and FORMS.candidates("mice") == FORMS.candidates("mouse")


def _extract(words: list[str], categories: list[str], dependencies: tuple[str, ...] = (),
             parse_status: str = "complete") -> CountDecision:
    alignment = [{"source_index": i, "text": w, "clip_index": i, "clip_lexical_piece_indices": [i]}
                 for i, w in enumerate(words, 1)]
    nouns = [i for i, category in enumerate(categories, 1) if category == "N"]
    return extract_count_constraints(
        synthetic_sentence(" ".join(words), categories), marked_dependencies(*dependencies), alignment,
        PDIndices(exist=nouns), parse_status=parse_status, unresolved_modifiers=(), text_token_count=len(words) + 2,
        noun_forms=FORMS)


@pytest.mark.parametrize(("numeral", "count"), [("one", 1), ("three", 3), ("ten", 10), ("7", 7)])
def test_explicit_cardinal_binds_its_noun(numeral: str, count: int) -> None:
    decision = _extract([numeral, "dogs"], ["N/N", "N"], (f"{numeral}_1 (N{{Y}}/N{{Y}}<1>){{_}} 1 dogs_2 0",))
    assert decision.status == "ok" and decision.reasons == ("explicit_cardinal_noun_bindings",)
    (constraint,) = decision.constraints
    assert (constraint.count, constraint.noun_word, constraint.noun_anchor, constraint.lexical_piece_indices) == (
        count, "dogs", 2, (2,))
    assert CountDecision.from_record(decision.as_record()) == decision


@pytest.mark.parametrize(("words", "categories", "dependencies", "parse_status", "status", "reason"), [
    (["red", "dogs"], ["N/N", "N"], (), "complete", "no_explicit_cardinal", "no_explicit_cardinal"),
    (["many", "dogs"], ["N/N", "N"], (), "complete", "abstain_B0", "unsupported_quantifier"),
    (["twelve", "dogs"], ["N/N", "N"], (), "complete", "abstain_B0", "unsupported_cardinality"),
    (["two", "dogs"], ["N/N", "N"], (), "partial", "abstain_B0", "incomplete_parse"),
    (["two", "dogs"], ["N/N", "N"], (), "complete", "abstain_B0", "missing_or_distributed_cardinal_binding"),
    (["two", "dogs", "and", "three", "dogs"], ["N/N", "N", "conj", "N/N", "N"],
     ("two_1 (N{Y}/N{Y}<1>){_} 1 dogs_2 0", "three_4 (N{Y}/N{Y}<1>){_} 1 dogs_5 0"),
     "complete", "abstain_B0", "repeated_noun_referent_ambiguous"),
])
def test_ambiguous_counts_abstain(words, categories, dependencies, parse_status, status, reason) -> None:
    decision = _extract(words, categories, dependencies, parse_status)
    assert decision.status == status and reason in decision.reasons and decision.constraints == ()


def _map(point: tuple[int, int] = (6, 7), mean: float = .005) -> torch.Tensor:
    result = torch.full((16, 16), mean)
    result[point] += .1
    return result


def _attention(point: tuple[int, int] = (6, 7)) -> torch.Tensor:
    values = torch.full((16, 16, 77), .003)
    values[:, :, 2] = _map(point)
    values[:, :, 5] = _map((9, 9))
    return values.requires_grad_()


def _constraint(anchor: int = 2, count: int = 2) -> CountConstraint:
    return CountConstraint(count, anchor - 1, anchor, f"noun{anchor}", anchor, (anchor,), 12)


@pytest.mark.parametrize("requested", range(1, 11))
def test_target_has_the_requested_centres_and_equal_masses(requested: int) -> None:
    target = make_target(_map(), requested)
    assert len(set(target.centers)) == requested and not target.density.requires_grad
    assert target.density.sum() == pytest.approx(1.0, abs=2e-6)
    for index in range(requested):
        assert float(target.density[target.region_assignment == index].sum()) == pytest.approx(1 / requested, abs=3e-4)


def test_symmetric_kl_formula() -> None:
    raw = _map((3, 3)).requires_grad_()
    target = make_target(_map(), 2)
    observed = distribution(raw)
    expected = (0.5 * (target.density - observed) * (target.density.log() - observed.log())).sum()
    assert torch.equal(count_divergence(raw, target), expected.clamp_min(0.0))
    gradient = torch.autograd.grad(count_divergence(raw, target), raw)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


@pytest.mark.parametrize("branch", ["disabled", "ineligible", "outside_steps", "weak"])
def test_inactive_count_returns_the_same_base_loss_object(branch: str) -> None:
    attention = _attention()
    if branch == "weak":
        attention = (attention.detach() * 1e-4).requires_grad_()
    state = CountState("image", () if branch == "ineligible" else (_constraint(),), enabled=branch != "disabled")
    base = attention.square().sum()
    assert state.augment(base, attention, outer_step_index=10 if branch == "outside_steps" else 0) is base
    assert not state.targets and state.active_evaluations == 0


def test_first_ready_evaluation_freezes_the_targets_of_all_nouns() -> None:
    state = CountState("image", (_constraint(), _constraint(5)))
    weak = _attention().detach().clone()
    weak[:, :, 5] = 0
    weak.requires_grad_()
    weak_base = weak.square().sum()
    assert state.augment(weak_base, weak, outer_step_index=0) is weak_base and not state.targets
    attention = _attention()
    base = attention.square().sum()
    result = state.augment(base, attention, outer_step_index=1)
    assert (state.first_active_step, state.first_active_evaluation, len(state.targets)) == (1, 2, 2)
    assert result is not base and result.requires_grad
    targets = state.targets
    changed = _attention((1, 1))
    state.augment(changed.square().sum(), changed, outer_step_index=9)
    assert state.targets is targets
    assert state.augment(base, attention, outer_step_index=10) is base and state.expired
    with pytest.raises(ValueError, match="monotonic"):
        state.augment(base, attention, outer_step_index=9)


def test_batch_states_are_independent_and_absent_channel_is_a_no_op() -> None:
    def guidance(enabled: bool) -> CountGuidance:
        decision = CountDecision("ok", ("explicit_cardinal_noun_bindings",), (_constraint(),))
        return CountGuidance(enabled, decision, "a" * 64, "b" * 64, "c" * 64)

    states = fresh_count_states([guidance(True), guidance(False)], ["row0", "row1"])
    attention = torch.stack([_attention().detach(), _attention().detach()]).requires_grad_()
    bases = [attention[0].sum(), attention[1].sum()]
    result = augment_count_losses(bases, attention, states, outer_step_index=0)
    assert result[0] is not bases[0] and result[1] is bases[1]
    assert states[1].inactive_calls == {"disabled": 1}
    losses: list[Any] = [torch.tensor(1.0)]
    assert augment_count_losses(losses, None, None, outer_step_index=0) is losses
    record = guidance(True).as_record()
    assert record["schema"] == "holsd.sd14_count_job_guidance.v1"
    assert record["numeric_definition_sha256"] == numeric_definition_sha256()
