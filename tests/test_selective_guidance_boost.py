"""Selective Guidance Boost: base routes from the derivation, the attribute_v2 boost rule,
route guidance strengths and the frozen policy records."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from types import MappingProxyType
from typing import Any

import pytest

from conftest import WordTokenizer, marked_dependencies, synthetic_sentence
from ccg_diffusion.guidance.count_loss import NounForms
from ccg_diffusion.guidance.predicated_diffusion_loss import WholeWordSpans
from ccg_diffusion.guidance.selective_guidance_boost import (
    BASE_SCALE,
    BOOSTED_UPDATE_CAP,
    BOOST_SCALE_MULTIPLIER,
    DEFAULT_UPDATE_CAP,
    GuidanceDecision,
    GuidanceRoute,
    ImageGuidance,
    SelectiveBoostGuidance,
    apply_selective_guidance_boost,
    boost_policy_sha256,
    derive_base_decision,
    derive_image_guidance,
)
from ccg_diffusion.provenance import canonical_json
from ccg_diffusion.targets.lexicon import NominalModifierLexicon
from ccg_diffusion.targets.pd_indices import PDIndices

LEXICON = NominalModifierLexicon(nouns=frozenset({"school", "gold", "small"}), adjectives=frozenset({"gold", "small"}))
NOUN_FORMS = NounForms(frozenset({"box", "vase"}), (("s", ""),), MappingProxyType({}))


def _alignment(words: list[str], modifiers: tuple[int, ...]) -> list[dict[str, Any]]:
    rows, cursor = [], 0
    for index, word in enumerate(words, 1):
        rows.append({"source_index": index, "prompt_word_index": index, "text": word, "clip_index": index,
                     "clip_lexical_piece_indices": [index], "char_start": cursor, "char_end": cursor + len(word),
                     "structural_role": "modifier" if index in modifiers else "entity",
                     "pd_roles": ["corr_modifier"] if index in modifiers else []})
        cursor += len(word) + 1
    return rows


def _decide(words: list[str], categories: list[str], modifiers: tuple[int, ...], entities: tuple[int, ...],
            dependencies: tuple[str, ...] = (), **kwargs: Any) -> tuple[GuidanceDecision, tuple]:
    indices = PDIndices(exist=list(entities), corr=[[m, entities[-1]] for m in modifiers])
    return derive_base_decision(
        sentence=synthetic_sentence(" ".join(words), categories), dependencies=marked_dependencies(*dependencies),
        alignment=_alignment(words, modifiers), indices=indices, parse_status=kwargs.get("parse_status", "complete"),
        unresolved_modifiers=kwargs.get("unresolved", ()), compound_lexicon=LEXICON)


def _guidance(profile: str = "B0", policy: str = "routing_v1", relations: tuple = (), **evidence: Any) -> ImageGuidance:
    decision = GuidanceDecision(profile, ("synthetic",), canonical_json({"parse_status": "complete", **evidence}))
    spans = WholeWordSpans.from_mapping({2: (1, 2), 3: (3,)}, text_token_count=5)
    return ImageGuidance(policy, decision, "a" * 64, "b" * 64, spans, relations)


def _boost(base: ImageGuidance, word: str, *, count_eligible: bool = False, role: str = "modifier"
           ) -> SelectiveBoostGuidance:
    rows = [{"source_index": 1, "text": word, "clip_index": 2, "structural_role": role, "pd_roles": ["corr_modifier"]}]
    return apply_selective_guidance_boost(base, rows, whole_word=True, count_eligible=count_eligible)


def test_route_labels() -> None:
    assert [route.value for route in GuidanceRoute] == ["B0", "P1", "P2"]


@pytest.mark.parametrize("modifier", ["red", "hexagonal", "wooden", "rough", "Teal"])
def test_visual_modifier_gives_the_boosted_route(modifier: str) -> None:
    decision, _ = _decide([modifier, "box"], ["N/N", "N"], (1,), (2,))
    assert decision.profile == "P2"
    boosted = _boost(_guidance(), modifier)
    assert (boosted.effective_profile, boosted.scale_multiplier, boosted.max_update_ratio) == ("P2", 4.0, 0.06)
    record = boosted.as_record()
    assert record["scale_factor"] == 80.0 and record["pooling_applied"] and record["whole_word"]
    assert record["kernel_policy"] == "routing_whole_word_v1" and record["routing_strategy"] == "attribute_v2"
    assert record["policy_sha256"] == boost_policy_sha256()


def test_whole_word_spans_only_on_boosted_routes_with_whole_word() -> None:
    boosted = _boost(_guidance(), "red")
    assert boosted.active_pair_spans is boosted.pair_spans
    no_pooling = apply_selective_guidance_boost(_guidance(), [], whole_word=False, count_eligible=False)
    assert no_pooling.active_pair_spans is None and no_pooling.policy == "routing_v1"
    assert replace(_guidance("P2"), policy="routing_v1").active_pair_spans is None
    assert replace(_guidance("P2"), policy="routing_whole_word_v1").active_pair_spans is not None


@pytest.mark.parametrize(("word", "role"), [("happy", "modifier"), ("glass", "entity"), ("school", "modifier")])
def test_non_visual_bindings_keep_the_default_route(word: str, role: str) -> None:
    assert _boost(_guidance(), word, role=role).effective_profile == "B0"


def test_count_eligible_images_keep_the_default_guidance_strength() -> None:
    result = _boost(_guidance(), "red", count_eligible=True)
    assert result.effective_profile == "B0" and result.decision.reasons == ("retain_COUNT_B0_scale_and_cap",)
    assert result.as_record()["scale_factor"] == 20.0 and result.active_pair_spans is None
    with pytest.raises(ValueError, match="COUNT"):
        _boost(_guidance("P2"), "red", count_eligible=True)


@pytest.mark.parametrize(("profile", "scale", "cap"), [
    ("B0", BASE_SCALE, DEFAULT_UPDATE_CAP), ("P2", BASE_SCALE * BOOST_SCALE_MULTIPLIER, BOOSTED_UPDATE_CAP)])
def test_route_guidance_strengths_equal_the_kernel_constants(profile: str, scale: float, cap: float) -> None:
    guidance = _guidance(profile)
    assert (guidance.as_record()["scale_factor"], guidance.max_update_ratio) == (scale, cap)


@pytest.mark.parametrize(("words", "categories", "modifiers", "entities", "reason"), [
    (["two", "red", "cars"], ["N/N", "N/N", "N"], (1, 2), (3,), "count_semantics_not_yet_supported"),
    (["red", "car", "behind", "bus"], ["N/N", "N", "(NP\\NP)/NP", "N"], (1,), (2, 4),
     "depth_semantics_not_yet_supported"),
    (["red", "dog", "runs"], ["N/N", "N", "S[dcl]\\NP"], (1,), (2,), "action_semantics_not_yet_supported"),
    (["red", "car", "inside", "box"], ["N/N", "N", "(NP\\NP)/NP", "N"], (1,), (2, 4),
     "unsupported_or_mixed_relation_structure"),
])
def test_vetoes_of_the_base_route(words, categories, modifiers, entities, reason) -> None:
    decision, _ = _decide(words, categories, modifiers, entities)
    assert decision.profile == "B0" and reason in decision.reasons


@pytest.mark.parametrize("evidence", [{"parse_status": "partial"}, {"unresolved_modifiers": ["red"]},
                                      {"count_tokens": ["two"]}, {"depth_cues": ["behind"]}])
def test_syntax_count_and_depth_exclusions_survive_the_boost(evidence: dict) -> None:
    result = _boost(_guidance(**evidence), "red")
    assert result.effective_profile == "B0"
    assert result.decision.reasons == ("retain_syntax_count_and_depth_exclusions",)


def test_action_and_uncovered_attachment_do_not_veto_the_boost() -> None:
    decision, _ = _decide(["red", "dog", "runs"], ["N/N", "N", "S[dcl]\\NP"], (1,), (2,))
    assert decision.profile == "B0"
    base = _guidance(action_tokens=["runs"], uncovered_attachment_source_indices=[5])
    assert _boost(base, "red").effective_profile == "P2"


def test_derivation_relations_give_the_spatial_relation_route_and_survive_the_boost() -> None:
    words = ["a", "giraffe", "next", "to", "a", "lamp"]
    deps = ("next_3 ((NP{Y}\\NP{Y}<1>){_}/PP{Z}<2>){_} 1 giraffe_2 0",
            "next_3 ((NP{Y}\\NP{Y}<1>){_}/PP{Z}<2>){_} 2 to_4 0", "to_4 (PP{_}/NP{Y}<1>){_} 1 lamp_6 0")
    decision, relations = _decide(words, ["NP[nb]/N", "N", "(NP\\NP)/PP", "PP/NP", "NP[nb]/N", "N"], (), (2, 6),
                                  deps)
    assert decision.profile == "P1"
    assert [(r.label, r.subject_clip_index, r.object_clip_index) for r in relations] == [("next_to", 2, 6)]
    base = ImageGuidance("routing_v1", decision, "a" * 64, "b" * 64,
                         WholeWordSpans.from_mapping({}, text_token_count=8), relations)
    boosted = _boost(base, "red")
    assert boosted.effective_profile == "P1" and boosted.relations == relations
    assert boosted.as_record()["relation_parameters"]["weight"] == 6.0
    with pytest.raises(ValueError, match="P1 requires"):
        replace(base, derived_relations=())
    assert _decide(words, ["NP[nb]/N", "N", "(NP\\NP)/PP", "PP/NP", "NP[nb]/N", "N"], (), (2, 6), deps,
                   parse_status="partial")[0].profile == "B0"


def test_derive_image_guidance_golden() -> None:
    prompt = "a red box and a blue vase"
    words = prompt.split()
    sentence = synthetic_sentence(prompt, ["NP[nb]/N", "N/N", "N", "conj", "NP[nb]/N", "N/N", "N"])
    deps = marked_dependencies("red_2 (N{Y}/N{Y}<1>){_} 1 box_3 0", "blue_6 (N{Y}/N{Y}<1>){_} 1 vase_7 0")
    alignment = _alignment(words, (2, 6))
    for row in alignment:
        if row["text"] in {"a", "and"}:
            row["structural_role"] = "out_of_contract"
    indices = PDIndices(exist=[3, 7], corr=[[2, 3], [6, 7]], leak=[[2, 7], [6, 3]])
    guidance, count = derive_image_guidance(
        prompt, sentence, deps, alignment, indices, parse_status="complete", unresolved_modifiers=[],
        tokenizer=WordTokenizer(), noun_forms=NOUN_FORMS, compound_lexicon=LEXICON)
    assert guidance.effective_profile == "P2" and guidance.active_pair_spans is not None
    assert count.decision.status == "no_explicit_cardinal" and count.enabled
    original = json.loads(guidance.decision.evidence_json)["original_decision"]
    assert original["profile"] == "P2"
    digest = hashlib.sha256(canonical_json([guidance.as_record(), count.as_record()]).encode()).hexdigest()
    assert digest == GOLDEN_IMAGE_GUIDANCE_SHA256


# Canonical JSON of the two records above (a synthetic input); the records of the evaluated runs are
# compared with recompiled plans in test_frozen_compositional_plans.py.
GOLDEN_IMAGE_GUIDANCE_SHA256 = "a22fa53201e0047dce77cf37cf02544fcf19d6647acd501608640ba9a32d0e4c"
