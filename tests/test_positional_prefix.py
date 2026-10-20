"""Positional Prefix: eligibility, prefix text, token remapping, the K/V hook and the record-only depth/clause
screening."""
from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType
from typing import Any

import pytest
import torch

from conftest import (
    FakeSDPipe,
    WordTokenizer,
    one_token_alignment,
    phrase_plan_of,
    prefix_guidance_record,
    prefix_source,
    screening_stubs,
)
from ccg_diffusion.guidance._record_only_screening import screen_layouts
from ccg_diffusion.guidance.count_loss import NounForms
from ccg_diffusion.guidance.positional_prefix import (
    APPLIED_SELECTION,
    SHAPE_CONJUNCTION_REASON,
    UNSELECTED,
    PrefixLayoutPlan,
    ProjectionContextCache,
    build_phrase_plan,
    build_prefix_plan,
    prefix_applies,
    prefix_conditioning,
    selection_record,
)
from ccg_diffusion.parsing.ccg import AutoLeaf, AutoSentence, MarkedDependency
from ccg_diffusion.provenance import canonical_json, sha256_text

NOUN_FORMS = NounForms(frozenset({"dog", "cat", "front", "box", "vase"}), (("s", ""),), MappingProxyType({}))


def prefix_of(args: dict[str, Any], phrase=None) -> PrefixLayoutPlan:
    depth, clause = screening_stubs(args)
    return build_prefix_plan(args["prompt"], args["alignment"], args["program"], phrase or phrase_plan_of(args),
                             WordTokenizer(), depth=depth, clause=clause)


def test_prefix_text_mapping_and_groups() -> None:
    args = prefix_source()
    plan = prefix_of(args)
    assert plan.conditioning == "On the left side, a blue box. On the right side, a red vase. " + args["prompt"]
    assert prefix_applies(plan) and plan.relation == "left" and plan.reason == "permitted_conjunction_prefix"
    mapping = dict(plan.mapping)
    old, new = (WordTokenizer()(text)["input_ids"] for text in (args["prompt"], plan.conditioning))
    assert set(mapping) == set(range(1, len(old) - 1)) and len(set(mapping.values())) == len(mapping)
    assert all(old[i] == new[j] for i, j in mapping.items())
    assert plan.mapped_program == {"exist": [mapping[3], mapping[7]],
                                   "corr": [[mapping[2], mapping[3]], [mapping[6], mapping[7]]],
                                   "leak": [[mapping[2], mapping[7]], [mapping[6], mapping[3]]], "possession": []}
    assert plan.groups == (((mapping[3],), (mapping[2],)), ((mapping[7],), (mapping[6],)))
    assert selection_record(plan) == APPLIED_SELECTION
    assert PrefixLayoutPlan.from_record(plan.as_record()) == plan


@pytest.mark.parametrize(("first", "second", "reason"), [
    ("blue", "red", "two_color_or_texture_nominals_in_conjunction"),
    ("wooden", "leather", "two_color_or_texture_nominals_in_conjunction"),
    ("round", "red", SHAPE_CONJUNCTION_REASON), ("small", "wooden", SHAPE_CONJUNCTION_REASON),
    ("square", "round", SHAPE_CONJUNCTION_REASON),
])
def test_color_texture_shape_and_size_conjunctions_qualify(first: str, second: str, reason: str) -> None:
    args = prefix_source(first, second)
    plan = phrase_plan_of(args)
    assert plan.reason == reason and [p.text for p in plan.phrases] == [f"a {first} box", f"a {second} vase"]
    assert prefix_of(args, plan).conditioning.startswith(f"On the left side, a {first} box. ")


@pytest.mark.parametrize("guard", ["count", "count_cue", "P1", "partial", "unresolved", "depth", "or", "relation",
                                   "ambiguous"])
def test_guards_keep_the_prompt_unprefixed(guard: str) -> None:
    args = prefix_source(connection={"or": "or", "relation": "above"}.get(guard, "and"))
    evidence = args["guidance"]["decision"]["evidence"]["original_decision"]["evidence"]
    if guard == "count":
        args["count"]["decision"]["constraints"] = [{"count": 2}]
    elif guard == "count_cue":
        evidence["count_tokens"] = ["two"]
    elif guard == "P1":
        args["guidance"]["effective_profile"] = "P1"
    elif guard == "partial":
        evidence["parse_status"] = "partial"
    elif guard == "unresolved":
        evidence["unresolved_modifiers"] = ["blue"]
    elif guard == "depth":
        evidence["depth_cues"] = ["behind"]
    elif guard == "ambiguous":
        args["alignment"][4]["clip_index"] = 3
    plan = prefix_of(args)
    assert not prefix_applies(plan) and plan.conditioning is None and selection_record(plan) == UNSELECTED
    if guard == "relation":
        assert phrase_plan_of(args).reason == "two_visual_nominals_with_relation"
        assert plan.reason == "retain_other_relational_input"


def test_source_identity_mismatch_is_rejected() -> None:
    args = prefix_source()
    with pytest.raises(ValueError, match="source differs"):
        build_phrase_plan(args["prompt"] + " altered", args["alignment"], args["program"], args["guidance"],
                          args["count"], WordTokenizer())
    depth, clause = screening_stubs(args)
    with pytest.raises(ValueError, match="different source"):
        build_prefix_plan(args["prompt"], args["alignment"], args["program"], phrase_plan_of(args), WordTokenizer(),
                          depth=replace(depth, prompt_sha256="0" * 64), clause=clause)


def _depth_source() -> dict[str, Any]:
    words = "a dog behind a cat".split()
    categories = ["N/N", "N", "(NP\\NP)/NP", "N/N", "N"]
    leaves = tuple(AutoLeaf(i, category, word, (), 0) for i, (word, category) in enumerate(zip(words, categories), 1))
    dependencies = [MarkedDependency(3, "behind", categories[2], 1, 2, "dog", 0),
                    MarkedDependency(3, "behind", categories[2], 2, 5, "cat", 0)]
    prompt = " ".join(words)
    program = {"exist": [2, 5], "corr": [], "leak": [], "possession": []}
    return {"prompt": prompt, "sentence": AutoSentence("ID=test.1", "", None, leaves), "dependencies": dependencies,
            "alignment": one_token_alignment(words, {i: "entity" for i in range(1, 6)}), "program": program,
            "guidance": prefix_guidance_record(prompt, program, "B0"), "count": {"decision": {"constraints": []}}}


def test_depth_screening_is_record_only() -> None:
    args = _depth_source()
    depth, clause = screen_layouts(args["prompt"], args["sentence"], args["dependencies"], args["alignment"],
                                   args["program"], args["guidance"], args["count"], WordTokenizer(), NOUN_FORMS)
    assert depth.active and depth.relation == "behind" and not clause.active
    phrase = phrase_plan_of(args)
    assert phrase.reason == "fewer_than_two_visual_nominals"
    screened = build_prefix_plan(args["prompt"], args["alignment"], args["program"], phrase, WordTokenizer(),
                                 depth=depth, clause=clause)
    assert screened.conditioning is not None and screened.source_branch == "depth" and not prefix_applies(screened)
    assert screened.source_plan_sha256 == sha256_text(canonical_json(
        {"depth": depth.as_record(), "clause": clause.as_record(), "phrase": phrase.as_record()}))
    assert not prefix_applies(prefix_of(args, phrase))


def test_prefix_conditioning_replaces_only_active_conditional_rows() -> None:
    args = prefix_source()
    plan = prefix_of(args)
    pipe = FakeSDPipe(torch.bfloat16)
    context = torch.full((4, 77, 4), -100.0, dtype=torch.bfloat16)
    with prefix_conditioning(pipe, [args["prompt"]] * 2, [plan, replace(plan, conditioning=None)],
                             device="cpu") as audit:
        result = pipe.unet.attn2.to_k(context)
        assert torch.equal(result[[0, 1, 3]], context[[0, 1, 3]]) and not torch.equal(result[2], context[2])
        assert torch.equal(pipe.unet.attn2.to_v(context), result)
    assert audit["active_rows"] == [0] and audit["projection_calls"] == 2 and pipe.calls == 1
    assert audit["context_cache_hits"] == 1 and audit["context_cache_verifications"] == 1
    assert torch.equal(pipe.unet.attn2.to_k(context), context)
    with pytest.raises(ValueError, match="identity"):
        with prefix_conditioning(pipe, ["other prompt"], [plan], device="cpu"):
            pass


def test_projection_cache_verifies_the_first_hit() -> None:
    cache = ProjectionContextCache()
    context = torch.arange(4 * 7 * 3).reshape(4, 7, 3).float()

    def modify(x: torch.Tensor) -> torch.Tensor:
        result = x.clone()
        result[2, 1:4] = 1
        return result

    for _ in range(4):
        assert torch.equal(cache.replace(context, modify), modify(context))
    assert (cache.builds, cache.hits, cache.verifications) == (2, 3, 1)
    context.add_(1.0)
    assert torch.equal(cache.replace(context, modify), modify(context)) and cache.builds == 3


@pytest.mark.requires_clip_tokenizer
def test_clip_prefix_of_the_paper_example() -> None:
    from ccg_diffusion.targets.tokenizer import load_sd14_tokenizer

    tokenizer = load_sd14_tokenizer()
    args = prefix_source("blue", "green")
    args["prompt"] = prompt = "a blue banana and a green vase"
    cursor = 0
    for row, word in zip(args["alignment"], prompt.split()):
        row.update(text=word, char_start=cursor, char_end=cursor + len(word))
        cursor += len(word) + 1
    args["guidance"]["source_prompt_sha256"] = sha256_text(prompt)
    assert len(tokenizer(prompt)["input_ids"]) == 9
    phrase = build_phrase_plan(prompt, args["alignment"], args["program"], args["guidance"], args["count"], tokenizer)
    depth, clause = screening_stubs(args)
    plan = build_prefix_plan(prompt, args["alignment"], args["program"], phrase, tokenizer, depth=depth, clause=clause)
    assert plan.conditioning == "On the left side, a blue banana. On the right side, a green vase. " + prompt
    assert plan.mapped_program["exist"] == [8, 17]
