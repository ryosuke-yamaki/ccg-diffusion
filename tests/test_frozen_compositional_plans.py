"""Frozen T2I-CompBench++ plans: recompiled plans equal the records of the evaluated runs.

``fixtures/t2i_frozen_plans.json`` holds, for twelve prompts (per category a
prompt with and one without the Positional Prefix, the prompts of the GPU
checks, a depth and a clause screening case), the recorded Stage 1 AUTO line,
Stage 2 marked dependencies and Stage 3 guidance targets, the evaluated SD1.4
decision records (route, count, plan) and the evaluated Sana final_shape plan
row; ``sources`` names every source file with its sha256. Sana plan fields
that repeat other fixture fields are stored once (``SANA_COPIES``).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from conftest import FakeNLP
from ccg_diffusion import single_prompt
from ccg_diffusion.compositional import sd14
from ccg_diffusion.compositional.ablation import apply_ablation
from ccg_diffusion.compositional.plans import evaluated_operation_projection, operation_projection, plan_prompt
from ccg_diffusion.guidance.region_assignment_loss import in_source_coordinates
from ccg_diffusion.methods import EVALUATED_T2I_CONDITIONS, T2IAblation
from ccg_diffusion.parsing.ccg import auto_sentence, parse_marked_dependency
from ccg_diffusion.parsing.frontend import FrontEndResult
from ccg_diffusion.parsing.sentences import split_and_tokenize
from ccg_diffusion.provenance import read_jsonl
from ccg_diffusion.targets.guidance_targets import CompiledGuidanceTargets
from ccg_diffusion.targets.pd_indices import PDIndices

FIXTURE = json.loads((Path(__file__).parent / "fixtures/t2i_frozen_plans.json").read_text())
PROMPTS = {item["prompt_id"]: item for item in FIXTURE["prompts"]}
SANA_COPIES = ("prompt_id", "prompt", "prompt_sha256", "conditioning_sha256", "source_program",
               "source_program_sha256", "source_indices", "source_alignment", "source_guidance", "source_count",
               "source_prefix")


@pytest.fixture(scope="module")
def tokenizer() -> Any:
    from ccg_diffusion.targets.tokenizer import load_sd14_tokenizer

    return load_sd14_tokenizer()


@pytest.fixture(scope="module")
def gemma_tokenizer() -> Any:
    from ccg_diffusion.backbones.sana.pipeline import load_sana_tokenizer

    return load_sana_tokenizer()


def _inputs(item: dict[str, Any]) -> tuple[Any, list[Any], CompiledGuidanceTargets]:
    sentence = auto_sentence(item["auto_id"], item["derivation"])
    dependencies = [parse_marked_dependency(line) for line in item["dependencies"]]
    targets = CompiledGuidanceTargets(PDIndices(**item["targets"]["pd_indices"]), tuple(item["targets"]["alignment"]),
                                      item["trace"])
    return sentence, dependencies, targets


def _evaluated_row(item: dict[str, Any]) -> dict[str, Any]:
    return {"guidance": item["evaluated"]["guidance"], "count": item["evaluated"]["count"],
            "plans": {"final_shape": item["evaluated"]["plan"]}}


def test_fixture_covers_every_route_and_plan_branch() -> None:
    plans = [item["evaluated"]["plan"]["source_plan"] for item in PROMPTS.values()]
    assert {item["evaluated"]["guidance"]["effective_profile"] for item in PROMPTS.values()} == {"B0", "P1", "P2"}
    assert {plan["prefix"]["source_branch"] for plan in plans} == {"phrase", "depth", "clause", "unchanged"}
    assert any(item["evaluated"]["count"]["decision"]["constraints"] for item in PROMPTS.values())
    assert {"t2icb_color_0265", "t2icb_shape_0002", "t2icb_spatial_0299", "t2icb_numeracy_0299"} <= set(PROMPTS)


@pytest.mark.requires_clip_tokenizer
@pytest.mark.parametrize("prompt_id", sorted(PROMPTS))
def test_plan_prompt_reproduces_the_evaluated_sd14_records(prompt_id: str, tokenizer: Any) -> None:
    item = PROMPTS[prompt_id]
    sentence, dependencies, targets = _inputs(item)
    planned = plan_prompt(item["prompt"], sentence, dependencies, targets, tokenizer)
    evaluated = item["evaluated"]
    assert planned.guidance.as_record() == evaluated["guidance"]
    assert planned.count.as_record() == evaluated["count"]
    for name in ("phrase", "depth", "clause", "prefix"):
        assert getattr(planned.plan, name).as_record() == evaluated["plan"]["source_plan"][name]
    assert planned.plan.positional_prefix == evaluated["plan"]["active"]
    projection = operation_projection(planned.plan.as_record(), planned.guidance.as_record(), planned.count.as_record())
    assert projection == evaluated_operation_projection(_evaluated_row(item), record_format="shape_v2")


@pytest.mark.requires_clip_tokenizer
def test_ablations_change_only_the_executed_operations(tokenizer: Any) -> None:
    item = PROMPTS["t2icb_color_0265"]  # boosted (P2) with the Positional Prefix
    sentence, dependencies, targets = _inputs(item)
    planned = plan_prompt(item["prompt"], sentence, dependencies, targets, tokenizer)
    plan = planned.plan.as_record()
    program = item["targets"]["pd_indices"]
    source_groups = in_source_coordinates(planned.plan.prefix, program).groups
    for condition_id in EVALUATED_T2I_CONDITIONS[:8]:
        ablation = T2IAblation.from_condition_id(condition_id)
        guidance, count = apply_ablation(ablation, {"k": planned.guidance}, {"k": planned.count})
        projection = operation_projection(plan, guidance["k"].as_record(), count["k"].as_record(), ablation)
        assert projection["route"] == ("P2" if ablation.selective_guidance_boost else "B0")
        assert projection["whole_word_pooling"] is ablation.selective_guidance_boost
        assert (projection["positional_prefix"] is not None) is ablation.positional_prefix
        assert (projection["region_assignment"] is not None) is ablation.region_assignment_loss
        if ablation.region_assignment_loss and not ablation.positional_prefix:
            # Region-only: the same word groups and sides, located on the unaltered prompt.
            groups = [[list(word) for word in group] for group in source_groups]
            assert projection["region_assignment"]["groups"] == groups
        evaluated = evaluated_operation_projection(
            {**_evaluated_row(item), "guidance": guidance["k"].as_record()}, record_format="shape_v2",
            ablation=ablation)
        assert projection == evaluated


@pytest.mark.requires_clip_tokenizer
@pytest.mark.requires_gemma_tokenizer
@pytest.mark.parametrize("prompt_id", sorted(PROMPTS))
def test_sana_plans_reproduce_the_evaluated_final_shape_rows(prompt_id: str, tokenizer: Any,
                                                             gemma_tokenizer: Any) -> None:
    from ccg_diffusion.backbones.sana.pipeline import COMPLEX_HUMAN_INSTRUCTION
    from ccg_diffusion.compositional.sana import compile_plan

    item = PROMPTS[prompt_id]
    sentence, dependencies, targets = _inputs(item)
    plan = compile_plan(prompt_id=prompt_id, prompt=item["prompt"], sentence=sentence, dependencies=dependencies,
                        compiled=targets, clip_tokenizer=tokenizer, gemma_tokenizer=gemma_tokenizer,
                        instruction=COMPLEX_HUMAN_INSTRUCTION)
    evaluated = item["evaluated"]
    copies = {"prompt_id": prompt_id, "prompt": item["prompt"], "prompt_sha256": item["targets"]["prompt_sha256"],
              "conditioning_sha256": hashlib.sha256(item["sana_plan"]["conditioning"].encode()).hexdigest(),
              "source_program": item["targets"]["pd_indices"],
              "source_program_sha256": item["targets"]["program_sha256"],
              "source_indices": item["trace"]["source_indices"], "source_alignment": item["targets"]["alignment"],
              "source_guidance": evaluated["guidance"], "source_count": evaluated["count"],
              "source_prefix": evaluated["plan"]["source_plan"]["prefix"]}
    assert set(copies) == set(SANA_COPIES)
    assert json.dumps(plan, sort_keys=True) == json.dumps({**item["sana_plan"], **copies}, sort_keys=True)


def test_ablation_populations_are_whole_chunks_of_the_scored_categories() -> None:
    canonical = read_jsonl(sd14.DEFAULT_IMAGE_MANIFEST)
    expected = {"gs0_p1_r1": "26d204b33dc48aa5b3ac8316ce78428d8db577c6287b4b95761e54b21a9e32f8",
                "no_2d": "782210f9ccd21c388dcbdc1a706d9847e42fde1ad324274d134a5a492311d966",
                "no_count": "80eb1462860f35c3f7369efa90006d6c228520e21bf5ac1bdc54e8e8b058c7a0"}
    for condition_id, digest in expected.items():
        scored, generated = sd14.chunk_closure(canonical, T2IAblation.from_condition_id(condition_id).scored_categories)
        assert (len(scored), len(generated)) == ((9000, 9008) if condition_id.startswith("gs") else (3000, 3008))
        assert hashlib.sha256(json.dumps(generated, separators=(",", ":")).encode()).hexdigest() == digest
        assert len(generated) % sd14.CHUNK_ROWS == 0 and set(scored) <= set(generated)


@pytest.mark.parametrize("command", ["prepare", "generate", "audit", "prompt"])
def test_every_subcommand_help_formats(command: str) -> None:
    from scripts.t2i_compbench import run_ccg_diffusion

    with pytest.raises(SystemExit) as stop:
        run_ccg_diffusion.main([command, "--help"])
    assert stop.value.code == 0


# --- single_prompt with a fake front end -------------------------------------------------------------------


class FakeFrontEnd:
    """Returns the recorded parse of one fixture prompt, or a failed parse."""

    def __init__(self, item: dict[str, Any] | None, prompt: str) -> None:
        self.item, self.prompt = item, prompt

    def parse(self, prompt: str, *, sentence_id: str) -> FrontEndResult:
        assert prompt == self.prompt and sentence_id == single_prompt.SENTENCE_ID
        tokens = tuple(token.text for token in split_and_tokenize(prompt)[0].tokens)
        if self.item is None:
            id_line = f"ID={sentence_id} PARSER=HOLCCG NUMPARSE=1"
            derivation = f"(<T FAIL 0 {len(tokens)}> " + " ".join(f"(<L POS POS POS {t} POS>)" for t in tokens) + ")"
            dependencies, status, root = (), "fail", "FAIL"
        else:
            id_line, derivation = self.item["auto_id"], self.item["derivation"]
            dependencies = tuple(parse_marked_dependency(line) for line in self.item["dependencies"])
            status, root = "complete", auto_sentence(id_line, derivation).root.category
        coverage = {"sentence_index": 1, "auto_id": id_line, "root_category": root, "parse_status": status,
                    "all_generate_units_success": self.item is not None,
                    "any_generate_unit_success": self.item is not None, "prompt": " ".join(tokens)}
        return FrontEndResult(tokens=tokens, id_line=id_line, derivation=derivation,
                              sentence=auto_sentence(id_line, derivation), dependencies=dependencies,
                              dependency_lines=tuple(self.item["dependencies"]) if self.item else (),
                              coverage=coverage, parse_record_status=status)


def _fake_backend(monkeypatch: pytest.MonkeyPatch, calls: list) -> None:
    monkeypatch.setattr(single_prompt, "load_backbone", lambda backbone, **kwargs: f"{backbone} model")
    monkeypatch.setattr(single_prompt, "generate_sd14", lambda *args, **kwargs: (
        calls.append(("sd14", args)) or "guided image", {"kernel": "predicated_stable_batch"}))
    monkeypatch.setattr(single_prompt, "generate_stock", lambda *args, **kwargs: (
        calls.append(("stock", args)) or "stock image", {"kernel": "stock"}))


@pytest.mark.requires_clip_tokenizer
def test_single_prompt_plan_path_equals_the_evaluated_records(monkeypatch: pytest.MonkeyPatch) -> None:
    item = PROMPTS["t2icb_color_0265"]
    lemmas = item["trace"]["possession_lemma_annotations"]
    assert lemmas == []
    monkeypatch.setattr("ccg_diffusion.targets.guidance_targets.load_possession_lemmatizer",
                        lambda model="en_core_web_sm": FakeNLP([]))
    calls: list = []
    _fake_backend(monkeypatch, calls)
    result = single_prompt.generate(item["prompt"], frontend=FakeFrontEnd(item, item["prompt"]), seed=42)
    record = result.record
    assert result.image == "guided image" and record["paper_exact"] is False and record["guidance"] == "method_guided"
    assert record["guidance_targets"]["pd_indices"] == item["targets"]["pd_indices"]
    assert record["plan"]["guidance"] == item["evaluated"]["guidance"]
    assert record["plan"]["count"] == item["evaluated"]["count"]
    assert record["plan"]["plan"]["prefix"] == item["evaluated"]["plan"]["source_plan"]["prefix"]
    assert [name for name, _ in calls] == ["sd14"]
    removed = single_prompt.generate(item["prompt"], frontend=FakeFrontEnd(item, item["prompt"]),
                                     remove=["selective_guidance_boost"])
    assert removed.record["plan"]["guidance"]["effective_profile"] == "B0"
    assert removed.record["ablation"] == {"guidance_strength": False, "prefix": True, "region": True}


@pytest.mark.requires_clip_tokenizer
def test_single_prompt_falls_back_to_stock_without_guidance_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    prompt = "a fluffy cat"
    monkeypatch.setattr("ccg_diffusion.targets.guidance_targets.load_possession_lemmatizer",
                        lambda model="en_core_web_sm": FakeNLP([]))
    calls: list = []
    _fake_backend(monkeypatch, calls)
    result = single_prompt.generate(prompt, backbone="sana", frontend=FakeFrontEnd(None, prompt))
    assert result.image == "stock image" and [name for name, _ in calls] == ["stock"]
    assert result.record["guidance"] == "stock_vanilla_fallback"
    assert result.record["fallback_reason"] == "stage1_parse_failure" and result.record["plan"] is None
    with pytest.raises(ValueError, match="exactly one sentence"):
        single_prompt.generate("A cat. A dog.", frontend=FakeFrontEnd(None, "A cat. A dog."))
