"""Disambiguating Paraphrase: object cleft / PP fronting and guidance-index remapping."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
from conftest import PieceClipTokenizer, auto_leaf, marked_dependencies

from ccg_diffusion.config import SD14_GENERATION, TAB_PARAPHRASE_RULE
from ccg_diffusion.parsing.ccg import auto_sentence
from ccg_diffusion.pp_attachment import disambiguating_paraphrase as paraphrase
from ccg_diffusion.provenance import canonical_json, digest, sha256_text
from ccg_diffusion.targets.guidance_targets import derive_guidance_targets
from ccg_diffusion.targets.pd_indices import program_sha256

FIXTURES = Path(__file__).parent / "fixtures"


def _tree(category: str, *children: str, head: int = 0) -> str:
    return f"(<T {category} {head} {len(children)}> {' '.join(children)})"


def _np(phrase: str) -> str:
    words = phrase.split()
    noun = auto_leaf("N", words[-1])
    determiner = len(words) > 1 and words[0].lower() in {"the", "a", "an", "her"}
    for adjective in reversed(words[int(determiner):-1]):
        noun = _tree("N", auto_leaf("N/N", adjective), noun, head=1)
    if determiner:
        return _tree("NP", auto_leaf("NP[nb]/N", words[0]), noun, head=1)
    return _tree("NP", noun)


def _inputs(prompt: str, sentence: Any, lines: list[str]) -> dict[str, Any]:
    """Guidance targets with attachment possession of one interpretation, in the paraphrase's input form."""
    compiled = derive_guidance_targets(prompt, sentence, marked_dependencies(*lines), PieceClipTokenizer(),
                                       attachment_possession=True)
    target_row = {"prompt_id": "paraphrase-test_s0", "prompt_sha256": sha256_text(prompt),
                  "program_sha256": program_sha256(compiled.indices), "alignment": list(compiled.alignment),
                  "pd_indices": asdict(compiled.indices)}
    return {"prompt": prompt, "target_row": target_row, "sentence": sentence,
            "conversion_trace_row": {"conversion_trace": compiled.trace,
                                     "conversion_trace_sha256": digest(compiled.trace)}}


def _case(reading: str, *, site_word: str = "with", matrix_object: str = "the boy",
          site_object: str = "a yellow shirt", spatial: bool = False) -> dict[str, Any]:
    """A clause "The girl <verb> [next to] <object> <site> <PP object>" parsed in one interpretation."""
    verb = "stands" if spatial else "leaves"
    vp_category = "S[dcl]\\NP"
    matrix_category = vp_category if spatial else f"({vp_category})/NP"
    prefix = "The girl " + verb + (" next to " if spatial else " ") + matrix_object
    prompt = prefix + " " + site_word + " " + site_object
    words = prompt.split()
    site = len(prefix.split()) + 1
    holder = site - 1
    object_ = len(words)
    site_category = "(NP\\NP)/NP" if reading == "np_attach" else "((S\\NP)\\(S\\NP))/NP"
    suffix_category = "NP\\NP" if reading == "np_attach" else "(S\\NP)\\(S\\NP)"
    suffix = _tree(suffix_category, auto_leaf(site_category, site_word), _np(site_object))
    matrix_object_node = _tree("NP", _np(matrix_object), suffix) if reading == "np_attach" else _np(matrix_object)
    if spatial:
        next_node = _tree("(S\\NP)\\(S\\NP)", auto_leaf("((S\\NP)\\(S\\NP))/PP", "next"),
                          _tree("PP", auto_leaf("PP/NP", "to"), matrix_object_node))
        matrix = _tree(vp_category, auto_leaf(matrix_category, verb), next_node)
    else:
        matrix = _tree(vp_category, auto_leaf(matrix_category, verb), matrix_object_node)
    if reading == "vp_attach":
        matrix = _tree(vp_category, matrix, suffix)
    sentence = auto_sentence("ID=paraphrase-test", _tree("S[dcl]", _np("The girl"), matrix, head=1))
    verb_marked = "(S[dcl]{_}\\NP{Y}<1>){_}"
    if not spatial:
        verb_marked = f"({verb_marked}/NP{{Z}}<2>){{_}}"
    lines = [f"{verb}_3 {verb_marked} 1 girl_2 0"]
    if spatial:
        next_marked = "(((S[X]{Y}\\NP{Z}){Y}\\(S[X]{Y}<1>\\NP{Z}){Y}){_}/PP{W}<2>){_}"
        lines += [f"next_4 {next_marked} 1 {verb}_3 0", f"next_4 {next_marked} 2 to_5 0",
                  f"to_5 (PP{{_}}/NP{{Y}}<1>){{_}} 1 {words[holder - 1]}_{holder} 0"]
    else:
        lines.append(f"{verb}_3 {verb_marked} 2 {words[holder - 1]}_{holder} 0")
    attachment = ("((NP{Y}\\NP{Y}<1>){_}/NP{Z}<2>){_}" if reading == "np_attach" else
                  "(((S[X]{Y}\\NP{Z}){Y}\\(S[X]{Y}<1>\\NP{Z}){Y}){_}/NP{W}<2>){_}")
    modified = holder if reading == "np_attach" else 3
    lines += [f"{site_word}_{site} {attachment} 1 {words[modified - 1]}_{modified} 0",
              f"{site_word}_{site} {attachment} 2 {words[object_ - 1]}_{object_} 0"]
    for leaf in sentence.leaves:
        if leaf.category == "N/N":
            target = holder if leaf.source_index < site else object_
            lines.append(f"{leaf.word}_{leaf.source_index} (N{{Y}}/N{{Y}}<1>){{_}} 1 {words[target - 1]}_{target} 0")
    return {**_inputs(prompt, sentence, lines), "site_source_index": site, "tokenizer": PieceClipTokenizer()}


@pytest.fixture
def unpinned_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paraphrase, "require_sd14_tokenizer", lambda _tokenizer: SD14_GENERATION.tokenizer_identity())


def _assert_exact_remapping(source: dict[str, Any], result: paraphrase.Paraphrase) -> None:
    rows = result.trace["source_mapping"]
    assert sorted(row["source_index"] for row in rows) == list(range(1, len(source["sentence"].leaves) + 1))
    inverse = {row["conditioning_clip_index"]: row["source_clip_index"] for row in rows}
    actual, expected = asdict(result.indices), source["target_row"]["pd_indices"]
    assert sorted(inverse[index] for index in actual["exist"]) == sorted(expected["exist"])
    for field in ("corr", "leak", "possession"):
        assert sorted([[inverse[left], inverse[right]] for left, right in actual[field]]) == sorted(expected[field])
    for row in rows:
        start, end = row["conditioning_char_span"]
        assert result.prompt[start:end] == row["conditioning_text"]
        assert row["conditioning_text"].casefold() == row["source_text"].casefold()


@pytest.mark.parametrize(("reading", "site", "site_object", "expected"), [
    ("np_attach", "with", "a yellow shirt", "It is the boy with a yellow shirt that the girl leaves."),
    ("vp_attach", "with", "a yellow shirt", "With a yellow shirt, the girl leaves the boy."),
    ("np_attach", "in", "her pajamas", "It is the boy in her pajamas that the girl leaves."),
    ("vp_attach", "in", "her pajamas", "In her pajamas, the girl leaves the boy."),
])
def test_cleft_and_fronting_keep_every_source_word_and_guidance_target(
    unpinned_tokenizer: None, reading: str, site: str, site_object: str, expected: str,
) -> None:
    source = _case(reading, site_word=site, site_object=site_object)
    before = canonical_json([source["target_row"], source["conversion_trace_row"]])
    result = paraphrase.paraphrase_interpretation(**source)
    assert result.prompt == expected
    assert result.trace["applicable"] is True and result.trace["rule"] == TAB_PARAPHRASE_RULE
    assert result.trace["realization"] == ("pp_object_np_cleft" if reading == "np_attach" else "pp_event_fronting")
    assert result.program_sha256 == program_sha256(result.indices)
    _assert_exact_remapping(source, result)
    assert canonical_json([source["target_row"], source["conversion_trace_row"]]) == before


def test_cleft_moves_the_whole_modified_np_and_keeps_the_relation_preposition(unpinned_tokenizer: None) -> None:
    source = _case("np_attach", matrix_object="the yellow cage", site_object="a cat", spatial=True)
    result = paraphrase.paraphrase_interpretation(**source)
    assert result.prompt == "It is the yellow cage with a cat that the girl stands next to."
    assert result.trace["holder_np_source_span"] == [6, 8]
    _assert_exact_remapping(source, result)


def test_unsupported_site_keeps_the_source_prompt_and_targets(unpinned_tokenizer: None) -> None:
    source = _case("np_attach", site_word="wearing")
    result = paraphrase.paraphrase_interpretation(**source)
    assert result.prompt == source["prompt"]
    assert result.trace["applicable"] is False and result.trace["reason"] == "unsupported_site_lexical_form"
    assert asdict(result.indices) == source["target_row"]["pd_indices"] and result.trace["indices_changed"] is False


@pytest.mark.parametrize("index", [0, 1])
def test_progressive_clause_is_outside_the_rule(unpinned_tokenizer: None, index: int) -> None:
    data = json.loads((FIXTURES / "tab_progressive_standing.json").read_text())[index]
    sentence = auto_sentence(data["auto_id"], data["auto"])
    source = _inputs(data["prompt"], sentence, data["dependencies"])
    result = paraphrase.paraphrase_interpretation(
        **source, site_source_index=data["site_source_index"], tokenizer=PieceClipTokenizer())
    assert result.prompt == data["prompt"] and result.trace["applicable"] is False
    assert result.trace["reason"] == "unsupported_matrix_predicate_tense"


def test_source_job_records_the_default_tab_strength(unpinned_tokenizer: None) -> None:
    source = _case("vp_attach")
    result = paraphrase.paraphrase_interpretation(**source)
    job = paraphrase.source_job(tab_id="tab-pp-x", reading="vp_attach", paraphrase=result,
                                source_program_sha256="0" * 64, tokenizer=PieceClipTokenizer(),
                                evaluation={"target_setup_index": 1})
    assert job["prompt"] == result.prompt and job["relations"] == [] and job["target_setup_index"] == 1
    assert job["loss_config"]["scale_factor"] == 10.0 and job["loss_config"]["max_update_ratio"] == 0.02
    assert job["loss_config"]["holder_aware_weight"] == 0.0 and job["pd_indices"] == asdict(result.indices)


@pytest.mark.requires_clip_tokenizer
@pytest.mark.requires_spacy_model
def test_paper_parses_reproduce_the_bundle_paraphrases() -> None:
    """The recorded constrained parses of tab-pp-000/021/037 give the bundle's paraphrase traces byte for byte."""
    from ccg_diffusion.guidance.count_loss import load_noun_forms
    from ccg_diffusion.pp_attachment.compile import interpretation_from_parse
    from ccg_diffusion.pp_attachment.jobs import load_frozen_bundle
    from ccg_diffusion.pp_attachment.supertag_constrained_parsing import PrepositionSite
    from ccg_diffusion.targets.guidance_targets import load_possession_lemmatizer
    from ccg_diffusion.targets.tokenizer import load_sd14_tokenizer

    bundle = load_frozen_bundle()
    rows = {(r["tab_id"], r["reading"]): r for r in bundle.interpretation_rows}
    tokenizer, nlp, forms = load_sd14_tokenizer(), load_possession_lemmatizer(), load_noun_forms()
    for parse in json.loads((FIXTURES / "tab_pp_parses.json").read_text()):
        frozen = rows[parse["tab_id"], parse["reading"]]
        site = PrepositionSite(**{**frozen["preposition"], "frame_repair": None})
        template = bundle.interpretations[parse["tab_id"], parse["reading"]].source_job
        evaluation = {k: template[k] for k in ("tab_category", "tab_requires_reading", "tab_evaluation_mode",
                                               "target_setup_index", "evaluation_setup_indices",
                                               "target_setup_mapping_evidence_sha256")}
        interpretation = interpretation_from_parse(
            tab_id=parse["tab_id"], reading=parse["reading"], prompt=frozen["prompt"],
            item_index=parse["item_index"], site=site,
            sentence=auto_sentence(parse["id_line"], parse["derivation"]),
            dependencies=marked_dependencies(*parse["dependencies"]), tokenizer=tokenizer, nlp=nlp,
            noun_forms=forms, evaluation=evaluation)
        assert canonical_json(interpretation.paraphrase) == canonical_json(frozen["paraphrase"])
        assert interpretation.targets.pd_indices == frozen["guidance_targets"]["pd_indices"]
        assert interpretation.targets.boost == frozen["guidance_targets"]["boost"]
        assert list(interpretation.targets.spatial_relations) == frozen["guidance_targets"]["spatial_relations"]
        assert interpretation.standard_program == frozen["standard_program"]
        assert {k: interpretation.source_job[k] for k in template if k in interpretation.source_job} == {
            k: template[k] for k in template if k in interpretation.source_job}
