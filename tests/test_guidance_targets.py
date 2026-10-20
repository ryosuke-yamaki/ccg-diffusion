"""Stage 3 guidance targets I_D = (E, C, K, P) and the guidance-target files."""
from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any, Sequence

import pytest
from conftest import FakeNLP, FakeSpacyToken, PieceClipTokenizer, marked_dependencies, synthetic_sentence

from ccg_diffusion.config import SD14_GENERATION
from ccg_diffusion.parsing.ccg import MarkedDepsError
from ccg_diffusion.targets import guidance_targets as targets
from ccg_diffusion.targets.pd_indices import PDIndices, program_sha256

NP_ATTACHMENT = "((NP{Y}\\NP{Y}<1>){_}/NP{Z}<2>){_}"
BANANA_AND_VASE = (
    "a blue banana and a green vase",
    ["NP[nb]/N", "N/N", "N", "conj", "NP[nb]/N", "N/N", "N"],
    [
        "a_1 (NP[nb]{Y}/N{Y}<1>){_} 1 banana_3 0",
        "blue_2 (N{Y}/N{Y}<1>){_} 1 banana_3 0",
        "and_4 conj 1 banana_3 0",
        "and_4 conj 1 vase_7 0",
        "a_5 (NP[nb]{Y}/N{Y}<1>){_} 1 vase_7 0",
        "green_6 (N{Y}/N{Y}<1>){_} 1 vase_7 0",
    ],
)


def _derive(prompt: str, categories: Sequence[str], lines: Sequence[str], **kwargs: Any) -> Any:
    return targets.derive_guidance_targets(
        prompt, synthetic_sentence(prompt, categories), marked_dependencies(*lines), PieceClipTokenizer(), **kwargs)


def test_paper_example_modifier_of_modifier_chains_to_the_noun() -> None:
    compiled = _derive(
        "a light blue banana",
        ["NP[nb]/N", "(N/N)/(N/N)", "N/N", "N"],
        [
            "a_1 (NP[nb]{Y}/N{Y}<1>){_} 1 banana_4 0",
            "light_2 ((N{Y}/N{Y})/(N{Y}/N{Y})<1>){_} 1 blue_3 0",
            "blue_3 (N{Y}/N{Y}<1>){_} 1 banana_4 0",
        ],
    )
    assert compiled.indices == PDIndices(exist=[4], corr=[[2, 4], [3, 4]], leak=[], possession=[])


def test_paper_example_two_objects() -> None:
    compiled = _derive(*BANANA_AND_VASE)
    # E = {banana, vase}; C = {(blue, banana), (green, vase)}; K = {(blue, vase), (green, banana)}; P = {}.
    assert compiled.indices == PDIndices(exist=[3, 7], corr=[[2, 3], [6, 7]], leak=[[2, 7], [6, 3]], possession=[])
    assert len(compiled.trace["leaf_dispositions"]) == 7
    assert len(compiled.trace["dependency_dispositions"]) == 6
    assert compiled.trace["method"] == compiled.trace["profile"] == "deps2index"
    assert "attachment_possession_frames" not in compiled.trace


def test_possession_only_for_the_four_lemmas() -> None:
    case = ("woman holding yoga mat", ["N", "(S[ng]\\NP)/NP", "N/N", "N"], [
        "holding_2 ((S[ng]{Y}\\NP{Z}<1>){Y}/NP{W}<2>){_} 1 woman_1 0",
        "holding_2 ((S[ng]{Y}\\NP{Z}<1>){Y}/NP{W}<2>){_} 2 mat_4 0",
        "yoga_3 (N{Y}/N{Y}<1>){_} 1 mat_4 0",
    ])
    assert _derive(*case).indices.possession == []
    assert _derive(*case, possession_lemmas_by_source={2: "hold"}).indices == PDIndices(
        exist=[1, 4], corr=[[3, 4]], leak=[[3, 1]], possession=[[1, 4]])
    with pytest.raises(MarkedDepsError, match="outside the permitted possession set"):
        _derive(*case, possession_lemmas_by_source={2: "carry"})
    assert targets.POSSESSION_LEMMAS == frozenset({"have", "wear", "grasp", "hold"})
    nlp = FakeNLP([FakeSpacyToken("woman", "woman", 0), FakeSpacyToken("holding", "hold", 6)])
    assert targets.possession_lemma_annotations_from_spacy(case[0], nlp) == {2: "hold"}


def test_no_atomic_noun_is_an_unsupported_prompt() -> None:
    with pytest.raises(targets.UnsupportedPromptProjectionError, match="no atomic-N entity"):
        _derive("he", ["NP"], [])


def test_interpretation_targets_add_attachment_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(targets, "require_sd14_tokenizer", lambda tokenizer: {})
    prompt, categories = "man with hat", ["N", "(NP\\NP)/NP", "N"]
    lines = [f"with_2 {NP_ATTACHMENT} 1 man_1 0", f"with_2 {NP_ATTACHMENT} 2 hat_3 0"]
    args = (prompt, synthetic_sentence(prompt, categories), marked_dependencies(*lines), PieceClipTokenizer())
    interpretation = targets.sd14_interpretation_guidance_targets(*args)
    standard = targets.sd14_guidance_targets(*args)
    assert interpretation.indices == PDIndices(exist=[1, 3], corr=[], leak=[], possession=[[1, 3]])
    assert interpretation.trace["attachment_possession_frames"][0]["pairs"] == [[1, 3]]
    assert interpretation.trace["invariants"]["attachment_possession_rule"] == "attachment_functor_possession"
    assert standard.indices.possession == [] and "attachment_possession_frames" not in standard.trace


def test_program_sha256_golden() -> None:
    assert program_sha256(PDIndices(exist=[1], corr=[[2, 1]], leak=[], possession=[])) == (
        "bcf16040fb1adb087e163ddfe8439811c0214c131cf7b9b8dcbe9fdb7ea7d9f8")


def _target_rows(monkeypatch: pytest.MonkeyPatch, coverage: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    monkeypatch.setattr(targets, "require_sd14_tokenizer", lambda tokenizer: {})
    prompt, categories, lines = BANANA_AND_VASE
    return targets.build_guidance_target_rows(
        [{"experiment": "t2icb_color", "prompt_id": "t2icb_color_0000", "prompt": prompt, "seed": 0}],
        [synthetic_sentence(prompt, categories, root_category="NP")], [marked_dependencies(*lines)],
        PieceClipTokenizer(), FakeNLP([]), category="color", coverage_rows=[coverage],
        tokenizer_identity=SD14_GENERATION.tokenizer_identity())


def _coverage(**changes: Any) -> dict[str, Any]:
    return {"sentence_index": 1, "auto_id": "ID=test.1", "root_category": "NP", "parse_status": "complete",
            "all_generate_units_success": True, "any_generate_unit_success": True,
            "prompt": BANANA_AND_VASE[0], **changes}


def test_guidance_target_rows_have_the_recorded_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    rows, traces = _target_rows(monkeypatch, _coverage())
    row = rows[0]
    assert set(row) == {"prompt_id", "provider", "prompt_sha256", "status", "pd_indices", "program_sha256",
                        "alignment", "parser_provenance"}
    assert (row["provider"], row["status"]) == ("deps2index_sd14", "ok")
    assert row["pd_indices"] == {"exist": [3, 7], "corr": [[2, 3], [6, 7]], "leak": [[2, 7], [6, 3]],
                                 "possession": []}
    assert set(row["parser_provenance"]) == {
        "provider", "method", "canonical_arm_id", "sidecar_provider", "backbone", "tokenizer", "target_dataset",
        "target_category", "artifact_index", "sentence_index", "raw_line_index", "source_token_map", "auto_id",
        "parse_status", "projection_version", "possession_lemma_set", "spacy_syntax_used", "spacy_model",
        "spacy_model_version", "conversion_trace_sha256"}
    assert row["parser_provenance"]["source_token_map"] == "deterministic_sentence_tokenizer"
    assert set(traces[0]) == {"prompt_id", "prompt_sha256", "program_sha256", "conversion_trace_sha256",
                              "conversion_trace"}
    assert traces[0]["conversion_trace_sha256"] == row["parser_provenance"]["conversion_trace_sha256"]
    targets.require_target_row_identity(row, expected_provider="deps2index_sd14")
    with pytest.raises(ValueError, match="provider mismatch"):
        targets.require_target_row_identity(row, expected_provider="spacy_sd14")


def test_stage2_failure_becomes_a_failure_row(monkeypatch: pytest.MonkeyPatch) -> None:
    rows, traces = _target_rows(monkeypatch, _coverage(all_generate_units_success=False,
                                                       any_generate_unit_success=False))
    assert (rows[0]["status"], rows[0]["failure_code"], rows[0]["program_sha256"]) == (
        "stage2_dependency_failure", "no_successful_dependency_unit", None)
    assert traces[0]["conversion_trace"]["stage"] == "stage2"
    with pytest.raises(targets.GuidanceTargetError, match="impossible Stage 2 coverage state"):
        _target_rows(monkeypatch, _coverage(any_generate_unit_success=False))


def test_expand_to_image_rows_and_collapse_prompt_only(tmp_path: Path) -> None:
    from ccg_diffusion.provenance import sha256_text

    base = [{"experiment": "t2icb_color", "prompt_id": "t2icb_color_0000", "prompt": "a red car", "seed": 0}]
    source = [{"prompt_id": "t2icb_color_0000", "provider": "deps2index_sd14", "status": "ok",
               "prompt_sha256": sha256_text("a red car")}]
    images = [{"experiment": "t2icb_color", "prompt_id": f"t2icb_color_0000_s{seed}", "prompt": "a red car",
               "seed": seed} for seed in (42, 43)]
    rows, used = targets.expand_to_image_rows(base, source, images)
    assert used == {"t2icb_color_0000"}
    assert rows[1] == {**source[0], "prompt_id": "t2icb_color_0000_s43", "source_sidecar_prompt_id": "t2icb_color_0000"}
    with pytest.raises(targets.GuidanceTargetError, match="has no source program"):
        targets.expand_to_image_rows(base, source, [{**images[0], "prompt": "a blue car"}])

    blind = [{"prompt_id": f"p0_s{seed}", "prompt": "a red car", "seed": 0} for seed in range(42, 52)]
    assert targets.collapse_prompt_only(blind, location=tmp_path) == {
        "p0": {"prompt_id": "p0", "prompt": "a red car", "seed": 0}}
    with pytest.raises(targets.SourceManifestError, match="all ten seed suffixes"):
        targets.collapse_prompt_only(blind[:9], location=tmp_path)


def test_frozen_t2i_parse_bundle(tmp_path: Path) -> None:
    bundle = targets.load_t2i_parse_bundle(verify=True)
    assert bundle["manifest"]["schema"] == "ccg_diffusion.t2i_parse_bundle.v1"
    assert len(bundle["manifest"]["files"]) == 24 and set(bundle["categories"]) == set(targets.T2I_COMPBENCH_CATEGORIES)
    written = targets.unpack_t2i_parse_bundle(tmp_path, categories=("color",))
    assert written["color"]["markedup_deps"] == tmp_path / "generate/color.markedup_deps"
    entry = next(e for e in bundle["manifest"]["files"] if e["path"] == "auto/color.auto.gz")
    assert entry["raw_sha256"] == "d66bf7742cc598bb5dcc14b18e61ba876e721e54c682ec7bec51a8365c03570b"
    assert gzip.decompress(bundle["categories"]["color"]["auto"].read_bytes()) == (tmp_path / "color.auto").read_bytes()
    targets.unpack_t2i_parse_bundle(tmp_path, categories=("color",))  # identical bytes are kept
    (tmp_path / "color.auto").write_text("changed")
    with pytest.raises(targets.GuidanceTargetError, match="refusing to overwrite"):
        targets.unpack_t2i_parse_bundle(tmp_path, categories=("color",))
