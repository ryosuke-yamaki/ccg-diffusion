"""Stage 1 corpus and metadata, and Stage 2 C&C ``generate`` marked dependencies with PARTIAL aggregation."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Sequence

import pytest

from ccg_diffusion.config import HOLCCG_PARSE, HOLCCG_REVISION
from ccg_diffusion.parsing.dependencies import (
    AGGREGATION_NOTE,
    AutoRecord,
    GenerateUnit,
    SentencePlan,
    build_sentence_plans,
    load_aligned_dependency_blocks,
    load_auto_records,
    prepare_markedup_file,
    reindex_dependency_line,
    split_ccgbank_deps,
    validate_aggregated_dependency_blocks,
    write_ccgbank_deps,
)
from ccg_diffusion.parsing.derivation import extract_partial_subtrees, parse_auto_tree_with_spans
from ccg_diffusion.parsing.sentences import (
    SentenceMapError,
    build_sentence_map_rows,
    load_sentence_map,
    sentence_map_row_from_payload,
    split_and_tokenize,
    split_sentences,
    tokenize_for_parse,
    validate_sentence_map_group,
    write_parse_corpus,
)

ROOT = Path(__file__).resolve().parents[1]
# Recorded first sentence of the T2I-CompBench++ color category (complete parse).
COLOR_1 = (
    "(<T NP 0 2> (<T NP 1 2> (<L NP[nb]/N POS POS a NP[nb]/N>) (<T N 1 2> (<L N/N POS POS green N/N>) "
    "(<L N POS POS bench N>) ) ) (<T NP[conj] 1 2> (<L conj POS POS and conj>) (<T NP 1 2> "
    "(<L NP[nb]/N POS POS a NP[nb]/N>) (<T N 1 2> (<L N/N POS POS blue N/N>) (<L N POS POS bowl N>) ) ) ) )"
)
COLOR_1_DEPENDENCIES = [
    "and_4 conj 1 bowl_7 0",
    "and_4 conj 1 bench_3 0",
    "a_1 (NP[nb]{Y}/N{Y}<1>){_} 1 bench_3 0",
    "green_2 (N{Y}/N{Y}<1>){_} 1 bench_3 0",
    "a_5 (NP[nb]{Y}/N{Y}<1>){_} 1 bowl_7 0",
    "blue_6 (N{Y}/N{Y}<1>){_} 1 bowl_7 0",
]
PARTIAL_LEFT = "(<T NP 1 2> (<L NP[nb]/N POS POS a NP[nb]/N>) (<L N POS POS cat N>) )"
PARTIAL_RIGHT = "(<L S[dcl]\\NP POS POS sleeps S[dcl]\\NP>)"


# --- sentence split and parse tokenization ------------------------------------------------------------


def test_sentence_split_and_parse_tokenization() -> None:
    prompt = "A dog runs.  A cat sleeps!  Who knows?"
    assert [(s.text, s.char_start, s.char_end) for s in split_sentences(prompt)] == [
        ("A dog runs.", 0, 11), ("A cat sleeps!", 13, 26), ("Who knows?", 28, 38)]
    assert [s.text for s in split_sentences('The sign reads "Stop." A car waits.')] == [
        'The sign reads "Stop."', "A car waits."]
    assert [s.text for s in split_sentences('A sign proclaims "Welcome Friends!" to all. It glows.')] == [
        'A sign proclaims "Welcome Friends!" to all.', "It glows."]
    assert [s.text for s in split_sentences("It stands 37.5 inches tall.It is blue.")] == [
        "It stands 37.5 inches tall.It is blue."]
    assert [t.text for t in tokenize_for_parse('Two pairs (blue, red) sit; a "Merryman" plate.')] == [
        "Two", "pairs", "(", "blue", ",", "red", ")", "sit", ";", "a", '"', "Merryman", '"', "plate", "."]
    assert [t.text for t in tokenize_for_parse("the day's high-top sneakers aren't")] == [
        "the", "day", "'s", "high-top", "sneakers", "aren't"]
    assert [t.text for t in tokenize_for_parse("the cat’s objects' sandals—one")] == [
        "the", "cat", "’s", "objects'", "sandals—one"]


@pytest.mark.parametrize("prompt", ["a red dog", 'The sign reads "Stop." The day\'s light fades!', "It’s here. No…"])
def test_parse_tokens_are_exact_prompt_substrings(prompt: str) -> None:
    for sentence in split_and_tokenize(prompt):
        assert prompt[sentence.char_start:sentence.char_end] == sentence.text
        assert all(prompt[t.char_start:t.char_end] == t.text for t in sentence.tokens)
        assert sentence.raw_line == " ".join(t.text for t in sentence.tokens)


def test_sentence_map_rows_and_validation(tmp_path: Path) -> None:
    prompt = "A dog runs. A cat sleeps."
    rows = build_sentence_map_rows("p0", prompt, raw_line_offset=0)
    assert [row["raw_line_index"] for row in rows] == [0, 1]
    assert rows[1]["tokens"][1] == {"text": "cat", "char_start": 14, "char_end": 17}
    path = tmp_path / "map.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    loaded = load_sentence_map(path)
    validate_sentence_map_group(prompt, loaded)
    with pytest.raises(SentenceMapError, match="text 'cat' != prompt span"):
        validate_sentence_map_group("A dog runs. A cow sleeps.", loaded)
    with pytest.raises(SentenceMapError, match="key mismatch"):
        sentence_map_row_from_payload({**rows[0], "extra": 1}, location="row")


def test_write_parse_corpus_bytes(tmp_path: Path) -> None:
    manifest = tmp_path / "prompts.jsonl"
    rows = [{"prompt_id": "p0", "prompt": "a red dog's bone"}, {"prompt_id": "p1", "prompt": "Dogs run. Cats sit."}]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    report = write_parse_corpus(prompt_manifest=manifest, raw_output=tmp_path / "p.raw",
                                source_map_output=tmp_path / "p.source_map.jsonl",
                                report_output=tmp_path / "p.report.json", expected_count=2, overwrite=False)
    assert (tmp_path / "p.raw").read_bytes() == b"a red dog 's bone\nDogs run .\nCats sit .\n"
    assert (tmp_path / "p.source_map.jsonl").read_text(encoding="utf-8").splitlines()[0] == (
        '{"prompt": "a red dog\'s bone", "prompt_id": "p0"}')
    assert [json.loads(line)["raw_line_index"]
            for line in (tmp_path / "p.sentence_map.jsonl").read_text().splitlines()] == [0, 1, 2]
    assert report["schema"] == "holsd.native_prompt_corpus.v2"
    assert (report["sentence_count"], report["multi_sentence_prompt_count"]) == (3, 1)
    assert report["raw"]["sha256"] == hashlib.sha256((tmp_path / "p.raw").read_bytes()).hexdigest()


# --- Stage 1 metadata and configuration ---------------------------------------------------------------


def test_parse_config_equals_the_paper_parse_contract() -> None:
    from omegaconf import OmegaConf

    cfg = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/parsing/holccg.yaml"))
    assert cfg["parsing"] == HOLCCG_PARSE.inference_settings()
    assert (cfg["model_dir"], cfg["model_revision"]) == (HOLCCG_PARSE.checkpoint, HOLCCG_PARSE.revision)


class _FakeParser:
    """``HolCCGParser`` stand-in: the first sentence parses, the next is a skimmer result, the rest fail."""

    def __init__(self, device: str) -> None:
        self.device = device
        self.model = SimpleNamespace(tokenizer=SimpleNamespace(model_max_length=1_000_000_000))

    @classmethod
    def from_pretrained(cls, model_dir: str, *, options: Any, device: str, strict: bool = True) -> "_FakeParser":
        return cls(device)

    def iter_parse_chunks(self, sentences: Any, *, id_start: int = 1,
                          sentence_ids: Sequence[str] | None = None) -> Iterator[Any]:
        from holccg.parsing.inference import ParsedSentence, ParseStatus

        for offset, tokens in enumerate(sentences):
            tokens = tuple(tokens)
            status = (ParseStatus.SUCCESS, ParseStatus.PARTIAL, ParseStatus.FAIL)[min(offset, 2)]
            yield ParsedSentence(
                index=id_start + offset, sentence_id=sentence_ids[offset], tokens=tokens, status=status,
                auto=f"(<L N POS POS {tokens[0]} N>)" if offset == 0 else None,
                partial_autos=(f"(<L N POS POS {tokens[0]} N>)",) if offset == 1 else (),
            )


@pytest.mark.requires_holccg
@pytest.mark.parametrize("local_checkpoint", [True, False])
def test_parse_corpus_writes_auto_and_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                               local_checkpoint: bool) -> None:
    from ccg_diffusion.parsing import holccg

    snapshot = tmp_path / "hub/snapshots" / HOLCCG_REVISION
    snapshot.mkdir(parents=True)
    model_dir = str(snapshot) if local_checkpoint else HOLCCG_PARSE.checkpoint
    monkeypatch.setattr(holccg, "HolCCGParser", _FakeParser)
    monkeypatch.setattr(holccg, "align_encoder_dtype", lambda model: model)
    monkeypatch.setattr(holccg, "resolve_checkpoint", lambda value, **kwargs: snapshot)
    raw, auto = tmp_path / "prompts.raw", tmp_path / "prompts.auto"
    raw.write_text("cat\ndog runs\nbird\n", encoding="utf-8")
    meta = holccg.parse_corpus(model_dir=model_dir, input_raw_path=str(raw), output_auto_path=str(auto),
                               options=holccg.inference_options(HOLCCG_PARSE), device="cpu", progress=False)

    from holccg.parsing.render import fallback_right_branching

    assert auto.read_text(encoding="utf-8").splitlines() == [
        "ID=prompts.1 PARSER=HOLCCG NUMPARSE=1", "(<L N POS POS cat N>)",
        "ID=prompts.2 PARSER=HOLCCG NUMPARSE=1", "(<T PARTIAL 0 1> (<L N POS POS dog N>) )",
        "ID=prompts.3 PARSER=HOLCCG NUMPARSE=1", fallback_right_branching(["bird"], "FAIL", "POS").strip(),
    ]
    assert json.loads(holccg.auto_meta_path(auto).read_text()) == meta
    assert meta["model_dir"] == model_dir
    assert meta["resolved_revision"] == (None if local_checkpoint else HOLCCG_REVISION)
    assert meta["holccg_version"] == importlib.metadata.version("holccg")
    assert meta["parsing"] == HOLCCG_PARSE.inference_settings()
    assert meta["input_raw_sha256"] == hashlib.sha256(raw.read_bytes()).hexdigest()
    assert meta["output_auto_sha256"] == hashlib.sha256(auto.read_bytes()).hexdigest()
    assert meta["sentence_counts"] == {"total": 3, "ok": 1, "partial": 1, "fail": 1, "error": 0, "overlen": 0}
    assert datetime.fromisoformat(meta["created_at_utc"]).tzinfo is not None


@pytest.mark.requires_holccg
def test_combine_partial_autos_wraps_one_fragment() -> None:
    from ccg_diffusion.parsing.holccg import combine_partial_autos

    assert combine_partial_autos([PARTIAL_RIGHT], "PARTIAL") == f"(<T PARTIAL 0 1> {PARTIAL_RIGHT} )"


# --- Stage 2 ------------------------------------------------------------------------------------------


def test_partial_fragments_keep_their_serialization_and_spans(tmp_path: Path) -> None:
    root = parse_auto_tree_with_spans(f"(<T PARTIAL 0 2> {PARTIAL_LEFT} {PARTIAL_RIGHT} )")
    subtrees = extract_partial_subtrees(root, "PARTIAL")
    assert [(node.start, node.end, node.auto) for node in subtrees] == [(1, 2, PARTIAL_LEFT), (3, 3, PARTIAL_RIGHT)]

    auto = tmp_path / "x.auto"
    auto.write_text(f"ID=x.1 NUMPARSE=1\n(<T PARTIAL 0 1> {PARTIAL_RIGHT} )\nID=x.2 NUMPARSE=1\n{COLOR_1}\n")
    plans = build_sentence_plans(load_auto_records(auto), "PARTIAL", {"FAIL", "OVERLEN"})
    assert [(plan.parse_status, len(plan.units)) for plan in plans] == [("partial", 1), ("complete", 1)]
    assert plans[0].units[0].derivation == PARTIAL_RIGHT


def test_dependency_reindexing_and_cross_fragment_rejection() -> None:
    assert reindex_dependency_line("which_1 ((NP{Y}\\NP{Y})/NP{Z}) 2 object_name_3 0", 4) == (
        "which_5 ((NP{Y}\\NP{Y})/NP{Z}) 2 object_name_7 0")
    units = [GenerateUnit(1, 1, 1, 2, 1, 2, "NP", "partial", ""), GenerateUnit(2, 1, 2, 2, 3, 4, "NP", "partial", "")]
    plan = SentencePlan(AutoRecord(1, "ID=t.1", ""), "PARTIAL", 4, "partial", units)
    assert validate_aggregated_dependency_blocks([plan], [["l_1 NP 1 m_2 0", "r_3 NP 1 m_4 0"]]) == 2
    with pytest.raises(ValueError, match="crosses synthetic PARTIAL subtree"):
        validate_aggregated_dependency_blocks([plan], [["l_1 NP 1 r_3 0"]])


def test_markedup_overlay_adds_a_missing_category_once(tmp_path: Path) -> None:
    category = "((NP\\NP)/(S[dcl]\\NP))\\((NP\\NP)/NP)"
    (tmp_path / "base").write_text("N\n  0 N{_}\n", encoding="utf-8")
    (tmp_path / "additions").write_text(f"# overlay\n\n{category}\n  3 ignore\n  1 ignore\n", encoding="utf-8")
    assert prepare_markedup_file(tmp_path / "base", tmp_path / "additions", tmp_path / "e1") == [category]
    assert prepare_markedup_file(tmp_path / "e1", tmp_path / "additions", tmp_path / "e2") == []
    assert (tmp_path / "e2").read_text(encoding="utf-8").count(category + "\n") == 1


def test_dependency_file_round_trip_keeps_empty_leading_blocks(tmp_path: Path) -> None:
    path = tmp_path / "x.markedup_deps"
    write_ccgbank_deps(path, ["# generate -e ...", AGGREGATION_NOTE], [[], ["red_1 (N{Y}/N{Y}<1>){_} 1 car_2 0"]])
    assert split_ccgbank_deps(path) == (["# generate -e ...", AGGREGATION_NOTE, ""],
                                        [[], ["red_1 (N{Y}/N{Y}<1>){_} 1 car_2 0"]])
    blocks = load_aligned_dependency_blocks(path, 2)
    assert blocks[0] == [] and blocks[1][0].argument_word == "car"
    assert AGGREGATION_NOTE == "# PARTIAL fragments aggregated with sentence-global token indices by HolSD"


@pytest.mark.requires_holccg
def test_generate_dependencies_aggregates_fragments(tmp_path: Path) -> None:
    from ccg_diffusion.parsing.dependencies import generate_dependencies, generate_marked_dependencies

    auto, raw = tmp_path / "prompts.auto", tmp_path / "prompts.raw"
    auto_lines = ["ID=prompts.1 PARSER=HOLCCG NUMPARSE=1", COLOR_1,
                  "ID=prompts.2 PARSER=HOLCCG NUMPARSE=1", f"(<T PARTIAL 0 2> {PARTIAL_LEFT} {PARTIAL_RIGHT} )"]
    raw_lines = ["a green bench and a blue bowl", "a cat sleeps"]
    auto.write_text("\n".join(auto_lines) + "\n", encoding="utf-8")
    raw.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")
    summary = generate_dependencies(auto_path=auto, output_dir=tmp_path / "generate", raw_path=raw)

    header, blocks = split_ccgbank_deps(tmp_path / "generate/prompts.markedup_deps")
    assert header[-2:] == [AGGREGATION_NOTE, ""]
    assert blocks[0] == COLOR_1_DEPENDENCIES
    assert all(line.split()[3].endswith(("_1", "_2", "_3")) for line in blocks[1])
    rows = [json.loads(line) for line in (tmp_path / "generate/prompts.coverage_rows.jsonl").read_text().splitlines()]
    assert [(row["parse_status"], row["fragment_count"], row["prompt"]) for row in rows] == [
        ("complete", 1, raw_lines[0]), ("partial", 2, raw_lines[1])]
    assert summary["markedup"]["resource"] == "hol_en"
    assert summary["markedup"]["additions"] == "configs/parsing/candc_markedup_additions"
    in_process = generate_marked_dependencies(auto_lines, raw_lines)
    assert [[f"{d.head_word}_{d.head_index}" for d in block] for block in in_process] == [
        [line.split()[0] for line in block] for block in blocks]
