"""Supertag-Constrained Parsing: the forced-supertag hook and the derivation helpers.

The Hol-CCG tests avoid loading a checkpoint: ``HolCCG.__init__`` (tokenizer +
encoder) is bypassed and only the attributes the public decoder reads are set.
Fake ``_encode_words`` / classifiers are plain instance attributes, which is the
same ``vars(model)`` override path the forced-leaf hook relies on.
"""
from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from ccg_diffusion.parsing.derivation import (
    INTERPRETATION_SUPERTAGS,
    iter_leaves,
    leaf_at_position,
    leaf_position,
    node_category,
    path_to_position,
)

WORD_VOCAB: Dict[str, int] = {"<unk>": 0, "NP": 1, "N": 2, "(NP\\NP)/NP": 3}
PHRASE_VOCAB: Dict[str, int] = {"<unk>": 0, "NP": 1, "S": 2}
MODEL_DIM = 4


@pytest.fixture
def holccg_module() -> Any:
    pytest.importorskip("holccg")
    from ccg_diffusion.parsing import holccg

    return holccg


def _default_probs() -> Any:
    import torch

    # Top-1 is "N" (idx 2); "NP" (idx 1) clears a 0.1 stag_threshold; the rest stay below it.
    return torch.tensor([[0.01, 0.30, 0.60, 0.09]] * 4)


def _stub_model(holccg_module: Any, *, probs: Optional[Any] = None) -> Any:
    """A :class:`SupertagConstrainedHolCCG` built without ``__init__``."""
    import torch
    from holccg.models.persistence import HolCCGConfig

    model = holccg_module.SupertagConstrainedHolCCG.__new__(holccg_module.SupertagConstrainedHolCCG)
    model.training = False
    model.word_category2index = dict(WORD_VOCAB)
    model.phrase_category2index = dict(PHRASE_VOCAB)
    model.config = HolCCGConfig()
    model.vector_norm = getattr(model.config, "normalization_norm", 30.0)
    model.word_index2category = [None] * (max(WORD_VOCAB.values()) + 1)
    for cat, idx in WORD_VOCAB.items():
        model.word_index2category[idx] = cat
    model.phrase_index2category = [None] * (max(PHRASE_VOCAB.values()) + 1)
    for cat, idx in PHRASE_VOCAB.items():
        model.phrase_index2category[idx] = cat
    model.binary_rule = {}
    model.head_info = {}
    model._compiled_binary_rule = None
    model._compiled_arrays = None
    model._final_cat_to_id = {}
    model._num_final_cats = 0
    model._cat_to_id = {}
    model._by_left_int = {}
    model._by_left_int_entries = 0
    model._rule_type_cache = {}
    model._winner_select_cky = True
    model._cross_sentence_cky = True
    model._parse_grammar_mode = "treebank"
    model._candc_grammar = None
    model._word_mask_ids = None
    model._word_mask_tensor = None

    def encode_words(words: List[str], *, autocast_bf16: bool = False) -> Any:
        base = torch.arange(1, MODEL_DIM + 1, dtype=torch.float32)
        return torch.stack([base + index for index in range(len(words))], dim=0)

    word_probs = _default_probs() if probs is None else probs
    model._encode_words = encode_words
    model.word_category_classifier = lambda x: torch.log(word_probs[: x.size(0)])
    return model


def _add_grammar(model: Any) -> None:
    """``N N -> NP`` and ``NP N -> S``: forcing position 0 to NP turns the root from NP into S."""
    import torch

    model.binary_rule = {("N", "N"): ["NP"], ("NP", "N"): ["S"]}
    model.head_info = {("N", "N", "NP"): 0, ("NP", "N", "S"): 1}
    model.span_existence_classifier = lambda x: torch.tensor([[0.0, 5.0]] * x.size(0))
    model.phrase_category_classifier = lambda x: torch.zeros(x.size(0), len(model.phrase_category2index))


def _leaf_chart(model: Any, words: List[str], stag_threshold: float) -> Dict[Any, Any]:
    return model._build_leaf_chart(words, model._encode_words(words), stag_threshold)


@pytest.mark.requires_holccg
def test_no_forced_supertags_takes_the_stock_path(holccg_module: Any) -> None:
    model = _stub_model(holccg_module)
    seen: Dict[str, Any] = {}

    def record(sentence: str, **kwargs: Any) -> Dict[str, Any]:
        seen["hooked"] = "_build_leaf_chart" in vars(model)
        seen["kwargs"] = kwargs
        return {"status": "ok"}

    model._parse_single = record
    assert model.parse("a b", forced_supertags=None) == {"status": "ok"}
    assert seen["hooked"] is False and "forced_supertags" not in seen["kwargs"]
    assert "_build_leaf_chart" not in vars(model)


@pytest.mark.requires_holccg
def test_forced_position_gets_a_single_category_cell(holccg_module: Any) -> None:
    model = _stub_model(holccg_module)
    model.__dict__["_build_leaf_chart"] = holccg_module._make_forced_leaf_hook(model, {1: "NP"})
    chart = _leaf_chart(model, ["a", "b", "c"], 0.1)

    forced_cell = chart[(1, 2)]
    assert list(forced_cell.best_category) == ["NP"]
    forced = forced_cell.best_category["NP"]
    assert (forced.is_leaf, forced.word, forced.head, forced.span, forced.span_ll) == (True, "b", 1, (1, 2), 0.0)
    # Above the floor, so the classifier probability is used (float32 softmax round trip).
    assert forced.cat_ll == pytest.approx(math.log(0.30 + 1e-12), abs=1e-6)
    for span in [(0, 1), (2, 3)]:
        assert set(chart[span].best_category) == {"N", "NP"}


@pytest.mark.requires_holccg
def test_zero_probability_forced_category_uses_the_floor(holccg_module: Any) -> None:
    import torch

    model = _stub_model(holccg_module, probs=torch.tensor([[0.01, 0.30, 0.69, 0.0]] * 2))
    model.__dict__["_build_leaf_chart"] = holccg_module._make_forced_leaf_hook(model, {0: "(NP\\NP)/NP"})
    category = _leaf_chart(model, ["a", "b"], 0.1)[(0, 1)].best_category["(NP\\NP)/NP"]
    assert category.cat_ll == pytest.approx(math.log(1e-3 + 1e-12), abs=1e-9)


@pytest.mark.requires_holccg
def test_unknown_forced_category_is_rejected(holccg_module: Any) -> None:
    model = _stub_model(holccg_module)
    with pytest.raises(ValueError, match=r"NOT_A_CATEGORY.*position 2"):
        model.parse("a b c", forced_supertags={2: "NOT_A_CATEGORY"})
    assert "_build_leaf_chart" not in vars(model)


@pytest.mark.requires_holccg
@pytest.mark.parametrize("batched", [True, False])
def test_forced_supertag_changes_the_derivation_and_leaves_no_state(holccg_module: Any, batched: bool) -> None:
    model = _stub_model(holccg_module)
    _add_grammar(model)
    kwargs = dict(stag_threshold=0.5, phrase_threshold=0.01, span_threshold=0.01, apply_skimmer=False,
                  batched=batched)
    unforced = model.parse("a b", **kwargs)
    assert unforced["root_cat"].cat == "NP"
    forced = model.parse("a b", forced_supertags={0: "NP"}, **kwargs)
    assert forced["root_cat"].cat == "S"
    assert "(<L NP POS POS a NP>)" in forced["auto"]
    assert "_build_leaf_chart" not in vars(model)
    assert model.parse("a b", **kwargs)["auto"] == unforced["auto"]


@pytest.mark.requires_holccg
def test_supertag_constrained_parse_passes_the_contract(holccg_module: Any) -> None:
    from ccg_diffusion.config import HOLCCG_SUPERTAG_CONSTRAINED_PARSE, HOLCCG_TAB_SITE_PARSE

    calls: List[Dict[str, Any]] = []
    parser = SimpleNamespace(parse=lambda sentence, **kwargs: calls.append({"sentence": sentence, **kwargs}))
    holccg_module.supertag_constrained_parse(parser, "a b", {1: "NP"})
    holccg_module.site_parse(parser, "a b")
    flags = {"return_chart": False, "force_eval": True, "use_no_grad": True}
    assert calls[0] == {"sentence": "a b", "forced_supertags": {1: "NP"},
                        **HOLCCG_SUPERTAG_CONSTRAINED_PARSE.parsing_kwargs(), **flags}
    assert calls[1] == {"sentence": "a b", **HOLCCG_TAB_SITE_PARSE.parsing_kwargs(), **flags}
    assert calls[0]["phrase_threshold"] == 1e-6 and calls[0]["apply_skimmer"] is False


@pytest.mark.requires_holccg
def test_reentrant_forced_parse_is_rejected(holccg_module: Any) -> None:
    model = _stub_model(holccg_module)
    model.__dict__["_build_leaf_chart"] = holccg_module._make_forced_leaf_hook(model, {0: "NP"})
    with pytest.raises(RuntimeError, match="forced_supertags parse is already active"):
        model.parse("a b", forced_supertags={0: "NP"})
    assert "_build_leaf_chart" in vars(model)


@pytest.mark.requires_holccg
def test_compat_guard(holccg_module: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    import holccg

    assert holccg.__version__ in holccg_module.TESTED_HOLCCG_VERSIONS
    holccg_module.check_holccg_compat()
    monkeypatch.setattr(holccg, "__version__", "9.9.9", raising=False)
    with pytest.raises(RuntimeError, match="validated against holccg"):
        holccg_module.check_holccg_compat()


def test_interpretation_supertags_are_the_paper_categories() -> None:
    assert dict(INTERPRETATION_SUPERTAGS) == {"np_attach": r"(NP\NP)/NP", "vp_attach": r"((S\NP)\(S\NP))/NP"}


def _chart_node(cat: str, span: tuple, *children: Any) -> SimpleNamespace:
    left, right = (list(children) + [None, None])[:2]
    return SimpleNamespace(cat=cat, span=span, is_leaf=not children, left_child=left, right_child=right)


def test_chart_tree_walks() -> None:
    boy, looks = _chart_node("N", (0, 1)), _chart_node(r"S[dcl]\NP", (1, 2))
    root = _chart_node("S[dcl]", (0, 2), boy, looks)
    assert [node_category(leaf) for leaf in iter_leaves(root)] == ["N", r"S[dcl]\NP"]
    assert leaf_position(looks) == 1 and leaf_position(root) is None
    assert leaf_at_position(root, 1) is looks and leaf_at_position(root, 5) is None
    assert path_to_position(root, 1) == [root, looks]


# --- choice of the ambiguous preposition (pp_attachment.supertag_constrained_parsing) ---------------------


def test_preposition_category_arithmetic() -> None:
    from ccg_diffusion.pp_attachment import supertag_constrained_parsing as scp

    assert scp.delete_argument_slot(r"((S[dcl]\NP)/PP)/NP", "PP") == r"(S[dcl]\NP)/NP"
    assert scp.delete_argument_slot(r"(S[dcl]\NP)/NP", "PP") is None
    assert scp.stranded_arguments(r"S[dcl]/PP", "S[dcl]") == [("/", "PP")]
    assert scp.stranded_arguments("NP", "S[dcl]") is None
    assert scp._nominal_core_variant_category(r"(NP\NP)/NP") == r"(N\N)/NP"
    assert scp._nominal_core_variant_category(r"NP\NP") == r"N\N"
    assert scp._anchor_chain(r"((S\NP)\(S\NP))/NP", frozenset()) == (("/", "NP"),)
    assert scp._anchor_chain(r"(NP\NP)/NP", frozenset()) == (("/", "NP"),)
    assert scp._anchor_chain("N", frozenset()) is None  # nominal atoms never stop the peel
    assert scp._anchor_chain(r"(S[ng]\NP)/NP", frozenset({r"S[ng]\NP"})) == (("/", "NP"),)


def test_schema_pairs_and_type_change_stops_come_from_the_lexicon() -> None:
    from ccg_diffusion.pp_attachment import supertag_constrained_parsing as scp

    grammars: list[str] = []
    parser = SimpleNamespace(
        word_category2index={r"(NP\NP)/NP": 0, r"((S\NP)\(S\NP))/NP": 1, r"NP\NP": 2, r"(S\NP)\(S\NP)": 3,
                             r"(NP\NP)/PP": 4, "N": 5, r"S[ng]\NP-->NP\NP": 6},
        binary_rule={("A", "B"): [r"S[ng]\NP-->NP\NP", r"S[ng]\NP-->(S\NP)\(S\NP)", r"N-->NP"]},
        set_parse_grammar=grammars.append)
    grammar = scp.load_grammar(parser)
    assert grammars == ["treebank"]
    assert sorted(grammar.pairs_by_chain) == [(), (("/", "NP"),)]
    pair = grammar.pairs_by_chain[(("/", "NP"),)]
    assert (pair.category("np_attach"), pair.category("vp_attach")) == (r"(NP\NP)/NP", r"((S\NP)\(S\NP))/NP")
    assert pair.rightward_argument_count == 1 and pair.chain_id == "/NP"
    assert grammar.typechange_stops == frozenset({r"S[ng]\NP"})


def test_nesting_keeps_the_outermost_site() -> None:
    from conftest import auto_leaf

    from ccg_diffusion.pp_attachment import supertag_constrained_parsing as scp

    def tree(category: str, *children: str) -> str:
        return f"(<T {category} 0 {len(children)}> {' '.join(children)})"

    inner = tree(r"NP\NP", auto_leaf(r"(NP\NP)/NP", "with"), auto_leaf("NP", "dog"))
    outer = tree(r"NP\NP", auto_leaf(r"(NP\NP)/NP", "near"), tree("NP", auto_leaf("N", "cage"), inner))
    auto = tree("NP", auto_leaf("N", "girl"), outer)
    sites = [{"position": 1, "word": "near", "readings": {"np_attach": {"auto": auto}, "vp_attach": {"auto": auto}}},
             {"position": 3, "word": "with", "readings": {"np_attach": {"auto": auto}, "vp_attach": {"auto": auto}}}]
    assert scp.modifier_projection_span(auto, position=1, core=r"NP\NP") == (2, 5)
    assert scp.modifier_projection_span(auto, position=3, core=r"NP\NP") == (4, 5)
    assert [site["word"] for site in scp._apply_nesting(sites)] == ["near"]


def test_site_dependencies_resolve_to_nominal_entities() -> None:
    from conftest import marked_dependencies

    from ccg_diffusion.pp_attachment import supertag_constrained_parsing as scp

    attachment = r"((NP{Y}\NP{Y}<1>){_}/NP{Z}<2>){_}"
    deps = marked_dependencies(f"with_2 {attachment} 1 girl_1 0", f"with_2 {attachment} 2 a_3 0",
                               r"a_3 (NP[nb]{Y}/N{Y}<1>){_} 1 dog_4 0")
    leaves = ["N", r"(NP\NP)/NP", "NP[nb]/N", "N"]
    assert scp.resolve_endpoint(3, dependencies_by_head={3: deps[2:]}, leaf_categories=leaves) == 4
    assert scp.site_dependency_signature(deps, site_index=2, leaf_categories=leaves) == ((1, 1), (2, 4))
    assert scp._leaf_categories_from_auto("(<L N POS POS girl N>) (<L NP[nb]/N POS POS a NP[nb]/N>)") == [
        "N", "NP[nb]/N"]


def test_progressive_standing_interpretations_differ_only_at_the_preposition() -> None:
    import json
    from pathlib import Path

    from ccg_diffusion.pp_attachment import supertag_constrained_parsing as scp

    data = json.loads((Path(__file__).parent / "fixtures/tab_progressive_standing.json").read_text())
    np_leaves, vp_leaves = (scp._leaf_categories_from_auto(row["auto"]) for row in data)
    site = data[0]["site_source_index"] - 1
    assert (np_leaves[site], vp_leaves[site]) == (r"(NP\NP)/NP", r"((S\NP)\(S\NP))/NP")
    assert [a for i, a in enumerate(np_leaves) if i != site] == [b for i, b in enumerate(vp_leaves) if i != site]


def test_auto_lines_number_the_items_from_one() -> None:
    from ccg_diffusion.pp_attachment import supertag_constrained_parsing as scp

    derivations = [scp.ConstrainedDerivation("np_attach", {1: r"(NP\NP)/NP"}, f"(<L N POS POS w{i} N>) ", "N")
                   for i in range(2)]
    assert scp.auto_lines(derivations) == ["ID=prompts.1 PARSER=HOLCCG NUMPARSE=1", "(<L N POS POS w0 N>)",
                                           "ID=prompts.2 PARSER=HOLCCG NUMPARSE=1", "(<L N POS POS w1 N>)"]
