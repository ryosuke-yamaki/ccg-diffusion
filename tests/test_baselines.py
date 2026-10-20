"""The T2I baselines: method kernels, input files, arm registry, SD1.4 batching and the Sana adaptations."""
from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from nltk.tree import Tree

from ccg_diffusion.baselines import (
    attend_and_excite,
    composable_diffusion,
    preprocessing,
    registry,
    sana_plans,
    structure_diffusion,
    syngen,
)
from ccg_diffusion.baselines.sample_local_guidance import sample_local_latent_update
from ccg_diffusion.provenance import repository_root

ROOT = repository_root()


class FakeClipTokenizer:
    """CLIP-like pieces (``word</w>``; ``sunflower`` -> ``sun`` + ``flower</w>``) with BOS=0 and EOS=1."""

    model_max_length = 77

    def __init__(self) -> None:
        self._piece_to_id = {"<|startoftext|>": 0, "<|endoftext|>": 1}
        self._id_to_piece = {0: "<|startoftext|>", 1: "<|endoftext|>"}

    def _pieces(self, text: str) -> list[str]:
        pieces = []
        for match in re.finditer(r"[A-Za-z]+|[^\w\s]", text):
            token = match.group(0).lower()
            pieces.extend(("sun", "flower</w>") if token == "sunflower" else (f"{token}</w>",))
        return pieces

    def _id(self, piece: str) -> int:
        if piece not in self._piece_to_id:
            index = len(self._piece_to_id) + 10
            self._piece_to_id[piece] = index
            self._id_to_piece[index] = piece
        return self._piece_to_id[piece]

    def __call__(self, text: str, **kwargs: Any) -> SimpleNamespace:
        ids = [self._id(piece) for piece in self._pieces(text)]
        return SimpleNamespace(input_ids=[0, *ids, 1] if kwargs.get("add_special_tokens", True) else ids)

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [self._id_to_piece[index] for index in ids]


@dataclass
class Token:
    """spaCy token stand-in with the attributes the extractors read."""

    text: str
    pos_: str
    dep_: str
    idx: int
    children: list["Token"] = field(default_factory=list)


def _binding_doc() -> tuple[str, tuple[Token, ...]]:
    prompt = "a pink sunflower and a yellow flamingo"
    values = (("a", "DET", "det"), ("pink", "ADJ", "amod"), ("sunflower", "NOUN", "nsubj"), ("and", "CCONJ", "cc"),
              ("a", "DET", "det"), ("yellow", "ADJ", "amod"), ("flamingo", "NOUN", "conj"))
    tokens, cursor = [], 0
    for text, pos, dep in values:
        index = prompt.index(text, cursor)
        tokens.append(Token(text, pos, dep, index))
        cursor = index + len(text)
    tokens[2].children.append(tokens[1])
    tokens[6].children.append(tokens[5])
    return prompt, tuple(tokens)


# --- 1. Composable Diffusion -----------------------------------------------------------------------------


def test_composable_split_and_composition() -> None:
    plan = composable_diffusion.split_prompt("A candy bowl AND a rock_and_roll poster and a chair.")
    assert plan.components == ("A candy bowl", "a rock_and_roll poster", "a chair.")
    assert plan.weights == (7.5, 7.5, 7.5)
    assert composable_diffusion.split_prompt("A Red Firetruck.").components == ("A Red Firetruck.",)

    unconditional = torch.tensor([[1.0, -2.0]])
    conditional = torch.tensor([[[3.0, 4.0]]])
    one = composable_diffusion.compose_noise_predictions(
        unconditional, conditional, torch.tensor([[True]]), torch.tensor([[7.5]]))
    torch.testing.assert_close(one, unconditional + 7.5 * (conditional[:, 0] - unconditional), rtol=0, atol=0)

    packed = composable_diffusion.pack_plans([composable_diffusion.split_prompt("a cat"),
                                              composable_diffusion.split_prompt("a dog and a ball")])
    assert packed.texts_by_slot == (("a cat", "a dog"), ("", "a ball"))
    batch_conditional = torch.tensor([[[2.0], [9999.0]], [[3.0], [4.0]]])
    batch_unconditional = torch.tensor([[1.0], [10.0]])
    composed = composable_diffusion.compose_noise_predictions(
        batch_unconditional, batch_conditional, packed.component_mask, packed.weights)
    torch.testing.assert_close(composed[0], torch.tensor([1.0 + 7.5 * 1.0]), rtol=0, atol=0)


# --- 2. StructureDiffusion -------------------------------------------------------------------------------


def test_structure_noun_phrases_and_batched_attention() -> None:
    tree = Tree.fromstring(
        "(ROOT (NP (NP (DT a) (JJ red) (NN fire) (NN truck)) (CC and) (NP (DT a) (JJ green) (NN bird))))")
    tokens = ("a</w>", "red</w>", "fire</w>", "truck</w>", "and</w>", "a</w>", "green</w>", "bird</w>")
    prompt = "a red fire truck and a green bird"
    texts, spans, nps, np_spans = structure_diffusion.extract_noun_phrases(tree, prompt, tokens)
    assert texts == (prompt, "a red fire truck", "a green bird")
    assert spans == ((0, 8), (0, 4), (5, 8)) and (nps, np_spans) == (texts, spans)
    assert structure_diffusion.has_full_np_coordination(tree) is True
    assert structure_diffusion.normalize_prompt("  A Red Bird...  ") == "a red bird"

    torch.manual_seed(0)
    query = torch.randn(4, 2, 5, 3)
    keys, values = torch.randn(4, 3, 2, 7, 3), torch.randn(4, 3, 2, 7, 3)
    key_mask = torch.tensor([[True, True, True], [True, False, False], [True, True, False], [True, True, True]])
    value_mask = torch.tensor([[True, True, True], [True, False, False], [True, True, False], [True, False, False]])
    batched = structure_diffusion.compute_structured_attention(
        query, keys, values, key_mask, value_mask, scale=0.5)
    for index in range(4):
        single = structure_diffusion.compute_structured_attention(
            query[index:index + 1], keys[index:index + 1], values[index:index + 1],
            key_mask[index:index + 1], value_mask[index:index + 1], scale=0.5)
        torch.testing.assert_close(batched[index:index + 1], single, rtol=0, atol=0)


# --- 3. Attend-and-Excite --------------------------------------------------------------------------------


def test_attend_and_excite_targets_objective_and_schedule() -> None:
    prompt = "A sunflower and a sunflower beside Paris"
    first = prompt.index("sunflower")
    second = prompt.index("sunflower", first + 1)
    doc = (Token("sunflower", "NOUN", "nsubj", first), Token("sunflower", "NOUN", "conj", second),
           Token("Paris", "PROPN", "pobj", prompt.index("Paris")))
    plan = attend_and_excite.extract_noun_targets(prompt, doc=doc, tokenizer=FakeClipTokenizer())
    assert [target.clip_indices for target in plan.targets] == [(2, 3), (6, 7)]
    assert plan.selected_clip_indices == (3, 7) and plan.status == "ok"

    maps = torch.zeros(16, 16, 5)
    maps[3, 4, 2] = 1.0
    (maximum,) = attend_and_excite.max_attention(maps, [2], smooth_attentions=False)
    expected = torch.softmax(maps[3, 4, 1:-1] * 100, dim=-1)[1]
    torch.testing.assert_close(maximum, expected)
    torch.testing.assert_close(attend_and_excite.sample_loss(maps, [2], smooth_attentions=False), 1 - expected)
    assert attend_and_excite.step_size(10) == pytest.approx(20 * math.sqrt(1 - 0.5 * 10 / 49))
    losses = (torch.tensor(0.95), torch.tensor(0.96), torch.tensor(0.99))
    mask = attend_and_excite.refinement_mask(losses, threshold=0.05, iteration_counts=[0, 0, 20])
    assert mask == (False, True, False)
    assert attend_and_excite.select_mean_winner((4, 5, 6), (0.2, 0.3, 0.3)) == 5
    identifier = attend_and_excite.occurrence_id(prompt, occurrence_index=0, text="sunflower",
                                                 character_span=(2, 11), candidate_token_indices=(2, 3))
    assert identifier.startswith("ae_noun_") and len(identifier) == 8 + 64
    assert attend_and_excite.no_target_id("x").startswith("ae_no_target_")


# --- 4. SynGen -------------------------------------------------------------------------------------------


def test_syngen_extractors_loss_and_batching() -> None:
    prompt, doc = _binding_doc()
    plan = syngen.extract_bindings(prompt, doc=doc, tokenizer=FakeClipTokenizer())
    assert [group.extractor for group in plan.source_groups] == ["standard", "standard", "with_verbs"]
    assert [group.members for group in plan.bindings] == [((7,), (8,)), ((2,), (3, 4))]
    # The official ``_extract_with_verbs`` returns after the first eligible noun.
    with_verbs = [group for group in syngen.extract_parser_groups(doc) if group.extractor == "with_verbs"]
    assert [group.texts for group in with_verbs] == [("pink", "sunflower")]

    p, q = torch.tensor([0.25, 0.75]), torch.tensor([0.5, 0.5])
    expected = ((p * (p / q).log()).sum() + (q * (q / p).log()).sum()) / 2
    torch.testing.assert_close(syngen.symmetric_kl(p, q), expected)

    torch.manual_seed(1)
    maps = torch.rand(2, 16, 16, 77)
    batched = syngen.batch_losses(maps, [plan, plan])
    for index in range(2):
        torch.testing.assert_close(batched[index], syngen.sample_loss(maps[index], plan), rtol=0, atol=0)


# --- 5. Per-image latent update --------------------------------------------------------------------------


def test_sample_local_update_keeps_the_single_image_gradient() -> None:
    latents = torch.arange(1.0, 9.0).reshape(2, 4).requires_grad_(True)
    losses = [latents[0].square().sum(), 3 * latents[1].square().sum()]
    updated = sample_local_latent_update(latents, losses, 0.5, active_mask=[True, False])
    torch.testing.assert_close(updated[0], latents[0] - 0.5 * 2 * latents[0], rtol=0, atol=0)
    torch.testing.assert_close(updated[1], latents[1], rtol=0, atol=0)


# --- 6. Predicated Diffusion targets ---------------------------------------------------------------------


@pytest.mark.requires_spacy_model
@pytest.mark.requires_clip_tokenizer
def test_predicated_diffusion_spacy_targets_and_seed_expansion() -> None:
    from ccg_diffusion.baselines.predicated_diffusion import SpacyProvider, base_prompt_rows, extract_rows
    from ccg_diffusion.targets.guidance_targets import expand_to_image_rows
    from ccg_diffusion.targets.tokenizer import load_sd14_tokenizer

    provider = SpacyProvider(load_sd14_tokenizer())
    result = provider.extract("a green bench and a blue bowl")
    assert result.status == "ok"
    assert (result.pd_indices.exist, result.pd_indices.corr) == ([3, 7], [[2, 3], [6, 7]])
    assert (result.pd_indices.leak, result.pd_indices.possession) == ([[2, 7], [6, 3]], [])
    image_rows = [{"experiment": "t2icb_color", "prompt_id": f"t2icb_color_0000_s{seed}",
                   "prompt": "a green bench and a blue bowl", "seed": 0} for seed in (42, 43)]
    base = base_prompt_rows(image_rows)
    assert base == [{"experiment": "t2icb_color", "prompt_id": "t2icb_color_0000",
                     "prompt": "a green bench and a blue bowl", "seed": 0}]
    rows, failures = extract_rows(base, provider)
    expanded, used = expand_to_image_rows(base, rows, image_rows)
    assert failures == 0 and used == {"t2icb_color_0000"}
    assert [(row["prompt_id"], row["source_sidecar_prompt_id"]) for row in expanded] == [
        ("t2icb_color_0000_s42", "t2icb_color_0000"), ("t2icb_color_0000_s43", "t2icb_color_0000")]


# --- 7. Arm registry -------------------------------------------------------------------------------------


def test_registry_keeps_the_recorded_arm_identities() -> None:
    assert registry.SD14_SIX_ARM_METHODS == (
        "sd14_composable", "sd14_structure", "sd14_attend_excite", "sd14_syngen", "sd14_pd_spacy", "sd14_pd_holccg")
    assert dict(registry.SOURCE_COMMIT_BY_ARM) == {
        "sd14_composable": composable_diffusion.SOURCE_COMMIT,
        "sd14_structure": structure_diffusion.SOURCE_COMMIT,
        "sd14_attend_excite": attend_and_excite.SOURCE_COMMIT,
        "sd14_syngen": syngen.SOURCE_COMMIT,
        "sd14_pd_spacy": "af06f1be19dc6f0be468b271714c4904e623928e",
        "sd14_pd_holccg": "af06f1be19dc6f0be468b271714c4904e623928e",
    }
    assert sana_plans.METHODS == tuple(registry.T2I_BASELINE_ARMS["sana"][method] for method in (
        "composable_diffusion", "structure_diffusion", "attend_and_excite", "syngen", "predicated_diffusion"))


# --- 8. Input rows ---------------------------------------------------------------------------------------


def test_input_records_round_trip_through_parse_plan() -> None:
    assert preprocessing.INPUT_ROW_FIELDS == {
        "experiment", "prompt_id", "prompt", "seed", "status", "method", "source_commit", "plan"}
    tokenizer = FakeClipTokenizer()
    prompt, doc = _binding_doc()
    records = {
        "sd14_composable": preprocessing.composable_record(prompt),
        "sd14_attend_excite": preprocessing.attend_and_excite_record(prompt, doc=doc, tokenizer=tokenizer),
        "sd14_syngen": preprocessing.syngen_record(prompt, doc=doc, tokenizer=tokenizer),
    }
    for arm, (status, payload) in records.items():
        stage = "candidate" if arm == "sd14_attend_excite" else "final"
        plan = preprocessing.parse_plan(json.loads(json.dumps(payload)), arm=arm, prompt=prompt,
                                        location="row", stage=stage)
        preprocessing.validate_row_status(arm, status=status, plan=plan, location="row")
    rows = preprocessing.expand_rows(
        [{"experiment": "t2icb_color", "prompt_id": "x_s42", "prompt": prompt, "seed": 42}],
        arm="sd14_syngen", plans_by_prompt={prompt: records["sd14_syngen"]})
    assert set(rows[0]) == preprocessing.INPUT_ROW_FIELDS and rows[0]["source_commit"] == syngen.SOURCE_COMMIT
    with pytest.raises(preprocessing.BaselineInputError):
        preprocessing.parse_plan({"components": ["a cat"], "weights": [1.0]}, arm="sd14_composable",
                                 prompt="a cat", location="row")


def test_parser_environment_strictness() -> None:
    recorded = dict(preprocessing.RECORDED_PARSER_ENVIRONMENT["sd14_syngen"], spacy_gpu_id=-1)
    preprocessing.validate_parser_environment(recorded, arm="sd14_syngen")
    reinstalled = {**recorded, "spacy_distribution_file_count": 990, "spacy_distribution_manifest_sha256": "0" * 64}
    with pytest.raises(preprocessing.BaselineInputError):
        preprocessing.validate_parser_environment(reinstalled, arm="sd14_syngen")
    preprocessing.validate_parser_environment(reinstalled, arm="sd14_syngen", strict=False)


# --- 9. Separate environments ----------------------------------------------------------------------------


def test_scripts_of_the_separate_environments_parse_as_their_python() -> None:
    ast.parse((ROOT / "scripts/t2i_compbench/prepare_baseline_inputs.py").read_text(), feature_version=(3, 11))
    ast.parse((ROOT / "scripts/t2i_compbench/official_reference/generate.py").read_text(), feature_version=(3, 10))
    assert "ccg_diffusion" not in (ROOT / "scripts/t2i_compbench/official_reference/generate.py").read_text()


# --- 10. SD1.4 physical batches --------------------------------------------------------------------------


def test_external_batches_never_mix_guided_and_fallback_rows() -> None:
    from ccg_diffusion.backbones.jobs import Job
    from ccg_diffusion.baselines.sd14_generation import PreparedJob, pad_batch, plan_batches

    def job(index: int, status: str) -> PreparedJob:
        row = Job("t2icb_color", f"p{index}", "a cat", 42, [], [], [], [], "sd14_external_sidecar")
        return PreparedJob(row, "sd14_syngen", status, None, "a", "b", "c")

    jobs = [job(index, status) for index, status in enumerate(["ok"] * 5 + ["no_target"] * 2 + ["ok"])]
    batches = plan_batches(jobs, 4)
    assert [[item.job.prompt_id for item in batch] for batch in batches] == [
        ["p0", "p1", "p2", "p3"], ["p4"], ["p5", "p6"], ["p7"]]
    padded, padding = pad_batch(batches[2], 4)
    assert padding == 2 and [item.job.prompt_id for item in padded] == ["p5", "p6", "p6", "p6"]


# --- Sana adaptations ------------------------------------------------------------------------------------


def test_sana_alignment_objectives_and_updates() -> None:
    from ccg_diffusion.baselines.sana_generation import StructureProcessor, update_latents

    offsets = {1: (0, 3), 2: (4, 7), 3: (8, 11), 4: (11, 15)}
    decoded = {1: "cat", 2: "and", 3: "sun", 4: "fish"}
    assert sana_plans.span_tokens("cat and sunfish", [8, 15], offsets, decoded) == [3, 4]
    with pytest.raises(ValueError, match="incomplete"):
        sana_plans.span_tokens("cat and sunfish", [8, 15], {3: (8, 11)}, decoded)
    # A repeated word maps to the occurrence at its character span, not to the first same-spelling token.
    repeated = {1: (0, 3), 2: (4, 7), 3: (8, 11)}
    assert sana_plans.span_tokens("cat and cat", [8, 11], repeated, {1: "cat", 2: "and", 3: "cat"}) == [3]

    maps = torch.full((16, 16, 8), 1. / 8, requires_grad=True)
    loss = sana_plans.sample_loss(maps, {"method": "sana_attend_excite", "lexical_indices": [2, 3, 4]}, [4])
    gradient = torch.autograd.grad(loss, maps)[0]
    assert gradient[..., 4].abs().sum() > 0 and gradient[..., [0, 1, 5, 6, 7]].count_nonzero() == 0

    maps = torch.rand(16, 16, 5, requires_grad=True)
    plan = {"method": "sana_pd_spacy", "text_token_count": 5,
            "pd_indices": {"exist": [4], "corr": [[1, 4]], "leak": [], "possession": []}}
    loss = sana_plans.sample_loss(maps, plan)
    assert torch.isfinite(loss) and torch.autograd.grad(loss, maps)[0][..., 4].abs().sum() > 0

    for cap in (None, .02):
        initial = torch.arange(1, 17, dtype=torch.float32).reshape(4, 1, 2, 2)
        batch = initial.clone().requires_grad_(True)
        result, _ = update_latents(batch, [row.square().sum() * (i + 1) for i, row in enumerate(batch)], 20,
                                   [True, False, True, True], cap=cap)
        for index in range(4):
            single = initial[index:index + 1].clone().requires_grad_(True)
            expected, _ = update_latents(single, [single.square().sum() * (index + 1)], 20, [index != 1], cap=cap)
            torch.testing.assert_close(result[index:index + 1], expected, rtol=0, atol=0)

    torch.manual_seed(4)
    states, full = torch.randn(4, 3, 6), torch.randn(4, 4, 6)
    keys, values = torch.randn(2, 4, 6), torch.randn(1, 4, 6)
    attn = SimpleNamespace(to_q=torch.nn.Identity(), to_k=torch.nn.Identity(), to_v=torch.nn.Identity(),
                           norm_k=None, norm_q=None, heads=1, scale=6 ** -.5,
                           to_out=[torch.nn.Identity(), torch.nn.Identity()], rescale_output_factor=1.)

    def original(a: Any, hidden: torch.Tensor, encoder: torch.Tensor, attention_mask: Any = None) -> torch.Tensor:
        return (hidden @ encoder.transpose(-1, -2) * a.scale).softmax(-1) @ encoder

    actual = StructureProcessor(original, keys, values, 2)(attn, states, full)
    conditional = torch.stack([(states[2:] @ key.T * attn.scale).softmax(-1) @ values[0] for key in keys]).mean(0)
    torch.testing.assert_close(actual, torch.cat([original(attn, states[:2], full[:2]), conditional]))


def test_sana_syngen_uses_gemma_character_spans(monkeypatch: pytest.MonkeyPatch) -> None:
    prompt = "red cat and blue dog"
    spans = {1: (0, 3), 2: (4, 7), 3: (8, 11), 4: (12, 16), 5: (17, 20)}
    decoded = {i: prompt[a:b] for i, (a, b) in spans.items()}
    monkeypatch.setattr(sana_plans, "selected_token_offsets", lambda *a, **kw: (decoded, spans))
    source = {"plan": {"source_groups": [{"extractor": "verb_root", "texts": ["blue", "dog"],
                                         "character_spans": [[12, 16], [17, 20]]}]}}
    plan = sana_plans.compile_plan("sana_syngen", prompt, source, tokenizer=None, clip_tokenizer=None, instruction=[])
    assert plan["bindings"] == [[[4], [5]]] and plan["status"] == "ok"


def test_clip_tokenizer_class_per_call_site() -> None:
    """CLIPTokenizerFast only for the Sana baseline plans and the Sana TAB Predicated Diffusion plans."""
    fast_calls = set()
    for path in sorted((ROOT / "src/ccg_diffusion").rglob("*.py")) + sorted((ROOT / "scripts").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", None)) == (
                    "load_sd14_tokenizer"):
                fast_calls |= {(path.relative_to(ROOT).as_posix(), ast.literal_eval(kw.value))
                               for kw in node.keywords if kw.arg == "fast"}
    assert fast_calls == {("src/ccg_diffusion/baselines/sana_plans.py", True),
                          ("src/ccg_diffusion/baselines/tab_sana.py", True)}


# --- TAB comparison methods ------------------------------------------------------------------------------


class SequentialWordTokenizer:
    """CLIP-like token counts for the TAB manifests: BOS, one id per word or punctuation mark, EOS."""

    model_max_length = 77

    def __call__(self, text: str, **kwargs: Any) -> dict[str, list[int]]:
        return {"input_ids": [0, *range(1, 1 + len(re.findall(r"\w+|[^\w\s]", text))), 1]}


def test_gold_visual_setup_prompt() -> None:
    from ccg_diffusion.baselines.gold_visual_setup import gold_visual_setup_prompt, terminate_sentence

    assert gold_visual_setup_prompt("The girl stands next to the cage with a dog", "dog is in the cage") == (
        "The girl stands next to the cage with a dog. dog is in the cage.")
    assert terminate_sentence(" Is the dog in the cage? ") == "Is the dog in the cage?"


def test_tab_sd14_stock_manifests_cover_all_315_items() -> None:
    from ccg_diffusion.baselines.tab_sd14 import build_baseline_manifests
    from ccg_diffusion.evaluation.tab import ITEMS
    from ccg_diffusion.provenance import read_jsonl

    vanilla, gold, records = build_baseline_manifests(read_jsonl(ITEMS), tokenizer=SequentialWordTokenizer())
    assert (len(vanilla), len(gold), len(records)) == (1575, 3150, 4725)
    assert vanilla[0] == {"experiment": "tab_ppvp_vanilla", "prompt_id": "tab-pp-000_ambiguous_s0",
                          "prompt": "The girl stands next to the cage with a dog", "seed": 0}
    assert gold[0] == {"experiment": "tab_ppvp_tied_oracle", "prompt_id": "tab-pp-000_setup0_s0",
                       "prompt": "The girl stands next to the cage with a dog. dog is in the cage.", "seed": 0}


def test_tab_pd_jobs_and_physical_batches() -> None:
    from collections import Counter

    from ccg_diffusion.baselines.gold_visual_setup import pd_baseline_batches, pd_baseline_jobs
    from ccg_diffusion.evaluation.tab import load_pp_items

    jobs = pd_baseline_jobs(list(load_pp_items().values()))
    assert Counter(job["arm"] for job in jobs) == {"pd_spacy": 370, "pd_spacy_gold": 740}
    assert len({job["key"] for job in jobs}) == 222 and len({job["image_id"] for job in jobs}) == 1110
    assert jobs[0]["image_id"] == "pd_spacy_tab-pp-000_None_s0"
    assert jobs[370]["image_id"] == "pd_spacy_gold_tab-pp-000_0_s0"
    # SD1.4: consecutive physical batches of 4 per arm (0-92 and 93-277; batch 92 holds 2 jobs).
    sd14 = pd_baseline_batches(jobs, "sd14")
    assert len(sd14) == 278 and [len(batch) for batch in sd14].count(4) == 277 and len(sd14[92]) == 2
    assert {batch[0]["arm"] for batch in sd14[:93]} == {"pd_spacy"}
    assert {job["arm"] for batch in sd14[93:] for job in batch} == {"pd_spacy_gold"}
    # Sana: per prompt key, seeds 0-3 then seed 4.
    sana = pd_baseline_batches(jobs, "sana")
    assert len(sana) == 444 and [[job["seed"] for job in batch] for batch in sana[:2]] == [[0, 1, 2, 3], [4]]
