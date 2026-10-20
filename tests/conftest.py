"""Shared fakes, builders and markers for the CPU test suite.

Tests import the helpers directly (``from conftest import PieceClipTokenizer``).
Markers (registered in ``pytest.ini``) skip a test when an optional resource is
missing:

* ``requires_clip_tokenizer``: the pinned SD1.4 CLIP tokenizer is in the Hugging Face cache;
* ``requires_gemma_tokenizer``: the pinned Sana Gemma tokenizer is in the Hugging Face cache;
* ``requires_holccg``: the ``holccg`` package is installed;
* ``requires_spacy_model``: ``en_core_web_sm`` 3.8.0 is installed.

torch is imported only by the fakes that need it.
"""
from __future__ import annotations

import ast
import importlib.util
import re
from dataclasses import dataclass
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Sequence

import pytest

from ccg_diffusion.config import SANA_GENERATION, SD14_GENERATION
from ccg_diffusion.parsing.ccg import AutoSentence, MarkedDependency, auto_sentence, parse_marked_dependency


# --- tokenizers ---------------------------------------------------------------------------------------


class Encoding(dict):
    """Tokenizer output usable both as ``encoding["input_ids"]`` and ``encoding.input_ids``."""

    @property
    def input_ids(self) -> list[int]:
        return self["input_ids"]


class PieceClipTokenizer:
    """Deterministic tokenizer with CLIP's BOS/content/EOS layout.

    Each alphanumeric run and each punctuation mark is one piece; piece IDs are
    assigned in order of first use, after BOS=0 and EOS=1.
    """

    model_max_length = 77
    bos_token_id = 0
    eos_token_id = 1

    def __init__(self) -> None:
        self._piece_to_id: dict[str, int] = {}
        self._id_to_piece: dict[int, str] = {0: "<bos>", 1: "<eos>"}

    def _pieces(self, text: str) -> list[str]:
        return re.findall(r"[A-Za-z0-9]+|[^A-Za-z0-9\s]", text.lower())

    def _piece_id(self, piece: str) -> int:
        if piece not in self._piece_to_id:
            token_id = len(self._piece_to_id) + 2
            self._piece_to_id[piece] = token_id
            self._id_to_piece[token_id] = piece
        return self._piece_to_id[piece]

    def __call__(self, text: str, *, add_special_tokens: bool = True, truncation: bool = False) -> Encoding:
        assert not truncation
        ids = [self._piece_id(piece) for piece in self._pieces(text)]
        return Encoding(input_ids=[0, *ids, 1] if add_special_tokens else ids)

    def decode(self, token_id: int) -> str:
        return self._id_to_piece[token_id]


class WordTokenizer:
    """One token per word or punctuation mark; the ID is the sum of the character codes."""

    def __call__(self, text: str, **kwargs: Any) -> dict[str, list[int]]:
        return {"input_ids": [0] + [sum(map(ord, word)) for word in re.findall(r"\w+|[^\w\s]", text.lower())] + [9999]}


# --- synthetic CCG derivations and marked dependencies --------------------------------------------------


def auto_leaf(category: str, word: str) -> str:
    """One AUTO leaf ``(<L category POS POS word category>)``."""

    return f"(<L {category} POS POS {word} {category}>)"


def synthetic_sentence(
    prompt: str, categories: Sequence[str], *, root_category: str = "S", id_line: str = "ID=test.1",
) -> AutoSentence:
    """A flat AUTO derivation whose leaves are the whitespace tokens of ``prompt``."""

    words = prompt.split()
    assert len(words) == len(categories)
    derivation = (
        f"(<T {root_category} 0 {len(words)}> "
        + " ".join(auto_leaf(category, word) for category, word in zip(categories, words))
        + ")"
    )
    return auto_sentence(id_line, derivation)


def marked_dependencies(*lines: str) -> list[MarkedDependency]:
    """Parse ``generate -e`` lines such as ``"green_2 (N{Y}/N{Y}<1>){_} 1 bench_3 0"``."""

    return [parse_marked_dependency(line) for line in lines]


# --- Positional Prefix sources (shared by the Positional Prefix and Region Assignment Loss tests) --------


def one_token_alignment(words: list[str], roles: dict[int, str]) -> list[dict[str, Any]]:
    """Alignment rows with one CLIP token per word: ``clip_index`` equals the 1-based word position."""

    rows, cursor = [], 0
    for index, word in enumerate(words, 1):
        rows.append({"source_index": index, "text": word, "structural_role": roles.get(index, "out_of_contract"),
                     "clip_index": index, "clip_lexical_piece_indices": [index], "char_start": cursor,
                     "char_end": cursor + len(word)})
        cursor += len(word) + 1
    return rows


def prefix_guidance_record(prompt: str, program: dict[str, Any], profile: str) -> dict[str, Any]:
    """The fields of a ``SelectiveBoostGuidance`` record that the Positional Prefix rules read."""

    from ccg_diffusion.provenance import canonical_json, sha256_text

    evidence = {"count_tokens": [], "parse_status": "complete", "unresolved_modifiers": [], "depth_cues": []}
    return {"source_prompt_sha256": sha256_text(prompt), "source_program_sha256": sha256_text(canonical_json(program)),
            "effective_profile": profile, "decision": {"evidence": {"original_decision": {"evidence": evidence}}}}


def prefix_source(first: str = "blue", second: str = "red", connection: str = "and") -> dict[str, Any]:
    """``a <first> box <connection> a <second> vase`` with two modifier bindings (route P2)."""

    prompt = f"a {first} box {connection} a {second} vase"
    program = {"exist": [3, 7], "corr": [[2, 3], [6, 7]], "leak": [[2, 7], [6, 3]], "possession": []}
    roles = {2: "modifier", 6: "modifier", 3: "entity", 7: "entity"}
    return {"prompt": prompt, "alignment": one_token_alignment(prompt.split(), roles), "program": program,
            "guidance": prefix_guidance_record(prompt, program, "P2"), "count": {"decision": {"constraints": []}}}


def phrase_plan_of(source: dict[str, Any]) -> Any:
    """The Positional Prefix phrase plan of a ``prefix_source`` with the word tokenizer."""

    from ccg_diffusion.guidance.positional_prefix import build_phrase_plan

    return build_phrase_plan(source["prompt"], source["alignment"], source["program"], source["guidance"],
                             source["count"], WordTokenizer())


def screening_stubs(source: dict[str, Any]) -> tuple[Any, Any]:
    """Inactive record-only depth and clause plans with the identity of ``source``."""

    from ccg_diffusion.guidance._record_only_screening import ClauseLayoutPlan, DepthLayoutPlan
    from ccg_diffusion.provenance import canonical_json, sha256_text

    identity = (sha256_text(source["prompt"]), sha256_text(canonical_json(source["program"])))
    return (DepthLayoutPlan(*identity, None, None, None, (), (), None, None, "inactive", "{}"),
            ClauseLayoutPlan(*identity, None, None, None, (), (), None, 0.0, "inactive", "{}"))


# --- sealed source files ------------------------------------------------------------------------------


def import_closure(module: Path) -> set[str]:
    """Repository-relative paths of the ccg_diffusion modules that ``module`` imports, transitively.

    Only absolute imports are followed; package ``__init__.py`` files are excluded.
    """
    root = Path(__file__).resolve().parents[1]
    seen: set[Path] = set()
    todo = [module]
    while todo:
        path = todo.pop()
        if path in seen:
            continue
        seen.add(path)
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module, *(f"{node.module}.{alias.name}" for alias in node.names)]
            for name in names:
                target = root / "src" / Path(*name.split("."))
                for candidate in (target / "__init__.py", target.with_suffix(".py")):
                    if name.startswith("ccg_diffusion") and candidate.is_file():
                        todo.append(candidate)
    return {path.relative_to(root).as_posix() for path in seen if path.name != "__init__.py"}


# --- fake models --------------------------------------------------------------------------------------


class FakeSDPipe:
    """SD1.4-shaped stand-in: identity ``unet.attn2.to_k``/``to_v`` and a counting ``encode_prompt``."""

    def __init__(self, dtype: Any = None, hidden_size: int = 4) -> None:
        import torch

        self.dtype = dtype if dtype is not None else torch.float32
        self.hidden_size = hidden_size
        self.unet = torch.nn.Module()
        self.unet.attn2 = torch.nn.Module()
        self.unet.attn2.to_v = torch.nn.Identity()
        self.unet.attn2.to_k = torch.nn.Identity()
        self.calls = 0

    def encode_prompt(self, prompts: list[str], **kwargs: Any) -> tuple[Any, None]:
        import torch

        self.calls += 1
        rows = [torch.full((77, self.hidden_size), sum(map(ord, p)), dtype=self.dtype) for p in prompts]
        return torch.stack(rows), None


@dataclass(frozen=True)
class FakeSpacyToken:
    text: str
    lemma_: str
    idx: int


class FakeNLP:
    """A spaCy pipeline stand-in that returns fixed tokens for any prompt."""

    def __init__(self, tokens: Sequence[FakeSpacyToken]) -> None:
        self.tokens = tokens

    def __call__(self, _prompt: str) -> Sequence[FakeSpacyToken]:
        return self.tokens


# --- Sana fakes ---------------------------------------------------------------------------------------


class FakeGemmaTokenizer:
    """SentencePiece-like stand-in for Sana's Gemma tokenizer: BOS=2, PAD=0, no EOS.

    A piece is a word with its leading whitespace, or trailing whitespace on its
    own; so the instruction prompt, which ends in a space, loses its last piece
    once the prompt is appended, as with Gemma. Piece IDs follow first use.
    """

    def __init__(self) -> None:
        self.padding_side = "left"
        self._ids: dict[str, int] = {}
        self._pieces: dict[int, str] = {0: "<pad>", 2: "<bos>"}

    def _encode(self, text: str) -> list[tuple[int, int, int]]:
        tokens = [(2, 0, 0)]
        for match in re.finditer(r"\s*\S+|\s+", text):
            piece = match.group(0)
            if piece not in self._ids:
                self._ids[piece] = len(self._ids) + 3
                self._pieces[self._ids[piece]] = piece
            tokens.append((self._ids[piece], match.start(), match.end()))
        return tokens

    def encode(self, text: str) -> list[int]:
        return [token_id for token_id, _, _ in self._encode(text)]

    def __call__(self, text: str, *, padding: str, max_length: int, truncation: bool, add_special_tokens: bool,
                 return_tensors: str, return_offsets_mapping: bool = False) -> Any:
        import torch

        assert padding == "max_length" and truncation and add_special_tokens and return_tensors == "pt"
        tokens = self._encode(text)[:max_length]
        padding_tokens = [(0, 0, 0)] * (max_length - len(tokens))
        tokens = tokens + padding_tokens if self.padding_side == "right" else padding_tokens + tokens
        encoding = SimpleNamespace(input_ids=torch.tensor([[t[0] for t in tokens]]),
                                   attention_mask=torch.tensor([[int(t[0] != 0) for t in tokens]]))
        if return_offsets_mapping:
            encoding.offset_mapping = torch.tensor([[[t[1], t[2]] for t in tokens]])
        return encoding

    def decode(self, token_ids: Sequence[int]) -> str:
        return "".join(self._pieces[int(token_id)] for token_id in token_ids)


class FakeSanaVae:
    """DC-AE stand-in with 32x compression: the first three latent channels, upsampled and squashed."""

    def __init__(self) -> None:
        import torch

        self.dtype = torch.float32
        self.config = SimpleNamespace(scaling_factor=0.41407, encoder_block_out_channels=[1] * 6)

    def to(self, dtype: Any) -> "FakeSanaVae":
        self.dtype = dtype
        return self

    def decode(self, latents: Any, return_dict: bool = False) -> tuple[Any]:
        import torch

        return (torch.nn.functional.interpolate(latents[:, :3], scale_factor=32, mode="nearest").tanh(),)


# Scheduler of the pinned Sana 0.6B 512px checkpoint (scheduler/scheduler_config.json).
SANA_SCHEDULER_CONFIG = {
    "algorithm_type": "dpmsolver++", "beta_end": 0.02, "beta_schedule": "linear", "beta_start": 0.0001,
    "dynamic_thresholding_ratio": 0.995, "euler_at_final": False, "final_sigmas_type": "zero", "flow_shift": 3.0,
    "lower_order_final": True, "num_train_timesteps": 1000, "prediction_type": "flow_prediction",
    "rescale_betas_zero_snr": False, "sample_max_value": 1.0, "solver_order": 2, "solver_type": "midpoint",
    "steps_offset": 0, "thresholding": False, "timestep_spacing": "linspace", "use_flow_sigmas": True,
}
SANA_TEXT_AXIS = 32  # columns of the fake Gemma encodings
SANA_CAPTION_CHANNELS = 16


def tiny_sana_pipe(*, seed: int = 0, dtype: Any = None) -> Any:
    """A real ``SanaPipeline`` around an 8-block, randomly initialized transformer (16x16 latent grid).

    The scheduler is Sana's own DPM-Solver++ flow configuration; the tokenizer
    and the VAE are the fakes above; there is no text encoder (encodings are
    supplied by the caller). No model weights are downloaded.
    """
    import torch
    from diffusers import DPMSolverMultistepScheduler, SanaPipeline, SanaTransformer2DModel

    torch.manual_seed(seed)
    transformer = SanaTransformer2DModel(
        in_channels=4, out_channels=4, num_attention_heads=2, attention_head_dim=8, num_layers=8,
        num_cross_attention_heads=2, cross_attention_head_dim=8, cross_attention_dim=16,
        caption_channels=SANA_CAPTION_CHANNELS, sample_size=16, patch_size=1,
    )
    if dtype is not None:
        transformer.to(dtype)
    pipe = SanaPipeline(tokenizer=FakeGemmaTokenizer(), text_encoder=None, vae=FakeSanaVae(),
                        transformer=transformer, scheduler=DPMSolverMultistepScheduler(**SANA_SCHEDULER_CONFIG))
    pipe.transformer.eval().requires_grad_(False)
    return pipe


def sana_prompt_inputs(prompt: str = "a red cube and a blue sphere", *, seed: int = 7) -> tuple[dict[str, Any], Any]:
    """A guidance plan and a cached prompt encoding of ``SANA_TEXT_AXIS`` columns for ``prompt``.

    The real tokens are BOS plus one per word; the negative encoding has only BOS.
    """
    import torch

    generator = torch.Generator().manual_seed(seed)
    real = 1 + len(prompt.split())
    embeds = torch.randn(1, SANA_TEXT_AXIS, SANA_CAPTION_CHANNELS, generator=generator)
    mask = torch.zeros(1, SANA_TEXT_AXIS, dtype=torch.int64)
    mask[0, :real] = 1
    negative = torch.randn(1, SANA_TEXT_AXIS, SANA_CAPTION_CHANNELS, generator=generator)
    negative_mask = torch.zeros(1, SANA_TEXT_AXIS, dtype=torch.int64)
    negative_mask[0, 0] = 1
    program = {"exist": [2, 6], "corr": [], "leak": [], "possession": []}
    plan = {"conditioning": prompt, "text_token_count": real, "source_program": program,
            "lexical_projection": [{"source_clip_anchor": 2, "target_span": [2]},
                                   {"source_clip_anchor": 6, "target_span": [6]}]}
    return plan, (embeds, mask, negative, negative_mask)


class QuadraticSanaObjective:
    """``SanaGuidanceObjective`` stand-in: ``weight * sum(maps[..., 1:] ** 2)``; records the steps it saw."""

    def __init__(self, plan: dict[str, Any], indices: Any, *, scale_multiplier: float = 1.0,
                 max_update_ratio: float = 0.02, weight: float = 1.0) -> None:
        self.plan = plan
        self.indices = indices
        self.scale_multiplier = scale_multiplier
        self.max_update_ratio = max_update_ratio
        self.weight = weight
        self.update_diagnostics: dict[str, Any] = {}
        self.steps: list[int] = []
        self.losses: list[float] = []

    def __call__(self, attention: Any, step_index: int) -> Any:
        loss = self.weight * attention[..., 1:].square().sum()
        self.steps.append(step_index)
        self.losses.append(float(loss.detach()))
        return loss


# --- markers -----------------------------------------------------------------------------------------


def _cached(repo_id: str, filename: str, revision: str) -> bool:
    from huggingface_hub import try_to_load_from_cache

    return isinstance(try_to_load_from_cache(repo_id, filename, revision=revision), str)


def _clip_tokenizer_available() -> bool:
    return _cached(SD14_GENERATION.model_id, f"{SD14_GENERATION.tokenizer_subfolder}/vocab.json",
                   SD14_GENERATION.model_revision)


def _gemma_tokenizer_available() -> bool:
    return _cached(SANA_GENERATION.model_id, "tokenizer/tokenizer.json", SANA_GENERATION.model_revision)


def _holccg_available() -> bool:
    return importlib.util.find_spec("holccg") is not None


def _spacy_model_available() -> bool:
    try:
        return metadata.version("en-core-web-sm") == "3.8.0"
    except metadata.PackageNotFoundError:
        return False


REQUIREMENTS: dict[str, tuple[Callable[[], bool], str]] = {
    "requires_clip_tokenizer": (_clip_tokenizer_available, "the pinned SD1.4 CLIP tokenizer is not cached"),
    "requires_gemma_tokenizer": (_gemma_tokenizer_available, "the pinned Sana Gemma tokenizer is not cached"),
    "requires_holccg": (_holccg_available, "holccg is not installed"),
    "requires_spacy_model": (_spacy_model_available, "en_core_web_sm 3.8.0 is not installed"),
}


@lru_cache(maxsize=None)
def available(marker: str) -> bool:
    return REQUIREMENTS[marker][0]()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        for marker, (_, reason) in REQUIREMENTS.items():
            if item.get_closest_marker(marker) is not None and not available(marker):
                item.add_marker(pytest.mark.skip(reason=reason))
