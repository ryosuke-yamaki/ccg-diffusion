"""Positional Prefix (paper Sec. IV-B2, Eq. 4) for two coordinated noun phrases.

Eligibility (``build_phrase_plan``): a complete parse without count words,
count constraints, depth words or an existing Spatial Relation route, and two
contiguous noun phrases whose nouns carry color, texture, shape or size
modifiers, joined by "and". For an eligible prompt ``c`` with phrases NP1 and
NP2 (in source order), the conditioning text is::

    "On the left side, NP1. On the right side, NP2. " + c

``build_prefix_plan`` maps every original lexical CLIP position bijectively to
its copy in the prefix (the two noun phrases) or in the unchanged prompt
suffix, translates all guidance targets (E, C, K, P) and whole-word spans
accordingly, and records the word groups of the Region Assignment Loss (the
noun and its modifiers, left group first). ``prefix_applies`` is true exactly
for such phrase-branch plans; other recorded plan branches (depth, clause) are
record-only (see ``_record_only_screening``).

``prefix_conditioning`` is the SD1.4 K/V hook: it encodes the conditioning
texts of the active rows once per physical batch and replaces the conditional
cross-attention K/V inputs of those rows; the classifier-free-guidance
unconditional rows and all other rows are unchanged.
"""
from __future__ import annotations

import json
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterator, Mapping, NamedTuple, Protocol, Sequence

import torch

from ccg_diffusion.guidance._record_only_screening import ClauseLayoutPlan, DepthLayoutPlan, screen_layouts
from ccg_diffusion.guidance.count_loss import NounForms
from ccg_diffusion.guidance.selective_guidance_boost import ATTRIBUTE_ADDITIONS, VISUAL_ADJECTIVES, GuidanceRoute
from ccg_diffusion.parsing.ccg import AutoSentence, MarkedDependency
from ccg_diffusion.provenance import canonical_json, sha256_text

PHRASE_PLAN_VERSION = "holsd.sd14_phrase_binding.v1"
PREFIX_PLAN_VERSION = "holsd.sd14_prefix_layout.v1"
TEXTURE_WORDS = frozenset().union(
    VISUAL_ADJECTIVES["material_adjective"], VISUAL_ADJECTIVES["surface"], ATTRIBUTE_ADDITIONS["material"])
SHAPE_WORDS = VISUAL_ADJECTIVES["shape"] | ATTRIBUTE_ADDITIONS["size_and_shape"]
ATTRIBUTE_WORDS = VISUAL_ADJECTIVES["color"] | TEXTURE_WORDS | SHAPE_WORDS
RELATIONAL_WORDS = frozenset("above below over under underneath atop beside near inside outside behind on".split())
# Reason of a conjunction in which a shape or size modifier binds one of the two nouns.
SHAPE_CONJUNCTION_REASON = "two_attribute_nominals_in_conjunction"
# Position cues per plan relation. Only "left" (the phrase branch) is ever applied;
# the other entries appear in record-only depth/clause-branch plans.
LOCATIONS = {"left": ("On the left side, ", "On the right side, "),
             "right": ("On the right side, ", "On the left side, "),
             "top": ("In the upper part of the image, ", "In the lower part of the image, "),
             "bottom": ("In the lower part of the image, ", "In the upper part of the image, "),
             "next_to": ("On the left side, ", "On the right side, "),
             "depth": ("In the foreground, close to the camera, ", "Farther away in the background, ")}
# Recorded selection of the prompt's recipe branch (KEEP).
APPLIED_SELECTION = {"branch": "phrase", "experiment": "prefix_layout", "condition": "source_prefix_regions"}
UNSELECTED = {"branch": "unchanged", "experiment": "none", "condition": "incumbent_repeat"}


# --- eligibility: two coordinated noun phrases ----------------------------------------------------------


@dataclass(frozen=True)
class PhraseSpan:
    """An original contiguous modifier-bearing nominal, including CLIP BOS offset."""

    text: str
    token_indices: tuple[int, ...]
    source_indices: tuple[int, ...]
    entity_anchor: int

    def as_record(self) -> dict[str, Any]:
        return {**asdict(self), "token_indices": list(self.token_indices), "source_indices": list(self.source_indices)}


@dataclass(frozen=True)
class PhraseBindingPlan:
    prompt_sha256: str
    program_sha256: str
    alignment_sha256: str
    phrases: tuple[PhraseSpan, ...]
    reason: str

    @property
    def active(self) -> bool:
        return bool(self.phrases)

    def as_record(self) -> dict[str, Any]:
        return {"schema": PHRASE_PLAN_VERSION, "prompt_sha256": self.prompt_sha256,
                "program_sha256": self.program_sha256, "alignment_sha256": self.alignment_sha256,
                "reason": self.reason, "phrases": [p.as_record() for p in self.phrases]}


def build_phrase_plan(
    prompt: str,
    alignment: Sequence[Mapping[str, Any]],
    program: Mapping[str, Any],
    guidance: Mapping[str, Any],
    count: Mapping[str, Any],
    tokenizer: Any,
) -> PhraseBindingPlan:
    """Find the two modifier-bearing noun phrases of a coordination from authenticated source coordinates.

    ``guidance`` and ``count`` are the ``as_record()`` dicts of the prompt's
    ``SelectiveBoostGuidance`` and ``CountGuidance``.
    """
    identity = (sha256_text(prompt), sha256_text(canonical_json(program)), sha256_text(canonical_json(alignment)))

    def inactive(reason: str) -> PhraseBindingPlan:
        return PhraseBindingPlan(*identity, (), reason)

    if guidance["source_prompt_sha256"] != identity[0] or guidance["source_program_sha256"] != identity[1]:
        raise ValueError("phrase source differs from the incumbent guidance")
    evidence = guidance["decision"]["evidence"]["original_decision"]["evidence"]
    if count["decision"]["constraints"] or evidence["count_tokens"]:
        return inactive("retain_COUNT_or_count_cue")
    if guidance["effective_profile"] == GuidanceRoute.SPATIAL_RELATION:
        return inactive("retain_existing_P1")
    if evidence["parse_status"] != "complete" or evidence["unresolved_modifiers"]:
        return inactive("retain_incomplete_or_unresolved_source")
    if evidence["depth_cues"]:
        return inactive("depth_requires_separate_intervention")
    tokens = tokenizer(prompt, truncation=False)["input_ids"]
    if len(tokens) > 77:
        raise ValueError("full prompt exceeds CLIP context")
    projected = [r for r in alignment if r.get("clip_index") is not None]
    by_anchor = {r["clip_index"]: r for r in projected}
    if len(by_anchor) != len(projected):
        return inactive("ambiguous_source_anchors")
    groups: dict[int, list[int]] = {}
    for modifier, entity in program["corr"]:
        groups.setdefault(entity, []).append(modifier)
    phrases, visual_groups = [], []
    for entity, modifiers in sorted(groups.items()):
        entity_row = by_anchor[entity]
        if entity_row["structural_role"] != "entity" or entity not in program["exist"]:
            raise ValueError("phrase lacks a source-bound entity")
        words = {by_anchor[m]["text"].lower() for m in modifiers
                 if by_anchor[m]["structural_role"] == "modifier"}
        if not words.intersection(ATTRIBUTE_WORDS):
            continue
        start = min(by_anchor[m]["source_index"] for m in modifiers)
        end = entity_row["source_index"]
        if start >= end or any(by_anchor[m]["source_index"] >= end for m in modifiers):
            return inactive("noncontiguous_or_postnominal_binding")
        rows = [r for r in alignment if start <= r["source_index"] <= end]
        if any(r.get("clip_index") is None or r["structural_role"] not in {"modifier", "entity"} for r in rows):
            return inactive("phrase_crosses_non_nominal_source")
        if any(r["structural_role"] == "entity" and r["clip_index"] != entity for r in rows):
            return inactive("phrase_contains_another_entity")
        # Include the adjacent article, when present, without inventing prompt tokens.
        previous = next((r for r in alignment if r["source_index"] == start - 1), None)
        if previous is not None and previous["text"].lower() in {"a", "an", "the"}:
            rows.insert(0, previous)
        pieces = tuple(i for r in rows for i in r["clip_lexical_piece_indices"])
        if not pieces or pieces != tuple(range(pieces[0], pieces[-1] + 1)):
            return inactive("noncontiguous_lexical_pieces")
        text = prompt[rows[0]["char_start"]:rows[-1]["char_end"]]
        if tokenizer(text, truncation=False)["input_ids"][1:-1] != tokens[pieces[0]:pieces[-1] + 1]:
            raise ValueError("isolated phrase differs from full-prompt lexical token identity")
        phrases.append(PhraseSpan(text, pieces, tuple(r["source_index"] for r in rows), entity))
        visual_groups.append(words.intersection(ATTRIBUTE_WORDS))
    if len(phrases) < 2 or len(set(p.entity_anchor for p in phrases)) != len(phrases):
        return inactive("fewer_than_two_visual_nominals")
    pieces = [i for phrase in phrases for i in phrase.token_indices]
    if len(pieces) != len(set(pieces)):
        return inactive("overlapping_phrases")
    middle = [r["text"].lower() for r in alignment
              if phrases[0].source_indices[-1] < r["source_index"] < phrases[-1].source_indices[0]]
    relational = bool(RELATIONAL_WORDS.intersection(middle)) or any(
        tuple(middle[i:i + 2]) in {("next", "to"), ("left", "of"), ("right", "of"), ("top", "of")}
        for i in range(len(middle) - 1))
    if relational:
        return PhraseBindingPlan(*identity, tuple(phrases), "two_visual_nominals_with_relation")
    if "and" not in middle:
        return inactive("retain_shape_size_or_unsupported_connection")
    # Every visual group intersects ATTRIBUTE_WORDS; a group without shape/size
    # words therefore intersects the color and texture words.
    if any(words.intersection(SHAPE_WORDS) for words in visual_groups):
        return PhraseBindingPlan(*identity, tuple(phrases), SHAPE_CONJUNCTION_REASON)
    return PhraseBindingPlan(*identity, tuple(phrases), "two_color_or_texture_nominals_in_conjunction")


# --- prefix text and token remapping --------------------------------------------------------------------


def lexical_mapping(
    prompt: str, prefix: str, alignment: Sequence[Mapping[str, Any]],
    copies: Sequence[tuple[Sequence[int], int]], tokenizer: Any,
) -> tuple[list[int], list[int], dict[int, int]]:
    """Map original tokens to copied nominals or the unchanged prompt suffix.

    A copy pairs original word indices with its character offset in the prefix.
    Every lexical piece is checked against the actual tokenizer in both contexts.
    """
    conditioning = prefix + prompt
    old_ids = tokenizer(prompt, truncation=False)["input_ids"]
    new_ids = tokenizer(conditioning, truncation=False)["input_ids"]
    if len(new_ids) > 77:
        raise ValueError("caption exceeds CLIP's lexical window")
    prefix_ids = tokenizer(prefix, truncation=False)["input_ids"]
    offset = len(prefix_ids) - 2
    if (new_ids[:offset + 1] != prefix_ids[:-1]
            or new_ids[offset + 1:-1] != old_ids[1:-1]):
        raise ValueError("caption changed the original-prompt suffix tokens")
    mapping = {i: i + offset for i in range(1, len(old_ids) - 1)}
    by_source = {row["source_index"]: row for row in alignment}
    copied: set[int] = set()
    for source_indices, start in copies:
        if not source_indices or copied.intersection(source_indices):
            raise ValueError("empty or overlapping nominal copies")
        copied.update(source_indices)
        original_start = by_source[source_indices[0]]["char_start"]
        for source_index in source_indices:
            row = by_source[source_index]
            begin = start + row["char_start"] - original_start
            end = start + row["char_end"] - original_start
            before = tokenizer(conditioning[:begin], truncation=False)["input_ids"][:-1]
            through = tokenizer(conditioning[:end], truncation=False)["input_ids"][:-1]
            positions = list(range(len(before), len(through)))
            pieces = row["clip_lexical_piece_indices"]
            if (new_ids[:len(before)] != before or new_ids[:len(through)] != through
                    or len(positions) != len(pieces)
                    or [old_ids[i] for i in pieces] != [new_ids[i] for i in positions]):
                raise ValueError("caption nominal changed a source lexical token")
            mapping.update(zip(pieces, positions, strict=True))
    if len(set(mapping.values())) != len(mapping) or any(not 1 <= i < len(new_ids) - 1 for i in mapping.values()):
        raise ValueError("caption mapping is not one-to-one in the real lexical window")
    return old_ids, new_ids, mapping


@dataclass(frozen=True)
class PrefixLayoutPlan:
    """The Positional Prefix plan of one prompt (field names and order are recorded)."""

    prompt_sha256: str
    program_sha256: str
    alignment_sha256: str
    conditioning: str | None
    original_token_count: int
    conditioning_token_count: int
    source_branch: str
    relation: str | None
    mapping: tuple[tuple[int, int], ...]
    mapped_program_json: str
    groups: tuple[tuple[tuple[int, ...], ...], ...]
    phantom_anchor: int | None
    source_plan_sha256: str
    reason: str

    @property
    def active(self) -> bool:
        return self.conditioning is not None

    @property
    def mapped_program(self) -> dict[str, Any]:
        return json.loads(self.mapped_program_json)

    def as_record(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("mapped_program_json")
        return {**value, "schema": PREFIX_PLAN_VERSION, "mapping": [list(pair) for pair in self.mapping],
                "mapped_program": self.mapped_program, "groups": [[list(p) for p in g] for g in self.groups]}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> PrefixLayoutPlan:
        """Rebuild a plan from its ``as_record()`` dict (as stored in TAB job rows)."""
        record = dict(record)
        record.pop("schema", None)
        record["mapped_program_json"] = canonical_json(record.pop("mapped_program"))
        record["groups"] = tuple(tuple(tuple(p) for p in group) for group in record["groups"])
        record["mapping"] = tuple(tuple(p) for p in record["mapping"])
        return cls(**record)


def build_prefix_plan(
    prompt: str, alignment: Sequence[Mapping[str, Any]], program: Mapping[str, Any],
    phrase: PhraseBindingPlan, tokenizer: Any, *, depth: DepthLayoutPlan, clause: ClauseLayoutPlan,
) -> PrefixLayoutPlan:
    """Compile the Positional Prefix plan from the phrase plan and the record-only depth/clause plans."""
    identity = (sha256_text(prompt), sha256_text(canonical_json(program)), sha256_text(canonical_json(alignment)))
    if (phrase.prompt_sha256, phrase.program_sha256, phrase.alignment_sha256) != identity:
        raise ValueError("prefix source differs from its authenticated phrase plan")
    if any((plan.prompt_sha256, plan.program_sha256) != identity[:2] for plan in (depth, clause)):
        raise ValueError("prefix plans refer to different source inputs")
    old_ids = tokenizer(prompt, truncation=False)["input_ids"]
    source_records = {"depth": depth.as_record(), "clause": clause.as_record(), "phrase": phrase.as_record()}
    digest = sha256_text(canonical_json(source_records))

    def inactive(reason: str) -> PrefixLayoutPlan:
        return PrefixLayoutPlan(*identity, None, len(old_ids), 0, "unchanged", None, (), "{}", (), None, digest, reason)

    by_source = {r["source_index"]: r for r in alignment}
    by_anchor = {r["clip_index"]: r for r in alignment if r.get("clip_index") is not None}
    phantom = None
    if depth.active:
        nominals = []
        for anchor in (depth.front_anchor, depth.back_anchor):
            row = by_anchor[anchor]
            rows = [row]
            previous = by_source.get(row["source_index"] - 1)
            if previous is not None and previous["text"].lower() in {"a", "an", "the"}:
                rows.insert(0, previous)
            nominals.append(PhraseSpan(prompt[rows[0]["char_start"]:row["char_end"]],
                                       tuple(i for r in rows for i in r["clip_lexical_piece_indices"]),
                                       tuple(r["source_index"] for r in rows), anchor))
        relation, branch, phantom = "depth", "depth", depth.phantom_anchor
    elif phrase.active and len(phrase.phrases) == 2:
        if clause.active:
            entities = {p.entity_anchor: p for p in phrase.phrases}
            if set(entities) != {clause.subject_anchor, clause.object_anchor}:
                return inactive("clause_endpoints_differ_from_nominals")
            nominals = [entities[clause.subject_anchor], entities[clause.object_anchor]]
            relation, branch, phantom = clause.relation, "clause", clause.phantom_anchor
        elif phrase.reason in {"two_color_or_texture_nominals_in_conjunction", SHAPE_CONJUNCTION_REASON}:
            nominals = list(phrase.phrases)
            relation, branch = "left", "phrase"
        else:
            return inactive("retain_other_relational_input")
    else:
        return inactive("requires_two_supported_source_nominals")
    if relation not in LOCATIONS:
        return inactive("unsupported_source_direction")
    prefix, copies = "", []
    for nominal, cue in zip(nominals, LOCATIONS[relation], strict=True):
        prefix += cue
        copies.append((nominal, len(prefix)))
        prefix += nominal.text + ". "
    conditioning = prefix + prompt
    new_ids = tokenizer(conditioning, truncation=False)["input_ids"]
    if len(new_ids) > 77:
        return inactive("conditioning_would_truncate")
    _, _, mapping = lexical_mapping(
        prompt, prefix, alignment, [(nominal.source_indices, start) for nominal, start in copies], tokenizer)
    mapped = {"exist": [mapping[i] for i in program["exist"]]}
    for field in ("corr", "leak", "possession"):
        mapped[field] = [[mapping[i] for i in pair] for pair in program[field]]
    groups = []
    for nominal in nominals:
        anchors = [nominal.entity_anchor] + [m for m, n in program["corr"] if n == nominal.entity_anchor]
        if len(anchors) != len(set(anchors)):
            raise ValueError("duplicate nominal attention anchor")
        words = []
        for anchor in anchors:
            pieces = by_anchor[anchor]["clip_lexical_piece_indices"]
            if not pieces or pieces[-1] != anchor or not set(pieces) <= set(nominal.token_indices):
                raise ValueError("prefix word span differs from its nominal")
            words.append(tuple(mapping[i] for i in pieces))
        groups.append(tuple(words))
    return PrefixLayoutPlan(*identity, conditioning, len(old_ids), len(new_ids), branch, relation,
                            tuple(sorted(mapping.items())), canonical_json(mapped), tuple(groups),
                            mapping[phantom] if phantom is not None else None, digest,
                            "source_bound_spatial_prefix" if branch != "phrase" else "permitted_conjunction_prefix")


def prefix_applies(plan: PrefixLayoutPlan) -> bool:
    """Whether the Positional Prefix and the Region Assignment Loss apply to the prompt."""
    return plan.active and plan.source_branch == "phrase"


def selection_record(plan: PrefixLayoutPlan) -> dict[str, str]:
    """The recorded recipe-branch selection of the prompt."""
    return dict(APPLIED_SELECTION) if prefix_applies(plan) else dict(UNSELECTED)


class PrefixDerivation(NamedTuple):
    phrase: PhraseBindingPlan
    depth: DepthLayoutPlan
    clause: ClauseLayoutPlan
    prefix: PrefixLayoutPlan


def derive_positional_prefix(
    prompt: str, sentence: AutoSentence, dependencies: Sequence[MarkedDependency],
    alignment: Sequence[Mapping[str, Any]], program: Mapping[str, Any],
    guidance: Mapping[str, Any], count: Mapping[str, Any], tokenizer: Any, noun_forms: NounForms,
) -> PrefixDerivation:
    """Phrase plan, record-only depth/clause plans and the Positional Prefix plan of one prompt.

    ``guidance`` and ``count`` are the ``as_record()`` dicts of the prompt's
    ``SelectiveBoostGuidance`` and ``CountGuidance``; ``tokenizer`` is the
    pinned SD1.4 CLIP tokenizer.
    """
    phrase = build_phrase_plan(prompt, alignment, program, guidance, count, tokenizer)
    depth, clause = screen_layouts(prompt, sentence, dependencies, alignment, program, guidance, count,
                                   tokenizer, noun_forms)
    prefix = build_prefix_plan(prompt, alignment, program, phrase, tokenizer, depth=depth, clause=clause)
    return PrefixDerivation(phrase, depth, clause, prefix)


# --- SD1.4 K/V conditioning hook ------------------------------------------------------------------------


class ProjectionContextCache:
    """Reuse identical projection inputs, respecting tensor versions and autograd.

    Strong references prevent storage-pointer reuse. Shape, stride and version
    distinguish views and in-place writes. Grad-bearing and inference tensors
    bypass caching, so no graph is retained across backward passes. The cache is
    local to the hook context and contains at most four small text tensors.
    Each entry is checked against the original computation at its first hit,
    including on the actual GPU, before later immutable reuse is permitted.
    """

    def __init__(self, limit: int = 4) -> None:
        if type(limit) is not int or limit < 1:
            raise ValueError("context cache limit must be positive")
        self.limit = limit
        self.entries: OrderedDict[tuple[object, ...], tuple[torch.Tensor, torch.Tensor, int, bool]] = OrderedDict()
        self.builds = 0
        self.hits = 0
        self.verifications = 0

    def replace(self, original: torch.Tensor, build: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        key = None
        if not original.requires_grad:
            try:
                version = original._version
            except RuntimeError:  # Inference tensors have no version counter.
                pass
            else:
                key = (original.device, original.dtype, original.data_ptr(), tuple(original.shape),
                       tuple(original.stride()), version)
        if key is not None and key in self.entries:
            entry = self.entries[key]
            if entry[1]._version == entry[2]:
                if not entry[3]:
                    reference = build(original)
                    self.builds += 1
                    self.verifications += 1
                    if not torch.equal(entry[1], reference):
                        raise ValueError("cached text context differs from the original replacement computation")
                    self.entries[key] = (*entry[:3], True)
                self.hits += 1
                self.entries.move_to_end(key)
                return entry[1]
            del self.entries[key]
        result = build(original)
        self.builds += 1
        if key is not None and not result.requires_grad:
            try:
                result_version = result._version
            except RuntimeError:
                pass
            else:
                self.entries[key] = (original, result, result_version, False)
                self.entries.move_to_end(key)
                while len(self.entries) > self.limit:
                    self.entries.popitem(last=False)
        return result


class ConditioningPlan(Protocol):
    """Read-only inputs of a conditional text replacement (``PrefixLayoutPlan`` satisfies it)."""
    @property
    def prompt_sha256(self) -> str: ...

    @property
    def conditioning(self) -> str | None: ...

    @property
    def active(self) -> bool: ...


@contextmanager
def prefix_conditioning(
    pipe: Any, prompts: Sequence[str], plans: Sequence[ConditioningPlan], *, device: str,
) -> Iterator[dict[str, Any]]:
    """Use the Positional Prefix text only for the conditional K/V rows of active plans.

    All texts of the physical batch are encoded in one ``encode_prompt`` call
    (inactive rows encode their own prompt), so the batch composition of the
    text encoder is fixed. The hooks are removed on exit.
    """
    if len(prompts) != len(plans):
        raise ValueError("invalid prefix context batch")
    if any(p.prompt_sha256 != sha256_text(text) for p, text in zip(plans, prompts)):
        raise ValueError("prefix context prompt identity differs")
    active = [i for i, plan in enumerate(plans) if plan.active]
    audit = {"projection_calls": 0, "active_rows": active, "context_cache_builds": 0,
             "context_cache_hits": 0, "context_cache_verifications": 0}
    if not active:
        yield audit
        return
    with torch.no_grad():
        texts = [p.conditioning if i in active else prompts[i] for i, p in enumerate(plans)]
        context, _ = pipe.encode_prompt(texts, device=device, num_images_per_prompt=1,
                                        do_classifier_free_guidance=True)
    if not bool(torch.isfinite(context).all()):
        raise FloatingPointError("prefix context contains NaN/Inf")
    size = len(prompts)
    cache = ProjectionContextCache()

    def replace_context(module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        original = args[0]
        if original.shape[1:] != context.shape[1:] or original.shape[0] not in {size, 2 * size}:
            raise ValueError("unexpected prefix K/V projection shape")
        if original.dtype != context.dtype or original.device != context.device:
            raise ValueError("unexpected prefix K/V precision/device")
        offset = size if original.shape[0] == 2 * size else 0

        def build(source: torch.Tensor) -> torch.Tensor:
            output = source.clone()
            output[[i + offset for i in active]] = context[active]
            return output

        output = cache.replace(original, build)
        audit["context_cache_builds"], audit["context_cache_hits"] = cache.builds, cache.hits
        audit["context_cache_verifications"] = cache.verifications
        audit["projection_calls"] += 1
        return (output, *args[1:])

    modules = [m for name, m in pipe.unet.named_modules() if name.endswith(("attn2.to_k", "attn2.to_v"))]
    if not modules:
        raise ValueError("prefix conditioning requires SD cross-attention K/V projections")
    handles = [module.register_forward_pre_hook(replace_context) for module in modules]
    try:
        yield audit
    finally:
        for handle in handles:
            handle.remove()
