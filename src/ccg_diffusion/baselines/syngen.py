"""SynGen (Rassin et al., 2023): spaCy binding groups and the symmetric-KL binding loss.

The three official spaCy extractors (in official order and with their quirks),
the official wordpiece alignment and ``unify_lists`` merge, and the
positive/negative symmetric-KL loss mirror commit
``50b8eab175dc9b31ce9c7c4b2eba28c817fc440e`` with ``include_entities=False``.
The parse and alignment state is held per prompt in an immutable
``SynGenPlan``; the official pipeline's cross-sample state sharing and
last-sample-only append are not reproduced.
"""
from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import Any, Iterable, Mapping, Protocol, Sequence, Union

import torch
import torch.distributions as distributions


SOURCE_REPOSITORY = "RoyiRa/Linguistic-Binding-in-Diffusion-Models"
SOURCE_COMMIT = "50b8eab175dc9b31ce9c7c4b2eba28c817fc440e"
ATTENTION_RESOLUTION = 16
INTERVENTION_STEPS = 25
STEP_SIZE = 20.0
INCLUDE_ENTITIES = False
_START_TOKEN = "<|startoftext|>"
_END_TOKEN = "<|endoftext|>"
_MODIFIER_DEPS = ("amod", "nmod", "compound", "npadvmod", "advmod", "acomp")


class AlignmentError(ValueError):
    """The official SynGen wordpiece scan cannot align a spaCy member."""


class _SpacyToken(Protocol):
    text: str
    pos_: str
    dep_: str
    idx: int

    @property
    def children(self) -> Iterable["_SpacyToken"]:
        ...


class _Tokenizer(Protocol):
    model_max_length: int

    def __call__(self, text: str, **kwargs: Any) -> Any:
        ...

    def convert_ids_to_tokens(self, ids: Sequence[int]) -> Sequence[str]:
        ...


@dataclass(frozen=True)
class ParserGroup:
    """One ordered spaCy group produced by an official extractor."""

    extractor: str
    texts: tuple[str, ...]
    character_spans: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class BindingGroup:
    """One aligned modifier(s)-then-entity group in CLIP token space."""

    members: tuple[tuple[int, ...], ...]

    @property
    def entity(self) -> tuple[int, ...]:
        return self.members[-1]

    @property
    def modifiers(self) -> tuple[tuple[int, ...], ...]:
        return self.members[:-1]

    @property
    def flattened_indices(self) -> tuple[int, ...]:
        return tuple(index for member in self.members for index in member)


@dataclass(frozen=True)
class SynGenPlan:
    """Validated, sample-local SynGen parse and alignment state."""

    prompt: str
    bindings: tuple[BindingGroup, ...]
    attention_token_indices: tuple[int, ...]
    source_groups: tuple[ParserGroup, ...]
    status: str
    include_entities: bool = INCLUDE_ENTITIES
    source_commit: str = SOURCE_COMMIT


_MutableIndex = Union[int, list[int]]


def _input_ids(encoded: Any) -> list[int]:
    if isinstance(encoded, Mapping):
        ids = encoded.get("input_ids")
    else:
        ids = getattr(encoded, "input_ids", None)
    if ids is None:
        raise TypeError("tokenizer output does not contain input_ids")
    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().tolist()
    if ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise ValueError("expected tokenizer output for exactly one prompt")
        ids = ids[0]
    if not isinstance(ids, (list, tuple)):
        raise TypeError("tokenizer input_ids must be a sequence")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in ids):
        raise TypeError("tokenizer input_ids must contain integers")
    return list(ids)


def _wordpieces_to_indices(tokenizer: _Tokenizer, prompt: str) -> dict[int, str]:
    ids = _input_ids(tokenizer(prompt))
    tokens = list(tokenizer.convert_ids_to_tokens(ids))
    if len(tokens) != len(ids):
        raise ValueError("tokenizer returned a mismatched token string sequence")
    return dict(enumerate(tokens))


def _official_align_wordpieces(
    wordpieces: Mapping[int, str],
    start_index: int,
    target_word: str,
) -> list[int]:
    indices = [start_index]
    combined = wordpieces[start_index].replace("</w>", "")
    for wordpiece_index in range(start_index + 1, len(wordpieces)):
        if combined.lower() == target_word.lower():
            break
        next_piece = wordpieces[wordpiece_index].replace("</w>", "")
        if (
            target_word.lower().startswith(combined.lower() + next_piece.lower())
            and next_piece.lower() != target_word.lower()
        ):
            combined += wordpieces[wordpiece_index].replace("</w>", "")
            indices.append(wordpiece_index)
        else:
            indices = []
            break
    return indices


def _extract_standard(doc: Sequence[_SpacyToken]) -> list[list[_SpacyToken]]:
    subtrees = []
    for word in doc:
        if word.pos_ not in ("NOUN", "PROPN") or word.dep_ in _MODIFIER_DEPS:
            continue
        subtree = []
        stack = []
        for child in word.children:
            if child.dep_ in _MODIFIER_DEPS:
                subtree.append(child)
                stack.extend(child.children)
        while stack:
            node = stack.pop()
            if node.dep_ in _MODIFIER_DEPS or node.dep_ == "conj":
                subtree.append(node)
                stack.extend(node.children)
        if subtree:
            subtree.append(word)
            subtrees.append(subtree)
    return subtrees


def _extract_with_verb_root(doc: Sequence[_SpacyToken]) -> list[list[_SpacyToken]]:
    subtrees = []
    for word in doc:
        subtree = []
        stack = []
        if word.pos_ != "AUX" or word.dep_ in _MODIFIER_DEPS:
            continue
        for child in word.children:
            if child.dep_ in _MODIFIER_DEPS or child.pos_ in ("NOUN", "PROPN"):
                if child.pos_ not in ("AUX", "VERB"):
                    subtree.append(child)
                stack.extend(child.children)
        if len(subtree) < 2:
            continue
        while stack:
            node = stack.pop()
            if node.dep_ in _MODIFIER_DEPS or node.dep_ == "conj":
                if node.pos_ != "AUX":
                    subtree.append(node)
                stack.extend(node.children)
        if subtree:
            # This condition is unreachable for the AUX root, but it is part
            # of the official implementation and intentionally retained.
            if word.pos_ != "AUX":
                subtree.append(word)
            subtrees.append(subtree)
    return subtrees


def _extract_with_verbs(doc: Sequence[_SpacyToken]) -> list[list[_SpacyToken]]:
    """Preserve SynGen's official return-inside-the-first-token quirk."""
    subtrees = []
    modifier_deps = _MODIFIER_DEPS + ("relcl",)
    for word in doc:
        if word.pos_ not in ("NOUN", "PROPN") or word.dep_ in modifier_deps:
            continue
        subtree = []
        stack = []
        for child in word.children:
            if child.dep_ in modifier_deps:
                if child.pos_ not in ("AUX", "VERB"):
                    subtree.append(child)
                stack.extend(child.children)
        while stack:
            node = stack.pop()
            if node.dep_ in modifier_deps or node.dep_ == "conj":
                if node.pos_ not in ("AUX", "VERB"):
                    subtree.append(node)
                stack.extend(node.children)
        if subtree:
            subtree.append(word)
            subtrees.append(subtree)
        # The official function returns after the first eligible noun, not
        # after the first document token and not after scanning all nouns.
        return subtrees
    return subtrees


def extract_parser_groups(doc: Iterable[_SpacyToken]) -> tuple[ParserGroup, ...]:
    """Run all three official extractors and retain their original ordering."""
    tokens = tuple(doc)
    groups = []
    for extractor, extracted in (
        ("standard", _extract_standard(tokens)),
        ("verb_root", _extract_with_verb_root(tokens)),
        ("with_verbs", _extract_with_verbs(tokens)),
    ):
        for members in extracted:
            groups.append(
                ParserGroup(
                    extractor=extractor,
                    texts=tuple(member.text for member in members),
                    character_spans=tuple(
                        (int(member.idx), int(member.idx) + len(member.text))
                        for member in members
                    ),
                )
            )
    return tuple(groups)


def _align_groups(
    tokenizer: _Tokenizer,
    prompt: str,
    groups: Sequence[Sequence[_SpacyToken]],
) -> list[list[_MutableIndex]]:
    """Mirror official ``_align_indices`` including occurrence bookkeeping."""
    wordpieces = _wordpieces_to_indices(tokenizer, prompt)
    paired_indices = []
    collected_spacy_indices: set[int] = set()

    for group in groups:
        current: list[_MutableIndex] = []
        for member in group:
            for index, raw_wordpiece in wordpieces.items():
                if raw_wordpiece in (_START_TOKEN, _END_TOKEN):
                    continue
                wordpiece = raw_wordpiece.replace("</w>", "")
                if member.text.lower() == wordpiece.lower():
                    if index not in current and index not in collected_spacy_indices:
                        current.append(index)
                        break
                elif (
                    member.text.lower().startswith(wordpiece.lower())
                    and wordpiece.lower() != member.text.lower()
                ):
                    aligned = _official_align_wordpieces(
                        wordpieces,
                        index,
                        member.text,
                    )
                    if (
                        aligned
                        and aligned not in current
                        and all(piece not in collected_spacy_indices for piece in aligned)
                    ):
                        current.append(aligned)
                        break
        for collected in current:
            if isinstance(collected, list):
                collected_spacy_indices.update(collected)
            else:
                collected_spacy_indices.add(collected)
        if current:
            paired_indices.append(current)
    return paired_indices


def _flatten_mutable(values: Sequence[Any]) -> Iterable[int]:
    for value in values:
        if isinstance(value, list):
            yield from _flatten_mutable(value)
        else:
            yield int(value)


def official_unify_lists(
    collections: Sequence[Sequence[list[_MutableIndex]]],
) -> list[list[_MutableIndex]]:
    """Mirror official ``unify_lists`` merge order without sharing state."""
    pending = []
    for collection in collections:
        pending.extend([list(group) for group in collection])
    changed = True
    while changed:
        changed = False
        merged = []
        while pending:
            first = pending.pop(0)
            was_merged = False
            first_indices = set(_flatten_mutable(first))
            for index, other in enumerate(pending):
                if not first_indices.isdisjoint(set(_flatten_mutable(other))):
                    pending[index] = first + [item for item in other if item not in first]
                    changed = True
                    was_merged = True
                    break
            if not was_merged:
                merged.append(first)
        pending = merged
    return pending


def _as_binding(group: Sequence[_MutableIndex]) -> BindingGroup:
    members = tuple(
        tuple(member) if isinstance(member, list) else (member,)
        for member in group
    )
    return BindingGroup(members=members)


def extract_bindings(
    prompt: str,
    *,
    doc: Iterable[_SpacyToken],
    tokenizer: _Tokenizer,
    include_entities: bool = INCLUDE_ENTITIES,
) -> SynGenPlan:
    """Build the official SynGen binding state for one prompt."""
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    if include_entities is not INCLUDE_ENTITIES:
        raise ValueError("the six-arm experiment fixes SynGen include_entities=false")

    tokens = tuple(doc)
    extraction_results = (
        ("standard", _extract_standard(tokens)),
        ("verb_root", _extract_with_verb_root(tokens)),
        ("with_verbs", _extract_with_verbs(tokens)),
    )
    source_groups = []
    aligned_collections = []
    for extractor, groups in extraction_results:
        for members in groups:
            source_groups.append(
                ParserGroup(
                    extractor=extractor,
                    texts=tuple(member.text for member in members),
                    character_spans=tuple(
                        (int(member.idx), int(member.idx) + len(member.text))
                        for member in members
                    ),
                )
            )
        aligned = _align_groups(tokenizer, prompt, groups)
        if aligned:
            aligned_collections.append(aligned)

    bindings = tuple(
        _as_binding(group)
        for group in official_unify_lists(aligned_collections)
    )
    wordpieces = _wordpieces_to_indices(tokenizer, prompt)
    attention_indices = tuple(
        index
        for index, wordpiece in wordpieces.items()
        if wordpiece not in (_START_TOKEN, _END_TOKEN)
    )
    plan = SynGenPlan(
        prompt=prompt,
        bindings=bindings,
        attention_token_indices=attention_indices,
        source_groups=tuple(source_groups),
        status="ok" if bindings else "no_binding_pair",
    )
    validate_plan(plan, max_token_count=tokenizer.model_max_length)
    return plan


def validate_plan(plan: SynGenPlan, *, max_token_count: int) -> None:
    """Fail closed on malformed or incompatible SynGen prompt state."""
    if plan.source_commit != SOURCE_COMMIT:
        raise ValueError("unexpected SynGen source commit")
    if plan.include_entities is not INCLUDE_ENTITIES:
        raise ValueError("SynGen include_entities must be false")
    if plan.status not in {"ok", "no_binding_pair"}:
        raise ValueError(f"invalid SynGen sidecar status: {plan.status!r}")
    if (plan.status == "ok") != bool(plan.bindings):
        raise ValueError("SynGen sidecar status and bindings disagree")
    if isinstance(max_token_count, bool) or not isinstance(max_token_count, int):
        raise TypeError("max_token_count must be an integer")
    if tuple(sorted(set(plan.attention_token_indices))) != plan.attention_token_indices:
        raise ValueError("attention_token_indices must be unique and increasing")
    lexical_set = set(plan.attention_token_indices)
    for binding in plan.bindings:
        if len(binding.members) < 2:
            raise ValueError("primary SynGen bindings require a modifier and entity")
        for member in binding.members:
            if not member or tuple(sorted(set(member))) != member:
                raise ValueError("SynGen member indices must be unique and increasing")
            if any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or index <= 0
                or index >= max_token_count - 1
                or index not in lexical_set
                for index in member
            ):
                raise ValueError("SynGen binding index is outside the lexical CLIP input")


def symmetric_kl(
    first_attention_map: torch.Tensor,
    second_attention_map: torch.Tensor,
) -> torch.Tensor:
    """Compute SynGen's average ``KL(p||q)`` / ``KL(q||p)`` objective."""
    first = (
        first_attention_map.to(dtype=torch.float32).reshape(-1)
        if first_attention_map.ndim > 1
        else first_attention_map.to(dtype=torch.float32)
    )
    second = (
        second_attention_map.to(dtype=torch.float32).reshape(-1)
        if second_attention_map.ndim > 1
        else second_attention_map.to(dtype=torch.float32)
    )
    first_distribution = distributions.Categorical(probs=first)
    second_distribution = distributions.Categorical(probs=second)
    return (
        distributions.kl_divergence(first_distribution, second_distribution)
        + distributions.kl_divergence(second_distribution, first_distribution)
    ) / 2


def _official_tensor_max(values: Sequence[torch.Tensor]) -> torch.Tensor:
    """Reproduce Python ``max`` tie-breaking from the official loss."""
    if not values:
        raise ValueError("official tensor max requires at least one value")
    maximum = values[0]
    for value in values[1:]:
        if bool((value > maximum).detach().item()):
            maximum = value
    return maximum


def _positive_loss(
    attention_maps: Sequence[torch.Tensor],
    modifier: tuple[int, ...],
    entity: tuple[int, ...],
) -> torch.Tensor:
    losses = [
        symmetric_kl(attention_maps[source], attention_maps[destination])
        for source, destination in itertools.product(modifier, entity)
    ]
    return _official_tensor_max(losses)


def _outside_source_loss(
    attention_maps: Sequence[torch.Tensor],
    source: tuple[int, ...],
    outside_indices: Sequence[int],
) -> torch.Tensor:
    outside_losses = []
    for outside_index in outside_indices:
        piece_losses = [
            symmetric_kl(attention_maps[piece], attention_maps[outside_index])
            for piece in source
        ]
        outside_losses.append(_official_tensor_max(piece_losses))
    return torch.stack(outside_losses).sum()


def _negative_loss(
    attention_maps: Sequence[torch.Tensor],
    *,
    modifier: tuple[int, ...],
    entity: tuple[int, ...],
    binding: BindingGroup,
    attention_token_indices: Sequence[int],
) -> torch.Tensor:
    binding_indices = set(binding.flattened_indices)
    outside_indices = tuple(
        index for index in attention_token_indices if index not in binding_indices
    )
    if not outside_indices:
        return torch.zeros((), dtype=attention_maps[0].dtype, device=attention_maps[0].device)

    entity_loss = -_outside_source_loss(attention_maps, entity, outside_indices) / len(outside_indices)
    if modifier:
        modifier_loss = -_outside_source_loss(attention_maps, modifier, outside_indices) / len(outside_indices)
        return (modifier_loss + entity_loss) / 2
    return entity_loss


def sample_loss(
    attention_maps: torch.Tensor,
    plan: SynGenPlan,
) -> torch.Tensor:
    """Compute official positive-plus-negative SynGen loss for one sample."""
    if attention_maps.ndim != 3:
        raise ValueError("attention_maps must have shape [height, width, tokens]")
    if tuple(attention_maps.shape[:2]) != (
        ATTENTION_RESOLUTION,
        ATTENTION_RESOLUTION,
    ):
        raise ValueError("SynGen attention maps must be exactly 16x16")
    if not bool(torch.isfinite(attention_maps.float()).all().item()):
        raise FloatingPointError("SynGen attention maps contain NaN or Inf")
    validate_plan(plan, max_token_count=attention_maps.shape[-1])
    if not plan.bindings:
        raise ValueError("stock fallback sidecars do not have a SynGen loss")

    # Keep KL probability normalization and loss reduction out of BF16.
    scaled_maps = attention_maps.to(dtype=torch.float32) * 100
    per_token_maps = tuple(
        scaled_maps[:, :, index] for index in range(scaled_maps.shape[-1])
    )
    loss = torch.zeros((), dtype=torch.float32, device=attention_maps.device)
    for binding in plan.bindings:
        entity = binding.entity
        modifiers = binding.modifiers
        if not modifiers:
            loss = loss + _negative_loss(
                per_token_maps,
                modifier=(),
                entity=entity,
                binding=binding,
                attention_token_indices=plan.attention_token_indices,
            )
            continue
        for modifier in modifiers:
            loss = loss + _positive_loss(per_token_maps, modifier, entity)
            loss = loss + _negative_loss(
                per_token_maps,
                modifier=modifier,
                entity=entity,
                binding=binding,
                attention_token_indices=plan.attention_token_indices,
            )
    return loss


def batch_losses(
    attention_maps: torch.Tensor,
    plans: Sequence[SynGenPlan],
) -> tuple[torch.Tensor, ...]:
    """Compute sample-local SynGen losses for a physical image batch."""
    if attention_maps.ndim != 4:
        raise ValueError("attention_maps must have shape [batch, height, width, tokens]")
    if len(plans) != attention_maps.shape[0]:
        raise ValueError("sidecars must have one entry per sample")
    return tuple(
        sample_loss(sample_maps, plan)
        for sample_maps, plan in zip(attention_maps, plans)
    )
