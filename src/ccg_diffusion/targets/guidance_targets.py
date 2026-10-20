r"""Stage 3: guidance targets I_D = (E, C, K, P) from a CCG derivation, and their files.

Guidance targets (paper Sec. IV-A) are CLIP token indices derived from the
Hol-CCG derivation and the C&C ``generate -e`` marked dependencies of a prompt:

* E (``exist``): the nouns, i.e. the atomic ``N`` leaves;
* C (``corr``): modifier -> noun pairs, following modifier-of-modifier chains
  (``light blue banana`` gives both ``(light, banana)`` and ``(blue, banana)``);
* K (``leak``): each modifier paired with every noun it does not modify;
* P (``possession``): subject -> object pairs of ``have``/``wear``/``grasp``/
  ``hold`` (the only word comparison, by spaCy lemma), plus, for the TAB
  interpretations, the optional category-only attachment-functor rule.

The conversion is intentionally small:

* AUTO leaves are only a complete source-token/category ledger;
* C&C ``generate -e`` rows are the only syntactic relation evidence;
* the only semantic word comparison is the possession lemma set;
* the output is the unchanged four-field :class:`PDIndices`.

No nearest-token repair, partial-parse fallback, pronoun list, numeral list,
relation-word list, or extra loss/index type is used. Inputs which cannot
justify an index remain visible in the conversion trace instead of being
silently guessed.

The module also holds the guidance-target files of the SD1.4 pipeline: the
per-category Stage 3 rows (:func:`build_guidance_target_rows`) and their
metadata and identity gates, the Stage 1-3 source-manifest reader
(:func:`load_source_prompts`), the seed expansion to image rows
(:func:`expand_to_image_rows`) and the frozen T2I-CompBench++ parse bundle
(:func:`load_t2i_parse_bundle`). It uses only the standard library; spaCy and
tokenizer objects are passed in.
"""
from __future__ import annotations

import gzip
import importlib.metadata
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence

from ccg_diffusion import provenance
from ccg_diffusion.config import (
    SD14_CCG_ARM,
    SD14_GENERATION,
    SD14_GUIDANCE_TARGET_PROVIDER,
    SD14_PREDICATED_DIFFUSION_ARM,
    SD14_PREDICATED_DIFFUSION_PROVIDER,
    T2I_COMPBENCH_CATEGORIES,
    T2I_COMPBENCH_SEEDS,
)
from ccg_diffusion.parsing.ccg import (
    AutoSentence,
    CCGCategory,
    ClipSourceAnchor,
    MarkedDependency,
    MarkedDepsError,
    align_whitespace_tokens_to_clip,
    is_degree_modifier,
    is_noun_modifier,
    is_passive_verb_frame,
    is_predicative_adjective,
    load_auto_sentences,
    load_marked_dependency_blocks,
    parse_ccg_category,
    tokenizer_ids,
)
from ccg_diffusion.parsing.sentences import (
    ParseToken,
    SentenceMapRow,
    group_sentence_map,
    split_and_tokenize,
    tokenize_for_parse,
    validate_sentence_map_group,
)
from ccg_diffusion.provenance import (
    artifact_record,
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_jsonl,
    canonical_json,
    read_jsonl,
    resolve_recorded_path,
    sha256_bytes,
    sha256_text,
    verify_artifact_record,
)
from ccg_diffusion.targets.lexicon import lexical_resource_path
from ccg_diffusion.targets.pd_indices import PDIndices, pd_indices_payload, program_sha256
from ccg_diffusion.targets.tokenizer import require_sd14_tokenizer


class UnsupportedPromptProjectionError(MarkedDepsError):
    """The parsed prompt has no entity that the PD program can represent.

    This is deliberately narrower than :class:`MarkedDepsError`.  Callers may
    classify this corpus-level limitation as a pre-generation coverage
    failure, while tokenizer, alignment, dependency, category, and program
    invariant errors must still stop the artifact build.
    """


METHOD_ID = "deps2index"
PROFILE_ID = METHOD_ID
POSSESSION_LEMMAS = frozenset({"have", "wear", "grasp", "hold"})
PROJECTION_VERSION = "source_graph_to_full_prompt_clip_v2"
ATTACHMENT_POSSESSION_RULE_ID = "attachment_functor_possession"
ATTACHMENT_COMPLEMENT_ATOMS = frozenset({"NP", "PP"})
_SLOT_ANNOTATION_RE = re.compile(r"<(\d+)>")


@dataclass(frozen=True)
class CompiledGuidanceTargets:
    """One sentence's guidance targets I_D = (E, C, K, P), its CLIP alignment and its audit trace."""

    indices: PDIndices
    alignment: tuple[Mapping[str, Any], ...]
    trace: Mapping[str, Any]


@dataclass
class _DependencyDisposition:
    dependency_index: int
    dependency: MarkedDependency
    marked_role: str
    uses: set[str] = field(default_factory=set)
    reasons: set[str] = field(default_factory=set)

    def use(self, name: str) -> None:
        self.uses.add(name)

    def reject(self, reason: str) -> None:
        self.reasons.add(reason)

    def payload(self) -> dict[str, Any]:
        if self.uses:
            status = "used"
        elif any(reason.startswith("unresolved_") for reason in self.reasons):
            status = "unresolved"
        else:
            status = "not_used"
        dependency = self.dependency
        return {
            "dependency_index": self.dependency_index,
            "status": status,
            "marked_role": self.marked_role,
            "uses": sorted(self.uses),
            "reasons": sorted(self.reasons),
            "head_source": dependency.head_index,
            "head_word": dependency.head_word,
            "marked_category": dependency.marked_category,
            "slot": dependency.slot,
            "argument_source": dependency.argument_index,
            "argument_word": dependency.argument_word,
            "rule_id": dependency.rule_id,
            "long_range_category": dependency.long_range_category,
        }


@dataclass(frozen=True)
class _CoordinationGroup:
    members: tuple[int, ...]
    coordinator_sources: tuple[int, ...]
    dependency_indices: tuple[int, ...]
    member_role: str

    def payload(self) -> dict[str, Any]:
        return {
            "members": list(self.members),
            "coordinator_sources": list(self.coordinator_sources),
            "dependency_indices": list(self.dependency_indices),
            "member_role": self.member_role,
        }


@dataclass(frozen=True)
class _EntityResolution:
    entities: Mapping[int, frozenset[int]]
    paths: tuple[Mapping[str, Any], ...]
    visited_sources: tuple[int, ...]


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[int, int] = {}

    def find(self, item: int) -> int:
        self.parent.setdefault(item, item)
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        smaller, larger = sorted((left_root, right_root))
        self.parent[larger] = smaller


def _is_atomic_n(category: CCGCategory) -> bool:
    return category.is_atomic and category.base_atom == "N"


def _is_atomic_np(category: CCGCategory) -> bool:
    return category.is_atomic and category.base_atom == "NP"


def _is_coordination(category: CCGCategory) -> bool:
    return category.is_atomic and category.base_atom == "conj"


def _is_feature_preserving_noun_modifier(category: CCGCategory) -> bool:
    """Return whether a nominal endofunctor preserves the atomic-N feature."""

    return bool(
        is_noun_modifier(category)
        and category.result is not None
        and category.argument is not None
        and category.result.feature == category.argument.feature
    )


def _is_higher_order_noun_modifier(category: CCGCategory) -> bool:
    """Return whether the category maps one identical nominal modifier to itself."""

    return bool(
        is_degree_modifier(category)
        and category.result is not None
        and category.argument is not None
        and category.result == category.argument
        and _is_feature_preserving_noun_modifier(category.result)
    )


def _category_role(category: CCGCategory) -> str:
    if _is_atomic_n(category):
        return "entity"
    if (
        _is_feature_preserving_noun_modifier(category)
        or is_predicative_adjective(category)
    ):
        return "modifier"
    if _is_higher_order_noun_modifier(category):
        return "modifier_of_modifier"
    if _is_coordination(category):
        return "coordination"
    if category.is_atomic and category.base_atom == "NP":
        return "atomic_np"
    return "out_of_contract"


def _contains_s_np_subject_frame(category: CCGCategory) -> bool:
    current = category
    while not current.is_atomic:
        if (
            current.slash == "\\"
            and current.result is not None
            and current.argument is not None
            and current.result.is_atomic
            and current.result.base_atom == "S"
            and _is_atomic_np(current.argument)
        ):
            return True
        if current.result is None:
            return False
        current = current.result
    return False


def _is_backward_np_modifier(category: CCGCategory) -> bool:
    return bool(
        category.slash == "\\"
        and category.result is not None
        and category.argument is not None
        and _is_atomic_np(category.result)
        and _is_atomic_np(category.argument)
    )


def _is_verb_phrase_shape(category: CCGCategory) -> bool:
    r"""Return whether the category is an ``S...\NP`` predicate body.

    The ``S`` feature is not inspected, so ``S[dcl]``, ``S[ng]``, and the bare
    ``S`` of a marked category all qualify.  The argument must be atomic ``NP``.
    """

    return bool(
        not category.is_atomic
        and category.slash == "\\"
        and category.result is not None
        and category.argument is not None
        and category.result.is_atomic
        and category.result.base_atom == "S"
        and _is_atomic_np(category.argument)
    )


def _is_attachment_modifiable_core(category: CCGCategory) -> bool:
    """Return whether an attachment may modify this category."""

    return bool(
        _is_atomic_n(category)
        or _is_atomic_np(category)
        or _is_verb_phrase_shape(category)
    )


@dataclass(frozen=True)
class _AttachmentFunctorLayout:
    r"""The slot layout of one recognised ``(X\X)/Y`` attachment functor."""

    modified_slot: int
    complement_slot: int
    modified_core_role: str
    complement_atom: str

    def payload(self) -> dict[str, Any]:
        return {
            "rule_id": ATTACHMENT_POSSESSION_RULE_ID,
            "modified_slot": self.modified_slot,
            "complement_slot": self.complement_slot,
            "modified_core_role": self.modified_core_role,
            "complement_atom": self.complement_atom,
        }


def _attachment_functor_layout(
    category: CCGCategory,
) -> Optional[_AttachmentFunctorLayout]:
    r"""Recognise ``(X\X)/Y`` by category alone and return its slot layout.

    ``Y`` must be an atomic ``NP`` or ``PP`` complement, and ``X`` must be a
    nominal or predicate body that an attachment can modify.

    This function returns the layout the recognised shape implies: an
    attachment functor declares exactly two arguments - the endofunctor core's
    argument ``X`` and the complement ``Y`` - and ``X`` is written before ``Y``
    in ``(X\X)/Y``, so ``X`` takes the lower slot.  Whether the input really
    carries those two slot numbers in that order is not assumed here:
    :func:`parse_ccg_category` deletes ``<n>`` before this recognition runs, so
    :func:`_require_attachment_slot_annotations` re-reads the raw markedup
    string and rejects any category that does not declare them.
    """

    if category.is_atomic or category.slash != "/":
        return None
    complement = category.argument
    core = category.result
    if complement is None or core is None:
        return None
    if (
        not complement.is_atomic
        or complement.base_atom not in ATTACHMENT_COMPLEMENT_ATOMS
    ):
        return None
    if (
        core.is_atomic
        or core.slash != "\\"
        or core.result is None
        or core.argument is None
        or core.result != core.argument
    ):
        return None
    modified = core.argument
    if not _is_attachment_modifiable_core(modified):
        return None
    declared_arguments = (("modified", modified), ("complement", complement))
    slot_by_role = {
        role: slot
        for slot, (role, _declared) in enumerate(declared_arguments, start=1)
    }
    return _AttachmentFunctorLayout(
        modified_slot=slot_by_role["modified"],
        complement_slot=slot_by_role["complement"],
        modified_core_role=(
            "verb_phrase"
            if _is_verb_phrase_shape(modified)
            else ("atomic_n" if _is_atomic_n(modified) else "atomic_np")
        ),
        complement_atom=str(complement.base_atom),
    )


def _require_attachment_slot_annotations(
    marked_category: str, *, location: str
) -> None:
    r"""Require the two slot numbers the recognised layout reads off ``(X\X)/Y``.

    ``parse_ccg_category`` strips every ``<n>`` annotation, so the recognised
    layout would otherwise be a convention this module assumes rather than one
    the input declares.  The raw markedup string is therefore re-read here: an
    attachment functor must annotate slot ``1`` and slot ``2`` exactly once
    each, and ``<1>`` must precede ``<2>`` so that the earlier-written modified
    core ``X`` is the one holding slot 1 and the complement ``Y`` holds slot 2.
    A category that declares anything else is not the frame this rule claims to
    recognise, so it stops the build instead of being reinterpreted.
    """

    slots = _SLOT_ANNOTATION_RE.findall(marked_category)
    if sorted(slots) != ["1", "2"]:
        raise MarkedDepsError(
            f"{location}: attachment functor must declare slots 1 and 2 exactly "
            f"once each, got {slots}: {marked_category!r}"
        )
    if slots.index("1") > slots.index("2"):
        raise MarkedDepsError(
            f"{location}: attachment functor declares <2> before <1>, so the "
            "modified core does not hold slot 1: "
            f"{marked_category!r}"
        )


def _parse_category(value: str, *, location: str) -> CCGCategory:
    try:
        return parse_ccg_category(value)
    except MarkedDepsError as exc:
        raise MarkedDepsError(f"{location}: {exc}") from exc


def _validate_possession_annotations(
    annotations: Mapping[int, str], leaf_by_source: Mapping[int, Any]
) -> dict[int, str]:
    normalized: dict[int, str] = {}
    for source, raw_lemma in annotations.items():
        if type(source) is not int or source not in leaf_by_source:
            raise MarkedDepsError(
                f"possession lemma source is outside AUTO leaves: {source!r}"
            )
        if not isinstance(raw_lemma, str):
            raise MarkedDepsError(
                f"possession lemma at source {source} must be a string"
            )
        lemma = raw_lemma.lower()
        if lemma not in POSSESSION_LEMMAS:
            raise MarkedDepsError(
                "Stage 3 received a lemma outside the permitted possession set: "
                f"{source}:{raw_lemma!r}"
            )
        normalized[source] = lemma
    return normalized


def possession_lemma_annotations_from_spacy(
    prompt: str,
    nlp: Any,
    *,
    source_tokens: Sequence[ParseToken] | None = None,
) -> dict[int, str]:
    """Return only source-token annotations matching the four allowed lemmas.

    spaCy dependency labels and POS tags are deliberately not inspected.  The
    helper uses token character spans only to attach a matching allowed lemma
    to the parse-source positions shared by AUTO and marked dependencies.
    """

    if source_tokens is None:
        source_tokens = tuple(
            ParseToken(
                text=match.group(0),
                char_start=match.start(),
                char_end=match.end(),
            )
            for match in re.finditer(r"\S+", prompt)
        )
    annotations: dict[int, str] = {}
    doc = nlp(prompt)
    for token in doc:
        lemma = str(getattr(token, "lemma_", "")).lower()
        if lemma not in POSSESSION_LEMMAS:
            continue
        token_start = int(token.idx)
        token_end = token_start + len(str(token.text))
        containing = [
            source_index
            for source_index, source_token in enumerate(source_tokens, start=1)
            if source_token.char_start <= token_start
            and token_end <= source_token.char_end
        ]
        if len(containing) != 1:
            raise MarkedDepsError(
                "spaCy possession lemma cannot be aligned to one source token: "
                f"{token.text!r}@{token_start}:{token_end}"
            )
        source = containing[0]
        previous = annotations.get(source)
        if previous is not None and previous != lemma:
            raise MarkedDepsError(
                f"conflicting possession lemmas at source {source}: {previous}/{lemma}"
            )
        annotations[source] = lemma
    return annotations


def _build_coordination_groups(
    dispositions: Sequence[_DependencyDisposition],
    leaf_roles: Mapping[int, str],
) -> tuple[_CoordinationGroup, ...]:
    union_find = _UnionFind()
    members_by_coordinator: dict[int, set[int]] = {}
    dependency_indices_by_coordinator: dict[int, set[int]] = {}
    for disposition in dispositions:
        dependency = disposition.dependency
        if disposition.marked_role != "coordination":
            continue
        if dependency.slot != 1:
            disposition.reject("unexpected_coordination_slot")
            continue
        members_by_coordinator.setdefault(dependency.head_index, set()).add(
            dependency.argument_index
        )
        dependency_indices_by_coordinator.setdefault(
            dependency.head_index, set()
        ).add(disposition.dependency_index)

    for coordinator, members in sorted(members_by_coordinator.items()):
        ordered = sorted(members)
        if len(ordered) < 2:
            for dependency_index in dependency_indices_by_coordinator[coordinator]:
                dispositions[dependency_index].reject(
                    "coordination_group_requires_two_members"
                )
            continue
        for member in ordered[1:]:
            union_find.union(ordered[0], member)

    members_by_root: dict[int, set[int]] = {}
    coordinators_by_root: dict[int, set[int]] = {}
    dependencies_by_root: dict[int, set[int]] = {}
    for coordinator, members in sorted(members_by_coordinator.items()):
        if len(members) < 2:
            continue
        root = union_find.find(min(members))
        members_by_root.setdefault(root, set()).update(members)
        coordinators_by_root.setdefault(root, set()).add(coordinator)
        dependencies_by_root.setdefault(root, set()).update(
            dependency_indices_by_coordinator[coordinator]
        )

    groups: list[_CoordinationGroup] = []
    for root in sorted(members_by_root, key=lambda item: min(members_by_root[item])):
        members = tuple(sorted(members_by_root[root]))
        roles = {leaf_roles[member] for member in members}
        member_role = next(iter(roles)) if len(roles) == 1 else "mixed"
        group = _CoordinationGroup(
            members=members,
            coordinator_sources=tuple(sorted(coordinators_by_root[root])),
            dependency_indices=tuple(sorted(dependencies_by_root[root])),
            member_role=member_role,
        )
        groups.append(group)
        for dependency_index in group.dependency_indices:
            dispositions[dependency_index].use("coordination_group")
    return tuple(groups)


def _resolve_entity_endpoint(
    source: int,
    *,
    entity_sources: set[int],
    dependencies_by_head: Mapping[int, Sequence[tuple[int, MarkedDependency]]],
    dispositions: Sequence[_DependencyDisposition],
) -> _EntityResolution:
    resolved: dict[int, set[int]] = {}
    paths: list[Mapping[str, Any]] = []
    visited_sources: set[int] = set()
    stack: list[tuple[int, tuple[int, ...], frozenset[int]]] = [
        (source, (source,), frozenset())
    ]
    while stack:
        current, source_path, dependency_path = stack.pop()
        visited_sources.add(current)
        if current in entity_sources:
            resolved.setdefault(current, set()).update(dependency_path)
            paths.append(
                {
                    "source_path": list(source_path),
                    "resolved_entity": current,
                    "dependency_indices": sorted(dependency_path),
                }
            )
            continue

        for dependency_index, dependency in sorted(
            dependencies_by_head.get(current, ()),
            key=lambda item: (item[1].slot, item[1].argument_index, item[0]),
            reverse=True,
        ):
            if dependency.slot != 1:
                continue
            argument = dependency.argument_index
            if argument in source_path:
                continue
            stack.append(
                (
                    argument,
                    (*source_path, argument),
                    dependency_path | {dependency_index},
                )
            )

    frozen = {
        endpoint: frozenset(evidence)
        for endpoint, evidence in sorted(resolved.items())
    }
    used_dependency_indices = {
        dependency_index
        for evidence in frozen.values()
        for dependency_index in evidence
    }
    for dependency_index in used_dependency_indices:
        dispositions[dependency_index].use("endpoint_resolution")
    return _EntityResolution(
        entities=frozen,
        paths=tuple(paths),
        visited_sources=tuple(sorted(visited_sources)),
    )


def _nominal_modifier_subject_binding(
    sentence: AutoSentence,
    predicate_source: int,
    *,
    entity_sources: set[int],
) -> Optional[Mapping[str, Any]]:
    r"""Bind an unsaturated clause subject from an explicit AUTO ``NP\NP``.

    The innermost nominal-modifier constituent containing the predicate wins.
    Its parent must take its lexical head from outside that constituent, and
    that head must already be an atomic-N entity in the same parser fragment.
    """

    candidates: list[tuple[int, int, Mapping[str, Any]]] = []
    leaf_by_source = {leaf.source_index: leaf for leaf in sentence.leaves}
    predicate_fragment = leaf_by_source[predicate_source].fragment_index

    def predicate_heads_subject_frame(node: Any) -> bool:
        category = _parse_category(
            node.category,
            location=(
                "AUTO subject-binding node "
                f"{node.start}:{node.end} for source {predicate_source}"
            ),
        )
        if (
            node.head_index == predicate_source
            and _contains_s_np_subject_frame(category)
        ):
            return True
        return any(
            child.start <= predicate_source <= child.end
            and predicate_heads_subject_frame(child)
            for child in node.children
        )

    def visit(node: Any, depth: int) -> None:
        for child in node.children:
            if not child.start <= predicate_source <= child.end:
                continue
            child_category = _parse_category(
                child.category,
                location=f"AUTO child {child.start}:{child.end}",
            )
            external_head = node.head_index
            if (
                _is_backward_np_modifier(child_category)
                and not child.start <= external_head <= child.end
                and external_head in entity_sources
                and leaf_by_source[external_head].fragment_index == predicate_fragment
                and predicate_heads_subject_frame(child)
            ):
                candidates.append(
                    (
                        depth,
                        child.end - child.start,
                        {
                            "predicate_source": predicate_source,
                            "subject_source": external_head,
                            "modifier_category": child.category,
                            "modifier_span": [child.start, child.end],
                            "parent_category": node.category,
                            "evidence": "auto_nominal_modifier_attachment",
                        },
                    )
                )
            visit(child, depth + 1)

    visit(sentence.root, 0)
    if not candidates:
        return None
    candidates.sort(
        key=lambda item: (-item[0], item[1], int(item[2]["subject_source"]))
    )
    return candidates[0][2]


def _reject_attachment_frame(
    by_slot: Mapping[int, Sequence[tuple[int, MarkedDependency]]],
    dispositions: Sequence[_DependencyDisposition],
    reason: str,
    *,
    slots: Sequence[int],
) -> None:
    """Record ``reason`` on the dependencies of the functor's declared slots.

    A dependency in any other slot is left to the general disposition
    fallback, because this rule makes no claim about it.
    """

    for slot in slots:
        for dependency_index, _dependency in by_slot.get(slot, ()):
            dispositions[dependency_index].reject(reason)


def _attachment_frame_is_passive(
    modified_dependencies: Sequence[tuple[int, MarkedDependency]],
    *,
    functor_category: CCGCategory,
    leaf_categories: Mapping[int, CCGCategory],
    marked_categories: Sequence[CCGCategory],
    dependencies_by_head: Mapping[int, Sequence[tuple[int, MarkedDependency]]],
) -> bool:
    """Return whether an attachment modifies a passive predicate.

    This is the category-feature counterpart of the possession lemma path's
    voice check.  Both ends of the frame are inspected: the functor's own
    marked category, whose modified core is passive in
    ``((S[pss]\\NP)\\(S[pss]\\NP))/NP``, and the modified endpoint's AUTO leaf
    category together with the marked categories of the dependencies it heads.
    Either end alone is enough to reject the frame.
    """

    if is_passive_verb_frame(functor_category):
        return True
    for _dependency_index, dependency in modified_dependencies:
        modified_source = dependency.argument_index
        categories = [
            leaf_categories[modified_source],
            *[
                marked_categories[headed_index]
                for headed_index, _headed in dependencies_by_head.get(
                    modified_source, ()
                )
            ],
        ]
        if any(is_passive_verb_frame(category) for category in categories):
            return True
    return False


def _emit_attachment_possession(  # noqa: C901
    *,
    dependencies: Sequence[MarkedDependency],
    marked_categories: Sequence[CCGCategory],
    leaf_by_source: Mapping[int, Any],
    leaf_categories: Mapping[int, CCGCategory],
    entity_sources: set[int],
    dependencies_by_head: Mapping[int, Sequence[tuple[int, MarkedDependency]]],
    dispositions: Sequence[_DependencyDisposition],
    possession_sources: dict[tuple[int, int], set[int]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Add the possession pairs an attachment functor justifies by category.

    ``possession_sources`` is the map the possession lemma path has already
    filled, so a pair both paths justify stays one pair with merged evidence.
    Endpoints are resolved with the same slot-1 chain the rest of the module
    uses, and a frame that cannot resolve both endpoints is recorded instead of
    being repaired.

    The two conditions that would make a recorded frame meaningless are
    category invariants rather than coverage limits, so they stop the build:
    a recognised functor that does not declare slots 1 and 2 in that order, and
    one head carrying two different attachment functor categories.
    """

    frames: list[dict[str, Any]] = []
    stats = {
        "candidate_heads": 0,
        "fired_heads": 0,
        "heads_missing_a_slot": 0,
        "heads_with_a_passive_frame": 0,
        "heads_with_unresolved_holder": 0,
        "heads_with_unresolved_object": 0,
        "heads_with_only_self_pairs": 0,
        "pairs_added": 0,
    }
    candidates: dict[int, dict[str, Any]] = {}
    for dependency_index, dependency in enumerate(dependencies):
        category = marked_categories[dependency_index]
        layout = _attachment_functor_layout(category)
        if layout is None:
            continue
        location = f"dependency {dependency_index} marked category"
        _require_attachment_slot_annotations(
            dependency.marked_category, location=location
        )
        candidate = candidates.setdefault(
            dependency.head_index,
            {
                "layout": layout,
                "category": category,
                "marked_category": dependency.marked_category,
                "by_slot": {},
            },
        )
        # One head is one frame, so every recognised row it heads must be the
        # same functor.  Two different marked categories on one head would make
        # the slot layout depend on which row was read first, and would pair
        # endpoints across two frames, so they stop the build.
        if candidate["marked_category"] != dependency.marked_category:
            raise MarkedDepsError(
                f"{location}: source {dependency.head_index} heads two different "
                "attachment functor categories: "
                f"{candidate['marked_category']!r} and "
                f"{dependency.marked_category!r}"
            )
        candidate["by_slot"].setdefault(dependency.slot, []).append(
            (dependency_index, dependency)
        )

    for head_source, candidate in sorted(candidates.items()):
        layout = candidate["layout"]
        by_slot: dict[int, list[tuple[int, MarkedDependency]]] = candidate["by_slot"]
        stats["candidate_heads"] += 1
        frame: dict[str, Any] = {
            "head_source": head_source,
            "head_word": leaf_by_source[head_source].word,
            "auto_category": leaf_by_source[head_source].category,
            "marked_category": candidate["marked_category"],
            **layout.payload(),
            "slots_present": sorted(by_slot),
            "raw_holder_sources": sorted(
                dependency.argument_index
                for _index, dependency in by_slot.get(layout.modified_slot, ())
            ),
            "raw_object_sources": sorted(
                dependency.argument_index
                for _index, dependency in by_slot.get(layout.complement_slot, ())
            ),
            "fired": False,
            "reason": None,
            "holder_sources": [],
            "object_sources": [],
            "holder_resolution_paths": [],
            "object_resolution_paths": [],
            "pairs": [],
        }
        declared_slots = (layout.modified_slot, layout.complement_slot)
        missing_slots = [slot for slot in declared_slots if slot not in by_slot]
        if missing_slots:
            reason = "attachment_missing_slot_" + "_".join(
                str(slot) for slot in missing_slots
            )
            stats["heads_missing_a_slot"] += 1
            _reject_attachment_frame(
                by_slot, dispositions, reason, slots=declared_slots
            )
            frame["reason"] = reason
            frames.append(frame)
            continue
        if _attachment_frame_is_passive(
            by_slot[layout.modified_slot],
            functor_category=candidate["category"],
            leaf_categories=leaf_categories,
            marked_categories=marked_categories,
            dependencies_by_head=dependencies_by_head,
        ):
            stats["heads_with_a_passive_frame"] += 1
            _reject_attachment_frame(
                by_slot,
                dispositions,
                "passive_attachment_frame",
                slots=declared_slots,
            )
            frame["reason"] = "passive_attachment_frame"
            frames.append(frame)
            continue

        resolved: dict[str, dict[int, set[int]]] = {"holder": {}, "object": {}}
        resolution_paths: dict[str, list[Mapping[str, Any]]] = {
            "holder": [],
            "object": [],
        }
        for role, slot in (
            ("holder", layout.modified_slot),
            ("object", layout.complement_slot),
        ):
            for dependency_index, dependency in by_slot[slot]:
                resolution = _resolve_entity_endpoint(
                    dependency.argument_index,
                    entity_sources=entity_sources,
                    dependencies_by_head=dependencies_by_head,
                    dispositions=dispositions,
                )
                resolution_paths[role].extend(resolution.paths)
                if not resolution.entities:
                    dispositions[dependency_index].reject(
                        f"unresolved_attachment_{role}"
                    )
                for entity_source, path_evidence in resolution.entities.items():
                    resolved[role].setdefault(entity_source, set()).update(
                        {dependency_index, *path_evidence}
                    )

        frame_pairs: list[tuple[int, int]] = []
        for holder_source, holder_evidence in sorted(resolved["holder"].items()):
            for object_source, object_evidence in sorted(resolved["object"].items()):
                if holder_source == object_source:
                    continue
                pair = (holder_source, object_source)
                if pair not in possession_sources:
                    stats["pairs_added"] += 1
                evidence = holder_evidence | object_evidence
                possession_sources.setdefault(pair, set()).update(evidence)
                frame_pairs.append(pair)
                for dependency_index in evidence:
                    dispositions[dependency_index].use("possession")
        if frame_pairs:
            stats["fired_heads"] += 1
            reason = None
        elif not resolved["holder"]:
            stats["heads_with_unresolved_holder"] += 1
            reason = "unresolved_attachment_holder"
        elif not resolved["object"]:
            stats["heads_with_unresolved_object"] += 1
            reason = "unresolved_attachment_object"
        else:
            stats["heads_with_only_self_pairs"] += 1
            reason = "attachment_holder_equals_object"
        if reason is not None:
            _reject_attachment_frame(
                by_slot, dispositions, reason, slots=declared_slots
            )
        frame.update(
            {
                "fired": bool(frame_pairs),
                "reason": reason,
                "holder_sources": sorted(resolved["holder"]),
                "object_sources": sorted(resolved["object"]),
                "holder_resolution_paths": list(resolution_paths["holder"]),
                "object_resolution_paths": list(resolution_paths["object"]),
                "pairs": [list(pair) for pair in sorted(set(frame_pairs))],
            }
        )
        frames.append(frame)
    return frames, stats


def _default_source_tokens(prompt: str) -> tuple[ParseToken, ...]:
    return tuple(
        ParseToken(
            text=match.group(0),
            char_start=match.start(),
            char_end=match.end(),
        )
        for match in re.finditer(r"\S+", prompt)
    )


def _validate_source_tokens(
    prompt: str,
    source_tokens: Sequence[ParseToken],
    sentence: AutoSentence,
) -> None:
    if len(source_tokens) != len(sentence.leaves):
        raise MarkedDepsError(
            "parse-source token/AUTO leaf count mismatch: "
            f"{len(source_tokens)}/{len(sentence.leaves)}"
        )
    previous_end = 0
    for source_index, (source_token, leaf) in enumerate(
        zip(source_tokens, sentence.leaves), start=1
    ):
        if source_token.text != leaf.word:
            raise MarkedDepsError(
                f"parse-source token/AUTO surface mismatch at source {source_index}: "
                f"{source_token.text!r}/{leaf.word!r}"
            )
        if not (
            previous_end <= source_token.char_start
            < source_token.char_end
            <= len(prompt)
        ):
            raise MarkedDepsError(
                f"invalid parse-source char span at source {source_index}"
            )
        if prompt[source_token.char_start:source_token.char_end] != source_token.text:
            raise MarkedDepsError(
                f"parse-source char span does not reproduce source {source_index}"
            )
        previous_end = source_token.char_end


def _source_has_lexical_clip_piece(source_token: ParseToken, tokenizer: Any) -> bool:
    token_ids = tokenizer_ids(
        tokenizer(source_token.text, add_special_tokens=False)
    )
    return any(
        any(char.isalnum() for char in str(tokenizer.decode(token_id)))
        for token_id in token_ids
    )


def _full_prompt_anchors_by_source(
    prompt: str,
    source_tokens: Sequence[ParseToken],
    tokenizer: Any,
) -> tuple[
    tuple[ClipSourceAnchor, ...],
    dict[int, ClipSourceAnchor],
    dict[int, bool],
]:
    prompt_anchors = align_whitespace_tokens_to_clip(prompt, tokenizer)
    anchor_by_source: dict[int, ClipSourceAnchor] = {}
    projectable_by_source: dict[int, bool] = {}
    for source_index, source_token in enumerate(source_tokens, start=1):
        containing = [
            anchor
            for anchor in prompt_anchors
            if anchor.char_start <= source_token.char_start
            and source_token.char_end <= anchor.char_end
        ]
        if len(containing) != 1:
            raise MarkedDepsError(
                "parse-source token cannot be mapped to one full-prompt word: "
                f"{source_index}:{source_token.text!r}"
            )
        anchor = containing[0]
        anchor_by_source[source_index] = anchor
        projectable_by_source[source_index] = bool(
            anchor.lexical_piece_indices
            and _source_has_lexical_clip_piece(source_token, tokenizer)
        )
    return prompt_anchors, anchor_by_source, projectable_by_source


def _alignment_payload(
    sentence: AutoSentence,
    source_tokens: Sequence[ParseToken],
    anchor_by_source: Mapping[int, ClipSourceAnchor],
    projectable_by_source: Mapping[int, bool],
    leaf_roles: Mapping[int, str],
    selected_roles: Mapping[int, set[str]],
) -> tuple[Mapping[str, Any], ...]:
    payload: list[Mapping[str, Any]] = []
    for leaf, source_token in zip(sentence.leaves, source_tokens):
        anchor = anchor_by_source[leaf.source_index]
        roles = sorted(selected_roles.get(leaf.source_index, ()))
        projectable = bool(projectable_by_source[leaf.source_index])
        payload.append(
            {
                "status": "projected" if projectable else "unprojectable",
                "source_index": leaf.source_index,
                "text": leaf.word,
                "auto_category": leaf.category,
                "projected_categories": list(leaf.projected_categories),
                "fragment_index": leaf.fragment_index,
                "structural_role": leaf_roles[leaf.source_index],
                "pd_roles": roles or ["non_pd_token"],
                "char_start": source_token.char_start,
                "char_end": source_token.char_end,
                "prompt_word_index": anchor.source_index,
                "prompt_word": anchor.surface,
                "clip_piece_indices": list(anchor.piece_indices),
                "clip_lexical_piece_indices": list(anchor.lexical_piece_indices),
                "clip_index": anchor.anchor_index if projectable else None,
                "anchor_kind": anchor.anchor_kind,
                "pd_anchor_policy": (
                    "full_prompt_word_final_lexical_piece_once"
                    if projectable
                    else "unprojectable_source"
                ),
            }
        )
    return tuple(payload)


def _validate_pd_indices(indices: PDIndices, *, clip_full_length: int) -> None:
    pairs = {
        "corr": {tuple(pair) for pair in indices.corr},
        "leak": {tuple(pair) for pair in indices.leak},
        "possession": {tuple(pair) for pair in indices.possession},
    }
    for field_name, field_pairs in pairs.items():
        if any(left == right for left, right in field_pairs):
            raise MarkedDepsError(f"{field_name} contains a self-pair")
    overlap = pairs["corr"] & pairs["leak"]
    if overlap:
        raise MarkedDepsError(f"corr/leak guidance conflict: {sorted(overlap)!r}")
    exist = set(indices.exist)
    if any(head not in exist for _modifier, head in pairs["corr"]):
        raise MarkedDepsError("corr contains a non-entity head")
    if any(target not in exist for _modifier, target in pairs["leak"]):
        raise MarkedDepsError("leak contains a non-entity target")
    if any(
        holder not in exist or object_ not in exist
        for holder, object_ in pairs["possession"]
    ):
        raise MarkedDepsError("possession contains a non-entity endpoint")
    all_indices = set(indices.exist)
    for field_pairs in pairs.values():
        for left, right in field_pairs:
            all_indices.update((left, right))
    if any(index <= 0 or index >= clip_full_length - 1 for index in all_indices):
        raise MarkedDepsError("PD index is outside the usable CLIP content window")


def _project_source_graph(
    *,
    source_tokens: Sequence[ParseToken],
    entity_sources: set[int],
    corr_sources: Mapping[tuple[int, int], set[int]],
    leak_sources: set[tuple[int, int]],
    possession_sources: Mapping[tuple[int, int], set[int]],
    anchor_by_source: Mapping[int, ClipSourceAnchor],
    projectable_by_source: Mapping[int, bool],
) -> tuple[PDIndices, Mapping[str, Any]]:
    """Project one completed source graph to full-prompt CLIP anchors once."""

    unprojectable_contributions = {
        "exist": 0,
        "corr": 0,
        "leak": 0,
        "possession": 0,
    }
    collapsed_self_pairs = {"corr": 0, "leak": 0, "possession": 0}

    def clip_index(source: int) -> Optional[int]:
        if not projectable_by_source[source]:
            return None
        return anchor_by_source[source].anchor_index

    exist_set: set[int] = set()
    for source in entity_sources:
        projected = clip_index(source)
        if projected is None:
            unprojectable_contributions["exist"] += 1
            continue
        exist_set.add(projected)

    def project_pairs(
        field_name: str,
        source_pairs: Sequence[tuple[int, int]],
    ) -> set[tuple[int, int]]:
        projected_pairs: set[tuple[int, int]] = set()
        for left_source, right_source in source_pairs:
            left = clip_index(left_source)
            right = clip_index(right_source)
            if left is None or right is None:
                unprojectable_contributions[field_name] += 1
                continue
            if left == right:
                collapsed_self_pairs[field_name] += 1
                continue
            projected_pairs.add((left, right))
        return projected_pairs

    corr_set = project_pairs("corr", sorted(corr_sources))
    leak_set = project_pairs("leak", sorted(leak_sources))
    possession_set = project_pairs("possession", sorted(possession_sources))
    corr_leak_collisions = sorted(corr_set & leak_set)
    leak_set -= corr_set

    indices = PDIndices(
        exist=sorted(exist_set),
        corr=[
            list(pair)
            for pair in sorted(corr_set, key=lambda pair: (pair[1], pair[0]))
        ],
        leak=[list(pair) for pair in sorted(leak_set)],
        possession=[list(pair) for pair in sorted(possession_set)],
    )
    selected_sources = set(entity_sources)
    for pairs in (corr_sources, leak_sources, possession_sources):
        for left, right in pairs:
            selected_sources.update((left, right))
    unprojectable_sources = [
        {
            "source_index": source,
            "text": source_tokens[source - 1].text,
            "reason": "unprojectable_source",
        }
        for source in sorted(selected_sources)
        if not projectable_by_source[source]
    ]
    prompt_word_by_source = {
        str(source): anchor_by_source[source].source_index
        for source in range(1, len(source_tokens) + 1)
    }
    return indices, {
        "version": PROJECTION_VERSION,
        "source_graph_completed_before_projection": True,
        "clip_projection_passes": 1,
        "prompt_word_by_parse_source": prompt_word_by_source,
        "unprojectable_sources": unprojectable_sources,
        "unprojectable_contributions": unprojectable_contributions,
        "collapsed_self_pairs": collapsed_self_pairs,
        "corr_leak_collisions_removed": [list(pair) for pair in corr_leak_collisions],
    }


def derive_guidance_targets(  # noqa: C901
    prompt: str,
    sentence: AutoSentence,
    dependencies: Sequence[MarkedDependency],
    tokenizer: Any,
    *,
    source_tokens: Sequence[ParseToken] | None = None,
    possession_lemmas_by_source: Mapping[int, str] | None = None,
    parse_status: Optional[str] = None,
    attachment_possession: bool = False,
) -> CompiledGuidanceTargets:
    """Derive the guidance targets I_D = (E, C, K, P) of one sentence (paper Sec. IV-A).

    ``attachment_possession=False`` (the default) is the standard transform.
    ``attachment_possession=True`` additionally applies the category-only
    attachment-functor possession rule and records its frames, statistics, and
    rule ID in the trace (used for the TAB interpretations).
    """

    resolved_source_tokens = (
        tuple(source_tokens)
        if source_tokens is not None
        else _default_source_tokens(prompt)
    )
    leaves = sentence.leaves
    _validate_source_tokens(prompt, resolved_source_tokens, sentence)
    if [leaf.source_index for leaf in leaves] != list(range(1, len(leaves) + 1)):
        raise MarkedDepsError("AUTO leaf source indices must be contiguous and one-based")

    leaf_by_source = {leaf.source_index: leaf for leaf in leaves}
    leaf_categories = {
        leaf.source_index: _parse_category(
            leaf.category, location=f"AUTO leaf {leaf.source_index}"
        )
        for leaf in leaves
    }
    leaf_roles = {
        source: _category_role(category)
        for source, category in leaf_categories.items()
    }
    possession_lemmas = _validate_possession_annotations(
        possession_lemmas_by_source or {}, leaf_by_source
    )

    marked_categories: list[CCGCategory] = []
    dispositions: list[_DependencyDisposition] = []
    dependencies_by_head: dict[int, list[tuple[int, MarkedDependency]]] = {}
    for dependency_index, dependency in enumerate(dependencies):
        for source, word, endpoint_role in (
            (dependency.head_index, dependency.head_word, "head"),
            (dependency.argument_index, dependency.argument_word, "argument"),
        ):
            leaf = leaf_by_source.get(source)
            if leaf is None:
                raise MarkedDepsError(
                    f"dependency {dependency_index} {endpoint_role} is outside AUTO leaves"
                )
            if leaf.word != word:
                raise MarkedDepsError(
                    f"dependency {dependency_index} {endpoint_role} surface mismatch: "
                    f"{word!r} != {leaf.word!r}"
                )
        category = _parse_category(
            dependency.marked_category,
            location=f"dependency {dependency_index} marked category",
        )
        marked_categories.append(category)
        disposition = _DependencyDisposition(
            dependency_index=dependency_index,
            dependency=dependency,
            marked_role=_category_role(category),
        )
        dispositions.append(disposition)
        dependencies_by_head.setdefault(dependency.head_index, []).append(
            (dependency_index, dependency)
        )

    entity_sources = {
        source for source, role in leaf_roles.items() if role == "entity"
    }
    if not entity_sources:
        raise UnsupportedPromptProjectionError(
            "no atomic-N entity exists; the no-fallback Stage 3 contract cannot compile"
        )

    coordination_groups = _build_coordination_groups(
        dispositions, leaf_roles
    )

    corr_sources: dict[tuple[int, int], set[int]] = {}
    modifier_resolution_trace: list[dict[str, Any]] = []
    for dependency_index, dependency in enumerate(dependencies):
        if not (
            dispositions[dependency_index].marked_role == "modifier"
            and dependency.slot == 1
        ):
            continue
        if leaf_roles[dependency.head_index] != "modifier":
            raise MarkedDepsError(
                "marked/AUTO modifier category mismatch at source "
                f"{dependency.head_index}"
            )
        modifier_sources = (dependency.head_index,)
        resolution = _resolve_entity_endpoint(
            dependency.argument_index,
            entity_sources=entity_sources,
            dependencies_by_head=dependencies_by_head,
            dispositions=dispositions,
        )
        if not resolution.entities:
            dispositions[dependency_index].reject(
                "unresolved_modifier_entity_endpoint"
            )
        for modifier_source in modifier_sources:
            for entity_source, path_evidence in resolution.entities.items():
                evidence = {
                    dependency_index,
                    *path_evidence,
                }
                corr_sources.setdefault(
                    (modifier_source, entity_source), set()
                ).update(evidence)
                for evidence_index in evidence:
                    dispositions[evidence_index].use("corr")
        modifier_resolution_trace.append(
            {
                "kind": "direct",
                "dependency_index": dependency_index,
                "modifier_sources": list(modifier_sources),
                "raw_argument_source": dependency.argument_index,
                "resolved_entity_sources": sorted(resolution.entities),
                "resolution_paths": list(resolution.paths),
                "visited_sources": list(resolution.visited_sources),
            }
        )

    higher_order_resolution_trace: list[dict[str, Any]] = []
    for dependency_index, dependency in enumerate(dependencies):
        if not (
            dispositions[dependency_index].marked_role == "modifier_of_modifier"
            and dependency.slot == 1
        ):
            continue
        if leaf_roles[dependency.head_index] != "modifier_of_modifier":
            raise MarkedDepsError(
                "marked/AUTO higher-order modifier category mismatch at source "
                f"{dependency.head_index}"
            )
        resolution = _resolve_entity_endpoint(
            dependency.argument_index,
            entity_sources=entity_sources,
            dependencies_by_head=dependencies_by_head,
            dispositions=dispositions,
        )
        if not resolution.entities:
            dispositions[dependency_index].reject(
                "unresolved_higher_order_modifier_entity_endpoint"
            )
        for entity_source, path_evidence in resolution.entities.items():
            evidence = {
                dependency_index,
                *path_evidence,
            }
            corr_sources.setdefault(
                (dependency.head_index, entity_source), set()
            ).update(evidence)
            for evidence_index in evidence:
                dispositions[evidence_index].use("corr")
        higher_order_resolution_trace.append(
            {
                "kind": "higher_order",
                "dependency_index": dependency_index,
                "modifier_sources": [dependency.head_index],
                "raw_argument_source": dependency.argument_index,
                "resolved_entity_sources": sorted(resolution.entities),
                "resolution_paths": list(resolution.paths),
                "visited_sources": list(resolution.visited_sources),
            }
        )

    possession_sources: dict[tuple[int, int], set[int]] = {}
    possession_trace: list[dict[str, Any]] = []
    auto_subject_bindings: list[Mapping[str, Any]] = []
    for predicate_source, lemma in sorted(possession_lemmas.items()):
        predicate_dependencies = dependencies_by_head.get(predicate_source, ())
        predicate_categories = [
            leaf_categories[predicate_source],
            *[
                marked_categories[dependency_index]
                for dependency_index, _dependency in predicate_dependencies
            ],
        ]
        passive = any(is_passive_verb_frame(category) for category in predicate_categories)
        by_slot: dict[int, list[tuple[int, MarkedDependency]]] = {1: [], 2: []}
        for dependency_index, dependency in predicate_dependencies:
            if dependency.slot in by_slot:
                by_slot[dependency.slot].append((dependency_index, dependency))
        if passive:
            for dependency_index, _dependency in by_slot[1] + by_slot[2]:
                dispositions[dependency_index].reject("passive_possession_frame")
            possession_trace.append(
                {
                    "predicate_source": predicate_source,
                    "lemma": lemma,
                    "passive": True,
                    "subject_sources": [],
                    "object_sources": [],
                    "pairs": [],
                    "reason": "passive_possession_frame",
                }
            )
            continue

        slot_resolution: dict[int, dict[int, set[int]]] = {1: {}, 2: {}}
        slot_paths: dict[int, list[Mapping[str, Any]]] = {1: [], 2: []}
        for slot in (1, 2):
            for dependency_index, dependency in by_slot[slot]:
                resolution = _resolve_entity_endpoint(
                    dependency.argument_index,
                    entity_sources=entity_sources,
                    dependencies_by_head=dependencies_by_head,
                    dispositions=dispositions,
                )
                slot_paths[slot].extend(resolution.paths)
                if not resolution.entities:
                    dispositions[dependency_index].reject(
                        f"unresolved_possession_slot_{slot}_endpoint"
                    )
                for entity_source, path_evidence in resolution.entities.items():
                    slot_resolution[slot].setdefault(entity_source, set()).update(
                        {dependency_index, *path_evidence}
                    )

        subject_binding: Optional[Mapping[str, Any]] = None
        if not slot_resolution[1]:
            subject_binding = _nominal_modifier_subject_binding(
                sentence,
                predicate_source,
                entity_sources=entity_sources,
            )
            if subject_binding is not None:
                subject_source = int(subject_binding["subject_source"])
                slot_resolution[1][subject_source] = set()
                slot_paths[1].append(
                    {
                        "source_path": [predicate_source, subject_source],
                        "resolved_entity": subject_source,
                        "dependency_indices": [],
                        "evidence": subject_binding["evidence"],
                    }
                )
                auto_subject_bindings.append(subject_binding)

        frame_pairs: list[tuple[int, int]] = []
        for subject_source, subject_evidence in slot_resolution[1].items():
            for object_source, object_evidence in slot_resolution[2].items():
                if subject_source == object_source:
                    continue
                pair = (subject_source, object_source)
                evidence = subject_evidence | object_evidence
                possession_sources.setdefault(pair, set()).update(evidence)
                frame_pairs.append(pair)
                for dependency_index in evidence:
                    dispositions[dependency_index].use("possession")
        if not frame_pairs:
            reason = (
                "unresolved_possession_subject_and_object"
                if not slot_resolution[1] and not slot_resolution[2]
                else "unresolved_possession_subject"
                if not slot_resolution[1]
                else "unresolved_possession_object"
                if not slot_resolution[2]
                else "possession_self_pair_only"
            )
            for dependency_index, _dependency in by_slot[1] + by_slot[2]:
                dispositions[dependency_index].reject(reason)
        else:
            reason = None
        possession_trace.append(
            {
                "predicate_source": predicate_source,
                "lemma": lemma,
                "passive": False,
                "raw_subject_sources": sorted(
                    dependency.argument_index for _index, dependency in by_slot[1]
                ),
                "raw_object_sources": sorted(
                    dependency.argument_index for _index, dependency in by_slot[2]
                ),
                "subject_sources": sorted(slot_resolution[1]),
                "object_sources": sorted(slot_resolution[2]),
                "subject_resolution_paths": slot_paths[1],
                "object_resolution_paths": slot_paths[2],
                "nominal_modifier_subject_binding": subject_binding,
                "pairs": [list(pair) for pair in sorted(set(frame_pairs))],
                "reason": reason,
            }
        )

    attachment_frames: list[dict[str, Any]] = []
    attachment_stats: dict[str, int] = {}
    if attachment_possession:
        # After the lemma frames so both paths fill one possession map, and
        # before the fallback loop below so the rule's evidence is recorded as
        # a use rather than as a reject.
        attachment_frames, attachment_stats = _emit_attachment_possession(
            dependencies=dependencies,
            marked_categories=marked_categories,
            leaf_by_source=leaf_by_source,
            leaf_categories=leaf_categories,
            entity_sources=entity_sources,
            dependencies_by_head=dependencies_by_head,
            dispositions=dispositions,
            possession_sources=possession_sources,
        )

    for disposition in dispositions:
        if not disposition.uses and not disposition.reasons:
            if disposition.marked_role == "modifier" and disposition.dependency.slot != 1:
                disposition.reject("modifier_dependency_is_not_slot_1")
            elif disposition.marked_role == "modifier_of_modifier":
                disposition.reject("higher_order_modifier_dependency_is_not_slot_1")
            elif disposition.marked_role == "coordination":
                disposition.reject("coordination_did_not_form_a_group")
            else:
                disposition.reject("fixed_pd_contract_has_no_index_for_category_or_slot")

    unresolved_modifiers = [
        {
            "source_index": source,
            "category": leaf_by_source[source].category,
            "reason": "no_resolved_slot_1_entity",
        }
        for source, role in sorted(leaf_roles.items())
        if role in {"modifier", "modifier_of_modifier"}
        and not any(modifier == source for modifier, _entity in corr_sources)
    ]

    leak_sources: set[tuple[int, int]] = set()
    correct_heads_by_modifier: dict[int, set[int]] = {}
    for modifier_source, entity_source in corr_sources:
        correct_heads_by_modifier.setdefault(modifier_source, set()).add(entity_source)
    for modifier_source, correct_heads in correct_heads_by_modifier.items():
        for entity_source in entity_sources:
            if entity_source == modifier_source or entity_source in correct_heads:
                continue
            leak_sources.add((modifier_source, entity_source))

    coordination_evidence = []
    for group in coordination_groups:
        member_set = set(group.members)
        coordination_evidence.append(
            {
                **group.payload(),
                "corr_pairs_explicit_in_marked_dependencies": [
                    [modifier, entity]
                    for modifier, entity in sorted(corr_sources)
                    if modifier in member_set or entity in member_set
                ],
                "possession_pairs_explicit_in_marked_dependencies": [
                    [holder, object_]
                    for holder, object_ in sorted(possession_sources)
                    if holder in member_set or object_ in member_set
                ],
                "implicit_pairs_added": 0,
            }
        )

    selected_roles: dict[int, set[str]] = {}
    for source in entity_sources:
        selected_roles.setdefault(source, set()).add("exist")
    for modifier, entity in corr_sources:
        selected_roles.setdefault(modifier, set()).add("corr_modifier")
        selected_roles.setdefault(entity, set()).add("corr_entity")
    for modifier, entity in leak_sources:
        selected_roles.setdefault(modifier, set()).add("leak_modifier")
        selected_roles.setdefault(entity, set()).add("leak_entity")
    for holder, object_ in possession_sources:
        selected_roles.setdefault(holder, set()).add("possession_holder")
        selected_roles.setdefault(object_, set()).add("possession_object")

    prompt_anchors, anchor_by_source, projectable_by_source = (
        _full_prompt_anchors_by_source(prompt, resolved_source_tokens, tokenizer)
    )
    indices, projection_trace = _project_source_graph(
        source_tokens=resolved_source_tokens,
        entity_sources=entity_sources,
        corr_sources=corr_sources,
        leak_sources=leak_sources,
        possession_sources=possession_sources,
        anchor_by_source=anchor_by_source,
        projectable_by_source=projectable_by_source,
    )
    clip_full_length = max(
        anchor.piece_indices[-1] for anchor in prompt_anchors
    ) + 2
    _validate_pd_indices(indices, clip_full_length=clip_full_length)

    leaf_dispositions: list[dict[str, Any]] = []
    for leaf in leaves:
        role = leaf_roles[leaf.source_index]
        pd_roles = sorted(selected_roles.get(leaf.source_index, ()))
        if pd_roles:
            reason = None
        elif role in {"modifier", "modifier_of_modifier"}:
            reason = "no_resolved_slot_1_entity"
        elif role == "atomic_np":
            reason = "atomic_NP_cannot_distinguish_proper_name_from_pronoun_without_extra_lexical_evidence"
        else:
            reason = "fixed_pd_contract_has_no_index_for_category"
        leaf_dispositions.append(
            {
                "source_index": leaf.source_index,
                "text": leaf.word,
                "auto_category": leaf.category,
                "structural_role": role,
                "pd_roles": pd_roles,
                "reason": reason,
            }
        )

    trace: dict[str, Any] = {
        "method": METHOD_ID,
        "profile": PROFILE_ID,
        "parse_status": parse_status
        or ("partial" if sentence.root.category == "PARTIAL" else "complete"),
        "auto_id": sentence.id_line,
        "auto_token_count": len(leaves),
        "dependency_count": len(dependencies),
        "clip_full_length": clip_full_length,
        "possession_lemma_set": sorted(POSSESSION_LEMMAS),
        "possession_lemma_annotations": [
            {"source_index": source, "lemma": lemma}
            for source, lemma in sorted(possession_lemmas.items())
        ],
        "source_indices": {
            "entity": sorted(entity_sources),
            "corr": [list(pair) for pair in sorted(corr_sources)],
            "leak": [list(pair) for pair in sorted(leak_sources)],
            "possession": [list(pair) for pair in sorted(possession_sources)],
        },
        "projection": projection_trace,
        "coordination_evidence": coordination_evidence,
        "modifier_resolution": modifier_resolution_trace,
        "higher_order_modifier_resolution": higher_order_resolution_trace,
        "possession_frames": possession_trace,
        "nominal_modifier_subject_bindings": [
            dict(binding) for binding in auto_subject_bindings
        ],
        "unresolved_modifiers": unresolved_modifiers,
        "leaf_dispositions": leaf_dispositions,
        "dependency_dispositions": [
            disposition.payload() for disposition in dispositions
        ],
        "invariants": {
            "four_pd_fields_only": True,
            "corr_leak_disjoint": True,
            "all_pair_entity_endpoints_in_exist": True,
            "no_self_pairs": True,
            "all_leaves_have_dispositions": len(leaf_dispositions) == len(leaves),
            "all_dependencies_have_dispositions": len(dispositions) == len(dependencies),
            "lexical_semantic_comparisons": sorted(POSSESSION_LEMMAS),
            "fallback_edges": 0,
            "coordination_implicit_pairs": 0,
            "nominal_modifier_subject_binding_count": len(auto_subject_bindings),
            "source_graph_completed_before_projection": True,
            "clip_projection_passes": 1,
            "feature_preserving_nominal_modifier_only": True,
            "higher_order_nominal_composition": True,
        },
    }
    if attachment_possession:
        trace["attachment_possession_frames"] = attachment_frames
        trace["attachment_possession_stats"] = attachment_stats
        trace["invariants"]["attachment_possession_rule"] = (
            ATTACHMENT_POSSESSION_RULE_ID
        )
    return CompiledGuidanceTargets(
        indices=indices,
        alignment=_alignment_payload(
            sentence,
            resolved_source_tokens,
            anchor_by_source,
            projectable_by_source,
            leaf_roles,
            selected_roles,
        ),
        trace=trace,
    )


# --- SD1.4 transforms and the identity of guidance-target files ---------------------------------------

# Providers of the guidance-target files the SD1.4 driver reads, and their arms.
TARGET_FILE_ARMS: Mapping[str, str] = MappingProxyType(
    {
        SD14_GUIDANCE_TARGET_PROVIDER: SD14_CCG_ARM,
        SD14_PREDICATED_DIFFUSION_PROVIDER: SD14_PREDICATED_DIFFUSION_ARM,
    }
)


def sd14_guidance_targets(
    prompt: str,
    sentence: AutoSentence,
    dependencies: Sequence[MarkedDependency],
    tokenizer: Any,
    *,
    source_tokens: Sequence[ParseToken] | None = None,
    possession_lemmas_by_source: Mapping[int, str] | None = None,
    parse_status: str | None = None,
) -> CompiledGuidanceTargets:
    """:func:`derive_guidance_targets` behind the pinned SD1.4 CLIP tokenizer identity."""

    require_sd14_tokenizer(tokenizer)

    return derive_guidance_targets(
        prompt,
        sentence,
        dependencies,
        tokenizer,
        source_tokens=source_tokens,
        possession_lemmas_by_source=possession_lemmas_by_source,
        parse_status=parse_status,
    )


def sd14_interpretation_guidance_targets(
    prompt: str,
    sentence: AutoSentence,
    dependencies: Sequence[MarkedDependency],
    tokenizer: Any,
    *,
    source_tokens: Sequence[ParseToken] | None = None,
    possession_lemmas_by_source: Mapping[int, str] | None = None,
    parse_status: str | None = None,
) -> CompiledGuidanceTargets:
    """:func:`sd14_guidance_targets` plus the attachment-possession rule (TAB interpretations).

    The attachment frames of the trace identify the holder and the object of
    the PP attachment. The possession lemma path is unchanged.
    """

    require_sd14_tokenizer(tokenizer)

    return derive_guidance_targets(
        prompt,
        sentence,
        dependencies,
        tokenizer,
        source_tokens=source_tokens,
        possession_lemmas_by_source=possession_lemmas_by_source,
        parse_status=parse_status,
        attachment_possession=True,
    )


def _require_provider(expected_provider: str) -> None:
    if expected_provider not in TARGET_FILE_ARMS:
        raise ValueError(f"unsupported SD1.4 sidecar provider: {expected_provider}")


def require_target_file_metadata(
    metadata: Mapping[str, Any],
    *,
    expected_provider: str,
) -> None:
    """Reject guidance-target file metadata not sealed to the SD1.4 backbone, tokenizer and provider.

    ``canonical_arm_id`` is not checked here: every recorded SD1.4 file
    metadata predates the field. The row gate checks the arm of every row.
    """

    _require_provider(expected_provider)
    expected = {
        "backbone": "sd14",
        "provider": expected_provider,
        "tokenizer": SD14_GENERATION.tokenizer_identity(),
    }
    mismatches = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"SD1.4 sidecar metadata identity mismatch: {mismatches}")


def require_target_row_identity(
    row: Mapping[str, Any],
    *,
    expected_provider: str,
) -> None:
    """Reject a guidance-target row carrying another backbone's or an unpinned provenance."""

    _require_provider(expected_provider)
    parser_provenance = row.get("parser_provenance")
    if not isinstance(parser_provenance, Mapping):
        raise ValueError("SD1.4 sidecar row parser_provenance must be an object")
    expected_row_provider = row.get("provider")
    if expected_row_provider != expected_provider:
        raise ValueError(
            "SD1.4 sidecar row provider mismatch: "
            f"expected {expected_provider!r}, got {expected_row_provider!r}"
        )
    expected = {
        "backbone": "sd14",
        "sidecar_provider": expected_provider,
        "tokenizer": SD14_GENERATION.tokenizer_identity(),
        "canonical_arm_id": TARGET_FILE_ARMS[expected_provider],
    }
    mismatches = {
        key: {"expected": value, "actual": parser_provenance.get(key)}
        for key, value in expected.items()
        if parser_provenance.get(key) != value
    }
    if mismatches:
        raise ValueError(f"SD1.4 sidecar row identity mismatch: {mismatches}")


# --- per-category guidance-target files (T2I-CompBench++) ---------------------------------------------

PROMPT_ONLY_FIELDS = frozenset({"experiment", "prompt_id", "prompt", "seed"})
_FAIL_ROOT_CATEGORIES = frozenset({"FAIL", "OVERLEN"})
_PARTIAL_ROOT_CATEGORY = "PARTIAL"
# Stage 2 permits other root sets on its command line, but its coverage summary
# does not seal them, so only the producer defaults are accepted here.
_VALID_COVERAGE_STATES = frozenset(
    {
        ("complete", False, False),
        ("complete", True, True),
        ("partial", False, False),
        ("partial", True, False),
        ("partial", True, True),
        ("fail", False, False),
    }
)


class GuidanceTargetError(RuntimeError):
    """A target-corpus or Stage 3 artifact contract was violated."""


def _validate_coverage_state(
    coverage_row: Mapping[str, Any],
    *,
    auto_root_category: str,
    location: str,
) -> tuple[str, bool, bool]:
    """Validate one Stage 2 coverage state before failure classification."""

    parse_status = coverage_row.get("parse_status")
    any_success = coverage_row.get("any_generate_unit_success")
    all_success = coverage_row.get("all_generate_units_success")
    root_category = coverage_row.get("root_category")
    if type(parse_status) is not str or parse_status not in {
        "complete",
        "partial",
        "fail",
    }:
        raise GuidanceTargetError(
            f"{location}.parse_status must be complete, partial, or fail"
        )
    if type(any_success) is not bool:
        raise GuidanceTargetError(
            f"{location}.any_generate_unit_success must be a boolean"
        )
    if type(all_success) is not bool:
        raise GuidanceTargetError(
            f"{location}.all_generate_units_success must be a boolean"
        )
    state = (parse_status, any_success, all_success)
    if state not in _VALID_COVERAGE_STATES:
        raise GuidanceTargetError(
            f"{location} has an impossible Stage 2 coverage state: {state}"
        )
    if type(root_category) is not str or not root_category:
        raise GuidanceTargetError(
            f"{location}.root_category must be non-empty text"
        )
    if root_category != auto_root_category:
        raise GuidanceTargetError(
            f"{location}.root_category does not match AUTO: "
            f"{root_category!r}/{auto_root_category!r}"
        )
    expected_status = (
        "fail"
        if auto_root_category in _FAIL_ROOT_CATEGORIES
        else (
            "partial"
            if auto_root_category == _PARTIAL_ROOT_CATEGORY
            else "complete"
        )
    )
    if parse_status != expected_status:
        raise GuidanceTargetError(
            f"{location}.parse_status does not match AUTO root category: "
            f"{parse_status!r}/{auto_root_category!r}"
        )
    return state


def target_file_metadata_path(output: Path) -> Path:
    """``<output>.meta.json``."""
    return output.with_name(output.name + ".meta.json")


def target_file_trace_path(output: Path) -> Path:
    """``<output>.conversion-trace.jsonl``."""
    return output.with_name(output.name + ".conversion-trace.jsonl")


def load_category_prompt_rows(
    path: Path,
    *,
    expected_count: int = 300,
) -> tuple[str, list[dict[str, Any]]]:
    """Load exactly one fixed T2I-CompBench++ prompt-only category."""

    raw_rows = read_jsonl(path, reject_duplicate_keys=True)
    if len(raw_rows) != expected_count:
        raise GuidanceTargetError(
            f"{path}: expected {expected_count} rows, found {len(raw_rows)}"
        )
    rows: list[dict[str, Any]] = []
    categories: set[str] = set()
    prompt_ids: set[str] = set()
    for row_number, raw_row in enumerate(raw_rows, start=1):
        row = dict(raw_row)
        if set(row) != PROMPT_ONLY_FIELDS:
            raise GuidanceTargetError(
                f"{path}:{row_number}: prompt-only fields must be "
                f"{sorted(PROMPT_ONLY_FIELDS)}"
            )
        if row["seed"] != 0 or type(row["seed"]) is not int:
            raise GuidanceTargetError(
                f"{path}:{row_number}: Stage 3 prompt-only seed must be integer 0"
            )
        for field_name in ("experiment", "prompt_id", "prompt"):
            if not isinstance(row[field_name], str) or not row[field_name]:
                raise GuidanceTargetError(
                    f"{path}:{row_number}: {field_name} must be a non-empty string"
                )
        experiment = row["experiment"]
        if not experiment.startswith("t2icb_"):
            raise GuidanceTargetError(
                f"{path}:{row_number}: experiment is not T2I-CompBench++"
            )
        category = experiment.removeprefix("t2icb_")
        if category not in T2I_COMPBENCH_CATEGORIES:
            raise GuidanceTargetError(
                f"{path}:{row_number}: unsupported T2I-CompBench++ category {category!r}"
            )
        if not row["prompt_id"].startswith(experiment + "_"):
            raise GuidanceTargetError(
                f"{path}:{row_number}: prompt_id does not match experiment"
            )
        if row["prompt_id"] in prompt_ids:
            raise GuidanceTargetError(
                f"{path}:{row_number}: duplicate prompt_id {row['prompt_id']!r}"
            )
        prompt_ids.add(row["prompt_id"])
        categories.add(category)
        rows.append(row)
    if len(categories) != 1:
        raise GuidanceTargetError(
            f"{path}: one invocation must contain exactly one category; found {sorted(categories)}"
        )
    return next(iter(categories)), rows


def build_guidance_target_rows(  # noqa: C901
    prompt_rows: Sequence[Mapping[str, Any]],
    auto_sentences: Sequence[AutoSentence],
    dependency_blocks: Sequence[Sequence[MarkedDependency]],
    tokenizer: Any,
    nlp: Any,
    *,
    category: str,
    coverage_rows: Sequence[Mapping[str, Any]],
    tokenizer_identity: Mapping[str, Any],
    sentence_map_rows: Sequence[SentenceMapRow] | None = None,
    target_dataset: str = "T2I-CompBench++",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compile the SD1.4 guidance-target rows and conversion-trace rows of one prompt list.

    Stage 1-2 failures become rows with an empty program and a failure status
    (``stage1_parse_failure``, ``stage2_dependency_failure``,
    ``stage3_projection_failure``) instead of stopping the build. Without
    ``sentence_map_rows`` every prompt is tokenized here (single-prompt front
    end); with them, the recorded sentence map is the token source.
    """

    if not isinstance(target_dataset, str) or not target_dataset:
        raise GuidanceTargetError("target_dataset must be non-empty text")
    expected_identity = SD14_GENERATION.tokenizer_identity()
    if tokenizer_identity != expected_identity:
        invalid = {"tokenizer_identity": {"expected": expected_identity, "actual": tokenizer_identity}}
        raise GuidanceTargetError(f"SD1.4 Stage 3 identity tuple mismatch: {invalid}")

    if not (
        len(prompt_rows) == len(auto_sentences) == len(dependency_blocks)
    ):
        raise GuidanceTargetError(
            "prompt/AUTO/dependency block count mismatch: "
            f"{len(prompt_rows)}/{len(auto_sentences)}/{len(dependency_blocks)}"
        )
    if len(coverage_rows) != len(prompt_rows):
        raise GuidanceTargetError(
            "prompt/coverage row count mismatch: "
            f"{len(prompt_rows)}/{len(coverage_rows)}"
        )

    if sentence_map_rows is None:
        source_token_rows: list[tuple[ParseToken, ...]] = []
        map_provenance: list[Mapping[str, Any]] = []
        coverage_sentence_indices: list[int | None] = []
        for prompt_row in prompt_rows:
            prompt = str(prompt_row["prompt"])
            tokenized_sentences = split_and_tokenize(prompt)
            if len(tokenized_sentences) != 1:
                raise GuidanceTargetError(
                    "the T2I-CompBench++ Stage 3 entry point requires exactly "
                    f"one parsed sentence per prompt: {prompt_row['prompt_id']!r}"
                )
            tokenized = tokenized_sentences[0]
            source_token_rows.append(tokenized.tokens)
            coverage_sentence_indices.append(None)
            map_provenance.append(
                {
                    "sentence_index": tokenized.sentence_index,
                    "raw_line_index": None,
                    "source": "deterministic_sentence_tokenizer",
                }
            )
    else:
        groups = group_sentence_map(sentence_map_rows)
        if len(groups) != len(prompt_rows):
            raise GuidanceTargetError(
                "prompt/sentence-map group count mismatch: "
                f"{len(prompt_rows)}/{len(groups)}"
            )
        source_token_rows = []
        map_provenance = []
        coverage_sentence_indices = []
        for prompt_position, (prompt_row, (map_prompt_id, group)) in enumerate(
            zip(prompt_rows, groups)
        ):
            prompt_id = str(prompt_row["prompt_id"])
            prompt = str(prompt_row["prompt"])
            if map_prompt_id != prompt_id:
                raise GuidanceTargetError(
                    "prompt/sentence-map order mismatch: "
                    f"{prompt_id!r}/{map_prompt_id!r}"
                )
            if len(group) != 1:
                raise GuidanceTargetError(
                    "the T2I-CompBench++ Stage 3 entry point requires exactly "
                    f"one parsed sentence per prompt: {prompt_id!r} has {len(group)}"
                )
            validate_sentence_map_group(prompt, group)
            map_row = group[0]
            if map_row.raw_line_index != prompt_position:
                raise GuidanceTargetError(
                    "prompt/sentence-map raw line mismatch: "
                    f"{prompt_id!r}/{map_row.raw_line_index}/{prompt_position}"
                )
            source_token_rows.append(map_row.tokens)
            # Stage 2 coverage uses the one-based global .auto sentence
            # ordinal.  This corpus has exactly one sentence per raw prompt,
            # while sentence_map.sentence_index is local to each prompt.
            coverage_sentence_indices.append(map_row.raw_line_index + 1)
            map_provenance.append(
                {
                    "sentence_index": map_row.sentence_index,
                    "raw_line_index": map_row.raw_line_index,
                    "source": "canonical_sentence_map",
                }
            )

    spacy_meta = getattr(nlp, "meta", {})
    if all(index is not None for index in coverage_sentence_indices):
        coverage_by_sentence_index: dict[int, Mapping[str, Any]] = {}
        for row_number, coverage_row in enumerate(coverage_rows, start=1):
            coverage_sentence_index = coverage_row.get("sentence_index")
            if type(coverage_sentence_index) is not int or coverage_sentence_index < 1:
                raise GuidanceTargetError(
                    f"coverage row {row_number}: sentence_index must be positive"
                )
            if coverage_sentence_index in coverage_by_sentence_index:
                raise GuidanceTargetError(
                    "duplicate coverage sentence_index: "
                    f"{coverage_sentence_index}"
                )
            coverage_by_sentence_index[coverage_sentence_index] = coverage_row
        expected_coverage_indices = {
            int(index) for index in coverage_sentence_indices if index is not None
        }
        if set(coverage_by_sentence_index) != expected_coverage_indices:
            raise GuidanceTargetError(
                "sentence-map/coverage index mismatch: "
                f"missing={sorted(expected_coverage_indices - set(coverage_by_sentence_index))[:10]}, "
                f"extra={sorted(set(coverage_by_sentence_index) - expected_coverage_indices)[:10]}"
            )
        resolved_coverage_rows = [
            coverage_by_sentence_index[int(index)]
            for index in coverage_sentence_indices
            if index is not None
        ]
    else:
        # Callers without a canonical sentence map (the single-prompt front
        # end) pass the coverage rows in prompt order.
        resolved_coverage_rows = list(coverage_rows)

    for row_number, (coverage_row, sentence) in enumerate(
        zip(resolved_coverage_rows, auto_sentences), start=1
    ):
        if not isinstance(coverage_row, Mapping):
            raise GuidanceTargetError(
                f"coverage row {row_number} must be an object"
            )
        _validate_coverage_state(
            coverage_row,
            auto_root_category=sentence.root.category,
            location=f"coverage row {row_number}",
        )
    target_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    for artifact_index, (
        prompt_row,
        sentence,
        dependencies,
        source_tokens,
        source_map,
        coverage_row,
    ) in enumerate(
        zip(
            prompt_rows,
            auto_sentences,
            dependency_blocks,
            source_token_rows,
            map_provenance,
            resolved_coverage_rows,
        ),
        start=1,
    ):
        prompt = str(prompt_row["prompt"])
        prompt_digest = sha256_text(prompt)
        coverage_prompt = coverage_row.get("prompt")
        parse_sentence = " ".join(token.text for token in source_tokens)
        if (
            source_map["source"] == "canonical_sentence_map"
            and not isinstance(coverage_prompt, str)
        ):
            raise GuidanceTargetError(
                f"coverage prompt missing for {prompt_row['prompt_id']!r}"
            )
        if coverage_prompt is not None and coverage_prompt != parse_sentence:
            raise GuidanceTargetError(
                f"coverage prompt mismatch for {prompt_row['prompt_id']!r}"
            )
        if (
            source_map["source"] == "canonical_sentence_map"
            and coverage_row.get("auto_id") != sentence.id_line
        ):
            raise GuidanceTargetError(
                f"coverage AUTO mismatch for {prompt_row['prompt_id']!r}"
            )
        parse_status = str(coverage_row["parse_status"])
        base_provenance: dict[str, Any] = {
            "provider": PROFILE_ID,
            "method": METHOD_ID,
            "canonical_arm_id": SD14_CCG_ARM,
            "sidecar_provider": SD14_GUIDANCE_TARGET_PROVIDER,
            "backbone": "sd14",
            "tokenizer": dict(tokenizer_identity),
            "target_dataset": target_dataset,
            "target_category": category,
            "artifact_index": artifact_index,
            "sentence_index": source_map["sentence_index"],
            "raw_line_index": source_map["raw_line_index"],
            "source_token_map": source_map["source"],
            "auto_id": sentence.id_line,
            "parse_status": parse_status,
            "projection_version": PROJECTION_VERSION,
            "possession_lemma_set": sorted(POSSESSION_LEMMAS),
            "spacy_syntax_used": False,
            "spacy_model": spacy_meta.get("name"),
            "spacy_model_version": spacy_meta.get("version"),
        }
        failure_status: str | None = None
        failure_code: str | None = None
        failure_error: str | None = None
        if parse_status == "fail":
            failure_status = "stage1_parse_failure"
            failure_code = "no_complete_or_partial_parse"
            failure_error = "HolCCG produced no complete or partial parse"
        elif coverage_row["any_generate_unit_success"] is False:
            failure_status = "stage2_dependency_failure"
            failure_code = "no_successful_dependency_unit"
            failure_error = "C&C generate produced no successful dependency unit"

        compiled = None
        if failure_status is None:
            annotations = possession_lemma_annotations_from_spacy(
                prompt, nlp, source_tokens=source_tokens
            )
            try:
                compiled = sd14_guidance_targets(
                    prompt,
                    sentence,
                    dependencies,
                    tokenizer,
                    source_tokens=source_tokens,
                    possession_lemmas_by_source=annotations,
                    parse_status=parse_status,
                )
            except UnsupportedPromptProjectionError as exc:
                failure_status = "stage3_projection_failure"
                failure_code = "unsupported_prompt_no_atomic_entity"
                failure_error = str(exc)

        if failure_status is not None:
            assert failure_code is not None
            assert failure_error is not None
            failure_trace = {
                "status": failure_status,
                "failure_code": failure_code,
                "stage": failure_status.split("_", 1)[0],
                "error": failure_error,
                "parse_status": parse_status,
                "coverage_row": dict(coverage_row),
            }
            trace_digest = sha256_text(canonical_json(failure_trace))
            target_rows.append(
                {
                    "prompt_id": prompt_row["prompt_id"],
                    "provider": SD14_GUIDANCE_TARGET_PROVIDER,
                    "prompt_sha256": prompt_digest,
                    "status": failure_status,
                    "pd_indices": {
                        "exist": [],
                        "corr": [],
                        "leak": [],
                        "possession": [],
                    },
                    "program_sha256": None,
                    "alignment": [],
                    "error": failure_error,
                    "failure_code": failure_code,
                    "parser_provenance": {
                        **base_provenance,
                        "failure_stage": failure_trace["stage"],
                        "failure_code": failure_code,
                        "conversion_trace_sha256": trace_digest,
                    },
                }
            )
            trace_rows.append(
                {
                    "prompt_id": prompt_row["prompt_id"],
                    "prompt_sha256": prompt_digest,
                    "program_sha256": None,
                    "conversion_trace_sha256": trace_digest,
                    "conversion_trace": failure_trace,
                }
            )
            continue

        assert compiled is not None
        program_digest = program_sha256(compiled.indices)
        trace_digest = sha256_text(canonical_json(compiled.trace))
        target_rows.append(
            {
                "prompt_id": prompt_row["prompt_id"],
                "provider": SD14_GUIDANCE_TARGET_PROVIDER,
                "prompt_sha256": prompt_digest,
                "status": "ok",
                "pd_indices": pd_indices_payload(compiled.indices),
                "program_sha256": program_digest,
                "alignment": [dict(item) for item in compiled.alignment],
                "parser_provenance": {
                    **base_provenance,
                    "parse_status": compiled.trace["parse_status"],
                    "conversion_trace_sha256": trace_digest,
                },
            }
        )
        trace_rows.append(
            {
                "prompt_id": prompt_row["prompt_id"],
                "prompt_sha256": prompt_digest,
                "program_sha256": program_digest,
                "conversion_trace_sha256": trace_digest,
                "conversion_trace": dict(compiled.trace),
            }
        )
    return target_rows, trace_rows


def load_possession_lemmatizer(model: str = "en_core_web_sm") -> Any:
    """Load the spaCy pipeline whose lemmatizer finds the possession verbs (parser, NER disabled)."""
    import spacy

    nlp = spacy.load(model, disable=["parser", "ner", "textcat"])
    if "lemmatizer" not in nlp.pipe_names:
        raise GuidanceTargetError(
            f"spaCy model {model!r} has no lemmatizer"
        )
    return nlp


def guidance_target_file_metadata(
    *,
    category: str,
    rows: Sequence[Mapping[str, Any]],
    output_record: provenance.ArtifactRecord,
    trace_record: provenance.ArtifactRecord,
    source_paths: Sequence[Path],
    tokenizer_identity: Mapping[str, Any],
    nlp: Any,
    spacy_model: str,
) -> dict[str, Any]:
    """Metadata of one SD1.4 guidance-target file (``<output>.meta.json``); paths are repository-relative."""
    root = provenance.repository_root()
    source_artifacts = []
    for path in source_paths:
        record = artifact_record(path).to_dict(relative_to=root)
        source_artifacts.append({"path": record["path"], "sha256": record["sha256"]})
    return {
        "artifact": "deps2index_sidecar",
        "method": METHOD_ID,
        "profile": SD14_GUIDANCE_TARGET_PROVIDER,
        "provider": SD14_GUIDANCE_TARGET_PROVIDER,
        "backbone": "sd14",
        "canonical_arm_id": SD14_CCG_ARM,
        "projection_version": PROJECTION_VERSION,
        "target_dataset": "T2I-CompBench++",
        "target_category": category,
        "row_count": len(rows),
        "status_counts": dict(
            sorted(Counter(str(row["status"]) for row in rows).items())
        ),
        "source_artifacts": source_artifacts,
        "output": output_record.to_dict(relative_to=root),
        "conversion_trace": trace_record.to_dict(relative_to=root),
        "tokenizer_model": SD14_GENERATION.model_id,
        "tokenizer_revision": SD14_GENERATION.model_revision,
        "tokenizer_subfolder": SD14_GENERATION.tokenizer_subfolder,
        "tokenizer": dict(tokenizer_identity),
        "spacy_version": importlib.metadata.version("spacy"),
        "spacy_model": spacy_model,
        "spacy_model_version": getattr(nlp, "meta", {}).get("version"),
        "spacy_usage": "lemmatizer_only_for_have_wear_grasp_hold",
        "pd_loss_contract_changed": False,
        "stage1_or_stage2_invoked": False,
    }


# --- Stage 1-3 source manifest -----------------------------------------------------------------------

# A source manifest names, per category, the Stage 1-3 files of one run.
SOURCE_MANIFEST_SCHEMA = "holsd.sd14_semantic_sources.v1"
SOURCE_FIELDS = frozenset({"auto", "marked_deps", "prompt_only", "pd_sidecar", "sentence_map"})


class SourceManifestError(ValueError):
    """A Stage 1-3 source manifest or one of its files failed an input or provenance check."""


@dataclass(frozen=True)
class SourcePrompt:
    """One verified base prompt of a source manifest with its stored Stage 1-3 records."""

    prompt_id: str
    prompt: str
    sentence: AutoSentence
    dependencies: tuple[MarkedDependency, ...]
    targets: CompiledGuidanceTargets
    pd_row: Mapping[str, Any]
    source_bundle_index: int
    source_row_index: int


def _root(repository_root: Path | None = None) -> Path:
    return (repository_root or provenance.repository_root()).resolve()


def _path(value: Any, *, repository_root: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise SourceManifestError("artifact path must be a nonempty string")
    return resolve_recorded_path(value, repository_root=repository_root)


def _object(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceManifestError(f"cannot read JSON object: {path}") from exc
    if not isinstance(result, dict):
        raise SourceManifestError(f"JSON object required: {path}")
    return result


def _rows(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceManifestError(f"cannot read JSONL: {path}") from exc
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise SourceManifestError(f"nonempty JSONL objects required: {path}")
    return rows


def _record(path: Path, *, repository_root: Path) -> dict[str, Any]:
    return artifact_record(path.resolve()).to_dict(relative_to=repository_root)


def _require_artifact(record: Any, *, repository_root: Path) -> Path:
    try:
        return verify_artifact_record(record, repository_root=repository_root)
    except ValueError as exc:
        raise SourceManifestError(str(exc)) from exc


def _unique_rows(rows: list[dict[str, Any]], *, location: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row.get("prompt_id")
        if not isinstance(key, str) or not key or key in result:
            raise SourceManifestError(f"invalid or duplicate prompt_id in {location}")
        result[key] = row
    return result


def collapse_prompt_only(rows: list[dict[str, Any]], *, location: Path) -> dict[str, dict[str, Any]]:
    """Collapse the canonical ten seed-suffixed blind rows into base prompts without reading labels."""

    _unique_rows(rows, location=location)
    result: dict[str, dict[str, Any]] = {}
    suffixes: dict[str, set[int]] = {}
    expanded: bool | None = None
    for row in rows:
        if not isinstance(row.get("prompt"), str) or not row["prompt"] or row.get("seed") != 0:
            raise SourceManifestError("selected inputs must be blind prompt-only rows with seed zero")
        prompt_id = row["prompt_id"]
        base_id, marker, suffix = prompt_id.rpartition("_s")
        is_expanded = bool(marker and suffix.isascii() and suffix.isdigit())
        if expanded is not None and expanded != is_expanded:
            raise SourceManifestError("blind population cannot mix base and seed-suffixed prompt IDs")
        expanded = is_expanded
        key = base_id if is_expanded else prompt_id
        if is_expanded:
            seed = int(suffix)
            if seed not in T2I_COMPBENCH_SEEDS or seed in suffixes.setdefault(key, set()):
                raise SourceManifestError("expanded blind rows require unique seed suffixes 42 through 51")
            suffixes[key].add(seed)
        if key in result and result[key]["prompt"] != row["prompt"]:
            raise SourceManifestError("seed-suffixed rows disagree on blind prompt text")
        result.setdefault(key, {"prompt_id": key, "prompt": row["prompt"], "seed": 0})
    if expanded and any(seeds != set(T2I_COMPBENCH_SEEDS) for seeds in suffixes.values()):
        raise SourceManifestError("expanded blind population must cover all ten seed suffixes per prompt")
    return result


def load_source_prompts(  # noqa: C901
    source_manifest: Path,
    prompt_only: Path,
    tokenizer: Any,
    *,
    repository_root: Path | None = None,
) -> tuple[list[SourcePrompt], list[dict[str, Any]]]:
    """Read and verify the Stage 1-3 records of every base prompt selected by ``prompt_only``.

    For each source bundle of the manifest the guidance-target file metadata
    must seal the bundle's inputs, and for each selected prompt the rows must
    share identity and order, Stage 3 recompiled from the AUTO derivation and
    marked dependencies must equal the stored row and trace, and the prompt must
    fit the CLIP window. Returns the prompts in ``prompt_only`` order and the
    sorted artifact records of every file read. Nothing is written.
    """

    root = _root(repository_root)
    require_sd14_tokenizer(tokenizer)
    manifest = _object(source_manifest)
    if set(manifest) != {"schema", "sources"} or manifest["schema"] != SOURCE_MANIFEST_SCHEMA:
        raise SourceManifestError("unsupported semantic source-manifest schema or fields")
    if not isinstance(manifest["sources"], list) or not manifest["sources"]:
        raise SourceManifestError("source manifest must contain a nonempty source list")
    selected = collapse_prompt_only(_rows(prompt_only), location=prompt_only)

    lexical_path = lexical_resource_path()
    artifacts = {
        str(path.resolve()): _record(path, repository_root=root)
        for path in (source_manifest, prompt_only, lexical_path)
    }
    derived: dict[str, SourcePrompt] = {}
    seen_target_files: set[Path] = set()
    for source_index, source in enumerate(manifest["sources"]):
        if not isinstance(source, dict) or set(source) != SOURCE_FIELDS:
            raise SourceManifestError("source bundle fields differ from the closed source contract")
        paths = {key: _path(value, repository_root=root) for key, value in source.items()}
        if paths["pd_sidecar"] in seen_target_files:
            raise SourceManifestError("source manifest contains a duplicate PD sidecar")
        seen_target_files.add(paths["pd_sidecar"])
        pd_meta_path = Path(str(paths["pd_sidecar"]) + ".meta.json")
        pd_meta = _object(pd_meta_path)
        require_target_file_metadata(pd_meta, expected_provider=SD14_GUIDANCE_TARGET_PROVIDER)
        if _require_artifact(pd_meta.get("output"), repository_root=root) != paths["pd_sidecar"]:
            raise SourceManifestError("source PD metadata names a different output")
        declared_sources = pd_meta.get("source_artifacts")
        if not isinstance(declared_sources, list):
            raise SourceManifestError("source PD metadata must seal its input artifacts")
        source_paths = {_require_artifact(record, repository_root=root) for record in declared_sources}
        if not {paths[key] for key in SOURCE_FIELDS - {"pd_sidecar"}}.issubset(source_paths):
            raise SourceManifestError("source manifest differs from the inputs sealed by the PD producer")
        trace_path = _require_artifact(pd_meta.get("conversion_trace"), repository_root=root)
        for path in (*paths.values(), pd_meta_path, trace_path, *source_paths):
            artifacts[str(path)] = _record(path, repository_root=root)
        prompts = _rows(paths["prompt_only"])
        programs = _rows(paths["pd_sidecar"])
        sentences = load_auto_sentences(paths["auto"])
        dependencies = load_marked_dependency_blocks(paths["marked_deps"])
        maps = _rows(paths["sentence_map"])
        trace_rows = _rows(trace_path)
        counts = [len(prompts), len(programs), len(sentences), len(dependencies), len(maps), len(trace_rows)]
        if len(set(counts)) != 1:
            raise SourceManifestError(f"source bundle row counts disagree: {counts}")
        _unique_rows(prompts, location=paths["prompt_only"])
        for row_index, (prompt_row, pd_row, sentence, deps, token_map, trace_row) in enumerate(
            zip(prompts, programs, sentences, dependencies, maps, trace_rows)
        ):
            prompt_id = prompt_row["prompt_id"]
            if prompt_id not in selected:
                continue
            if prompt_id in derived:
                raise SourceManifestError("a selected prompt occurs in more than one source bundle")
            prompt = selected[prompt_id]["prompt"]
            if prompt_row.get("prompt") != prompt or prompt_row.get("seed") != 0:
                raise SourceManifestError("source prompt differs from the selected blind prompt")
            require_target_row_identity(pd_row, expected_provider=SD14_GUIDANCE_TARGET_PROVIDER)
            if pd_row.get("status") != "ok":
                raise SourceManifestError("semantic guidance requires valid PD programs; no generation fallback")
            digest = sha256_text(prompt)
            if any(row.get("prompt_id") != prompt_id for row in (pd_row, token_map, trace_row)):
                raise SourceManifestError("source rows do not share the prompt identity and order")
            if pd_row.get("prompt_sha256") != digest or trace_row.get("prompt_sha256") != digest:
                raise SourceManifestError("source prompt digest mismatch")
            parser_provenance = pd_row["parser_provenance"]
            if parser_provenance.get("auto_id") != sentence.id_line or token_map.get("raw_line_index") != row_index:
                raise SourceManifestError("source AUTO/sentence-map identity mismatch")
            if token_map.get("sentence_index") != 0:
                raise SourceManifestError(
                    f"CCG Diffusion takes a prompt of exactly one sentence; {prompt_id} has more than one")
            source_tokens = tuple(ParseToken(**token) for token in token_map.get("tokens", ()))
            if source_tokens != tuple(tokenize_for_parse(prompt)):
                raise SourceManifestError("source sentence map differs from canonical character tokenization")
            if token_map.get("sentence_char_start") != 0 or token_map.get("sentence_char_end") != len(prompt):
                raise SourceManifestError("source sentence map must cover the complete prompt")
            trace = trace_row.get("conversion_trace")
            if not isinstance(trace, dict):
                raise SourceManifestError("source conversion trace must be an object")
            trace_digest = sha256_text(canonical_json(trace))
            if trace_digest != trace_row.get("conversion_trace_sha256") or (
                trace_digest != parser_provenance.get("conversion_trace_sha256")
            ):
                raise SourceManifestError("source conversion-trace digest mismatch")
            lemma_rows = trace.get("possession_lemma_annotations", ())
            lemmas = {row["source_index"]: row["lemma"] for row in lemma_rows}
            if len(lemmas) != len(lemma_rows):
                raise SourceManifestError("source trace contains duplicate possession-lemma annotations")
            compiled = derive_guidance_targets(
                prompt, sentence, deps, tokenizer, source_tokens=source_tokens,
                possession_lemmas_by_source=lemmas, parse_status=parser_provenance.get("parse_status"),
            )
            program = asdict(compiled.indices)
            program_digest = sha256_text(canonical_json(program))
            if program != pd_row.get("pd_indices") or program_digest != pd_row.get("program_sha256"):
                raise SourceManifestError("recompiled four-field PD program differs from the source")
            if trace_row.get("program_sha256") != program_digest:
                raise SourceManifestError("source trace program digest mismatch")
            if canonical_json(compiled.alignment) != canonical_json(pd_row.get("alignment")):
                raise SourceManifestError("recompiled lexical alignment differs from the source")
            if canonical_json(compiled.trace) != canonical_json(trace):
                raise SourceManifestError("recompiled Stage 3 trace differs from the sealed source")
            token_ids = tokenizer(prompt, add_special_tokens=True, truncation=False)["input_ids"]
            if not isinstance(token_ids, list) or not 3 <= len(token_ids) <= 77:
                raise SourceManifestError("prompt must fit the pinned CLIP lexical window")
            if token_ids[0] != tokenizer.bos_token_id or token_ids[-1] != tokenizer.eos_token_id:
                raise SourceManifestError("prompt token layout differs from CLIP BOS/EOS")
            if tokenizer.eos_token_id in token_ids[1:-1] or tokenizer.bos_token_id in token_ids[1:]:
                raise SourceManifestError("special tokens may not occur in lexical prompt content")
            derived[prompt_id] = SourcePrompt(
                prompt_id=prompt_id,
                prompt=prompt,
                sentence=sentence,
                dependencies=tuple(deps),
                targets=CompiledGuidanceTargets(
                    indices=PDIndices(**pd_row["pd_indices"]), alignment=pd_row["alignment"], trace=trace,
                ),
                pd_row=pd_row,
                source_bundle_index=source_index,
                source_row_index=row_index,
            )
    if set(derived) != set(selected):
        raise SourceManifestError(f"source bundles lack selected prompts: {sorted(set(selected) - set(derived))[:8]}")
    return [derived[key] for key in selected], [artifacts[key] for key in sorted(artifacts)]


# --- seed expansion to image rows ---------------------------------------------------------------------


def expand_to_image_rows(
    source_prompt_rows: Sequence[Mapping[str, Any]],
    source_rows: Sequence[Mapping[str, Any]],
    target_prompt_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], set[str]]:
    """Replicate base-prompt rows to image rows, joining by ``(experiment, sha256(prompt))``.

    Each output row is the source row with ``prompt_id`` set to the image ID and
    ``source_sidecar_prompt_id`` (KEEP key) naming the base prompt. Returns the
    rows in target order and the base prompt IDs used.
    """

    rows_by_id: dict[str, Mapping[str, Any]] = {}
    for row_number, row in enumerate(source_rows, start=1):
        prompt_id = row.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise GuidanceTargetError(
                f"source sidecar row {row_number}: invalid prompt_id"
            )
        if prompt_id in rows_by_id:
            raise GuidanceTargetError(
                f"source sidecar row {row_number}: duplicate prompt_id {prompt_id!r}"
            )
        rows_by_id[prompt_id] = row

    source_ids = {str(row["prompt_id"]) for row in source_prompt_rows}
    if source_ids != set(rows_by_id):
        raise GuidanceTargetError(
            "source prompt/sidecar key mismatch: "
            f"missing={sorted(source_ids - set(rows_by_id))[:10]}, "
            f"extra={sorted(set(rows_by_id) - source_ids)[:10]}"
        )

    source_by_key: dict[tuple[str, str], tuple[str, Mapping[str, Any]]] = {}
    for prompt_row in source_prompt_rows:
        prompt_id = str(prompt_row["prompt_id"])
        digest = sha256_text(str(prompt_row["prompt"]))
        source_row = rows_by_id[prompt_id]
        if source_row.get("prompt_sha256") != digest:
            raise GuidanceTargetError(
                f"source prompt hash mismatch for {prompt_id!r}"
            )
        key = (str(prompt_row["experiment"]), digest)
        if key in source_by_key:
            raise GuidanceTargetError(
                "ambiguous duplicate source prompt text within experiment: "
                f"{key[0]!r}/{digest}"
            )
        source_by_key[key] = (prompt_id, source_row)

    providers = {row.get("provider") for row in source_rows}
    if len(providers) != 1 or not isinstance(next(iter(providers)), str):
        raise GuidanceTargetError(
            f"source sidecar must have one non-empty provider, found {providers}"
        )

    output: list[dict[str, Any]] = []
    used_source_ids: set[str] = set()
    for target_row in target_prompt_rows:
        digest = sha256_text(str(target_row["prompt"]))
        key = (str(target_row["experiment"]), digest)
        source = source_by_key.get(key)
        if source is None:
            raise GuidanceTargetError(
                "target prompt has no source program: "
                f"{target_row['prompt_id']!r}/{digest}"
            )
        source_prompt_id, source_row = source
        expanded = dict(source_row)
        expanded["prompt_id"] = target_row["prompt_id"]
        expanded["source_sidecar_prompt_id"] = source_prompt_id
        output.append(expanded)
        used_source_ids.add(source_prompt_id)
    return output, used_source_ids


def write_image_rows_file(
    output: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    provider: str,
    tokenizer_identity: Mapping[str, Any],
    parser_provenance: Mapping[str, Any],
    target_prompt_manifest: Path,
    used_source_prompt_count: int,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write image rows as compact canonical JSONL plus ``<output>.meta.json``; return the metadata.

    ``parser_provenance`` records where the rows came from (for example the
    Stage 1-3 files or the baseline's extraction settings). Paths are
    repository-relative when inside the checkout.
    """
    if {row.get("provider") for row in rows} != {provider}:
        raise GuidanceTargetError("expanded sidecar provider does not match source metadata")
    record = atomic_write_jsonl(output, rows, overwrite=overwrite)
    root = provenance.repository_root()
    target = artifact_record(target_prompt_manifest).to_dict(relative_to=root)
    metadata = {
        "artifact": "expanded_pd_sidecar",
        "backbone": "sd14",
        "provider": provider,
        "tokenizer": dict(tokenizer_identity),
        "row_count": len(rows),
        "selected_source_prompt_count": used_source_prompt_count,
        "status_counts": dict(sorted(Counter(str(row["status"]) for row in rows).items())),
        "parser_provenance": dict(parser_provenance),
        "target_prompt_manifest": {"path": target["path"], "sha256": target["sha256"]},
        "output": record.to_dict(relative_to=root),
    }
    atomic_write_json(target_file_metadata_path(output), metadata, overwrite=overwrite)
    return metadata


# --- frozen T2I-CompBench++ parse bundle --------------------------------------------------------------

# Recorded Stage 1 AUTO, Stage 2 marked dependencies and coverage rows of the
# eight categories, as used by every evaluated T2I run (gzip level 9, mtime 0).
T2I_PARSE_BUNDLE_DIR = "data/t2i_compbench"
T2I_PARSE_BUNDLE_SCHEMA = "ccg_diffusion.t2i_parse_bundle.v1"
# Bundle file kind -> path inside a Stage 1-3 directory (the recorded layout).
T2I_PARSE_BUNDLE_STAGE_PATHS: Mapping[str, str] = MappingProxyType(
    {
        "auto": "{category}.auto",
        "markedup_deps": "generate/{category}.markedup_deps",
        "coverage_rows": "generate/{category}.coverage_rows.jsonl",
    }
)


def t2i_parse_bundle_stage_path(stage_dir: Path, category: str, kind: str) -> Path:
    """Where a bundle file lives in a Stage 1-3 directory."""
    return stage_dir / T2I_PARSE_BUNDLE_STAGE_PATHS[kind].format(category=category)


def t2i_parse_bundle_file_name(category: str, kind: str) -> str:
    """Bundle-relative name of a compressed file, e.g. ``auto/color.auto.gz``."""
    return f"{kind}/{Path(T2I_PARSE_BUNDLE_STAGE_PATHS[kind].format(category=category)).name}.gz"


def load_t2i_parse_bundle(*, verify: bool = True) -> dict[str, Any]:
    """The bundle manifest and, per category and kind, the path of its compressed file.

    With ``verify`` every compressed file must match the manifest's gz and raw
    (decompressed) sha256.
    """
    root = provenance.repository_root() / T2I_PARSE_BUNDLE_DIR
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != T2I_PARSE_BUNDLE_SCHEMA:
        raise GuidanceTargetError(f"unsupported T2I parse bundle schema: {manifest.get('schema')!r}")
    categories: dict[str, dict[str, Path]] = {}
    for entry in manifest["files"]:
        path = root / entry["path"]
        if verify:
            payload = path.read_bytes()
            if sha256_bytes(payload) != entry["gz_sha256"]:
                raise GuidanceTargetError(f"T2I parse bundle file differs from its manifest: {entry['path']}")
            if sha256_bytes(gzip.decompress(payload)) != entry["raw_sha256"]:
                raise GuidanceTargetError(f"T2I parse bundle content differs from its manifest: {entry['path']}")
        categories.setdefault(entry["category"], {})[entry["kind"]] = path
    return {"root": root, "manifest": manifest, "categories": categories}


def unpack_t2i_parse_bundle(
    stage_dir: Path, *, categories: Sequence[str] = T2I_COMPBENCH_CATEGORIES,
) -> dict[str, dict[str, Path]]:
    """Write the recorded Stage 1-2 files of ``categories`` into ``stage_dir`` (byte-identical).

    Writes ``<stage>/<cat>.auto``, ``<stage>/generate/<cat>.markedup_deps`` and
    ``<stage>/generate/<cat>.coverage_rows.jsonl``. An existing file is kept when
    its bytes are equal and refused otherwise.
    """
    bundle = load_t2i_parse_bundle(verify=True)
    written: dict[str, dict[str, Path]] = {}
    for category in categories:
        for kind, gz_path in bundle["categories"][category].items():
            payload = gzip.decompress(gz_path.read_bytes())
            target = t2i_parse_bundle_stage_path(stage_dir, category, kind)
            if target.exists():
                if target.read_bytes() != payload:
                    raise GuidanceTargetError(f"refusing to overwrite a different file: {target}")
            else:
                atomic_write_bytes(target, payload, overwrite=False)
            written.setdefault(category, {})[kind] = target
    return written
