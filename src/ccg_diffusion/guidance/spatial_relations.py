"""Closed-set 2D relation phrases read from the CCG derivation (input of the Spatial Relation Loss).

The closed relation set is the surface inventory of the T2I-CompBench++
``spatial`` templates and of the official 2D-spatial metric:

    on the left of | on the right of | on the top of | on the bottom of
    next to        | near            | on side of

Extraction is derivation-driven:

1. Find the attachment functor leaf whose category is ``(A\\A)/B`` with
   ``A`` a nominal (``N`` / ``NP``).  Its slot-1 dependency argument is the
   modified nominal head — the *subject* entity ``X``.
2. Follow its slot-2 argument.  ``near`` takes the object entity directly;
   ``next`` takes a ``PP/NP`` preposition (``to``) whose slot-1 argument is the
   object entity; ``on`` takes a relational nominal (``left`` / ``right`` /
   ``top`` / ``bottom`` / ``side``) which an ``of`` functor further modifies,
   and that ``of`` functor's slot-2 argument is the object entity ``Y``.
3. The surface span from the attachment functor up to the start of ``Y``'s NP
   must be one of the seven closed-set phrases.

The error texts of ``extract_relation_triples_from_derivation`` are recorded in
guidance decisions (``relation_extraction_note``) and must not change.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, Optional, Sequence

from ccg_diffusion.parsing.ccg import (
    AutoSentence,
    CCGCategory,
    MarkedDependency,
    is_atomic_nominal,
    parse_ccg_category,
)

RelationLabel = Literal["left", "right", "top", "bottom", "next_to", "near", "side_of"]

# The closed relation set.  Keys are the surface phrases that occupy the span
# between the two entity NPs in the official ``spatial`` templates.
RELATION_SURFACE_TO_LABEL: Mapping[str, RelationLabel] = {
    "on the left of": "left",
    "on the right of": "right",
    "on the top of": "top",
    "on the bottom of": "bottom",
    "next to": "next_to",
    "near": "near",
    "on side of": "side_of",
}
RELATION_LABELS: tuple[RelationLabel, ...] = (
    "left",
    "right",
    "top",
    "bottom",
    "next_to",
    "near",
    "side_of",
)
# Labels whose metric branch tests a signed centre offset on one axis.
DIRECTIONAL_LABELS: frozenset[str] = frozenset({"left", "right", "top", "bottom"})
# Labels whose metric branch tests centre proximity on either axis.
PROXIMITY_LABELS: frozenset[str] = frozenset({"next_to", "near", "side_of"})

_DETERMINERS: frozenset[str] = frozenset({"a", "an", "the"})


class RelationExtractionError(ValueError):
    """A prompt does not expose a closed-set spatial relation triple."""


@dataclass(frozen=True)
class RelationTriple:
    """One ``(subject, label, object)`` relation over source-token positions.

    ``subject_source_index`` / ``object_source_index`` are 1-based positions in
    the parsed sentence, the same index space the marked dependencies and the
    guidance-target ``alignment`` rows use (``source_index``).
    ``extraction`` is always ``"derivation"`` (recorded value).
    """

    subject_source_index: int
    object_source_index: int
    label: RelationLabel
    subject_word: str
    object_word: str
    relation_surface: str
    relation_source_indices: tuple[int, ...]
    extraction: str
    # The atomic nominal *inside* the relation expression ("left" in "on the
    # left of"); ``None`` for the relations whose surface carries no nominal.
    relational_nominal_source_index: Optional[int] = None
    relational_nominal_word: Optional[str] = None


@dataclass(frozen=True)
class RelationConstraint:
    """A relation triple projected onto CLIP token columns for the loss.

    ``subject_clip_index`` / ``object_clip_index`` index the full 77-column
    CLIP token axis, like ``PDIndices`` entries, so the Spatial Relation Loss
    reads the same aggregated attention tensor as the Predicated Diffusion loss.
    """

    label: RelationLabel
    subject_clip_index: int
    object_clip_index: int

    def __post_init__(self) -> None:
        if self.label not in RELATION_LABELS:
            raise ValueError(f"unknown relation label: {self.label!r}")
        for name in ("subject_clip_index", "object_clip_index"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative int, got {value!r}")
        if self.subject_clip_index == self.object_clip_index:
            raise ValueError("relation endpoints must be distinct CLIP columns")


def is_nominal_attachment_functor(category: CCGCategory) -> bool:
    """Return whether ``category`` is ``(A\\A)/B`` with ``A`` a nominal atom."""

    if category.slash != "/" or category.result is None:
        return False
    modifier = category.result
    if modifier.slash != "\\" or modifier.result is None or modifier.argument is None:
        return False
    left, right = modifier.result, modifier.argument
    if not (left.is_atomic and right.is_atomic):
        return False
    return left.base_atom == right.base_atom and left.base_atom in {"N", "NP"}


def _dependency_argument(
    dependencies: Sequence[MarkedDependency],
    head_index: int,
    slot: int,
) -> Optional[int]:
    """Return the unique argument position filling ``slot`` of ``head_index``."""

    matches = {
        dep.argument_index
        for dep in dependencies
        if dep.head_index == head_index and dep.slot == slot
    }
    if len(matches) != 1:
        return None
    return next(iter(matches))


def _np_start_index(leaves: Mapping[int, Any], entity_index: int) -> int:
    """Return where the entity's NP starts, i.e. its determiner when present."""

    previous = leaves.get(entity_index - 1)
    if previous is None:
        return entity_index
    if previous.word.lower() in _DETERMINERS:
        return entity_index - 1
    return entity_index


def _surface_span(
    leaves: Mapping[int, Any],
    start_index: int,
    stop_index: int,
) -> tuple[str, tuple[int, ...]]:
    """Join leaf words over ``[start_index, stop_index)`` in source order."""

    indices = tuple(range(start_index, stop_index))
    words = [leaves[index].word.lower() for index in indices]
    return " ".join(words), indices


def _is_entity_leaf(leaf: Any) -> bool:
    return is_atomic_nominal(parse_ccg_category(leaf.category))


def extract_relation_triples_from_derivation(
    sentence: AutoSentence,
    dependencies: Sequence[MarkedDependency],
) -> tuple[RelationTriple, ...]:
    """Return every closed-set relation triple encoded by one parsed sentence.

    Raises :class:`RelationExtractionError` when no attachment functor yields a
    closed-set relation between two atomic nominal entities. Results retain
    source order and exact duplicates are removed.
    """

    leaves = {leaf.source_index: leaf for leaf in sentence.leaves}
    reasons: list[str] = []
    triples: list[RelationTriple] = []
    seen: set[tuple[int, int, RelationLabel, tuple[int, ...]]] = set()
    for leaf in sentence.leaves:
        try:
            category = parse_ccg_category(leaf.category)
        except ValueError:
            continue
        if not is_nominal_attachment_functor(category):
            continue
        subject_index = _dependency_argument(dependencies, leaf.source_index, 1)
        complement_index = _dependency_argument(dependencies, leaf.source_index, 2)
        if subject_index is None or complement_index is None:
            reasons.append(f"functor {leaf.word}_{leaf.source_index} lacks both slots")
            continue

        object_index, nominal_index = _resolve_object_endpoint(
            leaves, dependencies, leaf, complement_index
        )
        if object_index is None:
            reasons.append(
                f"functor {leaf.word}_{leaf.source_index} has no object endpoint"
            )
            continue
        if subject_index not in leaves or object_index not in leaves:
            continue
        if not (
            _is_entity_leaf(leaves[subject_index])
            and _is_entity_leaf(leaves[object_index])
        ):
            reasons.append(
                f"endpoints {subject_index}/{object_index} are not atomic nominals"
            )
            continue
        if not subject_index < leaf.source_index < object_index:
            reasons.append("relation span is not between the two entities")
            continue

        surface, indices = _surface_span(
            leaves, leaf.source_index, _np_start_index(leaves, object_index)
        )
        label = RELATION_SURFACE_TO_LABEL.get(surface)
        if label is None:
            reasons.append(f"surface {surface!r} is outside the closed relation set")
            continue
        triple = RelationTriple(
            subject_source_index=subject_index,
            object_source_index=object_index,
            label=label,
            subject_word=leaves[subject_index].word,
            object_word=leaves[object_index].word,
            relation_surface=surface,
            relation_source_indices=indices,
            extraction="derivation",
            relational_nominal_source_index=nominal_index,
            relational_nominal_word=(
                None if nominal_index is None else leaves[nominal_index].word
            ),
        )

        identity = (
            triple.subject_source_index,
            triple.object_source_index,
            triple.label,
            triple.relation_source_indices,
        )
        if identity not in seen:
            triples.append(triple)
            seen.add(identity)

    if not triples:
        detail = "; ".join(reasons) if reasons else "no nominal attachment functor"
        raise RelationExtractionError(f"no closed-set relation triple: {detail}")
    return tuple(triples)


def _resolve_object_endpoint(
    leaves: Mapping[int, Any],
    dependencies: Sequence[MarkedDependency],
    functor_leaf: Any,
    complement_index: int,
) -> tuple[Optional[int], Optional[int]]:
    """Walk the functor's slot-2 complement down to the object entity.

    Returns ``(object_source_index, relational_nominal_source_index)``.

    ``near a Y``          -> the complement already is ``Y``; no nominal.
    ``next to a Y``       -> the complement is ``to`` (``PP/NP``); take its
    slot 1; no nominal.
    ``on the left of a Y``-> the complement is an atomic nominal the ``of``
    functor takes as its slot-1 argument; the object is that ``of`` functor's
    slot-2 argument and the complement is the relational nominal.
    """

    complement = leaves.get(complement_index)
    if complement is None:
        return None, None

    complement_category = parse_ccg_category(complement.category)
    if (
        complement_category.slash == "/"
        and complement_category.result is not None
        and complement_category.result.is_atomic
        and complement_category.result.base_atom == "PP"
    ):
        return _dependency_argument(dependencies, complement_index, 1), None

    if not _is_entity_leaf(complement):
        return None, None

    # The complement is an atomic nominal.  If a further ``(A\A)/B`` functor
    # takes it as its slot-1 argument, the complement heads the "the <nominal>
    # of" chain and the real object entity is that functor's slot-2 argument.
    for dependency in dependencies:
        if dependency.slot != 1 or dependency.argument_index != complement_index:
            continue
        head_leaf = leaves.get(dependency.head_index)
        if head_leaf is None or dependency.head_index <= complement_index:
            continue
        try:
            head_category = parse_ccg_category(head_leaf.category)
        except ValueError:
            continue
        if not is_nominal_attachment_functor(head_category):
            continue
        object_index = _dependency_argument(dependencies, dependency.head_index, 2)
        if object_index is not None:
            return object_index, complement_index

    return complement_index, None


def clip_index_by_source_index(alignment: Iterable[Mapping[str, Any]]) -> dict[int, int]:
    """Map authoritative ``source_index`` to ``clip_index`` fail-closed."""

    mapping: dict[int, int] = {}
    for row_index, entry in enumerate(alignment):
        source_index = entry.get("source_index")
        prompt_word_index = entry.get("prompt_word_index")
        clip_index = entry.get("clip_index")
        if (
            isinstance(source_index, bool)
            or not isinstance(source_index, int)
            or source_index < 1
            or isinstance(prompt_word_index, bool)
            or not isinstance(prompt_word_index, int)
            or prompt_word_index < 1
            or isinstance(clip_index, bool)
            or not isinstance(clip_index, int)
            or clip_index < 0
        ):
            raise RelationExtractionError(
                f"alignment row {row_index} has invalid source/prompt/CLIP indices"
            )
        if prompt_word_index != source_index:
            raise RelationExtractionError(
                f"alignment row {row_index} source_index/prompt_word_index mismatch: "
                f"{source_index}/{prompt_word_index}"
            )
        previous = mapping.get(source_index)
        if previous is not None and previous != clip_index:
            raise RelationExtractionError(
                f"alignment source_index {source_index} maps to multiple CLIP indices"
            )
        mapping[source_index] = clip_index
    return mapping


def project_relation_triple(
    triple: RelationTriple,
    alignment: Iterable[Mapping[str, Any]],
) -> RelationConstraint:
    """Project a triple's entity positions onto CLIP token columns."""

    mapping = clip_index_by_source_index(alignment)
    try:
        subject_clip = mapping[triple.subject_source_index]
        object_clip = mapping[triple.object_source_index]
    except KeyError as exc:
        raise RelationExtractionError(
            f"relation endpoint {exc.args[0]} has no CLIP alignment row"
        ) from exc
    return RelationConstraint(
        label=triple.label,
        subject_clip_index=subject_clip,
        object_clip_index=object_clip,
    )
