"""Record-only. Its records are hashed into `PrefixLayoutPlan.source_plan_sha256`;
it never changes which prompt receives the Positional Prefix (arguments (a) and (b) below).

This module reproduces the historical depth and clause layout screening of the
development repository. Its two plan records (``DepthLayoutPlan``,
``ClauseLayoutPlan``) are part of every evaluated Positional Prefix plan: the
plan digest ``source_plan_sha256`` covers them, and depth- or clause-active
prompts record a depth/clause-branch prefix plan. They are not a paper
component and never change an executed operation:

* (a) Depth is inert. ``screen_depth`` is active only when the route is
  ``B0`` and the guidance targets have no correspondence, leak or possession
  pair. The Positional Prefix needs two correspondence groups, so
  ``build_phrase_plan`` returns ``fewer_than_two_visual_nominals`` for every
  depth-active prompt; conversely every prompt whose prefix is applied leaves
  ``screen_depth`` before any extraction or error.
* (b) Clause is inert. An applied prefix requires the word "and" between
  the two phrases, and ``screen_clause`` returns ``unsupported_scope`` for any
  prompt that contains "and". A clause-active plan only changes prefix plans
  with a relational phrase reason, which are never applied.

Only ``positional_prefix.derive_positional_prefix`` calls this module.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from typing import Any, Literal, Mapping, Sequence

from ccg_diffusion.guidance.count_loss import NounForms
from ccg_diffusion.guidance.selective_guidance_boost import COPULA_FORMS, GuidanceRoute
from ccg_diffusion.guidance.spatial_relations import is_nominal_attachment_functor
from ccg_diffusion.parsing.ccg import AutoSentence, CCGCategory, MarkedDependency, parse_ccg_category
from ccg_diffusion.provenance import canonical_json, sha256_text
from ccg_diffusion.targets.pd_indices import PDIndices

DEPTH_PLAN_VERSION = "holsd.sd14_depth_layout.v1"
CLAUSE_PLAN_VERSION = "holsd.sd14_clause_layout.v1"
TOKEN_AXIS_SIZE = 77
NEGATION_OR_COORDINATION = frozenset(("not", "n't", "never", "no", "neither", "nor", "without", "and", "or"))
DIRECT = {"above": "top", "over": "top", "atop": "top", "below": "bottom", "under": "bottom",
          "underneath": "bottom", "near": "near", "beside": "next_to"}
REGIONS = {"left": "left", "right": "right", "top": "top", "bottom": "bottom", "side": "side_of"}
INTRANSITIVE = frozenset("hang hangs hanging hung rest rests resting lie lies lying stand stands standing "
                         "sit sits sitting placed positioned located perch perches perched".split())


# --- typed depth records -----------------------------------------------------------------------------


def _positive_integer(value: Any) -> bool:
    return type(value) is int and value > 0


def validate_depth_dependency_source(dependency: MarkedDependency) -> None:
    """Reject Python bool/float aliases before they can name source coordinates."""
    if not isinstance(dependency, MarkedDependency) or any(
        not _positive_integer(value) for value in
        (dependency.head_index, dependency.argument_index, dependency.slot)
    ):
        raise ValueError("DEPTH dependency coordinates and slots must be strict positive integers")
    if any(not isinstance(value, str) or not value.strip()
           for value in (dependency.head_word, dependency.argument_word, dependency.marked_category)):
        raise ValueError("DEPTH dependency words and category must be nonempty strings")
    if type(dependency.rule_id) is not int or (
        dependency.long_range_category is not None
        and (not isinstance(dependency.long_range_category, str) or not dependency.long_range_category.strip())
    ):
        raise ValueError("DEPTH dependency rule and optional long-range category have invalid types")


def _record_fields(record: Mapping[str, Any], cls: type[Any]) -> None:
    if not isinstance(record, Mapping) or set(record) != {field.name for field in fields(cls)}:
        raise ValueError(f"serialized {cls.__name__} fields differ from its typed contract")


@dataclass(frozen=True)
class DepthEndpoint:
    """One named source noun and its entire verified lexical CLIP span."""

    role: Literal["subject", "object"]
    source_index: int
    word: str
    category: str
    lemma_candidates: tuple[str, ...]
    clip_anchor: int
    lexical_piece_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.role not in {"subject", "object"} or not _positive_integer(self.source_index):
            raise ValueError("DEPTH endpoint needs an explicit role and positive source position")
        if not isinstance(self.word, str) or not self.word or not isinstance(self.category, str):
            raise ValueError("DEPTH endpoint needs its lexical source word and category")
        category = parse_ccg_category(self.category)
        if not category.is_atomic or category.base_atom != "N":
            raise ValueError("DEPTH endpoint must be an atomic lexical noun")
        if (not isinstance(self.lemma_candidates, tuple) or not self.lemma_candidates
                or any(not isinstance(value, str) or not value for value in self.lemma_candidates)
                or tuple(sorted(set(self.lemma_candidates))) != self.lemma_candidates):
            raise ValueError("noun lemma candidates must be immutable, nonempty and canonical")
        span = self.lexical_piece_indices
        if (not isinstance(span, tuple) or not span or any(not _positive_integer(value) for value in span)
                or tuple(sorted(set(span))) != span or not _positive_integer(self.clip_anchor)
                or self.clip_anchor not in span):
            raise ValueError("DEPTH endpoint needs an ordered full lexical span containing its anchor")

    def as_record(self) -> dict[str, Any]:
        return {**asdict(self), "lemma_candidates": list(self.lemma_candidates),
                "lexical_piece_indices": list(self.lexical_piece_indices)}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> DepthEndpoint:
        _record_fields(record, cls)
        if not isinstance(record["lemma_candidates"], list) or not isinstance(record["lexical_piece_indices"], list):
            raise ValueError("serialized DEPTH noun lists must retain their order")
        return cls(**{**record, "lemma_candidates": tuple(record["lemma_candidates"]),
                      "lexical_piece_indices": tuple(record["lexical_piece_indices"])})


@dataclass(frozen=True)
class DepthConstraint:
    """Exactly one typed nominal relation; larger inverse depth means nearer."""

    relation: Literal["behind", "in_front_of"]
    inverse_depth_sign: int
    cue_source_index: int
    endpoints: tuple[DepthEndpoint, DepthEndpoint]
    dependency_evidence: tuple[MarkedDependency, ...]
    text_token_count: int

    def __post_init__(self) -> None:
        signs = {"behind": -1, "in_front_of": 1}
        if (self.relation not in signs or type(self.inverse_depth_sign) is not int
                or self.inverse_depth_sign != signs[self.relation] or not _positive_integer(self.cue_source_index)):
            raise ValueError("DEPTH relation sign or cue position differs from inverse-depth semantics")
        if (not isinstance(self.endpoints, tuple) or len(self.endpoints) != 2
                or any(not isinstance(value, DepthEndpoint) for value in self.endpoints)
                or tuple(value.role for value in self.endpoints) != ("subject", "object")):
            raise ValueError("DEPTH requires an ordered immutable subject/object pair")
        subject, obj = self.endpoints
        if len({subject.source_index, obj.source_index, self.cue_source_index}) != 3:
            raise ValueError("DEPTH cue and endpoint source positions must be distinct")
        if set(subject.lemma_candidates).intersection(obj.lemma_candidates):
            raise ValueError("DEPTH repeated noun referents must abstain")
        if type(self.text_token_count) is not int or not 3 <= self.text_token_count <= TOKEN_AXIS_SIZE:
            raise ValueError("DEPTH text layout must include valid CLIP BOS/lexical/EOS positions")
        pieces = (*subject.lexical_piece_indices, *obj.lexical_piece_indices)
        if len(pieces) != len(set(pieces)) or any(index >= self.text_token_count - 1 for index in pieces):
            raise ValueError("DEPTH noun spans may not overlap or include EOS/padding")
        evidence = self.dependency_evidence
        if (not isinstance(evidence, tuple) or len(evidence) != (2 if self.relation == "behind" else 4)
                or any(not isinstance(dep, MarkedDependency) for dep in evidence)
                or len(set(evidence)) != len(evidence)):
            raise ValueError("DEPTH must retain the complete unique typed dependency evidence")
        for dependency in evidence:
            validate_depth_dependency_source(dependency)
        if any(type(dep.slot) is not int or dep.slot not in {1, 2} or dep.long_range_category is not None
               for dep in evidence):
            raise ValueError("DEPTH evidence supports only unambiguous local slots one and two")

    def as_record(self) -> dict[str, Any]:
        return {**asdict(self), "endpoints": [value.as_record() for value in self.endpoints],
                "dependency_evidence": [asdict(value) for value in self.dependency_evidence]}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> DepthConstraint:
        _record_fields(record, cls)
        if not isinstance(record["endpoints"], list) or not isinstance(record["dependency_evidence"], list):
            raise ValueError("serialized DEPTH endpoints/evidence require ordered lists")
        dependencies = []
        for value in record["dependency_evidence"]:
            _record_fields(value, MarkedDependency)
            dependencies.append(MarkedDependency(**value))
        return cls(**{**record, "endpoints": tuple(DepthEndpoint.from_record(value) for value in record["endpoints"]),
                      "dependency_evidence": tuple(dependencies)})


@dataclass(frozen=True)
class DepthDecision:
    """Depth relation screening result: eligible, inactive or abstain."""

    status: Literal["eligible", "inactive", "abstain"]
    reasons: tuple[str, ...]
    constraints: tuple[DepthConstraint, ...]

    def __post_init__(self) -> None:
        if self.status not in {"eligible", "inactive", "abstain"}:
            raise ValueError("unsupported DEPTH source status")
        if (not isinstance(self.reasons, tuple) or not self.reasons
                or any(not isinstance(reason, str) or not reason for reason in self.reasons)):
            raise ValueError("DEPTH reasons must be immutable nonempty strings")
        if (not isinstance(self.constraints, tuple)
                or any(not isinstance(value, DepthConstraint) for value in self.constraints)
                or len(self.constraints) != int(self.status == "eligible")):
            raise ValueError("only an eligible DEPTH decision may contain exactly one relation")

    def as_record(self) -> dict[str, Any]:
        return {"status": self.status, "reasons": list(self.reasons),
                "constraints": [value.as_record() for value in self.constraints]}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> DepthDecision:
        _record_fields(record, cls)
        if not isinstance(record["reasons"], list) or not isinstance(record["constraints"], list):
            raise ValueError("serialized DEPTH decisions require ordered lists")
        return cls(record["status"], tuple(record["reasons"]),
                   tuple(DepthConstraint.from_record(value) for value in record["constraints"]))


def category_matches_source(marked: str, source: str) -> bool:
    """Keep slash/argument structure exact; unify C&C X and unspecified features.

    C&C's marked inventory uses S[X] where an AUTO leaf may use S, S[ng]
    or S[pss]. Repeated X occurrences in one inventory category must bind
    to the same explicit source feature. Explicit conflicting features fail.
    """
    bindings: dict[str, str] = {}

    def match(left: CCGCategory, right: CCGCategory) -> bool:
        if left.is_atomic != right.is_atomic or left.slash != right.slash:
            return False
        if left.is_atomic:
            if left.base_atom != right.base_atom:
                return False
            a, b = left.feature, right.feature
            if a == "X" and b not in {None, "X"}:
                key = left.base_atom
                if key in bindings and bindings[key] != b:
                    return False
                bindings[key] = b
                return True
            return a in {None, "X"} or b in {None, "X"} or a == b
        return match(left.result, right.result) and match(left.argument, right.argument)

    return match(parse_ccg_category(marked), parse_ccg_category(source))


# --- depth relation extraction -----------------------------------------------------------------------


def _decision(status: str, reason: str, constraints: list[dict[str, Any]] | None = None) -> DepthDecision:
    return DepthDecision.from_record({"status": status, "reasons": [reason], "constraints": constraints or []})


def _nominal_preposition(category: CCGCategory) -> bool:
    modifier = category.result
    argument = category.argument
    if category.slash != "/" or modifier is None or argument is None:
        return False
    if not argument.is_atomic or argument.base_atom not in {"N", "NP"}:
        return False
    if modifier.slash != "\\" or modifier.result is None or modifier.argument is None:
        return False
    left, right = modifier.result, modifier.argument
    return left.is_atomic and right.is_atomic and left.base_atom == right.base_atom and left.base_atom in {"N", "NP"}


def extract_depth(
    sentence: AutoSentence,
    dependencies: Sequence[MarkedDependency],
    alignment: Sequence[Mapping[str, Any]],
    indices: PDIndices,
    *,
    parse_status: str,
    unresolved_modifiers: Sequence[Any],
    text_token_count: int,
    noun_forms: NounForms,
) -> DepthDecision:
    """Only typed parse/PD inputs enter this rule; no ID, category, gold, score or raw prompt."""
    for leaf in sentence.leaves:
        if type(leaf.source_index) is not int or leaf.source_index <= 0 or any(
            not isinstance(value, str) or not value.strip() for value in (leaf.word, leaf.category)
        ):
            raise ValueError("DEPTH AUTO source coordinates, words and categories have invalid types")
    for entry in alignment:
        if (not isinstance(entry, Mapping) or type(entry.get("source_index")) is not int
                or entry["source_index"] <= 0 or not isinstance(entry.get("text"), str) or not entry["text"].strip()):
            raise ValueError("DEPTH alignment source coordinates and words have invalid types")
    for dependency in dependencies:
        validate_depth_dependency_source(dependency)
    leaves = {leaf.source_index: leaf for leaf in sentence.leaves}
    aligned = {entry["source_index"]: entry for entry in alignment}
    if len(leaves) != len(sentence.leaves) or len(aligned) != len(alignment) or set(leaves) != set(aligned):
        raise ValueError("source alignment is not one-to-one and complete")
    for index, leaf in leaves.items():
        if aligned[index]["text"] != leaf.word:
            raise ValueError("alignment source word differs from AUTO")
    for dep in dependencies:
        if dep.head_index not in leaves or dep.argument_index not in leaves:
            raise ValueError("dependency references an absent source token")
        if dep.head_word != leaves[dep.head_index].word or dep.argument_word != leaves[dep.argument_index].word:
            raise ValueError("dependency token identity differs from AUTO")
    cues = [leaf for leaf in leaves.values() if leaf.word.lower() in {"behind", "front"}]
    if not cues:
        return _decision("inactive", "no_supported_depth_cue")
    if parse_status != "complete":
        return _decision("abstain", "incomplete_parse")
    if unresolved_modifiers:
        return _decision("abstain", "unresolved_modifier")
    if any(leaf.word.lower() in NEGATION_OR_COORDINATION for leaf in leaves.values()):
        return _decision("abstain", "unsupported_negation_or_coordination_scope")
    if len(cues) != 1:
        return _decision("abstain", "multiple_depth_cues")
    cue = cues[0]

    def preposition_edges(head: int) -> tuple[MarkedDependency, MarkedDependency] | None:
        edges = [dep for dep in dependencies if dep.head_index == head]
        if len(edges) != 2 or sorted(dep.slot for dep in edges) != [1, 2]:
            return None
        category = parse_ccg_category(leaves[head].category)
        if not _nominal_preposition(category):
            return None
        if any(
            parse_ccg_category(dep.marked_category) != category or dep.long_range_category is not None for dep in edges
        ):
            return None
        return tuple(sorted(edges, key=lambda dep: dep.slot))

    if cue.word.lower() == "behind":
        edges = preposition_edges(cue.source_index)
        if edges is None:
            return _decision("abstain", "missing_ambiguous_or_unsupported_behind_binding")
        subject, obj = edges[0].argument_index, edges[1].argument_index
        label, sign = "behind", -1
        evidence = list(edges)
    else:
        front_category = parse_ccg_category(cue.category)
        if not front_category.is_atomic or front_category.base_atom != "N":
            return _decision("abstain", "front_is_not_a_relational_nominal")
        in_edges = [preposition_edges(leaf.source_index) for leaf in leaves.values() if leaf.word.lower() == "in"]
        in_edges = [edges for edges in in_edges if edges is not None and edges[1].argument_index == cue.source_index]
        of_edges = [preposition_edges(leaf.source_index) for leaf in leaves.values() if leaf.word.lower() == "of"]
        of_edges = [edges for edges in of_edges if edges is not None and edges[0].argument_index == cue.source_index]
        if len(in_edges) != 1 or len(of_edges) != 1:
            return _decision("abstain", "missing_ambiguous_or_unsupported_in_front_of_chain")
        left, right = in_edges[0], of_edges[0]
        if not left[0].head_index < cue.source_index < right[0].head_index:
            return _decision("abstain", "in_front_of_source_order_mismatch")
        subject, obj = left[0].argument_index, right[1].argument_index
        label, sign = "in_front_of", 1
        evidence = [*left, *right]
    if subject == obj or cue.source_index in {subject, obj}:
        return _decision("abstain", "identical_or_cue_endpoint")
    endpoint_leaves = [leaves[subject], leaves[obj]]
    if any(
        not (parse_ccg_category(leaf.category).is_atomic and parse_ccg_category(leaf.category).base_atom == "N")
        for leaf in endpoint_leaves
    ):
        return _decision("abstain", "endpoint_is_not_an_atomic_lexical_noun")
    noun_leaves = [
        leaf
        for leaf in leaves.values()
        if parse_ccg_category(leaf.category).is_atomic and parse_ccg_category(leaf.category).base_atom == "N"
    ]
    forms = {leaf.source_index: noun_forms.candidates(leaf.word) for leaf in noun_leaves}
    if any(
        forms[endpoint.source_index].intersection(forms[other.source_index])
        for endpoint in endpoint_leaves
        for other in noun_leaves
        if other.source_index != endpoint.source_index
    ):
        return _decision("abstain", "repeated_or_morphologically_ambiguous_noun_referent")
    endpoints = []
    seen_pieces: set[int] = set()
    for role, leaf in zip(("subject", "object"), endpoint_leaves):
        row = aligned[leaf.source_index]
        span = tuple(row["clip_lexical_piece_indices"])
        anchor = row["clip_index"]
        if anchor not in indices.exist:
            return _decision("abstain", "endpoint_missing_verified_PD_existence_anchor")
        if (
            not span
            or tuple(sorted(set(span))) != span
            or anchor not in span
            or any(type(index) is not int or not 1 <= index < text_token_count - 1 for index in span)
        ):
            raise ValueError("noun lexical span differs from the actual CLIP text window")
        if seen_pieces.intersection(span):
            raise ValueError("different depth endpoints have overlapping lexical token spans")
        seen_pieces.update(span)
        endpoints.append({
            "role": role,
            "source_index": leaf.source_index,
            "word": leaf.word,
            "category": leaf.category,
            "lemma_candidates": sorted(forms[leaf.source_index]),
            "clip_anchor": anchor,
            "lexical_piece_indices": list(span),
        })
    return _decision(
        "eligible",
        "one_unique_nominal_depth_relation",
        [{
            "relation": label,
            "inverse_depth_sign": sign,
            "cue_source_index": cue.source_index,
            "endpoints": endpoints,
            "dependency_evidence": [asdict(dep) for dep in evidence],
            "text_token_count": text_token_count,
        }],
    )


# --- depth layout plan -------------------------------------------------------------------------------


@dataclass(frozen=True)
class DepthLayoutPlan:
    prompt_sha256: str
    program_sha256: str
    conditioning: str | None
    front_anchor: int | None
    back_anchor: int | None
    front_pieces: tuple[int, ...]
    back_pieces: tuple[int, ...]
    phantom_anchor: int | None
    relation: str | None
    reason: str
    source_evidence_json: str

    @property
    def active(self) -> bool:
        return self.conditioning is not None

    def as_record(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("source_evidence_json")
        value.update({"schema": DEPTH_PLAN_VERSION, "front_pieces": list(self.front_pieces),
                      "back_pieces": list(self.back_pieces),
                      "source_evidence": json.loads(self.source_evidence_json)})
        return value


def screen_depth(
    prompt: str, sentence: AutoSentence, dependencies: Sequence[MarkedDependency],
    alignment: Sequence[Mapping[str, Any]], program: Mapping[str, Any],
    guidance: Mapping[str, Any], count: Mapping[str, Any], tokenizer: Any, noun_forms: NounForms,
) -> DepthLayoutPlan:
    identity = (sha256_text(prompt), sha256_text(canonical_json(program)))

    def inactive(reason: str) -> DepthLayoutPlan:
        return DepthLayoutPlan(*identity, None, None, None, (), (), None, None, reason, "{}")

    if guidance["source_prompt_sha256"] != identity[0] or guidance["source_program_sha256"] != identity[1]:
        raise ValueError("depth conditioning source differs from the incumbent")
    evidence = guidance["decision"]["evidence"]["original_decision"]["evidence"]
    if (count["decision"]["constraints"] or evidence["count_tokens"]
            or guidance["effective_profile"] != GuidanceRoute.DEFAULT
            or evidence["parse_status"] != "complete" or evidence["unresolved_modifiers"]):
        return inactive("retain_COUNT_non_B0_or_unresolved_source")
    if any(program[field] for field in ("corr", "leak", "possession")):
        return inactive("retain_modified_or_possession_input")
    leaves = [leaf for leaf in sentence.leaves if leaf.word not in {".", "!"}]
    words = [leaf.word.lower() for leaf in leaves]
    by_source = {r["source_index"]: r for r in alignment}
    by_leaf = {leaf.source_index: leaf for leaf in sentence.leaves}
    if any(word in {"not", "no", "and", "or", "neither", "without"} for word in words):
        return inactive("unsupported_scope")
    token_ids = tokenizer(prompt, truncation=False)["input_ids"]
    # Also validate the no-cue passive case: source coordinates and dependency
    # token identities must agree before any special-case binding is accepted.
    derived = extract_depth(sentence, dependencies, alignment, PDIndices(**program),
                            parse_status="complete", unresolved_modifiers=(), text_token_count=len(token_ids),
                            noun_forms=noun_forms)
    phantom = None
    if "hidden" in words:
        # A noun hidden by a noun: use passive-subject and by-argument dependency roles.
        if len(leaves) != 6 or words[0] not in {"a", "an", "the"} or words[4] not in {"a", "an", "the"}:
            return inactive("not_simple_hidden_by")
        if words[2:4] != ["hidden", "by"] or leaves[2].category != "S[pss]\\NP":
            return inactive("unsupported_passive_depth_predicate")
        if any(parse_ccg_category(leaves[i].category).base_atom != "N" for i in (1, 5)):
            return inactive("unsupported_depth_noun")
        back_source, front_source = leaves[1].source_index, leaves[5].source_index
        hidden, by = leaves[2].source_index, leaves[3].source_index
        required = {(hidden, 1, back_source), (by, 1, hidden), (by, 2, front_source)}
        edges = [d for d in dependencies if d.head_index in {hidden, by}]
        if ({(d.head_index, d.slot, d.argument_index) for d in edges} != required
                or len(edges) != 3 or any(d.long_range_category is not None for d in edges)
                or any(not category_matches_source(d.marked_category, by_leaf[d.head_index].category)
                       for d in edges)):
            return inactive("unresolved_hidden_by_arguments")
        relation = "hidden_by"
        trace = [asdict(d) for d in edges]
    else:
        if not (len(leaves) == 5 and words[2] == "behind" or
                len(leaves) == 7 and words[2:5] == ["in", "front", "of"]):
            return inactive("not_simple_supported_depth_relation")
        if not derived.constraints:
            return inactive("unsupported_or_ambiguous_depth_binding")
        constraint = derived.constraints[0]
        subject, obj = constraint.endpoints
        if constraint.inverse_depth_sign == 1:
            front_source, back_source = subject.source_index, obj.source_index
            phantom = by_source[constraint.cue_source_index]["clip_index"]
        else:
            front_source, back_source = obj.source_index, subject.source_index
        relation = constraint.relation
        trace = constraint.as_record()
    front, back = by_source[front_source], by_source[back_source]
    if noun_forms.candidates(front["text"]).intersection(noun_forms.candidates(back["text"])):
        return inactive("repeated_or_morphologically_ambiguous_referents")
    anchors = [front["clip_index"], back["clip_index"]]
    expected_exist = set(anchors) | ({phantom} if phantom is not None else set())
    if set(program["exist"]) != expected_exist or any(a is None for a in anchors):
        return inactive("additional_or_unprojectable_noun")
    seen_pieces: set[int] = set()
    for endpoint in (front, back):
        pieces = endpoint["clip_lexical_piece_indices"]
        if (not pieces or pieces != sorted(set(pieces)) or endpoint["clip_index"] not in pieces
                or any(type(i) is not int or not 1 <= i < len(token_ids) - 1 for i in pieces)
                or seen_pieces.intersection(pieces)):
            raise ValueError("depth endpoint lexical span differs from the CLIP text window")
        seen_pieces.update(pieces)
    prefix = prompt.rstrip() + (" " if prompt.rstrip().endswith(".") else ". ")
    conditioning = (f"{prefix}The {front['text']} is in the foreground, close to the camera. "
                    f"The {back['text']} is in the background, partially occluded by the {front['text']}.")
    new_ids = tokenizer(conditioning, truncation=False)["input_ids"]
    if len(new_ids) > 77:
        return inactive("conditioning_would_truncate")
    if new_ids[:len(token_ids) - 1] != token_ids[:-1]:
        raise ValueError("depth conditioning changed an original PD token")
    return DepthLayoutPlan(*identity, conditioning, *anchors, tuple(front["clip_lexical_piece_indices"]),
                           tuple(back["clip_lexical_piece_indices"]), phantom, relation,
                           "simple_source_bound_depth_relation", canonical_json(trace))


# --- clause layout plan ------------------------------------------------------------------------------


@dataclass(frozen=True)
class ClauseLayoutPlan:
    prompt_sha256: str
    program_sha256: str
    relation: str | None
    subject_anchor: int | None
    object_anchor: int | None
    subject_pieces: tuple[int, ...]
    object_pieces: tuple[int, ...]
    phantom_anchor: int | None
    relation_weight: float
    reason: str
    source_evidence_json: str

    @property
    def active(self) -> bool:
        return self.relation is not None

    def as_record(self) -> dict[str, Any]:
        record = asdict(self)
        record.pop("source_evidence_json")
        return {**record, "schema": CLAUSE_PLAN_VERSION, "subject_pieces": list(self.subject_pieces),
                "object_pieces": list(self.object_pieces), "source_evidence": json.loads(self.source_evidence_json)}


def screen_clause(
    prompt: str, sentence: AutoSentence, dependencies: Sequence[MarkedDependency],
    alignment: Sequence[Mapping[str, Any]], program: Mapping[str, Any],
    guidance: Mapping[str, Any], count: Mapping[str, Any], tokenizer: Any, noun_forms: NounForms,
) -> ClauseLayoutPlan:
    identity = (sha256_text(prompt), sha256_text(canonical_json(program)))

    def inactive(reason: str) -> ClauseLayoutPlan:
        return ClauseLayoutPlan(*identity, None, None, None, (), (), None, 0.0, reason, "{}")

    if guidance["source_prompt_sha256"] != identity[0] or guidance["source_program_sha256"] != identity[1]:
        raise ValueError("clause source differs from incumbent guidance")
    evidence = guidance["decision"]["evidence"]["original_decision"]["evidence"]
    if (count["decision"]["constraints"] or evidence["count_tokens"]
            or guidance["effective_profile"] == GuidanceRoute.SPATIAL_RELATION
            or evidence["parse_status"] != "complete" or evidence["unresolved_modifiers"] or evidence["depth_cues"]):
        return inactive("retain_COUNT_P1_or_unresolved_depth_source")
    if program["possession"]:
        return inactive("retain_possession_source")
    leaves = {leaf.source_index: leaf for leaf in sentence.leaves}
    words = {i: leaf.word.lower() for i, leaf in leaves.items()}
    if set(words.values()).intersection({"not", "no", "and", "or", "neither", "without"}):
        return inactive("unsupported_scope")
    token_count = len(tokenizer(prompt, truncation=False)["input_ids"])
    # The shared typed extractor authenticates AUTO/alignment/dependency identities
    # even when there is no depth cue. Its depth decision is not used here.
    extract_depth(sentence, dependencies, alignment, PDIndices(**program), parse_status="complete",
                  unresolved_modifiers=(), text_token_count=token_count, noun_forms=noun_forms)
    aligned = {r["source_index"]: r for r in alignment}
    cats = {i: parse_ccg_category(leaf.category) for i, leaf in leaves.items()}
    if any(not category_matches_source(d.marked_category, leaves[d.head_index].category) for d in dependencies):
        return inactive("unsupported_dependency_category")

    def argument(head: int, slot: int) -> int | None:
        matches = [d.argument_index for d in dependencies if d.head_index == head and d.slot == slot]
        return matches[0] if len(matches) == 1 else None

    for dependency in dependencies:
        if dependency.long_range_category is None:
            continue
        # C&C explicitly marks the inherited subject of 'was hung' / 'was next'.
        # Accept only a unique direct copular auxiliary with that same subject;
        # arbitrary long-range or coordinated control remains unsupported.
        parents = [d.head_index for d in dependencies
                   if d.slot == 2 and d.argument_index == dependency.head_index
                   and words[d.head_index] in COPULA_FORMS
                   and argument(d.head_index, 1) == dependency.argument_index
                   and category_matches_source(dependency.long_range_category, leaves[d.head_index].category)]
        if dependency.slot != 1 or len(parents) != 1:
            return inactive("unsupported_long_range_subject")

    def noun(index: int | None) -> bool:
        return index in cats and cats[index].is_atomic and cats[index].base_atom == "N"

    def pp(index: int | None) -> bool:
        return index in cats and leaves[index].category == "PP/NP"

    # Candidates carry actual source indices; no surface template supplies a noun.
    candidates = []
    for cue, word in words.items():
        subject, obj, phantom, verb = None, None, None, None
        label = DIRECT.get(word)
        if word == "next" and leaves[cue].category == "(S[adj]\\NP)/PP":
            to = argument(cue, 2)
            if pp(to) and words[to] == "to":
                subject, obj, label = argument(cue, 1), argument(to, 1), "next_to"
        elif label or word == "on":
            if is_nominal_attachment_functor(cats[cue]):
                subject, obj = argument(cue, 1), argument(cue, 2)
            elif pp(cue):
                parents = [d.head_index for d in dependencies if d.argument_index == cue and d.slot == 2
                           and words[d.head_index] in COPULA_FORMS
                           and leaves[d.head_index].category == "(S[dcl]\\NP)/PP"]
                if len(parents) == 1:
                    subject, obj = argument(parents[0], 1), argument(cue, 1)
            elif leaves[cue].category == "((S\\NP)\\(S\\NP))/NP":
                verb = argument(cue, 1)
                if verb in words and words[verb] in INTRANSITIVE:
                    subject, obj = argument(verb, 1), argument(cue, 2)
            if word == "on" and noun(obj) and words[obj] in REGIONS:
                phantom = obj
                of = [i for i in words if words[i] == "of" and is_nominal_attachment_functor(cats[i])
                      and argument(i, 1) == phantom]
                if len(of) == 1:
                    obj, label = argument(of[0], 2), REGIONS[words[phantom]]
        if label and noun(subject) and noun(obj) and subject != obj:
            candidates.append((subject, obj, label, phantom))
    if len(candidates) != 1:
        return inactive("no_unique_supported_clause_relation")
    subject, obj, label, phantom = candidates[0]
    if noun_forms.candidates(words[subject]).intersection(noun_forms.candidates(words[obj])):
        return inactive("repeated_or_ambiguous_nouns")
    endpoints = [aligned[subject], aligned[obj]]
    anchors = [r["clip_index"] for r in endpoints]
    phantom_anchor = aligned[phantom]["clip_index"] if phantom is not None else None
    expected = set(anchors) | ({phantom_anchor} if phantom is not None else set())
    if (None in expected or len(expected) != len(anchors) + int(phantom is not None)
            or set(program["exist"]) != expected):
        return inactive("additional_or_unprojectable_entity")
    modified = {entity for modifier, entity in program["corr"]}
    if not set(anchors).issubset(modified) or phantom_anchor in modified:
        return inactive("two_modified_entities_required")
    used = set()
    for row in endpoints:
        pieces = row["clip_lexical_piece_indices"]
        if (not pieces or pieces != sorted(set(pieces)) or row["clip_index"] != pieces[-1]
                or any(type(i) is not int or not 1 <= i < token_count - 1 for i in pieces)
                or used.intersection(pieces)):
            raise ValueError("invalid clause endpoint lexical span")
        used.update(pieces)
    trace = {"subject_source_index": subject, "object_source_index": obj, "phantom_source_index": phantom,
             "dependencies": [asdict(d) for d in dependencies]}
    return ClauseLayoutPlan(*identity, label, *anchors, tuple(endpoints[0]["clip_lexical_piece_indices"]),
                            tuple(endpoints[1]["clip_lexical_piece_indices"]), phantom_anchor,
                            1.5 if guidance["effective_profile"] == GuidanceRoute.BOOSTED else 6.0,
                            "two_modified_nouns_with_source_bound_relation", canonical_json(trace))


def screen_layouts(
    prompt: str, sentence: AutoSentence, dependencies: Sequence[MarkedDependency],
    alignment: Sequence[Mapping[str, Any]], program: Mapping[str, Any],
    guidance: Mapping[str, Any], count: Mapping[str, Any], tokenizer: Any, noun_forms: NounForms,
) -> tuple[DepthLayoutPlan, ClauseLayoutPlan]:
    """The depth and clause plan records of one prompt (they never both apply)."""
    arguments = (prompt, sentence, dependencies, alignment, program, guidance, count, tokenizer, noun_forms)
    depth = screen_depth(*arguments)
    clause = screen_clause(*arguments)
    if depth.active and clause.active:
        raise ValueError("depth and clause rules unexpectedly overlap")
    return depth, clause
