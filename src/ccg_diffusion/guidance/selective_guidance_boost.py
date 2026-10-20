"""Selective Guidance Boost (paper Sec. IV-B1): one guidance route per image from the derivation.

Each image gets one route, recorded under its historical labels:

* ``B0`` (default): guidance strength (s, delta) = (20, 0.02) on T2I-CompBench++
  (TAB PP uses (10, 0.02) through ``pp_attachment.objective.TabGuidanceStrength``);
* ``P1``: default strength plus the Spatial Relation Loss on the
  derivation-verified 2D relations;
* ``P2`` (boosted): (80, 0.06), i.e. scale multiplier 4, with whole-word
  pooling of the correspondence/leak endpoints when the kernel policy is
  ``routing_whole_word_v1``.

The decision has two stages. ``derive_base_decision`` reads the derivation:
incomplete parses, unresolved modifiers, count words, depth words ("behind",
"in front of", ...), action verbs and uncovered attachments keep ``B0``;
verified closed-set 2D relations give ``P1``; visual modifier bindings give
``P2``. ``apply_selective_guidance_boost`` then applies the ``attribute_v2``
eligibility rule of the paper configurations: count-eligible images and
syntax/count/depth exclusions keep ``B0``, ``P1`` is retained, and any
verified color/shape/size/material/surface modifier boosts the image to ``P2``.
``derive_image_guidance`` runs both stages and derives the image's count
guidance (Count Loss) from the same inputs.

``base_policy_definition()`` and ``boost_policy_definition()`` are frozen
records whose sha256 values are written into every guidance record. They are
kept byte-identical, including text that names strategies that are not
implemented here (only ``attribute_v2`` is).
"""
from __future__ import annotations

import enum
import json
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from ccg_diffusion.guidance.count_loss import (
    CountGuidance,
    NounForms,
    extract_count_constraints,
    load_noun_forms,
)
from ccg_diffusion.guidance.predicated_diffusion_loss import WholeWordSpans, whole_word_spans
from ccg_diffusion.guidance.spatial_relation_loss import SPATIAL_RELATION_LOSS_PARAMETERS
from ccg_diffusion.guidance.spatial_relations import (
    RelationConstraint,
    RelationExtractionError,
    extract_relation_triples_from_derivation,
    is_nominal_attachment_functor,
)
from ccg_diffusion.parsing.ccg import AutoSentence, CCGCategory, MarkedDependency, parse_ccg_category
from ccg_diffusion.provenance import canonical_json, sha256_text
from ccg_diffusion.targets.lexicon import (
    NominalModifierLexicon,
    lexical_resource_definition,
    load_nominal_modifier_lexicon,
)
from ccg_diffusion.targets.pd_indices import PDIndices


class GuidanceRoute(enum.StrEnum):
    """Per-image guidance routes; the values are the recorded labels."""

    DEFAULT = "B0"
    SPATIAL_RELATION = "P1"
    BOOSTED = "P2"


BASE_POLICY_VERSION = "holsd.sd14_semantic_routing.v1"
LEXICON_VERSION = "holsd.generic_visual_adjectives.v1"
BOOST_POLICY_VERSION = "holsd.sd14_routing_comparison.v1"
ROUTING_STRATEGY = "attribute_v2"
GUIDANCE_POLICIES = ("routing_v1", "routing_whole_word_v1")
WHOLE_WORD_POOLING_SCOPE = "mean_raw_lexical_attention_for_corr_leak_endpoints_on_P2_rows_only"
BASE_SCALE = 20.0
BOOST_SCALE_MULTIPLIER = 4.0
DEFAULT_UPDATE_CAP = 0.02
BOOSTED_UPDATE_CAP = 0.06

VISUAL_ADJECTIVES: Mapping[str, frozenset[str]] = MappingProxyType({
    "color": frozenset(
        "black white gray grey red orange yellow green blue purple pink brown "
        "violet indigo cyan magenta turquoise teal maroon crimson scarlet "
        "burgundy lavender lilac mauve beige tan cream ivory gold golden "
        "silver silvery bronze copper coral peach salmon navy azure cerulean "
        "ochre olive amber aquamarine chartreuse vermilion".split()
    ),
    "shape": frozenset(
        "round circular square rectangular triangular oval elliptical spherical "
        "cubic cubical cylindrical conical pyramidal polygonal pentagonal "
        "hexagonal heptagonal octagonal nonagonal decagonal trapezoidal "
        "rhomboid rhomboidal toroidal ovoid hemispherical semicircular "
        "diamond-shaped heart-shaped star-shaped crescent-shaped egg-shaped "
        "bell-shaped pear-shaped teardrop-shaped cone-shaped cylinder-shaped "
        "cube-shaped disk-shaped disc-shaped doughnut-shaped donut-shaped "
        "ring-shaped leaf-shaped flower-shaped cross-shaped spiral-shaped".split()
    ),
    # Derivational material adjectives qualify; bare compound nouns do not.
    "material_adjective": frozenset(
        "wooden metallic glassy rubbery papery waxy stony rocky sandy earthen silken cottony".split()
    ),
    "surface": frozenset(
        "smooth rough bumpy ridged ribbed grooved wrinkled crumpled creased "
        "fluffy furry fuzzy hairy woolly woolen silky velvety leathery grainy "
        "gritty coarse porous pitted cracked flaky scaly pebbled speckled "
        "striped spotted dotted checked checkered plaid woven braided knitted "
        "crocheted lacy embroidered corrugated polished glossy shiny matte "
        "reflective frosted translucent transparent opaque iridescent "
        "sparkling glittery glittering mottled marbled dimpled tufted quilted".split()
    ),
})
# Words added by the attribute_v2 eligibility rule.
ATTRIBUTE_ADDITIONS: Mapping[str, frozenset[str]] = MappingProxyType({
    "size_and_shape": frozenset(
        "big small large little tiny huge short tall long wide narrow thick thin "
        "broad slender flat curved straight pointed elongated oblong "
        "diamond heart star crescent teardrop pear ring bell cone cylinder cube sphere pyramid".split()
    ),
    "material": frozenset(
        "wood metal glass plastic rubber leather fabric paper stone ceramic porcelain "
        "clay cotton wool silk velvet linen denim satin canvas steel iron aluminum aluminium "
        "brass copper bronze silver gold concrete marble granite bamboo wicker rattan foam "
        "cork acrylic nylon polyester suede fur".split()
    ),
})
COUNT_WORDS = frozenset(
    "zero one two three four five six seven eight nine ten eleven twelve "
    "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty "
    "thirty forty fifty sixty seventy eighty ninety hundred thousand million "
    "dozen pair couple several many few multiple both".split()
)
DEPTH_WORDS = frozenset({"behind", "foreground", "background"})
DEPTH_SEQUENCES = (("in", "front", "of"), ("in", "back", "of"))
COPULA_FORMS = frozenset("be am is are was were been being".split())
AUXILIARY_FORMS = frozenset(
    "have has had do does did can could may might must shall should will would".split()
)
_ROUTES = frozenset(route.value for route in GuidanceRoute)


def base_policy_definition() -> dict[str, Any]:
    """Frozen record of the base routing policy (its sha256 is recorded)."""

    return {
        "schema": BASE_POLICY_VERSION,
        "lexicon": {
            "schema": LEXICON_VERSION,
            "source": "manually_declared_generic_visual_adjectives_not_harvested_from_benchmark",
            "entries": {group: sorted(words) for group, words in VISUAL_ADJECTIVES.items()},
            "matching": "exact_lowercase_source_token_and_HolCCG_corr_modifier_role",
        },
        "neutral_compound_lexicon": lexical_resource_definition(),
        "eligibility_amendment": "input_census_informed_before_generation_neutral_noun_only_context",
        "material_noun_scope": "material_nouns_alone_do_not_establish_a_visual_binding",
        "P2_dose_scope": "entire_original_four_field_PD_loss_including_existence_and_neutral_compound_bindings",
        "count_words": sorted(COUNT_WORDS),
        "count_digit_rule": "any_digit_in_any_source_token",
        "depth_words": sorted(DEPTH_WORDS),
        "depth_sequences": [list(words) for words in DEPTH_SEQUENCES],
        "copula_forms": sorted(COPULA_FORMS),
        "auxiliary_forms": sorted(AUXILIARY_FORMS),
        "action_rule": "non_adjectival_S_backslash_NP_result_spine_except_copula_or_clausal_auxiliary",
        "priority": [
            "partial_parse_or_unresolved_modifier_or_count_or_depth_or_action_or_uncovered_attachment: B0",
            "one_or_more_derivation_verified_closed_2D_relations: P1",
            "at_least_one_visual_corr_modifier_and_all_others_visual_or_noun_only_compound_modifiers: P2",
            "otherwise: B0",
        ],
        "profiles": {
            "B0": {"scale_factor": 20.0, "max_update_ratio": 0.02, "relation_weight": 0.0},
            "P1": {"scale_factor": 20.0, "max_update_ratio": 0.02, **dict(SPATIAL_RELATION_LOSS_PARAMETERS)},
            "P2": {"scale_factor": 80.0, "scale_multiplier": 4.0, "max_update_ratio": 0.06,
                   "relation_weight": 0.0},
        },
        "pooling_scope": WHOLE_WORD_POOLING_SCOPE,
        "pooling_unchanged_terms": ["exist", "possession"],
        "default_or_B0_or_P1_pooling": "off",
        "generation_failure_fallback": "forbidden",
    }


def base_policy_sha256() -> str:
    """Content-address the base policy independently of surrounding implementation."""

    return sha256_text(canonical_json(base_policy_definition()))


def boost_policy_definition() -> dict[str, Any]:
    """Frozen record of the boost policy (its sha256 is recorded); only attribute_v2 is implemented."""
    return {
        "schema": BOOST_POLICY_VERSION,
        "base_policy_sha256": base_policy_sha256(),
        "strategies": ["p2_all_except_count", "current", "attribute_v2"],
        "attribute_additions": {key: sorted(words) for key, words in ATTRIBUTE_ADDITIONS.items()},
        "attribute_evidence": "exact source token with verified corr_modifier and structural modifier role",
        "current": "unchanged v1 decision",
        "p2_all_except_count": "P2 for every prompt without eligible COUNT constraints; P1 disabled",
        "attribute_v2_priority": [
            "eligible COUNT constraints: B0",
            "retain v1 P1 routes",
            "incomplete parse, unresolved binding, count cue or depth cue: B0",
            "one or more verified visual/size/shape/material corr modifiers: P2",
            "otherwise: B0",
        ],
        "attribute_v2_relaxed_vetoes": [
            "an unrelated unknown corr modifier no longer vetoes a verified visual binding",
            "action or an uncovered attachment no longer vetoes a verified visual binding",
        ],
        "P2_objective": "entire original PD loss; no per-binding dose or new loss term",
        "whole_word": "existing raw lexical attention mean at corr/leak endpoints on P2 rows only",
        "COUNT": "same v3 constraints and numerics, enabled for all conditions, eligible rows retain B0",
        "DEPTH": False,
        "ACTION": False,
        "source_of_hypothesis": "post-hoc diagnosis of prior full results, tested on the fixed exploratory subset",
    }


def boost_policy_sha256() -> str:
    return sha256_text(canonical_json(boost_policy_definition()))


def _verb_body(category: CCGCategory) -> bool:
    return bool(
        category.slash == "\\"
        and category.result is not None
        and category.result.is_atomic
        and category.result.base_atom == "S"
        and category.result.feature != "adj"
        and category.argument is not None
        and category.argument.is_atomic
        and category.argument.base_atom == "NP"
    )


def _action_category(category: CCGCategory, word: str) -> bool:
    if word in COPULA_FORMS:
        return False
    current = category
    while not current.is_atomic:
        if (
            current.result is not None
            and current.argument is not None
            and current.result == current.argument
        ):
            return False
        if (
            word in AUXILIARY_FORMS
            and current.argument is not None
            and _verb_body(current.argument)
        ):
            return False
        if _verb_body(current):
            return True
        if current.result is None:
            return False
        current = current.result
    return False


def _contains_sequence(words: Sequence[str], sequence: Sequence[str]) -> bool:
    return any(tuple(words[index:index + len(sequence)]) == tuple(sequence)
               for index in range(len(words) - len(sequence) + 1))


@dataclass(frozen=True)
class GuidanceDecision:
    """One input-derived route and the complete evidence used to choose it."""

    profile: str
    reasons: tuple[str, ...]
    evidence_json: str

    def __post_init__(self) -> None:
        if self.profile not in _ROUTES or not self.reasons:
            raise ValueError("guidance decision requires a closed profile and explicit reason")
        evidence = json.loads(self.evidence_json)
        if not isinstance(evidence, dict) or canonical_json(evidence) != self.evidence_json:
            raise ValueError("guidance evidence must be a canonical JSON object")

    def as_record(self) -> dict[str, Any]:
        return {"profile": self.profile, "reasons": list(self.reasons),
                "evidence": json.loads(self.evidence_json)}


def derive_base_decision(
    *,
    sentence: AutoSentence,
    dependencies: Sequence[MarkedDependency],
    alignment: Sequence[Mapping[str, Any]],
    indices: PDIndices,
    parse_status: str,
    unresolved_modifiers: Sequence[Any],
    compound_lexicon: NominalModifierLexicon | None = None,
) -> tuple[GuidanceDecision, tuple[RelationConstraint, ...]]:
    """Choose the base route from the derivation, the marked dependencies and the guidance targets.

    Unsupported semantics give an explicit ``B0`` decision, never a parser
    failure or a stock-generation fallback.
    """

    source = {entry["source_index"]: entry for entry in alignment}
    if set(source) != {leaf.source_index for leaf in sentence.leaves}:
        raise ValueError("routing alignment must cover the exact AUTO source indices")
    words = [leaf.word.lower() for leaf in sentence.leaves]
    count_tokens = [word for word in words if word in COUNT_WORDS or any(char.isdigit() for char in word)]
    depth_cues = [word for word in words if word in DEPTH_WORDS]
    depth_cues.extend(" ".join(sequence) for sequence in DEPTH_SEQUENCES if _contains_sequence(words, sequence))
    action_tokens = [
        leaf.word for leaf in sentence.leaves
        if _action_category(parse_ccg_category(leaf.category), leaf.word.lower())
    ]
    attachment_sources = {
        leaf.source_index for leaf in sentence.leaves
        if is_nominal_attachment_functor(parse_ccg_category(leaf.category))
    }
    relation_error: str | None = None
    try:
        triples = extract_relation_triples_from_derivation(sentence, dependencies)
    except RelationExtractionError as exc:
        triples = ()
        relation_error = str(exc)
    covered_sources = {index for triple in triples for index in triple.relation_source_indices}
    uncovered_attachments = sorted(attachment_sources - covered_sources)
    relations: list[RelationConstraint] = []
    for triple in triples:
        if triple.extraction != "derivation":
            raise ValueError("routing rejects a surface-fallback relation")
        subject = source[triple.subject_source_index]
        object_ = source[triple.object_source_index]
        if (
            subject["text"] != triple.subject_word
            or object_["text"] != triple.object_word
            or subject["clip_index"] not in indices.exist
            or object_["clip_index"] not in indices.exist
        ):
            raise ValueError("typed relation endpoints must match the verified PD entities")
        relations.append(RelationConstraint(
            label=triple.label,
            subject_clip_index=subject["clip_index"],
            object_clip_index=object_["clip_index"],
        ))

    modifier_anchors = {pair[0] for pair in indices.corr}
    modifiers = [
        entry for entry in alignment
        if entry.get("clip_index") in modifier_anchors and "corr_modifier" in entry.get("pd_roles", ())
    ]
    represented_anchors = {entry["clip_index"] for entry in modifiers}
    if represented_anchors != modifier_anchors:
        raise ValueError("corr modifier anchors are absent from the verified source alignment")
    lexical_index = compound_lexicon or load_nominal_modifier_lexicon()
    visual: list[dict[str, Any]] = []
    neutral_compounds: list[dict[str, Any]] = []
    unknown: list[str] = []
    for entry in modifiers:
        groups = [group for group, entries in VISUAL_ADJECTIVES.items() if entry["text"].lower() in entries]
        if entry["structural_role"] == "modifier" and groups:
            visual.append({"source_index": entry["source_index"], "word": entry["text"],
                           "clip_index": entry["clip_index"], "groups": groups})
        elif entry["structural_role"] == "modifier" and lexical_index.is_noun_only(entry["text"]):
            neutral_compounds.append({"source_index": entry["source_index"], "word": entry["text"],
                                      "clip_index": entry["clip_index"], "rule": "noun_entry_no_adjective_entry"})
        else:
            unknown.append(entry["text"])

    vetoes: list[str] = []
    if parse_status != "complete":
        vetoes.append("incomplete_parse")
    if unresolved_modifiers:
        vetoes.append("unresolved_modifier_binding")
    if count_tokens:
        vetoes.append("count_semantics_not_yet_supported")
    if depth_cues:
        vetoes.append("depth_semantics_not_yet_supported")
    if action_tokens:
        vetoes.append("action_semantics_not_yet_supported")
    if uncovered_attachments:
        vetoes.append("unsupported_or_mixed_relation_structure")
    if vetoes:
        profile, reasons = GuidanceRoute.DEFAULT, tuple(vetoes)
    elif relations:
        profile, reasons = GuidanceRoute.SPATIAL_RELATION, ("verified_2D_relation_at_baseline_dose",)
    elif visual and not unknown and indices.corr:
        profile, reasons = GuidanceRoute.BOOSTED, ("visual_binding_with_only_visual_or_noun_only_compound_modifiers",)
    else:
        profile = GuidanceRoute.DEFAULT
        reasons = ("unknown_or_nonvisual_corr_modifier",) if unknown else ("no_supported_guidance_evidence",)
    evidence = {
        "parse_status": parse_status,
        "unresolved_modifiers": list(unresolved_modifiers),
        "count_tokens": count_tokens,
        "depth_cues": depth_cues,
        "action_tokens": action_tokens,
        "uncovered_attachment_source_indices": uncovered_attachments,
        "visual_modifiers": visual,
        "neutral_noun_compound_modifiers": neutral_compounds,
        "unknown_or_nonvisual_modifiers": unknown,
        "relations": [asdict(triple) for triple in triples],
        "relation_extraction_note": relation_error,
    }
    return GuidanceDecision(profile, reasons, canonical_json(evidence)), tuple(relations)


@dataclass(frozen=True)
class ImageGuidance:
    """The route of one image, passed to the guided denoising kernel.

    ``policy`` is the kernel policy: ``routing_whole_word_v1`` pools whole words
    on ``P2`` rows, ``routing_v1`` never pools.
    """

    policy: str
    decision: GuidanceDecision
    source_prompt_sha256: str
    source_program_sha256: str
    pair_spans: WholeWordSpans
    derived_relations: tuple[RelationConstraint, ...]

    def __post_init__(self) -> None:
        if self.policy not in GUIDANCE_POLICIES:
            raise ValueError("unsupported SD1.4 guidance policy")
        if not isinstance(self.pair_spans, WholeWordSpans):
            raise ValueError("guidance requires validated pair spans")
        if not isinstance(self.derived_relations, tuple) or any(
            not isinstance(relation, RelationConstraint) for relation in self.derived_relations
        ):
            raise ValueError("derived relations must be immutable typed constraints")
        for value in (self.source_prompt_sha256, self.source_program_sha256):
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError("guidance source identities must be SHA-256 digests")
        if (self.decision.profile == GuidanceRoute.SPATIAL_RELATION) and not self.derived_relations:
            raise ValueError("P1 requires derivation-verified relations")

    @property
    def effective_profile(self) -> str:
        return self.decision.profile

    @property
    def scale_multiplier(self) -> float:
        return 4.0 if self.effective_profile == GuidanceRoute.BOOSTED else 1.0

    @property
    def max_update_ratio(self) -> float:
        return 0.06 if self.effective_profile == GuidanceRoute.BOOSTED else 0.02

    @property
    def relations(self) -> tuple[RelationConstraint, ...]:
        return self.derived_relations if self.effective_profile == GuidanceRoute.SPATIAL_RELATION else ()

    @property
    def active_pair_spans(self) -> WholeWordSpans | None:
        if self.policy == "routing_whole_word_v1" and self.effective_profile == GuidanceRoute.BOOSTED:
            return self.pair_spans
        return None

    def as_record(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "policy_sha256": base_policy_sha256(),
            "decision": self.decision.as_record(),
            "effective_profile": self.effective_profile,
            "source_prompt_sha256": self.source_prompt_sha256,
            "source_program_sha256": self.source_program_sha256,
            "scale_factor": 20.0 * self.scale_multiplier,
            "scale_multiplier": self.scale_multiplier,
            "max_update_ratio": self.max_update_ratio,
            "relations": [asdict(relation) for relation in self.relations],
            "relation_parameters": dict(SPATIAL_RELATION_LOSS_PARAMETERS) if self.relations else None,
            "pooling_scope": WHOLE_WORD_POOLING_SCOPE,
            "pooling_applied": self.active_pair_spans is not None,
            "pair_token_spans": self.active_pair_spans.as_record() if self.active_pair_spans is not None else None,
        }


@dataclass(frozen=True)
class SelectiveBoostGuidance(ImageGuidance):
    """The route after the Selective Guidance Boost rule; records the boost policy."""

    whole_word: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if type(self.whole_word) is not bool:
            raise ValueError("untyped whole-word factor")
        expected = "routing_whole_word_v1" if self.whole_word else "routing_v1"
        if self.policy != expected:
            raise ValueError("kernel pooling preset differs from the declared whole-word factor")

    def as_record(self) -> dict[str, Any]:
        result = super().as_record()
        result.update({
            "policy": BOOST_POLICY_VERSION,
            "kernel_policy": self.policy,
            "policy_sha256": boost_policy_sha256(),
            "routing_strategy": ROUTING_STRATEGY,
            "whole_word": self.whole_word,
        })
        return result


def apply_selective_guidance_boost(
    base: ImageGuidance,
    alignment: Sequence[Mapping[str, Any]],
    *,
    whole_word: bool,
    count_eligible: bool,
) -> SelectiveBoostGuidance:
    """Apply the ``attribute_v2`` eligibility rule to a base route; the caller authenticates alignment."""
    if type(count_eligible) is not bool:
        raise ValueError("COUNT eligibility must be an explicit boolean")
    old = base.decision
    evidence = json.loads(old.evidence_json)
    vocabulary = set().union(*VISUAL_ADJECTIVES.values(), *ATTRIBUTE_ADDITIONS.values())
    visual = [
        {"source_index": row["source_index"], "word": row["text"], "clip_index": row["clip_index"]}
        for row in alignment
        if row.get("structural_role") == "modifier"
        and "corr_modifier" in row.get("pd_roles", ())
        and row["text"].lower() in vocabulary
    ]
    if count_eligible:
        if old.profile != GuidanceRoute.DEFAULT:
            raise ValueError("COUNT source no longer shares its required original B0 route")
        profile, reasons = GuidanceRoute.DEFAULT, ("retain_COUNT_B0_scale_and_cap",)
    elif old.profile == GuidanceRoute.SPATIAL_RELATION:
        profile, reasons = old.profile, old.reasons
    elif (evidence.get("parse_status") != "complete" or evidence.get("unresolved_modifiers")
          or evidence.get("count_tokens") or evidence.get("depth_cues")):
        profile, reasons = GuidanceRoute.DEFAULT, ("retain_syntax_count_and_depth_exclusions",)
    elif visual:
        profile = GuidanceRoute.BOOSTED
        reasons = ("verified_visual_binding_despite_unrelated_modifier_action_or_attachment",)
    else:
        profile, reasons = GuidanceRoute.DEFAULT, ("no_verified_visual_binding",)
    decision = GuidanceDecision(profile, reasons, canonical_json({
        "original_decision": old.as_record(),
        "attribute_v2_visual_modifiers": visual,
        "count_eligible": count_eligible,
    }))
    return SelectiveBoostGuidance(
        policy="routing_whole_word_v1" if whole_word else "routing_v1",
        decision=decision,
        source_prompt_sha256=base.source_prompt_sha256,
        source_program_sha256=base.source_program_sha256,
        pair_spans=base.pair_spans,
        derived_relations=base.derived_relations,
        whole_word=whole_word,
    )


def derive_image_guidance(
    prompt: str,
    sentence: AutoSentence,
    dependencies: Sequence[MarkedDependency],
    alignment: Sequence[Mapping[str, Any]],
    indices: PDIndices,
    *,
    parse_status: str,
    unresolved_modifiers: Sequence[Any],
    tokenizer: Any,
    noun_forms: NounForms | None = None,
    compound_lexicon: NominalModifierLexicon | None = None,
) -> tuple[SelectiveBoostGuidance, CountGuidance]:
    """The route (with whole-word pooling) and the count guidance of one prompt.

    ``tokenizer`` is the pinned SD1.4 CLIP tokenizer; ``alignment`` and
    ``indices`` are the prompt's guidance targets on its CLIP token axis.
    """
    program = asdict(indices)
    prompt_hash, program_hash = sha256_text(prompt), sha256_text(canonical_json(program))
    token_count = len(tokenizer(prompt, truncation=False)["input_ids"])
    decision, relations = derive_base_decision(
        sentence=sentence, dependencies=dependencies, alignment=alignment, indices=indices,
        parse_status=parse_status, unresolved_modifiers=unresolved_modifiers, compound_lexicon=compound_lexicon)
    base = ImageGuidance(
        "routing_v1", decision, prompt_hash, program_hash,
        whole_word_spans(alignment, indices, text_token_count=token_count), relations)
    forms = noun_forms if noun_forms is not None else load_noun_forms()
    count_decision = extract_count_constraints(
        sentence, dependencies, alignment, indices, parse_status=parse_status,
        unresolved_modifiers=unresolved_modifiers, text_token_count=token_count, noun_forms=forms)
    count = CountGuidance(
        True, count_decision, prompt_hash, program_hash, sha256_text(canonical_json(alignment)))
    guidance = apply_selective_guidance_boost(base, list(alignment), whole_word=True,
                                              count_eligible=bool(count_decision.constraints))
    return guidance, count
