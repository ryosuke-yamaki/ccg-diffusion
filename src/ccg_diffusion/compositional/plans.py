"""Per-prompt CCG Diffusion component plan for T2I-CompBench++ (paper Sec. IV-B).

For one prompt with its CCG derivation, marked dependencies and guidance
targets I_D = (E, C, K, P) (Sec. IV-A), ``plan_prompt`` decides which
components apply and with which targets:

* the Selective Guidance Boost route (``B0`` default, ``P1`` Spatial Relation
  Loss, ``P2`` boosted with whole-word pooling) with its relations and
  whole-word spans, and the Count Loss constraints
  (``guidance.selective_guidance_boost.derive_image_guidance``);
* the Positional Prefix and the Region Assignment Loss word groups
  (``guidance.positional_prefix.derive_positional_prefix``), together with the
  record-only depth and clause screening records that are hashed into the
  prefix plan.

Plans are on the SD1.4 CLIP token axis; ``compositional.sana`` projects them to
Sana's Gemma tokens. Plans are those of the full method under every ablation
(``compositional.ablation``).

``operation_projection`` is the executed-operation projection of one prompt:
everything the SD1.4 kernel consumes per prompt beyond the guidance targets and
the seed. Fresh plans are compared with the decision files of the evaluated
runs through it (``evaluated_operation_projection``; used by
``scripts/paper/check_reproducibility.py t2i-plans``), because the evaluated
runs recorded the same operations in older record shapes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from ccg_diffusion.guidance._record_only_screening import ClauseLayoutPlan, DepthLayoutPlan
from ccg_diffusion.guidance.count_loss import CountGuidance, NounForms, load_noun_forms
from ccg_diffusion.guidance.positional_prefix import (
    PhraseBindingPlan,
    PrefixLayoutPlan,
    derive_positional_prefix,
    prefix_applies,
)
from ccg_diffusion.guidance.region_assignment_loss import REGION_WEIGHT
from ccg_diffusion.guidance.selective_guidance_boost import SelectiveBoostGuidance, derive_image_guidance
from ccg_diffusion.methods import FULL_T2I, T2IAblation
from ccg_diffusion.parsing.ccg import AutoSentence, MarkedDependency
from ccg_diffusion.provenance import require
from ccg_diffusion.targets.guidance_targets import CompiledGuidanceTargets
from ccg_diffusion.targets.lexicon import NominalModifierLexicon

PLAN_SCHEMA = "ccg_diffusion.t2i_compositional_plan.v1"


@dataclass(frozen=True)
class CompositionalPlan:
    """The Positional Prefix plan of one prompt and the records it was derived from."""

    phrase: PhraseBindingPlan  # the two coordinated noun phrases (eligibility)
    depth: DepthLayoutPlan  # record-only screening
    clause: ClauseLayoutPlan  # record-only screening
    prefix: PrefixLayoutPlan  # prefix text, token remapping and Region Assignment Loss word groups

    @property
    def positional_prefix(self) -> bool:
        """Whether the Positional Prefix and the Region Assignment Loss apply to the prompt."""
        return prefix_applies(self.prefix)

    def as_record(self) -> dict[str, Any]:
        return {"schema": PLAN_SCHEMA, "positional_prefix": self.positional_prefix,
                "phrase": self.phrase.as_record(), "depth": self.depth.as_record(),
                "clause": self.clause.as_record(), "prefix": self.prefix.as_record()}


@dataclass(frozen=True)
class PromptPlan:
    """Route, Count Loss and Positional Prefix plan of one prompt (full method)."""

    guidance: SelectiveBoostGuidance
    count: CountGuidance
    plan: CompositionalPlan


def derive_plan(
    prompt: str, sentence: AutoSentence, dependencies: Sequence[MarkedDependency],
    alignment: Sequence[Mapping[str, Any]], program: Mapping[str, Any],
    guidance: SelectiveBoostGuidance, count: CountGuidance, tokenizer: Any, noun_forms: NounForms,
) -> CompositionalPlan:
    """The Positional Prefix plan of one prompt from its route and count guidance."""
    return CompositionalPlan(*derive_positional_prefix(
        prompt, sentence, dependencies, alignment, program, guidance.as_record(), count.as_record(), tokenizer,
        noun_forms))


def plan_prompt(
    prompt: str, sentence: AutoSentence, dependencies: Sequence[MarkedDependency],
    targets: CompiledGuidanceTargets, tokenizer: Any, *,
    noun_forms: NounForms | None = None, lexicon: NominalModifierLexicon | None = None,
) -> PromptPlan:
    """All component decisions of one prompt; ``tokenizer`` is the pinned SD1.4 CLIP tokenizer."""
    forms = noun_forms if noun_forms is not None else load_noun_forms()
    guidance, count = derive_image_guidance(
        prompt, sentence, dependencies, targets.alignment, targets.indices,
        parse_status=targets.trace["parse_status"], unresolved_modifiers=targets.trace["unresolved_modifiers"],
        tokenizer=tokenizer, noun_forms=forms, compound_lexicon=lexicon)
    plan = derive_plan(prompt, sentence, dependencies, targets.alignment, asdict(targets.indices),
                       guidance, count, tokenizer, forms)
    return PromptPlan(guidance, count, plan)


# --- executed-operation projection ------------------------------------------------------------------------


def operation_projection(
    plan: Mapping[str, Any], guidance: Mapping[str, Any], count: Mapping[str, Any],
    ablation: T2IAblation = FULL_T2I,
) -> dict[str, Any]:
    """What the SD1.4 kernel executes for one prompt under ``ablation``.

    ``plan`` is a ``CompositionalPlan`` record (or the legacy adapter of
    ``evaluated_operation_projection``); ``guidance`` and ``count`` are the
    executed (ablated) records. Covers the route (step size and cap), whole-word
    spans, Spatial Relation Loss relations, Count Loss constraints, the text sent
    to CLIP with its token remapping and executed targets, and the Region
    Assignment Loss groups (in source coordinates when the Positional Prefix is off).
    """
    prefix = plan["prefix"]
    use_prefix = plan["positional_prefix"] and ablation.positional_prefix
    use_region = plan["positional_prefix"] and ablation.region_assignment_loss
    if plan["positional_prefix"]:
        require(prefix["phantom_anchor"] is None, "an applied Positional Prefix plan never has a phantom anchor")
    inverse = {target: source for source, target in prefix["mapping"]}
    return {
        "prompt_sha256": guidance["source_prompt_sha256"],
        "program_sha256": guidance["source_program_sha256"],
        "route": guidance["effective_profile"],
        "scale_factor": guidance["scale_factor"], "max_update_ratio": guidance["max_update_ratio"],
        "whole_word_pooling": guidance["pooling_applied"], "pair_token_spans": guidance["pair_token_spans"],
        "spatial_relations": guidance["relations"], "spatial_relation_parameters": guidance["relation_parameters"],
        "count_constraints": count["decision"]["constraints"] if count["enabled"] else [],
        "positional_prefix": None if not use_prefix else {
            "conditioning": prefix["conditioning"], "mapping": prefix["mapping"],
            "conditioning_token_count": prefix["conditioning_token_count"],
            "executed_program": prefix["mapped_program"]},
        "region_assignment": None if not use_region else {
            "relation": prefix["relation"], "weight": REGION_WEIGHT,
            "groups": prefix["groups"] if use_prefix else
            [[[inverse[i] for i in word] for word in group] for group in prefix["groups"]]},
    }


# Decision-file formats of the evaluated SD1.4 runs (record shapes of the development repository):
#   shape_v2               20260925_sd14_shape_prefix_region_v1 (full method), plan in plans.final_shape
#   shape_v2_final_repeat  the same file, plan in plans.final_repeat (the Positional Prefix scope before shape/size)
#   selected_v1            20260909_sd14_targeted_selected_full_v1 (full method before the shape/size scope)
#   component_ablation     20260925_sd14_component_ablation_v1 (guidance "on"/"off" per Selective Guidance Boost)
#   objective_ablation     20260926_sd14_objective_ablation_v1 (guidance/count per "standard"/"no_2d"/"no_count")
#   routing_v2             20260909_sd14_routing_comparison_full_v2, rows "revised_word" (route and count only)
EVALUATED_FORMATS = ("shape_v2", "shape_v2_final_repeat", "selected_v1", "component_ablation",
                     "objective_ablation", "routing_v2")
# Keys of the projection that a routing_v2 row cannot provide (it has no plan).
PREFIX_KEYS = ("positional_prefix", "region_assignment")


def evaluated_operation_projection(
    row: Mapping[str, Any], *, record_format: str, ablation: T2IAblation = FULL_T2I,
) -> dict[str, Any]:
    """``operation_projection`` of one row of an evaluated decision file (``EVALUATED_FORMATS``).

    The evaluated rows store the plan as a selected-plan record; only the
    phrase branch was selectable, so an active selection is a Positional
    Prefix. ``routing_v2`` rows carry no plan: their projection lacks ``PREFIX_KEYS``.
    """
    guidance, count = row["guidance"], row["count"]
    if record_format == "routing_v2":
        projection = operation_projection({"positional_prefix": False, "prefix": {"mapping": []}}, guidance, count,
                                          ablation)
        for key in PREFIX_KEYS:
            del projection[key]
        return projection
    if record_format in ("shape_v2", "shape_v2_final_repeat"):
        selected = row["plans"]["final_shape" if record_format == "shape_v2" else "final_repeat"]
    elif record_format in ("selected_v1", "component_ablation", "objective_ablation"):
        selected = row["plan"]
    else:
        raise ValueError(f"unknown evaluated decision format {record_format!r}; known: {EVALUATED_FORMATS}")
    if record_format == "component_ablation":
        guidance = guidance["on" if ablation.selective_guidance_boost else "off"]
    elif record_format == "objective_ablation":
        key = ablation.condition_id if ablation.condition_id in ("no_2d", "no_count") else "standard"
        guidance, count = guidance[key], count[key]
    legacy = {"positional_prefix": selected["active"] and selected["source_choice"]["branch"] == "phrase",
              "prefix": selected["source_plan"]["prefix"]}
    return operation_projection(legacy, guidance, count, ablation)
