"""Effect of the T2I-CompBench++ ablation switches on the per-image guidance (paper Tables 3, 4 and 7).

``methods.T2IAblation`` names the switches. Plans never depend on a switch:
the Positional Prefix eligibility is always that of the full method, so every
condition shares the same per-prompt plans. The switches change what runs:

* Selective Guidance Boost off: boosted images (route ``P2``) fall back to the
  default strength (route ``B0``: scale 20, cap 0.02) and no image pools whole
  words; Spatial Relation Loss (``P1``) and Count Loss images keep their route.
* Spatial Relation Loss off: ``P1`` images keep the default strength and lose
  the relation term (route ``B0``).
* Count Loss off: the count constraints stay in the record but never enter the
  loss.
* Positional Prefix and Region Assignment Loss act in the SD1.4 batch adapter
  (``compositional.sd14.run_batch``) and in the Sana plans
  (``compositional.sana.compile_plan``).

The reason strings and evidence keys below are recorded in the guidance records
(KEEP values).
"""
from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from ccg_diffusion.guidance.count_loss import CountGuidance
from ccg_diffusion.guidance.selective_guidance_boost import GuidanceDecision, GuidanceRoute, SelectiveBoostGuidance
from ccg_diffusion.methods import T2IAblation
from ccg_diffusion.provenance import canonical_json

BOOST_OFF_REASON = "guidance_strength_off_base_dose"
SPATIAL_OFF_REASON = "spatial_loss_off_same_base_dose"


def without_selective_guidance_boost(spec: SelectiveBoostGuidance) -> SelectiveBoostGuidance:
    """Boost off: a boosted image gets the default guidance strength and cap; no image pools whole words."""
    decision = spec.decision
    if spec.effective_profile == GuidanceRoute.BOOSTED:
        decision = GuidanceDecision(GuidanceRoute.DEFAULT, (BOOST_OFF_REASON,),
                                    canonical_json({"guidance_strength_ablation": spec.decision.as_record()}))
    return SelectiveBoostGuidance(
        policy="routing_v1", decision=decision, source_prompt_sha256=spec.source_prompt_sha256,
        source_program_sha256=spec.source_program_sha256, pair_spans=spec.pair_spans,
        derived_relations=spec.derived_relations, whole_word=False)


def without_spatial_relation_loss(spec: SelectiveBoostGuidance) -> SelectiveBoostGuidance:
    """Spatial Relation Loss off: a ``P1`` image keeps its default guidance strength and loses the relation term."""
    if spec.effective_profile != GuidanceRoute.SPATIAL_RELATION:
        return spec
    decision = GuidanceDecision(GuidanceRoute.DEFAULT, (SPATIAL_OFF_REASON,),
                                canonical_json({"spatial_loss_ablation": spec.decision.as_record()}))
    return SelectiveBoostGuidance(
        policy=spec.policy, decision=decision, source_prompt_sha256=spec.source_prompt_sha256,
        source_program_sha256=spec.source_program_sha256, pair_spans=spec.pair_spans,
        derived_relations=spec.derived_relations, whole_word=spec.whole_word)


def without_count_loss(spec: CountGuidance) -> CountGuidance:
    """Count Loss off: the constraints are kept for the record but never enter the loss."""
    return replace(spec, enabled=False)


def apply_ablation(
    ablation: T2IAblation,
    guidance: Mapping[str, SelectiveBoostGuidance],
    count: Mapping[str, CountGuidance],
) -> tuple[dict[str, SelectiveBoostGuidance], dict[str, CountGuidance]]:
    """The route and count guidance that the images of each base prompt execute under ``ablation``."""
    executed = dict(guidance)
    if not ablation.selective_guidance_boost:
        executed = {key: without_selective_guidance_boost(spec) for key, spec in executed.items()}
    if not ablation.spatial_relation_loss:
        executed = {key: without_spatial_relation_loss(spec) for key, spec in executed.items()}
    counted = dict(count) if ablation.count_loss else {key: without_count_loss(spec) for key, spec in count.items()}
    return executed, counted
