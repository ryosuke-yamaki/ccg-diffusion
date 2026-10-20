"""Compose CCG Diffusion's TAB components into the generation job of one interpretation.

For one interpretation of a TAB PP prompt (paper Sec. IV-C) the job gets:

* the Disambiguating Paraphrase of the prompt and its remapped guidance
  targets (the job template ``Interpretation.source_job``);
* the Selective Guidance Boost when the guidance targets are boosted (scale
  80, cap 0.06 instead of 10, 0.02);
* the ``next_to`` Spatial Relation Loss;
* the Holder-Object Region Assignment (region layout, weight 0.25);
* the attachment possession removed from the guidance targets (the standard
  program of the interpretation has none).

The Table 6 switches (``methods.TabAblation``) remove the paraphrase (the job
conditions on the source prompt with the same guidance targets) or the region
layout. ``interpretation_from_parse`` builds an ``Interpretation`` from a
Supertag-Constrained derivation (the clean path); the frozen bundle stores the
same fields for the paper runs.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from ccg_diffusion.guidance.selective_guidance_boost import GuidanceRoute, derive_image_guidance
from ccg_diffusion.methods import TabAblation
from ccg_diffusion.parsing.ccg import AutoSentence, MarkedDependency
from ccg_diffusion.parsing.sentences import tokenize_for_parse
from ccg_diffusion.pp_attachment.disambiguating_paraphrase import paraphrase_interpretation, source_job
from ccg_diffusion.pp_attachment.holder_object import region_layout
from ccg_diffusion.pp_attachment.spatial import next_to_relations
from ccg_diffusion.pp_attachment.supertag_constrained_parsing import PrepositionSite
from ccg_diffusion.provenance import digest, sha256_text
from ccg_diffusion.targets.guidance_targets import (
    CompiledGuidanceTargets,
    possession_lemma_annotations_from_spacy,
    sd14_guidance_targets,
    sd14_interpretation_guidance_targets,
)
from ccg_diffusion.targets.pd_indices import program_sha256

TRANSFER_SCHEMA = "holsd.tab_pp_t2i_transfer.v1"
ABLATION_SCHEMA = "holsd.tab_pp_ablation.v1"
RECIPES = {"combined_region_only": ("p2", "spatial", "region"), "combined_no_layout": ("p2", "spatial")}


@dataclass(frozen=True)
class InterpretationTargets:
    """Guidance targets I_D of one interpretation on the source prompt.

    ``pd_indices`` include the attachment possession, ``alignment`` holds the
    full CLIP alignment entries, ``boost`` is the Selective Guidance Boost
    decision and ``spatial_relations`` are the ``next_to`` constraints (source
    CLIP indices). ``record_sha256`` identifies the record they come from
    (recorded as ``transfer_trace.source_audit_sha256``).
    """

    prompt: str
    alignment: tuple[Mapping[str, Any], ...]
    pd_indices: Mapping[str, Any]
    boost: bool
    spatial_relations: tuple[Mapping[str, Any], ...]
    record_sha256: str


@dataclass(frozen=True)
class Interpretation:
    """Everything the compile needs for one interpretation (``reading`` ``np_attach`` or ``vp_attach``)."""

    tab_id: str
    reading: str
    source_job: Mapping[str, Any]
    paraphrase: Mapping[str, Any]
    targets: InterpretationTargets
    standard_program: Mapping[str, Any]


def source_identity_trace(trace: Mapping[str, Any]) -> dict[str, Any]:
    """Describe the source prompt with a forced reading's source bindings (Disambiguating Paraphrase off).

    The realization is kept: it certifies that the two forced readings attach
    the PP to different holders. Only the conditioning text is replaced.
    """
    result = deepcopy(dict(trace))
    result.update(conditioning_prompt_sha256=trace["source_prompt_sha256"],
                  program_sha256=trace["source_program_sha256"],
                  source_to_conditioning_clip={k: int(k) for k in trace["source_to_conditioning_clip"]})
    for entry in result.get("source_mapping") or ():
        entry.update(conditioning_clip_index=entry["source_clip_index"],
                     conditioning_char_span=list(entry["source_char_span"]),
                     conditioning_text=entry["source_text"])
    return result


def _compile_job(
    source: Mapping[str, Any], selected: Mapping[str, Any], opposite: Mapping[str, Any],
    targets: InterpretationTargets, tokenizer: Any, recipe: str,
) -> dict[str, Any]:
    """Apply the recipe's components (boost, region layout, spatial relations) to one PP job template."""
    row = deepcopy(dict(source))
    features = RECIPES[recipe]
    trace: dict[str, Any] = {"schema": TRANSFER_SCHEMA, "recipe": recipe, "evaluation_outcomes_consumed": False}
    row["transfer_trace"] = trace
    if source["source_program_sha256"] != digest(targets.pd_indices):
        raise ValueError("TAB historical and audited source programs differ")
    trace["source_audit_sha256"] = targets.record_sha256
    if "p2" in features and targets.boost:
        row["loss_config"].update(scale_factor=80.0, max_update_ratio=0.06)
        trace["p2"] = True
    mapping = {i: i for i in range(1, len(row["token_ids"]) - 1)}
    if "region" in features:
        plan, remapped, reason = region_layout(row, selected, opposite, targets, tokenizer)
        trace["layout_reason"] = reason
        if plan is not None:
            mapping = remapped
            row.update(prompt=plan.conditioning, token_ids=tokenizer(plan.conditioning)["input_ids"],
                       pd_indices=plan.mapped_program, program_sha256=digest(plan.mapped_program))
            trace["layout"] = plan.as_record()
            row["transfer_region"] = {"plan": plan.as_record(), "weight": 0.25}
    if "spatial" in features and targets.spatial_relations:
        original_to_current = selected.get("source_to_conditioning_clip")
        if original_to_current is None:
            original_to_current = {str(i): i for i in range(1, len(source["token_ids"]) - 1)}
        row["relations"] = [{
            "label": rel["label"],
            "subject_clip_index": mapping[int(original_to_current[str(rel["subject_clip_index"])])],
            "object_clip_index": mapping[int(original_to_current[str(rel["object_clip_index"])])],
        } for rel in targets.spatial_relations]
        row["loss_config"]["relation_loss_weight"] = 6.0
        trace["spatial"] = True
    return row


def compile_interpretation(
    selected: Interpretation, opposite: Interpretation, tokenizer: Any, ablation: TabAblation,
) -> dict[str, Any]:
    """The TAB job of the ``selected`` interpretation (``opposite`` is the other interpretation)."""
    recipe = ablation.recipe
    targets = selected.targets
    row = dict(selected.source_job)
    selected_trace, opposite_trace = selected.paraphrase, opposite.paraphrase
    source_to_current = selected_trace["source_to_conditioning_clip"]
    if not ablation.disambiguating_paraphrase:
        prompt = targets.prompt
        if sha256_text(prompt) != selected_trace["source_prompt_sha256"]:
            raise ValueError("audit and conditioning trace refer to different source prompts")
        row.update(prompt=prompt, token_ids=tokenizer(prompt)["input_ids"],
                   pd_indices=deepcopy(dict(targets.pd_indices)), program_sha256=digest(targets.pd_indices))
        selected_trace, opposite_trace = source_identity_trace(selected_trace), source_identity_trace(opposite_trace)
        source_to_current = selected_trace["source_to_conditioning_clip"]
    result = _compile_job(row, selected_trace, opposite_trace, targets, tokenizer, recipe)
    if result["prompt"] != row["prompt"]:
        raise ValueError("ablation cells must not add a layout prefix")
    standard_program = selected.standard_program
    if standard_program["possession"]:
        raise ValueError("Poss. off requires the reading's standard PD program")

    def to_current(i: int) -> int:
        return int(source_to_current[str(i)])

    # Rewritten programs list indices in conditioning order; compare them as sets.
    expected = {"exist": sorted(to_current(i) for i in standard_program["exist"]),
                **{field: sorted([to_current(a), to_current(b)] for a, b in standard_program[field])
                   for field in ("corr", "leak")}}
    removed = result["pd_indices"]["possession"]
    stripped = {**result["pd_indices"], "possession": []}
    if {field: sorted(stripped[field]) for field in expected} != expected:
        raise ValueError("standard PD program differs from the attachment program without possession")
    result.update(pd_indices=stripped, program_sha256=digest(stripped))
    result["transfer_trace"]["ablation"] = {
        "schema": ABLATION_SCHEMA, "cell": ablation.cell_id, "switches": ablation.switches, "base_recipe": recipe,
        "conditioning": "interpretation_rewrite" if ablation.disambiguating_paraphrase else "source_prompt",
        "possession": "standard_pd_rule",
        "removed_possession": removed,
        "phrase_value_mixing": False,
    }
    return result


def compile_auxiliary_job(source_job: Mapping[str, Any], ablation: TabAblation) -> dict[str, Any]:
    """A VP row of the 3,150-job layout: the frozen job with only the transfer trace added (never generated)."""
    row = deepcopy(dict(source_job))
    row["transfer_trace"] = {"schema": TRANSFER_SCHEMA, "recipe": ablation.recipe,
                             "evaluation_outcomes_consumed": False}
    return row


# --- clean path ---------------------------------------------------------------------------------------


def interpretation_targets(
    prompt: str, sentence: AutoSentence, dependencies: Sequence[MarkedDependency], tokenizer: Any, nlp: Any,
) -> tuple[CompiledGuidanceTargets, CompiledGuidanceTargets]:
    """Stage 3 of one interpretation: ``(guidance targets with attachment possession, standard targets)``."""
    tokens = tuple(tokenize_for_parse(prompt))
    lemmas = possession_lemma_annotations_from_spacy(prompt, nlp, source_tokens=tokens)
    compiled = sd14_interpretation_guidance_targets(
        prompt, sentence, dependencies, tokenizer, source_tokens=tokens,
        possession_lemmas_by_source=lemmas, parse_status="complete")
    standard = sd14_guidance_targets(
        prompt, sentence, dependencies, tokenizer, source_tokens=tokens,
        possession_lemmas_by_source=lemmas, parse_status="complete")
    return compiled, standard


def interpretation_from_parse(
    *, tab_id: str, reading: str, prompt: str, item_index: int, site: PrepositionSite, sentence: AutoSentence,
    dependencies: Sequence[MarkedDependency], tokenizer: Any, nlp: Any, noun_forms: Any,
    evaluation: Mapping[str, Any],
) -> Interpretation:
    """Build one interpretation from its Supertag-Constrained derivation and marked dependencies (clean path).

    ``sentence`` carries the AUTO ID of the 1-based PP item index
    (``supertag_constrained_parsing.AUTO_ID``); ``evaluation`` is the
    interpretation's evaluation identity from the reading -> setup mapping.
    """
    if sentence.id_line != f"ID=prompts.{item_index} PARSER=HOLCCG NUMPARSE=1":
        raise ValueError("interpretation AUTO id does not name the PP item index")
    compiled, standard = interpretation_targets(prompt, sentence, dependencies, tokenizer, nlp)
    trace = compiled.trace
    guidance, _count = derive_image_guidance(
        prompt, sentence, dependencies, compiled.alignment, compiled.indices, parse_status=trace["parse_status"],
        unresolved_modifiers=trace["unresolved_modifiers"], tokenizer=tokenizer, noun_forms=noun_forms)
    boost = guidance.effective_profile == GuidanceRoute.BOOSTED
    alignment = list(compiled.alignment)
    pd_indices = asdict(compiled.indices)
    relations, _evidence = next_to_relations(
        prompt=prompt, site_source_index=site.source_index, alignment=alignment, pd_indices=pd_indices,
        conversion_trace=trace, sentence=sentence, dependencies=dependencies)
    target_row = {"prompt_id": f"{tab_id}_{reading}_s0", "prompt_sha256": sha256_text(prompt),
                  "program_sha256": program_sha256(compiled.indices), "alignment": alignment,
                  "pd_indices": pd_indices}
    conversion_row = {"conversion_trace": trace, "conversion_trace_sha256": digest(trace)}
    paraphrase = paraphrase_interpretation(
        prompt=prompt, target_row=target_row, conversion_trace_row=conversion_row,
        site_source_index=site.source_index, sentence=sentence, tokenizer=tokenizer)
    job = source_job(tab_id=tab_id, reading=reading, paraphrase=paraphrase,
                     source_program_sha256=target_row["program_sha256"], tokenizer=tokenizer,
                     evaluation=evaluation)
    record = {"prompt": prompt, "alignment": alignment, "pd_indices": pd_indices, "boost": boost,
              "spatial_relations": relations, "conversion_trace": trace}
    targets = InterpretationTargets(prompt, tuple(alignment), pd_indices, boost, tuple(relations), digest(record))
    return Interpretation(tab_id, reading, job, dict(paraphrase.trace), targets, asdict(standard.indices))
