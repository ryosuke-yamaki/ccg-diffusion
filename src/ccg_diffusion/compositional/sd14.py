"""CCG Diffusion on Stable Diffusion v1.4 for T2I-CompBench++: prepare, generate and audit a run.

Layout of run ``R`` (``output/generation/R``)::

    inputs/image_guidance_targets.sd14.jsonl (+ .meta.json)   guidance targets of the 24,000 images
    [C/]protocol.json, [C/]inputs/decisions.jsonl               full method, or ablation condition C
    [C/]images/sd14_pd_holccg/...                               images, metadata and run locks of 4 shards
    [C/]control/                                                shard receipts and audits

A verification subset generated with ``output_dir`` keeps its images, shard
receipts and audit under that directory (``<output_dir>/control/``) and writes
nothing into the run.

``prepare`` reads the Stage 1-3 records of every prompt through a source
manifest (Stage 3 is recompiled and checked), derives each prompt's plan
(``plans.plan_prompt``), applies the ablation (``ablation.apply_ablation``) and
freezes the protocol: configuration, source seal, the digests of the
per-prompt route and count records, and the image population. The image-row
guidance-target file is written once per run and shared by its conditions.

``generate`` re-derives the plans, checks them against the decisions file and
runs one of the four shards through the SD1.4 batch driver in the evaluated
physical batches (image ``i`` in shard ``i % 4``, four consecutive images of a
shard per batch). ``run_batch`` is the driver's batch function: it applies the
Positional Prefix (paper Eq. 4) as a K/V conditioning hook and the Region
Assignment Loss (Eq. 5-6) as a loss override, according to the ablation. An
ablation condition generates the whole 16-image chunks (4 shards x 4 images)
that contain its scored categories, so every image keeps its batch peers.

``run_batch`` is the batch adapter of every evaluated condition. The full
method's evaluated adapter also entered two hooks with zero strength (regional
attention 0, phrase values 0), which return before changing anything, so it
made the same kernel call.
"""
from __future__ import annotations

import functools
import math
import re
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image

from ccg_diffusion.backbones.sd14 import driver
from ccg_diffusion.backbones.sd14.inputs import load_t2icb_dataset_provenance
from ccg_diffusion.backbones.sd14.pipeline import generate_predicated_batch
from ccg_diffusion.backbones.sd14.records import current_gpu_uuid, run_lock_path, scheduler_record, shard_suffix
from ccg_diffusion.compositional.ablation import apply_ablation
from ccg_diffusion.compositional.plans import CompositionalPlan, plan_prompt
from ccg_diffusion.config import (
    HOLCCG_PACKAGE_VERSION,
    SD14_CCG_ARM,
    SD14_GENERATION,
    SD14_GUIDANCE_TARGET_PROVIDER,
    SD14_SCHEDULER_CONFIG_SHA256,
    T2I_COMPBENCH_DATASET_DIR,
)
from ccg_diffusion.guidance.count_loss import (
    MIN_MEAN_ATTENTION,
    MIN_SPATIAL_RANGE,
    CountGuidance,
    load_noun_forms,
)
from ccg_diffusion.guidance.positional_prefix import prefix_conditioning
from ccg_diffusion.guidance.region_assignment_loss import REGION_WEIGHT, PrefixRegionLoss, in_source_coordinates
from ccg_diffusion.guidance.selective_guidance_boost import GuidanceRoute, SelectiveBoostGuidance
from ccg_diffusion.methods import FULL_T2I, T2I_SD14, T2IAblation, get_config
from ccg_diffusion.provenance import (
    artifact_record,
    atomic_write_json,
    atomic_write_jsonl,
    digest,
    implementation_record,
    read_json_object,
    read_jsonl,
    repo_relative,
    repository_root,
    require,
    resolve_recorded_path,
    sha256_bytes,
    sha256_file,
    source_tree_snapshot,
    verify_artifact_record,
    verify_implementation_record,
)
from ccg_diffusion.targets.guidance_targets import (
    SourcePrompt,
    expand_to_image_rows,
    load_source_prompts,
    load_t2i_parse_bundle,
    target_file_metadata_path,
    write_image_rows_file,
)
from ccg_diffusion.targets.tokenizer import load_sd14_tokenizer

CONFIGURATION_ID = T2I_SD14
ARM_ID = SD14_CCG_ARM  # driver method and output directory (KEEP)
NUM_SHARDS = 4
CHUNK_ROWS = 16  # 4 shards x physical batch 4: canonical chunk c is batch c of every shard
PROTOCOL_SCHEMA = "ccg_diffusion.t2i_sd14_protocol.v1"
SOURCE_SEAL_SCHEMA = "ccg_diffusion.source_seal.v1"
RECEIPT_SCHEMA = "ccg_diffusion.t2i_sd14_shard_receipt.v1"
AUDIT_SCHEMA = "ccg_diffusion.t2i_sd14_audit.v1"
OPERATION_KEY = "t2i_operation"  # key in the per-image semantic_guidance diagnostics
OPERATION_SCHEMA = "ccg_diffusion.t2i_operation.v1"
DATASET_DIR = repository_root() / T2I_COMPBENCH_DATASET_DIR
DEFAULT_IMAGE_MANIFEST = DATASET_DIR / "independent_seed_manifest.jsonl"
TARGET_FILE = "inputs/image_guidance_targets.sd14.jsonl"
# The files of this run type beyond the driver's own (driver.SD14_DRIVER_COMPONENT_FILES):
# the ccg_diffusion import closure of this module and the command line.
COMPONENT_FILES = (
    "scripts/t2i_compbench/run_ccg_diffusion.py",
    "src/ccg_diffusion/compositional/ablation.py",
    "src/ccg_diffusion/compositional/plans.py",
    "src/ccg_diffusion/compositional/sd14.py",
    "src/ccg_diffusion/guidance/_record_only_screening.py",
    "src/ccg_diffusion/guidance/positional_prefix.py",
    "src/ccg_diffusion/guidance/region_assignment_loss.py",
    "src/ccg_diffusion/methods.py",
)
# Kernel arguments of every batch (paper Table 8; the driver passes exactly these).
BATCH_ARGUMENTS = {
    "steps": 50, "guidance_scale": 7.5, "height": 512, "width": 512, "attention_res": 16,
    "max_iter_to_alter": 25, "scale_factor": 20.0, "scale_range": (1.0, 0.5), "thresholds": {0: 0.0},
    "refinement_updates_per_threshold": 5, "strict_finite": True, "max_update_ratio": 0.02, "max_latent_abs": 20.0,
}
BATCH_INPUTS = frozenset({"device", "initial_latents", "diagnostics", "image_guidance", "count_guidance"})


def component_files() -> tuple[str, ...]:
    """Source files sealed into a protocol and into every run lock: ``COMPONENT_FILES`` and the driver's."""
    return tuple(sorted({*COMPONENT_FILES, *driver.SD14_DRIVER_COMPONENT_FILES}))


def _root(repository_root_: Path | None) -> Path:
    return (repository_root_ or repository_root()).resolve()


def _record(path: Path, root: Path) -> dict[str, Any]:
    return artifact_record(path.resolve()).to_dict(relative_to=root)


def run_dir(run_id: str, ablation: T2IAblation = FULL_T2I, *, repository_root: Path | None = None) -> Path:
    """``output/generation/<run_id>`` for the full method, ``.../<run_id>/<condition_id>`` for an ablation."""
    root = _root(repository_root) / "output/generation" / run_id
    return root if ablation.is_full else root / ablation.condition_id


# --- population -------------------------------------------------------------------------------------------


def validate_image_rows(canonical: Sequence[Mapping[str, Any]], *, num_shards: int) -> dict[str, str]:
    """Check that the image rows fill physical batches of 4 in every shard; return base prompt ID -> prompt."""
    require(type(num_shards) is int and num_shards > 0, "num_shards must be a positive integer")
    require(bool(canonical) and len(canonical) % (4 * num_shards) == 0,
            "fresh T2I population must contain complete physical B4 batches in each shard")
    prompts: dict[str, str] = {}
    seen: set[str] = set()
    for row in canonical:
        require(set(row) == {"prompt_id", "prompt", "seed", "experiment"}, "unexpected generation input fields")
        key, sep, seed = row["prompt_id"].rpartition("_s")
        require(bool(key) and sep == "_s" and seed.isdigit() and type(row["seed"]) is int
                and int(seed) == row["seed"] and row["seed"] >= 0, "explicit image IDs must end in their seed")
        require(row["prompt_id"] not in seen, "duplicate image ID")
        require(isinstance(row["prompt"], str) and bool(row["prompt"]), "empty prompt")
        require(key not in prompts or prompts[key] == row["prompt"], "image rows disagree on original text")
        seen.add(row["prompt_id"])
        prompts[key] = row["prompt"]
    return prompts


def chunk_closure(canonical: Sequence[Mapping[str, Any]], categories: Iterable[str]) -> tuple[list[str], list[str]]:
    """Scored images of ``categories`` and the whole 16-image chunks that contain them (the generated images)."""
    experiments = {f"t2icb_{category}" for category in categories}
    scored = [row["prompt_id"] for row in canonical if row["experiment"] in experiments]
    chosen = set(scored)
    generated = []
    for offset in range(0, len(canonical), CHUNK_ROWS):
        chunk = canonical[offset:offset + CHUNK_ROWS]
        if any(row["prompt_id"] in chosen for row in chunk):
            require(len(chunk) == CHUNK_ROWS, "selected chunk does not contain four complete physical batches")
            generated.extend(row["prompt_id"] for row in chunk)
    return scored, generated


# --- plans ------------------------------------------------------------------------------------------------


def derive(
    sources: Sequence[SourcePrompt], prompts: Mapping[str, str], tokenizer: Any, ablation: T2IAblation,
) -> tuple[dict[str, SelectiveBoostGuidance], dict[str, CountGuidance], dict[str, CompositionalPlan]]:
    """Full-method plans of every base prompt, and the route and count guidance that ``ablation`` executes."""
    require({source.prompt_id: source.prompt for source in sources} == dict(prompts),
            "source prompts differ from the image manifest")
    forms = load_noun_forms()
    guidance, count, plans = {}, {}, {}
    for source in sources:
        planned = plan_prompt(source.prompt, source.sentence, source.dependencies, source.targets, tokenizer,
                              noun_forms=forms)
        key = source.prompt_id
        guidance[key], count[key], plans[key] = planned.guidance, planned.count, planned.plan
    guidance, count = apply_ablation(ablation, guidance, count)
    return guidance, count, plans


def decision_records(
    guidance: Mapping[str, SelectiveBoostGuidance], count: Mapping[str, CountGuidance],
    plans: Mapping[str, CompositionalPlan],
) -> list[dict[str, Any]]:
    """One row per base prompt: plan, executed route record and executed count record."""
    return [{"prompt_id": key, "plan": plan.as_record(), "guidance": guidance[key].as_record(),
             "count": count[key].as_record()} for key, plan in plans.items()]


def _census(guidance: Mapping[str, SelectiveBoostGuidance], count: Mapping[str, CountGuidance],
            plans: Mapping[str, CompositionalPlan]) -> dict[str, Any]:
    return {"routes": dict(sorted(Counter(spec.effective_profile for spec in guidance.values()).items())),
            "count_status": dict(sorted(Counter(spec.decision.status for spec in count.values()).items())),
            "count_prompts": sum(bool(spec.decision.constraints) for spec in count.values()),
            "count_constraints": sum(len(spec.decision.constraints) for spec in count.values()),
            "positional_prefix": sum(plan.positional_prefix for plan in plans.values())}


# --- Stage 1 parser provenance ------------------------------------------------------------------------------


def parser_provenance(source_manifest: Path, *, repository_root: Path | None = None) -> list[dict[str, Any]]:
    """The Stage 1 metadata of every source bundle, checked against the configuration's parser.

    Stage 1 writes ``<auto>.meta.json``. A stage directory unpacked from the
    frozen T2I parse bundle (``data/t2i_compbench``) has none: the tracked
    metadata that the bundle manifest names for that AUTO file is used, and the
    stage directory's ``<cat>.raw`` must have the recorded raw sha256.
    """
    root = _root(repository_root)
    parser = get_config(CONFIGURATION_ID).parser
    bundle_metadata = {
        entry["raw_sha256"]: next(s["meta"] for s in entry["sources"] if s["field"] == "output_auto_sha256")
        for entry in load_t2i_parse_bundle(verify=False)["manifest"]["files"] if entry["kind"] == "auto"}
    result = []
    for source in read_json_object(source_manifest)["sources"]:
        auto = resolve_recorded_path(source["auto"], repository_root=root)
        metadata_path = Path(str(auto) + ".meta.json")
        auto_sha256 = sha256_file(auto)
        bundled = not metadata_path.is_file() and auto_sha256 in bundle_metadata
        if bundled:
            metadata_path = root / bundle_metadata[auto_sha256]
        require(metadata_path.is_file(), "explicit Stage 1 AUTO metadata is required for fresh paper generation")
        record = read_json_object(metadata_path)
        require(record.get("artifact") == "holccg_parse_auto" and record.get("schema_version") == 1,
                "unsupported Stage 1 provenance schema")
        require(record.get("output_auto_sha256") == auto_sha256, "Stage 1 AUTO provenance differs")
        require(isinstance(record.get("model_dir"), str) and bool(record["model_dir"])
                and isinstance(record.get("holccg_version"), str), "Stage 1 checkpoint/package is unrecorded")
        require(record["model_dir"] == parser.checkpoint and record["holccg_version"] == HOLCCG_PACKAGE_VERSION,
                "Stage 1 model/package differs from the paper configuration")
        required = {"stag_threshold", "phrase_threshold", "span_threshold", "apply_skimmer",
                    "grammar", "chunk_size", "min_rule_freq"}
        require(required <= set(record.get("parsing", {})), "Stage 1 effective settings are incomplete")
        require(all(record["parsing"][key] == getattr(parser, key) for key in required),
                "Stage 1 settings differ from the paper configuration")
        if record.get("resolved_revision") is not None:
            require(record["resolved_revision"] == parser.revision, "Stage 1 checkpoint revision differs from the pin")
        if bundled:
            raw = auto.with_suffix(".raw")
        else:
            raw = resolve_recorded_path(record["input_raw_path"], repository_root=root)
        require(raw.is_file() and sha256_file(raw) == record["input_raw_sha256"], "Stage 1 raw source differs")
        result.append({"metadata": _record(metadata_path, root), "raw": _record(raw, root), "record": record,
                       "frozen_parse_bundle": bundled,
                       "resolved_revision_status": "recorded" if record.get("resolved_revision") else
                       "not_recorded_in_supplied_stage1_metadata"})
    return result


# --- source seal ------------------------------------------------------------------------------------------


def _seal(*, allow_dirty: bool, repository_root: Path) -> dict[str, Any]:
    """The committed component files at HEAD, or (``allow_dirty``) their working-tree bytes."""
    if allow_dirty:
        return source_tree_snapshot(component_files(), repository_root=repository_root)
    return implementation_record(component_files(), schema=SOURCE_SEAL_SCHEMA, repository_root=repository_root)


def _sealed_files(source: Mapping[str, Any]) -> dict[str, str]:
    if "components" in source:
        return {path: value["sha256"] for path, value in source["components"].items()}
    return dict(source["files"])


def _verify_source(source: Mapping[str, Any], *, allow_dirty: bool, repository_root: Path) -> None:
    """Committed seal: the protocol commit is HEAD and the files are its blobs. ``allow_dirty``: same bytes."""
    if allow_dirty:
        current = {path: sha256_file(repository_root / path) for path in component_files()}
        require(current == _sealed_files(source), "source files changed since the protocol was prepared")
        return
    require("components" in source, "the protocol was prepared with --allow-dirty; generate it with --allow-dirty")
    verify_implementation_record(dict(source), component_files(), schema=SOURCE_SEAL_SCHEMA, at_head=True,
                                 repository_root=repository_root)


# --- prepare ----------------------------------------------------------------------------------------------


def write_image_guidance_targets(
    run_root: Path, sources: Sequence[SourcePrompt], canonical: Sequence[Mapping[str, Any]],
    image_manifest: Path, source_manifest: Path, *, repository_root: Path | None = None,
) -> Path:
    """The guidance-target row of every image (``{**Stage 3 row, prompt_id, source_sidecar_prompt_id}``).

    Written once per run; a later condition reuses it after checking its bytes
    and its source manifest against the ``.meta.json``.
    """
    root = _root(repository_root)
    path = run_root / TARGET_FILE
    source_record = _record(source_manifest, root)
    if path.exists():
        metadata = read_json_object(target_file_metadata_path(path))
        verify_artifact_record(metadata["output"], repository_root=root)
        require(metadata["parser_provenance"]["source_manifest"]["sha256"] == source_record["sha256"],
                "the run's image-row guidance targets come from another source manifest")
        return path
    experiments = {row["prompt_id"].rsplit("_s", 1)[0]: row["experiment"] for row in canonical}
    rows, used = expand_to_image_rows(
        [{"prompt_id": s.prompt_id, "prompt": s.prompt, "experiment": experiments[s.prompt_id]} for s in sources],
        [s.pd_row for s in sources], canonical)
    write_image_rows_file(
        path, rows, provider=SD14_GUIDANCE_TARGET_PROVIDER, tokenizer_identity=SD14_GENERATION.tokenizer_identity(),
        parser_provenance={"source_manifest": source_record}, target_prompt_manifest=image_manifest,
        used_source_prompt_count=len(used))
    return path


def prepare(
    *, run_id: str, source_manifest: Path, tokenizer: Any, ablation: T2IAblation = FULL_T2I,
    image_manifest: Path = DEFAULT_IMAGE_MANIFEST, allow_dirty: bool = False, repository_root: Path | None = None,
) -> Path:
    """Freeze the protocol of the full method or of one ablation condition; return its path.

    ``tokenizer`` is the pinned SD1.4 CLIP tokenizer (slow). Nothing of an
    evaluated run is read; the plans come from the Stage 1-3 records only.
    """
    require(re.fullmatch(r"\d{8}_[a-zA-Z0-9_-]+", run_id) is not None, "run ID must be YYYYMMDD_slug")
    root = _root(repository_root)
    canonical = read_jsonl(image_manifest)
    prompts = validate_image_rows(canonical, num_shards=NUM_SHARDS)
    dataset = load_t2icb_dataset_provenance(image_manifest, repository_root=root).run_record()
    sources, source_artifacts = load_source_prompts(
        source_manifest, image_manifest.parent / "prompt_only.jsonl", tokenizer, repository_root=root)
    stage1 = parser_provenance(source_manifest, repository_root=root)
    source = _seal(allow_dirty=allow_dirty, repository_root=root)
    directory = run_dir(run_id, ablation, repository_root=root)
    protocol_path = directory / "protocol.json"
    require(not protocol_path.exists(), f"protocol already exists: {protocol_path}")
    guidance, count, plans = derive(sources, prompts, tokenizer, ablation)
    decisions = directory / "inputs/decisions.jsonl"
    atomic_write_jsonl(decisions, decision_records(guidance, count, plans), overwrite=False)
    targets = write_image_guidance_targets(run_dir(run_id, repository_root=root), sources, canonical,
                                           image_manifest, source_manifest, repository_root=root)
    scored, generated = (None, None) if ablation.is_full else chunk_closure(canonical, ablation.scored_categories)
    configuration = get_config(CONFIGURATION_ID)
    protocol = {
        "schema": PROTOCOL_SCHEMA, "run_id": run_id, "configuration_id": CONFIGURATION_ID,
        "configuration": configuration.as_record(), "configuration_sha256": configuration.sha256,
        "condition_id": ablation.condition_id, "ablation": asdict(ablation), "components": ablation.components(),
        "export_arm": "ccg_diffusion" if ablation.is_full else ablation.condition_id,
        "source": source, "stage1_provenance": stage1, "num_shards": NUM_SHARDS, "dataset": dataset,
        "image_count": len(canonical), "image_ids": generated, "scored_image_ids": scored,
        "selected_image_ids_sha256": digest(tuple(generated)) if generated is not None else None,
        "per_prompt_guidance_sha256": digest({key: spec.as_record() for key, spec in sorted(guidance.items())}),
        "per_prompt_count_guidance_sha256": digest({key: spec.as_record() for key, spec in sorted(count.items())}),
        "census": _census(guidance, count, plans),
        "inputs": {"image_manifest": _record(image_manifest, root), "source_manifest": _record(source_manifest, root),
                   "image_guidance_targets": _record(targets, root),
                   "image_guidance_targets_metadata": _record(target_file_metadata_path(targets), root),
                   "decisions": _record(decisions, root)},
        "source_artifacts": source_artifacts,
    }
    _verify_source(source, allow_dirty=allow_dirty, repository_root=root)
    atomic_write_json(protocol_path, protocol, overwrite=False)
    return protocol_path


def load_protocol(path: Path, *, allow_dirty: bool = False, repository_root: Path | None = None) -> dict[str, Any]:
    """Read a protocol and check the configuration, the source seal and every input it names."""
    root = _root(repository_root)
    protocol = read_json_object(path)
    configuration = get_config(CONFIGURATION_ID)
    require(protocol.get("schema") == PROTOCOL_SCHEMA and protocol.get("configuration") == configuration.as_record()
            and protocol.get("configuration_sha256") == configuration.sha256, "paper configuration changed")
    _verify_source(protocol["source"], allow_dirty=allow_dirty, repository_root=root)
    for record in (*protocol["inputs"].values(), *protocol["source_artifacts"]):
        verify_artifact_record(record, repository_root=root)
    image_manifest = resolve_recorded_path(protocol["inputs"]["image_manifest"]["path"], repository_root=root)
    require(load_t2icb_dataset_provenance(image_manifest, repository_root=root).run_record() == protocol["dataset"],
            "dataset provenance changed")
    for stage1 in protocol["stage1_provenance"]:
        verify_artifact_record(stage1["metadata"], repository_root=root)
        verify_artifact_record(stage1["raw"], repository_root=root)
    return protocol


# --- the batch function -------------------------------------------------------------------------------------


def loss_override(ablation: T2IAblation, plan: CompositionalPlan, program: Mapping[str, Any]) -> Any:
    """The loss of one image: ``None`` keeps the kernel's Predicated Diffusion loss on the source prompt."""
    if not plan.positional_prefix or not (ablation.positional_prefix or ablation.region_assignment_loss):
        return None
    if ablation.positional_prefix:
        return PrefixRegionLoss(plan.prefix,
                                region_weight=REGION_WEIGHT if ablation.region_assignment_loss else 0.0)
    return PrefixRegionLoss(in_source_coordinates(plan.prefix, program), region_weight=REGION_WEIGHT)


def run_batch(ablation: T2IAblation, pipe: Any, jobs: Sequence[Any], plans: Sequence[CompositionalPlan],
              **kwargs: Any) -> Any:
    """One physical batch: Positional Prefix K/V hook and loss overrides around the guided kernel.

    ``kwargs`` are the driver's kernel arguments and pass through unchanged,
    plus ``loss_overrides`` when some image of the batch has an override.
    Records the executed operation of every image under ``OPERATION_KEY``.
    """
    require(len(plans) == len(jobs), "ablation batch requires one selected plan per physical row")
    require("loss_overrides" not in kwargs, "another loss override is already installed")
    programs = [{"exist": job.attention_exist_indices, "corr": job.attention_corr_indices,
                 "leak": job.attention_leak_indices, "possession": job.attention_possession_indices} for job in jobs]
    contexts = [plan.prefix if plan.positional_prefix and ablation.positional_prefix
                else replace(plan.prefix, conditioning=None) for plan in plans]
    overrides = [loss_override(ablation, plan, program) for plan, program in zip(plans, programs)]
    if any(override is not None for override in overrides):
        kwargs["loss_overrides"] = overrides
    with prefix_conditioning(pipe, [job.prompt for job in jobs], contexts, device=kwargs["device"]) as hook:
        result = generate_predicated_batch(pipe, jobs, **kwargs)
    for i, (diagnostic, plan, override) in enumerate(zip(kwargs["diagnostics"], plans, overrides, strict=True)):
        diagnostic.semantic_guidance[OPERATION_KEY] = {
            "schema": OPERATION_SCHEMA, "plan_sha256": digest(plan.as_record()),
            "condition_id": ablation.condition_id, "components": ablation.components(),
            "prefix_applied": i in hook["active_rows"],
            "projection_calls": hook["projection_calls"] if i in hook["active_rows"] else 0,
            "loss_override": override is not None,
            "executed_program_sha256": override.executed_program_sha256 if override is not None else None,
            "region_weight": override.region_weight if override is not None else 0.0,
            "loss_evaluations": override.evaluations if override is not None else 0,
            "geometry_sum": override.geometry_sum if override is not None else 0.0,
            "context_cache": {"builds": hook["context_cache_builds"], "hits": hook["context_cache_hits"],
                              "verifications": hook["context_cache_verifications"]}}
    return result


def check_batch(pipe: Any, jobs: Sequence[Any], kwargs: Mapping[str, Any]) -> None:
    """Reject a batch whose kernel arguments, size or scheduler differ from the paper configuration."""
    require(set(kwargs) == set(BATCH_ARGUMENTS) | BATCH_INPUTS
            and all(kwargs[key] == value for key, value in BATCH_ARGUMENTS.items()),
            "actual batch arguments differ from the paper configuration")
    require(len(jobs) == SD14_GENERATION.batch_size, "a T2I batch has four images")
    require(getattr(pipe, "safety_checker", None) is None,
            "safety-checker configuration differs from the evaluated runs")
    require(scheduler_record(pipe) == {"class": "PNDMScheduler", "config_sha256": SD14_SCHEDULER_CONFIG_SHA256,
                                       "prediction_type": "epsilon"}, "loaded scheduler differs from the paper")


def verify_operation(ablation: T2IAblation, plan: Mapping[str, Any], diagnostics: Mapping[str, Any]) -> bool:
    """Check that one generated image executed exactly its planned prefix/region operation; return override."""
    selected = plan["positional_prefix"]
    prefixed = selected and ablation.positional_prefix
    regional = selected and ablation.region_assignment_loss
    override = prefixed or regional
    operation = diagnostics["semantic_guidance"][OPERATION_KEY]
    require(operation["schema"] == OPERATION_SCHEMA and operation["plan_sha256"] == digest(plan)
            and operation["condition_id"] == ablation.condition_id
            and operation["components"] == ablation.components() and operation["prefix_applied"] == prefixed
            and (operation["projection_calls"] > 0) == prefixed and operation["loss_override"] == override
            and (operation["loss_evaluations"] > 0) == override
            and operation["region_weight"] == (REGION_WEIGHT if regional else 0.0),
            "executed ablation operation differs from its plan")
    if prefixed:
        require(plan["prefix"]["phantom_anchor"] is None
                and operation["executed_program_sha256"] == digest(plan["prefix"]["mapped_program"]),
                "prefixed row did not execute the mapped program")
    elif override:
        require(operation["executed_program_sha256"] == plan["prefix"]["program_sha256"],
                "region-only row did not keep the source program")
    geometry = operation["geometry_sum"]
    require(math.isfinite(geometry) and geometry >= 0 and (geometry > 0) == regional,
            "region loss evaluation differs from the plan")
    return override


def _checked_batch(ablation: T2IAblation, plans: Mapping[str, CompositionalPlan], pipe: Any, jobs: Sequence[Any],
                   **kwargs: Any) -> Any:
    check_batch(pipe, jobs, kwargs)
    batch_plans = [plans[job.prompt_id.rsplit("_s", 1)[0]] for job in jobs]
    result = run_batch(ablation, pipe, jobs, batch_plans, **kwargs)
    for plan, diagnostic in zip(batch_plans, kwargs["diagnostics"], strict=True):
        require(diagnostic.captured_attention_processor_count == SD14_GENERATION.expected_attention_processor_count,
                "actual attention processor count differs from the paper configuration")
        verify_operation(ablation, plan.as_record(), {"semantic_guidance": diagnostic.semantic_guidance})
    return result


# --- generate ---------------------------------------------------------------------------------------------


def run_request(
    *, run_id: str, ablation: T2IAblation = FULL_T2I, shard_index: int, device: str,
    batches: Sequence[int] | None = None, allow_dirty: bool = False, output_dir: Path | None = None,
    local_files_only: bool = True, strict_runtime: bool = True, repository_root: Path | None = None,
) -> driver.SD14Run:
    """The driver request of one shard of a prepared run (CPU only).

    The plans are derived again from the Stage 1-3 records and must equal the
    decisions file. ``batches`` runs only these physical batches of the shard
    (batch ``c`` of every shard is canonical chunk ``c``); ``output_dir``
    replaces ``<run>/[<condition>/]images``. ``strict_runtime=False`` records
    the software versions without enforcing ``SD14_GENERATOR_RUNTIME_PINS``.
    """
    root = _root(repository_root)
    directory = run_dir(run_id, ablation, repository_root=root)
    protocol_path = directory / "protocol.json"
    protocol = load_protocol(protocol_path, allow_dirty=allow_dirty, repository_root=root)
    require(type(shard_index) is int and 0 <= shard_index < NUM_SHARDS, "invalid shard index")
    inputs = {key: resolve_recorded_path(record["path"], repository_root=root)
              for key, record in protocol["inputs"].items()}
    canonical = read_jsonl(inputs["image_manifest"])
    prompts = validate_image_rows(canonical, num_shards=NUM_SHARDS)
    tokenizer = load_sd14_tokenizer(local_files_only=local_files_only)
    sources, _ = load_source_prompts(inputs["source_manifest"], inputs["image_manifest"].parent / "prompt_only.jsonl",
                                     tokenizer, repository_root=root)
    guidance, count, plans = derive(sources, prompts, tokenizer, ablation)
    require(decision_records(guidance, count, plans) == read_jsonl(inputs["decisions"]),
            "paper plan/guidance rederivation changed")
    selected = protocol["image_ids"]
    out_dir = output_dir if output_dir is not None else directory / "images"
    return driver.SD14Run(
        method=ARM_ID, prompt_manifest=inputs["image_manifest"], out_dir=out_dir,
        dataset=load_t2icb_dataset_provenance(inputs["image_manifest"], repository_root=root),
        target_file=inputs["image_guidance_targets"], num_shards=NUM_SHARDS, shard_index=shard_index,
        expected_total=len(selected) if selected is not None else len(canonical), device=device,
        local_files_only=local_files_only, strict_runtime=strict_runtime,
        resume=run_lock_path(out_dir, ARM_ID, shard_index, NUM_SHARDS).exists(), allow_dirty=allow_dirty,
        extra_component_files=COMPONENT_FILES, image_guidance=guidance, count_guidance=count,
        batch_fn=functools.partial(_checked_batch, ablation, plans),
        selected_image_ids=tuple(selected) if selected is not None else None,
        batches=tuple(batches) if batches is not None else None,
        extra_record={"protocol": _record(protocol_path, root), "configuration_id": CONFIGURATION_ID,
                      "condition_id": ablation.condition_id, "source": protocol["source"],
                      "decisions": protocol["inputs"]["decisions"]})


def generate(
    *, run_id: str, ablation: T2IAblation = FULL_T2I, shard_index: int, device: str,
    batches: Sequence[int] | None = None, allow_dirty: bool = False, output_dir: Path | None = None,
    local_files_only: bool = True, strict_runtime: bool = True,
) -> Path:
    """Generate (or resume) one shard of a prepared run; return the output directory.

    Writes the shard receipt to ``[<condition>/]control/``, or to
    ``<output_dir>/control/`` for a verification subset.
    """
    import torch

    request = run_request(run_id=run_id, ablation=ablation, shard_index=shard_index, device=device,
                          batches=batches, allow_dirty=allow_dirty, output_dir=output_dir,
                          local_files_only=local_files_only, strict_runtime=strict_runtime)
    gpu_uuid = current_gpu_uuid() if torch.cuda.is_available() else None
    request = replace(request, extra_record={**request.extra_record, "gpu_uuid": gpu_uuid})
    started = datetime.now(timezone.utc).isoformat()
    driver.run(request)
    require((current_gpu_uuid() if torch.cuda.is_available() else None) == gpu_uuid,
            "CUDA context GPU changed during generation")
    directory = run_dir(run_id, ablation)
    # The protocol must still hold after the last image was saved; only then is the shard receipt written.
    load_protocol(directory / "protocol.json", allow_dirty=allow_dirty)
    control = (output_dir if output_dir is not None else directory) / "control"
    atomic_write_json(control / f"{request.out_dir.name}.shard_{shard_index}.json", {
        "schema": RECEIPT_SCHEMA, "status": "complete", "source": request.extra_record["source"],
        "gpu_uuid": gpu_uuid, "started_at_utc": started, "ended_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": request.extra_record["protocol"], "output_dir": repo_relative(request.out_dir),
        "shard_index": shard_index, "batches": list(batches) if batches is not None else None})
    return request.out_dir


# --- audit ------------------------------------------------------------------------------------------------


def _finite(value: Any, label: str, *, nonnegative: bool = False) -> float:
    require(type(value) in (int, float) and math.isfinite(value), f"COUNT {label} is not finite numeric evidence")
    require(not nonnegative or value >= 0, f"COUNT {label} is negative")
    return float(value)


def verify_guidance_operation(diagnostics: Mapping[str, Any], expected: Mapping[str, Any]) -> dict[str, Any]:
    """Compare the kernel's recorded route, step sizes, caps, relations and pooling with the route record."""
    semantic = diagnostics.get("semantic_guidance")
    if not isinstance(semantic, Mapping):
        raise ValueError("observed semantic kernel diagnostics are required")
    for field, value in expected.items():
        if semantic.get(field) != value:
            raise ValueError(f"kernel semantic setting differs from frozen input: {field}")
    caps = semantic.get("update_cap_trajectory")
    steps = semantic.get("update_step_size_trajectory")
    if not isinstance(caps, list) or not isinstance(steps, list) or len(caps) != len(steps):
        raise ValueError("observed sample-local step/cap trajectories are required")
    applied_updates = int(diagnostics.get("guidance_updates_applied", 0))
    if applied_updates > 0 and not steps:
        raise ValueError("applied guidance updates have no observed step/cap trajectory")
    expected_cap = float(expected["max_update_ratio"])
    expected_scale = float(expected["scale_factor"])

    def finite_number(value: Any, *, label: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"expected finite number for {label}")
        return float(value)

    for cap in caps:
        if finite_number(cap, label="observed update cap") != expected_cap:
            raise ValueError("kernel applied another track's update cap")
    for step in steps:
        value = finite_number(step, label="observed step size")
        if not expected_scale * math.sqrt(0.5) - 1e-6 <= value <= expected_scale + 1e-6:
            raise ValueError("kernel step size is outside its declared profile schedule")
    if steps and not math.isclose(float(steps[0]), expected_scale, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("first observed update does not use the declared scale factor")
    maximum = finite_number(diagnostics.get("max_applied_update_ratio"), label="applied update ratio")
    if maximum > expected_cap + 1e-6:
        raise ValueError("observed applied update exceeds the per-track cap")

    relation = diagnostics.get("relation_guidance")
    if expected["effective_profile"] == GuidanceRoute.SPATIAL_RELATION:
        if not isinstance(relation, Mapping) or int(relation.get("evaluations", 0)) <= 0:
            raise ValueError("P1 policy did not evaluate the typed relation loss")
        for field, value in expected["relation_parameters"].items():
            if relation.get(field) != value:
                raise ValueError(f"observed P1 parameter differs: {field}")
        if relation.get("relations") != expected["relations"]:
            raise ValueError("observed P1 relation endpoints differ from frozen input")
        trajectory = relation.get("relation_loss_trajectory")
        if not isinstance(trajectory, list) or len(trajectory) != relation["evaluations"]:
            raise ValueError("P1 relation evaluation trajectory is incomplete")
        for value in trajectory:
            finite_number(value, label="P1 relation loss")
    elif relation is not None and int(relation.get("evaluations", 0)):
        raise ValueError("relation guidance executed on a B0/P2 track")

    guards = diagnostics.get("pd_loss_guard_counts", {})
    calls = guards.get("whole_word_pair_loss_calls", 0)
    reductions = guards.get("whole_word_multi_piece_map_reductions", 0)
    if type(calls) is not int or type(reductions) is not int or min(calls, reductions) < 0:
        raise ValueError("invalid observed whole-word operation counters")
    spans = expected["pair_token_spans"]
    multi_piece = bool(spans and any(len(entry["lexical_piece_indices"]) > 1 for entry in spans["spans"]))
    if expected["pooling_applied"]:
        if calls <= 0:
            raise ValueError("P2 pooling was enabled but no pair pooling operation executed")
        if multi_piece and reductions <= 0:
            raise ValueError("the multi-piece P2 case never executed a mean map reduction")
    elif calls or reductions:
        raise ValueError("whole-word pooling executed on an inactive policy/profile")
    return {"effective_profile": expected["effective_profile"], "updates": len(steps),
            "pooling_calls": calls, "multi_piece_reductions": reductions,
            "multi_piece_pooling_observed": bool(multi_piece and reductions > 0)}


def verify_count_operation(  # noqa: C901
    diagnostics: Mapping[str, Any], expected: Mapping[str, Any], row: Mapping[str, Any],
) -> dict[str, int]:
    """Check the Count Loss state of one image against its count record (Eq. 8; gate, trajectory, targets)."""
    count = diagnostics.get("count_guidance")
    require(isinstance(count, Mapping)
            and all(digest(count.get(key)) == digest(value) for key, value in expected.items()),
            "observed COUNT kernel spec differs from its source-derived constraint population")
    state = count.get("state", {})
    enabled = expected["enabled"]
    constraints = expected["decision"]["constraints"]
    identity = f"{row['experiment']}/{row['prompt_id']}/seed{row['seed']}/physical_row{row['physical_batch_position']}"
    require(state.get("identity") == identity
            and state.get("enabled") is enabled
            and state.get("constraints") == constraints
            and type(state.get("nonfinite_failures")) is int
            and state["nonfinite_failures"] == 0
            and state.get("resume") == "whole_image_only; unfinished images restart from initial seed with fresh state",
            "COUNT state identity, finite history or fresh-state resume contract changed")
    for field in ("calls", "evaluations", "active_evaluations", "skipped_evaluations"):
        require(type(state.get(field)) is int and state[field] >= 0, "COUNT state counter is invalid")
    trajectory, targets = state.get("loss_trajectory"), state.get("targets")
    require(isinstance(trajectory, list) and isinstance(targets, list)
            and len(trajectory) == state["evaluations"]
            and state["active_evaluations"] + state["skipped_evaluations"] == state["evaluations"],
            "COUNT evaluation trajectory and counters disagree")
    active = [item for item in trajectory if item.get("status") == "active"]
    require(len(active) == state["active_evaluations"], "COUNT active trajectory is incomplete")
    previous_step = -1
    for index, item in enumerate(trajectory, 1):
        step = item.get("outer_step_index")
        require(type(step) is int and 0 <= step <= 9 and previous_step <= step
                and type(item.get("evaluation")) is int and item["evaluation"] == index
                and item.get("status") in {"active", "weak_or_flat"},
                "COUNT evaluates outside steps 0 through 9 or repeats an evaluation ordinal")
        previous_step = step
        base = _finite(item.get("base_loss"), "base loss")
        added = _finite(item.get("count_loss"), "added loss", nonnegative=True)
        combined = _finite(item.get("combined_loss"), "combined loss")
        require(math.isclose(combined, base + added, rel_tol=2e-5, abs_tol=2e-6),
                "COUNT observed combined loss differs from base plus added density loss")
        evidence = item.get("evidence")
        require(isinstance(evidence, list) and len(evidence) == len(constraints),
                "COUNT gate dropped a counted noun from the all-noun evidence population")
        for noun in evidence:
            mean = _finite(noun.get("mean_attention"), "noun mean attention", nonnegative=True)
            spread = _finite(noun.get("spatial_range"), "noun spatial range", nonnegative=True)
            ready = mean >= MIN_MEAN_ATTENTION and spread >= MIN_SPATIAL_RANGE
            require(noun.get("ready") is ready
                    and math.isclose(_finite(noun.get("mass_factor"), "COUNT detached mass factor"),
                                     min(1.0, 77 * mean), abs_tol=1e-7),
                    "COUNT numerical all-noun gate or detached mass factor changed")
        require((item["status"] == "active") == all(noun["ready"] for noun in evidence),
                "COUNT accepted a partial noun subset or rejected ready evidence")
        if item["status"] == "weak_or_flat":
            require(added == 0 and combined == base, "COUNT weak/flat observation altered the incumbent loss")
    if not enabled or not constraints:
        require(not trajectory and not targets and state.get("first_active_step") is None,
                "COUNT controls or source-ineligible images executed the added loss")
    if active:
        require(state.get("first_active_step") == active[0]["outer_step_index"]
                and state.get("first_active_evaluation") == active[0]["evaluation"]
                and len(targets) == len(constraints),
                "COUNT first activation or frozen target population changed")
    else:
        require(not targets and state.get("first_active_step") is None and state.get("first_active_evaluation") is None,
                "COUNT target exists without active evidence")
    for constraint, target in zip(constraints, targets):
        n = constraint["count"]
        centers, sigmas = target.get("centers"), target.get("sigma")
        require(isinstance(centers, list) and len(centers) == n and len({tuple(p) for p in centers}) == n
                and all(len(p) == 2 and all(type(x) is int and 0 <= x < 16 for x in p) for p in centers)
                and isinstance(sigmas, list) and len(sigmas) == n,
                "COUNT target centers or widths are incomplete")
        density = np.asarray(target.get("density"), dtype=np.float32)
        assignment = np.asarray(target.get("region_assignment"), dtype=np.int64)
        require(density.shape == assignment.shape == (16, 16) and np.isfinite(density).all()
                and (density > 0).all() and abs(float(density.sum()) - 1) < 2e-6
                and int(assignment.min()) >= 0 and int(assignment.max()) < n,
                "COUNT frozen target density or nearest-region assignment is invalid")
        require(sha256_bytes(density.tobytes()) == target.get("sha256")
                and sha256_bytes(assignment.tobytes()) == target.get("region_assignment_sha256"),
                "COUNT saved target or region-assignment bytes changed")
    return {"active_images": int(bool(active)), "active_evaluations": len(active)}


def _metadata_rows(out_dir: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in sorted((out_dir / ARM_ID).glob("*/metadata.shard_*.jsonl")):
        for row in read_jsonl(path):
            require(row["prompt_id"] not in rows, "duplicate generated image")
            rows[row["prompt_id"]] = row
    return rows


def audit(  # noqa: C901
    *, run_id: str, ablation: T2IAblation = FULL_T2I, batches: Sequence[int] | None = None,
    output_dir: Path | None = None, repository_root: Path | None = None,
) -> dict[str, Any]:
    """Check every generated image: placement, numerics, source seal, image file and executed operations.

    ``batches``/``output_dir`` audit a subset written by ``generate`` with the
    same arguments. Writes ``[<condition>/]control/audit_images.json``, or
    ``<output_dir>/control/audit_<output dir name>.json`` for a subset.
    """
    root = _root(repository_root)
    directory = run_dir(run_id, ablation, repository_root=root)
    protocol = read_json_object(directory / "protocol.json")
    require(protocol["condition_id"] == ablation.condition_id, "protocol belongs to another condition")
    out_dir = output_dir if output_dir is not None else directory / "images"
    manifest = resolve_recorded_path(protocol["inputs"]["image_manifest"]["path"], repository_root=root)
    canonical = {row["prompt_id"]: row for row in read_jsonl(manifest)}
    population = protocol["image_ids"] if protocol["image_ids"] is not None else list(canonical)
    decisions = {row["prompt_id"]: row for row in read_jsonl(
        resolve_recorded_path(protocol["inputs"]["decisions"]["path"], repository_root=root))}
    sealed = _sealed_files(protocol["source"])
    images = _metadata_rows(out_dir)
    expected: dict[str, tuple[int, int, int, list[str]]] = {}
    for shard in range(NUM_SHARDS):
        shard_ids = population[shard::NUM_SHARDS]
        indices = range(math.ceil(len(shard_ids) / 4)) if batches is None else batches
        for batch in indices:
            members = shard_ids[4 * batch:4 * batch + 4]
            for slot, key in enumerate(members):
                expected[key] = (shard, batch, slot, members)
    require(set(images) == set(expected), "generated population is incomplete or contains extra images")
    operations: Counter[str] = Counter()
    locks: dict[str, Any] = {}
    for key, (shard, batch, slot, members) in expected.items():
        row, source = images[key], canonical[key]
        require(all(row[field] == source[field] for field in ("prompt", "seed", "experiment")), "prompt/seed differs")
        require(row["shard_index"] == shard and row["physical_batch_position"] == slot
                and row["generation_batch_id"] == f"{shard_suffix(shard, NUM_SHARDS)}:batch_{batch:06d}",
                "physical slot differs from the planned batch")
        peers = sorted(k for k, r in images.items() if r["generation_batch_id"] == row["generation_batch_id"])
        require(peers == sorted(members), "physical batch members differ from the plan")
        require(row["batch_size_effective"] == 4 and row["strict_finite"] is True
                and row["fallback_used"] is False and row["height"] == row["width"] == 512
                and row["steps"] == 50 and row["guidance_scale"] == 7.5
                and row["precision_profile_resolved"] == "bf16_stable"
                and row["scheduler"]["class"] == "PNDMScheduler", "numeric generation contract changed")
        if row["run_lock_path"] not in locks:
            lock_path = resolve_recorded_path(row["run_lock_path"], repository_root=root)
            require(sha256_file(lock_path) == row["run_lock_sha256"], "run lock changed")
            locks[row["run_lock_path"]] = read_json_object(lock_path)["configuration"]
        configuration = locks[row["run_lock_path"]]
        require(configuration["component_sha256"] == sealed, "generation source differs from the protocol seal")
        if "components" in protocol["source"]:
            require(configuration["repo_git_commit"] == protocol["source"]["repo_git_commit"]
                    and configuration["components_match_repo_git_commit"] is True, "generation source changed")
        image = resolve_recorded_path(row["image_path"], repository_root=root)
        require(sha256_file(image) == row["image_sha256"], "generated image changed")
        with Image.open(image) as png:
            require(png.format == "PNG" and png.size == (512, 512), "PNG shape or format changed")
            png.verify()
        decision = decisions[key.rsplit("_s", 1)[0]]
        diagnostics = row["generation_diagnostics"]
        verify_guidance_operation(diagnostics, decision["guidance"])
        verify_count_operation(diagnostics, decision["count"], row)
        override = verify_operation(ablation, decision["plan"], diagnostics)
        if not ablation.spatial_relation_loss:
            require(not (diagnostics.get("relation_guidance") or {}).get("relations"),
                    "a Spatial Relation Loss term ran without the Spatial Relation Loss")
        if not ablation.count_loss:
            require(diagnostics["count_guidance"]["state"]["enabled"] is False, "COUNT ran without the Count Loss")
        operations[("prefix" if override and ablation.positional_prefix else "region_only" if override else "plain")
                   + f"|{decision['guidance']['effective_profile']}"] += 1
    result = {"schema": AUDIT_SCHEMA, "status": "PASS", "run_id": run_id, "condition_id": ablation.condition_id,
              "output_dir": repo_relative(out_dir, repository_root=root),
              "batches": list(batches) if batches is not None else None, "images": len(expected),
              "operations": dict(sorted(operations.items()))}
    control = (output_dir if output_dir is not None else directory) / "control"
    atomic_write_json(control / f"audit_{out_dir.name}.json", result)
    return result
