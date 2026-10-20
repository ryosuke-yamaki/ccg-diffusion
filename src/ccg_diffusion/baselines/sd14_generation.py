"""SD1.4 generation of the four external T2I-CompBench++ baselines and the Attend-and-Excite subtoken selection.

Composable Diffusion, StructureDiffusion, Attend-and-Excite and SynGen run
under the shared SD1.4 contract of ``config.SD14_GENERATION`` (PNDM 50 steps,
CFG 7.5, 512x512, BF16 model with FP32 master latents, scheduler and losses):

* every image row of the generation manifest is joined to exactly one row of
  the arm's input file (``baselines.preprocessing``) before a model is loaded;
* rows run in manifest order in physical batches of 4 (the last real row
  repeats as padding; padded outputs are discarded); guided rows and
  stock-fallback rows (``status != "ok"``) never share a batch;
* ``run`` writes ``<out>/<arm>/run_lock.shard_<i>-of-<n>.json`` and one
  metadata row per image (``metadata_schema_version`` 3), like the six-arm
  run ``20260812_t2icb_sd14_six_arm_full_v2``.

The SD1.4 Predicated Diffusion baseline (``sd14_pd_spacy``) runs through the
SD1.4 batch driver instead; ``driver_plan_projection`` is its CPU plan check.
``select_attend_and_excite_subtokens`` turns the A&E candidate input file into
the final one: for a noun spanning several CLIP subtokens it keeps the subtoken
with the highest mean, over seeds 42-51, of the smoothed step-0 maximum.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import torch
from PIL import Image
from tqdm import tqdm

from ccg_diffusion.backbones.jobs import Job, image_path
from ccg_diffusion.backbones.sd14.driver import (
    SD14_DRIVER_COMPONENT_FILES,
    SD14_PD_BATCH_PADDING_POLICY,
    SD14Run,
    initial_latents,
    plan_run,
    prepare_sd14_runtime,
)
from ccg_diffusion.backbones.sd14.inputs import T2ICBDatasetProvenance, load_t2icb_dataset_provenance
from ccg_diffusion.backbones.sd14.pipeline import decode_latents, encode_prompts, generate_stock_batch, tensor_sha256
from ccg_diffusion.backbones.sd14.records import (
    append_metadata,
    device_record,
    ensure_run_lock,
    image_is_valid,
    load_metadata_records,
    peak_memory,
    reset_peak_memory,
    run_lock_path,
    save_image_atomic,
    sd14_generator_runtime_environment,
    shard_metadata_path,
    shard_suffix,
    synchronize_device,
    validate_clip_token_contract,
)
from ccg_diffusion.baselines import attend_and_excite, composable_diffusion, preprocessing, structure_diffusion, syngen
from ccg_diffusion.baselines.registry import EXTERNAL_ARMS, SOURCE_COMMIT_BY_ARM
from ccg_diffusion.baselines.sample_local_guidance import sample_local_latent_update
from ccg_diffusion.config import SD14_GENERATION
from ccg_diffusion.guidance.attention import (
    ATTENTION_PLACES,
    AttentionStore,
    aggregate_attention_batched,
    register_attention_control,
)
from ccg_diffusion.guidance.latent_update import GenerationDiagnostics
from ccg_diffusion.provenance import (
    SourceSealError,
    artifact_record,
    atomic_write_json,
    atomic_write_jsonl,
    canonical_json,
    git_head,
    git_head_components_record,
    portable_path,
    read_jsonl,
    repo_relative,
    repository_root,
    resolve_recorded_path,
    sha256_file,
)
from ccg_diffusion.provenance import digest as sha256_json
from ccg_diffusion.targets.tokenizer import (
    load_sd14_tokenizer,
    require_clip_bpe_behavior,
    require_clip_tokenizer_identity,
)

# Result-affecting source files sealed into every run lock: the SD1.4 driver closure, the baseline
# modules and the CLI that builds the run.
COMPONENT_FILES = tuple(sorted({
    *SD14_DRIVER_COMPONENT_FILES,
    "src/ccg_diffusion/baselines/attend_and_excite.py",
    "src/ccg_diffusion/baselines/composable_diffusion.py",
    "src/ccg_diffusion/baselines/preprocessing.py",
    "src/ccg_diffusion/baselines/registry.py",
    "src/ccg_diffusion/baselines/sample_local_guidance.py",
    "src/ccg_diffusion/baselines/sd14_generation.py",
    "src/ccg_diffusion/baselines/structure_diffusion.py",
    "src/ccg_diffusion/baselines/syngen.py",
    "scripts/t2i_compbench/run_baselines.py",
}))
# Also sealed into the A&E selection evidence.
SELECTION_COMPONENT_FILES = tuple(sorted({*COMPONENT_FILES, "scripts/t2i_compbench/prepare_baseline_inputs.py"}))
BATCH_PADDING_POLICY = SD14_PD_BATCH_PADDING_POLICY  # "repeat_last_real_job_discard_padded_outputs" (KEEP)
RUN_LOCK_SCHEMA_VERSION = 3  # KEEP
METADATA_SCHEMA_VERSION = 3  # KEEP


class BaselineGenerationError(RuntimeError):
    """The baseline input, its selection evidence or the generation contract is invalid."""


def _artifact(path: Path) -> dict[str, Any]:
    """``{"path", "sha256", "size_bytes"}`` of an existing file (portable path)."""

    record = artifact_record(path).to_dict()
    return {**record, "path": portable_path(path)}


def _read_rows(path: Path, *, label: str) -> list[dict[str, Any]]:
    try:
        rows = read_jsonl(path, reject_duplicate_keys=True)
    except (OSError, ValueError) as exc:
        raise BaselineGenerationError(f"invalid {label}: {path}: {exc}") from exc
    if not rows:
        raise BaselineGenerationError(f"empty {label}: {path}")
    return rows


def _reject_duplicate_json_keys(
    pairs: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    """Reject duplicate keys at every object depth parsed by ``json.loads``."""

    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise BaselineGenerationError(f"duplicate JSON key: {key!r}")
        payload[key] = value
    return payload


def _existing_artifact(value: Any, *, location: str) -> tuple[dict[str, Any], Path]:
    """Check a recorded artifact against the file it names.

    A recorded absolute ``/workspace/holsd/...`` path that does not exist is
    re-rooted at the repository root; sha256 and size are always enforced.
    """

    record = preprocessing.parse_artifact_record(value, location=location)
    path = resolve_recorded_path(record["path"])
    try:
        actual_size = path.stat().st_size
        actual_digest = sha256_file(path)
    except OSError as exc:
        raise BaselineGenerationError(f"cannot read sealed artifact {path}") from exc
    if actual_size != record["size_bytes"] or actual_digest != record["sha256"]:
        raise BaselineGenerationError(
            f"{location} does not seal the referenced artifact: {path}"
        )
    return record, path


def _source_record(component_files: Sequence[str], *, allow_dirty: bool) -> dict[str, Any]:
    """Seal the component files to Git HEAD, or record their working-tree bytes under ``allow_dirty``."""

    if not allow_dirty:
        try:
            record = git_head_components_record(component_files)
        except SourceSealError as exc:
            raise BaselineGenerationError(str(exc)) from exc
        return {
            "repo_git_commit": record["repo_git_commit"],
            "component_sha256": record["component_sha256"],
            "components_match_repo_git_commit": True,
            "dirty": False,
        }
    root = repository_root()
    return {
        "repo_git_commit": git_head(root),
        "component_sha256": {path: sha256_file(root / path) for path in component_files},
        "components_match_repo_git_commit": None,
        "dirty": True,
    }


# --- prepared jobs ---------------------------------------------------------------------------------------


@dataclass
class BaselineDiagnostics(GenerationDiagnostics):
    """Method-specific counters appended to the common generation ledger."""

    guidance_loss_evaluations: int = 0
    first_guidance_loss: float | None = None
    last_guidance_loss: float | None = None
    min_guidance_loss: float | None = None
    max_guidance_loss: float | None = None
    refinement_rounds: int = 0
    method_guidance_updates: int = 0
    expected_attention_processor_count: int = 0
    valid_batch_size: int = 1
    batch_padding_count: int = 0
    _guidance_loss_sum: float = field(default=0.0, repr=False)
    _finite_guidance_loss_count: int = field(default=0, repr=False)

    def record_guidance_loss(self, loss: torch.Tensor) -> None:
        """Record one finite scalar loss without changing its graph."""

        self.guidance_loss_evaluations += 1
        value = float(loss.detach().float().item())
        if not math.isfinite(value):
            return
        self._guidance_loss_sum += value
        self._finite_guidance_loss_count += 1
        if self.first_guidance_loss is None:
            self.first_guidance_loss = value
            self.min_guidance_loss = value
            self.max_guidance_loss = value
        self.last_guidance_loss = value
        self.min_guidance_loss = min(float(self.min_guidance_loss), value)
        self.max_guidance_loss = max(float(self.max_guidance_loss), value)

    def as_dict(self) -> dict[str, Any]:
        payload = super().as_dict()
        payload["mean_guidance_loss"] = (
            self._guidance_loss_sum / self._finite_guidance_loss_count
            if self._finite_guidance_loss_count
            else None
        )
        return payload


@dataclass(frozen=True)
class PreparedJob:
    """One manifest row joined to its input-file row, with the reconstructed method plan."""

    job: Job
    method: str
    status: str
    plan: Any
    prompt_sha256: str
    sidecar_row_sha256: str
    program_sha256: str

    @property
    def guidance_applied(self) -> bool:
        return self.status == preprocessing.GUIDED_STATUS

    @property
    def guidance_mode(self) -> str:
        return "method_guided" if self.guidance_applied else "stock_vanilla_fallback"

    @property
    def actual_conditioning_prompt(self) -> str:
        if self.method == "sd14_structure" and self.guidance_applied:
            if not isinstance(self.plan, structure_diffusion.StructurePlan):
                raise BaselineGenerationError("invalid Structure Diffusion plan type")
            return self.plan.method_prompt
        return self.job.prompt


def _job_from_prompt_row(row: Mapping[str, Any], *, location: str) -> Job:
    preprocessing.require_exact_fields(row, preprocessing.PROMPT_FIELDS, location=location)
    return Job(
        experiment=preprocessing.require_text(row["experiment"], location=f"{location}.experiment"),
        prompt_id=preprocessing.require_text(row["prompt_id"], location=f"{location}.prompt_id"),
        prompt=preprocessing.require_text(row["prompt"], location=f"{location}.prompt"),
        seed=preprocessing.require_integer(row["seed"], location=f"{location}.seed"),
        attention_exist_indices=[],
        attention_corr_indices=[],
        attention_leak_indices=[],
        attention_possession_indices=[],
        source="sd14_external_sidecar",
    )


def _job_key(job: Job) -> tuple[str, str, int]:
    return job.experiment, job.prompt_id, job.seed


# --- input files and the A&E selection evidence ---------------------------------------------------------


def load_selection_evidence(
    metadata: Mapping[str, Any],
    *,
    prompt_manifest: Path,
    blind_prompt_manifest: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Mapping[str, Any]],
    dict[str, Mapping[str, Any]],
    dict[tuple[str, str, int], Mapping[str, Any]],
]:
    """Load and fully recompute the v3 subtoken-selection decisions."""

    metadata_record, path = _existing_artifact(
        metadata.get("selection_evidence"),
        location="sidecar metadata.selection_evidence",
    )
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (OSError, json.JSONDecodeError, BaselineGenerationError) as exc:
        raise BaselineGenerationError(
            f"invalid A&E selection evidence at {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise BaselineGenerationError("A&E selection evidence must be an object")
    required_fields = frozenset(
        {
            "schema",
            "created_at_utc",
            "selection_protocol",
            "method",
            "backbone",
            "model_id",
            "model_revision",
            "precision_profile",
            "pipeline_dtypes",
            "scheduler",
            "selection_denoising_step_index",
            "selection_scheduler_timestep",
            "selection_seeds",
            "attention",
            "tie_break",
            "tokenizer",
            "inputs",
            "runtime",
            "summary",
            "gold_inputs_used",
            "local_implementation",
            "occurrences",
            "no_target_prompts",
        }
    )
    preprocessing.require_exact_fields(payload, required_fields, location="selection evidence")
    expected_scalars = {
        "schema": preprocessing.SELECTION_EVIDENCE_SCHEMA,
        "selection_protocol": attend_and_excite.ENSEMBLE_SELECTION_PROTOCOL,
        "method": "sd14_attend_excite",
        "backbone": "sd14",
        "model_id": SD14_GENERATION.model_id,
        "model_revision": SD14_GENERATION.model_revision,
        "selection_denoising_step_index": 0,
        "selection_seeds": list(attend_and_excite.SELECTION_SEEDS),
        "tie_break": "maximum_10_seed_mean_then_smallest_token_index",
        "tokenizer": SD14_GENERATION.tokenizer_identity(),
        "gold_inputs_used": False,
    }
    mismatches = {
        key: {"expected": expected, "actual": payload.get(key)}
        for key, expected in expected_scalars.items()
        if payload.get(key) != expected
    }
    if mismatches:
        raise BaselineGenerationError(
            f"A&E selection evidence identity mismatch: {mismatches}"
        )
    if not isinstance(payload.get("created_at_utc"), str):
        raise BaselineGenerationError("selection evidence.created_at_utc must be text")
    local_implementation = preprocessing.validate_recorded_local_implementation(
        payload.get("local_implementation"),
        location="selection evidence.local_implementation",
    )
    if metadata.get("local_implementation") != local_implementation:
        raise BaselineGenerationError(
            "selection evidence local implementation differs from sidecar metadata"
        )
    timestep = payload.get("selection_scheduler_timestep")
    if timestep != 981:
        raise BaselineGenerationError(
            "selection evidence must use SD1.4 PNDM step-0 timestep 981"
        )
    precision = payload.get("precision_profile")
    expected_precision = {
        "name": "bf16_stable",
        "model_dtype": "bfloat16",
        "master_latent_dtype": "float32",
        "scheduler_dtype": "float32",
    }
    if not isinstance(precision, dict) or any(
        precision.get(key) != expected
        for key, expected in expected_precision.items()
    ):
        raise BaselineGenerationError(
            "A&E selection evidence precision is not BF16/FP32 stable"
        )
    scheduler = payload.get("scheduler")
    if not isinstance(scheduler, dict) or any(
        scheduler.get(key) != expected
        for key, expected in {
            "class": "PNDMScheduler",
            "prediction_type": SD14_GENERATION.prediction_type,
        }.items()
    ):
        raise BaselineGenerationError("A&E selection evidence scheduler is not SD1.4 PNDM")
    scheduler_digest = scheduler.get("config_sha256") if isinstance(scheduler, dict) else None
    if not isinstance(scheduler_digest, str) or len(scheduler_digest) != 64:
        raise BaselineGenerationError("A&E selection scheduler config is not sealed")
    dtypes = payload.get("pipeline_dtypes")
    if not isinstance(dtypes, dict):
        raise BaselineGenerationError("selection evidence.pipeline_dtypes must be an object")
    for component in ("unet", "text_encoder", "vae"):
        component_record = dtypes.get(component)
        if not isinstance(component_record, dict) or component_record.get(
            "parameter_dtypes"
        ) != ["bfloat16"]:
            raise BaselineGenerationError(
                f"selection evidence {component} is not exclusively BF16"
            )
        if component_record.get("requires_grad") is not False:
            raise BaselineGenerationError(
                f"selection evidence {component} was not frozen"
            )
    attention = payload.get("attention")
    if not isinstance(attention, dict):
        raise BaselineGenerationError("selection evidence.attention must be an object")
    expected_attention = {
        "resolution": SD14_GENERATION.attention_res,
        "from_where": list(ATTENTION_PLACES),
        "text_attention_transform": "softmax(raw_attention[:, :, 1:-1] * 100)",
        "smoothing": {
            "kernel": "official Attend-and-Excite GaussianSmoothing",
            "kernel_size": 3,
            "sigma": 0.5,
        },
    }
    if any(attention.get(key) != expected for key, expected in expected_attention.items()):
        raise BaselineGenerationError("A&E selection attention contract mismatch")
    processors = attention.get("captured_attention_processors")
    expected_processors = [
        "down_blocks.2.attentions.0.transformer_blocks.0.attn2.processor",
        "down_blocks.2.attentions.1.transformer_blocks.0.attn2.processor",
        "up_blocks.1.attentions.0.transformer_blocks.0.attn2.processor",
        "up_blocks.1.attentions.1.transformer_blocks.0.attn2.processor",
        "up_blocks.1.attentions.2.transformer_blocks.0.attn2.processor",
    ]
    if processors not in (expected_processors, []):
        raise BaselineGenerationError(
            "A&E selection evidence has an invalid processor population"
        )
    if attention.get("captured_attention_processor_count") != len(processors):
        raise BaselineGenerationError(
            "A&E selection processor count differs from its identities"
        )
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict):
        raise BaselineGenerationError("selection evidence.inputs must be an object")
    preprocessing.require_exact_fields(
        inputs,
        frozenset(
            {
                "generation_manifest",
                "blind_prompt_manifest",
                "candidate_sidecar",
                "candidate_sidecar_metadata",
                "candidate_metadata_identity_sha256",
                "gold_inputs_used",
            }
        ),
        location="selection evidence.inputs",
    )
    if inputs.get("gold_inputs_used") is not False:
        raise BaselineGenerationError("selection evidence inputs used gold data")
    generation_record, _generation_path = _existing_artifact(
        inputs["generation_manifest"],
        location="selection evidence.inputs.generation_manifest",
    )
    blind_record, _blind_path = _existing_artifact(
        inputs["blind_prompt_manifest"],
        location="selection evidence.inputs.blind_prompt_manifest",
    )
    if generation_record["sha256"] != sha256_file(prompt_manifest):
        raise BaselineGenerationError("selection evidence generation manifest differs")
    if blind_record["sha256"] != sha256_file(blind_prompt_manifest):
        raise BaselineGenerationError("selection evidence blind manifest differs")
    evidence_candidate_record, _candidate_path = _existing_artifact(
        inputs["candidate_sidecar"],
        location="selection evidence.inputs.candidate_sidecar",
    )
    evidence_candidate_metadata_record, _candidate_metadata_path = (
        _existing_artifact(
            inputs["candidate_sidecar_metadata"],
            location="selection evidence.inputs.candidate_sidecar_metadata",
        )
    )
    if evidence_candidate_record != preprocessing.parse_artifact_record(
        metadata.get("candidate_sidecar"),
        location="sidecar metadata.candidate_sidecar",
    ):
        raise BaselineGenerationError("selection evidence candidate sidecar differs")
    if evidence_candidate_metadata_record != preprocessing.parse_artifact_record(
        metadata.get("candidate_sidecar_metadata"),
        location="sidecar metadata.candidate_sidecar_metadata",
    ):
        raise BaselineGenerationError(
            "selection evidence candidate metadata differs"
        )
    candidate_rows = _read_rows(
        _candidate_path,
        label="A&E candidate sidecar referenced by selection evidence",
    )
    candidate_row_by_key: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for row_number, candidate_row in enumerate(candidate_rows, start=1):
        candidate_location = f"{_candidate_path}:{row_number}"
        preprocessing.require_exact_fields(
            candidate_row,
            preprocessing.INPUT_ROW_FIELDS,
            location=candidate_location,
        )
        key = (
            preprocessing.require_text(
                candidate_row["experiment"],
                location=f"{candidate_location}.experiment",
            ),
            preprocessing.require_text(
                candidate_row["prompt_id"],
                location=f"{candidate_location}.prompt_id",
            ),
            preprocessing.require_integer(
                candidate_row["seed"],
                location=f"{candidate_location}.seed",
            ),
        )
        if key in candidate_row_by_key:
            raise BaselineGenerationError(
                f"duplicate referenced candidate identity: {key}"
            )
        candidate_row_by_key[key] = candidate_row
    identity_digest = inputs.get("candidate_metadata_identity_sha256")
    if (
        not isinstance(identity_digest, str)
        or len(identity_digest) != 64
        or any(character not in "0123456789abcdef" for character in identity_digest)
    ):
        raise BaselineGenerationError(
            "selection evidence candidate metadata identity is invalid"
        )
    try:
        candidate_metadata_payload = json.loads(
            _candidate_metadata_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (OSError, json.JSONDecodeError, BaselineGenerationError) as exc:
        raise BaselineGenerationError(
            "cannot parse selection evidence candidate metadata"
        ) from exc
    if sha256_json(candidate_metadata_payload) != identity_digest:
        raise BaselineGenerationError(
            "selection evidence candidate metadata identity does not recompute"
        )
    runtime = payload.get("runtime")
    if not isinstance(runtime, dict):
        raise BaselineGenerationError("selection evidence.runtime must be an object")
    preprocessing.require_exact_fields(
        runtime,
        frozenset({"device", "determinism", "physical_batch_size"}),
        location="selection evidence.runtime",
    )
    if runtime.get("physical_batch_size") != SD14_GENERATION.batch_size:
        raise BaselineGenerationError("selection evidence did not use physical batch 4")
    occurrences = preprocessing.require_list(
        payload["occurrences"], location="selection evidence.occurrences"
    )
    if occurrences and processors != expected_processors:
        raise BaselineGenerationError(
            "A&E selection occurrences require the canonical five processors"
        )
    occurrence_by_id: dict[str, Mapping[str, Any]] = {}
    occurrence_prompt_hashes: set[str] = set()
    occurrence_count_by_prompt: dict[str, int] = {}
    latent_hash_by_seed: dict[int, str] = {}
    for occurrence_index, raw in enumerate(occurrences):
        location = f"selection evidence.occurrences[{occurrence_index}]"
        if not isinstance(raw, dict):
            raise BaselineGenerationError(f"{location} must be an object")
        preprocessing.require_exact_fields(
            raw,
            frozenset(
                {
                    "occurrence_id",
                    "experiment",
                    "prompt",
                    "prompt_sha256",
                    "text",
                    "character_span",
                    "candidate_token_indices",
                    "seed_scores",
                    "candidate_mean_smoothed_max_attention",
                    "selected_token_index",
                }
            ),
            location=location,
        )
        occurrence_id = preprocessing.require_text(
            raw["occurrence_id"], location=f"{location}.occurrence_id"
        )
        preprocessing.require_text(raw["experiment"], location=f"{location}.experiment")
        if occurrence_id in occurrence_by_id:
            raise BaselineGenerationError(f"duplicate evidence occurrence: {occurrence_id}")
        prompt = preprocessing.require_text(raw["prompt"], location=f"{location}.prompt")
        prompt_digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if raw["prompt_sha256"] != prompt_digest:
            raise BaselineGenerationError(f"{location}.prompt_sha256 mismatch")
        candidate_indices = preprocessing.parse_indices(
            raw["candidate_token_indices"],
            location=f"{location}.candidate_token_indices",
        )
        seed_scores = preprocessing.require_list(raw["seed_scores"], location=f"{location}.seed_scores")
        if len(seed_scores) != len(attend_and_excite.SELECTION_SEEDS):
            raise BaselineGenerationError(f"{location} does not contain ten seed scores")
        values_by_candidate: list[list[float]] = [list() for _ in candidate_indices]
        for position, seed_score in enumerate(seed_scores):
            seed_location = f"{location}.seed_scores[{position}]"
            if not isinstance(seed_score, dict):
                raise BaselineGenerationError(f"{seed_location} must be an object")
            preprocessing.require_exact_fields(
                seed_score,
                frozenset(
                    {
                        "seed",
                        "initial_latent_sha256",
                        "candidate_smoothed_max_attention",
                    }
                ),
                location=seed_location,
            )
            seed = seed_score["seed"]
            if seed != attend_and_excite.SELECTION_SEEDS[position]:
                raise BaselineGenerationError(f"{seed_location}.seed ordering mismatch")
            latent_digest = seed_score["initial_latent_sha256"]
            if (
                not isinstance(latent_digest, str)
                or len(latent_digest) != 64
                or any(character not in "0123456789abcdef" for character in latent_digest)
            ):
                raise BaselineGenerationError(f"{seed_location} latent hash is invalid")
            previous_digest = latent_hash_by_seed.setdefault(seed, latent_digest)
            if previous_digest != latent_digest:
                raise BaselineGenerationError(
                    f"seed {seed} has inconsistent initial latent hashes"
                )
            raw_values = preprocessing.require_list(
                seed_score["candidate_smoothed_max_attention"],
                location=f"{seed_location}.candidate_smoothed_max_attention",
            )
            if len(raw_values) != len(candidate_indices):
                raise BaselineGenerationError(f"{seed_location} score width mismatch")
            for candidate_position, value in enumerate(raw_values):
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise BaselineGenerationError(f"{seed_location} scores must be numeric")
                numeric = float(value)
                if not math.isfinite(numeric):
                    raise BaselineGenerationError(f"{seed_location} scores must be finite")
                values_by_candidate[candidate_position].append(numeric)
        recomputed_means = [statistics.fmean(values) for values in values_by_candidate]
        stored_means = preprocessing.require_list(
            raw["candidate_mean_smoothed_max_attention"],
            location=f"{location}.candidate_mean_smoothed_max_attention",
        )
        if stored_means != recomputed_means:
            raise BaselineGenerationError(f"{location} candidate means do not recompute")
        selected = candidate_indices[
            min(
                range(len(candidate_indices)),
                key=lambda index: (-recomputed_means[index], candidate_indices[index]),
            )
        ]
        if raw["selected_token_index"] != selected:
            raise BaselineGenerationError(f"{location} winner/tie-break mismatch")
        occurrence_by_id[occurrence_id] = raw
        occurrence_prompt_hashes.add(prompt_digest)
        occurrence_count_by_prompt[prompt] = occurrence_count_by_prompt.get(prompt, 0) + 1

    no_target_rows = preprocessing.require_list(
        payload["no_target_prompts"],
        location="selection evidence.no_target_prompts",
    )
    no_target_by_prompt_sha256: dict[str, Mapping[str, Any]] = {}
    no_target_ids: set[str] = set()
    for no_target_index, raw in enumerate(no_target_rows):
        location = f"selection evidence.no_target_prompts[{no_target_index}]"
        if not isinstance(raw, dict):
            raise BaselineGenerationError(f"{location} must be an object")
        preprocessing.require_exact_fields(
            raw,
            frozenset(
                {
                    "no_target_id",
                    "experiment",
                    "prompt",
                    "prompt_sha256",
                    "status",
                    "candidate_program_sha256",
                    "seed_rows",
                }
            ),
            location=location,
        )
        prompt = preprocessing.require_text(raw["prompt"], location=f"{location}.prompt")
        prompt_digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if raw["prompt_sha256"] != prompt_digest:
            raise BaselineGenerationError(f"{location}.prompt_sha256 mismatch")
        expected_no_target_id = f"ae_no_target_{prompt_digest}"
        if raw["no_target_id"] != expected_no_target_id:
            raise BaselineGenerationError(f"{location}.no_target_id mismatch")
        if prompt_digest in occurrence_prompt_hashes:
            raise BaselineGenerationError(
                f"{location} is also represented by guided occurrences"
            )
        if prompt_digest in no_target_by_prompt_sha256:
            raise BaselineGenerationError(f"duplicate no-target prompt: {prompt!r}")
        if raw["status"] != "no_noun_target":
            raise BaselineGenerationError(f"{location}.status mismatch")
        program_digest = raw["candidate_program_sha256"]
        if (
            not isinstance(program_digest, str)
            or len(program_digest) != 64
            or any(character not in "0123456789abcdef" for character in program_digest)
        ):
            raise BaselineGenerationError(
                f"{location}.candidate_program_sha256 is invalid"
            )
        seed_rows = preprocessing.require_list(raw["seed_rows"], location=f"{location}.seed_rows")
        if len(seed_rows) != len(attend_and_excite.SELECTION_SEEDS):
            raise BaselineGenerationError(f"{location} must seal ten fallback rows")
        for position, seed_row in enumerate(seed_rows):
            seed_location = f"{location}.seed_rows[{position}]"
            if not isinstance(seed_row, dict):
                raise BaselineGenerationError(f"{seed_location} must be an object")
            preprocessing.require_exact_fields(
                seed_row,
                frozenset({"seed", "prompt_id", "candidate_sidecar_row_sha256"}),
                location=seed_location,
            )
            if seed_row["seed"] != attend_and_excite.SELECTION_SEEDS[position]:
                raise BaselineGenerationError(f"{seed_location}.seed ordering mismatch")
            preprocessing.require_text(seed_row["prompt_id"], location=f"{seed_location}.prompt_id")
            row_digest = seed_row["candidate_sidecar_row_sha256"]
            if (
                not isinstance(row_digest, str)
                or len(row_digest) != 64
                or any(character not in "0123456789abcdef" for character in row_digest)
            ):
                raise BaselineGenerationError(f"{seed_location} row hash is invalid")
        no_target_by_prompt_sha256[prompt_digest] = raw
        no_target_ids.add(expected_no_target_id)
        occurrence_count_by_prompt[prompt] = 0
    summary = payload.get("summary")
    expected_summary_fields = frozenset(
        {
            "unique_prompt_count",
            "guided_prompt_count",
            "no_target_prompt_count",
            "occurrence_count",
            "single_subtoken_occurrence_count",
            "multi_subtoken_occurrence_count",
            "changed_from_candidate_final_subtoken_count",
            "guided_row_count",
            "stock_fallback_row_count",
            "prompt_occurrence_count_sha256",
        }
    )
    if not isinstance(summary, dict):
        raise BaselineGenerationError("A&E selection evidence summary census mismatch")
    preprocessing.require_exact_fields(summary, expected_summary_fields, location="selection evidence.summary")
    single_count = sum(
        len(raw["candidate_token_indices"]) == 1 for raw in occurrences
    )
    multi_count = len(occurrences) - single_count
    changed_count = sum(
        raw["selected_token_index"] != raw["candidate_token_indices"][-1]
        for raw in occurrences
    )
    expected_summary = {
        "unique_prompt_count": len(occurrence_prompt_hashes) + len(no_target_rows),
        "guided_prompt_count": len(occurrence_prompt_hashes),
        "no_target_prompt_count": len(no_target_rows),
        "occurrence_count": len(occurrences),
        "single_subtoken_occurrence_count": single_count,
        "multi_subtoken_occurrence_count": multi_count,
        "changed_from_candidate_final_subtoken_count": changed_count,
        "guided_row_count": (
            len(occurrence_prompt_hashes) * len(attend_and_excite.SELECTION_SEEDS)
        ),
        "stock_fallback_row_count": (
            len(no_target_rows) * len(attend_and_excite.SELECTION_SEEDS)
        ),
        "prompt_occurrence_count_sha256": sha256_json(occurrence_count_by_prompt),
    }
    if summary != expected_summary:
        raise BaselineGenerationError(
            f"A&E selection evidence summary mismatch: "
            f"expected={expected_summary}, actual={summary}"
        )
    if metadata.get("selection_summary") != summary:
        raise BaselineGenerationError(
            "A&E sidecar metadata selection summary differs from evidence"
        )
    return (
        metadata_record,
        occurrence_by_id,
        no_target_by_prompt_sha256,
        candidate_row_by_key,
    )


def _validate_plan_evidence(
    raw_plan: Mapping[str, Any],
    plan: attend_and_excite.AttendAndExcitePlan,
    *,
    experiment: str,
    prompt: str,
    metadata_evidence_record: Mapping[str, Any],
    occurrence_by_id: Mapping[str, Mapping[str, Any]],
    no_target_by_prompt_sha256: Mapping[str, Mapping[str, Any]],
    location: str,
) -> tuple[set[str], set[str]]:
    plan_record = preprocessing.parse_artifact_record(
        raw_plan.get("selection_evidence"),
        location=f"{location}.selection_evidence",
    )
    if plan_record != metadata_evidence_record:
        raise BaselineGenerationError(
            f"{location}.selection_evidence differs from metadata"
        )
    raw_targets = preprocessing.require_list(raw_plan["targets"], location=f"{location}.targets")
    prompt_digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    if not plan.targets:
        no_target = no_target_by_prompt_sha256.get(prompt_digest)
        expected = {
            "experiment": experiment,
            "prompt": prompt,
            "prompt_sha256": prompt_digest,
            "status": "no_noun_target",
            "no_target_id": f"ae_no_target_{prompt_digest}",
        }
        if no_target is None:
            raise BaselineGenerationError(
                f"{location} stock fallback lacks matching no-target evidence"
            )
        preprocessing.require_text(no_target["experiment"], location=f"{location}.experiment")
        if any(
            no_target.get(key) != value for key, value in expected.items()
        ):
            raise BaselineGenerationError(
                f"{location} stock fallback lacks matching no-target evidence"
            )
        if raw_targets or raw_plan["indices_to_alter"]:
            raise BaselineGenerationError(
                f"{location} stock fallback must have empty targets and indices"
            )
        return set(), {str(no_target["no_target_id"])}
    if prompt_digest in no_target_by_prompt_sha256:
        raise BaselineGenerationError(
            f"{location} guided plan is also marked as a no-target fallback"
        )
    referenced: set[str] = set()
    for target_index, (raw_target, target) in enumerate(zip(raw_targets, plan.targets)):
        target_location = f"{location}.targets[{target_index}]"
        occurrence_id = raw_target["selection_evidence_occurrence_id"]
        expected_occurrence_id = attend_and_excite.occurrence_id(
            prompt,
            occurrence_index=target_index,
            text=target.text,
            character_span=(target.char_start, target.char_end),
            candidate_token_indices=target.clip_indices,
        )
        if occurrence_id != expected_occurrence_id:
            raise BaselineGenerationError(
                f"{target_location} occurrence identity does not recompute"
            )
        if occurrence_id in referenced:
            raise BaselineGenerationError(
                f"duplicate occurrence reference in one prompt: {occurrence_id}"
            )
        evidence = occurrence_by_id.get(occurrence_id)
        if evidence is None:
            raise BaselineGenerationError(
                f"{target_location} references absent selection evidence"
            )
        expected = {
            "experiment": experiment,
            "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "text": target.text,
            "character_span": [target.char_start, target.char_end],
            "candidate_token_indices": list(target.clip_indices),
            "selected_token_index": target.selected_clip_index,
        }
        mismatches = {
            key: {"expected": value, "actual": evidence.get(key)}
            for key, value in expected.items()
            if evidence.get(key) != value
        }
        if mismatches:
            raise BaselineGenerationError(
                f"{target_location} differs from selection evidence: {mismatches}"
            )
        referenced.add(occurrence_id)
    return referenced, set()


def load_jobs(
    prompt_manifest: Path,
    blind_prompt_manifest: Path,
    input_file: Path,
    *,
    arm: str,
    stage: Literal["candidate", "final"] = "final",
    strict_environment: bool = True,
) -> tuple[list[PreparedJob], dict[str, Any]]:
    """Join the arm's input file to every row of the generation manifest; return the jobs and its metadata.

    Every row's plan is rebuilt and validated (``preprocessing.parse_plan``);
    the final A&E input is also checked against its selection evidence.
    """

    if arm not in EXTERNAL_ARMS:
        raise BaselineGenerationError(f"unsupported external method: {arm!r}")
    if stage not in {"candidate", "final"}:
        raise BaselineGenerationError(
            f"unsupported sidecar stage: {stage!r}"
        )
    if stage == "candidate" and arm != "sd14_attend_excite":
        raise BaselineGenerationError("candidate sidecar stage is specific to A&E")
    prompt_rows = _read_rows(prompt_manifest, label="prompt manifest")
    blind_rows = _read_rows(blind_prompt_manifest, label="blind prompt manifest")
    preprocessing.validate_blind_generation_manifests(blind_rows, prompt_rows)
    input_rows = _read_rows(input_file, label="external sidecar")
    metadata = preprocessing.load_input_metadata(
        input_file,
        prompt_manifest=prompt_manifest,
        blind_prompt_manifest=blind_prompt_manifest,
        arm=arm,
        row_count=len(input_rows),
        stage=stage,
        strict_environment=strict_environment,
    )
    evidence_record: dict[str, Any] | None = None
    occurrence_by_id: dict[str, Mapping[str, Any]] = {}
    no_target_by_prompt_sha256: dict[str, Mapping[str, Any]] = {}
    candidate_row_by_key: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    if arm == "sd14_attend_excite" and stage == "final":
        (
            evidence_record,
            occurrence_by_id,
            no_target_by_prompt_sha256,
            candidate_row_by_key,
        ) = load_selection_evidence(
            metadata,
            prompt_manifest=prompt_manifest,
            blind_prompt_manifest=blind_prompt_manifest,
        )

    manifest_jobs = []
    manifest_by_key: dict[tuple[str, str, int], Job] = {}
    prompt_ids: set[str] = set()
    for row_number, row in enumerate(prompt_rows, start=1):
        job = _job_from_prompt_row(row, location=f"{prompt_manifest}:{row_number}")
        key = _job_key(job)
        if key in manifest_by_key or job.prompt_id in prompt_ids:
            raise BaselineGenerationError(f"duplicate prompt identity in manifest: {key}")
        manifest_by_key[key] = job
        prompt_ids.add(job.prompt_id)
        manifest_jobs.append(job)

    input_row_by_key: dict[tuple[str, str, int], tuple[dict[str, Any], int]] = {}
    for row_number, row in enumerate(input_rows, start=1):
        location = f"{input_file}:{row_number}"
        preprocessing.require_exact_fields(row, preprocessing.INPUT_ROW_FIELDS, location=location)
        key = (
            preprocessing.require_text(row["experiment"], location=f"{location}.experiment"),
            preprocessing.require_text(row["prompt_id"], location=f"{location}.prompt_id"),
            preprocessing.require_integer(row["seed"], location=f"{location}.seed"),
        )
        if key in input_row_by_key:
            raise BaselineGenerationError(f"duplicate sidecar identity: {key}")
        input_row_by_key[key] = (row, row_number)

    manifest_keys = set(manifest_by_key)
    input_keys = set(input_row_by_key)
    if manifest_keys != input_keys:
        raise BaselineGenerationError(
            "manifest/sidecar identity mismatch: "
            f"missing={sorted(manifest_keys - input_keys)[:5]}, "
            f"extra={sorted(input_keys - manifest_keys)[:5]}"
        )

    prepared = []
    program_by_prompt: dict[str, str] = {}
    referenced_occurrence_ids: set[str] = set()
    referenced_no_target_ids: set[str] = set()
    for job in manifest_jobs:
        key = _job_key(job)
        row, row_number = input_row_by_key[key]
        location = f"{input_file}:{row_number}"
        if row["prompt"] != job.prompt:
            raise BaselineGenerationError(f"sidecar prompt differs at {location}")
        row_method = preprocessing.require_text(row["method"], location=f"{location}.method")
        if row_method != arm:
            raise BaselineGenerationError(
                f"sidecar method differs at {location}: {row_method!r}"
            )
        source_commit = preprocessing.require_text(
            row["source_commit"], location=f"{location}.source_commit"
        )
        if source_commit != SOURCE_COMMIT_BY_ARM[arm]:
            raise BaselineGenerationError(f"sidecar source commit differs at {location}")
        status = preprocessing.require_text(row["status"], location=f"{location}.status")
        if status not in preprocessing.ROW_STATUSES[arm]:
            raise BaselineGenerationError(
                f"unsupported {arm} status at {location}: {status!r}"
            )
        raw_plan = row["plan"]
        if not isinstance(raw_plan, dict):
            raise BaselineGenerationError(f"{location}.plan must be an object")
        plan = preprocessing.parse_plan(
            raw_plan,
            arm=arm,
            prompt=job.prompt,
            location=f"{location}.plan",
            stage=stage,
        )
        if arm == "sd14_attend_excite" and stage == "final":
            if evidence_record is None:
                raise AssertionError("A&E final stage omitted selection evidence")
            occurrence_ids, no_target_ids = _validate_plan_evidence(
                raw_plan,
                plan,
                experiment=job.experiment,
                prompt=job.prompt,
                metadata_evidence_record=evidence_record,
                occurrence_by_id=occurrence_by_id,
                no_target_by_prompt_sha256=no_target_by_prompt_sha256,
                location=f"{location}.plan",
            )
            referenced_occurrence_ids.update(occurrence_ids)
            referenced_no_target_ids.update(no_target_ids)
            if not plan.targets:
                no_target = no_target_by_prompt_sha256[
                    hashlib.sha256(job.prompt.encode("utf-8")).hexdigest()
                ]
                matching_seed_rows = [
                    value
                    for value in no_target["seed_rows"]
                    if value["seed"] == job.seed
                ]
                if len(matching_seed_rows) != 1:
                    raise BaselineGenerationError(
                        f"{location} no-target evidence omitted seed row"
                    )
                seed_evidence = matching_seed_rows[0]
                candidate_key = (job.experiment, job.prompt_id, job.seed)
                candidate_row = candidate_row_by_key.get(candidate_key)
                if candidate_row is None:
                    raise BaselineGenerationError(
                        f"{location} no-target candidate row is absent"
                    )
                if (
                    seed_evidence["prompt_id"] != job.prompt_id
                    or seed_evidence["candidate_sidecar_row_sha256"]
                    != sha256_json(candidate_row)
                    or candidate_row.get("prompt") != job.prompt
                    or candidate_row.get("status") != "no_target"
                ):
                    raise BaselineGenerationError(
                        f"{location} no-target provenance differs from candidate"
                    )
        preprocessing.validate_row_status(arm, status=status, plan=plan, location=location)
        program_sha256 = sha256_json(raw_plan)
        previous_program = program_by_prompt.setdefault(job.prompt, program_sha256)
        if previous_program != program_sha256:
            raise BaselineGenerationError(
                f"seed-expanded sidecar program drift for prompt {job.prompt!r}"
            )
        prepared.append(
            PreparedJob(
                job=job,
                method=arm,
                status=status,
                plan=plan,
                prompt_sha256=hashlib.sha256(job.prompt.encode("utf-8")).hexdigest(),
                sidecar_row_sha256=sha256_json(row),
                program_sha256=program_sha256,
            )
        )
    if arm == "sd14_attend_excite" and stage == "final":
        evidence_ids = set(occurrence_by_id)
        if referenced_occurrence_ids != evidence_ids:
            raise BaselineGenerationError(
                "A&E final sidecar/evidence occurrence population mismatch: "
                f"missing={sorted(evidence_ids - referenced_occurrence_ids)[:5]}, "
                f"extra={sorted(referenced_occurrence_ids - evidence_ids)[:5]}"
            )
        no_target_ids = {
            str(value["no_target_id"])
            for value in no_target_by_prompt_sha256.values()
        }
        if referenced_no_target_ids != no_target_ids:
            raise BaselineGenerationError(
                "A&E final sidecar/evidence no-target population mismatch: "
                f"missing={sorted(no_target_ids - referenced_no_target_ids)[:5]}, "
                f"extra={sorted(referenced_no_target_ids - no_target_ids)[:5]}"
            )
    return prepared, metadata


# --- physical batches and generation kernels -------------------------------------------------------------


def plan_batches(
    jobs: Sequence[PreparedJob], batch_size: int
) -> list[list[PreparedJob]]:
    """Plan stable manifest-order batches without mixing guided/fallback rows."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise BaselineGenerationError("batch_size must be a positive integer")
    batches = []
    cursor = 0
    while cursor < len(jobs):
        first = jobs[cursor]
        batch = []
        while (
            cursor + len(batch) < len(jobs)
            and len(batch) < batch_size
            and jobs[cursor + len(batch)].guidance_applied == first.guidance_applied
        ):
            batch.append(jobs[cursor + len(batch)])
        batches.append(batch)
        cursor += len(batch)
    return batches


def pad_batch(
    batch: Sequence[PreparedJob], batch_size: int
) -> tuple[list[PreparedJob], int]:
    """Pad one uniform execution batch without adding observable samples.

    Guided and stock-fallback rows cannot share a call because their kernels
    differ.  A status boundary may therefore leave fewer than four real rows.
    Repeating the last row is numerically sample-local for SD1.4 and keeps the
    actual tensor batch at the preregistered physical size; padded outputs are
    discarded by the caller.
    """

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise BaselineGenerationError("batch_size must be a positive integer")
    if not batch:
        raise BaselineGenerationError("cannot pad an empty batch")
    if len(batch) > batch_size:
        raise BaselineGenerationError(
            f"batch exceeds physical batch size: {len(batch)} > {batch_size}"
        )
    if len({prepared.guidance_applied for prepared in batch}) != 1:
        raise BaselineGenerationError("cannot pad a mixed guided/fallback batch")
    padding_count = batch_size - len(batch)
    return [*batch, *([batch[-1]] * padding_count)], padding_count


def _initial_latent_batch(
    pipe: Any,
    prepared_jobs: Sequence[PreparedJob],
    *,
    device: str,
    height: int,
    width: int,
    steps: int,
) -> tuple[torch.Tensor, list[BaselineDiagnostics]]:
    tracks = []
    diagnostics = []
    for prepared in prepared_jobs:
        _, scaled, digest = initial_latents(
            pipe,
            prepared.job,
            device=device,
            dtype=torch.float32,
            height=height,
            width=width,
            steps=steps,
        )
        tracks.append(scaled)
        diagnostics.append(
            BaselineDiagnostics(initial_latent_sha256=digest)
        )
    return torch.cat(tracks), diagnostics


def _finish_diagnostics(
    diagnostics: Sequence[BaselineDiagnostics],
    latents: torch.Tensor,
    *,
    kernel: str,
    strict_finite: bool,
) -> None:
    for index, record in enumerate(diagnostics):
        record.generation_kernel = kernel
        record.strict_finite_applied = strict_finite
        record.effective_batch_size = len(diagnostics)
        record.final_latent_sha256 = tensor_sha256(latents[index:index + 1])


def _strict_finite(tensor: torch.Tensor, *, label: str) -> None:
    if not bool(torch.isfinite(tensor.float()).all().item()):
        raise BaselineGenerationError(f"{label} contains NaN or Inf")


def _encode_texts(pipe: Any, texts: Sequence[str], device: torch.device) -> torch.Tensor:
    conditioning = encode_prompts(
        pipe,
        list(texts),
        device=device,
    )
    return conditioning.conditional_prompt_embeds


def generate_composable_batch(
    pipe: Any,
    prepared_jobs: Sequence[PreparedJob],
    *,
    device: str,
    initial_latents_batch: torch.Tensor,
    diagnostics: Sequence[BaselineDiagnostics],
    strict_finite: bool = True,
) -> list[Image.Image]:
    """Run the official ordered score composition on a physical batch."""

    if not prepared_jobs or any(not job.guidance_applied for job in prepared_jobs):
        raise BaselineGenerationError("Composable Diffusion requires guided jobs")
    plans = []
    for prepared in prepared_jobs:
        if not isinstance(prepared.plan, composable_diffusion.ComposablePlan):
            raise BaselineGenerationError("invalid Composable Diffusion plan")
        plans.append(prepared.plan)
    batch = composable_diffusion.pack_plans(plans, device=device)
    device_object = torch.device(device)
    unconditional = _encode_texts(pipe, [""] * len(plans), device_object)
    component_embeddings = [
        _encode_texts(pipe, list(texts), device_object)
        for texts in batch.texts_by_slot
    ]
    pipe.scheduler.set_timesteps(SD14_GENERATION.steps, device=device)
    latents = initial_latents_batch.detach().clone().to(device_object, dtype=torch.float32)

    with torch.no_grad():
        for timestep in pipe.scheduler.timesteps:
            latent_input = pipe.scheduler.scale_model_input(latents, timestep).to(
                dtype=unconditional.dtype
            )
            unconditional_prediction = pipe.unet(
                latent_input,
                timestep,
                encoder_hidden_states=unconditional,
            ).sample
            conditional_predictions = []
            for embeddings in component_embeddings:
                conditional_predictions.append(
                    pipe.unet(
                        latent_input,
                        timestep,
                        encoder_hidden_states=embeddings,
                    ).sample
                )
            composed = composable_diffusion.compose_noise_predictions(
                unconditional_prediction,
                torch.stack(conditional_predictions, dim=1),
                batch.component_mask,
                batch.weights,
                stable_fp32=True,
            )
            if strict_finite:
                _strict_finite(composed, label="Composable noise prediction")
            latents = pipe.scheduler.step(composed, timestep, latents).prev_sample.float()
            if strict_finite:
                _strict_finite(latents, label="Composable scheduler output")
            for index, record in enumerate(diagnostics):
                record.max_latent_abs = max(
                    record.max_latent_abs,
                    float(latents[index].abs().max().item()),
                )

        images = decode_latents(pipe, latents)
    _finish_diagnostics(
        diagnostics,
        latents,
        kernel="sd14_composable_batch",
        strict_finite=strict_finite,
    )
    return images


def generate_structure_batch(
    pipe: Any,
    prepared_jobs: Sequence[PreparedJob],
    *,
    device: str,
    initial_latents_batch: torch.Tensor,
    diagnostics: Sequence[BaselineDiagnostics],
    strict_finite: bool = True,
) -> list[Image.Image]:
    """Run phrase-conditioned structured cross-attention for guided rows."""

    if not prepared_jobs or any(not job.guidance_applied for job in prepared_jobs):
        raise BaselineGenerationError("Structure Diffusion requires guided jobs")
    plans = []
    for prepared in prepared_jobs:
        if not isinstance(prepared.plan, structure_diffusion.StructurePlan):
            raise BaselineGenerationError("invalid Structure Diffusion plan")
        plans.append(prepared.plan)
    device_object = torch.device(device)
    prompt_conditioning = encode_prompts(
        pipe,
        [plan.method_prompt for plan in plans],
        device=device_object,
    )
    contexts = []
    for plan in plans:
        embeddings = _encode_texts(pipe, list(plan.conditioning_texts), device_object)
        contexts.append(
            structure_diffusion.build_conditioning(
                embeddings,
                plan.token_spans,
                conjunction=plan.conjunction,
            )
        )
    packed = structure_diffusion.pack_conditioning(contexts)
    registration = structure_diffusion.register_attention(pipe.unet, stable_fp32=True)
    registration.controller.set_conditioning(
        packed,
        do_classifier_free_guidance=True,
    )
    latents = initial_latents_batch.detach().clone().to(device_object, dtype=torch.float32)
    pipe.scheduler.set_timesteps(SD14_GENERATION.steps, device=device)
    try:
        with torch.no_grad():
            for timestep in pipe.scheduler.timesteps:
                latent_input = torch.cat([latents, latents])
                latent_input = pipe.scheduler.scale_model_input(
                    latent_input, timestep
                ).to(dtype=prompt_conditioning.cfg_prompt_embeds.dtype)
                noise_prediction = pipe.unet(
                    latent_input,
                    timestep,
                    encoder_hidden_states=prompt_conditioning.cfg_prompt_embeds,
                ).sample.float()
                noise_unconditional, noise_conditional = noise_prediction.chunk(2)
                noise_prediction = noise_unconditional + (
                    SD14_GENERATION.guidance_scale
                    * (noise_conditional - noise_unconditional)
                )
                if strict_finite:
                    _strict_finite(noise_prediction, label="Structure noise prediction")
                latents = pipe.scheduler.step(
                    noise_prediction, timestep, latents
                ).prev_sample.float()
                if strict_finite:
                    _strict_finite(latents, label="Structure scheduler output")
                for index, record in enumerate(diagnostics):
                    record.max_latent_abs = max(
                        record.max_latent_abs,
                        float(latents[index].abs().max().item()),
                    )
            images = decode_latents(pipe, latents)
    finally:
        registration.controller.clear()
        registration.restore(pipe.unet)
    _finish_diagnostics(
        diagnostics,
        latents,
        kernel="sd14_structure_batch",
        strict_finite=strict_finite,
    )
    return images


def capture_attention_maps(
    pipe: Any,
    prompts: Sequence[str],
    latents: torch.Tensor,
    timestep: torch.Tensor,
    *,
    device: str,
    controller: AttentionStore | None = None,
) -> tuple[torch.Tensor, AttentionStore]:
    """Return one official A&E guided-forward map batch for audit probes.

    The helper intentionally performs no latent update.  It is used by the
    multi-subtoken gate and by the generation loop, keeping both probes on the
    exact same selective-attention path.
    """

    device_object = torch.device(device)
    store = controller or AttentionStore(max_size=32)
    original_processors = (
        dict(pipe.unet.attn_processors) if controller is None else None
    )
    if controller is None:
        register_attention_control(
            pipe.unet,
            store,
            attention_res=SD14_GENERATION.attention_res,
        )
    embeddings = _encode_texts(pipe, list(prompts), device_object)
    try:
        store.reset()
        store.set_capture_enabled(True)
        model_input = pipe.scheduler.scale_model_input(latents, timestep).to(
            dtype=embeddings.dtype
        )
        pipe.unet(model_input, timestep, encoder_hidden_states=embeddings).sample
        store.finish_selective_forward()
        maps = aggregate_attention_batched(
            store,
            res=SD14_GENERATION.attention_res,
            batch_size=len(prompts),
        )
    finally:
        if original_processors is not None:
            pipe.unet.set_attn_processor(dict(original_processors))
    return torch.stack(maps).detach().float(), store


def probe_subtoken_maxima(
    pipe: Any,
    plans: Sequence[attend_and_excite.AttendAndExcitePlan],
    latents: torch.Tensor,
    timestep: torch.Tensor,
    *,
    device: str,
) -> list[dict[str, Any]]:
    """Probe every candidate subtoken with the official smoothed A&E max.

    Results are JSON-compatible and occurrence-local.  This is the screening
    gate hook for auditing the automated final-subtoken selection before any
    image generation is accepted.
    """

    if not plans or latents.shape[0] != len(plans):
        raise BaselineGenerationError(
            "sidecars and latents must form one non-empty probe batch"
        )
    for plan in plans:
        attend_and_excite.validate_plan(plan, max_token_count=77)
        if plan.status != "ok":
            raise BaselineGenerationError("A&E probe requires guided sidecars")
    maps, store = capture_attention_maps(
        pipe,
        [plan.prompt for plan in plans],
        latents,
        timestep,
        device=device,
    )
    processors = list(store.last_capture_processor_names)
    if len(processors) != SD14_GENERATION.expected_attention_processor_count:
        raise BaselineGenerationError(
            "A&E probe captured an unexpected processor population: "
            f"{len(processors)} != "
            f"{SD14_GENERATION.expected_attention_processor_count}"
        )
    records = []
    for sample_index, (plan, sample_maps) in enumerate(zip(plans, maps)):
        targets = []
        for occurrence_index, target in enumerate(plan.targets):
            maxima = attend_and_excite.max_attention(
                sample_maps,
                target.clip_indices,
                smooth_attentions=True,
                sigma=0.5,
                kernel_size=3,
            )
            values = [float(maximum.detach().item()) for maximum in maxima]
            selected_position = target.clip_indices.index(target.selected_clip_index)
            targets.append(
                {
                    "occurrence_index": occurrence_index,
                    "text": target.text,
                    "character_span": [target.char_start, target.char_end],
                    "candidate_token_indices": list(target.clip_indices),
                    "candidate_smoothed_max_attention": values,
                    "selected_token_index": target.selected_clip_index,
                    "selected_smoothed_max_attention": values[selected_position],
                    "best_candidate_token_index": target.clip_indices[
                        max(range(len(values)), key=values.__getitem__)
                    ],
                }
            )
        records.append(
            {
                "sample_index": sample_index,
                "prompt_sha256": hashlib.sha256(
                    plan.prompt.encode("utf-8")
                ).hexdigest(),
                "captured_attention_processors": processors,
                "targets": targets,
            }
        )
    return records


def _capture_guidance_maps(
    pipe: Any,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    embeddings: torch.Tensor,
    controller: AttentionStore,
) -> torch.Tensor:
    controller.reset()
    controller.set_capture_enabled(True)
    model_input = pipe.scheduler.scale_model_input(latents, timestep).to(
        dtype=embeddings.dtype
    )
    pipe.unet(model_input, timestep, encoder_hidden_states=embeddings).sample
    controller.finish_selective_forward()
    maps = aggregate_attention_batched(
        controller,
        res=SD14_GENERATION.attention_res,
        batch_size=latents.shape[0],
    )
    return torch.stack(maps).float()


def _record_losses(
    diagnostics: Sequence[BaselineDiagnostics],
    losses: Sequence[torch.Tensor],
) -> None:
    for record, loss in zip(diagnostics, losses):
        record.record_guidance_loss(loss)


def _attend_and_excite_refinement(
    pipe: Any,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    embeddings: torch.Tensor,
    target_indices: Sequence[Sequence[int]],
    controller: AttentionStore,
    *,
    threshold: float,
    step_size: float,
    initial_active_mask: Sequence[bool],
    diagnostics: Sequence[BaselineDiagnostics],
    strict_finite: bool,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Perform official threshold refinement with sample-local active masks."""

    if len(initial_active_mask) != latents.shape[0] or any(
        not isinstance(active, bool) for active in initial_active_mask
    ):
        raise BaselineGenerationError(
            "A&E initial_active_mask must contain one bool per sample"
        )
    iteration_counts = [0] * latents.shape[0]
    working = latents
    # The caller entered this helper because the pre-refinement loss exceeded
    # the threshold.  Official A&E therefore performs the first loop body
    # unconditionally; it tests the newly computed pre-update loss only at the
    # next while boundary.
    active = list(initial_active_mask)
    if not any(active):
        raise BaselineGenerationError(
            "A&E refinement requires at least one initially active sample"
        )
    while True:
        working = working.detach().clone().requires_grad_(True)
        maps = _capture_guidance_maps(
            pipe, working, timestep, embeddings, controller
        )
        losses = attend_and_excite.batch_losses(maps, target_indices)
        _record_losses(diagnostics, losses)
        working = sample_local_latent_update(
            working,
            losses,
            step_size,
            active_mask=active,
        )
        if strict_finite:
            _strict_finite(working, label="A&E refinement latent")
        for index, is_active in enumerate(active):
            if is_active:
                iteration_counts[index] += 1
                diagnostics[index].refinement_rounds += 1
                diagnostics[index].method_guidance_updates += 1
        # The official loop detaches before its next attention forward.
        working = working.detach()
        eligible = list(
            attend_and_excite.refinement_mask(
                losses,
                threshold=threshold,
                iteration_counts=iteration_counts,
                max_refinement_steps=attend_and_excite.MAX_REFINEMENT_STEPS,
            )
        )
        active = [was_active and can_continue for was_active, can_continue in zip(active, eligible)]
        if not any(active):
            break

    # Official helper performs one final forward after exiting and returns its
    # loss so the caller can execute the usual outer update.
    working = working.detach().clone().requires_grad_(True)
    maps = _capture_guidance_maps(pipe, working, timestep, embeddings, controller)
    final_losses = attend_and_excite.batch_losses(maps, target_indices)
    _record_losses(diagnostics, final_losses)
    return working, final_losses


def generate_attention_guided_batch(
    pipe: Any,
    prepared_jobs: Sequence[PreparedJob],
    *,
    device: str,
    initial_latents_batch: torch.Tensor,
    diagnostics: Sequence[BaselineDiagnostics],
    strict_finite: bool = True,
) -> list[Image.Image]:
    """Generate a uniform A&E or SynGen batch with sample-local gradients."""

    if not prepared_jobs or any(not job.guidance_applied for job in prepared_jobs):
        raise BaselineGenerationError("attention guidance requires guided jobs")
    method = prepared_jobs[0].method
    if method not in {"sd14_attend_excite", "sd14_syngen"}:
        raise BaselineGenerationError(f"unsupported attention-guided method: {method}")
    if any(job.method != method for job in prepared_jobs):
        raise BaselineGenerationError("one batch cannot mix external methods")

    prompts = [job.job.prompt for job in prepared_jobs]
    device_object = torch.device(device)
    conditioning = encode_prompts(
        pipe,
        prompts,
        device=device_object,
    )
    conditional_embeddings = conditioning.conditional_prompt_embeds
    cfg_embeddings = conditioning.cfg_prompt_embeds
    controller = AttentionStore(max_size=32)
    original_processors = dict(pipe.unet.attn_processors)
    register_attention_control(
        pipe.unet,
        controller,
        attention_res=SD14_GENERATION.attention_res,
    )
    expected_processors = SD14_GENERATION.expected_attention_processor_count
    latents = initial_latents_batch.detach().clone().to(device_object, dtype=torch.float32)
    pipe.scheduler.set_timesteps(SD14_GENERATION.steps, device=device)

    if method == "sd14_attend_excite":
        ae_plans = []
        for prepared in prepared_jobs:
            if not isinstance(prepared.plan, attend_and_excite.AttendAndExcitePlan):
                raise BaselineGenerationError("invalid A&E sidecar type")
            ae_plans.append(prepared.plan)
        target_indices = [plan.selected_clip_indices for plan in ae_plans]
    else:
        syngen_plans = []
        for prepared in prepared_jobs:
            if not isinstance(prepared.plan, syngen.SynGenPlan):
                raise BaselineGenerationError("invalid SynGen sidecar type")
            syngen_plans.append(prepared.plan)

    try:
        for step_index, timestep in enumerate(pipe.scheduler.timesteps):
            if step_index < attend_and_excite.INTERVENTION_STEPS:
                working = latents.detach().clone().requires_grad_(True)
                maps = _capture_guidance_maps(
                    pipe,
                    working,
                    timestep,
                    conditional_embeddings,
                    controller,
                )
                if len(controller.last_capture_processor_names) != expected_processors:
                    raise BaselineGenerationError(
                        "unexpected attention processor count: "
                        f"{len(controller.last_capture_processor_names)} != "
                        f"{expected_processors}"
                    )
                for record in diagnostics:
                    record.expected_attention_processor_count = expected_processors
                    record.captured_attention_processors = list(
                        controller.last_capture_processor_names
                    )
                    record.captured_attention_processor_count = len(
                        controller.last_capture_processor_names
                    )
                if method == "sd14_attend_excite":
                    losses = attend_and_excite.batch_losses(maps, target_indices)
                    _record_losses(diagnostics, losses)
                    if step_index in attend_and_excite.THRESHOLDS:
                        active = attend_and_excite.refinement_mask(
                            losses,
                            threshold=attend_and_excite.THRESHOLDS[step_index],
                            iteration_counts=[0] * len(losses),
                        )
                        if any(active):
                            working, losses = _attend_and_excite_refinement(
                                pipe,
                                working,
                                timestep,
                                conditional_embeddings,
                                target_indices,
                                controller,
                                threshold=attend_and_excite.THRESHOLDS[step_index],
                                step_size=attend_and_excite.step_size(step_index),
                                initial_active_mask=active,
                                diagnostics=diagnostics,
                                strict_finite=strict_finite,
                            )
                    # Official A&E always performs the outer update after any
                    # threshold refinement while i < max_iter_to_alter.
                    active = tuple(bool(loss.detach().item() != 0.0) for loss in losses)
                    latents = sample_local_latent_update(
                        working,
                        losses,
                        attend_and_excite.step_size(step_index),
                        active_mask=active,
                    ).detach()
                    for index, is_active in enumerate(active):
                        if is_active:
                            diagnostics[index].method_guidance_updates += 1
                else:
                    losses = syngen.batch_losses(maps, syngen_plans)
                    _record_losses(diagnostics, losses)
                    active = tuple(bool(loss.detach().item() != 0.0) for loss in losses)
                    latents = sample_local_latent_update(
                        working,
                        losses,
                        syngen.STEP_SIZE,
                        active_mask=active,
                    ).detach()
                    for index, is_active in enumerate(active):
                        if is_active:
                            diagnostics[index].method_guidance_updates += 1
                if strict_finite:
                    _strict_finite(latents, label=f"{method} guided latent")

            controller.set_capture_enabled(False)
            latent_input = torch.cat([latents, latents])
            latent_input = pipe.scheduler.scale_model_input(
                latent_input, timestep
            ).to(dtype=cfg_embeddings.dtype)
            with torch.no_grad():
                noise_prediction = pipe.unet(
                    latent_input,
                    timestep,
                    encoder_hidden_states=cfg_embeddings,
                ).sample.float()
            noise_unconditional, noise_conditional = noise_prediction.chunk(2)
            noise_prediction = noise_unconditional + (
                SD14_GENERATION.guidance_scale
                * (noise_conditional - noise_unconditional)
            )
            if strict_finite:
                _strict_finite(noise_prediction, label=f"{method} noise prediction")
            latents = pipe.scheduler.step(
                noise_prediction, timestep, latents
            ).prev_sample.float()
            if strict_finite:
                _strict_finite(latents, label=f"{method} scheduler output")
            for index, record in enumerate(diagnostics):
                record.max_latent_abs = max(
                    record.max_latent_abs,
                    float(latents[index].abs().max().item()),
                )
        with torch.no_grad():
            images = decode_latents(pipe, latents)
    finally:
        pipe.unet.set_attn_processor(dict(original_processors))

    _finish_diagnostics(
        diagnostics,
        latents,
        kernel=f"{method}_batch",
        strict_finite=strict_finite,
    )
    return images


def generate_batch(
    pipe: Any,
    prepared_jobs: Sequence[PreparedJob],
    *,
    device: str,
) -> list[tuple[Image.Image, BaselineDiagnostics]]:
    """Dispatch one uniform batch to a guided kernel or shared stock path."""

    if not prepared_jobs:
        raise BaselineGenerationError("prepared_jobs must not be empty")
    if any(job.method != prepared_jobs[0].method for job in prepared_jobs):
        raise BaselineGenerationError("one batch cannot mix methods")
    if any(
        job.guidance_applied != prepared_jobs[0].guidance_applied
        for job in prepared_jobs
    ):
        raise BaselineGenerationError("one batch cannot mix guided and fallback rows")
    latents, diagnostics = _initial_latent_batch(
        pipe,
        prepared_jobs,
        device=device,
        height=SD14_GENERATION.height,
        width=SD14_GENERATION.width,
        steps=SD14_GENERATION.steps,
    )
    if not prepared_jobs[0].guidance_applied:
        output = generate_stock_batch(
            pipe,
            [prepared.job for prepared in prepared_jobs],
            device=device,
            steps=SD14_GENERATION.steps,
            guidance_scale=SD14_GENERATION.guidance_scale,
            initial_latents=latents,
            diagnostics=diagnostics,
            strict_finite=True,
        )
        for record in diagnostics:
            record.generation_kernel = "sd14_external_stock_fallback_batch"
        return list(zip(output.images, diagnostics))

    method = prepared_jobs[0].method
    if method == "sd14_composable":
        images = generate_composable_batch(
            pipe,
            prepared_jobs,
            device=device,
            initial_latents_batch=latents,
            diagnostics=diagnostics,
        )
    elif method == "sd14_structure":
        images = generate_structure_batch(
            pipe,
            prepared_jobs,
            device=device,
            initial_latents_batch=latents,
            diagnostics=diagnostics,
        )
    else:
        images = generate_attention_guided_batch(
            pipe,
            prepared_jobs,
            device=device,
            initial_latents_batch=latents,
            diagnostics=diagnostics,
        )
    return list(zip(images, diagnostics))


def runtime_token_records(
    tokenizer: Any,
    jobs: Sequence[PreparedJob],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Check the CLIP tokenizer identity and every method index against the conditioning prompt's tokens.

    Returns the per-image token records (conditioning prompt, CLIP token
    counts, truncation) and the tokenizer identity.
    """

    try:
        tokenizer_identity = require_clip_tokenizer_identity(
            tokenizer,
            expected=SD14_GENERATION.tokenizer_identity(),
        )
        require_clip_bpe_behavior(tokenizer)
    except ValueError as exc:
        raise BaselineGenerationError(str(exc)) from exc
    records = {}
    for prepared in jobs:
        conditioning_prompt = prepared.actual_conditioning_prompt
        encoded = tokenizer(
            conditioning_prompt,
            add_special_tokens=True,
            padding=False,
            truncation=False,
        )["input_ids"]
        if encoded and isinstance(encoded[0], list):
            if len(encoded) != 1:
                raise BaselineGenerationError(
                    f"unexpected tokenization batch for {prepared.job.prompt_id}"
                )
            encoded = encoded[0]
        untruncated_count = len(encoded)
        effective_count = min(untruncated_count, 77)
        usable_max = effective_count - 2
        semantic_indices: list[int] = []
        if isinstance(prepared.plan, attend_and_excite.AttendAndExcitePlan):
            semantic_indices.extend(prepared.plan.selected_clip_indices)
        elif isinstance(prepared.plan, syngen.SynGenPlan):
            semantic_indices.extend(
                index
                for binding in prepared.plan.bindings
                for index in binding.flattened_indices
            )
        elif isinstance(prepared.plan, structure_diffusion.StructurePlan):
            semantic_indices.extend(
                index + 1
                for start, end in prepared.plan.token_spans
                for index in range(start, end)
            )
        invalid = sorted(
            {index for index in semantic_indices if not 1 <= index <= usable_max}
        )
        if prepared.guidance_applied and invalid:
            raise BaselineGenerationError(
                f"method indices exceed runtime CLIP text for {prepared.job.prompt_id}: "
                f"invalid={invalid}, usable=[1,{usable_max}]"
            )
        records[prepared.job.prompt_id] = {
            "actual_conditioning_prompt_sha256": hashlib.sha256(
                conditioning_prompt.encode("utf-8")
            ).hexdigest(),
            "actual_conditioning_prompt_kind": (
                "structure_normalized_method_prompt"
                if prepared.method == "sd14_structure" and prepared.guidance_applied
                else "original_manifest_prompt"
            ),
            "clip_token_count": effective_count,
            "clip_untruncated_token_count": untruncated_count,
            "clip_truncated": untruncated_count > 77,
        }
    return records, tokenizer_identity


# --- one shard of one arm --------------------------------------------------------------------------------


@dataclass(frozen=True)
class RunOptions:
    """One shard of one external SD1.4 baseline arm.

    ``batches`` runs only these physical-batch indices of the shard; batch
    membership and the ``generation_batch_id`` numbering are unchanged.
    """

    arm: str
    input_file: Path
    out_dir: Path
    prompt_manifest: Path
    blind_prompt_manifest: Path
    device: str = "cuda:0"
    num_shards: int = 1
    shard_index: int = 0
    expected_total: int | None = None
    batches: tuple[int, ...] | None = None
    max_save_attempts: int = 2
    local_files_only: bool = True
    resume: bool = False
    strict_runtime: bool = True
    strict_environment: bool = True
    allow_dirty: bool = False


def select_jobs(options: RunOptions) -> tuple[list[PreparedJob], dict[str, Any]]:
    """Load the arm's jobs, check ``expected_total`` and keep this shard (``index % num_shards``)."""

    if options.num_shards < 1:
        raise BaselineGenerationError("--num-shards must be positive")
    if not 0 <= options.shard_index < options.num_shards:
        raise BaselineGenerationError(
            "--shard-index must satisfy 0 <= shard-index < num-shards"
        )
    if options.expected_total is not None and options.expected_total < 1:
        raise BaselineGenerationError("--expected-total must be positive")
    if options.max_save_attempts < 1:
        raise BaselineGenerationError("--max-save-attempts must be positive")
    jobs, input_metadata = load_jobs(
        options.prompt_manifest,
        options.blind_prompt_manifest,
        options.input_file,
        arm=options.arm,
        strict_environment=options.strict_environment,
    )
    if options.expected_total is not None and len(jobs) != options.expected_total:
        raise BaselineGenerationError(
            f"expected {options.expected_total} jobs before sharding, got {len(jobs)}"
        )
    jobs = [
        job
        for index, job in enumerate(jobs)
        if index % options.num_shards == options.shard_index
    ]
    if not jobs:
        raise BaselineGenerationError("selected shard contains no jobs")
    return jobs, input_metadata


def _selection_record(
    selected_jobs: Sequence[PreparedJob],
    token_records: Mapping[str, Any],
) -> dict[str, Any]:
    """Counts and digests of the selected, guided and stock-fallback jobs (run-lock fields)."""

    selected_keys = [list(_job_key(prepared.job)) for prepared in selected_jobs]
    guided_keys = [
        list(_job_key(prepared.job))
        for prepared in selected_jobs
        if prepared.guidance_applied
    ]
    fallback_keys = [
        list(_job_key(prepared.job))
        for prepared in selected_jobs
        if not prepared.guidance_applied
    ]
    program_records = [
        {
            "key": list(_job_key(prepared.job)),
            "program_sha256": prepared.program_sha256,
            "sidecar_row_sha256": prepared.sidecar_row_sha256,
        }
        for prepared in selected_jobs
    ]
    return {
        "token_records_sha256": sha256_json(token_records),
        "selected_job_count": len(selected_jobs),
        "selected_job_keys_sha256": sha256_json(selected_keys),
        "guided_job_count": len(guided_keys),
        "guided_job_keys_sha256": sha256_json(guided_keys),
        "stock_fallback_job_count": len(fallback_keys),
        "stock_fallback_job_keys_sha256": sha256_json(fallback_keys),
        "program_records_sha256": sha256_json(program_records),
    }


def build_run_configuration(
    options: RunOptions,
    *,
    selected_jobs: Sequence[PreparedJob],
    input_metadata: Mapping[str, Any],
    runtime: Any,
    tokenizer: Mapping[str, Any],
    token_records: Mapping[str, Any],
    device: Mapping[str, Any],
    dataset: T2ICBDatasetProvenance,
    generator_runtime_environment: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """The run-lock ``configuration`` of one shard (key set of schema version 3, plus ``dirty``)."""

    selection = _selection_record(selected_jobs, token_records)
    configuration = {
        "schema_version": RUN_LOCK_SCHEMA_VERSION,
        "backbone": "sd14",
        "method": options.arm,
        "method_source_commit": SOURCE_COMMIT_BY_ARM[options.arm],
        "model_contract": SD14_GENERATION.to_dict(),
        "precision_profile_requested": SD14_GENERATION.precision_profile,
        "precision_profile_resolved": runtime.precision.name,
        "precision_profile": runtime.precision.as_record(),
        "execution_mode": SD14_GENERATION.execution_mode,
        "physical_batch_size": SD14_GENERATION.batch_size,
        "batch_padding_policy": BATCH_PADDING_POLICY,
        "compile_mode": SD14_GENERATION.compile_mode,
        "memory_format": SD14_GENERATION.memory_format,
        "finite_guard": SD14_GENERATION.finite_guard,
        "scheduler": dict(runtime.scheduler),
        "tokenizer": dict(tokenizer),
        "token_records_sha256": selection["token_records_sha256"],
        "determinism": dict(runtime.determinism),
        "device": dict(device),
        "pipeline_dtypes": dict(runtime.pipeline_dtypes),
        "prompt_manifest": portable_path(options.prompt_manifest),
        "prompt_manifest_sha256": sha256_file(options.prompt_manifest),
        **dataset.run_record(),
        "generator_runtime_environment": dict(generator_runtime_environment),
        "blind_prompt_manifest": portable_path(options.blind_prompt_manifest),
        "blind_prompt_manifest_sha256": sha256_file(options.blind_prompt_manifest),
        "sidecar": portable_path(options.input_file),
        "sidecar_sha256": sha256_file(options.input_file),
        "sidecar_metadata_sha256": sha256_file(
            options.input_file.with_name(options.input_file.name + ".meta.json")
        ),
        "sidecar_metadata_identity_sha256": sha256_json(input_metadata),
        "sidecar_protocol": input_metadata["adapter_protocol"],
        "gold_inputs_used": False,
        **{key: value for key, value in selection.items() if key != "token_records_sha256"},
        "num_shards": options.num_shards,
        "shard_index": options.shard_index,
        # Recorded as null since this runner has no image or experiment subset.
        "max_images": None,
        "experiments": None,
        **source,
    }
    if options.batches is not None:
        configuration["batch_subset"] = list(options.batches)
    return configuration


def metadata_row(
    prepared: PreparedJob,
    path: Path,
    diagnostics: BaselineDiagnostics,
    options: RunOptions,
    *,
    runtime: Any,
    dataset: T2ICBDatasetProvenance,
    token_record: Mapping[str, Any],
    configuration_signature: str,
    lock_path: Path,
    lock_sha256: str,
    tokenizer: Mapping[str, Any],
    physical_batch_position: int,
) -> dict[str, Any]:
    """One image's metadata row (key set of metadata schema version 3)."""

    fallback = not prepared.guidance_applied
    attention_guided = prepared.method in {"sd14_attend_excite", "sd14_syngen"} and prepared.guidance_applied
    return {
        "metadata_schema_version": METADATA_SCHEMA_VERSION,
        "experiment": prepared.job.experiment,
        "prompt_id": prepared.job.prompt_id,
        "prompt": prepared.job.prompt,
        "seed": prepared.job.seed,
        **dataset.image_record(prepared.job),
        "physical_batch_position": physical_batch_position,
        "method": prepared.method,
        "backbone": "sd14",
        "method_source_commit": SOURCE_COMMIT_BY_ARM[prepared.method],
        "preprocessing_status": prepared.status,
        "guidance_mode": prepared.guidance_mode,
        "guidance_applied": prepared.guidance_applied,
        "fallback_used": fallback,
        "fallback_reason": prepared.status if fallback else None,
        "prompt_sha256": prepared.prompt_sha256,
        "sidecar_row_sha256": prepared.sidecar_row_sha256,
        "program_sha256": prepared.program_sha256,
        **dict(token_record),
        "image_path": repo_relative(path),
        "image_sha256": sha256_file(path),
        "model_id": SD14_GENERATION.model_id,
        "model_revision": SD14_GENERATION.model_revision,
        "precision_profile": runtime.precision.as_record(),
        "execution_mode": SD14_GENERATION.execution_mode,
        "batch_size_requested": SD14_GENERATION.batch_size,
        "batch_size_effective": diagnostics.effective_batch_size,
        "compile_mode": SD14_GENERATION.compile_mode,
        "memory_format": SD14_GENERATION.memory_format,
        "scheduler": dict(runtime.scheduler),
        "tokenizer": dict(tokenizer),
        "steps": SD14_GENERATION.steps,
        "guidance_scale": SD14_GENERATION.guidance_scale,
        "height": SD14_GENERATION.height,
        "width": SD14_GENERATION.width,
        "negative_prompt": SD14_GENERATION.negative_prompt,
        "safety_checker": SD14_GENERATION.safety_checker_enabled,
        "attention_res": SD14_GENERATION.attention_res if attention_guided else None,
        "attention_block_scope": SD14_GENERATION.attention_block_scope if attention_guided else None,
        "generation_kernel": diagnostics.generation_kernel,
        "generation_batch_id": diagnostics.generation_batch_id,
        "strict_finite": diagnostics.strict_finite_applied,
        "initial_latent_sha256": diagnostics.initial_latent_sha256,
        "generation_diagnostics": diagnostics.as_dict(),
        "prompt_manifest_sha256": sha256_file(options.prompt_manifest),
        "blind_prompt_manifest_sha256": sha256_file(options.blind_prompt_manifest),
        "sidecar_sha256": sha256_file(options.input_file),
        "sidecar_metadata_sha256": sha256_file(
            options.input_file.with_name(options.input_file.name + ".meta.json")
        ),
        "configuration_signature": configuration_signature,
        "run_lock_path": repo_relative(lock_path),
        "run_lock_sha256": lock_sha256,
        "num_shards": options.num_shards,
        "shard_index": options.shard_index,
    }


def _batch_rows(
    batches: Sequence[Sequence[Any]],
    *,
    shard_index: int,
    num_shards: int,
) -> list[dict[str, Any]]:
    """Per-image batch membership of a planned shard (the fields the metadata rows record)."""

    rows = []
    for batch_index, batch in enumerate(batches):
        for position, prepared in enumerate(batch):
            rows.append({
                "experiment": prepared.job.experiment,
                "prompt_id": prepared.job.prompt_id,
                "seed": prepared.job.seed,
                "generation_batch_id": f"{shard_suffix(shard_index, num_shards)}:batch_{batch_index:06d}",
                "physical_batch_position": position,
                "valid_batch_size": len(batch),
                "batch_padding_count": SD14_GENERATION.batch_size - len(batch),
                "guidance_mode": prepared.guidance_mode,
            })
    return rows


def plan_projection(options: RunOptions) -> dict[str, Any]:
    """The run-lock selection fields and the batch membership of one shard, without a GPU (CPU only)."""

    jobs, input_metadata = select_jobs(options)
    tokenizer = load_sd14_tokenizer(local_files_only=options.local_files_only)
    token_records, _identity = runtime_token_records(tokenizer, jobs)
    return {
        "method": options.arm,
        "method_source_commit": SOURCE_COMMIT_BY_ARM[options.arm],
        "model_contract": SD14_GENERATION.to_dict(),
        "physical_batch_size": SD14_GENERATION.batch_size,
        "batch_padding_policy": BATCH_PADDING_POLICY,
        "sidecar_sha256": sha256_file(options.input_file),
        "sidecar_protocol": input_metadata["adapter_protocol"],
        **_selection_record(jobs, token_records),
        "rows": _batch_rows(
            plan_batches(jobs, SD14_GENERATION.batch_size),
            shard_index=options.shard_index,
            num_shards=options.num_shards,
        ),
    }


def driver_plan_projection(request: SD14Run) -> dict[str, Any]:
    """The run-lock selection fields and batch membership of one SD1.4 driver shard (CPU only).

    Used for the Predicated Diffusion baseline (``sd14_pd_spacy``): the same
    expressions as ``records.run_configuration`` over ``driver.plan_run``.
    """

    plan = plan_run(request)
    tokenizer = load_sd14_tokenizer(local_files_only=request.local_files_only)
    token_counts, tokenizer_info, clip_truncation = validate_clip_token_contract(tokenizer, plan.jobs)

    def keys(jobs: Sequence[Any]) -> list[list[Any]]:
        return [[item.job.experiment, item.job.prompt_id, item.job.seed] for item in jobs]

    guided = [item for item in plan.jobs if item.guidance_applied]
    fallback = [item for item in plan.jobs if item.guidance_mode == "stock_vanilla_fallback"]
    metadata = request.target_file.with_name(request.target_file.name + ".meta.json")
    return {
        "method": request.method,
        "selected_job_count": len(plan.jobs),
        "selected_job_keys_sha256": sha256_json(keys(plan.jobs)),
        "guided_job_count": len(guided),
        "guided_job_keys_sha256": sha256_json(keys(guided)),
        "stock_vanilla_fallback_job_count": len(fallback),
        "stock_vanilla_fallback_job_keys_sha256": sha256_json(keys(fallback)),
        "pd_sidecar_sha256": sha256_file(request.target_file),
        "pd_sidecar_metadata_sha256": sha256_file(metadata),
        "clip_token_counts_sha256": sha256_json(token_counts),
        "clip_truncation_sha256": sha256_json(clip_truncation),
        "clip_truncated_prompt_count": sum(1 for record in clip_truncation.values() if record["truncated"]),
        "tokenizer": tokenizer_info,
        "rows": _batch_rows(plan.batches, shard_index=request.shard_index, num_shards=request.num_shards),
    }


def run(options: RunOptions) -> None:  # noqa: C901
    """Generate (or resume) one shard of one external arm: run lock, images and metadata rows."""

    dataset = load_t2icb_dataset_provenance(options.prompt_manifest)
    environment = sd14_generator_runtime_environment(device=options.device, strict=options.strict_runtime)
    jobs, input_metadata = select_jobs(options)
    source = _source_record(COMPONENT_FILES, allow_dirty=options.allow_dirty)
    runtime = prepare_sd14_runtime(options.device, local_files_only=options.local_files_only)
    pipe = runtime.pipe
    token_records, tokenizer = runtime_token_records(pipe.tokenizer, jobs)
    configuration = build_run_configuration(
        options,
        selected_jobs=jobs,
        input_metadata=input_metadata,
        runtime=runtime,
        tokenizer=tokenizer,
        token_records=token_records,
        device=device_record(options.device),
        dataset=dataset,
        generator_runtime_environment=environment,
        source=source,
    )
    lock_path = run_lock_path(
        options.out_dir, options.arm, options.shard_index, options.num_shards
    )
    lock = ensure_run_lock(lock_path, configuration, resume=options.resume)
    lock_sha256 = sha256_file(lock_path)
    configuration_signature = str(lock["configuration_signature"])
    metadata_paths = {
        experiment: shard_metadata_path(
            options.out_dir,
            options.arm,
            experiment,
            options.shard_index,
            options.num_shards,
        )
        for experiment in sorted({prepared.job.experiment for prepared in jobs})
    }
    if not options.resume:
        existing_metadata = [path for path in metadata_paths.values() if path.exists()]
        if existing_metadata:
            raise BaselineGenerationError(
                f"metadata already exists for new run: {existing_metadata[0]}"
            )
    records = load_metadata_records(
        metadata_paths.values(),
        expected_signature=configuration_signature,
    )
    expected_size = (
        SD14_GENERATION.width,
        SD14_GENERATION.height,
    )
    for prepared in jobs:
        path = image_path(options.out_dir, options.arm, prepared.job)
        existing = records.get(repo_relative(path))
        if existing is not None:
            if not options.resume:
                raise BaselineGenerationError(f"unexpected recorded image: {path}")
            if not image_is_valid(path, expected_size=expected_size):
                raise BaselineGenerationError(f"recorded image is invalid: {path}")
            if existing.get("image_sha256") != sha256_file(path):
                raise BaselineGenerationError(f"recorded image hash mismatch: {path}")
            continue
        if path.exists():
            if not options.resume:
                raise BaselineGenerationError(
                    f"unlocked image would be overwritten: {path}"
                )
            path.unlink()

    planned_batches = plan_batches(
        jobs,
        SD14_GENERATION.batch_size,
    )
    if options.batches is not None and any(index not in range(len(planned_batches)) for index in options.batches):
        raise BaselineGenerationError(
            f"batch indices outside the {len(planned_batches)} physical batches of this shard: {options.batches}"
        )
    selected_batches = [
        (batch_index, batch)
        for batch_index, batch in enumerate(planned_batches)
        if options.batches is None or batch_index in options.batches
    ]
    progress = tqdm(
        total=sum(
            1
            for _batch_index, batch in selected_batches
            for prepared in batch
            if repo_relative(image_path(options.out_dir, options.arm, prepared.job)) not in records
        ),
        desc=f"generate:{options.arm}:{options.shard_index}",
    )
    for batch_index, batch in selected_batches:
        execution_batch, padding_count = pad_batch(
            batch,
            SD14_GENERATION.batch_size,
        )
        paths = [image_path(options.out_dir, options.arm, prepared.job) for prepared in batch]
        missing = [repo_relative(path) not in records for path in paths]
        if not any(missing):
            continue
        generated: list[tuple[Image.Image, BaselineDiagnostics]] = []
        last_error: Exception | None = None
        for _attempt in range(options.max_save_attempts):
            try:
                synchronize_device(options.device)
                reset_peak_memory(options.device)
                wall_start = time.perf_counter()
                generated_all = generate_batch(
                    pipe,
                    execution_batch,
                    device=options.device,
                )
                if len(generated_all) != SD14_GENERATION.batch_size:
                    raise BaselineGenerationError(
                        "generation kernel returned the wrong physical batch size: "
                        f"{len(generated_all)} != {SD14_GENERATION.batch_size}"
                    )
                generated = generated_all[:len(batch)]
                synchronize_device(options.device)
                wall_time_ms = (time.perf_counter() - wall_start) * 1000.0
                peak_allocated, peak_reserved = peak_memory(options.device)
                batch_id = (
                    f"{shard_suffix(options.shard_index, options.num_shards)}:"
                    f"batch_{batch_index:06d}"
                )
                for _image, diagnostics in generated:
                    diagnostics.generation_batch_id = batch_id
                    diagnostics.effective_batch_size = len(execution_batch)
                    diagnostics.valid_batch_size = len(batch)
                    diagnostics.batch_padding_count = padding_count
                    diagnostics.batch_wall_time_ms = round(wall_time_ms, 4)
                    diagnostics.amortized_wall_time_ms = round(
                        wall_time_ms / len(execution_batch), 4
                    )
                    diagnostics.peak_cuda_memory_allocated_bytes = peak_allocated
                    diagnostics.peak_cuda_memory_reserved_bytes = peak_reserved
                for path, is_missing, (image, _diagnostics) in zip(
                    paths, missing, generated
                ):
                    if is_missing:
                        save_image_atomic(image, path, expected_size=expected_size)
                last_error = None
                break
            except (OSError, RuntimeError, ValueError) as exc:
                last_error = exc
                for path, is_missing in zip(paths, missing):
                    if is_missing and path.exists():
                        path.unlink()
        if last_error is not None:
            raise BaselineGenerationError(
                "failed to generate valid external-method batch after "
                f"{options.max_save_attempts} attempts: {paths[0]}"
            ) from last_error

        for physical_batch_position, (
            prepared,
            path,
            is_missing,
            (_image, diagnostics),
        ) in enumerate(zip(batch, paths, missing, generated)):
            if not is_missing:
                continue
            payload = metadata_row(
                prepared,
                path,
                diagnostics,
                options,
                runtime=runtime,
                dataset=dataset,
                token_record=token_records[prepared.job.prompt_id],
                configuration_signature=configuration_signature,
                lock_path=lock_path,
                lock_sha256=lock_sha256,
                tokenizer=tokenizer,
                physical_batch_position=physical_batch_position,
            )
            append_metadata(metadata_paths[prepared.job.experiment], payload)
            records[repo_relative(path)] = payload
        progress.update(sum(missing))
    progress.close()


# --- Attend-and-Excite subtoken selection ---------------------------------------------------------------

SELECTION_ARM = "sd14_attend_excite"


def _current_local_implementation() -> dict[str, Any]:
    """The committed source files of this selection (the ``local_implementation`` record)."""

    try:
        record = git_head_components_record(SELECTION_COMPONENT_FILES)
    except SourceSealError as exc:
        raise BaselineGenerationError(str(exc)) from exc
    return {
        "repo_git_commit": record["repo_git_commit"],
        "component_sha256": record["component_sha256"],
        "components_match_repo_git_commit": True,
    }


def _batches(
    items: Sequence[PreparedJob], size: int
) -> list[list[PreparedJob]]:
    return [list(items[index:index + size]) for index in range(0, len(items), size)]


def _selection_population(
    jobs: Sequence[PreparedJob],
) -> tuple[list[PreparedJob], dict[str, dict[int, PreparedJob]]]:
    """Require one seed 42-51 population for every unique blind prompt."""

    by_prompt: dict[str, dict[int, PreparedJob]] = {}
    program_by_prompt: dict[str, str] = {}
    for prepared in jobs:
        if prepared.method != SELECTION_ARM or not isinstance(prepared.plan, attend_and_excite.AttendAndExcitePlan):
            raise attend_and_excite.SubtokenSelectionError("candidate population contains a non-A&E row")
        if prepared.guidance_applied != bool(prepared.plan.targets):
            raise attend_and_excite.SubtokenSelectionError(
                "candidate guidance status and literal NOUN targets disagree"
            )
        prompt = prepared.job.prompt
        previous_program = program_by_prompt.setdefault(prompt, prepared.program_sha256)
        if previous_program != prepared.program_sha256:
            raise attend_and_excite.SubtokenSelectionError("candidate program drifted across seeds")
        seed_rows = by_prompt.setdefault(prompt, {})
        if prepared.job.seed in seed_rows:
            raise attend_and_excite.SubtokenSelectionError(
                f"duplicate prompt/seed in candidate population: {prompt!r}/{prepared.job.seed}"
            )
        seed_rows[prepared.job.seed] = prepared
    required_seeds = set(attend_and_excite.SELECTION_SEEDS)
    for prompt, seed_rows in by_prompt.items():
        if set(seed_rows) != required_seeds:
            raise attend_and_excite.SubtokenSelectionError(
                f"prompt does not have exactly seeds 42-51: {prompt!r}"
            )
    ordered = [
        by_prompt[prompt][seed]
        for prompt in by_prompt
        for seed in attend_and_excite.SELECTION_SEEDS
        if by_prompt[prompt][seed].guidance_applied
    ]
    return ordered, by_prompt


def build_occurrence_evidence(
    by_prompt: Mapping[str, Mapping[int, PreparedJob]],
    observations: Mapping[tuple[str, int], Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int]]:
    """Aggregate ten per-seed probes and return selections by occurrence ID."""

    occurrences: list[dict[str, Any]] = []
    selected_by_id: dict[str, int] = {}
    occurrence_count_by_prompt: dict[str, int] = {}
    for prompt, seed_rows in by_prompt.items():
        representative = seed_rows[attend_and_excite.SELECTION_SEEDS[0]]
        plan = representative.plan
        if not isinstance(plan, attend_and_excite.AttendAndExcitePlan):
            raise attend_and_excite.SubtokenSelectionError("representative A&E plan has invalid type")
        occurrence_count_by_prompt[prompt] = len(plan.targets)
        for target_position, target in enumerate(plan.targets):
            identifier = attend_and_excite.occurrence_id(
                prompt,
                occurrence_index=target_position,
                text=target.text,
                character_span=(target.char_start, target.char_end),
                candidate_token_indices=target.clip_indices,
            )
            seed_scores = []
            scores_by_candidate: list[list[float]] = [list() for _ in target.clip_indices]
            for seed in attend_and_excite.SELECTION_SEEDS:
                observation = observations.get((prompt, seed))
                if not isinstance(observation, Mapping):
                    raise attend_and_excite.SubtokenSelectionError(
                        f"missing probe observation for {prompt!r}/{seed}"
                    )
                raw_targets = observation.get("targets")
                if not isinstance(raw_targets, list) or len(raw_targets) != len(plan.targets):
                    raise attend_and_excite.SubtokenSelectionError("probe target population drifted")
                raw_target = raw_targets[target_position]
                expected_identity = {
                    "occurrence_index": target_position,
                    "text": target.text,
                    "character_span": [target.char_start, target.char_end],
                    "candidate_token_indices": list(target.clip_indices),
                }
                if any(raw_target.get(key) != value for key, value in expected_identity.items()):
                    raise attend_and_excite.SubtokenSelectionError(
                        "probe occurrence identity differs from candidate sidecar"
                    )
                raw_scores = raw_target.get("candidate_smoothed_max_attention")
                if not isinstance(raw_scores, list) or len(raw_scores) != len(
                    target.clip_indices
                ):
                    raise attend_and_excite.SubtokenSelectionError("probe candidate score width drifted")
                scores = [float(value) for value in raw_scores]
                for candidate_position, score in enumerate(scores):
                    scores_by_candidate[candidate_position].append(score)
                latent_digest = observation.get("initial_latent_sha256")
                if not isinstance(latent_digest, str) or len(latent_digest) != 64:
                    raise attend_and_excite.SubtokenSelectionError("probe omitted initial latent hash")
                seed_scores.append(
                    {
                        "seed": seed,
                        "initial_latent_sha256": latent_digest,
                        "candidate_smoothed_max_attention": scores,
                    }
                )
            means = [statistics.fmean(values) for values in scores_by_candidate]
            selected = attend_and_excite.select_mean_winner(target.clip_indices, means)
            if identifier in selected_by_id:
                raise attend_and_excite.SubtokenSelectionError(f"duplicate occurrence ID: {identifier}")
            selected_by_id[identifier] = selected
            occurrences.append(
                {
                    "occurrence_id": identifier,
                    "experiment": representative.job.experiment,
                    "prompt": prompt,
                    "prompt_sha256": representative.prompt_sha256,
                    "text": target.text,
                    "character_span": [target.char_start, target.char_end],
                    "candidate_token_indices": list(target.clip_indices),
                    "seed_scores": seed_scores,
                    "candidate_mean_smoothed_max_attention": means,
                    "selected_token_index": selected,
                }
            )
    return occurrences, selected_by_id, occurrence_count_by_prompt


def build_no_target_evidence(
    by_prompt: Mapping[str, Mapping[int, PreparedJob]],
) -> list[dict[str, Any]]:
    """Seal every ten-seed stock fallback whose candidate parse found no NOUN."""

    records = []
    for prompt, seed_rows in by_prompt.items():
        representative = seed_rows[attend_and_excite.SELECTION_SEEDS[0]]
        plan = representative.plan
        if not isinstance(plan, attend_and_excite.AttendAndExcitePlan):
            raise attend_and_excite.SubtokenSelectionError("no-target representative has invalid type")
        if plan.targets:
            continue
        if representative.guidance_applied or plan.status != "no_noun_target":
            raise attend_and_excite.SubtokenSelectionError("invalid candidate no-target status")
        seed_provenance = []
        for seed in attend_and_excite.SELECTION_SEEDS:
            prepared = seed_rows[seed]
            prepared_plan = prepared.plan
            if (
                prepared.guidance_applied
                or not isinstance(prepared_plan, attend_and_excite.AttendAndExcitePlan)
                or prepared_plan.targets
                or prepared_plan.status != "no_noun_target"
                or prepared.program_sha256 != representative.program_sha256
            ):
                raise attend_and_excite.SubtokenSelectionError(
                    f"no-target fallback drifted across seeds for {prompt!r}"
                )
            seed_provenance.append(
                {
                    "seed": seed,
                    "prompt_id": prepared.job.prompt_id,
                    "candidate_sidecar_row_sha256": prepared.sidecar_row_sha256,
                }
            )
        records.append(
            {
                "no_target_id": attend_and_excite.no_target_id(prompt),
                "experiment": representative.job.experiment,
                "prompt": prompt,
                "prompt_sha256": representative.prompt_sha256,
                "status": plan.status,
                "candidate_program_sha256": representative.program_sha256,
                "seed_rows": seed_provenance,
            }
        )
    return records


def finalize_rows(
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    evidence_record: Mapping[str, Any],
    selected_by_id: Mapping[str, int],
) -> list[dict[str, Any]]:
    """The final A&E input rows: each target keeps its candidates and records the selected subtoken."""

    rows = []
    program_by_prompt: dict[str, str] = {}
    for row_number, raw_row in enumerate(candidate_rows, start=1):
        row = dict(raw_row)
        if set(row) != preprocessing.INPUT_ROW_FIELDS or row.get("method") != SELECTION_ARM:
            raise attend_and_excite.SubtokenSelectionError(
                f"candidate row schema/method mismatch at row {row_number}"
            )
        prompt = str(row["prompt"])
        raw_plan = row.get("plan")
        if not isinstance(raw_plan, Mapping):
            raise attend_and_excite.SubtokenSelectionError(f"candidate plan missing at row {row_number}")
        plan = dict(raw_plan)
        targets = []
        selected_indices = []
        raw_targets = plan.get("targets")
        if not isinstance(raw_targets, list):
            raise attend_and_excite.SubtokenSelectionError(f"candidate targets missing at row {row_number}")
        for target_position, raw_target_value in enumerate(raw_targets):
            if not isinstance(raw_target_value, Mapping):
                raise attend_and_excite.SubtokenSelectionError("candidate target must be an object")
            raw_target = dict(raw_target_value)
            identifier = attend_and_excite.occurrence_id(
                prompt,
                occurrence_index=target_position,
                text=str(raw_target["text"]),
                character_span=raw_target["character_span"],
                candidate_token_indices=raw_target["token_indices"],
            )
            try:
                selected = selected_by_id[identifier]
            except KeyError as exc:
                raise attend_and_excite.SubtokenSelectionError(
                    f"candidate target lacks selection evidence: {identifier}"
                ) from exc
            raw_target["selected_token_index"] = selected
            raw_target["selection_evidence_occurrence_id"] = identifier
            targets.append(raw_target)
            selected_indices.append(selected)
        plan["targets"] = targets
        plan["indices_to_alter"] = selected_indices
        plan["selection_protocol"] = attend_and_excite.ENSEMBLE_SELECTION_PROTOCOL
        plan["selection_evidence"] = dict(evidence_record)
        program_digest = hashlib.sha256(canonical_json(plan).encode("utf-8")).hexdigest()
        previous_digest = program_by_prompt.setdefault(prompt, program_digest)
        if previous_digest != program_digest:
            raise attend_and_excite.SubtokenSelectionError(
                f"final program is seed-dependent for prompt {prompt!r}"
            )
        row["plan"] = plan
        rows.append(row)
    return rows


def select_attend_and_excite_subtokens(  # noqa: C901
    *,
    generation_manifest: Path,
    blind_prompt_manifest: Path,
    input_dir: Path,
    device: str = "cuda:0",
    batch_size: int = 4,
    expected_prompt_count: int = 2400,
    expected_occurrence_count: int | None = None,
    local_files_only: bool = True,
    resume: bool = False,
    strict_environment: bool = True,
) -> dict[str, Any]:
    """Write the final A&E input file, its selection evidence and metadata from the candidate input.

    Every candidate subtoken of every noun occurrence is probed at the first
    PNDM timestep (981) for seeds 42-51 with the official smoothed A&E
    maximum; the subtoken with the largest 10-seed mean is selected (smallest
    index on an exact tie). The written input is reloaded through ``load_jobs``,
    which recomputes every mean and winner.
    """

    if batch_size < 1:
        raise attend_and_excite.SubtokenSelectionError("--batch-size must be positive")
    candidate_input = preprocessing.input_path(input_dir, SELECTION_ARM, stage="candidate")
    output_input = preprocessing.input_path(input_dir, SELECTION_ARM, stage="final")
    evidence_output = output_input.with_name(f"{output_input.stem}.selection_evidence.json")
    metadata_output = output_input.with_name(output_input.name + ".meta.json")
    existing = [path.exists() for path in (output_input, evidence_output, metadata_output)]
    if any(existing):
        if not resume:
            raise FileExistsError(
                next(path for path in (output_input, evidence_output, metadata_output) if path.exists())
            )
        if not all(existing):
            raise attend_and_excite.SubtokenSelectionError(
                "--resume refuses a partial v3 artifact set; choose new output paths"
            )
        jobs, _metadata = load_jobs(
            generation_manifest, blind_prompt_manifest, output_input, arm=SELECTION_ARM,
            strict_environment=strict_environment)
        if len(jobs) != expected_prompt_count * len(attend_and_excite.SELECTION_SEEDS):
            raise attend_and_excite.SubtokenSelectionError("resumed v3 sidecar has unexpected row count")
        return {"resumed": True, "row_count": len(jobs)}

    local_implementation = _current_local_implementation()
    all_jobs, candidate_metadata = load_jobs(
        generation_manifest,
        blind_prompt_manifest,
        candidate_input,
        arm=SELECTION_ARM,
        stage="candidate",
        strict_environment=strict_environment,
    )
    ordered_jobs, by_prompt = _selection_population(all_jobs)
    if len(by_prompt) != expected_prompt_count:
        raise attend_and_excite.SubtokenSelectionError(
            f"expected {expected_prompt_count} prompts, got {len(by_prompt)}"
        )

    runtime = prepare_sd14_runtime(device, local_files_only=local_files_only)
    pipe = runtime.pipe
    tokenizer_identity = require_clip_tokenizer_identity(
        pipe.tokenizer,
        expected=SD14_GENERATION.tokenizer_identity(),
    )
    pipe.scheduler.set_timesteps(SD14_GENERATION.steps, device=device)
    timestep = pipe.scheduler.timesteps[0]
    timestep_value = int(timestep.item())

    observations: dict[tuple[str, int], dict[str, Any]] = {}
    latent_hash_by_seed: dict[int, str] = {}
    captured_processors: tuple[str, ...] | None = None
    for batch in tqdm(_batches(ordered_jobs, batch_size), desc="A&E v3 selection"):
        latent_tracks = []
        plans = []
        for prepared in batch:
            _raw, scaled, latent_digest = initial_latents(
                pipe,
                prepared.job,
                device=device,
                dtype=torch.float32,
                height=SD14_GENERATION.height,
                width=SD14_GENERATION.width,
                steps=SD14_GENERATION.steps,
            )
            if latent_digest != tensor_sha256(scaled):
                raise attend_and_excite.SubtokenSelectionError("initial latent digest changed")
            previous_digest = latent_hash_by_seed.setdefault(prepared.job.seed, latent_digest)
            if previous_digest != latent_digest:
                raise attend_and_excite.SubtokenSelectionError(
                    f"seed {prepared.job.seed} generated inconsistent initial latents"
                )
            latent_tracks.append(scaled)
            plans.append(prepared.plan)
        pipe.scheduler.set_timesteps(SD14_GENERATION.steps, device=device)
        if int(pipe.scheduler.timesteps[0].item()) != timestep_value:
            raise attend_and_excite.SubtokenSelectionError("first scheduler timestep drifted")
        with torch.no_grad():
            raw_records = probe_subtoken_maxima(
                pipe,
                plans,
                torch.cat(latent_tracks),
                pipe.scheduler.timesteps[0],
                device=device,
            )
        for prepared, raw_record in zip(batch, raw_records):
            processors = raw_record.get("captured_attention_processors")
            if not isinstance(processors, list):
                raise attend_and_excite.SubtokenSelectionError("probe omitted processor identities")
            processor_tuple = tuple(processors)
            if captured_processors is None:
                captured_processors = processor_tuple
            elif captured_processors != processor_tuple:
                raise attend_and_excite.SubtokenSelectionError("captured processor identities drifted")
            key = (prepared.job.prompt, prepared.job.seed)
            if key in observations:
                raise attend_and_excite.SubtokenSelectionError(f"duplicate probe observation: {key}")
            observations[key] = {
                **raw_record,
                "initial_latent_sha256": latent_hash_by_seed[prepared.job.seed],
            }

    occurrences, selected_by_id, occurrence_counts = build_occurrence_evidence(
        by_prompt,
        observations,
    )
    no_target_prompts = build_no_target_evidence(by_prompt)
    if expected_occurrence_count is not None and len(occurrences) != expected_occurrence_count:
        raise attend_and_excite.SubtokenSelectionError(
            f"expected {expected_occurrence_count} occurrences, got {len(occurrences)}"
        )
    if occurrences and captured_processors is None:
        raise attend_and_excite.SubtokenSelectionError("selection captured no attention processors")
    if captured_processors is None:
        captured_processors = ()
    changed_count = sum(
        occurrence["selected_token_index"]
        != occurrence["candidate_token_indices"][-1]
        for occurrence in occurrences
    )
    multi_count = sum(
        len(occurrence["candidate_token_indices"]) > 1
        for occurrence in occurrences
    )
    candidate_metadata_path = candidate_input.with_name(candidate_input.name + ".meta.json")
    candidate_input_record = _artifact(candidate_input)
    evidence_payload = {
        "schema": preprocessing.SELECTION_EVIDENCE_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_protocol": attend_and_excite.ENSEMBLE_SELECTION_PROTOCOL,
        "method": SELECTION_ARM,
        "backbone": "sd14",
        "model_id": SD14_GENERATION.model_id,
        "model_revision": SD14_GENERATION.model_revision,
        "precision_profile": runtime.precision.as_record(),
        "pipeline_dtypes": dict(runtime.pipeline_dtypes),
        "scheduler": dict(runtime.scheduler),
        "selection_denoising_step_index": 0,
        "selection_scheduler_timestep": timestep_value,
        "selection_seeds": list(attend_and_excite.SELECTION_SEEDS),
        "attention": {
            "resolution": SD14_GENERATION.attention_res,
            "from_where": list(ATTENTION_PLACES),
            "captured_attention_processors": list(captured_processors),
            "captured_attention_processor_count": len(captured_processors),
            "text_attention_transform": "softmax(raw_attention[:, :, 1:-1] * 100)",
            "smoothing": {
                "kernel": "official Attend-and-Excite GaussianSmoothing",
                "kernel_size": 3,
                "sigma": 0.5,
            },
        },
        "tie_break": "maximum_10_seed_mean_then_smallest_token_index",
        "tokenizer": tokenizer_identity,
        "inputs": {
            "generation_manifest": _artifact(generation_manifest),
            "blind_prompt_manifest": _artifact(blind_prompt_manifest),
            "candidate_sidecar": candidate_input_record,
            "candidate_sidecar_metadata": _artifact(candidate_metadata_path),
            "candidate_metadata_identity_sha256": hashlib.sha256(
                canonical_json(candidate_metadata).encode("utf-8")
            ).hexdigest(),
            "gold_inputs_used": False,
        },
        "runtime": {
            "device": device_record(device),
            "determinism": dict(runtime.determinism),
            "physical_batch_size": batch_size,
        },
        "summary": {
            "unique_prompt_count": len(by_prompt),
            "guided_prompt_count": len(by_prompt) - len(no_target_prompts),
            "no_target_prompt_count": len(no_target_prompts),
            "occurrence_count": len(occurrences),
            "single_subtoken_occurrence_count": len(occurrences) - multi_count,
            "multi_subtoken_occurrence_count": multi_count,
            "changed_from_candidate_final_subtoken_count": changed_count,
            "guided_row_count": len(ordered_jobs),
            "stock_fallback_row_count": (
                len(no_target_prompts) * len(attend_and_excite.SELECTION_SEEDS)
            ),
            "prompt_occurrence_count_sha256": hashlib.sha256(
                canonical_json(occurrence_counts).encode("utf-8")
            ).hexdigest(),
        },
        "gold_inputs_used": False,
        "local_implementation": local_implementation,
        "occurrences": occurrences,
        "no_target_prompts": no_target_prompts,
    }
    atomic_write_json(evidence_output, evidence_payload, overwrite=False)
    evidence_record = _artifact(evidence_output)
    candidate_rows = [dict(row) for row in read_jsonl(candidate_input, reject_duplicate_keys=True)]
    final_rows = finalize_rows(
        candidate_rows,
        evidence_record=evidence_record,
        selected_by_id=selected_by_id,
    )
    atomic_write_jsonl(output_input, final_rows, overwrite=False)
    unique_status_counts = Counter(
        seed_rows[attend_and_excite.SELECTION_SEEDS[0]].status
        for seed_rows in by_prompt.values()
    )
    metadata_payload = {
        "artifact": "sd14_external_method_sidecar",
        "backbone": "sd14",
        "method": SELECTION_ARM,
        "source_commit": candidate_metadata["source_commit"],
        "adapter_protocol": preprocessing.ATTEND_EXCITE_FINAL_PROTOCOL,
        "schema_fields": sorted(preprocessing.INPUT_ROW_FIELDS),
        "blind_prompt_manifest": candidate_metadata["blind_prompt_manifest"],
        "generation_manifest": candidate_metadata["generation_manifest"],
        "row_count": len(final_rows),
        "unique_prompt_count": len(by_prompt),
        "status_counts": dict(sorted(Counter(row["status"] for row in final_rows).items())),
        "unique_prompt_status_counts": dict(sorted(unique_status_counts.items())),
        "output": _artifact(output_input),
        "model_id": SD14_GENERATION.model_id,
        "model_revision": SD14_GENERATION.model_revision,
        "tokenizer": tokenizer_identity,
        "parser_environment": candidate_metadata["parser_environment"],
        "adapter_environment": candidate_metadata["adapter_environment"],
        "candidate_sidecar": candidate_input_record,
        "candidate_sidecar_metadata": _artifact(candidate_metadata_path),
        "selection_evidence": evidence_record,
        "selection_summary": dict(evidence_payload["summary"]),
        "gold_inputs_used": False,
        "local_implementation": local_implementation,
    }
    atomic_write_json(metadata_output, metadata_payload, overwrite=False)
    validated, _metadata = load_jobs(
        generation_manifest, blind_prompt_manifest, output_input, arm=SELECTION_ARM,
        strict_environment=strict_environment)
    if len(validated) != len(final_rows):
        raise attend_and_excite.SubtokenSelectionError("post-write v3 validation row count drifted")
    return {
        "output_sidecar": portable_path(output_input),
        "output_sidecar_sha256": metadata_payload["output"]["sha256"],
        "selection_evidence": evidence_record["path"],
        "selection_evidence_sha256": evidence_record["sha256"],
        **evidence_payload["summary"],
        "row_count": len(final_rows),
    }
