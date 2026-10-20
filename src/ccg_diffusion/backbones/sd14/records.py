"""Records of an SD1.4 run: runtime environment, run lock, metadata ledger and image files.

A run of one shard writes ``<out>/<method>/run_lock.shard_<i>-of-<n>.json``
(the run ``configuration`` and its signature) and appends one metadata row per
image to ``<out>/<method>/<experiment>/metadata.shard_<i>-of-<n>.jsonl``. The
run lock fixes every setting of the run (model, precision, scheduler, CLIP
tokenizer, guidance schedule, dataset, guidance-target file, selected images,
source seal); a resumed run must reproduce it exactly. Paths in new records
are repository-relative.
"""
from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import diffusers
import torch
import transformers
from diffusers import PNDMScheduler, StableDiffusionPipeline
from PIL import Image, ImageStat

from ccg_diffusion.backbones.sd14.inputs import (
    DatasetProvenance,
    GenerationContractError,
    PreparedJob,
    _read_jsonl,
    _require_text,
)
from ccg_diffusion.backbones.sd14.pipeline import SD14_ATTENTION_BLOCK_SCOPE, SD14_EXPECTED_ATTENTION_PROCESSORS
from ccg_diffusion.config import SD14_ARM_PROVIDERS, SD14_GENERATION, SD14_GUIDANCE_SCHEDULE
from ccg_diffusion.guidance.latent_update import GenerationDiagnostics
from ccg_diffusion.guidance.precision import PrecisionProfile
from ccg_diffusion.guidance.predicated_diffusion_loss import PD_LEAK_WEIGHT
from ccg_diffusion.guidance.selective_guidance_boost import BASE_SCALE
from ccg_diffusion.provenance import repo_relative, sha256_file
from ccg_diffusion.provenance import digest as sha256_json
from ccg_diffusion.targets.tokenizer import require_clip_tokenizer_identity

SD14_RUN_LOCK_SCHEMA_VERSION = 4
SD14_METADATA_SCHEMA_VERSION = 4
# Software of the evaluated SD1.4 runs. ``strict=True`` enforces these pins;
# the NVIDIA driver version is recorded, never enforced.
SD14_GENERATOR_RUNTIME_PINS = MappingProxyType({
    "python": "3.12.3",
    "torch": "2.13.0+cu129",
    "diffusers": "0.39.0",
    "transformers": "5.14.1",
    "cuda_runtime": "12.9",
    "cudnn": 92000,
})


# --- runtime environment -------------------------------------------------------------------------------


def _sd14_generator_runtime_actual(*, device: str) -> dict[str, Any]:
    """Read the active SD1.4 generator environment after CUDA preflight."""

    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise GenerationContractError(
            "the accepted SD1.4 six-arm run requires a CUDA generator device"
        )
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise GenerationContractError("cannot query the NVIDIA driver version") from exc
    driver_versions = sorted(
        {line.strip() for line in completed.stdout.splitlines() if line.strip()}
    )
    if len(driver_versions) != 1:
        raise GenerationContractError(
            f"all visible GPUs must share one driver version: {driver_versions!r}"
        )
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "diffusers": diffusers.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "nvidia_driver": driver_versions[0],
    }


def sd14_generator_runtime_environment(*, device: str, strict: bool = True) -> dict[str, Any]:
    """Return the T2I generator environment; ``strict`` requires the pinned software versions."""

    actual = _sd14_generator_runtime_actual(device=device)
    mismatches = {
        name: {"expected": required, "actual": actual.get(name)}
        for name, required in SD14_GENERATOR_RUNTIME_PINS.items()
        if actual.get(name) != required
    }
    if strict and mismatches:
        raise GenerationContractError(
            f"SD1.4 generator runtime does not match the pinned environment: {mismatches}"
        )
    return {
        "schema": "ccg_diffusion.sd14_runtime_environment.v1",
        **actual,
        "pins_enforced": strict,
        "python_executable": str(Path(sys.executable).resolve()),
        "cuda_device_count": torch.cuda.device_count(),
    }


def sd14_tab_generator_runtime_environment(*, device: str) -> dict[str, Any]:
    """Record, without pinning, the TAB SD1.4 runtime."""

    actual = _sd14_generator_runtime_actual(device=device)
    return {
        "schema": "ccg_diffusion.sd14_tab_runtime_environment.v1",
        "verification": "record_only_for_new_tab_experiment",
        **actual,
        "python_executable": str(Path(sys.executable).resolve()),
        "cuda_device_count": torch.cuda.device_count(),
    }


# --- pipeline records ----------------------------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _canonical_scheduler_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Canonicalize order-free diffusers scheduler provenance fields."""

    result = _jsonable(dict(config))
    default_values = result.get("_use_default_values")
    if isinstance(default_values, list):
        result["_use_default_values"] = sorted(default_values)
    return result


def scheduler_record(pipe: StableDiffusionPipeline) -> dict[str, Any]:
    """The PNDM scheduler class, the sha256 of its canonical config and its prediction type."""

    if not isinstance(pipe.scheduler, PNDMScheduler):
        raise GenerationContractError(
            "sd14 experiment requires PNDMScheduler, "
            f"got {type(pipe.scheduler).__name__}"
        )
    config = _canonical_scheduler_config(pipe.scheduler.config)
    prediction_type = config.get("prediction_type")
    if prediction_type != SD14_GENERATION.prediction_type:
        raise GenerationContractError(
            f"sd14 experiment requires {SD14_GENERATION.prediction_type} prediction, "
            f"got {prediction_type!r}"
        )
    return {
        "class": type(pipe.scheduler).__name__,
        "config_sha256": sha256_json(config),
        "prediction_type": prediction_type,
    }


def device_record(device_name: str) -> dict[str, Any]:
    device = torch.device(device_name)
    record: dict[str, Any] = {"type": device.type, "requested": device_name}
    if device.type == "cuda":
        index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        properties = torch.cuda.get_device_properties(index)
        record.update(
            {
                "index": index,
                "name": properties.name,
                "compute_capability": [properties.major, properties.minor],
                "total_memory": properties.total_memory,
            }
        )
    return record


def pipeline_dtype_record(pipe: Any) -> dict[str, Any]:
    """Return the model-component dtype/freeze ledger for provenance."""

    record: dict[str, Any] = {}
    for name in ("unet", "text_encoder", "vae"):
        component = getattr(pipe, name, None)
        if component is None:
            record[name] = None
            continue
        parameter_source = getattr(component, "parameters", None)
        parameters = list(parameter_source()) if callable(parameter_source) else []
        dtypes = sorted(
            {
                str(parameter.dtype).removeprefix("torch.")
                for parameter in parameters
            }
        )
        record[name] = {
            "class": type(component).__name__,
            "parameter_dtypes": dtypes,
            "training": (
                bool(component.training) if hasattr(component, "training") else None
            ),
            "requires_grad": (
                any(parameter.requires_grad for parameter in parameters)
                if parameters
                else None
            ),
        }
    return record


def latent_abs_bound_record(profile: PrecisionProfile, pipe: Any) -> dict[str, Any]:
    """The absolute latent guard of the run: the precision profile's bound (20) is inherited."""

    return {
        "source": "precision_profile",
        "requested_max_latent_abs": profile.max_latent_abs,
        "inherited_max_latent_abs": profile.max_latent_abs,
        "scheduler_init_noise_sigma": float(pipe.scheduler.init_noise_sigma),
        "sigma_multiplier": None,
        "policy": "inherit_precision_profile",
        "reason": "No backbone/method-specific latent guard amendment applies.",
        "resolved_max_latent_abs": profile.max_latent_abs,
    }


def validate_clip_token_contract(
    tokenizer: Any,
    jobs: Sequence[PreparedJob],
) -> tuple[dict[str, int], dict[str, Any], dict[str, dict[str, Any]]]:
    """Record CLIP truncation and reject guidance targets outside the lexical columns.

    Prompts longer than the CLIP window are conditioned under the standard
    diffusers truncation; the per-job truncation flag and the untruncated count
    go into the run lock and the metadata. Guidance-target indices must name a
    lexical column ``[1, token_count - 2]``: index 0 is the CLIP BOS column and
    ``token_count - 1`` the EOS column.
    """
    model_max_length = int(tokenizer.model_max_length)
    if model_max_length != 77:
        raise GenerationContractError(
            f"CLIP tokenizer must have model_max_length=77, got {model_max_length}"
        )

    token_counts: dict[str, int] = {}
    truncation: dict[str, dict[str, Any]] = {}
    for prepared in jobs:
        job = prepared.job
        encoded = tokenizer(
            job.prompt,
            add_special_tokens=True,
            padding=False,
            truncation=False,
        )
        input_ids = encoded["input_ids"]
        if input_ids and isinstance(input_ids[0], list):
            if len(input_ids) != 1:
                raise GenerationContractError(
                    f"unexpected batched tokenization for {job.prompt_id}"
                )
            input_ids = input_ids[0]
        untruncated_token_count = len(input_ids)
        token_count = min(untruncated_token_count, model_max_length)
        token_counts[job.prompt_id] = token_count
        truncation[job.prompt_id] = {
            "truncated": untruncated_token_count > model_max_length,
            "untruncated_token_count": untruncated_token_count,
        }

        if prepared.provider is None:
            continue
        indices = [*job.attention_exist_indices]
        for pairs in (
            job.attention_corr_indices,
            job.attention_leak_indices,
            job.attention_possession_indices,
        ):
            indices.extend(index for pair in pairs for index in pair)
        # Index zero is CLIP BOS and token_count - 1 is EOS. Neither may
        # participate in a semantic PD predicate.
        usable_max_index = token_count - 2
        invalid = sorted({index for index in indices if not 1 <= index <= usable_max_index})
        if invalid:
            raise GenerationContractError(
                f"PD indices do not name lexical CLIP tokens for {job.prompt_id}: "
                f"invalid={invalid}, usable window is [1, {usable_max_index}] "
                f"(index 0 is CLIP BOS and index {token_count - 1} is the EOS column); "
                f"clip_token_count={token_count}, "
                f"untruncated_token_count={untruncated_token_count}"
            )

    tokenizer_info: dict[str, Any] = {
        "class": type(tokenizer).__name__,
        "model_max_length": model_max_length,
    }
    try:
        runtime_identity = require_clip_tokenizer_identity(
            tokenizer,
            expected=SD14_GENERATION.tokenizer_identity(),
        )
    except ValueError as exc:
        raise GenerationContractError(str(exc)) from exc
    tokenizer_info["pinned_identity"] = runtime_identity
    return token_counts, tokenizer_info, truncation


# --- run lock and metadata -----------------------------------------------------------------------------


def shard_suffix(shard_index: int, num_shards: int) -> str:
    return f"shard_{shard_index:05d}-of-{num_shards:05d}"


def run_lock_path(
    out_dir: Path, method: str, shard_index: int, num_shards: int
) -> Path:
    return out_dir / method / f"run_lock.{shard_suffix(shard_index, num_shards)}.json"


def shard_metadata_path(
    out_dir: Path,
    method: str,
    experiment: str,
    shard_index: int,
    num_shards: int,
) -> Path:
    return (
        out_dir
        / method
        / experiment
        / f"metadata.{shard_suffix(shard_index, num_shards)}.jsonl"
    )


def _job_keys(jobs: Sequence[PreparedJob]) -> list[list[Any]]:
    return [[item.job.experiment, item.job.prompt_id, item.job.seed] for item in jobs]


def run_configuration(
    *,
    method: str,
    runtime: Any,
    device: Mapping[str, Any],
    local_files_only: bool,
    tokenizer: Mapping[str, Any],
    token_counts: Mapping[str, int],
    clip_truncation: Mapping[str, Mapping[str, Any]],
    prompt_manifest: Path,
    dataset: DatasetProvenance,
    generator_runtime_environment: Mapping[str, Any],
    target_file: Path | None,
    index_failure_policy: str,
    num_shards: int,
    shard_index: int,
    selected_jobs: Sequence[PreparedJob],
    batch_padding_policy: str,
    source: Mapping[str, Any],
    runner_file: Path,
    kernel_file: Path,
    batch_subset: Sequence[int] | None,
    method_record: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """The run-lock ``configuration`` of one shard; its canonical sha256 is the run's signature.

    ``runtime`` is the ``driver.SD14Runtime`` of the run (precision profile,
    latent bound, determinism, scheduler and pipeline dtype records);
    ``source`` is the source seal of the component files.
    """

    predicated = SD14_ARM_PROVIDERS[method] is not None
    schedule = SD14_GUIDANCE_SCHEDULE
    target_metadata = (
        target_file.with_name(target_file.name + ".meta.json") if target_file is not None else None
    )
    return {
        "schema_version": SD14_RUN_LOCK_SCHEMA_VERSION,
        "backbone": "sd14",
        "method": method,
        "model_id": SD14_GENERATION.model_id,
        "model_revision": SD14_GENERATION.model_revision,
        "local_files_only": local_files_only,
        "dtype": runtime.precision.model_dtype_name,
        "precision_profile_requested": SD14_GENERATION.precision_profile,
        "precision_profile_resolved": runtime.precision.name,
        "precision_profile": runtime.precision.as_record(),
        "latent_abs_bound": dict(runtime.latent_abs_bound),
        "execution_mode": SD14_GENERATION.execution_mode,
        "batch_size": SD14_GENERATION.batch_size,
        "physical_batch_size": SD14_GENERATION.batch_size if predicated else None,
        "batch_padding_policy": batch_padding_policy if predicated else None,
        "device": dict(device),
        "pipeline_dtypes": dict(runtime.pipeline_dtypes),
        "scheduler": dict(runtime.scheduler),
        "tokenizer": dict(tokenizer),
        "clip_token_counts_sha256": sha256_json(token_counts),
        "clip_truncation_sha256": sha256_json(clip_truncation),
        "clip_truncated_prompt_count": sum(
            1 for record in clip_truncation.values() if record["truncated"]
        ),
        "steps": SD14_GENERATION.steps,
        "guidance_scale": SD14_GENERATION.guidance_scale,
        "height": SD14_GENERATION.height,
        "width": SD14_GENERATION.width,
        "negative_prompt": None,
        "safety_checker": False,
        "attention_guidance_level": "token" if predicated else None,
        "attention_res": SD14_GENERATION.attention_res if predicated else None,
        "attention_block_scope": SD14_ATTENTION_BLOCK_SCOPE if predicated else None,
        "guided_scheduler_input_scaling": True if predicated else None,
        "expected_attention_processor_count": (
            SD14_GENERATION.expected_attention_processor_count if predicated else None
        ),
        "expected_attention_processors": list(SD14_EXPECTED_ATTENTION_PROCESSORS) if predicated else None,
        "existence_normalization": None,
        "max_iter_to_alter": schedule.guided_steps if predicated else None,
        "binding_loss": "pd" if predicated else None,
        "leak_weight": PD_LEAK_WEIGHT if predicated else None,
        "scale_factor": BASE_SCALE if predicated else None,
        "scale_range": list(schedule.scale_range) if predicated else None,
        "threshold_steps": list(schedule.refinement_steps) if predicated else None,
        "refinement_updates_per_threshold": schedule.refinement_updates if predicated else 0,
        "strict_finite": SD14_GENERATION.finite_guard == "strict",
        "finite_guard": SD14_GENERATION.finite_guard,
        "determinism": dict(runtime.determinism),
        "prompt_manifest": repo_relative(prompt_manifest),
        "prompt_manifest_sha256": sha256_file(prompt_manifest),
        **dataset.run_record(),
        "generator_runtime_environment": dict(generator_runtime_environment),
        "pd_sidecar": repo_relative(target_file) if target_file is not None else None,
        "pd_sidecar_sha256": sha256_file(target_file) if target_file is not None else None,
        "pd_sidecar_metadata_sha256": (
            sha256_file(target_metadata)
            if target_metadata is not None and target_metadata.is_file()
            else None
        ),
        "provider": SD14_ARM_PROVIDERS[method],
        "index_failure_policy": index_failure_policy,
        "guided_job_count": sum(1 for item in selected_jobs if item.guidance_applied),
        "guided_job_keys_sha256": sha256_json(_job_keys([item for item in selected_jobs if item.guidance_applied])),
        "stock_vanilla_fallback_job_count": sum(
            1 for item in selected_jobs if item.guidance_mode == "stock_vanilla_fallback"
        ),
        "stock_vanilla_fallback_job_keys_sha256": sha256_json(
            _job_keys([item for item in selected_jobs if item.guidance_mode == "stock_vanilla_fallback"])
        ),
        "num_shards": num_shards,
        "shard_index": shard_index,
        "selected_job_count": len(selected_jobs),
        "selected_job_keys_sha256": sha256_json(_job_keys(selected_jobs)),
        "batch_subset": list(batch_subset) if batch_subset is not None else None,
        "source_provenance_schema": source["schema"],
        "repo_git_commit": source["repo_git_commit"],
        "component_sha256": dict(source["component_sha256"]),
        "components_match_repo_git_commit": source["components_match_repo_git_commit"],
        "dirty": source["dirty"],
        "runner_sha256": sha256_file(runner_file),
        "kernel_sha256": sha256_file(kernel_file),
        "method_record": dict(method_record) if method_record is not None else None,
    }


def ensure_run_lock(
    path: Path, configuration: Mapping[str, Any], *, resume: bool
) -> dict[str, Any]:
    signature = sha256_json(configuration)
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise GenerationContractError(f"invalid existing run lock: {path}") from exc
        if not resume:
            raise GenerationContractError(
                f"run lock already exists; resume only the identical run: {path}"
            )
        if existing.get("configuration_signature") != signature:
            raise GenerationContractError(
                f"resume configuration does not match existing run lock: {path}"
            )
        if existing.get("configuration") != dict(configuration):
            raise GenerationContractError(
                f"run lock payload differs despite its signature: {path}"
            )
        return existing

    if resume:
        raise GenerationContractError(f"cannot resume without run lock: {path}")
    payload = {
        "configuration_signature": signature,
        "configuration": dict(configuration),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
    return payload


def load_metadata_records(
    paths: Iterable[Path], *, expected_signature: str
) -> dict[str, dict[str, Any]]:
    """Recorded metadata rows of a run, keyed by ``image_path``."""

    records: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        for row in _read_jsonl(path, label="generation metadata"):
            image = _require_text(
                row.get("image_path"), location=f"metadata row in {path}.image_path"
            )
            if image in records:
                raise GenerationContractError(f"duplicate metadata image_path: {image}")
            if row.get("configuration_signature") != expected_signature:
                raise GenerationContractError(
                    f"metadata configuration signature mismatch at {path}: {image}"
                )
            records[image] = row
    return records


def image_is_valid(path: Path, *, expected_size: tuple[int, int]) -> bool:
    if not path.is_file():
        return False
    try:
        with Image.open(path) as image:
            if image.format != "PNG" or image.size != expected_size:
                return False
            image.load()
            stat = ImageStat.Stat(image.convert("RGB"))
    except (OSError, ValueError):
        return False
    mean = sum(stat.mean) / len(stat.mean)
    std = sum(stat.stddev) / len(stat.stddev)
    return 1.0 <= mean <= 254.0 and std >= 1.0


def save_image_atomic(
    image: Image.Image, path: Path, *, expected_size: tuple[int, int]
) -> None:
    """Save a PNG with PIL defaults (the PNG bytes are compared) and validate it before renaming."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        image.save(temporary, format="PNG")
        if not image_is_valid(temporary, expected_size=expected_size):
            raise GenerationContractError(f"generated image failed validation: {path}")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def append_metadata(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(dict(payload), ensure_ascii=False, sort_keys=True) + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def metadata_payload(
    prepared: PreparedJob,
    path: Path,
    diagnostics: GenerationDiagnostics,
    *,
    method: str,
    runtime: Any,
    dataset: DatasetProvenance,
    configuration: Mapping[str, Any],
    configuration_signature: str,
    run_lock: Path,
    run_lock_sha256: str,
    clip_token_count: int,
    clip_truncated: bool,
    clip_untruncated_token_count: int,
    physical_batch_position: int,
) -> dict[str, Any]:
    """One image's metadata row; run-level fields are taken from the run-lock ``configuration``."""

    job = prepared.job
    guidance_applied = prepared.guidance_applied
    fallback_used = prepared.guidance_mode == "stock_vanilla_fallback"
    schedule = SD14_GUIDANCE_SCHEDULE
    return {
        "metadata_schema_version": SD14_METADATA_SCHEMA_VERSION,
        **job.__dict__,
        **dataset.image_record(job),
        "physical_batch_position": physical_batch_position,
        "method": method,
        "provider": prepared.provider,
        "extraction_status": prepared.extraction_status,
        "extraction_error": prepared.extraction_error,
        "sidecar_row_sha256": prepared.sidecar_row_sha256,
        "guidance_mode": prepared.guidance_mode,
        "guidance_applied": guidance_applied,
        "fallback_used": fallback_used,
        "fallback_reason": prepared.extraction_status if fallback_used else None,
        "effective_generation_path": prepared.guidance_mode,
        "index_failure_policy": configuration["index_failure_policy"],
        "prompt_sha256": prepared.prompt_sha256,
        "program_sha256": prepared.program_sha256,
        "pd_indices_sha256": prepared.pd_indices_sha256,
        "image_path": repo_relative(path),
        "image_sha256": sha256_file(path),
        "backbone": "sd14",
        "model_id": SD14_GENERATION.model_id,
        "model_revision": SD14_GENERATION.model_revision,
        "dtype": runtime.precision.model_dtype_name,
        "precision_profile_requested": SD14_GENERATION.precision_profile,
        "precision_profile_resolved": runtime.precision.name,
        "precision_profile": runtime.precision.as_record(),
        "latent_abs_bound": dict(runtime.latent_abs_bound),
        "execution_mode": SD14_GENERATION.execution_mode,
        "batch_size": SD14_GENERATION.batch_size,
        "batch_size_requested": SD14_GENERATION.batch_size,
        "batch_size_effective": diagnostics.effective_batch_size,
        "pipeline_dtypes": dict(runtime.pipeline_dtypes),
        "scheduler": dict(runtime.scheduler),
        "tokenizer": dict(configuration["tokenizer"]),
        "clip_token_count": clip_token_count,
        "clip_truncated": clip_truncated,
        "clip_untruncated_token_count": clip_untruncated_token_count,
        "steps": SD14_GENERATION.steps,
        "guidance_scale": SD14_GENERATION.guidance_scale,
        "height": SD14_GENERATION.height,
        "width": SD14_GENERATION.width,
        "negative_prompt": None,
        "safety_checker": False,
        "attention_guidance_level": "token" if guidance_applied else None,
        "attention_res": SD14_GENERATION.attention_res if guidance_applied else None,
        "attention_block_scope": SD14_ATTENTION_BLOCK_SCOPE if guidance_applied else None,
        "guided_scheduler_input_scaling": (
            diagnostics.guided_scheduler_input_scaling if guidance_applied else None
        ),
        "captured_attention_processors": (
            diagnostics.captured_attention_processors if guidance_applied else []
        ),
        "captured_attention_processor_count": (
            diagnostics.captured_attention_processor_count if guidance_applied else 0
        ),
        "existence_normalization": None,
        "max_iter_to_alter": schedule.guided_steps if guidance_applied else None,
        "binding_loss": "pd" if guidance_applied else None,
        "leak_weight": PD_LEAK_WEIGHT if guidance_applied else None,
        "scale_factor": BASE_SCALE if guidance_applied else None,
        "scale_range": list(schedule.scale_range) if guidance_applied else None,
        "threshold_steps": list(schedule.refinement_steps) if guidance_applied else None,
        "refinement_updates_per_threshold": schedule.refinement_updates if guidance_applied else 0,
        # Actual per-image execution: the kernel that generated this image and
        # whether its strict finite checks ran.
        "generation_kernel": diagnostics.generation_kernel,
        "generation_batch_id": diagnostics.generation_batch_id,
        "strict_finite": diagnostics.strict_finite_applied,
        "initial_latent_sha256": diagnostics.initial_latent_sha256,
        "generation_diagnostics": diagnostics.as_dict(),
        "prompt_manifest_sha256": configuration["prompt_manifest_sha256"],
        "pd_sidecar_sha256": configuration["pd_sidecar_sha256"],
        "pd_sidecar_metadata_sha256": configuration["pd_sidecar_metadata_sha256"],
        "configuration_signature": configuration_signature,
        "run_lock_path": repo_relative(run_lock),
        "run_lock_sha256": run_lock_sha256,
        "num_shards": configuration["num_shards"],
        "shard_index": configuration["shard_index"],
    }


# --- devices -------------------------------------------------------------------------------------------


def synchronize_device(device_name: str) -> None:
    """Synchronize only the selected CUDA device for wall-time measurement."""

    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak_memory(device_name: str) -> None:
    """Start a per-generation CUDA allocator peak measurement."""

    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory(device_name: str) -> tuple[int | None, int | None]:
    """Return allocator peaks, or nulls for a non-CUDA device."""

    device = torch.device(device_name)
    if device.type != "cuda":
        return None, None
    return (
        int(torch.cuda.max_memory_allocated(device)),
        int(torch.cuda.max_memory_reserved(device)),
    )


def canonical_gpu_uuid(raw: str) -> str:
    """Normalize torch's bare/full GPU UUID without accepting aliases or MIG IDs."""

    match = re.fullmatch(
        r"(?:GPU-)?([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})",
        raw,
    )
    if match is None:
        raise GenerationContractError("actual torch GPU UUID has an unsupported or malformed representation")
    return f"GPU-{match.group(1).lower()}"


def current_gpu_uuid() -> str:
    """The canonical UUID of CUDA device 0 of this process."""

    return canonical_gpu_uuid(str(torch.cuda.get_device_properties(0).uuid))
