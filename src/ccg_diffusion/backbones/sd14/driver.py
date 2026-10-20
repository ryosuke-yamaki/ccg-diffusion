"""SD1.4 batch driver: one shard of one arm over a sealed image manifest.

``run(SD14Run(...))`` verifies the dataset and the guidance-target file,
seals the source files to Git ``HEAD``, keeps the evaluated physical batches
(``index % num_shards`` sharding, consecutive guided/stock grouping in
batches of 4, guided batches padded with their last real image), calls the
guided denoising kernel (or ``SD14Run.batch_fn``), validates every image's
diagnostics and appends the per-image metadata under a run lock.
``plan_run`` is the CPU-only planning half.

Arms (``config.SD14_ARM_PROVIDERS``): ``sd14_pd_holccg`` (CCG Diffusion on
T2I-CompBench++, with per-prompt ``image_guidance``/``count_guidance``),
``sd14_pd_spacy`` (Predicated Diffusion baseline), ``sd14_vanilla`` and
``sd14_tied_oracle`` (TAB Vanilla and VS-TIED + Gold-visual setup, stock
denoising). The TAB CCG Diffusion and TAB Predicated Diffusion runners call
``pipeline.generate_predicated_batch`` directly with ``prepare_sd14_runtime``
and ``initial_latents``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
from tqdm import tqdm

from ccg_diffusion.backbones.jobs import Job, image_path
from ccg_diffusion.backbones.sd14.inputs import (
    SD14_PD_SOURCE_PROVENANCE_SCHEMA,
    DatasetProvenance,
    GenerationContractError,
    PreparedJob,
    T2ICBDatasetProvenance,
    load_prepared_jobs,
)
from ccg_diffusion.backbones.sd14.pipeline import (
    SD14_EXPECTED_ATTENTION_PROCESSORS,
    generate_predicated_batch,
    generate_stock_batch,
    load_pipe,
    tensor_sha256,
)
from ccg_diffusion.backbones.sd14.records import (
    append_metadata,
    device_record,
    ensure_run_lock,
    image_is_valid,
    latent_abs_bound_record,
    load_metadata_records,
    metadata_payload,
    peak_memory,
    pipeline_dtype_record,
    reset_peak_memory,
    run_configuration,
    run_lock_path,
    save_image_atomic,
    scheduler_record,
    sd14_generator_runtime_environment,
    sd14_tab_generator_runtime_environment,
    shard_metadata_path,
    shard_suffix,
    synchronize_device,
    validate_clip_token_contract,
)
from ccg_diffusion.config import SD14_ARM_PROVIDERS, SD14_CCG_ARM, SD14_GENERATION, SD14_GUIDANCE_SCHEDULE
from ccg_diffusion.guidance.count_loss import CountGuidance
from ccg_diffusion.guidance.latent_update import GenerationDiagnostics
from ccg_diffusion.guidance.precision import (
    PROFILES,
    PrecisionProfile,
    configure_determinism,
    freeze_pipeline_for_latent_grad,
    resolve_precision_profile,
)
from ccg_diffusion.guidance.predicated_diffusion_loss import PAIR_BOUNDARY_MARGIN
from ccg_diffusion.guidance.selective_guidance_boost import BASE_SCALE, GuidanceRoute, ImageGuidance
from ccg_diffusion.provenance import (
    SourceSealError,
    canonical_json,
    git_head,
    git_head_components_record,
    repo_relative,
    repository_root,
    sha256_file,
    sha256_text,
)

# The ccg_diffusion files the driver imports (module and function level, ``__init__`` excluded),
# sealed into every run lock; tests/test_guided_denoising_loop.py checks it against the import closure.
SD14_DRIVER_COMPONENT_FILES = (
    "src/ccg_diffusion/backbones/jobs.py",
    "src/ccg_diffusion/backbones/sd14/driver.py",
    "src/ccg_diffusion/backbones/sd14/inputs.py",
    "src/ccg_diffusion/backbones/sd14/pipeline.py",
    "src/ccg_diffusion/backbones/sd14/records.py",
    "src/ccg_diffusion/config.py",
    "src/ccg_diffusion/guidance/attention.py",
    "src/ccg_diffusion/guidance/count_loss.py",
    "src/ccg_diffusion/guidance/latent_update.py",
    "src/ccg_diffusion/guidance/precision.py",
    "src/ccg_diffusion/guidance/predicated_diffusion_loss.py",
    "src/ccg_diffusion/guidance/selective_guidance_boost.py",
    "src/ccg_diffusion/guidance/spatial_relation_loss.py",
    "src/ccg_diffusion/guidance/spatial_relations.py",
    "src/ccg_diffusion/parsing/ccg.py",
    "src/ccg_diffusion/parsing/sentences.py",
    "src/ccg_diffusion/provenance.py",
    "src/ccg_diffusion/targets/guidance_targets.py",
    "src/ccg_diffusion/targets/lexicon.py",
    "src/ccg_diffusion/targets/pd_indices.py",
    "src/ccg_diffusion/targets/tokenizer.py",
)
SD14_PD_BATCH_PADDING_POLICY = "repeat_last_real_job_discard_padded_outputs"
_STOCK_ATTENTION_PROCESSORS_ATTR = "_ccg_diffusion_stock_attention_processors"


@dataclass(frozen=True)
class SD14Run:
    """One shard of one SD1.4 arm.

    ``image_guidance``/``count_guidance`` map each base prompt ID to its route
    and count guidance (CCG Diffusion on T2I-CompBench++); ``batch_fn``
    replaces ``generate_predicated_batch`` (same arguments) and
    ``selected_image_ids`` selects an ordered sub-population before sharding.
    ``batches`` runs only these physical-batch indices of the shard; planning,
    batch membership and ``generation_batch_id`` numbering are unchanged.
    ``extra_record`` is written under the run-lock key ``method_record``.
    """

    method: str
    prompt_manifest: Path
    out_dir: Path
    dataset: DatasetProvenance
    target_file: Path | None = None
    index_failure_policy: str = "error"
    num_shards: int = 1
    shard_index: int = 0
    expected_total: int | None = None
    device: str = "cuda:0"
    local_files_only: bool = True
    resume: bool = False
    strict_runtime: bool = True
    allow_dirty: bool = False
    extra_component_files: tuple[str, ...] = ()
    image_guidance: Mapping[str, ImageGuidance] | None = None
    count_guidance: Mapping[str, CountGuidance] | None = None
    batch_fn: Callable[..., Any] | None = None
    selected_image_ids: tuple[str, ...] | None = None
    batches: tuple[int, ...] | None = None
    extra_record: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class RunPlan:
    """The images of one shard (manifest order), their physical batches and the sealed source files."""

    jobs: list[PreparedJob]
    batches: list[list[PreparedJob]]
    component_files: tuple[str, ...]


@dataclass(frozen=True)
class SD14Runtime:
    """The loaded SD1.4 pipeline and the records of its precision, determinism and scheduler."""

    pipe: StableDiffusionPipeline
    precision: PrecisionProfile
    determinism: dict[str, Any]
    latent_abs_bound: dict[str, Any]
    scheduler: dict[str, Any]
    pipeline_dtypes: dict[str, Any]


def prepare_sd14_runtime(device: str, *, local_files_only: bool = True) -> SD14Runtime:
    """Configure determinism, then load and freeze the pinned SD1.4 pipeline in BF16.

    ``configure_determinism`` runs before the pipeline is loaded and before any
    CUDA matmul of the process (it sets process-global cuBLAS/cuDNN state).
    """

    precision = resolve_precision_profile("bf16_stable", device=torch.device(device))
    if precision != PROFILES["bf16_stable"] or precision.pair_boundary_margin != PAIR_BOUNDARY_MARGIN:
        raise GenerationContractError("SD1.4 generation requires the bf16_stable precision profile")
    determinism = configure_determinism(precision)
    pipe = load_pipe(
        SD14_GENERATION.model_id,
        device,
        precision.model_dtype,
        revision=SD14_GENERATION.model_revision,
        local_files_only=local_files_only,
    )
    latent_abs_bound = latent_abs_bound_record(precision, pipe)
    freeze_pipeline_for_latent_grad(pipe)
    pipeline_dtypes = pipeline_dtype_record(pipe)
    scheduler = scheduler_record(pipe)
    return SD14Runtime(pipe, precision, determinism, latent_abs_bound, scheduler, pipeline_dtypes)


def initial_latents(
    pipe: StableDiffusionPipeline,
    job: Job,
    *,
    device: str,
    dtype: torch.dtype,
    height: int,
    width: int,
    steps: int,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Return raw noise, scheduler-scaled noise, and the scaled-noise digest."""
    pipe.scheduler.set_timesteps(steps, device=device)
    shape = (
        1,
        pipe.unet.config.in_channels,
        height // pipe.vae_scale_factor,
        width // pipe.vae_scale_factor,
    )
    generator = torch.Generator(device=device).manual_seed(job.seed)
    raw = torch.randn(shape, generator=generator, device=device, dtype=dtype)
    scaled = raw * pipe.scheduler.init_noise_sigma
    return raw, scaled, tensor_sha256(scaled)


def source_prompt_id(job: Job) -> str:
    """Return the base prompt ID of a seed-suffixed image ID (``<prompt_id>_s<seed>``)."""

    suffix = f"_s{job.seed}"
    return job.prompt_id[:-len(suffix)] if job.prompt_id.endswith(suffix) else job.prompt_id


def _is_predicated(method: str) -> bool:
    return SD14_ARM_PROVIDERS[method] is not None


# --- physical batches ----------------------------------------------------------------------------------


def _collect_generation_batch(
    pending: Sequence[PreparedJob], cursor: int, batch_size: int
) -> list[PreparedJob]:
    """Return the next consecutive batch sharing one generation path.

    Jobs are grouped, in manifest order, while they stay on the same path
    (guided vs stock), so a batch never mixes guidance with the stock denoiser
    and a trailing group of one to three jobs is simply a shorter batch.
    """

    first = pending[cursor]
    if batch_size <= 1:
        return [first]
    batch: list[PreparedJob] = []
    while (
        cursor + len(batch) < len(pending)
        and len(batch) < batch_size
        and pending[cursor + len(batch)].guidance_applied == first.guidance_applied
    ):
        batch.append(pending[cursor + len(batch)])
    return batch


def plan_generation_batches(
    jobs: Sequence[PreparedJob], batch_size: int
) -> list[list[PreparedJob]]:
    """Fix the batch composition over the full selected job list.

    The plan is a pure function of (jobs, batch_size): it never looks at which
    rows are already recorded, so a resume reproduces exactly the batch
    boundaries of the interrupted run.
    """

    batches: list[list[PreparedJob]] = []
    cursor = 0
    while cursor < len(jobs):
        batch = _collect_generation_batch(jobs, cursor, batch_size)
        batches.append(batch)
        cursor += len(batch)
    return batches


def pad_physical_batch(
    prepared_jobs: Sequence[PreparedJob], *, physical_batch_size: int
) -> tuple[list[PreparedJob], int]:
    """Pad one uniform logical batch without changing any real job identity.

    The last real job is repeated only in ephemeral denoiser inputs. Padded
    images and diagnostics are discarded, while the first ``len(prepared)``
    tracks retain their manifest order and independently seeded latents.
    """

    jobs = list(prepared_jobs)
    if not jobs:
        raise GenerationContractError("cannot pad an empty generation batch")
    if physical_batch_size < 1 or len(jobs) > physical_batch_size:
        raise GenerationContractError(
            "invalid SD1.4 PD physical batch request: "
            f"real={len(jobs)}, physical={physical_batch_size}"
        )
    padding_count = physical_batch_size - len(jobs)
    jobs.extend([jobs[-1]] * padding_count)
    return jobs, padding_count


def _remember_stock_attention_processors(pipe: StableDiffusionPipeline) -> None:
    """Snapshot the pipeline's original attention processors exactly once."""

    if hasattr(pipe, _STOCK_ATTENTION_PROCESSORS_ATTR):
        return
    processors = getattr(pipe.unet, "attn_processors", None)
    if not isinstance(processors, Mapping) or not processors:
        raise GenerationContractError(
            "pipeline UNet does not expose its stock attention processors"
        )
    setattr(pipe, _STOCK_ATTENTION_PROCESSORS_ATTR, dict(processors))


def _restore_stock_attention_processors(pipe: StableDiffusionPipeline) -> None:
    """Restore the exact processors captured before the guidance processors were registered."""

    _remember_stock_attention_processors(pipe)
    processors = getattr(pipe, _STOCK_ATTENTION_PROCESSORS_ATTR)
    setter = getattr(pipe.unet, "set_attn_processor", None)
    if not callable(setter):
        raise GenerationContractError(
            "cannot restore the stock attention processors for parser fallback"
        )
    # diffusers consumes mapping entries while installing them, so pass a copy.
    setter(dict(processors))


def _validate_guided_diagnostics(diagnostics: GenerationDiagnostics) -> None:
    """Require the guidance schedule, the five 16x16 processors and no finite-guard activity."""

    finite_failures = (
        diagnostics.finite_guard_nonfinite_loss
        + diagnostics.finite_guard_nonfinite_gradient
        + diagnostics.finite_guard_nonfinite_candidate
    )
    schedule = SD14_GUIDANCE_SCHEDULE
    expected_outer = min(SD14_GENERATION.steps, schedule.guided_steps)
    expected_refinements = sum(
        schedule.refinement_updates
        for step in set(schedule.refinement_steps)
        if 0 <= step < expected_outer
    )
    if diagnostics.outer_updates_attempted != expected_outer:
        raise GenerationContractError(
            "unexpected outer guidance count: "
            f"{diagnostics.outer_updates_attempted} != {expected_outer}"
        )
    if diagnostics.refinement_updates_attempted != expected_refinements:
        raise GenerationContractError(
            "unexpected refinement count: "
            f"{diagnostics.refinement_updates_attempted} != {expected_refinements}"
        )
    if not diagnostics.guided_scheduler_input_scaling:
        raise GenerationContractError(
            "guided UNet did not apply scheduler model-input scaling"
        )
    expected_names = set(SD14_EXPECTED_ATTENTION_PROCESSORS)
    captured_names = set(diagnostics.captured_attention_processors)
    if (
        diagnostics.captured_attention_processor_count
        != SD14_GENERATION.expected_attention_processor_count
        or captured_names != expected_names
    ):
        raise GenerationContractError(
            "unexpected SD1.4 16x16 attention processors: "
            f"captured={sorted(captured_names)}, "
            f"expected={sorted(expected_names)}"
        )
    if finite_failures:
        raise GenerationContractError(
            "strict PD generation reported finite-guard activity"
        )


def generate_image_batch(
    runtime: SD14Runtime,
    prepared_jobs: Sequence[PreparedJob],
    *,
    request: SD14Run,
    physical_batch_size: int,
    device: str,
) -> list[tuple[Image.Image, GenerationDiagnostics]]:
    """Generate one physical batch with per-image ledgers; padded images are discarded.

    A batch is either entirely guided (``generate_predicated_batch`` or
    ``request.batch_fn``) or entirely stock (``generate_stock_batch``: Vanilla
    images and parser fallbacks); the two paths never mix inside one batch.
    Batches of the guided arms are padded to ``physical_batch_size``.
    """

    pipe = runtime.pipe
    real_job_count = len(prepared_jobs)
    runtime_jobs = list(prepared_jobs)
    padding_count = 0
    if _is_predicated(request.method):
        runtime_jobs, padding_count = pad_physical_batch(
            runtime_jobs,
            physical_batch_size=physical_batch_size,
        )
    guided = runtime_jobs[0].guidance_applied
    if any(prepared.guidance_applied != guided for prepared in runtime_jobs):
        raise GenerationContractError(
            "batched inference cannot mix guided and stock jobs"
        )

    scaled_tracks: list[torch.Tensor] = []
    diagnostics: list[GenerationDiagnostics] = []
    for prepared in runtime_jobs:
        job = prepared.job
        torch.manual_seed(job.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(job.seed)
        _, scaled, latent_digest = initial_latents(
            pipe,
            job,
            device=device,
            dtype=torch.float32,
            height=SD14_GENERATION.height,
            width=SD14_GENERATION.width,
            steps=SD14_GENERATION.steps,
        )
        scaled_tracks.append(scaled)
        diagnostics.append(
            GenerationDiagnostics(initial_latent_sha256=latent_digest)
        )

    if not guided:
        # Stock path: attention capture, losses and latent updates never run.
        # A guided arm may still carry the capture processors of an earlier
        # guided batch, so restore the stock processors first.
        if _is_predicated(request.method):
            _restore_stock_attention_processors(pipe)
        output = generate_stock_batch(
            pipe,
            [prepared.job for prepared in runtime_jobs],
            device=device,
            steps=SD14_GENERATION.steps,
            guidance_scale=SD14_GENERATION.guidance_scale,
            initial_latents=torch.cat(scaled_tracks),
            diagnostics=diagnostics,
            strict_finite=SD14_GENERATION.finite_guard == "strict",
        )
        if len(output.images) != len(runtime_jobs):
            raise GenerationContractError(
                f"expected {len(runtime_jobs)} images, got {len(output.images)}"
            )
        for record, expected_digest in zip(
            diagnostics,
            (tensor_sha256(track) for track in scaled_tracks),
        ):
            if record.initial_latent_sha256 != expected_digest:
                raise GenerationContractError(
                    "batched generation kernel changed the initial latent contract"
                )
        generated = list(zip(output.images, diagnostics))[:real_job_count]
        for _image, record in generated:
            record.effective_batch_size = len(runtime_jobs)
            record.valid_batch_size = real_job_count
            record.batch_padding_count = padding_count
        return generated

    _remember_stock_attention_processors(pipe)
    guidance_options: dict[str, Any] = {}
    if request.image_guidance is not None:
        guidance_options["image_guidance"] = [
            request.image_guidance[source_prompt_id(prepared.job)] for prepared in runtime_jobs
        ]
    if request.count_guidance is not None:
        guidance_options["count_guidance"] = [
            request.count_guidance[source_prompt_id(prepared.job)] for prepared in runtime_jobs
        ]
    schedule = SD14_GUIDANCE_SCHEDULE
    batch_generator = request.batch_fn or generate_predicated_batch
    output = batch_generator(
        pipe,
        [prepared.job for prepared in runtime_jobs],
        device=device,
        steps=SD14_GENERATION.steps,
        guidance_scale=SD14_GENERATION.guidance_scale,
        height=SD14_GENERATION.height,
        width=SD14_GENERATION.width,
        attention_res=SD14_GENERATION.attention_res,
        max_iter_to_alter=schedule.guided_steps,
        scale_factor=BASE_SCALE,
        scale_range=schedule.scale_range,
        thresholds={step: 0.0 for step in schedule.refinement_steps},
        refinement_updates_per_threshold=schedule.refinement_updates,
        initial_latents=torch.cat(scaled_tracks),
        diagnostics=diagnostics,
        strict_finite=SD14_GENERATION.finite_guard == "strict",
        max_update_ratio=runtime.precision.max_update_ratio,
        max_latent_abs=runtime.precision.max_latent_abs,
        **guidance_options,
    )
    if len(output.images) != len(runtime_jobs):
        raise GenerationContractError(
            f"expected {len(runtime_jobs)} images, got {len(output.images)}"
        )
    for record, expected_digest in zip(
        diagnostics,
        (tensor_sha256(track) for track in scaled_tracks),
    ):
        if record.initial_latent_sha256 != expected_digest:
            raise GenerationContractError(
                "batched generation kernel changed the initial latent contract"
            )
        _validate_guided_diagnostics(record)
    generated = list(zip(output.images, diagnostics))[:real_job_count]
    for _image, record in generated:
        record.effective_batch_size = len(runtime_jobs)
        record.valid_batch_size = real_job_count
        record.batch_padding_count = padding_count
    return generated


# --- request validation and planning -------------------------------------------------------------------


def _validate_request(request: SD14Run) -> None:  # noqa: C901
    if request.method not in SD14_ARM_PROVIDERS:
        raise GenerationContractError(f"unsupported method: {request.method}")
    t2i = isinstance(request.dataset, T2ICBDatasetProvenance)
    if _is_predicated(request.method) and not t2i:
        raise GenerationContractError(f"{request.method} requires T2I-CompBench++ dataset provenance")
    if not _is_predicated(request.method) and t2i:
        raise GenerationContractError(f"{request.method} requires TAB stock-baseline provenance")
    if request.index_failure_policy == "stock_vanilla" and request.method != "sd14_pd_spacy":
        raise GenerationContractError("the stock_vanilla fallback is defined only for sd14_pd_spacy")
    if request.prompt_manifest.resolve() != request.dataset.prompt_manifest_path.resolve():
        raise GenerationContractError(
            f"prompt manifest {request.prompt_manifest} differs from the dataset provenance manifest"
        )
    guidance_fields = (
        request.image_guidance, request.count_guidance, request.batch_fn, request.selected_image_ids,
    )
    if any(value is not None for value in guidance_fields) and (request.method != SD14_CCG_ARM or not t2i):
        raise GenerationContractError("per-job semantic guidance requires its closed SD1.4 throughput invocation")
    if request.image_guidance is not None and (
        not request.image_guidance
        or any(
            not isinstance(key, str) or not key or not isinstance(spec, ImageGuidance)
            for key, spec in request.image_guidance.items()
        )
    ):
        raise GenerationContractError("job guidance must map source prompt IDs to typed contracts")
    if request.count_guidance is not None and (
        request.image_guidance is None or not request.count_guidance or any(
            not isinstance(key, str) or not key or not isinstance(spec, CountGuidance)
            for key, spec in request.count_guidance.items()
        )
    ):
        raise GenerationContractError("COUNT guidance requires typed source IDs and the adopted base guidance")
    if request.batch_fn is not None and (not callable(request.batch_fn) or request.image_guidance is None):
        raise GenerationContractError("batch adapter requires callable and source-bound semantic guidance")
    if request.selected_image_ids is not None:
        selected = request.selected_image_ids
        if (
            not isinstance(selected, tuple) or not selected
            or any(not isinstance(value, str) or not value for value in selected)
            or len(set(selected)) != len(selected)
        ):
            raise GenerationContractError("selected image IDs must be nonempty and unique")
        if request.image_guidance is None:
            raise GenerationContractError("image selection requires source-bound per-prompt guidance")
    if request.num_shards < 1:
        raise GenerationContractError("num_shards must be >= 1")
    if not 0 <= request.shard_index < request.num_shards:
        raise GenerationContractError("shard_index must be in [0, num_shards)")
    if request.expected_total is not None and request.expected_total < 1:
        raise GenerationContractError("expected_total must be positive")
    if request.batches is not None and (
        not isinstance(request.batches, tuple) or not request.batches
        or len(set(request.batches)) != len(request.batches)
    ):
        raise GenerationContractError("batches must be a nonempty tuple of unique batch indices")


def _component_files(request: SD14Run) -> tuple[str, ...]:
    """Driver files plus the caller's files, which must not overlap them."""

    combined = (*SD14_DRIVER_COMPONENT_FILES, *request.extra_component_files)
    if len(set(combined)) != len(combined):
        raise GenerationContractError(
            "extension component files overlap the standard component seal"
        )
    return combined


def _bind_image_guidance(
    jobs: Sequence[PreparedJob], image_guidance: Mapping[str, ImageGuidance] | None,
) -> None:
    """Bind every seed track to its source-verified immutable prompt contract."""

    if image_guidance is None:
        return
    source_ids: set[str] = set()
    for prepared in jobs:
        source_id = source_prompt_id(prepared.job)
        source_ids.add(source_id)
        spec = image_guidance.get(source_id)
        if spec is None:
            raise GenerationContractError(f"semantic guidance lacks source prompt: {source_id}")
        if not prepared.guidance_applied or prepared.provider != "deps2index_sd14":
            raise GenerationContractError("semantic guidance requires valid SD1.4 HolCCG PD programs")
        if (
            prepared.prompt_sha256 != spec.source_prompt_sha256
            or prepared.program_sha256 != spec.source_program_sha256
            or prepared.pd_indices_sha256 != spec.source_program_sha256
        ):
            raise GenerationContractError(f"semantic guidance source identity mismatch: {source_id}")
    if source_ids != set(image_guidance):
        raise GenerationContractError("semantic guidance must cover the exact unfiltered prompt population")


def _bind_count_guidance(
    jobs: Sequence[PreparedJob],
    image_guidance: Mapping[str, ImageGuidance] | None,
    count_guidance: Mapping[str, CountGuidance] | None,
) -> None:
    """Bind the complete COUNT population before any selection or sharding."""

    if count_guidance is None:
        return
    source_ids = set()
    for prepared in jobs:
        source_id = source_prompt_id(prepared.job)
        source_ids.add(source_id)
        spec, semantic_spec = count_guidance.get(source_id), image_guidance.get(source_id)
        if spec is None or semantic_spec is None:
            raise GenerationContractError(f"COUNT guidance lacks source prompt: {source_id}")
        if not prepared.guidance_applied or prepared.provider != "deps2index_sd14" or (
            prepared.prompt_sha256 != spec.source_prompt_sha256
            or prepared.program_sha256 != spec.source_program_sha256
            or prepared.pd_indices_sha256 != spec.source_program_sha256
            or semantic_spec.source_prompt_sha256 != spec.source_prompt_sha256
            or semantic_spec.source_program_sha256 != spec.source_program_sha256
        ):
            raise GenerationContractError(f"COUNT source identity mismatch: {source_id}")
        if spec.decision.constraints and (
            semantic_spec.effective_profile != GuidanceRoute.DEFAULT or semantic_spec.scale_multiplier != 1.0
            or semantic_spec.max_update_ratio != 0.02
            or any(value.noun_anchor not in prepared.job.attention_exist_indices for value in spec.decision.constraints)
        ):
            raise GenerationContractError("eligible COUNT rows must retain source anchors and adopted B0 scale/cap")
    if source_ids != set(count_guidance) or source_ids != set(image_guidance):
        raise GenerationContractError("COUNT requires the complete unfiltered source population")


def _select_image_jobs(
    jobs: list[PreparedJob], selected_image_ids: tuple[str, ...] | None,
) -> list[PreparedJob]:
    """Select an ordered subset of the canonical images; the canonical order is enforced."""

    if selected_image_ids is None:
        return jobs
    positions = {prepared.job.prompt_id: index for index, prepared in enumerate(jobs)}
    if len(positions) != len(jobs):
        raise GenerationContractError("canonical jobs contain duplicate image IDs")
    if any(prompt_id not in positions for prompt_id in selected_image_ids):
        raise GenerationContractError("selected image IDs contain an unknown canonical job")
    ordinals = [positions[prompt_id] for prompt_id in selected_image_ids]
    if ordinals != sorted(ordinals):
        raise GenerationContractError("selected image IDs reorder the canonical image population")
    return [jobs[index] for index in ordinals]


def _method_record(request: SD14Run) -> dict[str, Any] | None:
    """``extra_record`` plus the digests of the bound guidance and of the image selection."""

    record = dict(request.extra_record or {})
    digests = {}
    if request.image_guidance is not None:
        digests["per_prompt_guidance_sha256"] = sha256_text(canonical_json(
            {key: spec.as_record() for key, spec in sorted(request.image_guidance.items())}))
    if request.count_guidance is not None:
        digests["per_prompt_count_guidance_sha256"] = sha256_text(canonical_json(
            {key: spec.as_record() for key, spec in sorted(request.count_guidance.items())}))
    if request.selected_image_ids is not None:
        digests["selected_image_ids_sha256"] = sha256_text(canonical_json(request.selected_image_ids))
    for key in digests:
        if key in record:
            raise GenerationContractError(f"{key} is an extension-owned field")
    record.update(digests)
    return record or None


def _source_record(component_files: Sequence[str], *, allow_dirty: bool) -> dict[str, Any]:
    """Seal the component files to Git HEAD, or record their working-tree bytes under ``allow_dirty``."""

    if not allow_dirty:
        try:
            record = git_head_components_record(component_files)
        except SourceSealError as exc:
            raise GenerationContractError(str(exc)) from exc
        return {**record, "dirty": False}
    root = repository_root()
    return {
        "schema": SD14_PD_SOURCE_PROVENANCE_SCHEMA,
        "repo_git_commit": git_head(root),
        "component_sha256": {path: sha256_file(root / path) for path in component_files},
        "components_match_repo_git_commit": None,
        "dirty": True,
    }


def plan_run(request: SD14Run) -> RunPlan:
    """Load, bind, select and shard the images of the request and fix its physical batches (CPU only)."""

    _validate_request(request)
    component_files = _component_files(request)
    jobs = load_prepared_jobs(
        request.target_file,
        request.prompt_manifest,
        method=request.method,
        index_failure_policy=request.index_failure_policy,
        dataset=request.dataset,
    )
    _bind_image_guidance(jobs, request.image_guidance)
    _bind_count_guidance(jobs, request.image_guidance, request.count_guidance)
    jobs = _select_image_jobs(jobs, request.selected_image_ids)
    if request.expected_total is not None and len(jobs) != request.expected_total:
        raise GenerationContractError(
            f"expected {request.expected_total} input jobs before sharding, got {len(jobs)}"
        )
    jobs = [
        item
        for index, item in enumerate(jobs)
        if index % request.num_shards == request.shard_index
    ]
    if not jobs:
        raise GenerationContractError("selected shard contains no jobs")
    batches = plan_generation_batches(jobs, SD14_GENERATION.batch_size)
    if request.batches is not None and any(index not in range(len(batches)) for index in request.batches):
        raise GenerationContractError(
            f"batch indices outside the {len(batches)} physical batches of this shard: {request.batches}"
        )
    return RunPlan(jobs, batches, component_files)


def run(request: SD14Run) -> None:  # noqa: C901
    """Generate (or resume) one shard: run lock, images and metadata rows."""

    plan = plan_run(request)
    if isinstance(request.dataset, T2ICBDatasetProvenance):
        environment = sd14_generator_runtime_environment(device=request.device, strict=request.strict_runtime)
    else:
        environment = sd14_tab_generator_runtime_environment(device=request.device)
    source = _source_record(plan.component_files, allow_dirty=request.allow_dirty)
    runtime = prepare_sd14_runtime(request.device, local_files_only=request.local_files_only)
    token_counts, tokenizer, clip_truncation = validate_clip_token_contract(runtime.pipe.tokenizer, plan.jobs)
    configuration = run_configuration(
        method=request.method,
        runtime=runtime,
        device=device_record(request.device),
        local_files_only=request.local_files_only,
        tokenizer=tokenizer,
        token_counts=token_counts,
        clip_truncation=clip_truncation,
        prompt_manifest=request.prompt_manifest,
        dataset=request.dataset,
        generator_runtime_environment=environment,
        target_file=request.target_file,
        index_failure_policy=request.index_failure_policy,
        num_shards=request.num_shards,
        shard_index=request.shard_index,
        selected_jobs=plan.jobs,
        batch_padding_policy=SD14_PD_BATCH_PADDING_POLICY,
        source=source,
        runner_file=Path(__file__),
        kernel_file=Path(__file__).with_name("pipeline.py"),
        batch_subset=request.batches,
        method_record=_method_record(request),
    )
    lock_path = run_lock_path(
        request.out_dir, request.method, request.shard_index, request.num_shards
    )
    lock = ensure_run_lock(lock_path, configuration, resume=request.resume)
    lock_digest = sha256_file(lock_path)
    configuration_signature = str(lock["configuration_signature"])

    metadata_paths = {
        experiment: shard_metadata_path(
            request.out_dir,
            request.method,
            experiment,
            request.shard_index,
            request.num_shards,
        )
        for experiment in sorted({item.job.experiment for item in plan.jobs})
    }
    if not request.resume:
        existing = [path for path in metadata_paths.values() if path.exists()]
        if existing:
            raise GenerationContractError(
                f"metadata already exists for new run: {existing[0]}"
            )
    records = load_metadata_records(
        metadata_paths.values(), expected_signature=configuration_signature
    )
    image_size = (SD14_GENERATION.width, SD14_GENERATION.height)

    for prepared in plan.jobs:
        path = image_path(request.out_dir, request.method, prepared.job)
        recorded = records.get(repo_relative(path))
        if recorded is not None:
            if not request.resume:
                raise GenerationContractError(
                    f"unexpected recorded image in new run: {path}"
                )
            if not image_is_valid(path, expected_size=image_size):
                raise GenerationContractError(
                    f"recorded image is invalid on resume: {path}"
                )
            if recorded.get("image_sha256") != sha256_file(path):
                raise GenerationContractError(
                    f"recorded image hash mismatch on resume: {path}"
                )
            continue
        if path.exists():
            if not request.resume:
                raise GenerationContractError(
                    f"unlocked existing image would be overwritten: {path}"
                )
            # A matching run lock plus no metadata means the previous process
            # stopped between image replacement and ledger append. Regenerate
            # deterministically so all required diagnostics are recovered.
            path.unlink()

    selected_batches = [
        (batch_index, batch)
        for batch_index, batch in enumerate(plan.batches)
        if request.batches is None or batch_index in request.batches
    ]
    progress = tqdm(
        total=sum(
            1
            for _batch_index, batch in selected_batches
            for item in batch
            if repo_relative(image_path(request.out_dir, request.method, item.job)) not in records
        ),
        desc=f"generate:{request.method}:{request.shard_index}",
    )
    for batch_index, batch in selected_batches:
        generation_batch_id = (
            f"{shard_suffix(request.shard_index, request.num_shards)}:"
            f"batch_{batch_index:06d}"
        )
        batch_paths = [
            image_path(request.out_dir, request.method, item.job) for item in batch
        ]
        missing_flags = [repo_relative(path) not in records for path in batch_paths]
        if not any(missing_flags):
            continue

        try:
            synchronize_device(request.device)
            reset_peak_memory(request.device)
            wall_start = time.perf_counter()
            generated = generate_image_batch(
                runtime, batch, request=request,
                physical_batch_size=SD14_GENERATION.batch_size, device=request.device,
            )
            synchronize_device(request.device)
            wall_time_ms = (time.perf_counter() - wall_start) * 1000.0
            peak_allocated, peak_reserved = peak_memory(request.device)
            for image, diagnostics in generated:
                diagnostics.generation_batch_id = generation_batch_id
                if _is_predicated(request.method):
                    expected_padding_count = SD14_GENERATION.batch_size - len(batch)
                    if (
                        diagnostics.effective_batch_size != SD14_GENERATION.batch_size
                        or diagnostics.valid_batch_size != len(batch)
                        or diagnostics.batch_padding_count
                        != expected_padding_count
                    ):
                        raise GenerationContractError(
                            "SD1.4 PD denoiser did not honor its physical "
                            "batch contract"
                        )
                else:
                    diagnostics.effective_batch_size = len(batch)
                    diagnostics.valid_batch_size = len(batch)
                    diagnostics.batch_padding_count = 0
                diagnostics.batch_wall_time_ms = round(wall_time_ms, 4)
                diagnostics.amortized_wall_time_ms = round(
                    wall_time_ms / diagnostics.effective_batch_size, 4
                )
                diagnostics.peak_cuda_memory_allocated_bytes = peak_allocated
                diagnostics.peak_cuda_memory_reserved_bytes = peak_reserved
            for path, missing, (image, _diagnostics) in zip(
                batch_paths, missing_flags, generated
            ):
                if missing:
                    save_image_atomic(image, path, expected_size=image_size)
        except (OSError, GenerationContractError) as exc:
            for path, missing in zip(batch_paths, missing_flags):
                if missing and path.exists():
                    path.unlink()
            raise GenerationContractError(f"failed to generate valid image batch: {batch_paths[0]}") from exc

        for physical_batch_position, (
            prepared,
            path,
            missing,
            (_image, diagnostics),
        ) in enumerate(zip(batch, batch_paths, missing_flags, generated)):
            if not missing:
                continue
            job = prepared.job
            payload = metadata_payload(
                prepared,
                path,
                diagnostics,
                method=request.method,
                runtime=runtime,
                dataset=request.dataset,
                configuration=configuration,
                configuration_signature=configuration_signature,
                run_lock=lock_path,
                run_lock_sha256=lock_digest,
                clip_token_count=token_counts[job.prompt_id],
                clip_truncated=bool(clip_truncation[job.prompt_id]["truncated"]),
                clip_untruncated_token_count=int(
                    clip_truncation[job.prompt_id]["untruncated_token_count"]
                ),
                physical_batch_position=physical_batch_position,
            )
            append_metadata(metadata_paths[job.experiment], payload)
            records[repo_relative(path)] = payload
        progress.update(sum(missing_flags))
    progress.close()
