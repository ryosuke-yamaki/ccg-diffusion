"""SD1.4 guided denoising kernel of CCG Diffusion (one physical batch) and the stock batch kernel.

``generate_predicated_batch`` denoises a batch of independent images with
PNDM for 50 steps and classifier-free guidance 7.5. During the first 25 steps
(and five extra times at step 0) it captures the 16x16 cross-attention maps of
the conditional forward, evaluates each image's loss and moves each image's
FP32 master latent along the negative gradient (paper Eq. 3)::

    x_t <- x_t - eta_t * grad L,   eta_t = s * sqrt(linspace(1, 0.5, T)[t])

with a per-image relative-L2 cap delta on the step (``guidance.latent_update``).
The loss of an image is the Predicated Diffusion loss L_PD of its guidance
targets, or the image's loss override (Positional Prefix + Region Assignment
Loss, or the TAB objective); routed images add the Spatial Relation Loss
(route P1, weight 6) and then the Count Loss. The route of an image
(``ImageGuidance``: B0 default, P1 Spatial Relation, P2 Selective Guidance
Boost) sets its step multiplier and cap. Images never interact: one batched
backward of the summed losses gives each image its own gradient.

The U-Net, text encoder and VAE run in the pipeline dtype (BF16); latents, the
scheduler, the attention-probability reduction, every loss and the update stay
in FP32 (``bf16_stable``). Non-finite values abort the batch (strict finite
guards). ``generate_stock_batch`` is the same denoising loop without guidance,
used for Vanilla images and Predicated Diffusion parser fallbacks.

Helpers are module attributes so that tests can replace them with
``monkeypatch.setattr(pipeline, name, ...)``.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Sequence

import torch
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from PIL import Image

from ccg_diffusion.backbones.jobs import Job
from ccg_diffusion.guidance.attention import (
    AttentionStore,
    aggregate_attention_batched,
    register_attention_control,
)
from ccg_diffusion.guidance.count_loss import CountGuidance, augment_count_losses, fresh_count_states
from ccg_diffusion.guidance.latent_update import (
    GenerationDiagnostics,
    finite_and_max_abs_by_track,
    record_pd_loss,
    update_latents_batched,
)
from ccg_diffusion.guidance.predicated_diffusion_loss import compute_pd_loss
from ccg_diffusion.guidance.selective_guidance_boost import GuidanceRoute, ImageGuidance
from ccg_diffusion.guidance.spatial_relation_loss import (
    SPATIAL_RELATION_LOSS_PARAMETERS,
    SPATIAL_RELATION_LOSS_WEIGHT,
    compute_spatial_relation_loss,
)
from ccg_diffusion.targets.tokenizer import require_sd14_tokenizer

# Recorded name of the capture scope: every cross-attention processor at the
# 16x16 attention resolution (KEEP value).
SD14_ATTENTION_BLOCK_SCOPE = "all_at_attention_res"
# The five 16x16 cross-attention processors of the SD1.4 U-Net (KEEP values).
SD14_EXPECTED_ATTENTION_PROCESSORS = (
    "down_blocks.2.attentions.0.transformer_blocks.0.attn2.processor",
    "down_blocks.2.attentions.1.transformer_blocks.0.attn2.processor",
    "up_blocks.1.attentions.0.transformer_blocks.0.attn2.processor",
    "up_blocks.1.attentions.1.transformer_blocks.0.attn2.processor",
    "up_blocks.1.attentions.2.transformer_blocks.0.attn2.processor",
)


def record_relation_loss(
    diagnostics: GenerationDiagnostics | None,
    loss: torch.Tensor,
    *,
    base_loss: torch.Tensor | None = None,
    combined_loss: torch.Tensor | None = None,
) -> None:
    """Append distinct relation and optional combined-loss audit values."""

    if diagnostics is None or diagnostics.relation_guidance is None:
        return
    ledger = diagnostics.relation_guidance
    value = float(loss.detach().float().item())
    ledger["evaluations"] = int(ledger.get("evaluations", 0)) + 1
    if not math.isfinite(value):
        ledger["nonfinite_evaluations"] = int(ledger.get("nonfinite_evaluations", 0)) + 1
        return
    trajectory = ledger.setdefault("trajectory", [])
    trajectory.append(round(value, 8))
    ledger.setdefault("relation_loss_trajectory", []).append(round(value, 8))
    weight = float(ledger.get("weight", 0.0))
    weighted_value = weight * value
    ledger.setdefault("weighted_relation_loss_trajectory", []).append(
        round(weighted_value, 8)
    )
    ledger["last_weighted_relation_loss"] = weighted_value
    if ledger.get("first_loss") is None:
        ledger["first_loss"] = value
        ledger["min_loss"] = value
        ledger["max_loss"] = value
    ledger["last_loss"] = value
    ledger["min_loss"] = min(float(ledger["min_loss"]), value)
    ledger["max_loss"] = max(float(ledger["max_loss"]), value)
    ledger["loss_sum"] = float(ledger.get("loss_sum", 0.0)) + value
    ledger["mean_loss"] = ledger["loss_sum"] / len(trajectory)
    for name, tensor in (
        ("base_pd_loss", base_loss),
        ("combined_loss", combined_loss),
    ):
        if tensor is None:
            continue
        item = float(tensor.detach().float().item())
        if not math.isfinite(item):
            continue
        ledger.setdefault(f"{name}_trajectory", []).append(round(item, 8))
        ledger[f"last_{name}"] = item


def _audit_batched_tensor(
    tensor: torch.Tensor,
    diagnostics: Sequence[GenerationDiagnostics],
    *,
    strict_finite: bool,
    counter_name: str,
    label: str,
    maximum_name: str | None = None,
) -> None:
    """Audit a PD batch while retaining per-image counters and maxima."""

    finite, maxima = finite_and_max_abs_by_track(tensor)
    nonfinite_indices: list[int] = []
    for index, (record, is_finite, maximum) in enumerate(
        zip(diagnostics, finite, maxima)
    ):
        if is_finite:
            if maximum_name is not None:
                setattr(
                    record,
                    maximum_name,
                    max(float(getattr(record, maximum_name)), maximum),
                )
            continue
        setattr(record, counter_name, int(getattr(record, counter_name)) + 1)
        nonfinite_indices.append(index)
    if nonfinite_indices and strict_finite:
        raise FloatingPointError(
            f"non-finite Predicated Diffusion {label} at batch indices "
            f"{nonfinite_indices}"
        )


class _PhaseTimer:
    """Accumulate CUDA-event or CPU wall timings with one final sync."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.cuda_enabled = device.type == "cuda" and torch.cuda.is_available()
        self.cuda_events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        self.cpu_elapsed: dict[str, float] = {}

    @contextlib.contextmanager
    def phase(self, name: str) -> Iterator[None]:
        if self.cuda_enabled:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            stream = torch.cuda.current_stream(self.device)
            start.record(stream)
            try:
                yield
            finally:
                end.record(stream)
                self.cuda_events.append((name, start, end))
            return

        start_time = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            self.cpu_elapsed[name] = self.cpu_elapsed.get(name, 0.0) + elapsed_ms

    def finish(self) -> dict[str, float]:
        timings = dict(self.cpu_elapsed)
        if self.cuda_events:
            torch.cuda.synchronize(self.device)
            for name, start, end in self.cuda_events:
                timings[name] = timings.get(name, 0.0) + start.elapsed_time(end)
        return {name: round(value, 4) for name, value in sorted(timings.items())}


@dataclass(frozen=True)
class PromptConditioning:
    """Text conditioning of the guided (conditional only) and CFG U-Net calls."""

    conditional_prompt_embeds: torch.Tensor
    cfg_prompt_embeds: torch.Tensor


def scale_guided_model_input(
    pipe: Any,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    *,
    dtype: torch.dtype | None,
) -> torch.Tensor:
    """Map a master latent to the scheduler's differentiable U-Net input.

    For PNDM ``scale_model_input`` is the identity; it is kept so that the
    attention-capture forward and the denoising forward use the same scheduler
    coordinates, and the gradient reaches the FP32 master latent through it.
    """

    model_input = pipe.scheduler.scale_model_input(latents, timestep)
    return model_input if dtype is None else model_input.to(dtype=dtype)


def load_pipe(
    model_id: str,
    device: str,
    dtype: torch.dtype,
    *,
    revision: str | None = None,
    local_files_only: bool = False,
) -> StableDiffusionPipeline:
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        revision=revision,
        local_files_only=local_files_only,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def encode_prompts(
    pipe: Any,
    prompts: str | Sequence[str],
    *,
    device: torch.device,
) -> PromptConditioning:
    """Encode SD1.4 prompts into guided and CFG U-Net inputs (empty negative prompt)."""

    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompts,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt=None,
    )
    return PromptConditioning(
        conditional_prompt_embeds=prompt_embeds,
        cfg_prompt_embeds=torch.cat([negative_prompt_embeds, prompt_embeds]),
    )


def decode_latents(pipe: Any, latents: torch.Tensor) -> list[Image.Image]:
    """Decode FP32 latents with the VAE: cast to the VAE dtype, then divide by the scaling factor."""

    vae_dtype = next(pipe.vae.parameters()).dtype
    vae_latents = latents.to(dtype=vae_dtype)
    vae_latents = vae_latents / pipe.vae.config.scaling_factor
    image = pipe.vae.decode(
        vae_latents, return_dict=False
    )[0]
    image = pipe.image_processor.postprocess(image, output_type="pil")
    return image


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash a tensor's shape, dtype, and exact CPU bytes."""
    detached = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(tuple(detached.shape)).encode("ascii"))
    digest.update(str(detached.dtype).encode("ascii"))
    digest.update(detached.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def record_attention_capture(
    controller: AttentionStore,
    diagnostics: Sequence[GenerationDiagnostics],
) -> None:
    """Persist and cross-step-check the exact processors used by the losses."""

    names = list(controller.last_capture_processor_names)
    if not names:
        raise RuntimeError("guided UNet forward captured no cross-attention processors")
    for record in diagnostics:
        if record.captured_attention_processors:
            if record.captured_attention_processors != names:
                raise RuntimeError(
                    "captured attention processor set changed between guided forwards"
                )
        else:
            record.captured_attention_processors = list(names)
            record.captured_attention_processor_count = len(names)
            record.attention_block_scope = SD14_ATTENTION_BLOCK_SCOPE


@torch.enable_grad()
def generate_predicated_batch(
    pipe: Any,
    jobs: Sequence[Job],
    *,
    device: str,
    steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    attention_res: int,
    max_iter_to_alter: int,
    scale_factor: float,
    scale_range: tuple[float, float],
    thresholds: dict[int, float],
    refinement_updates_per_threshold: int = 5,
    initial_latents: torch.Tensor,
    diagnostics: Sequence[GenerationDiagnostics],
    strict_finite: bool,
    max_update_ratio: float | None,
    max_latent_abs: float | None,
    image_guidance: Sequence[ImageGuidance] | None = None,
    count_guidance: Sequence[CountGuidance] | None = None,
    loss_overrides: Sequence[Callable[..., torch.Tensor] | None] | None = None,
) -> StableDiffusionPipelineOutput:
    """Guided denoising of one physical batch of independent images (paper Eq. 3).

    The first ``max_iter_to_alter`` steps update each image's latent with step
    ``eta_t = scale_factor * sqrt(torch.linspace(*scale_range, steps)[t])``;
    at every step in ``thresholds`` the update is repeated
    ``refinement_updates_per_threshold`` more times with the same step.

    ``image_guidance`` gives one route per image (B0/P1/P2): the image's step is
    ``eta_t * scale_multiplier`` (4 on P2 images) and its cap is the route's
    ``max_update_ratio`` (0.06 on P2 images); P1 images add the Spatial
    Relation Loss with weight 6; P2 images with whole-word spans pool the
    pieces of each correspondence/leak endpoint word. Routes require the base
    scale 20. ``count_guidance`` (requires ``image_guidance``) adds the Count
    Loss after the relation term. ``loss_overrides`` replaces L_PD of an image
    by its callable (called like ``compute_pd_loss``).
    """

    update_options: dict[str, Any] = {}
    pair_spans_by_job = [None] * len(jobs)
    relation_constraints = None
    if image_guidance is not None:
        if len(image_guidance) != len(jobs) or any(not isinstance(spec, ImageGuidance) for spec in image_guidance):
            raise ValueError("semantic guidance must provide one typed contract per job")
        if scale_factor != 20.0 or max_update_ratio != 0.02 or not strict_finite:
            raise ValueError("semantic guidance requires unchanged SD1.4 B0 settings and owns its per-job extensions")
        require_sd14_tokenizer(pipe.tokenizer)
        for job, spec in zip(jobs, image_guidance):
            program = {
                "exist": job.attention_exist_indices,
                "corr": job.attention_corr_indices,
                "leak": job.attention_leak_indices,
                "possession": job.attention_possession_indices,
            }
            program_digest = hashlib.sha256(
                json.dumps(program, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            if spec.source_program_sha256 != program_digest or (
                spec.source_prompt_sha256 != hashlib.sha256(job.prompt.encode("utf-8")).hexdigest()
            ):
                raise ValueError("per-job semantic guidance source identity mismatch")
        update_options = {
            "step_size_multipliers": tuple(spec.scale_multiplier for spec in image_guidance),
            "per_job_max_update_ratios": tuple(spec.max_update_ratio for spec in image_guidance),
        }
        pair_spans_by_job = [spec.active_pair_spans for spec in image_guidance]
        if any(spec.relations for spec in image_guidance):
            relation_constraints = [spec.relations for spec in image_guidance]

    count_states = None
    if count_guidance is not None:
        if image_guidance is None or len(count_guidance) != len(jobs) or attention_res != 16:
            raise ValueError("COUNT requires aligned semantic guidance and SD1.4 16x16 attention")
        for job, semantic_spec, count_spec in zip(jobs, image_guidance, count_guidance):
            if not isinstance(count_spec, CountGuidance) or (
                count_spec.source_prompt_sha256 != semantic_spec.source_prompt_sha256
                or count_spec.source_program_sha256 != semantic_spec.source_program_sha256
            ):
                raise ValueError("COUNT and adopted semantic guidance must share the same source identity")
            if count_spec.decision.constraints and (
                semantic_spec.effective_profile != GuidanceRoute.DEFAULT or semantic_spec.scale_multiplier != 1.0
                or semantic_spec.max_update_ratio != 0.02
                or any(spec.noun_anchor not in job.attention_exist_indices for spec in count_spec.decision.constraints)
            ):
                raise ValueError("eligible COUNT nouns must retain source existence anchors and B0 scale/cap")
        count_states = fresh_count_states(
            count_guidance,
            [f"{job.experiment}/{job.prompt_id}/seed{job.seed}/physical_row{index}" for index, job in enumerate(jobs)],
        )

    if loss_overrides is not None:
        if len(loss_overrides) != len(jobs):
            raise ValueError("pd_loss_overrides must have one entry per job")
        if any(override is not None and not callable(override) for override in loss_overrides):
            raise ValueError("pd_loss_overrides entries must be callable or None")

    batch_size = len(jobs)
    if batch_size < 1:
        raise ValueError("jobs must not be empty")
    if len(diagnostics) != batch_size or initial_latents.shape[0] != batch_size:
        raise ValueError("jobs, diagnostics, and initial_latents must share batch size")
    if any(not job.pd_indices().has_any() for job in jobs):
        raise ValueError("batched PD inference requires guidance indices for every job")

    device_object = torch.device(device)
    timer = _PhaseTimer(device_object)

    def timed(name: str) -> contextlib.AbstractContextManager[None]:
        return timer.phase(name)

    with timed("text_encode"):
        conditioning = encode_prompts(
            pipe,
            [job.prompt for job in jobs],
            device=device_object,
        )
    text_embeddings = conditioning.cfg_prompt_embeds
    cond_embeds = conditioning.conditional_prompt_embeds

    controller = AttentionStore()
    register_attention_control(
        pipe.unet,
        controller,
        attention_res=attention_res,
    )
    pipe.scheduler.set_timesteps(steps, device=device)
    timesteps = pipe.scheduler.timesteps
    latents = initial_latents.detach().clone().to(device_object, dtype=torch.float32)
    for index, record in enumerate(diagnostics):
        record.effective_batch_size = batch_size
        record.generation_kernel = "predicated_stable_batch"
        record.strict_finite_applied = strict_finite
        record.initial_latent_sha256 = tensor_sha256(latents[index:index + 1])

    if image_guidance is not None:
        for record, spec in zip(diagnostics, image_guidance):
            record.semantic_guidance = {
                **spec.as_record(),
                "resolved_scale_factor": scale_factor * spec.scale_multiplier,
                "resolved_max_update_ratio": spec.max_update_ratio,
                "update_step_size_trajectory": [],
                "update_cap_trajectory": [],
            }

    pd_indices = [job.pd_indices() for job in jobs]
    # A float32 linspace, never a Python formula: the step sizes are part of the numerics.
    scale_values = torch.linspace(
        scale_range[0], scale_range[1], steps, device="cpu"
    ).tolist()

    relation_enabled = relation_constraints is not None
    if relation_enabled:
        for record, constraints in zip(diagnostics, relation_constraints):
            relations = tuple(constraints or ())
            if not relations:
                continue
            record.relation_guidance = {
                **SPATIAL_RELATION_LOSS_PARAMETERS,
                "status": "ok",
                "relations": [
                    {
                        "label": relation.label,
                        "subject_clip_index": relation.subject_clip_index,
                        "object_clip_index": relation.object_clip_index,
                    }
                    for relation in relations
                ],
                "evaluations": 0,
                "trajectory": [],
                "first_loss": None,
                "last_loss": None,
                "min_loss": None,
                "max_loss": None,
                "loss_sum": 0.0,
                "mean_loss": None,
            }

    overrides = loss_overrides if loss_overrides is not None else [None] * batch_size

    def compute_current_losses(step_index: int) -> list[torch.Tensor]:
        maps = aggregate_attention_batched(controller, res=attention_res, batch_size=batch_size)
        base_losses = [
            (compute_pd_loss if override is None else override)(
                attention,
                indices,
                attention_res=attention_res,
                guard_counts=record.pd_loss_guard_counts,
                **({"pair_token_spans": spans} if spans is not None else {}),
            )
            for attention, indices, record, spans, override in zip(
                maps,
                pd_indices,
                diagnostics,
                pair_spans_by_job,
                overrides,
            )
        ]
        if not relation_enabled:
            # With the Count Loss, the recorded PD loss is the base loss before the Count term.
            if count_states is not None:
                for record, loss in zip(diagnostics, base_losses):
                    record_pd_loss(record, loss)
            return augment_count_losses(base_losses, maps, count_states, outer_step_index=step_index)
        combined: list[torch.Tensor] = []
        for attention, base_loss, constraints, record in zip(
            maps, base_losses, relation_constraints, diagnostics
        ):
            relations = tuple(constraints or ())
            record_pd_loss(record, base_loss)
            if not relations:
                combined.append(base_loss)
                continue
            relation_loss = compute_spatial_relation_loss(
                attention,
                relations,
                attention_res=attention_res,
            )
            total_loss = base_loss + SPATIAL_RELATION_LOSS_WEIGHT * relation_loss
            record_relation_loss(
                record,
                relation_loss,
                base_loss=base_loss,
                combined_loss=total_loss,
            )
            combined.append(total_loss)
        # The Count Loss is added after the Spatial Relation term.
        return augment_count_losses(combined, maps, count_states, outer_step_index=step_index)

    for step_index, timestep in enumerate(timesteps):
        if step_index < max_iter_to_alter:
            for record in diagnostics:
                record.outer_updates_attempted += 1
            latents = latents.clone().detach().requires_grad_(True)
            controller.reset()
            controller.set_capture_enabled(True)
            guided_input = scale_guided_model_input(
                pipe,
                latents,
                timestep,
                dtype=cond_embeds.dtype,
            )
            for record in diagnostics:
                record.guided_scheduler_input_scaling = True
            with timed("guided_unet"):
                pipe.unet(
                    guided_input,
                    timestep,
                    encoder_hidden_states=cond_embeds,
                ).sample
            controller.finish_selective_forward()
            record_attention_capture(controller, diagnostics)
            with timed("pd_loss"):
                losses = compute_current_losses(step_index)
            if not relation_enabled and count_states is None:
                for record, loss in zip(diagnostics, losses):
                    record_pd_loss(record, loss)
            step_size = scale_factor * math.sqrt(scale_values[step_index])
            with timed("latent_update"):
                latents = update_latents_batched(
                    latents,
                    losses,
                    step_size,
                    diagnostics=diagnostics,
                    strict_finite=strict_finite,
                    max_update_ratio=max_update_ratio,
                    max_latent_abs=max_latent_abs,
                    **update_options,
                )

            if step_index in thresholds:
                for _ in range(refinement_updates_per_threshold):
                    for record in diagnostics:
                        record.refinement_updates_attempted += 1
                    latents = latents.clone().detach().requires_grad_(True)
                    controller.reset()
                    controller.set_capture_enabled(True)
                    guided_input = scale_guided_model_input(
                        pipe,
                        latents,
                        timestep,
                        dtype=cond_embeds.dtype,
                    )
                    with timed("guided_unet"):
                        pipe.unet(
                            guided_input,
                            timestep,
                            encoder_hidden_states=cond_embeds,
                        ).sample
                    controller.finish_selective_forward()
                    record_attention_capture(controller, diagnostics)
                    with timed("pd_loss"):
                        losses = compute_current_losses(step_index)
                    if not relation_enabled and count_states is None:
                        for record, loss in zip(diagnostics, losses):
                            record_pd_loss(record, loss)
                    with timed("latent_update"):
                        latents = update_latents_batched(
                            latents,
                            losses,
                            step_size,
                            diagnostics=diagnostics,
                            strict_finite=strict_finite,
                            max_update_ratio=max_update_ratio,
                            max_latent_abs=max_latent_abs,
                            **update_options,
                        )

        with torch.no_grad():
            controller.set_capture_enabled(False)
            latent_model_input = torch.cat([latents.detach()] * 2)
            latent_model_input = pipe.scheduler.scale_model_input(
                latent_model_input, timestep
            ).to(dtype=text_embeddings.dtype)
            denoiser = pipe.unet
            with timed("denoise_unet"):
                noise_pred = denoiser(
                    latent_model_input,
                    timestep,
                    encoder_hidden_states=text_embeddings,
                ).sample.float()
            noise_uncond, noise_text = noise_pred.chunk(2)
            noise_pred = noise_uncond + guidance_scale * (
                noise_text - noise_uncond
            )
            _audit_batched_tensor(
                noise_pred,
                diagnostics,
                strict_finite=strict_finite,
                counter_name="finite_guard_nonfinite_noise_pred",
                label="CFG noise prediction",
            )
            with timed("scheduler"):
                latents = pipe.scheduler.step(
                    noise_pred, timestep, latents.detach()
                ).prev_sample
            _audit_batched_tensor(
                latents,
                diagnostics,
                strict_finite=strict_finite,
                counter_name="finite_guard_nonfinite_scheduler_output",
                label="scheduler output",
                maximum_name="max_post_scheduler_latent_abs",
            )

    with torch.no_grad():
        for index, record in enumerate(diagnostics):
            record.final_latent_sha256 = tensor_sha256(latents[index:index + 1])
        with timed("vae_decode"):
            images = decode_latents(pipe, latents)

    if count_states is not None:
        for record, spec, state in zip(diagnostics, count_guidance, count_states):
            record.count_guidance = {
                **spec.as_record(), "state": state.audit_record(),
                "gradient_audit_enabled": False,
            }

    phase_timings = timer.finish()
    for record in diagnostics:
        record.phase_timings_ms = dict(phase_timings)
    return StableDiffusionPipelineOutput(images=images, nsfw_content_detected=None)


def generate_stock_batch(
    pipe: Any,
    jobs: Sequence[Job],
    *,
    device: str,
    steps: int,
    guidance_scale: float,
    initial_latents: torch.Tensor,
    diagnostics: Sequence[GenerationDiagnostics],
    strict_finite: bool,
) -> StableDiffusionPipelineOutput:
    """Stable stock (non-guided) denoising for multiple prompts in one batch.

    Shares the mixed-precision contract of :func:`generate_predicated_batch`:
    the master latent track and the scheduler step stay FP32, the U-Net /
    text encoder run in the pipeline's model dtype, the CFG-combined noise
    prediction is cast back to FP32, and the VAE conversion happens only at
    decode time.  No attention capture, loss, or latent update runs here; the
    caller is responsible for having the stock attention processors installed.
    A non-finite noise prediction or scheduler output aborts immediately
    (fail closed) under ``strict_finite``.
    """

    batch_size = len(jobs)
    if batch_size < 1:
        raise ValueError("jobs must not be empty")
    if len(diagnostics) != batch_size or initial_latents.shape[0] != batch_size:
        raise ValueError("jobs, diagnostics, and initial_latents must share batch size")

    device_object = torch.device(device)
    timer = _PhaseTimer(device_object)

    def timed(name: str) -> contextlib.AbstractContextManager[None]:
        return timer.phase(name)

    with timed("text_encode"):
        conditioning = encode_prompts(
            pipe,
            [job.prompt for job in jobs],
            device=device_object,
        )
    text_embeddings = conditioning.cfg_prompt_embeds

    pipe.scheduler.set_timesteps(steps, device=device)
    timesteps = pipe.scheduler.timesteps
    latents = initial_latents.detach().clone().to(device_object, dtype=torch.float32)
    for index, record in enumerate(diagnostics):
        record.effective_batch_size = batch_size
        record.generation_kernel = "stock_stable_batch"
        record.strict_finite_applied = strict_finite
        record.initial_latent_sha256 = tensor_sha256(latents[index:index + 1])

    with torch.no_grad():
        for timestep in timesteps:
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = pipe.scheduler.scale_model_input(
                latent_model_input, timestep
            ).to(dtype=text_embeddings.dtype)
            denoiser = pipe.unet
            with timed("denoise_unet"):
                noise_pred = denoiser(
                    latent_model_input,
                    timestep,
                    encoder_hidden_states=text_embeddings,
                ).sample.float()
            noise_uncond, noise_text = noise_pred.chunk(2)
            noise_pred = noise_uncond + guidance_scale * (
                noise_text - noise_uncond
            )
            if strict_finite and not torch.isfinite(noise_pred).all():
                raise RuntimeError(
                    "stock batch produced a non-finite noise prediction"
                )
            with timed("scheduler"):
                latents = pipe.scheduler.step(
                    noise_pred, timestep, latents
                ).prev_sample
            if strict_finite and not torch.isfinite(latents).all():
                raise RuntimeError(
                    "stock batch produced a non-finite scheduler output"
                )
            for index, record in enumerate(diagnostics):
                record.max_latent_abs = max(
                    record.max_latent_abs,
                    float(latents[index].float().abs().max().item()),
                )

        for index, record in enumerate(diagnostics):
            record.final_latent_sha256 = tensor_sha256(latents[index:index + 1])
        with timed("vae_decode"):
            images = decode_latents(pipe, latents)

    phase_timings = timer.finish()
    for record in diagnostics:
        record.phase_timings_ms = dict(phase_timings)
    return StableDiffusionPipelineOutput(images=images, nsfw_content_detected=None)
