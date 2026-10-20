"""Sana guided denoising loop: Predicated Diffusion latent updates from an image-local objective (paper Eq. 2-3).

``generate_guided`` mirrors ``SanaPipeline.__call__`` (flow-matching scheduler,
classifier-free guidance, resolution binning, VAE decode) and inserts, before
each guided step, one differentiable conditional forward that captures the
cross-attention maps of blocks [0, 8) at 16x16. Each image's objective (the
CCG Diffusion loss of T2I-CompBench++ or TAB PP) turns its maps into a loss,
and the latents take one capped gradient step (``guidance.latent_update``).
Paper Table 8: all 20 steps guided with step size ``20 * sqrt(s_t)``
(``s_t`` from 1.0 to 0.5), 5 extra updates at step 0, cap 0.02 per image
(0.06 on Selective Guidance Boost images, whose step is also multiplied by 4).

The images of one call share one prompt (and one plan); they differ by seed.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Optional, Protocol, Sequence

import torch
from diffusers.pipelines.sana.pipeline_sana import ASPECT_RATIO_512_BIN, retrieve_timesteps

from ccg_diffusion.backbones.sana.attention import (
    SanaAttentionStore,
    SanaCaptureComplete,
    aggregate_sana_attention,
    register_sana_attention_store,
)
from ccg_diffusion.backbones.sana.pipeline import DETERMINISM_MESSAGE, SanaEncoding, decode_latents, tensor_sha256
from ccg_diffusion.backbones.sana.tokens import decoded_prompt_tokens, pd_payload, preprocess_prompt
from ccg_diffusion.config import SANA_GENERATION
from ccg_diffusion.guidance.latent_update import GenerationDiagnostics, update_latents_batched
from ccg_diffusion.targets.pd_indices import PDIndices

DEFAULT_BLOCK_RANGE = tuple(range(0, 8))


class SanaGuidanceObjective(Protocol):
    """Per-image guidance objective of the Sana loop.

    Implemented by ``compositional.sana.CompositionalLoss`` (T2I-CompBench++)
    and ``pp_attachment.sana.SanaPPAttachmentLoss`` (TAB PP). ``plan`` must
    provide ``conditioning``, ``text_token_count``, ``source_program`` and
    ``lexical_projection``; ``generate_guided`` writes ``update_diagnostics``.
    """

    plan: Mapping[str, Any]
    indices: PDIndices
    scale_multiplier: float
    max_update_ratio: float
    update_diagnostics: dict[str, Any]

    def __call__(self, attention: torch.Tensor, step_index: int) -> torch.Tensor:
        """Loss of one image from its ``[16, 16, K]`` attention maps."""
        ...


@dataclass
class SanaGuidanceDiagnostics:
    """Per-call diagnostics of ``generate_guided`` (the recorded T2I metadata ``diagnostics``)."""

    real_token_count: int = 0
    decoded_tokens: dict[int, str] = field(default_factory=dict)
    source_pd_indices: Optional[dict[str, Any]] = None
    sana_pd_indices: Optional[dict[str, Any]] = None
    token_index_map: dict[Any, Any] = field(default_factory=dict)
    token_layout_verified: bool = False
    actual_attention_res: Optional[int] = None
    loss_trace: list[float] = field(default_factory=list)
    vae_decode_fallback: bool = False
    vae_decode_initial_nonfinite: bool = False
    initial_latent_hash: Optional[str] = None
    initial_latent_sha256: Optional[str] = None
    initial_latent_shape: Optional[list[int]] = None
    initial_latent_dtype: Optional[str] = None
    final_latent_hash: Optional[str] = None
    ambiguous_prompt_embedding_hash_before: Optional[str] = None
    ambiguous_prompt_embedding_hash_after: Optional[str] = None

    def compact(self, *, include_trace: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "real_token_count": self.real_token_count,
            "decoded_tokens": self.decoded_tokens,
            "source_pd_indices": self.source_pd_indices,
            "sana_pd_indices": self.sana_pd_indices,
            "token_index_map": self.token_index_map,
            "token_layout_verified": self.token_layout_verified,
            "actual_attention_res": self.actual_attention_res,
            "loss_first": self.loss_trace[0] if self.loss_trace else None,
            "loss_last": self.loss_trace[-1] if self.loss_trace else None,
            "loss_steps": len(self.loss_trace),
            "vae_decode_fallback": self.vae_decode_fallback,
            "vae_decode_initial_nonfinite": self.vae_decode_initial_nonfinite,
            "initial_latent_hash": self.initial_latent_hash,
            "initial_latent_sha256": self.initial_latent_sha256,
            "initial_latent_shape": self.initial_latent_shape,
            "initial_latent_dtype": self.initial_latent_dtype,
            "final_latent_hash": self.final_latent_hash,
            "ambiguous_prompt_embedding_hash_before": self.ambiguous_prompt_embedding_hash_before,
            "ambiguous_prompt_embedding_hash_after": self.ambiguous_prompt_embedding_hash_after,
        }
        if include_trace:
            payload["loss_trace"] = self.loss_trace
        return payload


def _aspect_ratio_bin(sample_size: int) -> dict[str, list[float]]:
    if sample_size == 16:
        return ASPECT_RATIO_512_BIN
    raise ValueError(f"invalid Sana transformer sample_size={sample_size}")


@torch.enable_grad()
def generate_guided(
    pipe: Any,
    conditioning: str,
    objectives: Sequence[SanaGuidanceObjective],
    generators: list[torch.Generator],
    *,
    prompt_encoding: SanaEncoding,
    instruction: Sequence[str],
    negative_prompt: str = "",
    steps: int = SANA_GENERATION.steps,
    guidance_scale: float = SANA_GENERATION.guidance_scale,
    height: int = SANA_GENERATION.height,
    width: int = SANA_GENERATION.width,
    block_range: Sequence[int] = DEFAULT_BLOCK_RANGE,
    scale_factor: float = SANA_GENERATION.scale_factor,
    guided_steps: int = SANA_GENERATION.guided_steps,
    refinement_steps: Mapping[int, float] = MappingProxyType({0: 0.0}),
) -> tuple[list[Any], SanaGuidanceDiagnostics]:
    """Generate one image per objective (and generator) for ``conditioning`` with latent guidance.

    ``prompt_encoding`` is the cached ``encode_prompt`` output of
    ``conditioning`` (see ``pipeline.encode_texts``). At each step ``i <
    guided_steps`` the latents take one update of size ``scale_factor *
    sqrt(s_i)``; at the steps in ``refinement_steps`` five more updates follow.
    Each objective's ``update_diagnostics`` receives its per-image update ledger
    and its initial and final latent hashes.
    """
    assert not torch.are_deterministic_algorithms_enabled(), DETERMINISM_MESSAGE
    objectives = tuple(objectives)
    batch_size = len(objectives)
    if any(item.plan != objectives[0].plan for item in objectives):
        raise ValueError("Sana compositional batches require one shared prompt plan")
    if pipe.transformer.training or any(p.requires_grad for p in pipe.transformer.parameters()):
        raise ValueError("text KV caching requires an evaluation-mode frozen transformer")
    if getattr(pipe.transformer, "gradient_checkpointing", False):
        raise ValueError("early guidance exit is incompatible with gradient checkpointing")
    if isinstance(conditioning, list):
        raise ValueError("generate_guided expects a single prompt string")

    orig_height, orig_width = height, width
    height, width = pipe.image_processor.classify_height_width_bin(
        height, width, ratios=_aspect_ratio_bin(pipe.transformer.config.sample_size)
    )

    pipe.check_inputs(
        conditioning,
        height,
        width,
        ["latents"],
        negative_prompt,
        None,
        None,
        None,
        None,
    )
    pipe._guidance_scale = guidance_scale
    pipe._attention_kwargs = None
    pipe._interrupt = False

    device = next(pipe.transformer.parameters()).device
    do_cfg = pipe.do_classifier_free_guidance

    with torch.no_grad():
        prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = (
            value.to(device) if value is not None else None for value in prompt_encoding)
        model_dtype = next(pipe.transformer.parameters()).dtype
        prompt_embeds = prompt_embeds.detach().to(model_dtype)
        prompt_attention_mask = prompt_attention_mask.detach()
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.detach().to(model_dtype)
        if negative_prompt_attention_mask is not None:
            negative_prompt_attention_mask = negative_prompt_attention_mask.detach()

    # Padding has zero softmax mass. Keep the same Gemma embeddings while
    # avoiding 300-column attention graphs for short prompts.
    active_tokens = int(prompt_attention_mask.sum(dim=-1).max())
    if negative_prompt_attention_mask is not None:
        active_tokens = max(active_tokens, int(negative_prompt_attention_mask.sum(dim=-1).max()))
    prompt_embeds = prompt_embeds[:, :active_tokens]
    prompt_attention_mask = prompt_attention_mask[:, :active_tokens]
    if negative_prompt_embeds is not None:
        negative_prompt_embeds = negative_prompt_embeds[:, :active_tokens]
        negative_prompt_attention_mask = negative_prompt_attention_mask[:, :active_tokens]
    if batch_size > 1:
        prompt_embeds = prompt_embeds.repeat(batch_size, 1, 1)
        prompt_attention_mask = prompt_attention_mask.repeat(batch_size, 1)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.repeat(batch_size, 1, 1)
            negative_prompt_attention_mask = negative_prompt_attention_mask.repeat(batch_size, 1)
    cond_mask = prompt_attention_mask[:1]
    real_token_count = int(cond_mask.reshape(-1).sum().item())
    effective_prompt = preprocess_prompt(pipe, conditioning)
    decoded_tokens = decoded_prompt_tokens(pipe, effective_prompt, cond_mask, instruction=instruction)
    plan = objectives[0].plan
    indices = objectives[0].indices
    if plan["conditioning"] != conditioning or plan["text_token_count"] != real_token_count:
        raise ValueError("runtime conditioning or PD indices differ from the Sana compositional plan")
    diagnostics = SanaGuidanceDiagnostics(
        real_token_count=real_token_count,
        decoded_tokens=decoded_tokens,
        source_pd_indices=plan["source_program"],
        sana_pd_indices=pd_payload(indices),
        token_index_map={row["source_clip_anchor"]: row["target_span"][-1] for row in plan["lexical_projection"]},
        token_layout_verified=True,
    )
    diagnostics.ambiguous_prompt_embedding_hash_before = tensor_sha256(prompt_embeds)
    stable_diagnostics = [GenerationDiagnostics() for _ in objectives]

    def update(current: torch.Tensor, loss: torch.Tensor, step: float) -> torch.Tensor:
        return update_latents_batched(
            current, list(loss.unbind()), step, diagnostics=stable_diagnostics, strict_finite=True,
            max_update_ratio=.02, max_latent_abs=None,
            step_size_multipliers=[item.scale_multiplier for item in objectives],
            per_job_max_update_ratios=[item.max_update_ratio for item in objectives])

    if do_cfg:
        text_embeddings = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        text_attention_mask = torch.cat(
            [negative_prompt_attention_mask, prompt_attention_mask], dim=0
        )
    else:
        text_embeddings = prompt_embeds
        text_attention_mask = prompt_attention_mask

    timesteps, num_inference_steps = retrieve_timesteps(pipe.scheduler, steps, device, None, None)
    pipe._num_timesteps = len(timesteps)
    latent_channels = pipe.transformer.config.in_channels
    with torch.no_grad():
        latents = pipe.prepare_latents(
            batch_size,
            latent_channels,
            height,
            width,
            torch.float32,
            device,
            generators,
            None,
        )
    diagnostics.initial_latent_hash = tensor_sha256(latents)
    initial_latent = latents.detach().to(
        device="cpu", dtype=torch.float32
    ).contiguous()
    diagnostics.initial_latent_sha256 = hashlib.sha256(
        initial_latent.numpy().tobytes()
    ).hexdigest()
    diagnostics.initial_latent_shape = list(initial_latent.shape)
    diagnostics.initial_latent_dtype = "float32"
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generators, 0.0)
    # s_t of the step size: a float32 linspace, read back as Python floats.
    scale_values = torch.linspace(1.0, 0.5, num_inference_steps, device="cpu").tolist()

    original_attn_processors = dict(pipe.transformer.attn_processors)
    store = SanaAttentionStore()
    store.cache_text_kv = True
    if not block_range:
        raise ValueError("efficient capture needs a nonempty block range")
    store.selected_blocks = set(block_range)
    store.stop_after = max(block_range)

    try:
        register_sana_attention_store(pipe.transformer, store)
        timestep_scale = pipe.transformer.config.timestep_scale

        def guidance_forward(model_input: torch.Tensor, timestep: torch.Tensor) -> None:
            # Conditional forward that stops after the last selected block.
            store.enabled = True
            try:
                pipe.transformer(
                    model_input, encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask, timestep=timestep,
                    return_dict=False)
            except SanaCaptureComplete:
                pass
            if len(store.attention_store) != len(store.selected_blocks):
                raise RuntimeError("Sana forward did not capture all requested guidance blocks")

        def compute_current_loss(step_index: int) -> torch.Tensor:
            maps = aggregate_sana_attention(store, block_range)[..., :real_token_count].float()
            diagnostics.actual_attention_res = int(maps.shape[1])
            return torch.stack([item(maps[index], step_index) for index, item in enumerate(objectives)])

        for i, t in enumerate(timesteps):
            if pipe.interrupt:
                continue

            if indices.has_any() and i < guided_steps:
                latents = latents.clone().detach().requires_grad_(True)
                store.reset()
                latent_model_input = latents.to(prompt_embeds.dtype)
                timestep = (
                    t.expand(latent_model_input.shape[0]).to(latents.dtype)
                    * timestep_scale
                )
                guidance_forward(latent_model_input, timestep)
                pipe.transformer.zero_grad(set_to_none=True)
                loss = compute_current_loss(i)
                diagnostics.loss_trace.append(float(loss.detach().mean().cpu()))
                step_size = scale_factor * math.sqrt(scale_values[i])
                latents = update(latents, loss, step_size)

                if i in refinement_steps:
                    for _ in range(5):
                        latents = latents.clone().detach().requires_grad_(True)
                        store.reset()
                        latent_model_input = latents.to(prompt_embeds.dtype)
                        timestep = (
                            t.expand(latent_model_input.shape[0]).to(latents.dtype)
                            * timestep_scale
                        )
                        guidance_forward(latent_model_input, timestep)
                        pipe.transformer.zero_grad(set_to_none=True)
                        loss = compute_current_loss(i)
                        diagnostics.loss_trace.append(float(loss.detach().mean().cpu()))
                        latents = update(latents, loss, step_size)

            with torch.no_grad():
                store.reset()  # Release guidance graphs before CFG denoising.
                store.enabled = False
                latent_model_input = (
                    torch.cat([latents.detach()] * 2) if do_cfg else latents.detach()
                )
                latent_model_input = latent_model_input.to(prompt_embeds.dtype)
                timestep = (
                    t.expand(latent_model_input.shape[0]).to(latents.dtype)
                    * timestep_scale
                )
                noise_pred = pipe.transformer(
                    latent_model_input,
                    encoder_hidden_states=text_embeddings,
                    encoder_attention_mask=text_attention_mask,
                    timestep=timestep,
                    return_dict=False,
                )[0].float()
                if do_cfg:
                    noise_uncond, noise_text = noise_pred.chunk(2)
                    noise_pred = noise_uncond + guidance_scale * (
                        noise_text - noise_uncond
                    )
                if pipe.transformer.config.out_channels // 2 == latent_channels:
                    noise_pred = noise_pred.chunk(2, dim=1)[0]
                if not torch.isfinite(noise_pred).all():
                    raise FloatingPointError("nonfinite Sana scheduler input")
                previous_latents = latents.detach()
                next_latents = pipe.scheduler.step(
                    noise_pred,
                    t,
                    previous_latents,
                    **extra_step_kwargs,
                    return_dict=False,
                )[0]
                if not torch.isfinite(next_latents).all():
                    raise FloatingPointError("nonfinite Sana scheduler output")
                latents = next_latents.float()

        diagnostics.final_latent_hash = tensor_sha256(latents)
        with torch.no_grad():
            image = decode_latents(pipe, latents, diagnostics)
            image = pipe.image_processor.resize_and_crop_tensor(image, orig_width, orig_height)
            images = pipe.image_processor.postprocess(image, output_type="pil")
    finally:
        pipe.transformer.set_attn_processor(original_attn_processors)

    for index, (item, record) in enumerate(zip(objectives, stable_diagnostics)):
        item.update_diagnostics = record.as_dict()
        item.update_diagnostics["initial_latent_sha256"] = tensor_sha256(initial_latent[index:index + 1])
        item.update_diagnostics["final_latent_sha256"] = tensor_sha256(latents[index:index + 1])
    diagnostics.ambiguous_prompt_embedding_hash_after = tensor_sha256(prompt_embeds)

    pipe.maybe_free_model_hooks()
    return images, diagnostics
