"""Sana 0.6B (512px) backbone: loading, instruction prompt, prompt encodings, stock generation, VAE decoding.

The checkpoint is ``Efficient-Large-Model/Sana_600M_512px_diffusers`` at the
pinned revision of ``config.SANA_GENERATION``, run in FP16 with FP32 latents
(paper Table 8). Sana prepends its own instruction prompt
(``COMPLEX_HUMAN_INSTRUCTION``) to every prompt before Gemma encodes it.

Determinism: the evaluated Sana runs only disabled TF32 and the cuDNN
autotuner; they never enabled deterministic algorithms. SD1.4's
``guidance.precision.configure_determinism`` is process-global, so Sana code
never calls it and refuses to generate when it is in force (run SD1.4 and Sana
in separate processes).

diffusers and transformers are imported inside the loaders, so this module
imports with torch alone.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from ccg_diffusion.config import SANA_GENERATION

# Sana's instruction prompt, verbatim; its value is recorded in run protocols.
COMPLEX_HUMAN_INSTRUCTION = [
    (
        "Given a user prompt, generate an 'Enhanced prompt' that provides detailed visual descriptions suitable "
        "for image generation. Evaluate the level of detail in the user prompt:"
    ),
    (
        "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial "
        "relationships to create vivid and concrete scenes."
    ),
    (
        "- If the prompt is already detailed, refine and enhance the existing details slightly without "
        "overcomplicating."
    ),
    "Here are examples of how to transform or refine prompts:",
    (
        "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping "
        "peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers."
    ),
    (
        "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing "
        "street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering "
        "glass skyscrapers."
    ),
    (
        "Please generate only the enhanced description for the prompt below and avoid including any additional "
        "commentary or evaluations:"
    ),
    "User Prompt: ",
]

# (prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask), on the CPU.
SanaEncoding = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

DETERMINISM_MESSAGE = "Sana generation must run in a process where SD1.4 configure_determinism was never called"


def load_sana_pipeline(device: str | torch.device = "cuda", *, local_files_only: bool = True) -> Any:
    """Load the pinned Sana 0.6B 512px pipeline in FP16, frozen, with TF32 and cuDNN autotuning off."""
    from diffusers import SanaPipeline

    pipeline = SanaPipeline.from_pretrained(
        SANA_GENERATION.model_id,
        torch_dtype=torch.float16,
        revision=SANA_GENERATION.model_revision,
        local_files_only=local_files_only,
    )
    pipeline.to(device)
    for component in (pipeline.transformer, pipeline.text_encoder, pipeline.vae):
        component.eval().requires_grad_(False)
    pipeline.set_progress_bar_config(disable=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    return pipeline


def load_sana_tokenizer(*, local_files_only: bool = True) -> Any:
    """Sana's Gemma tokenizer (``GemmaTokenizerFast``) at the pinned revision."""
    from transformers import GemmaTokenizerFast

    return GemmaTokenizerFast.from_pretrained(
        SANA_GENERATION.model_id,
        revision=SANA_GENERATION.model_revision,
        subfolder="tokenizer",
        local_files_only=local_files_only,
    )


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Sana layout: sha256 of the dtype string, ``str(tuple(shape))`` and the raw bytes.

    It differs from the SD1.4 ``tensor_sha256`` (shape first); every recorded
    Sana latent and text-encoding hash uses this layout.
    """
    value = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(str(tuple(value.shape)).encode("utf-8"))
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def initial_latents(pipe: Any, seeds: Sequence[int], device: str) -> torch.Tensor:
    """FP32 initial latents of a 512x512 batch, one generator seeded per image."""
    return pipe.prepare_latents(
        len(seeds), pipe.transformer.config.in_channels, 512, 512, torch.float32, device,
        [torch.Generator(device=device).manual_seed(seed) for seed in seeds], None)


def encode_texts(
    pipe: Any,
    texts: Mapping[str, str],
    *,
    device: str,
    instruction: Sequence[str] = COMPLEX_HUMAN_INSTRUCTION,
    negative_prompt: str = "",
    offload_text_encoder: bool = True,
) -> dict[str, SanaEncoding]:
    """Gemma encodings of ``texts`` (key -> text), in the caller's key order, stored on the CPU.

    The negative prompt is encoded once, with the first text, and reused for
    every later text. Afterwards the text encoder moves to the CPU, unless
    ``offload_text_encoder`` is False: a later ``generate_stock_from_encoding``
    on the same pipeline needs it on the GPU, because diffusers places the
    supplied encodings on the text encoder's device (the TAB runs kept it there).
    """
    encodings: dict[str, SanaEncoding] = {}
    negative: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for key, text in texts.items():
            encoding = pipe.encode_prompt(
                text, True, negative_prompt=negative_prompt, num_images_per_prompt=1, device=device,
                clean_caption=False, max_sequence_length=300, complex_human_instruction=list(instruction),
                **negative)
            if any(not torch.isfinite(value).all() for value in encoding):
                raise FloatingPointError(f"nonfinite text encoding: {key}")
            encodings[key] = tuple(value.detach().cpu() for value in encoding)
            if not negative:
                negative = {"negative_prompt_embeds": encoding[2], "negative_prompt_attention_mask": encoding[3]}
    del negative
    if offload_text_encoder:
        pipe.text_encoder.to("cpu")
    torch.cuda.empty_cache()
    return encodings


def generate_stock_from_text(
    pipe: Any,
    prompt: str,
    generators: list[torch.Generator],
    *,
    device: str,
    steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    instruction: Sequence[str],
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Vanilla Sana from the prompt text (T2I-CompBench++): the unmodified diffusers pipeline.

    Returns the images and one audit record per image with the initial and
    final latent hashes.
    """
    assert not torch.are_deterministic_algorithms_enabled(), DETERMINISM_MESSAGE
    initial = pipe.prepare_latents(
        len(generators), pipe.transformer.config.in_channels, height, width,
        torch.float32, device, generators, None)
    audits = [{"guidance_mode": "stock_vanilla",
               "initial_latent_sha256": tensor_sha256(value.unsqueeze(0))} for value in initial]

    def audit_final(_pipe: Any, step: int, timestep: Any, values: dict[str, Any]) -> dict[str, Any]:
        if step == steps - 1:
            if not torch.isfinite(values["latents"]).all():
                raise FloatingPointError("nonfinite stock Sana output latent")
            for record, value in zip(audits, values["latents"]):
                record["final_latent_sha256"] = tensor_sha256(value.unsqueeze(0))
        return values

    with torch.inference_mode():
        result = pipe(
            [prompt] * len(generators), negative_prompt=[""] * len(generators),
            num_inference_steps=steps, guidance_scale=guidance_scale,
            height=height, width=width, generator=generators, latents=initial,
            use_resolution_binning=True,
            clean_caption=False, complex_human_instruction=instruction,
            callback_on_step_end=audit_final, callback_on_step_end_tensor_inputs=["latents"])
    return result.images, audits


def generate_stock_from_encoding(
    pipe: Any,
    encoding: SanaEncoding,
    generators: list[torch.Generator],
    *,
    device: str,
    steps: int,
    guidance_scale: float,
    height: int,
    width: int,
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Vanilla Sana from a cached encoding (TAB Vanilla and Gold-visual setup).

    The full 300-token encoding is repeated per image. Returns the images and
    one audit record per image with the initial and final latent hashes.
    """
    assert not torch.are_deterministic_algorithms_enabled(), DETERMINISM_MESSAGE
    initial = pipe.prepare_latents(len(generators), pipe.transformer.config.in_channels,
                                   height, width, torch.float32, device, generators, None)
    audits = [{"initial_latent_sha256": tensor_sha256(v.unsqueeze(0))} for v in initial]

    def audit_final(_pipe: Any, step: int, timestep: Any, values: dict[str, Any]) -> dict[str, Any]:
        if not torch.isfinite(values["latents"]).all():
            raise FloatingPointError("nonfinite stock Sana latent")
        if step == steps - 1:
            for d, value in zip(audits, values["latents"]):
                d["final_latent_sha256"] = tensor_sha256(value.unsqueeze(0))
        return values

    emb, mask, neg, negmask = [v.to(device).repeat(len(generators), *([1] * (v.ndim - 1)))
                               for v in encoding]
    with torch.inference_mode():
        result = pipe(prompt_embeds=emb, prompt_attention_mask=mask, negative_prompt=None,
                      negative_prompt_embeds=neg, negative_prompt_attention_mask=negmask,
                      num_inference_steps=steps, guidance_scale=guidance_scale,
                      height=height, width=width, generator=generators, latents=initial,
                      use_resolution_binning=True, callback_on_step_end=audit_final,
                      callback_on_step_end_tensor_inputs=["latents"])
    return result.images, audits


@dataclass
class VaeDecodeRecord:
    """Whether the FP16 VAE decode produced non-finite pixels and was retried in FP32."""

    vae_decode_initial_nonfinite: bool = False
    vae_decode_fallback: bool = False


def decode_latents(pipe: Any, latents: torch.Tensor, record: Any) -> torch.Tensor:
    """Decode latents, retrying non-finite low-precision VAE output in FP32.

    ``record`` is any object with the two ``VaeDecodeRecord`` flags.
    """
    original_dtype = pipe.vae.dtype
    scaled = latents.to(original_dtype) / pipe.vae.config.scaling_factor
    image = pipe.vae.decode(scaled, return_dict=False)[0]
    if torch.isfinite(image).all():
        return image

    record.vae_decode_initial_nonfinite = True
    if original_dtype == torch.float32:
        raise RuntimeError("Sana VAE produced non-finite output in float32")

    record.vae_decode_fallback = True
    try:
        pipe.vae.to(dtype=torch.float32)
        image = pipe.vae.decode(
            latents.float() / pipe.vae.config.scaling_factor,
            return_dict=False,
        )[0]
    finally:
        pipe.vae.to(dtype=original_dtype)
    if not torch.isfinite(image).all():
        raise RuntimeError("Sana VAE produced non-finite output after float32 fallback")
    return image
