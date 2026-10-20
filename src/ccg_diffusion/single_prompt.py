"""CCG Diffusion on one arbitrary text-to-image prompt, with Stable Diffusion v1.4 or Sana 0.6B.

``generate(prompt, backbone="sd14"|"sana")`` runs the whole method for one
prompt and one seed: Hol-CCG parse and C&C marked dependencies
(``parsing.frontend.CCGFrontEnd``), guidance targets I_D = (E, C, K, P)
(Stage 3), the component plan (``compositional.plans.plan_prompt``), the
ablation switches (``remove``), and guided denoising on the backbone (the
T2I-CompBench++ batch function on SD1.4, ``compositional.sana`` on Sana). If
Stage 1-3 yields no usable guidance targets, the image is generated without
guidance and the record says why.

The result is not paper-exact (``"paper_exact": false``): the paper runs used
the frozen parses of the benchmark prompts and their physical batches of four
images, and a re-parse on other hardware can flip near-tie supertags.

Heavy packages (torch, diffusers, transformers, holccg, spaCy) are imported
inside ``generate``. SD1.4 configures its process-global determinism before
the parse, so do not run Sana later in the same process.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Sequence

from ccg_diffusion.methods import T2IAblation
from ccg_diffusion.parsing.sentences import split_and_tokenize

if TYPE_CHECKING:
    from ccg_diffusion.parsing.frontend import CCGFrontEnd

SENTENCE_ID = "prompt.1"
PROMPT_ID = "prompt"


@dataclass(frozen=True)
class SinglePromptResult:
    """The generated image and its record (parse, guidance targets, plan, diagnostics)."""

    image: Any  # PIL.Image.Image
    record: dict[str, Any]


def guidance_targets(prompt: str, parsed: Any, tokenizer: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Stage 3 of one parsed prompt: its guidance-target row and conversion trace (``status`` ``ok`` or a failure)."""
    from ccg_diffusion.config import SD14_GENERATION
    from ccg_diffusion.targets.guidance_targets import build_guidance_target_rows, load_possession_lemmatizer

    rows, traces = build_guidance_target_rows(
        [{"prompt_id": PROMPT_ID, "prompt": prompt, "seed": 0, "experiment": "prompt"}], [parsed.sentence],
        [parsed.dependencies], tokenizer, load_possession_lemmatizer(), category="prompt",
        coverage_rows=[parsed.coverage], tokenizer_identity=SD14_GENERATION.tokenizer_identity())
    return rows[0], traces[0]["conversion_trace"]


def generate(
    prompt: str, *, backbone: Literal["sd14", "sana"] = "sd14", seed: int = 42, device: str = "cuda:0",
    remove: Sequence[str] = (), frontend: CCGFrontEnd | None = None, local_files_only: bool = True,
) -> SinglePromptResult:
    """Generate one image of ``prompt`` with CCG Diffusion (``remove`` names T2I ablation switches).

    The prompt must be a single sentence; a multi-sentence prompt raises ``ValueError``.
    """
    from ccg_diffusion.compositional.ablation import apply_ablation
    from ccg_diffusion.compositional.plans import plan_prompt
    from ccg_diffusion.parsing.frontend import CCGFrontEnd
    from ccg_diffusion.targets.guidance_targets import CompiledGuidanceTargets
    from ccg_diffusion.targets.pd_indices import PDIndices
    from ccg_diffusion.targets.tokenizer import load_sd14_tokenizer

    if backbone not in ("sd14", "sana"):
        raise ValueError(f"unknown backbone {backbone!r}")
    sentences = split_and_tokenize(prompt)
    if len(sentences) != 1:
        raise ValueError(f"CCG Diffusion takes a prompt of exactly one sentence; got {len(sentences)} sentences")
    ablation = T2IAblation.from_removed(remove)
    # SD1.4: configure determinism before the Hol-CCG parse and any other CUDA work (it is process-global).
    model = load_backbone(backbone, device=device, local_files_only=local_files_only)
    frontend = frontend if frontend is not None else CCGFrontEnd(device, local_files_only=local_files_only)
    parsed = frontend.parse(prompt, sentence_id=SENTENCE_ID)
    tokenizer = load_sd14_tokenizer(local_files_only=local_files_only)
    row, trace = guidance_targets(prompt, parsed, tokenizer)
    record: dict[str, Any] = {
        "paper_exact": False, "backbone": backbone, "prompt": prompt, "seed": seed,
        "ablation": ablation.components(), "parse_status": trace.get("parse_status"),
        "derivation": parsed.derivation, "dependencies": list(parsed.dependency_lines), "guidance_targets": row,
    }
    if row["status"] != "ok":
        image, diagnostics = generate_stock(backbone, model, prompt, seed, device=device)
        record.update(guidance="stock_vanilla_fallback", fallback_reason=row["status"], plan=None,
                      diagnostics=diagnostics)
        return SinglePromptResult(image, record)
    targets = CompiledGuidanceTargets(PDIndices(**row["pd_indices"]), tuple(row["alignment"]), trace)
    if backbone == "sd14":
        planned = plan_prompt(prompt, parsed.sentence, parsed.dependencies, targets, tokenizer)
        executed, counted = apply_ablation(ablation, {PROMPT_ID: planned.guidance}, {PROMPT_ID: planned.count})
        plan = {"plan": planned.plan.as_record(), "guidance": executed[PROMPT_ID].as_record(),
                "count": counted[PROMPT_ID].as_record()}
        image, diagnostics = generate_sd14(model, row, planned.plan, executed[PROMPT_ID], counted[PROMPT_ID],
                                           ablation, prompt, seed, device=device)
    else:
        from ccg_diffusion.backbones.sana.pipeline import COMPLEX_HUMAN_INSTRUCTION, load_sana_tokenizer
        from ccg_diffusion.compositional.sana import compile_plan

        plan = compile_plan(prompt_id=PROMPT_ID, prompt=prompt, sentence=parsed.sentence,
                            dependencies=parsed.dependencies, compiled=targets, clip_tokenizer=tokenizer,
                            gemma_tokenizer=load_sana_tokenizer(local_files_only=local_files_only),
                            instruction=COMPLEX_HUMAN_INSTRUCTION,
                            ablation=None if ablation.is_full else ablation)
        image, diagnostics = generate_sana(model, plan, seed, device=device)
    record.update(guidance="method_guided", plan=plan, diagnostics=diagnostics)
    return SinglePromptResult(image, record)


def load_backbone(backbone: str, *, device: str, local_files_only: bool) -> Any:
    """SD1.4: ``driver.prepare_sd14_runtime`` (determinism first); Sana: the pinned pipeline."""
    if backbone == "sd14":
        from ccg_diffusion.backbones.sd14.driver import prepare_sd14_runtime

        return prepare_sd14_runtime(device, local_files_only=local_files_only)
    from ccg_diffusion.backbones.sana.pipeline import load_sana_pipeline

    return load_sana_pipeline(device, local_files_only=local_files_only)


def _sd14_job(prompt: str, seed: int, program: dict[str, Any]) -> Any:
    from ccg_diffusion.backbones.jobs import Job

    return Job(experiment=PROMPT_ID, prompt_id=f"{PROMPT_ID}_s{seed}", prompt=prompt, seed=seed,
               attention_exist_indices=program["exist"], attention_corr_indices=program["corr"],
               attention_leak_indices=program["leak"], attention_possession_indices=program["possession"],
               source="single_prompt")


def _sd14_latents(runtime: Any, jobs: Sequence[Any], *, device: str) -> tuple[Any, list[Any]]:
    """Seeded initial latents and fresh diagnostics, as the SD1.4 driver makes them."""
    import torch

    from ccg_diffusion.backbones.sd14.driver import initial_latents
    from ccg_diffusion.config import SD14_GENERATION
    from ccg_diffusion.guidance.latent_update import GenerationDiagnostics

    tracks, diagnostics = [], []
    for job in jobs:
        torch.manual_seed(job.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(job.seed)
        _, scaled, digest = initial_latents(runtime.pipe, job, device=device, dtype=torch.float32,
                                            height=SD14_GENERATION.height, width=SD14_GENERATION.width,
                                            steps=SD14_GENERATION.steps)
        tracks.append(scaled)
        diagnostics.append(GenerationDiagnostics(initial_latent_sha256=digest))
    return torch.cat(tracks), diagnostics


def generate_sd14(runtime: Any, row: dict[str, Any], plan: Any, guidance: Any, count: Any, ablation: T2IAblation,
                  prompt: str, seed: int, *, device: str) -> tuple[Any, dict[str, Any]]:
    """One guided SD1.4 image: the job repeated to a physical batch of four, run by ``compositional.sd14.run_batch``."""
    from ccg_diffusion.backbones.sd14.driver import pad_physical_batch
    from ccg_diffusion.compositional.sd14 import run_batch
    from ccg_diffusion.config import SD14_GENERATION, SD14_GUIDANCE_SCHEDULE
    from ccg_diffusion.guidance.selective_guidance_boost import BASE_SCALE

    jobs, _ = pad_physical_batch([_sd14_job(prompt, seed, row["pd_indices"])],
                                 physical_batch_size=SD14_GENERATION.batch_size)
    latents, diagnostics = _sd14_latents(runtime, jobs, device=device)
    schedule = SD14_GUIDANCE_SCHEDULE
    output = run_batch(
        ablation, runtime.pipe, jobs, [plan] * len(jobs), device=device, steps=SD14_GENERATION.steps,
        guidance_scale=SD14_GENERATION.guidance_scale, height=SD14_GENERATION.height, width=SD14_GENERATION.width,
        attention_res=SD14_GENERATION.attention_res, max_iter_to_alter=schedule.guided_steps,
        scale_factor=BASE_SCALE, scale_range=schedule.scale_range,
        thresholds={step: 0.0 for step in schedule.refinement_steps},
        refinement_updates_per_threshold=schedule.refinement_updates, initial_latents=latents,
        diagnostics=diagnostics, strict_finite=True, max_update_ratio=runtime.precision.max_update_ratio,
        max_latent_abs=runtime.precision.max_latent_abs, image_guidance=[guidance] * len(jobs),
        count_guidance=[count] * len(jobs))
    return output.images[0], diagnostics[0].as_dict()


def generate_sana(pipe: Any, plan: dict[str, Any], seed: int, *, device: str) -> tuple[Any, dict[str, Any]]:
    """One guided Sana image with ``compositional.sana.CompositionalLoss``."""
    import torch

    from ccg_diffusion.backbones.sana.guidance import generate_guided
    from ccg_diffusion.backbones.sana.pipeline import COMPLEX_HUMAN_INSTRUCTION, encode_texts
    from ccg_diffusion.compositional.sana import CompositionalLoss

    encoding = encode_texts(pipe, {PROMPT_ID: plan["conditioning"]}, device=device)[PROMPT_ID]
    loss = CompositionalLoss(plan, identity=f"{PROMPT_ID}_s{seed}")
    images, diagnostics = generate_guided(pipe, plan["conditioning"], [loss],
                                          [torch.Generator(device=device).manual_seed(seed)],
                                          prompt_encoding=encoding, instruction=COMPLEX_HUMAN_INSTRUCTION)
    return images[0], {**diagnostics.compact(include_trace=True), "compositional": loss.audit_record()}


def generate_stock(backbone: str, model: Any, prompt: str, seed: int, *, device: str) -> tuple[Any, dict[str, Any]]:
    """One image without guidance (the fallback when Stage 1-3 gives no usable guidance targets)."""
    import torch

    from ccg_diffusion.config import SANA_GENERATION, SD14_GENERATION

    if backbone == "sd14":
        from ccg_diffusion.backbones.sd14.pipeline import generate_stock_batch

        job = _sd14_job(prompt, seed, {"exist": [], "corr": [], "leak": [], "possession": []})
        latents, diagnostics = _sd14_latents(model, [job], device=device)
        output = generate_stock_batch(model.pipe, [job], device=device, steps=SD14_GENERATION.steps,
                                      guidance_scale=SD14_GENERATION.guidance_scale, initial_latents=latents,
                                      diagnostics=diagnostics, strict_finite=True)
        return output.images[0], diagnostics[0].as_dict()
    from ccg_diffusion.backbones.sana.pipeline import COMPLEX_HUMAN_INSTRUCTION, generate_stock_from_text

    images, records = generate_stock_from_text(
        model, prompt, [torch.Generator(device=device).manual_seed(seed)], device=device,
        steps=SANA_GENERATION.steps, guidance_scale=SANA_GENERATION.guidance_scale, height=SANA_GENERATION.height,
        width=SANA_GENERATION.width, instruction=COMPLEX_HUMAN_INSTRUCTION)
    return images[0], records[0]
