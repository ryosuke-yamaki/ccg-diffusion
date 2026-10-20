"""SD1.4 guided denoising loop (with the capped latent update) and batch driver on CPU fakes.

The kernel runs with a fake U-Net, scheduler and attention aggregation (each
image's "attention maps" are a function of its own latent), so the schedule,
per-image guidance strengths and caps, finite guards, loss overrides and physical batching
are checked without model weights.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import pytest
import torch

from conftest import FakeSDPipe, import_closure
from ccg_diffusion import config
from ccg_diffusion.backbones.jobs import Job
from ccg_diffusion.backbones.sd14 import driver, pipeline
from ccg_diffusion.backbones.sd14.inputs import GenerationContractError, PreparedJob, load_prepared_jobs
from ccg_diffusion.backbones.sd14.records import ensure_run_lock, scheduler_record
from ccg_diffusion.guidance.latent_update import GenerationDiagnostics
from ccg_diffusion.guidance.precision import PROFILES
from ccg_diffusion.guidance.predicated_diffusion_loss import WholeWordSpans, compute_pd_loss
from ccg_diffusion.guidance.selective_guidance_boost import GuidanceDecision, ImageGuidance

ROOT = Path(__file__).resolve().parents[1]


# --- fakes -------------------------------------------------------------------------------------------------


class FakeScheduler:
    init_noise_sigma = 1.0

    def __init__(self, *, nonfinite_rows: Sequence[int] = ()) -> None:
        self.nonfinite_rows = tuple(nonfinite_rows)
        self.timesteps = torch.empty(0)

    def set_timesteps(self, steps: int, *, device: str) -> None:
        self.timesteps = torch.arange(steps - 1, -1, -1)

    def scale_model_input(self, sample: torch.Tensor, _timestep: torch.Tensor) -> torch.Tensor:
        return sample

    def step(self, noise: torch.Tensor, _timestep: torch.Tensor, latents: torch.Tensor) -> SimpleNamespace:
        result = latents - 0.01 * noise
        for row in self.nonfinite_rows:
            result[row] = float("nan")
        return SimpleNamespace(prev_sample=result)


class FakeUNet:
    """Counts guided (gradient) and CFG forwards; records the processor the store captured."""

    def __init__(self, *, nonfinite_rows: Sequence[int] = ()) -> None:
        self.nonfinite_rows = tuple(nonfinite_rows)
        self.guided_forwards = 0
        self.cfg_forwards = 0
        self.attention_latents: torch.Tensor | None = None
        self.controller: Any = None
        self.attn_processors = {"down.attn2.processor": "stock"}
        self.installed: list[dict] = []

    def set_attn_processor(self, processors: dict) -> None:
        self.installed.append(dict(processors))

    def __call__(self, sample: torch.Tensor, _timestep: torch.Tensor, **_kwargs: Any) -> SimpleNamespace:
        if sample.requires_grad:
            self.guided_forwards += 1
            self.attention_latents = sample
            self.controller.record_capture_processor("down.attn2.processor")
            return SimpleNamespace(sample=torch.zeros_like(sample))
        self.cfg_forwards += 1
        noise = sample.clone()
        for row in self.nonfinite_rows:
            noise[row] = float("nan")
        return SimpleNamespace(sample=noise)


def _patch_kernel(monkeypatch: pytest.MonkeyPatch, unet: FakeUNet) -> None:
    def encode(_pipe: Any, prompts: Sequence[str], **_kwargs: Any) -> pipeline.PromptConditioning:
        return pipeline.PromptConditioning(torch.zeros(len(prompts), 1, 1), torch.zeros(2 * len(prompts), 1, 1))

    def aggregate(*_args: Any, **_kwargs: Any) -> torch.Tensor:
        field = unet.attention_latents[:, 0]
        return torch.stack((torch.zeros_like(field), field.sigmoid(), (0.4 - field.square()).sigmoid(),
                            torch.zeros_like(field)), dim=-1)

    monkeypatch.setattr(pipeline, "encode_prompts", encode)
    monkeypatch.setattr(pipeline, "register_attention_control",
                        lambda model, controller, **_kwargs: setattr(model, "controller", controller))
    monkeypatch.setattr(pipeline, "aggregate_attention_batched", aggregate)
    monkeypatch.setattr(pipeline, "decode_latents", lambda _pipe, latents: [row.detach().clone() for row in latents])
    monkeypatch.setattr(pipeline, "require_sd14_tokenizer", lambda _tokenizer: None)


def _job(index: int, *, exist: Sequence[int] = (1,)) -> Job:
    return Job(experiment="loop", prompt_id=f"prompt_{index}", prompt=f"object {index}", seed=index,
               attention_exist_indices=list(exist), attention_corr_indices=[], attention_leak_indices=[],
               attention_possession_indices=[], source="test")


LATENTS = torch.tensor([0.1, 0.2, 0.3, 0.5, -0.2, 0.4, 0.6, 0.8, 0.3, -0.1, 0.2, 0.7]).reshape(3, 1, 2, 2)


def _run(monkeypatch: pytest.MonkeyPatch, *, rows: Sequence[int] = (0, 1), steps: int = 2, unet: FakeUNet | None = None,
         scheduler: FakeScheduler | None = None, **extra: Any) -> tuple[list[torch.Tensor], list, FakeUNet]:
    unet = unet or FakeUNet()
    _patch_kernel(monkeypatch, unet)
    records = [GenerationDiagnostics() for _ in rows]
    kwargs = {
        "device": "cpu", "steps": steps, "guidance_scale": 7.5, "height": 16, "width": 16, "attention_res": 2,
        "max_iter_to_alter": 1, "scale_factor": 20.0, "scale_range": (1.0, 0.5), "thresholds": {},
        "initial_latents": LATENTS[list(rows)], "diagnostics": records, "strict_finite": True,
        "max_update_ratio": 0.02, "max_latent_abs": None,
    }
    kwargs.update(extra)
    pipe = SimpleNamespace(unet=unet, scheduler=scheduler or FakeScheduler(), tokenizer=None)
    result = pipeline.generate_predicated_batch(pipe, [_job(row) for row in rows], **kwargs)
    return result.images, records, unet


def _semantic(records: Sequence[GenerationDiagnostics]) -> list[dict]:
    result = [record.as_dict() for record in records]
    for record in result:
        record.pop("phase_timings_ms")
    return result


def _route(job: Job, profile: str) -> ImageGuidance:
    program = json.dumps({"exist": job.attention_exist_indices, "corr": job.attention_corr_indices,
                          "leak": job.attention_leak_indices, "possession": job.attention_possession_indices},
                         ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return ImageGuidance("routing_v1", GuidanceDecision(profile, ("test",), "{}"),
                         hashlib.sha256(job.prompt.encode()).hexdigest(), hashlib.sha256(program.encode()).hexdigest(),
                         WholeWordSpans(4, ()), ())


# --- the guided denoising kernel ----------------------------------------------------------------------------


def test_schedule_of_guided_forwards_and_step_sizes(monkeypatch: pytest.MonkeyPatch) -> None:
    steps: list[float] = []
    original = pipeline.update_latents_batched

    def record(latents: torch.Tensor, losses: list, step_size: float, **kwargs: Any) -> torch.Tensor:
        steps.append(step_size)
        return original(latents, losses, step_size, **kwargs)

    monkeypatch.setattr(pipeline, "update_latents_batched", record)
    _, records, unet = _run(monkeypatch, rows=(0, 1, 2), steps=50, max_iter_to_alter=25, thresholds={0: 0.0},
                            refinement_updates_per_threshold=5)
    assert (unet.guided_forwards, unet.cfg_forwards) == (30, 50)
    assert all(r.pd_loss_evaluations == 30 and r.outer_updates_attempted == 25 for r in records)
    assert all(r.refinement_updates_attempted == 5 for r in records)
    assert all(r.generation_kernel == "predicated_stable_batch" for r in records)
    values = torch.linspace(1.0, 0.5, 50).tolist()
    assert steps == [20.0 * math.sqrt(values[0])] * 6 + [20.0 * math.sqrt(values[t]) for t in range(1, 25)]


def test_mixed_routes_equal_independent_images_and_keep_their_caps(monkeypatch: pytest.MonkeyPatch) -> None:
    profiles = ("P2", "B0")
    batch, records, _ = _run(monkeypatch, image_guidance=[_route(_job(i), p) for i, p in enumerate(profiles)])
    for index, profile in enumerate(profiles):
        alone, alone_records, _ = _run(monkeypatch, rows=(index,), image_guidance=[_route(_job(index), profile)])
        assert torch.equal(batch[index], alone[0])
        in_batch, single = _semantic(records[index:index + 1])[0], _semantic(alone_records)[0]
        assert (in_batch.pop("effective_batch_size"), single.pop("effective_batch_size")) == (2, 1)
        assert in_batch == single
    assert records[0].semantic_guidance["update_step_size_trajectory"] == [80.0]
    assert records[0].semantic_guidance["update_cap_trajectory"] == [0.06]
    assert records[1].semantic_guidance["update_step_size_trajectory"] == [20.0]
    assert records[0].max_applied_update_ratio <= 0.06 + 1e-6 and records[1].max_applied_update_ratio <= 0.02 + 1e-6
    with pytest.raises(ValueError, match="unchanged SD1.4 B0 settings"):
        _run(monkeypatch, scale_factor=10.0, image_guidance=[_route(_job(i), "B0") for i in (0, 1)])


def test_finite_guards_fail_closed_with_per_image_counters(monkeypatch: pytest.MonkeyPatch) -> None:
    records = [GenerationDiagnostics(), GenerationDiagnostics()]
    with pytest.raises(FloatingPointError, match=r"CFG noise prediction at batch indices \[1\]"):
        _run(monkeypatch, unet=FakeUNet(nonfinite_rows=(3,)), max_iter_to_alter=0, diagnostics=records)
    assert [r.finite_guard_nonfinite_noise_pred for r in records] == [0, 1]
    records = [GenerationDiagnostics(), GenerationDiagnostics()]
    with pytest.raises(FloatingPointError, match=r"scheduler output at batch indices \[0\]"):
        _run(monkeypatch, scheduler=FakeScheduler(nonfinite_rows=(0,)), max_iter_to_alter=0, diagnostics=records)
    assert [r.finite_guard_nonfinite_scheduler_output for r in records] == [1, 0]
    unet = FakeUNet(nonfinite_rows=(0,))
    _patch_kernel(monkeypatch, unet)
    pipe = SimpleNamespace(unet=unet, scheduler=FakeScheduler())
    with pytest.raises(RuntimeError, match="non-finite noise prediction"):
        pipeline.generate_stock_batch(pipe, [_job(0)], device="cpu", steps=2, guidance_scale=7.5,
                                      initial_latents=LATENTS[:1], diagnostics=[GenerationDiagnostics()],
                                      strict_finite=True)


@pytest.mark.parametrize("overrides", [None, [None, None]])
def test_absent_overrides_are_bit_exact(monkeypatch: pytest.MonkeyPatch, overrides: Any) -> None:
    baseline, before, _ = _run(monkeypatch)
    actual, after, _ = _run(monkeypatch, loss_overrides=overrides)
    assert all(torch.equal(a, b) for a, b in zip(actual, baseline))
    assert _semantic(after) == _semantic(before)


def test_an_override_changes_only_its_image_and_receives_the_loss_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    baseline, before, _ = _run(monkeypatch)
    calls: list[dict] = []

    def override(maps: torch.Tensor, indices: Any, **kwargs: Any) -> torch.Tensor:
        calls.append(kwargs)
        return compute_pd_loss(maps, indices, **kwargs) + (maps[:, :, 1] - 0.7).square().mean()

    actual, after, _ = _run(monkeypatch, loss_overrides=[override, None])
    assert not torch.equal(actual[0], baseline[0]) and torch.equal(actual[1], baseline[1])
    assert _semantic(after[1:]) == _semantic(before[1:])
    assert calls == [{"attention_res": 2, "guard_counts": after[0].pd_loss_guard_counts}]


# --- the batch driver ----------------------------------------------------------------------------------------


def _prepared(index: int, *, guided: bool, provider: str | None = "spacy_sd14") -> PreparedJob:
    job = _job(index, exist=(1,) if guided else ())
    status = None if provider is None else ("ok" if guided else "alignment_error")
    return PreparedJob(job=job, prompt_sha256="0" * 64, program_sha256="1" * 64 if guided else None,
                       pd_indices_sha256=None, provider=provider, extraction_status=status)


def test_batches_group_by_path_and_guided_arms_pad_to_four(monkeypatch: pytest.MonkeyPatch) -> None:
    jobs = [_prepared(i, guided=g) for i, g in enumerate((True, True, False, False, False, True))]
    assert [[p.job.prompt_id for p in batch] for batch in driver.plan_generation_batches(jobs, 4)] == [
        ["prompt_0", "prompt_1"], ["prompt_2", "prompt_3", "prompt_4"], ["prompt_5"]]
    padded, count = driver.pad_physical_batch(jobs[2:5], physical_batch_size=4)
    assert count == 1 and padded[3] is jobs[4] and padded[:3] == jobs[2:5]

    calls: list[tuple[str, list[str]]] = []

    def fake_latents(_pipe: Any, job: Job, **_kwargs: Any) -> tuple:
        value = torch.full((1, 1, 2, 2), float(job.seed))
        return value, value, pipeline.tensor_sha256(value)

    def guided(_pipe: Any, batch: list[Job], *, diagnostics: list, **_kwargs: Any) -> SimpleNamespace:
        calls.append(("guided", [job.prompt_id for job in batch]))
        for record in diagnostics:
            record.outer_updates_attempted, record.refinement_updates_attempted = 25, 5
            record.guided_scheduler_input_scaling = True
            record.captured_attention_processors = list(pipeline.SD14_EXPECTED_ATTENTION_PROCESSORS)
            record.captured_attention_processor_count = 5
        return SimpleNamespace(images=[job.prompt_id for job in batch])

    def stock(_pipe: Any, batch: list[Job], **_kwargs: Any) -> SimpleNamespace:
        calls.append(("stock", [job.prompt_id for job in batch]))
        return SimpleNamespace(images=[job.prompt_id for job in batch])

    monkeypatch.setattr(driver, "initial_latents", fake_latents)
    monkeypatch.setattr(driver, "generate_predicated_batch", guided)
    monkeypatch.setattr(driver, "generate_stock_batch", stock)
    unet = FakeUNet()
    runtime = driver.SD14Runtime(SimpleNamespace(unet=unet), PROFILES["bf16_stable"], {}, {}, {}, {})

    def batch(method: str, prepared: list) -> list:
        request = driver.SD14Run(method=method, prompt_manifest=Path("m"), out_dir=Path("o"), dataset=None)
        return driver.generate_image_batch(runtime, prepared, request=request, physical_batch_size=4, device="cpu")

    first = batch("sd14_pd_spacy", jobs[:2])
    assert [image for image, _ in first] == ["prompt_0", "prompt_1"]
    assert [(r.effective_batch_size, r.valid_batch_size, r.batch_padding_count) for _, r in first] == [(4, 2, 2)] * 2
    batch("sd14_pd_spacy", jobs[2:5])
    assert unet.installed == [{"down.attn2.processor": "stock"}]
    vanilla = [_prepared(i, guided=False, provider=None) for i in range(3)]
    third = batch("sd14_vanilla", vanilla)
    assert [(r.effective_batch_size, r.valid_batch_size, r.batch_padding_count) for _, r in third] == [(3, 3, 0)] * 3
    assert calls == [("guided", ["prompt_0", "prompt_1", "prompt_1", "prompt_1"]),
                     ("stock", ["prompt_2", "prompt_3", "prompt_4", "prompt_4"]),
                     ("stock", ["prompt_0", "prompt_1", "prompt_2"])]
    with pytest.raises(GenerationContractError, match="cannot mix guided and stock"):
        batch("sd14_pd_spacy", [jobs[1], jobs[2]])


def _target_row(prompt_id: str, prompt: str, *, status: str = "ok") -> dict:
    pd = {"exist": [1], "corr": [], "leak": [], "possession": []} if status == "ok" else \
        {"exist": [], "corr": [], "leak": [], "possession": []}
    program = hashlib.sha256(json.dumps(pd, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"prompt_id": prompt_id, "provider": "spacy_sd14", "status": status,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "program_sha256": program if status == "ok" else None, "pd_indices": pd,
            "error": None if status == "ok" else "alignment failed",
            "parser_provenance": {"backbone": "sd14", "sidecar_provider": "spacy_sd14",
                                  "tokenizer": config.SD14_GENERATION.tokenizer_identity(),
                                  "canonical_arm_id": "sd14_pd_spacy"}}


def _write_inputs(tmp_path: Path, rows: list[dict]) -> tuple[Path, Path]:
    manifest = tmp_path / "images.jsonl"
    manifest.write_text("".join(json.dumps({"experiment": "e", "prompt_id": f"p{i}", "prompt": f"object {i}",
                                            "seed": i}) + "\n" for i in range(len(rows))))
    target = tmp_path / "targets.jsonl"
    target.write_text("".join(json.dumps(row) + "\n" for row in rows))
    (tmp_path / "targets.jsonl.meta.json").write_text(json.dumps(
        {"backbone": "sd14", "provider": "spacy_sd14", "tokenizer": config.SD14_GENERATION.tokenizer_identity()}))
    return manifest, target


def test_guidance_target_file_contract(tmp_path: Path) -> None:
    dataset = SimpleNamespace(require_prepared_jobs=lambda jobs: None)
    rows = [_target_row("p0", "object 0"), _target_row("p1", "object 1", status="alignment_error")]
    manifest, target = _write_inputs(tmp_path, rows)
    with pytest.raises(GenerationContractError, match="is not usable: status='alignment_error'"):
        load_prepared_jobs(target, manifest, method="sd14_pd_spacy", index_failure_policy="error", dataset=dataset)
    jobs = load_prepared_jobs(target, manifest, method="sd14_pd_spacy", index_failure_policy="stock_vanilla",
                              dataset=dataset)
    assert [job.guidance_mode for job in jobs] == ["token_predicated", "stock_vanilla_fallback"]
    assert jobs[1].extraction_error == "alignment failed" and jobs[1].program_sha256 is None
    assert jobs[0].pd_indices_sha256 == jobs[0].program_sha256 == rows[0]["program_sha256"]
    assert jobs[0].sidecar_row_sha256 == hashlib.sha256(
        json.dumps(rows[0], ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    for broken, message in (
        ({**rows[0], "provider": "deps2index_sd14"}, "does not match 'spacy_sd14'"),
        ({**rows[0], "prompt_sha256": "f" * 64}, "prompt hash mismatch"),
        ({**rows[0], "program_sha256": "e" * 64}, "does not match pd_indices"),
    ):
        manifest, target = _write_inputs(tmp_path, [broken, rows[1]])
        with pytest.raises(GenerationContractError, match=message):
            load_prepared_jobs(target, manifest, method="sd14_pd_spacy", index_failure_policy="stock_vanilla",
                               dataset=dataset)
    with pytest.raises(GenerationContractError, match="must not receive a guidance-target file"):
        load_prepared_jobs(target, manifest, method="sd14_vanilla", index_failure_policy="error", dataset=dataset)
    manifest, target = _write_inputs(tmp_path, [rows[0], {**rows[1], "status": "spacy_error"}])
    with pytest.raises(GenerationContractError, match="'spacy_error' is not an allowed fallback"):
        load_prepared_jobs(target, manifest, method="sd14_pd_spacy", index_failure_policy="stock_vanilla",
                           dataset=dataset)


def test_component_files_are_the_import_closure_and_resume_requires_the_same_lock(tmp_path: Path) -> None:
    assert driver.SD14_DRIVER_COMPONENT_FILES == tuple(sorted(driver.SD14_DRIVER_COMPONENT_FILES))
    assert set(driver.SD14_DRIVER_COMPONENT_FILES) == import_closure(
        ROOT / "src/ccg_diffusion/backbones/sd14/driver.py")
    lock = tmp_path / "run_lock.json"
    with pytest.raises(GenerationContractError, match="cannot resume without run lock"):
        ensure_run_lock(lock, {"a": 1}, resume=True)
    created = ensure_run_lock(lock, {"a": 1}, resume=False)
    assert ensure_run_lock(lock, {"a": 1}, resume=True) == created
    with pytest.raises(GenerationContractError, match="run lock already exists"):
        ensure_run_lock(lock, {"a": 1}, resume=False)
    with pytest.raises(GenerationContractError, match="does not match existing run lock"):
        ensure_run_lock(lock, {"a": 2}, resume=True)


# --- the pinned scheduler record ------------------------------------------------------------------------------


@pytest.mark.requires_clip_tokenizer
def test_scheduler_record() -> None:
    from diffusers import PNDMScheduler

    scheduler = PNDMScheduler.from_pretrained(config.SD14_GENERATION.model_id, subfolder="scheduler",
                                              revision=config.SD14_GENERATION.model_revision, local_files_only=True)
    assert scheduler_record(SimpleNamespace(scheduler=scheduler)) == {
        "class": "PNDMScheduler", "config_sha256": config.SD14_SCHEDULER_CONFIG_SHA256, "prediction_type": "epsilon"}


# --- the T2I-CompBench++ batch function (compositional.sd14.run_batch) ----------------------------------------


def _compositional_plan(job: Job, *, applied: bool) -> Any:
    """A Positional Prefix plan on a synthetic coordination (or an unapplied plan) for ``job``."""
    from ccg_diffusion.compositional.plans import CompositionalPlan
    from ccg_diffusion.guidance._record_only_screening import ClauseLayoutPlan, DepthLayoutPlan
    from ccg_diffusion.guidance.positional_prefix import PhraseBindingPlan, PrefixLayoutPlan
    from ccg_diffusion.provenance import canonical_json, sha256_text

    program = {"exist": job.attention_exist_indices, "corr": job.attention_corr_indices,
               "leak": job.attention_leak_indices, "possession": job.attention_possession_indices}
    identity = (sha256_text(job.prompt), sha256_text(canonical_json(program)))
    mapping = ((1, 10), (2, 11), (3, 19), (4, 16), (5, 17))
    mapped = {"exist": [11, 17], "corr": [[10, 11], [16, 17]], "leak": [], "possession": []}
    prefix = PrefixLayoutPlan(
        *identity, sha256_text("[]"), "On the left side, red box. On the right side, blue ball. " + job.prompt
        if applied else None, 7, 21 if applied else 0, "phrase" if applied else "unchanged",
        "left" if applied else None, mapping if applied else (), canonical_json(mapped) if applied else "{}",
        (((10,), (11,)), ((16,), (17,))) if applied else (), None, "0" * 64, "test")
    return CompositionalPlan(PhraseBindingPlan(*identity, sha256_text("[]"), (), "test"),
                             DepthLayoutPlan(*identity, None, None, None, (), (), None, None, "test", "{}"),
                             ClauseLayoutPlan(*identity, None, None, None, (), (), None, 0.0, "test", "{}"), prefix)


@pytest.mark.parametrize("condition_id", ["gs1_p1_r1", "gs1_p1_r0", "gs1_p0_r1", "gs1_p0_r0", "no_count"])
@pytest.mark.parametrize("applied", [(True, False, False, False), (False, False, False, False)])
def test_run_batch_installs_the_planned_prefix_and_region_operations(
    monkeypatch: pytest.MonkeyPatch, condition_id: str, applied: tuple[bool, ...],
) -> None:
    from ccg_diffusion.compositional import sd14
    from ccg_diffusion.methods import T2IAblation

    ablation = T2IAblation.from_condition_id(condition_id)
    jobs = [Job(experiment="t2icb_color", prompt_id=f"p{i}_s{42 + i}", prompt="red box and blue ball", seed=42 + i,
                attention_exist_indices=[2, 5], attention_corr_indices=[[1, 2], [4, 5]], attention_leak_indices=[],
                attention_possession_indices=[], source="test") for i in range(4)]
    plans = [_compositional_plan(job, applied=flag) for job, flag in zip(jobs, applied)]
    calls: list[dict] = []

    def kernel(pipe: Any, batch: list[Job], **kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs)
        states = torch.zeros(2 * len(batch), 77, pipe.hidden_size)
        pipe.unet.attn2.to_k(states)
        pipe.unet.attn2.to_v(states)
        maps = torch.rand(16, 16, 77, generator=torch.Generator().manual_seed(3))
        for job, override in zip(batch, kwargs.get("loss_overrides") or [None] * len(batch)):
            if override is not None:
                override(maps, job.pd_indices(), attention_res=16)
        return SimpleNamespace(images=[None] * len(batch))

    monkeypatch.setattr(sd14, "generate_predicated_batch", kernel)
    diagnostics = [GenerationDiagnostics(semantic_guidance={}) for _ in jobs]
    sd14.run_batch(ablation, FakeSDPipe(), jobs, plans, device="cpu", diagnostics=diagnostics)
    override = applied[0] and (ablation.positional_prefix or ablation.region_assignment_loss)
    assert ("loss_overrides" in calls[0]) is override  # never an all-None list
    if override:
        assert [o is not None for o in calls[0]["loss_overrides"]] == [True, False, False, False]
    for plan, record in zip(plans, diagnostics):
        assert sd14.verify_operation(ablation, plan.as_record(), {"semantic_guidance": record.semantic_guidance}) == (
            plan.positional_prefix and override)
    with pytest.raises(ValueError, match="another loss override"):
        sd14.run_batch(ablation, FakeSDPipe(), jobs, plans, device="cpu", diagnostics=diagnostics, loss_overrides=[])


def test_t2i_run_seals_the_import_closure_of_its_modules() -> None:
    from ccg_diffusion.compositional import sd14

    assert sd14.COMPONENT_FILES == tuple(sorted(sd14.COMPONENT_FILES))
    assert not set(sd14.COMPONENT_FILES) & set(driver.SD14_DRIVER_COMPONENT_FILES)
    assert set(sd14.component_files()) == import_closure(ROOT / "src/ccg_diffusion/compositional/sd14.py") | {
        "scripts/t2i_compbench/run_ccg_diffusion.py"}
