"""Generation-time benchmark: prompt-to-image time of Vanilla, Predicated Diffusion and CCG Diffusion.

    python -m scripts.paper.compute_cost benchmark --backbone sd14 --out DIR [--modes b1 b4] [--limit N]
    python -m scripts.paper.compute_cost benchmark --backbone sana --out DIR [--modes b1 b4] [--limit N]
    python -m scripts.paper.compute_cost summarize DIR          # writes DIR/summary.json

``benchmark`` times each method on T2I-CompBench++ prompts (by default the 30
per category of ``data/compute_cost/prompts.json``). The timed region starts
when the prompt string is handed to a method and ends when its images exist as
PIL images in host memory: it includes parsing, guidance-target and plan
construction, text encoding, denoising, VAE decoding and PIL conversion; it
excludes model loading, PNG writing and every verification. Methods:

* ``vanilla``: the backbone without guidance;
* ``spacy_pd``: Predicated Diffusion with its spaCy guidance targets;
* ``ours``: CCG Diffusion (Hol-CCG parse, C&C marked dependencies, guidance
  targets, component plan and guided denoising).

Mode ``b1`` generates one image (seed 42) per prompt, ``b4`` four (seeds
42-45). The methods rotate per prompt; a GPU guard (NVML) waits until no other
process computes on the measured GPU and re-measures prompt groups that shared
it. Run one backbone per process (the SD1.4 determinism settings are
process-global). Unless ``--no-verify`` is given, every result is then compared,
outside the timed region, with the Stage 1-3 outputs, plans and images of the
evaluated T2I-CompBench++ runs under ``--reference-root``; without those run
outputs, pass ``--no-verify``. ``DIR/<backbone>/`` receives ``timings.jsonl``,
``verification.jsonl``, ``gpu_guard.jsonl``, ``environment.json`` and the images.

``summarize`` writes ``DIR/summary.json`` (per backbone, mode and method:
seconds per image, latency percentiles, pre-generation time and its share,
phase means; per-pair time ratios with a paired prompt bootstrap). The
generation time per image of a method is its ``s_per_image`` in mode ``b1``;
the benchmark's measured and warm-up prompts are the defaults
(``data/compute_cost/``).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterator, Sequence

import numpy as np

# The measured prompts (30 per category) and the warm-up prompts of the generation-time benchmark.
DEFAULT_PROMPTS = "data/compute_cost/prompts.json"
DEFAULT_WARMUP = "data/compute_cost/warmup.json"

# --------------------------------------------------------------------------------------------------------
# summarize
# --------------------------------------------------------------------------------------------------------

GENERATION_PHASES = {"generation", "text_encode"}
# Method pairs compared in summary.json; a pair with a method that has no timings is skipped.
COMPARISONS = {"sd14": [("vanilla", "ours"), ("spacy_pd", "ours"), ("vanilla_table2", "ours")],
               "sana": [("vanilla", "ours"), ("spacy_pd", "ours")]}


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def bootstrap_ratio(base: np.ndarray, other: np.ndarray, *, seed: int = 0, draws: int = 20000) -> dict[str, float]:
    """Ratio of total time (other / base) with a paired prompt bootstrap."""
    rng = np.random.default_rng(seed)
    n = len(base)
    picks = rng.integers(0, n, size=(draws, n))
    ratios = other[picks].sum(1) / base[picks].sum(1)
    return {"ratio": float(other.sum() / base.sum()),
            "ci95": [float(np.quantile(ratios, .025)), float(np.quantile(ratios, .975))]}


def summarize(directory: Path) -> dict[str, Any]:
    """Per-backbone method summaries, pairwise comparisons and verification counts of one benchmark directory."""
    result: dict[str, Any] = {}
    for backbone_dir in sorted(p for p in directory.iterdir() if (p / "timings.jsonl").exists()):
        backbone = backbone_dir.name
        records = rows(backbone_dir / "timings.jsonl")
        by: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
        for r in records:
            by[r["mode"], r["method"]][r["prompt_id"]] = r
        summary: dict[str, Any] = {}
        for (mode, method), items in sorted(by.items()):
            total = np.array([r["total_s"] for r in items.values()])
            images = sum(r["images"] for r in items.values())
            text_side = np.array([sum(v for k, v in r["phases_s"].items() if k not in GENERATION_PHASES)
                                  for r in items.values()])
            phases: dict[str, float] = defaultdict(float)
            for r in items.values():
                for k, v in r["phases_s"].items():
                    phases[k] += v
            categories: dict[str, list[float]] = defaultdict(list)
            for r in items.values():
                categories[r["category"]].append(r["total_s"])
            routes: dict[str, list[float]] = defaultdict(list)
            for r in items.values():
                if "route" in r:
                    routes[f"{r['route']}{'+prefix' if r.get('prefix_active') else ''}"
                           f"{'+count' if r.get('count_constraints') else ''}"].append(r["total_s"])
            summary.setdefault(mode, {})[method] = {
                "prompts": len(items), "images": images, "total_s": float(total.sum()),
                "images_per_s": images / float(total.sum()), "s_per_image": float(total.sum()) / images,
                "prompt_latency_s": {"mean": float(total.mean()), "median": float(np.median(total)),
                                     "p90": float(np.quantile(total, .9)), "min": float(total.min()),
                                     "max": float(total.max())},
                "pre_generation_s": {"mean": float(text_side.mean()), "median": float(np.median(text_side)),
                                     "max": float(text_side.max()),
                                     "share_of_total": float(text_side.sum() / total.sum())},
                "phase_mean_s": {k: v / len(items) for k, v in sorted(phases.items())},
                "category_mean_s": {k: float(np.mean(v)) for k, v in sorted(categories.items())},
                "route_mean_s": {k: {"n": len(v), "mean": float(np.mean(v))} for k, v in sorted(routes.items())},
                "peak_allocated_gib_max": max(r["peak_allocated_gib"] for r in items.values()),
                "fallback_prompts": sum(1 for r in items.values() if r.get("guided") is False),
            }
        comparisons: dict[str, Any] = {}
        for mode in summary:
            for base, other in COMPARISONS[backbone]:
                if (mode, base) not in by or (mode, other) not in by:
                    continue
                keys = sorted(set(by[mode, base]) & set(by[mode, other]))
                b = np.array([by[mode, base][k]["total_s"] for k in keys])
                o = np.array([by[mode, other][k]["total_s"] for k in keys])
                per_category = {}
                for category in sorted({by[mode, base][k]["category"] for k in keys}):
                    ck = [k for k in keys if by[mode, base][k]["category"] == category]
                    per_category[category] = float(sum(by[mode, other][k]["total_s"] for k in ck)
                                                   / sum(by[mode, base][k]["total_s"] for k in ck))
                comparisons.setdefault(mode, {})[f"{other}_vs_{base}"] = {
                    "prompts": len(keys), **bootstrap_ratio(b, o),
                    "throughput_ratio": float(b.sum() / o.sum()),
                    "extra_s_per_prompt_mean": float((o - b).mean()),
                    "ratio_by_category": per_category}
        verification = summarize_verification(backbone_dir / "verification.jsonl")
        result[backbone] = {"methods": summary, "comparisons": comparisons, "verification": verification}
    return result


def summarize_verification(path: Path) -> dict[str, Any]:
    """Counts of the boolean verification fields and of the image comparisons per mode and method."""
    counts: dict[str, dict[str, Any]] = defaultdict(lambda: defaultdict(int))
    for r in rows(path):
        key = f"{r['mode']}/{r['method']}"
        for field, value in r.items():
            if isinstance(value, bool):
                counts[key][f"{field}:{value}"] += 1
            elif field == "status":
                counts[key][f"status:{value}"] += 1
        for image in r["images"]:
            if image is None:
                counts[key]["image:no_reference"] += 1
            elif image.get("exact"):
                counts[key]["image:exact"] += 1
            else:
                counts[key]["image:differs"] += 1
                counts[key].setdefault("image_psnr_differs", []).append(image.get("psnr"))
    out = {}
    for key, value in counts.items():
        psnr = value.pop("image_psnr_differs", [])
        out[key] = dict(value)
        if psnr:
            out[key]["image_psnr_differs_median"] = float(np.median(psnr))
    return out


# --------------------------------------------------------------------------------------------------------
# benchmark: timing helpers (torch and the models are imported inside the functions that need them)
# --------------------------------------------------------------------------------------------------------


def roundtrip(value: Any) -> Any:
    """Match the JSON file boundary that the recorded artifacts cross between stages."""
    return json.loads(json.dumps(value))


@contextlib.contextmanager
def backend(*, deterministic: bool) -> Iterator[None]:
    """Temporarily set ``torch.use_deterministic_algorithms`` (the Stage 1 parse ran without it)."""
    import torch

    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(deterministic)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous)


class GpuGuard:
    """Detect other processes computing on the measured GPU so that shared intervals can be re-measured.

    ``presence`` treats any other process holding a context as interference. ``activity`` treats an
    other process as interference only when NVML reports SM or memory utilization samples for it; an
    idle context is not scheduled on the GPU and does not share its time slices.
    """

    def __init__(self, uuid: str, *, quiet_s: float, policy: str, log: Callable[[dict[str, Any]], None]) -> None:
        import pynvml
        import torch

        self.nvml, self.quiet_s, self.policy, self.log = pynvml, quiet_s, policy, log
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByUUID(uuid.encode())
        before = set(self.pids())
        torch.zeros(1, device="cuda")
        while True:
            pids = self.pids()
            new = [pid for pid in pids if pid not in before]
            if len(pids) == 1 or (policy == "activity" and len(new) == 1):
                self.own = pids[0] if len(pids) == 1 else new[0]
                break
            log({"guard": "waiting_to_identify_own_process", "pids": pids})
            time.sleep(5)
        self.seen: set[int] = set()
        self.last_foreign = time.time()
        self.last_sample_us = int(time.time() * 1e6)
        self._stop: Any = None

    def pids(self) -> list[int]:
        return [p.pid for p in self.nvml.nvmlDeviceGetComputeRunningProcesses(self.handle)]

    def _active(self) -> list[int]:
        try:
            samples = self.nvml.nvmlDeviceGetProcessUtilization(self.handle, self.last_sample_us)
        except self.nvml.NVMLError_NotFound:
            return []
        if samples:
            self.last_sample_us = max(self.last_sample_us, max(s.timeStamp for s in samples))
        return sorted({s.pid for s in samples if s.pid != self.own and (s.smUtil > 0 or s.memUtil > 0)})

    def foreign(self) -> list[int]:
        if self.policy == "activity":
            others = self._active()
        else:
            others = [pid for pid in self.pids() if pid != self.own]
        if others:
            self.last_foreign = time.time()
        return others

    def wait_clear(self) -> float:
        """Return once no other process has been seen on the GPU for ``quiet_s`` seconds."""
        started = time.time()
        while True:
            others = self.foreign()
            if others:
                self.log({"guard": "foreign_process", "pids": others})
                time.sleep(5)
                continue
            if time.time() - self.last_foreign >= self.quiet_s:
                return time.time() - started
            time.sleep(1)

    def start(self) -> None:
        import threading

        self.seen = set()
        self._stop = threading.Event()

        def watch() -> None:
            while not self._stop.is_set():
                self.seen.update(self.foreign())
                self._stop.wait(0.5)

        self._thread = threading.Thread(target=watch, daemon=True)
        self._thread.start()

    def stop(self) -> set[int]:
        self._stop.set()
        self._thread.join()
        self.seen.update(self.foreign())
        return set(self.seen)


class Clock:
    """Wall clock with CUDA synchronization at every phase boundary."""

    def __init__(self) -> None:
        import torch

        self.cuda = torch.cuda
        self.phases: dict[str, float] = {}
        self.cuda.synchronize()
        self.start = self.last = time.perf_counter()

    def mark(self, name: str) -> None:
        self.cuda.synchronize()
        now = time.perf_counter()
        self.phases[name] = self.phases.get(name, 0.0) + now - self.last
        self.last = now

    def total(self) -> float:
        self.cuda.synchronize()
        return time.perf_counter() - self.start


# --------------------------------------------------------------------------------------------------------
# benchmark: Hol-CCG front end shared by both backbones (Stage 1-3 of the paper pipeline)
# --------------------------------------------------------------------------------------------------------


class FrontEnd:
    """Stage 1 Hol-CCG parse, Stage 2 C&C ``generate`` and Stage 3 guidance targets for one prompt.

    ``parsing.frontend.CCGFrontEnd`` holds the parser and one resident
    ``generate`` engine per output mode (models load once, like any other
    model); Stage 3 is ``build_guidance_target_rows`` without a sentence map,
    as for an arbitrary prompt. The parse runs without deterministic algorithms,
    as the recorded Stage 1 did.
    """

    def __init__(self, device: str, clip: Any) -> None:
        from ccg_diffusion.config import HOLCCG_PARSE
        from ccg_diffusion.parsing.frontend import CCGFrontEnd
        from ccg_diffusion.targets.guidance_targets import load_possession_lemmatizer
        from ccg_diffusion.targets.tokenizer import require_sd14_tokenizer

        self.settings = {"checkpoint": HOLCCG_PARSE.checkpoint, "revision": HOLCCG_PARSE.revision,
                         **{key: getattr(HOLCCG_PARSE, key) for key in (
                             "stag_threshold", "phrase_threshold", "span_threshold", "apply_skimmer",
                             "min_rule_freq", "grammar", "chunk_size")}}
        with backend(deterministic=False):
            self.engine = CCGFrontEnd(device)
        self.resource_sha256 = self.engine.resource_sha256
        self.clip, self.identity = clip, require_sd14_tokenizer(clip)
        self.nlp = load_possession_lemmatizer()

    def run(self, prompt: str, prompt_id: str, category: str, index: int, clock: Clock) -> SimpleNamespace:
        from ccg_diffusion.targets.guidance_targets import build_guidance_target_rows

        with backend(deterministic=False):
            parsed = self.engine.parse(prompt, sentence_id=f"{category}.{index + 1}")
        clock.mark("stage1_2_holccg_parse_candc_generate")
        targets, traces = build_guidance_target_rows(
            [{"prompt_id": prompt_id, "prompt": prompt}], [parsed.sentence], [list(parsed.dependencies)], self.clip,
            self.nlp, category=category, coverage_rows=[parsed.coverage], tokenizer_identity=self.identity)
        row, trace = roundtrip(targets[0]), roundtrip(traces[0]["conversion_trace"])
        clock.mark("stage3_guidance_targets")
        return SimpleNamespace(derivation=parsed.derivation, sentence=parsed.sentence,
                               dependencies=list(parsed.dependencies), row=row, trace=trace)


def compiled_targets(row: dict[str, Any], trace: dict[str, Any]) -> Any:
    """The Stage 3 result of one prompt as ``CompiledGuidanceTargets`` (requires a valid program)."""
    from ccg_diffusion.targets.guidance_targets import CompiledGuidanceTargets
    from ccg_diffusion.targets.pd_indices import PDIndices

    if row["status"] != "ok":
        raise ValueError(f"CCG Diffusion requires valid guidance targets: {row['prompt_id']} {row['status']}")
    return CompiledGuidanceTargets(PDIndices(**row["pd_indices"]), tuple(row["alignment"]), trace)


# --------------------------------------------------------------------------------------------------------
# benchmark: Stable Diffusion v1.4
# --------------------------------------------------------------------------------------------------------


class SD14Bench:
    """The three methods on SD1.4."""

    methods = ("vanilla", "spacy_pd", "ours")

    def __init__(self, device: str) -> None:
        from ccg_diffusion.backbones.sd14.driver import prepare_sd14_runtime
        from ccg_diffusion.baselines.predicated_diffusion import SpacyProvider
        from ccg_diffusion.guidance.count_loss import load_noun_forms
        from ccg_diffusion.targets.lexicon import load_nominal_modifier_lexicon
        from ccg_diffusion.targets.tokenizer import load_sd14_tokenizer

        self.device = device
        # Determinism is configured before the pipeline load and before any CUDA matmul.
        self.runtime = prepare_sd14_runtime(device)
        self.clip = load_sd14_tokenizer()
        self.frontend = FrontEnd(device, self.clip)
        self.spacy = SpacyProvider(self.clip)
        self.forms = load_noun_forms()
        self.lexicon = load_nominal_modifier_lexicon()
        self.plans: dict[str, Any] = {}

    def environment(self) -> dict[str, Any]:
        from ccg_diffusion.config import SD14_GENERATION, SD14_GUIDANCE_SCHEDULE
        from ccg_diffusion.methods import T2I_SD14, get_config

        return {"configuration_id": T2I_SD14, "configuration_sha256": get_config(T2I_SD14).sha256,
                "generation": asdict(SD14_GENERATION), "guidance_schedule": asdict(SD14_GUIDANCE_SCHEDULE),
                "scheduler": self.runtime.scheduler, "precision_profile": self.runtime.precision.as_record(),
                "determinism": self.runtime.determinism, "latent_bound": self.runtime.latent_abs_bound,
                "parser": self.frontend.settings, "candc_resource_sha256": self.frontend.resource_sha256}

    def _request(self, method: str, **guidance: Any) -> Any:
        """The driver request whose fields ``generate_image_batch`` reads (no run directory is written)."""
        from ccg_diffusion.backbones.sd14.driver import SD14Run
        from ccg_diffusion.config import SD14_CCG_ARM, SD14_PREDICATED_DIFFUSION_ARM

        arm = SD14_CCG_ARM if method == "ours" else SD14_PREDICATED_DIFFUSION_ARM
        return SD14Run(method=arm, prompt_manifest=Path(os.devnull), out_dir=Path(os.devnull), dataset=None,
                       device=self.device, **guidance)

    def _prepared(self, item: dict[str, Any], seeds: Sequence[int], row: dict[str, Any] | None,
                  policy: str) -> list[Any]:
        from ccg_diffusion.backbones.sd14.inputs import prepared_job_from_target_row

        return [prepared_job_from_target_row(row, experiment=f"t2icb_{item['category']}",
                                             prompt_id=f"{item['prompt_id']}_s{seed}", prompt=item["prompt"],
                                             seed=seed, index_failure_policy=policy) for seed in seeds]

    def _generate(self, request: Any, jobs: list[Any]) -> list[Any]:
        from ccg_diffusion.backbones.sd14.driver import generate_image_batch

        return [image for image, _ in generate_image_batch(self.runtime, jobs, request=request,
                                                           physical_batch_size=len(jobs), device=self.device)]

    def _ours_batch(self, pipe: Any, jobs: Sequence[Any], **kwargs: Any) -> Any:
        """``compositional.sd14``'s checked batch function; the batch check applies to batches of four."""
        from ccg_diffusion.compositional import sd14
        from ccg_diffusion.config import SD14_GENERATION
        from ccg_diffusion.methods import FULL_T2I
        from ccg_diffusion.provenance import require

        if len(jobs) == SD14_GENERATION.batch_size:
            sd14.check_batch(pipe, jobs, kwargs)
        batch_plans = [self.plans[job.prompt_id.rsplit("_s", 1)[0]] for job in jobs]
        result = sd14.run_batch(FULL_T2I, pipe, jobs, batch_plans, **kwargs)
        for plan, diagnostic in zip(batch_plans, kwargs["diagnostics"], strict=True):
            require(diagnostic.captured_attention_processor_count == SD14_GENERATION.expected_attention_processor_count,
                    "actual attention processor count differs from the paper configuration")
            sd14.verify_operation(FULL_T2I, plan.as_record(), {"semantic_guidance": diagnostic.semantic_guidance})
        return result

    def vanilla(self, item: dict[str, Any], seeds: Sequence[int], clock: Clock) -> tuple[list[Any], dict]:
        images = self._generate(self._request("vanilla"), self._prepared(item, seeds, None, "error"))
        clock.mark("generation")
        return images, {"guidance": "stock"}

    def spacy_pd(self, item: dict[str, Any], seeds: Sequence[int], clock: Clock) -> tuple[list[Any], dict]:
        prompt, prompt_id = item["prompt"], item["prompt_id"]
        row = roundtrip(self.spacy.extract(prompt).target_row(prompt_id=prompt_id, prompt=prompt))
        clock.mark("spacy_parse_index")
        jobs = self._prepared(item, seeds, row, "stock_vanilla")
        images = self._generate(self._request("spacy_pd"), jobs)
        clock.mark("generation")
        return images, {"status": row["status"], "guided": jobs[0].guidance_applied, "target_row": row}

    def ours(self, item: dict[str, Any], seeds: Sequence[int], clock: Clock) -> tuple[list[Any], dict]:
        from ccg_diffusion.compositional.plans import plan_prompt

        prompt, prompt_id = item["prompt"], item["prompt_id"]
        front = self.frontend.run(prompt, prompt_id, item["category"], item["index"], clock)
        planned = plan_prompt(prompt, front.sentence, front.dependencies, compiled_targets(front.row, front.trace),
                              self.clip, noun_forms=self.forms, lexicon=self.lexicon)
        clock.mark("plan")
        self.plans = {prompt_id: planned.plan}
        request = self._request("ours", image_guidance={prompt_id: planned.guidance},
                                count_guidance={prompt_id: planned.count}, batch_fn=self._ours_batch)
        images = self._generate(request, self._prepared(item, seeds, front.row, "error"))
        clock.mark("generation")
        return images, {"front": front, "plan": planned.plan.as_record(), "guidance": planned.guidance.as_record(),
                        "count": planned.count.as_record(), "route": planned.guidance.effective_profile,
                        "prefix_active": planned.plan.positional_prefix,
                        "count_constraints": len(planned.count.decision.constraints)}


# --------------------------------------------------------------------------------------------------------
# benchmark: Sana 0.6B
# --------------------------------------------------------------------------------------------------------


class SanaBench:
    """The three methods on Sana 0.6B (T2I-CompBench++ settings of ``config.SANA_GENERATION``)."""

    methods = ("vanilla", "spacy_pd", "ours")

    def __init__(self, device: str) -> None:
        from ccg_diffusion.backbones.sana.pipeline import (
            COMPLEX_HUMAN_INSTRUCTION, load_sana_pipeline, load_sana_tokenizer)
        from ccg_diffusion.baselines.predicated_diffusion import SpacyProvider
        from ccg_diffusion.guidance.count_loss import load_noun_forms
        from ccg_diffusion.targets.tokenizer import load_sd14_tokenizer

        self.device, self.instruction = device, COMPLEX_HUMAN_INSTRUCTION
        self.pipe = load_sana_pipeline(device)
        self.gemma = load_sana_tokenizer()
        self.clip = load_sd14_tokenizer()
        self.frontend = FrontEnd(device, self.clip)
        self.spacy = SpacyProvider(self.clip)
        self.forms = load_noun_forms()
        # Both guided runners encode the empty negative prompt once and reuse it for every prompt.
        import torch

        with torch.no_grad():
            values = self.pipe.encode_prompt("a photo", True, negative_prompt="", device=device, clean_caption=False,
                                             max_sequence_length=300, complex_human_instruction=self.instruction)
        self.negative = {"negative_prompt_embeds": values[2], "negative_prompt_attention_mask": values[3]}

    def environment(self) -> dict[str, Any]:
        from ccg_diffusion.config import SANA_GENERATION
        from ccg_diffusion.methods import T2I_SANA, get_config

        return {"configuration_id": T2I_SANA, "configuration_sha256": get_config(T2I_SANA).sha256,
                "generation": asdict(SANA_GENERATION), "parser": self.frontend.settings,
                "candc_resource_sha256": self.frontend.resource_sha256,
                "scheduler": type(self.pipe.scheduler).__name__}

    def _encode(self, text: str) -> tuple[Any, ...]:
        import torch

        with torch.no_grad():
            values = self.pipe.encode_prompt(text, True, negative_prompt="", num_images_per_prompt=1,
                                             device=self.device, clean_caption=False, max_sequence_length=300,
                                             complex_human_instruction=self.instruction, **self.negative)
        return tuple(value.detach() if value is not None else None for value in values)

    def _generators(self, seeds: Sequence[int]) -> list[Any]:
        import torch

        return [torch.Generator(device=self.device).manual_seed(seed) for seed in seeds]

    def vanilla(self, item: dict[str, Any], seeds: Sequence[int], clock: Clock) -> tuple[list[Any], dict]:
        from ccg_diffusion.backbones.sana.pipeline import generate_stock_from_text
        from ccg_diffusion.config import SANA_GENERATION as S

        images, _ = generate_stock_from_text(
            self.pipe, item["prompt"], self._generators(seeds), device=self.device, steps=S.steps,
            guidance_scale=S.guidance_scale, height=S.height, width=S.width, instruction=self.instruction)
        clock.mark("generation")
        return list(images), {"guidance": "stock"}

    def spacy_pd(self, item: dict[str, Any], seeds: Sequence[int], clock: Clock) -> tuple[list[Any], dict]:
        from ccg_diffusion.baselines.sana_generation import SanaBaselineRuntime
        from ccg_diffusion.baselines.sana_plans import compile_plan

        prompt, prompt_id = item["prompt"], item["prompt_id"]
        # The paper's plans were compiled from the seed-42 row of the expanded spaCy guidance-target file.
        row = roundtrip(self.spacy.extract(prompt).target_row(prompt_id=f"{prompt_id}_s42", prompt=prompt))
        plan = roundtrip(compile_plan("sana_pd_spacy", prompt, row, tokenizer=self.gemma, clip_tokenizer=self.clip,
                                      instruction=self.instruction))
        clock.mark("spacy_parse_index")
        encodings = {plan["conditioning"]: self._encode(plan["conditioning"])}
        clock.mark("text_encode")
        images, _ = SanaBaselineRuntime(self.pipe, plan, encodings, device=self.device).generate(
            list(seeds), selected=[])
        clock.mark("generation")
        return list(images), {"status": plan["status"], "guided": plan["status"] == "ok", "target_row": row,
                              "plan": plan}

    def ours(self, item: dict[str, Any], seeds: Sequence[int], clock: Clock) -> tuple[list[Any], dict]:
        from ccg_diffusion.backbones.sana.guidance import generate_guided
        from ccg_diffusion.compositional.sana import CompositionalLoss, compile_plan

        prompt, prompt_id = item["prompt"], item["prompt_id"]
        front = self.frontend.run(prompt, prompt_id, item["category"], item["index"], clock)
        plan = roundtrip(compile_plan(
            prompt_id=prompt_id, prompt=prompt, sentence=front.sentence, dependencies=front.dependencies,
            compiled=compiled_targets(front.row, front.trace), clip_tokenizer=self.clip,
            gemma_tokenizer=self.gemma, instruction=self.instruction, noun_forms=self.forms))
        clock.mark("plan")
        encoding = self._encode(plan["conditioning"])
        clock.mark("text_encode")
        losses = [CompositionalLoss(plan, identity=f"{prompt_id}_s{seed}") for seed in seeds]
        images, _ = generate_guided(self.pipe, plan["conditioning"], losses, self._generators(seeds),
                                    prompt_encoding=encoding, instruction=self.instruction)
        clock.mark("generation")
        return list(images), {"front": front, "plan": plan, "route": plan["source_guidance"]["effective_profile"],
                              "prefix_active": plan["prefix_active"], "count_constraints": len(plan["counts"])}


# --------------------------------------------------------------------------------------------------------
# benchmark: verification against the paper's recorded artifacts (outside timing)
# --------------------------------------------------------------------------------------------------------

STAGE123 = {"sd14": "output/generation/20260812_t2icb_sd14_six_arm_full_v1/input/stage123",
            "sana": "output/generation/20260919_sana06_native512_full_v1/stage123"}
SD14_DECISIONS = "output/generation/20260925_sd14_shape_prefix_region_v1/input/decisions.jsonl"
SANA_PLANS = "output/generation/20260925_sana06_shape_prefix_region_v1/final_shape_protocol.plans.jsonl"
SANA_BASELINE_PLANS = "output/generation/20260919_sana_five_baselines_full_v1/protocol.plans.jsonl"
REFERENCE_IMAGES = {
    ("sd14", "spacy_pd"): ["output/generation/20260812_t2icb_sd14_six_arm_full_v2/sd14_pd_spacy/t2icb_{cat}"],
    ("sd14", "ours"): ["output/generation/20260925_sd14_shape_prefix_region_v1/composed/final_shape/sd14_pd_holccg/"
                       "t2icb_{cat}",
                       "output/generation/20260909_sd14_targeted_selected_full_v1/composed/selected/sd14_pd_holccg/"
                       "t2icb_{cat}"],
    ("sana", "vanilla"): ["output/generation/20260919_sana06_native512_full_v1/vanilla/{cat}/sana_baseline/"
                          "t2icb_{cat}"],
    ("sana", "spacy_pd"): ["output/generation/20260919_sana_five_baselines_full_v1/sana_pd_spacy/{cat}/sana_pd_spacy/"
                           "t2icb_{cat}"],
    ("sana", "ours"): ["output/generation/20260925_sana06_shape_prefix_region_v1/combined/sana_ccg_compositional_v1/"
                       "t2icb_{cat}",
                       "output/generation/20260919_sana06_native512_full_v1/final/{cat}/sana_ccg_compositional_v1/"
                       "t2icb_{cat}"],
}


def missing_references(backbone: str, root: Path, categories: Sequence[str]) -> list[str]:
    """The recorded files that ``References`` reads for ``categories`` and that are missing under ``root``."""
    stage123 = root / STAGE123[backbone]
    files = [SD14_DECISIONS] if backbone == "sd14" else [SANA_PLANS, SANA_BASELINE_PLANS]
    missing = [name for name in files if not (root / name).is_file()]
    for category in categories:
        for name in (f"{category}.auto", f"{category}.markedup_deps", f"{category}.deps2index_sd14.jsonl",
                     f"{category}.deps2index_sd14.jsonl.conversion-trace.jsonl"):
            if not (stage123 / name).exists() and next(stage123.rglob(name), None) is None:
                missing.append(f"{STAGE123[backbone]}/{name}")
        if not (root / STAGE123["sd14"] / f"{category}.spacy_sd14.jsonl").exists():
            missing.append(f"{STAGE123['sd14']}/{category}.spacy_sd14.jsonl")
    return missing


class References:
    """The recorded Stage 1-3 outputs, plans and images of the evaluated runs under ``root``."""

    def __init__(self, backbone: str, root: Path) -> None:
        from ccg_diffusion.provenance import read_jsonl

        self.backbone, self.root = backbone, root
        self.stage123 = root / STAGE123[backbone]
        self._cache: dict[str, Any] = {}
        if backbone == "sd14":
            self.decisions = {r["prompt_id"]: r for r in read_jsonl(root / SD14_DECISIONS)}
        else:
            self.plans = {r["prompt_id"]: r for r in read_jsonl(root / SANA_PLANS)}
            self.spacy_plans = {r["prompt_id"]: r for r in read_jsonl(root / SANA_BASELINE_PLANS)
                                if r["method"] == "sana_pd_spacy"}

    def _file(self, name: str) -> Path:
        direct = self.stage123 / name
        if direct.exists():
            return direct
        found = sorted(self.stage123.rglob(name))
        if len(found) != 1:
            raise FileNotFoundError(f"{name}: {found}")
        return found[0]

    def category(self, category: str) -> dict[str, Any]:
        if category not in self._cache:
            from ccg_diffusion.parsing.ccg import load_auto_sentences, load_marked_dependency_blocks
            from ccg_diffusion.provenance import read_jsonl

            targets = self._file(f"{category}.deps2index_sd14.jsonl")
            self._cache[category] = {
                "auto": [s.derivation for s in load_auto_sentences(self._file(f"{category}.auto"))],
                "deps": load_marked_dependency_blocks(self._file(f"{category}.markedup_deps")),
                "targets": read_jsonl(targets),
                "trace": read_jsonl(Path(str(targets) + ".conversion-trace.jsonl")),
                "spacy": read_jsonl(self.root / STAGE123["sd14"] / f"{category}.spacy_sd14.jsonl"),
            }
        return self._cache[category]

    def check(self, method: str, item: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
        from ccg_diffusion.compositional.plans import evaluated_operation_projection, operation_projection
        from ccg_diffusion.provenance import canonical_json

        ref = self.category(item["category"])
        i = item["index"]
        result: dict[str, Any] = {}
        if method == "ours":
            front = info["front"]
            if ref["targets"][i]["prompt_id"] != item["prompt_id"]:
                raise ValueError("reference row order differs")
            result["auto_equal"] = front.derivation == ref["auto"][i]
            result["deps_equal"] = front.dependencies == ref["deps"][i]
            result["pd_indices_equal"] = front.row["pd_indices"] == ref["targets"][i]["pd_indices"]
            result["alignment_equal"] = front.row["alignment"] == ref["targets"][i]["alignment"]
            result["status"] = front.row["status"]
            result["trace_equal"] = canonical_json(front.trace) == canonical_json(ref["trace"][i]["conversion_trace"])
            if self.backbone == "sd14":
                decision = self.decisions[item["prompt_id"]]
                # The evaluated decision rows store the plan in their own record shape: compare what is executed.
                result["plan_equal"] = canonical_json(operation_projection(
                    info["plan"], info["guidance"], info["count"])) == canonical_json(
                    evaluated_operation_projection(decision, record_format="shape_v2"))
                result["guidance_equal"] = canonical_json(info["guidance"]) == canonical_json(decision["guidance"])
                result["count_equal"] = canonical_json(info["count"]) == canonical_json(decision["count"])
            else:
                result["plan_equal"] = canonical_json(info["plan"]) == canonical_json(self.plans[item["prompt_id"]])
        elif method == "spacy_pd":
            spacy_ref = ref["spacy"][i]
            if spacy_ref["prompt_id"] != item["prompt_id"]:
                raise ValueError("reference spaCy row order differs")
            row = info["target_row"]
            result["status"] = row["status"]
            result["pd_indices_equal"] = row["pd_indices"] == spacy_ref["pd_indices"]
            result["status_equal"] = row["status"] == spacy_ref["status"]
            result["alignment_equal"] = row["alignment"] == spacy_ref["alignment"]
            if self.backbone == "sana":
                reference = self.spacy_plans[item["prompt_id"]]
                keys = ("conditioning", "pd_indices", "status", "fallback_reason", "text_token_count",
                        "lexical_indices", "token_offsets")
                result["plan_equal"] = all(canonical_json(info["plan"].get(k)) == canonical_json(reference.get(k))
                                           for k in keys)
        return result

    def image_path(self, method: str, item: dict[str, Any], seed: int) -> Path | None:
        for template in REFERENCE_IMAGES.get((self.backbone, method), []):
            directory = self.root / template.format(cat=item["category"])
            found = sorted(directory.glob(f"{item['prompt_id']}_s{seed}_*/seed_{seed:05d}.png"))
            if found:
                return found[0]
        return None


def compare_image(image: Any, path: Path | None) -> dict[str, Any] | None:
    from PIL import Image

    if path is None:
        return None
    mine = np.asarray(image.convert("RGB"), dtype=np.int16)
    with Image.open(path) as handle:
        ref = np.asarray(handle.convert("RGB"), dtype=np.int16)
    if mine.shape != ref.shape:
        return {"path": str(path), "shape_mismatch": [list(mine.shape), list(ref.shape)]}
    diff = np.abs(mine - ref)
    mse = float((diff.astype(np.float64) ** 2).mean())
    return {"path": str(path), "exact": bool(not diff.any()), "max_abs": int(diff.max()),
            "mean_abs": float(diff.mean()), "psnr": float("inf") if mse == 0 else float(10 * np.log10(255 ** 2 / mse))}


# --------------------------------------------------------------------------------------------------------
# benchmark: driver
# --------------------------------------------------------------------------------------------------------


def completed_groups(out: Path, method_count: int) -> set[tuple[str, str]]:
    """Prompt groups with every method recorded; records of an interrupted group are set aside."""
    import shutil
    from collections import Counter

    path = out / "timings.jsonl"
    if not path.exists():
        return set()
    counts = Counter((r["mode"], r["prompt_id"]) for r in rows(path))
    complete = {key for key, count in counts.items() if count == method_count}
    partial = set(counts) - complete
    if partial:
        stamp = int(time.time())
        for name in ("timings.jsonl", "verification.jsonl"):
            target = out / name
            if target.exists():
                shutil.copy2(target, out / f"{name}.before_resume_{stamp}")
                kept = [r for r in rows(target) if (r["mode"], r["prompt_id"]) not in partial]
                target.write_text("".join(json.dumps(r) + "\n" for r in kept))
    return complete


def environment_record(device: str) -> dict[str, Any]:
    import diffusers
    import spacy
    import torch
    import transformers

    from ccg_diffusion.provenance import repository_root

    def run(command: list[str]) -> str:
        try:
            return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            return f"unavailable: {exc}"

    root = str(repository_root())
    cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")), None)
    return {"hostname": platform.node(), "python": sys.version.split()[0], "torch": torch.__version__,
            "cuda": torch.version.cuda, "diffusers": diffusers.__version__, "transformers": transformers.__version__,
            "spacy": spacy.__version__, "gpu": torch.cuda.get_device_name(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "cpu": cpu, "cpu_count": os.cpu_count(),
            "torch_threads": torch.get_num_threads(), "loadavg": os.getloadavg(),
            "git_head": run(["git", "-C", root, "rev-parse", "HEAD"]),
            "git_status_src_scripts": run(["git", "-C", root, "status", "--porcelain", "--", "src", "scripts"]),
            "nvidia_smi": run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,utilization.gpu,"
                               "clocks.sm,clocks.max.sm,temperature.gpu,power.draw",
                               "--format=csv,noheader"])}


def select_prompts(prompts: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    """The first ``limit`` prompts of every category, in file order (all prompts without ``limit``)."""
    if limit is None:
        return prompts
    taken: dict[str, int] = defaultdict(int)
    selected = []
    for item in prompts:
        if taken[item["category"]] < limit:
            taken[item["category"]] += 1
            selected.append(item)
    return selected


def benchmark(args: argparse.Namespace) -> None:  # noqa: C901
    import torch

    from ccg_diffusion.provenance import repository_root

    device = "cuda:0"
    prompts = select_prompts(json.loads(args.prompts.read_text()), args.limit)
    warmup = json.loads(args.warmup.read_text())
    reference_root = args.reference_root or repository_root()
    if not args.no_verify:
        categories = list(dict.fromkeys(item["category"] for item in prompts))
        if missing := missing_references(args.backbone, reference_root, categories):
            raise SystemExit(f"the recorded runs to verify against are missing under {reference_root} "
                             f"({len(missing)} files, first {missing[0]}); they are not distributed with this "
                             "repository: pass --no-verify to skip the comparison, or --reference-root DIR")
    out = args.out / args.backbone
    out.mkdir(parents=True, exist_ok=args.resume)
    bench_class = SD14Bench if args.backbone == "sd14" else SanaBench
    methods = tuple(args.methods or bench_class.methods)
    done = completed_groups(out, len(methods)) if args.resume else set()
    environment_path = out / "environment.json"
    if environment_path.exists():
        environment_path = out / f"environment_resume_{int(time.time())}.json"
    guard_log = (out / "gpu_guard.jsonl").open("a")

    def log(event: dict[str, Any]) -> None:
        guard_log.write(json.dumps({"time_unix": time.time(), **event}) + "\n")
        guard_log.flush()

    log({"guard": "policy", "policy": args.guard_policy, "quiet_seconds": args.quiet_seconds,
         "resumed_complete_groups": len(done)})
    # Created before any CUDA context of this process so that its own NVML PID is the new one.
    guard = GpuGuard(os.environ["CUDA_VISIBLE_DEVICES"], quiet_s=args.quiet_seconds, policy=args.guard_policy,
                     log=log)
    log({"guard": "own_process", "pid": guard.own})
    started = time.time()
    environment: dict[str, Any] = {"started_at_unix": started, "system_before": environment_record(device),
                                   "resumed_complete_groups": len(done), "methods": list(methods)}
    load_started = time.perf_counter()
    bench = bench_class(device)
    environment["model_load_seconds"] = time.perf_counter() - load_started
    environment["backbone"] = bench.environment()
    environment_path.write_text(json.dumps(environment, indent=2, default=str))
    references = None if args.no_verify else References(args.backbone, reference_root)
    timings = (out / "timings.jsonl").open("a")
    checks = (out / "verification.jsonl").open("a")
    for mode in args.modes:
        seeds = [42] if mode == "b1" else [42, 43, 44, 45]
        if all((mode, item["prompt_id"]) in done for item in prompts):
            continue
        for phase_name, population in (("warmup", warmup), ("measure", prompts)):
            for position, item in enumerate(population):
                if phase_name == "measure" and (mode, item["prompt_id"]) in done:
                    continue
                order = [methods[(position + k) % len(methods)] for k in range(len(methods))]
                for attempt in range(1, args.max_attempts + 1):
                    waited = guard.wait_clear()
                    loadavg_before = os.getloadavg()
                    guard.start()
                    group = []
                    for slot, method in enumerate(order):
                        torch.cuda.reset_peak_memory_stats()
                        clock = Clock()
                        images, info = getattr(bench, method)(item, seeds, clock)
                        total = clock.total()
                        peak = torch.cuda.max_memory_allocated() / 2 ** 30
                        if len(images) != len(seeds):
                            raise ValueError(f"{method} returned {len(images)} images for {len(seeds)} seeds")
                        group.append((slot, method, images, info, total, dict(clock.phases), peak))
                    intruders = guard.stop()
                    if not intruders:
                        break
                    log({"guard": "discarded_group", "mode": mode, "phase": phase_name,
                         "prompt_id": item["prompt_id"], "attempt": attempt, "pids": sorted(intruders)})
                else:
                    raise RuntimeError(f"GPU stayed shared for {args.max_attempts} attempts: {item['prompt_id']}")
                if phase_name == "warmup":
                    continue
                for slot, method, images, info, total, phases, peak in group:
                    record = {
                        "backbone": args.backbone, "mode": mode, "method": method, "prompt_id": item["prompt_id"],
                        "category": item["category"], "seeds": seeds, "images": len(images),
                        "order_slot": slot, "total_s": total, "phases_s": phases,
                        "peak_allocated_gib": peak, "time_unix": time.time(), "attempt": attempt,
                        "guard_wait_s": waited, "loadavg_before": loadavg_before}
                    for key in ("route", "prefix_active", "count_constraints", "status", "guided"):
                        if key in info:
                            record[key] = info[key]
                    timings.write(json.dumps(record) + "\n")
                    timings.flush()
                    verification: dict[str, Any] = {"mode": mode, "method": method, "prompt_id": item["prompt_id"],
                                                    "category": item["category"]}
                    if references is not None:
                        verification.update(references.check(method, item, info))
                    verification["images"] = [
                        compare_image(image, references.image_path(method, item, seed) if references else None)
                        for image, seed in zip(images, seeds)]
                    checks.write(json.dumps(verification) + "\n")
                    checks.flush()
                    if not args.no_save_images:
                        directory = out / "images" / mode / method
                        directory.mkdir(parents=True, exist_ok=True)
                        for image, seed in zip(images, seeds):
                            image.save(directory / f"{item['prompt_id']}_s{seed}.png")
                print(json.dumps({"mode": mode, "phase": phase_name, "position": position,
                                  "prompt_id": item["prompt_id"], "elapsed_min": (time.time() - started) / 60}),
                      flush=True)
    timings.close()
    checks.close()
    guard_log.close()
    environment["system_after"] = environment_record(device)
    environment["finished_at_unix"] = time.time()
    environment_path.write_text(json.dumps(environment, indent=2, default=str))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    bench = sub.add_parser("benchmark", help="time the three methods on one backbone (one GPU, one process)")
    bench.add_argument("--backbone", choices=("sd14", "sana"), required=True)
    bench.add_argument("--out", type=Path, required=True, help="output directory (DIR/<backbone>/ is written)")
    bench.add_argument("--prompts", type=Path, default=Path(DEFAULT_PROMPTS),
                       help="JSON list of {category, index, prompt_id, prompt} (default: the benchmark prompts)")
    bench.add_argument("--warmup", type=Path, default=Path(DEFAULT_WARMUP), help="warm-up prompts (not recorded)")
    bench.add_argument("--modes", nargs="+", default=["b1", "b4"], choices=("b1", "b4"),
                       help="b1: one image (seed 42) per prompt; b4: four images (seeds 42-45)")
    bench.add_argument("--methods", nargs="+", choices=("vanilla", "spacy_pd", "ours"),
                       help="default: all three, rotated per prompt")
    bench.add_argument("--limit", type=int, help="only the first N prompts of each category")
    bench.add_argument("--no-save-images", action="store_true")
    bench.add_argument("--no-verify", action="store_true",
                       help="skip the comparison with the evaluated runs' Stage 1-3 outputs, plans and images")
    bench.add_argument("--reference-root", type=Path,
                       help="checkout whose output/generation holds the evaluated runs (default: this checkout)")
    bench.add_argument("--quiet-seconds", type=float, default=30.0,
                       help="required interval without other processes on the GPU before each prompt group")
    bench.add_argument("--max-attempts", type=int, default=200)
    bench.add_argument("--guard-policy", choices=("presence", "activity"), default="activity")
    bench.add_argument("--resume", action="store_true", help="skip prompt groups already recorded in --out")
    summary = sub.add_parser("summarize", help="write DIR/summary.json from DIR/<backbone>/timings.jsonl")
    summary.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    if args.command == "benchmark":
        benchmark(args)
        return 0
    result = summarize(args.directory)
    (args.directory / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
