"""Sana TAB comparison methods: Vanilla, VS-TIED + Gold-visual setup and Predicated Diffusion (+Gold).

Both stock arms use the Sana TAB run layout and generation loop of
``pp_attachment.sana`` (``write_run``, ``generate_run``): one plan per prompt
key (``vanilla_<tab_id>_None``, ``gold_<tab_id>_<setup>``), seeds 0-4 as one
batch of 4 and one of 1, no guidance.

Predicated Diffusion and Gold-visual setup + Predicated Diffusion share the
run layout of the SD1.4 runs (``baselines.tab_sd14``) and their spaCy guidance
targets: ``compile_pd_plans`` projects each prompt's targets from SD1.4 CLIP
indices onto Sana's Gemma token axis (the ``sana_pd_spacy`` adaptation of
``baselines.sana_plans``), and ``generate_pd_run`` runs them with the Sana
Predicated Diffusion update of ``baselines.sana_generation`` (scale 20, 5 extra
updates at step 0, per-image cap 0.02), per prompt key as seeds 0-3 and seed 4.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from ccg_diffusion.backbones.sana.pipeline import COMPLEX_HUMAN_INSTRUCTION
from ccg_diffusion.baselines.gold_visual_setup import PD_ARMS, pd_baseline_batches, pd_baseline_jobs, stock_jobs
from ccg_diffusion.baselines.sana_plans import compile_plan
from ccg_diffusion.baselines.tab_sd14 import (
    append_pd_record,
    load_pd_run,
    pd_records,
    spacy_target_rows,
    write_generation_summary,
    write_pd_run,
)
from ccg_diffusion.config import SANA_GENERATION
from ccg_diffusion.evaluation.tab import ITEMS, load_pp_items
from ccg_diffusion.pp_attachment.sana import write_run
from ccg_diffusion.provenance import canonical_json, repo_relative, repository_root, sha256_file, sha256_text

STOCK_ARMS = ("vanilla", "gold")
# Generation settings of the Sana Predicated Diffusion runs (recorded in the run manifest).
PD_GENERATION = {**asdict(SANA_GENERATION), "method": "sana_pd_spacy", "pd_scale": 20.0, "pd_cap": 0.02,
                 "pd_step0_extra_refinements": 5, "batch_policy": "per prompt: seeds 0-3, then seed 4",
                 "instruction": list(COMPLEX_HUMAN_INSTRUCTION), "negative_prompt": ""}


def prepare_stock_run(run_id: str, *, arms: Sequence[str], dataset_verification: Mapping[str, Any],
                      allow_dirty: bool = False) -> Path:
    """Prepare a Sana TAB run of the stock arms ``vanilla`` and/or ``gold`` over the 74 PP items."""
    if not arms or set(arms) - set(STOCK_ARMS):
        raise ValueError(f"stock arms must be among {STOCK_ARMS}")
    plans, jobs = stock_jobs(list(load_pp_items().values()), arms)
    inputs = {"kind": "dataset", "path": repo_relative(ITEMS.parent),
              "files": {repo_relative(ITEMS): sha256_file(ITEMS)}}
    return write_run(run_id, plans=plans, jobs=jobs, arms=tuple(arms), inputs_record=inputs,
                     dataset_verification=dataset_verification, allow_dirty=allow_dirty)


# --- Predicated Diffusion and Gold-visual setup + Predicated Diffusion -----------------------------------


def compile_pd_plans(jobs: Sequence[Mapping[str, Any]], spacy_rows: Mapping[str, Mapping[str, Any]], *,
                     tokenizer: Any, clip_tokenizer: Any) -> dict[str, dict[str, Any]]:
    """One ``sana_pd_spacy`` plan per prompt key; every projection must keep guidance targets."""
    prompts = {j["key"]: j["prompt"] for j in jobs}
    plans = {}
    for key, row in spacy_rows.items():
        plan = compile_plan("sana_pd_spacy", prompts[key], row, tokenizer=tokenizer, clip_tokenizer=clip_tokenizer,
                            instruction=COMPLEX_HUMAN_INSTRUCTION)
        if plan["status"] != "ok":
            raise ValueError(f"Sana PD projection fell back for {key}: {plan['fallback_reason']}")
        plans[key] = {**plan, "key": key}
    return plans


def prepare_pd_run(run_id: str, *, dataset_verification: Mapping[str, Any], allow_dirty: bool = False) -> Path:
    """Prepare the Sana Predicated Diffusion run of both arms (370 + 740 jobs, 222 plans)."""
    from ccg_diffusion.backbones.sana.pipeline import load_sana_tokenizer
    from ccg_diffusion.targets.tokenizer import load_sd14_tokenizer

    jobs = pd_baseline_jobs(list(load_pp_items().values()))
    targets = spacy_target_rows(jobs)
    # CLIPTokenizerFast, as for the Sana five-baseline plans (sana_plans.load_tokenizers).
    plans = compile_pd_plans(jobs, targets, tokenizer=load_sana_tokenizer(),
                             clip_tokenizer=load_sd14_tokenizer(fast=True))
    return write_pd_run(run_id, backbone="sana", jobs=jobs, targets=targets, plans=plans, generation=PD_GENERATION,
                        dataset_verification=dataset_verification, allow_dirty=allow_dirty)


def generate_pd_run(run_dir: Path, *, device: str = "cuda:0", arms: Sequence[str] = PD_ARMS,
                    keys: Sequence[str] | None = None) -> None:
    """Generate the pending images of ``arms`` (or of the prompt ``keys``) of a prepared Sana run.

    Each key keeps its batches (seeds 0-3, then seed 4); a resumed batch must
    reproduce its completed images.
    """
    from ccg_diffusion.backbones.sana.pipeline import encode_texts, load_sana_pipeline
    from ccg_diffusion.baselines.sana_generation import SanaBaselineRuntime

    manifest, jobs, _, plans = load_pd_run(run_dir)
    selected = [b for b in pd_baseline_batches(jobs, "sana") if b[0]["arm"] in arms]
    if keys is not None:
        unknown = set(keys) - {b[0]["key"] for b in selected}
        if unknown:
            raise ValueError(f"plan keys outside the arms {list(arms)}: {sorted(unknown)}")
        selected = [b for b in selected if b[0]["key"] in keys]
    done = pd_records(run_dir, jobs, complete=False)
    pending = [b for b in selected if not all(j["image_id"] in done for j in b)]
    if not pending:
        return
    c = manifest["generation"]
    pipe = load_sana_pipeline(device)
    texts = sorted({plans[b[0]["key"]]["conditioning"] for b in pending})
    encodings = encode_texts(pipe, {text: text for text in texts}, device=device, instruction=c["instruction"],
                             negative_prompt=c["negative_prompt"])
    root = repository_root()
    started = time.monotonic()
    for batch_jobs in pending:
        plan = plans[batch_jobs[0]["key"]]
        tick = time.monotonic()
        images, diagnostics = SanaBaselineRuntime(pipe, plan, encodings, device=device).generate(
            [j["seed"] for j in batch_jobs], selected=[])
        seconds = (time.monotonic() - tick) / len(batch_jobs)
        for job, image, diagnostic in zip(batch_jobs, images, diagnostics, strict=True):
            path = run_dir / "images" / job["arm"] / (job["image_id"] + ".png")
            temporary = path.with_suffix(".tmp.png")
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(temporary)
            if job["image_id"] in done:
                if sha256_file(temporary) != done[job["image_id"]]["image_sha256"]:
                    raise ValueError("resumed batch differs from its completed peer")
                temporary.unlink()
                continue
            temporary.replace(path)
            record = {**job, "image_path": repo_relative(path, repository_root=root), "image_sha256": sha256_file(path),
                      "conditioning": plan["conditioning"], "plan_sha256": sha256_text(canonical_json(plan)),
                      "pd_indices": plan["pd_indices"], "diagnostics": diagnostic,
                      "physical_batch_size": len(batch_jobs), "source_commit": manifest["source"]["commit"],
                      "seconds": seconds}
            append_pd_record(run_dir, record)
            done[job["image_id"]] = record
        print(json.dumps({"key": batch_jobs[0]["key"], "completed": len(done), "total": len(jobs),
                          "seconds": time.monotonic() - started}), flush=True)
    if len(done) == len(jobs):
        write_generation_summary(run_dir, jobs, started)
