"""CCG Diffusion for TAB PP on Sana 0.6B: Gemma projection, objective, run layout and generation loop.

A compiled TAB job (CLIP token indices, ``compile.compile_interpretation``) is
projected onto Sana's Gemma token axis: every CLIP word anchor maps to the
Gemma pieces that cover the same characters, guidance targets and relations
move to the last piece of their word, and the region groups keep whole words.
``SanaPPAttachmentLoss`` is the per-image objective (``L_PD + 6 L_spatial +
0.25 L_region``, Gemma layout, whole-word pooling on boosted rows) with the
per-interpretation strength (scale 10 or 80 as a multiplier of the loop's
base scale 20, cap 0.02 or 0.06).

One run layout and one generation loop serve the arms ``guided`` (CCG
Diffusion), ``vanilla`` and ``gold`` (VS-TIED + Gold-visual setup, built in
``baselines.tab_sana``): ``output/generation/<run>/{plans.jsonl,
metadata_jobs.jsonl, run_manifest.json, text_encoding_manifest.json,
images/<arm>/<image_id>.png, metadata.jsonl}``. Each prompt key is generated
as seeds 0-3 in one batch and seed 4 alone. Sana TAB images are bit-exact only
on the GPU type of the paper runs (RTX 4090).

``diffusers`` is imported only by the generation functions.
"""
from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from ccg_diffusion.backbones.sana.pipeline import COMPLEX_HUMAN_INSTRUCTION
from ccg_diffusion.backbones.sana.tokens import normalise_token, selected_token_offsets
from ccg_diffusion.config import SANA_GENERATION
from ccg_diffusion.evaluation.tab import load_pp_items, pp_profile, write_trial_plan
from ccg_diffusion.guidance.predicated_diffusion_loss import SanaWholeWordSpans
from ccg_diffusion.guidance.region_assignment_loss import RegionGroups
from ccg_diffusion.guidance.selective_guidance_boost import GuidanceRoute
from ccg_diffusion.guidance.spatial_relations import RelationConstraint
from ccg_diffusion.methods import TabAblation
from ccg_diffusion.parsing.ccg import align_whitespace_tokens_to_clip
from ccg_diffusion.pp_attachment.compile import Interpretation, compile_interpretation
from ccg_diffusion.pp_attachment.jobs import (
    BUNDLE_DIR,
    MANIFEST_FILE,
    READINGS,
    load_clean_interpretations,
    load_frozen_bundle,
    run_directory,
    verify_hashes,
)
from ccg_diffusion.pp_attachment.objective import tab_objective
from ccg_diffusion.provenance import (
    atomic_write_json,
    atomic_write_jsonl,
    read_jsonl,
    repo_relative,
    repository_root,
    sha256_file,
    source_tree_snapshot,
)
from ccg_diffusion.targets.pd_indices import PDIndices

PLAN_SCHEMA = "holsd.sana_tab_pp.v1"
RUN_SCHEMA = "ccg_diffusion.sana_tab_run.v1"
BATCH_POLICY = "same conditioning, seeds 0-3 then seed4; no historical SD batch reuse"
ARMS = ("vanilla", "gold", "guided")
TRIAL_PROFILE = "sana-tab-pp"


def project_interpretation(
    row: Mapping[str, Any], clip_tokenizer: Any, gemma_tokenizer: Any, instruction: Sequence[str],
) -> dict[str, Any]:
    """The Sana plan of one compiled TAB job (no setup or question text is read)."""
    prompt = row["prompt"]
    if clip_tokenizer(prompt, truncation=False)["input_ids"] != row["token_ids"]:
        raise ValueError("TAB conditioning token IDs changed")
    if row.get("phrase_conditioning") is not None:
        raise ValueError("unsupported TAB plan: phrase_conditioning is set")
    decoded, offsets = selected_token_offsets(gemma_tokenizer, prompt, instruction=instruction,
                                              max_sequence_length=300)
    spans, owners = {}, {}
    anchors = align_whitespace_tokens_to_clip(prompt, clip_tokenizer)
    for anchor in anchors:
        if not anchor.lexical_piece_indices:
            continue
        pieces = tuple(i for i, (a, b) in sorted(offsets.items())
                       if min(b, anchor.char_end) > max(a, anchor.char_start) and normalise_token(decoded[i]))
        if not pieces or normalise_token("".join(decoded[i] for i in pieces)) != normalise_token(anchor.surface):
            raise ValueError(f"Gemma lexical projection failed: {anchor.surface}")
        spans[anchor.anchor_index] = pieces
        owners.update({i: anchor.anchor_index for i in anchor.lexical_piece_indices})

    def endpoint(i: int) -> int:
        return spans[owners[i]][-1]

    original = row["pd_indices"]
    indices = {"exist": [endpoint(i) for i in original["exist"]],
               **{field: [[endpoint(a), endpoint(b)] for a, b in original[field]]
                  for field in ("corr", "leak", "possession")}}
    count = max(offsets) + 1
    pair_spans = SanaWholeWordSpans.from_mapping(
        {endpoint(i): spans[owners[i]] for pair in (*original["corr"], *original["leak"]) for i in pair},
        text_token_count=count)
    groups = []
    if row.get("transfer_region"):
        groups = [[list(spans[owners[word[-1]]]) for word in group]
                  for group in row["transfer_region"]["plan"]["groups"]]
    return {"schema": PLAN_SCHEMA, "conditioning": prompt,
            "pd_indices": indices,
            "source_program": original, "text_token_count": count,
            "lexical_projection": [{"source_clip_anchor": i, "target_span": list(s)} for i, s in spans.items()],
            "token_spans": pair_spans.as_record(), "loss_config": dict(row["loss_config"]),
            "relations": [{**r, "subject_clip_index": endpoint(r["subject_clip_index"]),
                           "object_clip_index": endpoint(r["object_clip_index"])} for r in row["relations"]],
            "region_groups": groups, "phrase": None, "transfer_trace": row.get("transfer_trace")}


class SanaPPAttachmentLoss:
    """Per-image TAB objective on Sana (``backbones.sana.guidance.SanaGuidanceObjective``)."""

    def __init__(self, plan: Mapping[str, Any], batch_size: int = 1) -> None:
        if plan["schema"] != PLAN_SCHEMA:
            raise ValueError("unsupported TAB Sana plan")
        self.plan = plan
        self.batch_size = batch_size
        self.indices = PDIndices(**plan["pd_indices"])
        self.route = GuidanceRoute.BOOSTED if plan["loss_config"]["scale_factor"] == 80 else GuidanceRoute.DEFAULT
        self.spans = SanaWholeWordSpans.from_mapping(
            {r["anchor"]: tuple(r["lexical_piece_indices"]) for r in plan["token_spans"]["spans"]},
            text_token_count=plan["text_token_count"]) if self.route == GuidanceRoute.BOOSTED else None
        self.relations = tuple(RelationConstraint(**r) for r in plan["relations"])
        self.region = RegionGroups(bool(plan["region_groups"]), "left",
                                   tuple(tuple(tuple(w) for w in g) for g in plan["region_groups"]))
        self.trace: list[dict[str, Any]] = []
        self.update_diagnostics: dict[str, Any] = {}

    @property
    def scale_multiplier(self) -> float:
        return self.plan["loss_config"]["scale_factor"] / 20.

    @property
    def max_update_ratio(self) -> float:
        return self.plan["loss_config"]["max_update_ratio"]

    def __call__(self, attention: torch.Tensor, step_index: int) -> torch.Tensor:
        if attention.shape[-1] != self.plan["text_token_count"]:
            raise ValueError("runtime Gemma token count differs")
        work = attention.float()
        if work.shape[:2] != (16, 16):
            raise ValueError("Sana TAB attention maps must be 16x16")
        total, spatial, region = tab_objective(
            work, self.indices, self.relations, self.region if self.region.active else None,
            attention_res=16, special_layout="leading_bos", pair_token_spans=self.spans)
        self.trace.append({"step": step_index, "spatial": float(spatial.detach()),
                           "region": float(region.detach()), "total": float(total.detach())})
        return total

    def audit_record(self) -> dict[str, Any]:
        self.update_diagnostics.update(
            strict_finite_applied=True, effective_batch_size=self.batch_size, valid_batch_size=self.batch_size,
            generation_kernel="sana_tab_fp32_master", pd_loss_evaluations=len(self.trace),
            captured_attention_processors=list(range(8)), captured_attention_processor_count=8,
        )
        return {"route": self.route, "scale_multiplier": self.scale_multiplier,
                "max_update_ratio": self.max_update_ratio, "loss_trace": self.trace,
                "update_diagnostics": self.update_diagnostics}


def guided_plans_and_jobs(
    *, specs: Sequence[Mapping[str, Any]], interpretations: Mapping[tuple[str, str], Interpretation],
    ablation: TabAblation, clip_tokenizer: Any, gemma_tokenizer: Any,
    instruction: Sequence[str] = COMPLEX_HUMAN_INSTRUCTION,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Plans (one per interpretation) and jobs (seeds 0-4) of the guided arm, items in dataset order."""
    plans, jobs = [], []
    for spec in specs:
        tab_id = spec["tab_id"]
        for reading, opposite in zip(READINGS, READINGS[::-1]):
            key = f"guided_{tab_id}_{reading}"
            setup = interpretations[tab_id, reading].source_job["target_setup_index"]
            transferred = compile_interpretation(interpretations[tab_id, reading], interpretations[tab_id, opposite],
                                                 clip_tokenizer, ablation)
            plan = project_interpretation(transferred, clip_tokenizer, gemma_tokenizer, instruction)
            plan.update(key=key, arm="guided")
            plans.append(json.loads(json.dumps(plan)))
            for seed in range(5):
                jobs.append({"key": key, "arm": "guided", "condition": "guided", "tab_id": tab_id,
                             "tab_category": spec["tab_category"], "seed": seed, "reading": reading,
                             "tab_requires_reading": True, "target_setup_index": setup,
                             "tab_evaluation_mode": "targeted_setup", "evaluation_setup_indices": [setup],
                             "image_id": f"{key}_s{seed}"})
    return plans, jobs


def census(plans: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Counts of the guided interpretations per guidance component."""
    return {"p2_readings": sum(p["loss_config"]["scale_factor"] == 80 for p in plans),
            "spatial_readings": sum(bool(p["relations"]) for p in plans),
            "region_readings": sum(bool(p["region_groups"]) for p in plans),
            "phrase_readings": sum(p["phrase"] is not None for p in plans),
            "possession_readings": sum(bool(p["pd_indices"]["possession"]) for p in plans)}


# --- run layout ----------------------------------------------------------------------------------------


def write_run(
    run_id: str, *, plans: Sequence[Mapping[str, Any]], jobs: Sequence[Mapping[str, Any]], arms: Sequence[str],
    inputs_record: Mapping[str, Any], dataset_verification: Mapping[str, Any], allow_dirty: bool = False,
    ablation: TabAblation | None = None,
) -> Path:
    """Write the plans, jobs and run manifest of a Sana TAB run (any of the arms guided/vanilla/gold).

    The manifest seals the source tree (``provenance.source_tree_snapshot``);
    a dirty tree needs ``allow_dirty``.
    """
    if set(arms) - set(ARMS) or {j["arm"] for j in jobs} != set(arms):
        raise ValueError(f"Sana TAB arms must be among {ARMS} and match the jobs")
    source = source_tree_snapshot()
    if source["dirty"] and not allow_dirty:
        raise ValueError("commit the implementation before preparing a Sana TAB run (or pass --allow-dirty)")
    output = run_directory(run_id)
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(output / "plans.jsonl", plans, overwrite=False)
    atomic_write_jsonl(output / "metadata_jobs.jsonl", jobs, overwrite=False)
    guided = [p for p in plans if p["arm"] == "guided"]
    manifest = {"schema": RUN_SCHEMA, "run_id": run_id, "arms": list(arms),
                "generation_suffix": "", "negative_prompt": "", "source": source,
                "generation": asdict(SANA_GENERATION), "instruction": list(COMPLEX_HUMAN_INSTRUCTION),
                "dataset": dict(dataset_verification), "inputs": dict(inputs_record),
                "plans_sha256": sha256_file(output / "plans.jsonl"),
                "jobs_sha256": sha256_file(output / "metadata_jobs.jsonl"),
                "counts": dict(Counter(j["arm"] for j in jobs)), "census": census(guided),
                "batch_policy": BATCH_POLICY}
    if ablation is not None:
        manifest.update(recipe=f"tab_pp_ablation:{ablation.cell_id}",
                        ablation={"cell": ablation.cell_id, "switches": ablation.switches})
    atomic_write_json(output / "run_manifest.json", manifest, overwrite=False)
    return output


def inputs_record(inputs: str | Path) -> dict[str, Any]:
    """Kind, path and sha256 of the TAB inputs of a run (frozen bundle manifest or clean-path file)."""
    if inputs == "frozen":
        path = BUNDLE_DIR / MANIFEST_FILE
        return {"kind": "frozen", "path": repo_relative(BUNDLE_DIR), "files": {repo_relative(path): sha256_file(path)}}
    path = Path(inputs) / "interpretations.jsonl"
    return {"kind": "clean", "path": repo_relative(Path(inputs)), "files": {repo_relative(path): sha256_file(path)}}


def prepare_run(
    *, run_id: str, ablation: TabAblation = TabAblation(), inputs: str | Path = "frozen",
    dataset_verification: Mapping[str, Any], allow_dirty: bool = False,
) -> Path:
    """Prepare the guided arm of one TAB PP ablation cell (``c1r1p0m0``, ...) on Sana (148 plans, 740 jobs)."""
    from ccg_diffusion.backbones.sana.pipeline import load_sana_tokenizer
    from ccg_diffusion.targets.tokenizer import load_sd14_tokenizer

    if inputs == "frozen":
        interpretations = load_frozen_bundle().interpretations
    else:
        interpretations = load_clean_interpretations(Path(inputs))
    specs = list(load_pp_items().values())
    plans, jobs = guided_plans_and_jobs(specs=specs, interpretations=interpretations, ablation=ablation,
                                        clip_tokenizer=load_sd14_tokenizer(), gemma_tokenizer=load_sana_tokenizer())
    return write_run(run_id, plans=plans, jobs=jobs, arms=("guided",), inputs_record=inputs_record(inputs),
                     dataset_verification=dataset_verification, allow_dirty=allow_dirty, ablation=ablation)


def load_run(run_id: str) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """``(manifest, jobs, plans by key)`` of a prepared run; the source tree and inputs must be unchanged."""
    output = run_directory(run_id)
    manifest = json.loads((output / "run_manifest.json").read_text())
    current = source_tree_snapshot()
    if current["files"] != manifest["source"]["files"] or current["dirty"] != manifest["source"]["dirty"]:
        raise ValueError("implementation differs from prepared source")
    verify_hashes(manifest["inputs"]["files"])
    for name, field in (("plans.jsonl", "plans_sha256"), ("metadata_jobs.jsonl", "jobs_sha256")):
        if sha256_file(output / name) != manifest[field]:
            raise ValueError(f"prepared {name} changed")
    plans = {p["key"]: p for p in read_jsonl(output / "plans.jsonl")}
    return manifest, read_jsonl(output / "metadata_jobs.jsonl"), plans


def generation_records(output: Path, jobs: Sequence[Mapping[str, Any]], *, complete: bool) -> dict[str, Any]:
    """The recorded images of a run by image ID, checked against the jobs and the image files."""
    path = output / "metadata.jsonl"
    records = read_jsonl(path) if path.exists() else []
    result = {r["image_id"]: r for r in records}
    planned = {j["image_id"]: j for j in jobs}
    if len(result) != len(records) or set(result) - set(planned) or (complete and set(result) != set(planned)):
        raise ValueError("generation population is incomplete, duplicated or unplanned")
    root = repository_root()
    for identity, record in result.items():
        if any(record[k] != v for k, v in planned[identity].items()):
            raise ValueError("generation metadata differs from prepared job")
        if sha256_file(root / record["image_path"]) != record["image_sha256"]:
            raise ValueError("generated image changed")
    return result


def generate_run(run_id: str, *, device: str = "cuda:0", arms: Sequence[str] = ("guided",),
                 keys: Sequence[str] | None = None) -> None:
    """Generate the pending images of ``arms`` (or of the prompt ``keys``); each key keeps its 0-3/4 batches."""
    from ccg_diffusion.backbones.sana.guidance import generate_guided
    from ccg_diffusion.backbones.sana.pipeline import (
        encode_texts,
        generate_stock_from_encoding,
        load_sana_pipeline,
        tensor_sha256,
    )

    output = run_directory(run_id)
    manifest, jobs, plans = load_run(run_id)
    if keys is not None and set(keys) - set(plans):
        raise ValueError(f"unknown plan keys: {sorted(set(keys) - set(plans))}")
    completed = generation_records(output, jobs, complete=False)
    pending = [j for j in jobs if j["image_id"] not in completed and j["arm"] in arms
               and (keys is None or j["key"] in keys)]
    if not pending:
        return
    c = manifest["generation"]
    pipe = load_sana_pipeline(device)
    started = time.monotonic()
    cache = encode_texts(pipe, {key: plans[key]["conditioning"] for key in dict.fromkeys(j["key"] for j in pending)},
                         device=device, instruction=manifest["instruction"], negative_prompt="",
                         offload_text_encoder=False)
    encoding_audit = {key: {name: tensor_sha256(value) for name, value in zip(
        ("positive", "positive_mask", "negative", "negative_mask"), encoding)} for key, encoding in cache.items()}
    audit_path = output / "text_encoding_manifest.json"
    prior_audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
    if any(key in prior_audit and prior_audit[key] != value for key, value in encoding_audit.items()):
        raise ValueError("resumed text encoding changed")
    atomic_write_json(audit_path, {**prior_audit, **encoding_audit}, overwrite=True)
    print(json.dumps({"encoded": len(cache), "seconds": time.monotonic() - started}), flush=True)
    root = repository_root()
    for key in dict.fromkeys(j["key"] for j in pending):
        plan = plans[key]
        group = [j for j in jobs if j["key"] == key]
        for offset in range(0, len(group), c["batch_size"]):
            # Retain original batch membership on resume, including committed peers.
            batch = group[offset:offset + c["batch_size"]]
            if all(j["image_id"] in completed for j in batch):
                continue
            generators = [torch.Generator(device=device).manual_seed(j["seed"]) for j in batch]
            torch.cuda.reset_peak_memory_stats()
            tick = time.monotonic()
            if plan["arm"] == "guided":
                losses = [SanaPPAttachmentLoss(plan, len(batch)) for _ in batch]
                images, _ = generate_guided(
                    pipe, plan["conditioning"], losses, generators, prompt_encoding=cache[key],
                    instruction=manifest["instruction"], negative_prompt="", steps=c["steps"],
                    guidance_scale=c["guidance_scale"], height=c["height"], width=c["width"],
                    block_range=tuple(range(8)), scale_factor=20., guided_steps=c["guided_steps"],
                    refinement_steps={0: 0.})
                diagnostics = [loss.audit_record() for loss in losses]
            else:
                images, diagnostics = generate_stock_from_encoding(
                    pipe, cache[key], generators, device=device, steps=c["steps"],
                    guidance_scale=c["guidance_scale"], height=c["height"], width=c["width"])
            elapsed = time.monotonic() - tick
            for j, image, diagnostic in zip(batch, images, diagnostics):
                path = output / "images" / j["arm"] / (j["image_id"] + ".png")
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".tmp.png")
                image.save(temporary)
                image_hash = sha256_file(temporary)
                if j["image_id"] in completed:
                    temporary.unlink()
                    if image_hash != completed[j["image_id"]]["image_sha256"]:
                        raise ValueError("resumed batch differs from committed peer")
                    continue
                if path.exists() and sha256_file(path) != image_hash:
                    temporary.unlink()
                    raise ValueError("uncommitted image differs from regenerated batch")
                temporary.replace(path)
                record = {**j, "image_path": repo_relative(path, repository_root=root), "image_sha256": image_hash,
                          "conditioning": plan["conditioning"], "diagnostics": diagnostic,
                          "negative_prompt": "", "source_commit": manifest["source"]["commit"],
                          "physical_batch_size": len(batch), "seconds": elapsed / len(batch),
                          "peak_memory_bytes": torch.cuda.max_memory_allocated()}
                with (output / "metadata.jsonl").open("a") as stream:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                completed[j["image_id"]] = record
            print(json.dumps({"completed": len(completed), "total": len(jobs), "key": key,
                              "seconds_per_image": elapsed / len(batch)}), flush=True)
    if len(completed) == len(jobs):
        generation_records(output, jobs, complete=True)
        atomic_write_json(output / "generation_summary.json", {"images": len(completed),
                          "metadata_sha256": sha256_file(output / "metadata.jsonl"),
                          "seconds": time.monotonic() - started}, overwrite=False)


def pinned_projection(plan: Mapping[str, Any]) -> dict[str, Any]:
    """The generation-affecting fields of a Sana TAB plan (compared between the clean and the frozen path)."""
    return {k: plan[k] for k in ("conditioning", "lexical_projection", "token_spans", "text_token_count",
                                 "region_groups", "relations", "loss_config", "pd_indices", "source_program")}


def preflight(run_id: str) -> dict[str, Any]:
    """Compare a prepared guided run with the evaluated plans (frozen inputs) or the frozen plans (clean inputs)."""
    from ccg_diffusion.backbones.sana.pipeline import load_sana_tokenizer
    from ccg_diffusion.targets.tokenizer import load_sd14_tokenizer

    manifest, _, plans = load_run(run_id)
    cell = manifest["ablation"]["cell"]
    bundle = load_frozen_bundle()
    reference = bundle.reference["sana"]["cells"][cell]
    if manifest["inputs"]["kind"] == "frozen":
        for field in ("plans_sha256", "jobs_sha256", "census"):
            if manifest[field] != reference[field]:
                raise ValueError(f"Sana {field} differs from the evaluated {cell} run")
        return {"run_id": run_id, "cell": cell, "inputs": "frozen", "plans_sha256": manifest["plans_sha256"],
                "jobs_sha256": manifest["jobs_sha256"], "census": manifest["census"]}
    frozen, _ = guided_plans_and_jobs(
        specs=list(load_pp_items().values()), interpretations=bundle.interpretations,
        ablation=TabAblation.from_cell_id(cell), clip_tokenizer=load_sd14_tokenizer(),
        gemma_tokenizer=load_sana_tokenizer())
    differing = [p["key"] for p in frozen if pinned_projection(plans[p["key"]]) != pinned_projection(p)]
    if differing:
        raise ValueError(f"clean-path Sana plans differ from the frozen plans: {differing[:5]}")
    return {"run_id": run_id, "cell": cell, "inputs": "clean", "pinned_projection_equal": len(frozen)}


def validate_guided_record(record: Mapping[str, Any], plan: Mapping[str, Any]) -> None:
    """Check one guided image: 25 applied updates, strict finite, cap, route, loss schedule and active terms."""
    d = record["diagnostics"]
    u, trace = d["update_diagnostics"], d["loss_trace"]
    if (u["guidance_updates_applied"] != 25 or not u["strict_finite_applied"]
            or any(v for k, v in u.items() if k.startswith("finite_guard_"))):
        raise ValueError("guidance schedule or finite guard differs")
    if u["max_applied_update_ratio"] > plan["loss_config"]["max_update_ratio"] + 1e-7:
        raise ValueError("guidance cap exceeded")
    if d["route"] != (GuidanceRoute.BOOSTED if plan["loss_config"]["scale_factor"] == 80 else GuidanceRoute.DEFAULT):
        raise ValueError("guidance route differs from the plan")
    if [t["step"] for t in trace] != [0] * 6 + list(range(1, 20)):
        raise ValueError("guidance loss schedule differs")
    if any(not math.isfinite(t[k]) for t in trace for k in ("total", "spatial", "region")):
        raise ValueError("nonfinite loss")
    # Active terms carry a positive -log mass (region) or overlap (spatial) part, so all-zero means off.
    if (bool(plan["region_groups"]) != any(t["region"] for t in trace)
            or bool(plan["relations"]) != any(t["spatial"] for t in trace)):
        raise ValueError("loss term does not match the plan")


def audit_run(run_id: str) -> dict[str, Any]:
    """Check every guided image of a complete run and that seeds share initial latents across interpretations."""
    output = run_directory(run_id)
    _, jobs, plans = load_run(run_id)
    records = generation_records(output, jobs, complete=True)
    initial: dict[int, set[str]] = defaultdict(set)
    for record in records.values():
        if record["arm"] != "guided":
            continue
        validate_guided_record(record, plans[record["key"]])
        initial[record["seed"]].add(record["diagnostics"]["update_diagnostics"]["initial_latent_sha256"])
    if any(len(v) != 1 for v in initial.values()):
        raise ValueError("initial latents are not shared by the images of one seed")
    return {"run_id": run_id, "images": len(records),
            "routes": dict(Counter(r["diagnostics"]["route"] for r in records.values() if r["arm"] == "guided")),
            "initial_latent_sha256_by_seed": {str(k): sorted(v)[0] for k, v in sorted(initial.items())}}


def merge_run(run_id: str) -> Path:
    """Write the trial plan of a complete run (``output/vqa/<run>/trial_plan.jsonl``) for the judges."""
    output = run_directory(run_id)
    _, jobs, _ = load_run(run_id)
    records = list(generation_records(output, jobs, complete=True).values())
    path = repository_root() / "output/vqa" / run_id / "trial_plan.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_trial_plan(path, records, profile=pp_profile(TRIAL_PROFILE), with_source_metadata=True)
    return path
