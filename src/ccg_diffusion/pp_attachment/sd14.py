"""CCG Diffusion for TAB PP on Stable Diffusion v1.4: prepare, generate and merge one Table 6 cell.

``prepare_run`` compiles the jobs of one cell into the 3,150-job layout (or the
740 PP jobs of the clean path) and seals the plan. ``generate`` runs the
original physical batches of 4 that hold PP jobs (0-184) through the guided
denoising kernel: the TAB guidance strength per batch (kernel scale 20 with
route multipliers 0.5/4.0 when a batch has a boosted row, scale 10 without
routes otherwise) and the TAB objective as the per-image loss override of rows
with a region layout or relations. SD1.4 TAB images are bit-exact only on the
GPU type of the paper runs (RTX 4090) under the determinism settings recorded
in the plan. ``merge`` collects the receipts into ``metadata.jsonl`` and writes
the trial plan the judges read.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from ccg_diffusion import config
from ccg_diffusion.backbones.jobs import Job
from ccg_diffusion.evaluation.tab import pp_profile, write_trial_plan
from ccg_diffusion.methods import TAB_SD14, TabAblation
from ccg_diffusion.pp_attachment.jobs import (
    BUNDLE_DIR,
    INTERPRETATIONS_FILE,
    MANIFEST_FILE,
    PLAN_SCHEMA,
    PP_JOBS,
    REFERENCE_FILE,
    SD14_SOURCE_FILES,
    SOURCE_JOBS_FILE,
    TAB_SD14_GENERATION_RECORD,
    check_population,
    clean_layout,
    load_clean_interpretations,
    load_frozen_bundle,
    load_run,
    place_jobs,
    receipt_path,
    run_directory,
    selected_batches,
    validate_diagnostics,
    verify_hashes,
    verify_receipt,
    write_plan,
)
from ccg_diffusion.pp_attachment.objective import TabObjective, guidance_strength
from ccg_diffusion.provenance import (
    atomic_write_json,
    git_head,
    immutable_json,
    immutable_rows,
    repo_relative,
    repository_root,
    require_committed,
    sha256_file,
)
from ccg_diffusion.targets.pd_indices import PD_FIELDS
from ccg_diffusion.targets.tokenizer import load_sd14_tokenizer, require_sd14_tokenizer

EXPERIMENT = "tab_pp_t2i_transfer"  # recorded Job experiment of the SD1.4 TAB images
TRIAL_PROFILE = "tab-pp-transfer-74-five-seed"


def _input_files(inputs: str | Path) -> dict[str, str]:
    """Repository-relative input files of a run and their sha256 (re-verified by ``load_run``)."""
    if inputs == "frozen":
        names = (MANIFEST_FILE, SOURCE_JOBS_FILE, INTERPRETATIONS_FILE, REFERENCE_FILE)
        return {repo_relative(BUNDLE_DIR / name): sha256_file(BUNDLE_DIR / name) for name in names}
    directory = Path(inputs)
    return {repo_relative(path): sha256_file(path)
            for path in (directory / "interpretations.jsonl", directory / "parse_manifest.json")}


def prepare_run(run_id: str, *, ablation: TabAblation = TabAblation(), inputs: str | Path = "frozen",
                allow_dirty: bool = False) -> Path:
    """Compile and seal the jobs of one Table 6 cell; returns the run directory.

    ``inputs`` is ``"frozen"`` (the bundle ``data/tab_pp``) or the output
    directory of ``scripts/tab_pp/parse_interpretations.py``. Without
    ``allow_dirty`` every sealed source file must be committed at ``HEAD``.
    """
    run_dir = run_directory(run_id)
    tokenizer = load_sd14_tokenizer()
    require_sd14_tokenizer(tokenizer)
    if inputs == "frozen":
        bundle = load_frozen_bundle()
        interpretations, source_jobs = bundle.interpretations, list(bundle.source_jobs)
    else:
        interpretations = load_clean_interpretations(Path(inputs))
        source_jobs = clean_layout(interpretations)
    rows = place_jobs(interpretations=interpretations, source_jobs=source_jobs, ablation=ablation, run_id=run_id,
                      tokenizer=tokenizer)
    census = check_population(rows, ablation=ablation, interpretations=interpretations, source_jobs=source_jobs)
    if not allow_dirty:
        require_committed(SD14_SOURCE_FILES)
    root = repository_root()
    plan: dict[str, Any] = {
        "schema": PLAN_SCHEMA, "run_id": run_id, "configuration": TAB_SD14, "cell": ablation.cell_id,
        "ablation": {"disambiguating_paraphrase": ablation.disambiguating_paraphrase,
                     "holder_object_region_assignment": ablation.holder_object_region_assignment},
        "inputs": {"kind": "frozen" if inputs == "frozen" else "clean",
                   "path": repo_relative(BUNDLE_DIR if inputs == "frozen" else Path(inputs)),
                   "files": _input_files(inputs)},
        "generation": dict(TAB_SD14_GENERATION_RECORD), "determinism": dict(config.SD14_DETERMINISM),
        "git_commit": git_head(root), "dirty": allow_dirty,
        "source_sha256": {name: sha256_file(root / name) for name in SD14_SOURCE_FILES},
        "population_jobs": len(rows), "selected_physical_batches": selected_batches(rows), "census": census,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    immutable_rows(run_dir / "metadata_jobs.jsonl", rows)
    plan["jobs_sha256"] = sha256_file(run_dir / "metadata_jobs.jsonl")
    write_plan(run_dir, plan)
    return run_dir


def generate(run_id: str, *, device: str = "cuda:0", batches: Sequence[int] | None = None,
             output_dir: Path | None = None) -> None:
    """Generate the selected physical batches of a prepared run (all of them, or ``batches``).

    ``output_dir`` writes images and receipts of a verification subset outside
    the run directory.
    """
    import diffusers
    import torch
    import transformers

    from ccg_diffusion.backbones.sd14.driver import initial_latents, prepare_sd14_runtime
    from ccg_diffusion.backbones.sd14.pipeline import generate_predicated_batch
    from ccg_diffusion.backbones.sd14.records import save_image_atomic
    from ccg_diffusion.guidance.latent_update import GenerationDiagnostics

    run_dir = run_directory(run_id)
    plan, rows, plan_hash = load_run(run_dir)
    out = run_dir if output_dir is None else output_dir
    selected = plan["selected_physical_batches"]
    if batches is not None:
        if not set(batches) <= set(selected):
            raise ValueError(f"batches must be a subset of the planned physical batches 0-{selected[-1]}")
        selected = [b for b in selected if b in set(batches)]
    torch.set_num_threads(4)
    runtime = prepare_sd14_runtime(device)
    if not runtime.determinism == plan["determinism"] == dict(config.SD14_DETERMINISM):
        raise ValueError("SD1.4 determinism differs from the TAB plan")
    c = plan["generation"]
    versions = {"torch": torch.__version__, "diffusers": diffusers.__version__,
                "transformers": transformers.__version__}
    if any(c[k] != v for k, v in versions.items()):
        raise ValueError(f"software versions differ from the TAB plan: {versions}")
    pipe = runtime.pipe
    if runtime.scheduler["config_sha256"] != c["scheduler_config_sha256"]:
        raise ValueError("SD1.4 scheduler differs from the TAB plan")
    immutable_json(out / "runtime.json", {
        "plan_sha256": plan_hash, "versions": versions, "gpu": torch.cuda.get_device_name(device),
        "gpu_uuid": str(torch.cuda.get_device_properties(device).uuid), "scheduler": runtime.scheduler,
        "determinism": runtime.determinism,
    })
    cell = plan["cell"]
    root = repository_root()
    completed = 0
    started = time.monotonic()
    for batch_id in selected:
        real = rows[4 * batch_id:4 * batch_id + 4]
        receipt = receipt_path(out, cell, batch_id)
        if receipt.exists():
            verify_receipt(receipt, real, plan_hash)
            completed += 1
            continue
        verify_hashes(plan["source_sha256"])
        batch = real + [real[-1]] * (4 - len(real))
        jobs = [Job(experiment=EXPERIMENT, prompt_id=r["image_id"], prompt=r["prompt"], seed=r["seed"],
                    source=run_id, **{"attention_" + f + "_indices": r["pd_indices"][f] for f in PD_FIELDS})
                for r in batch]
        latent, diagnostics = [], []
        for job in jobs:
            torch.manual_seed(job.seed)
            torch.cuda.manual_seed_all(job.seed)
            _, z, noise_hash = initial_latents(
                pipe, job, device=device, dtype=torch.float32, height=512, width=512, steps=50)
            latent.append(z)
            diagnostics.append(GenerationDiagnostics(
                initial_latent_sha256=noise_hash, generation_batch_id=f"{run_id}:{cell}:{batch_id}",
                valid_batch_size=len(real), batch_padding_count=4 - len(real)))
        losses = [TabObjective(r) if r.get("transfer_region") or r["relations"] else None for r in batch]
        # Multiplication by 0.5 retains the default scale 10 exactly; boosted rows use 80/.06.
        use_guidance_strength = any(r["loss_config"]["scale_factor"] == 80.0 for r in batch)
        result = generate_predicated_batch(
            pipe, jobs, device=device, steps=50, guidance_scale=7.5, height=512, width=512,
            attention_res=16, max_iter_to_alter=25, scale_factor=20.0 if use_guidance_strength else 10.0,
            scale_range=(1., .5), thresholds={0: 0.}, refinement_updates_per_threshold=5,
            initial_latents=torch.cat(latent), diagnostics=diagnostics, strict_finite=True, max_update_ratio=.02,
            max_latent_abs=20., image_guidance=[guidance_strength(r) for r in batch] if use_guidance_strength else None,
            count_guidance=None, loss_overrides=losses if any(losses) else None,
        )
        records = []
        for i, row in enumerate(real):
            image_path = out / "images" / (row["image_id"] + ".png")
            save_image_atomic(result.images[i], image_path, expected_size=(512, 512))
            record = {**row, "image_path": repo_relative(image_path, repository_root=root),
                      "image_sha256": sha256_file(image_path), "generation_origin": "generated",
                      "generation_diagnostics": diagnostics[i].as_dict(),
                      "transfer_diagnostics": losses[i].diagnostics() if losses[i] else {},
                      "physical_batch_size": 4, "valid_batch_size": len(real), "padding_count": 4 - len(real)}
            validate_diagnostics(record)
            records.append(record)
        immutable_json(receipt, {"plan_sha256": plan_hash, "records": records})
        completed += 1
        atomic_write_json(out / "progress.json", {
            "batch_id": batch_id, "completed_batches": completed, "planned_batches": len(selected),
            "seconds": time.monotonic() - started})
        if completed % 10 == 0:
            print(json.dumps({"cell": cell, "completed_batches": completed}), flush=True)
    print(json.dumps({"cell": cell, "status": "complete", "batches": completed}), flush=True)


def merge(run_id: str) -> Path:
    """Write ``metadata.jsonl``, ``run_manifest.json`` and the trial plan of a complete run."""
    run_dir = run_directory(run_id)
    plan, rows, plan_hash = load_run(run_dir)
    records = []
    for batch_id in plan["selected_physical_batches"]:
        records.extend(verify_receipt(receipt_path(run_dir, plan["cell"], batch_id),
                                      rows[4 * batch_id:4 * batch_id + 4], plan_hash))
    pp = [r for r in records if r["tab_id"].startswith("tab-pp-")]
    if len(pp) != PP_JOBS:
        raise ValueError("TAB run is incomplete")
    immutable_rows(run_dir / "metadata.jsonl", records)
    immutable_json(run_dir / "run_manifest.json", {
        "status": "complete", "plan_sha256": plan_hash, "PP_images": len(pp),
        "auxiliary_images": len(records) - len(pp), "metadata_sha256": sha256_file(run_dir / "metadata.jsonl"),
        "counts": dict(Counter(r["generation_origin"] for r in records)),
    })
    vqa_dir = repository_root() / "output/vqa" / run_id
    vqa_dir.mkdir(parents=True, exist_ok=True)
    write_trial_plan(vqa_dir / "trial_plan.jsonl", pp, profile=pp_profile(TRIAL_PROFILE), with_source_metadata=False)
    return run_dir / "metadata.jsonl"
