"""TAB PP job population, frozen input bundle, run plans and receipts (SD1.4 layout).

The paper's SD1.4 TAB runs place every job at its position in a fixed 3,150-job
layout: the 315 TAB items in dataset order x interpretations ``np_attach`` /
``vp_attach`` x seeds 0-4. The 740 PP jobs occupy positions 0-739, so physical
batches of 4 (``position // 4``) 0-184 hold only PP jobs and are the only ones
generated. The 2,410 VP rows keep their frozen job (never generated); they are
kept so that a prepared job file equals the evaluated one row for row.

Inputs come either from the frozen bundle ``data/tab_pp/`` (the paper jobs,
byte-equal to the evaluated runs) or from ``scripts/tab_pp/parse_interpretations.py``
(the clean path from Supertag-Constrained Parsing; 740 PP rows at the same
positions). ``data/tab_pp/reference.json`` holds the digests of the evaluated
jobs that ``preflight`` compares with.
"""
from __future__ import annotations

import gzip
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from ccg_diffusion.methods import TabAblation
from ccg_diffusion.pp_attachment.compile import (
    Interpretation,
    InterpretationTargets,
    compile_auxiliary_job,
    compile_interpretation,
)
from ccg_diffusion.provenance import (
    canonical_json,
    digest,
    immutable_json,
    read_jsonl,
    repository_root,
    sha256_bytes,
    sha256_file,
)

IDENTITY = ("arm", "run_id", "condition", "image_id")
POPULATION_JOBS = 3150
PP_JOBS = 740
PHYSICAL_BATCH = 4
READINGS = ("np_attach", "vp_attach")
SEEDS = tuple(range(5))
EFFECTIVE_FIELDS = ("prompt", "token_ids", "pd_indices", "seed", "loss_config", "relations",
                    "phrase_conditioning", "transfer_region")  # phrase_conditioning is always None
BUNDLE_DIR = repository_root() / "data/tab_pp"
BUNDLE_SCHEMA = "ccg_diffusion.tab_pp_bundle.v1"
SOURCE_JOBS_FILE = "source_jobs.jsonl.gz"
INTERPRETATIONS_FILE = "interpretations.jsonl.gz"
REFERENCE_FILE = "reference.json"
MANIFEST_FILE = "manifest.json"
PLAN_SCHEMA = "ccg_diffusion.tab_pp.sd14_run.v1"
CLEAN_INTERPRETATIONS_FILE = "interpretations.jsonl"
RESULT_FIELDS = ("image_path", "image_sha256", "generation_origin", "generation_diagnostics", "phrase_diagnostics",
                 "transfer_diagnostics", "physical_batch_size", "valid_batch_size", "padding_count")
_RUN_ID = re.compile(r"[0-9]{8}_[a-z0-9][a-z0-9_-]*")

# The SD1.4 TAB generation settings recorded in every TAB plan (verbatim).
TAB_SD14_GENERATION_RECORD = MappingProxyType({
    "model_id": "CompVis/stable-diffusion-v1-4",
    "model_revision": "133a221b8aa7292a167afc5127cb63fb5005638b",
    "height": 512,
    "width": 512,
    "steps": 50,
    "guidance_scale": 7.5,
    "precision_profile": "bf16_stable",
    "execution_mode": "throughput",
    "physical_batch_size": 4,
    "padding_policy": "repeat_last_real_job_discard_padded_outputs",
    "compile_mode": "off",
    "memory_format": "contiguous",
    "strict_finite": True,
    "attention_res": 16,
    "attention_block_scope": "all_at_attention_res",
    "attention_processors": 5,
    "max_iter_to_alter": 25,
    "scale_range": [1.0, 0.5],
    "thresholds": {"0": 0.0},
    "refinement_updates_per_threshold": 5,
    "guidance_updates": 30,
    "max_latent_abs": 20.0,
    "negative_prompt": None,
    "scheduler": "PNDMScheduler",
    "scheduler_prediction_type": "epsilon",
    "scheduler_config_sha256": "c6589022d9d8432e019b2750354e061eec62aed48d1a7e82c036dc40f33a534f",
    "torch": "2.13.0+cu129",
    "transformers": "5.14.1",
    "diffusers": "0.39.0",
})

# Files whose bytes an SD1.4 TAB plan seals (``plan.source_sha256``): the ccg_diffusion import closure of
# ``pp_attachment/sd14.py``, the CLI and the bundle manifest. ``generate`` re-verifies them before each batch.
SD14_SOURCE_FILES = (
    "data/tab_pp/manifest.json",
    "scripts/tab_pp/run_ccg_diffusion.py",
    "src/ccg_diffusion/backbones/jobs.py",
    "src/ccg_diffusion/backbones/sd14/driver.py",
    "src/ccg_diffusion/backbones/sd14/inputs.py",
    "src/ccg_diffusion/backbones/sd14/pipeline.py",
    "src/ccg_diffusion/backbones/sd14/records.py",
    "src/ccg_diffusion/config.py",
    "src/ccg_diffusion/evaluation/tab.py",
    "src/ccg_diffusion/guidance/_record_only_screening.py",
    "src/ccg_diffusion/guidance/attention.py",
    "src/ccg_diffusion/guidance/count_loss.py",
    "src/ccg_diffusion/guidance/latent_update.py",
    "src/ccg_diffusion/guidance/positional_prefix.py",
    "src/ccg_diffusion/guidance/precision.py",
    "src/ccg_diffusion/guidance/predicated_diffusion_loss.py",
    "src/ccg_diffusion/guidance/region_assignment_loss.py",
    "src/ccg_diffusion/guidance/selective_guidance_boost.py",
    "src/ccg_diffusion/guidance/spatial_relation_loss.py",
    "src/ccg_diffusion/guidance/spatial_relations.py",
    "src/ccg_diffusion/methods.py",
    "src/ccg_diffusion/parsing/ccg.py",
    "src/ccg_diffusion/parsing/derivation.py",
    "src/ccg_diffusion/parsing/holccg.py",
    "src/ccg_diffusion/parsing/sentences.py",
    "src/ccg_diffusion/pp_attachment/compile.py",
    "src/ccg_diffusion/pp_attachment/disambiguating_paraphrase.py",
    "src/ccg_diffusion/pp_attachment/holder_object.py",
    "src/ccg_diffusion/pp_attachment/jobs.py",
    "src/ccg_diffusion/pp_attachment/objective.py",
    "src/ccg_diffusion/pp_attachment/sd14.py",
    "src/ccg_diffusion/pp_attachment/spatial.py",
    "src/ccg_diffusion/pp_attachment/supertag_constrained_parsing.py",
    "src/ccg_diffusion/provenance.py",
    "src/ccg_diffusion/targets/guidance_targets.py",
    "src/ccg_diffusion/targets/lexicon.py",
    "src/ccg_diffusion/targets/pd_indices.py",
    "src/ccg_diffusion/targets/tokenizer.py",
)


def effective_inputs(row: Mapping[str, Any]) -> dict[str, Any]:
    """The fields of a job that determine its image (hashed as ``effective_sha256``)."""
    return {k: row.get(k) for k in EFFECTIVE_FIELDS}


def historical_input_signature(row: Mapping[str, Any]) -> str:
    """``source_standard_input_signature`` of a job template with its seed (recorded lineage digest)."""
    return digest({
        "prompt": row["prompt"],
        "token_ids": row["token_ids"],
        "pd_indices": row["pd_indices"],
        "seed": row["seed"],
        "loss_config": row["loss_config"],
        "relations": row.get("relations", []),
        "holder_aware_constraints": row.get("holder_aware_constraints", []),
        "generation": dict(TAB_SD14_GENERATION_RECORD),
    })


def opposite_reading(reading: str) -> str:
    return "vp_attach" if reading == "np_attach" else "np_attach"


# --- interpretations and the frozen bundle -----------------------------------------------------------


def interpretation_from_row(row: Mapping[str, Any], source_job: Mapping[str, Any]) -> Interpretation:
    """An ``Interpretation`` from one interpretation row (bundle or clean-path file) and its job template."""
    targets = row["guidance_targets"]
    return Interpretation(
        row["tab_id"], row["reading"], dict(source_job), row["paraphrase"],
        InterpretationTargets(targets["prompt"], tuple(targets["alignment"]), targets["pd_indices"],
                              targets["boost"], tuple(targets["spatial_relations"]), targets["record_sha256"]),
        row["standard_program"],
    )


@dataclass(frozen=True)
class FrozenBundle:
    """The paper's TAB PP inputs: 3,150 source jobs (original order), 148 interpretations, reference digests."""

    source_jobs: tuple[Mapping[str, Any], ...]
    interpretations: Mapping[tuple[str, str], Interpretation]
    interpretation_rows: tuple[Mapping[str, Any], ...]
    reference: Mapping[str, Any]
    manifest: Mapping[str, Any]


def read_gzip_rows(path: Path, *, sha256: str | None = None, gzip_sha256: str | None = None) -> list[dict]:
    """Rows of a gzip-compressed JSONL file, optionally checked against the raw and gzip sha256."""
    data = path.read_bytes()
    raw = gzip.decompress(data)
    if gzip_sha256 is not None and sha256_bytes(data) != gzip_sha256:
        raise ValueError(f"frozen TAB bundle file changed: {path}")
    if sha256 is not None and sha256_bytes(raw) != sha256:
        raise ValueError(f"frozen TAB bundle content changed: {path}")
    return [json.loads(line) for line in raw.decode("utf-8").splitlines()]


def load_frozen_bundle(path: Path = BUNDLE_DIR, *, verify: bool = True) -> FrozenBundle:
    """Load ``data/tab_pp/``; ``verify`` checks every file against ``manifest.json``."""
    manifest = json.loads((path / MANIFEST_FILE).read_text(encoding="utf-8"))
    if manifest.get("schema") != BUNDLE_SCHEMA:
        raise ValueError("unsupported TAB PP bundle manifest")
    files = manifest["files"]

    def rows(name: str) -> list[dict]:
        record = files[name]
        result = read_gzip_rows(path / name, sha256=record["sha256"] if verify else None,
                                gzip_sha256=record["gzip_sha256"] if verify else None)
        if len(result) != record["rows"]:
            raise ValueError(f"frozen TAB bundle row count changed: {name}")
        return result

    if verify and sha256_file(path / REFERENCE_FILE) != files[REFERENCE_FILE]["sha256"]:
        raise ValueError("frozen TAB bundle reference changed")
    source_jobs = tuple(rows(SOURCE_JOBS_FILE))
    interpretation_rows = tuple(rows(INTERPRETATIONS_FILE))
    templates: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in source_jobs:
        templates.setdefault((row["tab_id"], row["reading"]), row)
    interpretations = {(row["tab_id"], row["reading"]): interpretation_from_row(row, templates[row["tab_id"],
                                                                                               row["reading"]])
                       for row in interpretation_rows}
    reference = json.loads((path / REFERENCE_FILE).read_text(encoding="utf-8"))
    return FrozenBundle(source_jobs, interpretations, interpretation_rows, reference, manifest)


def load_clean_interpretations(path: Path) -> dict[tuple[str, str], Interpretation]:
    """The interpretations written by ``scripts/tab_pp/parse_interpretations.py`` (``<dir>/interpretations.jsonl``)."""
    rows = read_jsonl(path / CLEAN_INTERPRETATIONS_FILE)
    return {(row["tab_id"], row["reading"]): interpretation_from_row(row, row["source_job"]) for row in rows}


# --- job placement -----------------------------------------------------------------------------------


def clean_layout(interpretations: Mapping[tuple[str, str], Interpretation]) -> list[dict[str, Any]]:
    """The 740 PP positions of the clean path: items in dataset order x interpretations x seeds 0-4."""
    rows = []
    for (tab_id, reading), interpretation in interpretations.items():
        for seed in SEEDS:
            row = {**interpretation.source_job, "seed": seed}
            row["source_standard_input_signature"] = historical_input_signature(row)
            rows.append(row)
    return rows


def place_jobs(
    *, interpretations: Mapping[tuple[str, str], Interpretation], source_jobs: Sequence[Mapping[str, Any]],
    ablation: TabAblation, run_id: str, tokenizer: Any,
) -> list[dict[str, Any]]:
    """Compile every job of one Table 6 cell at its original position (frozen 3,150 or clean 740 layout).

    One job is compiled per (item, interpretation) from its first (seed-0)
    position; every position copies it with its own seed and identity.
    """
    cell = ablation.cell_id
    jobs, cached = [], {}
    for position, source in enumerate(source_jobs):
        key = source["tab_id"], source["reading"]
        if key not in cached:
            if source["tab_id"].startswith("tab-pp-"):
                cached[key] = compile_interpretation(
                    interpretations[key], interpretations[key[0], opposite_reading(key[1])], tokenizer, ablation)
            else:
                cached[key] = compile_auxiliary_job(source, ablation)
        row = deepcopy(cached[key])
        row.update(seed=source["seed"], arm=cell, run_id=run_id, condition=f"{run_id}:{cell}",
                   source_standard_input_signature=source["source_standard_input_signature"],
                   image_id=f"{cell}_{key[0]}_{key[1]}_s{source['seed']}",
                   original_position=position, physical_batch_index=position // 4,
                   physical_batch_position=position % 4)
        row["effective_sha256"] = digest(effective_inputs(row))
        jobs.append(row)
    return jobs


def selected_batches(rows: Sequence[Mapping[str, Any]]) -> list[int]:
    """The physical batches that hold PP jobs (the generated ones)."""
    return sorted({row["physical_batch_index"] for row in rows if row["tab_id"].startswith("tab-pp-")})


def check_population(
    rows: Sequence[Mapping[str, Any]], *, ablation: TabAblation,
    interpretations: Mapping[tuple[str, str], Interpretation], source_jobs: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    """Check positions, identity and PP coverage of compiled jobs; return the census."""
    if len(rows) != len(source_jobs) or len(rows) not in (POPULATION_JOBS, PP_JOBS):
        raise ValueError("TAB jobs must keep every source position (3,150 frozen or 740 clean)")
    pp = []
    for i, (row, source) in enumerate(zip(rows, source_jobs, strict=True)):
        if (row["original_position"], row["physical_batch_index"], row["physical_batch_position"]) != (i, i // 4,
                                                                                                       i % 4):
            raise ValueError("TAB jobs changed the original physical-batch membership")
        if [row[k] for k in ("tab_id", "reading", "seed")] != [source[k] for k in ("tab_id", "reading", "seed")]:
            raise ValueError("TAB jobs changed the source layout")
        if row["effective_sha256"] != digest(effective_inputs(row)):
            raise ValueError(f"effective_sha256 differs from the effective inputs: {row['image_id']}")
        if row["tab_id"].startswith("tab-pp-"):
            pp.append(row)
        elif effective_inputs(row) != effective_inputs(source):
            raise ValueError(f"auxiliary VP job changed its frozen inputs: {row['image_id']}")
    if len({(r["tab_id"], r["reading"], r["seed"]) for r in pp}) != PP_JOBS or selected_batches(rows) != list(
            range(PP_JOBS // PHYSICAL_BATCH)):
        raise ValueError("TAB jobs do not cover the 740 PP jobs in physical batches 0-184")
    source = [row["prompt"] == interpretations[row["tab_id"], row["reading"]].targets.prompt for row in pp]
    paraphrase = [row["prompt"] == interpretations[row["tab_id"], row["reading"]].source_job["prompt"] for row in pp]
    expected_prompt = paraphrase if ablation.disambiguating_paraphrase else source
    if not all(expected_prompt):
        raise ValueError("TAB PP job does not condition on its interpretation's prompt")
    return {
        "pp_images": len(pp),
        "p2_images": sum(row["loss_config"]["scale_factor"] == 80 for row in pp),
        "spatial_images": sum(bool(row["relations"]) for row in pp),
        "region_images": sum(bool(row.get("transfer_region")) for row in pp),
        "possession_pairs": sum(len(row["pd_indices"]["possession"]) for row in pp),
        "source_prompt_images": sum(source),
        "interpretation_prompt_images": sum(paraphrase),
        "prefix_images": sum(not (a or b) for a, b in zip(source, paraphrase)),
    }


# --- reference digests -------------------------------------------------------------------------------


def _strip_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if k not in IDENTITY}


def _rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    return sha256_bytes("".join(canonical_json(_strip_identity(row)) + "\n" for row in rows).encode("utf-8"))


def reference_digests(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Digests of a 3,150-job cell without run identity: all rows, each PP physical batch, the VP rows."""
    return {
        "rows_sha256": _rows_sha256(rows),
        "pp_batch_sha256": [_rows_sha256(rows[4 * b:4 * b + 4]) for b in range(PP_JOBS // PHYSICAL_BATCH)],
        "auxiliary_rows_sha256": _rows_sha256(rows[PP_JOBS:]),
    }


def compare_with_reference(
    rows: Sequence[Mapping[str, Any]], reference_cell: Mapping[str, Any], *, evaluated_batches: Path | None = None,
) -> dict[str, Any]:
    """Require frozen jobs to equal the evaluated jobs of the cell (run identity excluded).

    ``evaluated_batches`` (the evaluated run's ``batches/<cell>`` receipts) adds a
    field-by-field comparison of the PP rows, for readable differences.
    """
    report: dict[str, Any] = {"pp_rows_compared_with_receipts": 0}
    if evaluated_batches is not None and evaluated_batches.is_dir():
        for batch in range(PP_JOBS // PHYSICAL_BATCH):
            receipt = json.loads((evaluated_batches / f"batch_manifest_{batch:05d}.json").read_text())
            for record, row in zip(receipt["records"], rows[4 * batch:4 * batch + 4], strict=True):
                expected = _strip_identity({k: v for k, v in record.items() if k not in RESULT_FIELDS})
                actual = _strip_identity(row)
                if expected != actual:
                    fields = sorted(k for k in expected.keys() | actual.keys() if expected.get(k) != actual.get(k))
                    raise ValueError(f"TAB job {row['image_id']} differs from the evaluated receipt in {fields}")
                report["pp_rows_compared_with_receipts"] += 1
    digests = reference_digests(rows)
    if digests["pp_batch_sha256"] != reference_cell["pp_batch_sha256"]:
        first = next(b for b, (a, e) in enumerate(zip(digests["pp_batch_sha256"], reference_cell["pp_batch_sha256"]))
                     if a != e)
        raise ValueError(f"TAB PP physical batch {first} differs from the evaluated jobs")
    for name in ("auxiliary_rows_sha256", "rows_sha256"):
        if digests[name] != reference_cell[name]:
            raise ValueError(f"TAB jobs differ from the evaluated jobs ({name})")
    return {**report, "rows_sha256": digests["rows_sha256"], "pp_batches_equal": len(digests["pp_batch_sha256"]),
            "matched": True}


def pinned_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    """The generation-affecting fields of an SD1.4 TAB job compared between the clean and the frozen path.

    Lineage digests of the frozen inputs (``followup_trace``,
    ``standard_intervention_trace_sha256``, ``source_standard_input_signature``,
    ``target_setup_mapping_evidence_sha256``, ``transfer_trace.source_audit_sha256``,
    the layout ``source_plan_sha256`` and ``effective_sha256``) are excluded.
    """
    result = {k: deepcopy(row.get(k)) for k in (
        "prompt", "token_ids", "pd_indices", "program_sha256", "source_program_sha256", "seed", "loss_config",
        "relations", "transfer_region", "transfer_trace", "physical_batch_index", "physical_batch_position",
        "original_position", "tab_id", "reading", "target_setup_index", "evaluation_setup_indices",
        "tab_evaluation_mode", "tab_requires_reading", "tab_category")}
    if result["transfer_region"]:
        result["transfer_region"]["plan"].pop("source_plan_sha256")
    result["transfer_trace"].pop("source_audit_sha256", None)
    if "layout" in result["transfer_trace"]:
        result["transfer_trace"]["layout"].pop("source_plan_sha256")
    return result


# --- runs ----------------------------------------------------------------------------------------------


def run_directory(run_id: str) -> Path:
    """``output/generation/<run_id>`` of a ``<YYYYMMDD>_<slug>`` run ID."""
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError("run id must match [0-9]{8}_[a-z0-9][a-z0-9_-]*")
    return repository_root() / "output/generation" / run_id


def receipt_path(run_dir: Path, cell: str, batch: int) -> Path:
    return run_dir / "batches" / cell / f"batch_manifest_{batch:05d}.json"


def write_plan(run_dir: Path, plan: Mapping[str, Any]) -> None:
    immutable_json(run_dir / "plan.json", dict(plan))


def verify_hashes(hashes: Mapping[str, str]) -> None:
    """Require each repository-relative file to still have its recorded sha256."""
    root = repository_root()
    for name, expected in hashes.items():
        if sha256_file(root / name) != expected:
            raise ValueError(f"Frozen input/source changed: {name}")


def load_run(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    """``(plan, jobs, plan sha256)`` of a prepared SD1.4 TAB run, with its inputs re-verified."""
    plan = json.loads((run_dir / "plan.json").read_text())
    verify_hashes(plan["inputs"]["files"])
    if sha256_file(run_dir / "metadata_jobs.jsonl") != plan["jobs_sha256"]:
        raise ValueError("prepared TAB jobs changed")
    rows = read_jsonl(run_dir / "metadata_jobs.jsonl")
    if len(rows) != plan["population_jobs"] or any(r["effective_sha256"] != digest(effective_inputs(r))
                                                   for r in rows):
        raise ValueError("prepared TAB jobs are incomplete or changed")
    return plan, rows, sha256_file(run_dir / "plan.json")


def validate_diagnostics(row: Mapping[str, Any]) -> None:
    """Require strict finite guidance, 30 applied updates, 5 attention processors and the update cap."""
    diagnostic = row["generation_diagnostics"]
    if not diagnostic["strict_finite_applied"] or diagnostic["guidance_updates_applied"] != 30:
        raise ValueError("Strict finite or 30-update contract failed")
    bad = {
        key: value
        for key, value in diagnostic.items()
        if (key.startswith("finite_guard_nonfinite") or key == "latent_bound_violations") and value
    }
    if bad or diagnostic["captured_attention_processor_count"] != 5:
        raise ValueError(f"Finite/attention contract failed: {bad}")
    if diagnostic["max_applied_update_ratio"] > row["loss_config"]["max_update_ratio"] + 1e-7:
        raise ValueError("Relative latent-update bound exceeded")


def verify_receipt(path: Path, jobs: Sequence[Mapping[str, Any]], plan_sha256: str) -> list[dict[str, Any]]:
    """The records of one batch receipt, checked against the plan, the jobs and the stored images."""
    saved = json.loads(path.read_text())
    if saved["plan_sha256"] != plan_sha256 or len(saved["records"]) != len(jobs):
        raise ValueError("Batch receipt does not belong to the exact plan")
    root = repository_root()
    for record, job in zip(saved["records"], jobs, strict=True):
        if any(record.get(k) != v for k, v in job.items()):
            raise ValueError("Stored generation inputs differ")
        if sha256_file(root / record["image_path"]) != record["image_sha256"]:
            raise ValueError("Stored image bytes differ")
        validate_diagnostics(record)
    return saved["records"]


def preflight(run_id: str) -> dict[str, Any]:
    """Compare a prepared SD1.4 TAB run with the evaluated jobs (``data/tab_pp/reference.json``).

    A frozen-input run must equal the evaluated jobs of its cell row for row
    (run identity excluded). A clean-path run (740 PP jobs) is compared with
    the frozen jobs of the same cell on ``pinned_projection``.
    """
    run_dir = run_directory(run_id)
    plan, rows, _ = load_run(run_dir)
    bundle = load_frozen_bundle()
    ablation = TabAblation.from_cell_id(plan["cell"])
    sd14 = bundle.reference["sd14"]
    if plan["inputs"]["kind"] == "frozen":
        evaluated = repository_root() / "output/generation" / sd14["run_id"] / "batches" / plan["cell"]
        report = compare_with_reference(rows, sd14["cells"][plan["cell"]], evaluated_batches=evaluated)
        if plan["census"] != sd14["cells"][plan["cell"]]["census"]:
            raise ValueError("TAB census differs from the evaluated cell")
        return {"run_id": run_id, "cell": plan["cell"], "inputs": "frozen", **report, "census": plan["census"]}
    from ccg_diffusion.targets.tokenizer import load_sd14_tokenizer

    frozen = place_jobs(interpretations=bundle.interpretations, source_jobs=bundle.source_jobs, ablation=ablation,
                        run_id=run_id, tokenizer=load_sd14_tokenizer())[:PP_JOBS]
    differing = [row["image_id"] for row, reference in zip(rows, frozen, strict=True)
                 if pinned_projection(row) != pinned_projection(reference)]
    if differing:
        raise ValueError(f"clean-path TAB jobs differ from the frozen jobs on the pinned projection: {differing[:5]}")
    return {"run_id": run_id, "cell": plan["cell"], "inputs": "clean", "pinned_projection_equal": len(rows),
            "census": plan["census"]}
