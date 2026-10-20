"""Export the images of one method or ablation condition (arm) into the official T2I-CompBench++ ``samples/`` layout.

The official evaluators read the prompt and the image index back from the file
name ``<prompt>_<index:06d>.png`` (exactly one underscore) and expect image
``prompt_position * 10 + seed_position``. ``export_samples`` links every
generated image of one category under that name in
``output/vqa/<run>/t2icb_work/<arm>/<category>/samples/`` and writes the
``metric_stage_manifest.json`` that seals each link with the image's sha256 and
the population identity (``dataset.expected_population_identity``).

Registered arm IDs (``EXPORT_METHODS``) must equal their method; ablation
condition aliases (e.g. ``gs0_p1_r1``) keep the truthful method.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from ccg_diffusion.backbones.jobs import Job, image_path
from ccg_diffusion.evaluation.t2i_compbench.dataset import (
    CATEGORIES, CONTRACT_NAME, DATASET_DIR, DATASET_ID, DATASET_VERSION, REPO_ROOT, SEEDS, SELECTION_NAME,
    T2ICompBenchError,
)
from ccg_diffusion.provenance import sha256_file

IMAGES_PER_PROMPT = len(SEEDS)
# Method directory names of the generation runs (recorded arm IDs).
EXPORT_METHODS = (
    "sd14_pd_holccg", "sd14_pd_spacy", "sd14_composable", "sd14_structure", "sd14_attend_excite",
    "sd14_syngen", "sana_baseline", "sana_ccg_compositional_v1", "sana_ccg_compositional_shape_v1",
    "sana_ccg_compositional_ablation_v1", "sana_ccg_compositional_no_prefix_v1", "sana_composable",
    "sana_structure", "sana_attend_excite", "sana_syngen", "sana_pd_spacy",
)
SEED_PROMPT_ID_RE = re.compile(r"^(?P<base>.+)_s(?P<seed>\d+)$")


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise T2ICompBenchError(f"generation manifest not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise T2ICompBenchError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise T2ICompBenchError(f"non-object row at {path}:{line_number}")
            rows.append(row)
    return rows


def category_from_experiment(experiment: str) -> str:
    prefix = "t2icb_"
    if not experiment.startswith(prefix):
        raise T2ICompBenchError(f"unexpected experiment: {experiment!r}")
    category = experiment[len(prefix):]
    if category not in CATEGORIES:
        raise T2ICompBenchError(f"unknown category in experiment: {experiment!r}")
    return category


def grouped_prompt_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[list[Mapping[str, Any]]]]:
    """Group manifest rows by category and first-seen prompt order."""

    grouped: dict[str, dict[str, list[Mapping[str, Any]]]] = {
        category: {} for category in CATEGORIES
    }
    for row in rows:
        category = category_from_experiment(str(row.get("experiment")))
        prompt_id = str(row.get("prompt_id"))
        match = SEED_PROMPT_ID_RE.match(prompt_id)
        if match is None or int(match.group("seed")) != row.get("seed"):
            raise T2ICompBenchError(
                f"{category}: prompt_id must end in its seed: {prompt_id!r}"
            )
        grouped[category].setdefault(match.group("base"), []).append(row)

    result: dict[str, list[list[Mapping[str, Any]]]] = {}
    for category, prompts in grouped.items():
        result[category] = list(prompts.values())
        for prompt_rows in result[category]:
            prompts_seen = {str(row.get("prompt")) for row in prompt_rows}
            seeds = [row.get("seed") for row in prompt_rows]
            if len(prompts_seen) != 1:
                raise T2ICompBenchError(
                    f"{category}: prompt rows disagree on prompt text"
                )
            if seeds != list(SEEDS):
                raise T2ICompBenchError(
                    f"{category}/{prompt_rows[0].get('prompt_id')}: expected seeds "
                    f"{list(SEEDS)}, got {seeds}"
                )
    return result


def source_path(
    generation_out_dir: Path,
    method: str,
    row: Mapping[str, Any],
) -> Path:
    """The generated image of one manifest row (the generators' naming rule)."""

    job = Job(
        experiment=str(row["experiment"]),
        prompt_id=str(row["prompt_id"]),
        prompt=str(row["prompt"]),
        seed=int(row["seed"]),
        attention_exist_indices=[],
        attention_corr_indices=[],
        attention_leak_indices=[],
        attention_possession_indices=[],
        source="subset_export",
    )
    return image_path(generation_out_dir, method, job)


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def export_samples(
    category: str,
    *,
    run_id: str,
    arm: str,
    method: str,
    generation_out_dir: Path,
    dataset_dir: Path = DATASET_DIR,
    generation_manifest: Path | None = None,
    vqa_root: Path = REPO_ROOT / "output/vqa",
) -> dict[str, Any]:
    """Link one category of ``arm`` into the official layout and write its metric stage manifest."""

    if arm in EXPORT_METHODS and arm != method:
        raise T2ICompBenchError(
            f"registered arm export requires --arm == --method; got "
            f"arm={arm!r}, method={method!r}"
        )
    if category not in CATEGORIES:
        raise T2ICompBenchError(f"unknown categories: {[category]}")
    canonical = dataset_dir.resolve() == DATASET_DIR.resolve()
    if generation_manifest is None:
        generation_manifest = dataset_dir / "independent_seed_manifest.jsonl"
    prompt_groups = grouped_prompt_rows(load_rows(generation_manifest))[category]
    if not prompt_groups:
        raise T2ICompBenchError(f"generation manifest has no {category} rows")
    prompt_file = dataset_dir / "prompts" / f"{category}_val.txt"
    selection_manifest = dataset_dir / SELECTION_NAME
    if not prompt_file.is_file():
        raise T2ICompBenchError(f"sampled prompt file not found: {prompt_file}")
    prompts = prompt_file.read_text(encoding="utf-8").splitlines()
    grouped_prompts = [str(group[0]["prompt"]) for group in prompt_groups]
    if grouped_prompts != prompts:
        raise T2ICompBenchError(
            f"{category}: generation manifest order differs from {prompt_file}"
        )

    work_dir = vqa_root / run_id / "t2icb_work" / arm / category
    samples_dir = work_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    plan: list[dict[str, Any]] = []
    for prompt_position, group in enumerate(prompt_groups):
        for seed_position, row in enumerate(group):
            index = prompt_position * IMAGES_PER_PROMPT + seed_position
            prompt = str(row["prompt"])
            name = f"{prompt}_{index:06d}.png"
            if name.count("_") != 1:
                raise T2ICompBenchError(
                    f"official sample name must contain exactly one underscore: {name!r}"
                )
            source = source_path(generation_out_dir, method, row)
            if not source.is_file():
                raise T2ICompBenchError(f"source image not found: {source}")
            plan.append(
                {
                    "prompt_id": str(row["prompt_id"]),
                    "prompt": prompt,
                    "seed": int(row["seed"]),
                    "index": index,
                    "name": name,
                    "source": source,
                }
            )

    for entry in plan:
        link = samples_dir / str(entry["name"])
        target = Path(entry["source"]).resolve()
        if link.is_symlink():
            if Path(os.readlink(link)).resolve() != target:
                raise T2ICompBenchError(
                    f"existing link {link} points to a different source"
                )
            continue
        if link.exists():
            raise T2ICompBenchError(f"existing sample is not a symlink: {link}")
        link.symlink_to(target)

    planned_names = {str(entry["name"]) for entry in plan}
    actual_names = {entry.name for entry in samples_dir.iterdir()}
    stray = sorted(actual_names - planned_names)
    if stray:
        raise T2ICompBenchError(
            f"{samples_dir} contains files outside this export; first: {stray[0]}"
        )
    if len(actual_names) != len(plan):
        raise T2ICompBenchError(
            f"{samples_dir}: expected {len(plan)} samples, found {len(actual_names)}"
        )

    manifest = {
        "schema": "holsd.t2icb_subset_export.v2",
        "run_id": run_id,
        "arm": arm,
        "method": method,
        "category": category,
        "screening_dataset_id": None,
        "screening_dataset_version": None,
        "dataset_id": DATASET_ID if canonical else None,
        "dataset_version": DATASET_VERSION if canonical else None,
        "prompt_count": len(prompt_groups),
        "seeds": list(SEEDS),
        "image_count": len(plan),
        "generation_out_dir": str(generation_out_dir.resolve()),
        "generation_manifest": {
            "path": str(generation_manifest.resolve()),
            "sha256": sha256_file(generation_manifest),
        },
        "selection_manifest": (
            {
                "path": str(selection_manifest.resolve()),
                "sha256": sha256_file(selection_manifest),
            }
            if selection_manifest.is_file()
            else None
        ),
        "dataset_contract": (
            {
                "path": str((dataset_dir / CONTRACT_NAME).resolve()),
                "sha256": sha256_file(dataset_dir / CONTRACT_NAME),
            }
            if (dataset_dir / CONTRACT_NAME).is_file()
            else None
        ),
        "prompt_file": str(prompt_file.resolve()),
        "prompt_file_sha256": sha256_file(prompt_file),
        "population_identity_sha256": hashlib.sha256(
            json.dumps(
                [
                    {
                        "prompt_id": entry["prompt_id"],
                        "prompt": entry["prompt"],
                        "seed": entry["seed"],
                        "index": entry["index"],
                    }
                    for entry in plan
                ],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "samples_dir": str(samples_dir.resolve()),
        "images": [
            {
                "prompt_id": entry["prompt_id"],
                "seed": entry["seed"],
                "index": entry["index"],
                "name": entry["name"],
                "source": str(Path(entry["source"]).resolve()),
                "sha256": sha256_file(Path(entry["source"])),
            }
            for entry in plan
        ],
    }
    write_json_atomic(work_dir / "metric_stage_manifest.json", manifest)
    return manifest
