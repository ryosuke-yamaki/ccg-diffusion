#!/usr/bin/env python3
"""Prepare the Stage 1-3 inputs of the full T2I-CompBench++ dataset in one stage directory.

1. Per category, the 300 seed-0 base prompts (``<cat>.prompt_only.jsonl``) of
   the frozen dataset and ``input_population.json``, which records how they
   relate to the 24,000-row seed manifest (seeds 42-51).
2. Per category, the Stage 1 parse corpus: ``<cat>.raw`` (one tokenized
   sentence per line), ``<cat>.source_map.jsonl``, ``<cat>.sentence_map.jsonl``
   and ``<cat>.report.json``.
3. ``--parse-inputs frozen`` (default): the recorded Stage 1 AUTO, Stage 2
   marked dependencies and coverage rows of every evaluated T2I run are
   unpacked from ``data/t2i_compbench/`` (byte-identical), so the rest of the
   pipeline runs on CPU without re-parsing (a re-parse on other hardware can
   flip near-tie supertags). ``--parse-inputs parse`` prints the Stage 1 (GPU)
   and Stage 2 commands instead.
4. ``source_manifest.json``: per category the Stage 1-3 files, including the
   guidance-target file that Stage 3 (``scripts/ccg/build_guidance_targets.py``,
   printed at the end) writes next.

Run from the repository root::

    python -m scripts.t2i_compbench.prepare_prompts --stage-dir output/generation/RUN_ID/stage123
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from ccg_diffusion.config import T2I_COMPBENCH_CATEGORIES, T2I_COMPBENCH_DATASET_DIR, T2I_COMPBENCH_SEEDS
from ccg_diffusion.parsing.sentences import write_parse_corpus
from ccg_diffusion.provenance import (
    artifact_record,
    atomic_write_json,
    atomic_write_jsonl,
    read_jsonl,
    repo_relative,
    repository_root,
)
from ccg_diffusion.targets.guidance_targets import SOURCE_MANIFEST_SCHEMA, unpack_t2i_parse_bundle

DATASET_DIR = Path(__file__).resolve().parents[2] / T2I_COMPBENCH_DATASET_DIR
PROMPT_FIELDS = frozenset({"experiment", "prompt_id", "prompt", "seed"})


class InputPreparationError(RuntimeError):
    """The base or generation population violates the paired-seed contract."""


def _load(path: Path, *, label: str) -> list[dict[str, Any]]:
    rows = [dict(row) for row in read_jsonl(path, reject_duplicate_keys=True)]
    seen: set[str] = set()
    for row_number, row in enumerate(rows, start=1):
        if set(row) != PROMPT_FIELDS:
            raise InputPreparationError(
                f"{label} {path}:{row_number} fields must be {sorted(PROMPT_FIELDS)}"
            )
        if any(not isinstance(row[field], str) or not row[field] for field in ("experiment", "prompt_id", "prompt")):
            raise InputPreparationError(f"invalid text field in {label} row {row_number}")
        if isinstance(row["seed"], bool) or not isinstance(row["seed"], int):
            raise InputPreparationError(f"invalid seed in {label} row {row_number}")
        if row["prompt_id"] in seen:
            raise InputPreparationError(f"duplicate prompt_id in {label}: {row['prompt_id']}")
        seen.add(row["prompt_id"])
    if not rows:
        raise InputPreparationError(f"empty {label}: {path}")
    return rows


def _category(row: Mapping[str, Any]) -> str:
    experiment = str(row["experiment"])
    if not experiment.startswith("t2icb_"):
        raise InputPreparationError(f"invalid T2I-CompBench++ experiment: {experiment}")
    category = experiment.removeprefix("t2icb_")
    if category not in T2I_COMPBENCH_CATEGORIES:
        raise InputPreparationError(f"unsupported category: {category}")
    return category


def _file_record(path: Path) -> dict[str, str]:
    record = artifact_record(path).to_dict(relative_to=repository_root())
    return {"path": record["path"], "sha256": record["sha256"]}


def _manifest_path(path: Path) -> str:
    """Repository-relative path inside the checkout, absolute otherwise."""
    try:
        return repo_relative(path)
    except ValueError:
        return str(path.resolve())


def _write_category_prompts(
    blind_prompt_manifest: Path, generation_manifest: Path, stage_dir: Path, *, overwrite: bool,
) -> dict[str, int]:
    """Split the blind seed manifest into per-category base prompts (seed 0) and write ``input_population.json``."""
    blind_rows = _load(blind_prompt_manifest, label="blind prompt manifest")
    generation_rows = _load(generation_manifest, label="generation manifest")
    generation_by_id = {str(row["prompt_id"]): row for row in generation_rows}
    if set(generation_by_id) != {str(row["prompt_id"]) for row in blind_rows}:
        raise InputPreparationError("blind and generation prompt_id populations differ")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for blind_row in blind_rows:
        if blind_row["seed"] != 0:
            raise InputPreparationError("blind prompt manifest rows must use seed 0")
        generation_row = generation_by_id[str(blind_row["prompt_id"])]
        if any(
            blind_row[field] != generation_row[field]
            for field in ("experiment", "prompt_id", "prompt")
        ):
            raise InputPreparationError(
                f"blind/generation identity mismatch: {blind_row['prompt_id']}"
            )
        seed = int(generation_row["seed"])
        if seed not in T2I_COMPBENCH_SEEDS or not str(blind_row["prompt_id"]).endswith(
            f"_s{seed}"
        ):
            raise InputPreparationError(
                f"generation seed/id mismatch: {blind_row['prompt_id']}/{seed}"
            )
        grouped.setdefault(
            (str(blind_row["experiment"]), str(blind_row["prompt"])), []
        ).append(generation_row)

    base_rows = []
    for key, group in grouped.items():
        seeds = {int(row["seed"]) for row in group}
        if len(group) != 10 or seeds != set(T2I_COMPBENCH_SEEDS):
            raise InputPreparationError(f"prompt does not contain seeds 42-51: {key}")
        base_ids = {
            str(row["prompt_id"])[: -len(f"_s{row['seed']}")]
            for row in group
        }
        if len(base_ids) != 1:
            raise InputPreparationError(f"base prompt_id changes across seeds: {key}")
        base_rows.append(
            {
                "experiment": key[0],
                "prompt_id": next(iter(base_ids)),
                "prompt": key[1],
                "seed": 0,
            }
        )

    base_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in base_rows:
        key = (str(row["experiment"]), str(row["prompt"]))
        if key in base_by_key:
            raise InputPreparationError(f"duplicate base prompt text within category: {key}")
        base_by_key[key] = row

    generation_counts: Counter[tuple[str, str]] = Counter()
    for row in generation_rows:
        key = (str(row["experiment"]), str(row["prompt"]))
        if key not in base_by_key:
            raise InputPreparationError(f"generation row has no base prompt: {row['prompt_id']}")
        base_id = str(base_by_key[key]["prompt_id"])
        expected_id = f"{base_id}_s{row['seed']}"
        if row["prompt_id"] != expected_id:
            raise InputPreparationError(
                f"generation prompt_id does not encode base/seed: {row['prompt_id']} != {expected_id}"
            )
        generation_counts[key] += 1
    if set(generation_counts) != set(base_by_key):
        raise InputPreparationError("base and generation prompt populations differ")
    wrong_counts = {key: count for key, count in generation_counts.items() if count != 10}
    if wrong_counts:
        raise InputPreparationError(f"every base prompt requires ten seed rows: {wrong_counts}")

    stage_dir.mkdir(parents=True, exist_ok=True)
    root = repository_root()
    category_counts: dict[str, int] = {}
    outputs = []
    for category in T2I_COMPBENCH_CATEGORIES:
        selected = [row for row in base_rows if _category(row) == category]
        if not selected:
            raise InputPreparationError(f"category has no prompts: {category}")
        output = stage_dir / f"{category}.prompt_only.jsonl"
        record = atomic_write_jsonl(output, selected, overwrite=overwrite)
        category_counts[category] = len(selected)
        outputs.append(record.to_dict(relative_to=root))

    atomic_write_json(
        stage_dir / "input_population.json",
        {
            "artifact": "sd14_t2icb_stage123_input_population",
            "blind_prompt_manifest": _file_record(blind_prompt_manifest),
            "generation_manifest": _file_record(generation_manifest),
            "base_prompt_count": len(base_rows),
            "generation_row_count": len(generation_rows),
            "seeds_per_prompt": 10,
            "category_counts": category_counts,
            "outputs": outputs,
            "gold_inputs_used": False,
        },
        overwrite=overwrite,
    )
    return category_counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage-dir", type=Path, required=True, help="Stage 1-3 directory to write")
    parser.add_argument("--blind-prompt-manifest", type=Path, default=DATASET_DIR / "prompt_only.jsonl",
                        help="blind seed manifest (prompt_id <base>_s<seed>, seed 0)")
    parser.add_argument("--generation-manifest", type=Path, default=DATASET_DIR / "independent_seed_manifest.jsonl",
                        help="seed manifest of the 24,000 images")
    parser.add_argument("--parse-inputs", choices=("frozen", "parse"), default="frozen",
                        help="frozen: unpack the recorded Stage 1-2 outputs; parse: print the Stage 1-2 commands")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    stage = args.stage_dir
    counts = _write_category_prompts(args.blind_prompt_manifest, args.generation_manifest, stage,
                                     overwrite=args.overwrite)
    for category in T2I_COMPBENCH_CATEGORIES:
        base = stage / category
        write_parse_corpus(
            prompt_manifest=Path(f"{base}.prompt_only.jsonl"),
            raw_output=Path(f"{base}.raw"),
            source_map_output=Path(f"{base}.source_map.jsonl"),
            sentence_map_output=Path(f"{base}.sentence_map.jsonl"),
            report_output=Path(f"{base}.report.json"),
            expected_count=counts[category],
            overwrite=args.overwrite,
        )
    if args.parse_inputs == "frozen":
        unpack_t2i_parse_bundle(stage)
    sources = [
        {
            "prompt_only": _manifest_path(stage / f"{category}.prompt_only.jsonl"),
            "auto": _manifest_path(stage / f"{category}.auto"),
            "sentence_map": _manifest_path(stage / f"{category}.sentence_map.jsonl"),
            "marked_deps": _manifest_path(stage / "generate" / f"{category}.markedup_deps"),
            "pd_sidecar": _manifest_path(stage / f"{category}.deps2index_sd14.jsonl"),
        }
        for category in T2I_COMPBENCH_CATEGORIES
    ]
    atomic_write_json(stage / "source_manifest.json", {"schema": SOURCE_MANIFEST_SCHEMA, "sources": sources},
                      overwrite=args.overwrite)

    print(f"[prepare_prompts] wrote {stage} ({args.parse_inputs} parse inputs); next, per category:")
    for category in T2I_COMPBENCH_CATEGORIES:
        base = f"{stage}/{category}"
        if args.parse_inputs == "parse":
            print(f"python -m scripts.ccg.parse --input-raw-path {base}.raw --output-auto-path {base}.auto")
            print(f"python -m scripts.ccg.generate_dependencies --auto-path {base}.auto --raw-path {base}.raw "
                  f"--output-dir {stage}/generate --apply-skimmer")
        print(f"python -m scripts.ccg.build_guidance_targets --prompt-only {base}.prompt_only.jsonl "
              f"--auto-path {base}.auto --sentence-map-path {base}.sentence_map.jsonl "
              f"--markedup-deps-path {stage}/generate/{category}.markedup_deps "
              f"--coverage-rows-path {stage}/generate/{category}.coverage_rows.jsonl "
              f"--output {base}.deps2index_sd14.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
