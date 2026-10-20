#!/usr/bin/env python3
"""Build (or check) the full T2I-CompBench++ dataset: 300 test prompts per category x seeds 42-51.

    python -m scripts.data.build_t2i_compbench --output-dir DIR      # build from the official clone
    python -m scripts.data.build_t2i_compbench --check               # rebuild in a temporary directory and
                                                                     # compare every file with the frozen dataset

Reads the eight ``<category>_val.txt`` files of the pinned official clone
(their sha256 is checked against the tracked source pin), keeps every prompt
in source order (the ``equal_interval`` selection of 300 of 300 prompts, kept
with its recorded seed so the selection manifest is reproduced byte for byte),
and writes the prompt files, ``selected_prompt_manifest.jsonl``,
``prompt_only.jsonl``, ``independent_seed_manifest.jsonl`` and
``selection_manifest.json``. The hand-authored ``dataset_contract.json`` is
never written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

COMPLEX_AUXILIARY_FILES = (
    "complex_val_spatial.txt",
    "complex_val_action.txt",
    "complex_val_spatialaction.txt",
)
OFFICIAL_PROMPTS = 300  # test prompts per category in the official files
PROMPTS_PER_CATEGORY = 300
SELECTION_POLICY = "equal_interval"
SAMPLE_SEED = 20260812


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_official_prompts(path: Path, *, expected_sha256: str) -> list[str]:
    """Read one ``<category>_val.txt`` the way the official generation loop does.

    The digest is checked on the raw bytes before any decoding, the decoded
    text is split with ``splitlines()`` (universal-newline semantics drop the
    ``\\r`` of the CRLF files), and each line keeps only the text before a tab,
    mirroring the official ``d.strip("\\n").split("\\t")[0]``.
    """
    from ccg_diffusion.evaluation.t2i_compbench.dataset import T2ICompBenchError

    if not path.is_file():
        raise T2ICompBenchError(f"official prompt file not found: {path}")
    raw = path.read_bytes()
    digest = sha256_bytes(raw)
    if digest != expected_sha256:
        raise T2ICompBenchError(
            f"{path}: sha256 {digest} does not match the pinned {expected_sha256}"
        )
    prompts = [line.split("\t")[0] for line in raw.decode("utf-8").splitlines()]
    if len(prompts) != OFFICIAL_PROMPTS:
        raise T2ICompBenchError(
            f"{path}: expected {OFFICIAL_PROMPTS} prompts, found {len(prompts)}"
        )
    for line_number, prompt in enumerate(prompts):
        if not prompt:
            raise T2ICompBenchError(f"{path}: empty prompt at line {line_number}")
        if "_" in prompt:
            raise T2ICompBenchError(
                f"{path}: prompt at line {line_number} contains an underscore, which "
                f"breaks the official '<prompt>_<index>.png' name contract: {prompt!r}"
            )
        if "/" in prompt:
            raise T2ICompBenchError(
                f"{path}: prompt at line {line_number} contains a path separator: "
                f"{prompt!r}"
            )
    return prompts


def jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        for row in rows
    )


def write_bytes_atomic(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, path)
    return sha256_bytes(payload)


def load_pinned_val_txt(pin_json: Path) -> dict[str, Any]:
    from ccg_diffusion.evaluation.t2i_compbench.dataset import CATEGORIES, T2ICompBenchError

    if not pin_json.is_file():
        raise T2ICompBenchError(f"source-pin mirror not found: {pin_json}")
    payload = json.loads(pin_json.read_text(encoding="utf-8"))
    val_txt = payload.get("val_txt")
    if not isinstance(val_txt, dict):
        raise T2ICompBenchError(f"{pin_json}: missing 'val_txt' section")
    missing = sorted(set(CATEGORIES) - set(val_txt))
    if missing:
        raise T2ICompBenchError(f"{pin_json}: no pinned digest for categories {missing}")
    return val_txt


def select_line_indices(*, category_index: int) -> list[int]:
    """Sorted line indices of the ``equal_interval`` selection (one per interval: all 300 lines)."""

    category_seed = SAMPLE_SEED + (category_index + 1) * 1_000_003
    generator = random.Random(category_seed)
    selected = []
    for interval in range(PROMPTS_PER_CATEGORY):
        start = interval * OFFICIAL_PROMPTS // PROMPTS_PER_CATEGORY
        stop = (interval + 1) * OFFICIAL_PROMPTS // PROMPTS_PER_CATEGORY
        selected.append(generator.randrange(start, stop))
    return sorted(selected)


def prompt_row(category: str, line_index: int, prompt: str, *, seed: int) -> dict[str, Any]:
    return {
        "experiment": f"t2icb_{category}",
        "prompt_id": f"t2icb_{category}_{line_index:04d}",
        "prompt": prompt,
        "seed": seed,
    }


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def build(clone_dir: Path, pin_json: Path, output_dir: Path) -> None:
    """Write every dataset file except the contract into ``output_dir``."""
    from ccg_diffusion.evaluation.t2i_compbench.dataset import CATEGORIES, SEEDS, T2ICompBenchError

    pinned = load_pinned_val_txt(pin_json)

    selected_prompt_rows: list[dict[str, Any]] = []
    prompt_only_rows: list[dict[str, Any]] = []
    independent_seed_rows: list[dict[str, Any]] = []
    selected: dict[str, list[dict[str, Any]]] = {}
    source_hashes: dict[str, str] = {}
    prompt_file_hashes: dict[str, str] = {}
    source_auxiliary_hashes: dict[str, str] = {}

    prompts_dir = output_dir / "prompts"
    for category_index, category in enumerate(CATEGORIES):
        expected_sha256 = str(pinned[category]["sha256"])
        source_path = (
            clone_dir / "examples" / "dataset" / f"{category}_val.txt"
        )
        prompts = read_official_prompts(
            source_path,
            expected_sha256=expected_sha256,
        )
        line_indices = select_line_indices(category_index=category_index)
        category_rows: list[dict[str, Any]] = []
        selected_prompts: list[str] = []
        for line_index in line_indices:
            prompt = prompts[line_index]
            base = prompt_row(category, line_index, prompt, seed=0)
            selected_prompt_rows.append(base)
            selected_prompts.append(prompt)
            category_rows.append(
                {
                    "line_index": line_index,
                    "prompt_id": base["prompt_id"],
                    "prompt": prompt,
                }
            )
            for seed in SEEDS:
                seed_prompt_id = f"{base['prompt_id']}_s{seed}"
                independent_seed_rows.append(
                    {
                        **base,
                        "prompt_id": seed_prompt_id,
                        "seed": seed,
                    }
                )
                prompt_only_rows.append(
                    {
                        **base,
                        "prompt_id": seed_prompt_id,
                    }
                )
        prompt_bytes = ("\n".join(selected_prompts) + "\n").encode("utf-8")
        prompt_name = f"{category}_val.txt"
        prompt_file_hashes[prompt_name] = write_bytes_atomic(
            prompts_dir / prompt_name,
            prompt_bytes,
        )
        selected[category] = category_rows
        source_hashes[category] = expected_sha256

    # The official 3-in-1 evaluator treats the spatial/action files as
    # membership sets, not as 300-row arrays aligned to complex_val.txt.
    # Preserve the complete pinned-clone files byte-for-byte.
    official_dataset_dir = clone_dir / "examples" / "dataset"
    for name in COMPLEX_AUXILIARY_FILES:
        source_path = official_dataset_dir / name
        if not source_path.is_file():
            raise T2ICompBenchError(f"official complex auxiliary file not found: {source_path}")
        payload = source_path.read_bytes()
        if not payload:
            raise T2ICompBenchError(f"official complex auxiliary file is empty: {source_path}")
        digest = write_bytes_atomic(prompts_dir / name, payload)
        prompt_file_hashes[name] = digest
        source_auxiliary_hashes[name] = digest

    selected_prompt_payload = jsonl_bytes(selected_prompt_rows)
    prompt_only_payload = jsonl_bytes(prompt_only_rows)
    generation_payload = jsonl_bytes(independent_seed_rows)
    selected_prompt_name = "selected_prompt_manifest.jsonl"
    prompt_only_name = "prompt_only.jsonl"
    generation_name = "independent_seed_manifest.jsonl"
    selected_prompt_sha256 = write_bytes_atomic(
        output_dir / selected_prompt_name,
        selected_prompt_payload,
    )
    prompt_only_sha256 = write_bytes_atomic(
        output_dir / prompt_only_name,
        prompt_only_payload,
    )
    generation_sha256 = write_bytes_atomic(
        output_dir / generation_name,
        generation_payload,
    )

    manifest = {
        "schema": "holsd.t2icb_subset.v1",
        "selection_policy": SELECTION_POLICY,
        "sample_seed": SAMPLE_SEED,
        "categories": list(CATEGORIES),
        "prompts_per_category": PROMPTS_PER_CATEGORY,
        "prompt_count": len(selected_prompt_rows),
        "seeds": list(SEEDS),
        "images_per_prompt": len(SEEDS),
        "generation_row_count": len(independent_seed_rows),
        "selected": selected,
        "source_txt_sha256": source_hashes,
        "source_complex_auxiliary_sha256": source_auxiliary_hashes,
        "prompt_files_sha256": prompt_file_hashes,
        "selected_prompt_manifest": {
            "path": selected_prompt_name,
            "sha256": selected_prompt_sha256,
            "rows": len(selected_prompt_rows),
        },
        "prompt_only_manifest": {
            "path": prompt_only_name,
            "sha256": prompt_only_sha256,
            "rows": len(prompt_only_rows),
        },
        "independent_seed_manifest": {
            "path": generation_name,
            "sha256": generation_sha256,
            "rows": len(independent_seed_rows),
        },
    }
    write_bytes_atomic(output_dir / "selection_manifest.json", canonical_json_bytes(manifest))
    print(
        f"wrote {len(selected_prompt_rows)} prompts and "
        f"{len(independent_seed_rows)} independent-seed rows to {output_dir}",
        flush=True,
    )


def check(clone_dir: Path, pin_json: Path) -> dict[str, Any]:
    """Rebuild in a temporary directory and compare every file with the frozen dataset."""
    from ccg_diffusion.evaluation.t2i_compbench.dataset import DATASET_DIR, EXPECTED_SELECTION_SHA256

    with tempfile.TemporaryDirectory() as tmp:
        build(clone_dir, pin_json, Path(tmp))
        built = {p.relative_to(tmp).as_posix(): sha256_bytes(p.read_bytes()) for p in Path(tmp).rglob("*")
                 if p.is_file()}
    frozen = {p.relative_to(DATASET_DIR).as_posix(): sha256_bytes(p.read_bytes()) for p in DATASET_DIR.rglob("*")
              if p.is_file() and p.name != "dataset_contract.json"}
    return {"selection_manifest_sha256": built.get("selection_manifest.json"),
            "expected_selection_sha256": EXPECTED_SELECTION_SHA256, "files": len(built),
            "differing_files": sorted(name for name in set(built) | set(frozen) if built.get(name) != frozen.get(name)),
            "equal": built == frozen and built.get("selection_manifest.json") == EXPECTED_SELECTION_SHA256}


def main(argv: Sequence[str] | None = None) -> int:
    from ccg_diffusion.evaluation.t2i_compbench.dataset import DATASET_DIR, REPO_ROOT, SOURCE_PIN_PATH

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clone-dir", type=Path, default=REPO_ROOT / "external/t2i_compbench/T2I-CompBench")
    parser.add_argument("--pin-json", type=Path, default=REPO_ROOT / SOURCE_PIN_PATH)
    parser.add_argument("--output-dir", type=Path, help="where to build (not the frozen dataset directory)")
    parser.add_argument("--check", action="store_true", help="rebuild in a temporary directory and compare")
    args = parser.parse_args(argv)
    if args.check:
        result = check(args.clone_dir, args.pin_json)
        print(json.dumps(result, indent=1), flush=True)
        return 0 if result["equal"] else 1
    if args.output_dir is None:
        parser.error("--output-dir is required without --check")
    if args.output_dir.resolve() == DATASET_DIR.resolve():
        parser.error("refusing to overwrite the frozen dataset; use --check to compare with it")
    build(args.clone_dir, args.pin_json, args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
