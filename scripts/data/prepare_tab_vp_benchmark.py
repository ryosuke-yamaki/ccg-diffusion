"""Fetch and freeze the official TAB syntax-VP benchmark.

The source is pinned to the same official commit as the syntax-PP slice
(``prepare_tab_pp_benchmark.py``).  This script deliberately performs no
linguistic rewriting: the ambiguous prompt, visual setups, commonsense labels,
and VQA questions are copied verbatim from the official CSV into a
machine-readable JSONL specification.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data.prepare_tab_pp_benchmark import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    TAB_COMMIT,
    TAB_LICENSE,
    TAB_LICENSE_RELATIVE_PATH,
    TAB_LICENSE_SHA256,
    TAB_REPOSITORY,
    download_verified,
    parse_official_list,
    raw_url,
    sha256_file,
    write_json,
    write_jsonl,
)

TAB_VP_RELATIVE_PATH = "benchmark/data/syntax-vp.csv"
TAB_VP_SHA256 = "e22909d54669372e31a8c06ce0cc2ca54a71c9cef50ab8a04aedaef1a87ec1ee"
EXPECTED_VP_PROMPTS = 243
EXPECTED_VISUAL_SETUPS_PER_PROMPT = 2
VP_SPEC_SCHEMA_VERSION = "tab-vp-spec-v1"
VP_TAB_CATEGORY = "Syntax Verb Phrase (VP)"
VP_TAB_ID_PREFIX = "tab-vp"
VP_SPEC_FILENAME = "tab_vp_spec.jsonl"
VP_MANIFEST_FILENAME = "source_manifest_vp.json"


def derive_official_spec_rows(
    csv_path: Path,
    *,
    schema_version: str,
    tab_category: str,
    tab_id_prefix: str,
    expected_prompts: int,
) -> list[dict[str, Any]]:
    """Derive verbatim spec rows from one official TAB category CSV.

    Parameterized twin of ``prepare_tab_pp_benchmark.load_official_pp`` (same
    validation, same row structure) so the VP slice and future re-derivation
    checks share one implementation.  No hash gate here; pinned callers verify
    the source digest before calling.
    """
    rows: list[dict[str, Any]] = []
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["Example", "Visual-setups", "CS-UCS", "questions"]:
            raise ValueError(f"unexpected TAB schema: {reader.fieldnames!r}")
        for row_index, source in enumerate(reader):
            line_number = row_index + 2
            setups = parse_official_list(
                source["Visual-setups"], field="Visual-setups", line_number=line_number
            )
            labels = parse_official_list(
                source["CS-UCS"], field="CS-UCS", line_number=line_number
            )
            questions = parse_official_list(
                source["questions"], field="questions", line_number=line_number
            )
            if not (
                len(setups)
                == len(labels)
                == len(questions)
                == EXPECTED_VISUAL_SETUPS_PER_PROMPT
            ):
                raise ValueError(
                    f"line {line_number} does not contain exactly two aligned interpretations"
                )
            if any(label not in {"CS", "UCS"} for label in labels):
                raise ValueError(f"line {line_number} contains an unknown CS-UCS label")
            targets = [
                {
                    "setup_index": index,
                    "visual_setup": setups[index],
                    "commonsense_label": labels[index],
                    "question": questions[index],
                }
                for index in range(EXPECTED_VISUAL_SETUPS_PER_PROMPT)
            ]
            rows.append(
                {
                    "schema_version": schema_version,
                    "tab_id": f"{tab_id_prefix}-{row_index:03d}",
                    "tab_category": tab_category,
                    "source_row_index": row_index,
                    "source_csv_line": line_number,
                    "ambiguous_prompt": source["Example"],
                    "targets": targets,
                    "official_split": None,
                }
            )
    if len(rows) != expected_prompts:
        raise ValueError(
            f"expected {expected_prompts} TAB prompts in {csv_path.name}, found {len(rows)}"
        )
    return rows


def load_official_vp(csv_path: Path) -> list[dict[str, Any]]:
    """Load and strictly validate the 243-row official syntax-VP benchmark."""
    if sha256_file(csv_path) != TAB_VP_SHA256:
        raise ValueError(f"{csv_path} is not the pinned official syntax-vp.csv")
    return derive_official_spec_rows(
        csv_path,
        schema_version=VP_SPEC_SCHEMA_VERSION,
        tab_category=VP_TAB_CATEGORY,
        tab_id_prefix=VP_TAB_ID_PREFIX,
        expected_prompts=EXPECTED_VP_PROMPTS,
    )


def prepare(output_dir: Path, *, no_download: bool = False) -> dict[str, Any]:
    """Fetch, verify, and derive the immutable VP evaluation specification."""
    source_dir = output_dir / "source"
    csv_path = source_dir / "syntax-vp.csv"
    license_path = source_dir / "LICENSE.md"
    if not no_download:
        download_verified(raw_url(TAB_VP_RELATIVE_PATH), csv_path, TAB_VP_SHA256)
        download_verified(raw_url(TAB_LICENSE_RELATIVE_PATH), license_path, TAB_LICENSE_SHA256)
    if not csv_path.is_file() or not license_path.is_file():
        raise FileNotFoundError("pinned TAB source files are absent; rerun without --no-download")
    if sha256_file(license_path) != TAB_LICENSE_SHA256:
        raise ValueError("TAB license file hash does not match the pinned source")

    rows = load_official_vp(csv_path)
    spec_path = output_dir / VP_SPEC_FILENAME
    write_jsonl(spec_path, rows)
    manifest = {
        "schema_version": "tab-source-manifest-v1",
        "dataset": "Text-to-image Ambiguity Benchmark (TAB)",
        "benchmark_slice": "main/syntax-vp",
        "source_repository": TAB_REPOSITORY,
        "source_commit": TAB_COMMIT,
        "source_csv_relative_path": TAB_VP_RELATIVE_PATH,
        "source_csv_url": raw_url(TAB_VP_RELATIVE_PATH),
        "source_csv_sha256": TAB_VP_SHA256,
        "source_license_relative_path": TAB_LICENSE_RELATIVE_PATH,
        "source_license_url": raw_url(TAB_LICENSE_RELATIVE_PATH),
        "source_license_sha256": TAB_LICENSE_SHA256,
        "license": TAB_LICENSE,
        "official_prompt_count": len(rows),
        "official_visual_setup_count": sum(len(row["targets"]) for row in rows),
        "official_split": None,
        "split_note": (
            "The official release supplies one benchmark table and category files, not "
            "train/dev/test assignments. Category files are not dataset splits."
        ),
        "derived_spec": str(spec_path.resolve()),
        "derived_spec_sha256": sha256_file(spec_path),
        "derivation": "verbatim field extraction; no prompt/setup/question rewriting",
    }
    write_json(output_dir / VP_MANIFEST_FILENAME, manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="verify and rebuild from already downloaded pinned source files",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    manifest = prepare(args.output_dir.resolve(), no_download=args.no_download)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
