"""Fetch and freeze the official TAB syntax-PP benchmark.

The source is pinned to the commit cited by the ACL 2023 TAB release.  This
script deliberately performs no linguistic rewriting: the ambiguous prompt,
visual setups, commonsense labels, and VQA questions are copied verbatim from
the official CSV into a machine-readable JSONL specification.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
TAB_REPOSITORY = "https://github.com/amazon-science/tab-dataset"
TAB_COMMIT = "2e9d70c3c44037ef7e6a2c96bf8efb31bd6aed0e"
TAB_PP_RELATIVE_PATH = "benchmark/data/syntax-pp.csv"
TAB_LICENSE_RELATIVE_PATH = "LICENSE.md"
TAB_PP_SHA256 = "2979c282212c6d343988d13d96647a58da3cecfb83e2780334471423862ca5f5"
TAB_LICENSE_SHA256 = "b8019c123e9f9b6c3c677f1b37ac0cc0a652535a5c15ec68ba324c09e476f5d4"
TAB_LICENSE = "CC BY-SA 4.0"
EXPECTED_PP_PROMPTS = 74
EXPECTED_VISUAL_SETUPS_PER_PROMPT = 2
DEFAULT_OUTPUT_DIR = REPO_ROOT / "dataset/tab/official/amazon-science-tab-dataset-2e9d70c3"


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of *path*."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_url(relative_path: str) -> str:
    """Build a raw GitHub URL pinned to the official commit."""
    return f"https://raw.githubusercontent.com/amazon-science/tab-dataset/{TAB_COMMIT}/{relative_path}"


def download_verified(url: str, destination: Path, expected_sha256: str) -> None:
    """Download one source file atomically and verify its pinned digest."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
        with urllib.request.urlopen(url) as response:
            shutil.copyfileobj(response, temporary)
    try:
        actual = sha256_file(temporary_path)
        if actual != expected_sha256:
            raise ValueError(
                f"source hash mismatch for {url}: expected {expected_sha256}, found {actual}"
            )
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def parse_official_list(raw: str, *, field: str, line_number: int) -> list[str]:
    """Parse TAB's unquoted bracket-list serialization without rewriting it."""
    if not raw.startswith("[") or not raw.endswith("]"):
        raise ValueError(f"line {line_number} {field} is not a TAB bracket-list: {raw!r}")
    values = raw[1:-1].split(",")
    if any(value == "" for value in values):
        raise ValueError(f"line {line_number} {field} contains an empty element")
    # The official syntax-PP file has no comma inside an element.  Strip only
    # the delimiter-adjacent space introduced by its list serialization.
    return [value.strip() for value in values]


def load_official_pp(csv_path: Path) -> list[dict[str, Any]]:
    """Load and strictly validate the 74-row official syntax-PP benchmark."""
    if sha256_file(csv_path) != TAB_PP_SHA256:
        raise ValueError(f"{csv_path} is not the pinned official syntax-pp.csv")
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
            if not (len(setups) == len(labels) == len(questions) == EXPECTED_VISUAL_SETUPS_PER_PROMPT):
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
                    "schema_version": "tab-pp-spec-v1",
                    "tab_id": f"tab-pp-{row_index:03d}",
                    "tab_category": "Syntax Prepositional Phrase (PP)",
                    "source_row_index": row_index,
                    "source_csv_line": line_number,
                    "ambiguous_prompt": source["Example"],
                    "targets": targets,
                    "official_split": None,
                }
            )
    if len(rows) != EXPECTED_PP_PROMPTS:
        raise ValueError(f"expected {EXPECTED_PP_PROMPTS} TAB PP prompts, found {len(rows)}")
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def prepare(output_dir: Path, *, no_download: bool = False) -> dict[str, Any]:
    """Fetch, verify, and derive the immutable PP evaluation specification."""
    source_dir = output_dir / "source"
    csv_path = source_dir / "syntax-pp.csv"
    license_path = source_dir / "LICENSE.md"
    if not no_download:
        download_verified(raw_url(TAB_PP_RELATIVE_PATH), csv_path, TAB_PP_SHA256)
        download_verified(raw_url(TAB_LICENSE_RELATIVE_PATH), license_path, TAB_LICENSE_SHA256)
    if not csv_path.is_file() or not license_path.is_file():
        raise FileNotFoundError("pinned TAB source files are absent; rerun without --no-download")
    if sha256_file(license_path) != TAB_LICENSE_SHA256:
        raise ValueError("TAB license file hash does not match the pinned source")

    rows = load_official_pp(csv_path)
    spec_path = output_dir / "tab_pp_spec.jsonl"
    write_jsonl(spec_path, rows)
    manifest = {
        "schema_version": "tab-source-manifest-v1",
        "dataset": "Text-to-image Ambiguity Benchmark (TAB)",
        "benchmark_slice": "main/syntax-pp",
        "source_repository": TAB_REPOSITORY,
        "source_commit": TAB_COMMIT,
        "source_csv_relative_path": TAB_PP_RELATIVE_PATH,
        "source_csv_url": raw_url(TAB_PP_RELATIVE_PATH),
        "source_csv_sha256": TAB_PP_SHA256,
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
    write_json(output_dir / "source_manifest.json", manifest)
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
