#!/usr/bin/env python3
"""Verify the frozen ``tab-ppvp-disambiguation`` dataset against its contract."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data.prepare_tab_pp_benchmark import (  # noqa: E402
    TAB_PP_SHA256,
    load_official_pp,
    sha256_file,
)
from scripts.data.prepare_tab_vp_benchmark import (  # noqa: E402
    TAB_VP_SHA256,
    load_official_vp,
)
from scripts.data.prepare_tab_ppvp_disambiguation_dataset import (  # noqa: E402
    CANONICAL_RELATIVE_ROOT,
    CONTRACT_FILENAME,
    CONTRACT_SCHEMA,
    DATASET_ID,
    DATASET_VERSION,
    DEFAULT_DATASET_DIR,
    ITEMS_FILENAME,
    PROMPT_ONLY_FILENAME,
    SEED_MANIFEST_FILENAME,
    build_prompt_only,
    build_seed_manifest,
    category_stats,
    clip_token_check,
    select_items,
    serialize_jsonl,
)


class TabPpvpDatasetError(RuntimeError):
    """The frozen dataset differs from its committed contract."""


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise TabPpvpDatasetError(f"required JSON file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TabPpvpDatasetError(f"invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise TabPpvpDatasetError(f"expected a JSON object: {path}")
    return payload


def require_equal(actual: Any, expected: Any, *, label: str) -> None:
    if actual != expected:
        raise TabPpvpDatasetError(
            f"{label} differs from the frozen contract: {actual!r} != {expected!r}"
        )


def require_file_bytes(path: Path, expected_text: str, *, label: str) -> None:
    if not path.is_file():
        raise TabPpvpDatasetError(f"{label} file not found: {path}")
    if path.read_text(encoding="utf-8") != expected_text:
        raise TabPpvpDatasetError(f"{label} differs from its frozen re-derivation: {path}")


def _verify_sources(contract: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    sources = contract["sources"]
    pp_csv = REPO_ROOT / sources["syntax_pp_csv"]["path"]
    vp_csv = REPO_ROOT / sources["syntax_vp_csv"]["path"]
    require_equal(sources["syntax_pp_csv"]["sha256"], TAB_PP_SHA256, label="pp csv pin")
    require_equal(sources["syntax_vp_csv"]["sha256"], TAB_VP_SHA256, label="vp csv pin")
    require_equal(sha256_file(pp_csv), TAB_PP_SHA256, label="pp csv sha256")
    require_equal(sha256_file(vp_csv), TAB_VP_SHA256, label="vp csv sha256")

    pp_rows = load_official_pp(pp_csv)
    vp_rows = load_official_vp(vp_csv)
    for slice_key, rows in (("tab_pp_spec", pp_rows), ("tab_vp_spec", vp_rows)):
        spec_path = REPO_ROOT / sources[slice_key]["path"]
        require_file_bytes(spec_path, serialize_jsonl(rows), label=slice_key)
        require_equal(
            sha256_file(spec_path), sources[slice_key]["sha256"], label=f"{slice_key} sha256"
        )
    return pp_rows, vp_rows


def _verify_file(
    dataset_dir: Path,
    contract: dict[str, Any],
    key: str,
    filename: str,
    rows: Sequence[dict[str, Any]],
) -> None:
    entry = contract["files"][key]
    require_equal(entry["path"], filename, label=f"files.{key}.path")
    require_equal(entry["rows"], len(rows), label=f"files.{key}.rows")
    path = dataset_dir / filename
    require_file_bytes(path, serialize_jsonl(rows), label=f"files.{key}")
    require_equal(sha256_file(path), entry["sha256"], label=f"files.{key}.sha256")


def verify_dataset(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    *,
    enforce_canonical_path: bool = True,
    with_clip_token_check: bool = False,
) -> dict[str, Any]:
    """Re-derive every dataset file from the pinned sources and compare."""
    dataset_dir = dataset_dir.resolve()
    contract = load_json(dataset_dir / CONTRACT_FILENAME)
    require_equal(contract.get("schema"), CONTRACT_SCHEMA, label="contract schema")
    require_equal(contract.get("dataset_id"), DATASET_ID, label="dataset_id")
    require_equal(contract.get("version"), DATASET_VERSION, label="version")
    require_equal(contract.get("status"), "frozen", label="status")
    require_equal(contract.get("root"), CANONICAL_RELATIVE_ROOT, label="contract root")
    if enforce_canonical_path:
        require_equal(
            dataset_dir,
            (REPO_ROOT / CANONICAL_RELATIVE_ROOT).resolve(),
            label="dataset directory",
        )

    pp_rows, vp_rows = _verify_sources(contract)
    items, exclusions = select_items([*pp_rows, *vp_rows])
    require_equal(
        contract["selection"]["excluded_items"], exclusions, label="selection.excluded_items"
    )

    population = contract["population"]
    require_equal(population["item_count"], len(items), label="population.item_count")
    require_equal(
        population["target_count"],
        sum(len(item["targets"]) for item in items),
        label="population.target_count",
    )
    require_equal(population["categories"], category_stats(items), label="population.categories")

    seed_rows = build_seed_manifest(items)
    require_equal(
        sorted({row["seed"] for row in seed_rows}),
        population["image_seeds"],
        label="population.image_seeds",
    )
    _verify_file(dataset_dir, contract, "items", ITEMS_FILENAME, items)
    _verify_file(dataset_dir, contract, "independent_seed_manifest", SEED_MANIFEST_FILENAME, seed_rows)
    _verify_file(
        dataset_dir, contract, "prompt_only", PROMPT_ONLY_FILENAME, build_prompt_only(seed_rows)
    )

    expected_inventory = sorted(
        [CONTRACT_FILENAME, ITEMS_FILENAME, SEED_MANIFEST_FILENAME, PROMPT_ONLY_FILENAME]
    )
    actual_inventory = sorted(
        path.relative_to(dataset_dir).as_posix()
        for path in dataset_dir.rglob("*")
        if path.is_file()
    )
    require_equal(actual_inventory, expected_inventory, label="dataset file inventory")

    if with_clip_token_check:
        recorded = contract["clip_token_check"]
        if recorded.get("status") != "checked":
            raise TabPpvpDatasetError(
                f"contract records clip_token_check as {recorded.get('status')!r}"
            )
        require_equal(clip_token_check(items), recorded, label="clip_token_check")

    return {
        "schema": CONTRACT_SCHEMA,
        "dataset_id": DATASET_ID,
        "version": DATASET_VERSION,
        "items": len(items),
        "targets": sum(len(item["targets"]) for item in items),
        "excluded_items": len(exclusions),
        "seed_manifest_rows": len(seed_rows),
        "files": len(actual_inventory),
        "clip_token_check": "recomputed" if with_clip_token_check else "recorded_only",
        "status": "verified",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument(
        "--with-clip-token-check",
        action="store_true",
        help="also recompute the CLIP token bound check (loads the pinned tokenizer)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from ccg_diffusion.evaluation import tab

    args = build_parser().parse_args(argv)
    result = verify_dataset(
        args.dataset_dir, with_clip_token_check=args.with_clip_token_check
    )
    # The hash-level runtime check of the judges must agree with the full re-derivation.
    if {**result, "clip_token_check": "recorded_only"} != tab.verify_dataset(args.dataset_dir):
        raise TabPpvpDatasetError("full re-derivation and ccg_diffusion.evaluation.tab.verify_dataset disagree")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
