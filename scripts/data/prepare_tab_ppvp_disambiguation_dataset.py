"""Build the frozen ``tab-ppvp-disambiguation`` derived dataset.

Combines the pinned official TAB syntax-PP and syntax-VP specs into one
immutable experiment population: verbatim item rows, the per-item image-seed
manifest (seeds 42-45, one image each), and a dataset contract that records
sources, selection rules, exclusions, and a CLIP token-length bound check for
the disambiguated-prompt concatenation.  No prompt, setup, or question text is
rewritten.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ccg_diffusion import config  # noqa: E402
from scripts.data.prepare_tab_pp_benchmark import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as OFFICIAL_TAB_DIR,
    TAB_COMMIT,
    TAB_LICENSE,
    TAB_PP_RELATIVE_PATH,
    TAB_PP_SHA256,
    TAB_REPOSITORY,
    load_official_pp,
    sha256_file,
    write_json,
)
from scripts.data.prepare_tab_vp_benchmark import (  # noqa: E402
    TAB_VP_RELATIVE_PATH,
    TAB_VP_SHA256,
    VP_SPEC_FILENAME,
    load_official_vp,
)

DATASET_ID = config.TAB_DATASET_ID
DATASET_VERSION = config.TAB_DATASET_VERSION
CONTRACT_SCHEMA = "holsd.tab_ppvp_disambiguation_contract.v1"
CANONICAL_RELATIVE_ROOT = config.TAB_DATASET_DIR
DEFAULT_DATASET_DIR = REPO_ROOT / CANONICAL_RELATIVE_ROOT
CREATED_AT = "2026-08-17"

EXPERIMENT_ID = "tab_ppvp"
IMAGE_SEEDS = (42, 43, 44, 45)
CLIP_MAX_TOKENS = 77
# Upper-bound form only: the VS-TIED + Gold-visual setup baseline
# (``ccg_diffusion.baselines.gold_visual_setup``) defines its own concatenation;
# this template exists so the freeze can prove that even a period-joined
# concatenation stays inside the CLIP content window.
CONCATENATION_TEMPLATE = "{prompt}. {setup}."

PP_SPEC_FILENAME = "tab_pp_spec.jsonl"
ITEMS_FILENAME = "items.jsonl"
SEED_MANIFEST_FILENAME = "independent_seed_manifest.jsonl"
PROMPT_ONLY_FILENAME = "prompt_only.jsonl"
CONTRACT_FILENAME = "dataset_contract.json"

SELECTION_RULES = (
    "exactly_two_aligned_targets_per_item",
    "drop_exact_duplicate_rows_keep_first_occurrence",
    "drop_prompt_collisions_with_different_targets_keep_first_occurrence",
)


class TabPpvpDatasetBuildError(RuntimeError):
    """The derived dataset cannot be built from the pinned official sources."""


def serialize_jsonl(rows: Sequence[dict[str, Any]]) -> str:
    """Serialize rows exactly like ``prepare_tab_pp_benchmark.write_jsonl``."""
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )


def write_jsonl_file(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialize_jsonl(rows), encoding="utf-8")


def require_spec_matches(
    spec_path: Path, rows: Sequence[dict[str, Any]], *, label: str
) -> None:
    """Require the on-disk spec to be byte-identical to its re-derivation."""
    if not spec_path.is_file():
        raise TabPpvpDatasetBuildError(f"{label} spec is absent: {spec_path}")
    if spec_path.read_text(encoding="utf-8") != serialize_jsonl(rows):
        raise TabPpvpDatasetBuildError(
            f"{label} spec differs from its pinned re-derivation: {spec_path}"
        )


def select_items(
    rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply the frozen selection rules and report every exclusion.

    Duplicate handling exists because the guidance-target files are keyed on
    the prompt text (``ccg_diffusion.targets.guidance_targets.expand_to_image_rows``
    joins on ``(experiment, sha256(prompt))``): two items sharing one prompt
    cannot coexist.
    """
    items: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    first_by_row: dict[tuple[str, str], str] = {}
    first_by_prompt: dict[str, str] = {}
    for row in rows:
        prompt = row["ambiguous_prompt"]
        row_key = (prompt, json.dumps(row["targets"], ensure_ascii=False, sort_keys=True))
        if row_key in first_by_row:
            exclusions.append(
                {
                    "tab_id": row["tab_id"],
                    "reason": "exact_duplicate",
                    "duplicate_of": first_by_row[row_key],
                }
            )
            continue
        if prompt in first_by_prompt:
            exclusions.append(
                {
                    "tab_id": row["tab_id"],
                    "reason": "prompt_collision_different_targets",
                    "collides_with": first_by_prompt[prompt],
                }
            )
            continue
        first_by_row[row_key] = row["tab_id"]
        first_by_prompt[prompt] = row["tab_id"]
        items.append(row)
    return items, exclusions


def build_seed_manifest(items: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per (item, seed), mirroring the T2I-CompBench++ 4-field manifest schema."""
    return [
        {
            "experiment": EXPERIMENT_ID,
            "prompt": item["ambiguous_prompt"],
            "prompt_id": f"{item['tab_id']}_s{seed}",
            "seed": seed,
        }
        for item in items
        for seed in IMAGE_SEEDS
    ]


def build_prompt_only(seed_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Blind Stage 1-3 input: the seed manifest rows with the seed zeroed."""
    return [dict(row, seed=0) for row in seed_rows]


def category_stats(items: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for item in items:
        entry = stats.setdefault(
            item["tab_category"],
            {"items": 0, "targets": 0, "commonsense_labels": {"CS": 0, "UCS": 0}},
        )
        entry["items"] += 1
        entry["targets"] += len(item["targets"])
        for target in item["targets"]:
            entry["commonsense_labels"][target["commonsense_label"]] += 1
    return stats


def hyphenated_participle_item_ids(items: Sequence[dict[str, Any]]) -> list[str]:
    """Items whose only participle-like token is hyphenated (detector risk).

    A Stage-1 participle detector that matches whole tokens ending in ``-ing``
    misses forms like ``picking-up``.  Recorded for preflight expectations
    only — the rows stay in the dataset.
    """
    flagged: list[str] = []
    for item in items:
        tokens = [token.strip(".,").lower() for token in item["ambiguous_prompt"].split()]
        plain = [token for token in tokens if len(token) > 3 and token.endswith("ing")]
        hyphenated = [
            token
            for token in tokens
            if "-" in token
            and any(part.endswith("ing") and len(part) > 3 for part in token.split("-"))
        ]
        if hyphenated and not plain:
            flagged.append(item["tab_id"])
    return flagged


def clip_token_check(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Bound-check ambiguous and concatenated prompts against the CLIP window."""
    from transformers import CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(
        config.SD14_GENERATION.model_id,
        subfolder=config.SD14_GENERATION.tokenizer_subfolder,
        revision=config.SD14_GENERATION.model_revision,
    )

    def token_count(text: str) -> int:
        return len(tokenizer(text).input_ids)

    ambiguous_max = max(token_count(item["ambiguous_prompt"]) for item in items)
    concatenated_max = max(
        token_count(
            CONCATENATION_TEMPLATE.format(
                prompt=item["ambiguous_prompt"], setup=target["visual_setup"]
            )
        )
        for item in items
        for target in item["targets"]
    )
    payload = {
        "status": "checked",
        "tokenizer": {
            "model_id": config.SD14_GENERATION.model_id,
            "revision": config.SD14_GENERATION.model_revision,
            "subfolder": config.SD14_GENERATION.tokenizer_subfolder,
        },
        "max_length_limit": CLIP_MAX_TOKENS,
        "concatenation_template": CONCATENATION_TEMPLATE,
        "ambiguous_prompt_max_tokens": ambiguous_max,
        "concatenated_max_tokens": concatenated_max,
        "all_within_limit": ambiguous_max <= CLIP_MAX_TOKENS
        and concatenated_max <= CLIP_MAX_TOKENS,
    }
    if not payload["all_within_limit"]:
        raise TabPpvpDatasetBuildError(
            f"CLIP token bound exceeded: {payload!r}"
        )
    return payload


def build_dataset(
    output_dir: Path = DEFAULT_DATASET_DIR,
    *,
    official_dir: Path = OFFICIAL_TAB_DIR,
    skip_clip_token_check: bool = False,
) -> dict[str, Any]:
    """Build every dataset file and return the written contract."""
    pp_csv = official_dir / "source" / "syntax-pp.csv"
    vp_csv = official_dir / "source" / "syntax-vp.csv"
    pp_spec = official_dir / PP_SPEC_FILENAME
    vp_spec = official_dir / VP_SPEC_FILENAME

    pp_rows = load_official_pp(pp_csv)
    vp_rows = load_official_vp(vp_csv)
    require_spec_matches(pp_spec, pp_rows, label="syntax-PP")
    require_spec_matches(vp_spec, vp_rows, label="syntax-VP")

    items, exclusions = select_items([*pp_rows, *vp_rows])
    seed_rows = build_seed_manifest(items)
    prompt_only_rows = build_prompt_only(seed_rows)
    token_check = (
        {"status": "skipped", "reason": "--skip-clip-token-check"}
        if skip_clip_token_check
        else clip_token_check(items)
    )

    items_path = output_dir / ITEMS_FILENAME
    seed_path = output_dir / SEED_MANIFEST_FILENAME
    prompt_only_path = output_dir / PROMPT_ONLY_FILENAME
    write_jsonl_file(items_path, items)
    write_jsonl_file(seed_path, seed_rows)
    write_jsonl_file(prompt_only_path, prompt_only_rows)

    contract = {
        "schema": CONTRACT_SCHEMA,
        "dataset_id": DATASET_ID,
        "version": DATASET_VERSION,
        "status": "frozen",
        "root": CANONICAL_RELATIVE_ROOT,
        "created_at": CREATED_AT,
        "purpose": (
            "Fixed TAB syntax-PP/VP population for the SD1.4 syntactic-ambiguity "
            "disambiguation experiment (vanilla / tied-oracle / derivation-conditioned "
            "PD arms) under the ACL 2023 TAB faithful-image VQA protocol."
        ),
        "population": {
            "experiment_id": EXPERIMENT_ID,
            "categories": category_stats(items),
            "item_count": len(items),
            "target_count": sum(len(item["targets"]) for item in items),
            "targets_per_item": 2,
            "image_seeds": list(IMAGE_SEEDS),
            "images_per_item_per_condition": len(IMAGE_SEEDS),
        },
        "sources": {
            "snapshot_root": str(official_dir.relative_to(REPO_ROOT)),
            "source_repository": TAB_REPOSITORY,
            "source_commit": TAB_COMMIT,
            "license": TAB_LICENSE,
            "syntax_pp_csv": {
                "official_relative_path": TAB_PP_RELATIVE_PATH,
                "path": str(pp_csv.relative_to(REPO_ROOT)),
                "sha256": TAB_PP_SHA256,
            },
            "syntax_vp_csv": {
                "official_relative_path": TAB_VP_RELATIVE_PATH,
                "path": str(vp_csv.relative_to(REPO_ROOT)),
                "sha256": TAB_VP_SHA256,
            },
            "tab_pp_spec": {
                "schema_version": "tab-pp-spec-v1",
                "path": str(pp_spec.relative_to(REPO_ROOT)),
                "sha256": sha256_file(pp_spec),
            },
            "tab_vp_spec": {
                "schema_version": "tab-vp-spec-v1",
                "path": str(vp_spec.relative_to(REPO_ROOT)),
                "sha256": sha256_file(vp_spec),
            },
        },
        "selection": {
            "rules": list(SELECTION_RULES),
            "excluded_items": exclusions,
        },
        "files": {
            "items": {
                "path": ITEMS_FILENAME,
                "rows": len(items),
                "sha256": sha256_file(items_path),
            },
            "independent_seed_manifest": {
                "path": SEED_MANIFEST_FILENAME,
                "rows": len(seed_rows),
                "sha256": sha256_file(seed_path),
            },
            "prompt_only": {
                "path": PROMPT_ONLY_FILENAME,
                "rows": len(prompt_only_rows),
                "sha256": sha256_file(prompt_only_path),
            },
        },
        "clip_token_check": token_check,
        "notes": [
            "Item rows are verbatim copies of the pinned official spec rows; no text is rewritten.",
            (
                "Hyphenated-participle items (Stage-1 participle detector matches plain "
                "-ing tokens only): "
                + json.dumps(hyphenated_participle_item_ids(items))
            ),
            (
                "Parse-dependent gates (forced-parse realizability, index differentiation) "
                "are applied downstream at endpoint freeze; they reference this dataset id "
                "and never rewrite it."
            ),
        ],
        "usage_rules": [
            (
                "Arm comparisons are paired on identical (item, seed); seeds 42-45 are "
                "used exactly once per item per condition."
            ),
            (
                "The faithful-image metric follows the ACL 2023 TAB protocol "
                "(scripts/analysis/run_tab_vqa.py); the opposite-setup question is "
                "diagnostic only."
            ),
            (
                "Judge comparisons stay inside one judge series; no absolute-accuracy "
                "comparison across judges."
            ),
            "Report per tab_category; do not average PP and VP into one score.",
            (
                "Any prompt/target/seed change requires a new dataset id or version; "
                "never overwrite this directory."
            ),
        ],
    }
    write_json(output_dir / CONTRACT_FILENAME, contract)
    return contract


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--official-dir", type=Path, default=OFFICIAL_TAB_DIR)
    parser.add_argument(
        "--skip-clip-token-check",
        action="store_true",
        help="record the token bound check as skipped (offline rebuild)",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    contract = build_dataset(
        args.output_dir.resolve(),
        official_dir=args.official_dir.resolve(),
        skip_clip_token_check=args.skip_clip_token_check,
    )
    print(json.dumps(contract, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
