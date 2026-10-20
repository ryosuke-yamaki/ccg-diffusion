"""TAB PP evaluation: dataset, VQA prompt and answer adapter, trial plans, scoring and judgments.

Sections, in order:

* the frozen ``tab-ppvp-disambiguation`` dataset (``verify_dataset`` checks
  the contract and every file hash without re-deriving from the official CSV);
* the VQA adapter shared by both judges: the official intention question plus
  ``Answer with only Yes or No.``, and the leading standalone Yes/No answer;
* image coverage validation and trial plans of a generation run (one trial =
  one image + one official question; ``write_trial_plan`` writes the
  ``trial_plan.jsonl`` the judges read);
* panels and Yes-rate scoring (per item: both interpretations x seeds);
* the compact per-trial judgments of the paper's TAB PP conditions
  (``output/analysis/20261006_paper_significance/tab_trials``), from which the
  Table 5 kappa column and the Table 5/6 tests are recomputed;
* the deterministic reading -> setup mapping (``np_attach``/``vp_attach`` to
  TAB setup 0/1).
"""
from __future__ import annotations

import gzip
import hashlib
import itertools
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from ccg_diffusion import config
from ccg_diffusion.provenance import (
    canonical_json, immutable_rows, read_jsonl, repository_root, sha256_bytes, sha256_file, sha256_text,
)

REPO_ROOT = repository_root()

# --- dataset -------------------------------------------------------------------------------------------

DATASET_ID = config.TAB_DATASET_ID
DATASET_VERSION = config.TAB_DATASET_VERSION
DATASET_DIR = REPO_ROOT / config.TAB_DATASET_DIR
ITEMS = DATASET_DIR / "items.jsonl"
CONTRACT_SCHEMA = "holsd.tab_ppvp_disambiguation_contract.v1"
CONTRACT_FILENAME = "dataset_contract.json"
DATASET_FILES = (
    ("items", "items.jsonl"),
    ("independent_seed_manifest", "independent_seed_manifest.jsonl"),
    ("prompt_only", "prompt_only.jsonl"),
)
PP_CATEGORY = "Syntax Prepositional Phrase (PP)"
PP_ITEMS = 74
SEEDS = config.TAB_PP_SEEDS


def _require_equal(actual: Any, expected: Any, *, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} differs from the frozen contract: {actual!r} != {expected!r}")


def verify_dataset(dataset_dir: Path = DATASET_DIR) -> dict[str, Any]:
    """Check the frozen dataset contract, every file's sha256 and row count, and the 4-file inventory.

    The full re-derivation from the official CSV files is
    ``scripts/data/verify_tab_ppvp_disambiguation_dataset.py``. The returned
    record is the one the judge runs store as ``dataset_verification``.
    """

    dataset_dir = dataset_dir.resolve()
    contract = json.loads((dataset_dir / CONTRACT_FILENAME).read_text(encoding="utf-8"))
    _require_equal(contract.get("schema"), CONTRACT_SCHEMA, label="contract schema")
    _require_equal(contract.get("dataset_id"), DATASET_ID, label="dataset_id")
    _require_equal(contract.get("version"), DATASET_VERSION, label="version")
    _require_equal(contract.get("status"), "frozen", label="status")
    _require_equal(contract.get("root"), config.TAB_DATASET_DIR, label="contract root")
    for key, filename in DATASET_FILES:
        entry = contract["files"][key]
        _require_equal(entry["path"], filename, label=f"files.{key}.path")
        path = dataset_dir / filename
        _require_equal(len(read_jsonl(path)), entry["rows"], label=f"files.{key}.rows")
        _require_equal(sha256_file(path), entry["sha256"], label=f"files.{key}.sha256")
    inventory = sorted(path.relative_to(dataset_dir).as_posix() for path in dataset_dir.rglob("*") if path.is_file())
    _require_equal(inventory, sorted([CONTRACT_FILENAME, *(name for _, name in DATASET_FILES)]),
                   label="dataset file inventory")
    items = read_jsonl(dataset_dir / "items.jsonl")
    population = contract["population"]
    _require_equal(len(items), population["item_count"], label="population.item_count")
    _require_equal(sum(len(item["targets"]) for item in items), population["target_count"],
                   label="population.target_count")
    return {
        "schema": CONTRACT_SCHEMA,
        "dataset_id": DATASET_ID,
        "version": DATASET_VERSION,
        "items": len(items),
        "targets": sum(len(item["targets"]) for item in items),
        "excluded_items": len(contract["selection"]["excluded_items"]),
        "seed_manifest_rows": contract["files"]["independent_seed_manifest"]["rows"],
        "files": len(inventory),
        "clip_token_check": "recorded_only",
        "status": "verified",
    }


def load_pp_items() -> dict[str, dict[str, Any]]:
    """The 74 PP items of the dataset, by ``tab_id``."""

    items = {row["tab_id"]: row for row in read_jsonl(ITEMS) if row["tab_id"].startswith("tab-pp-")}
    if len(items) != PP_ITEMS:
        raise ValueError(f"Expected all {PP_ITEMS} official PP items")
    return items


# --- VQA prompt and answer adapter -----------------------------------------------------------------------

# ``question`` is inserted byte-for-byte. The suffix is the minimum adapter a
# generative judge needs (the TAB paper's ViLT has a closed VQA answer head).
VQA_PROMPT_TEMPLATE = "{question}\nAnswer with only Yes or No."
PROMPT_TEMPLATE_SHA256 = hashlib.sha256(VQA_PROMPT_TEMPLATE.encode("utf-8")).hexdigest()
_LEADING_ANSWER_RE = re.compile(r"^\s*(yes|no)(?=\b|[.!?])", flags=re.IGNORECASE)


def normalize_yes_no(raw_response: str) -> str | None:
    """Map only a leading standalone Yes/No to the binary answer; anything else is invalid (``None``)."""
    match = _LEADING_ANSWER_RE.search(raw_response)
    return None if match is None else match.group(1).lower()


def render_vqa_prompt(question: str) -> str:
    """Preserve the official question and add only the answer suffix."""
    rendered = VQA_PROMPT_TEMPLATE.format(question=question)
    if not rendered.startswith(question) or rendered[: len(question)] != question:
        raise AssertionError("official TAB question changed while rendering the Qwen adapter")
    return rendered


# --- image coverage and trial plans ----------------------------------------------------------------------


@dataclass(frozen=True)
class TabCoverageProfile:
    """Fixed population and image-panel contract for one TAB evaluation series."""

    name: str
    default_spec: Path
    full_item_count: int
    full_category_counts: Mapping[str, int]
    image_seeds: tuple[int, ...]
    requires_reading: bool

    @property
    def images_per_target_setup(self) -> int:
        return len(self.image_seeds)

    @property
    def images_per_item(self) -> int:
        return 2 * self.images_per_target_setup


TAB_PPVP_5SEED_PROFILE = TabCoverageProfile(
    name="tab-ppvp-5seed",
    default_spec=ITEMS,
    full_item_count=315,
    full_category_counts={
        "Syntax Prepositional Phrase (PP)": 74,
        "Syntax Verb Phrase (VP)": 241,
    },
    image_seeds=(0, 1, 2, 3, 4),
    requires_reading=True,
)
# Names of the 74-item PP restriction recorded in the coverage records of the
# evaluated runs: SD1.4 CCG Diffusion, Sana, and the Predicated Diffusion baselines.
PP_PROFILE_NAMES = ("tab-pp-transfer-74-five-seed", "sana-tab-pp", "tab-pp-pd-baselines")
TARGETED_SETUP_MODE = "targeted_setup"
AMBIGUOUS_BOTH_SETUPS_MODE = "ambiguous_both_setups"
TAB_EVALUATION_MODES = frozenset(
    {TARGETED_SETUP_MODE, AMBIGUOUS_BOTH_SETUPS_MODE}
)


def pp_profile(name: str) -> TabCoverageProfile:
    """The 74-item PP restriction of the 5-seed profile under one of ``PP_PROFILE_NAMES``."""

    if name not in PP_PROFILE_NAMES:
        raise ValueError(f"unknown TAB PP coverage profile {name!r}")
    return replace(TAB_PPVP_5SEED_PROFILE, name=name, full_item_count=74,
                   full_category_counts={PP_CATEGORY: 74})


def _validate_specs(
    specs: Sequence[dict[str, Any]],
    *,
    profile: TabCoverageProfile,
    benchmark_scope: str,
) -> dict[str, dict[str, Any]]:
    """Validate the official TAB rows before they are paired with images."""

    spec_by_id: dict[str, dict[str, Any]] = {}
    for spec in specs:
        tab_id = str(spec.get("tab_id", ""))
        if not tab_id:
            raise ValueError(f"TAB spec row lacks tab_id: {spec!r}")
        if tab_id in spec_by_id:
            raise ValueError(f"duplicate TAB spec id: {tab_id!r}")
        category = spec.get("tab_category")
        if not isinstance(category, str) or not category:
            raise ValueError(f"TAB spec row lacks tab_category: {spec!r}")
        targets = spec.get("targets")
        if not isinstance(targets, list) or len(targets) != 2:
            raise ValueError(f"TAB spec must provide exactly two targets: {tab_id!r}")
        setup_indices = [target.get("setup_index") for target in targets]
        if setup_indices != [0, 1]:
            raise ValueError(
                f"TAB targets must be ordered by setup_index [0, 1]: {tab_id!r} has {setup_indices!r}"
            )
        for target in targets:
            if not isinstance(target.get("question"), str) or not target["question"]:
                raise ValueError(f"TAB target lacks official question: {tab_id!r}")
        spec_by_id[tab_id] = spec

    if benchmark_scope == "full":
        category_counts = Counter(str(row["tab_category"]) for row in spec_by_id.values())
        if len(spec_by_id) != profile.full_item_count or dict(category_counts) != dict(
            profile.full_category_counts
        ):
            raise ValueError(
                f"coverage profile {profile.name!r} requires {profile.full_item_count} TAB rows "
                f"with category counts {dict(profile.full_category_counts)!r}; found "
                f"{len(spec_by_id)} rows and {dict(category_counts)!r}"
            )
    return spec_by_id


def validate_image_coverage(
    metadata: Sequence[dict[str, Any]],
    spec_ids: set[str],
    *,
    benchmark_scope: str,
    profile: TabCoverageProfile,
    spec_by_id: Mapping[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Enforce physical-image and question-pairing populations per condition."""
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    physical_keys: set[tuple[str, str]] = set()
    physical_paths: set[tuple[str, str]] = set()
    for row in metadata:
        required = {"condition", "tab_id", "image_path", "image_id", "seed"}
        missing = required - set(row)
        if missing:
            raise ValueError(f"image metadata is missing fields {sorted(missing)}: {row!r}")
        if row["tab_id"] not in spec_ids:
            raise ValueError(f"unknown TAB id: {row['tab_id']!r}")
        if type(row["seed"]) is not int or row["seed"] not in profile.image_seeds:
            raise ValueError(
                f"seed must be one of {list(profile.image_seeds)!r} for {profile.name}: {row!r}"
            )
        explicit_mode = row.get("tab_evaluation_mode")
        mode = TARGETED_SETUP_MODE if explicit_mode is None else explicit_mode
        if mode not in TAB_EVALUATION_MODES:
            raise ValueError(f"unsupported tab_evaluation_mode: {row!r}")
        reading_required = row.get(
            "tab_requires_reading",
            profile.requires_reading if explicit_mode is None else False,
        )
        if type(reading_required) is not bool:
            raise ValueError(f"tab_requires_reading must be boolean: {row!r}")
        if mode == TARGETED_SETUP_MODE:
            target_setup = row.get("target_setup_index")
            if type(target_setup) is not int or target_setup not in {0, 1}:
                raise ValueError(f"target_setup_index must be 0 or 1: {row!r}")
            evaluation_indices = row.get("evaluation_setup_indices", [target_setup])
            if evaluation_indices != [target_setup]:
                raise ValueError(
                    "targeted_setup requires exactly its intended official question: "
                    f"{row!r}"
                )
        else:
            if row.get("target_setup_index") is not None:
                raise ValueError(
                    "ambiguous_both_setups has no intended target_setup_index: "
                    f"{row!r}"
                )
            if row.get("evaluation_setup_indices") != [0, 1]:
                raise ValueError(
                    "ambiguous_both_setups must explicitly evaluate setups [0, 1]: "
                    f"{row!r}"
                )
            if reading_required:
                raise ValueError("ambiguous vanilla images cannot require a forced reading")
        if reading_required and row.get("reading") not in {"np_attach", "vp_attach"}:
            raise ValueError(
                "reading must be one of ['np_attach', 'vp_attach'] for "
                f"{profile.name}: {row!r}"
            )
        if not reading_required and row.get("reading") is not None:
            raise ValueError(f"sidecar-free TAB condition must not declare a reading: {row!r}")
        key = (str(row["condition"]), str(row["image_id"]))
        if key in physical_keys:
            raise ValueError(f"duplicate physical image key: {key!r}")
        physical_keys.add(key)
        path_key = (str(row["condition"]), str(Path(str(row["image_path"])).resolve()))
        if path_key in physical_paths:
            raise ValueError(f"one physical image path is assigned more than once: {path_key!r}")
        physical_paths.add(path_key)
        by_condition[str(row["condition"])].append(row)

    condition_reports: dict[str, Any] = {}
    for condition, rows in sorted(by_condition.items()):
        modes = {
            TARGETED_SETUP_MODE
            if row.get("tab_evaluation_mode") is None
            else str(row["tab_evaluation_mode"])
            for row in rows
        }
        reading_requirements = {
            bool(
                row.get(
                    "tab_requires_reading",
                    profile.requires_reading
                    if row.get("tab_evaluation_mode") is None
                    else False,
                )
            )
            for row in rows
        }
        if len(modes) != 1 or len(reading_requirements) != 1:
            raise ValueError(
                f"condition {condition!r} mixes TAB evaluation/readings contracts"
            )
        mode = next(iter(modes))
        reading_required = next(iter(reading_requirements))
        counts: Counter[str] = Counter(str(row["tab_id"]) for row in rows)
        observed_ids = set(counts)
        if benchmark_scope == "full" and observed_ids != spec_ids:
            missing = sorted(spec_ids - observed_ids)
            extra = sorted(observed_ids - spec_ids)
            raise ValueError(
                f"condition {condition!r} is not the full {profile.full_item_count}-item "
                f"{profile.name} benchmark; "
                f"missing={missing}, extra={extra}"
            )
        if mode == TARGETED_SETUP_MODE:
            target_counts: Counter[tuple[str, int]] = Counter(
                (str(row["tab_id"]), int(row["target_setup_index"]))
                for row in rows
            )
            expected_images_per_item = profile.images_per_item
            bad_targets = {
                f"{tab_id}:{setup_index}": target_counts[(tab_id, setup_index)]
                for tab_id in observed_ids
                for setup_index in (0, 1)
                if target_counts[(tab_id, setup_index)]
                != profile.images_per_target_setup
            }
            seed_sets: dict[tuple[str, int | None], set[int]] = defaultdict(set)
            for row in rows:
                seed_sets[(str(row["tab_id"]), int(row["target_setup_index"]))].add(
                    int(row["seed"])
                )
        else:
            expected_images_per_item = profile.images_per_target_setup
            bad_targets = {}
            seed_sets = defaultdict(set)
            for row in rows:
                seed_sets[(str(row["tab_id"]), None)].add(int(row["seed"]))
        bad_total = {
            tab_id: count
            for tab_id, count in counts.items()
            if count != expected_images_per_item
        }
        if bad_total or bad_targets:
            if mode == TARGETED_SETUP_MODE:
                raise ValueError(
                    f"condition {condition!r} violates "
                    f"{profile.images_per_target_setup} images per selected target setup: "
                    f"totals={bad_total}, targets={bad_targets}"
                )
            raise ValueError(
                f"condition {condition!r} violates its {mode} physical-image panel: "
                f"totals={bad_total}"
            )
        bad_seed_sets = {
            f"{tab_id}:{'shared' if setup_index is None else setup_index}": sorted(seeds)
            for (tab_id, setup_index), seeds in sorted(
                seed_sets.items(), key=lambda item: (item[0][0], str(item[0][1]))
            )
            if seeds != set(profile.image_seeds)
        }
        if bad_seed_sets:
            raise ValueError(
                f"condition {condition!r} does not use exact seeds {list(profile.image_seeds)!r}: "
                f"{bad_seed_sets}"
            )

        if reading_required:
            reading_counts: Counter[tuple[str, str]] = Counter(
                (str(row["tab_id"]), str(row["reading"])) for row in rows
            )
            reading_seeds: dict[tuple[str, str], set[int]] = defaultdict(set)
            reading_targets: dict[tuple[str, str], set[int]] = defaultdict(set)
            target_readings: dict[tuple[str, int], set[str]] = defaultdict(set)
            for row in rows:
                tab_id = str(row["tab_id"])
                reading = str(row["reading"])
                target_setup = int(row["target_setup_index"])
                reading_seeds[(tab_id, reading)].add(int(row["seed"]))
                reading_targets[(tab_id, reading)].add(target_setup)
                target_readings[(tab_id, target_setup)].add(reading)
            expected_readings = {"np_attach", "vp_attach"}
            bad_readings = {
                tab_id: sorted(
                    reading for item_id, reading in reading_counts if item_id == tab_id
                )
                for tab_id in observed_ids
                if {reading for item_id, reading in reading_counts if item_id == tab_id}
                != expected_readings
            }
            bad_reading_counts = {
                f"{tab_id}:{reading}": count
                for (tab_id, reading), count in reading_counts.items()
                if count != profile.images_per_target_setup
            }
            bad_reading_seeds = {
                f"{tab_id}:{reading}": sorted(seeds)
                for (tab_id, reading), seeds in reading_seeds.items()
                if seeds != set(profile.image_seeds)
            }
            bad_reading_target_mapping = {
                f"{tab_id}:{reading}": sorted(targets)
                for (tab_id, reading), targets in reading_targets.items()
                if len(targets) != 1
            }
            bad_target_reading_mapping = {
                f"{tab_id}:{target}": sorted(readings)
                for (tab_id, target), readings in target_readings.items()
                if len(readings) != 1
            }
            if (
                bad_readings
                or bad_reading_counts
                or bad_reading_seeds
                or bad_reading_target_mapping
                or bad_target_reading_mapping
            ):
                raise ValueError(
                    f"condition {condition!r} does not provide one 5-seed panel for each "
                    "reading and a one-to-one reading/setup mapping: "
                    f"readings={bad_readings}, counts={bad_reading_counts}, "
                    f"seeds={bad_reading_seeds}, reading_targets={bad_reading_target_mapping}, "
                    f"target_readings={bad_target_reading_mapping}"
                )

        category_images: dict[str, int] = {}
        if spec_by_id is not None:
            category_images = dict(
                Counter(str(spec_by_id[str(row["tab_id"])]["tab_category"]) for row in rows)
            )
        condition_reports[condition] = {
            "prompts": len(observed_ids),
            "images": len(rows),
            "physical_images_per_prompt": expected_images_per_item,
            "primary_trials": (
                len(rows) * 2 if mode == AMBIGUOUS_BOTH_SETUPS_MODE else len(rows)
            ),
            "primary_trials_per_prompt": profile.images_per_item,
            "evaluation_mode": mode,
            "requires_reading": reading_required,
            "images_per_prompt_with_both_targets": expected_images_per_item,
            "images_per_target_setup": (
                profile.images_per_target_setup
                if mode == TARGETED_SETUP_MODE
                else None
            ),
            "image_seeds": list(profile.image_seeds),
            "category_images": category_images,
        }
    if not condition_reports:
        raise ValueError("image metadata is empty")
    return {
        "benchmark_scope": benchmark_scope,
        "coverage_profile": profile.name,
        "conditions": condition_reports,
    }


def build_trial_plan(
    metadata: Sequence[dict[str, Any]],
    specs: Sequence[dict[str, Any]],
    *,
    benchmark_scope: str,
    include_opposite_diagnostic: bool,
    profile: TabCoverageProfile,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """One trial per generated image and official question, sorted by condition, item and image.

    Targeted images get the question of their intended setup (and, with
    ``include_opposite_diagnostic``, the other one as a diagnostic); ambiguous
    images (Vanilla, Predicated Diffusion) get both questions as targets.
    """
    spec_by_id = _validate_specs(specs, profile=profile, benchmark_scope=benchmark_scope)
    coverage = validate_image_coverage(
        metadata,
        set(spec_by_id),
        benchmark_scope=benchmark_scope,
        profile=profile,
        spec_by_id=spec_by_id,
    )
    trials: list[dict[str, Any]] = []
    for image in metadata:
        spec = spec_by_id[str(image["tab_id"])]
        evaluation_mode = str(
            image.get("tab_evaluation_mode", TARGETED_SETUP_MODE)
        )
        generation_target_index = image.get("target_setup_index")
        if evaluation_mode == AMBIGUOUS_BOTH_SETUPS_MODE:
            roles = [("target", 0), ("target", 1)]
        else:
            target_index = int(generation_target_index)
            roles = [("target", target_index)]
            if include_opposite_diagnostic:
                roles.append(("opposite_diagnostic", 1 - target_index))
        image_path = Path(str(image["image_path"]))
        if not image_path.is_absolute():
            image_path = REPO_ROOT / image_path
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        for evaluation_role, question_index in roles:
            target = spec["targets"][question_index]
            question = str(target["question"])
            prompt = render_vqa_prompt(question)
            trial_id = sha256_bytes(
                (
                    f"{image['condition']}\0{image['image_id']}\0{evaluation_role}\0"
                    f"{question_index}\0{question}"
                ).encode("utf-8")
            )
            trials.append(
                {
                    "schema_version": "tab-vqa-trial-v2",
                    "trial_id": trial_id,
                    "evaluation_role": evaluation_role,
                    "paper_primary_metric_member": evaluation_role == "target",
                    "condition": image["condition"],
                    "tab_id": image["tab_id"],
                    "tab_category": spec["tab_category"],
                    "ambiguous_prompt": spec["ambiguous_prompt"],
                    # For ambiguous vanilla there is no intended generation
                    # target.  Each released question is nevertheless one
                    # primary setup-specific trial, so this field follows the
                    # question index while generation_target_setup_index stays
                    # null.
                    "target_setup_index": (
                        question_index
                        if evaluation_mode == AMBIGUOUS_BOTH_SETUPS_MODE
                        else int(generation_target_index)
                    ),
                    "generation_target_setup_index": generation_target_index,
                    "question_setup_index": question_index,
                    "tab_evaluation_mode": evaluation_mode,
                    "visual_setup": target["visual_setup"],
                    "commonsense_label": target["commonsense_label"],
                    "official_question": question,
                    "official_question_sha256": sha256_bytes(question.encode("utf-8")),
                    "qwen_prompt": prompt,
                    "qwen_prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
                    "qwen_prompt_template_sha256": PROMPT_TEMPLATE_SHA256,
                    "image_path": str(image_path.resolve()),
                    "image_sha256": sha256_file(image_path),
                    "image_id": image["image_id"],
                    "seed": image["seed"],
                    "reading": image.get("reading"),
                    "source_metadata": image,
                }
            )
    trials.sort(
        key=lambda row: (
            row["condition"],
            row["tab_id"],
            str(row["reading"]),
            row["image_id"],
            row["evaluation_role"],
            row["question_setup_index"],
        )
    )
    if len({row["trial_id"] for row in trials}) != len(trials):
        raise ValueError("duplicate trial IDs")
    return trials, coverage


def write_trial_plan(
    path: Path,
    metadata: Sequence[dict[str, Any]],
    *,
    profile: TabCoverageProfile,
    with_source_metadata: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Write the trial plan of a TAB PP generation run (target and opposite-diagnostic trials).

    ``metadata`` are the run's generated-image records. Two recorded layouts:

    * ``with_source_metadata=False`` (SD1.4 CCG Diffusion runs): rows without
      ``source_metadata``, all target trials first, each part in build order;
    * ``with_source_metadata=True`` (Sana runs and the Predicated Diffusion
      baselines): rows keep the image record as ``source_metadata`` and are
      sorted by role, condition, item, question setup, interpretation and seed.

    An existing file must already hold exactly these bytes. Returns the trials
    and the coverage record.
    """

    specs = [row for row in read_jsonl(profile.default_spec) if row["tab_id"].startswith("tab-pp-")]
    trials, coverage = build_trial_plan(
        metadata, specs, benchmark_scope="full", include_opposite_diagnostic=True, profile=profile)
    if with_source_metadata:
        trials.sort(key=lambda t: (t["evaluation_role"] != "target", t["condition"], t["tab_id"],
                                   t["question_setup_index"], str(t["reading"]), t["seed"]))
    else:
        trials = [{k: v for k, v in t.items() if k != "source_metadata"} for t in trials]
        trials = [t for t in trials if t["evaluation_role"] == "target"] + [
            t for t in trials if t["evaluation_role"] == "opposite_diagnostic"]
    immutable_rows(path, trials)
    return trials, coverage


# --- panels and Yes-rate scoring -------------------------------------------------------------------------


def trial_key(row: Mapping[str, Any]) -> tuple[str, int, int]:
    """``(tab_id, question_setup_index, seed)`` of one primary trial."""
    return row["tab_id"], int(row["question_setup_index"]), int(row["seed"])


def validate_panel(rows: Sequence[Mapping[str, Any]], items: Sequence[str], seeds: Sequence[int]) -> None:
    """Require exactly one trial per item x question setup (0, 1) x seed."""
    keys = [trial_key(r) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != set(itertools.product(items, (0, 1), seeds)):
        raise ValueError("Incomplete, duplicate or unexpected TAB item/setup/seed panel")


def item_means(predictions: Sequence[Mapping[str, Any]], condition: str) -> tuple[list[str], Any]:
    """Sorted item IDs and per-item Yes rates (10 trials each) of one condition; the significance unit."""
    import numpy as np

    rows = [r for r in predictions if r["condition"] == condition]
    if len(rows) != 2 * PP_ITEMS * len(SEEDS) or any(r["invalid_answer"] for r in rows):
        raise ValueError(f"{condition}: expected {2 * PP_ITEMS * len(SEEDS)} valid judgments")
    by_item = defaultdict(list)
    for r in rows:
        by_item[r["tab_id"]].append(float(r["answered_yes"]))
    items = sorted(by_item)
    if len(items) != PP_ITEMS or not all(len(by_item[i]) == 2 * len(SEEDS) for i in items):
        raise ValueError(f"{condition}: expected {PP_ITEMS} items with {2 * len(SEEDS)} trials each")
    return items, np.array([np.mean(by_item[i]) for i in items])


def score_conditions(
    predictions: Sequence[Mapping[str, Any]],
    conditions: Sequence[str],
    *,
    seed_counts: Sequence[int] = (5,),
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    """Yes counts, Yes rate (item mean x 100) and a 74-item bootstrap CI per condition and seed count."""
    import numpy as np

    items = sorted({r["tab_id"] for r in predictions})
    bootstrap = np.random.default_rng(bootstrap_seed).integers(0, len(items), size=(20000, len(items)))
    scores = []
    for count in seed_counts:
        for condition in conditions:
            panel = [r for r in predictions if r["condition"] == condition and r["seed"] < count]
            validate_panel(panel, items, list(range(count)))
            values = np.array([sum(r["answered_yes"] for r in panel if r["tab_id"] == item) / (count * 2)
                               for item in items])
            scores.append({"condition": condition, "seed_count": count, "yes": sum(r["answered_yes"] for r in panel),
                           "trials": len(panel), "percent": float(values.mean() * 100),
                           "invalid": sum(r["invalid_answer"] for r in panel),
                           "item_bootstrap_ci95_percent": (np.quantile(values[bootstrap].mean(axis=1), [.025, .975])
                                                           * 100).tolist()})
    return scores


# --- per-trial judgments of the paper's TAB PP conditions ----------------------------------------------

# A compact projection of the judgments behind Tables 5 and 6, tracked so a
# public clone can recompute the Table 5 kappa column and the Table 5/6 tests.
# ``manifest.json`` names the full prediction files it was projected from and
# their sha256 (built by ``scripts/maintainer/build_tab_trial_judgments.py``).
TRIAL_JUDGMENTS_DIR = REPO_ROOT / "output/analysis/20261006_paper_significance/tab_trials"
TRIAL_JUDGMENTS_SCHEMA = "ccg_diffusion.tab_trial_judgments.v1"
TRIAL_JUDGMENT_FIELDS = (
    "judge", "run", "condition", "trial_id", "tab_id", "question_setup_index", "seed", "answer", "invalid_answer",
)
ASTRA_JUDGE_NAME = config.TAB_JUDGE
QWEN38_JUDGE_NAME = "qwen3.8-flash-next"


def project_trial_judgments(rows: Sequence[Mapping[str, Any]], *, judge: str, run: str) -> list[dict[str, Any]]:
    """The tracked projection of full prediction rows (input order kept)."""
    return [{"judge": judge, "run": run, **{k: row[k] for k in TRIAL_JUDGMENT_FIELDS[2:]}} for row in rows]


def load_trial_judgments(directory: Path = TRIAL_JUDGMENTS_DIR) -> list[dict[str, Any]]:
    """Read the tracked per-trial judgments after checking them against their manifest.

    Every row gains ``answered_yes`` (``answer == "yes"``), the field the
    scoring and kappa functions read.
    """

    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    record = manifest["file"]
    payload = (directory / record["path"]).read_bytes()
    if manifest["schema"] != TRIAL_JUDGMENTS_SCHEMA or sha256_bytes(payload) != record["sha256_gz"]:
        raise ValueError(f"TAB trial judgments differ from their manifest: {directory}")
    text = gzip.decompress(payload)
    if sha256_bytes(text) != record["sha256"]:
        raise ValueError(f"TAB trial judgments differ from their manifest: {directory}")
    rows = [json.loads(line) for line in text.decode("utf-8").splitlines()]
    if len(rows) != record["rows"] or any(set(row) != set(TRIAL_JUDGMENT_FIELDS) for row in rows):
        raise ValueError(f"TAB trial judgments have unexpected rows: {directory}")
    return [{**row, "answered_yes": row["answer"] == "yes"} for row in rows]


# --- reading -> setup mapping ----------------------------------------------------------------------------
#
# The TAB PP generation produces one image per forced CCG interpretation. TAB,
# however, labels its two intended visual readings only as setup ``0`` and
# ``1``. The functions below are the deterministic bridge between those
# namespaces. They do not inspect images and have no per-item overrides: both
# official textual target channels must independently support the same
# low/high attachment-to-setup bijection or the item is rejected.


TAB_READING_SETUP_MAPPING_RULE_ID = "tab_ppvp_two_channel_attachment_setup_bijection_v1"
READINGS = ("np_attach", "vp_attach")
TARGET_CHANNELS = ("visual_setup", "question")
_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?", re.IGNORECASE)
_EDGE_PUNCTUATION = ".,;:!?()[]{}\"'"
_RELATIVE_MARKERS = frozenset({"that", "where", "which", "who"})
_HIGH_COREFERENCE_CUES = frozenset({"her", "his", "their", "someone"})
_HIGH_EVENT_CUES = frozenset({"holding", "used"})


class TabSetupMappingError(RuntimeError):
    """TAB reading-to-setup evidence is incomplete or internally inconsistent."""


def normalize_anchor(text: str) -> str:
    """Normalize one structural anchor without stemming or synonym expansion."""

    if not isinstance(text, str):
        raise TabSetupMappingError("anchor text must be a string")
    normalized = text.strip(_EDGE_PUNCTUATION).casefold()
    if normalized.endswith("'s") and len(normalized) > 2:
        normalized = normalized[:-2]
    if not normalized:
        raise TabSetupMappingError("anchor text is empty after edge-punctuation normalization")
    return normalized


def _words(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        raise TabSetupMappingError("target text must be non-empty")
    return [normalize_anchor(match.group(0)) for match in _WORD_RE.finditer(text)]


def _longest_common_prefix(left: Sequence[str], right: Sequence[str]) -> tuple[str, ...]:
    length = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        length += 1
    return tuple(left[:length])


def _last_direction_anchor(
    tokens: Sequence[str], *, low_anchor: str, high_anchor: str
) -> str | None:
    matches: list[tuple[int, str]] = []
    for position, token in enumerate(tokens):
        if token == low_anchor:
            matches.append((position, "low"))
        if token == high_anchor:
            matches.append((position, "high"))
    return None if not matches else max(matches, key=lambda entry: entry[0])[1]


def _orientation_sign(attachment: str) -> int:
    if attachment == "low":
        return 1
    if attachment == "high":
        return -1
    raise TabSetupMappingError(f"unknown attachment direction: {attachment!r}")


def channel_decision(
    target_text_by_setup: Mapping[int, str], *, low_anchor: str, high_anchor: str
) -> dict[str, Any]:
    """Apply the fixed M4-style contrastive rule to one official TAB channel."""

    if set(target_text_by_setup) != {0, 1}:
        raise TabSetupMappingError("target channel must contain exactly setup indices 0 and 1")
    anchors = {low_anchor, high_anchor}
    tokens = {
        setup_index: [
            token[:-1] if token.endswith("s") and token[:-1] in anchors else token
            for token in _words(target_text_by_setup[setup_index])
        ]
        for setup_index in (0, 1)
    }
    common_prefix = _longest_common_prefix(tokens[0], tokens[1])
    prefix_length = len(common_prefix)
    residuals = {
        setup_index: tokens[setup_index][prefix_length:] for setup_index in (0, 1)
    }
    carrier = _last_direction_anchor(
        common_prefix, low_anchor=low_anchor, high_anchor=high_anchor
    )
    negated = {setup_index: "not" in residuals[setup_index] for setup_index in (0, 1)}
    polarity_contrast = negated[0] != negated[1] and carrier is not None
    evidence: dict[int, dict[str, Any]] = {}
    scores: dict[int, int] = {}

    for setup_index in (0, 1):
        residual = residuals[setup_index]
        low_mentions = sum(token == low_anchor for token in residual)
        high_mentions = sum(token == high_anchor for token in residual)
        anchor_score = low_mentions - high_mentions
        polarity_score = 0
        if polarity_contrast:
            polarity_score = _orientation_sign(str(carrier)) * (
                -1 if negated[setup_index] else 1
            )
        relative_score = 0
        if residual and residual[0] in _RELATIVE_MARKERS and carrier is not None:
            relative_score = _orientation_sign(str(carrier))
        participial_score = (
            -1 if residual and len(residual[0]) > 4 and residual[0].endswith("ing") else 0
        )
        score = anchor_score + polarity_score + relative_score + participial_score
        support: list[str] = []
        if low_mentions or polarity_score > 0 or relative_score > 0:
            support.append("low")
        if (
            high_mentions
            or polarity_score < 0
            or relative_score < 0
            or participial_score < 0
            or any(token in _HIGH_COREFERENCE_CUES for token in residual)
            or any(token in _HIGH_EVENT_CUES for token in residual)
        ):
            support.append("high")
        scores[setup_index] = score
        evidence[setup_index] = {
            "residual_tokens": residual,
            "low_anchor_mentions": low_mentions,
            "high_anchor_mentions": high_mentions,
            "negated": negated[setup_index],
            "anchor_score": anchor_score,
            "polarity_score": polarity_score,
            "relative_score": relative_score,
            "participial_score": participial_score,
            "score": score,
            "supported_attachments": support,
        }

    if scores[0] == scores[1]:
        return {
            "status": "nonunique",
            "common_prefix_tokens": list(common_prefix),
            "common_prefix_carrier": carrier,
            "polarity_contrast": polarity_contrast,
            "scores": scores,
            "attachment_to_setup": None,
            "options": evidence,
        }
    low_setup = 0 if scores[0] > scores[1] else 1
    high_setup = 1 if low_setup == 0 else 0
    return {
        "status": "mapped",
        "common_prefix_tokens": list(common_prefix),
        "common_prefix_carrier": carrier,
        "polarity_contrast": polarity_contrast,
        "scores": scores,
        "attachment_to_setup": {"low": low_setup, "high": high_setup},
        "options": evidence,
    }


def _targets_by_setup(item: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    targets = item.get("targets")
    if not isinstance(targets, list) or len(targets) != 2:
        raise TabSetupMappingError("TAB item must have exactly two targets")
    by_setup: dict[int, Mapping[str, Any]] = {}
    for target in targets:
        if not isinstance(target, Mapping):
            raise TabSetupMappingError("TAB target must be an object")
        setup_index = target.get("setup_index")
        if type(setup_index) is not int or setup_index not in {0, 1}:
            raise TabSetupMappingError("TAB setup_index must be exactly 0 or 1")
        if setup_index in by_setup:
            raise TabSetupMappingError(f"duplicate TAB setup_index {setup_index}")
        for channel in TARGET_CHANNELS:
            if not isinstance(target.get(channel), str) or not str(target[channel]).strip():
                raise TabSetupMappingError(f"TAB target {setup_index} has no {channel}")
        by_setup[setup_index] = target
    if set(by_setup) != {0, 1}:
        raise TabSetupMappingError("TAB targets do not form a 0/1 pair")
    return by_setup


def _site_pairs(row: Mapping[str, Any], reading: str) -> list[Mapping[str, Any]]:
    possession_pairs = row.get("possession_pairs")
    if not isinstance(possession_pairs, Mapping):
        raise TabSetupMappingError("syntax evidence lacks possession_pairs")
    pairs = possession_pairs.get(reading)
    if not isinstance(pairs, list):
        raise TabSetupMappingError(f"syntax evidence lacks {reading} possession pairs")
    site_pairs = [
        pair
        for pair in pairs
        if isinstance(pair, Mapping) and pair.get("from_site_attachment") is True
    ]
    if len(site_pairs) != len(pairs):
        # Non-site attachment possession pairs are informative for PD but do
        # not determine the reading-to-setup mapping.
        return site_pairs
    return site_pairs


def _single_holder(pairs: Sequence[Mapping[str, Any]], *, reading: str) -> str:
    holders: set[str] = set()
    for pair in pairs:
        holder_word = pair.get("holder_word")
        if not isinstance(holder_word, str):
            raise TabSetupMappingError(f"{reading} site pair has no holder_word")
        holders.add(normalize_anchor(holder_word))
    if len(holders) != 1:
        raise TabSetupMappingError(
            f"{reading} must have exactly one site possession holder, found {sorted(holders)}"
        )
    return next(iter(holders))


def _anchors_from_syntax(row: Mapping[str, Any]) -> dict[str, Any]:
    prompt = row.get("prompt")
    site = row.get("site")
    if not isinstance(prompt, str) or not isinstance(site, Mapping):
        raise TabSetupMappingError("syntax evidence lacks prompt or site")
    position = site.get("position")
    if type(position) is not int:
        raise TabSetupMappingError("syntax site position must be an integer")
    prompt_tokens = prompt.split()
    if position <= 0 or position >= len(prompt_tokens):
        raise TabSetupMappingError("syntax site position is not an interior prompt token")
    site_word = site.get("word")
    if not isinstance(site_word, str) or normalize_anchor(prompt_tokens[position]) != normalize_anchor(site_word):
        raise TabSetupMappingError("syntax site word does not match prompt at site position")
    low_anchor = normalize_anchor(prompt_tokens[position - 1])
    high_anchor = _single_holder(_site_pairs(row, "vp_attach"), reading="vp_attach")
    if low_anchor == high_anchor:
        raise TabSetupMappingError("low and high attachment anchors collapse to one word")

    np_pairs = _site_pairs(row, "np_attach")
    if np_pairs:
        np_holder = _single_holder(np_pairs, reading="np_attach")
        if np_holder != low_anchor:
            raise TabSetupMappingError(
                f"np_attach site holder {np_holder!r} does not equal low anchor {low_anchor!r}"
            )
        np_support = "site_holder_matches_low_anchor"
    else:
        branch_kind = row.get("site_branch_kind")
        reasons = row.get("site_frame_reasons")
        np_reasons = reasons.get("np_attach") if isinstance(reasons, Mapping) else None
        if (
            branch_kind != "one_sided_missing_slot"
            or not isinstance(np_reasons, list)
            or "attachment_missing_slot_1" not in np_reasons
        ):
            raise TabSetupMappingError(
                "np_attach lacks a site pair without the sealed missing-slot disposition"
            )
        np_support = "sealed_one_sided_missing_slot_low_anchor"
    return {
        "site_position": position,
        "site_word": normalize_anchor(str(site_word)),
        "low_anchor": low_anchor,
        "high_anchor": high_anchor,
        "np_attach_support": np_support,
        "vp_attach_support": "site_holder",
    }


def map_tab_item_readings_to_setups(
    item: Mapping[str, Any], syntax_evidence: Mapping[str, Any]
) -> dict[str, Any]:
    """Map ``np_attach``/``vp_attach`` to TAB setup 0/1 with two-channel evidence.

    ``np_attach`` is structurally the low attachment and ``vp_attach`` the
    high attachment.  That fact alone is insufficient: the official TAB
    ``visual_setup`` and ``question`` strings must each make the same strict,
    directly supported low/high setup distinction.
    """

    tab_id = item.get("tab_id")
    if not isinstance(tab_id, str) or not tab_id:
        raise TabSetupMappingError("TAB item has no tab_id")
    if syntax_evidence.get("tab_id") != tab_id:
        raise TabSetupMappingError("TAB item and syntax evidence tab_id differ")
    item_prompt = item.get("ambiguous_prompt")
    syntax_prompt = syntax_evidence.get("prompt")
    if not isinstance(item_prompt, str) or not isinstance(syntax_prompt, str):
        raise TabSetupMappingError("TAB item or syntax evidence has no prompt")
    if item_prompt.strip() != syntax_prompt:
        raise TabSetupMappingError("TAB item prompt does not match syntax evidence after edge stripping")
    if syntax_evidence.get("differs") is not True:
        raise TabSetupMappingError("syntax evidence does not record a PD reading difference")
    if syntax_evidence.get("changed_fields") != ["possession"]:
        raise TabSetupMappingError("syntax evidence changed fields are not exactly possession")

    targets = _targets_by_setup(item)
    anchors = _anchors_from_syntax(syntax_evidence)
    channels: dict[str, dict[str, Any]] = {}
    for channel in TARGET_CHANNELS:
        channels[channel] = channel_decision(
            {setup_index: str(targets[setup_index][channel]) for setup_index in (0, 1)},
            low_anchor=str(anchors["low_anchor"]),
            high_anchor=str(anchors["high_anchor"]),
        )
    unmapped = [channel for channel, decision in channels.items() if decision["status"] != "mapped"]
    if unmapped:
        raise TabSetupMappingError(f"TAB target channel has no strict setup mapping: {unmapped}")
    maps = [channels[channel]["attachment_to_setup"] for channel in TARGET_CHANNELS]
    if maps[0] != maps[1]:
        raise TabSetupMappingError("TAB visual_setup/question channels disagree about low/high mapping")
    attachment_to_setup = dict(maps[0])
    unsupported: list[dict[str, Any]] = []
    for channel in TARGET_CHANNELS:
        for attachment, setup_index in attachment_to_setup.items():
            supported = channels[channel]["options"][setup_index]["supported_attachments"]
            if attachment not in supported:
                unsupported.append(
                    {"channel": channel, "setup_index": setup_index, "attachment": attachment}
                )
    if unsupported:
        raise TabSetupMappingError(
            f"TAB channel mapping lacks direct attachment support: {unsupported}"
        )

    reading_to_target_setup_index = {
        "np_attach": int(attachment_to_setup["low"]),
        "vp_attach": int(attachment_to_setup["high"]),
    }
    if set(reading_to_target_setup_index.values()) != {0, 1}:
        raise TabSetupMappingError("reading-to-setup mapping is not a 0/1 bijection")
    evidence = {
        "tab_id": tab_id,
        "prompt": syntax_prompt,
        "targets": {
            str(setup_index): {
                channel: targets[setup_index][channel] for channel in TARGET_CHANNELS
            }
            for setup_index in (0, 1)
        },
        "anchors": anchors,
        "channels": channels,
        "reading_to_target_setup_index": reading_to_target_setup_index,
    }
    return {
        "tab_id": tab_id,
        "rule_id": TAB_READING_SETUP_MAPPING_RULE_ID,
        "reading_to_target_setup_index": reading_to_target_setup_index,
        "anchors": anchors,
        "channels": channels,
        "evidence_sha256": sha256_text(canonical_json(evidence)),
    }
