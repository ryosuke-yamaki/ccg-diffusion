"""SD1.4 TAB comparison methods (paper Table 5).

Vanilla and VS-TIED + Gold-visual setup use stock denoising.
``write_baseline_inputs`` writes the prompt manifests of the two stock arms
over all 315 TAB items x seeds 0-4 (Vanilla: the ambiguous prompt, 1,575
images; Gold-visual setup: the prompt plus each gold visual setup, 3,150
images), their shared provenance records and a derived contract.
``driver_request`` turns one arm into an ``SD14Run`` of the SD1.4 batch driver
(physical batches of 4 in manifest order, so VP rows are part of the layout).
``load_tab_baseline_provenance`` re-verifies the inputs and gives the driver the
per-image TAB fields.

Predicated Diffusion and Gold-visual setup + Predicated Diffusion form one run
of their own (arms ``pd_spacy`` and ``pd_spacy_gold``: the 74 PP items x seeds
0-4, 370 and 740 images). Their guidance targets are the spaCy targets of the
original paper on the text that conditions the generator
(``spacy_target_rows``, SD1.4 CLIP indices), applied with the T2I settings of
the Predicated Diffusion baseline (``SD14_PD_SETTINGS``). ``prepare_pd_run``
writes ``metadata_jobs.jsonl``, ``pd_spacy_sidecar.jsonl`` and
``run_manifest.json``; ``generate_pd_baselines`` runs the guided denoising
kernel over physical batches of 4 per arm; ``merge_pd_run`` writes the trial
plan the judges read. The Sana runs (``baselines.tab_sana``) adapt the same
targets and share this run layout (``write_pd_run``, ``load_pd_run``,
``pd_records``, ``merge_pd_run``).
"""
from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from ccg_diffusion.backbones.jobs import Job
from ccg_diffusion.baselines.gold_visual_setup import (
    PD_ARMS,
    gold_visual_setup_prompt,
    pd_baseline_batches,
    pd_baseline_jobs,
)
from ccg_diffusion.baselines.predicated_diffusion import PROVIDER, SPACY_MODEL, SpacyProvider
from ccg_diffusion.config import (
    SD14_DETERMINISM,
    SD14_GENERATION,
    SD14_GOLD_VISUAL_SETUP_ARM,
    SD14_MAX_LATENT_ABS,
    SD14_SCHEDULER_CONFIG_SHA256,
    SD14_VANILLA_ARM,
    TAB_DATASET_ID,
    TAB_DATASET_VERSION,
)
from ccg_diffusion.evaluation.tab import ITEMS, load_pp_items, pp_profile, write_trial_plan
from ccg_diffusion.provenance import (
    atomic_write_json,
    atomic_write_jsonl,
    read_jsonl,
    repo_relative,
    repository_root,
    resolve_recorded_path,
    sha256_file,
    sha256_text,
    source_tree_snapshot,
)
from ccg_diffusion.targets.pd_indices import PD_FIELDS

SOURCE_CONTRACT_SCHEMA = "holsd.tab_ppvp_disambiguation_contract.v1"
SOURCE_DATASET_ID = TAB_DATASET_ID
DERIVED_CONTRACT_SCHEMA = "holsd.tab_ppvp_sd14_baselines_5seed_contract.v1"
DERIVED_DATASET_ID = "tab-ppvp-sd14-baselines-5seed"
INPUTS_SUMMARY_SCHEMA = "holsd.tab_ppvp_sd14_baseline_inputs.v1"
GENERATION_SEEDS = (0, 1, 2, 3, 4)
VANILLA_EXPERIMENT = "tab_ppvp_vanilla"
GOLD_VISUAL_SETUP_EXPERIMENT = "tab_ppvp_tied_oracle"
TAB_ITEM_COUNT = 315
BASELINE_ARMS = {"vanilla": (SD14_VANILLA_ARM, 1575), "gold_visual_setup": (SD14_GOLD_VISUAL_SETUP_ARM, 3150)}
# Files beyond the driver's own seal that every TAB stock-arm run lock records.
BASELINE_COMPONENT_FILES = (
    "scripts/tab_pp/run_baselines.py",
    "src/ccg_diffusion/baselines/gold_visual_setup.py",
    "src/ccg_diffusion/baselines/tab_sd14.py",
)
PROMPT_FIELDS = frozenset({"experiment", "prompt_id", "prompt", "seed"})
RECORD_FIELDS = frozenset(
    {
        "arm_id",
        "experiment",
        "prompt_id",
        "prompt",
        "seed",
        "tab_id",
        "tab_category",
        "evaluation_mode",
        "evaluation_setup_indices",
        "target_setup_index",
        "visual_setup",
        "prompt_construction",
        "source_prompt_sha256",
        "generation_prompt_sha256",
        "visual_setup_sha256",
        "prompt_stripped",
        "clip_token_count",
    }
)


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{label} not found: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicates
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _load_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"{label} not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank row in {label}: {path}:{line_number}")
            try:
                row = json.loads(line, object_pairs_hook=_reject_duplicates)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON in {label}: {path}:{line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"non-object row in {label}: {path}:{line_number}")
            rows.append(row)
    return rows


def _resolve(value: Any, *, label: str) -> Path:
    """A recorded path: repository-relative, or a historical absolute path (re-rooted when absent)."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty path")
    return resolve_recorded_path(value)


def _require_artifact(contract: Mapping[str, Any], name: str, *, explicit_path: Path) -> None:
    artifacts = contract.get("artifacts")
    record = artifacts.get(name) if isinstance(artifacts, Mapping) else None
    if not isinstance(record, Mapping):
        raise ValueError(f"TAB baseline contract lacks artifacts.{name}")
    recorded_path = _resolve(record.get("path"), label=f"artifacts.{name}.path")
    if recorded_path != explicit_path.resolve():
        raise ValueError(
            f"TAB baseline {name} path differs from its contract: "
            f"{explicit_path.resolve()} != {recorded_path}"
        )
    expected_sha = record.get("sha256")
    if not isinstance(expected_sha, str) or sha256_file(explicit_path) != expected_sha:
        raise ValueError(f"TAB baseline {name} SHA-256 differs from its contract")


def _load_source_items(
    source_contract_path: Path,
) -> tuple[dict[str, dict[str, Any]], str, Path, str, int]:
    source = _load_json(source_contract_path, label="frozen TAB dataset contract")
    if (
        source.get("schema") != SOURCE_CONTRACT_SCHEMA
        or source.get("dataset_id") != SOURCE_DATASET_ID
        or source.get("version") != 1
        or source.get("status") != "frozen"
    ):
        raise ValueError("frozen TAB dataset identity is inconsistent")
    files = source.get("files")
    item_record = files.get("items") if isinstance(files, Mapping) else None
    if not isinstance(item_record, Mapping):
        raise ValueError("frozen TAB contract lacks files.items")
    items_path = (source_contract_path.parent / item_record.get("path")).resolve()
    expected_sha = item_record.get("sha256")
    if not isinstance(expected_sha, str) or sha256_file(items_path) != expected_sha:
        raise ValueError("frozen TAB items differ from their contract")
    item_rows = _load_jsonl(items_path, label="frozen TAB items")
    if len(item_rows) != 315 or item_record.get("rows") != 315:
        raise ValueError("frozen TAB source must contain exactly 315 items")
    items: dict[str, dict[str, Any]] = {}
    for row in item_rows:
        tab_id = row.get("tab_id")
        prompt = row.get("ambiguous_prompt")
        category = row.get("tab_category")
        targets = row.get("targets")
        if (
            not isinstance(tab_id, str)
            or not tab_id
            or tab_id in items
            or not isinstance(prompt, str)
            or not prompt.strip()
            or not isinstance(category, str)
            or not category
            or not isinstance(targets, list)
            or len(targets) != 2
        ):
            raise ValueError(f"invalid frozen TAB item: {tab_id!r}")
        targets_by_index: dict[int, dict[str, Any]] = {}
        for target in targets:
            if not isinstance(target, Mapping):
                raise ValueError(f"invalid frozen TAB target: {tab_id!r}")
            setup_index = target.get("setup_index")
            visual_setup = target.get("visual_setup")
            if (
                type(setup_index) is not int
                or setup_index not in {0, 1}
                or setup_index in targets_by_index
                or not isinstance(visual_setup, str)
                or not visual_setup
            ):
                raise ValueError(f"invalid frozen TAB target: {tab_id!r}")
            targets_by_index[setup_index] = dict(target)
        if set(targets_by_index) != {0, 1}:
            raise ValueError(f"TAB item lacks setup 0 or 1: {tab_id}")
        items[tab_id] = {
            "source_prompt": prompt,
            "generation_prompt": prompt.strip(),
            "prompt_stripped": prompt != prompt.strip(),
            "tab_category": category,
            "source_row_index": row.get("source_row_index"),
            "targets": targets_by_index,
        }
    return (
        items,
        sha256_file(source_contract_path),
        items_path,
        str(expected_sha),
        int(source["version"]),
    )


@dataclass(frozen=True)
class TABBaselineDatasetProvenance:
    """Verified source/derived identities and per-image VQA pairing metadata."""

    source_contract_path: Path
    source_contract_sha256: str
    items_path: Path
    items_sha256: str
    derived_contract_path: Path
    derived_contract_sha256: str
    records_path: Path
    records_sha256: str
    prompt_manifest_path: Path
    prompt_manifest_sha256: str
    arm_id: str
    image_records: Mapping[tuple[str, str, int], Mapping[str, Any]]

    def run_record(self) -> dict[str, Any]:
        primary_trials = sum(
            len(record["evaluation_setup_indices"])
            for record in self.image_records.values()
        )
        return {
            "dataset_id": SOURCE_DATASET_ID,
            "dataset_version": TAB_DATASET_VERSION,
            "derived_generation_id": DERIVED_DATASET_ID,
            "derived_generation_schema": DERIVED_CONTRACT_SCHEMA,
            "derived_generation_contract": repo_relative(self.derived_contract_path),
            "derived_generation_contract_sha256": self.derived_contract_sha256,
            "dataset_contract": repo_relative(self.source_contract_path),
            "dataset_contract_sha256": self.source_contract_sha256,
            "tab_items": repo_relative(self.items_path),
            "tab_items_sha256": self.items_sha256,
            "tab_generation_records": repo_relative(self.records_path),
            "tab_generation_records_sha256": self.records_sha256,
            "tab_generation_manifest": repo_relative(self.prompt_manifest_path),
            "tab_generation_manifest_sha256": self.prompt_manifest_sha256,
            "tab_generation_population": {
                "item_count": 315,
                "image_seeds": list(GENERATION_SEEDS),
                "physical_images": len(self.image_records),
                "primary_vqa_trials": primary_trials,
            },
            "canonical_arm_id": self.arm_id,
            "provider": None,
            "stock_denoising": True,
        }

    def image_record(self, job: Job) -> dict[str, Any]:
        key = (job.experiment, job.prompt_id, job.seed)
        record = self.image_records.get(key)
        if record is None:
            raise ValueError(f"job is absent from verified TAB baseline population: {key!r}")
        return {
            **record,
            "dataset_id": SOURCE_DATASET_ID,
            "dataset_version": TAB_DATASET_VERSION,
            "derived_generation_id": DERIVED_DATASET_ID,
            "condition": self.arm_id,
            "image_id": job.prompt_id,
            "tab_evaluation_mode": record["evaluation_mode"],
            "tab_requires_reading": False,
        }

    def require_prepared_jobs(self, prepared_jobs: Sequence[Any]) -> None:
        observed: set[tuple[str, str, int]] = set()
        for prepared in prepared_jobs:
            job = getattr(prepared, "job", None)
            if job is None:
                raise ValueError("TAB baseline preflight received an invalid prepared job")
            key = (job.experiment, job.prompt_id, job.seed)
            record = self.image_records.get(key)
            if record is None:
                raise ValueError(f"prepared job is absent from TAB baseline records: {key!r}")
            if getattr(prepared, "provider", None) is not None:
                raise ValueError(f"TAB stock baseline unexpectedly received a provider: {key!r}")
            if getattr(prepared, "prompt_sha256", None) != record["generation_prompt_sha256"]:
                raise ValueError(f"TAB baseline prepared prompt hash mismatch: {key!r}")
            observed.add(key)
        if observed != set(self.image_records):
            raise ValueError("TAB prepared-job population does not equal its selected baseline arm")


def load_tab_baseline_provenance(
    prompt_manifest: Path,
    *,
    dataset_contract: Path,
    generation_records: Path,
    preparation_contract: Path,
    arm_id: str,
) -> TABBaselineDatasetProvenance:
    """Load and recompute the exact stock-baseline population contract."""

    if arm_id not in {SD14_VANILLA_ARM, SD14_GOLD_VISUAL_SETUP_ARM}:
        raise ValueError(f"unsupported TAB baseline arm: {arm_id}")
    preparation_contract = preparation_contract.resolve()
    derived = _load_json(preparation_contract, label="TAB baseline preparation contract")
    if (
        derived.get("schema") != DERIVED_CONTRACT_SCHEMA
        or derived.get("dataset_id") != DERIVED_DATASET_ID
        or derived.get("version") != 1
        or derived.get("status") != "derived"
    ):
        raise ValueError("TAB baseline derived contract identity is inconsistent")
    source_record = derived.get("source_dataset")
    source_contract_record = (
        source_record.get("contract") if isinstance(source_record, Mapping) else None
    )
    if not isinstance(source_contract_record, Mapping):
        raise ValueError("TAB baseline contract lacks source dataset contract identity")
    if _resolve(source_contract_record.get("path"), label="source dataset contract path") != dataset_contract.resolve():
        raise ValueError("explicit TAB source contract differs from derived contract")
    source_sha = source_contract_record.get("sha256")
    if not isinstance(source_sha, str) or sha256_file(dataset_contract) != source_sha:
        raise ValueError("TAB source dataset contract SHA-256 differs from derived contract")
    items, source_contract_sha, items_path, items_sha, source_version = _load_source_items(
        dataset_contract.resolve()
    )
    if source_version != 1 or source_contract_sha != source_sha:
        raise ValueError("TAB source dataset version/hash drifted")
    source_items_record = (
        source_record.get("items") if isinstance(source_record, Mapping) else None
    )
    if (
        not isinstance(source_items_record, Mapping)
        or _resolve(source_items_record.get("path"), label="source TAB items path") != items_path
        or source_items_record.get("sha256") != items_sha
        or source_items_record.get("rows") != 315
    ):
        raise ValueError("derived TAB baseline contract has inconsistent source items")

    artifact_name = (
        "vanilla_manifest"
        if arm_id == SD14_VANILLA_ARM
        else "tied_oracle_manifest"
    )
    _require_artifact(
        derived,
        artifact_name,
        explicit_path=prompt_manifest,
    )
    _require_artifact(
        derived,
        "records",
        explicit_path=generation_records,
    )
    generation = derived.get("generation")
    expected_generation = {
        "backbone": "sd14",
        "checkpoint": SD14_GENERATION.model_id,
        "revision": SD14_GENERATION.model_revision,
        "tokenizer": SD14_GENERATION.tokenizer_identity(),
        "stock_denoising": True,
        "pd_sidecar": None,
    }
    if not isinstance(generation, Mapping) or any(
        generation.get(key) != value for key, value in expected_generation.items()
    ):
        raise ValueError("TAB baseline SD1.4 generation identity is inconsistent")
    seed_override = derived.get("seed_override")
    population = derived.get("population")
    conditions = derived.get("conditions")
    comparison_pairing = derived.get("comparison_pairing")
    expected_conditions = {
        SD14_VANILLA_ARM: {
            "experiment": VANILLA_EXPERIMENT,
            "evaluation_mode": "ambiguous_both_setups",
            "physical_images": 1575,
            "primary_vqa_trials": 3150,
        },
        SD14_GOLD_VISUAL_SETUP_ARM: {
            "experiment": GOLD_VISUAL_SETUP_EXPERIMENT,
            "evaluation_mode": "targeted_setup",
            "physical_images": 3150,
            "primary_vqa_trials": 3150,
        },
    }
    if (
        not isinstance(seed_override, Mapping)
        or seed_override.get("approved_by") != "user"
        or seed_override.get("derived_generation_seeds") != list(GENERATION_SEEDS)
        or not isinstance(population, Mapping)
        or population.get("items") != 315
        or population.get("image_seeds") != list(GENERATION_SEEDS)
        or population.get("physical_images") != 4725
        or population.get("primary_vqa_trials") != 6300
        or not isinstance(conditions, Mapping)
        or comparison_pairing
        != {
            "item_key": "tab_id",
            "image_seed_key": "seed",
            "same_seed_uses_same_initial_latent_across_arms": True,
            "primary_vqa_trials_per_item_per_condition": 10,
            "vanilla_image_reuse": (
                "one ambiguous-prompt physical image is paired with setup questions 0 and 1"
            ),
        }
    ):
        raise ValueError("TAB baseline population/seed contract is inconsistent")
    for condition, expected in expected_conditions.items():
        actual = conditions.get(condition)
        if not isinstance(actual, Mapping) or any(
            actual.get(key) != value for key, value in expected.items()
        ):
            raise ValueError(f"TAB baseline condition contract drifted: {condition}")

    manifest_rows = _load_jsonl(prompt_manifest.resolve(), label="TAB baseline prompt manifest")
    record_rows = _load_jsonl(generation_records.resolve(), label="TAB baseline records")
    if len(record_rows) != 4725 or any(set(row) != RECORD_FIELDS for row in record_rows):
        raise ValueError("shared TAB baseline ledger must contain 4,725 exact-schema rows")
    record_arm_counts: dict[str, int] = {}
    for row in record_rows:
        row_arm = row.get("arm_id")
        if row_arm not in {SD14_VANILLA_ARM, SD14_GOLD_VISUAL_SETUP_ARM}:
            raise ValueError(f"shared TAB baseline ledger has unknown arm: {row_arm!r}")
        record_arm_counts[str(row_arm)] = record_arm_counts.get(str(row_arm), 0) + 1
    if record_arm_counts != {
        SD14_VANILLA_ARM: 1575,
        SD14_GOLD_VISUAL_SETUP_ARM: 3150,
    }:
        raise ValueError("shared TAB baseline ledger arm populations are inconsistent")
    selected_records = [row for row in record_rows if row.get("arm_id") == arm_id]
    expected_count = 1575 if arm_id == SD14_VANILLA_ARM else 3150
    if len(manifest_rows) != expected_count or len(selected_records) != expected_count:
        raise ValueError(f"{arm_id} must contain exactly {expected_count} physical images")
    expected_keys: dict[tuple[str, str, int], dict[str, Any]] = {}
    for tab_id, item in items.items():
        if arm_id == SD14_VANILLA_ARM:
            expected_prompts = [
                (
                    VANILLA_EXPERIMENT,
                    f"{tab_id}_ambiguous_s{seed}",
                    seed,
                    item["generation_prompt"],
                    None,
                    None,
                    [0, 1],
                    "ambiguous_both_setups",
                )
                for seed in GENERATION_SEEDS
            ]
        else:
            expected_prompts = []
            for setup_index in (0, 1):
                visual_setup = item["targets"][setup_index]["visual_setup"]
                prompt = gold_visual_setup_prompt(
                    item["generation_prompt"], visual_setup
                )
                expected_prompts.extend(
                    (
                        GOLD_VISUAL_SETUP_EXPERIMENT,
                        f"{tab_id}_setup{setup_index}_s{seed}",
                        seed,
                        prompt,
                        setup_index,
                        visual_setup,
                        [setup_index],
                        "targeted_setup",
                    )
                    for seed in GENERATION_SEEDS
                )
        for (
            experiment,
            prompt_id,
            seed,
            prompt,
            target_index,
            visual_setup,
            eval_indices,
            mode,
        ) in expected_prompts:
            expected_keys[(experiment, prompt_id, seed)] = {
                "prompt": prompt,
                "tab_id": tab_id,
                "tab_category": item["tab_category"],
                "target_setup_index": target_index,
                "visual_setup": visual_setup,
                "evaluation_setup_indices": eval_indices,
                "evaluation_mode": mode,
                "generation_prompt_sha256": sha256_text(prompt),
                "source_prompt_sha256": sha256_text(item["source_prompt"]),
                "visual_setup_sha256": (
                    None if visual_setup is None else sha256_text(visual_setup)
                ),
                "prompt_stripped": item["prompt_stripped"],
                "prompt_construction": (
                    "released_ambiguous_prompt_strip_only"
                    if arm_id == SD14_VANILLA_ARM
                    else "canonical_ambiguous_period_space_visual_setup_period"
                ),
            }

    actual_manifest: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in manifest_rows:
        if set(row) != PROMPT_FIELDS:
            raise ValueError("TAB baseline manifest must use the exact generic four-field schema")
        key = (row.get("experiment"), row.get("prompt_id"), row.get("seed"))
        if key in actual_manifest:
            raise ValueError(f"duplicate TAB baseline manifest key: {key!r}")
        actual_manifest[key] = row
    actual_records: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in selected_records:
        key = (row.get("experiment"), row.get("prompt_id"), row.get("seed"))
        if key in actual_records:
            raise ValueError(f"duplicate TAB baseline record key: {key!r}")
        actual_records[key] = row
    if set(actual_manifest) != set(expected_keys) or set(actual_records) != set(expected_keys):
        raise ValueError(f"{arm_id} manifest/record keys do not equal the frozen population")
    for key, expected in expected_keys.items():
        manifest = actual_manifest[key]
        record = actual_records[key]
        if manifest["prompt"] != expected["prompt"]:
            raise ValueError(f"TAB baseline prompt changed: {key!r}")
        checks = {
            **expected,
            "arm_id": arm_id,
            "experiment": key[0],
            "prompt_id": key[1],
            "seed": key[2],
            "prompt": expected["prompt"],
        }
        mismatches = {
            field: {"expected": value, "actual": record.get(field)}
            for field, value in checks.items()
            if record.get(field) != value
        }
        if mismatches:
            raise ValueError(f"TAB baseline record mismatch for {key!r}: {mismatches}")
        clip_token_count = record.get("clip_token_count")
        if (
            type(clip_token_count) is not int
            or clip_token_count < 3
            or clip_token_count > SD14_GENERATION.tokenizer_model_max_length
        ):
            raise ValueError(f"TAB baseline CLIP token count is invalid for {key!r}")

    return TABBaselineDatasetProvenance(
        source_contract_path=dataset_contract.resolve(),
        source_contract_sha256=source_contract_sha,
        items_path=items_path,
        items_sha256=items_sha,
        derived_contract_path=preparation_contract,
        derived_contract_sha256=sha256_file(preparation_contract),
        records_path=generation_records.resolve(),
        records_sha256=sha256_file(generation_records),
        prompt_manifest_path=prompt_manifest.resolve(),
        prompt_manifest_sha256=sha256_file(prompt_manifest),
        arm_id=arm_id,
        image_records=actual_records,
    )


# --- inputs of the two stock arms -----------------------------------------------------------------------


def _token_count(tokenizer: Any, prompt: str) -> int:
    encoded = tokenizer(
        prompt,
        add_special_tokens=True,
        padding=False,
        truncation=False,
    )["input_ids"]
    if encoded and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise ValueError("unexpected batched SD1.4 tokenization")
        encoded = encoded[0]
    return len(encoded)


def build_baseline_manifests(
    items: Sequence[Mapping[str, Any]],
    *,
    tokenizer: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Prompt manifests of Vanilla and Gold-visual setup and their shared provenance records."""

    vanilla_manifest: list[dict[str, Any]] = []
    gold_visual_setup_manifest: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    max_length = int(tokenizer.model_max_length)
    for item in items:
        tab_id = str(item["tab_id"])
        source_prompt = str(item["ambiguous_prompt"])
        ambiguous_prompt = source_prompt.strip()
        prompt_stripped = ambiguous_prompt != source_prompt
        ambiguous_tokens = _token_count(tokenizer, ambiguous_prompt)
        if ambiguous_tokens > max_length:
            raise ValueError(f"{tab_id}: ambiguous prompt has {ambiguous_tokens} tokens, limit {max_length}")
        for seed in GENERATION_SEEDS:
            prompt_id = f"{tab_id}_ambiguous_s{seed}"
            manifest = {
                "experiment": VANILLA_EXPERIMENT,
                "prompt_id": prompt_id,
                "prompt": ambiguous_prompt,
                "seed": seed,
            }
            vanilla_manifest.append(manifest)
            records.append(
                {
                    "arm_id": SD14_VANILLA_ARM,
                    **manifest,
                    "tab_id": tab_id,
                    "tab_category": item["tab_category"],
                    "evaluation_mode": "ambiguous_both_setups",
                    "evaluation_setup_indices": [0, 1],
                    "target_setup_index": None,
                    "visual_setup": None,
                    "prompt_construction": "released_ambiguous_prompt_strip_only",
                    "source_prompt_sha256": sha256_text(source_prompt),
                    "generation_prompt_sha256": sha256_text(ambiguous_prompt),
                    "visual_setup_sha256": None,
                    "prompt_stripped": prompt_stripped,
                    "clip_token_count": ambiguous_tokens,
                }
            )

        targets = item.get("targets")
        if not isinstance(targets, list) or len(targets) != 2:
            raise ValueError(f"{tab_id}: expected two released visual setups")
        for target in targets:
            setup_index = target.get("setup_index")
            visual_setup = target.get("visual_setup")
            if type(setup_index) is not int or setup_index not in {0, 1}:
                raise ValueError(f"{tab_id}: setup_index must be 0 or 1")
            if not isinstance(visual_setup, str) or not visual_setup:
                raise ValueError(f"{tab_id}: visual_setup must be non-empty")
            disambiguated_prompt = gold_visual_setup_prompt(ambiguous_prompt, visual_setup)
            tied_tokens = _token_count(tokenizer, disambiguated_prompt)
            if tied_tokens > max_length:
                raise ValueError(f"{tab_id}/setup{setup_index}: prompt has {tied_tokens} tokens, limit {max_length}")
            for seed in GENERATION_SEEDS:
                prompt_id = f"{tab_id}_setup{setup_index}_s{seed}"
                manifest = {
                    "experiment": GOLD_VISUAL_SETUP_EXPERIMENT,
                    "prompt_id": prompt_id,
                    "prompt": disambiguated_prompt,
                    "seed": seed,
                }
                gold_visual_setup_manifest.append(manifest)
                records.append(
                    {
                        "arm_id": SD14_GOLD_VISUAL_SETUP_ARM,
                        **manifest,
                        "tab_id": tab_id,
                        "tab_category": item["tab_category"],
                        "evaluation_mode": "targeted_setup",
                        "evaluation_setup_indices": [setup_index],
                        "target_setup_index": setup_index,
                        "visual_setup": visual_setup,
                        "prompt_construction": (
                            "canonical_ambiguous_period_space_visual_setup_period"
                        ),
                        "source_prompt_sha256": sha256_text(source_prompt),
                        "generation_prompt_sha256": sha256_text(disambiguated_prompt),
                        "visual_setup_sha256": sha256_text(visual_setup),
                        "prompt_stripped": prompt_stripped,
                        "clip_token_count": tied_tokens,
                    }
                )

    expected_vanilla = TAB_ITEM_COUNT * len(GENERATION_SEEDS)
    expected_gold_visual_setup = TAB_ITEM_COUNT * 2 * len(GENERATION_SEEDS)
    if len(vanilla_manifest) != expected_vanilla or len(gold_visual_setup_manifest) != expected_gold_visual_setup:
        raise ValueError("TAB baseline population size is inconsistent")
    if len(records) != expected_vanilla + expected_gold_visual_setup:
        raise ValueError("TAB baseline record population size is inconsistent")
    if any(set(row) != PROMPT_FIELDS for row in (*vanilla_manifest, *gold_visual_setup_manifest)):
        raise ValueError("generic generation manifest fields changed")
    if any(set(row) != RECORD_FIELDS for row in records):
        raise ValueError("TAB baseline provenance record fields changed")
    keys = [(row["experiment"], row["prompt_id"], row["seed"]) for row in records]
    if len(keys) != len(set(keys)):
        raise ValueError("TAB baseline job keys are not unique")
    return vanilla_manifest, gold_visual_setup_manifest, records


def baseline_input_paths(run_id: str) -> dict[str, Path]:
    """``output/generation/<run_id>/baseline_inputs/`` files of the two stock arms."""

    output_dir = repository_root() / "output/generation" / run_id / "baseline_inputs"
    return {
        "vanilla_manifest": output_dir / "tab_sd14_vanilla_manifest.jsonl",
        "tied_oracle_manifest": output_dir / "tab_sd14_tied_oracle_manifest.jsonl",
        "records": output_dir / "tab_sd14_baseline_records.jsonl",
        "contract": output_dir / "tab_sd14_baselines_5seed_contract.json",
        "summary": output_dir / "tab_sd14_baseline_inputs.summary.json",
    }


def write_baseline_inputs(
    run_id: str, items: Sequence[Mapping[str, Any]], tokenizer_identity: Mapping[str, Any], tokenizer: Any,
    dataset_contract: Path,
) -> dict[str, Any]:
    """Write the manifests, records, derived contract and summary of the two stock arms; return the summary."""

    paths = baseline_input_paths(run_id)
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite TAB baseline artifacts: " + ", ".join(existing))
    source_contract = json.loads(dataset_contract.read_text(encoding="utf-8"))
    items_path = (dataset_contract.parent / source_contract["files"]["items"]["path"]).resolve()
    vanilla_manifest, gold_visual_setup_manifest, records = build_baseline_manifests(items, tokenizer=tokenizer)
    root = repository_root()
    vanilla_record = atomic_write_jsonl(paths["vanilla_manifest"], vanilla_manifest, overwrite=False)
    gold_visual_setup_record = atomic_write_jsonl(paths["tied_oracle_manifest"], gold_visual_setup_manifest,
                                                  overwrite=False)
    records_record = atomic_write_jsonl(paths["records"], records, overwrite=False)
    contract = {
        "schema": DERIVED_CONTRACT_SCHEMA,
        "dataset_id": DERIVED_DATASET_ID,
        "version": 1,
        "status": "derived",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Paired five-seed stock SD1.4 TAB comparison: ambiguous-prompt vanilla "
            "and official-visual-setup concatenation (TIED oracle upper bound)."
        ),
        "source_dataset": {
            "dataset_id": SOURCE_DATASET_ID,
            "version": source_contract.get("version"),
            "status": source_contract.get("status"),
            "source_image_seeds": source_contract.get("population", {}).get("image_seeds"),
            "contract": {"path": repo_relative(dataset_contract), "sha256": sha256_file(dataset_contract)},
            "items": {"path": repo_relative(items_path), "sha256": sha256_file(items_path), "rows": len(items)},
        },
        "seed_override": {
            "approved_by": "user",
            "source_seeds": source_contract.get("population", {}).get("image_seeds"),
            "derived_generation_seeds": list(GENERATION_SEEDS),
            "reason": "Comparison uses the same user-requested 0--4 panel as HolCCG forced.",
        },
        "generation": {
            "backbone": "sd14",
            "checkpoint": SD14_GENERATION.model_id,
            "revision": SD14_GENERATION.model_revision,
            "tokenizer": dict(tokenizer_identity),
            "stock_denoising": True,
            "pd_sidecar": None,
        },
        "conditions": {
            SD14_VANILLA_ARM: {
                "experiment": VANILLA_EXPERIMENT,
                "evaluation_mode": "ambiguous_both_setups",
                "physical_images": len(vanilla_manifest),
                "primary_vqa_trials": len(vanilla_manifest) * 2,
                "prompt_construction": "released_ambiguous_prompt_strip_only",
            },
            SD14_GOLD_VISUAL_SETUP_ARM: {
                "experiment": GOLD_VISUAL_SETUP_EXPERIMENT,
                "evaluation_mode": "targeted_setup",
                "physical_images": len(gold_visual_setup_manifest),
                "primary_vqa_trials": len(gold_visual_setup_manifest),
                "prompt_construction": "canonical_ambiguous_period_space_visual_setup_period",
                "claim_boundary": "TIED oracle upper bound only; QA-TIED/VS-TIED interaction is not reproduced.",
            },
        },
        "population": {
            "items": len(items),
            "image_seeds": list(GENERATION_SEEDS),
            "physical_images": len(vanilla_manifest) + len(gold_visual_setup_manifest),
            "primary_vqa_trials": len(vanilla_manifest) * 2 + len(gold_visual_setup_manifest),
        },
        "comparison_pairing": {
            "item_key": "tab_id",
            "image_seed_key": "seed",
            "same_seed_uses_same_initial_latent_across_arms": True,
            "primary_vqa_trials_per_item_per_condition": 10,
            "vanilla_image_reuse": "one ambiguous-prompt physical image is paired with setup questions 0 and 1",
        },
        "artifacts": {
            "vanilla_manifest": vanilla_record.to_dict(relative_to=root),
            "tied_oracle_manifest": gold_visual_setup_record.to_dict(relative_to=root),
            "records": records_record.to_dict(relative_to=root),
        },
    }
    atomic_write_json(paths["contract"], contract, overwrite=False)
    summary = {
        "schema": INPUTS_SUMMARY_SCHEMA,
        "run_id": run_id,
        "dataset_id": DERIVED_DATASET_ID,
        "source_dataset_id": SOURCE_DATASET_ID,
        "items": len(items),
        "seeds": list(GENERATION_SEEDS),
        "condition_physical_images": {
            SD14_VANILLA_ARM: len(vanilla_manifest),
            SD14_GOLD_VISUAL_SETUP_ARM: len(gold_visual_setup_manifest),
        },
        "condition_primary_vqa_trials": {
            SD14_VANILLA_ARM: len(vanilla_manifest) * 2,
            SD14_GOLD_VISUAL_SETUP_ARM: len(gold_visual_setup_manifest),
        },
        "tab_category_items": dict(Counter(str(item["tab_category"]) for item in items)),
        "contract": repo_relative(paths["contract"]),
        "artifacts": contract["artifacts"],
    }
    atomic_write_json(paths["summary"], summary, overwrite=False)
    return summary


def driver_request(
    run_id: str, method: str, *, out_dir: Path | None = None, device: str = "cuda:0", num_shards: int = 1,
    shard_index: int = 0, batches: Sequence[int] | None = None, allow_dirty: bool = False,
) -> tuple[Any, TABBaselineDatasetProvenance]:
    """The ``SD14Run`` of one stock arm (``vanilla`` or ``gold_visual_setup``) and its dataset provenance.

    Inputs come from ``output/generation/<run_id>/baseline_inputs``; images go
    to ``out_dir`` (default: the run directory).
    """
    from ccg_diffusion.backbones.sd14.driver import SD14Run
    from ccg_diffusion.evaluation.tab import CONTRACT_FILENAME, DATASET_DIR

    arm, expected_total = BASELINE_ARMS[method]
    paths = baseline_input_paths(run_id)
    manifest = paths["vanilla_manifest" if arm == SD14_VANILLA_ARM else "tied_oracle_manifest"]
    provenance = load_tab_baseline_provenance(
        manifest, dataset_contract=DATASET_DIR / CONTRACT_FILENAME, generation_records=paths["records"],
        preparation_contract=paths["contract"], arm_id=arm)
    request = SD14Run(
        method=arm, prompt_manifest=manifest, out_dir=out_dir or paths["contract"].parent.parent,
        dataset=provenance, expected_total=expected_total, device=device, num_shards=num_shards,
        shard_index=shard_index, batches=None if batches is None else tuple(batches), allow_dirty=allow_dirty,
        extra_component_files=BASELINE_COMPONENT_FILES)
    return request, provenance


# --- Predicated Diffusion and Gold-visual setup + Predicated Diffusion -----------------------------------

PD_RUN_SCHEMA = "ccg_diffusion.tab_pp_pd_baselines_run.v1"
PD_TARGETS_FILE = "pd_spacy_sidecar.jsonl"
PD_EXPERIMENT = "tab_pp_pd_baselines"  # Job.experiment of the kernel jobs; not written to the image records
TRIAL_PROFILE = "tab-pp-pd-baselines"
# The T2I settings of the SD1.4 Predicated Diffusion baseline (recorded in the run manifest; the kernel
# reads them from there). binding_loss and leak_weight are the fixed L_PD of the kernel.
SD14_PD_SETTINGS = {"scale_factor": 20.0, "scale_range": [1.0, 0.5], "max_iter_to_alter": 25, "threshold_steps": [0],
                    "refinement_updates_per_threshold": 5, "binding_loss": "pd", "leak_weight": 0.3,
                    "max_update_ratio": 0.02}


def spacy_target_rows(jobs: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """One ``spacy_sd14`` guidance-target row per prompt key, exactly as the T2I Predicated Diffusion rows."""
    from ccg_diffusion.targets.tokenizer import load_sd14_tokenizer

    provider = SpacyProvider(load_sd14_tokenizer())
    result = {}
    for job in jobs:
        if job["key"] not in result:
            result[job["key"]] = provider.extract(job["prompt"]).target_row(prompt_id=job["key"], prompt=job["prompt"])
    # T2I sends alignment_error rows to stock generation; every TAB PP prompt parses, so fail closed instead.
    failed = sorted(key for key, row in result.items() if row["status"] != "ok")
    if failed:
        raise ValueError(f"spaCy PD extraction failed for {len(failed)} prompts: {failed[:3]}")
    return result


def pd_census(targets: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Prompt count, statuses and the prompts with possession or correspondence targets."""
    return {"prompts": len(targets), "status": dict(Counter(r["status"] for r in targets.values())),
            "with_possession": sum(bool(r["pd_indices"]["possession"]) for r in targets.values()),
            "with_corr": sum(bool(r["pd_indices"]["corr"]) for r in targets.values())}


def write_pd_run(
    run_id: str, *, backbone: str, jobs: Sequence[Mapping[str, Any]], targets: Mapping[str, Mapping[str, Any]],
    generation: Mapping[str, Any], dataset_verification: Mapping[str, Any],
    plans: Mapping[str, Mapping[str, Any]] | None = None, allow_dirty: bool = False,
) -> Path:
    """Write the jobs, guidance targets, Sana plans (if any) and manifest of a Predicated Diffusion run.

    The manifest seals the source tree (``provenance.source_tree_snapshot``);
    a dirty tree needs ``allow_dirty``. Returns the run directory.
    """
    source = source_tree_snapshot()
    if source["dirty"] and not allow_dirty:
        raise ValueError("commit the implementation before preparation (or pass --allow-dirty)")
    run_dir = repository_root() / "output/generation" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(run_dir / "metadata_jobs.jsonl", jobs, overwrite=False)
    atomic_write_jsonl(run_dir / PD_TARGETS_FILE, [targets[k] for k in sorted(targets)], overwrite=False)
    files = ["metadata_jobs.jsonl", PD_TARGETS_FILE]
    if plans:
        atomic_write_jsonl(run_dir / "plans.jsonl", [plans[k] for k in sorted(plans)], overwrite=False)
        files.append("plans.jsonl")
    manifest = {
        "schema": PD_RUN_SCHEMA, "run_id": run_id, "backbone": backbone, "arms": list(PD_ARMS), "source": source,
        "generation": dict(generation), "dataset": dict(dataset_verification),
        "spacy": {"provider": PROVIDER, "model": SPACY_MODEL, "census": pd_census(targets),
                  "parser_provenance": next(iter(targets.values()))["parser_provenance"]},
        "input_sha256": {repo_relative(ITEMS): sha256_file(ITEMS)},
        "artifact_sha256": {name: sha256_file(run_dir / name) for name in files},
        "counts": dict(Counter(j["arm"] for j in jobs)),
    }
    atomic_write_json(run_dir / "run_manifest.json", manifest, overwrite=False)
    return run_dir


def prepare_pd_run(run_id: str, *, dataset_verification: Mapping[str, Any], allow_dirty: bool = False) -> Path:
    """Prepare the SD1.4 Predicated Diffusion run of both arms (370 + 740 jobs)."""
    jobs = pd_baseline_jobs(list(load_pp_items().values()))
    c = SD14_GENERATION
    generation = {"model_id": c.model_id, "model_revision": c.model_revision, "height": c.height, "width": c.width,
                  "steps": c.steps, "guidance_scale": c.guidance_scale, "precision_profile": c.precision_profile,
                  "physical_batch_size": c.batch_size,
                  "padding_policy": "repeat_last_real_job_discard_padded_outputs",
                  "attention_res": c.attention_res, "attention_block_scope": c.attention_block_scope,
                  "max_latent_abs": SD14_MAX_LATENT_ABS, "scheduler_config_sha256": SD14_SCHEDULER_CONFIG_SHA256,
                  "determinism": dict(SD14_DETERMINISM), **SD14_PD_SETTINGS}
    return write_pd_run(run_id, backbone="sd14", jobs=jobs, targets=spacy_target_rows(jobs), generation=generation,
                        dataset_verification=dataset_verification, allow_dirty=allow_dirty)


def load_pd_run(
    run_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """``(manifest, jobs, guidance targets by key, Sana plans by key)``; sources and prepared files are unchanged."""
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    current = source_tree_snapshot()
    if current["files"] != manifest["source"]["files"] or current["dirty"] != manifest["source"]["dirty"]:
        raise ValueError("implementation differs from prepared source")
    for group, base in (("input_sha256", repository_root()), ("artifact_sha256", run_dir)):
        for name, digest in manifest[group].items():
            if sha256_file(base / name) != digest:
                raise ValueError(f"prepared file changed: {name}")
    targets = {r["prompt_id"]: r for r in read_jsonl(run_dir / PD_TARGETS_FILE)}
    plans = {p["key"]: p for p in read_jsonl(run_dir / "plans.jsonl")} if (run_dir / "plans.jsonl").exists() else {}
    return manifest, read_jsonl(run_dir / "metadata_jobs.jsonl"), targets, plans


def pd_records(output: Path, jobs: Sequence[Mapping[str, Any]], *, complete: bool) -> dict[str, dict[str, Any]]:
    """The recorded images of a run by image ID; every image file is unchanged.

    With ``complete`` every job has exactly one record that repeats the job's fields.
    """
    path = output / "metadata.jsonl"
    records = read_jsonl(path) if path.exists() else []
    result = {r["image_id"]: r for r in records}
    if len(result) != len(records):
        raise ValueError("duplicate generation metadata")
    root = repository_root()
    for record in result.values():
        if sha256_file(root / record["image_path"]) != record["image_sha256"]:
            raise ValueError(f"generated image changed: {record['image_path']}")
    if complete:
        if set(result) != {j["image_id"] for j in jobs}:
            raise ValueError("generation incomplete")
        for job in jobs:
            if any(result[job["image_id"]][k] != v for k, v in job.items()):
                raise ValueError(f"generation metadata differs from its job: {job['image_id']}")
    return result


def append_pd_record(output: Path, record: Mapping[str, Any]) -> None:
    with (output / "metadata.jsonl").open("a") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")


def write_generation_summary(output: Path, jobs: Sequence[Mapping[str, Any]], started: float) -> None:
    """Write ``generation_summary.json`` of a run whose every job has its image."""
    done = pd_records(output, jobs, complete=True)
    if not (output / "generation_summary.json").exists():
        atomic_write_json(output / "generation_summary.json", {
            "images": len(done), "metadata_sha256": sha256_file(output / "metadata.jsonl"),
            "seconds": time.monotonic() - started}, overwrite=False)


def validate_pd_diagnostics(row: Mapping[str, Any]) -> None:
    """Require strict finite guidance, 30 applied updates, 5 attention processors and the update cap.

    The same checks as ``pp_attachment.jobs.validate_diagnostics`` (baselines do not import pp_attachment).
    """
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


def generate_pd_baselines(
    run_dir: Path, *, device: str = "cuda:0", arms: Sequence[str] = PD_ARMS, batches: Sequence[int] | None = None,
    output_dir: Path | None = None,
) -> None:
    """Generate the pending images of ``arms`` of a prepared SD1.4 run through the guided denoising kernel.

    The run's physical batches of 4 are consecutive per arm (0-92 Predicated
    Diffusion, 93-277 Gold-visual setup + Predicated Diffusion; a short last
    batch repeats its last job); ``batches`` selects some of them.
    ``output_dir`` writes images and metadata of a verification subset
    outside the run directory. The kernel runs with the run's T2I settings
    (scale 20, no routes, no loss overrides, no Count Loss).
    """
    import diffusers
    import torch
    import transformers

    from ccg_diffusion.backbones.sd14.driver import initial_latents, prepare_sd14_runtime
    from ccg_diffusion.backbones.sd14.pipeline import generate_predicated_batch
    from ccg_diffusion.backbones.sd14.records import SD14_GENERATOR_RUNTIME_PINS, save_image_atomic
    from ccg_diffusion.guidance.latent_update import GenerationDiagnostics

    manifest, jobs, targets, _ = load_pd_run(run_dir)
    output = run_dir if output_dir is None else output_dir
    done = pd_records(output, jobs, complete=False)
    selected = [(index, batch) for index, batch in enumerate(pd_baseline_batches(jobs, "sd14"))
                if batch[0]["arm"] in arms]
    if batches is not None:
        if not set(batches) <= {index for index, _ in selected}:
            raise ValueError(f"batches must be physical batches of the arms {list(arms)}: "
                             f"{selected[0][0]}-{selected[-1][0]}")
        selected = [(index, batch) for index, batch in selected if index in set(batches)]
    torch.set_num_threads(4)
    runtime = prepare_sd14_runtime(device)
    c = manifest["generation"]
    if not runtime.determinism == c["determinism"] == dict(SD14_DETERMINISM):
        raise ValueError("SD1.4 determinism differs from the prepared run")
    versions = {"torch": torch.__version__, "diffusers": diffusers.__version__,
                "transformers": transformers.__version__}
    if any(SD14_GENERATOR_RUNTIME_PINS[k] != v for k, v in versions.items()):
        raise ValueError(f"runtime differs from the prepared contract: {versions}")
    if runtime.scheduler["config_sha256"] != c["scheduler_config_sha256"]:
        raise ValueError("scheduler configuration differs from the contract")
    pipe = runtime.pipe
    root = repository_root()
    run_id = run_dir.name
    started = time.monotonic()
    for index, batch_jobs in selected:
        if all(j["image_id"] in done for j in batch_jobs):
            continue
        padded = batch_jobs + [batch_jobs[-1]] * (4 - len(batch_jobs))
        kernel_jobs = [Job(experiment=PD_EXPERIMENT, prompt_id=j["image_id"], prompt=j["prompt"],
                           seed=j["seed"], source=run_id,
                           **{"attention_" + f + "_indices": targets[j["key"]]["pd_indices"][f] for f in PD_FIELDS})
                       for j in padded]
        latent, diagnostics = [], []
        for job in kernel_jobs:
            torch.manual_seed(job.seed)
            torch.cuda.manual_seed_all(job.seed)
            _, z, noise_hash = initial_latents(pipe, job, device=device, dtype=torch.float32, height=512,
                                               width=512, steps=c["steps"])
            latent.append(z)
            diagnostics.append(GenerationDiagnostics(
                initial_latent_sha256=noise_hash, generation_batch_id=f"{run_id}:{index}",
                valid_batch_size=len(batch_jobs), batch_padding_count=4 - len(batch_jobs)))
        tick = time.monotonic()
        result = generate_predicated_batch(
            pipe, kernel_jobs, device=device, steps=c["steps"], guidance_scale=c["guidance_scale"], height=512,
            width=512, attention_res=c["attention_res"], max_iter_to_alter=c["max_iter_to_alter"],
            scale_factor=c["scale_factor"], scale_range=tuple(c["scale_range"]),
            thresholds={s: 0. for s in c["threshold_steps"]},
            refinement_updates_per_threshold=c["refinement_updates_per_threshold"],
            initial_latents=torch.cat(latent), diagnostics=diagnostics, strict_finite=True,
            max_update_ratio=c["max_update_ratio"], max_latent_abs=c["max_latent_abs"],
            image_guidance=None, count_guidance=None)
        seconds = (time.monotonic() - tick) / len(batch_jobs)
        for i, job in enumerate(batch_jobs):
            if job["image_id"] in done:
                continue
            path = output / "images" / job["arm"] / (job["image_id"] + ".png")
            save_image_atomic(result.images[i], path, expected_size=(512, 512))
            record = {**job, "image_path": repo_relative(path, repository_root=root), "image_sha256": sha256_file(path),
                      "pd_indices": targets[job["key"]]["pd_indices"],
                      "loss_config": {"scale_factor": c["scale_factor"], "max_update_ratio": c["max_update_ratio"]},
                      "generation_diagnostics": diagnostics[i].as_dict(), "physical_batch_index": index,
                      "physical_batch_size": 4, "valid_batch_size": len(batch_jobs),
                      "source_commit": manifest["source"]["commit"], "determinism": runtime.determinism,
                      "seconds": seconds}
            validate_pd_diagnostics(record)
            append_pd_record(output, record)
            done[job["image_id"]] = record
        print(json.dumps({"batch": index, "completed": len(done), "total": len(jobs),
                          "seconds": time.monotonic() - started}), flush=True)
    if len(done) == len(jobs):
        write_generation_summary(output, jobs, started)


def merge_pd_run(run_dir: Path) -> Path:
    """Check a complete Predicated Diffusion run and write its trial plan (``output/vqa/<run>/trial_plan.jsonl``).

    Every image of one seed must share one initial latent; Sana images must
    show the 25 finite guidance updates (20 steps, 5 extra at step 0).
    """
    manifest, jobs, _, _ = load_pd_run(run_dir)
    records = pd_records(run_dir, jobs, complete=True)
    latents: dict[int, set[str]] = defaultdict(set)
    for record in records.values():
        diagnostics = record["generation_diagnostics" if manifest["backbone"] == "sd14" else "diagnostics"]
        latents[record["seed"]].add(diagnostics["initial_latent_sha256"])
        if manifest["backbone"] == "sana" and (diagnostics["updates"] != 25 or len(diagnostics["loss_trace"]) != 25
                                               or not all(math.isfinite(v) for v in diagnostics["loss_trace"])):
            raise ValueError(f"Sana PD guidance schedule differs: {record['image_id']}")
    if any(len(v) != 1 for v in latents.values()):
        raise ValueError("initial latents are not shared by the images of one seed")
    path = repository_root() / "output/vqa" / run_dir.name / "trial_plan.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_trial_plan(path, list(records.values()), profile=pp_profile(TRIAL_PROFILE), with_source_metadata=True)
    return path
