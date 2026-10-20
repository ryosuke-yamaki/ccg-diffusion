"""T2I-CompBench++ constants and the frozen full dataset (2,400 test prompts x seeds 42-51).

* Constants of the benchmark: categories, seeds, the official repository pin
  and the four evaluator weights (``scripts/setup/setup_t2i_compbench.py``
  installs exactly these).
* ``verify_dataset`` checks the frozen ``t2i-compbench-full-v1`` dataset
  against its contract: selection manifest sha256, prompt files, the three
  generation manifests and the exact file inventory.
* ``expected_population_identity`` recomputes the identity of one category's
  3,000 generated images (prompt order x seed order) that every evaluation
  manifest records.
"""
from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from ccg_diffusion import config
from ccg_diffusion.provenance import read_jsonl, repository_root, sha256_file

REPO_ROOT = repository_root()
CATEGORIES: tuple[str, ...] = config.T2I_COMPBENCH_CATEGORIES
SEEDS: tuple[int, ...] = config.T2I_COMPBENCH_SEEDS
PROMPTS_PER_CATEGORY = 300
IMAGES_PER_PROMPT = 10
DATASET_ID = config.T2I_COMPBENCH_DATASET_ID
DATASET_VERSION = config.T2I_COMPBENCH_DATASET_VERSION
DATASET_DIR = REPO_ROOT / config.T2I_COMPBENCH_DATASET_DIR
SOURCE_REPOSITORY = "https://github.com/Karine-Huang/T2I-CompBench"
SOURCE_COMMIT = "1b7094991a57f3c22abdd4f6e8ba6c1a15517073"
# Tracked record of the official clone, prompt files, evaluator venv and weights.
SOURCE_PIN_PATH = Path("output/analysis/t2i_compbench/t2icb_source_pin.json")
# ``link`` is relative to the clone root, or None for the torch.hub cache (BLIP resolves its
# checkpoint URL through timm's download_cached_file, which reads torch.hub.get_dir()).
EVALUATOR_WEIGHTS: tuple[dict[str, Any], ...] = (
    {
        "name": "Unified_learned_OCIM_RS200_6x+2x.pth",
        "url": "https://huggingface.co/shikunl/prismer/resolve/main/expert_weights/"
               "Unified_learned_OCIM_RS200_6x%2B2x.pth",
        "bytes": 483254220,
        "sha256": "b19a836811c0d0d7aebf04f83ff4ea1da77ccf57b85cd65819746c1fccbff258",
        "link": "UniDet_eval/experts/expert_weights/Unified_learned_OCIM_RS200_6x+2x.pth",
        "used_by": "2D_spatial_eval.py (UniDet RS200 detector)",
        "official_source": True,
    },
    {
        "name": "dpt_hybrid-midas-501f0c75.pt",
        "url": "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/"
               "dpt_hybrid-midas-501f0c75.pt",
        "bytes": 492757791,
        "sha256": "501f0c75b3bca7daec6b3682c5054c09b366765aef6fa3a09d03a5cb4b230853",
        "link": "UniDet_eval/experts/expert_weights/dpt_hybrid-midas-501f0c75.pt",
        "used_by": "3D_spatial_eval.py (MiDaS depth expert)",
        "official_source": True,
    },
    {
        "name": "Unified_learned_OCIM_R50_6x+2x.pth",
        # The official README points at a Google Drive id that now refuses anonymous access
        # ("Cannot retrieve the public link of the file"), and shikunl/prismer carries only the
        # RS200 weight.  This third-party mirror is byte-verified against the pinned sha256 and
        # structurally checked against the official R50 config; see `deviations` in the pin JSON.
        "url": "https://huggingface.co/datasets/nqbinh/misc_upload/resolve/main/"
               "Unified_learned_OCIM_R50_6x%2B2x.pth",
        "bytes": 287883795,
        "sha256": "630dbe1eb012a8985016b7b8ca96aa7f9db588ac8921604cf9eea81d6c9f6e21",
        "link": "UniDet_eval/experts/expert_weights/Unified_learned_OCIM_R50_6x+2x.pth",
        "used_by": "numeracy_eval.py, 3D_spatial_eval.py (UniDet R50 detector)",
        "official_source": False,
    },
    {
        "name": "model_base_vqa_capfilt_large.pth",
        "url": "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/"
               "model_base_vqa_capfilt_large.pth",
        "bytes": 1446244375,
        "sha256": "7a7d546209f1ccfa8b3cd3a0138c53e0d1e95e4a4bc280bef8f67e20fe4925ae",
        "link": None,
        "used_by": "BLIP_vqa.py (disentangled BLIP-VQA judge)",
        "official_source": True,
    },
)

DATASET_SCHEMA = "holsd.t2icb_full_dataset_contract.v1"
SELECTION_SCHEMA = "holsd.t2icb_subset.v1"
SELECTION_POLICY = "equal_interval"
SAMPLE_SEED = 20260812
EXPECTED_PROMPT_COUNT = len(CATEGORIES) * PROMPTS_PER_CATEGORY
EXPECTED_IMAGE_COUNT = EXPECTED_PROMPT_COUNT * len(SEEDS)
EXPECTED_SELECTION_SHA256 = (
    "959de73018528db02bccd911b76b691f5a48241924471e2301804250563eb7c2"
)
CANONICAL_RELATIVE_ROOT = config.T2I_COMPBENCH_DATASET_DIR
CONTRACT_NAME = "dataset_contract.json"
SELECTION_NAME = "selection_manifest.json"
MANIFEST_KEYS = (
    "selected_prompt_manifest",
    "prompt_only_manifest",
    "independent_seed_manifest",
)
PROMPT_ROW_FIELDS = frozenset({"experiment", "prompt_id", "prompt", "seed"})


class T2ICompBenchError(RuntimeError):
    """A T2I-CompBench++ dataset, export, evaluation or result violates its contract."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise T2ICompBenchError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise T2ICompBenchError(f"required JSON file not found: {path}")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except json.JSONDecodeError as exc:
        raise T2ICompBenchError(f"invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise T2ICompBenchError(f"expected a JSON object: {path}")
    return payload


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise T2ICompBenchError(f"required JSONL file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
            except json.JSONDecodeError as exc:
                raise T2ICompBenchError(
                    f"invalid JSONL row at {path}:{line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise T2ICompBenchError(
                    f"non-object JSONL row at {path}:{line_number}"
                )
            rows.append(row)
    return rows


def require_equal(actual: Any, expected: Any, *, label: str) -> None:
    if actual != expected:
        raise T2ICompBenchError(
            f"{label} differs from the frozen contract: {actual!r} != {expected!r}"
        )


def verify_hash(path: Path, expected: Any, *, label: str) -> None:
    if not path.is_file():
        raise T2ICompBenchError(f"{label} not found: {path}")
    require_equal(sha256_file(path), expected, label=f"{label} sha256")


def resolve_repo_path(repo_root: Path, relative_path: Any, *, label: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise T2ICompBenchError(f"{label} must be a repository-relative path")
    path = (repo_root / relative_path).resolve()
    try:
        path.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise T2ICompBenchError(f"{label} escapes the repository") from exc
    return path


def verify_contract(
    contract: Mapping[str, Any],
    *,
    dataset_dir: Path,
    repo_root: Path,
    enforce_canonical_path: bool,
) -> dict[str, Any]:
    require_equal(contract.get("schema"), DATASET_SCHEMA, label="contract schema")
    require_equal(contract.get("dataset_id"), DATASET_ID, label="dataset id")
    require_equal(contract.get("version"), DATASET_VERSION, label="dataset version")
    require_equal(contract.get("status"), "frozen", label="dataset status")
    require_equal(contract.get("root"), CANONICAL_RELATIVE_ROOT, label="contract root")
    if enforce_canonical_path:
        require_equal(
            dataset_dir.resolve(),
            (repo_root / CANONICAL_RELATIVE_ROOT).resolve(),
            label="dataset directory",
        )
    require_equal(
        contract.get("population"),
        {
            "categories": list(CATEGORIES),
            "prompts_per_category": PROMPTS_PER_CATEGORY,
            "prompt_count": EXPECTED_PROMPT_COUNT,
            "selection_policy": "all official prompts in source order",
            "image_seeds": list(SEEDS),
            "images_per_prompt": len(SEEDS),
            "images_per_arm": EXPECTED_IMAGE_COUNT,
        },
        label="contract population",
    )
    selection_record = contract.get("selection_manifest")
    if not isinstance(selection_record, dict):
        raise T2ICompBenchError("contract selection_manifest must be an object")
    require_equal(
        selection_record,
        {
            "path": f"{CANONICAL_RELATIVE_ROOT}/{SELECTION_NAME}",
            "sha256": EXPECTED_SELECTION_SHA256,
        },
        label="contract selection manifest",
    )
    source = contract.get("source")
    if not isinstance(source, dict):
        raise T2ICompBenchError("contract source must be an object")
    pin_path = resolve_repo_path(repo_root, source.get("pin_path"), label="source pin")
    verify_hash(pin_path, source.get("pin_sha256"), label="source pin")
    source_pin = load_json(pin_path)
    require_equal(source.get("repository"), source_pin.get("repo"), label="source repository")
    require_equal(source.get("commit"), source_pin.get("commit"), label="source commit")
    return source_pin


def _validate_prompt_row(row: Mapping[str, Any], *, label: str) -> None:
    require_equal(set(row), PROMPT_ROW_FIELDS, label=f"{label} fields")
    for field in ("experiment", "prompt_id", "prompt"):
        if not isinstance(row[field], str) or not row[field]:
            raise T2ICompBenchError(f"{label}.{field} must be non-empty text")
    if isinstance(row["seed"], bool) or not isinstance(row["seed"], int):
        raise T2ICompBenchError(f"{label}.seed must be an integer")


def verify_selection(
    selection: Mapping[str, Any],
    *,
    dataset_dir: Path,
    source_pin: Mapping[str, Any],
) -> set[Path]:
    require_equal(selection.get("schema"), SELECTION_SCHEMA, label="selection schema")
    require_equal(selection.get("selection_policy"), SELECTION_POLICY, label="selection policy")
    require_equal(selection.get("sample_seed"), SAMPLE_SEED, label="sample seed")
    require_equal(selection.get("categories"), list(CATEGORIES), label="categories")
    require_equal(selection.get("prompts_per_category"), PROMPTS_PER_CATEGORY, label="prompts per category")
    require_equal(selection.get("prompt_count"), EXPECTED_PROMPT_COUNT, label="prompt count")
    require_equal(selection.get("seeds"), list(SEEDS), label="image seeds")
    require_equal(selection.get("images_per_prompt"), len(SEEDS), label="images per prompt")
    require_equal(selection.get("generation_row_count"), EXPECTED_IMAGE_COUNT, label="generation row count")

    source_val = source_pin.get("val_txt")
    if not isinstance(source_val, dict):
        raise T2ICompBenchError("source pin val_txt must be an object")
    require_equal(
        selection.get("source_txt_sha256"),
        {category: source_val[category]["sha256"] for category in CATEGORIES},
        label="source prompt hashes",
    )

    expected_files = {dataset_dir / CONTRACT_NAME, dataset_dir / SELECTION_NAME}
    selected = selection.get("selected")
    prompt_hashes = selection.get("prompt_files_sha256")
    if not isinstance(selected, dict) or not isinstance(prompt_hashes, dict):
        raise T2ICompBenchError("selection rows and prompt hashes must be objects")

    expected_base_rows: list[dict[str, Any]] = []
    for category in CATEGORIES:
        rows = selected.get(category)
        if not isinstance(rows, list):
            raise T2ICompBenchError(f"selection rows missing for {category}")
        require_equal(len(rows), PROMPTS_PER_CATEGORY, label=f"{category} row count")
        require_equal(
            [row.get("line_index") for row in rows],
            list(range(PROMPTS_PER_CATEGORY)),
            label=f"{category} source positions",
        )
        expected_ids = [
            f"t2icb_{category}_{position:04d}"
            for position in range(PROMPTS_PER_CATEGORY)
        ]
        require_equal(
            [row.get("prompt_id") for row in rows],
            expected_ids,
            label=f"{category} prompt ids",
        )
        prompt_path = dataset_dir / "prompts" / f"{category}_val.txt"
        expected_files.add(prompt_path)
        verify_hash(prompt_path, prompt_hashes.get(prompt_path.name), label=f"{category} prompt file")
        prompts = prompt_path.read_text(encoding="utf-8").splitlines()
        require_equal(prompts, [row.get("prompt") for row in rows], label=f"{category} prompt text")
        expected_base_rows.extend(
            {
                "experiment": f"t2icb_{category}",
                "prompt_id": prompt_id,
                "prompt": prompt,
                "seed": 0,
            }
            for prompt_id, prompt in zip(expected_ids, prompts)
        )

    for prompt_name, expected_hash in prompt_hashes.items():
        prompt_path = dataset_dir / "prompts" / prompt_name
        expected_files.add(prompt_path)
        verify_hash(prompt_path, expected_hash, label=f"prompt file {prompt_name}")

    manifest_rows: dict[str, list[dict[str, Any]]] = {}
    for key in MANIFEST_KEYS:
        record = selection.get(key)
        if not isinstance(record, dict):
            raise T2ICompBenchError(f"selection {key} must be an object")
        relative_path = record.get("path")
        if not isinstance(relative_path, str) or Path(relative_path).name != relative_path:
            raise T2ICompBenchError(f"selection {key}.path must be a root file name")
        path = dataset_dir / relative_path
        expected_files.add(path)
        verify_hash(path, record.get("sha256"), label=key)
        rows = load_jsonl(path)
        require_equal(len(rows), record.get("rows"), label=f"{key} rows")
        for row_number, row in enumerate(rows, start=1):
            _validate_prompt_row(row, label=f"{key}:{row_number}")
        manifest_rows[key] = rows

    require_equal(
        manifest_rows["selected_prompt_manifest"],
        expected_base_rows,
        label="selected prompt manifest content",
    )
    expected_blind: list[dict[str, Any]] = []
    expected_generation: list[dict[str, Any]] = []
    for base in expected_base_rows:
        for seed in SEEDS:
            prompt_id = f"{base['prompt_id']}_s{seed}"
            expected_blind.append({**base, "prompt_id": prompt_id})
            expected_generation.append({**base, "prompt_id": prompt_id, "seed": seed})
    require_equal(manifest_rows["prompt_only_manifest"], expected_blind, label="blind manifest content")
    require_equal(
        manifest_rows["independent_seed_manifest"],
        expected_generation,
        label="generation manifest content",
    )
    return expected_files


def verify_dataset(
    dataset_dir: Path = DATASET_DIR,
    *,
    repo_root: Path = REPO_ROOT,
    enforce_canonical_path: bool = True,
) -> dict[str, Any]:
    """Verify the frozen full dataset against its contract; return the verification record."""

    dataset_dir = dataset_dir.resolve()
    repo_root = repo_root.resolve()
    contract = load_json(dataset_dir / CONTRACT_NAME)
    source_pin = verify_contract(
        contract,
        dataset_dir=dataset_dir,
        repo_root=repo_root,
        enforce_canonical_path=enforce_canonical_path,
    )
    selection_path = dataset_dir / SELECTION_NAME
    verify_hash(selection_path, EXPECTED_SELECTION_SHA256, label="selection manifest")
    expected_files = verify_selection(
        load_json(selection_path),
        dataset_dir=dataset_dir,
        source_pin=source_pin,
    )
    actual_files = {path for path in dataset_dir.rglob("*") if path.is_file()}
    require_equal(
        sorted(path.relative_to(dataset_dir).as_posix() for path in actual_files),
        sorted(path.relative_to(dataset_dir).as_posix() for path in expected_files),
        label="dataset file inventory",
    )
    return {
        "schema": DATASET_SCHEMA,
        "dataset_id": DATASET_ID,
        "version": DATASET_VERSION,
        "selection_manifest_sha256": EXPECTED_SELECTION_SHA256,
        "categories": len(CATEGORIES),
        "prompts": EXPECTED_PROMPT_COUNT,
        "images_per_arm": EXPECTED_IMAGE_COUNT,
        "files": len(actual_files),
        "status": "verified",
    }


def load_generation_rows(dataset_dir: Path = DATASET_DIR) -> list[dict[str, Any]]:
    """The 24,000 rows of ``independent_seed_manifest.jsonl`` (one per prompt and seed)."""

    return read_jsonl(dataset_dir / "independent_seed_manifest.jsonl")


@functools.lru_cache(maxsize=None)
def expected_population_identity(category: str, dataset_dir: Path = DATASET_DIR) -> str:
    """sha256 of one category's image population in the official sample order.

    Rows of ``t2icb_<category>`` are grouped by base prompt in first-seen order;
    image ``index = prompt_position * 10 + seed_position``. The serialization is
    the one every exported ``metric_stage_manifest.json`` records.
    """

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in load_generation_rows(dataset_dir):
        if row["experiment"] == f"t2icb_{category}":
            groups.setdefault(row["prompt_id"].rsplit("_s", 1)[0], []).append(row)
    population = [
        {"prompt_id": row["prompt_id"], "prompt": row["prompt"], "seed": row["seed"],
         "index": prompt_position * IMAGES_PER_PROMPT + seed_position}
        for prompt_position, rows in enumerate(groups.values())
        for seed_position, row in enumerate(rows)
    ]
    return hashlib.sha256(
        json.dumps(population, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
