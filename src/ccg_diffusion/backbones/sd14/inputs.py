"""Inputs of the SD1.4 batch driver: dataset provenance, prepared jobs and guidance-target files.

* ``load_t2icb_dataset_provenance`` verifies the frozen T2I-CompBench++
  dataset contract and that the image manifest reproduces the sealed prompt
  selection and seed order exactly; ``T2ICBDatasetProvenance`` then supplies
  the dataset fields of the run lock and of every metadata row. TAB runs pass
  ``baselines.tab_sd14.TABBaselineDatasetProvenance``; both satisfy
  ``DatasetProvenance``.
* ``load_prepared_jobs`` joins the image manifest with a guidance-target file
  (one row per image, sealed to the SD1.4 tokenizer and to its provider) and
  returns one ``PreparedJob`` per image. A row whose guidance targets could not
  be derived is accepted only under the ``stock_vanilla`` policy of the
  Predicated Diffusion baseline; the image is then generated without guidance.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from ccg_diffusion.backbones.jobs import Job
from ccg_diffusion.config import SD14_ARM_PROVIDERS
from ccg_diffusion.provenance import repo_relative, sha256_file, sha256_text
from ccg_diffusion.provenance import digest as sha256_json
from ccg_diffusion.provenance import repository_root as checkout_root
from ccg_diffusion.targets.guidance_targets import require_target_file_metadata, require_target_row_identity
from ccg_diffusion.targets.pd_indices import PD_FIELDS

SD14_PD_SOURCE_PROVENANCE_SCHEMA = "ccg_diffusion.git_head_components.v1"
# Guidance-target statuses that may fall back to stock generation, per provider.
STOCK_FALLBACK_STATUSES_BY_PROVIDER = {"spacy_sd14": frozenset({"alignment_error"})}
INDEX_FAILURE_POLICIES = ("error", "stock_vanilla")
PROMPT_FIELDS = frozenset({"experiment", "prompt_id", "prompt", "seed"})
T2ICB_DATASET_CONTRACT_SCHEMAS = frozenset({"holsd.t2icb_full_dataset_contract.v1"})


class GenerationContractError(ValueError):
    """Raised when an input or output violates the locked experiment contract."""


# --- prepared jobs -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PreparedJob:
    """One manifest image joined with its guidance-target row (``provider`` is None for stock arms)."""

    job: Job
    prompt_sha256: str
    program_sha256: str | None
    pd_indices_sha256: str | None
    provider: str | None
    extraction_status: str | None = None
    extraction_error: str | None = None
    sidecar_row_sha256: str | None = None
    index_failure_policy: str = "error"

    @property
    def guidance_mode(self) -> str:
        """Return the effective per-image generation path."""

        if self.provider is None:
            return "stock_vanilla"
        if self.extraction_status == "ok":
            return "token_predicated"
        return "stock_vanilla_fallback"

    @property
    def guidance_applied(self) -> bool:
        return self.guidance_mode == "token_predicated"


def prepared_job_from_target_row(
    row: Mapping[str, Any] | None,
    *,
    experiment: str,
    prompt_id: str,
    prompt: str,
    seed: int,
    index_failure_policy: str = "error",
) -> PreparedJob:
    """Join one image with its validated guidance-target row (``None`` for stock arms)."""

    digest = sha256_text(prompt)
    if row is None:
        pd = {field: [] for field in PD_FIELDS}
        provider = program_digest = indices_digest = None
        status = error = row_digest = None
        index_failure_policy = "error"
    else:
        if row["prompt_sha256"] != digest:
            raise GenerationContractError(
                f"prompt hash mismatch for {prompt_id}: manifest={digest}, "
                f"sidecar={row['prompt_sha256']}"
            )
        pd = row["pd_indices"]
        provider = row["provider"]
        status = str(row["status"])
        # A failure row has no program and carries the parser error.
        program_digest = row["program_sha256"] if status == "ok" else None
        error = row["error"] if status != "ok" else None
        indices_digest = sha256_json(pd) if program_digest is not None else None
        row_digest = sha256_json(row)
    return PreparedJob(
        job=Job(
            experiment=experiment,
            prompt_id=prompt_id,
            prompt=prompt,
            seed=seed,
            attention_exist_indices=pd["exist"],
            attention_corr_indices=pd["corr"],
            attention_leak_indices=pd["leak"],
            attention_possession_indices=pd["possession"],
            source=provider or "prompt_only",
        ),
        prompt_sha256=digest,
        program_sha256=program_digest,
        pd_indices_sha256=indices_digest,
        provider=provider,
        extraction_status=status,
        extraction_error=error,
        sidecar_row_sha256=row_digest,
        index_failure_policy=index_failure_policy,
    )


def _require_target_file_identity(target_file: Path, *, expected_provider: str) -> None:
    """Reject guidance-target files compiled under any other tokenizer identity."""

    metadata_path = target_file.with_name(target_file.name + ".meta.json")
    if not metadata_path.is_file():
        raise GenerationContractError(
            f"SD1.4 sidecar metadata is required: {metadata_path}"
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GenerationContractError(
            f"invalid SD1.4 sidecar metadata: {metadata_path}"
        ) from exc
    if not isinstance(metadata, dict):
        raise GenerationContractError("SD1.4 sidecar metadata must be an object")
    try:
        require_target_file_metadata(
            metadata,
            expected_provider=expected_provider,
        )
    except ValueError as exc:
        raise GenerationContractError(str(exc)) from exc


def _require_target_row_identity(
    row: Mapping[str, Any],
    *,
    location: str,
    expected_provider: str,
) -> None:
    try:
        require_target_row_identity(
            row,
            expected_provider=expected_provider,
        )
    except ValueError as exc:
        raise GenerationContractError(f"{location}: {exc}") from exc


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise GenerationContractError(f"missing {label}: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise GenerationContractError(
                    f"blank row in {label} at {path}:{line_number}"
                )
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise GenerationContractError(
                    f"invalid JSON in {label} at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise GenerationContractError(
                    f"non-object row in {label} at {path}:{line_number}"
                )
            rows.append(row)
    return rows


def _require_text(value: Any, *, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise GenerationContractError(f"{location} must be a non-empty string")
    return value


def _validate_pd_indices(
    raw: Any, *, location: str, allow_empty: bool = False
) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != set(PD_FIELDS):
        raise GenerationContractError(
            f"{location} must contain exactly {sorted(PD_FIELDS)}"
        )

    exist = raw["exist"]
    if not isinstance(exist, list) or any(
        type(index) is not int or index < 1 for index in exist
    ):
        raise GenerationContractError(
            f"{location}.exist must be a list of positive integers"
        )

    normalized: dict[str, Any] = {"exist": list(exist)}
    for field in ("corr", "leak", "possession"):
        pairs = raw[field]
        if not isinstance(pairs, list):
            raise GenerationContractError(f"{location}.{field} must be a list")
        normalized_pairs: list[list[int]] = []
        for pair_index, pair in enumerate(pairs):
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or any(type(index) is not int or index < 1 for index in pair)
            ):
                raise GenerationContractError(
                    f"{location}.{field}[{pair_index}] must be two positive integers"
                )
            normalized_pairs.append(list(pair))
        normalized[field] = normalized_pairs

    flattened = [*normalized["exist"]]
    for field in ("corr", "leak", "possession"):
        flattened.extend(index for pair in normalized[field] for index in pair)
    if not flattened and not allow_empty:
        raise GenerationContractError(f"{location} contains no guidance indices")
    return normalized


def _validate_target_row(
    row: Mapping[str, Any], *, location: str, expected_provider: str, index_failure_policy: str,
) -> None:
    """Check one guidance-target row: provider, tokenizer identity, digests, and the failure policy."""

    provider = _require_text(
        row.get("provider"), location=f"{location}.provider"
    )
    if provider != expected_provider:
        raise GenerationContractError(
            f"{location} provider {provider!r} does not match {expected_provider!r}"
        )
    _require_target_row_identity(
        row,
        location=location,
        expected_provider=expected_provider,
    )
    status = _require_text(
        row.get("status"), location=f"{location}.status"
    )
    digest = _require_text(
        row.get("prompt_sha256"), location=f"{location}.prompt_sha256"
    )
    if len(digest) != 64 or any(
        char not in "0123456789abcdef" for char in digest
    ):
        raise GenerationContractError(
            f"{location}.prompt_sha256 is not lowercase SHA-256"
        )
    if status == "ok":
        if row.get("failure_code") is not None:
            raise GenerationContractError(
                f"{location}.failure_code must be null for status=ok"
            )
        program_digest = _require_text(
            row.get("program_sha256"),
            location=f"{location}.program_sha256",
        )
        if len(program_digest) != 64 or any(
            char not in "0123456789abcdef" for char in program_digest
        ):
            raise GenerationContractError(
                f"{location}.program_sha256 is not lowercase SHA-256"
            )
        indices = _validate_pd_indices(
            row.get("pd_indices"), location=f"{location}.pd_indices"
        )
        computed_program_digest = sha256_json(indices)
        if program_digest != computed_program_digest:
            raise GenerationContractError(
                f"{location}.program_sha256 does not match pd_indices: "
                f"declared={program_digest}, computed={computed_program_digest}"
            )
        return
    if index_failure_policy != "stock_vanilla":
        raise GenerationContractError(
            f"{location} is not usable: status={status!r}"
        )
    allowed_statuses = STOCK_FALLBACK_STATUSES_BY_PROVIDER.get(
        expected_provider, frozenset()
    )
    if status not in allowed_statuses:
        raise GenerationContractError(
            f"{location} status {status!r} is not an allowed "
            f"fallback for provider {expected_provider!r}"
        )
    if row.get("program_sha256") is not None:
        raise GenerationContractError(
            f"{location}.program_sha256 must be null for parser failure"
        )
    indices = _validate_pd_indices(
        row.get("pd_indices"),
        location=f"{location}.pd_indices",
        allow_empty=True,
    )
    if any(indices[field] for field in PD_FIELDS):
        raise GenerationContractError(
            f"{location}.pd_indices must be empty for parser failure"
        )
    _require_text(
        row.get("error"), location=f"{location}.error"
    )


def load_prepared_jobs(
    target_file: Path | None,
    prompt_manifest: Path,
    *,
    method: str,
    index_failure_policy: str,
    dataset: "DatasetProvenance",
) -> list[PreparedJob]:
    """Load strict prompt-only rows and join a provider-locked guidance-target file."""

    if method not in SD14_ARM_PROVIDERS:
        raise GenerationContractError(f"unsupported method: {method}")
    expected_provider = SD14_ARM_PROVIDERS[method]
    if expected_provider is None and target_file is not None:
        raise GenerationContractError(f"{method} must not receive a guidance-target file")
    if expected_provider is not None and target_file is None:
        raise GenerationContractError(f"{method} requires a guidance-target file")
    if index_failure_policy not in INDEX_FAILURE_POLICIES:
        raise GenerationContractError(
            "index_failure_policy must be error or stock_vanilla"
        )
    if expected_provider is None and index_failure_policy != "error":
        raise GenerationContractError(
            f"{method} does not accept an index failure policy"
        )

    prompt_rows = _read_jsonl(prompt_manifest, label="prompt-only manifest")
    prompt_ids: set[str] = set()
    prompt_keys: set[tuple[str, str, int]] = set()
    for row_index, row in enumerate(prompt_rows):
        location = f"prompt-only manifest row {row_index}"
        if set(row) != PROMPT_FIELDS:
            raise GenerationContractError(
                f"{location} must contain exactly {sorted(PROMPT_FIELDS)}; "
                f"found {sorted(row)}"
            )
        _require_text(row["experiment"], location=f"{location}.experiment")
        prompt_id = _require_text(row["prompt_id"], location=f"{location}.prompt_id")
        _require_text(row["prompt"], location=f"{location}.prompt")
        if type(row["seed"]) is not int or row["seed"] < 0:
            raise GenerationContractError(
                f"{location}.seed must be a non-negative integer"
            )
        key = (row["experiment"], prompt_id, row["seed"])
        if key in prompt_keys:
            raise GenerationContractError(f"duplicate prompt key: {key}")
        if prompt_id in prompt_ids:
            raise GenerationContractError(
                f"prompt_id must be unique for sidecar joining: {prompt_id}"
            )
        prompt_keys.add(key)
        prompt_ids.add(prompt_id)

    target_rows: dict[str, dict[str, Any]] = {}
    if target_file is not None:
        _require_target_file_identity(target_file, expected_provider=expected_provider)
        for row_index, row in enumerate(_read_jsonl(target_file, label="PD sidecar")):
            location = f"PD sidecar row {row_index}"
            prompt_id = _require_text(
                row.get("prompt_id"), location=f"{location}.prompt_id"
            )
            if prompt_id in target_rows:
                raise GenerationContractError(
                    f"duplicate PD sidecar prompt_id: {prompt_id}"
                )
            _validate_target_row(
                row, location=location, expected_provider=expected_provider,
                index_failure_policy=index_failure_policy,
            )
            target_rows[prompt_id] = row
        missing = sorted(prompt_ids - set(target_rows))
        extra = sorted(set(target_rows) - prompt_ids)
        if missing or extra:
            raise GenerationContractError(
                f"PD sidecar key mismatch: missing={missing[:10]}, extra={extra[:10]}"
            )

    prepared = [
        prepared_job_from_target_row(
            target_rows.get(row["prompt_id"]),
            experiment=row["experiment"],
            prompt_id=row["prompt_id"],
            prompt=row["prompt"],
            seed=row["seed"],
            index_failure_policy=index_failure_policy,
        )
        for row in prompt_rows
    ]
    try:
        dataset.require_prepared_jobs(prepared)
    except ValueError as exc:
        raise GenerationContractError(str(exc)) from exc
    return prepared


# --- dataset provenance --------------------------------------------------------------------------------


class DatasetProvenance(Protocol):
    """Dataset identity of a run: run-lock fields, per-image fields and a population check."""

    prompt_manifest_path: Path

    def run_record(self) -> dict[str, Any]: ...

    def image_record(self, job: Job) -> dict[str, Any]: ...

    def require_prepared_jobs(self, prepared_jobs: Sequence[PreparedJob]) -> None: ...


@dataclass(frozen=True)
class T2ICBDatasetProvenance:
    """Sealed T2I-CompBench++ dataset identity and per-image source rows."""

    dataset_id: str
    dataset_version: int
    contract_path: Path
    contract_sha256: str
    selection_manifest_path: Path
    selection_manifest_sha256: str
    source_repository: str
    source_commit: str
    source_pin_path: Path
    source_pin_sha256: str
    image_records: Mapping[tuple[str, str, int], Mapping[str, Any]]
    prompt_manifest_path: Path

    def run_record(self) -> dict[str, Any]:
        """Return the immutable dataset record stored in every shard lock."""

        return {
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "dataset_contract": repo_relative(self.contract_path),
            "dataset_contract_sha256": self.contract_sha256,
            "selection_manifest": repo_relative(self.selection_manifest_path),
            "selection_manifest_sha256": self.selection_manifest_sha256,
            "source_repository": self.source_repository,
            "source_commit": self.source_commit,
            "source_pin": repo_relative(self.source_pin_path),
            "source_pin_sha256": self.source_pin_sha256,
        }

    def image_record(self, job: Job) -> dict[str, Any]:
        """Resolve one manifest job to its sealed original prompt line."""

        key = (job.experiment, job.prompt_id, job.seed)
        record = self.image_records.get(key)
        if record is None:
            raise GenerationContractError(
                f"job is absent from sealed dataset selection: {key!r}"
            )
        return dict(record)

    def require_prepared_jobs(self, prepared_jobs: Sequence[PreparedJob]) -> None:
        """Nothing to check: the manifest was verified against the sealed selection when loaded."""


def _load_json_object_no_duplicates(path: Path, *, label: str) -> dict[str, Any]:
    """Read one JSON object while rejecting duplicate keys."""

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise GenerationContractError(
                    f"duplicate key {key!r} in {label}: {path}"
                )
            result[key] = value
        return result

    if not path.is_file():
        raise GenerationContractError(f"{label} not found: {path}")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise GenerationContractError(f"invalid {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise GenerationContractError(f"{label} must be a JSON object: {path}")
    return payload


def _resolve_dataset_path(value: Any, *, base: Path, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise GenerationContractError(f"{label} must be a non-empty path")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_t2icb_dataset_provenance(  # noqa: C901
    prompt_manifest: Path,
    *,
    repository_root: Path | None = None,
) -> T2ICBDatasetProvenance:
    """Verify the frozen dataset contract and exact generation population.

    The original official line index is taken from the content-addressed
    selection manifest, never inferred only from a prompt-id suffix.  The
    complete image manifest is then reconstructed from those selected rows and
    the sealed seed order, making the mapping an executable precondition of
    generation rather than a descriptive note.
    """

    resolved_manifest = prompt_manifest.resolve()
    dataset_root = resolved_manifest.parent
    root = (
        repository_root.resolve()
        if repository_root is not None
        else checkout_root()
    )
    contract_path = dataset_root / "dataset_contract.json"
    contract = _load_json_object_no_duplicates(
        contract_path, label="T2I-CompBench++ dataset contract"
    )
    if contract.get("schema") not in T2ICB_DATASET_CONTRACT_SCHEMAS:
        raise GenerationContractError(
            f"unsupported T2I-CompBench++ dataset contract schema: "
            f"{contract.get('schema')!r}"
        )
    if contract.get("status") != "frozen":
        raise GenerationContractError("T2I-CompBench++ dataset must be frozen")
    dataset_id = contract.get("dataset_id")
    dataset_version = contract.get("version")
    if not isinstance(dataset_id, str) or not dataset_id:
        raise GenerationContractError("dataset_id must be non-empty")
    if type(dataset_version) is not int or dataset_version < 1:
        raise GenerationContractError("dataset version must be a positive integer")
    recorded_root = _resolve_dataset_path(
        contract.get("root"), base=root, label="dataset contract root"
    )
    if recorded_root != dataset_root:
        raise GenerationContractError(
            f"dataset contract root is {recorded_root}, expected {dataset_root}"
        )

    selection_record = contract.get("selection_manifest")
    if not isinstance(selection_record, dict):
        raise GenerationContractError("dataset contract has no selection manifest")
    selection_path = _resolve_dataset_path(
        selection_record.get("path"), base=root, label="selection manifest path"
    )
    selection_sha256 = selection_record.get("sha256")
    if not isinstance(selection_sha256, str) or len(selection_sha256) != 64:
        raise GenerationContractError("selection manifest SHA-256 is invalid")
    if not selection_path.is_file() or sha256_file(selection_path) != selection_sha256:
        raise GenerationContractError(
            f"selection manifest is absent or changed: {selection_path}"
        )
    selection = _load_json_object_no_duplicates(
        selection_path, label="T2I-CompBench++ selection manifest"
    )
    population = contract.get("population")
    if not isinstance(population, dict):
        raise GenerationContractError("dataset population contract is missing")
    categories = population.get("categories")
    seeds = population.get("image_seeds")
    selected = selection.get("selected")
    if not isinstance(categories, list) or not all(
        isinstance(category, str) and category for category in categories
    ):
        raise GenerationContractError("dataset categories are invalid")
    if not isinstance(seeds, list) or any(type(seed) is not int for seed in seeds):
        raise GenerationContractError("dataset image seeds are invalid")
    if seeds != list(range(42, 52)) or len(set(seeds)) != len(seeds):
        raise GenerationContractError(
            f"six-arm dataset requires ordered image seeds 42--51, got {seeds!r}"
        )
    if not isinstance(selected, dict) or set(selected) != set(categories):
        raise GenerationContractError("selection categories do not match the contract")

    generation_record = selection.get("independent_seed_manifest")
    if not isinstance(generation_record, dict):
        raise GenerationContractError("selection manifest lacks generation input")
    recorded_generation = _resolve_dataset_path(
        generation_record.get("path"),
        base=selection_path.parent,
        label="selected generation manifest path",
    )
    if recorded_generation != resolved_manifest:
        raise GenerationContractError(
            f"selected generation manifest is {recorded_generation}, "
            f"but runner received {resolved_manifest}"
        )
    manifest_sha256 = sha256_file(resolved_manifest)
    if generation_record.get("sha256") != manifest_sha256:
        raise GenerationContractError("generation manifest SHA differs from selection seal")

    expected_rows: list[dict[str, Any]] = []
    image_records: dict[tuple[str, str, int], dict[str, Any]] = {}
    for category in categories:
        category_rows = selected.get(category)
        if not isinstance(category_rows, list):
            raise GenerationContractError(
                f"selection rows for {category!r} must be a list"
            )
        for position, raw in enumerate(category_rows):
            if not isinstance(raw, dict):
                raise GenerationContractError(
                    f"selection row {category}:{position} is not an object"
                )
            base_prompt_id = raw.get("prompt_id")
            prompt = raw.get("prompt")
            source_line_index = raw.get("line_index")
            if (
                not isinstance(base_prompt_id, str)
                or not base_prompt_id
                or not isinstance(prompt, str)
                or not prompt
                or type(source_line_index) is not int
                or source_line_index < 0
            ):
                raise GenerationContractError(
                    f"invalid selection row at {category}:{position}"
                )
            experiment = f"t2icb_{category}"
            for seed in seeds:
                prompt_id = f"{base_prompt_id}_s{seed}"
                row = {
                    "experiment": experiment,
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "seed": seed,
                }
                key = (experiment, prompt_id, seed)
                if key in image_records:
                    raise GenerationContractError(
                        f"duplicate selected generation key: {key!r}"
                    )
                expected_rows.append(row)
                image_records[key] = {
                    "dataset_id": dataset_id,
                    "dataset_version": dataset_version,
                    "source_line_index": source_line_index,
                }

    actual_rows: list[dict[str, Any]] = []

    def reject_manifest_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise GenerationContractError(
                    f"duplicate generation manifest key: {key!r}"
                )
            result[key] = value
        return result

    with resolved_manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise GenerationContractError(
                    f"blank generation manifest row at line {line_number}"
                )
            try:
                row = json.loads(
                    line, object_pairs_hook=reject_manifest_duplicates
                )
            except json.JSONDecodeError as exc:
                raise GenerationContractError(
                    f"invalid generation manifest JSON at line {line_number}"
                ) from exc
            if not isinstance(row, dict) or set(row) != PROMPT_FIELDS:
                raise GenerationContractError(
                    f"generation manifest row {line_number} has invalid fields"
                )
            actual_rows.append(row)
    if actual_rows != expected_rows:
        mismatch = next(
            (
                index
                for index, pair in enumerate(zip(actual_rows, expected_rows))
                if pair[0] != pair[1]
            ),
            min(len(actual_rows), len(expected_rows)),
        )
        raise GenerationContractError(
            "generation manifest does not exactly reproduce the sealed selection; "
            f"first mismatch row={mismatch}, actual_rows={len(actual_rows)}, "
            f"expected_rows={len(expected_rows)}"
        )
    if generation_record.get("rows") != len(expected_rows):
        raise GenerationContractError("sealed generation row count is inconsistent")
    if population.get("images_per_arm") != len(expected_rows):
        raise GenerationContractError("dataset images_per_arm is inconsistent")

    source = contract.get("source")
    if not isinstance(source, dict):
        raise GenerationContractError("dataset source identity is missing")
    source_repository = source.get("repository")
    source_commit = source.get("commit")
    if not isinstance(source_repository, str) or not source_repository:
        raise GenerationContractError("dataset source repository is invalid")
    if not isinstance(source_commit, str) or len(source_commit) != 40:
        raise GenerationContractError("dataset source commit is invalid")
    source_pin_path = _resolve_dataset_path(
        source.get("pin_path"), base=root, label="dataset source pin path"
    )
    source_pin_sha256 = source.get("pin_sha256")
    if not isinstance(source_pin_sha256, str) or len(source_pin_sha256) != 64:
        raise GenerationContractError("dataset source pin SHA-256 is invalid")
    if (
        not source_pin_path.is_file()
        or sha256_file(source_pin_path) != source_pin_sha256
    ):
        raise GenerationContractError(
            f"dataset source pin is absent or changed: {source_pin_path}"
        )
    source_pin = _load_json_object_no_duplicates(
        source_pin_path, label="T2I-CompBench++ source pin"
    )
    if (
        source_pin.get("repo") != source_repository
        or source_pin.get("commit") != source_commit
    ):
        raise GenerationContractError(
            "dataset source pin repository/commit differs from the dataset contract"
        )
    return T2ICBDatasetProvenance(
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        contract_path=contract_path.resolve(),
        contract_sha256=sha256_file(contract_path),
        selection_manifest_path=selection_path,
        selection_manifest_sha256=selection_sha256,
        source_repository=source_repository,
        source_commit=source_commit,
        source_pin_path=source_pin_path,
        source_pin_sha256=source_pin_sha256,
        image_records=image_records,
        prompt_manifest_path=resolved_manifest,
    )
