"""Blind preprocessing of the T2I-CompBench++ prompts for the four external SD1.4 baselines.

Composable Diffusion, StructureDiffusion, Attend-and-Excite and SynGen each turn
a prompt into a method input (``plan``): component prompts, Stanza 1.4.2 noun
phrases, spaCy 3.6.1 ``en_core_web_trf`` noun targets, or spaCy binding groups.
``build_external_inputs`` parses every unique prompt once, without category
labels or gold annotations, and expands the plan to every image row of the
generation manifest. One input file per arm is written as compact canonical
JSONL with a ``.meta.json`` that records the parser and adapter environments:

* ``<input_dir>/external_sidecars/sd14_{composable,structure,syngen}.jsonl``;
* ``<input_dir>/ae_candidate/sd14_attend_excite.jsonl`` (every candidate
  subtoken; the final ``external_sidecars/sd14_attend_excite.jsonl`` is written
  by the GPU subtoken selection of ``baselines.sd14_generation``).

The building side runs in the isolated Python 3.11 adapter environment
(``scripts/setup/setup_baseline_env.sh``, ``docker/requirements-baselines*.txt``);
the reading side (``parse_plan``, ``load_input_metadata`` and the validators)
runs in the main environment before generation. The module therefore imports
only the standard library, torch/nltk (through the method modules),
``config``, ``provenance``, ``targets.tokenizer`` and the baseline method
modules at module level.
"""
from __future__ import annotations

import importlib.metadata
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

from ccg_diffusion.baselines import attend_and_excite, composable_diffusion, structure_diffusion, syngen
from ccg_diffusion.baselines.registry import EXTERNAL_ARMS, SOURCE_COMMIT_BY_ARM
from ccg_diffusion.config import SD14_GENERATION
from ccg_diffusion.provenance import (
    atomic_write_json,
    atomic_write_jsonl,
    canonical_json,
    portable_path,
    read_jsonl,
    repository_root,
    sha256_file,
    sha256_text,
)
from ccg_diffusion.targets.tokenizer import require_sd14_tokenizer

PROMPT_FIELDS = frozenset({"experiment", "prompt_id", "prompt", "seed"})
# Fields of every input row (KEEP).
INPUT_ROW_FIELDS = frozenset(
    {
        "experiment",
        "prompt_id",
        "prompt",
        "seed",
        "status",
        "method",
        "source_commit",
        "plan",
    }
)
SPACY_VERSION = "3.6.1"
SPACY_MODEL = "en_core_web_trf"
SPACY_MODEL_VERSION = "3.6.1"
EXTERNAL_INPUT_PROTOCOL = "sd14_external_sidecar_v2"  # KEEP
ATTEND_EXCITE_FINAL_PROTOCOL = "sd14_attend_excite_sidecar_v3"  # KEEP
SELECTION_EVIDENCE_SCHEMA = "holsd.sd14_attend_excite_subtoken_selection.v3"  # KEEP
GUIDED_STATUS = "ok"
ADAPTER_ENVIRONMENT_PACKAGES = (
    "diffusers",
    # ftfy keeps transformers' CLIPTokenizer on the openai BPE path; without it
    # the BasicTokenizer fallback splits contractions and produces indices in a
    # token space the SD1.4 generation runtime does not use.
    "ftfy",
    "huggingface-hub",
    "nltk",
    "numpy",
    "spacy",
    "spacy-transformers",
    "stanza",
    "torch",
    "transformers",
)
# Plan payload fields per arm (KEEP).
PLAN_FIELDS = {
    "sd14_composable": frozenset({"components", "weights"}),
    "sd14_structure": frozenset(
        {
            "original_prompt",
            "method_prompt",
            "first_sentence",
            "conditioning_texts",
            "token_spans",
            "np_texts",
            "np_token_spans",
            "conjunction",
            "stanza_version",
            "source_commit",
        }
    ),
    "sd14_syngen": frozenset(
        {
            "prompt",
            "bindings",
            "attention_token_indices",
            "source_groups",
            "status",
            "include_entities",
            "source_commit",
        }
    ),
}
ATTEND_EXCITE_CANDIDATE_PLAN_FIELDS = frozenset(
    {
        "prompt",
        "targets",
        "indices_to_alter",
        "status",
        "source_commit",
        "automation_source_commit",
    }
)
ATTEND_EXCITE_FINAL_PLAN_FIELDS = frozenset(
    {
        *ATTEND_EXCITE_CANDIDATE_PLAN_FIELDS,
        "selection_protocol",
        "selection_evidence",
    }
)
ATTEND_EXCITE_CANDIDATE_TARGET_FIELDS = frozenset(
    {"text", "character_span", "token_indices"}
)
ATTEND_EXCITE_FINAL_TARGET_FIELDS = frozenset(
    {
        *ATTEND_EXCITE_CANDIDATE_TARGET_FIELDS,
        "selected_token_index",
        "selection_evidence_occurrence_id",
    }
)
# Top-level row statuses per arm; every status other than ``ok`` is generated as stock Vanilla.
ROW_STATUSES = {
    "sd14_composable": frozenset({"ok"}),
    "sd14_structure": frozenset({"ok", "no_np"}),
    "sd14_attend_excite": frozenset({"ok", "no_target"}),
    "sd14_syngen": frozenset({"ok", "no_target"}),
}
# Parser installations of the evaluated input files (pinned digests, KEEP).
RECORDED_PARSER_ENVIRONMENT: Mapping[str, Mapping[str, Any]] = {
    "sd14_composable": {"parser_required": False},
    "sd14_structure": {
        "parser_required": True,
        "parser": "stanza",
        "stanza_version": "1.4.2",
        "stanza_processors": ["tokenize", "pos", "constituency"],
        "stanza_resource_file_count": 7,
        "stanza_resource_manifest_sha256": (
            "e63aa877d0d5fecd0dd946a168b8381cac102d667decdd877c7428b4634fe227"
        ),
        "stanza_distribution_file_count": 348,
        "stanza_distribution_manifest_sha256": (
            "e087d8fdf89b3499eff6090b22c581dfcc7224b7db1d10c6e33473f1861e500e"
        ),
        "stanza_torch_load_compatibility": (
            "trusted_stanza_1_4_2_checkpoints_weights_only_false_during_load"
        ),
    },
    **{
        arm: {
            "parser_required": True,
            "parser": "spacy",
            "spacy_version": "3.6.1",
            "spacy_model": "en_core_web_trf",
            "spacy_model_version": "3.6.1",
            "spacy_distribution_file_count": 992,
            "spacy_distribution_manifest_sha256": (
                "3009ae7c3be49aaf93b397762ec82a3552e7704d9b63d8079df2a0f5777462fd"
            ),
            "spacy_model_distribution_file_count": 36,
            "spacy_model_distribution_manifest_sha256": (
                "ac404477eba7823f2b6049be9531e4bffbd686fdbf1df7f6426af1fe8bad4646"
            ),
            "spacy_model_asset_file_count": 24,
            "spacy_model_asset_manifest_sha256": (
                "b2e144fe4a580708f3a9281b774eeff774ed2adb2979801fddada5296ec718f3"
            ),
        }
        for arm in ("sd14_attend_excite", "sd14_syngen")
    },
}


class BaselineInputError(ValueError):
    """A blind preprocessing input, a pinned parser contract or a method input file is invalid."""


def input_path(input_dir: Path, arm: str, *, stage: Literal["candidate", "final"] = "final") -> Path:
    """The input file of ``arm`` under ``input_dir`` (the A&E candidate stage has its own directory)."""

    if arm not in EXTERNAL_ARMS:
        raise BaselineInputError(f"unsupported external method: {arm!r}")
    if stage == "candidate":
        if arm != "sd14_attend_excite":
            raise BaselineInputError("candidate input stage is specific to A&E")
        return input_dir / "ae_candidate" / f"{arm}.jsonl"
    return input_dir / "external_sidecars" / f"{arm}.jsonl"


# --- building the input files (adapter environment) ------------------------------------------------------


def load_prompt_manifest(path: Path) -> list[dict[str, Any]]:
    """Load the exact four-field generation manifest without gold fields."""

    raw_rows = read_jsonl(path, reject_duplicate_keys=True)
    rows: list[dict[str, Any]] = []
    prompt_ids: set[str] = set()
    keys: set[tuple[str, str, int]] = set()
    for row_number, raw in enumerate(raw_rows, start=1):
        row = dict(raw)
        location = f"{path}:{row_number}"
        if set(row) != PROMPT_FIELDS:
            raise BaselineInputError(
                f"{location} must contain exactly {sorted(PROMPT_FIELDS)}"
            )
        experiment = require_text(row["experiment"], location=f"{location}.experiment")
        prompt_id = require_text(row["prompt_id"], location=f"{location}.prompt_id")
        prompt = require_text(row["prompt"], location=f"{location}.prompt")
        seed = row["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise BaselineInputError(f"{location}.seed must be a non-negative integer")
        key = (experiment, prompt_id, seed)
        if key in keys or prompt_id in prompt_ids:
            raise BaselineInputError(f"duplicate prompt identity at {location}: {key}")
        keys.add(key)
        prompt_ids.add(prompt_id)
        rows.append(
            {
                "experiment": experiment,
                "prompt_id": prompt_id,
                "prompt": prompt,
                "seed": seed,
            }
        )
    if not rows:
        raise BaselineInputError(f"empty prompt manifest: {path}")
    return rows


def composable_record(prompt: str) -> tuple[str, dict[str, Any]]:
    plan = composable_diffusion.split_prompt(prompt)
    return "ok", {
        "components": list(plan.components),
        "weights": list(plan.weights),
    }


def structure_record(
    prompt: str,
    *,
    preprocessor: structure_diffusion.StanzaConstituencyPreprocessor,
) -> tuple[str, dict[str, Any]]:
    plan = preprocessor(prompt)
    return ("ok" if plan.guidance_applied else "no_np"), {
        "original_prompt": plan.original_prompt,
        "method_prompt": plan.method_prompt,
        "first_sentence": plan.first_sentence,
        "conditioning_texts": list(plan.conditioning_texts),
        "token_spans": [list(span) for span in plan.token_spans],
        "np_texts": list(plan.np_texts),
        "np_token_spans": [list(span) for span in plan.np_token_spans],
        "conjunction": plan.conjunction,
        "stanza_version": plan.stanza_version,
        "source_commit": plan.source_commit,
    }


def attend_and_excite_record(prompt: str, *, doc: Any, tokenizer: Any) -> tuple[str, dict[str, Any]]:
    plan = attend_and_excite.extract_noun_targets(prompt, doc=doc, tokenizer=tokenizer)
    return ("ok" if plan.status == "ok" else "no_target"), {
        "prompt": plan.prompt,
        "targets": [
            {
                "text": target.text,
                "character_span": [target.char_start, target.char_end],
                "token_indices": list(target.clip_indices),
            }
            for target in plan.targets
        ],
        "indices_to_alter": list(plan.selected_clip_indices),
        "status": plan.status,
        "source_commit": plan.source_commit,
        "automation_source_commit": plan.automation_source_commit,
    }


def syngen_record(prompt: str, *, doc: Any, tokenizer: Any) -> tuple[str, dict[str, Any]]:
    plan = syngen.extract_bindings(prompt, doc=doc, tokenizer=tokenizer)
    return ("ok" if plan.status == "ok" else "no_target"), {
        "prompt": plan.prompt,
        "bindings": [
            {"members": [list(member) for member in binding.members]}
            for binding in plan.bindings
        ],
        "attention_token_indices": list(plan.attention_token_indices),
        "source_groups": [
            {
                "extractor": group.extractor,
                "texts": list(group.texts),
                "character_spans": [list(span) for span in group.character_spans],
            }
            for group in plan.source_groups
        ],
        "status": plan.status,
        "include_entities": plan.include_entities,
        "source_commit": plan.source_commit,
    }


def directory_manifest_sha256(path: Path) -> tuple[int, str]:
    """Number of files under ``path`` and the sha256 of their sorted (path, size, sha256) list."""

    if not path.is_dir():
        raise BaselineInputError(f"resource directory does not exist: {path}")
    files = []
    for candidate in sorted(item for item in path.rglob("*") if item.is_file()):
        files.append(
            {
                "path": str(candidate.relative_to(path)),
                "size": candidate.stat().st_size,
                "sha256": sha256_file(candidate),
            }
        )
    if not files:
        raise BaselineInputError(f"resource directory contains no files: {path}")
    return len(files), sha256_text(canonical_json(files))


def distribution_manifest_sha256(distribution_name: str) -> tuple[int, str]:
    """Seal every installed file recorded by an importlib distribution."""

    distribution = importlib.metadata.distribution(distribution_name)
    files = []
    for relative_path in sorted(distribution.files or (), key=str):
        candidate = Path(distribution.locate_file(relative_path))
        if not candidate.is_file():
            continue
        files.append(
            {
                "path": str(relative_path),
                "size": candidate.stat().st_size,
                "sha256": sha256_file(candidate),
            }
        )
    if not files:
        raise BaselineInputError(
            f"distribution contains no recorded files: {distribution_name}"
        )
    return len(files), sha256_text(canonical_json(files))


def adapter_environment_record() -> dict[str, Any]:
    """Record the isolated adapter environment (package versions and the ``pip freeze --all`` digest)."""

    versions = {
        package: importlib.metadata.version(package)
        for package in ADAPTER_ENVIRONMENT_PACKAGES
    }
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        check=True,
        capture_output=True,
        text=True,
    )
    freeze_lines = tuple(
        sorted(line.strip() for line in completed.stdout.splitlines() if line.strip())
    )
    pyvenv_config = Path(sys.prefix) / "pyvenv.cfg"
    return {
        "role": "adapter_preprocessing_only",
        "official_reference_environment": False,
        "python_executable": str(Path(sys.executable).resolve()),
        "python_executable_sha256": sha256_file(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "package_versions": versions,
        "pip_freeze_line_count": len(freeze_lines),
        "pip_freeze_sha256": sha256_text("\n".join(freeze_lines) + "\n"),
        "pyvenv_config_sha256": (
            sha256_file(pyvenv_config) if pyvenv_config.is_file() else None
        ),
    }


def load_adapter_tokenizer(*, local_files_only: bool) -> tuple[Any, dict[str, Any]]:
    """The pinned SD1.4 CLIP tokenizer built from its vocab/merges files (adapter environment)."""

    from huggingface_hub import snapshot_download
    from transformers import CLIPTokenizer

    contract = SD14_GENERATION
    snapshot = Path(snapshot_download(
        contract.model_id,
        revision=contract.model_revision,
        local_files_only=local_files_only,
        allow_patterns=[f"{contract.tokenizer_subfolder}/*"],
    ))
    tokenizer_dir = snapshot / contract.tokenizer_subfolder
    tokenizer = CLIPTokenizer(
        vocab_file=str(tokenizer_dir / "vocab.json"),
        merges_file=str(tokenizer_dir / "merges.txt"),
        model_max_length=contract.tokenizer_model_max_length,
        pad_token=contract.tokenizer_pad_token,
    )
    tokenizer.vocab_file = str(tokenizer_dir / "vocab.json")
    tokenizer.merges_file = str(tokenizer_dir / "merges.txt")
    try:
        identity = require_sd14_tokenizer(tokenizer)
    except ValueError as exc:
        raise BaselineInputError(str(exc)) from exc
    return tokenizer, identity


def load_spacy_parser(*, gpu_id: int) -> tuple[Any, dict[str, Any]]:
    """spaCy 3.6.1 ``en_core_web_trf`` and its installation record (A&E and SynGen inputs)."""

    import spacy

    if spacy.__version__ != SPACY_VERSION:
        raise BaselineInputError(
            f"A&E/SynGen require spacy=={SPACY_VERSION}, got {spacy.__version__}"
        )
    if gpu_id >= 0:
        if not spacy.require_gpu(gpu_id=gpu_id):
            raise BaselineInputError(f"spaCy could not acquire GPU {gpu_id}")
    nlp = spacy.load(SPACY_MODEL)
    model_version = str(getattr(nlp, "meta", {}).get("version"))
    if model_version != SPACY_MODEL_VERSION:
        raise BaselineInputError(
            f"{SPACY_MODEL} must be {SPACY_MODEL_VERSION}, got {model_version}"
        )
    spacy_file_count, spacy_digest = distribution_manifest_sha256("spacy")
    model_distribution_file_count, model_distribution_digest = (
        distribution_manifest_sha256("en-core-web-trf")
    )
    model_asset_file_count, model_asset_digest = directory_manifest_sha256(
        Path(nlp.path)
    )
    return nlp, {
        "parser_required": True,
        "parser": "spacy",
        "spacy_version": spacy.__version__,
        "spacy_model": SPACY_MODEL,
        "spacy_model_version": model_version,
        "spacy_gpu_id": gpu_id,
        "spacy_distribution_file_count": spacy_file_count,
        "spacy_distribution_manifest_sha256": spacy_digest,
        "spacy_model_distribution_file_count": model_distribution_file_count,
        "spacy_model_distribution_manifest_sha256": model_distribution_digest,
        "spacy_model_asset_file_count": model_asset_file_count,
        "spacy_model_asset_manifest_sha256": model_asset_digest,
    }


def load_stanza_preprocessor(
    *,
    resources_dir: Path,
    use_gpu: bool,
    download: bool,
    tokenizer: Any,
) -> tuple[structure_diffusion.StanzaConstituencyPreprocessor, dict[str, Any]]:
    """The Stanza 1.4.2 constituency pipeline and its installation record (StructureDiffusion inputs)."""

    import stanza
    import torch

    if stanza.__version__ != structure_diffusion.STANZA_VERSION:
        raise BaselineInputError(
            f"Structure Diffusion requires stanza=={structure_diffusion.STANZA_VERSION}, "
            f"got {stanza.__version__}"
        )
    if download:
        stanza.download(
            "en",
            model_dir=str(resources_dir),
            processors="tokenize,pos,constituency",
            verbose=True,
        )
    # Stanza 1.4.2 predates torch 2.6's weights_only=True default.  Its
    # official English checkpoints are trusted artifacts downloaded by
    # stanza.download, so narrowly restore the historical torch.load default
    # only while Stanza constructs this pipeline, then immediately restore it.
    original_torch_load = torch.load

    def trusted_stanza_torch_load(*load_args: Any, **load_kwargs: Any) -> Any:
        load_kwargs.setdefault("weights_only", False)
        return original_torch_load(*load_args, **load_kwargs)

    torch.load = trusted_stanza_torch_load
    try:
        pipeline = stanza.Pipeline(
            lang="en",
            processors="tokenize,pos,constituency",
            dir=str(resources_dir),
            use_gpu=use_gpu,
            verbose=False,
        )
    finally:
        torch.load = original_torch_load
    file_count, resource_digest = directory_manifest_sha256(resources_dir)
    distribution_file_count, distribution_digest = distribution_manifest_sha256(
        "stanza"
    )
    return structure_diffusion.StanzaConstituencyPreprocessor(
        pipeline,
        tokenizer,
        stanza_version=stanza.__version__,
    ), {
        "parser_required": True,
        "parser": "stanza",
        "stanza_version": stanza.__version__,
        "stanza_processors": ["tokenize", "pos", "constituency"],
        "stanza_resources_dir": str(resources_dir.resolve()),
        "stanza_resource_file_count": file_count,
        "stanza_resource_manifest_sha256": resource_digest,
        "stanza_distribution_file_count": distribution_file_count,
        "stanza_distribution_manifest_sha256": distribution_digest,
        "stanza_use_gpu": use_gpu,
        "stanza_torch_load_compatibility": (
            "trusted_stanza_1_4_2_checkpoints_weights_only_false_during_load"
        ),
    }


def expand_rows(
    generation_rows: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    plans_by_prompt: Mapping[str, tuple[str, Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """One input row per image row of the generation manifest, carrying its prompt's plan."""

    source_commit = SOURCE_COMMIT_BY_ARM[arm]
    output = []
    for manifest_row in generation_rows:
        prompt = str(manifest_row["prompt"])
        status, plan = plans_by_prompt[prompt]
        row = {
            **dict(manifest_row),
            "status": status,
            "method": arm,
            "source_commit": source_commit,
            "plan": dict(plan),
        }
        if set(row) != INPUT_ROW_FIELDS:
            raise AssertionError("external input row schema drifted")
        output.append(row)
    return output


def unique_prompts(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return list(dict.fromkeys(str(row["prompt"]) for row in rows))


def join_blind_and_generation(
    blind_rows: Sequence[Mapping[str, Any]],
    generation_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Require the prompt-only manifest (seed 0) to describe the generation manifest (seeds 42-51)."""

    blind_by_id: dict[str, Mapping[str, Any]] = {}
    for row in blind_rows:
        if row["seed"] != 0:
            raise BaselineInputError("blind prompt manifest must use seed 0")
        blind_by_id[str(row["prompt_id"])] = row
    generation_ids = {str(row["prompt_id"]) for row in generation_rows}
    if generation_ids != set(blind_by_id):
        raise BaselineInputError(
            "blind/generation prompt_id population mismatch: "
            f"missing={sorted(set(blind_by_id) - generation_ids)[:10]}, "
            f"extra={sorted(generation_ids - set(blind_by_id))[:10]}"
        )
    for row in generation_rows:
        blind = blind_by_id[str(row["prompt_id"])]
        if any(row[field] != blind[field] for field in ("experiment", "prompt", "prompt_id")):
            raise BaselineInputError(
                f"blind/generation prompt identity mismatch: {row['prompt_id']}"
            )
        seed = int(row["seed"])
        if seed not in range(42, 52) or not str(row["prompt_id"]).endswith(
            f"_s{seed}"
        ):
            raise BaselineInputError(
                "generation rows must encode one of seeds 42-51 in prompt_id: "
                f"{row['prompt_id']}/{seed}"
            )
    populations: dict[tuple[str, str], set[int]] = {}
    for row in generation_rows:
        key = (str(row["experiment"]), str(row["prompt"]))
        populations.setdefault(key, set()).add(int(row["seed"]))
    required_seeds = set(range(42, 52))
    invalid = [key for key, seeds in populations.items() if seeds != required_seeds]
    if invalid:
        raise BaselineInputError(
            f"every unique prompt requires all seeds 42-51: {invalid[:10]}"
        )


def load_parsers(
    arms: Sequence[str],
    *,
    tokenizer: Any,
    stanza_resources: Path,
    download_stanza: bool = False,
    stanza_gpu: bool = False,
    spacy_gpu_id: int = -1,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Load the parsers the arms need: ``({"stanza"|"spacy": parser}, {arm: parser_environment})``."""

    parsers: dict[str, Any] = {}
    environments: dict[str, dict[str, Any]] = {arm: {"parser_required": False} for arm in arms}
    if "sd14_structure" in arms:
        parsers["stanza"], environments["sd14_structure"] = load_stanza_preprocessor(
            resources_dir=stanza_resources,
            use_gpu=stanza_gpu,
            download=download_stanza,
            tokenizer=tokenizer,
        )
    spacy_arms = [arm for arm in ("sd14_attend_excite", "sd14_syngen") if arm in arms]
    if spacy_arms:
        parsers["spacy"], spacy_environment = load_spacy_parser(gpu_id=spacy_gpu_id)
        for arm in spacy_arms:
            environments[arm] = dict(spacy_environment)
    return parsers, environments


def build_external_inputs(
    *,
    blind_prompt_manifest: Path,
    generation_manifest: Path,
    input_dir: Path,
    arms: Sequence[str] = EXTERNAL_ARMS,
    stanza_resources: Path,
    download_stanza: bool = False,
    stanza_gpu: bool = False,
    spacy_gpu_id: int = -1,
    spacy_batch_size: int = 32,
    local_files_only: bool = True,
    overwrite: bool = False,
    strict_environment: bool = True,
) -> dict[str, Path]:
    """Write the input file and ``.meta.json`` of every requested external arm; return their paths.

    The parser and adapter environments are checked against the evaluated
    installation (``RECORDED_PARSER_ENVIRONMENT``) before anything is written;
    ``strict_environment=False`` compares parser versions, processors and model
    names only (``validate_parser_environment(strict=False)``).
    """

    blind_rows = load_prompt_manifest(blind_prompt_manifest)
    generation_rows = load_prompt_manifest(generation_manifest)
    join_blind_and_generation(blind_rows, generation_rows)
    prompts = unique_prompts(blind_rows)
    tokenizer, tokenizer_identity = load_adapter_tokenizer(local_files_only=local_files_only)
    adapter_environment = adapter_environment_record()
    arms = tuple(arms)
    parsers, parser_environments = load_parsers(
        arms,
        tokenizer=tokenizer,
        stanza_resources=stanza_resources,
        download_stanza=download_stanza,
        stanza_gpu=stanza_gpu,
        spacy_gpu_id=spacy_gpu_id,
    )
    for arm in arms:
        validate_parser_environment(parser_environments[arm], arm=arm, strict=strict_environment)
    validate_adapter_environment(adapter_environment)

    plans: dict[str, dict[str, tuple[str, Mapping[str, Any]]]] = {arm: {} for arm in arms}
    if "sd14_composable" in arms:
        plans["sd14_composable"] = {prompt: composable_record(prompt) for prompt in prompts}
    if "sd14_structure" in arms:
        for prompt in prompts:
            plans["sd14_structure"][prompt] = structure_record(prompt, preprocessor=parsers["stanza"])
    if "spacy" in parsers:
        documents: Iterable[Any] = parsers["spacy"].pipe(prompts, batch_size=spacy_batch_size)
        for prompt, document in zip(prompts, documents):
            if "sd14_attend_excite" in arms:
                plans["sd14_attend_excite"][prompt] = attend_and_excite_record(
                    prompt, doc=document, tokenizer=tokenizer)
            if "sd14_syngen" in arms:
                plans["sd14_syngen"][prompt] = syngen_record(prompt, doc=document, tokenizer=tokenizer)

    outputs: dict[str, Path] = {}
    for arm in arms:
        if set(plans[arm]) != set(prompts):
            raise BaselineInputError(f"incomplete unique-prompt plans for {arm}")
        rows = expand_rows(generation_rows, arm=arm, plans_by_prompt=plans[arm])
        output = input_path(input_dir, arm, stage="candidate" if arm == "sd14_attend_excite" else "final")
        metadata_output = output.with_name(output.name + ".meta.json")
        if not overwrite and (output.exists() or metadata_output.exists()):
            raise FileExistsError(output if output.exists() else metadata_output)
        output_record = atomic_write_jsonl(output, rows, overwrite=overwrite)
        atomic_write_json(
            metadata_output,
            {
                "artifact": "sd14_external_method_sidecar",
                "backbone": "sd14",
                "method": arm,
                "source_commit": SOURCE_COMMIT_BY_ARM[arm],
                "adapter_protocol": EXTERNAL_INPUT_PROTOCOL,
                "schema_fields": sorted(INPUT_ROW_FIELDS),
                "blind_prompt_manifest": {
                    "path": portable_path(blind_prompt_manifest),
                    "sha256": sha256_file(blind_prompt_manifest),
                },
                "generation_manifest": {
                    "path": portable_path(generation_manifest),
                    "sha256": sha256_file(generation_manifest),
                },
                "row_count": len(rows),
                "unique_prompt_count": len(prompts),
                "status_counts": dict(sorted(Counter(row["status"] for row in rows).items())),
                "unique_prompt_status_counts": dict(
                    sorted(Counter(status for status, _ in plans[arm].values()).items())
                ),
                "output": output_record.to_dict(relative_to=repository_root()),
                "model_id": SD14_GENERATION.model_id,
                "model_revision": SD14_GENERATION.model_revision,
                "tokenizer": tokenizer_identity,
                "parser_environment": parser_environments[arm],
                "adapter_environment": adapter_environment,
                "gold_inputs_used": False,
            },
            overwrite=overwrite,
        )
        outputs[arm] = output
    return outputs


def check_environment(
    *,
    arms: Sequence[str] = EXTERNAL_ARMS,
    stanza_resources: Path,
    stanza_gpu: bool = False,
    spacy_gpu_id: int = -1,
    local_files_only: bool = True,
    strict_environment: bool = True,
) -> dict[str, Any]:
    """Load the tokenizer and parsers and compare their records with the evaluated installation.

    ``strict_environment=False`` compares parser versions, processors and model
    names only. Writes nothing. Returns ``{"ok": bool, "mismatches": {...}}``.
    """

    tokenizer, _identity = load_adapter_tokenizer(local_files_only=local_files_only)
    _parsers, environments = load_parsers(
        arms, tokenizer=tokenizer, stanza_resources=stanza_resources,
        stanza_gpu=stanza_gpu, spacy_gpu_id=spacy_gpu_id)
    mismatches: dict[str, str] = {}
    for arm in arms:
        try:
            validate_parser_environment(environments[arm], arm=arm, strict=strict_environment)
        except BaselineInputError as exc:
            mismatches[arm] = str(exc)
    try:
        validate_adapter_environment(adapter_environment_record())
    except BaselineInputError as exc:
        mismatches["adapter_environment"] = str(exc)
    return {"ok": not mismatches, "mismatches": mismatches}


# --- reading and validating the input files (main environment) -------------------------------------------

# Field validators of input rows and their records; each raises ``BaselineInputError`` naming ``location``.


def require_exact_fields(
    payload: Mapping[str, Any],
    expected: frozenset[str],
    *,
    location: str,
) -> None:
    actual = set(payload)
    if actual != expected:
        raise BaselineInputError(
            f"{location} fields mismatch: expected={sorted(expected)}, "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def require_text(value: Any, *, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise BaselineInputError(f"{location} must be non-empty text")
    return value


def require_integer(value: Any, *, location: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise BaselineInputError(
            f"{location} must be an integer >= {minimum}"
        )
    return value


def require_bool(value: Any, *, location: str) -> bool:
    if not isinstance(value, bool):
        raise BaselineInputError(f"{location} must be boolean")
    return value


def require_list(value: Any, *, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise BaselineInputError(f"{location} must be a JSON array")
    return value


def parse_span(value: Any, *, location: str) -> tuple[int, int]:
    sequence = require_list(value, location=location)
    if len(sequence) != 2:
        raise BaselineInputError(f"{location} must have exactly two integers")
    start = require_integer(sequence[0], location=f"{location}[0]")
    end = require_integer(sequence[1], location=f"{location}[1]", minimum=1)
    if end <= start:
        raise BaselineInputError(f"{location} must be a non-empty interval")
    return start, end


def parse_indices(value: Any, *, location: str) -> tuple[int, ...]:
    sequence = require_list(value, location=location)
    indices = tuple(
        require_integer(item, location=f"{location}[{index}]", minimum=1)
        for index, item in enumerate(sequence)
    )
    if tuple(sorted(set(indices))) != indices:
        raise BaselineInputError(f"{location} must be unique and increasing")
    return indices


def parse_artifact_record(value: Any, *, location: str) -> dict[str, Any]:
    """Check the shape of a ``{"path", "sha256", "size_bytes"}`` artifact record."""

    if not isinstance(value, dict):
        raise BaselineInputError(f"{location} must be an object")
    require_exact_fields(
        value,
        frozenset({"path", "sha256", "size_bytes"}),
        location=location,
    )
    path = require_text(value["path"], location=f"{location}.path")
    digest = require_text(value["sha256"], location=f"{location}.sha256")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise BaselineInputError(f"{location}.sha256 must be lowercase SHA-256")
    size_bytes = require_integer(
        value["size_bytes"], location=f"{location}.size_bytes", minimum=1
    )
    return {"path": path, "sha256": digest, "size_bytes": size_bytes}


def _parse_composable_plan(
    payload: Mapping[str, Any], *, prompt: str, location: str
) -> composable_diffusion.ComposablePlan:
    require_exact_fields(payload, PLAN_FIELDS["sd14_composable"], location=location)
    raw_components = require_list(payload["components"], location=f"{location}.components")
    raw_weights = require_list(payload["weights"], location=f"{location}.weights")
    components = tuple(
        require_text(value, location=f"{location}.components[{index}]")
        for index, value in enumerate(raw_components)
    )
    weights = []
    for index, value in enumerate(raw_weights):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise BaselineInputError(
                f"{location}.weights[{index}] must be numeric"
            )
        weights.append(float(value))
    parsed = composable_diffusion.ComposablePlan(
        prompt=prompt,
        components=components,
        weights=tuple(weights),
    )
    expected = composable_diffusion.split_prompt(prompt)
    if parsed != expected:
        raise BaselineInputError(
            f"{location} does not match the frozen literal-and split/weight rule"
        )
    return parsed


def _parse_structure_plan(
    payload: Mapping[str, Any], *, prompt: str, location: str
) -> structure_diffusion.StructurePlan:
    require_exact_fields(payload, PLAN_FIELDS["sd14_structure"], location=location)
    original_prompt = require_text(
        payload["original_prompt"], location=f"{location}.original_prompt"
    )
    if original_prompt != prompt:
        raise BaselineInputError(f"{location}.original_prompt differs from top prompt")
    conditioning_texts = tuple(
        require_text(value, location=f"{location}.conditioning_texts[{index}]")
        for index, value in enumerate(
            require_list(
                payload["conditioning_texts"],
                location=f"{location}.conditioning_texts",
            )
        )
    )
    token_spans = tuple(
        parse_span(value, location=f"{location}.token_spans[{index}]")
        for index, value in enumerate(
            require_list(payload["token_spans"], location=f"{location}.token_spans")
        )
    )
    np_texts = tuple(
        require_text(value, location=f"{location}.np_texts[{index}]")
        for index, value in enumerate(
            require_list(payload["np_texts"], location=f"{location}.np_texts")
        )
    )
    np_token_spans = tuple(
        parse_span(value, location=f"{location}.np_token_spans[{index}]")
        for index, value in enumerate(
            require_list(
                payload["np_token_spans"], location=f"{location}.np_token_spans"
            )
        )
    )
    plan = structure_diffusion.StructurePlan(
        original_prompt=original_prompt,
        method_prompt=require_text(
            payload["method_prompt"], location=f"{location}.method_prompt"
        ),
        first_sentence=require_text(
            payload["first_sentence"], location=f"{location}.first_sentence"
        ),
        conditioning_texts=conditioning_texts,
        token_spans=token_spans,
        np_texts=np_texts,
        np_token_spans=np_token_spans,
        conjunction=require_bool(
            payload["conjunction"], location=f"{location}.conjunction"
        ),
        stanza_version=require_text(
            payload["stanza_version"], location=f"{location}.stanza_version"
        ),
        source_commit=require_text(
            payload["source_commit"], location=f"{location}.source_commit"
        ),
    )
    if plan.method_prompt != structure_diffusion.normalize_prompt(prompt):
        raise BaselineInputError(
            f"{location}.method_prompt differs from frozen normalization"
        )
    if plan.guidance_applied:
        if plan.method_prompt in plan.np_texts:
            expected_texts = plan.np_texts
            expected_spans = plan.np_token_spans
            if not expected_texts or expected_texts[0] != plan.method_prompt:
                raise BaselineInputError(
                    f"{location} full-prompt NP is not first in traversal order"
                )
        else:
            expected_texts = (plan.method_prompt, *plan.np_texts)
            expected_spans = plan.token_spans[:1] + plan.np_token_spans
        if plan.conditioning_texts != expected_texts:
            raise BaselineInputError(
                f"{location} NP conditioning rows do not match np_texts"
            )
        if plan.token_spans != expected_spans:
            raise BaselineInputError(
                f"{location} NP conditioning spans do not match np_token_spans"
            )
    elif len(plan.conditioning_texts) != 1 or len(plan.token_spans) != 1:
        raise BaselineInputError(
            f"{location} no-NP plan must contain only full-prompt conditioning"
        )
    return plan


def _parse_attend_and_excite_plan(
    payload: Mapping[str, Any], *, prompt: str, location: str, stage: str
) -> attend_and_excite.AttendAndExcitePlan:
    if stage == "candidate":
        plan_fields = ATTEND_EXCITE_CANDIDATE_PLAN_FIELDS
        target_fields = ATTEND_EXCITE_CANDIDATE_TARGET_FIELDS
        selection_protocol = attend_and_excite.CANDIDATE_SELECTION_PROTOCOL
    elif stage == "final":
        plan_fields = ATTEND_EXCITE_FINAL_PLAN_FIELDS
        target_fields = ATTEND_EXCITE_FINAL_TARGET_FIELDS
        selection_protocol = require_text(
            payload.get("selection_protocol"),
            location=f"{location}.selection_protocol",
        )
        if selection_protocol != attend_and_excite.ENSEMBLE_SELECTION_PROTOCOL:
            raise BaselineInputError(
                f"{location}.selection_protocol is not the frozen v3 ensemble rule"
            )
        parse_artifact_record(
            payload.get("selection_evidence"),
            location=f"{location}.selection_evidence",
        )
    else:
        raise BaselineInputError(f"invalid A&E input stage: {stage!r}")
    require_exact_fields(payload, plan_fields, location=location)
    plan_prompt = require_text(payload["prompt"], location=f"{location}.prompt")
    if plan_prompt != prompt:
        raise BaselineInputError(f"{location}.prompt differs from top prompt")
    targets = []
    for index, value in enumerate(
        require_list(payload["targets"], location=f"{location}.targets")
    ):
        target_location = f"{location}.targets[{index}]"
        if not isinstance(value, dict):
            raise BaselineInputError(f"{target_location} must be an object")
        require_exact_fields(
            value,
            target_fields,
            location=target_location,
        )
        char_start, char_end = parse_span(
            value["character_span"], location=f"{target_location}.character_span"
        )
        indices = parse_indices(
            value["token_indices"], location=f"{target_location}.token_indices"
        )
        selected_index = indices[-1]
        if stage == "final":
            selected_index = require_integer(
                value["selected_token_index"],
                location=f"{target_location}.selected_token_index",
                minimum=1,
            )
            if selected_index not in indices:
                raise BaselineInputError(
                    f"{target_location}.selected_token_index is outside token_indices"
                )
            require_text(
                value["selection_evidence_occurrence_id"],
                location=f"{target_location}.selection_evidence_occurrence_id",
            )
        targets.append(
            attend_and_excite.NounTarget(
                text=require_text(value["text"], location=f"{target_location}.text"),
                char_start=char_start,
                char_end=char_end,
                clip_indices=indices,
                selected_clip_index=selected_index,
            )
        )
    selected = parse_indices(
        payload["indices_to_alter"], location=f"{location}.indices_to_alter"
    )
    plan = attend_and_excite.AttendAndExcitePlan(
        prompt=plan_prompt,
        targets=tuple(targets),
        status=require_text(payload["status"], location=f"{location}.status"),
        source_commit=require_text(
            payload["source_commit"], location=f"{location}.source_commit"
        ),
        automation_source_commit=require_text(
            payload["automation_source_commit"],
            location=f"{location}.automation_source_commit",
        ),
        selection_protocol=selection_protocol,
    )
    if plan.selected_clip_indices != selected:
        raise BaselineInputError(
            f"{location}.indices_to_alter differs from selected target subtokens"
        )
    attend_and_excite.validate_plan(plan, max_token_count=77)
    return plan


def _parse_syngen_plan(
    payload: Mapping[str, Any], *, prompt: str, location: str
) -> syngen.SynGenPlan:
    require_exact_fields(payload, PLAN_FIELDS["sd14_syngen"], location=location)
    plan_prompt = require_text(payload["prompt"], location=f"{location}.prompt")
    if plan_prompt != prompt:
        raise BaselineInputError(f"{location}.prompt differs from top prompt")
    bindings = []
    for index, raw_binding in enumerate(
        require_list(payload["bindings"], location=f"{location}.bindings")
    ):
        binding_location = f"{location}.bindings[{index}]"
        if not isinstance(raw_binding, dict):
            raise BaselineInputError(f"{binding_location} must be an object")
        require_exact_fields(
            raw_binding, frozenset({"members"}), location=binding_location
        )
        members = tuple(
            parse_indices(value, location=f"{binding_location}.members[{member_index}]")
            for member_index, value in enumerate(
                require_list(
                    raw_binding["members"], location=f"{binding_location}.members"
                )
            )
        )
        bindings.append(syngen.BindingGroup(members=members))
    source_groups = []
    for index, raw_group in enumerate(
        require_list(payload["source_groups"], location=f"{location}.source_groups")
    ):
        group_location = f"{location}.source_groups[{index}]"
        if not isinstance(raw_group, dict):
            raise BaselineInputError(f"{group_location} must be an object")
        require_exact_fields(
            raw_group, frozenset({"extractor", "texts", "character_spans"}), location=group_location)
        texts = tuple(
            require_text(value, location=f"{group_location}.texts[{text_index}]")
            for text_index, value in enumerate(
                require_list(raw_group["texts"], location=f"{group_location}.texts")
            )
        )
        spans = tuple(
            parse_span(value, location=f"{group_location}.character_spans[{span_index}]")
            for span_index, value in enumerate(
                require_list(
                    raw_group["character_spans"],
                    location=f"{group_location}.character_spans",
                )
            )
        )
        if len(texts) != len(spans):
            raise BaselineInputError(
                f"{group_location} texts and character_spans differ in length"
            )
        for text, (start, end) in zip(texts, spans):
            if end > len(prompt) or prompt[start:end] != text:
                raise BaselineInputError(
                    f"{group_location} character span does not reproduce prompt text"
                )
        source_groups.append(
            syngen.ParserGroup(
                extractor=require_text(
                    raw_group["extractor"], location=f"{group_location}.extractor"
                ),
                texts=texts,
                character_spans=spans,
            )
        )
    plan = syngen.SynGenPlan(
        prompt=plan_prompt,
        bindings=tuple(bindings),
        attention_token_indices=parse_indices(
            payload["attention_token_indices"],
            location=f"{location}.attention_token_indices",
        ),
        source_groups=tuple(source_groups),
        status=require_text(payload["status"], location=f"{location}.status"),
        include_entities=require_bool(
            payload["include_entities"], location=f"{location}.include_entities"
        ),
        source_commit=require_text(
            payload["source_commit"], location=f"{location}.source_commit"
        ),
    )
    syngen.validate_plan(plan, max_token_count=77)
    return plan


def parse_plan(
    payload: Mapping[str, Any],
    *,
    arm: str,
    prompt: str,
    location: str,
    stage: str = "final",
) -> Any:
    """Rebuild and validate the method plan of one input row."""

    parsers = {
        "sd14_composable": _parse_composable_plan,
        "sd14_structure": _parse_structure_plan,
        "sd14_syngen": _parse_syngen_plan,
    }
    try:
        if arm == "sd14_attend_excite":
            parsed = _parse_attend_and_excite_plan(
                payload,
                prompt=prompt,
                location=location,
                stage=stage,
            )
        else:
            parser = parsers[arm]
            parsed = parser(payload, prompt=prompt, location=location)
    except KeyError as exc:
        raise BaselineInputError(f"unsupported external method: {arm}") from exc
    except (TypeError, ValueError) as exc:
        if isinstance(exc, BaselineInputError):
            raise
        raise BaselineInputError(f"invalid {arm} plan at {location}: {exc}") from exc
    return parsed


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except (OSError, json.JSONDecodeError, BaselineInputError) as exc:
        raise BaselineInputError(f"invalid {label} at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise BaselineInputError(f"{label} must be an object")
    return payload


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise BaselineInputError(f"duplicate JSON key: {key!r}")
        payload[key] = value
    return payload


def load_input_metadata(
    input_file: Path,
    *,
    prompt_manifest: Path,
    blind_prompt_manifest: Path,
    arm: str,
    row_count: int,
    stage: str,
    strict_environment: bool = True,
) -> dict[str, Any]:
    """Load and check ``<input_file>.meta.json`` against the arm, the manifests and the input bytes."""

    path = input_file.with_name(input_file.name + ".meta.json")
    payload = _read_json_object(path, label="input file metadata")
    expected_protocol = (
        ATTEND_EXCITE_FINAL_PROTOCOL
        if arm == "sd14_attend_excite" and stage == "final"
        else EXTERNAL_INPUT_PROTOCOL
    )
    required = {
        "artifact": "sd14_external_method_sidecar",
        "backbone": "sd14",
        "method": arm,
        "source_commit": SOURCE_COMMIT_BY_ARM[arm],
        "adapter_protocol": expected_protocol,
        "schema_fields": sorted(INPUT_ROW_FIELDS),
        "row_count": row_count,
        "model_id": SD14_GENERATION.model_id,
        "model_revision": SD14_GENERATION.model_revision,
        "tokenizer": SD14_GENERATION.tokenizer_identity(),
        "gold_inputs_used": False,
    }
    mismatches = {
        key: {"expected": expected, "actual": payload.get(key)}
        for key, expected in required.items()
        if payload.get(key) != expected
    }
    generation_record = payload.get("generation_manifest")
    if not isinstance(generation_record, dict):
        mismatches["generation_manifest"] = {
            "expected": "object with path and sha256",
            "actual": generation_record,
        }
    elif generation_record.get("sha256") != sha256_file(prompt_manifest):
        mismatches["generation_manifest.sha256"] = {
            "expected": sha256_file(prompt_manifest),
            "actual": generation_record.get("sha256"),
        }
    blind_record = payload.get("blind_prompt_manifest")
    if not isinstance(blind_record, dict):
        mismatches["blind_prompt_manifest"] = {
            "expected": "object with path and sha256",
            "actual": blind_record,
        }
    elif blind_record.get("sha256") != sha256_file(blind_prompt_manifest):
        mismatches["blind_prompt_manifest.sha256"] = {
            "expected": sha256_file(blind_prompt_manifest),
            "actual": blind_record.get("sha256"),
        }
    if mismatches:
        raise BaselineInputError(f"input file metadata mismatch: {mismatches}")
    output_record = payload.get("output")
    expected_output = {
        "sha256": sha256_file(input_file),
        "size_bytes": input_file.stat().st_size,
    }
    if not isinstance(output_record, dict) or any(
        output_record.get(key) != value for key, value in expected_output.items()
    ):
        raise BaselineInputError(
            "input file metadata output record does not seal the current JSONL"
        )
    validate_parser_environment(payload.get("parser_environment"), arm=arm, strict=strict_environment)
    validate_adapter_environment(payload.get("adapter_environment"))
    if arm == "sd14_attend_excite" and stage == "final":
        validate_recorded_local_implementation(
            payload.get("local_implementation"),
            location="input file metadata.local_implementation",
        )
        for name in ("selection_evidence", "candidate_sidecar", "candidate_sidecar_metadata"):
            parse_artifact_record(payload.get(name), location=f"input file metadata.{name}")
    return payload


def validate_adapter_environment(value: Any) -> None:
    """Require a content-addressed isolated preprocessing environment."""

    if not isinstance(value, dict):
        raise BaselineInputError("input file adapter_environment must be an object")
    required = {
        "role": "adapter_preprocessing_only",
        "official_reference_environment": False,
    }
    mismatches = {
        key: {"expected": expected, "actual": value.get(key)}
        for key, expected in required.items()
        if value.get(key) != expected
    }
    if not isinstance(value.get("python_executable"), str):
        mismatches["python_executable"] = {
            "expected": "path text",
            "actual": value.get("python_executable"),
        }
    if not isinstance(value.get("python_version"), str):
        mismatches["python_version"] = {
            "expected": "version text",
            "actual": value.get("python_version"),
        }
    for key in (
        "python_executable_sha256",
        "pip_freeze_sha256",
        "pyvenv_config_sha256",
    ):
        digest = value.get(key)
        if not isinstance(digest, str) or len(digest) != 64:
            mismatches[key] = {"expected": "SHA-256", "actual": digest}
    freeze_count = value.get("pip_freeze_line_count")
    if (
        isinstance(freeze_count, bool)
        or not isinstance(freeze_count, int)
        or freeze_count < 1
    ):
        mismatches["pip_freeze_line_count"] = {
            "expected": "positive integer",
            "actual": freeze_count,
        }
    package_versions = value.get("package_versions")
    expected_packages = set(ADAPTER_ENVIRONMENT_PACKAGES)
    if not isinstance(package_versions, dict) or set(package_versions) != expected_packages:
        mismatches["package_versions"] = {
            "expected": sorted(expected_packages),
            "actual": (
                sorted(package_versions) if isinstance(package_versions, dict) else package_versions
            ),
        }
    elif any(
        not isinstance(version, str) or not version
        for version in package_versions.values()
    ):
        mismatches["package_versions.values"] = {
            "expected": "non-empty version text",
            "actual": package_versions,
        }
    if mismatches:
        raise BaselineInputError(
            f"adapter preprocessing environment mismatch: {mismatches}"
        )


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def validate_parser_environment(value: Any, *, arm: str, strict: bool = True) -> None:
    """Require the parser pins used by each arm's input file.

    ``strict=False`` compares versions, processors and model names only and
    requires the installation digests to be SHA-256 values: a fresh parser
    installation may differ in recorded files while parsing identically. The
    digests depend on the installer and on the venv path (the spaCy
    distribution lists the ``bin/spacy`` launcher, which holds the absolute
    path of the venv's Python), so only the evaluated venv matches them
    strictly.
    """

    if not isinstance(value, dict):
        raise BaselineInputError("input file parser_environment must be an object")
    recorded = RECORDED_PARSER_ENVIRONMENT[arm]
    identity = {key: recorded[key] for key in ("parser_required", "parser") if key in recorded}
    parser_mismatches = {
        key: {"expected": expected, "actual": value.get(key)}
        for key, expected in identity.items()
        if value.get(key) != expected
    }
    if parser_mismatches:
        raise BaselineInputError(
            f"parser environment identity mismatch: {parser_mismatches}"
        )
    mismatches = {}
    for key, expected in recorded.items():
        if not strict and key.endswith(("_sha256", "_file_count")):
            if key.endswith("_sha256") and not _is_sha256(value.get(key)):
                mismatches[key] = {"expected": "SHA-256", "actual": value.get(key)}
            continue
        if value.get(key) != expected:
            mismatches[key] = {"expected": expected, "actual": value.get(key)}
    if arm == "sd14_structure":
        if not isinstance(value.get("stanza_resources_dir"), str):
            mismatches["stanza_resources_dir"] = {
                "expected": "path text",
                "actual": value.get("stanza_resources_dir"),
            }
        if not isinstance(value.get("stanza_use_gpu"), bool):
            mismatches["stanza_use_gpu"] = {
                "expected": "boolean",
                "actual": value.get("stanza_use_gpu"),
            }
    if arm in {"sd14_attend_excite", "sd14_syngen"}:
        gpu_id = value.get("spacy_gpu_id")
        if isinstance(gpu_id, bool) or not isinstance(gpu_id, int) or gpu_id < -1:
            mismatches["spacy_gpu_id"] = {
                "expected": "integer >= -1",
                "actual": gpu_id,
            }
    if mismatches:
        raise BaselineInputError(
            f"{arm} parser environment mismatch: {mismatches}"
        )


def validate_recorded_local_implementation(value: Any, *, location: str) -> dict[str, Any]:
    """Check the shape of a recorded ``local_implementation`` (commit and component digests).

    Historical records name the source paths of their commit, so only the
    shape is checked when such a record is loaded; new records are sealed to
    the current commit when they are written.
    """

    if not isinstance(value, dict):
        raise BaselineInputError(f"{location} must be an object")
    require_exact_fields(
        value,
        frozenset(
            {
                "repo_git_commit",
                "component_sha256",
                "components_match_repo_git_commit",
            }
        ),
        location=location,
    )
    commit = value.get("repo_git_commit")
    if not isinstance(commit, str) or len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise BaselineInputError(f"{location}.repo_git_commit must be a full commit")
    components = value.get("component_sha256")
    if value.get("components_match_repo_git_commit") is not True:
        raise BaselineInputError(
            f"{location}.components_match_repo_git_commit must be true"
        )
    if not isinstance(components, dict) or not components:
        raise BaselineInputError(f"{location}.component_sha256 must be non-empty")
    for relative_path, digest in components.items():
        if not _is_sha256(digest):
            raise BaselineInputError(
                f"{location}.component_sha256[{relative_path!r}] is invalid"
            )
    return dict(value)


def validate_blind_generation_manifests(
    blind_rows: Sequence[Mapping[str, Any]],
    generation_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Require the canonical prompt-only view of every generation row."""

    if len(blind_rows) != len(generation_rows):
        raise BaselineInputError(
            "blind/generation manifest row counts differ: "
            f"{len(blind_rows)} != {len(generation_rows)}"
        )
    blind_by_id: dict[str, Mapping[str, Any]] = {}
    for row_number, row in enumerate(blind_rows, start=1):
        location = f"blind prompt manifest:{row_number}"
        require_exact_fields(row, PROMPT_FIELDS, location=location)
        prompt_id = require_text(row["prompt_id"], location=f"{location}.prompt_id")
        if row["seed"] != 0:
            raise BaselineInputError(
                f"{location}.seed must be zero for parser blindness"
            )
        if prompt_id in blind_by_id:
            raise BaselineInputError(f"duplicate blind prompt_id: {prompt_id}")
        blind_by_id[prompt_id] = row
    for row_number, row in enumerate(generation_rows, start=1):
        location = f"generation manifest:{row_number}"
        require_exact_fields(row, PROMPT_FIELDS, location=location)
        prompt_id = require_text(row["prompt_id"], location=f"{location}.prompt_id")
        blind = blind_by_id.get(prompt_id)
        if blind is None:
            raise BaselineInputError(
                f"generation prompt_id is absent from blind manifest: {prompt_id}"
            )
        if any(
            blind[field] != row[field]
            for field in ("experiment", "prompt_id", "prompt")
        ):
            raise BaselineInputError(
                f"blind/generation prompt identity mismatch: {prompt_id}"
            )


def validate_row_status(
    arm: str, *, status: str, plan: Any, location: str
) -> None:
    """Require the top-level row status to agree with the plan (``ok`` iff the method guides)."""

    expected_guided: bool
    if arm == "sd14_composable":
        expected_guided = True
    elif arm == "sd14_structure":
        if not isinstance(plan, structure_diffusion.StructurePlan):
            raise BaselineInputError(f"invalid Structure plan at {location}")
        expected_guided = plan.guidance_applied
    elif arm == "sd14_attend_excite":
        if not isinstance(plan, attend_and_excite.AttendAndExcitePlan):
            raise BaselineInputError(f"invalid A&E plan at {location}")
        expected_guided = plan.status == "ok"
    elif arm == "sd14_syngen":
        if not isinstance(plan, syngen.SynGenPlan):
            raise BaselineInputError(f"invalid SynGen plan at {location}")
        expected_guided = plan.status == "ok"
    else:
        raise BaselineInputError(f"unsupported method: {arm}")
    if (status == GUIDED_STATUS) != expected_guided:
        raise BaselineInputError(
            f"top-level status and adapter plan disagree at {location}"
        )
