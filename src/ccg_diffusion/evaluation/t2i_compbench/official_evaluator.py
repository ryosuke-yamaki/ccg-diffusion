"""Run the official T2I-CompBench++ evaluators over a generation run (fail closed).

Every category routes to the upstream metric scripts named in the
T2I-CompBench++ readme (``CATEGORY_STEPS``; scores in ``METRICS``). The official
code runs verbatim out of its own virtualenv (Python 3.10) through
``subprocess``; this module only prepares the working tree, validates the image
population the evaluators assume, reads back the score the upstream script
wrote, and records provenance. Nothing here imports torch.

``validate_official_environment`` checks every external input first: the
tracked source pin, the official clone at the pinned commit, its prompt files,
the evaluator venv freeze and the four metric weights.

The image-name contract is load-bearing and is checked before any GPU work:
``BLIP_vqa.py`` reads the prompt back out of ``name.split("_")[0]`` and orders
images by ``int(name.split("_")[-1].split(".")[0])``, ``numeracy_eval.py`` reads
the index out of ``name.split(".png")[0].split("_")[1]``, and ``3_in_1.py``
assumes image ``i*10 + j`` belongs to prompt ``i`` of ``complex_val.txt``.

Each evaluated category gets ``output/vqa/<run>/t2i_compbench/[<arm>/]<category>/``
with ``vqa_result.json``, the score text file and ``eval_manifest.json``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Sequence

from ccg_diffusion.evaluation.t2i_compbench.dataset import (
    CATEGORIES, EVALUATOR_WEIGHTS, IMAGES_PER_PROMPT, REPO_ROOT, SOURCE_COMMIT, SOURCE_PIN_PATH, SOURCE_REPOSITORY,
    T2ICompBenchError,
)
from ccg_diffusion.evaluation.t2i_compbench.export import EXPORT_METHODS
from ccg_diffusion.provenance import sha256_file

GATE_TOLERANCE = 0.02
EVAL_MANIFEST_SCHEMA = "holsd.t2icb_eval.v2"
VENV_FREEZE_NAME = "venv_freeze.txt"
METRIC_STAGE_MANIFEST_NAME = "metric_stage_manifest.json"
METRIC_STAGE_SCHEMAS = frozenset({"holsd.t2icb_subset_export.v2"})
# Local code sealed into every evaluation manifest (``local_evaluator_provenance``).
LOCAL_EVALUATOR_COMPONENTS = (
    "scripts/t2i_compbench/evaluate.py",
    "scripts/t2i_compbench/export_samples.py",
    "src/ccg_diffusion/evaluation/t2i_compbench/dataset.py",
    "src/ccg_diffusion/evaluation/t2i_compbench/export.py",
    "src/ccg_diffusion/evaluation/t2i_compbench/official_evaluator.py",
    "src/ccg_diffusion/evaluation/t2i_compbench/results.py",
)
COMPLEX_AUXILIARY_PROMPTS = (
    "complex_val_action.txt",
    "complex_val_spatial.txt",
    "complex_val_spatialaction.txt",
)

# The average an official script prints and the mean of its own per-image json
# are the same arithmetic, so they may differ only by float accumulation order.
# A wider gap means the free-text score was mis-parsed or the format changed.
SCORE_AGREEMENT_TOLERANCE = 1e-4

# TABLE XIII of the T2I-CompBench++ paper, "Stable v1-4" row.
PUBLISHED_TARGETS: Mapping[str, float] = {
    "color": 0.3765,
    "shape": 0.3576,
    "texture": 0.4156,
    "spatial": 0.1246,
    "3d_spatial": 0.3030,
    "numeracy": 0.4456,
    "non_spatial": 0.3079,
    "complex": 0.3080,
}

SAMPLE_RE = re.compile(r"^(?P<prompt>.+)_(?P<index>\d{6})\.png$", re.DOTALL)

DATASET_DIR_TOKEN = "{dataset_dir}"


class MetricStep(NamedTuple):
    """One official script invocation.

    ``cwd`` matters: the UniDet scripts open ``configs/experts.yaml`` and
    ``dataset/detection_features.pt`` relative to ``UniDet_eval``, and
    ``numeracy_eval.py`` additionally opens ``../examples/dataset/new_objects.txt``.
    """

    name: str
    script: str
    cwd: str
    out_flag: str
    use_complex: bool = False
    extra: tuple[str, ...] = ()


class ScoreSource(NamedTuple):
    """Where a metric leaves its score, and the literal text in front of it."""

    metric: str
    score_path: str
    score_prefix: str
    result_path: str


_BLIP_STEP = MetricStep(
    name="blip_vqa",
    script="BLIPvqa_eval/BLIP_vqa.py",
    cwd="BLIPvqa_eval",
    out_flag="--out_dir",
)
_CLIP_STEP = MetricStep(
    name="clip_similarity",
    script="CLIPScore_eval/CLIP_similarity.py",
    cwd=".",
    out_flag="--outpath",
)

CATEGORY_STEPS: Mapping[str, tuple[MetricStep, ...]] = {
    "color": (_BLIP_STEP,),
    "shape": (_BLIP_STEP,),
    "texture": (_BLIP_STEP,),
    "spatial": (
        MetricStep(
            name="spatial_2d",
            script="UniDet_eval/2D_spatial_eval.py",
            cwd="UniDet_eval",
            out_flag="--outpath",
        ),
    ),
    "numeracy": (
        MetricStep(
            name="numeracy",
            script="UniDet_eval/numeracy_eval.py",
            cwd="UniDet_eval",
            out_flag="--outpath",
        ),
    ),
    "3d_spatial": (
        MetricStep(
            name="spatial_3d",
            script="UniDet_eval/3D_spatial_eval.py",
            cwd="UniDet_eval",
            out_flag="--outpath",
        ),
    ),
    "non_spatial": (_CLIP_STEP,),
    "complex": (
        _BLIP_STEP,
        MetricStep(
            name="spatial_2d",
            script="UniDet_eval/2D_spatial_eval.py",
            cwd="UniDet_eval",
            out_flag="--outpath",
            use_complex=True,
        ),
        _CLIP_STEP._replace(use_complex=True),
        MetricStep(
            name="three_in_one",
            script="3_in_1_eval/3_in_1.py",
            cwd="3_in_1_eval",
            out_flag="--outpath",
            extra=(f"--data_path={DATASET_DIR_TOKEN}",),
        ),
    ),
}

_BLIP_SCORE = ScoreSource(
    metric="B-VQA",
    score_path="annotation_blip/blip_vqa_score.txt",
    score_prefix="BLIP-VQA score:",
    result_path="annotation_blip/vqa_result.json",
)

METRICS: Mapping[str, ScoreSource] = {
    "color": _BLIP_SCORE,
    "shape": _BLIP_SCORE,
    "texture": _BLIP_SCORE,
    "spatial": ScoreSource(
        metric="UniDet-2D",
        score_path="labels/annotation_obj_detection_2d/avg_score.txt",
        score_prefix="score avg:",
        result_path="labels/annotation_obj_detection_2d/vqa_result.json",
    ),
    "numeracy": ScoreSource(
        metric="UniDet-numeracy",
        score_path="annotation_num/score.txt",
        score_prefix="avg:",
        result_path="annotation_num/vqa_result.json",
    ),
    "3d_spatial": ScoreSource(
        metric="UniDet-3D",
        score_path="labels/annotation_obj_detection_3d/avg_result.txt",
        score_prefix="avg score is ",
        result_path="labels/annotation_obj_detection_3d/vqa_result.json",
    ),
    "non_spatial": ScoreSource(
        metric="CLIPScore",
        score_path="annotation_clip/score_avg.txt",
        score_prefix="score avg:",
        result_path="annotation_clip/vqa_result.json",
    ),
    "complex": ScoreSource(
        metric="3-in-1",
        score_path="annotation_3_in_1/vqa_score.txt",
        score_prefix="score avg:",
        result_path="annotation_3_in_1/vqa_result.json",
    ),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise T2ICompBenchError(f"{label} not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise T2ICompBenchError(f"invalid JSON in {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise T2ICompBenchError(f"{label} must contain a JSON object: {path}")
    return payload


def _capture_command(
    argv: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        capture_output=True,
        text=True,
        check=False,
        env=None if env is None else dict(env),
    )


def _require_sha256(value: Any, *, label: str) -> str:
    text_value = str(value)
    if len(text_value) != 64 or set(text_value) - set("0123456789abcdef"):
        raise T2ICompBenchError(f"{label} is not a lowercase SHA-256 digest: {value!r}")
    return text_value


def _resolve_contract_path(value: str, *, repo_root: Path) -> Path:
    """Resolve tracked contract paths without depending on the caller's cwd."""

    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo_root.resolve() / path).resolve()


def validate_source_pin_metadata(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the tracked external-source contract and match it to setup constants."""

    root = repo_root.resolve()
    path = root / SOURCE_PIN_PATH
    pin = _load_json_object(path, label="T2I-CompBench++ source pin")
    if pin.get("repo") != SOURCE_REPOSITORY:
        raise T2ICompBenchError(
            f"source pin repo is {pin.get('repo')!r}, expected {SOURCE_REPOSITORY!r}"
        )
    if pin.get("commit") != SOURCE_COMMIT:
        raise T2ICompBenchError(
            f"source pin commit is {pin.get('commit')!r}, expected {SOURCE_COMMIT}"
        )

    expected_weights = {str(spec["name"]): spec for spec in EVALUATOR_WEIGHTS}
    raw_weights = pin.get("weights")
    if not isinstance(raw_weights, list):
        raise T2ICompBenchError("source pin weights must be a list")
    recorded_weights: dict[str, Mapping[str, Any]] = {}
    for raw in raw_weights:
        if not isinstance(raw, dict) or not isinstance(raw.get("name"), str):
            raise T2ICompBenchError("source pin contains a malformed weight record")
        name = raw["name"]
        if name in recorded_weights:
            raise T2ICompBenchError(f"source pin repeats weight {name!r}")
        recorded_weights[name] = raw
    if set(recorded_weights) != set(expected_weights):
        raise T2ICompBenchError(
            "source pin weight names do not match the setup contract: "
            f"recorded={sorted(recorded_weights)}, expected={sorted(expected_weights)}"
        )
    for name, expected in expected_weights.items():
        recorded = recorded_weights[name]
        for key in ("url", "bytes", "sha256", "official_source"):
            if recorded.get(key) != expected[key]:
                raise T2ICompBenchError(
                    f"source pin {name}/{key} is {recorded.get(key)!r}, "
                    f"expected {expected[key]!r}"
                )
        _require_sha256(recorded["sha256"], label=f"source pin {name}/sha256")

    venv = pin.get("venv")
    if not isinstance(venv, dict):
        raise T2ICompBenchError("source pin venv must be an object")
    for key in ("path", "interpreter", "python_version", "freeze_sha256"):
        if not isinstance(venv.get(key), str) or not venv[key]:
            raise T2ICompBenchError(f"source pin venv/{key} must be a non-empty string")
    _require_sha256(venv["freeze_sha256"], label="source pin venv/freeze_sha256")

    clone_path = pin.get("clone_path")
    if not isinstance(clone_path, str) or not clone_path:
        raise T2ICompBenchError("source pin clone_path must be a non-empty string")

    # The pin is itself part of the contract.  Refuse an untracked or locally
    # edited mirror even when its replacement JSON is superficially plausible.
    relative = str(SOURCE_PIN_PATH)
    tracked = _capture_command(["git", "-C", str(root), "ls-files", "--error-unmatch", relative])
    if tracked.returncode != 0:
        raise T2ICompBenchError(f"source pin is not tracked by git: {path}")
    for argv, label in (
        (["git", "-C", str(root), "diff", "--quiet", "--", relative], "working tree"),
        (["git", "-C", str(root), "diff", "--cached", "--quiet", "--", relative], "index"),
    ):
        status = _capture_command(argv)
        if status.returncode != 0:
            raise T2ICompBenchError(f"source pin differs from HEAD in the {label}: {path}")

    return pin, {"path": str(path), "sha256": sha256_file(path)}


def validate_official_clone(clone_dir: Path) -> dict[str, Any]:
    """Require the official commit and a clean tracked checkout."""

    clone = clone_dir.resolve()
    if not clone.is_dir():
        raise T2ICompBenchError(f"official clone not found: {clone}")
    head = _capture_command(["git", "-C", str(clone), "rev-parse", "HEAD"])
    if head.returncode != 0:
        raise T2ICompBenchError(f"cannot read official clone commit at {clone}: {head.stderr.strip()}")
    commit = head.stdout.strip()
    if commit != SOURCE_COMMIT:
        raise T2ICompBenchError(f"official clone is at {commit!r}, expected pinned commit {SOURCE_COMMIT}")
    origin = _capture_command(["git", "-C", str(clone), "remote", "get-url", "origin"])
    if origin.returncode != 0:
        raise T2ICompBenchError(f"official clone has no readable origin remote: {clone}")
    origin_url = origin.stdout.strip().removesuffix(".git")
    if origin_url != SOURCE_REPOSITORY.removesuffix(".git"):
        raise T2ICompBenchError(
            f"official clone origin is {origin.stdout.strip()!r}, expected {SOURCE_REPOSITORY!r}"
        )
    status = _capture_command(
        ["git", "-C", str(clone), "status", "--porcelain", "--untracked-files=no"]
    )
    if status.returncode != 0:
        raise T2ICompBenchError(f"cannot inspect official clone status at {clone}: {status.stderr.strip()}")
    if status.stdout.strip():
        first = status.stdout.splitlines()[0]
        raise T2ICompBenchError(f"official clone has tracked modifications ({first}); refusing evaluation")
    inventory = _capture_command(
        [
            "git",
            "-C",
            str(clone),
            "ls-files",
            "--others",
            "-z",
        ]
    )
    if inventory.returncode != 0:
        raise T2ICompBenchError(
            f"cannot inspect official clone untracked inventory at {clone}: "
            f"{inventory.stderr.strip()}"
        )
    allowed_weight_paths = {
        Path(str(spec["link"])).as_posix()
        for spec in EVALUATOR_WEIGHTS
        if spec["link"] is not None
    }
    untracked = [entry for entry in inventory.stdout.split("\0") if entry]
    unexpected = [
        relative
        for relative in untracked
        if Path(relative).as_posix() not in allowed_weight_paths
    ]
    if unexpected:
        raise T2ICompBenchError(
            f"official clone contains untracked runtime input {unexpected[0]!r}; "
            "only the exact pinned metric weight paths are allowed. Remove Python bytecode "
            "caches before evaluation; evaluator subprocesses set PYTHONDONTWRITEBYTECODE=1."
        )
    return {
        "path": str(clone),
        "repo": SOURCE_REPOSITORY,
        "commit": commit,
        "tracked_clean": True,
        "untracked_inventory": {
            "allowed_weight_paths": sorted(
                relative for relative in untracked if relative in allowed_weight_paths
            ),
            "unexpected_count": 0,
        },
    }


def validate_official_prompt_assets(
    clone_dir: Path,
    pin: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Verify every official validation prompt file recorded in the source pin."""

    recorded = pin.get("val_txt")
    if not isinstance(recorded, dict) or set(recorded) != set(CATEGORIES):
        raise T2ICompBenchError("source pin val_txt categories do not match the evaluator categories")
    result: list[dict[str, Any]] = []
    for category in CATEGORIES:
        spec = recorded[category]
        if not isinstance(spec, dict):
            raise T2ICompBenchError(f"source pin val_txt/{category} must be an object")
        path = prompt_file_path(clone_dir, category)
        if not path.is_file():
            raise T2ICompBenchError(f"official prompt asset not found: {path}")
        actual_sha = sha256_file(path)
        expected_sha = _require_sha256(
            spec.get("sha256"), label=f"source pin val_txt/{category}/sha256"
        )
        actual_count = len(path.read_text(encoding="utf-8").splitlines())
        if actual_sha != expected_sha or actual_count != spec.get("count"):
            raise T2ICompBenchError(
                f"official prompt asset drift for {category}: sha256={actual_sha}, "
                f"count={actual_count}; expected sha256={expected_sha}, count={spec.get('count')}"
            )
        result.append(
            {"category": category, "path": str(path), "sha256": actual_sha, "count": actual_count}
        )
    return result


def _verify_file_seal(path: Path, *, expected_sha: str, expected_bytes: int, label: str) -> None:
    if not path.is_file():
        raise T2ICompBenchError(f"{label} not found: {path}")
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise T2ICompBenchError(
            f"{label} byte size is {actual_bytes}, expected {expected_bytes}: {path}"
        )
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise T2ICompBenchError(
            f"{label} sha256 is {actual_sha}, expected {expected_sha}: {path}"
        )


def validate_official_weights(
    repo_root: Path,
    clone_dir: Path,
    venv_python: Path,
    pin: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Verify the four pinned weight blobs and the paths used by official code."""

    venv = pin["venv"]
    venv_dir = (repo_root.resolve() / str(venv["path"])).resolve()
    weights_dir = venv_dir.parent / "weights"
    recorded = {str(row["name"]): row for row in pin["weights"]}

    hub = _capture_command(
        [str(venv_python), "-c", "import torch; print(torch.hub.get_dir())"],
        env=subprocess_env(),
    )
    if hub.returncode != 0:
        raise T2ICompBenchError(f"cannot resolve eval-venv torch hub: {hub.stderr.strip()}")
    hub_checkpoints = Path(hub.stdout.strip()) / "checkpoints"

    result: list[dict[str, Any]] = []
    for setup_spec in EVALUATOR_WEIGHTS:
        name = str(setup_spec["name"])
        record = recorded[name]
        expected_sha = str(record["sha256"])
        expected_bytes = int(record["bytes"])
        source = weights_dir / name
        _verify_file_seal(
            source,
            expected_sha=expected_sha,
            expected_bytes=expected_bytes,
            label=f"pinned metric weight {name}",
        )
        linked = (
            clone_dir.resolve() / str(setup_spec["link"])
            if setup_spec["link"]
            else hub_checkpoints / name
        )
        if not linked.is_symlink():
            raise T2ICompBenchError(f"official metric weight path is not the pinned symlink: {linked}")
        if linked.resolve() != source.resolve():
            raise T2ICompBenchError(
                f"official metric weight link {linked} resolves to {linked.resolve()}, "
                f"expected {source.resolve()}"
            )
        _verify_file_seal(
            linked,
            expected_sha=expected_sha,
            expected_bytes=expected_bytes,
            label=f"linked metric weight {name}",
        )
        result.append(
            {
                "name": name,
                "path": str(source),
                "linked_path": str(linked),
                "sha256": expected_sha,
                "bytes": expected_bytes,
            }
        )
    return result


def validate_official_venv(
    repo_root: Path,
    venv_python: Path,
    pin: Mapping[str, Any],
) -> dict[str, Any]:
    """Match both the frozen record and the live environment to the pinned freeze."""

    root = repo_root.resolve()
    record = pin["venv"]
    expected_python_path = root / str(record["interpreter"])
    actual_python_path = venv_python if venv_python.is_absolute() else root / venv_python
    if actual_python_path.absolute() != expected_python_path.absolute():
        raise T2ICompBenchError(
            f"eval venv interpreter is {actual_python_path.absolute()}, expected pinned interpreter "
            f"{expected_python_path.absolute()}"
        )
    if not actual_python_path.is_file():
        raise T2ICompBenchError(f"eval venv interpreter not found: {actual_python_path}")

    version = _capture_command(
        [str(actual_python_path), "-c", "import sys; print(sys.version.split()[0])"],
        env=subprocess_env(),
    )
    if version.returncode != 0:
        raise T2ICompBenchError(f"cannot query eval venv Python: {version.stderr.strip()}")
    python_version = version.stdout.strip()
    if python_version != record["python_version"]:
        raise T2ICompBenchError(
            f"eval venv Python is {python_version!r}, expected {record['python_version']!r}"
        )

    expected_freeze_sha = str(record["freeze_sha256"])
    freeze_path = (root / str(record["path"])).resolve().parent / VENV_FREEZE_NAME
    if not freeze_path.is_file():
        raise T2ICompBenchError(f"pinned eval-venv freeze file not found: {freeze_path}")
    recorded_freeze_sha = sha256_file(freeze_path)
    if recorded_freeze_sha != expected_freeze_sha:
        raise T2ICompBenchError(
            f"eval-venv freeze record sha256 is {recorded_freeze_sha}, "
            f"expected {expected_freeze_sha}: {freeze_path}"
        )

    uv = shutil.which("uv")
    if uv is None:
        raise T2ICompBenchError("uv is required to verify the live official eval-venv freeze")
    # --color never keeps the freeze byte-stable across uv versions that emit
    # ANSI styling; the pinned sha256 covers the plain-text package list.
    live = _capture_command(
        [uv, "pip", "freeze", "--color", "never", "--python", str(actual_python_path)],
        env=subprocess_env(),
    )
    if live.returncode != 0:
        raise T2ICompBenchError(f"cannot freeze the live official eval venv: {live.stderr.strip()}")
    live_freeze_sha = hashlib.sha256(live.stdout.encode("utf-8")).hexdigest()
    if live_freeze_sha != expected_freeze_sha:
        raise T2ICompBenchError(
            f"live eval-venv freeze sha256 is {live_freeze_sha}, expected {expected_freeze_sha}"
        )
    return {
        "path": str(actual_python_path.absolute()),
        "python_version": python_version,
        "freeze_path": str(freeze_path),
        "freeze_sha256": expected_freeze_sha,
        "live_freeze_sha256": live_freeze_sha,
    }


def validate_official_environment(
    *,
    repo_root: Path,
    clone_dir: Path,
    venv_python: Path,
) -> dict[str, Any]:
    """Fail-closed preflight for every external input to official evaluation."""

    pin, pin_reference = validate_source_pin_metadata(repo_root)
    expected_clone = (repo_root.resolve() / str(pin["clone_path"])).resolve()
    if clone_dir.resolve() != expected_clone:
        raise T2ICompBenchError(f"official clone path is {clone_dir.resolve()}, expected {expected_clone}")
    clone = validate_official_clone(clone_dir)
    prompts = validate_official_prompt_assets(clone_dir, pin)
    venv = validate_official_venv(repo_root, venv_python, pin)
    weights = validate_official_weights(repo_root, clone_dir, venv_python, pin)
    return {
        "source_pin": pin_reference,
        "clone": clone,
        "venv": venv,
        "weights": weights,
        "official_prompt_assets": prompts,
    }


def local_evaluator_provenance(repo_root: Path) -> dict[str, Any]:
    """Seal clean local driver/exporter bytes to the checked-out commit."""

    root = repo_root.resolve()
    component_relpaths = LOCAL_EVALUATOR_COMPONENTS
    commit_result = _capture_command(["git", "-C", str(root), "rev-parse", "HEAD"])
    if commit_result.returncode != 0:
        raise T2ICompBenchError(f"cannot resolve local repository commit: {commit_result.stderr.strip()}")
    commit = commit_result.stdout.strip()
    components: list[dict[str, Any]] = []
    for relative in component_relpaths:
        path = root / relative
        if not path.is_file():
            raise T2ICompBenchError(f"local evaluator component not found: {path}")
        status = _capture_command(
            [
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--",
                relative,
            ]
        )
        if status.returncode != 0:
            raise T2ICompBenchError(f"cannot inspect local evaluator component status: {relative}")
        if status.stdout.strip():
            raise T2ICompBenchError(
                f"local evaluator component has uncommitted changes: {relative}; "
                "commit the reviewed evaluator before running official metrics"
            )
        head_blob = subprocess.run(
            ["git", "-C", str(root), "show", f"{commit}:{relative}"],
            capture_output=True,
            check=False,
        )
        if head_blob.returncode != 0:
            raise T2ICompBenchError(f"local evaluator component is not tracked at {commit}: {relative}")
        runtime_sha = sha256_file(path)
        commit_sha = hashlib.sha256(head_blob.stdout).hexdigest()
        if runtime_sha != commit_sha:
            raise T2ICompBenchError(
                f"local evaluator component bytes differ from {commit}: {relative}"
            )
        components.append(
            {
                "path": relative,
                "sha256": runtime_sha,
                "bytes": path.stat().st_size,
            }
        )
    return {
        "repo_path": str(root),
        "repo_git_commit": commit,
        "components_match_repo_git_commit": True,
        "components": components,
    }


def validate_local_evaluator_provenance(
    provenance: Mapping[str, Any],
    *,
    repo_root: Path,
) -> None:
    """Verify the live local files against a manifest's evaluator seals."""

    root = repo_root.resolve()
    if Path(str(provenance.get("repo_path", ""))).resolve() != root:
        raise T2ICompBenchError("local evaluator provenance repo_path does not identify this repository")
    commit = provenance.get("repo_git_commit")
    if not isinstance(commit, str) or len(commit) != 40:
        raise T2ICompBenchError("local evaluator provenance repo_git_commit must be a full commit")
    if provenance.get("components_match_repo_git_commit") is not True:
        raise T2ICompBenchError("local evaluator components_match_repo_git_commit must be true")
    components = provenance.get("components")
    if not isinstance(components, list) or not components:
        raise T2ICompBenchError("local evaluator provenance must seal at least one component")
    raw_paths = [
        raw.get("path") if isinstance(raw, dict) else None
        for raw in components
    ]
    if (
        len(raw_paths) != len(LOCAL_EVALUATOR_COMPONENTS)
        or set(raw_paths) != set(LOCAL_EVALUATOR_COMPONENTS)
    ):
        raise T2ICompBenchError(
            "local evaluator provenance must seal the exact component set"
        )
    paths: set[str] = set()
    for raw in components:
        if not isinstance(raw, dict) or not isinstance(raw.get("path"), str):
            raise T2ICompBenchError("local evaluator provenance contains a malformed component")
        relative = str(raw["path"])
        if relative in paths or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise T2ICompBenchError(f"invalid local evaluator component path: {relative!r}")
        paths.add(relative)
        expected_sha = _require_sha256(
            raw.get("sha256"), label=f"local evaluator component {relative}/sha256"
        )
        expected_bytes = raw.get("bytes")
        if not isinstance(expected_bytes, int) or expected_bytes < 0:
            raise T2ICompBenchError(f"local evaluator component {relative}/bytes is invalid")
        _verify_file_seal(
            root / relative,
            expected_sha=expected_sha,
            expected_bytes=expected_bytes,
            label=f"local evaluator component {relative}",
        )
        head_blob = subprocess.run(
            ["git", "-C", str(root), "show", f"{commit}:{relative}"],
            capture_output=True,
            check=False,
        )
        if head_blob.returncode != 0 or hashlib.sha256(head_blob.stdout).hexdigest() != expected_sha:
            raise T2ICompBenchError(
                f"local evaluator component seal does not match {commit}: {relative}"
            )


def prompt_file_path(clone_dir: Path, category: str) -> Path:
    return clone_dir / "examples" / "dataset" / f"{category}_val.txt"


def read_prompts(path: Path) -> list[str]:
    """Read prompts the way the official generation loop does.

    Six of the eight prompt files are CRLF and two are LF, so this has to stay
    a text-mode read: universal newlines drop the ``\\r`` that would otherwise
    survive into the comparison against the image file names.
    """
    if not path.is_file():
        raise T2ICompBenchError(f"prompt file not found: {path}")
    data = path.read_text(encoding="utf-8").splitlines()
    return [line.strip("\n").split("\t")[0] for line in data]


def build_commands(
    category: str,
    *,
    clone_dir: Path,
    venv_python: Path,
    work_dir: Path,
    dataset_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Render the official invocations for a category.

    ``--complex`` is emitted only for the complex category.  It is declared
    upstream as ``type=bool``, so ``--complex False`` would evaluate to True;
    the flag has to be absent, not falsy, to be off.
    """
    effective_dataset_dir = str(
        (dataset_dir or clone_dir / "examples" / "dataset").resolve()
    )
    commands: list[dict[str, Any]] = []
    for step in CATEGORY_STEPS[category]:
        argv = [
            str(venv_python),
            str((clone_dir / step.script).resolve()),
            f"{step.out_flag}={work_dir}",
        ]
        argv.extend(
            token.replace(DATASET_DIR_TOKEN, effective_dataset_dir)
            for token in step.extra
        )
        if step.use_complex:
            argv.extend(["--complex", "True"])
        commands.append(
            {
                "name": step.name,
                "argv": argv,
                "cwd": str((clone_dir / step.cwd).resolve()),
            }
        )
    return commands


def validate_samples(samples_dir: Path, prompts: Sequence[str]) -> dict[str, Any]:
    """Check the image population the official evaluators assume.

    Indices must cover ``0..N-1`` with no gaps, ``N`` must be ten per prompt,
    and every image ``i`` must carry prompt ``prompts[i // 10]`` so the 3-in-1
    block mapping and the BLIP prompt read-back both land on the intended
    prompt.  Checking only each block's first image is insufficient because
    BLIP parses the prompt independently from every file name.
    """
    if not samples_dir.is_dir():
        raise T2ICompBenchError(f"samples directory not found: {samples_dir}")
    names = sorted(entry.name for entry in samples_dir.iterdir() if entry.is_file())
    if not names:
        raise T2ICompBenchError(f"no images in {samples_dir}")

    expected = len(prompts) * IMAGES_PER_PROMPT
    by_index: dict[int, str] = {}
    for name in names:
        match = SAMPLE_RE.match(name)
        if match is None:
            raise T2ICompBenchError(f"{name!r} does not match '<prompt>_<000000>.png' in {samples_dir}")
        if name.count("_") != 1:
            raise T2ICompBenchError(
                f"{name!r} has {name.count('_')} underscores; numeracy_eval.py reads the index as "
                f"split('_')[1] while BLIP_vqa.py reads split('_')[-1], which agree only at exactly one"
            )
        index = int(match.group("index"))
        if index in by_index:
            raise T2ICompBenchError(f"duplicate sample index {index:06d} in {samples_dir}")
        by_index[index] = match.group("prompt")

    if len(names) != expected:
        raise T2ICompBenchError(
            f"expected {expected} images ({len(prompts)} prompts x {IMAGES_PER_PROMPT}), found {len(names)}"
        )
    missing = [i for i in range(expected) if i not in by_index]
    if missing:
        raise T2ICompBenchError(
            f"sample indices are not contiguous 0..{expected - 1}; first missing is {missing[0]:06d}"
        )
    for index in range(expected):
        prompt_position = index // IMAGES_PER_PROMPT
        prompt = prompts[prompt_position]
        found = by_index[index]
        if found != prompt:
            raise T2ICompBenchError(
                f"image {index:06d} carries prompt {found!r}, but the prompt file "
                f"has {prompt!r} at position {prompt_position}; the index-to-prompt mapping is broken"
            )
    return {"n_images": len(names), "n_prompts": len(prompts), "images_per_prompt": IMAGES_PER_PROMPT}


def _resolved_symlink_target(path: Path) -> Path:
    raw = Path(os.readlink(path))
    return (raw if raw.is_absolute() else path.parent / raw).resolve()


def validate_metric_stage_manifest(
    *,
    work_dir: Path,
    run_id: str,
    arm: str,
    category: str,
    prompt_file: Path,
    prompts: Sequence[str],
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Seal-check one exported ``samples/`` tree before official code reads it.

    The stage manifest is not merely descriptive: it is the allow-list for
    every symlink and the recorded SHA-256 of every generated source image.
    Both the symlink destination and the bytes reached through the staged name
    are checked, and the manifest itself is hashed before and after the scan.
    """

    resolved_work = work_dir.resolve()
    samples_dir = resolved_work / "samples"
    manifest_path = resolved_work / METRIC_STAGE_MANIFEST_NAME
    manifest_sha_before = sha256_file(manifest_path) if manifest_path.is_file() else None
    manifest = _load_json_object(manifest_path, label="metric stage manifest")
    schema = manifest.get("schema")
    if schema not in METRIC_STAGE_SCHEMAS:
        raise T2ICompBenchError(
            f"unexpected metric stage schema {schema!r} in {manifest_path}; "
            f"expected one of {sorted(METRIC_STAGE_SCHEMAS)}"
        )
    expected_identity = {"run_id": run_id, "arm": arm, "category": category}
    for key, expected in expected_identity.items():
        if manifest.get(key) != expected:
            raise T2ICompBenchError(
                f"metric stage {key} is {manifest.get(key)!r}, expected {expected!r}: {manifest_path}"
            )
    method = manifest.get("method")
    if arm in EXPORT_METHODS and method != arm:
        raise T2ICompBenchError(
            f"metric stage method is {method!r}, but registered arm {arm!r} requires method == arm"
        )
    if not isinstance(method, str) or not method:
        raise T2ICompBenchError(f"metric stage method must be a non-empty string: {manifest_path}")
    generation_out_dir = manifest.get("generation_out_dir")
    if not isinstance(generation_out_dir, str) or not Path(generation_out_dir).is_absolute():
        raise T2ICompBenchError(
            f"metric stage generation_out_dir must be an absolute path: {generation_out_dir!r}"
        )
    generation_root = Path(generation_out_dir).resolve()
    generation_manifest_record = manifest.get("generation_manifest")
    if not isinstance(generation_manifest_record, dict):
        raise T2ICompBenchError("metric stage must seal its source generation manifest")
    generation_manifest_text = generation_manifest_record.get("path")
    if not isinstance(generation_manifest_text, str) or not Path(generation_manifest_text).is_absolute():
        raise T2ICompBenchError("metric stage generation_manifest/path must be absolute")
    generation_manifest_path = Path(generation_manifest_text).resolve()
    generation_manifest_sha = _require_sha256(
        generation_manifest_record.get("sha256"),
        label="metric stage generation_manifest/sha256",
    )
    if not generation_manifest_path.is_file():
        raise T2ICompBenchError(f"source generation manifest not found: {generation_manifest_path}")
    current_generation_manifest_sha = sha256_file(generation_manifest_path)
    if current_generation_manifest_sha != generation_manifest_sha:
        raise T2ICompBenchError(
            f"source generation manifest sha256 is {current_generation_manifest_sha}, "
            f"expected {generation_manifest_sha}"
        )
    if Path(str(manifest.get("samples_dir", ""))).resolve() != samples_dir:
        raise T2ICompBenchError(
            f"metric stage samples_dir does not identify {samples_dir}: {manifest.get('samples_dir')!r}"
        )
    recorded_prompt_path = manifest.get("prompt_file")
    if not isinstance(recorded_prompt_path, str):
        raise T2ICompBenchError(f"metric stage prompt_file must be a path string: {manifest_path}")
    if Path(recorded_prompt_path).resolve() != prompt_file.resolve():
        raise T2ICompBenchError(
            f"metric stage prompt_file is {recorded_prompt_path!r}, expected {str(prompt_file.resolve())!r}"
        )
    prompt_sha = sha256_file(prompt_file)
    if manifest.get("prompt_file_sha256") != prompt_sha:
        raise T2ICompBenchError(
            f"metric stage prompt_file_sha256 is {manifest.get('prompt_file_sha256')!r}, "
            f"current prompt file is {prompt_sha}"
        )

    sealed_dataset_assets: dict[str, Any] = {}
    for key in ("selection_manifest", "dataset_contract"):
        record = manifest.get(key)
        if not isinstance(record, dict):
            raise T2ICompBenchError(f"metric stage must seal {key}")
        path_text = record.get("path")
        if not isinstance(path_text, str) or not Path(path_text).is_absolute():
            raise T2ICompBenchError(f"metric stage {key}/path must be an absolute path")
        path = Path(path_text).resolve()
        expected_sha = _require_sha256(record.get("sha256"), label=f"metric stage {key}/sha256")
        if not path.is_file():
            raise T2ICompBenchError(f"metric stage {key} not found: {path}")
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise T2ICompBenchError(
                f"metric stage {key} sha256 is {actual_sha}, expected {expected_sha}"
            )
        sealed_dataset_assets[key] = {"path": str(path), "sha256": expected_sha}

    dataset_contract = _load_json_object(
        Path(sealed_dataset_assets["dataset_contract"]["path"]),
        label="metric dataset contract",
    )
    selection_from_contract = dataset_contract.get("selection_manifest")
    if not isinstance(selection_from_contract, dict):
        raise T2ICompBenchError("metric dataset contract does not seal its selection manifest")
    selection_contract_path = _resolve_contract_path(
        str(selection_from_contract.get("path", "")), repo_root=repo_root
    )
    if (
        selection_contract_path
        != Path(sealed_dataset_assets["selection_manifest"]["path"])
        or selection_from_contract.get("sha256")
        != sealed_dataset_assets["selection_manifest"]["sha256"]
    ):
        raise T2ICompBenchError("metric stage dataset contract and selection manifest seals disagree")
    for stage_key, contract_key in (("dataset_id", "dataset_id"), ("dataset_version", "version")):
        if manifest.get(stage_key) != dataset_contract.get(contract_key):
            raise T2ICompBenchError(
                f"metric stage {stage_key} is {manifest.get(stage_key)!r}, dataset contract has "
                f"{dataset_contract.get(contract_key)!r}"
            )
    dataset_source = dataset_contract.get("source")
    if not isinstance(dataset_source, dict):
        raise T2ICompBenchError("metric dataset contract source must be an object")
    if (
        dataset_source.get("repository") != SOURCE_REPOSITORY
        or dataset_source.get("commit") != SOURCE_COMMIT
    ):
        raise T2ICompBenchError("metric dataset contract source repo/commit is not the official pin")
    source_pin_path_value = dataset_source.get("pin_path")
    if not isinstance(source_pin_path_value, str) or not source_pin_path_value:
        raise T2ICompBenchError("metric dataset contract source/pin_path must be non-empty")
    dataset_source_pin_path = _resolve_contract_path(
        source_pin_path_value, repo_root=repo_root
    )
    dataset_source_pin_sha = _require_sha256(
        dataset_source.get("pin_sha256"), label="metric dataset contract source/pin_sha256"
    )
    if (
        not dataset_source_pin_path.is_file()
        or sha256_file(dataset_source_pin_path) != dataset_source_pin_sha
    ):
        raise T2ICompBenchError("metric dataset contract source pin bytes do not match its seal")
    selection = _load_json_object(
        Path(sealed_dataset_assets["selection_manifest"]["path"]),
        label="metric selection manifest",
    )
    if selection.get("schema") != "holsd.t2icb_subset.v1":
        raise T2ICompBenchError(
            f"unexpected metric selection schema {selection.get('schema')!r}"
        )
    independent = selection.get("independent_seed_manifest")
    if not isinstance(independent, dict):
        raise T2ICompBenchError("metric selection manifest does not seal independent_seed_manifest")
    independent_path_value = independent.get("path")
    if not isinstance(independent_path_value, str) or not independent_path_value:
        raise T2ICompBenchError("selection independent_seed_manifest/path must be a non-empty string")
    independent_path = Path(independent_path_value)
    if not independent_path.is_absolute():
        independent_path = (
            Path(sealed_dataset_assets["selection_manifest"]["path"]).parent
            / independent_path
        )
    independent_path = independent_path.resolve()
    independent_sha = _require_sha256(
        independent.get("sha256"), label="selection independent_seed_manifest/sha256"
    )
    if independent_path != generation_manifest_path:
        raise T2ICompBenchError(
            f"metric stage generation_manifest is {generation_manifest_path}, but the sealed "
            f"selection requires {independent_path}"
        )
    if independent_sha != generation_manifest_sha:
        raise T2ICompBenchError(
            "metric stage generation_manifest sha256 differs from the sealed selection manifest"
        )
    if independent.get("rows") != len(generation_manifest_path.read_text(encoding="utf-8").splitlines()):
        raise T2ICompBenchError("source generation manifest row count differs from the sealed selection")
    prompt_hashes = selection.get("prompt_files_sha256")
    if not isinstance(prompt_hashes, dict) or prompt_hashes.get(prompt_file.name) != prompt_sha:
        raise T2ICompBenchError(
            f"metric prompt {prompt_file.name} sha256 {prompt_sha} does not match the sealed selection"
        )

    expected_images = len(prompts) * IMAGES_PER_PROMPT
    if manifest.get("prompt_count") != len(prompts):
        raise T2ICompBenchError(
            f"metric stage prompt_count is {manifest.get('prompt_count')!r}, expected {len(prompts)}"
        )
    if manifest.get("image_count") != expected_images:
        raise T2ICompBenchError(
            f"metric stage image_count is {manifest.get('image_count')!r}, expected {expected_images}"
        )
    seeds = manifest.get("seeds")
    if not isinstance(seeds, list) or len(seeds) != IMAGES_PER_PROMPT or len(set(seeds)) != len(seeds):
        raise T2ICompBenchError(
            f"metric stage seeds must contain {IMAGES_PER_PROMPT} distinct values: {seeds!r}"
        )
    if (
        selection.get("prompts_per_category") != len(prompts)
        or selection.get("images_per_prompt") != IMAGES_PER_PROMPT
        or selection.get("seeds") != seeds
    ):
        raise T2ICompBenchError(
            "metric stage prompt/seed population disagrees with the sealed selection manifest"
        )
    images = manifest.get("images")
    if not isinstance(images, list) or len(images) != expected_images:
        raise T2ICompBenchError(
            f"metric stage images must contain {expected_images} rows: {manifest_path}"
        )
    generation_rows: list[dict[str, Any]] = []
    with generation_manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                generation_row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise T2ICompBenchError(
                    f"invalid source generation manifest JSON at line {line_number}"
                ) from exc
            if not isinstance(generation_row, dict):
                raise T2ICompBenchError(
                    f"source generation manifest line {line_number} is not an object"
                )
            if generation_row.get("experiment") == f"t2icb_{category}":
                generation_rows.append(generation_row)
    if len(generation_rows) != expected_images:
        raise T2ICompBenchError(
            f"source generation manifest has {len(generation_rows)} {category} rows, "
            f"expected {expected_images}"
        )
    canonical_population = []
    for index, (raw_row, generation_row) in enumerate(zip(images, generation_rows)):
        if not isinstance(raw_row, dict):
            raise T2ICompBenchError("metric stage population contains a non-object image row")
        match = SAMPLE_RE.match(str(raw_row.get("name", "")))
        expected_generation_identity = {
            "prompt_id": generation_row.get("prompt_id"),
            "prompt": generation_row.get("prompt"),
            "seed": generation_row.get("seed"),
        }
        staged_identity = {
            "prompt_id": raw_row.get("prompt_id"),
            "prompt": None if match is None else match.group("prompt"),
            "seed": raw_row.get("seed"),
        }
        if staged_identity != expected_generation_identity:
            raise T2ICompBenchError(
                f"metric stage image {index:06d} identity {staged_identity!r} does not exact-join "
                f"the source generation manifest row {expected_generation_identity!r}"
            )
        canonical_population.append(
            {
                **staged_identity,
                "index": raw_row.get("index"),
            }
        )
    population_identity_sha256 = hashlib.sha256(
        json.dumps(
            canonical_population,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    recorded_population_sha = _require_sha256(
        manifest.get("population_identity_sha256"),
        label="metric stage population_identity_sha256",
    )
    if recorded_population_sha != population_identity_sha256:
        raise T2ICompBenchError(
            f"metric stage population identity is {recorded_population_sha}, "
            f"recomputed {population_identity_sha256}"
        )
    indices = [row.get("index") if isinstance(row, dict) else None for row in images]
    if indices != list(range(expected_images)):
        raise T2ICompBenchError("metric stage image rows are not ordered contiguously by index")

    manifest_names: set[str] = set()
    for index, raw_row in enumerate(images):
        if not isinstance(raw_row, dict):
            raise T2ICompBenchError(f"metric stage image row {index} is not an object")
        expected_prompt = prompts[index // IMAGES_PER_PROMPT]
        expected_seed = seeds[index % IMAGES_PER_PROMPT]
        name = raw_row.get("name")
        if not isinstance(name, str):
            raise T2ICompBenchError(f"metric stage image row {index} has no string name")
        match = SAMPLE_RE.match(name)
        if match is None or int(match.group("index")) != index:
            raise T2ICompBenchError(f"metric stage image row {index} has invalid name {name!r}")
        if match.group("prompt") != expected_prompt:
            raise T2ICompBenchError(
                f"metric stage image {index:06d} carries prompt {match.group('prompt')!r}, "
                f"expected {expected_prompt!r}"
            )
        if raw_row.get("seed") != expected_seed:
            raise T2ICompBenchError(
                f"metric stage image {index:06d} seed is {raw_row.get('seed')!r}, "
                f"expected {expected_seed!r}"
            )
        if name in manifest_names:
            raise T2ICompBenchError(f"metric stage repeats sample name {name!r}")
        manifest_names.add(name)

        source_text = raw_row.get("source")
        if not isinstance(source_text, str) or not Path(source_text).is_absolute():
            raise T2ICompBenchError(f"metric stage image {index:06d} source must be an absolute path")
        source = Path(source_text).resolve()
        if not source.is_relative_to(generation_root):
            raise T2ICompBenchError(
                f"metric stage image {index:06d} source {source} is outside "
                f"generation_out_dir {generation_root}"
            )
        recorded_sha = _require_sha256(
            raw_row.get("sha256"), label=f"metric stage image {index:06d}/sha256"
        )
        staged = samples_dir / name
        if not staged.is_symlink():
            raise T2ICompBenchError(f"staged metric image is not a symlink: {staged}")
        staged_target = _resolved_symlink_target(staged)
        if staged_target != source:
            raise T2ICompBenchError(
                f"staged metric image {staged} targets {staged_target}, recorded source is {source}"
            )
        if not source.is_file():
            raise T2ICompBenchError(f"recorded metric source image not found: {source}")
        source_sha = sha256_file(source)
        staged_sha = sha256_file(staged)
        if source_sha != recorded_sha or staged_sha != recorded_sha:
            raise T2ICompBenchError(
                f"staged metric image {index:06d} bytes drifted: recorded={recorded_sha}, "
                f"source={source_sha}, staged={staged_sha}"
            )

    if not samples_dir.is_dir():
        raise T2ICompBenchError(f"staged samples directory not found: {samples_dir}")
    actual_names = {entry.name for entry in samples_dir.iterdir()}
    if actual_names != manifest_names:
        missing = sorted(manifest_names - actual_names)
        stray = sorted(actual_names - manifest_names)
        raise T2ICompBenchError(
            f"staged samples differ from the metric manifest; missing={missing[:1]}, stray={stray[:1]}"
        )
    manifest_sha_after = sha256_file(manifest_path)
    if manifest_sha_before != manifest_sha_after:
        raise T2ICompBenchError(f"metric stage manifest changed during validation: {manifest_path}")
    return {
        "path": str(manifest_path),
        "schema": schema,
        "sha256": manifest_sha_after,
        "image_count": expected_images,
        "method": method,
        "generation_out_dir": str(generation_root),
        "generation_manifest": {
            "path": str(generation_manifest_path),
            "sha256": generation_manifest_sha,
        },
        "dataset_id": manifest.get("dataset_id"),
        "dataset_version": manifest.get("dataset_version"),
        "population_identity_sha256": population_identity_sha256,
        "dataset_source_pin": {
            "path": str(dataset_source_pin_path),
            "sha256": dataset_source_pin_sha,
            "repository": SOURCE_REPOSITORY,
            "commit": SOURCE_COMMIT,
        },
        **sealed_dataset_assets,
    }


def validate_complex_auxiliary_prompts(
    *,
    work_dir: Path,
    prompt_file: Path,
    category: str,
) -> list[dict[str, Any]]:
    """Seal the three auxiliary prompt files consumed by official 3-in-1."""

    if category != "complex":
        return []
    stage_path = work_dir.resolve() / METRIC_STAGE_MANIFEST_NAME
    stage = _load_json_object(stage_path, label="metric stage manifest")
    selection_record = stage.get("selection_manifest")
    contract_record = stage.get("dataset_contract")
    for label, record in (
        ("selection_manifest", selection_record),
        ("dataset_contract", contract_record),
    ):
        if not isinstance(record, dict):
            raise T2ICompBenchError(f"complex metric stage must seal {label}")
        path_text = record.get("path")
        if not isinstance(path_text, str) or not Path(path_text).is_absolute():
            raise T2ICompBenchError(f"complex metric stage {label}/path must be absolute")
        path = Path(path_text).resolve()
        expected = _require_sha256(record.get("sha256"), label=f"metric stage {label}/sha256")
        if not path.is_file() or sha256_file(path) != expected:
            raise T2ICompBenchError(f"complex metric stage {label} bytes do not match its seal: {path}")

    selection_path = Path(str(selection_record["path"])).resolve()
    selection = _load_json_object(selection_path, label="complex selection manifest")
    seals = selection.get("source_complex_auxiliary_sha256")
    if not isinstance(seals, dict) or set(seals) != set(COMPLEX_AUXILIARY_PROMPTS):
        raise T2ICompBenchError(
            "complex selection manifest must seal all official auxiliary prompt files"
        )
    result: list[dict[str, Any]] = []
    for name in COMPLEX_AUXILIARY_PROMPTS:
        expected = _require_sha256(seals.get(name), label=f"complex auxiliary {name}/sha256")
        path = prompt_file.parent / name
        if not path.is_file():
            raise T2ICompBenchError(f"complex auxiliary prompt file not found: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise T2ICompBenchError(
                f"complex auxiliary prompt {name} sha256 is {actual}, expected {expected}"
            )
        result.append({"path": str(path.resolve()), "sha256": actual})
    return result


def validate_curated_artifact_seals(
    manifest: Mapping[str, Any],
    curated_dir: Path,
) -> list[dict[str, Any]]:
    """Validate the two immutable score/result copies described by an eval manifest."""

    category = manifest.get("category")
    if category not in METRICS:
        raise T2ICompBenchError(f"eval manifest has unknown category {category!r}")
    source = METRICS[str(category)]
    expected = {
        "result": Path(source.result_path).name,
        "score": Path(source.score_path).name,
    }
    raw_artifacts = manifest.get("curated_artifacts")
    if not isinstance(raw_artifacts, list) or len(raw_artifacts) != len(expected):
        raise T2ICompBenchError("eval manifest must seal exactly the result and score curated artifacts")
    by_role: dict[str, dict[str, Any]] = {}
    for raw in raw_artifacts:
        if not isinstance(raw, dict) or raw.get("role") not in expected:
            raise T2ICompBenchError(f"malformed curated artifact seal: {raw!r}")
        role = str(raw["role"])
        if role in by_role:
            raise T2ICompBenchError(f"duplicate curated artifact role {role!r}")
        if raw.get("path") != expected[role]:
            raise T2ICompBenchError(
                f"curated {role} path is {raw.get('path')!r}, expected {expected[role]!r}"
            )
        expected_sha = _require_sha256(raw.get("sha256"), label=f"curated {role}/sha256")
        expected_bytes = raw.get("bytes")
        if not isinstance(expected_bytes, int) or expected_bytes < 0:
            raise T2ICompBenchError(f"curated {role}/bytes must be a non-negative integer")
        path = curated_dir.resolve() / expected[role]
        _verify_file_seal(
            path,
            expected_sha=expected_sha,
            expected_bytes=expected_bytes,
            label=f"curated {role} artifact",
        )
        by_role[role] = dict(raw)
    if set(by_role) != set(expected):
        raise T2ICompBenchError(f"curated artifact roles are {sorted(by_role)}, expected {sorted(expected)}")
    return [by_role[role] for role in ("result", "score")]


def validate_curated_score_semantics(
    manifest: Mapping[str, Any],
    curated_dir: Path,
) -> dict[str, Any]:
    """Reparse sealed score/result artifacts and match every recorded summary."""

    category = str(manifest["category"])
    source = METRICS[category]
    result_path = curated_dir.resolve() / Path(source.result_path).name
    score_path = curated_dir.resolve() / Path(source.score_path).name
    score = parse_score(score_path.read_text(encoding="utf-8"), source.score_prefix)
    distribution = summarize_answers(result_path)
    population = manifest.get("population")
    if not isinstance(population, dict) or population.get("n_images") != distribution["n_scored"]:
        raise T2ICompBenchError(
            f"curated result has {distribution['n_scored']} rows, but eval population is "
            f"{None if not isinstance(population, dict) else population.get('n_images')}"
        )
    if manifest.get("score") != score:
        raise T2ICompBenchError(f"eval manifest score is {manifest.get('score')!r}, curated score is {score!r}")
    if manifest.get("distribution") != distribution:
        raise T2ICompBenchError("eval manifest distribution differs from the sealed curated result")
    score_sources = manifest.get("score_sources")
    if not isinstance(score_sources, dict):
        raise T2ICompBenchError("eval manifest score_sources must be an object")
    difference = abs(distribution["mean"] - score)
    expected_values = {
        "official_score": score,
        "recomputed_mean": distribution["mean"],
        "abs_difference": difference,
        "agreement_tolerance": SCORE_AGREEMENT_TOLERANCE,
    }
    for key, expected in expected_values.items():
        if score_sources.get(key) != expected:
            raise T2ICompBenchError(
                f"eval manifest score_sources/{key} is {score_sources.get(key)!r}, expected {expected!r}"
            )
    if difference > SCORE_AGREEMENT_TOLERANCE:
        raise T2ICompBenchError(f"sealed curated score/result disagree by {difference:.3e}")
    return {"score": score, "distribution": distribution}


def validate_completed_eval_manifest(
    manifest_path: Path,
    *,
    run_id: str,
    arm: str | None,
    category: str,
    stage_reference: Mapping[str, Any] | None,
    repo_root: Path | None = None,
    official_preflight: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a completed category before it may be skipped or summarized."""

    manifest = _load_json_object(manifest_path, label="completed eval manifest")
    if manifest.get("schema") != EVAL_MANIFEST_SCHEMA:
        raise T2ICompBenchError(
            f"completed eval manifest schema is {manifest.get('schema')!r}, "
            f"expected {EVAL_MANIFEST_SCHEMA!r}"
        )
    for key, expected in (("run_id", run_id), ("arm", arm), ("category", category)):
        if manifest.get(key) != expected:
            raise T2ICompBenchError(
                f"completed eval manifest {key} is {manifest.get(key)!r}, expected {expected!r}"
            )
    recorded_stage = manifest.get("metric_stage_manifest")
    if stage_reference is None:
        if recorded_stage is not None:
            raise T2ICompBenchError("arm-less eval manifest unexpectedly records a metric stage manifest")
    else:
        if not isinstance(recorded_stage, dict):
            raise T2ICompBenchError("completed eval manifest does not seal its metric stage manifest")
        for key in (
            "path",
            "schema",
            "sha256",
            "image_count",
            "method",
            "generation_out_dir",
            "generation_manifest",
            "dataset_id",
            "dataset_version",
            "population_identity_sha256",
            "dataset_source_pin",
            "selection_manifest",
            "dataset_contract",
        ):
            if recorded_stage.get(key) != stage_reference.get(key):
                raise T2ICompBenchError(
                    f"completed eval metric stage {key} is {recorded_stage.get(key)!r}, "
                    f"current stage is {stage_reference.get(key)!r}"
                )
    validate_curated_artifact_seals(manifest, manifest_path.parent)
    validate_curated_score_semantics(manifest, manifest_path.parent)
    if official_preflight is not None and manifest.get("official_preflight") != dict(
        official_preflight
    ):
        raise T2ICompBenchError(
            "completed eval manifest official preflight identity differs from the current "
            "source pin/clone/venv/weight preflight"
        )
    if repo_root is not None:
        provenance = manifest.get("local_evaluator_provenance")
        if not isinstance(provenance, dict):
            raise T2ICompBenchError("completed eval manifest does not seal local evaluator provenance")
        validate_local_evaluator_provenance(provenance, repo_root=repo_root)
    return manifest


def link_samples(work_dir: Path, samples_dir: Path) -> None:
    link = work_dir / "samples"
    target = samples_dir.resolve()
    if link.is_symlink():
        if Path(os.readlink(link)).resolve() != target:
            raise T2ICompBenchError(f"{link} already points at {os.readlink(link)}, not {target}")
        return
    if link.exists():
        raise T2ICompBenchError(f"{link} exists and is not a symlink")
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target)


def require_fresh_metric_work_dir(work_dir: Path) -> None:
    """Refuse an official rerun that could consume cached/interrupted outputs.

    In particular, the upstream 3D evaluator skips depth inference whenever
    ``labels/depth`` already exists.  A reused category directory can therefore
    score stale maps.  Keeping only the staged inputs makes retries explicit:
    use ``--skip-completed`` for a sealed result, or remove/move the scratch
    outputs before requesting a fresh run.
    """

    if not work_dir.is_dir():
        return
    allowed = {"samples", METRIC_STAGE_MANIFEST_NAME}
    stale = sorted(entry.name for entry in work_dir.iterdir() if entry.name not in allowed)
    if stale:
        raise T2ICompBenchError(
            f"metric work directory contains stale/interrupted evaluator output {stale[0]!r}: "
            f"{work_dir}; refusing a non-skipped rerun"
        )


def parse_score(text: str, prefix: str) -> float:
    """Pull the score out of the free-text file the official script writes."""
    position = text.find(prefix)
    if position < 0:
        raise T2ICompBenchError(f"expected {prefix!r} in score file, got {text.strip()!r}")
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text[position + len(prefix):])
    if match is None:
        raise T2ICompBenchError(f"no number after {prefix!r} in {text.strip()!r}")
    return float(match.group(0))


def summarize_answers(result_path: Path) -> dict[str, Any]:
    """Per-image score distribution, recomputed from the metric's own json."""
    rows = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise T2ICompBenchError(f"{result_path} must hold a JSON list of score objects")
    question_ids = [row.get("question_id") for row in rows]
    if question_ids != list(range(len(rows))):
        raise T2ICompBenchError(
            f"{result_path} question_id values must be exactly ordered 0..{len(rows) - 1}"
        )
    answers: list[float] = []
    for index, row in enumerate(rows):
        if "answer" not in row:
            raise T2ICompBenchError(f"{result_path} row {index} has no answer")
        try:
            answer = float(row["answer"])
        except (TypeError, ValueError) as exc:
            raise T2ICompBenchError(
                f"{result_path} row {index} has a non-numeric answer {row['answer']!r}"
            ) from exc
        if not math.isfinite(answer):
            raise T2ICompBenchError(f"{result_path} row {index} has a non-finite answer {answer!r}")
        answers.append(answer)
    if not answers:
        raise T2ICompBenchError(f"{result_path} holds no scores")
    return {
        "n_scored": len(answers),
        "mean": statistics.fmean(answers),
        "median": statistics.median(answers),
        "min": min(answers),
        "max": max(answers),
        "zero_fraction": sum(1 for value in answers if value == 0.0) / len(answers),
    }


def read_score(work_dir: Path, category: str) -> dict[str, Any]:
    source = METRICS[category]
    score_file = work_dir / source.score_path
    result_file = work_dir / source.result_path
    if not score_file.is_file():
        raise T2ICompBenchError(f"score file not written: {score_file}")
    if not result_file.is_file():
        raise T2ICompBenchError(f"per-image result not written: {result_file}")
    official_score = parse_score(score_file.read_text(encoding="utf-8"), source.score_prefix)
    distribution = summarize_answers(result_file)
    difference = abs(distribution["mean"] - official_score)
    if difference > SCORE_AGREEMENT_TOLERANCE:
        raise T2ICompBenchError(
            f"{category}: {score_file} reports {official_score!r} but re-averaging {result_file} gives "
            f"{distribution['mean']!r} (difference {difference:.3e} > {SCORE_AGREEMENT_TOLERANCE:.0e}); "
            f"the score text was mis-parsed or its format changed. The official outputs are intact in "
            f"{work_dir}, so only the parse needs fixing."
        )
    return {
        "metric": source.metric,
        "score": official_score,
        "score_file": str(score_file),
        "result_file": str(result_file),
        "distribution": distribution,
        "official_score": official_score,
        "recomputed_mean": distribution["mean"],
        "abs_difference": difference,
        "agreement_tolerance": SCORE_AGREEMENT_TOLERANCE,
    }


def gate_verdict(category: str, score: float) -> dict[str, Any]:
    target = PUBLISHED_TARGETS[category]
    delta = score - target
    return {
        "target": target,
        "delta": delta,
        "pass": abs(delta) <= GATE_TOLERANCE + 1e-12,
    }


def no_target_verdict() -> dict[str, Any]:
    """Score-recording mode: the arm has no published row to compare against."""
    return {"target": None, "delta": None, "pass": None}


def work_root_dir(vqa_root: Path, run_id: str, arm: str | None) -> Path:
    root = vqa_root / run_id / "t2icb_work"
    return root / arm if arm else root


def curated_root_dir(vqa_root: Path, run_id: str, arm: str | None) -> Path:
    root = vqa_root / run_id / "t2i_compbench"
    return root / arm if arm else root


def subprocess_env() -> dict[str, str]:
    """Hand the evaluators a clean import path.

    ``PYTHONPATH=src`` and the container's ``LD_LIBRARY_PATH`` both leak the main
    python 3.12 environment into the evaluators' python 3.10 venv.
    """
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["LD_LIBRARY_PATH"] = ":".join(
        ("/usr/local/cuda/compat/lib", "/usr/local/nvidia/lib", "/usr/local/nvidia/lib64")
    )
    return env


def run_command(command: Mapping[str, Any], log_path: Path) -> dict[str, Any]:
    """Run one official script in the evaluator venv; its output goes to ``log_path``."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = utc_now()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"# cwd: {command['cwd']}\n# argv: {command['argv']}\n\n")
        log.flush()
        completed = subprocess.run(
            command["argv"],
            cwd=command["cwd"],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=subprocess_env(),
            check=False,
        )
    record = {
        "name": command["name"],
        "argv": list(command["argv"]),
        "cwd": command["cwd"],
        "returncode": completed.returncode,
        "log": str(log_path),
        "started_at": started,
        "ended_at": utc_now(),
    }
    if completed.returncode != 0:
        raise T2ICompBenchError(
            f"step {command['name']!r} exited {completed.returncode}; see {log_path}"
        )
    return record


def curate(work_dir: Path, curated_dir: Path, category: str) -> list[dict[str, Any]]:
    """Copy the small, final artifacts into the git-tracked tree."""
    source = METRICS[category]
    curated_dir.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, Any]] = []
    for role, relative in (("result", source.result_path), ("score", source.score_path)):
        destination = curated_dir / Path(relative).name
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_bytes((work_dir / relative).read_bytes())
        os.replace(temporary, destination)
        copied.append(
            {
                "role": role,
                "path": destination.name,
                "sha256": sha256_file(destination),
                "bytes": destination.stat().st_size,
            }
        )
    return copied


def evaluate_category(
    category: str,
    *,
    run_id: str,
    clone_dir: Path,
    venv_python: Path,
    generation_root: Path,
    vqa_root: Path,
    repo_root: Path,
    dry_run: bool,
    prompt_dir: Path | None = None,
    published_target_gate: bool = True,
    arm: str | None = None,
    official_preflight: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    work_dir = (work_root_dir(vqa_root, run_id, arm) / category).resolve()
    curated_dir = (curated_root_dir(vqa_root, run_id, arm) / category).resolve()
    if arm is None:
        samples_dir = generation_root / run_id / category / "samples"
    else:
        # export_samples has already materialised work_dir/samples as a
        # real directory of per-image symlinks.
        samples_dir = work_dir / "samples"
    dataset_dir = prompt_dir or clone_dir / "examples" / "dataset"
    prompt_file = dataset_dir / f"{category}_val.txt"
    prompts = read_prompts(prompt_file)

    preflight = (
        dict(official_preflight)
        if official_preflight is not None
        else validate_official_environment(
            repo_root=repo_root,
            clone_dir=clone_dir,
            venv_python=venv_python,
        )
    )

    commands = build_commands(
        category,
        clone_dir=clone_dir,
        venv_python=venv_python,
        work_dir=work_dir,
        dataset_dir=dataset_dir,
    )
    if dry_run:
        return {
            "category": category,
            "dry_run": True,
            "work_dir": str(work_dir),
            "samples_dir": str(samples_dir),
            "n_prompts": len(prompts),
            "expected_images": len(prompts) * IMAGES_PER_PROMPT,
            "commands": commands,
            "official_preflight": preflight,
        }

    require_fresh_metric_work_dir(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    if arm is None:
        link_samples(work_dir, samples_dir)
    elif not samples_dir.is_dir():
        raise T2ICompBenchError(
            f"samples not exported for arm {arm!r}: {samples_dir}; "
            "run scripts.t2i_compbench.export_samples first"
        )
    stage_reference = (
        validate_metric_stage_manifest(
            work_dir=work_dir,
            run_id=run_id,
            arm=arm,
            category=category,
            prompt_file=prompt_file,
            prompts=prompts,
            repo_root=repo_root,
        )
        if arm is not None
        else None
    )
    population = validate_samples(work_dir / "samples", prompts)
    complex_auxiliary_prompts = (
        validate_complex_auxiliary_prompts(
            work_dir=work_dir,
            prompt_file=prompt_file,
            category=category,
        )
        if arm is not None
        else []
    )

    started = utc_now()
    executed = [run_command(command, work_dir / "logs" / f"{command['name']}.log") for command in commands]
    ended = utc_now()

    # The generated source files remain symlinked while official code runs.
    # Recheck the complete stage after the subprocesses finish so an in-flight
    # source mutation cannot be curated as if it belonged to the pre-run seal.
    if arm is not None:
        stage_after = validate_metric_stage_manifest(
            work_dir=work_dir,
            run_id=run_id,
            arm=arm,
            category=category,
            prompt_file=prompt_file,
            prompts=prompts,
            repo_root=repo_root,
        )
        if stage_after != stage_reference:
            raise T2ICompBenchError("metric stage identity changed while official evaluation was running")

    scored = read_score(work_dir, category)
    if scored["distribution"]["n_scored"] != population["n_images"]:
        raise T2ICompBenchError(
            f"{category}: official result scored {scored['distribution']['n_scored']} images, "
            f"expected the complete population of {population['n_images']}"
        )
    verdict = (
        gate_verdict(category, scored["score"])
        if published_target_gate
        else no_target_verdict()
    )
    manifest: dict[str, Any] = {
        "schema": EVAL_MANIFEST_SCHEMA,
        "run_id": run_id,
        "arm": arm,
        "category": category,
        "metric": scored["metric"],
        "score": scored["score"],
        "started_at": started,
        "ended_at": ended,
        "commands": executed,
        "population": population,
        "score_sources": {
            "score_file": scored["score_file"],
            "result_file": scored["result_file"],
            "official_score": scored["official_score"],
            "recomputed_mean": scored["recomputed_mean"],
            "abs_difference": scored["abs_difference"],
            "agreement_tolerance": scored["agreement_tolerance"],
        },
        "distribution": scored["distribution"],
        "prompt_file": {
            "path": str(prompt_file),
            "sha256": sha256_file(prompt_file),
            "count": len(prompts),
        },
        "published_target_gate": published_target_gate,
        "samples_dir": str(samples_dir.resolve()),
        "work_dir": str(work_dir),
        "clone": preflight["clone"],
        "venv_python": preflight["venv"],
        "source_pin": preflight["source_pin"],
        "official_preflight": preflight,
        "metric_stage_manifest": stage_reference,
        "complex_auxiliary_prompts": complex_auxiliary_prompts,
        "local_evaluator_provenance": local_evaluator_provenance(repo_root),
        **verdict,
    }
    manifest["curated_artifacts"] = curate(work_dir, curated_dir, category)
    validate_curated_artifact_seals(manifest, curated_dir)
    validate_curated_score_semantics(manifest, curated_dir)
    manifest_path = curated_dir / "eval_manifest.json"
    write_json(manifest_path, manifest)
    validate_completed_eval_manifest(
        manifest_path,
        run_id=run_id,
        arm=arm,
        category=category,
        stage_reference=stage_reference,
        repo_root=repo_root,
        official_preflight=preflight,
    )
    return manifest


def summary_row(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "metric": manifest["metric"],
        "score": manifest["score"],
        "n_images": manifest["population"]["n_images"],
        "target": manifest["target"],
        "delta": manifest["delta"],
        "pass": manifest["pass"],
    }


def write_summary(
    vqa_root: Path,
    run_id: str,
    *,
    arm: str | None = None,
) -> dict[str, Any]:
    """Rebuild the run summary from whichever category manifests exist.

    Whether the published-target gate was applied is read back from the
    manifests themselves (a gated row carries a non-null ``target``), so a
    rebuild needs no memory of the original CLI flags.
    """
    curated_root = curated_root_dir(vqa_root, run_id, arm).resolve()
    categories: dict[str, Any] = {}
    for category in CATEGORIES:
        manifest_path = curated_root / category / "eval_manifest.json"
        if manifest_path.is_file():
            manifest = validate_completed_eval_manifest(
                manifest_path,
                run_id=run_id,
                arm=arm,
                category=category,
                stage_reference=(
                    _load_json_object(manifest_path, label="eval manifest used for summary").get(
                        "metric_stage_manifest"
                    )
                    if arm is not None
                    else None
                ),
            )
            categories[category] = summary_row(manifest)
    n_gated = sum(1 for row in categories.values() if row["target"] is not None)
    summary = {
        "schema": "holsd.t2icb_scores_summary.v1",
        "run_id": run_id,
        "arm": arm,
        "generated_at": utc_now(),
        "gate_tolerance": GATE_TOLERANCE if n_gated else None,
        "target_source": (
            "T2I-CompBench++ TABLE XIII, Stable v1-4 row"
            if n_gated
            else "none (score recording only)"
        ),
        "categories": categories,
        "n_pass": (
            sum(1 for row in categories.values() if row["pass"]) if n_gated else None
        ),
        "n_gated": n_gated,
        "n_evaluated": len(categories),
    }
    write_json(curated_root / "scores_summary.json", summary)
    return summary


def resolve_categories(requested: Sequence[str]) -> list[str]:
    if list(requested) == ["all"]:
        return list(CATEGORIES)
    unknown = [name for name in requested if name not in PUBLISHED_TARGETS]
    if unknown:
        raise T2ICompBenchError(f"unknown categories: {unknown}; choose from {list(CATEGORIES)} or 'all'")
    return list(requested)
