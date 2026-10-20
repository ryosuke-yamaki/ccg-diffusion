"""Curated T2I-CompBench++ results: per-image scores with order and seal verification.

Every evaluated category of a run has a curated copy of the official per-image
result, ``output/vqa/<run>/t2i_compbench/[<arm>/]<category>/vqa_result.json``,
next to its ``eval_manifest.json``. ``load_answers`` returns the 3,000 image
scores in the official order (prompt position x 10 + seed position) after
checking the result against the manifest's seal, the population, the prompt
file and the recorded population identity. ``prompt_score_matrix`` averages
the ten seeds of each prompt: the unit of every T2I significance test.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ccg_diffusion.evaluation.t2i_compbench.dataset import (
    DATASET_DIR, IMAGES_PER_PROMPT, PROMPTS_PER_CATEGORY, REPO_ROOT, SELECTION_NAME, SOURCE_PIN_PATH,
    T2ICompBenchError, expected_population_identity,
)
from ccg_diffusion.provenance import sha256_file

VQA_ROOT = REPO_ROOT / "output/vqa"
# Prefix of the work directories recorded by the evaluated runs.
RECORDED_VQA_PREFIX = "/workspace/holsd/output/vqa/"


def numeric_answers(path: Path) -> list[float]:
    """Finite per-image scores of one ``vqa_result.json`` whose question IDs are exactly 0..N-1."""

    if not path.is_file():
        raise T2ICompBenchError(f"metric result not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise T2ICompBenchError(f"metric result must be a non-empty list: {path}")
    answers: list[float] = []
    for position, row in enumerate(payload):
        if not isinstance(row, dict) or "answer" not in row:
            raise T2ICompBenchError(f"missing answer at {path}:{position}")
        if row.get("question_id") != position:
            raise T2ICompBenchError(
                f"question_id at {path}:{position} is {row.get('question_id')!r}; "
                f"expected contiguous index {position}"
            )
        try:
            value = float(row["answer"])
        except (TypeError, ValueError) as exc:
            raise T2ICompBenchError(
                f"non-numeric answer at {path}:{position}: {row['answer']!r}"
            ) from exc
        if not math.isfinite(value):
            raise T2ICompBenchError(f"non-finite answer at {path}:{position}")
        answers.append(value)
    return answers


def curated_result_path(run_id: str, arm: str | None, category: str, *, vqa_root: Path = VQA_ROOT) -> Path:
    """``output/vqa/<run>/t2i_compbench/[<arm>/]<category>/vqa_result.json``."""

    root = vqa_root / run_id / "t2i_compbench"
    return (root / arm if arm else root) / category / "vqa_result.json"


def recorded_work_dir(work_dir: str) -> tuple[str, str | None, str]:
    """``/workspace/holsd/output/vqa/<run>/t2icb_work/[<arm>/]<category>`` -> ``(run, arm, category)``."""

    parts = work_dir.removeprefix(RECORDED_VQA_PREFIX).split("/")
    if len(parts) not in (3, 4) or parts[1] != "t2icb_work":
        raise T2ICompBenchError(f"not a recorded T2I-CompBench++ work directory: {work_dir!r}")
    return parts[0], parts[2] if len(parts) == 4 else None, parts[-1]


def load_answers(run_id: str, arm: str | None, category: str, *, vqa_root: Path = VQA_ROOT,
                 verify: bool = True) -> np.ndarray:
    """The 3,000 curated image scores of one evaluated category, in file order (float64).

    With ``verify`` (default) the result must (1) have contiguous question IDs
    and finite answers, (2) carry the sha256 its ``eval_manifest.json`` seals,
    (3) cover 300 prompts x 10 images, (4) come from the pinned official or the
    frozen dataset prompt file, and (5) when the manifest records an exported
    population, match ``expected_population_identity``.
    """

    path = curated_result_path(run_id, arm, category, vqa_root=vqa_root)
    answers = numeric_answers(path)
    if len(answers) != PROMPTS_PER_CATEGORY * IMAGES_PER_PROMPT:
        raise T2ICompBenchError(f"expected {PROMPTS_PER_CATEGORY * IMAGES_PER_PROMPT} answers: {path}")
    if verify:
        manifest = json.loads((path.parent / "eval_manifest.json").read_text(encoding="utf-8"))
        if manifest["schema"] == "holsd.t2icb_eval.v1":
            # The SD1.4 Vanilla reference run predates the sealed manifests; it lists its curated files only.
            sealed = path.relative_to(vqa_root).as_posix() in [
                name.removeprefix(RECORDED_VQA_PREFIX) for name in manifest["curated_files"]]
        else:
            results = [a for a in manifest["curated_artifacts"] if a["role"] == "result"]
            sealed = len(results) == 1 and results[0]["path"] == path.name and results[0]["sha256"] == sha256_file(path)
        if not sealed:
            raise T2ICompBenchError(f"curated result does not match its eval manifest seal: {path}")
        if manifest["population"] != {"images_per_prompt": 10, "n_images": 3000, "n_prompts": 300}:
            raise T2ICompBenchError(f"unexpected evaluated population: {path}")
        pin = json.loads((REPO_ROOT / SOURCE_PIN_PATH).read_text(encoding="utf-8"))
        selection = json.loads((DATASET_DIR / SELECTION_NAME).read_text(encoding="utf-8"))
        accepted = {pin["val_txt"][category]["sha256"], selection["prompt_files_sha256"][f"{category}_val.txt"]}
        if manifest["prompt_file"]["sha256"] not in accepted:
            raise T2ICompBenchError(f"evaluated prompt file is neither the official nor the dataset file: {path}")
        stage = manifest.get("metric_stage_manifest")
        if stage and stage["population_identity_sha256"] != expected_population_identity(category):
            raise T2ICompBenchError(f"evaluated image population differs from the frozen dataset: {path}")
    return np.array(answers)


def prompt_score_matrix(answers: np.ndarray) -> np.ndarray:
    """Per-prompt mean of the ten seeds (300 prompts) in prompt order."""

    return answers.reshape(PROMPTS_PER_CATEGORY, IMAGES_PER_PROMPT).mean(axis=1)
