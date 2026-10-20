"""Statistics of the paper: permutation tests, Holm correction, significance marks, Cohen's kappa, bootstraps.

* ``paired_permutation_test`` (two-sided sign-flip test on the mean of paired
  differences) and ``label_permutation_test`` (two-sided test between two
  different prompt sets) produce every p-value of the method comparisons and
  ablations on T2I-CompBench++ and TAB PP. Both draw from an explicit
  ``numpy`` generator, so the call order that fixes every p-value is visible at
  the call site.
* ``holm`` and ``significance_mark`` turn p-values into significance marks.
* ``cohen_kappa`` measures the agreement of the two TAB PP judges (GPT-6 Astra
  and Qwen3.8-Flash-Next).
* ``prompt_means``, ``paired_prompt_bootstrap`` and ``prompt_mean_bootstrap``
  are the prompt-unit bootstraps of the ablation summaries.
* ``load_results_provenance`` reads the map from each reported result to the
  run that produced it (``RESULTS_PROVENANCE``), a record of those runs that is
  not distributed with this repository.

These numerics produced the paper numbers; do not reorder their expressions.
"""
from __future__ import annotations

import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ccg_diffusion.provenance import repository_root

N_PERMUTATIONS = 200_000
N_BOOTSTRAP = 20_000
SIGNIFICANCE_SEED = 20260927
PERMUTATION_CHUNK = 10_000
TIE_TOLERANCE = 1e-12
IMAGES_PER_PROMPT = 10
RESULTS_PROVENANCE = "docs/paper/results_provenance.json"


def paired_permutation_test(
    d: np.ndarray, rng: np.random.Generator, *, n_perm: int = N_PERMUTATIONS, n_boot: int = N_BOOTSTRAP,
) -> dict[str, Any]:
    """Two-sided sign-flip permutation test on the mean of the paired differences ``d``.

    Returns ``{"delta", "ci95", "p", "n", "wtl"}`` (key order is part of the
    serialized outputs); ``ci95`` is the percentile bootstrap over units and
    ``wtl`` counts wins, ties and losses.
    """

    n, observed = d.size, float(d.mean())
    extreme = 0
    for start in range(0, n_perm, PERMUTATION_CHUNK):
        signs = rng.choice(np.array([-1.0, 1.0]), size=(min(PERMUTATION_CHUNK, n_perm - start), n))
        extreme += int(np.sum(np.abs(signs @ d / n) >= abs(observed) - TIE_TOLERANCE))
    boot = d[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return {"delta": observed, "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "p": (extreme + 1) / (n_perm + 1), "n": n,
            "wtl": [int(np.sum(d > TIE_TOLERANCE)), int(np.sum(np.abs(d) <= TIE_TOLERANCE)),
                    int(np.sum(d < -TIE_TOLERANCE))]}


def label_permutation_test(
    a: np.ndarray, b: np.ndarray, rng: np.random.Generator, *,
    n_perm: int = N_PERMUTATIONS, n_boot: int = N_BOOTSTRAP,
) -> dict[str, Any]:
    """Two-sided label permutation test of ``mean(a) - mean(b)`` for two different unit sets."""

    pooled, n = np.concatenate([a, b]), a.size
    observed = float(a.mean() - b.mean())
    extreme = 0
    for start in range(0, n_perm, PERMUTATION_CHUNK):
        m = min(PERMUTATION_CHUNK, n_perm - start)
        perm = rng.permuted(np.broadcast_to(pooled, (m, pooled.size)), axis=1)
        stat = perm[:, :n].mean(axis=1) - perm[:, n:].mean(axis=1)
        extreme += int(np.sum(np.abs(stat) >= abs(observed) - TIE_TOLERANCE))
    boot = (a[rng.integers(0, a.size, size=(n_boot, a.size))].mean(axis=1)
            - b[rng.integers(0, b.size, size=(n_boot, b.size))].mean(axis=1))
    return {"delta": observed, "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "p": (extreme + 1) / (n_perm + 1), "n": [a.size, b.size]}


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm-adjusted p-values in input order (stable sort: ties keep their input order)."""

    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    running, adjusted = 0.0, [0.0] * len(p_values)
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def significance_mark(p: float) -> str:
    """``***`` p < .001, ``**`` p < .01, ``*`` p < .05, else ``n.s.``."""

    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def cohen_kappa(a: Sequence[bool], b: Sequence[bool]) -> tuple[float, float]:
    """Observed agreement and Cohen's kappa of two binary judgments of the same trials."""

    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return po, (po - pe) / (1 - pe)


def bootstrap_quantiles(values: Sequence[float]) -> list[float]:
    """2.5% and 97.5% order statistics of bootstrap estimates."""

    values = sorted(values)
    return [values[int(0.025 * len(values))], values[int(0.975 * len(values)) - 1]]


def prompt_means(answers: Sequence[float], *, expected_prompts: int) -> list[float]:
    """Per-prompt means of consecutive blocks of ten image scores."""

    expected_answers = expected_prompts * IMAGES_PER_PROMPT
    if len(answers) != expected_answers:
        raise ValueError(
            f"expected {expected_answers} answers for {expected_prompts} prompts, "
            f"found {len(answers)}"
        )
    return [
        statistics.fmean(answers[start:start + IMAGES_PER_PROMPT])
        for start in range(0, len(answers), IMAGES_PER_PROMPT)
    ]


def percentile(sorted_values: Sequence[float], probability: float) -> float:
    """Linear-interpolation percentile of already sorted values."""

    if not sorted_values:
        raise ValueError("percentile needs at least one value")
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(
        sorted_values[lower] * (1.0 - fraction)
        + sorted_values[upper] * fraction
    )


def paired_prompt_bootstrap(deltas: Sequence[float], *, iterations: int, seed: int) -> dict[str, Any]:
    """Bootstrap the mean paired difference with the prompt as the sample unit."""

    if not deltas:
        raise ValueError("paired bootstrap needs at least one prompt")
    if iterations < 1:
        raise ValueError("bootstrap iterations must be positive")
    generator = random.Random(seed)
    n_prompts = len(deltas)
    estimates = [
        statistics.fmean(deltas[generator.randrange(n_prompts)] for _ in range(n_prompts))
        for _ in range(iterations)
    ]
    estimates.sort()
    tolerance = 1e-12
    return {
        "unit": "prompt",
        "iterations": iterations,
        "seed": seed,
        "mean_delta": statistics.fmean(deltas),
        "ci95": [percentile(estimates, 0.025), percentile(estimates, 0.975)],
        "positive_prompt_count": sum(value > tolerance for value in deltas),
        "tied_prompt_count": sum(abs(value) <= tolerance for value in deltas),
        "negative_prompt_count": sum(value < -tolerance for value in deltas),
        "positive_prompt_fraction": sum(value > tolerance for value in deltas) / n_prompts,
    }


def prompt_mean_bootstrap(values: Sequence[float], *, iterations: int, seed: int) -> dict[str, Any]:
    """Bootstrap one arm's score with prompt, not image, as the sample unit."""

    if not values:
        raise ValueError("prompt bootstrap needs at least one prompt")
    if iterations < 1:
        raise ValueError("bootstrap iterations must be positive")
    generator = random.Random(seed)
    n_prompts = len(values)
    estimates = [
        statistics.fmean(values[generator.randrange(n_prompts)] for _ in range(n_prompts))
        for _ in range(iterations)
    ]
    estimates.sort()
    return {
        "unit": "prompt",
        "iterations": iterations,
        "seed": seed,
        "mean": statistics.fmean(values),
        "ci95": [percentile(estimates, 0.025), percentile(estimates, 0.975)],
    }


def load_results_provenance(path: Path | None = None) -> dict[str, Any]:
    """The map from each reported result to its run (``RESULTS_PROVENANCE``, not distributed)."""

    source = path if path is not None else repository_root() / RESULTS_PROVENANCE
    return json.loads(Path(source).read_text(encoding="utf-8"))
