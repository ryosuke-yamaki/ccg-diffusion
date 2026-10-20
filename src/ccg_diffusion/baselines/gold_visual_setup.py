"""VS-TIED + Gold-visual setup and the job layouts of the TAB comparison methods (backbone independent).

VS-TIED + Gold-visual setup conditions on the ambiguous prompt followed by the
gold visual setup sentence of the intended interpretation (an upper bound that
uses the dataset's answer). TAB's strings have no final punctuation; a period
ends each sentence, as in the figures of the TAB paper.

Job keys (recorded): Vanilla ``vanilla_<tab_id>_None``, Gold-visual setup
``gold_<tab_id>_<setup>``, Predicated Diffusion ``pd_spacy_<tab_id>_None`` and
Gold-visual setup + Predicated Diffusion ``pd_spacy_gold_<tab_id>_<setup>``;
image IDs append ``_s<seed>`` for seeds 0-4. A Vanilla or Predicated Diffusion
image is judged against both setups (``ambiguous_both_setups``).
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

PD_ARMS = ("pd_spacy", "pd_spacy_gold")
SEEDS = tuple(range(5))


def terminate_sentence(text: str) -> str:
    """``text`` stripped, with a final period unless it already ends a sentence."""
    text = text.strip()
    if not text:
        raise ValueError("TAB paraphrasing inputs must be nonempty")
    return text if text.endswith((".", "!", "?")) else text + "."


def gold_visual_setup_prompt(ambiguous_prompt: str, visual_setup: str) -> str:
    """The ambiguous prompt followed by the gold visual setup, each as a terminated sentence."""
    return f"{terminate_sentence(ambiguous_prompt)} {terminate_sentence(visual_setup)}"


def stock_jobs(specs: Sequence[Mapping[str, Any]], arms: Sequence[str]) -> tuple[list[dict], list[dict]]:
    """Plans (``conditioning``, ``key``, ``arm``) and jobs of the Vanilla and Gold-visual setup arms.

    Per item (dataset order): the Vanilla key, then one Gold key per setup.
    """
    plans, jobs = [], []
    for spec in specs:
        tab_id = spec["tab_id"]
        variants = [("vanilla", None, spec["ambiguous_prompt"].strip())]
        variants += [("gold", t["setup_index"],
                      gold_visual_setup_prompt(spec["ambiguous_prompt"].strip(), t["visual_setup"]))
                     for t in spec["targets"]]
        for arm, setup, conditioning in variants:
            if arm not in arms:
                continue
            key = f"{arm}_{tab_id}_{setup}"
            plans.append({"conditioning": conditioning, "key": key, "arm": arm})
            for seed in SEEDS:
                jobs.append({"key": key, "arm": arm, "condition": arm, "tab_id": tab_id,
                             "tab_category": spec["tab_category"], "seed": seed, "reading": None,
                             "tab_requires_reading": False, "target_setup_index": setup,
                             "tab_evaluation_mode": "ambiguous_both_setups" if arm == "vanilla" else "targeted_setup",
                             "evaluation_setup_indices": [0, 1] if arm == "vanilla" else [setup],
                             "image_id": f"{key}_s{seed}"})
    return plans, jobs


def pd_baseline_jobs(specs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Jobs of the Predicated Diffusion arms: ``pd_spacy`` like Vanilla, ``pd_spacy_gold`` like Gold (arm-major)."""
    jobs = []
    for arm in PD_ARMS:
        for spec in specs:
            ambiguous = spec["ambiguous_prompt"].strip()
            setups = [None] if arm == "pd_spacy" else [0, 1]
            for setup in setups:
                prompt = ambiguous if setup is None else gold_visual_setup_prompt(
                    ambiguous, spec["targets"][setup]["visual_setup"])
                key = f"{arm}_{spec['tab_id']}_{setup}"
                for seed in SEEDS:
                    jobs.append({"key": key, "arm": arm, "condition": arm, "tab_id": spec["tab_id"],
                                 "tab_category": spec["tab_category"], "seed": seed, "reading": None,
                                 "tab_requires_reading": False, "target_setup_index": setup,
                                 "tab_evaluation_mode": "ambiguous_both_setups" if setup is None else "targeted_setup",
                                 "evaluation_setup_indices": [0, 1] if setup is None else [setup],
                                 "image_id": f"{key}_s{seed}", "prompt": prompt})
    return jobs


def pd_baseline_batches(jobs: Sequence[Mapping[str, Any]], backbone: str) -> list[list[Mapping[str, Any]]]:
    """SD1.4: consecutive physical batches of 4 per arm; Sana: per prompt, seeds 0-3 then seed 4."""
    if backbone == "sd14":
        result = []
        for arm in PD_ARMS:
            group = [j for j in jobs if j["arm"] == arm]
            result += [group[i:i + 4] for i in range(0, len(group), 4)]
        return result
    by_key: dict[str, list[Mapping[str, Any]]] = {}
    for job in jobs:
        by_key.setdefault(job["key"], []).append(job)
    return [group[i:i + 4] for group in by_key.values() for i in range(0, len(group), 4)]
