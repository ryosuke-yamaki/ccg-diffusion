"""Per-image generation jobs and the on-disk image layout shared by every backbone.

The path rule of ``image_path`` is part of the recorded layout: exporters and
the paper evidence resolve images through it. The module does not import torch.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from ccg_diffusion.targets.pd_indices import PDIndices

MAX_IMAGE_DIR_COMPONENT_LENGTH = 220


@dataclass(frozen=True)
class Job:
    """One image to generate: prompt, seed and guidance targets (E, C, K, P) as CLIP token indices."""

    experiment: str
    prompt_id: str
    prompt: str
    seed: int
    attention_exist_indices: list[int]
    attention_corr_indices: list[list[int]]
    attention_leak_indices: list[list[int]]
    attention_possession_indices: list[list[int]]
    source: str

    def pd_indices(self) -> PDIndices:
        return PDIndices(
            exist=self.attention_exist_indices,
            corr=self.attention_corr_indices,
            leak=self.attention_leak_indices,
            possession=self.attention_possession_indices,
        )


def slugify(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    return text or "prompt"


def image_path(out_dir: Path, method: str, job: Job) -> Path:
    """``<out>/<method>/<experiment>/<prompt_id>_<slug>/seed_<seed>.png``.

    A directory component longer than 220 bytes keeps a truncated slug plus
    the first 16 hex digits of ``sha256(prompt)``.
    """

    slug = slugify(job.prompt)
    component = f"{job.prompt_id}_{slug}"
    if len(component.encode("utf-8")) > MAX_IMAGE_DIR_COMPONENT_LENGTH:
        digest = hashlib.sha256(job.prompt.encode("utf-8")).hexdigest()[:16]
        suffix = f"_{digest}"
        available = (
            MAX_IMAGE_DIR_COMPONENT_LENGTH
            - len(job.prompt_id.encode("utf-8"))
            - len(suffix)
            - 1
        )
        if available <= 0:
            raise ValueError("prompt_id is too long for the image directory contract")
        slug = slug[:available].rstrip("_") + suffix
        component = f"{job.prompt_id}_{slug}"
    return out_dir / method / job.experiment / component / f"seed_{job.seed:05d}.png"


def metadata_path(out_dir: Path, method: str, experiment: str) -> Path:
    return out_dir / method / experiment / "metadata.jsonl"
