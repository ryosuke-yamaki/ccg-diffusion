"""Port of the official T2I-CompBench++ reference generator (GORS_finetune/inference_eval.py).

The generation loop is a verbatim port of the official reference implementation
``GORS_finetune/inference_eval.py`` (T2I-CompBench pin
``1b7094991a57f3c22abdd4f6e8ba6c1a15517073``), with the model id as a
command-line argument.

Everything the downstream official evaluators depend on is kept byte-identical
in behaviour: prompt reading (``splitlines`` -> ``strip("\\n").split("\\t")[0]``),
per-prompt seed schedule (42, then 42+n+1 for the n-th of ``n_iter`` images),
the pipeline settings of the official script, and the
``{prompt}_{base_count:06}.png`` global running index under ``<outdir>/samples``.

Deviations from the official script, and why:

1. ``lora_diffusion`` import and the ``patch_pipe`` branch are removed. They only
   apply to GORS-finetuned LoRA checkpoints; this script generates the
   pretrained baseline, so the branch is always inactive and importing
   ``lora_diffusion`` would pull an unnecessary dependency into the eval venv.
2. Grid-image saving (``image_grid`` / ``*-grid-*.png``) is removed. Grids are
   not read by any official evaluator and would add 300 large PNGs per category.
3. Added beyond the official script: a run manifest, final count/contiguity
   assertions, and the resume guard described below. These are provenance and
   safety only; they do not touch the generation loop.

Resume guard (no official counterpart): ``base_count`` starts from
``len(os.listdir(sample_path))``, so restarting an interrupted run against a
non-empty output directory would silently re-generate from prompt 0 while
numbering from the old count, breaking the ``prompt i, image j -> i*n_iter+j``
mapping the evaluators rely on. On startup the existing samples are therefore
checked (count divisible by ``n_iter``, indices contiguous from 0, each file
name matching the prompt its index implies); the run then resumes at the next
prompt boundary, or stops with an explicit error if the check fails.

Runs in the evaluator venv of ``scripts/setup/setup_t2i_compbench.py``, not in
the main CCG Diffusion environment:

    env -u LD_LIBRARY_PATH -u PYTHONPATH \\
      external/t2i_compbench/.venv/bin/python \\
      scripts/t2i_compbench/official_reference/generate.py \\
      --from-file external/t2i_compbench/T2I-CompBench/examples/dataset/color_val.txt \\
      --outdir output/generation/RUN_ID/color
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import inspect
import json
import os
import re
import time
from typing import Any

import torch
from diffusers import DDIMScheduler, PNDMScheduler, StableDiffusionPipeline
from PIL import Image, ImageStat
from tqdm import tqdm, trange

SAMPLE_RE = re.compile(r"^(?P<prompt>.*)_(?P<index>\d{6})\.png$", re.DOTALL)
BLACK_MEAN_THRESHOLD = 5.0
BIT_REPRODUCIBILITY_NOTE = (
    "Images are not bit-reproducible on this stack: three same-seed generations of one prompt inside a "
    "single process gave three different images (max per-pixel diff 226, ~96% of pixels). torch 2.0.1's "
    "AttnProcessor2_0 (SDPA) is not bit-deterministic and the 30-step denoising loop amplifies it; the "
    "initial latent does follow the seed. Compare against published values on the per-category 3,000-image "
    "aggregate only, never per image. Determinism is deliberately not forced, to keep the official runtime "
    "behaviour."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model-id",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        help="diffusers model id",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        required=True,
        help="prompt txt, one prompt per line (official T2I-CompBench <category>_val.txt)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="run directory; images go to <outdir>/samples",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="num_inference_steps (default: the official script's)",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="ddim",
        choices=["ddim", "pndm"],
        help="scheduler (default: the official script's)",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=10,
        help="images per prompt (official: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="prompts per pipeline call (official: 1)",
    )
    return parser.parse_args()


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def detect_line_endings(path: str) -> str:
    """Report the line-ending form of a prompt file.

    The official T2I-CompBench prompt files are not uniform (6 of 8 categories
    are CRLF), which is invisible to the text-mode reader used here but breaks
    any reader that splits on "\\n" in binary mode.
    """
    with open(path, "rb") as handle:
        raw = handle.read()
    crlf = raw.count(b"\r\n")
    lf = raw.count(b"\n") - crlf
    cr = raw.count(b"\r") - crlf
    found = [name for name, count in (("crlf", crlf), ("lf", lf), ("cr", cr)) if count]
    if not found:
        return "none"
    if len(found) > 1:
        return "mixed(" + ",".join(f"{name}={count}" for name, count in
                                   (("crlf", crlf), ("lf", lf), ("cr", cr)) if count) + ")"
    return found[0]


def load_prompts(from_file: str, batch_size: int) -> list[list[str]]:
    """Read prompts exactly as the official script does."""
    with open(from_file, "r") as f:
        data = f.read().splitlines()
        data = [d.strip("\n").split("\t")[0] for d in data]
        data = [batch_size * [d] for d in data]
    return data


def resume_start_index(sample_path: str, data: list[list[str]], n_iter: int) -> int:
    """Validate already-present samples and return the prompt index to resume from.

    Raises SystemExit with an explicit message when the existing samples cannot
    be a prefix of this run's output.
    """
    entries = sorted(os.listdir(sample_path))
    if not entries:
        return 0

    non_png = [name for name in entries if not name.endswith(".png")]
    if non_png:
        raise SystemExit(
            f"resume check failed: {sample_path} holds {len(non_png)} non-png entries "
            f"(e.g. {non_png[0]!r}); base_count would be wrong. Move them out or use a fresh --outdir."
        )

    count = len(entries)
    if count % n_iter != 0:
        raise SystemExit(
            f"resume check failed: {count} existing samples is not a multiple of n_iter={n_iter}. "
            f"The interrupted prompt is half-written; delete its {count % n_iter} image(s) to resume "
            f"at a prompt boundary."
        )

    by_index: dict[int, str] = {}
    for name in entries:
        match = SAMPLE_RE.match(name)
        if match is None:
            raise SystemExit(f"resume check failed: {name!r} does not match '<prompt>_<000000>.png'.")
        index = int(match.group("index"))
        if index in by_index:
            raise SystemExit(f"resume check failed: duplicate sample index {index:06d}.")
        by_index[index] = match.group("prompt")

    missing = sorted(set(range(count)) - set(by_index))
    if missing:
        raise SystemExit(
            f"resume check failed: sample indices are not contiguous 0..{count - 1}; "
            f"first missing index is {missing[0]:06d}."
        )

    start = count // n_iter
    if start > len(data):
        raise SystemExit(
            f"resume check failed: {count} existing samples exceed {len(data)} prompts x {n_iter} images."
        )
    for index, prompt in by_index.items():
        expected = data[index // n_iter][0]
        if prompt != expected:
            raise SystemExit(
                f"resume check failed: sample {index:06d} was written for prompt {prompt!r}, but this "
                f"prompt file has {expected!r} at that position. The existing samples come from a "
                f"different prompt file or ordering."
            )

    print(f"resuming: {count} samples present, continuing from prompt {start}/{len(data)}", flush=True)
    return start


def count_black_images(sample_path: str, threshold: float) -> tuple[int, list[str]]:
    """Count near-black images."""
    black: list[str] = []
    for name in sorted(os.listdir(sample_path)):
        if not name.endswith(".png"):
            continue
        with Image.open(os.path.join(sample_path, name)) as img:
            mean = ImageStat.Stat(img.convert("L")).mean[0]
        if mean < threshold:
            black.append(name)
    return len(black), black[:50]


def main() -> None:
    opt = parse_args()
    started_at = time.time()

    data = load_prompts(opt.from_file, opt.batch_size)
    print(f"reading prompts from {opt.from_file}", flush=True)
    print(f"{len(data)} prompts x {opt.n_iter} images = {len(data) * opt.n_iter} expected samples", flush=True)

    scheduler_cls = DDIMScheduler if opt.scheduler == "ddim" else PNDMScheduler
    scheduler = scheduler_cls.from_pretrained(opt.model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(opt.model_id, scheduler=scheduler, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    guidance_scale = inspect.signature(pipe.__call__).parameters["guidance_scale"].default

    outpath = opt.outdir
    os.makedirs(outpath, exist_ok=True)
    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)

    start_prompt = resume_start_index(sample_path, data, opt.n_iter)
    base_count = len(os.listdir(sample_path))

    with torch.no_grad():
        for prompts in tqdm(data[start_prompt:], desc="data"):
            generator = torch.Generator(device="cuda").manual_seed(42)
            for n in trange(opt.n_iter, desc="Sampling", leave=False):
                image = pipe(prompts, num_inference_steps=opt.steps, generator=generator).images
                generator = torch.Generator(device="cuda").manual_seed(42 + n + 1)
                for i in range(len(image)):
                    image[i].save(os.path.join(sample_path, f"{prompts[0]}_{base_count:06}.png"))
                    base_count += 1

    finished_at = time.time()

    expected = len(data) * opt.n_iter
    written = sorted(name for name in os.listdir(sample_path) if name.endswith(".png"))
    assert len(written) == expected, f"expected {expected} samples, found {len(written)}"
    indices = sorted(int(SAMPLE_RE.match(name).group("index")) for name in written)
    assert indices == list(range(expected)), "sample indices are not a contiguous 0..N-1 run"

    black_count, black_examples = count_black_images(sample_path, BLACK_MEAN_THRESHOLD)

    manifest: dict[str, Any] = {
        "script": "scripts/t2i_compbench/official_reference/generate.py",
        "reference_implementation": "T2I-CompBench GORS_finetune/inference_eval.py",
        "reference_commit": "1b7094991a57f3c22abdd4f6e8ba6c1a15517073",
        "model_id": opt.model_id,
        "torch_version": torch.__version__,
        "diffusers_version": __import__("diffusers").__version__,
        "transformers_version": __import__("transformers").__version__,
        "gpu_name": torch.cuda.get_device_name(0),
        "scheduler_arg": opt.scheduler,
        "scheduler_class": type(pipe.scheduler).__name__,
        "scheduler_config": dict(pipe.scheduler.config),
        "num_inference_steps": opt.steps,
        "guidance_scale": guidance_scale,
        "n_iter": opt.n_iter,
        "batch_size": opt.batch_size,
        "seed_schedule": "torch.Generator('cuda').manual_seed(42) per prompt, then manual_seed(42+n+1) after "
                         "each image (effective seeds 42..51 for n_iter=10)",
        "safety_checker_enabled": pipe.safety_checker is not None,
        "prompt_file": os.path.abspath(opt.from_file),
        "prompt_file_sha256": sha256_file(opt.from_file),
        "prompt_file_line_endings": detect_line_endings(opt.from_file),
        "prompt_count": len(data),
        "image_count": len(written),
        "black_image_count": black_count,
        "black_image_examples": black_examples,
        "black_mean_threshold": BLACK_MEAN_THRESHOLD,
        "bit_reproducibility": BIT_REPRODUCIBILITY_NOTE,
        "resumed_from_prompt_index": start_prompt,
        "started_at": dt.datetime.fromtimestamp(started_at).astimezone().isoformat(),
        "finished_at": dt.datetime.fromtimestamp(finished_at).astimezone().isoformat(),
        "elapsed_seconds": round(finished_at - started_at, 1),
    }
    manifest_path = os.path.join(outpath, "generation_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, sort_keys=False)
        f.write("\n")

    print(
        f"done: {len(written)} samples in {sample_path}, black={black_count}, "
        f"elapsed={manifest['elapsed_seconds']}s, manifest={manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
