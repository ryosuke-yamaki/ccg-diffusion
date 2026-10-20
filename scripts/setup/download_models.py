#!/usr/bin/env python3
"""Download the pinned models of CCG Diffusion into the Hugging Face cache.

Each model is fetched at the revision used by the paper runs, so that parsing,
Stage 3 and generation can afterwards run offline (``HF_HUB_OFFLINE=1``):

* ``sd14``: Stable Diffusion v1.4, ``CompVis/stable-diffusion-v1-4`` (U-Net,
  VAE, CLIP text encoder and tokenizer, scheduler, feature extractor);
* ``sana``: Sana 0.6B, ``Efficient-Large-Model/Sana_600M_512px_diffusers``
  (transformer, VAE, Gemma text encoder and tokenizer, scheduler);
* ``holccg``: the Hol-CCG parser checkpoint ``ryosuke-yamaki/hol-ccg-roberta-large``.

The cache is ``$HF_HOME/hub`` (default ``~/.cache/huggingface/hub``). Run from
the repository root, without ``HF_HUB_OFFLINE``::

    python -m scripts.setup.download_models                 # all three models
    python -m scripts.setup.download_models --only holccg   # one model

The spaCy model, the T2I-CompBench++ evaluator weights and the baseline
adapter environment are installed by other commands, printed at the end.
"""
from __future__ import annotations

import argparse
import os
from typing import Sequence

from ccg_diffusion.config import HOLCCG_CHECKPOINT, HOLCCG_REVISION, SANA_GENERATION, SD14_GENERATION

# name -> (repository, revision, files to fetch; None fetches the whole repository)
MODELS: dict[str, tuple[str, str, list[str] | None]] = {
    "sd14": (SD14_GENERATION.model_id, SD14_GENERATION.model_revision, [
        "model_index.json",
        "feature_extractor/preprocessor_config.json",
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "text_encoder/model.safetensors",
        "tokenizer/merges.txt",
        "tokenizer/special_tokens_map.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/vocab.json",
        "unet/config.json",
        "unet/diffusion_pytorch_model.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    ]),
    "sana": (SANA_GENERATION.model_id, SANA_GENERATION.model_revision, [
        "model_index.json",
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "text_encoder/model.safetensors.index.json",
        "text_encoder/model-00001-of-00002.safetensors",
        "text_encoder/model-00002-of-00002.safetensors",
        "tokenizer/special_tokens_map.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer.model",
        "tokenizer/tokenizer_config.json",
        "transformer/config.json",
        "transformer/diffusion_pytorch_model.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    ]),
    "holccg": (HOLCCG_CHECKPOINT, HOLCCG_REVISION, None),
}

SPACY_MODEL_WHEEL = ("https://github.com/explosion/spacy-models/releases/download/"
                     "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl")
NEXT_STEPS = f"""\
Other pinned assets:
  spaCy model (Stage 3 possession lemmas, Predicated Diffusion baseline):
    pip install {SPACY_MODEL_WHEEL}
  T2I-CompBench++ evaluator (official clone, Python 3.10 venv and metric weights):
    python -m scripts.setup.setup_t2i_compbench
  T2I baseline preprocessing (Python 3.11 adapter venv, Stanza and en_core_web_trf):
    bash scripts/setup/setup_baseline_env.sh
Then set HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 for every run."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=sorted(MODELS), default=list(MODELS),
                        help="download only these models (default: all)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if os.environ.get("HF_HUB_OFFLINE", "0") not in ("", "0"):
        raise SystemExit("HF_HUB_OFFLINE is set; unset it to download models")
    from huggingface_hub import snapshot_download

    for name in args.only:
        repository, revision, files = MODELS[name]
        print(f"{name}: {repository}@{revision}", flush=True)
        path = snapshot_download(repository, revision=revision, allow_patterns=files)
        print(f"  -> {path}", flush=True)
    print(NEXT_STEPS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
