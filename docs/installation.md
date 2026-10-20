# Installation

CCG Diffusion runs from a source checkout: the package reads files of the
repository (`configs/`, `data/`, `dataset/derived/` and the frozen inputs under
`output/`, see [data.md](data.md)), so it is used either inside the runtime
container, which puts `src/` on `PYTHONPATH`, or after an editable install.
Every command in these docs runs from the repository root as
`python -m scripts.<dir>.<name>`.

## Requirements

* Linux x86_64 and an NVIDIA GPU with a driver for CUDA 12 (tested with
  drivers 560 and 580). Generation needs about 9-12 GiB of GPU memory per
  process (physical batches of 4 images). The reference hardware is A100 80GB
  GPUs for T2I-CompBench++ and an RTX 4090 24GB for TAB PP
  ([below](#hardware-and-determinism)).
* About 15 GB of disk for the three models and about 2.7 GB for the four
  T2I-CompBench++ metric weights. The CPU tests need neither.
* [`uv`](https://docs.astral.sh/uv/) for the evaluator and baseline
  environments.
* For the TAB PP judges: an OpenAI API key (GPT-6 Astra) and, for
  Qwen3.8-Flash-Next, an OpenAI-compatible server (see [below](#tab-pp-judges)).

## Option 1: the runtime container

```bash
cp .env.example .env                 # set NVIDIA_VISIBLE_DEVICES, HOST_UID/HOST_GID and, if needed, API keys
docker compose build runtime         # docker/Dockerfile; about 12 minutes after the base image download
docker compose run --rm runtime python -m pytest -q
docker compose run --rm runtime python -m scripts.setup.download_models
```

The image ([`docker/Dockerfile`](../docker/Dockerfile)) holds the software
environment of the experiments: NVIDIA PyTorch 24.12 base (Python 3.12), torch
2.13.0 built for CUDA 12.9, the pins of
[`docker/requirements.txt`](../docker/requirements.txt) (transformers 5.14.1,
diffusers 0.39.0, numpy 2.5.1, spaCy 3.8.14, ...), Hol-CCG (`holccg` 1.0.0 at
commit `0213ca85`) and `en_core_web_sm` 3.8.0; the build ends with `pip check`.
Models are not part of the image.
[`docker-compose.yml`](../docker-compose.yml) mounts the checkout at
`/workspace/${PROJECT_NAME}` and `DATASET_DIR` and `OUTPUT_DIR` over its
`dataset/` and `output/`, runs as a user with your UID/GID and keeps the Hugging
Face cache in the named volume `hf-cache`. Keep the defaults of `DATASET_DIR`
and `OUTPUT_DIR` unless you copy the tracked files of `dataset/` and `output/`
to the new locations ([data.md](data.md#frozen-inputs-under-output)).

`.env` is git-ignored. Besides the GPU selection and the UID/GID it holds the
credentials that the container passes on: `OPENAI_API_KEY` (only for the
GPT-6 Astra judge) and `HF_TOKEN` (optional; the pinned models are public).

**Optional agent overlay.** In-container tooling for coding agents is kept out
of the runtime image. To use it, copy
[`docker-compose.override.example.yml`](../docker-compose.override.example.yml)
to `docker-compose.override.yml` (git-ignored; Docker Compose merges it into
`docker-compose.yml` automatically) and follow the notes in its header; it
builds [`docker/Dockerfile.agent`](../docker/Dockerfile.agent) on top of the
runtime image. Nothing in this repository needs it.

## Option 2: a Python environment

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu129
pip install -e ".[dev]"
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```

[`pyproject.toml`](../pyproject.toml) declares the runtime pins and Hol-CCG at
its pinned commit; torch comes from the CUDA 12.9 index because the PyPI wheel
targets CUDA 13 and needs a driver >= 580. Install in editable mode only: a
non-editable install cannot find the repository files and stops with an error.

SD1.4 T2I-CompBench++ generation checks the software of the reference
environment: Python 3.12.3, torch 2.13.0+cu129, diffusers 0.39.0, transformers
5.14.1, CUDA runtime 12.9 and cuDNN `92000` (`SD14_GENERATOR_RUNTIME_PINS` in
[`backbones/sd14/records.py`](../src/ccg_diffusion/backbones/sd14/records.py)).
The runtime container provides them. With other versions, for example another
Python 3.12 patch release, pass `--no-strict-runtime` to
`python -m scripts.t2i_compbench.run_ccg_diffusion generate --backbone sd14` and
`python -m scripts.t2i_compbench.run_baselines --backbone sd14`: the versions
are then recorded with `pins_enforced: false` instead of enforced.

## Models

```bash
python -m scripts.setup.download_models                 # all three (about 15 GB)
python -m scripts.setup.download_models --only holccg   # one or more of sd14, sana, holccg
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

[`scripts/setup/download_models.py`](../scripts/setup/download_models.py)
downloads, into the Hugging Face cache (`$HF_HOME/hub`, default
`~/.cache/huggingface/hub`), the files that the code loads, at the pinned
revisions:

| Model | Repository @ revision |
|---|---|
| Stable Diffusion v1.4 (U-Net, VAE, CLIP text encoder and tokenizer) | `CompVis/stable-diffusion-v1-4` @ `133a221b8aa7292a167afc5127cb63fb5005638b` |
| Sana 0.6B (transformer, VAE, Gemma text encoder and tokenizer) | `Efficient-Large-Model/Sana_600M_512px_diffusers` @ `83d7a190bfd1fd070570a793d2dab5c7a3231b9d` |
| Hol-CCG parser | `ryosuke-yamaki/hol-ccg-roberta-large` @ `1888cd6343d8df52cf871e1a86924f500a6d7b6b` |

The equivalent with the Hugging Face CLI (whole repositories) is
`hf download <repository> --revision <revision>`. After the download, run with
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`: every command loads models from the
cache only, unless it is given `--allow-download`. The script prints the
commands for the remaining assets (spaCy model, evaluator, baseline
environment) at the end.

## Checks

```bash
python -m pytest -q                       # CPU only, under a minute
python -m ccg_diffusion.methods list      # the four configurations (T2I / TAB x SD1.4 / Sana)
python -m scripts.ccg.show_derivation --prompt "a blue banana and a green vase"
```

Tests that need the cached SD1.4 or Gemma tokenizer, `holccg` or the spaCy model
are skipped when the resource is missing.

## GPUs

List the GPUs with `nvidia-smi -L` and select one by UUID, e.g.
`CUDA_VISIBLE_DEVICES=GPU-xxxxxxxx-... python -m ... --device cuda:0`; do not
rely on GPU indices. Run one backbone per process: SD1.4 generation switches on
deterministic algorithms for the whole process
(`guidance.precision.configure_determinism`), and Sana generation refuses to
run in a process where they are on.

Generated images depend on the GPU type, and SD1.4 generation of
T2I-CompBench++ is not deterministic between runs on the A100
([below](#hardware-and-determinism)).

## Hardware and determinism

The experiments run on the following reference hardware; the times and the
determinism results below refer to it.

| Experiment | Hardware |
|---|---|
| T2I-CompBench++ generation (CCG Diffusion, compared methods, ablations) | NVIDIA A100 80GB (A100 SXM4 80GB), one process per GPU; four GPUs run the four SD1.4 shards in parallel. A process needs about 9-12 GiB of GPU memory ([requirements](#requirements)) |
| Generation-time benchmark (`python -m scripts.paper.compute_cost`) | one NVIDIA A100 80GB without other processes on it |
| TAB PP generation (CCG Diffusion, its ablation cells, compared methods) | one NVIDIA GeForce RTX 4090 24GB |
| GPT-6 Astra judge | OpenAI API access to `gpt-6-astra` (paid requests; `judge_astra run` needs `OPENAI_API_KEY` and a spending ceiling `--budget-usd`) |
| Qwen3.8-Flash-Next judge | two NVIDIA DGX Spark systems connected directly by a QSFP cable, serving the NVFP4 checkpoint with vLLM 0.30.0 (tensor parallel 2 with expert parallelism; [TAB PP judges](#tab-pp-judges)) |
| T2I-CompBench++ evaluator | one NVIDIA GPU, in its own Python 3.10 environment |
| Stages 1-3, plans, TAB PP jobs, tests | CPU |

A 512x512 PNG takes about 0.4 MB of disk, i.e. about 10 GB per method on
T2I-CompBench++. Approximate times on the hardware above:

| Work | GPU | Time or cost |
|---|---|---|
| SD1.4 CCG Diffusion, 24,000 images | A100 | about 12 GPU-hours (four shards: about 3 hours on four GPUs) |
| SD1.4 compared methods, 5 x 24,000 images | A100 | about 13 GPU-hours per method on average |
| SD1.4 attribute-binding ablation condition (`gs0_p1_r1`, `gs1_p0_r1`, `gs1_p1_r0`, `gs0_p0_r0`, `gs0_p0_r1`, `gs0_p1_r0` or `gs1_p0_r0`; color, shape and texture, about 9,000 images) | A100 | about 6 GPU-hours |
| SD1.4 `no_2d` or `no_count` (about 3,000 images) | A100 | about 1.5 GPU-hours |
| Sana Vanilla and CCG Diffusion, 2 x 24,000 images | A100 | about 8 GPU-hours |
| Sana compared methods, 5 x 24,000 images | A100 | about 5.5 GPU-hours per method |
| Sana attribute-binding ablation condition (9,000 images with `--keys`) | A100 | about 2.5 GPU-hours |
| Sana `no_2d` or `no_count` (3,000 images) | A100 | about 0.7 GPU-hours |
| TAB SD1.4 CCG Diffusion, one cell (`c1r1p0m0` or an ablation cell; 185 batches of 4) | RTX 4090 | about 20 minutes |
| TAB SD1.4 Vanilla and VS-TIED + Gold-visual setup (4,725 images) | RTX 4090 | about 1.5 hours |
| TAB SD1.4 Predicated Diffusion runs (1,110 images) | RTX 4090 | about 30 minutes |
| TAB Sana, each run (740 or 1,110 images) | RTX 4090 | about 5 minutes |
| GPT-6 Astra, the ten method-comparison conditions (5 methods x 2 backbones) and the six ablation cells (11,840 trials) | API | about 60 USD |
| Qwen3.8-Flash-Next, the ten method-comparison conditions (7,400 trials, `--workers 10`) | two DGX Spark | about 2 hours |

The SD1.4 Vanilla run (the official reference generator of T2I-CompBench++,
[t2i_compbench.md](t2i_compbench.md#vanilla-on-stable-diffusion-v14)) is not
included in these rows. In total, the T2I-CompBench++ rows take about 175 A100
GPU-hours, followed by about 160 evaluations of a category of 3,000 images with
the official evaluators; the TAB PP rows take about 4 hours on one RTX 4090,
followed by the two judgments. The Qwen3.8-Flash-Next judge re-judges the
trials of a finished GPT-6 Astra judgment, so the GPT-6 Astra judgment of the
ten TAB PP method-comparison conditions (5 methods x 2 backbones, 7,400 paid
trials, about 37 USD) must be complete first; without it neither the
Qwen3.8-Flash-Next judgment nor Cohen's kappa of the two judges can be
computed.

Everything before generation (the benchmark prompts, their frozen CCG
derivations, the guidance targets, the per-prompt plans and the TAB PP jobs) is
the same on every machine and can be checked on a CPU
([t2i_compbench.md](t2i_compbench.md#checking-the-preparation-without-a-gpu),
[tab_pp.md](tab_pp.md#checking-every-cell-on-a-cpu)). Repeated
generation behaves as follows:

| Generation | GPU type | Repeated generation |
|---|---|---|
| Sana 0.6B, T2I-CompBench++ | A100 | bit-identical, also across different A100 boards |
| Sana 0.6B, TAB PP | RTX 4090 | bit-identical |
| SD1.4, TAB PP | RTX 4090 | bit-identical (images, initial and final latents, first loss value) |
| SD1.4, T2I-CompBench++ | A100 | not bit-deterministic between runs: when the same 64 images are generated twice, 36 to 43 of them are identical (8 to 14 of 16 for ablation conditions); the initial latents, guidance targets, batch positions, first loss values and the numbers of loss evaluations and updates are always identical, so the scores reproduce statistically |
| SD1.4, T2I-CompBench++ | RTX 4090 | bit-identical between runs, but not equal to A100 images |

Images are not bit-identical across GPU types: for example, the SD1.4 TAB PP
images generated on an A100 differ from the RTX 4090 images in all 740 PP
trials, first loss values included; the scores of images generated on another
GPU type are a new measurement. Equality holds only for images generated in
their physical batches, which every command preserves (`--batches` and
`--keys` select whole batches). The deterministic algorithms that SD1.4
generation enables for the whole process ([GPUs](#gpus)) are recorded as
`config.SD14_DETERMINISM`. An interrupted Sana `generate` resumes with the
remaining images regrouped into new batches, so its images can differ from an
uninterrupted run.

The frozen inputs remove the parser from this list: the T2I-CompBench++
derivations ([`data/t2i_compbench/`](../data/t2i_compbench/README.md)) were
parsed on an A100 and the TAB PP interpretations
([`data/tab_pp/`](../data/tab_pp/README.md)) on an RTX 4090; a re-parse on
other hardware can flip near-tie supertags ([parsing.md](parsing.md)). The
judges are not deterministic (GPT-6 Astra is a hosted model; the Qwen judge
samples at temperature 1.0), so judging the same images again reproduces the
Yes rates up to judge variation.

## T2I-CompBench++ evaluator

```bash
python -m scripts.setup.setup_t2i_compbench               # clone, venv, weights; idempotent
python -m scripts.setup.setup_t2i_compbench --verify-only # check without installing
python -m scripts.setup.setup_t2i_compbench --smoke \
    --smoke-dir "$PWD/output/analysis/t2i_compbench_smoke"    # score the bundled samples (GPU)
```

[`scripts/setup/setup_t2i_compbench.py`](../scripts/setup/setup_t2i_compbench.py)
builds the official evaluation stack in isolation under `external/t2i_compbench/`
(git-ignored); the main environment is never touched:

1. clones `Karine-Huang/T2I-CompBench` at commit
   `1b7094991a57f3c22abdd4f6e8ba6c1a15517073` into
   `external/t2i_compbench/T2I-CompBench` (or checks an existing clone) and
   treats it as read-only;
2. creates a Python 3.10.20 venv `external/t2i_compbench/.venv` with `uv` and
   installs exactly the tracked freeze
   [`docker/t2i_compbench/venv_freeze.txt`](../docker/t2i_compbench/venv_freeze.txt)
   (torch 2.0.1+cu118, transformers 4.30.2, diffusers 0.15.0, spaCy 3.5.3;
   detectron2 0.6 is built from its pinned commit as a CPU extension). The
   freeze is derived from the official requirements as described in
   [`docker/t2i_compbench/requirements.eval.txt`](../docker/t2i_compbench/requirements.eval.txt);
3. downloads the four metric weights (UniDet RS200 and R50, MiDaS, BLIP-VQA)
   into `external/t2i_compbench/weights/` and checks their byte counts and
   sha256. The UniDet R50 weight comes from a third-party mirror, because the
   official download link refuses anonymous access;
4. links the UniDet and MiDaS weights into the clone
   (`UniDet_eval/experts/expert_weights/`) and the BLIP weight into the torch
   hub cache of the evaluator venv (`torch.hub.get_dir()/checkpoints`, by
   default `~/.cache/torch/hub/checkpoints`), where the official BLIP code
   looks for it;
5. requires the installed venv to equal the tracked freeze and checks the
   tracked source pin
   [`output/analysis/t2i_compbench/t2icb_source_pin.json`](../output/analysis/t2i_compbench/t2icb_source_pin.json)
   (clone, prompt files, venv, weights and every departure from the official
   requirements, under `deviations`). The pin is left unchanged; any difference
   apart from its retrieval time stops the setup, because the sha256 of the pin
   is part of the T2I-CompBench++ dataset contract.

The BLIP link lies outside the checkout. In the runtime container the torch hub
cache is not on a volume, so run the setup and the evaluation in the same
container, or run the setup again in a new one (it skips every step whose result
is already present). Call the venv's Python only as
`env -u PYTHONPATH -u LD_LIBRARY_PATH external/t2i_compbench/.venv/bin/python`:
the container's `PYTHONPATH` and `LD_LIBRARY_PATH` point at the main
environment's torch. `scripts/t2i_compbench/evaluate.py` does this itself.

## Baseline adapter environment

```bash
bash scripts/setup/setup_baseline_env.sh
```

[`scripts/setup/setup_baseline_env.sh`](../scripts/setup/setup_baseline_env.sh)
builds the Python 3.11 environment `external/baseline_env/.venv` from
[`docker/requirements-baselines.lock.txt`](../docker/requirements-baselines.lock.txt)
(transformers 4.30.2, Stanza 1.4.2, spaCy 3.6.1 with `en_core_web_trf` 3.6.1),
downloads the Stanza resources into `external/stanza_resources_1_4_2` and
compares the parser versions, processors and model names with the reference
ones. It is used only for the blind preprocessing of Composable Diffusion,
StructureDiffusion, Attend-and-Excite and SynGen on T2I-CompBench++
([t2i_compbench.md](t2i_compbench.md)). The pinned parser installation digests
(`RECORDED_PARSER_ENVIRONMENT` in
[`baselines/preprocessing.py`](../src/ccg_diffusion/baselines/preprocessing.py))
depend on the absolute venv path, so inputs built in a new venv need
`--no-strict-environment` (versions, processors and model names are still
compared).

## TAB PP judges

**GPT-6 Astra.** `python -m scripts.tab_pp.judge_astra run` sends requests to
the OpenAI API. It needs `OPENAI_API_KEY` (in `.env` for the container, or in
the environment) and `--budget-usd`, a hard spending ceiling of the invocation:
the worst-case cost of every in-flight request is reserved before it is sent,
and an unresolved request is never resent automatically. Every other
subcommand is offline.

**Qwen3.8-Flash-Next.** The judge is `nvidia/Qwen3.8-Flash-Next-NVFP4` at
revision `fc694b54fb0174e0913e6adf86691ef85a4ead47`, served by vLLM 0.30.0
(`vllm/vllm-openai:v0.30.0`) through an OpenAI-compatible endpoint. The
reference setup is two NVIDIA DGX Spark systems connected by a direct QSFP link
(tensor parallel 2 with expert parallelism).
[`scripts/setup/qwen38_flash_next/serve.sh`](../scripts/setup/qwen38_flash_next/serve.sh)
starts that two-node server from the head node; it needs Docker on both nodes,
passwordless SSH from the head to the worker and the checkpoint in the Hugging
Face cache of both nodes (the containers run offline). The node addresses and
network devices of the direct link come from the environment:

```bash
hf download nvidia/Qwen3.8-Flash-Next-NVFP4 --revision fc694b54fb0174e0913e6adf86691ef85a4ead47   # on both nodes
HEAD_IP=<head-ip> WORKER_IP=<worker-ip> IFACE=<interface> IB_HCA=<hca> \
    scripts/setup/qwen38_flash_next/serve.sh start     # also: stop, status
```

The server listens on port 8888 with the served model name
`qwen3.8-flash-next`. Before the start, `serve.sh` runs `~/qwen38fn/evict_ckpt.py`
on both nodes, which drops the checkpoint's pages from the page cache; it copies
[`evict_ckpt.py`](../scripts/setup/qwen38_flash_next/evict_ckpt.py) to the
worker, so place a copy at the same path on the head node. Any
OpenAI-compatible endpoint serving the same checkpoint can be used:
`python -m scripts.tab_pp.judge_qwen38 run --base-url http://<host>:8888`
([evaluation.md](evaluation.md)).
