# CCG Diffusion

**CCG Diffusion** is a training-free method that controls text-to-image
diffusion models with the Combinatory Categorial Grammar (CCG) derivation of the
prompt. The Hol-CCG parser parses the prompt into a derivation, and the
derivation and its marked dependencies drive two pathways into a frozen image
generator: (a) the prompt given to the text encoder is rewritten, and (b)
attention guidance enforces the Predicated Diffusion propositions whose
guidance targets are read off the derivation. The same two pathways serve two
tasks:

* **Compositional text-to-image generation** (T2I-CompBench++): the Selective
  Guidance Boost, the Positional Prefix, the Region Assignment Loss, the Spatial
  Relation Loss and the Count Loss target attribute binding, 2D spatial
  relations and object counts.
* **PP-attachment ambiguity resolution** (the PP-attachment category of TAB):
  Supertag-Constrained Parsing fixes the supertag of the preposition and yields
  the derivation of the intended interpretation, from which the Disambiguating
  Paraphrase and the Holder-Object Region Assignment are built.

The code runs both tasks on two backbones, Stable Diffusion v1.4 and Sana 0.6B.
The repository contains the implementation, the frozen inputs of the benchmark
runs (CCG derivations, TAB PP interpretations and jobs, benchmark populations)
and the command lines that prepare, generate and evaluate CCG Diffusion, its
component ablations and the compared methods on T2I-CompBench++ and TAB PP.

![Overview of CCG Diffusion: a CCG parser parses the prompt into a derivation D (under the specified NP or VP attachment for PP ambiguity resolution); D drives (a) prompt rewriting for the text encoder and (b) attention guidance of a frozen Stable Diffusion or Sana backbone](overview.svg)

*Overview of CCG Diffusion: the syntactic information read from the derivation
D reaches the image generator through (a) the input prompt of the text encoder
and (b) attention guidance.*

[`docs/method.md`](docs/method.md) describes every component with its module,
constants and implementation details.

---

## Repository layout

```text
src/ccg_diffusion/      the package (the library)
  config.py             generation and parse contracts
  methods.py            the four CCG Diffusion configurations (2 tasks x 2 backbones) and the ablation switches
  provenance.py         canonical JSON, atomic writers, sealing of the source files to the Git commit
  parsing/              Stage 1-2: Hol-CCG derivations, Supertag-Constrained Parsing, C&C marked dependencies
  targets/              Stage 3: guidance targets (E, C, K, P) as text-encoder token indices
  guidance/             Predicated Diffusion loss, latent update and the CCG Diffusion losses (numerical kernels)
  backbones/            guided denoising on Stable Diffusion v1.4 (sd14/) and Sana 0.6B (sana/)
  compositional/        CCG Diffusion on T2I-CompBench++: per-prompt plans, SD1.4 and Sana runs, ablations
  pp_attachment/        CCG Diffusion on TAB PP: interpretations, jobs, SD1.4 and Sana runs, ablations
  baselines/            the compared methods on T2I-CompBench++ and TAB PP
  evaluation/           T2I-CompBench++ evaluator driver, TAB PP judges, statistics
  single_prompt.py      CCG Diffusion on one arbitrary prompt
scripts/                command lines, run as `python -m scripts.<dir>.<name>` (catalogue: scripts/README.md)
  ccg/                  Stage 1-3 and derivation display
  t2i_compbench/        T2I-CompBench++ runs, baselines, sample export and official evaluation
  tab_pp/               TAB PP interpretations, runs, compared methods and judges
  data/                 build and verify the benchmark populations
  setup/                models, evaluator, baseline environment, judge server
  paper/                generation-time benchmark (compute_cost.py)
configs/parsing/        Hol-CCG parse configuration and the C&C markedup additions
data/                   frozen inputs: T2I-CompBench++ parses, TAB PP interpretations and jobs, benchmark prompts
dataset/derived/        the T2I-CompBench++ and TAB populations (prompts, seeds, dataset contracts)
docker/                 runtime image, pinned requirements, evaluator and baseline-environment pins, agent overlay
docs/                   documentation (see Documentation below)
tests/                  CPU test suite and its fixtures
output/                 generated runs (gitignored, except a few frozen inputs read by the code)
external/               evaluator clone and virtual environments created by scripts/setup/ (gitignored)
```

[`docs/data.md`](docs/data.md) describes `data/`, `dataset/` and `output/`.

---

## Setup (Docker — supported path)

The runtime image ([`docker/Dockerfile`](docker/Dockerfile)) is based on
`nvcr.io/nvidia/pytorch:24.12-py3` (Python 3.12) and adds torch 2.13.0 built for
CUDA 12.9, the pins of [`docker/requirements.txt`](docker/requirements.txt),
Hol-CCG (`holccg` 1.0.0 at its pinned commit) and the spaCy model
`en_core_web_sm` 3.8.0. Generation needs an NVIDIA GPU; the tests run on the CPU.
Clone the repository with Git: every run records the commit and seals its
source files to it.

```bash
git clone https://github.com/ryosuke-yamaki/ccg-diffusion.git && cd ccg-diffusion
cp .env.example .env                    # then adjust (see below)
docker compose build runtime
docker compose run --rm runtime bash    # a shell in the runtime container, at the repository root
```

Inside the container:

```bash
python -m scripts.setup.download_models          # SD1.4, Sana 0.6B and Hol-CCG at their pinned revisions (about 15 GB)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # from here on, models load from the cache only
python -m pytest -q
```

The Hugging Face cache lives in the named volume `hf-cache`, so the models
survive container re-creation. A command downloads a missing model only when it
is given `--allow-download`.

### `.env`

`.env` is gitignored; `cp .env.example .env` lists every key that
[`docker-compose.yml`](docker-compose.yml) reads, with working defaults:

* `HOST_UID` / `HOST_GID`: your host user (`id -u`, `id -g`), so that files
  written into the mounted checkout keep their owner.
* `NVIDIA_VISIBLE_DEVICES`: the GPUs passed to the container (`all`, indices
  or UUIDs from `nvidia-smi -L`).
* `DATASET_DIR` / `OUTPUT_DIR`: host directories mounted over `dataset/` and
  `output/` (default: the ones in the checkout).
* `OPENAI_API_KEY`: only for the GPT-6 Astra judge
  (`python -m scripts.tab_pp.judge_astra run`). `HF_TOKEN` is optional; the
  pinned models are public.

Inside the container, select a GPU by UUID (`CUDA_VISIBLE_DEVICES=GPU-...`,
then `--device cuda:0`) and run one backbone per process: SD1.4 switches on
deterministic algorithms for the whole process, and Sana refuses to run where
they are on.

### Without Docker

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu129
pip install -e ".[dev]"
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```

Install in editable mode only: the package reads repository files and stops
with an error in a non-editable install. SD1.4 T2I-CompBench++ generation checks
the software of the runtime image (Python 3.12.3, torch 2.13.0+cu129, diffusers
0.39.0, transformers 5.14.1, CUDA runtime 12.9, cuDNN `92000`); with other
versions pass `--no-strict-runtime`, which records them instead of enforcing
them. Details: [`docs/installation.md`](docs/installation.md).

### Additional environments

Only the experiments that need them:

* **T2I-CompBench++ evaluator**: `python -m scripts.setup.setup_t2i_compbench`
  clones the official repository at commit `1b709499` into `external/`, builds
  its isolated Python 3.10 environment and downloads the metric weights
  (`--verify-only` checks an existing installation).
* **Baseline adapter environment**: `bash scripts/setup/setup_baseline_env.sh`
  builds the Python 3.11 environment (Stanza 1.4.2, spaCy 3.6.1
  `en_core_web_trf`) for the preprocessing of four T2I-CompBench++ baselines;
  it needs [`uv`](https://docs.astral.sh/uv/).
* **Qwen3.8-Flash-Next judge**:
  [`scripts/setup/qwen38_flash_next/serve.sh`](scripts/setup/qwen38_flash_next/serve.sh)
  serves `nvidia/Qwen3.8-Flash-Next-NVFP4` with vLLM 0.30.0 on two nodes.

### Optional agent overlay

In-container tooling for coding agents is kept out of the runtime image. To
enable it, copy `docker-compose.override.example.yml` to
`docker-compose.override.yml` (gitignored, merged automatically by Docker
Compose) and follow the usage notes in its header; it builds
[`docker/Dockerfile.agent`](docker/Dockerfile.agent) on top of the runtime image
with the entrypoint [`docker/entrypoint.agent.sh`](docker/entrypoint.agent.sh).
Nothing in this repository needs it.

---

## Data

Everything the experiments read is in the repository or is downloaded by the
setup and data commands; model weights and the official benchmark files are
not redistributed.

* **T2I-CompBench++**
  ([`dataset/derived/t2i-compbench/t2i-compbench-full-v1/`](dataset/derived/t2i-compbench/t2i-compbench-full-v1/dataset_contract.json)):
  the official test split (commit `1b709499` of `Karine-Huang/T2I-CompBench`),
  300 prompts in each of the eight subcategories, each generated with the image
  seeds 42-51, i.e. 24,000 images per method. Verify it with
  `python -m scripts.data.verify_t2i_compbench`.
* **TAB**
  ([`dataset/derived/tab/tab-ppvp-disambiguation-v1/`](dataset/derived/tab/tab-ppvp-disambiguation-v1/dataset_contract.json)):
  the syntax PP and syntax VP items of TAB (commit `2e9d70c3` of
  `amazon-science/tab-dataset`) with their visual setups and intention questions,
  copied verbatim. The experiments use the 74 PP items: 2 interpretations x
  seeds 0-4 = 740 trials per method. Verify it with
  `python -m scripts.data.verify_tab_ppvp_disambiguation_dataset` after
  `python -m scripts.data.prepare_tab_pp_benchmark` and
  `python -m scripts.data.prepare_tab_vp_benchmark` have downloaded the official
  CSVs (generation and judging do not need them).
* **Frozen parses of T2I-CompBench++** ([`data/t2i_compbench/`](data/t2i_compbench/README.md)):
  the Stage 1 derivations, Stage 2 marked dependencies and coverage rows of the
  2,400 prompts. A re-parse on other hardware can flip near-tie supertags, so
  the T2I runs start from these files and Stages 1-3 run on the CPU.
* **Frozen TAB PP inputs** ([`data/tab_pp/`](data/tab_pp/README.md)): the 148
  interpretations of the 74 PP items, the source jobs and the digests that
  `preflight` compares prepared jobs with. `python -m scripts.tab_pp.parse_interpretations`
  derives the interpretations from the prompts instead.
* **Generation-time prompts** ([`data/compute_cost/`](data/compute_cost/prompts.json)):
  the 240 measured prompts (30 per category) and the warm-up prompts of the
  generation-time benchmark `python -m scripts.paper.compute_cost`.

`python -m scripts.data.build_t2i_compbench --check` rebuilds the
T2I-CompBench++ population from the evaluator clone and compares it;
`prepare_tab_pp_benchmark`, `prepare_tab_vp_benchmark` and
`prepare_tab_ppvp_disambiguation_dataset` (under `scripts/data/`) download the
official TAB files and rebuild the TAB population. See
[`docs/data.md`](docs/data.md).

---

## Quick start

One prompt. This path parses the prompt with Hol-CCG at run time and generates
a single image, whereas a benchmark run starts from the frozen parses of the
benchmark prompts and generates in physical batches of four images; the two
paths need not give the same image for the same prompt and seed:

```bash
python -m scripts.ccg.show_derivation --prompt "a blue banana and a green vase"
python -m scripts.t2i_compbench.run_ccg_diffusion prompt --backbone sd14 \
    --prompt "a blue banana and a green vase" --output output/analysis/one_prompt
```

```python
from ccg_diffusion import single_prompt

result = single_prompt.generate("three red apples on the left of a green vase", backbone="sana", seed=42)
result.image.save("apples.png")   # result.record: derivation, guidance targets, plan, diagnostics
```

The jobs of a TAB PP run are compiled on the CPU from the frozen inputs and
compared with the reference digests in `data/tab_pp/reference.json` (`R` stands
for a new run ID of the form `<YYYYMMDD>_<slug>`):

```bash
python -m scripts.tab_pp.run_ccg_diffusion prepare --backbone sd14 --run-id R
python -m scripts.tab_pp.run_ccg_diffusion preflight --backbone sd14 --run-id R   # prints PREFLIGHT_PASS
```

The configurations and the ablations can be listed without loading a model:
`python -m ccg_diffusion.methods list` prints the four configurations,
`inspect <config_id>` one configuration record, `ablations` every ablation
condition (T2I-CompBench++) and cell (TAB PP) with its `--remove` names, and
`table8` the generation settings and hyperparameters of every configuration
(one column per backbone) as JSON.
The complete command sequences of CCG Diffusion, the ablations, the compared
methods, scoring and judging are in [`docs/t2i_compbench.md`](docs/t2i_compbench.md),
[`docs/tab_pp.md`](docs/tab_pp.md) and [`docs/evaluation.md`](docs/evaluation.md).

---

## Hardware

| Experiment | Requirement |
|---|---|
| T2I-CompBench++ generation (CCG Diffusion, compared methods, ablations) | NVIDIA A100 80GB, one process per GPU (four GPUs run the four SD1.4 shards in parallel); a process needs about 9-12 GiB of GPU memory |
| Generation-time benchmark (`python -m scripts.paper.compute_cost`) | one NVIDIA A100 80GB without other processes on it |
| TAB PP generation (CCG Diffusion, ablation cells, compared methods) | one NVIDIA GeForce RTX 4090 24GB |
| GPT-6 Astra judge | OpenAI API access to `gpt-6-astra` (paid requests; `judge_astra run` needs `OPENAI_API_KEY` and a spending ceiling `--budget-usd`) |
| Qwen3.8-Flash-Next judge | two NVIDIA DGX Spark systems connected directly by a QSFP cable, serving the NVFP4 checkpoint with vLLM 0.30.0 (tensor parallel 2 with expert parallelism). The judge re-judges the trials of a finished GPT-6 Astra judgment, so it and Cohen's kappa of the two judges need the complete GPT-6 Astra judgment of the ten TAB PP method-comparison conditions (5 methods x 2 backbones, 7,400 paid trials, about 37 USD) first |
| T2I-CompBench++ evaluator | one NVIDIA GPU, in its own Python 3.10 environment |
| Stages 1-3, plans, TAB PP jobs, tests | CPU |

Plans, guidance targets and TAB PP jobs are computed from the frozen inputs and
are the same on any machine; the CPU checks in
[`docs/t2i_compbench.md`](docs/t2i_compbench.md#checking-the-preparation-without-a-gpu)
and [`docs/tab_pp.md`](docs/tab_pp.md#checking-every-cell-on-a-cpu) verify them
against reference digests. Generated images depend on the GPU type: on the GPU
types above, Sana images (both tasks) and SD1.4 TAB PP images are bit-identical
between runs, whereas SD1.4 on T2I-CompBench++ is not bit-deterministic between
runs on the A100, so its scores reproduce statistically. On other GPU types the
images, and hence the scores, are a new measurement. Times, costs and the
details are in [`docs/installation.md`](docs/installation.md#hardware-and-determinism).

---

## Statistical tests

The unit of every test is the prompt for T2I-CompBench++ (300 per subcategory,
mean over its ten seeds) and the item for TAB PP (74, mean over 2
interpretations x 5 seeds). Two methods evaluated on the same prompts are
compared with a two-sided paired sign-flip permutation test on the mean of the
per-unit differences; a contrast between two different prompt sets (for example
the effect of a component in two categories) uses a two-sided label permutation
test. Both use 200,000 permutations drawn from
`numpy.random.default_rng(20260927)`. The p-values are Holm-corrected within
each family of contrasts (for example, the eight subcategories of one
backbone). The functions are `paired_permutation_test`,
`label_permutation_test`, `holm`, `significance_mark` and `cohen_kappa` in
[`ccg_diffusion.evaluation.statistics`](src/ccg_diffusion/evaluation/statistics.py);
[`docs/evaluation.md`](docs/evaluation.md#statistical-tests) lists the families.

---

## Documentation

| File | Content |
|---|---|
| [`docs/installation.md`](docs/installation.md) | requirements, container and Python environment, models, evaluator, baseline environment, judges, hardware and determinism |
| [`docs/data.md`](docs/data.md) | `data/`, `dataset/` and `output/`; the T2I-CompBench++ and TAB populations |
| [`docs/parsing.md`](docs/parsing.md) | Stages 1-3, the frozen parse bundle, Supertag-Constrained Parsing |
| [`docs/method.md`](docs/method.md) | every component with its constants and modules; the generation settings and hyperparameters of each configuration |
| [`docs/t2i_compbench.md`](docs/t2i_compbench.md) | T2I-CompBench++ runs, ablations, compared methods, scoring |
| [`docs/tab_pp.md`](docs/tab_pp.md) | TAB PP runs, ablations, compared methods, judging |
| [`docs/evaluation.md`](docs/evaluation.md) | metrics, judges, statistical tests |
| [`docs/provenance.md`](docs/provenance.md) | glossary of the labels written into run records (arm, condition and cell IDs) |
| [`scripts/README.md`](scripts/README.md) | every command line |

---

## Tests

```bash
python -m pytest -q                         # CPU only, under a minute
python -m compileall -q src scripts tests
flake8 src scripts tests                    # .flake8: max line length 120
```

Tests that need the cached SD1.4 or Gemma tokenizer, `holccg` or the spaCy
model are skipped when the resource is missing (markers in
[`pytest.ini`](pytest.ini)). In the runtime container with the models
downloaded, every test runs and passes.

---

## License

The code and documentation written for this project are under the MIT License
([`LICENSE`](LICENSE)). The exceptions are listed in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md):

* ported code keeps the license of its source; the files derived from
  Predicated Diffusion and SynGen, which publish no license, are included with
  attribution and are not covered by the MIT License;
* the TAB-derived files (`dataset/derived/tab/`, `data/tab_pp/`) are under
  CC BY-SA 4.0;
* Stage 2 (marked dependencies), which every CCG Diffusion run uses, relies on
  the C&C `generate` port and the C&C grammar resources bundled in `holccg`;
  these are under the C&C tools licence (non-commercial use) and the Java C&C
  BSD 2-Clause licence, as stated in the `holccg` notices, and
  `configs/parsing/candc_markedup_additions` extends them under the same terms.
  The C&C tools licence permits use for education and research only and
  requires research that uses them to cite Clark and Curran, "Parsing the WSJ
  using CCG and Log-Linear Models" (ACL 2004);
* models and services keep their own licenses and terms.

---

## Acknowledgements

CCG Diffusion extends [Predicated Diffusion](https://github.com/tksmatsubara/PredicatedDiffusion),
whose implementation builds on [Attend-and-Excite](https://github.com/yuval-alaluf/Attend-and-Excite)
and [prompt-to-prompt](https://github.com/google/prompt-to-prompt). The compared
methods are ported from [Composable Diffusion](https://github.com/energy-based-model/Compositional-Visual-Generation-with-Composable-Diffusion-Models-PyTorch),
[StructureDiffusion](https://github.com/weixi-feng/Structured-Diffusion-Guidance),
Attend-and-Excite, [SynGen](https://github.com/RoyiRa/Linguistic-Binding-in-Diffusion-Models)
and Predicated Diffusion. The experiments use the benchmarks and evaluators of
[T2I-CompBench++](https://github.com/Karine-Huang/T2I-CompBench) and
[TAB](https://github.com/amazon-science/tab-dataset), the parser
[Hol-CCG](https://github.com/ryosuke-yamaki/hol-ccg) with the C&C `generate`
resources, [Diffusers](https://github.com/huggingface/diffusers) and
[vLLM](https://github.com/vllm-project/vllm). We thank the authors of all these
projects.

This work was supported by JST ACT-X, Japan, Grant Number JPMJAX24CS.

Provenance and third-party notices: see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
