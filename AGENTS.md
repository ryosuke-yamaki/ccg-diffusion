# AGENTS.md

Operational brief for coding agents working in this repository (`CLAUDE.md` is a
symlink to this file). Keep it thin: it points to the human docs rather than
restating them.

## Setup & run

- Docker is the supported environment (README "Setup"): `docker compose build runtime`,
  then work in `docker compose run --rm runtime bash`. Without Docker, use an
  editable install (`pip install -e ".[dev]"`); the package reads repository
  files, so a non-editable install fails on purpose.
- Command lines are `scripts/<dir>/<name>.py`, run from the repository root as
  `python -m scripts.<dir>.<name>` (catalogue: [`scripts/README.md`](scripts/README.md)).
  `pytest.ini` puts `src` and the repository root on the path; elsewhere use
  `PYTHONPATH=src:.`.
- Models load from the Hugging Face cache only: `python -m scripts.setup.download_models`,
  then `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`; a command downloads only with
  `--allow-download`.
- GPUs: check `nvidia-smi -L` and select by UUID (`CUDA_VISIBLE_DEVICES=GPU-...`,
  then `--device cuda:0`); do not assume GPU indices. Run one backbone per
  process: SD1.4 enables process-global deterministic algorithms, and Sana
  refuses to run where they are enabled.
- The four CCG Diffusion configurations and the ablations, without loading a
  model: `python -m ccg_diffusion.methods {list,inspect,catalog,ablations,table8}`.

## Verify (run after any code change)

```bash
python -m pytest -q                         # CPU only, under a minute
python -m compileall -q src scripts tests
flake8 src scripts tests                    # .flake8: max line length 120
```

Run the tests related to a change; run the whole suite for shared modules.
Tests that need the cached tokenizers, `holccg` or the spaCy model are skipped
when the resource is missing. No test may depend on untracked run outputs.

## Conventions

- Branch off `main`. Commit messages, comments, docstrings, error messages and
  docs in English. Type hints on new functions; match the error handling of the
  surrounding code.
- **Layering** (`tests/test_layering.py`): `src/` never imports `scripts/`;
  imports follow the allowed edges; package `__init__.py` files import nothing;
  the light modules import without torch, transformers, diffusers, holccg or
  spaCy; every CLI answers `--help` without them (`tests/test_datasets_and_cli.py`).
- **Numerics.** The guidance kernels (`src/ccg_diffusion/guidance/` and the
  guided denoising loops under `backbones/` and `baselines/`) determine the
  generated images bit for bit wherever generation is deterministic
  ([`docs/installation.md`](docs/installation.md#hardware-and-determinism)).
  Do not rewrite their arithmetic: keep operand order, dtype casts, in-place
  versus out-of-place operations, literals and loop order. Kernels own their
  constants; `methods.py` only describes the configurations, and the tests
  assert that both agree.
- **Recorded values.** String literals written into records stay verbatim:
  schema IDs (including the `holsd.*` prefix), arm, condition, cell, route,
  provider and profile labels (`B0`/`P1`/`P2`, `gs1_p0_r1`, `c1r1p0m0`, ...),
  reason strings, recorded error texts, dataset IDs. Identifiers in code use the
  component names of the method (`selective_guidance_boost`, `region_assignment_loss`,
  `disambiguating_paraphrase`, ...); [`docs/provenance.md`](docs/provenance.md) maps
  the recorded labels to them.
- **Records.** Write JSON with `provenance.canonical_json` and the atomic
  writers; store repository-relative paths. A run writes only into
  `output/{generation,vqa}/<run_id>` (`<YYYYMMDD>_<slug>`, shared by generation
  and evaluation) or `output/analysis/<topic>`; never modify the records of a
  finished run, start a new run ID instead.
- **Frozen inputs.** `data/`, `dataset/derived/`, `tests/fixtures/` and the few
  tracked files under `output/` are frozen: their bytes are pinned by manifests,
  dataset contracts, `data/tab_pp/reference.json` or the tests. Do not edit them.
- **Hol-CCG** is the external `holccg` package at a pinned commit. Parser
  changes belong upstream; the Supertag-Constrained Parsing extension lives in
  `parsing/holccg.py`.
- **Command lines:** hyphenated flags; `--run-id` derives every output path;
  `--allow-dirty` records an uncommitted working tree; subsets are `--batches`
  (SD1.4 physical batches) and `--keys` (Sana prompt keys); heavy imports go
  inside `main`.
- Third-party code keeps its attribution ([`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md));
  the files derived from Predicated Diffusion and SynGen are not under the MIT License.
- Never commit run outputs, model weights or `external/`; do not force-add
  ignored files. Personal settings (`.env`, `docker-compose.override.yml`, local
  agent files) stay untracked.

## Gotchas

- Runs seal their source files to the Git commit: `prepare` stops when those
  files have uncommitted changes unless given `--allow-dirty` (recorded; the T2I
  `generate` then needs it too). A prepared run records the bytes of its source
  files (for example `SD14_SOURCE_FILES` in `pp_attachment/jobs.py`) and
  `generate` re-verifies them, so do not edit those files between `prepare` and
  `generate`.
- SD1.4 T2I-CompBench++ generation enforces the runtime image's software pins
  (`SD14_GENERATOR_RUNTIME_PINS`); elsewhere pass `--no-strict-runtime`.
  Baseline inputs built in a new adapter venv need `--no-strict-environment`.
- A re-parse on other hardware can flip near-tie supertags; the T2I-CompBench++
  parses and TAB PP interpretations are frozen in `data/` and are the defaults
  (`prepare_prompts --parse-inputs frozen`,
  `tab_pp.run_ccg_diffusion prepare --inputs frozen`).
- The official T2I-CompBench++ evaluator runs in its own Python 3.10 venv:
  call it as `env -u PYTHONPATH -u LD_LIBRARY_PATH external/t2i_compbench/.venv/bin/python`
  (`scripts.t2i_compbench.evaluate` does this itself).
- `python -m scripts.tab_pp.judge_astra run` is the only command that sends paid
  API requests (`OPENAI_API_KEY`, `--budget-usd`); do not run it without the
  user's explicit approval. Every other `judge_astra` subcommand is offline.
- An interrupted Sana `generate` resumes, but regroups the remaining images
  into new batches, so the result is not guaranteed to be byte-identical to an
  uninterrupted run.

## Pointers

- [`README.md`](README.md): method summary, repository layout, setup, data, quick start, hardware
- [`scripts/README.md`](scripts/README.md): every command line
- [`docs/installation.md`](docs/installation.md): environments, models, evaluator, judges, hardware and determinism
- [`docs/data.md`](docs/data.md): `data/`, `dataset/`, `output/`
- [`docs/parsing.md`](docs/parsing.md): Stages 1-3, Supertag-Constrained Parsing
- [`docs/method.md`](docs/method.md): components, constants, hyperparameters
- [`docs/t2i_compbench.md`](docs/t2i_compbench.md), [`docs/tab_pp.md`](docs/tab_pp.md): run commands
- [`docs/evaluation.md`](docs/evaluation.md): metrics, judges, statistical tests
- [`docs/provenance.md`](docs/provenance.md): glossary of recorded labels
- [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md): third-party code, data, models and licenses
