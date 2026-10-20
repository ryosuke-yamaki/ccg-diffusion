# Data

## What lives where

| Directory | Content | In Git |
|---|---|---|
| `data/` | frozen input bundles: [`data/t2i_compbench/`](../data/t2i_compbench/README.md) (Stage 1-2 outputs of the 2,400 T2I-CompBench++ prompts), [`data/tab_pp/`](../data/tab_pp/README.md) (the TAB PP interpretations and source jobs, with the digests that `preflight` compares with) and `data/compute_cost/` (the prompts of the timing benchmark) | yes |
| `dataset/derived/` | the two benchmark populations, `t2i-compbench/t2i-compbench-full-v1/` and `tab/tab-ppvp-disambiguation-v1/` | yes |
| `dataset/tab/official/` | the official TAB sources, downloaded on demand | no |
| `configs/parsing/` | the Stage 1 parse configuration and the C&C markedup overlay of Stage 2 ([parsing.md](parsing.md)) | yes |
| `output/` | the runs you generate and evaluate, plus nine frozen inputs ([below](#frozen-inputs-under-output)) | only the frozen inputs |
| `external/` | local clones, virtual environments and weights (T2I-CompBench++ evaluator, baseline adapter environment; [installation.md](installation.md)) | no |

## Runs under `output/`

Every command that generates or evaluates takes a `--run-id` of the form
`<YYYYMMDD>_<slug>` and derives all of its paths from it:

* `output/generation/<run_id>/`: Stages 1-4 (parse inputs, plans, protocol,
  images and their metadata);
* `output/vqa/<run_id>/`: evaluation (exported samples, metric outputs, judge
  requests and answers), with the same `run_id` as the generation it scores;
* `output/analysis/<topic>/`: outputs that span several runs.

Since generation and evaluation of one experiment share the `run_id`,
`grep -r <run_id> output/` finds all of its files. Everything a run writes is
git-ignored. Records are JSON written with sorted keys
(`provenance.canonical_json`), store repository-relative paths and record the
sha256 of their inputs.

### Frozen inputs under `output/`

Nine tracked files under `output/` are inputs, not results, and must stay at
their paths:

| File | Read by |
|---|---|
| `output/analysis/t2i_compbench/t2icb_source_pin.json` | the pin of the official T2I-CompBench++ clone, prompt files, evaluator venv and metric weights. Its sha256 is part of the dataset contract (`source.pin_sha256`); `scripts.data.verify_t2i_compbench`, `scripts.setup.setup_t2i_compbench` and the evaluator check it |
| `output/generation/20260812_t2icb_sd14_six_arm_full_v1/input/stage123/<category>.auto.meta.json` (8 files) | the Stage 1 metadata (checkpoint, parse settings, input and output sha256) of the derivations in [`data/t2i_compbench/`](../data/t2i_compbench/README.md). The bundle manifest names these paths, and SD1.4 T2I `prepare` reads them as the parser provenance of the frozen derivations |

`docker-compose.yml` mounts `DATASET_DIR` and `OUTPUT_DIR` of `.env` over the
checkout's `dataset/` and `output/`. Pointing either at another directory
hides the tracked files below it (the frozen datasets, or the nine files above)
unless they are copied there with their relative paths, for example

```bash
(cd output && git ls-files -z . | xargs -0 cp --parents -t /path/to/new/output)
(cd dataset && git ls-files -z . | xargs -0 cp --parents -t /path/to/new/dataset)
```

## T2I-CompBench++

`dataset/derived/t2i-compbench/t2i-compbench-full-v1` (dataset ID
`t2i-compbench-full-v1`, version 1) is the official test split of
T2I-CompBench++ at commit `1b7094991a57f3c22abdd4f6e8ba6c1a15517073` of
`Karine-Huang/T2I-CompBench`: 300 prompts in each of the eight categories
(color, shape, texture, 2D-spatial `spatial`, 3D-spatial `3d_spatial`,
numeracy, non-spatial `non_spatial`, complex), 2,400 prompts, each generated
with the ten seeds 42-51, i.e. 24,000 images per method.

| File | Content |
|---|---|
| `prompts/<category>_val.txt` | the official prompt files (as read by the official evaluators) |
| `prompt_only.jsonl` | one row per prompt (seed 0), the input of Stages 1-3 |
| `independent_seed_manifest.jsonl` | the 24,000 image rows (`experiment`, `prompt_id`, `prompt`, `seed`) |
| `selected_prompt_manifest.jsonl`, `selection_manifest.json` | the selection record (all prompts in source order; sha256 `959de730...`) |
| `dataset_contract.json` | dataset identity, population, pairing and resampling rules |

```bash
python -m scripts.data.verify_t2i_compbench             # verify the frozen dataset against its contract
python -m scripts.data.build_t2i_compbench --check      # rebuild from the official clone and compare
```

The verifier checks the contract, the selection manifest, the prompt files, the
three manifests, the source pin and the exact file inventory of the directory,
so do not add files to it. `build_t2i_compbench` reads the official clone that
`scripts.setup.setup_t2i_compbench` installs ([installation.md](installation.md)).
Comparisons between methods pair images by category, prompt position and seed;
a significance test averages the ten seeds of a prompt and resamples prompts
([evaluation.md](evaluation.md)). Categories use different metrics and are
never averaged together.

## TAB PP

`dataset/derived/tab/tab-ppvp-disambiguation-v1` (dataset ID
`tab-ppvp-disambiguation`, version 1) holds the syntax PP and syntax VP items of
TAB (Text-to-image Ambiguity Benchmark, `amazon-science/tab-dataset` at commit
`2e9d70c3c44037ef7e6a2c96bf8efb31bd6aed0e`) with their visual setups and
intention questions copied verbatim: 315 items (74 PP, 241 VP), each with two
interpretations. The TAB PP experiments use the 74 PP items: 74 items x 2
interpretations x seeds 0-4 (`config.TAB_PP_SEEDS`) = 740 trials per method.

| File | Content |
|---|---|
| `items.jsonl` | the 315 items: prompt, the two visual setups and intention questions, commonsense labels |
| `independent_seed_manifest.jsonl`, `prompt_only.jsonl` | per-item seed rows (prompt IDs `tab-pp-000_s42`, ..., seeds 42-45) that are part of the frozen dataset and its contract; the TAB PP commands do not read them and use seeds 0-4 (`config.TAB_PP_SEEDS`) |
| `dataset_contract.json` | dataset identity, official source files with their sha256, selection rules and exclusions, CLIP token-length check |

The official sources go to `dataset/tab/official/amazon-science-tab-dataset-2e9d70c3/`
(not tracked). Download them before verifying or rebuilding the dataset:

```bash
python -m scripts.data.prepare_tab_pp_benchmark          # download and freeze the official syntax-PP CSV
python -m scripts.data.prepare_tab_vp_benchmark          # the same for syntax VP
python -m scripts.data.verify_tab_ppvp_disambiguation_dataset                     # verify the frozen dataset
python -m scripts.data.prepare_tab_ppvp_disambiguation_dataset --output-dir DIR   # rebuild it into DIR
```

The verifier checks the official CSVs against their pinned sha256 and
re-derives every file of the frozen dataset from them
(`--with-clip-token-check` also recomputes the token-length check with the
pinned SD1.4 tokenizer). Generation and judging do not need the official
sources: they read the frozen dataset and [`data/tab_pp/`](../data/tab_pp/README.md),
and the judges check the sha256 of the four dataset files and the file
inventory of the directory, so do not add files to it either.

### The 3,150-job layout

SD1.4 TAB runs place every job at a fixed position in a layout of 3,150 jobs:
the 315 items in dataset order x interpretations `np_attach`, `vp_attach` x
seeds 0-4. Physical batches of 4 are consecutive positions
(`physical_batch_index = position // 4`). The 740 PP jobs occupy positions
0-739, so batches 0-184 contain only PP jobs and are the only batches
generated; the 2,410 VP jobs are never generated. They are kept so that every
PP image keeps the batch of 4 it is generated in, which bit-exact regeneration
on the RTX 4090 requires, and so that a prepared job file can be compared row
for row with the reference digests ([tab_pp.md](tab_pp.md)). The SD1.4 Vanilla
and VS-TIED + Gold-visual setup generators likewise cover all 315 items (1,575
and 3,150 images); only the PP images are judged.

## Timing prompts

`data/compute_cost/prompts.json` holds the 240 T2I-CompBench++ prompts of the
timing benchmark (30 per category, seed 42) and `data/compute_cost/warmup.json`
its 8 warm-up prompts (the first prompt of each category). They are the
defaults of `python -m scripts.paper.compute_cost benchmark`, the
generation-time benchmark of Vanilla, Predicated Diffusion and CCG Diffusion
([evaluation.md](evaluation.md#generation-time-benchmark)).

## Licenses

TAB is distributed under CC BY-SA 4.0; `dataset/derived/tab/` and the derived
bundle `data/tab_pp/` are distributed under the same license. T2I-CompBench++
is MIT-licensed. See [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).
