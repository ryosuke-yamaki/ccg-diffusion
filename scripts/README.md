# Scripts

The command lines of CCG Diffusion. Run every Python script from the repository
root as `python -m scripts.<dir>.<name>` and the shell scripts with `bash`. Each
Python command line prints its full usage with `--help`, which works without
torch, diffusers, transformers, `holccg` or spaCy (`evict_ckpt.py` takes no
arguments). This page is only a map: the workflows are in
[`docs/`](../docs/) (linked per section). `R` stands for a run ID
`<YYYYMMDD>_<slug>`; a run writes under `output/generation/R/` and
`output/vqa/R/`.

## Parsing (`ccg/`)

Stages 1-3 of CCG Diffusion ([`docs/parsing.md`](../docs/parsing.md)).

| Script | Purpose |
|--------|---------|
| `parse.py` | Stage 1: parse a tokenized `.raw` corpus (one sentence per line) with Hol-CCG into CCGbank AUTO, with `<auto>.meta.json`; defaults from `configs/parsing/holccg.yaml`. |
| `generate_dependencies.py` | Stage 2: C&C `generate` marked dependencies and coverage rows of an AUTO corpus (`PARTIAL` derivations are generated fragment by fragment). |
| `build_guidance_targets.py` | Stage 3: the guidance targets (E, C, K, P) of one T2I-CompBench++ category from its Stage 1-2 outputs. |
| `show_derivation.py` | Print the derivation and the marked dependencies of one prompt; `--force-supertag INDEX CATEGORY` uses Supertag-Constrained Parsing, e.g. `--force-supertag 6 "(NP\NP)/NP"` or `"((S\NP)\(S\NP))/NP"` for the two interpretations of "the boy looks at the girl with a telescope". |

## T2I-CompBench++ (`t2i_compbench/`)

CCG Diffusion, its component ablations and the compared methods on
T2I-CompBench++, and the official scoring
([`docs/t2i_compbench.md`](../docs/t2i_compbench.md),
[`docs/evaluation.md`](../docs/evaluation.md)).

| Script | Purpose |
|--------|---------|
| `prepare_prompts.py` | Stage 1-3 inputs of the eight categories in one stage directory: prompt files, parse corpora, the frozen Stage 1-2 outputs of `data/t2i_compbench/` (`--parse-inputs parse` prints the Stage 1-2 commands instead) and the source manifest; prints the Stage 3 commands. |
| `run_ccg_diffusion.py` | CCG Diffusion on SD1.4 (four shards) or Sana: `prepare` (per-prompt plans and run protocol, CPU), `generate`, `audit`; `--remove` selects an ablation condition, `--method vanilla` Sana without guidance (Sana Vanilla); `prompt` generates one image of an arbitrary prompt with a live parse (not a benchmark run). |
| `prepare_baseline_inputs.py` | Inputs of the five compared methods: `external` (blind preprocessing in the Python 3.11 adapter environment), `select-attend-and-excite` (GPU), `predicated-diffusion` (spaCy guidance targets), `sana` (Sana plans of all five). |
| `run_baselines.py` | Generate one compared method: SD1.4 one shard of the 24,000 images, Sana one category at a time; `--plan-only` prints the SD1.4 batch membership without a GPU. |
| `audit_sana_baselines.py` | Check the finished Sana categories of one compared method (`--run-id`, `--method`, `--categories`) against the Sana CCG Diffusion run named by `--reference-run` (same initial latents). |
| `official_reference/generate.py` | Port of the official T2I-CompBench++ reference generator (`GORS_finetune/inference_eval.py`); run by `run_all_categories.sh`, needs the evaluator installation of `setup/setup_t2i_compbench.py`. |
| `official_reference/run_all_categories.sh` | `bash ... R <category> [<category> ...]`: runs `generate.py` over categories in series (SD1.4 Vanilla). |
| `export_samples.py` | Link the images of one method or ablation condition into the official `samples/` layout, sealed with their sha256. |
| `evaluate.py` | Run the official T2I-CompBench++ evaluators over a run (`--categories all`, `--dry-run` prints the plan) and keep the per-image scores. |

## TAB PP (`tab_pp/`)

CCG Diffusion, its ablation cells and the compared methods on TAB PP, and the
two judges ([`docs/tab_pp.md`](../docs/tab_pp.md),
[`docs/evaluation.md`](../docs/evaluation.md)).

| Script | Purpose |
|--------|---------|
| `run_ccg_diffusion.py` | CCG Diffusion on SD1.4 or Sana: `prepare` (jobs from the frozen inputs or a `parse_interpretations.py` directory, CPU), `preflight` (compare with the digests in `data/tab_pp/reference.json`), `generate`, `merge` (trial plan), `audit` (Sana); `--remove` selects an ablation cell (`c0r1p0m0`, `c1r0p0m0`, `c0r0p0m0`; default `c1r1p0m0`). |
| `parse_interpretations.py` | Derive the 148 interpretations of the 74 PP prompts with Supertag-Constrained Parsing, Stages 2-3, the Disambiguating Paraphrase and the Holder-Object Region Assignment; `--compare-frozen data/tab_pp` compares the result with the frozen inputs. |
| `run_baselines.py` | The compared methods (Vanilla, Predicated Diffusion, VS-TIED + Gold-visual setup, with and without Predicated Diffusion): `prepare`, `generate`, `merge`. |
| `judge_astra.py` | GPT-6 Astra judge: `prepare` (seal the trials), `preflight`, `run` (the only paid step; `--budget-usd`), `summarize` (Yes rates), `replay` (re-derive predictions from stored responses, offline). |
| `judge_qwen38.py` | Qwen3.8-Flash-Next judge through an OpenAI-compatible endpoint: `run` (`--base-url`; `--astra-run` names the GPT-6 Astra judgments whose trials are re-judged), `summarize`. |

## Benchmark data (`data/`)

The two benchmark populations under `dataset/derived/`
([`docs/data.md`](../docs/data.md)).

| Script | Purpose |
|--------|---------|
| `verify_t2i_compbench.py` | Verify the frozen T2I-CompBench++ dataset against its contract. |
| `build_t2i_compbench.py` | Build the T2I-CompBench++ dataset (300 test prompts per category x seeds 42-51) from the official clone; `--check` rebuilds it in a temporary directory and compares. |
| `prepare_tab_pp_benchmark.py` | Download and freeze the official TAB syntax-PP CSV at its pinned commit. |
| `prepare_tab_vp_benchmark.py` | The same for syntax VP. |
| `prepare_tab_ppvp_disambiguation_dataset.py` | Build the derived TAB dataset `tab-ppvp-disambiguation` from the two frozen specifications (`--output-dir DIR`). |
| `verify_tab_ppvp_disambiguation_dataset.py` | Verify the frozen TAB dataset against its contract and file hashes. |

## Setup (`setup/`)

Models and the external environments ([`docs/installation.md`](../docs/installation.md)).

| Script | Purpose |
|--------|---------|
| `download_models.py` | Download SD1.4, Sana 0.6B and the Hol-CCG checkpoint at their pinned revisions into the Hugging Face cache (`--only` selects models). |
| `setup_t2i_compbench.py` | Install the official T2I-CompBench++ evaluator: clone at its pinned commit, isolated Python 3.10 venv and metric weights under `external/`; `--verify-only`, `--smoke`. |
| `setup_baseline_env.sh` | Build the Python 3.11 adapter environment (Stanza 1.4.2, spaCy 3.6.1 `en_core_web_trf`) for `prepare_baseline_inputs.py external`; needs `uv`. |
| `qwen38_flash_next/serve.sh` | `start`, `stop` or `status` of the two-node vLLM 0.30.0 server of `nvidia/Qwen3.8-Flash-Next-NVFP4`; node addresses and interfaces come from `HEAD_IP`, `WORKER_IP`, `IFACE` and `IB_HCA`. |
| `qwen38_flash_next/evict_ckpt.py` | Drop the cached pages of that checkpoint from the page cache (no arguments). |

## Generation-time benchmark (`paper/`)

| Script | Purpose |
|--------|---------|
| `compute_cost.py` | Prompt-to-image time per image of Vanilla, Predicated Diffusion and CCG Diffusion on one backbone: `benchmark --backbone sd14 --out DIR --modes b1 --no-verify`, the same with `--backbone sana`, then `summarize DIR` (writes `DIR/summary.json`); the prompts are those of `data/compute_cost/` ([`docs/evaluation.md`](../docs/evaluation.md)). |

## Package command line

`python -m ccg_diffusion.methods` lists and inspects the four CCG Diffusion
configurations without loading any model: `list`, `inspect`, `catalog`,
`ablations` (every ablation condition and cell with its `--remove` names) and `table8`
(the generation settings and hyperparameters of every configuration as JSON).
