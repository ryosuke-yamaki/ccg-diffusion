# TAB PP attachment

This page generates the images of CCG Diffusion, of its component ablations and
of the compared methods on the 74 PP-attachment items of TAB (2 interpretations
x seeds 0-4 = 740 trials per method, [data.md](data.md#tab-pp)) and judges them
with GPT-6 Astra and Qwen3.8-Flash-Next. The method is described in
[method.md](method.md), the parsing in [parsing.md](parsing.md), the judges and
the Yes rate in [evaluation.md](evaluation.md#tab-pp), the statistical tests in
[evaluation.md](evaluation.md#statistical-tests).

Every command runs from the repository root. A run ID has the form
`<YYYYMMDD>_<slug>` (lowercase); one CCG Diffusion run holds one ablation cell
([below](#removing-components)) on one backbone. The examples use these run IDs:

```bash
T=20261020_tab_sd14          # SD1.4 CCG Diffusion (cell c1r1p0m0)
TS=20261020_tab_sana         # Sana CCG Diffusion (cell c1r1p0m0)
TB=20261020_tab_sd14_stock   # SD1.4 Vanilla and VS-TIED + Gold-visual setup
TP=20261020_tab_sd14_pd      # SD1.4 Predicated Diffusion and VS-TIED + Gold-visual setup + Predicated Diffusion
TSB=20261020_tab_sana_stock  # Sana Vanilla and VS-TIED + Gold-visual setup
TSP=20261020_tab_sana_pd     # Sana Predicated Diffusion and VS-TIED + Gold-visual setup + Predicated Diffusion
J=20261020_tab_astra         # GPT-6 Astra judgment of all conditions
Q=20261020_tab_qwen38        # Qwen3.8-Flash-Next judgment of the ten method-comparison conditions
```

## Methods and modules

| Method | Stable Diffusion v1.4 | Sana 0.6B |
|---|---|---|
| Vanilla | `baselines/tab_sd14.py` (SD1.4 batch driver, stock denoising) | `baselines/tab_sana.py` with the run layout of `pp_attachment/sana.py` |
| Predicated Diffusion | `baselines/tab_sd14.py` (guided denoising kernel) | `baselines/tab_sana.py` (Sana runtime of `baselines/sana_generation.py`) |
| VS-TIED + Gold-visual setup | `baselines/gold_visual_setup.py`, `baselines/tab_sd14.py` | `baselines/gold_visual_setup.py`, `baselines/tab_sana.py` |
| VS-TIED + Gold-visual setup + Predicated Diffusion | as Predicated Diffusion | as Predicated Diffusion |
| CCG Diffusion | `pp_attachment/` (`compile.py`, `jobs.py`, `sd14.py`) | `pp_attachment/` (`compile.py`, `sana.py`) |

Module paths are under `src/ccg_diffusion/`. The commands are
[`scripts/tab_pp/run_ccg_diffusion.py`](../scripts/tab_pp/run_ccg_diffusion.py),
[`scripts/tab_pp/run_baselines.py`](../scripts/tab_pp/run_baselines.py),
[`scripts/tab_pp/judge_astra.py`](../scripts/tab_pp/judge_astra.py) and
[`scripts/tab_pp/judge_qwen38.py`](../scripts/tab_pp/judge_qwen38.py).

## Two input paths

**Frozen path (default).** [`data/tab_pp/`](../data/tab_pp/README.md) holds the
148 interpretations of the 74 PP items (Supertag-Constrained Parsing, Stage 2
and Stage 3, the Disambiguating Paraphrase, the Holder-Object Region Assignment
and the `next_to` relations) and the 3,150 source jobs.
`run_ccg_diffusion prepare` (default `--inputs frozen`) compiles them into the
jobs of one ablation cell on the CPU (only the tokenizers are loaded), and
`preflight` compares them with the reference digests in
[`data/tab_pp/reference.json`](../data/tab_pp/reference.json).

**Clean path.** The interpretations are derived from the prompts:

```bash
python -m scripts.tab_pp.parse_interpretations --output output/analysis/20261020_tab_interpretations \
    --device cuda:0 --compare-frozen data/tab_pp
python -m scripts.tab_pp.run_ccg_diffusion prepare --backbone sd14 --run-id $T \
    --inputs output/analysis/20261020_tab_interpretations
```

`parse_interpretations` runs Supertag-Constrained Parsing with the choice of
the preposition, Stage 2 and Stage 3, the Disambiguating Paraphrase, the
Holder-Object Region Assignment and the `next_to` relations for the 74 PP items
and writes `interpretations.jsonl` (with the parse corpora, a parse manifest
and a summary) into a new directory; `--compare-frozen` also compiles the jobs
and Sana plans of every cell from both inputs and writes `comparison.json`. A
clean-path run has the 740 PP jobs at their positions of the 3,150-job layout
([data.md](data.md#the-3150-job-layout)).

The clean-path jobs equal the frozen ones on a pinned projection
(`pp_attachment.jobs.pinned_projection`): every field that affects generation
(prompt, token IDs, guidance targets, loss configuration, relations, region
layout, traces of the components, seed, batch position, evaluation setup).
Some recorded fields of the frozen jobs do not affect generation and are not
part of the projection, so a clean-path `metadata_jobs.jsonl` is not
byte-comparable with a frozen-path one, and `preflight` of a clean-path run
compares the pinned projection. Near-tie supertags can change the parse on
hardware other than an RTX 4090, and then the clean path can yield other
interpretations.

## CCG Diffusion on Stable Diffusion v1.4

```bash
python -m scripts.tab_pp.run_ccg_diffusion prepare --backbone sd14 --run-id $T
python -m scripts.tab_pp.run_ccg_diffusion preflight --backbone sd14 --run-id $T
python -m scripts.tab_pp.run_ccg_diffusion generate --backbone sd14 --run-id $T --device cuda:0
python -m scripts.tab_pp.run_ccg_diffusion merge --backbone sd14 --run-id $T
```

`prepare` writes `output/generation/$T/plan.json` and `metadata_jobs.jsonl` (the
3,150-job layout); it seals the implementation files to the Git commit and stops
when they have uncommitted changes, unless given `--allow-dirty`. `preflight`
requires the jobs to equal the reference digests row for row (run identity
excluded) and prints `PREFLIGHT_PASS`. `generate` runs the physical batches of 4
that contain PP jobs (0-184) through the guided denoising kernel, with the TAB
guidance strength of each batch ([method.md](method.md)), and writes one
receipt per batch (`batches/<cell>/batch_manifest_NNNNN.json`); `--batches
0,7,20` (or ranges, `0-3`) selects batches and `--output-dir DIR` writes a
subset outside the run. The job of item `i` (0-73 in dataset order),
interpretation `np_attach` or `vp_attach` and seed `s` sits at position
`10i + 5r + s` (`r` = 0 for `np_attach`, 1 for `vp_attach`), in physical batch
`position // 4`. `merge` checks every receipt and image and writes
`metadata.jsonl` and the trial plan `output/vqa/$T/trial_plan.jsonl`.

## CCG Diffusion on Sana 0.6B

```bash
python -m scripts.tab_pp.run_ccg_diffusion prepare --backbone sana --run-id $TS
python -m scripts.tab_pp.run_ccg_diffusion preflight --backbone sana --run-id $TS
python -m scripts.tab_pp.run_ccg_diffusion generate --backbone sana --run-id $TS --device cuda:0
python -m scripts.tab_pp.run_ccg_diffusion audit --backbone sana --run-id $TS
python -m scripts.tab_pp.run_ccg_diffusion merge --backbone sana --run-id $TS
```

The compiled jobs are projected onto Sana's Gemma tokens
(`pp_attachment.sana`). The run holds `plans.jsonl`, `metadata_jobs.jsonl`,
`run_manifest.json`, `text_encoding_manifest.json`, `images/guided/` and
`metadata.jsonl`; `preflight` compares the digests of the plans and jobs with
`data/tab_pp/reference.json` and prints `PREFLIGHT_PASS`. Each prompt key (e.g.
`guided_tab-pp-000_np_attach`) is generated as seeds 0-3 in one batch and seed 4
alone; `--keys` selects keys. `audit` checks every image record, and `merge`
writes the trial plan `output/vqa/$TS/trial_plan.jsonl`.

## Removing components

```bash
python -m scripts.tab_pp.run_ccg_diffusion prepare --backbone sd14 --run-id ${T}_c0r1p0m0 \
    --remove disambiguating_paraphrase
```

| `--remove` | Cell | Removed component |
|---|---|---|
| (none) | `c1r1p0m0` | none: the full configuration, which every other cell is compared with |
| `disambiguating_paraphrase` | `c0r1p0m0` | Disambiguating Paraphrase |
| `holder_object_region_assignment` | `c1r0p0m0` | Holder-Object Region Assignment |
| `disambiguating_paraphrase holder_object_region_assignment` | `c0r0p0m0` | both |

`--remove` is a `prepare` option; `preflight`, `generate`, `audit` and `merge`
follow the cell recorded in the run. Each cell is a run of its own on each
backbone (here `${T}_c0r1p0m0`, `${T}_c1r0p0m0`, `${T}_c0r0p0m0` on SD1.4 and
`${TS}_c0r1p0m0`, ... on Sana), with the same commands as above. Without the
paraphrase a job conditions on the source prompt with the same guidance
targets; without the Holder-Object Region Assignment it has no region layout,
so no Region Assignment Loss is applied. The other components (Selective
Guidance Boost, `next_to` Spatial Relation Loss) stay on. The cell IDs are
recorded labels: `c` = Disambiguating Paraphrase, `r` = Holder-Object Region
Assignment (1 = on); the suffix `p0m0` is part of the label
([glossary](provenance.md#glossary)). `python -m ccg_diffusion.methods
ablations` lists the cells with their `--remove` names.

## Checking every cell on a CPU

`prepare` and `preflight` need no GPU, only the cached SD1.4 and Gemma
tokenizers (`python -m scripts.setup.download_models`,
[installation.md](installation.md#models)). Each `preflight` compares the
prepared jobs (SD1.4) or plans and jobs (Sana) with the reference digests in
[`data/tab_pp/reference.json`](../data/tab_pp/reference.json) and prints
`PREFLIGHT_PASS`; this checks every backbone and cell:

```bash
for b in sd14 sana; do
  python -m scripts.tab_pp.run_ccg_diffusion prepare --backbone $b --run-id 20261020_tab_${b}_check
  python -m scripts.tab_pp.run_ccg_diffusion preflight --backbone $b --run-id 20261020_tab_${b}_check
  for remove in disambiguating_paraphrase holder_object_region_assignment both; do
    args=$remove; [ $remove = both ] && args="disambiguating_paraphrase holder_object_region_assignment"
    python -m scripts.tab_pp.run_ccg_diffusion prepare --backbone $b --run-id 20261020_tab_${b}_check_$remove \
        --remove $args
    python -m scripts.tab_pp.run_ccg_diffusion preflight --backbone $b --run-id 20261020_tab_${b}_check_$remove
  done
done
```

`prepare` seals its source files to the Git commit and stops when they have
uncommitted changes; to check a modified checkout, add `--allow-dirty` to each
`prepare`. The test suite (`python -m pytest -q`) also compares the compiled
jobs and Sana plans of all four cells with `data/tab_pp/reference.json`.

## Compared methods

| `--method` | Method | Prompt and guidance |
|---|---|---|
| `vanilla` | Vanilla | the ambiguous prompt; one image per seed, judged against both interpretations |
| `predicated_diffusion` | Predicated Diffusion | the ambiguous prompt with the spaCy guidance targets of the original Predicated Diffusion paper and its T2I settings (scale 20, cap 0.02); one image per seed, judged against both interpretations |
| `gold_visual_setup` | VS-TIED + Gold-visual setup | the prompt followed by the gold visual setup sentence of each interpretation (a period ends each sentence) |
| `gold_visual_setup_predicated_diffusion` | VS-TIED + Gold-visual setup + Predicated Diffusion | the same prompt with Predicated Diffusion |

TIED, proposed together with TAB (Mehrabi et al., ACL 2023;
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md#tab-text-to-image-ambiguity-benchmark)),
resolves an ambiguous prompt through interaction with the user; in VS-TIED the
user selects one of several visual setups, sentences that each describe a
possible interpretation, and the selected one is concatenated to the prompt.
With the gold visual setup the prompt receives the dataset's description of the
intended interpretation, so this method uses the benchmark's answer and serves
as an oracle reference, not as a parser-based method.

The two Predicated Diffusion methods form one run of their own (370 and 740
images); `prepare` takes both and `generate` runs one at a time. On SD1.4,
Vanilla and VS-TIED + Gold-visual setup cover all 315 TAB items in physical
batches of 4 (1,575 and 3,150 images, of which the 370 and 740 PP images are
judged); on Sana they cover the 74 PP items. `--batches` (SD1.4) and `--keys`
(Sana) select subsets. For item `tab-pp-<i>` (dataset order), seed `s` and
visual setup `v` (0 or 1), the SD1.4 physical batch is `(5i + s) // 4` for
Vanilla and `(10i + 5v + s) // 4` for VS-TIED + Gold-visual setup, each in its
own manifest; in the Predicated Diffusion run it is `(5i + s) // 4` for
`predicated_diffusion` (batches 0-92) and `93 + (10i + 5v + s) // 4` for
`gold_visual_setup_predicated_diffusion` (batches 93-277). The Sana keys are
`vanilla_tab-pp-<i>_None`, `gold_tab-pp-<i>_<v>`, `pd_spacy_tab-pp-<i>_None`
and `pd_spacy_gold_tab-pp-<i>_<v>`; each key runs seeds 0-3 in one batch and
seed 4 alone.

**SD1.4.**

```bash
python -m scripts.tab_pp.run_baselines prepare --backbone sd14 --method vanilla gold_visual_setup --run-id $TB
python -m scripts.tab_pp.run_baselines generate --backbone sd14 --method vanilla --run-id $TB --device cuda:0
python -m scripts.tab_pp.run_baselines generate --backbone sd14 --method gold_visual_setup --run-id $TB --device cuda:0

python -m scripts.tab_pp.run_baselines prepare --backbone sd14 --method predicated_diffusion \
    gold_visual_setup_predicated_diffusion --run-id $TP
python -m scripts.tab_pp.run_baselines generate --backbone sd14 --method predicated_diffusion --run-id $TP --device cuda:0
python -m scripts.tab_pp.run_baselines generate --backbone sd14 --method gold_visual_setup_predicated_diffusion \
    --run-id $TP --device cuda:0
python -m scripts.tab_pp.run_baselines merge --backbone sd14 --run-id $TP
```

The SD1.4 Vanilla and Gold-visual setup images go to
`output/generation/$TB/sd14_vanilla/tab_ppvp_vanilla/` and
`output/generation/$TB/sd14_tied_oracle/tab_ppvp_tied_oracle/` with one
metadata file each. `merge` does not apply to this run; its trial plan is
written from the PP rows of the two metadata files with the same function that
`merge` uses:

```bash
python - "$TB" <<'EOF'
import sys
from pathlib import Path

from ccg_diffusion.evaluation.tab import pp_profile, write_trial_plan
from ccg_diffusion.provenance import read_jsonl

run = sys.argv[1]
rows = [row for arm, experiment in (("sd14_vanilla", "tab_ppvp_vanilla"), ("sd14_tied_oracle", "tab_ppvp_tied_oracle"))
        for row in read_jsonl(Path(f"output/generation/{run}/{arm}/{experiment}/metadata.shard_00000-of-00001.jsonl"))
        if row["tab_id"].startswith("tab-pp-")]
Path(f"output/vqa/{run}").mkdir(parents=True, exist_ok=True)
write_trial_plan(Path(f"output/vqa/{run}/trial_plan.jsonl"), rows,
                 profile=pp_profile("tab-pp-transfer-74-five-seed"), with_source_metadata=True)
EOF
```

The trial plan has the conditions `sd14_vanilla` (Vanilla) and
`sd14_tied_oracle` (VS-TIED + Gold-visual setup); the Predicated Diffusion run
has `pd_spacy` and `pd_spacy_gold`.

**Sana.**

```bash
python -m scripts.tab_pp.run_baselines prepare --backbone sana --method vanilla gold_visual_setup --run-id $TSB
python -m scripts.tab_pp.run_baselines generate --backbone sana --method vanilla --run-id $TSB --device cuda:0
python -m scripts.tab_pp.run_baselines generate --backbone sana --method gold_visual_setup --run-id $TSB --device cuda:0
python -m scripts.tab_pp.run_baselines merge --backbone sana --run-id $TSB

python -m scripts.tab_pp.run_baselines prepare --backbone sana --method predicated_diffusion \
    gold_visual_setup_predicated_diffusion --run-id $TSP
python -m scripts.tab_pp.run_baselines generate --backbone sana --method predicated_diffusion --run-id $TSP \
    --device cuda:0
python -m scripts.tab_pp.run_baselines generate --backbone sana --method gold_visual_setup_predicated_diffusion \
    --run-id $TSP --device cuda:0
python -m scripts.tab_pp.run_baselines merge --backbone sana --run-id $TSP
```

The Sana trial plans have the conditions `vanilla` and `gold`, and `pd_spacy`
and `pd_spacy_gold`.

## Judging

Both judges see each trial as one image and one official TAB intention question
([evaluation.md](evaluation.md#tab-pp)). One GPT-6 Astra judgment run takes the
conditions of all generation runs, each under its own condition ID: the ten
conditions of the method comparison (the five methods Vanilla, Predicated
Diffusion, VS-TIED + Gold-visual setup, VS-TIED + Gold-visual setup + Predicated
Diffusion and CCG Diffusion in cell `c1r1p0m0`, on both backbones) and the
three ablation cells of each backbone, 16 conditions in all. The Qwen judge then
re-judges the ten method-comparison conditions of that run. The condition IDs
below are the ones `judge_qwen38` expects for them (`sd14_vanilla`, ...,
`sana_c1r1p0m0`); each `--condition` names `ID=RUN:SOURCE_CONDITION`, where
`SOURCE_CONDITION` is the condition of the generation run's trial plan (an SD1.4
CCG Diffusion run records `<run-id>:<cell>`, a Sana one `guided`).

```bash
C=()
for cell in c1r1p0m0 c0r1p0m0 c1r0p0m0 c0r0p0m0; do       # the full configuration and the three ablation cells
  t=$T; ts=$TS; [ $cell = c1r1p0m0 ] || { t=${T}_$cell; ts=${TS}_$cell; }
  C+=(--condition sd14_$cell=$t:$t:$cell --condition sana_$cell=$ts:guided)
done
C+=(--condition sd14_vanilla=$TB:sd14_vanilla --condition sd14_gold=$TB:sd14_tied_oracle
    --condition sd14_pd_spacy=$TP:pd_spacy --condition sd14_pd_spacy_gold=$TP:pd_spacy_gold
    --condition sana_vanilla=$TSB:vanilla --condition sana_gold=$TSB:gold
    --condition sana_pd_spacy=$TSP:pd_spacy --condition sana_pd_spacy_gold=$TSP:pd_spacy_gold)
python -m scripts.tab_pp.judge_astra prepare --run-id $J "${C[@]}"            # offline: seals the trials
python -m scripts.tab_pp.judge_astra preflight --output output/vqa/$J         # offline
python -m scripts.tab_pp.judge_astra run --output output/vqa/$J --budget-usd 100   # paid requests
python -m scripts.tab_pp.judge_astra summarize --output output/vqa/$J        # offline
```

`prepare` checks every image against its trial plan and writes
`output/vqa/$J/trial_plan.jsonl` and `preparation.json` (16 conditions x 740
trials). `run` needs `OPENAI_API_KEY`; it sends at most `--workers` (default
16) requests at a time, resumes from the stored responses and stops before the
spend could exceed `--budget-usd` (the 11,840 trials cost about 60 USD,
[evaluation.md](evaluation.md#gpt-6-astra)). `summarize` writes
`predictions_5seed.jsonl`, `summary_5seed.json` and `scores_5seed.csv` (Yes
count, trials and Yes rate per condition). To judge a single generation run
instead, `judge_astra prepare --generation-run RUN` takes the conditions of
`output/vqa/RUN/trial_plan.jsonl` under their own IDs into
`output/vqa/RUN/astra`.

```bash
A=()
for c in sd14_vanilla sd14_pd_spacy sd14_gold sd14_pd_spacy_gold sd14_c1r1p0m0 \
         sana_vanilla sana_pd_spacy sana_gold sana_pd_spacy_gold sana_c1r1p0m0; do
  A+=(--astra-run $c=$J)
done
python -m scripts.tab_pp.judge_qwen38 run --base-url http://<host>:8888 --run-id $Q --workers 10 "${A[@]}"
python -m scripts.tab_pp.judge_qwen38 summarize --run-id $Q
```

`judge_qwen38 run` takes the 7,400 trials of the ten method-comparison conditions from
`output/vqa/$J/predictions_5seed.jsonl` (`--astra-run CONDITION=RUN[:SOURCE_CONDITION]`
names, per condition, the GPT-6 Astra judgment directory under `output/vqa/`
and the condition ID used there; all ten are required), sends the same image and
question to the OpenAI-compatible server
([installation.md](installation.md#tab-pp-judges)), stores one raw response per
trial under `output/vqa/$Q/raw/` and resumes from them; `--limit N` judges only
N pending trials. `summarize` (offline) writes `predictions.jsonl` (each trial
with the Qwen answer and the GPT-6 Astra answer of the same trial),
`summary.json` and `scores.csv`. Cohen's kappa of the two judges is computed
from `predictions.jsonl` ([evaluation.md](evaluation.md#cohens-kappa)).

## Hardware and determinism

On an RTX 4090, SD1.4 and Sana TAB generation is bit-deterministic between
runs (PNG bytes, initial and final latents and the first loss value), provided
that every image is generated in its physical batch, which the commands above
preserve (`--batches` and `--keys` select whole batches). On another GPU type
(for example an A100) neither the images nor the first loss values equal those
of an RTX 4090. The frozen interpretations of `data/tab_pp/` were parsed on an
RTX 4090, so the frozen path does not depend on the parser's hardware. Run
one backbone per process: SD1.4 switches on deterministic algorithms for the
whole process (`config.SD14_DETERMINISM`), and Sana refuses to run where they
are on. The judges are not deterministic (GPT-6 Astra is a hosted model; the
Qwen judge samples at temperature 1.0), so a new judgment of the same images
reproduces the Yes rates up to judge variation. Generation times and judge
costs are listed in [installation.md](installation.md#hardware-and-determinism).
