# Evaluation

This page describes how generated images are scored: the official
T2I-CompBench++ evaluators, the two TAB PP judges and their agreement, the
statistical tests that compare methods and ablation conditions, and the
generation-time benchmark. The commands that generate the images are in
[t2i_compbench.md](t2i_compbench.md) and [tab_pp.md](tab_pp.md). Run IDs
below (`20261020_t2i_sd14`, ...) are placeholders of the form
`<YYYYMMDD>_<slug>`.

## T2I-CompBench++

Images are scored with the official T2I-CompBench++ evaluators at commit
`1b709499`, run unmodified from their clone and Python 3.10 venv
([installation.md](installation.md#t2i-compbench-evaluator)):

| Category | Metric | Official script |
|---|---|---|
| color, shape, texture | BLIP-VQA | `BLIPvqa_eval/BLIP_vqa.py` (8 passes) |
| 2D-spatial (`spatial`) | UniDet | `UniDet_eval/2D_spatial_eval.py` |
| 3D-spatial (`3d_spatial`) | UniDet with MiDaS depth | `UniDet_eval/3D_spatial_eval.py` |
| numeracy | UniDet | `UniDet_eval/numeracy_eval.py` |
| non-spatial (`non_spatial`) | CLIPScore | `CLIPScore_eval/CLIP_similarity.py` |
| complex | 3-in-1 | BLIP-VQA, then 2D-spatial and CLIPScore with `--complex True`, then `3_in_1_eval/3_in_1.py` |

```bash
python -m scripts.t2i_compbench.export_samples --run-id 20261020_t2i_sd14 --arm sd14_pd_holccg \
    --method sd14_pd_holccg --generation-out-dir output/generation/20261020_t2i_sd14/images
python -m scripts.t2i_compbench.evaluate --run-id 20261020_t2i_sd14 --arm sd14_pd_holccg --categories all
```

`export_samples` links every image of a category as `<prompt>_<index:06d>.png`
under `output/vqa/<run-id>/t2icb_work/<arm>/<category>/samples/` (image index =
prompt position x 10 + seed position) and seals the links with the image sha256
in `metric_stage_manifest.json`; the table in
[t2i_compbench.md](t2i_compbench.md#6-scoring) gives `--generation-out-dir`,
`--method` and `--arm` for each kind of run (`--arm` is the arm ID: a method or
an ablation condition, see the [glossary](provenance.md#glossary)). `evaluate`
([`evaluation/t2i_compbench/official_evaluator.py`](../src/ccg_diffusion/evaluation/t2i_compbench/official_evaluator.py))
checks the clone, its prompt files, the venv freeze, the four metric weights
and the image names before any GPU work, runs the official scripts through
`subprocess` with `PYTHONPATH` and `LD_LIBRARY_PATH` removed, and keeps
`vqa_result.json` (per-image scores), the metric's own average text and
`eval_manifest.json` under `output/vqa/<run-id>/t2i_compbench/<arm>/<category>/`.
The average printed by the official script is cross-checked against the mean of
the per-image scores (tolerance 1e-4), so a mis-parsed score fails the
category. `--skip-completed` leaves verified categories alone, `--dry-run`
prints the planned invocations. `t2icb_work/` is scratch space.

Without `--arm`, `evaluate` scores the official layout
`output/generation/<run-id>/<category>/samples/` (the SD1.4 Vanilla run of
[t2i_compbench.md](t2i_compbench.md#vanilla-on-stable-diffusion-v14)) against the official
prompt files and keeps the results under `output/vqa/<run-id>/t2i_compbench/<category>/`.
It also compares each category score with the published Stable Diffusion v1.4
scores of T2I-CompBench++ (`PUBLISHED_TARGETS` in `official_evaluator.py`) and
prints `target=... delta=... PASS` when they differ by at most 0.02
(`GATE_TOLERANCE`), `FAIL` otherwise, and a summary line `n_pass/n_gated within
+/-0.02`; the verdicts are kept in `eval_manifest.json` and `scores_summary.json`
and do not change the exit status. `--targets none` skips the comparison; runs
with `--arm` never apply it.

Notes on the official evaluators that the driver handles:

* The image name must contain exactly one underscore: `BLIP_vqa.py` reads the
  index as `split("_")[-1]`, `numeracy_eval.py` as `split("_")[1]`, and
  `3_in_1.py` assumes that image `i*10 + j` belongs to prompt `i`.
* `--complex` is declared with `type=bool`, so any value turns it on; the driver
  passes it only for the complex category.
* Six of the eight official prompt files have CRLF line endings; they are read
  in text mode, as the official loop does.

The score of a method or ablation condition on a category is the mean of its
3,000 per-image scores (`vqa_result.json`); the effect of an ablation condition
is its score minus that of the full configuration on the same category.
`evaluation.t2i_compbench.results.load_answers(run_id, arm, category)` reads a
`vqa_result.json` after checking its seal, its image population and its prompt
order, and `prompt_score_matrix` averages the ten seeds of each prompt, the
unit of every T2I test ([below](#statistical-tests)).

## TAB PP

One trial is one generated image and one official TAB intention question,
judged Yes or No by a VQA model. A method's Yes rate is the share of Yes over
its 740 trials (74 items x 2 interpretations x seeds 0-4). Each image of
CCG Diffusion and of VS-TIED + Gold-visual setup (with or without Predicated
Diffusion) is generated for one interpretation and judged with that
interpretation's question. Vanilla and Predicated Diffusion receive no
interpretation, so their single image per seed is judged with the questions of
both interpretations. The question sent is the official question followed by a
new line and `Answer with only Yes or No.`; the answer is the leading standalone
Yes or No, and any other answer is invalid and counts as not Yes
([`evaluation/tab.py`](../src/ccg_diffusion/evaluation/tab.py)). A generation
run writes its trials to `output/vqa/<run-id>/trial_plan.jsonl`
([tab_pp.md](tab_pp.md)).

### GPT-6 Astra

```bash
python -m scripts.tab_pp.judge_astra prepare --run-id 20261020_tab_astra --condition ID=RUN:SOURCE_CONDITION ...
python -m scripts.tab_pp.judge_astra prepare --generation-run RUN          # alternative: one generation run
python -m scripts.tab_pp.judge_astra preflight --output output/vqa/20261020_tab_astra
python -m scripts.tab_pp.judge_astra run --output output/vqa/20261020_tab_astra --budget-usd 100 [--workers 16]
python -m scripts.tab_pp.judge_astra summarize --output output/vqa/20261020_tab_astra
python -m scripts.tab_pp.judge_astra replay --output output/vqa/20261020_tab_astra --seed-count 5
```

[tab_pp.md](tab_pp.md#judging) gives the full `--condition` list of all 16
conditions (the ten method-comparison conditions and the six ablation cells).
The frozen protocol
([`evaluation/judges/astra.py`](../src/ccg_diffusion/evaluation/judges/astra.py),
`PROTOCOL`, sha256 `4bee59de5ff0ff4725f1a93ba7d12fd7a895b16cef62405b4e9df856f6503165`)
sends each trial as an independent Responses API request to `gpt-6-astra` with
reasoning effort `high`, the 512x512 PNG at original detail, at most 32,768
output tokens, no tools and `store=false`, and records the model version
returned for every response (`run` stops if it is not `gpt-6-astra`, unless
given `--allow-any-snapshot`). `prepare` (offline) checks every image against
its trial plan and seals the trials; the trials of all conditions are ordered
item by item, so that judge drift over time is not confounded with a condition.

Only `run` sends paid requests. It needs `OPENAI_API_KEY` and `--budget-usd`:
before each request it reserves the worst-case cost of one request (about
1.69 USD at the prices in `PRICING`), and it sends a request only while the
spend stored in the judgment directory plus the reserve of every in-flight
request stays within the budget. A trial costs about 0.005 USD (about 4 USD
per 740-trial condition), so a budget about 30 USD above the expected spend
lets 16 workers run in parallel. `run` resumes from the stored responses, never
resends a request whose response is missing (it stops and names the attempt to
inspect), stops at the first error after draining the requests in flight, and
pauses when a file `PAUSE` appears in the judgment directory; `--max-requests N`
sends at most N requests. `summarize` (offline) writes
`predictions_<n>seed.jsonl`, `summary_<n>seed.json` and `scores_<n>seed.csv`
(Yes count `yes`, `trials` and Yes rate `percent` per condition, under `scores`
in the summary, with an item-bootstrap interval
of 20,000 draws, `--bootstrap-seed`, default 20260923). `replay` re-derives the
predictions from the stored responses without network access and prints their
sha256 (`--expect-sha256 HEX` fails on another digest).

### Qwen3.8-Flash-Next

```bash
python -m scripts.tab_pp.judge_qwen38 run --base-url http://<host>:8888 --run-id 20261020_tab_qwen38 \
    --workers 10 --astra-run sd14_vanilla=20261020_tab_astra ...      # one --astra-run per method-comparison condition
python -m scripts.tab_pp.judge_qwen38 summarize --run-id 20261020_tab_qwen38
```

The open judge re-judges, unchanged (same images and questions as the GPT-6
Astra judgments), the 7,400 trials of the ten conditions of the TAB PP method
comparison: the five methods on both backbones
([tab_pp.md](tab_pp.md#judging)). It runs through an OpenAI-compatible
`/v1/chat/completions` endpoint
([`evaluation/judges/qwen38.py`](../src/ccg_diffusion/evaluation/judges/qwen38.py),
`PROTOCOL`): `nvidia/Qwen3.8-Flash-Next-NVFP4` at revision `fc694b54`, served by
vLLM 0.30.0 ([installation.md](installation.md#tab-pp-judges)) under the model
name `qwen3.8-flash-next` (`--served-model`), thinking enabled with reasoning
effort `xhigh` (`chat_template_kwargs`), the sampling recommended by the checkpoint (temperature
1.0, top-p 0.95, top-k 20), at most 32,768 tokens, the image before the
question. The answer is read from the message content (the reasoning is
excluded) with the same Yes/No rule; invalid or truncated answers stay in the
denominator.

`run` builds its trial plan from the `predictions_5seed.jsonl` of the GPT-6
Astra judgment named by `--astra-run CONDITION=RUN[:SOURCE_CONDITION]` for each
of the ten conditions `sd14_vanilla`, `sd14_pd_spacy`, `sd14_gold`,
`sd14_pd_spacy_gold`, `sd14_c1r1p0m0` and their `sana_` counterparts (`RUN` is a
judgment directory under `output/vqa/`, `SOURCE_CONDITION` the condition ID used
there, by default `CONDITION`); `--run-id` names the output directory
`output/vqa/<run-id>/` and is required with `--astra-run`. The ten conditions
need distinct GPT-6 Astra condition IDs, because the trial IDs digest them.
`run` stores one raw response per trial, resumes from them (`--limit N` judges
only N pending trials) and ends with `summarize`, which writes
`predictions.jsonl` (each trial with the Qwen answer `answered_yes` and the GPT-6
Astra answer `astra_answered_yes` of the same trial), `summary.json` (Yes rate
`rate_pct`, invalid and truncated answers and agreement with GPT-6 Astra per
condition) and `scores.csv`. With `--workers 10`, the two-node server of
[installation.md](installation.md#tab-pp-judges) judges the 7,400 trials in
about two hours.

### Cohen's kappa

Cohen's kappa between the two judges' Yes/No answers is computed over the 740
trials of each method-comparison condition, over the 3,700 trials of each
backbone and over all 7,400 trials, each with the share of agreeing trials.
`statistics.cohen_kappa(a, b)` returns the observed agreement
`po` and `(po - pe) / (1 - pe)`, where `pe` is the agreement expected from the
two judges' Yes rates; an invalid answer counts as No for both judges. From the
Qwen predictions:

```python
import json

from ccg_diffusion.evaluation.statistics import cohen_kappa

rows = [json.loads(line) for line in open("output/vqa/20261020_tab_qwen38/predictions.jsonl")]
groups = {"all": rows}
for row in rows:
    groups.setdefault(row["backbone"], []).append(row)    # recorded labels "SD1.4", "Sana 0.6B"
    groups.setdefault(row["condition"], []).append(row)
for name, trials in groups.items():
    agreement, kappa = cohen_kappa([r["astra_answered_yes"] for r in trials], [r["answered_yes"] for r in trials])
    print(f"{name:20} n={len(trials):5} agreement={100 * agreement:.1f}% kappa={kappa:.2f}")
```

## Statistical tests

Methods and ablation conditions are compared with permutation tests
implemented in
[`evaluation/statistics.py`](../src/ccg_diffusion/evaluation/statistics.py):

* **Unit.** The prompt for T2I-CompBench++ (300 per category; the score of a
  prompt is the mean of its ten seeds, `prompt_score_matrix`) and the item for
  TAB PP (74 items; the score of an item is the Yes rate of its 10 trials, 2
  interpretations x 5 seeds, `evaluation.tab.item_means`). Two methods are
  compared on the same units. `item_means` accepts a condition only when all
  740 of its trials have a valid answer, so a trial with an invalid or
  truncated answer is judged again before the tests: for the Qwen judge, delete
  its file `output/vqa/<run-id>/raw/<trial_id>.json` and run `judge_qwen38 run`
  again, which sends only the trials without a stored response and then
  summarizes.
* **Tests.** A contrast on the same units uses the two-sided paired sign-flip
  permutation test on the mean of the per-unit differences
  (`paired_permutation_test`): each permutation flips the sign of every
  difference independently. A contrast between two different prompt sets, for
  example the effect of a component in two categories, uses the two-sided label
  permutation test on the difference of the means (`label_permutation_test`).
  Both use 200,000 permutations; the p-value is `(k + 1) / (200,000 + 1)`, where
  `k` counts permutations whose statistic is at least as extreme as the
  observed one. Both also return a 95% percentile bootstrap interval over the
  units (20,000 draws).
* **Multiplicity.** p-values are Holm-corrected (`holm`) within each family:

  | Comparison | Contrast | Family (per backbone) |
  |---|---|---|
  | T2I-CompBench++ methods | the highest-mean method against every other method; CCG Diffusion against Predicated Diffusion | the 8 subcategories |
  | one attribute-binding component removed (`gs0_p1_r1`, `gs1_p0_r1`, `gs1_p1_r0`) | each condition minus the full configuration, on color, shape and texture | 9 (3 conditions x 3 categories) |
  | two or three attribute-binding components removed (`gs0_p0_r0`, `gs0_p0_r1`, `gs0_p1_r0`, `gs1_p0_r0`) | each condition minus the full configuration, on color, shape and texture | 12 (4 conditions x 3 categories) |
  | Spatial Relation Loss or Count Loss removed (`no_2d`, `no_count`) | each condition minus the full configuration, on `spatial` or `numeracy` | 2 |
  | TAB PP methods | as for the T2I-CompBench++ methods, for each judge | the 2 VQA models |
  | TAB PP ablation cells (`c0r1p0m0`, `c1r0p0m0`, `c0r0p0m0`) | each cell minus `c1r1p0m0` (GPT-6 Astra) | 3 |

  The highest-mean method is significantly best when its paired test against
  each other method is significant: the largest of those p-values,
  Holm-corrected over the family, is below 0.05. CCG Diffusion is
  significantly better than Predicated Diffusion when its mean is higher and
  the Holm-corrected p is below 0.05. `significance_mark` turns a
  Holm-corrected p-value into `***` (p < 0.001), `**` (p < 0.01), `*`
  (p < 0.05) or `n.s.`.
* **Random numbers.** The tests take an explicit generator; the examples below
  create `numpy.random.default_rng(SIGNIFICANCE_SEED)` (`SIGNIFICANCE_SEED =
  20260927`) once per group of tests. The p-values depend on the order in which
  the tests draw from it: with the same scores and the same order every p-value
  is reproduced exactly, and another order changes the p-values only by Monte
  Carlo error (about 0.0005 at p = 0.05).

Example: CCG Diffusion against Predicated Diffusion on T2I-CompBench++ with
SD1.4 (Holm over the eight subcategories), from the scores of
[t2i_compbench.md](t2i_compbench.md#6-scoring):

```python
import numpy as np

from ccg_diffusion.evaluation.statistics import SIGNIFICANCE_SEED, holm, paired_permutation_test
from ccg_diffusion.evaluation.t2i_compbench.results import load_answers, prompt_score_matrix

categories = ["color", "shape", "texture", "spatial", "3d_spatial", "numeracy", "non_spatial", "complex"]
rng = np.random.default_rng(SIGNIFICANCE_SEED)
tests = []
for c in categories:
    ours = prompt_score_matrix(load_answers("20261020_t2i_sd14", "sd14_pd_holccg", c))
    pd = prompt_score_matrix(load_answers("20261020_t2i_sd14_baselines", "sd14_pd_spacy", c))
    tests.append(paired_permutation_test(ours - pd, rng))
for c, test, p in zip(categories, tests, holm([t["p"] for t in tests])):
    print(c, round(test["delta"], 4), "higher, p < 0.05" if test["delta"] > 0 and p < 0.05 else "")
```

and the same contrast on TAB PP (Holm over the two judges of a backbone):

```python
import json

import numpy as np

from ccg_diffusion.evaluation.statistics import SIGNIFICANCE_SEED, holm, paired_permutation_test
from ccg_diffusion.evaluation.tab import item_means


def rows(path):
    return [json.loads(line) for line in open(path)]


judges = [rows("output/vqa/20261020_tab_astra/predictions_5seed.jsonl"),
          rows("output/vqa/20261020_tab_qwen38/predictions.jsonl")]
rng = np.random.default_rng(SIGNIFICANCE_SEED)
for backbone in ("sd14", "sana"):
    tests = [paired_permutation_test(item_means(p, f"{backbone}_c1r1p0m0")[1]
                                     - item_means(p, f"{backbone}_pd_spacy")[1], rng) for p in judges]
    print(backbone, [round(100 * t["delta"], 1) for t in tests], [p < 0.05 for p in holm([t["p"] for t in tests])])
```

## Generation-time benchmark

`scripts.paper.compute_cost` times Vanilla (`vanilla`), Predicated Diffusion
(`spacy_pd`) and CCG Diffusion (`ours`) on each backbone, from the prompt
string to PIL images in host memory: parsing, guidance targets and plans, text
encoding, denoising and VAE decoding are timed; model loading, PNG writing and
verification are not. The prompts are the 30 per category (240 in all) of
[`data/compute_cost/prompts.json`](../data/compute_cost/prompts.json), after the
warm-up prompts of `data/compute_cost/warmup.json`
([data.md](data.md#timing-prompts)); mode `b1` generates one image (seed 42) per
prompt, the methods rotate per prompt, and a GPU guard waits until no other
process computes on the measured GPU. `--no-verify` skips the comparison of
every result with the outputs of earlier runs under `--reference-root`, which a
fresh clone does not have.

```bash
export CUDA_VISIBLE_DEVICES=GPU-xxxxxxxx-...        # the UUID of one GPU
python -m scripts.paper.compute_cost benchmark --backbone sd14 --out output/analysis/20261020_compute_cost \
    --modes b1 --no-verify
python -m scripts.paper.compute_cost benchmark --backbone sana --out output/analysis/20261020_compute_cost \
    --modes b1 --no-verify
python -m scripts.paper.compute_cost summarize output/analysis/20261020_compute_cost
```

In `DIR/summary.json` (`DIR` is the `--out` directory),
`<backbone>.methods.b1.<method>.s_per_image` is the generation time per image
of a method (`vanilla`, `spacy_pd` or `ours`), in seconds;
`<backbone>.comparisons.b1.ours_vs_vanilla.ratio` and
`ours_vs_spacy_pd.ratio` are the time ratios of CCG Diffusion over Vanilla and
over Predicated Diffusion (the overhead over Predicated Diffusion is
`100 x (ratio - 1)` percent);
`<backbone>.methods.b1.<method>.pre_generation_s` gives the extra parsing time
(its `mean` for `ours` minus that for `spacy_pd`) and the share of parsing in
the whole run time (its `share_of_total` for `ours`).
