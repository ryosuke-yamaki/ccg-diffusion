# T2I-CompBench++

This page generates the images of CCG Diffusion, of its component ablations and
of the compared methods on the 2,400 test prompts of T2I-CompBench++ with the
ten seeds 42-51, i.e. 24,000 images per method ([data.md](data.md#t2i-compbench)),
and scores them with the official evaluators. Setup is in
[installation.md](installation.md), the metrics in
[evaluation.md](evaluation.md#t2i-compbench) and the statistical tests in
[evaluation.md](evaluation.md#statistical-tests).

Every command runs from the repository root. A run ID has the form
`<YYYYMMDD>_<slug>`; it derives every output path
(`output/generation/<run-id>/` for the images, `output/vqa/<run-id>/` for the
scores). The examples use these run IDs:

```bash
R=20261020_t2i_sd14              # SD1.4: Stages 1-3, CCG Diffusion and its ablations
RS=20261020_t2i_sana             # Sana: Vanilla, CCG Diffusion and its ablations
RB=20261020_t2i_sd14_baselines   # SD1.4: the five compared methods
RSB=20261020_t2i_sana_baselines  # Sana: the five compared methods
RV=20261020_t2i_sd14_vanilla     # SD1.4: Vanilla
S=output/generation/$R/stage123  # Stage 1-3 files shared by all runs
```

## Methods and modules

| Method | Stable Diffusion v1.4 | Sana 0.6B |
|---|---|---|
| Vanilla | [`scripts/t2i_compbench/official_reference/`](../scripts/t2i_compbench/official_reference/generate.py) | `compositional/sana.py` (`--method vanilla`) |
| Composable Diffusion | `baselines/composable_diffusion.py`, run by `baselines/sd14_generation.py` | `baselines/sana_plans.py`, `baselines/sana_generation.py` |
| StructureDiffusion | `baselines/structure_diffusion.py`, `baselines/sd14_generation.py` | as above |
| Attend-and-Excite | `baselines/attend_and_excite.py`, `baselines/sd14_generation.py` | as above |
| SynGen | `baselines/syngen.py`, `baselines/sd14_generation.py` | as above |
| Predicated Diffusion | `baselines/predicated_diffusion.py`, SD1.4 batch driver `backbones/sd14/driver.py` | as above |
| CCG Diffusion | `compositional/plans.py`, `compositional/sd14.py` | `compositional/plans.py`, `compositional/sana.py` |

Module paths are under `src/ccg_diffusion/`. The SD1.4 runs of
`run_ccg_diffusion` and `run_baselines` use `config.SD14_GENERATION` (PNDM 50
steps, CFG 7.5, 512x512, BF16 model with FP32 master latents, physical batches
of 4), and their Sana runs use `config.SANA_GENERATION` (20 steps, CFG 4.5,
512x512, FP16 model with FP32 master latents, batches of up to 4 seeds of one
prompt); [method.md](method.md) lists the hyperparameters, and
`python -m ccg_diffusion.methods table8` prints the generation settings and
hyperparameters of both backbones as JSON.

## 1. Prompts and guidance targets (Stages 1-3, CPU)

```bash
python -m scripts.t2i_compbench.prepare_prompts --stage-dir $S
for c in color shape texture spatial 3d_spatial numeracy non_spatial complex; do
  python -m scripts.ccg.build_guidance_targets --prompt-only $S/$c.prompt_only.jsonl --auto-path $S/$c.auto \
    --sentence-map-path $S/$c.sentence_map.jsonl --markedup-deps-path $S/generate/$c.markedup_deps \
    --coverage-rows-path $S/generate/$c.coverage_rows.jsonl --output $S/$c.deps2index_sd14.jsonl
done
```

`prepare_prompts` writes the per-category prompt files and parse corpora,
unpacks the frozen Stage 1-2 outputs (Hol-CCG derivations and C&C marked
dependencies of the 2,400 prompts) from
[`data/t2i_compbench/`](../data/t2i_compbench/README.md), and writes the source
manifest `$S/source_manifest.json` that every `prepare` below reads; it prints
the eight Stage 3 commands of the loop above. With the frozen inputs Stages 1-3
run on the CPU and give the same guidance targets on every machine. To parse
again instead, pass `--parse-inputs parse` and run the printed Stage 1 (GPU) and
Stage 2 commands before Stage 3; near-tie supertags can then differ from the
frozen derivations on other hardware ([parsing.md](parsing.md)).

## 2. CCG Diffusion on Stable Diffusion v1.4

```bash
python -m scripts.t2i_compbench.run_ccg_diffusion prepare --backbone sd14 --run-id $R \
    --source-manifest $S/source_manifest.json
for k in 0 1 2 3; do       # one GPU per shard; the shards can run in parallel (below)
  python -m scripts.t2i_compbench.run_ccg_diffusion generate --backbone sd14 --run-id $R --shard-index $k --device cuda:0
done
python -m scripts.t2i_compbench.run_ccg_diffusion audit --backbone sd14 --run-id $R
```

`prepare` (CPU) derives the plan of every prompt from the Stage 1-3 files,
writes `output/generation/$R/inputs/image_guidance_targets.sd14.jsonl` (the
guidance targets of the 24,000 images), freezes `protocol.json` and
`inputs/decisions.jsonl`, and prints digests of the per-prompt guidance
decisions ([below](#checking-the-preparation-without-a-gpu) gives their
expected values). `generate` runs one of four shards (image `i` of the
seed manifest goes to shard `i % 4`; four consecutive images of a shard form a
physical batch) and writes
`output/generation/$R/images/sd14_pd_holccg/t2icb_<category>/.../seed_000NN.png`
with one metadata row per image. `audit` checks the finished images against the
protocol.

**Sources.** `prepare` seals the implementation files to the Git commit and
stops when they have uncommitted changes, and `generate` requires the same
commit to be checked out; `--allow-dirty` on `prepare` and `generate` seals the
bytes of the working tree instead.

**Software pins.** SD1.4 generation (`run_ccg_diffusion generate --backbone
sd14` and `run_baselines --backbone sd14`) stops unless it runs on the software
of the runtime container, `SD14_GENERATOR_RUNTIME_PINS` in
[`backbones/sd14/records.py`](../src/ccg_diffusion/backbones/sd14/records.py):
Python 3.12.3, torch 2.13.0+cu129, diffusers 0.39.0, transformers 5.14.1, CUDA
runtime 12.9 and cuDNN `92000`. Elsewhere, `--no-strict-runtime` records the
versions with `pins_enforced: false` instead of enforcing them; the NVIDIA
driver version is always only recorded.

**Subsets.** `--batches 165,188` generates only those physical batches of the
shard, and `--output-dir DIR` writes them outside the run (`audit` takes the
same `--batches` and `--output-dir`). Rows `16c` to `16c+15` of the seed
manifest (chunk `c`) are batch `c` of every shard, so
`--batches c --shard-index 0..3` generates one chunk of 16 images with their
batch peers. The manifest lists the categories in the order color, shape,
texture, 2D-spatial, 3D-spatial, numeracy, non-spatial, complex, each as 300
prompts x seeds 42-51, so the image of prompt `p` (0-299) of category `k`
(0 for color, ..., 7 for complex) with seed `s` is row `3000k + 10p + (s - 42)`.

## 3. CCG Diffusion on Sana 0.6B

```bash
python -m scripts.t2i_compbench.run_ccg_diffusion prepare --backbone sana --run-id $RS \
    --source-manifest $S/source_manifest.json
python -m scripts.t2i_compbench.run_ccg_diffusion generate --backbone sana --run-id $RS --device cuda:0
python -m scripts.t2i_compbench.run_ccg_diffusion audit --backbone sana --run-id $RS
```

`prepare` (CPU) writes `final_shape_protocol.json` and
`final_shape_protocol.plans.jsonl` (the plans projected onto Sana's Gemma
tokens) and prints the sha256 of the plans; `generate` writes the images under
`output/generation/$RS/final_shape/` (seeds 42-45, 46-49 and 50-51 of a prompt
in batches of 4, 4 and 2). Sana runs are not sharded; `--keys
t2icb_color_0265,t2icb_shape_0002` generates all ten seeds of those prompts
(the same `--keys` goes to `audit`).

Vanilla on Sana uses `--method vanilla` on the same three commands; it
needs no `--source-manifest` and writes `vanilla_protocol.json` and
`output/generation/$RS/vanilla/`:

```bash
python -m scripts.t2i_compbench.run_ccg_diffusion prepare --backbone sana --run-id $RS --method vanilla
python -m scripts.t2i_compbench.run_ccg_diffusion generate --backbone sana --run-id $RS --method vanilla --device cuda:0
python -m scripts.t2i_compbench.run_ccg_diffusion audit --backbone sana --run-id $RS --method vanilla
```

## 4. Component ablations

Each ablation removes components with `--remove`; the same `--remove` goes to
`prepare`, `generate` and `audit`. The condition ID is derived from the removed
components, and each condition is scored on the categories listed for it:

| Removed | `--remove` | Condition | Scored categories |
|---|---|---|---|
| Selective Guidance Boost | `selective_guidance_boost` | `gs0_p1_r1` | color, shape, texture |
| Positional Prefix | `positional_prefix` | `gs1_p0_r1` | color, shape, texture |
| Region Assignment Loss | `region_assignment_loss` | `gs1_p1_r0` | color, shape, texture |
| all three | `selective_guidance_boost positional_prefix region_assignment_loss` | `gs0_p0_r0` | color, shape, texture |
| Boost and Prefix | `selective_guidance_boost positional_prefix` | `gs0_p0_r1` | color, shape, texture |
| Boost and Region Assignment Loss | `selective_guidance_boost region_assignment_loss` | `gs0_p1_r0` | color, shape, texture |
| Prefix and Region Assignment Loss | `positional_prefix region_assignment_loss` | `gs1_p0_r0` | color, shape, texture |
| Spatial Relation Loss | `spatial_relation_loss` | `no_2d` | 2D-spatial (`spatial`) |
| Count Loss | `count_loss` | `no_count` | numeracy |

The first seven rows are the attribute-binding ablations:
`gs<boost>_p<prefix>_r<region>` marks Selective Guidance Boost, Positional
Prefix and Region Assignment Loss with 1 (on) or 0 (off), so the full
configuration is `gs1_p1_r1`. The full configuration that every condition is
compared with is the CCG Diffusion run on the same backbone
([SD1.4](#2-ccg-diffusion-on-stable-diffusion-v14), [Sana](#3-ccg-diffusion-on-sana-06b)).
`python -m ccg_diffusion.methods ablations` lists the conditions with their
`--remove` names and scored categories. Plans never depend on the switches:
every condition uses the plans of the full method and changes only what runs
([`compositional/ablation.py`](../src/ccg_diffusion/compositional/ablation.py)).

**SD1.4.** A condition writes under `output/generation/$R/<condition>/` and
generates only the whole 16-image chunks that contain its scored categories,
so every image keeps its batch peers; `--batches` then counts batches within
those chunks.

```bash
python -m scripts.t2i_compbench.run_ccg_diffusion prepare --backbone sd14 --run-id $R \
    --source-manifest $S/source_manifest.json --remove positional_prefix
for k in 0 1 2 3; do
  python -m scripts.t2i_compbench.run_ccg_diffusion generate --backbone sd14 --run-id $R --shard-index $k \
      --device cuda:0 --remove positional_prefix
done
python -m scripts.t2i_compbench.run_ccg_diffusion audit --backbone sd14 --run-id $R --remove positional_prefix
```

**Sana.** A condition writes `<condition>_protocol.json` and
`output/generation/$RS/<condition>/`. Without `--keys` it generates all 24,000
images; to generate only the scored categories, pass their prompt IDs:

```bash
KEYS=$(python -c "print(','.join(f't2icb_{c}_{i:04d}' for c in ('color', 'shape', 'texture') for i in range(300)))")
python -m scripts.t2i_compbench.run_ccg_diffusion prepare --backbone sana --run-id $RS \
    --source-manifest $S/source_manifest.json --remove positional_prefix
python -m scripts.t2i_compbench.run_ccg_diffusion generate --backbone sana --run-id $RS --device cuda:0 \
    --remove positional_prefix --keys "$KEYS"
python -m scripts.t2i_compbench.run_ccg_diffusion audit --backbone sana --run-id $RS \
    --remove positional_prefix --keys "$KEYS"
```

For `no_2d` use the prompt IDs `t2icb_spatial_0000` to `t2icb_spatial_0299`,
for `no_count` `t2icb_numeracy_0000` to `t2icb_numeracy_0299`. A Sana output
directory keeps the prompt selection of its first `generate`.

## 5. Compared methods

### Stable Diffusion v1.4

```bash
# 1. blind preprocessing of Composable Diffusion, StructureDiffusion, Attend-and-Excite and SynGen
#    (Python 3.11 adapter environment, installation.md)
env -u LD_LIBRARY_PATH PYTHONPATH=src:. external/baseline_env/.venv/bin/python \
    -m scripts.t2i_compbench.prepare_baseline_inputs external --input-dir output/generation/$RB/input \
    --no-strict-environment
# 2. Attend-and-Excite subtoken selection (main environment, one GPU)
python -m scripts.t2i_compbench.prepare_baseline_inputs select-attend-and-excite \
    --input-dir output/generation/$RB/input --no-strict-environment
# 3. Predicated Diffusion guidance targets (spaCy en_core_web_sm 3.8.0, CPU)
python -m scripts.t2i_compbench.prepare_baseline_inputs predicated-diffusion --input-dir output/generation/$RB/input
# 4. generation: four shards per method, one GPU per shard
for m in composable_diffusion structure_diffusion attend_and_excite syngen predicated_diffusion; do
  for k in 0 1 2 3; do
    python -m scripts.t2i_compbench.run_baselines --backbone sd14 --method $m --run-id $RB \
        --num-shards 4 --shard-index $k --device cuda:0 --no-strict-environment
  done
done
```

`run_baselines --backbone sd14` reads `output/generation/$RB/input/` and writes
`output/generation/$RB/<arm>/`, one directory per method named by its arm ID
([glossary](provenance.md#glossary)): `sd14_composable`, `sd14_structure`,
`sd14_attend_excite`, `sd14_syngen`, `sd14_pd_spacy`. `--plan-only` prints the
batch membership of a shard without a GPU, `--batches` and `--output-dir`
generate a subset, and `--resume` continues an interrupted shard. The parser
installation digests pinned in the code depend on the absolute path of the
adapter venv, so inputs built in a new venv need `--no-strict-environment`
(versions, processors and model names are still compared,
[installation.md](installation.md#baseline-adapter-environment)).
`select-attend-and-excite` has no `--allow-dirty`: its output records its source
files sealed to the Git commit, and generation requires that seal.

Settings of `run_baselines --backbone sd14`: `CompVis/stable-diffusion-v1-4`
@ `133a221b`, PNDM 50 steps, CFG 7.5, 512x512, BF16 model with FP32 master
latents, scheduler and losses, physical batches of 4 (padded by repeating the
last real image; padded outputs discarded), guided and unguided images never
share a batch, seeds 42-51.

| Method | Inputs and settings |
|---|---|
| Composable Diffusion | prompt split on the word *and* (case-insensitive); weight 7.5 per component; a prompt without *and* is ordinary CFG |
| StructureDiffusion | Stanza 1.4.2 constituency parse of the normalized prompt; all noun phrases with at least two leaves; conjunction mode when a covering noun phrase coordinates noun phrases with *and*; no noun phrase gives an unguided image |
| Attend-and-Excite | every spaCy 3.6.1 `en_core_web_trf` NOUN (7,313 nouns, 327 of several CLIP tokens); for a noun of several tokens, the token with the highest mean over seeds 42-51 of the smoothed attention maximum at the first timestep; step `20 * sqrt(linspace(1, 0.5, 50)[t])` in the first 25 steps; thresholds {0: 0.05, 10: 0.5, 20: 0.8}; at most 20 refinements; no update cap |
| SynGen | the official spaCy extractors on `en_core_web_trf` (`include_entities=False`); step 20 in the first 25 steps; prompts without a binding pair are unguided (10,310 of 24,000 images) |
| Predicated Diffusion | spaCy 3.8.14 `en_core_web_sm` 3.8.0 with the hand-written rules of the original Predicated Diffusion paper (`baselines/predicated_diffusion.py`: every `NOUN`/`PROPN` is an entity, `amod`/`nmod`/`compound`/`acomp` modifiers bind to their noun, the subject and direct object of *have*/*wear*/*grasp*/*hold* form a possession); scale 20, cap 0.02, first 25 steps, 5 extra updates at step 0; prompts whose targets do not align with CLIP tokens are unguided (270 of 24,000 images) |

### Sana 0.6B

```bash
python -m scripts.t2i_compbench.prepare_baseline_inputs sana --run-id $RSB --input-dir output/generation/$RB/input
for m in composable_diffusion structure_diffusion attend_and_excite syngen predicated_diffusion; do
  python -m scripts.t2i_compbench.run_baselines --backbone sana --method $m --run-id $RSB --device cuda:0
  python -m scripts.t2i_compbench.audit_sana_baselines --run-id $RSB --method $m --reference-run $RS
done
```

None of the five methods has a Sana implementation; each is adapted
([`baselines/sana_plans.py`](../src/ccg_diffusion/baselines/sana_plans.py)):
the SD1.4 preprocessing of steps 1-3 above is reused (`--input-dir` points at
it) and every target is rebuilt on Sana's Gemma tokens, with the generation
settings of CCG Diffusion on Sana. `prepare_baseline_inputs sana` (CPU) writes
`output/generation/$RSB/protocol.json` and `protocol.plans.jsonl`. `run_baselines
--backbone sana` generates `output/generation/$RSB/<arm>/<category>/` one category
after the other (`--categories color shape` selects categories, `--keys`
prompts); arm IDs `sana_composable`, `sana_structure`, `sana_attend_excite`,
`sana_syngen`, `sana_pd_spacy`. `audit_sana_baselines` checks finished
categories (`--categories` selects them, default all eight): every image of the
method must have a finite metadata row matching its prompt, seed, model, plan
and protocol, and must start from the same initial latent as the image of the
same prompt and seed in the Sana CCG Diffusion run named by `--reference-run`,
so that run ([CCG Diffusion on Sana 0.6B](#3-ccg-diffusion-on-sana-06b)) must be
complete for the audited categories.

### Vanilla on Stable Diffusion v1.4

```bash
HF_HUB_OFFLINE=0 hf download CompVis/stable-diffusion-v1-4
bash scripts/t2i_compbench/official_reference/run_all_categories.sh $RV color shape texture spatial \
    3d_spatial non_spatial numeracy complex
python -m scripts.t2i_compbench.evaluate --run-id $RV --categories all
```

`run_all_categories.sh` requires the evaluator installation of
`scripts.setup.setup_t2i_compbench`
([installation.md](installation.md#t2i-compbench-evaluator)) and writes the
images to `output/generation/$RV/<category>/samples/`, which `evaluate` without
`--arm` scores ([evaluation.md](evaluation.md#t2i-compbench)) and compares with
the published Stable Diffusion v1.4 scores of T2I-CompBench++. The official
generator (`official_reference/generate.py`, run in the evaluator venv) loads
`CompVis/stable-diffusion-v1-4` at the default revision of the Hugging Face
repository in fp16, as the official script does, so it needs this download in
addition to the pinned revision that `download_models` fetches. It generates
whole categories; the image of prompt `p` (0-299 in the order of the official
test file) with seed `s` is `<prompt>_<10p + (s - 42):06d>.png`.

## 6. Scoring

```bash
python -m scripts.t2i_compbench.export_samples --run-id $R --arm sd14_pd_holccg --method sd14_pd_holccg \
    --generation-out-dir output/generation/$R/images
python -m scripts.t2i_compbench.evaluate --run-id $R --arm sd14_pd_holccg --categories all
```

`export_samples` links the images of one arm into the official `samples/`
layout under `output/vqa/<run-id>/t2icb_work/<arm>/`, and `evaluate --arm`
scores them into `output/vqa/<run-id>/t2i_compbench/<arm>/<category>/`. The
arguments for each kind of run:

| Images | `--run-id` | `--generation-out-dir` | `--method` | `--arm` | `--categories` |
|---|---|---|---|---|---|
| SD1.4 CCG Diffusion | `$R` | `output/generation/$R/images` | `sd14_pd_holccg` | `sd14_pd_holccg` | all |
| SD1.4 ablation `C` | `$R` | `output/generation/$R/C/images` | `sd14_pd_holccg` | `C` | its scored categories |
| SD1.4 compared method, arm `A` | `$RB` | `output/generation/$RB` | `A` | `A` | all |
| Sana CCG Diffusion | `$RS` | `output/generation/$RS/final_shape` | `sana_ccg_compositional_shape_v1` | `sana_ccg_compositional_shape_v1` | all |
| Sana ablation `C` | `$RS` | `output/generation/$RS/C` | `sana_ccg_compositional_ablation_v1` | `C` | its scored categories |
| Sana Vanilla | `$RS` | `output/generation/$RS/vanilla` | `sana_baseline` | `sana_baseline` | all |
| Sana compared method, arm `A`, category `c` | `$RSB` | `output/generation/$RSB/A/c` | `A` | `A` | `c` (one call per category) |

`--categories` takes category names (`color shape texture`); `all` above means
leaving it out of `export_samples` and passing `--categories all` to
`evaluate`. The SD1.4 Vanilla run needs no export
([Vanilla on Stable Diffusion v1.4](#vanilla-on-stable-diffusion-v14)).
[evaluation.md](evaluation.md#t2i-compbench) describes the metrics and the
result files.

## Several GPUs

There is no multi-GPU scheduler; run one process per GPU, selected by UUID
([installation.md](installation.md#gpus)), e.g. the four SD1.4 shards in
parallel:

```bash
GPU_UUIDS=($(nvidia-smi --query-gpu=uuid --format=csv,noheader))   # pick four compute GPUs
for k in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=${GPU_UUIDS[$k]} nohup python -m scripts.t2i_compbench.run_ccg_diffusion generate \
    --backbone sd14 --run-id $R --shard-index $k --device cuda:0 > shard$k.log 2>&1 &
done
```

The same applies to the shards of the SD1.4 compared methods, and to one
category per GPU for the Sana compared methods (`--categories`). A Sana
`generate` that is interrupted resumes from its metadata; the remaining images
of a prompt are regrouped into new batches, so a resumed run is not guaranteed
to be byte-identical to an uninterrupted one.
Generation times and the determinism of each backbone and GPU type are listed
in [installation.md](installation.md#hardware-and-determinism).

## Checking the preparation without a GPU

The steps of [Prompts and guidance targets](#1-prompts-and-guidance-targets-stages-1-3-cpu),
[CCG Diffusion on Stable Diffusion v1.4](#2-ccg-diffusion-on-stable-diffusion-v14) and
[CCG Diffusion on Sana 0.6B](#3-ccg-diffusion-on-sana-06b), up to each `prepare`,
need no GPU, only the cached SD1.4 and Gemma tokenizers
(`python -m scripts.setup.download_models`,
[installation.md](installation.md#models)), and give the same values on every
machine. The test suite (`python -m pytest -q`) compares the plans of selected
prompts with [`tests/fixtures/t2i_frozen_plans.json`](../tests/fixtures/t2i_frozen_plans.json);
the preparation of all 2,400 prompts on both backbones is checked with the run
IDs defined at the top of this page:

```bash
python -m scripts.t2i_compbench.prepare_prompts --stage-dir $S
for c in color shape texture spatial 3d_spatial numeracy non_spatial complex; do
  python -m scripts.ccg.build_guidance_targets --prompt-only $S/$c.prompt_only.jsonl --auto-path $S/$c.auto \
    --sentence-map-path $S/$c.sentence_map.jsonl --markedup-deps-path $S/generate/$c.markedup_deps \
    --coverage-rows-path $S/generate/$c.coverage_rows.jsonl --output $S/$c.deps2index_sd14.jsonl
done
python -m scripts.t2i_compbench.run_ccg_diffusion prepare --backbone sd14 --run-id $R --source-manifest $S/source_manifest.json
sha256sum output/generation/$R/inputs/image_guidance_targets.sd14.jsonl
python -m scripts.t2i_compbench.run_ccg_diffusion prepare --backbone sana --run-id $RS --source-manifest $S/source_manifest.json
```

| Value | sha256 |
|---|---|
| `per_prompt_guidance_sha256`, printed by `prepare --backbone sd14` | `e85b52aedaac92b9c86d182be407f4715b615eb6b0276363e07dc6e39a98477b` |
| `per_prompt_count_guidance_sha256`, printed by `prepare --backbone sd14` | `96ae56fa0f5a0f7bd816ae7b09a0b338066f0174304057e2f7e2ea41bd3aea4f` |
| `output/generation/$R/inputs/image_guidance_targets.sd14.jsonl` | `64e6900fc6763f269b082725a947c2b02b62cdfcb018b35c4a1c3b9bdd1dbac2` |
| `final_shape_protocol.plans.jsonl`, printed by `prepare --backbone sana` (`plans.sha256`) | `27ad7fcf43a67b89fce5208b7b6ad2e8a252f174e7c5e8a5522b7c09fa1343ec` |

The two printed SD1.4 values digest the Selective Guidance Boost and Count Loss
decisions of every prompt, the file digest covers the guidance targets of every
image, and the Sana value covers the complete plans (guidance targets on Gemma
tokens, Positional Prefix and Region Assignment Loss groups, Count Loss
constraints); the same runs can then be generated on a GPU. `prepare` seals its
source files to the Git commit and stops when they have uncommitted changes; to
check a modified checkout, add `--allow-dirty` to each `prepare`.
