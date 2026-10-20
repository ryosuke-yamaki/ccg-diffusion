# Frozen TAB PP inputs (`data/tab_pp/`)

The inputs from which the TAB PP jobs of CCG Diffusion on Stable Diffusion
v1.4 and Sana 0.6B (the full configuration, cell `c1r1p0m0`, and the ablation
cells `c0r1p0m0`, `c1r0p0m0` and `c0r0p0m0`) are compiled, frozen so that a
clone reproduces those jobs exactly, on the CPU (only the tokenizers are
loaded). The bundle is provided frozen; it is not rebuilt by any command of
this repository.

```bash
python -m scripts.tab_pp.run_ccg_diffusion prepare --backbone sd14 --run-id 20261020_tab_sd14
python -m scripts.tab_pp.run_ccg_diffusion preflight --backbone sd14 --run-id 20261020_tab_sd14
```

`prepare` (default `--inputs frozen`) compiles the jobs of cell `c1r1p0m0`
from this bundle (`--remove` selects the ablation cells), and `preflight`
compares them with `reference.json` and prints `PREFLIGHT_PASS` when they
equal the reference jobs apart from run identity. The same holds for
`--backbone sana`. The clean path,
`python -m scripts.tab_pp.parse_interpretations`, derives the interpretations
from the prompts with Supertag-Constrained Parsing; its
`--compare-frozen data/tab_pp` compares the result with this bundle
([docs/tab_pp.md](../../docs/tab_pp.md)).

## Files

| File | Content |
|---|---|
| `manifest.json` | `ccg_diffusion.tab_pp_bundle.v1`: per file the sha256 of the uncompressed bytes, of the gzip bytes and the row count, and the sha256 of the files the bundle was built from |
| `source_jobs.jsonl.gz` | The 3,150 source jobs in layout order: the 315 TAB items in dataset order x interpretations `np_attach`/`vp_attach` x seeds 0-4, without run identity (`arm`, `run_id`, `condition`, `image_id`) |
| `interpretations.jsonl.gz` | The 148 interpretations of the 74 PP items (dataset order, `np_attach` then `vp_attach`): `preposition` (the ambiguous preposition and the deciding tier), `paraphrase` (the Disambiguating Paraphrase trace), `guidance_targets` (prompt, CLIP alignment, the Stage 3 guidance targets, Selective Guidance Boost decision, `next_to` relations, digest of the source record), `standard_program` (the guidance targets that the jobs use) |
| `reference.json` | The digests that `preflight` compares with: per cell the SD1.4 job rows without run identity (all rows, each PP physical batch of 4, the VP rows) and their census, the Sana plans and jobs with their census and per-plan digests, and the inputs of the TAB comparison methods (SD1.4 Vanilla and Gold-visual setup manifests, Predicated Diffusion artifacts, Sana Vanilla and Gold-visual setup jobs and conditioning) |

The `.jsonl.gz` files are compressed with gzip level 9 and `mtime=0`; each line
is canonical JSON (sorted keys, compact separators).

## Verification

`ccg_diffusion.pp_attachment.jobs.load_frozen_bundle(verify=True)`, which
`prepare` calls, checks both sha256 and the row count of every file and the
sha256 of `reference.json` against `manifest.json`. An SD1.4 TAB plan seals
`manifest.json` together with its source files, and `generate` re-verifies the
seal before each batch. `python -m pytest -q tests/test_frozen_pp_attachment_plans.py`
compiles every cell from the bundle and compares the jobs and Sana
plans with `reference.json` on the CPU.

## The VP rows

PP jobs occupy positions 0-739 of the 3,150-job layout, so the physical
batches of 4 that contain them (batches 0-184) contain no VP job; the 2,410 VP
rows are never generated. They are kept so that a prepared job file has the
layout of the reference row for row (only run identity differs) and every PP
image keeps its batch of 4 ([docs/data.md](../../docs/data.md#the-3150-job-layout)).
Some recorded fields of the frozen jobs do not affect generation;
`pp_attachment.jobs.pinned_projection` lists the fields that do.

## License

The prompts, visual setups and questions come from the TAB benchmark
(Text-to-image Ambiguity Benchmark), distributed under CC BY-SA 4.0; this
bundle is a derived work under the same license
([`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md)).
