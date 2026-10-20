# Frozen T2I-CompBench++ parse inputs (`data/t2i_compbench/`)

The Stage 1 and Stage 2 outputs of the eight T2I-CompBench++ categories (2,400
test prompts) from which the T2I-CompBench++ runs of CCG Diffusion are
generated, on Stable Diffusion v1.4 and on Sana 0.6B. Hol-CCG supertagging has
near ties: a re-parse on other hardware can flip a few supertags and change the
derivation and the guidance targets of those prompts. The bundle removes this
dependency, so Stage 3 and the generation `prepare` steps run on the CPU and
give the same plans on every machine.

| Directory | File | Content |
|---|---|---|
| `auto/` | `<category>.auto.gz` | Stage 1 Hol-CCG derivations (CCGbank AUTO, 300 sentences) |
| `markedup_deps/` | `<category>.markedup_deps.gz` | Stage 2 C&C `generate -e` marked dependencies (300 blocks) |
| `coverage_rows/` | `<category>.coverage_rows.jsonl.gz` | Stage 2 per-sentence coverage rows (300 rows) |
| | `manifest.json` | the digests of every file (below) |

The bundle is provided frozen; it is not rebuilt by any command of this
repository. All 24 files are gzip level 9 with `mtime` 0, so the compressed
bytes are fixed as well.

## Manifest and verification

`manifest.json` (schema `ccg_diffusion.t2i_parse_bundle.v1`) lists, for every
file, its category and kind, the sha256 and size of the compressed and of the
decompressed bytes, the record count, the path at which the file is unpacked
into a Stage 1-3 directory (`stage_path`) and the tracked metadata that records
the same decompressed sha256 (`sources`). For the AUTO files this is the Stage 1
metadata `output/generation/20260812_t2icb_sd14_six_arm_full_v1/input/stage123/<category>.auto.meta.json`
(field `output_auto_sha256`), which also records the parse settings: Hol-CCG
checkpoint `ryosuke-yamaki/hol-ccg-roberta-large`, `holccg` 1.0.0, thresholds
0.01/0.001/0.001, skimmer on, treebank grammar, one sentence per batch. These
eight metadata files are tracked and must stay at that path
([docs/data.md](../../docs/data.md#frozen-inputs-under-output)).

`ccg_diffusion.targets.guidance_targets.load_t2i_parse_bundle(verify=True)`
checks every file against both sha256 of the manifest; unpacking always
verifies first. SD1.4 T2I `prepare` additionally checks each unpacked AUTO file
against its Stage 1 metadata and the parser settings of the configuration
`ccg_diffusion_t2i_sd14` (`python -m ccg_diffusion.methods inspect
ccg_diffusion_t2i_sd14`). `python -m pytest -q tests/test_guidance_targets.py`
includes the bundle check (CPU).

Two properties of the bytes are part of the frozen content:

- the `#` header lines of each `.markedup_deps` file contain the `generate`
  command with the absolute paths at which the files were produced; compare
  dependency files without these lines;
- each coverage row carries the keys `experiment`, `prompt_id` and `seed` of
  its prompt, and the 918 of the 2,400 rows whose tokenized sentence differs
  from the source prompt (for example `rack .` for `rack.`) also carry
  `source_prompt` (color 56, shape 82, texture 60, numeracy 120, non_spatial
  300, complex 300).

## Use

`python -m scripts.t2i_compbench.prepare_prompts --stage-dir DIR` (default
`--parse-inputs frozen`) writes the prompt files and parse corpus of every
category and unpacks this bundle into `DIR/<category>.auto`,
`DIR/generate/<category>.markedup_deps` and
`DIR/generate/<category>.coverage_rows.jsonl`; an existing file is kept when
its bytes are equal and refused otherwise. It then prints the Stage 3 command
of each category ([docs/parsing.md](../../docs/parsing.md),
[docs/t2i_compbench.md](../../docs/t2i_compbench.md)). In code,
`ccg_diffusion.targets.guidance_targets` provides
`load_t2i_parse_bundle(verify=True)` and `unpack_t2i_parse_bundle(stage_dir)`.

`--parse-inputs parse` prints the Stage 1 (GPU) and Stage 2 commands instead,
for a fresh parse of the prompts.

## License

The prompts are the official T2I-CompBench++ test prompts (MIT License); see
[`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md) for the C&C resources
used by Stage 2.
