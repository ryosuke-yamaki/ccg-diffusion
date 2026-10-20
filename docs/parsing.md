# Parsing: Stage 1-3 and Supertag-Constrained Parsing

CCG Diffusion reads all syntactic information from one CCG derivation per
prompt (one per interpretation for TAB PP). This page describes the three
parsing stages, the frozen T2I-CompBench++ parse bundle and
Supertag-Constrained Parsing with the choice of the ambiguous preposition.
How the guidance targets are used is in [method.md](method.md).

## Parser

The parser is Hol-CCG: the `holccg` package 1.0.0 (pinned commit
`0213ca85d2397c5a4fb2c03bc1b20cf1a6283d26`, installed as described in
[installation.md](installation.md)) with the checkpoint
`ryosuke-yamaki/hol-ccg-roberta-large` at revision
`1888cd6343d8df52cf871e1a86924f500a6d7b6b`. `ccg_diffusion.config` holds the
three parse contracts:

| Contract | Use | Thresholds (supertag / phrase / span) | Skimmer |
|---|---|---|---|
| `HOLCCG_PARSE` | Stage 1 of T2I-CompBench++ and of arbitrary prompts | 0.01 / 0.001 / 0.001 | on |
| `HOLCCG_TAB_SITE_PARSE` | unconstrained TAB parse that locates the preposition | 0.01 / 0.001 / 0.001 | off |
| `HOLCCG_SUPERTAG_CONSTRAINED_PARSE` | Supertag-Constrained Parsing of one TAB interpretation | 0.01 / 1e-6 / 1e-6 | off |

All three use the checkpoint's own binary rules (`grammar: treebank`) and one
sentence per batch (`chunk_size: 1`).

## Stage 1: derivations

```bash
python -m scripts.ccg.parse --input-raw-path STAGE/color.raw --output-auto-path STAGE/color.auto
```

[`scripts/ccg/parse.py`](../scripts/ccg/parse.py) parses a `.raw` corpus (one
whitespace-tokenized sentence per line) into CCGbank AUTO and writes
`<auto>.meta.json` (checkpoint, resolved revision, parse settings, input and
output sha256, sentence counts). Its defaults come from
[`configs/parsing/holccg.yaml`](../configs/parsing/holccg.yaml), which equals
`HOLCCG_PARSE`. With the skimmer on, a sentence without a complete derivation
yields a `PARTIAL` derivation of skimmed fragments instead of nothing.

Parse corpora are written by
[`parsing/sentences.py`](../src/ccg_diffusion/parsing/sentences.py): a prompt
is split into sentences at terminal punctuation and tokenized for parsing
(punctuation and the clitic *'s* split off, every token an exact substring of
the prompt with its character span). The generation prompt is never rewritten.

**Near ties.** The supertagger's scores of two categories can be almost equal.
A different GPU, a different batch composition (`chunk_size` other than 1) or a
different library build can flip such a near tie and change the derivation.
The derivations of the 2,400 T2I-CompBench++ prompts are therefore shipped
frozen (below), and a fresh parse is not guaranteed to reproduce them on other
hardware.

## Stage 2: marked dependencies

```bash
python -m scripts.ccg.generate_dependencies --auto-path STAGE/color.auto --raw-path STAGE/color.raw \
    --output-dir STAGE/generate --apply-skimmer
```

[`scripts/ccg/generate_dependencies.py`](../scripts/ccg/generate_dependencies.py)
turns each derivation into C&C `generate -e` marked dependencies (head word,
category with the argument slot, slot number, argument word) with the
pure-Python port of C&C `generate` in the `holccg` package; no external C&C
binary is used. The markedup categories are the `hol_en` resources of `holccg`
plus the overlay [`configs/parsing/candc_markedup_additions`](../configs/parsing/candc_markedup_additions).
A `PARTIAL` derivation is split into its fragments; each fragment is generated
on its own and its token indices are shifted back to sentence positions, and no
dependency between two fragments is ever added. The outputs are
`<prefix>.markedup_deps`, `<prefix>.ccgbank_deps`, `<prefix>.coverage_rows.jsonl`
and `<prefix>.coverage.json` (the share of complete parses and of sentences
with dependencies). The first lines of a `.markedup_deps` file are `#` header
lines that contain the `generate` command with the paths of the run; compare
dependency files without them.

## Stage 3: guidance targets

```bash
python -m scripts.ccg.build_guidance_targets --prompt-only STAGE/color.prompt_only.jsonl \
    --auto-path STAGE/color.auto --sentence-map-path STAGE/color.sentence_map.jsonl \
    --markedup-deps-path STAGE/generate/color.markedup_deps \
    --coverage-rows-path STAGE/generate/color.coverage_rows.jsonl --output STAGE/color.deps2index_sd14.jsonl
```

[`scripts/ccg/build_guidance_targets.py`](../scripts/ccg/build_guidance_targets.py)
writes one guidance-target row per prompt (provider `deps2index_sd14`:
E, C, K, P as SD1.4 CLIP token indices and the word-to-token alignment), its
conversion trace and `<output>.meta.json`. A prompt whose Stage 1-2 failed gets
a failure row; T2I-CompBench++ runs generate such prompts without guidance.
Stage 3 loads only the pinned SD1.4 CLIP tokenizer and the spaCy lemmatizer
(for the possession verbs).

## Frozen T2I-CompBench++ parse bundle

[`data/t2i_compbench/`](../data/t2i_compbench/README.md) holds the Stage 1
derivations, Stage 2 marked dependencies and coverage rows of the 2,400
T2I-CompBench++ prompts, the default parse inputs of the T2I-CompBench++ runs
on both backbones.
`python -m scripts.t2i_compbench.prepare_prompts --stage-dir DIR` (default
`--parse-inputs frozen`) writes the prompt files and parse corpus of the eight
categories into `DIR`, unpacks the bundle there and prints the Stage 3 command
of each category; everything after that runs on the CPU.
`--parse-inputs parse` prints the Stage 1 and Stage 2 commands instead.

## One prompt

```bash
python -m scripts.ccg.show_derivation --prompt "a blue banana and a green vase"
```

[`scripts/ccg/show_derivation.py`](../scripts/ccg/show_derivation.py) prints the
derivation tree and the marked dependencies of one prompt through the
single-prompt front end `parsing.frontend.CCGFrontEnd`, which keeps the parser
and the `generate` resources loaded and is also used by `single_prompt` and the
generation-time benchmark (`scripts/paper/compute_cost.py`).

## Supertag-Constrained Parsing

`SupertagConstrainedHolCCG` in
[`parsing/holccg.py`](../src/ccg_diffusion/parsing/holccg.py) extends the
public parser with `parse(..., forced_supertags={position: category})`: the
leaf cell of each forced token keeps only the given category, so the CKY search
returns the best derivation in which the token has that supertag.
`supertag_constrained_parse` calls it with `HOLCCG_SUPERTAG_CONSTRAINED_PARSE`.
For example, the two interpretations of the PP *with a telescope* (the
preposition is token 6, counted from 0) are

```bash
python -m scripts.ccg.show_derivation --prompt "the boy looks at the girl with a telescope" \
    --force-supertag 6 "(NP\NP)/NP"            # NP attachment
python -m scripts.ccg.show_derivation --prompt "the boy looks at the girl with a telescope" \
    --force-supertag 6 "((S\NP)\(S\NP))/NP"    # VP attachment
```

### Choosing the ambiguous preposition

A TAB prompt may contain several prepositions. `select_preposition` in
[`pp_attachment/supertag_constrained_parsing.py`](../src/ccg_diffusion/pp_attachment/supertag_constrained_parsing.py)
chooses the one whose attachment is ambiguous from the grammar alone (no gold
annotation is read):

1. **Schema pairs.** Every lexical category of the parser is peeled down to a
   noun-modifier core `NP\NP` or a verb-phrase-modifier core `(S\NP)\(S\NP)`;
   two categories with the same argument chain and different cores form a
   pair, e.g. `(NP\NP)/NP` and `((S\NP)\(S\NP))/NP`.
2. **Anchoring.** The category of each token in the unconstrained parse
   (`HOLCCG_TAB_SITE_PARSE`) is peeled until it reaches a self-modifier, a
   non-nominal atomic category or a type-change source; nominal projections
   (`N`, `NP`) do not stop the peeling (tier `t0`).
3. **Realizability.** A token is a candidate when Supertag-Constrained Parsing
   with each category of its pair keeps the forced leaf and derives the root
   category of the unconstrained parse.

The recorded tiers then decide:

| Tier | Recorded name | Rule |
|---|---|---|
| - | `single_site` | exactly one candidate |
| `a` | `frame_repair` | no candidate: a governor whose category strands a trailing argument in both constrained parses is re-forced with that argument deleted |
| `b1` | `nesting` | several candidates: drop a candidate nested inside another candidate's flipped constituent |
| `b2` | `vacuity` | then drop candidates whose two interpretations give the preposition the same resolved dependencies |
| `b3` | `nominal_core_variant` | if none is left: retry with the `N\N` core of the noun-modifier category, `(N\N)/NP` |

On the 74 PP items of TAB the preposition is *with* in 56 items and *in* in 18;
the decision is `single_site` for 69 items, `frame_repair` for `tab-pp-039`
and `tab-pp-045` (the verb *leaves* is re-forced from `((S[dcl]\NP)/PP)/NP` to
`(S[dcl]\NP)/NP`) and `nominal_core_variant` for `tab-pp-021`..`023`. The two
constrained derivations of the chosen preposition are the interpretations
`np_attach` and `vp_attach`.

`python -m scripts.tab_pp.parse_interpretations --output DIR` runs this
selection and Stages 2-3 for all 74 items (the clean path of TAB PP,
[tab_pp.md](tab_pp.md)); `--compare-frozen data/tab_pp` compares the result
with the frozen bundle. The frozen bundle
[`data/tab_pp/`](../data/tab_pp/README.md) stores the chosen preposition and
tier of every interpretation under `preposition`.
