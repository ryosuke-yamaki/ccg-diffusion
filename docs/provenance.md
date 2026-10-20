# Provenance: recorded labels

The records that the commands write (plans, protocols, job files, image
metadata, run manifests, judge requests and answers) identify methods,
conditions, routes and inputs by fixed labels. The labels are kept verbatim
because the digests of the records contain them; they are recorded labels, and
this page maps the ones that you meet in the outputs and on the command lines
to the component names used in these docs and to the code (module paths are
under `src/ccg_diffusion/`).

## Glossary

| Recorded value | Term in these docs | Code |
|---|---|---|
| `HolCCG`, `holccg` | Hol-CCG, the CCG parser | the `holccg` package; `parsing.holccg` |
| Stages 1-5 | parse, marked dependencies, guidance targets, generation, evaluation | [method.md](method.md) |
| route `B0` | default guidance strength | `GuidanceRoute.DEFAULT` |
| route `P1` | default strength plus the Spatial Relation Loss | `GuidanceRoute.SPATIAL_RELATION` |
| route `P2` | Selective Guidance Boost (boosted strength) | `GuidanceRoute.BOOSTED` |
| `profile`, `effective_profile` (fields of a guidance decision) | the route of the image | `GuidanceDecision.profile`, `ImageGuidance.effective_profile` |
| `attribute_v2` | the eligibility rule of the Selective Guidance Boost | `guidance.selective_guidance_boost.ROUTING_STRATEGY` |
| `routing_v1`, `routing_whole_word_v1` | kernel policies without and with whole-word pooling on `P2` images | `GUIDANCE_POLICIES` |
| COUNT, count guidance | Count Loss | `guidance.count_loss` |
| prefix layout, `prefix_layout`, `source_prefix_regions`, phrase branch | Positional Prefix with the Region Assignment Loss | `guidance.positional_prefix`, `guidance.region_assignment_loss` |
| region, `transfer_region`, attachment layout | Region Assignment Loss; on TAB the Holder-Object Region Assignment | `pp_attachment.holder_object` |
| `relation_loss`, `metric_aligned_v1` | Spatial Relation Loss | `guidance.spatial_relation_loss` |
| forced reparse, `forced_supertags` | Supertag-Constrained Parsing | `parsing.holccg.supertag_constrained_parse` |
| site | the ambiguous preposition | `PrepositionSite` |
| reading, `np_attach`, `vp_attach` | interpretation: NP attachment, VP attachment | `Interpretation.reading` |
| `tab_ppvp_holder_relative_pp_cleft_fronting_v1`, `pp_object_np_cleft`, `pp_event_fronting` | Disambiguating Paraphrase (object cleft for NP attachment, PP fronting for VP attachment) | `pp_attachment.disambiguating_paraphrase` |
| `tab_next_to_vp_attachment_slot1_resolution_v1` | the `next_to` relations of the Spatial Relation Loss on TAB | `pp_attachment.spatial` |
| PD index, sidecar, `pd_indices` | guidance targets I_D = (E, C, K, P) | `targets.guidance_targets`, `targets.pd_indices.PDIndices` |
| `exist`, `corr`, `leak`, `possession` | Existence, Correspondence, Leakage, Possession | `PDIndices` |
| setup, target question | visual setup, intention question | `evaluation.tab` |
| B4 | a physical batch of 4 images | `physical_batch_index`, `physical_batch_position` |
| t2icb, `spatial`, `3d_spatial`, `non_spatial` | T2I-CompBench++, 2D-spatial, 3D-spatial, non-spatial | |
| `sd14`, `sana`, `sana06`, `native512` | Stable Diffusion v1.4, Sana 0.6B (512px checkpoint) | |
| `gpt-6-astra-high` | GPT-6 Astra, reasoning effort high | `evaluation.judges.astra` |
| `qwen38fn`, `qwen3.8-flash-next` | Qwen3.8-Flash-Next (served name of `nvidia/Qwen3.8-Flash-Next-NVFP4`) | `evaluation.judges.qwen38` |
| `method` of the Qwen judge outputs (`summary.json`, `scores.csv`): `Vanilla`, `spaCy + PD`, `Gold-Setup Concatenation`, `spaCy + PD + Gold-Setup`, `PP-Attachment Control (ours)` | the TAB PP methods Vanilla, Predicated Diffusion, VS-TIED + Gold-visual setup, VS-TIED + Gold-visual setup + Predicated Diffusion, CCG Diffusion | `evaluation.judges.qwen38.TABLE` |

## Guidance-target providers

The `provider` of a guidance-target row (and `guidance_target_transform` in the
configuration records of `ccg_diffusion.methods`) names the Stage 3 conversion
that produced the targets; SD1.4 generation checks the provider of every row it
reads.

| Provider | Guidance targets of |
|---|---|
| `deps2index_sd14` | CCG Diffusion on T2I-CompBench++, Stable Diffusion v1.4 (SD1.4 CLIP token indices) |
| `deps2index_sana` | CCG Diffusion on T2I-CompBench++, Sana 0.6B (projected onto Gemma tokens) |
| `deps2index_syntax_ambiguity` | CCG Diffusion on TAB PP (one row per interpretation) |
| `spacy_sd14` | Predicated Diffusion with its spaCy targets on Stable Diffusion v1.4 |

## Methods, arms and conditions

An arm is one population of generated images of a run: a method, or in an
ablation run a condition. Its ID is the recorded field `arm`, names the image
directories and is the `--arm` of `scripts.t2i_compbench.export_samples` and
`scripts.t2i_compbench.evaluate` (`output/vqa/<run_id>/t2icb_work/<arm>/`).

**T2I-CompBench++: CCG Diffusion and the compared methods.**

| Method | Stable Diffusion v1.4 | Sana 0.6B |
|---|---|---|
| Vanilla | the official reference generator (`scripts/t2i_compbench/official_reference/`; no arm, official `samples/` layout) | `sana_baseline` (condition `vanilla`) |
| Composable Diffusion | `sd14_composable` | `sana_composable` |
| StructureDiffusion | `sd14_structure` | `sana_structure` |
| Attend-and-Excite | `sd14_attend_excite` | `sana_attend_excite` |
| SynGen | `sd14_syngen` | `sana_syngen` |
| Predicated Diffusion | `sd14_pd_spacy` | `sana_pd_spacy` |
| CCG Diffusion | `sd14_pd_holccg` (driver method and image directory) | `sana_ccg_compositional_shape_v1` (condition `final_shape`) |

`--method` of `export_samples` names the method directory of the generation
run (`sana_ccg_compositional_ablation_v1` for the Sana ablation conditions; the
table in [t2i_compbench.md](t2i_compbench.md#6-scoring) gives it for every kind
of run). `sana_ccg_compositional_v1` is the `profile_id` written into every Sana
T2I plan, whatever its condition.

**T2I-CompBench++ component ablations.** The condition ID is derived
from the removed components (`python -m ccg_diffusion.methods ablations` lists
them with their `--remove` names):

| Condition | Meaning |
|---|---|
| `gs{0,1}_p{0,1}_r{0,1}` | attribute-binding ablation: `gs` Selective Guidance Boost, `p` Positional Prefix, `r` Region Assignment Loss; 1 = on. `gs1_p1_r1` is the full method |
| `no_2d` | Spatial Relation Loss removed (scored on 2D-spatial) |
| `no_count` | Count Loss removed (scored on numeracy) |

**TAB PP: CCG Diffusion and the compared methods.**

| Method | Stable Diffusion v1.4 | Sana 0.6B |
|---|---|---|
| Vanilla | `sd14_vanilla` | `vanilla` |
| Predicated Diffusion | `pd_spacy` | `pd_spacy` |
| VS-TIED + Gold-visual setup | `sd14_tied_oracle` | `gold` |
| VS-TIED + Gold-visual setup + Predicated Diffusion | `pd_spacy_gold` | `pd_spacy_gold` |
| CCG Diffusion | the cell `c1r1p0m0` | `guided` |

The TAB cells `c1r1p0m0`, `c0r1p0m0`, `c1r0p0m0` and `c0r0p0m0` are recorded
cell labels (`TabAblation.cell_id`): `c` = Disambiguating Paraphrase, `r` =
Holder-Object Region Assignment, 1 = on; the suffix `p0m0` is part of the
label. `c1r1p0m0` is CCG Diffusion and the other three cells are its
component ablations (`python -m ccg_diffusion.methods ablations` lists them
with their `--remove` names). The recorded `transfer_trace.recipe` of a job is
`combined_region_only` (cells `c1r1p0m0`, `c0r1p0m0`) or `combined_no_layout`
(cells `c1r0p0m0`, `c0r0p0m0`), and `combined_region_only_c1r1p0m0`
(`config.TAB_PP_RECIPE`) is the recorded recipe label of CCG Diffusion.

## Configuration records

`python -m ccg_diffusion.methods inspect ID` prints the record of one of the
four configurations that `python -m ccg_diffusion.methods list` lists; its
`record_keys` lists the labels under which the configuration appears in
records: `recorded_profile_id` (the profile label of the SD1.4
configurations), `recorded_arm_id` and `run_arm`, `condition`,
`guidance_target_transform` and `transfer_recipe` (TAB). The records are
descriptive; the kernels own the constants
([method.md](method.md#hyperparameters-and-the-code)).

## Identifiers in the outputs

| Identifier | Form |
|---|---|
| run ID | `<YYYYMMDD>_<slug>`, shared by `output/generation/<run_id>` and `output/vqa/<run_id>` ([data.md](data.md)) |
| T2I-CompBench++ prompt | `t2icb_<category>_<NNNN>` (position in the official prompt file); an image row appends `_s<seed>`, seeds 42-51 |
| TAB item | `tab-pp-NNN` (dataset order); interpretations `np_attach` and `vp_attach`, seeds 0-4 |
| SD1.4 TAB image | `<cell>_<tab_id>_<reading>_s<seed>`, condition `<run_id>:<cell>` |
| Sana TAB prompt key | `guided_<tab_id>_<reading>` |
| TAB comparison-method job key | `vanilla_<tab_id>_None`, `gold_<tab_id>_<setup>`, `pd_spacy_<tab_id>_None`, `pd_spacy_gold_<tab_id>_<setup>`; image IDs append `_s<seed>` |

## Schema IDs

Schema, policy and lexicon IDs in the records (prefix `ccg_diffusion.` or
`holsd.`) are recorded labels; a record is read by the module that defines its
schema ID.
