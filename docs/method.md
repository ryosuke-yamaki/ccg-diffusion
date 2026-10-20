# Method

CCG Diffusion extends Predicated Diffusion with information read from a CCG
derivation of the prompt. The derivation reaches the image model through two
pathways: the text given to the text encoder (Positional Prefix, Disambiguating
Paraphrase) and attention guidance (the Predicated Diffusion loss and the
Spatial Relation, Region Assignment and Count losses, with the Selective
Guidance Boost choosing the guidance strength). This page describes each
component and names the module and the constants that implement it.

Pipeline stages used throughout the repository:

| Stage | What | Code |
|---|---|---|
| 1 | Hol-CCG parse of the prompt into a derivation (CCGbank AUTO) | `src/ccg_diffusion/parsing/holccg.py` |
| 2 | C&C `generate` marked dependencies of the derivation | `src/ccg_diffusion/parsing/dependencies.py` |
| 3 | Guidance targets I_D = (E, C, K, P) on SD1.4 CLIP token indices | `src/ccg_diffusion/targets/guidance_targets.py` |
| 4 | Generation with prompt rewriting and attention guidance | `src/ccg_diffusion/compositional/`, `src/ccg_diffusion/pp_attachment/`, `src/ccg_diffusion/backbones/` |
| 5 | Evaluation (official T2I-CompBench++ metrics, TAB judges) | `src/ccg_diffusion/evaluation/` |

Stages 1-3 are described in [parsing.md](parsing.md).

## Predicated Diffusion

Before each guided denoising step, each image's latent moves along the
negative gradient of its loss; at step j of the T denoising steps,

    x_t <- x_t - eta_t * grad L,    eta_t = s * sqrt(1 - 0.5 j / (T - 1))

The step `eta_t * grad L` is scaled down when its L2 norm exceeds `delta`
times the L2 norm of the latent; `(s, delta)` is the guidance strength. The
update is `update_latents_batched` in
[`guidance/latent_update.py`](../src/ccg_diffusion/guidance/latent_update.py):
one batched backward pass over the sum of the per-image losses gives every
image its own gradient, and the cap is applied per image. The step size is
computed as `s * sqrt(torch.linspace(1, 0.5, T)[j])`.

The loss L_PD is `compute_pd_loss` in
[`guidance/predicated_diffusion_loss.py`](../src/ccg_diffusion/guidance/predicated_diffusion_loss.py):
the weighted sum of the terms of the four proposition types of Predicated
Diffusion (existence, correspondence, leakage, possession) on the guidance
targets:

    L_exist(i)        = -log(1 - exp(mean_x log(1 - a_i(x))))
    L_corr(i, j)      = -log(1 - a_j (1 - a_i)) - log(1 - a_i (1 - a_j))   (mean over pixels)
    L_leak(i, j)      = -log(1 - a_i a_j)                                  (mean over pixels)
    L_poss(h, o)      = -log(1 - a_o (1 - a_h))                            (mean over pixels)

with the weights `gamma = (0.1, 1, 0.3, 1)` (`EXISTENCE_WEIGHT`, `PD_LEAK_WEIGHT`).
All loss arithmetic is FP32. The pair terms (correspondence, leak and
possession) use the bounded barrier `-log1p(-(1 - 1e-6) * product)`. The
existence term computes `mean_x log1p(1e-4 - a_i(x))`, clamps it to at most
`-1e-6` and returns `-log(-expm1(.))`. Its maps are softmax-normalized: a
softmax over 100 times the attention columns, on SD1.4 over the fixed 75
lexical CLIP columns (padding included, as in Predicated Diffusion), on Sana
over the Gemma columns after the leading BOS.

The attention maps are 16x16 cross-attention probabilities:

* Stable Diffusion v1.4: the mean over heads of the five `attn2` processors at
  16x16 ([`guidance/attention.py`](../src/ccg_diffusion/guidance/attention.py));
* Sana 0.6B: the cross-attention of transformer blocks [0, 8) at 16x16
  ([`backbones/sana/attention.py`](../src/ccg_diffusion/backbones/sana/attention.py)).

The guided denoising loops are `generate_predicated_batch` in
[`backbones/sd14/pipeline.py`](../src/ccg_diffusion/backbones/sd14/pipeline.py)
(PNDM, 50 steps, CFG 7.5, the first 25 steps guided, 5 extra updates at step 0)
and `generate_guided` in
[`backbones/sana/guidance.py`](../src/ccg_diffusion/backbones/sana/guidance.py)
(the checkpoint's DPM-Solver++ flow scheduler, 20 steps, CFG 4.5, all 20 steps
guided, 5 extra updates at step 0).

## Guidance targets from the CCG derivation

Stage 3 (`targets.guidance_targets`, provider `deps2index_sd14`) reads
I_D = (E, C, K, P) from the derivation and the marked dependencies:

* E (`exist`): the words with category `N` (nouns);
* C (`corr`): modifier -> noun pairs; a modifier of a modifier is paired with
  the noun it finally modifies, so *a light blue banana* gives
  `(light, banana)` and `(blue, banana)`;
* K (`leak`): every modifier paired with each noun it does not modify;
* P (`possession`): subject -> object pairs of *have*, *wear*, *grasp* and
  *hold* (the only word comparison, by spaCy lemma).

The targets are SD1.4 CLIP token indices (BOS at index 0). For *a blue banana
and a green vase*, `E = [3, 7]`, `C = [[2, 3], [6, 7]]`, `K = [[2, 7], [6, 3]]`
and `P = []` (see `tests/test_guidance_targets.py`). On Sana, every index is
projected to the Gemma tokens that cover the same characters
([`backbones/sana/tokens.py`](../src/ccg_diffusion/backbones/sana/tokens.py)).

A word that spans several tokens is represented by its last token in the
Predicated Diffusion loss and the Spatial Relation Loss, and by the mean of
all its tokens in the Count Loss and the Region Assignment Loss (and, under the
Selective Guidance Boost, in the correspondence and leakage terms; see below).

## Compositional text-to-image generation

The per-prompt plan, which decides which components apply and with which
targets, is `plan_prompt` in
[`compositional/plans.py`](../src/ccg_diffusion/compositional/plans.py). The
SD1.4 runs are in [`compositional/sd14.py`](../src/ccg_diffusion/compositional/sd14.py),
the Sana runs (with the Gemma projection of the plans and the per-image
objective `CompositionalLoss`) in
[`compositional/sana.py`](../src/ccg_diffusion/compositional/sana.py).

### Selective Guidance Boost

[`guidance/selective_guidance_boost.py`](../src/ccg_diffusion/guidance/selective_guidance_boost.py)
gives each image one route, recorded under its label:

| Route | Meaning | Strength `(s, delta)` |
|---|---|---|
| `B0` | default | (20, 0.02) on T2I-CompBench++; (10, 0.02) on TAB PP |
| `P1` | default strength plus the Spatial Relation Loss | as `B0` |
| `P2` | Selective Guidance Boost | (80, 0.06) on both tasks |

The decision has two stages. `derive_base_decision` reads the derivation:
incomplete parses, unresolved modifiers, count words, depth words, action
verbs and attachments not covered by a relation keep `B0`; 2D relations from
the closed set of the Spatial Relation Loss, linked to two nouns by the
derivation, give `P1`; visual modifier bindings give `P2`.
`apply_selective_guidance_boost` then applies the eligibility rule (recorded
name `attribute_v2`), in this order:

1. inputs eligible for the Count Loss keep `B0`;
2. `P1` inputs keep `P1` (default strength plus the Spatial Relation Loss);
3. incomplete parses, unresolved modifier bindings, count words and depth
   words (`DEPTH_WORDS`, `DEPTH_SEQUENCES`) keep `B0`;
4. a noun-bound modifier from the color/shape/size/material/surface vocabulary
   (`VISUAL_ADJECTIVES`, `ATTRIBUTE_ADDITIONS`) boosts the image to `P2`;
5. everything else keeps `B0`.

The boost multiplies the base scale 20 by `BOOST_SCALE_MULTIPLIER = 4.0` and
uses the cap `BOOSTED_UPDATE_CAP = 0.06`. `derive_image_guidance` runs both
stages and derives the Count Loss constraints from the same inputs. Each
decision is recorded with its reason strings and evidence.

Under the kernel policy `routing_whole_word_v1`, the correspondence and leakage
terms of boosted (`P2`) images read the mean of the raw maps of all tokens of
each endpoint word before min-max normalization (whole-word pooling, recorded
scope `mean_raw_lexical_attention_for_corr_leak_endpoints_on_P2_rows_only`);
existence and possession are unchanged, and `B0`/`P1` images use the last
token.

### Positional Prefix and Region Assignment Loss

[`guidance/positional_prefix.py`](../src/ccg_diffusion/guidance/positional_prefix.py)
decides eligibility (`build_phrase_plan`): a complete parse without count
words, count constraints, depth words or a Spatial Relation route, and two
contiguous, non-overlapping noun phrases whose nouns carry color, texture,
shape or size modifiers, joined by *and*. The conditioning text is

    "On the left side, NP1. On the right side, NP2. " + c

`build_prefix_plan` maps every lexical CLIP position of the prompt to its copy
in the prefix (the two noun phrases) or in the unchanged prompt suffix and
translates all guidance targets accordingly: the tokens of a noun phrase in the
prefix are the ones guided. On SD1.4, `prefix_conditioning` encodes the
conditioning text once per physical batch and replaces the conditional
cross-attention K/V inputs of the prefixed images; the unconditional rows of
classifier-free guidance and all other images are unchanged. On Sana the
prefixed text is what Gemma encodes. Positional Prefix plans also contain
record-only fields (among them `phantom_anchor`) that enter the plan digest
`source_plan_sha256` and never change an executed operation.

The Region Assignment Loss is `region_assignment_loss` in
[`guidance/region_assignment_loss.py`](../src/ccg_diffusion/guidance/region_assignment_loss.py).
The word groups are the noun and its modifiers of each noun phrase (left group
first). Each word map `A` is the mean of the word's token maps; with pixel
x-coordinates in [0, 1] and the soft half-plane mask
`M_L(x) = sigmoid((x_b - x) / kappa)` (`M_R = 1 - M_L`), the fraction of the
word's mass in its assigned half is `f = sum(A * M) / sum(A)`, and the loss is
the mean of `-log f` over all grouped words, with `x_b = 0.5`, `kappa = 0.08`
and weight `lambda_region = 0.25`. On SD1.4 the per-image loss of a prefixed
prompt is `PrefixRegionLoss`: L_PD on the remapped targets plus
`0.25 * L_region`.

### Spatial Relation Loss

[`guidance/spatial_relations.py`](../src/ccg_diffusion/guidance/spatial_relations.py)
detects the closed set of relation phrases (*on the left of*, *on the right
of*, *on the top of*, *on the bottom of*, *next to*, *near*, *on side of*)
from the derivation: the relation must link a subject noun and an object noun
through marked dependencies. [`guidance/spatial_relation_loss.py`](../src/ccg_diffusion/guidance/spatial_relation_loss.py)
min-max normalizes the map of each endpoint and takes its centroid
`(cx, cy)` in [0, 1] (`cy` grows downward; recorded geometry
`metric_aligned_v1`). Per relation, the loss is a
placement hinge plus the overlap term `mean(a_s * a_o)` of the normalized
subject and object maps: `relu(m - gap)` on the signed centroid offset along
the requested axis plus `relu(|off-axis offset| - |gap|)` for *left*, *right*,
*top* and *bottom*, and `relu(min(dx, dy) - r)` for *next to*, *near* and *on
side of*, with `m = 0.25` (`MARGIN`) and `r = 150/512` (`PROXIMITY_MAX`). The
loss is the mean over relations; callers add
`SPATIAL_RELATION_LOSS_WEIGHT * L_spatial` with `lambda_spatial = 6`. On SD1.4
the relation term is added to the base loss of `P1` images, before the Count
Loss.

### Count Loss

[`guidance/count_loss.py`](../src/ccg_diffusion/guidance/count_loss.py)
extracts a constraint `(noun, k)` when a cardinal from *one* to *ten* takes an
atomic `N` as its argument in a complete parse; ambiguous cases (vague
quantifiers, coordinated or modified cardinals, repeated referents,
incomplete parses) abstain. At the first loss evaluation within outer steps
0-9 at which every counted noun has a usable map (mean and range of the
smoothed map at least `1e-4`), the target `q_n` is frozen: `k` peaks of the
smoothed map with min-max-normalized salience at least `theta = 0.3`, at least
`min(16/k, 4)` pixels apart, as an equal-mass Gaussian mixture confined to
their Voronoi cells (width half the distance to the nearest other center, at
most 2 pixels). The loss adds `0.1 * mean_n(g_n * symmetric_KL(p_n, q_n))`
with `g_n = min(1, (77 / I) * sum_i smooth(A_n)[i])`, where `smooth(A_n)` is the
smoothed map of noun `n` over its `I` pixels (no gradient through `g_n`).

## PP ambiguity resolution

### Supertag-Constrained Parsing

`SupertagConstrainedHolCCG.parse(..., forced_supertags={position: category})`
in [`parsing/holccg.py`](../src/ccg_diffusion/parsing/holccg.py) restricts the
chart cell of a token to one category. Fixing the preposition to
`(NP\NP)/NP` gives the NP-attachment derivation, `((S\NP)\(S\NP))/NP` the
VP-attachment derivation (`parsing/derivation.py`). Which preposition is the
ambiguous one is decided from the grammar by `select_preposition` in
[`pp_attachment/supertag_constrained_parsing.py`](../src/ccg_diffusion/pp_attachment/supertag_constrained_parsing.py);
[parsing.md](parsing.md#supertag-constrained-parsing) describes the
parse settings, the decision tiers and the frame repair.

### Disambiguating Paraphrase and Holder-Object Region Assignment

[`pp_attachment/disambiguating_paraphrase.py`](../src/ccg_diffusion/pp_attachment/disambiguating_paraphrase.py)
rewrites the prompt with the derivation of the selected interpretation
(recorded rule `tab_ppvp_holder_relative_pp_cleft_fronting_v1`):

* NP attachment, object cleft (`pp_object_np_cleft`): *It is [object NP] [PP]
  that [words before the object NP].*
* VP attachment, PP fronting (`pp_event_fronting`): *[PP], [words before the PP].*

Every original word keeps its CLIP pieces exactly once and every guidance index
is remapped to the rewritten prompt. The rule handles PPs headed by *with* or
*in* in a finite declarative root clause with a listed present-tense
predicate. Other inputs keep the source prompt and get no region layout
(reason `requires_finite_declarative_clause`); among the TAB PP items these are
`tab-pp-021`, `tab-pp-022` and `tab-pp-023`, so the Region Assignment Loss
applies to 142 of the 148 interpretations.
[`pp_attachment/holder_object.py`](../src/ccg_diffusion/pp_attachment/holder_object.py)
identifies the object (the noun of the PP's NP argument), the holder of the
selected interpretation and the holder of the other interpretation, and builds
the two word groups of the Region Assignment Loss: the selected holder, the
object and their modifiers on one side, the other holder on the other side
(the two holders keep their source order, left to right).

The TAB objective of one image is `L_PD + 6 * L_spatial + 0.25 * L_region`
(`tab_objective` and `TabObjective` in
[`pp_attachment/objective.py`](../src/ccg_diffusion/pp_attachment/objective.py),
`SanaPPAttachmentLoss` in
[`pp_attachment/sana.py`](../src/ccg_diffusion/pp_attachment/sana.py)), with the
default strength (10, 0.02) and the Selective Guidance Boost (80, 0.06). On
SD1.4 the kernel applies a route's scale as `scale_factor * scale_multiplier`:
a physical batch of 4 that contains a boosted row runs at kernel scale 20 with
multipliers 0.5 (default rows, scale 10) and 4.0 (boosted rows, scale 80), and
a batch without one runs at kernel scale 10 without routes. The boosted rows
run under the kernel policy `routing_v1` (the boosted strength without
whole-word pooling). The spatial term uses the `next_to` relations of
[`pp_attachment/spatial.py`](../src/ccg_diffusion/pp_attachment/spatial.py),
which come from a TAB-specific rule
(`tab_next_to_vp_attachment_slot1_resolution_v1`): *next* must head a fired
verb-phrase attachment frame whose slots resolve to two nouns. The rule applies
to 12 interpretations (items 000, 001, 002, 021, 022, 023).
`compile_interpretation` in
[`pp_attachment/compile.py`](../src/ccg_diffusion/pp_attachment/compile.py)
composes these components into the generation job of one interpretation. The
paraphrase is never combined with the Positional Prefix. SD1.4 TAB runs place
the jobs in the 3,150-job layout (315 TAB items x 2 interpretations x 5 seeds,
PP jobs at positions 0-739), so that every image keeps its physical batch of 4
([data.md](data.md#the-3150-job-layout)).

## Hyperparameters and the code

`python -m ccg_diffusion.methods table8` prints the generation settings and
hyperparameters as JSON, one object per setting (`row`) with its SD1.4 and
Sana values, derived from the four configurations of
[`methods.py`](../src/ccg_diffusion/methods.py) (`ccg_diffusion_t2i_sd14`,
`ccg_diffusion_t2i_sana`, `ccg_diffusion_tab_sd14`, `ccg_diffusion_tab_sana`;
`python -m ccg_diffusion.methods list`). These records are descriptive: the
kernels own their constants, and `tests/test_config_and_provenance.py` asserts
that both agree.

| Setting (`row`) | Value (SD1.4 / Sana) | Constants |
|---|---|---|
| Resolution | 512x512 | `config.SD14_GENERATION.height/width`, `config.SANA_GENERATION.height/width` |
| Denoising steps | 50 (PNDM) / 20 | `SD14_GENERATION.steps`, `.scheduler_class`; `SANA_GENERATION.steps` |
| Classifier-free guidance | 7.5 / 4.5 | `SD14_GENERATION.guidance_scale`, `SANA_GENERATION.guidance_scale` |
| Guided steps | first 25 / all 20 | `config.SD14_GUIDANCE_SCHEDULE`, `config.SANA_GUIDANCE_SCHEDULE` (`guided_steps`) |
| Extra updates at step 0 | 5 / 5 | `GuidanceSchedule.refinement_steps`, `.refinement_updates` |
| Attention maps | 5 layers (16x16) / blocks [0, 8) (16x16) | `SD14_GENERATION.expected_attention_processor_count`, `.attention_res`; `SANA_GENERATION.block_range`, `.attention_res` |
| (s, delta), default, compositional | (20, 0.02) | `guidance.selective_guidance_boost.BASE_SCALE`, `DEFAULT_UPDATE_CAP` |
| (s, delta), default, PP ambiguity | (10, 0.02) | `pp_attachment.objective.TabGuidanceStrength` |
| (s, delta), boosted, both tasks | (80, 0.06) | `BOOST_SCALE_MULTIPLIER` x `BASE_SCALE`, `BOOSTED_UPDATE_CAP` |
| gamma_exist, gamma_corr, gamma_leak, gamma_poss | 0.1, 1, 0.3, 1 | `guidance.predicated_diffusion_loss.EXISTENCE_WEIGHT`, `PD_LEAK_WEIGHT` |
| lambda_region, x_b, kappa | 0.25, 0.5, 0.08 | `guidance.region_assignment_loss.REGION_WEIGHT`, `BOUNDARY`, `BOUNDARY_WIDTH` |
| lambda_spatial, m, r | 6, 0.25, 150/512 | `guidance.spatial_relation_loss.SPATIAL_RELATION_LOSS_WEIGHT`, `MARGIN`, `PROXIMITY_MAX` |
| lambda_count (steps), theta | 0.1 (0-9), 0.3 | `guidance.count_loss.COUNT_WEIGHT`, `ACTIVE_OUTER_STEPS`, `TAU` |

Further settings of the four configurations:

* SD1.4: `CompVis/stable-diffusion-v1-4` at revision `133a221b`, precision
  profile `bf16_stable` (U-Net, text encoder and VAE in BF16; master latent,
  scheduler, attention reduction, losses and update in FP32,
  [`guidance/precision.py`](../src/ccg_diffusion/guidance/precision.py)),
  physical batches of 4, no negative prompt.
* Sana: `Efficient-Large-Model/Sana_600M_512px_diffusers` at revision
  `83d7a190`, FP16 model with FP32 master latents and losses; Sana prepends its
  own instruction prompt (`backbones.sana.pipeline.COMPLEX_HUMAN_INSTRUCTION`)
  before Gemma encodes the text; the scheduler is the checkpoint's
  `DPMSolverMultistepScheduler` (flow prediction).
* Seeds: 42-51 for T2I-CompBench++, 0-4 for TAB PP.

## Arbitrary prompts

`ccg_diffusion.single_prompt.generate(prompt, backbone="sd14" | "sana", seed=42,
remove=())` runs the whole method on one prompt: the Stage 1-2 front end
(`parsing.frontend.CCGFrontEnd`), Stage 3, `plans.plan_prompt`, the ablation
switches and the backbone. The command line is
`python -m scripts.t2i_compbench.run_ccg_diffusion prompt`. The prompt must be
one sentence; a multi-sentence prompt is rejected. The result records
`"paper_exact": false`: the benchmark runs use the frozen parses of the
benchmark prompts and their physical batches of four images, and a re-parse on
other hardware can flip near-tie supertags. If Stage 1-3 yields no usable
guidance targets, the image is generated without guidance and the record says
why.
