"""CCG Diffusion on Sana 0.6B for T2I-CompBench++: Gemma-token plans, the image objective, prepare and generate.

``compile_plan`` takes the plan of a prompt on the SD1.4 CLIP token axis
(``plans.plan_prompt``: route, Count Loss constraints, Positional Prefix and
Region Assignment Loss groups) and projects every target onto Sana's selected
Gemma token axis: each CLIP word maps to the Gemma pieces that cover the same
characters of the conditioning text, and a target index is the last piece of
its word. The text Sana encodes is the prompt, or the Positional Prefix text
when the prefix applies; Sana's own Gemma encoding conditions the image.

``CompositionalLoss`` is the objective of one image from its 16x16 attention
maps: L_PD (``compute_pd_loss``; whole-word pooling on boosted images) + 6 x
Spatial Relation Loss (route P1) + 0.25 x Region Assignment Loss + Count
Loss. Its route sets the step multiplier and cap of the guided
loop (``backbones.sana.guidance.generate_guided``): x4 and 0.06 on boosted
images, x1 and 0.02 otherwise.

Run layout (``output/generation/<run_id>``): ``<C>_protocol.json``,
``<C>_protocol.plans.jsonl`` and images under ``<C>/`` with ``C`` =
``final_shape`` (the full method), an ablation condition ID (``gs1_p0_r1``,
``no_2d``, ...) or ``vanilla`` (stock Sana from the prompt). The plan records
keep the recorded values: condition ``final_shape``/``ablation``, profile
``sana_ccg_compositional_v1`` and the method directories below.
"""
from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from ccg_diffusion.backbones.jobs import Job, image_path, metadata_path
from ccg_diffusion.backbones.sana.pipeline import COMPLEX_HUMAN_INSTRUCTION, load_sana_tokenizer
from ccg_diffusion.backbones.sana.tokens import normalise_token, selected_token_offsets
from ccg_diffusion.compositional.plans import plan_prompt
from ccg_diffusion.config import SANA_GENERATION, T2I_COMPBENCH_DATASET_DIR
from ccg_diffusion.guidance.count_loss import CountState, NounForms, SanaCountConstraint, load_noun_forms
from ccg_diffusion.guidance.predicated_diffusion_loss import SanaWholeWordSpans, compute_pd_loss
from ccg_diffusion.guidance.region_assignment_loss import RegionGroups, region_assignment_loss
from ccg_diffusion.guidance.selective_guidance_boost import GuidanceRoute
from ccg_diffusion.guidance.spatial_relation_loss import SPATIAL_RELATION_LOSS_WEIGHT, compute_spatial_relation_loss
from ccg_diffusion.guidance.spatial_relations import RelationConstraint
from ccg_diffusion.methods import T2IAblation
from ccg_diffusion.parsing.ccg import AutoSentence, MarkedDependency, align_whitespace_tokens_to_clip
from ccg_diffusion.provenance import (
    atomic_write_json,
    atomic_write_jsonl,
    canonical_json,
    read_json_object,
    read_jsonl,
    repo_relative,
    repository_root,
    require,
    resolve_recorded_path,
    sha256_file,
    sha256_text,
    source_tree_snapshot,
    verify_artifact_record,
)
from ccg_diffusion.targets.guidance_targets import CompiledGuidanceTargets, collapse_prompt_only, load_source_prompts
from ccg_diffusion.targets.pd_indices import PDIndices
from ccg_diffusion.targets.tokenizer import load_sd14_tokenizer

SCHEMA = "holsd.sana_ccg_compositional.v1"  # plan rows (recorded value)
PROFILE_ID = "sana_ccg_compositional_v1"  # written in every plan, final_shape included (recorded value)
RUN_SCHEMA = "ccg_diffusion.sana_t2i_run.v1"
POPULATION_SCHEMA = "ccg_diffusion.t2i_population.v1"
METHOD_DIRS = {"final_shape": "sana_ccg_compositional_shape_v1",
               "ablation": "sana_ccg_compositional_ablation_v1",
               "vanilla": "sana_baseline"}  # output method directories (recorded values)
DATASET_DIR = repository_root() / T2I_COMPBENCH_DATASET_DIR
SEEDS_PER_PROMPT = 10


def compile_plan(
    *, prompt_id: str, prompt: str, sentence: AutoSentence, dependencies: Sequence[MarkedDependency],
    compiled: CompiledGuidanceTargets, clip_tokenizer: Any, gemma_tokenizer: Any, instruction: Sequence[str],
    noun_forms: NounForms | None = None, ablation: T2IAblation | None = None,
) -> dict[str, Any]:
    """The Sana plan of one prompt: the SD1.4 plan projected onto the Gemma token axis.

    Without ``ablation`` the plan is the full method (condition
    ``final_shape``); with it, the plan of that ablation condition (condition
    ``ablation``), which also records the switches. Eligibility never depends
    on a switch.
    """
    indices, alignment, trace = compiled.indices, compiled.alignment, compiled.trace
    program = asdict(indices)
    planned = plan_prompt(prompt, sentence, dependencies, compiled, clip_tokenizer, noun_forms=noun_forms)
    guidance, count_spec, source_prefix = planned.guidance, planned.count, planned.plan.prefix
    count_decision = count_spec.decision
    prompt_hash, program_hash = sha256_text(prompt), sha256_text(canonical_json(program))
    condition = "final_shape" if ablation is None else "ablation"
    components = ablation.components() if ablation is not None else None
    eligible = source_prefix.active and source_prefix.source_branch == "phrase"
    if components is not None:
        region_active, prefix_active = eligible and components["region"], eligible and components["prefix"]
    else:
        region_active = prefix_active = eligible
    conditioning = source_prefix.conditioning if prefix_active else prompt
    assert isinstance(conditioning, str)
    decoded, offsets = selected_token_offsets(gemma_tokenizer, conditioning, instruction=instruction)
    if not offsets:
        raise ValueError("Sana conditioning has no prompt tokens")
    native_count = max(offsets) + 1
    spans: dict[int, tuple[int, ...]] = {}
    anchors = align_whitespace_tokens_to_clip(conditioning, clip_tokenizer)
    for anchor in anchors:
        if not anchor.lexical_piece_indices:
            continue
        span = tuple(i for i, (start, end) in sorted(offsets.items())
                     if min(end, anchor.char_end) > max(start, anchor.char_start)
                     and normalise_token(decoded[i]))
        word = conditioning[anchor.char_start:anchor.char_end]
        if not span or normalise_token("".join(decoded[i] for i in span)) != normalise_token(word):
            raise ValueError(f"Gemma word alignment failed: {word!r}")
        spans[anchor.anchor_index] = span
    remapping = dict(source_prefix.mapping) if prefix_active else {}

    def word_span(index: int) -> tuple[int, ...]:
        return spans[remapping.get(index, index)]

    def endpoint(index: int) -> int:
        return word_span(index)[-1]

    native_program = {"exist": [endpoint(i) for i in program["exist"]]}
    native_program.update({field: [[endpoint(a), endpoint(b)] for a, b in program[field]]
                           for field in ("corr", "leak", "possession")})
    if prefix_active and source_prefix.phantom_anchor is not None:
        phantom = spans[source_prefix.phantom_anchor][-1]
        native_program["exist"] = [i for i in native_program["exist"] if i != phantom]
        native_program["leak"] = [pair for pair in native_program["leak"] if phantom not in pair]
    used_pairs = {i for pair in (*indices.corr, *indices.leak) for i in pair}
    pair_spans = SanaWholeWordSpans.from_mapping(
        {endpoint(i): word_span(i) for i in used_pairs}, text_token_count=native_count)
    pair_spans.require_coverage({i for pair in (*native_program["corr"], *native_program["leak"]) for i in pair},
                                token_axis_size=native_count)
    native_relations = [asdict(RelationConstraint(r.label, endpoint(r.subject_clip_index),
                                                  endpoint(r.object_clip_index))) for r in guidance.relations]
    native_counts = [asdict(SanaCountConstraint(
        c.count, c.numeral_source, c.noun_source, c.noun_word, endpoint(c.noun_anchor),
        word_span(c.noun_anchor), native_count)) for c in count_decision.constraints]
    groups = []
    if region_active:
        # Source groups refer to the prefixed copies. Without added text, invert
        # the authenticated lexical mapping before projecting the original words.
        group_anchors = anchors if prefix_active else align_whitespace_tokens_to_clip(
            source_prefix.conditioning, clip_tokenizer)
        owners = {piece: a.anchor_index for a in group_anchors for piece in a.lexical_piece_indices}
        inverse = {} if prefix_active else {target: source for source, target in source_prefix.mapping}
        for words in source_prefix.groups:
            groups.append([list(spans[owners[pieces[-1]] if prefix_active else inverse[owners[pieces[-1]]]])
                           for pieces in words])
    return {
        "schema": SCHEMA, "profile_id": PROFILE_ID, "condition": condition, "prompt_id": prompt_id, "prompt": prompt,
        "prompt_sha256": prompt_hash, "conditioning": conditioning,
        "conditioning_sha256": sha256_text(conditioning), "provider": "deps2index_sana",
        "source_program": program, "source_program_sha256": program_hash,
        "source_indices": trace["source_indices"], "source_alignment": alignment,
        "source_guidance": guidance.as_record(), "source_count": count_spec.as_record(),
        "source_prefix": source_prefix.as_record(), "pd_indices": native_program,
        "token_index_space": "sana_selected_bos_inclusive_v1", "text_token_count": native_count,
        "token_spans": pair_spans.as_record(), "relations": native_relations, "counts": native_counts,
        "prefix_active": prefix_active, "region_groups": groups,
        "region_relation": source_prefix.relation if region_active else None,
        **({"region_active": region_active} if components is not None else {}),
        "phrase_scope": "color_texture_shape_size",
        **({"components": dict(components)} if components is not None else {}),
        "lexical_projection": [{"source_clip_anchor": i, "target_span": list(word_span(i))}
                               for i in sorted({r["clip_index"] for r in alignment if r.get("clip_index")})],
    }


class CompositionalLoss:
    """The guidance objective of one Sana image (implements ``backbones.sana.guidance.SanaGuidanceObjective``)."""

    def __init__(self, plan: Mapping[str, Any], *, identity: str, region_weight: float = .25,
                 count_weight: float = .1, count_mass_scale: float = 77.) -> None:
        condition = plan["condition"]
        if plan["schema"] != SCHEMA or condition not in {"final_shape", "ablation"}:
            raise ValueError("unsupported Sana compositional contract")
        if (condition == "ablation") != ("components" in plan):
            raise ValueError("ablation plans and losses must agree on their component switches")
        if any(not math.isfinite(v) or v < 0 for v in (region_weight, count_weight, count_mass_scale)):
            raise ValueError("loss coefficients must be finite and nonnegative")
        self.plan = plan
        self.indices = PDIndices(**plan["pd_indices"])
        self.route = plan["source_guidance"]["effective_profile"]
        if (condition == "ablation" and not plan["components"]["guidance_strength"]
                and self.route == GuidanceRoute.BOOSTED):
            # Selective Guidance Boost off: default guidance strength and cap, no whole-word pooling; P1 and the
            # Count Loss remain.
            self.route = GuidanceRoute.DEFAULT
        spans = plan["token_spans"]
        self.pair_spans = SanaWholeWordSpans.from_mapping(
            {row["anchor"]: tuple(row["lexical_piece_indices"]) for row in spans["spans"]},
            text_token_count=plan["text_token_count"]) if self.route == GuidanceRoute.BOOSTED else None
        switches = plan.get("components", {})
        self.relations = tuple(RelationConstraint(**r) for r in plan["relations"]) if (
            self.route == GuidanceRoute.SPATIAL_RELATION and switches.get("spatial_loss", True)) else ()
        constraints = tuple(SanaCountConstraint(**{**c, "lexical_piece_indices": tuple(c["lexical_piece_indices"])})
                            for c in plan["counts"]) if switches.get("count_loss", True) else ()
        self.count_state = CountState(identity, constraints, token_axis_size=plan["text_token_count"],
                                      weight=count_weight, mass_scale=count_mass_scale)
        self.region = RegionGroups(bool(plan.get("region_active", plan["prefix_active"])),
                                   plan["region_relation"],
                                   tuple(tuple(tuple(word) for word in group) for group in plan["region_groups"]))
        self.region_weight = region_weight
        self.trace: list[dict[str, Any]] = []
        self.update_diagnostics: dict[str, Any] = {}

    @property
    def scale_multiplier(self) -> float:
        return 4. if self.route == GuidanceRoute.BOOSTED else 1.

    @property
    def max_update_ratio(self) -> float:
        return .06 if self.route == GuidanceRoute.BOOSTED else .02

    def __call__(self, attention: torch.Tensor, step_index: int) -> torch.Tensor:
        if attention.shape[-1] != self.plan["text_token_count"]:
            raise ValueError("runtime Gemma token count differs from the compiled plan")
        work = attention.float()
        if work.shape[:2] != (16, 16):
            raise ValueError("Sana 0.6B 512px attention maps are 16x16")
        loss = compute_pd_loss(work, self.indices, attention_res=16, special_layout="leading_bos",
                               pair_token_spans=self.pair_spans)
        pd_value = float(loss.detach())
        if self.relations:
            loss = loss + SPATIAL_RELATION_LOSS_WEIGHT * compute_spatial_relation_loss(work, self.relations)
        if self.region.active:
            loss = loss + self.region_weight * region_assignment_loss(work, self.region)
        loss = self.count_state.augment(loss, work, outer_step_index=step_index)
        self.trace.append({"step": step_index, "pd": pd_value, "total": float(loss.detach())})
        return loss

    def audit_record(self) -> dict[str, Any]:
        return {"route": self.route, "prefix_active": bool(self.plan["prefix_active"]),
                **({"components": self.plan["components"], "relations_applied": len(self.relations)}
                   if "components" in self.plan else {}),
                **({"region_active": self.region.active} if "region_active" in self.plan else {}),
                "scale_multiplier": self.scale_multiplier, "max_update_ratio": self.max_update_ratio,
                "loss_trace": self.trace, "count": self.count_state.audit_record(),
                "update_diagnostics": self.update_diagnostics}


# --- runs -------------------------------------------------------------------------------------------------


def verify_population(manifest: Path, prompt_only: Path) -> dict[str, Any]:
    """Every image row is ``<base prompt ID>_s<seed>`` of a blind prompt, and every blind prompt has images."""
    prompts = collapse_prompt_only(read_jsonl(prompt_only), location=prompt_only)
    jobs = read_jsonl(manifest)
    seen, covered = set(), set()
    for row in jobs:
        if set(row) != {"experiment", "prompt_id", "prompt", "seed"}:
            raise ValueError("custom generation rows require experiment, prompt_id, prompt, seed only")
        identity, seed = row["prompt_id"], row["seed"]
        if (not isinstance(identity, str) or Path(identity).name != identity
                or type(seed) is not int or seed < 0 or not identity.endswith(f"_s{seed}")):
            raise ValueError("custom image IDs must be <base_prompt_id>_s<seed>, with a nonnegative integer seed")
        base = identity.rsplit("_s", 1)[0]
        if identity in seen or base not in prompts or row["prompt"] != prompts[base]["prompt"]:
            raise ValueError("custom population has duplicate IDs or differs from blind prompts")
        experiment = row["experiment"]
        if (not isinstance(experiment, str) or experiment in {"", ".", ".."}
                or Path(experiment).name != experiment):
            raise ValueError("experiment must be a nonempty directory name")
        seen.add(identity)
        covered.add(base)
    if not jobs or covered != set(prompts):
        raise ValueError("custom manifest must cover every blind prompt")
    return {"schema": POPULATION_SCHEMA, "images": len(jobs), "prompts": len(prompts),
            "manifest_sha256": sha256_file(manifest), "prompt_only_sha256": sha256_file(prompt_only)}


def condition_name(method: str = "ccg_diffusion", ablation: T2IAblation | None = None) -> str:
    """``final_shape``, the ablation condition ID, or ``vanilla``."""
    if method == "vanilla":
        require(ablation is None, "Sana Vanilla has no ablation")
        return "vanilla"
    require(method == "ccg_diffusion", f"unknown Sana T2I method {method!r}")
    return "final_shape" if ablation is None else ablation.condition_id


def protocol_path(run_id: str, method: str = "ccg_diffusion", ablation: T2IAblation | None = None) -> Path:
    """``output/generation/<run_id>/<condition>_protocol.json``; the plans sit next to it (``.plans.jsonl``)."""
    return repository_root() / "output/generation" / run_id / f"{condition_name(method, ablation)}_protocol.json"


def prepare_run(
    *, run_id: str, method: str = "ccg_diffusion", ablation: T2IAblation | None = None,
    source_manifest: Path | None = None, allow_dirty: bool = False,
) -> Path:
    """Write the plans and the protocol of one condition for the full T2I-CompBench++ dataset; return the protocol.

    ``ablation`` selects an ablation condition (``T2IAblation()`` is the
    all-components cell ``gs1_p1_r1``, which the plans record as an
    ablation); without it the plans are the full method (``final_shape``).
    """
    require(re.fullmatch(r"\d{8}_[a-zA-Z0-9_-]+", run_id) is not None, "run ID must be YYYYMMDD_slug")
    path = protocol_path(run_id, method, ablation)
    require(not path.exists(), f"protocol already exists: {path}")
    manifest, blind = DATASET_DIR / "independent_seed_manifest.jsonl", DATASET_DIR / "prompt_only.jsonl"
    verification = verify_population(manifest, blind)
    source = source_tree_snapshot()
    require(allow_dirty or not source["dirty"], "preparation requires committed implementation sources")
    vanilla = method == "vanilla"
    condition = "vanilla" if vanilla else "final_shape" if ablation is None else "ablation"
    if vanilla:
        collapsed = collapse_prompt_only(read_jsonl(blind), location=blind)
        plans = [{**row, "conditioning": row["prompt"]} for row in collapsed.values()]
        artifacts = []
    else:
        require(source_manifest is not None, "guided preparation requires --source-manifest")
        clip, gemma = load_sd14_tokenizer(), load_sana_tokenizer()
        sources, artifacts = load_source_prompts(source_manifest, blind, clip)
        forms = load_noun_forms()
        plans = [compile_plan(prompt_id=s.prompt_id, prompt=s.prompt, sentence=s.sentence,
                              dependencies=s.dependencies, compiled=s.targets, clip_tokenizer=clip,
                              gemma_tokenizer=gemma, instruction=COMPLEX_HUMAN_INSTRUCTION, noun_forms=forms,
                              ablation=ablation) for s in sources]
    artifacts.append({"path": repo_relative(blind), "sha256": sha256_file(blind)})
    plan_path = path.with_suffix(".plans.jsonl")
    atomic_write_jsonl(plan_path, plans, overwrite=False)
    census = dict(Counter((p["source_guidance"]["effective_profile"] if not vanilla else "vanilla") for p in plans))
    contract = SANA_GENERATION
    protocol = {"schema": RUN_SCHEMA, "condition": condition,
                **({"components": ablation.components()} if ablation is not None else {}),
                "method": METHOD_DIRS[condition], "source": source, "dataset": verification,
                "manifest_sha256": sha256_file(manifest),
                "generation_manifest": repo_relative(manifest), "prompt_only": repo_relative(blind),
                "plans": {"path": repo_relative(plan_path), "sha256": sha256_file(plan_path)},
                "source_artifacts": artifacts, "model_id": contract.model_id, "model_revision": contract.model_revision,
                "checkpoint_resolution": contract.checkpoint_resolution,
                "instruction": COMPLEX_HUMAN_INSTRUCTION,
                "runtime": {
                    "execution_mode": contract.execution_mode if not vanilla else "reference",
                    "steps": contract.steps, "guidance_scale": contract.guidance_scale,
                    "height": contract.height, "width": contract.width,
                    "use_resolution_binning": contract.height == contract.checkpoint_resolution,
                    "scale_factor": contract.scale_factor if not vanilla else 0.,
                    "guided_steps": contract.guided_steps if not vanilla else 0,
                    "refinement_steps": list(contract.refinement_steps) if not vanilla else [],
                    "block_range": list(contract.block_range) if not vanilla else [],
                    "model_dtype": contract.model_dtype, "master_dtype": contract.master_dtype,
                    "attention_pooling": "mean_blocks_heads_then_area_16x16" if not vanilla else None,
                    "masked_padding": "trim_after_native_encoding" if not vanilla else "native_300",
                    "region_weight": .25 if not vanilla else 0., "count_weight": .1 if not vanilla else 0.,
                    "count_mass_scale": 77.,
                    "batch_size": contract.batch_size},
                "census": {**census, "count": sum(bool(p.get("counts")) for p in plans),
                           "prefix": sum(p.get("prefix_active", False) for p in plans)}}
    atomic_write_json(path, protocol, overwrite=False)
    return path


def _load(run_id: str, method: str, ablation: T2IAblation | None, allow_dirty: bool) -> tuple[Path, dict[str, Any]]:
    path = protocol_path(run_id, method, ablation)
    protocol = read_json_object(path)
    require(protocol["schema"] == RUN_SCHEMA, "unsupported Sana T2I protocol")
    if source_tree_snapshot() != protocol["source"]:
        raise ValueError("implementation differs from the prepared source")
    require(allow_dirty or not protocol["source"]["dirty"], "generation requires committed sources")
    plan_path = resolve_recorded_path(protocol["plans"]["path"])
    require(sha256_file(plan_path) == protocol["plans"]["sha256"], "compiled plans changed")
    for artifact in protocol["source_artifacts"]:
        verify_artifact_record(artifact)
    manifest = resolve_recorded_path(protocol["generation_manifest"])
    verification = verify_population(manifest, resolve_recorded_path(protocol["prompt_only"]))
    require(sha256_file(manifest) == protocol["manifest_sha256"] and verification == protocol["dataset"],
            "generation population changed")
    return path, protocol


def _select(jobs: list[dict[str, Any]], keys: Sequence[str] | None) -> tuple[list[dict[str, Any]], Any]:
    """The image rows of the selected base prompts (all seeds, so batch membership is unchanged)."""
    if keys is None:
        return jobs, None
    available = {row["prompt_id"].rsplit("_s", 1)[0] for row in jobs}
    require(bool(keys) and len(keys) == len(set(keys)) and set(keys) <= available,
            "prompt selection must contain unique, known base prompt IDs")
    jobs = [row for row in jobs if row["prompt_id"].rsplit("_s", 1)[0] in set(keys)]
    return jobs, {"keys": list(keys), "images": len(jobs)}


def generate_run(  # noqa: C901
    *, run_id: str, method: str = "ccg_diffusion", ablation: T2IAblation | None = None, device: str = "cuda:0",
    keys: Sequence[str] | None = None, output_dir: Path | None = None, allow_dirty: bool = False,
) -> Path:
    """Generate (or resume) the images of a prepared condition; return the output directory.

    ``keys`` (base prompt IDs) restricts the run to those prompts with all ten
    seeds. Images are batched per prompt in manifest order, four seeds per
    physical batch (42-45, 46-49, 50-51). Run Sana in its own process: it must
    not share one with SD1.4's ``configure_determinism``.
    """
    from ccg_diffusion.backbones.sana.guidance import generate_guided
    from ccg_diffusion.backbones.sana.pipeline import encode_texts, generate_stock_from_text, load_sana_pipeline

    path, protocol = _load(run_id, method, ablation, allow_dirty)
    vanilla = protocol["condition"] == "vanilla"
    plans = {p["prompt_id"]: p for p in read_jsonl(resolve_recorded_path(protocol["plans"]["path"]))}
    jobs, selection = _select(read_jsonl(resolve_recorded_path(protocol["generation_manifest"])), keys)
    output = output_dir if output_dir is not None else path.parent / condition_name(method, ablation)
    output.mkdir(parents=True, exist_ok=True)
    selection_path = output / "prompt_selection.json"
    if selection_path.exists() and read_json_object(selection_path) != selection:
        raise ValueError("output directory belongs to a different prompt selection")
    if selection is not None and not selection_path.exists():
        if (output / "generation_protocol.json").exists():
            raise ValueError("cannot add a selection to an existing unselected run")
        atomic_write_json(selection_path, selection, overwrite=False)
    receipt = output / "generation_protocol.json"
    if receipt.exists() and read_json_object(receipt) != protocol:
        raise ValueError("output directory belongs to a different protocol")
    if not receipt.exists():
        atomic_write_json(receipt, protocol, overwrite=False)
    runtime = protocol["runtime"]
    pipe = load_sana_pipeline(device)
    cache = {}
    if not vanilla:
        started = time.monotonic()
        selected = sorted({row["prompt_id"].rsplit("_s", 1)[0] for row in jobs})
        cache = encode_texts(pipe, {key: plans[key]["conditioning"] for key in selected}, device=device,
                             instruction=protocol["instruction"])
        atomic_write_json(output / "text_encoding_summary.json", {
            "prompts": len(cache), "seconds": time.monotonic() - started, "negative_prompt_forward_calls": 1,
        }, overwrite=True)
    completed = {}
    for metadata in (output / protocol["method"]).glob("*/metadata.jsonl"):
        completed.update({row["prompt_id"]: row for row in read_jsonl(metadata)})
    pending: dict[str, list[tuple[dict[str, Any], Job, Path]]] = {}
    for row in jobs:
        key = row["prompt_id"].rsplit("_s", 1)[0]
        job = Job(**row, attention_exist_indices=[], attention_corr_indices=[], attention_leak_indices=[],
                  attention_possession_indices=[], source="sana_compositional")
        destination = image_path(output, protocol["method"], job)
        old = completed.get(row["prompt_id"])
        if old is not None:
            if not destination.is_file() or sha256_file(destination) != old["image_sha256"]:
                raise ValueError("resume image missing or changed")
            continue
        if destination.exists():
            raise ValueError("image exists without authenticated metadata")
        pending.setdefault(key, []).append((row, job, destination))
    position = len(jobs) - sum(len(group) for group in pending.values())
    for key, group in pending.items():
        plan = plans[key]
        for offset in range(0, len(group), runtime["batch_size"]):
            batch = group[offset:offset + runtime["batch_size"]]
            losses = [CompositionalLoss(
                plan, identity=row["prompt_id"], region_weight=runtime["region_weight"],
                count_weight=runtime["count_weight"], count_mass_scale=runtime["count_mass_scale"])
                for row, _, _ in batch] if not vanilla else [None] * len(batch)
            started = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            generators = [torch.Generator(device=device).manual_seed(row["seed"]) for row, _, _ in batch]
            if vanilla:
                images, records = generate_stock_from_text(
                    pipe, plan["conditioning"], generators, device=device, steps=runtime["steps"],
                    guidance_scale=runtime["guidance_scale"], height=runtime["height"], width=runtime["width"],
                    instruction=protocol["instruction"])
            else:
                images, diagnostics = generate_guided(
                    pipe, plan["conditioning"], losses, generators, prompt_encoding=cache[key],
                    instruction=protocol["instruction"], steps=runtime["steps"],
                    guidance_scale=runtime["guidance_scale"], height=runtime["height"], width=runtime["width"],
                    block_range=tuple(range(*runtime["block_range"])), scale_factor=runtime["scale_factor"],
                    guided_steps=runtime["guided_steps"])
                records = [diagnostics.compact(include_trace=True)] * len(batch)
            seconds = time.monotonic() - started
            peak = torch.cuda.max_memory_allocated()
            for generated, loss, record, (row, job, destination) in zip(images, losses, records, batch):
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(".tmp.png")
                generated.save(temporary)
                temporary.replace(destination)
                entry = {**row, "method": protocol["method"], "image_path": repo_relative(destination),
                         "image_sha256": sha256_file(destination), "conditioning": plan["conditioning"],
                         "source_commit": protocol["source"]["commit"], "model_id": protocol["model_id"],
                         "model_revision": protocol["model_revision"], "runtime": runtime, "diagnostics": record,
                         "compositional": None if vanilla else loss.audit_record(), "seconds": seconds / len(batch),
                         "physical_batch_size": len(batch), "peak_memory_bytes": peak}
                with metadata_path(output, protocol["method"], job.experiment).open("a") as stream:
                    stream.write(json.dumps(entry, allow_nan=False) + "\n")
                    stream.flush()
            position += len(batch)
            print(json.dumps({"completed": position, "total": len(jobs), "prompt_id": key,
                              "route": "vanilla" if vanilla else losses[0].route,
                              "seconds_per_image": seconds / len(batch),
                              "physical_batch_size": len(batch), "peak_gb": peak / 1e9}), flush=True)
    atomic_write_json(output / "complete.json", {"images": len(jobs), "protocol_sha256": sha256_file(path)},
                      overwrite=True)
    return output


def audit_run(
    *, run_id: str, method: str = "ccg_diffusion", ablation: T2IAblation | None = None,
    keys: Sequence[str] | None = None, output_dir: Path | None = None,
) -> dict[str, Any]:
    """Every selected prompt has one metadata row per seed 42-51, and every recorded image has its sha256."""
    path = protocol_path(run_id, method, ablation)
    protocol = read_json_object(path)
    plans = {p["prompt_id"] for p in read_jsonl(resolve_recorded_path(protocol["plans"]["path"]))}
    jobs, _ = _select(read_jsonl(resolve_recorded_path(protocol["generation_manifest"])), keys)
    output = output_dir if output_dir is not None else path.parent / condition_name(method, ablation)
    rows: dict[str, dict[str, Any]] = {}
    for metadata in sorted((output / protocol["method"]).glob("*/metadata.jsonl")):
        for row in read_jsonl(metadata):
            require(row["prompt_id"] not in rows, f"duplicate metadata row: {row['prompt_id']}")
            rows[row["prompt_id"]] = row
    require(set(rows) == {row["prompt_id"] for row in jobs}, "generated population is incomplete or has extra images")
    seeds: dict[str, set[int]] = {}
    for row in jobs:
        key = row["prompt_id"].rsplit("_s", 1)[0]
        require(key in plans, f"image without a plan: {row['prompt_id']}")
        recorded = rows[row["prompt_id"]]
        require(all(recorded[field] == row[field] for field in ("experiment", "prompt", "seed")), "prompt/seed differs")
        image = resolve_recorded_path(recorded["image_path"])
        require(image.is_file() and sha256_file(image) == recorded["image_sha256"],
                f"recorded image missing or changed: {recorded['image_path']}")
        seeds.setdefault(key, set()).add(row["seed"])
    require(all(len(values) == SEEDS_PER_PROMPT for values in seeds.values()), "a prompt lacks some of its ten seeds")
    return {"status": "PASS", "run_id": run_id, "condition": condition_name(method, ablation),
            "output_dir": repo_relative(output), "prompts": len(seeds), "images": len(rows)}
