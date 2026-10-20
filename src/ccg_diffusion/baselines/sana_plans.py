"""Sana 0.6B adaptations of the five comparison methods: plans, objectives and the T2I run protocol.

None of Composable Diffusion, StructureDiffusion, Attend-and-Excite, SynGen and
Predicated Diffusion provides a Sana implementation; each is adapted here. The
methods reuse the SD1.4 blind preprocessing (``baselines.preprocessing`` and the
``sd14_pd_spacy`` input file), but their CLIP indices are only source
coordinates: every target is rebuilt from character spans on Sana's selected
Gemma token axis (``compile_plan``). ``sample_loss`` evaluates each method's own
objective on Sana's 16x16 cross-attention maps; the GPU loop is in
``baselines.sana_generation``.

``prepare_t2i_run`` checks the five input files against their metadata and
the T2I-CompBench++ population and writes ``protocol.json`` and
``protocol.plans.jsonl`` (one plan per method and base prompt).
diffusers is never imported here.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from ccg_diffusion.backbones.sana.pipeline import COMPLEX_HUMAN_INSTRUCTION
from ccg_diffusion.backbones.sana.tokens import normalise_token, project_target_row, selected_token_offsets
from ccg_diffusion.baselines import attend_and_excite, syngen
from ccg_diffusion.baselines.composable_diffusion import split_prompt
from ccg_diffusion.baselines.registry import SOURCE_COMMIT_BY_ARM
from ccg_diffusion.config import SANA_GENERATION, T2I_COMPBENCH_DATASET_DIR
from ccg_diffusion.guidance.predicated_diffusion_loss import compute_pd_loss
from ccg_diffusion.provenance import (
    atomic_write_json,
    atomic_write_jsonl,
    canonical_json,
    portable_path,
    read_jsonl,
    repository_root,
    resolve_recorded_path,
    sha256_file,
    sha256_text,
    source_tree_snapshot,
)
from ccg_diffusion.targets.pd_indices import PDIndices

METHODS = ("sana_composable", "sana_structure", "sana_attend_excite", "sana_syngen", "sana_pd_spacy")
SCHEMA = "holsd.sana_five_baselines_plan.v1"
# Map the SD 0/10/20 thresholds over its 25-step intervention window to
# Sana's 20-step intervention window.
AE_THRESHOLDS = {0: .05, 8: .5, 16: .8}
RUN_SCHEMA = "ccg_diffusion.sana_baselines_run.v1"
# The Sana T2I run whose CCG Diffusion images share the initial latents of the baseline images.
PRIMARY_REFERENCE_RUN = "20260919_sana06_native512_full_v1"
# Fixed generation settings of the five Sana baselines (recorded in the protocol).
T2I_RUNTIME = {"steps": 20, "guided_steps": 20, "guidance_scale": 4.5, "height": 512, "width": 512,
               "model_dtype": "float16", "master_dtype": "float32", "batch_size": 4,
               "tail_batch_size": 2, "block_range": [0, 8], "attention_res": 16,
               "pd_scale": 20., "pd_cap": .02, "pd_step0_extra_refinements": 5,
               "ae_thresholds": AE_THRESHOLDS, "ae_max_refinements": 20,
               "ae_selection": "Gemma_first_timestep_seed42_51_mean_smoothed_max",
               "ae_syngen_update_cap": None, "syngen_scale": 20.,
               "conditioning": "native_encode_prompt_with_fixed_instruction"}


def span_tokens(
    prompt: str, span: Sequence[int], offsets: Mapping[int, tuple[int, int]],
    decoded: Mapping[int, str],
) -> list[int]:
    """Find a complete lexical occurrence, never a same-spelling substitute."""
    start, end = span
    if not 0 <= start < end <= len(prompt):
        raise ValueError("invalid source character span")
    indices = [i for i, (a, b) in sorted(offsets.items()) if min(b, end) > max(a, start)]
    if not indices or normalise_token("".join(decoded[i] for i in indices)) != normalise_token(prompt[start:end]):
        raise ValueError(f"incomplete Gemma alignment for {prompt[start:end]!r}")
    return indices


def compile_plan(
    method: str, prompt: str, source: Mapping[str, Any], *, tokenizer: Any,
    clip_tokenizer: Any, instruction: Any,
) -> dict[str, Any]:
    """Compile semantic source records without reusing their runtime token IDs."""
    if method not in METHODS:
        raise ValueError(f"unsupported Sana method: {method}")
    original = source.get("plan", {})
    conditioning = original["method_prompt"] if method == "sana_structure" else prompt
    decoded, offsets = selected_token_offsets(
        tokenizer, conditioning, max_sequence_length=300, instruction=instruction)
    if not offsets:
        raise ValueError("empty Sana lexical input")
    plan: dict[str, Any] = {
        "schema": SCHEMA, "method": method, "prompt": prompt, "conditioning": conditioning,
        "source_sha256": sha256_text(canonical_json(source)), "text_token_count": max(offsets) + 1,
        "lexical_indices": sorted(offsets), "token_offsets": {str(k): list(v) for k, v in offsets.items()},
        "status": "ok", "fallback_reason": None,
    }
    if method == "sana_composable":
        plan["components"] = list(split_prompt(prompt).components)
        plan["weights"] = [SANA_GENERATION.guidance_scale] * len(plan["components"])
    elif method == "sana_attend_excite":
        plan["targets"] = [
            {"text": target["text"], "character_span": target["character_span"],
             "candidates": span_tokens(prompt, target["character_span"], offsets, decoded)}
            for target in original["targets"]]
        if not plan["targets"]:
            plan.update(status="fallback", fallback_reason="no_noun_target")
    elif method == "sana_syngen":
        collections = []
        for extractor in ("standard", "verb_root", "with_verbs"):
            groups = []
            for group in original["source_groups"]:
                if group["extractor"] != extractor:
                    continue
                for text, (start, end) in zip(group["texts"], group["character_spans"], strict=True):
                    if prompt[start:end] != text:
                        raise ValueError("SynGen parser occurrence differs from source prompt")
                members = [span_tokens(prompt, span, offsets, decoded) for span in group["character_spans"]]
                groups.append([member[0] if len(member) == 1 else member for member in members])
            collections.append(groups)
        unified = syngen.official_unify_lists(collections)
        plan["bindings"] = [[member if isinstance(member, list) else [member] for member in group]
                            for group in unified if len(group) >= 2]
        if not plan["bindings"]:
            plan.update(status="fallback", fallback_reason="no_binding_pair")
    elif method == "sana_pd_spacy":
        projected = project_target_row(
            prompt_id=source["prompt_id"], prompt=prompt, source_row=source, tokenizer=tokenizer,
            max_sequence_length=300, instruction=instruction,
            fallback_statuses=("alignment_error", "empty_pd"))
        plan["projection"] = projected
        plan["pd_indices"] = projected["sana_pd_indices"]
        if not PDIndices(**plan["pd_indices"]).has_any():
            plan.update(status="fallback", fallback_reason=projected["fallback_reason"] or "empty_pd")
    else:
        # CLIP spans locate source character ranges only. Gemma phrase inputs
        # and target ranges are then aligned independently at the character level.
        encoded = clip_tokenizer(conditioning, add_special_tokens=False, return_offsets_mapping=True)
        clip_offsets = encoded["offset_mapping"]
        phrases = []
        for text, (start, end) in zip(original["conditioning_texts"][1:], original["token_spans"][1:], strict=True):
            char_start, char_end = clip_offsets[start][0], clip_offsets[end - 1][1]
            original_text = conditioning[char_start:char_end]
            if "".join(text.split()) != "".join(original_text.split()):
                raise ValueError("Structure phrase no longer matches its source span")
            pd, po = selected_token_offsets(
                tokenizer, text, max_sequence_length=300, instruction=instruction)
            correspondence = []
            for index, (a, b) in sorted(offsets.items()):
                if not min(b, char_end) > max(a, char_start):
                    continue
                left = len("".join(conditioning[char_start:max(a, char_start)].split()))
                right = len("".join(conditioning[char_start:min(b, char_end)].split()))
                pieces = [i for i, (c, d) in sorted(po.items())
                          if min(len("".join(text[:d].split())), right)
                          > max(len("".join(text[:c].split())), left)]
                if not pieces:
                    raise ValueError("Structure phrase has an unmapped Gemma piece")
                correspondence.append([index, pieces])
            phrases.append({"text": text, "character_span": [char_start, char_end],
                            "mapping": correspondence, "text_token_count": max(po) + 1})
        plan.update(phrases=phrases, conjunction=original["conjunction"])
        if not original["np_texts"]:
            plan.update(status="fallback", fallback_reason="no_np")
    return plan


def ae_maps(maps: torch.Tensor, plan: Mapping[str, Any]) -> torch.Tensor:
    """Present the shared A&E loss with BOS/prompt/trailing-special layout.

    Sana has a leading BOS and no trailing EOS. Explicit zero sentinels are
    excluded by the shared kernel; masked padding and instruction residue do
    not enter its lexical softmax.
    """
    lexical = maps[..., plan["lexical_indices"]]
    sentinel = torch.zeros_like(lexical[..., :1])
    return torch.cat([sentinel, lexical, sentinel], dim=-1)


def ae_maxima(maps: torch.Tensor, plan: Mapping[str, Any], indices: Sequence[int]) -> tuple[torch.Tensor, ...]:
    lookup = {raw: index + 1 for index, raw in enumerate(plan["lexical_indices"])}
    return attend_and_excite.max_attention(ae_maps(maps, plan), [lookup[i] for i in indices])


def sample_loss(maps: torch.Tensor, plan: Mapping[str, Any], selected: Sequence[int] = ()) -> torch.Tensor:
    """Retain each method's objective and sample-local gradient magnitude."""
    if plan["method"] == "sana_attend_excite":
        lookup = {raw: index + 1 for index, raw in enumerate(plan["lexical_indices"])}
        return attend_and_excite.sample_loss(ae_maps(maps, plan), [lookup[i] for i in selected])
    if plan["method"] == "sana_syngen":
        syngen_plan = syngen.SynGenPlan(
            prompt=plan["prompt"], bindings=tuple(
                syngen.BindingGroup(tuple(tuple(m) for m in group)) for group in plan["bindings"]),
            attention_token_indices=tuple(plan["lexical_indices"]), source_groups=(), status="ok")
        # The unchanged validator reserves a trailing special slot. Append a
        # zero sentinel that is never used by the SynGen pair/negative lists.
        return syngen.sample_loss(torch.cat([maps, torch.zeros_like(maps[..., :1])], -1), syngen_plan)
    if plan["method"] == "sana_pd_spacy":
        return compute_pd_loss(maps, PDIndices(**plan["pd_indices"]), attention_res=16,
                               special_layout="leading_bos")
    raise ValueError("method does not use an attention objective")


# --- T2I-CompBench++ run protocol -------------------------------------------------------------------------


def source_path(input_dir: Path, method: str) -> Path:
    """The SD1.4 input file a Sana baseline adapts (``sana_<m>`` reads ``sd14_<m>``)."""

    old = method.replace("sana_", "sd14_", 1)
    parent = input_dir if method == "sana_pd_spacy" else input_dir / "external_sidecars"
    return parent / (old + ".jsonl")


def load_tokenizers(*, local_files_only: bool = True) -> tuple[Any, Any]:
    """Sana's Gemma tokenizer and the SD1.4 ``CLIPTokenizerFast`` (character offsets for StructureDiffusion)."""

    from ccg_diffusion.backbones.sana.pipeline import load_sana_tokenizer
    from ccg_diffusion.targets.tokenizer import load_sd14_tokenizer

    return load_sana_tokenizer(local_files_only=local_files_only), load_sd14_tokenizer(
        fast=True, local_files_only=local_files_only)


def prepare_t2i_run(
    *,
    run_dir: Path,
    input_dir: Path,
    dataset_verification: Mapping[str, Any],
    allow_dirty: bool = False,
    local_files_only: bool = True,
) -> dict[str, Any]:
    """Compile the Sana plans of the five baselines and write ``protocol.json`` + ``protocol.plans.jsonl``.

    ``input_dir`` holds the SD1.4 inputs (``external_sidecars/sd14_*.jsonl``
    and ``sd14_pd_spacy.jsonl``); each is checked against its ``.meta.json``,
    its source commit and the T2I-CompBench++ population. One plan is compiled
    per method and base prompt. ``dataset_verification`` is the record of the
    full-dataset verifier.
    """

    dataset = repository_root() / T2I_COMPBENCH_DATASET_DIR
    manifest, blind_manifest = dataset / "independent_seed_manifest.jsonl", dataset / "prompt_only.jsonl"
    jobs = read_jsonl(manifest)
    blind = {row["prompt_id"]: row for row in read_jsonl(blind_manifest)}
    expected = {row["prompt_id"]: row for row in jobs}
    if len(expected) != len(jobs) or set(expected) != set(blind):
        raise ValueError("duplicate or mismatched blind/generation population")
    source = source_tree_snapshot()
    if source["dirty"] and not allow_dirty:
        raise ValueError("production preparation requires committed sources (or --allow-dirty)")
    tokenizer, clip = load_tokenizers(local_files_only=local_files_only)
    artifacts, plans, failures = [], [], []
    for method in METHODS:
        path = source_path(input_dir, method)
        meta_path = Path(str(path) + ".meta.json")
        meta = json.loads(meta_path.read_text())
        if meta["output"]["sha256"] != sha256_file(path):
            raise ValueError(f"source sidecar hash differs: {path}")
        old_method = method.replace("sana_", "sd14_", 1)
        if method != "sana_pd_spacy":
            if (meta["method"] != old_method or meta["source_commit"] != SOURCE_COMMIT_BY_ARM[old_method]
                    or meta["gold_inputs_used"] is not False
                    or meta["generation_manifest"]["sha256"] != sha256_file(manifest)
                    or meta["blind_prompt_manifest"]["sha256"] != sha256_file(blind_manifest)):
                raise ValueError("source method or blind population mismatch")
        artifacts.extend({"path": portable_path(p), "sha256": sha256_file(p)} for p in (path, meta_path))
        seen, bases = set(), {}
        for row in read_jsonl(path):
            key = row["prompt_id"]
            if key in seen or key not in expected:
                raise ValueError("duplicate or unknown source row")
            seen.add(key)
            prompt = expected[key]["prompt"]
            if blind[key]["prompt"] != prompt or blind[key]["seed"] != 0:
                raise ValueError("source manifest is not blind")
            if method == "sana_pd_spacy":
                if row["provider"] != "spacy_sd14" or row["prompt_sha256"] != sha256_text(prompt):
                    raise ValueError("spaCy source identity differs")
            elif (row["prompt"] != prompt or row["method"] != old_method
                  or row["seed"] != expected[key]["seed"] or row["experiment"] != expected[key]["experiment"]):
                raise ValueError("source row differs from paired manifest")
            base = key.rsplit("_s", 1)[0]
            semantic = {k: v for k, v in row.items() if k not in {"prompt_id", "seed"}}
            digest = sha256_text(canonical_json(semantic))
            if base in bases:
                if bases[base] != digest:
                    raise ValueError("semantic preprocessing varies across seeds")
                continue
            bases[base] = digest
            try:
                plan = compile_plan(method, prompt, row, tokenizer=tokenizer, clip_tokenizer=clip,
                                    instruction=COMPLEX_HUMAN_INSTRUCTION)
                plans.append({**plan, "prompt_id": base, "experiment": expected[key]["experiment"]})
            except (ValueError, KeyError, IndexError) as exc:
                failures.append({"method": method, "prompt_id": base, "error": str(exc)})
        if seen != set(expected) or len(seen) != meta["row_count"]:
            raise ValueError("source sidecar does not cover the entire population")
    protocol_path = run_dir / "protocol.json"
    if failures:
        atomic_write_json(protocol_path.with_suffix(".errors.json"), failures, overwrite=True)
        raise ValueError(f"{len(failures)} native plan compilation failures; inspect .errors.json")
    plan_path = protocol_path.with_suffix(".plans.jsonl")
    atomic_write_jsonl(plan_path, plans, overwrite=False)
    protocol = {
        "schema": RUN_SCHEMA, "source": source,
        "dataset": T2I_COMPBENCH_DATASET_DIR, "dataset_verification": dict(dataset_verification),
        "manifest_sha256": sha256_file(manifest), "blind_sha256": sha256_file(blind_manifest),
        "source_artifacts": artifacts,
        "plans": {"path": portable_path(plan_path), "sha256": sha256_file(plan_path)},
        "model_id": SANA_GENERATION.model_id, "model_revision": SANA_GENERATION.model_revision,
        "instruction": COMPLEX_HUMAN_INSTRUCTION, "runtime": T2I_RUNTIME,
        "primary_reference_run": PRIMARY_REFERENCE_RUN,
        "census": dict(Counter(p["method"] + ":" + p["status"] for p in plans)),
    }
    atomic_write_json(protocol_path, protocol, overwrite=False)
    return protocol


def authenticate_t2i_protocol(protocol_path: Path, *, allow_dirty: bool = False) -> dict[str, Any]:
    """Load a run protocol and require the pinned model, runtime, source tree, plans and dataset bytes."""

    protocol = json.loads(protocol_path.read_text())
    if (protocol["schema"] != RUN_SCHEMA
            or protocol["model_id"] != SANA_GENERATION.model_id
            or protocol["model_revision"] != SANA_GENERATION.model_revision
            or protocol["instruction"] != COMPLEX_HUMAN_INSTRUCTION):
        raise ValueError("protocol does not describe the pinned Sana baseline model")
    runtime = protocol["runtime"]
    fixed = {"steps": 20, "guided_steps": 20, "guidance_scale": 4.5, "height": 512, "width": 512,
             "model_dtype": "float16", "master_dtype": "float32", "batch_size": 4, "tail_batch_size": 2,
             "block_range": [0, 8], "attention_res": 16, "pd_scale": 20., "pd_cap": .02,
             "pd_step0_extra_refinements": 5, "ae_max_refinements": 20,
             "ae_syngen_update_cap": None, "syngen_scale": 20.}
    if (any(runtime[k] != v for k, v in fixed.items())
            or runtime["ae_thresholds"] != {str(k): v for k, v in AE_THRESHOLDS.items()}):
        raise ValueError("protocol runtime differs from the implemented comparison")
    if protocol["source"] != source_tree_snapshot():
        raise ValueError("source differs from prepared protocol")
    if protocol["source"]["dirty"] and not allow_dirty:
        raise ValueError("the protocol was prepared from uncommitted sources; pass --allow-dirty")
    for artifact in [protocol["plans"], *protocol["source_artifacts"]]:
        if sha256_file(resolve_recorded_path(artifact["path"])) != artifact["sha256"]:
            raise ValueError(f"changed source artifact: {artifact['path']}")
    dataset = resolve_recorded_path(protocol["dataset"])
    if (sha256_file(dataset / "independent_seed_manifest.jsonl") != protocol["manifest_sha256"]
            or sha256_file(dataset / "prompt_only.jsonl") != protocol["blind_sha256"]):
        raise ValueError("changed dataset")
    return protocol
