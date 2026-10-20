"""Sana 0.6B execution of the five comparison methods and the T2I-CompBench++ baseline run.

``SanaBaselineRuntime`` runs one plan of ``baselines.sana_plans`` for a batch of
seeds with the method's own update rule on Sana's cross-attention maps
(blocks [0, 8), 16x16):

* Attend-and-Excite: smoothed-max loss, step ``20 * sqrt(linspace(1, 0.5, 20)[t])``,
  up to 20 refinements at steps 0/8/16, the Gemma subtoken chosen over seeds 42-51;
* SynGen: symmetric-KL binding loss with the constant step 20;
* Predicated Diffusion: L_PD, step as A&E, 5 extra updates at step 0 and the
  per-image relative-L2 cap 0.02;
* Composable Diffusion: linear composition of the component flow predictions;
* StructureDiffusion: phrase-averaged cross-attention values (``StructureProcessor``).

These updates sum the per-image losses and are not the capped
``guidance.latent_update`` of CCG Diffusion. ``generate_t2i_run`` writes one
method's images and metadata for one category of the T2I-CompBench++ run
(batches of seeds 42-45, 46-49, 50-51 per prompt); ``audit_t2i_category``
checks a finished category against the protocol and the paired CCG Diffusion
run. The TAB Predicated Diffusion baselines (``baselines.tab_sana``) reuse the
runtime.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from ccg_diffusion.backbones.jobs import Job, image_path, metadata_path
from ccg_diffusion.backbones.sana.attention import (
    SanaAttentionStore, SanaCaptureComplete, register_sana_attention_store, aggregate_sana_attention,
)
from ccg_diffusion.backbones.sana.pipeline import (
    DETERMINISM_MESSAGE,
    VaeDecodeRecord,
    decode_latents,
    encode_texts,
    initial_latents,
    load_sana_pipeline,
    tensor_sha256,
)
from ccg_diffusion.baselines.sana_plans import AE_THRESHOLDS, ae_maxima, authenticate_t2i_protocol, sample_loss
from ccg_diffusion.config import SANA_GENERATION as CONTRACT
from ccg_diffusion.provenance import (
    atomic_write_json,
    canonical_json,
    portable_path,
    read_jsonl,
    repository_root,
    resolve_recorded_path,
    sha256_file,
    sha256_text,
)


def finite(value: torch.Tensor, label: str) -> torch.Tensor:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"nonfinite Sana {label}")
    return value


def update_latents(
    latents: torch.Tensor, losses: Sequence[torch.Tensor], step_size: float,
    active: Sequence[bool], *, cap: float | None,
) -> tuple[torch.Tensor, list[float]]:
    """Sum independent sample losses; bound only PD, never A&E or SynGen."""
    indices = [i for i, enabled in enumerate(active) if enabled and losses[i].requires_grad]
    if not indices:
        return latents.detach(), [0.] * len(active)
    gradient = torch.autograd.grad(sum(losses[i] for i in indices), latents)[0].float()
    finite(gradient, "guidance gradient")
    mask = torch.tensor(active, device=latents.device).reshape(-1, 1, 1, 1)
    delta = gradient * step_size * mask
    norm = latents.detach().float().flatten(1).norm(dim=1).clamp_min(1e-12)
    ratio = delta.flatten(1).norm(dim=1) / norm
    if cap is not None:
        delta = delta * (cap / ratio.clamp_min(1e-12)).clamp(max=1).reshape(-1, 1, 1, 1)
    return finite(latents.detach().float() - delta, "updated latent"), ratio.detach().cpu().tolist()


class StructureProcessor:
    """Average phrase-conditioned cross-attention while preserving unconditional CFG."""

    def __init__(self, original: Any, keys: torch.Tensor, values: torch.Tensor, batch: int) -> None:
        self.original, self.keys, self.values, self.batch = original, keys, values, batch
        self.cached: list[tuple[torch.Tensor, torch.Tensor]] | None = None

    def __call__(self, attn: Any, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor,
                 attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch = self.batch
        if hidden_states.shape[0] != 2 * batch:
            raise ValueError("Structure processor requires [unconditional, conditional] CFG batches")
        unconditional = self.original(attn, hidden_states[:batch], encoder_hidden_states[:batch],
                                      attention_mask=None if attention_mask is None else attention_mask[:batch])
        query = attn.to_q(hidden_states[batch:])
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        query = query.view(batch, -1, attn.heads, query.shape[-1] // attn.heads).transpose(1, 2)
        if self.cached is None:
            self.cached = []
            for index in range(max(len(self.keys), len(self.values))):
                key = attn.to_k(self.keys[min(index, len(self.keys) - 1):min(index, len(self.keys) - 1) + 1])
                value = attn.to_v(self.values[min(index, len(self.values) - 1):min(index, len(self.values) - 1) + 1])
                if attn.norm_k is not None:
                    key = attn.norm_k(key)
                self.cached.append((key.view(1, -1, attn.heads, query.shape[-1]).transpose(1, 2),
                                    value.view(1, -1, attn.heads, query.shape[-1]).transpose(1, 2)))
        outputs = []
        for key, value in self.cached:
            score = query.float() @ key.transpose(-1, -2).float() * attn.scale
            if attention_mask is not None:
                bias = attn.prepare_attention_mask(attention_mask[batch:], key.shape[-2], batch)
                score = score + bias.view(batch, attn.heads, -1, key.shape[-2]).float()
            outputs.append(score.softmax(-1).to(value.dtype) @ value)
        combined = torch.stack(outputs).float().mean(0).to(query.dtype)
        combined = combined.transpose(1, 2).reshape(batch, -1, attn.heads * query.shape[-1])
        conditional = attn.to_out[1](attn.to_out[0](combined)) / attn.rescale_output_factor
        return torch.cat([unconditional, conditional])


class SanaBaselineRuntime:
    """Own processors and immutable encodings for a single prompt invocation."""

    def __init__(self, pipe: Any, plan: Mapping[str, Any], encodings: Mapping[str, tuple[torch.Tensor, ...]],
                 *, device: str) -> None:
        self.pipe, self.plan, self.device = pipe, plan, device
        self.encodings = encodings
        self.store = SanaAttentionStore()
        self.store.selected_blocks = set(range(8))
        self.store.stop_after = 7
        # Cache the text keys/values of the evaluated FP32 attention path.
        self.store.cache_text_kv = True

    def encoding(self, text: str, batch: int, *, trim: bool = True) -> tuple[torch.Tensor, ...]:
        values = [v.to(self.device) for v in self.encodings[text]]
        length = max(int(values[1].sum()), int(values[3].sum())) if trim else values[0].shape[1]
        return tuple(v[:, :length].repeat(batch, 1, 1) if v.ndim == 3
                     else v[:, :length].repeat(batch, 1) for v in values)

    def forward(self, latents: torch.Tensor, timestep: torch.Tensor,
                encoding: tuple[torch.Tensor, ...], *, cfg: bool) -> torch.Tensor:
        positive, mask, negative, negative_mask = encoding
        model_input = torch.cat([latents] * 2) if cfg else latents
        embeddings = torch.cat([negative, positive]) if cfg else positive
        masks = torch.cat([negative_mask, mask]) if cfg else mask
        prediction = self.pipe.transformer(
            model_input.to(self.pipe.transformer.dtype),
            encoder_hidden_states=embeddings.to(self.pipe.transformer.dtype),
            encoder_attention_mask=masks,
            timestep=timestep.expand(len(model_input)).to(latents.dtype) * self.pipe.transformer.config.timestep_scale,
            return_dict=False)[0].float()
        finite(prediction, "prediction")
        if cfg:
            unconditional, conditional = prediction.chunk(2)
            prediction = unconditional + CONTRACT.guidance_scale * (conditional - unconditional)
        return prediction

    def capture(self, latents: torch.Tensor, timestep: torch.Tensor,
                encoding: tuple[torch.Tensor, ...]) -> torch.Tensor:
        self.store.reset()
        self.store.enabled = True
        try:
            self.forward(latents, timestep, encoding, cfg=False)
        except SanaCaptureComplete:
            pass
        if set(self.store.attention_store) != {f"block_{i}" for i in range(8)}:
            raise RuntimeError("incomplete Sana attention capture")
        return aggregate_sana_attention(self.store, range(8))[..., :self.plan["text_token_count"]].float()

    def select_targets(self) -> dict[str, Any]:
        """Freeze A&E Gemma subtoken choice using all ten initial seeds."""
        candidates = [target["candidates"] for target in self.plan["targets"]]
        if all(len(indices) == 1 for indices in candidates):
            return {"selected": [indices[0] for indices in candidates],
                    "measurements": [], "seeds": list(range(42, 52))}
        original = dict(self.pipe.transformer.attn_processors)
        measurements, initial_hashes = [], []
        self.store.cache_text_kv = False
        try:
            register_sana_attention_store(self.pipe.transformer, self.store)
            self.pipe.scheduler.set_timesteps(CONTRACT.steps, device=self.device)
            for start in range(42, 52, 4):
                seeds = list(range(start, min(start + 4, 52)))
                latent = initial_latents(self.pipe, seeds, self.device)
                initial_hashes.extend(tensor_sha256(row[None]) for row in latent)
                with torch.no_grad():
                    encoding = self.encoding(self.plan["conditioning"], len(seeds))
                    maps = self.capture(latent, self.pipe.scheduler.timesteps[0], encoding)
                    measurements.extend([[[float(x) for x in ae_maxima(row, self.plan, indices)]
                                          for indices in candidates] for row in maps])
            means = [[sum(row[t][p] for row in measurements) / 10 for p in range(len(indices))]
                     for t, indices in enumerate(candidates)]
            return {"selected": [indices[max(range(len(indices)), key=lambda i: (mean[i], -indices[i]))]
                                 for indices, mean in zip(candidates, means)],
                    "measurements": measurements, "means": means, "seeds": list(range(42, 52)),
                    "initial_latent_sha256": initial_hashes}
        finally:
            self.pipe.transformer.set_attn_processor(original)
            self.store.reset()
            self.store.cache_text_kv = True

    def structure_context(self, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        full = self.encodings[self.plan["conditioning"]][0].to(self.device)[:, :length]
        aligned, combined = [full], full.clone()
        for phrase in self.plan["phrases"]:
            embedding = self.encodings[phrase["text"]][0].to(self.device)
            context = full.clone()
            for target, indices in phrase["mapping"]:
                replacement = embedding[:, indices].float().mean(1).to(full.dtype)
                context[:, target] = replacement
                combined[:, target] = replacement
            aligned.append(context)
        keys = torch.cat(aligned) if self.plan["conjunction"] else full
        values = combined if self.plan["conjunction"] else torch.cat(aligned)
        project = self.pipe.transformer.caption_projection
        norm = self.pipe.transformer.caption_norm
        return norm(project(keys)), norm(project(values))

    def generate(self, seeds: Sequence[int], *, selected: Sequence[int] = (),
                 output_type: str = "pil") -> tuple[Any, list[dict[str, Any]]]:
        assert not torch.are_deterministic_algorithms_enabled(), DETERMINISM_MESSAGE
        method, batch = self.plan["method"], len(seeds)
        original = dict(self.pipe.transformer.attn_processors)
        latent = initial_latents(self.pipe, seeds, self.device)
        records = [{"initial_latent_sha256": tensor_sha256(row[None]), "updates": 0,
                    "refinements": 0, "max_update_ratio_before_cap": 0., "loss_trace": []} for row in latent]
        guided = self.plan["status"] == "ok" and method in {"sana_attend_excite", "sana_syngen", "sana_pd_spacy"}
        encoding = self.encoding(self.plan["conditioning"], batch, trim=guided or method == "sana_structure")
        if int(encoding[1][0].sum()) != self.plan["text_token_count"]:
            raise ValueError("prepared Gemma token layout differs from runtime encoding")
        self.pipe.scheduler.set_timesteps(CONTRACT.steps, device=self.device)
        try:
            if guided:
                register_sana_attention_store(self.pipe.transformer, self.store)
            elif method == "sana_structure" and self.plan["status"] == "ok":
                with torch.no_grad():
                    keys, values = self.structure_context(encoding[0].shape[1])
                self.pipe.transformer.set_attn_processor({
                    name: StructureProcessor(proc, keys, values, batch) if name.endswith("attn2.processor") else proc
                    for name, proc in original.items()})
            for step, timestep in enumerate(self.pipe.scheduler.timesteps):
                if guided:
                    latent = latent.detach().requires_grad_(True)
                    maps = self.capture(latent, timestep, encoding)
                    losses = [sample_loss(row, self.plan, selected) for row in maps]
                    dose = 20. * math.sqrt(float(torch.linspace(1., .5, CONTRACT.steps)[step]))
                    if method == "sana_syngen":
                        dose = 20.  # Official SynGen uses a constant step size.
                    if method == "sana_attend_excite" and step in AE_THRESHOLDS:
                        for _ in range(20):
                            active = [float(loss.detach()) > 1. - AE_THRESHOLDS[step] for loss in losses]
                            if not any(active):
                                break
                            latent, ratios = update_latents(latent, losses, dose, active, cap=None)
                            for i, enabled in enumerate(active):
                                records[i]["refinements"] += int(enabled)
                            latent.requires_grad_(True)
                            maps = self.capture(latent, timestep, encoding)
                            losses = [sample_loss(row, self.plan, selected) for row in maps]
                    repeats = 6 if method == "sana_pd_spacy" and step == 0 else 1
                    for repeat in range(repeats):
                        if repeat:
                            latent = latent.detach().requires_grad_(True)
                            maps = self.capture(latent, timestep, encoding)
                            losses = [sample_loss(row, self.plan, selected) for row in maps]
                        latent, ratios = update_latents(
                            latent, losses, dose, [True] * batch, cap=.02 if method == "sana_pd_spacy" else None)
                        for i, loss in enumerate(losses):
                            records[i]["loss_trace"].append(float(loss.detach()))
                            records[i]["updates"] += 1
                            records[i]["max_update_ratio_before_cap"] = max(
                                records[i]["max_update_ratio_before_cap"], ratios[i])
                    del maps, losses
                    self.store.reset()
                    self.store.enabled = False
                with torch.no_grad():
                    if method == "sana_composable":
                        # Apply the published linear composition to the flow
                        # prediction expected by Sana's scheduler.
                        components = self.plan["components"]
                        if len(components) == 1:
                            component = self.encoding(components[0], batch, trim=False)
                            prediction = self.forward(latent, timestep, component, cfg=True)
                        else:
                            p, m, n, nm = encoding
                            unconditional = self.forward(latent, timestep, (n, nm, n, nm), cfg=False)
                            prediction = unconditional.clone()
                            for text, weight in zip(components, self.plan["weights"], strict=True):
                                encoding_component = self.encoding(text, batch, trim=False)
                                component = self.forward(latent, timestep, encoding_component, cfg=False)
                                prediction += weight * (component - unconditional)
                    else:
                        prediction = self.forward(latent, timestep, encoding, cfg=True)
                    next_latent = self.pipe.scheduler.step(prediction, timestep, latent.detach(), return_dict=False)[0]
                    latent = finite(next_latent.float(), "scheduler output")
            for record, row in zip(records, latent):
                record["final_latent_sha256"] = tensor_sha256(row[None])
            if output_type == "latent":
                return latent, records
            with torch.no_grad():
                diagnostics = VaeDecodeRecord()
                images = decode_latents(self.pipe, latent, diagnostics)
                images = self.pipe.image_processor.resize_and_crop_tensor(images, 512, 512)
                images = self.pipe.image_processor.postprocess(images, output_type="pil")
            for record in records:
                record["vae_decode_fallback"] = diagnostics.vae_decode_fallback
            return images, records
        finally:
            self.pipe.transformer.set_attn_processor(original)
            self.store.reset()


# --- T2I-CompBench++ run ----------------------------------------------------------------------------------


def generate_t2i_run(  # noqa: C901
    *,
    protocol: Path,
    method: str,
    output: Path,
    device: str = "cuda:0",
    categories: Sequence[str] | None = None,
    keys: Sequence[str] | None = None,
    allow_dirty: bool = False,
) -> None:
    """Generate (or resume) the images of one method for the selected base prompts.

    ``categories`` and ``keys`` (base prompt IDs) select whole prompts; every
    prompt runs seeds 42-51 in batches of 4, 4 and 2. Images and metadata go
    to ``image_path(output, method, job)`` and ``<output>/<method>/<experiment>/metadata.jsonl``.
    """

    protocol_record = authenticate_t2i_protocol(protocol, allow_dirty=allow_dirty)
    dataset = resolve_recorded_path(protocol_record["dataset"])
    jobs = read_jsonl(dataset / "independent_seed_manifest.jsonl")
    plans = {p["prompt_id"]: p for p in read_jsonl(resolve_recorded_path(protocol_record["plans"]["path"]))
             if p["method"] == method}
    selected = sorted(p for p in plans if not categories
                      or plans[p]["experiment"].removeprefix("t2icb_") in categories)
    if keys is not None:
        if len(keys) != len(set(keys)) or not set(keys) <= set(selected):
            raise ValueError("invalid prompt selection")
        selected = [p for p in selected if p in keys]
    if not selected:
        raise ValueError("empty generation selection")
    jobs = [r for r in jobs if r["prompt_id"].rsplit("_s", 1)[0] in set(selected)]
    output.mkdir(parents=True, exist_ok=True)
    receipt = {"protocol_sha256": sha256_file(protocol), "method": method,
               "selected": selected, "images": len(jobs)}
    receipt_path = output / "generation_receipt.json"
    if receipt_path.exists() and json.loads(receipt_path.read_text()) != receipt:
        raise ValueError("output belongs to a different run/selection")
    if not receipt_path.exists():
        atomic_write_json(receipt_path, receipt, overwrite=False)
    old = {}
    for path in (output / method).glob("*/metadata.jsonl"):
        for row in read_jsonl(path):
            if row["prompt_id"] in old:
                raise ValueError("duplicate generation metadata")
            old[row["prompt_id"]] = row
    if not set(old) <= {r["prompt_id"] for r in jobs}:
        raise ValueError("unexpected generation metadata")
    pending = {}
    for row in jobs:
        job = Job(**row, attention_exist_indices=[], attention_corr_indices=[], attention_leak_indices=[],
                  attention_possession_indices=[], source="sana_five_baselines")
        destination = image_path(output, method, job)
        if row["prompt_id"] in old:
            record = old[row["prompt_id"]]
            if (not destination.is_file() or sha256_file(destination) != record["image_sha256"]
                    or record["protocol_sha256"] != receipt["protocol_sha256"]):
                raise ValueError("resume image or metadata authentication failed")
        elif destination.exists():
            raise ValueError("image without completed metadata; preserve it before recovery")
        pending.setdefault(row["prompt_id"].rsplit("_s", 1)[0], []).append((row, job, destination))
    pipe = load_sana_pipeline(device)
    texts = set()
    for key in selected:
        plan = plans[key]
        texts.add(plan["conditioning"])
        texts.update(plan.get("components", []))
        texts.update(p["text"] for p in plan.get("phrases", []))
    encodings = encode_texts(pipe, {text: text for text in sorted(texts)}, device=device,
                             instruction=protocol_record["instruction"])
    completed = len(old)
    print(json.dumps({"encoded_texts": len(texts), "pending_images": len(jobs) - completed}), flush=True)
    for key, group in pending.items():
        plan = plans[key]
        runtime = SanaBaselineRuntime(pipe, plan, encodings, device=device)
        selection = {}
        if method == "sana_attend_excite" and plan["status"] == "ok":
            selection_path = output / "selection" / (key + ".json")
            if selection_path.exists():
                selection = json.loads(selection_path.read_text())
                if selection["plan_sha256"] != sha256_text(canonical_json(plan)):
                    raise ValueError("A&E selection belongs to another plan")
            else:
                selection = {**runtime.select_targets(), "plan_sha256": sha256_text(canonical_json(plan)),
                             "protocol_sha256": receipt["protocol_sha256"]}
                atomic_write_json(selection_path, selection, overwrite=False)
        for start in range(0, len(group), 4):
            batch = group[start:start + 4]
            done = [r["prompt_id"] in old for r, _, _ in batch]
            if all(done):
                continue
            # Reproduce the original seed/batch slots even after a partially
            # written batch; authenticated completed images must match exactly.
            before = time.monotonic()
            images, diagnostics = runtime.generate(
                [r["seed"] for r, _, _ in batch], selected=selection.get("selected", []))
            for i, (image, diagnostic, item) in enumerate(zip(images, diagnostics, batch, strict=True)):
                row, job, destination = item
                destination.parent.mkdir(parents=True, exist_ok=True)
                temp = destination.with_suffix(".tmp.png")
                image.save(temp)
                if done[i]:
                    if sha256_file(temp) != old[row["prompt_id"]]["image_sha256"]:
                        raise ValueError("partial batch replay differs from completed image")
                    temp.unlink()
                    continue
                temp.replace(destination)
                record = {**row, "method": method, "conditioning": plan["conditioning"],
                          "guidance_applied": plan["status"] == "ok", "fallback_reason": plan["fallback_reason"],
                          "source_commit": protocol_record["source"]["commit"],
                          "protocol_sha256": receipt["protocol_sha256"],
                          "plan_sha256": sha256_text(canonical_json(plan)), "runtime": protocol_record["runtime"],
                          "model_id": CONTRACT.model_id, "model_revision": CONTRACT.model_revision,
                          "image_path": portable_path(destination), "image_sha256": sha256_file(destination),
                          "diagnostics": diagnostic, "physical_batch_size": len(batch), "batch_slot": i,
                          "device": device, "gpu_name": torch.cuda.get_device_name(device),
                          "ae_selection_sha256": sha256_text(canonical_json(selection)) if selection else None,
                          "seconds": (time.monotonic() - before) / len(batch)}
                with metadata_path(output, method, job.experiment).open("a") as stream:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                    stream.flush()
                completed += 1
            print(json.dumps({"completed": completed, "total": len(jobs), "prompt": key,
                              "seconds": time.monotonic() - before}), flush=True)
    atomic_write_json(output / "complete.json", {**receipt, "completed": completed}, overwrite=True)


def _all_finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_all_finite(v) for v in value.values())
    if isinstance(value, list):
        return all(_all_finite(v) for v in value)
    return True


def audit_t2i_category(  # noqa: C901
    *,
    run: Path,
    method: str,
    category: str,
    reference_root: Path | None = None,
    reference_run: str | None = None,
) -> dict[str, Any]:
    """Check one finished category of one method and write ``generation_audit.json``.

    Every manifest image of the category must have one finite metadata row
    whose prompt, seed, model, plan and protocol match; the batch slots must be
    seeds 42-45, 46-49, 50-51; and every initial latent must equal the one of
    the paired CCG Diffusion image of the reference run (``reference_run``,
    default the protocol's ``primary_reference_run``, under ``reference_root``,
    default the repository root). The reference images are read from the
    recorded layout ``final/<category>/<method>/<experiment>/`` or from that of
    ``run_ccg_diffusion generate --backbone sana``
    (``final_shape/<method>/t2icb_<category>/``).
    """

    from PIL import Image

    protocol_path = run / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    dataset = resolve_recorded_path(protocol["dataset"])
    expected = {r["prompt_id"]: r for r in read_jsonl(dataset / "independent_seed_manifest.jsonl")
                if r["experiment"] == "t2icb_" + category}
    output = run / method / category
    completion = json.loads((output / "complete.json").read_text())
    if completion["completed"] != len(expected):
        raise ValueError("generation population is incomplete")
    plans = {p["prompt_id"]: p for p in read_jsonl(resolve_recorded_path(protocol["plans"]["path"]))
             if p["method"] == method}
    reference = (reference_root or repository_root()) / "output/generation" / (
        reference_run or protocol["primary_reference_run"])
    reference_rows = {}
    reference_files = list((reference / "final" / category).glob("*/*/metadata.jsonl")) + list(
        (reference / "final_shape").glob(f"*/t2icb_{category}/metadata.jsonl"))
    if not reference_files:
        raise ValueError(f"no CCG Diffusion metadata of category {category} in the reference run {reference}; "
                         "name a Sana CCG Diffusion run of this category with "
                         "`python -m scripts.t2i_compbench.audit_sana_baselines --reference-run`")
    for path in reference_files:
        for r in read_jsonl(path):
            if r["prompt_id"] in reference_rows:
                raise ValueError("duplicate reference metadata")
            reference_rows[r["prompt_id"]] = r
    seen, fallback, updates = set(), 0, 0
    files = list(output.glob("*/*/metadata.jsonl"))
    for metadata in files:
        for row in read_jsonl(metadata):
            key = row["prompt_id"]
            if key in seen or key not in expected or not _all_finite(row):
                raise ValueError("duplicate, unexpected or nonfinite generated row")
            seen.add(key)
            plan = plans[key.rsplit("_s", 1)[0]]
            if any(row[k] != v for k, v in expected[key].items()):
                raise ValueError("generated prompt/seed differs from manifest")
            for field_name in ("runtime", "model_id", "model_revision"):
                if row[field_name] != protocol[field_name]:
                    raise ValueError(f"generation contract mismatch: {field_name}")
            if (row["protocol_sha256"] != sha256_file(protocol_path)
                    or row["plan_sha256"] != sha256_text(canonical_json(plan))
                    or row["conditioning"] != plan["conditioning"]
                    or row["guidance_applied"] != (plan["status"] == "ok")
                    or row["fallback_reason"] != plan["fallback_reason"]
                    or row["source_commit"] != protocol["source"]["commit"]):
                raise ValueError("generation provenance mismatch")
            physical = 4 if row["seed"] < 50 else 2
            if row["physical_batch_size"] != physical or row["batch_slot"] != (row["seed"] - 42) % 4:
                raise ValueError("seed/batch slots differ from the reference experiment")
            if key not in reference_rows:
                raise ValueError(f"the reference run {reference} has no image {key}")
            ref = reference_rows[key]
            if any(ref[k] != row[k] for k in ("prompt", "seed", "model_id", "model_revision")):
                raise ValueError("reference model or image population differs")
            initial = ref["compositional"]["update_diagnostics"]["initial_latent_sha256"]
            if initial != row["diagnostics"]["initial_latent_sha256"]:
                raise ValueError("initial latent differs from the paired proposal")
            if len(row["diagnostics"]["final_latent_sha256"]) != 64:
                raise ValueError("missing final latent audit")
            path = resolve_recorded_path(row["image_path"])
            if sha256_file(path) != row["image_sha256"]:
                raise ValueError("image hash mismatch")
            with Image.open(path) as im:
                im.load()
                if im.size != (512, 512) or im.mode != "RGB":
                    raise ValueError("invalid generated image format")
            fallback += int(not row["guidance_applied"])
            updates += row["diagnostics"]["updates"]
    if seen != set(expected):
        raise ValueError("generated population does not match the entire category")
    result = {"status": "passed", "method": method, "category": category,
              "images": len(seen), "fallback_images": fallback, "updates": updates,
              "all_initial_latents_match_reference": True, "protocol_sha256": sha256_file(protocol_path),
              "metadata": [{"path": portable_path(p), "sha256": sha256_file(p)} for p in files],
              "reference_metadata": [{"path": portable_path(p), "sha256": sha256_file(p)} for p in reference_files]}
    atomic_write_json(output / "generation_audit.json", result, overwrite=True)
    return result
