"""Clean path of TAB PP: Supertag-Constrained Parsing of the 74 PP prompts into 148 interpretations.

    python -m scripts.tab_pp.parse_interpretations --output DIR [--device cuda:0] [--compare-frozen data/tab_pp]

For each PP item (dataset order) the ambiguous preposition is chosen from the
grammar and the prompt is parsed once per interpretation with the
preposition's supertag fixed (``pp_attachment.supertag_constrained_parsing``).
Stage 2 runs over each interpretation's 74-sentence corpus
(``DIR/<interpretation>/prompts.auto``); Stage 3, the Selective Guidance
Boost decision, the ``next_to`` relations, the reading -> setup mapping and
the Disambiguating Paraphrase follow. ``DIR/interpretations.jsonl`` feeds
``run_ccg_diffusion.py prepare --inputs DIR``. ``--compare-frozen`` writes
``DIR/comparison.json`` (the clean interpretations and the jobs and Sana plans
compiled from them vs the frozen bundle) and fails when a required check fails.
The constrained parses have near-tied supertags: they reproduce the bundle
only on the paper's hardware (RTX 4090).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

READINGS = ("np_attach", "vp_attach")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True, help="output directory (must not exist)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--compare-frozen", type=Path, help="frozen bundle directory to compare with (data/tab_pp)")
    return parser


def _quiet_generator() -> Any:
    """In-memory Stage 2 without the per-call ``generate`` command echo."""
    import contextlib
    import io

    from ccg_diffusion.parsing.dependencies import generate_marked_dependencies

    def generate(auto_lines: Sequence[str], raw_lines: Sequence[str]) -> list:
        with contextlib.redirect_stdout(io.StringIO()):
            return generate_marked_dependencies(auto_lines, raw_lines)

    return generate


def parse(output: Path, device: str) -> list[dict[str, Any]]:  # noqa: C901
    """Write the parse corpora, Stage 2 files and ``interpretations.jsonl``; return the interpretation rows."""
    import contextlib
    import io
    from dataclasses import asdict

    import torch

    from ccg_diffusion import config
    from ccg_diffusion.evaluation.tab import load_pp_items, map_tab_item_readings_to_setups
    from ccg_diffusion.guidance.count_loss import load_noun_forms
    from ccg_diffusion.parsing.ccg import load_auto_sentences
    from ccg_diffusion.parsing.dependencies import generate_dependencies, load_aligned_dependency_blocks
    from ccg_diffusion.parsing.holccg import load_holccg, resolve_checkpoint
    from ccg_diffusion.pp_attachment import compile as tab_compile
    from ccg_diffusion.pp_attachment import supertag_constrained_parsing as scp
    from ccg_diffusion.provenance import atomic_write_json, atomic_write_jsonl, repo_relative, sha256_file
    from ccg_diffusion.targets.guidance_targets import load_possession_lemmatizer
    from ccg_diffusion.targets.pd_indices import program_sha256
    from ccg_diffusion.targets.tokenizer import load_sd14_tokenizer

    output.mkdir(parents=True, exist_ok=False)
    items = list(load_pp_items().values())
    torch.set_num_threads(4)
    parser = load_holccg(device=device)
    grammar = scp.load_grammar(parser)
    generator = _quiet_generator()
    sites, derivations = {}, {}
    for item in items:
        prompt = item["ambiguous_prompt"].strip()
        sites[item["tab_id"]], derivations[item["tab_id"]] = scp.select_preposition(
            parser, prompt, grammar, generate_dependencies=generator)
        print(json.dumps({"tab_id": item["tab_id"], "site": sites[item["tab_id"]].position,
                          "tier": sites[item["tab_id"]].tier}), flush=True)
    sentences, blocks = {}, {}
    for reading in READINGS:
        directory = output / reading
        directory.mkdir()
        lines = scp.auto_lines([derivations[item["tab_id"]][reading] for item in items], first_index=1)
        (directory / "prompts.auto").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (directory / "prompts.raw").write_text(
            "\n".join(item["ambiguous_prompt"].strip() for item in items) + "\n", encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            generate_dependencies(auto_path=directory / "prompts.auto", output_dir=directory / "generate",
                                  raw_path=directory / "prompts.raw")
        sentences[reading] = load_auto_sentences(directory / "prompts.auto")
        blocks[reading] = load_aligned_dependency_blocks(directory / "generate/prompts.markedup_deps", len(items))

    tokenizer = load_sd14_tokenizer()
    nlp = load_possession_lemmatizer()
    noun_forms = load_noun_forms()
    rows = []
    for index, item in enumerate(items):
        tab_id, prompt, site = item["tab_id"], item["ambiguous_prompt"].strip(), sites[item["tab_id"]]
        compiled = {reading: tab_compile.interpretation_targets(
            prompt, sentences[reading][index], blocks[reading][index], tokenizer, nlp)[0] for reading in READINGS}
        evidence = scp.setup_mapping_evidence(
            tab_id, prompt, site, {r: compiled[r].trace for r in READINGS},
            {r: compiled[r].alignment for r in READINGS},
            {r: {"pd_indices": asdict(compiled[r].indices),
                 "program_sha256": program_sha256(compiled[r].indices)} for r in READINGS})
        mapping = map_tab_item_readings_to_setups(item, evidence)
        for reading in READINGS:
            setup = mapping["reading_to_target_setup_index"][reading]
            evaluation = {"tab_category": item["tab_category"], "tab_requires_reading": True,
                          "tab_evaluation_mode": "targeted_setup", "target_setup_index": setup,
                          "evaluation_setup_indices": [setup],
                          "target_setup_mapping_evidence_sha256": mapping["evidence_sha256"]}
            interpretation = tab_compile.interpretation_from_parse(
                tab_id=tab_id, reading=reading, prompt=prompt, item_index=index + 1, site=site,
                sentence=sentences[reading][index], dependencies=blocks[reading][index], tokenizer=tokenizer,
                nlp=nlp, noun_forms=noun_forms, evaluation=evaluation)
            targets = interpretation.targets
            rows.append({
                "tab_id": tab_id, "reading": reading, "prompt": prompt,
                "preposition": {**{k: v for k, v in asdict(site).items() if k != "frame_repair"},
                                "frame_repair": None if site.frame_repair is None else asdict(site.frame_repair)},
                "paraphrase": interpretation.paraphrase,
                "guidance_targets": {"prompt": targets.prompt, "alignment": list(targets.alignment),
                                     "pd_indices": targets.pd_indices, "boost": targets.boost,
                                     "spatial_relations": list(targets.spatial_relations),
                                     "record_sha256": targets.record_sha256},
                "standard_program": interpretation.standard_program,
                "source_job": interpretation.source_job,
            })
    atomic_write_jsonl(output / "interpretations.jsonl", rows, overwrite=False)
    checkpoint = resolve_checkpoint()
    atomic_write_json(output / "parse_manifest.json", {
        "schema": "ccg_diffusion.tab_pp_parse.v1", "checkpoint": config.HOLCCG_CHECKPOINT,
        "resolved_revision": Path(checkpoint).name, "holccg_version": config.HOLCCG_PACKAGE_VERSION,
        "site_parse": config.HOLCCG_TAB_SITE_PARSE.to_dict(),
        "supertag_constrained_parse": config.HOLCCG_SUPERTAG_CONSTRAINED_PARSE.to_dict(),
        "device": device, "gpu": torch.cuda.get_device_name(device) if device.startswith("cuda") else None,
        "files": {repo_relative(path): sha256_file(path) for path in sorted(output.rglob("*"))
                  if path.is_file() and path.name != "parse_manifest.json"},
    }, overwrite=False)
    atomic_write_json(output / "summary.json", {
        "items": len(items), "interpretations": len(rows),
        "tiers": dict(sorted(_count(site.tier for site in sites.values()).items())),
        "realizations": dict(sorted(_count(str(r["paraphrase"].get("realization")) for r in rows).items())),
        "boost": sum(r["guidance_targets"]["boost"] for r in rows),
        "next_to": sum(bool(r["guidance_targets"]["spatial_relations"]) for r in rows),
    }, overwrite=False)
    return rows


def _count(values: Any) -> dict[str, int]:
    from collections import Counter

    return dict(Counter(values))


def compare(output: Path, bundle_dir: Path) -> dict[str, Any]:  # noqa: C901
    """Compare the clean interpretations, jobs and Sana plans with the frozen bundle."""
    from collections import Counter

    from ccg_diffusion.backbones.sana.pipeline import load_sana_tokenizer
    from ccg_diffusion.evaluation.tab import load_pp_items
    from ccg_diffusion.methods import PAPER_TAB_CELLS, TabAblation
    from ccg_diffusion.pp_attachment import jobs, sana
    from ccg_diffusion.provenance import atomic_write_json, canonical_json, read_jsonl
    from ccg_diffusion.targets.tokenizer import load_sd14_tokenizer

    bundle = jobs.load_frozen_bundle(bundle_dir)
    frozen_rows = {(r["tab_id"], r["reading"]): r for r in bundle.interpretation_rows}
    clean_rows = {(r["tab_id"], r["reading"]): r for r in read_jsonl(output / "interpretations.jsonl")}
    keys = list(frozen_rows)
    checks: dict[str, int] = Counter()
    if list(clean_rows) != keys:
        raise SystemExit("clean interpretations do not cover the 148 frozen interpretations in order")
    alignment_fields = ("source_index", "text", "char_start", "char_end", "clip_index", "clip_lexical_piece_indices")
    job_fields = ("prompt", "token_ids", "pd_indices", "program_sha256", "source_program_sha256", "loss_config",
                  "relations", "tab_category", "tab_requires_reading", "tab_evaluation_mode", "target_setup_index",
                  "evaluation_setup_indices")
    for key in keys:
        clean, frozen = clean_rows[key], frozen_rows[key]
        checks["preposition"] += clean["preposition"] == frozen["preposition"]
        checks["auto_vs_paraphrase_source_auto"] += (clean["paraphrase"]["source_auto_derivation_sha256"]
                                                     == frozen["paraphrase"]["source_auto_derivation_sha256"])
        for name in ("pd_indices", "boost", "spatial_relations"):
            checks[f"guidance_targets.{name}"] += clean["guidance_targets"][name] == frozen["guidance_targets"][name]
        checks["standard_program"] += clean["standard_program"] == frozen["standard_program"]
        checks["alignment_text_char_clip"] += (
            [{k: e[k] for k in alignment_fields} for e in clean["guidance_targets"]["alignment"]]
            == [{k: e[k] for k in alignment_fields} for e in frozen["guidance_targets"]["alignment"]])
        checks["alignment_full"] += clean["guidance_targets"]["alignment"] == frozen["guidance_targets"]["alignment"]
        checks["paraphrase_trace"] += canonical_json(clean["paraphrase"]) == canonical_json(frozen["paraphrase"])
        template = bundle.interpretations[key].source_job
        checks["source_job_fields"] += all(clean["source_job"][k] == template[k] for k in job_fields)
        checks["target_setup_mapping_evidence_sha256"] += (clean["source_job"]["target_setup_mapping_evidence_sha256"]
                                                           == template["target_setup_mapping_evidence_sha256"])
    report: dict[str, Any] = {"interpretations": len(keys), "items": len(keys) // 2,
                              "preposition_items": checks["preposition"] // 2, "checks": dict(checks)}
    clean_interpretations = jobs.load_clean_interpretations(output)
    clip, gemma = load_sd14_tokenizer(), load_sana_tokenizer()
    specs = list(load_pp_items().values())
    report["cells"] = {}
    for cell in PAPER_TAB_CELLS:
        ablation = TabAblation.from_cell_id(cell)
        frozen = jobs.place_jobs(interpretations=bundle.interpretations, source_jobs=bundle.source_jobs,
                                 ablation=ablation, run_id="clean_vs_frozen", tokenizer=clip)[:jobs.PP_JOBS]
        clean = jobs.place_jobs(interpretations=clean_interpretations,
                                source_jobs=jobs.clean_layout(clean_interpretations), ablation=ablation,
                                run_id="clean_vs_frozen", tokenizer=clip)
        excluded: dict[str, int] = Counter()
        for a, b in zip(clean, frozen, strict=True):
            for field in ("followup_trace", "standard_intervention_trace_sha256", "source_standard_input_signature",
                          "target_setup_mapping_evidence_sha256", "effective_sha256"):
                excluded[field] += a.get(field) != b.get(field)
            excluded["transfer_trace.source_audit_sha256"] += (a["transfer_trace"].get("source_audit_sha256")
                                                               != b["transfer_trace"].get("source_audit_sha256"))
            if a.get("transfer_region"):
                excluded["transfer_region.plan.source_plan_sha256"] += (
                    a["transfer_region"]["plan"]["source_plan_sha256"]
                    != b["transfer_region"]["plan"]["source_plan_sha256"])
        clean_plans, _ = sana.guided_plans_and_jobs(specs=specs, interpretations=clean_interpretations,
                                                    ablation=ablation, clip_tokenizer=clip, gemma_tokenizer=gemma)
        frozen_plans, _ = sana.guided_plans_and_jobs(specs=specs, interpretations=bundle.interpretations,
                                                     ablation=ablation, clip_tokenizer=clip, gemma_tokenizer=gemma)
        report["cells"][cell] = {
            "sd14_jobs_pinned_projection_equal": sum(jobs.pinned_projection(a) == jobs.pinned_projection(b)
                                                     for a, b in zip(clean, frozen)),
            "sd14_jobs": len(clean),
            "sd14_excluded_field_differences": dict(excluded),
            "sana_plans_pinned_projection_equal": sum(sana.pinned_projection(a) == sana.pinned_projection(b)
                                                      for a, b in zip(clean_plans, frozen_plans, strict=True)),
            "sana_plans": len(clean_plans),
        }
    required = {
        "preposition": len(keys), "auto_vs_paraphrase_source_auto": len(keys),
        "guidance_targets.pd_indices": len(keys), "guidance_targets.boost": len(keys),
        "guidance_targets.spatial_relations": len(keys), "standard_program": len(keys),
        "alignment_text_char_clip": len(keys), "paraphrase_trace": len(keys), "source_job_fields": len(keys),
    }
    failed = [name for name, expected in required.items() if checks[name] != expected]
    failed += [f"{cell}.{name}" for cell, record in report["cells"].items()
               for name, total in (("sd14_jobs_pinned_projection_equal", "sd14_jobs"),
                                   ("sana_plans_pinned_projection_equal", "sana_plans"))
               if record[name] != record[total]]
    report["failed"] = failed
    report["passed"] = not failed
    atomic_write_json(output / "comparison.json", report, overwrite=False)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    parse(args.output, args.device)
    if args.compare_frozen is not None:
        report = compare(args.output, args.compare_frozen)
        print(json.dumps(report, indent=2))
        if not report["passed"]:
            raise SystemExit(f"clean path differs from the frozen bundle: {report['failed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
