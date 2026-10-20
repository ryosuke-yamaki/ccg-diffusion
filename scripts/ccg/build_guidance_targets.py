#!/usr/bin/env python3
"""Stage 3: SD1.4 guidance targets I_D = (E, C, K, P) of one T2I-CompBench++ category.

Reads the category's prompt-only rows, Stage 1 AUTO, sentence map, Stage 2
marked dependencies and coverage rows, and writes the guidance-target rows
(provider ``deps2index_sd14``), their conversion traces and ``<output>.meta.json``.
Stage 1-2 failures become failure rows. No parser, ``generate`` or diffusion
model runs here; the pinned SD1.4 CLIP tokenizer and the spaCy lemmatizer
(possession verbs only) are loaded. Run from the repository root, e.g.::

    python -m scripts.ccg.build_guidance_targets --prompt-only STAGE/color.prompt_only.jsonl \\
        --auto-path STAGE/color.auto --sentence-map-path STAGE/color.sentence_map.jsonl \\
        --markedup-deps-path STAGE/generate/color.markedup_deps \\
        --coverage-rows-path STAGE/generate/color.coverage_rows.jsonl --output STAGE/color.deps2index_sd14.jsonl
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from ccg_diffusion.config import SD14_GENERATION
from ccg_diffusion.parsing.ccg import load_auto_sentences, load_marked_dependency_blocks
from ccg_diffusion.parsing.sentences import load_sentence_map
from ccg_diffusion.provenance import atomic_write_json, atomic_write_jsonl, read_jsonl
from ccg_diffusion.targets import guidance_targets as targets
from ccg_diffusion.targets.tokenizer import load_sd14_tokenizer, require_clip_tokenizer_identity


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt-only", type=Path, required=True, help="<cat>.prompt_only.jsonl (300 base prompts)")
    parser.add_argument("--auto-path", type=Path, required=True, help="Stage 1 AUTO")
    parser.add_argument("--sentence-map-path", type=Path, required=True, help="<cat>.sentence_map.jsonl")
    parser.add_argument("--markedup-deps-path", type=Path, required=True, help="Stage 2 <cat>.markedup_deps")
    parser.add_argument("--coverage-rows-path", type=Path, required=True,
                        help="Stage 2 <cat>.coverage_rows.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="guidance-target rows (JSONL)")
    parser.add_argument("--diagnostics-output", type=Path, default=None,
                        help="conversion traces (default: <output>.conversion-trace.jsonl)")
    parser.add_argument("--expected-count", type=int, default=300)
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument("--allow-download", action="store_true",
                        help="download the pinned SD1.4 tokenizer if it is not in the Hugging Face cache")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    trace_output = args.diagnostics_output or targets.target_file_trace_path(args.output)
    metadata_output = targets.target_file_metadata_path(args.output)
    existing = [str(path) for path in (args.output, trace_output, metadata_output) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "refusing to overwrite Stage 3 artifacts without --overwrite: " + ", ".join(existing)
        )
    category, prompt_rows = targets.load_category_prompt_rows(args.prompt_only, expected_count=args.expected_count)
    auto_sentences = load_auto_sentences(args.auto_path)
    sentence_map_rows = load_sentence_map(args.sentence_map_path)
    dependency_blocks = load_marked_dependency_blocks(args.markedup_deps_path)
    coverage_rows = read_jsonl(args.coverage_rows_path, reject_duplicate_keys=True)
    tokenizer = load_sd14_tokenizer(local_files_only=not args.allow_download)
    try:
        tokenizer_identity = require_clip_tokenizer_identity(tokenizer, expected=SD14_GENERATION.tokenizer_identity())
    except ValueError as exc:
        raise targets.GuidanceTargetError(str(exc)) from exc
    nlp = targets.load_possession_lemmatizer(args.spacy_model)
    rows, trace_rows = targets.build_guidance_target_rows(
        prompt_rows,
        auto_sentences,
        dependency_blocks,
        tokenizer,
        nlp,
        category=category,
        coverage_rows=coverage_rows,
        tokenizer_identity=tokenizer_identity,
        sentence_map_rows=sentence_map_rows,
    )

    trace_record = atomic_write_jsonl(trace_output, trace_rows, overwrite=args.overwrite)
    output_record = atomic_write_jsonl(args.output, rows, overwrite=args.overwrite)
    source_paths = [args.prompt_only, args.auto_path, args.sentence_map_path, args.markedup_deps_path,
                    args.coverage_rows_path]
    stage1_meta = targets.target_file_metadata_path(args.auto_path)  # <auto>.meta.json, when Stage 1 ran here
    if stage1_meta.is_file():
        source_paths.append(stage1_meta)
    metadata = targets.guidance_target_file_metadata(
        category=category,
        rows=rows,
        output_record=output_record,
        trace_record=trace_record,
        source_paths=source_paths,
        tokenizer_identity=tokenizer_identity,
        nlp=nlp,
        spacy_model=args.spacy_model,
    )
    atomic_write_json(metadata_output, metadata, overwrite=args.overwrite)
    print(f"[build_guidance_targets] {category}: {metadata['status_counts']} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
