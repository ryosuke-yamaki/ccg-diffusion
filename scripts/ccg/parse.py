#!/usr/bin/env python3
"""Stage 1: parse a tokenized .raw corpus (one sentence per line) with Hol-CCG into CCGbank AUTO.

Writes the AUTO file and ``<auto>.meta.json`` (checkpoint, resolved revision,
parse settings, input/output sha256, sentence counts). The defaults come from
``configs/parsing/holccg.yaml``, which holds the paper's parse contract
``ccg_diffusion.config.HOLCCG_PARSE`` (thresholds 0.01/0.001/0.001, skimmer on,
treebank grammar, one sentence per batch).

Run from the repository root, e.g.::

    python -m scripts.ccg.parse --input-raw-path STAGE/color.raw --output-auto-path STAGE/color.auto
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "parsing" / "holccg.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML config (OmegaConf)")
    parser.add_argument("--override", nargs="*", default=None,
                        help="OmegaConf dotlist overrides, e.g. runtime.device=cpu")
    parser.add_argument("--model-dir", "--model_dir", dest="model_dir", default=None,
                        help="Hugging Face repo id or local checkpoint directory (default: config model_dir)")
    parser.add_argument("--input-raw-path", "--input_raw_path", dest="input_raw_path", default=None,
                        help="input .raw corpus, one tokenized sentence per line")
    parser.add_argument("--output-auto-path", "--output_auto_path", dest="output_auto_path", default=None,
                        help="output AUTO file")
    parser.add_argument("--device", default=None, help="cuda | cuda:0 | cpu (default: config runtime.device)")
    parser.add_argument("--max-sentences", "--max_sentences", dest="max_sentences", type=int, default=None,
                        help="parse only the first N sentences")
    parser.add_argument("--allow-download", action="store_true",
                        help="download the pinned checkpoint revision if it is not in the Hugging Face cache")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from omegaconf import OmegaConf

    from ccg_diffusion.config import HOLCCG_REVISION, HolCCGParseContract
    from ccg_diffusion.parsing.holccg import inference_options, parse_corpus, resolve_chunk_size, resolve_grammar

    cfg = OmegaConf.load(args.config)
    if args.override:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.override))
    for key in ("model_dir", "input_raw_path", "output_auto_path"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    if args.device is not None:
        cfg.runtime.device = args.device
    if args.max_sentences is not None:
        cfg.runtime.max_sentences = args.max_sentences
    for key in ("model_dir", "input_raw_path", "output_auto_path"):
        if not cfg.get(key):
            raise ValueError(f"{key} is required (pass --{key.replace('_', '-')})")

    contract = HolCCGParseContract(
        str(cfg.model_dir),
        str(cfg.get("model_revision") or HOLCCG_REVISION),
        float(cfg.parsing.stag_threshold),
        float(cfg.parsing.phrase_threshold),
        float(cfg.parsing.span_threshold),
        apply_skimmer=bool(cfg.parsing.apply_skimmer),
        grammar=resolve_grammar(cfg.parsing.get("grammar")),
        chunk_size=resolve_chunk_size(cfg.parsing.get("chunk_size")),
        min_rule_freq=int(cfg.parsing.min_rule_freq),
    )
    max_sentences = cfg.runtime.max_sentences
    meta = parse_corpus(
        model_dir=contract.checkpoint,
        input_raw_path=str(cfg.input_raw_path),
        output_auto_path=str(cfg.output_auto_path),
        options=inference_options(contract),
        device=str(cfg.runtime.device),
        revision=contract.revision,
        local_files_only=not args.allow_download,
        id_prefix=None if cfg.id.prefix is None else str(cfg.id.prefix),
        id_start=int(cfg.id.start),
        parser_name=str(cfg.output.parser_name),
        partial_root_cat=str(cfg.output.partial_root_cat),
        fail_root_cat=str(cfg.output.fail_root_cat),
        fail_pos=str(cfg.output.fail_pos),
        overlen_root_cat=str(cfg.output.overlen_root_cat),
        encoding=str(cfg.input.encoding),
        skip_empty=bool(cfg.input.skip_empty),
        max_sentences=None if max_sentences is None else int(max_sentences),
        progress=bool(cfg.runtime.progress),
    )
    counts = meta["sentence_counts"]
    print(
        f"[parse] done: ok={counts['ok']} partial={counts['partial']} fail={counts['fail']} "
        f"error={counts['error']} overlen={counts['overlen']} (device={cfg.runtime.device}, "
        f"grammar={contract.grammar}, chunk_size={contract.chunk_size}, revision={meta['resolved_revision']}, "
        f"auto={cfg.output_auto_path})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
