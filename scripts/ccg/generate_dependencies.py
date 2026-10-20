#!/usr/bin/env python3
"""Stage 2: C&C ``generate`` marked dependencies of an AUTO corpus, with PARTIAL-fragment aggregation.

Complete derivations go through the pure-Python C&C ``generate`` port of the
``holccg`` package. A ``PARTIAL`` derivation is split into its skimmer
fragments; each fragment is generated on its own and its dependencies are
re-indexed to sentence positions (no dependency crosses two fragments).

Writes ``<prefix>.markedup_deps``, ``<prefix>.ccgbank_deps``,
``<prefix>.coverage_rows.jsonl``, ``<prefix>.coverage.json`` and intermediate
files into ``--output-dir`` and prints the coverage summary. Relative paths
resolve against the repository root. Run from the repository root, e.g.::

    python -m scripts.ccg.generate_dependencies --auto-path STAGE/color.auto --raw-path STAGE/color.raw \\
        --output-dir STAGE/generate --apply-skimmer
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from ccg_diffusion.parsing.dependencies import generate_dependencies

REPO_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--auto-path", "--auto_path", dest="auto_path", required=True, help="Stage 1 AUTO input")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", required=True, help="output directory")
    parser.add_argument("--output-prefix", "--output_prefix", dest="output_prefix", default=None,
                        help="output file stem (default: AUTO stem)")
    parser.add_argument("--markedup-additions-path", "--markedup_additions_path", dest="markedup_additions_path",
                        default="configs/parsing/candc_markedup_additions",
                        help="markedup overlay added to the bundled hol_en database ('' for none)")
    parser.add_argument("--partial-root-cat", "--partial_root_cat", dest="partial_root_cat", default="PARTIAL")
    parser.add_argument("--fail-root-cats", "--fail_root_cats", dest="fail_root_cats", nargs="*",
                        default=["FAIL", "OVERLEN"])
    parser.add_argument("--raw-path", "--raw_path", dest="raw_path", default=None,
                        help="Stage 1 .raw input; adds the sentence text to every coverage row")
    parser.add_argument("--dataset-name", "--dataset_name", dest="dataset_name", default=None,
                        help="recorded in the summary only")
    parser.add_argument("--checkpoint", default=None, help="recorded in the summary only")
    parser.add_argument("--stag-threshold", "--stag_threshold", dest="stag_threshold", type=float, default=None,
                        help="recorded in the summary only")
    parser.add_argument("--phrase-threshold", "--phrase_threshold", dest="phrase_threshold", type=float,
                        default=None, help="recorded in the summary only")
    parser.add_argument("--span-threshold", "--span_threshold", dest="span_threshold", type=float, default=None,
                        help="recorded in the summary only")
    parser.add_argument("--apply-skimmer", "--apply_skimmer", dest="apply_skimmer", action="store_true",
                        help="recorded in the summary only")
    return parser


def _repository_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = generate_dependencies(
        auto_path=_repository_path(args.auto_path),
        output_dir=_repository_path(args.output_dir),
        additions_path=_repository_path(args.markedup_additions_path) if args.markedup_additions_path else None,
        output_prefix=args.output_prefix,
        partial_root_cat=args.partial_root_cat,
        fail_root_cats=args.fail_root_cats,
        raw_path=_repository_path(args.raw_path) if args.raw_path else None,
        dataset_name=args.dataset_name,
        checkpoint=args.checkpoint,
        stag_threshold=args.stag_threshold,
        phrase_threshold=args.phrase_threshold,
        span_threshold=args.span_threshold,
        apply_skimmer=args.apply_skimmer,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
