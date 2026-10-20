"""Audit finished T2I-CompBench++ categories of a Sana 0.6B compared method against a Sana CCG Diffusion run.

    python -m scripts.t2i_compbench.audit_sana_baselines --run-id R --method M --reference-run C [--categories ...]

The check of ``run_baselines.py --backbone sana --audit``, with the paired run
named explicitly: every image of method ``M`` in run ``R`` must have a finite
metadata row matching its prompt, seed, model, plan and protocol, and must share
its initial latent with the paired CCG Diffusion image of run ``C``
(``output/generation/C``, e.g. a run of ``run_ccg_diffusion.py generate --backbone
sana``). ``run_baselines.py --audit`` compares with the reference run recorded in
the protocol instead.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ccg_diffusion.baselines.registry import T2I_BASELINE_ARMS
from ccg_diffusion.config import T2I_COMPBENCH_CATEGORIES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", required=True, help="the baseline run output/generation/<run-id>")
    parser.add_argument("--method", choices=sorted(T2I_BASELINE_ARMS["sana"]), required=True)
    parser.add_argument("--reference-run", required=True,
                        help="the paired Sana CCG Diffusion run in output/generation")
    parser.add_argument("--categories", nargs="+", choices=T2I_COMPBENCH_CATEGORIES)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    from ccg_diffusion.baselines.sana_generation import audit_t2i_category

    arm = T2I_BASELINE_ARMS["sana"][args.method]
    run_dir = Path("output/generation") / args.run_id
    for category in args.categories or T2I_COMPBENCH_CATEGORIES:
        print(json.dumps(audit_t2i_category(run=run_dir, method=arm, category=category,
                                            reference_run=args.reference_run)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
