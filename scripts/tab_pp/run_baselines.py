"""TAB PP comparison methods (paper Table 5) on Stable Diffusion v1.4 and Sana 0.6B.

    python -m scripts.tab_pp.run_baselines prepare --backbone sd14 --method vanilla gold_visual_setup --run-id RUN
    python -m scripts.tab_pp.run_baselines generate --backbone sd14 --method vanilla --run-id RUN --device cuda:0
        [--batches 0] [--output-dir DIR]
    python -m scripts.tab_pp.run_baselines prepare --backbone sana --method vanilla gold_visual_setup --run-id RUN
    python -m scripts.tab_pp.run_baselines generate --backbone sana --method vanilla --run-id RUN --device cuda:0
        [--keys vanilla_tab-pp-000_None]
    python -m scripts.tab_pp.run_baselines prepare --backbone sd14 --method predicated_diffusion
        gold_visual_setup_predicated_diffusion --run-id RUN
    python -m scripts.tab_pp.run_baselines generate --backbone sd14 --method gold_visual_setup_predicated_diffusion
        --run-id RUN --device cuda:0 [--batches 93]
    python -m scripts.tab_pp.run_baselines generate --backbone sana --method predicated_diffusion --run-id RUN
        --device cuda:0 [--keys pd_spacy_tab-pp-000_None]
    python -m scripts.tab_pp.run_baselines merge --backbone sana --run-id RUN

Methods: ``vanilla`` (the ambiguous prompt), ``gold_visual_setup`` (VS-TIED +
Gold-visual setup: the prompt followed by the gold visual setup of each
interpretation), ``predicated_diffusion`` (the ambiguous prompt with the spaCy
guidance targets of the original Predicated Diffusion paper) and
``gold_visual_setup_predicated_diffusion`` (VS-TIED + Gold-visual setup +
Predicated Diffusion).

On SD1.4 Vanilla and Gold-visual setup run the stock batch driver over all 315
TAB items (1,575 and 3,150 images in physical batches of 4, manifest order;
``--batches`` selects physical-batch indices); on Sana they use the TAB run
layout of CCG Diffusion (74 PP items; ``--keys`` selects prompt keys).
The two Predicated Diffusion methods form one run of their own on either
backbone (74 PP items, 370 and 740 images): ``prepare`` takes both, and
``generate`` runs one of them. On SD1.4 the run's physical batches of 4 are
0-92 (``predicated_diffusion``) and 93-277
(``gold_visual_setup_predicated_diffusion``); on Sana each prompt key runs seeds
0-3, then seed 4. ``merge`` writes the trial plan
(``output/vqa/<run-id>/trial_plan.jsonl``) of a complete Sana stock run or
Predicated Diffusion run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from scripts.tab_pp.run_ccg_diffusion import parse_indices, run_lock

STOCK_METHODS = ("vanilla", "gold_visual_setup")
PD_METHODS = ("predicated_diffusion", "gold_visual_setup_predicated_diffusion")
SANA_ARMS = {"vanilla": "vanilla", "gold_visual_setup": "gold"}
PD_ARMS = {"predicated_diffusion": "pd_spacy", "gold_visual_setup_predicated_diffusion": "pd_spacy_gold"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "generate", "merge"):
        command = sub.add_parser(name)
        command.add_argument("--backbone", choices=("sd14", "sana"), required=True)
        command.add_argument("--run-id", required=True)
        if name == "prepare":
            command.add_argument("--method", nargs="+", choices=STOCK_METHODS + PD_METHODS, required=True,
                                 help="stock methods, or both Predicated Diffusion methods (one run)")
            command.add_argument("--allow-dirty", action="store_true",
                                 help="prepare from a working tree with uncommitted changes (recorded)")
        if name == "generate":
            command.add_argument("--method", choices=STOCK_METHODS + PD_METHODS, required=True)
            command.add_argument("--device", default="cuda:0")
            command.add_argument("--batches", type=parse_indices, help="SD1.4: physical-batch indices, e.g. 0-3")
            command.add_argument("--keys", type=lambda text: text.split(","), help="Sana: comma-separated plan keys")
            command.add_argument("--output-dir", type=Path, help="SD1.4: write a verification subset here")
            command.add_argument("--allow-dirty", action="store_true",
                                 help="SD1.4 Vanilla and Gold-visual setup: seal the working-tree bytes instead of "
                                      "the committed sources (other runs keep the state sealed at prepare)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare" and set(args.method) & set(PD_METHODS) and set(args.method) != set(PD_METHODS):
        raise SystemExit("predicated_diffusion and gold_visual_setup_predicated_diffusion form one run of their "
                         "own: prepare it with --method predicated_diffusion gold_visual_setup_predicated_diffusion")
    if args.command == "generate" and (args.batches and args.backbone != "sd14" or args.keys and args.backbone != "sana"
                                       or args.output_dir and args.backbone != "sd14"):
        raise SystemExit("--batches and --output-dir are SD1.4 options, --keys is a Sana option")

    with run_lock(args.run_id):
        if args.command == "merge":
            merge(args)
        elif (args.method[0] if args.command == "prepare" else args.method) in PD_METHODS:
            run_predicated_diffusion(args)
        else:
            run_stock(args)
    return 0


def run_stock(args: argparse.Namespace) -> None:
    """Vanilla and VS-TIED + Gold-visual setup."""
    from ccg_diffusion.evaluation.tab import CONTRACT_FILENAME, DATASET_DIR, ITEMS, verify_dataset

    if args.backbone == "sd14":
        from ccg_diffusion.baselines import tab_sd14

        if args.command == "prepare":
            from ccg_diffusion.config import SD14_GENERATION
            from ccg_diffusion.provenance import read_jsonl
            from ccg_diffusion.targets.tokenizer import load_sd14_tokenizer, require_clip_tokenizer_identity

            verify_dataset()
            tokenizer = load_sd14_tokenizer()
            identity = require_clip_tokenizer_identity(tokenizer, expected=SD14_GENERATION.tokenizer_identity())
            summary = tab_sd14.write_baseline_inputs(args.run_id, read_jsonl(ITEMS), identity, tokenizer,
                                                     DATASET_DIR / CONTRACT_FILENAME)
            print(json.dumps(summary, indent=2))
        else:
            from ccg_diffusion.backbones.sd14 import driver

            request, _ = tab_sd14.driver_request(args.run_id, args.method, out_dir=args.output_dir,
                                                 device=args.device, batches=args.batches,
                                                 allow_dirty=args.allow_dirty)
            driver.run(request)
        return

    from ccg_diffusion.baselines import tab_sana
    from ccg_diffusion.pp_attachment import sana

    if args.command == "prepare":
        path = tab_sana.prepare_stock_run(args.run_id, arms=tuple(SANA_ARMS[m] for m in args.method),
                                          dataset_verification=verify_dataset(), allow_dirty=args.allow_dirty)
        print(json.dumps({"run_dir": str(path)}))
    else:
        sana.generate_run(args.run_id, device=args.device, arms=(SANA_ARMS[args.method],), keys=args.keys)


def run_predicated_diffusion(args: argparse.Namespace) -> None:
    """Predicated Diffusion and Gold-visual setup + Predicated Diffusion (one run, both methods)."""
    from ccg_diffusion.pp_attachment.jobs import run_directory

    if args.command == "prepare":
        from ccg_diffusion.evaluation.tab import verify_dataset

        if args.backbone == "sd14":
            from ccg_diffusion.baselines.tab_sd14 import prepare_pd_run
        else:
            from ccg_diffusion.baselines.tab_sana import prepare_pd_run
        path = prepare_pd_run(args.run_id, dataset_verification=verify_dataset(), allow_dirty=args.allow_dirty)
        manifest = json.loads((path / "run_manifest.json").read_text())
        print(json.dumps({"run_dir": str(path), "counts": manifest["counts"],
                          "artifact_sha256": manifest["artifact_sha256"]}, indent=2))
    elif args.backbone == "sd14":
        from ccg_diffusion.baselines.tab_sd14 import generate_pd_baselines

        generate_pd_baselines(run_directory(args.run_id), device=args.device, arms=(PD_ARMS[args.method],),
                              batches=args.batches, output_dir=args.output_dir)
    else:
        from ccg_diffusion.baselines.tab_sana import generate_pd_run

        generate_pd_run(run_directory(args.run_id), device=args.device, arms=(PD_ARMS[args.method],),
                        keys=args.keys)


def merge(args: argparse.Namespace) -> None:
    """Trial plan of a complete Predicated Diffusion run (either backbone) or Sana stock run."""
    from ccg_diffusion.baselines import tab_sd14
    from ccg_diffusion.pp_attachment import sana
    from ccg_diffusion.pp_attachment.jobs import run_directory

    run_dir = run_directory(args.run_id)
    if (run_dir / tab_sd14.PD_TARGETS_FILE).exists():
        backbone = json.loads((run_dir / "run_manifest.json").read_text())["backbone"]
        if backbone != args.backbone:
            raise SystemExit(f"{args.run_id} is a {backbone} run, not --backbone {args.backbone}")
        print(tab_sd14.merge_pd_run(run_dir))
    elif args.backbone == "sana":
        print(sana.merge_run(args.run_id))
    else:
        raise SystemExit("merge writes the trial plan of a Predicated Diffusion run or a Sana stock run; "
                         "the SD1.4 stock methods record metadata per shard")


if __name__ == "__main__":
    raise SystemExit(main())
