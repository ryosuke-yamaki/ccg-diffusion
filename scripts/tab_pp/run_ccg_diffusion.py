"""CCG Diffusion on TAB PP (paper Tables 5 and 6) for Stable Diffusion v1.4 and Sana 0.6B.

    python -m scripts.tab_pp.run_ccg_diffusion prepare --backbone sd14 --run-id 20261006_tab_sd14 [--inputs frozen|DIR]
        [--remove disambiguating_paraphrase holder_object_region_assignment] [--allow-dirty]
    python -m scripts.tab_pp.run_ccg_diffusion preflight --backbone sd14 --run-id 20261006_tab_sd14
    python -m scripts.tab_pp.run_ccg_diffusion generate --backbone sd14 --run-id 20261006_tab_sd14 --device cuda:0
        [--batches 0,7,20] [--output-dir DIR]
    python -m scripts.tab_pp.run_ccg_diffusion generate --backbone sana --run-id 20261006_tab_sana --device cuda:0
        [--keys guided_tab-pp-000_np_attach,guided_tab-pp-003_np_attach]
    python -m scripts.tab_pp.run_ccg_diffusion merge --backbone sd14 --run-id 20261006_tab_sd14
    python -m scripts.tab_pp.run_ccg_diffusion audit --backbone sana --run-id 20261006_tab_sana

``prepare`` compiles the 148 interpretations of the 74 PP items from the frozen
bundle ``data/tab_pp`` (or from the output directory of
``scripts/tab_pp/parse_interpretations.py``) into the jobs of one Table 6 cell;
``--remove`` drops the Disambiguating Paraphrase and/or the Holder-Object
Region Assignment (cells c0r1p0m0, c1r0p0m0, c0r0p0m0; default c1r1p0m0).
``preflight`` compares the prepared jobs with the evaluated paper jobs
(``data/tab_pp/reference.json``). ``generate`` runs the SD1.4 physical batches
of 4 (``--batches``: indices 0-184, comma list or ranges) or the Sana prompt
keys (``--keys``: all seeds of each key). ``merge`` writes the images'
metadata and the trial plan (``output/vqa/<run-id>/trial_plan.jsonl``) of a
complete run. Images are bit-exact only on the paper's GPU type (RTX 4090).
"""
from __future__ import annotations

import argparse
import fcntl
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

REMOVABLE = ("disambiguating_paraphrase", "holder_object_region_assignment")


def parse_indices(text: str) -> list[int]:
    """``"0,7,20"`` or ``"0-3,7"`` -> sorted unique batch indices."""
    result: set[int] = set()
    for part in text.split(","):
        start, _, end = part.partition("-")
        result.update(range(int(start), int(end or start) + 1))
    return sorted(result)


@contextmanager
def run_lock(run_id: str) -> Iterator[None]:
    """Exclusive lock on ``output/generation/<run-id>/.lock`` (one process per run)."""
    from ccg_diffusion.pp_attachment.jobs import run_directory

    directory = run_directory(run_id)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "preflight", "generate", "merge", "audit"):
        command = sub.add_parser(name)
        command.add_argument("--backbone", choices=("sd14", "sana"), required=True)
        command.add_argument("--run-id", required=True)
        if name == "prepare":
            command.add_argument("--inputs", default="frozen",
                                 help="'frozen' (data/tab_pp) or a parse_interpretations.py output directory")
            command.add_argument("--remove", nargs="+", choices=REMOVABLE, default=(),
                                 help="TAB components to remove (Table 6 cells)")
            command.add_argument("--allow-dirty", action="store_true",
                                 help="prepare from a working tree with uncommitted changes (recorded)")
        if name == "generate":
            command.add_argument("--device", default="cuda:0")
            command.add_argument("--batches", type=parse_indices, help="SD1.4: physical-batch indices, e.g. 0,7,20")
            command.add_argument("--keys", type=lambda text: text.split(","), help="Sana: comma-separated plan keys")
            command.add_argument("--output-dir", type=Path,
                                 help="SD1.4: write images and receipts of a verification subset here")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "generate" and (args.batches and args.backbone != "sd14"
                                       or args.keys and args.backbone != "sana"
                                       or args.output_dir and args.backbone != "sd14"):
        raise SystemExit("--batches and --output-dir are SD1.4 options, --keys is a Sana option")
    if args.command == "audit" and args.backbone != "sana":
        raise SystemExit("audit is a Sana command (the SD1.4 receipts are verified by merge)")

    from ccg_diffusion.methods import TabAblation
    from ccg_diffusion.pp_attachment import jobs, sana, sd14

    if args.command == "preflight":
        report = (jobs if args.backbone == "sd14" else sana).preflight(args.run_id)
        print(json.dumps({**report, "status": "PREFLIGHT_PASS"}, indent=2))
        return 0
    if args.command == "audit":
        print(json.dumps(sana.audit_run(args.run_id), indent=2))
        return 0
    with run_lock(args.run_id):
        if args.command == "prepare":
            ablation = TabAblation.from_removed(args.remove)
            inputs = args.inputs if args.inputs == "frozen" else Path(args.inputs)
            if args.backbone == "sd14":
                path = sd14.prepare_run(args.run_id, ablation=ablation, inputs=inputs, allow_dirty=args.allow_dirty)
            else:
                from ccg_diffusion.evaluation.tab import verify_dataset

                path = sana.prepare_run(run_id=args.run_id, ablation=ablation, inputs=inputs,
                                        dataset_verification=verify_dataset(), allow_dirty=args.allow_dirty)
            print(json.dumps({"run_dir": str(path), "cell": ablation.cell_id}))
        elif args.command == "generate":
            if args.backbone == "sd14":
                sd14.generate(args.run_id, device=args.device, batches=args.batches, output_dir=args.output_dir)
            else:
                sana.generate_run(args.run_id, device=args.device, keys=args.keys)
        elif args.backbone == "sd14":
            print(sd14.merge(args.run_id))
        else:
            print(sana.merge_run(args.run_id))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
