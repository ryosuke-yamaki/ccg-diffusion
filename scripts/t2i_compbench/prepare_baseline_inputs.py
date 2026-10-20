"""Prepare the inputs of the T2I-CompBench++ baselines (Table 2).

Order of the subcommands:

1. ``external`` (Python 3.11 adapter environment, ``scripts/setup/setup_baseline_env.sh``):
   blind preprocessing of the 2,400 prompts for Composable Diffusion,
   StructureDiffusion (Stanza 1.4.2), Attend-and-Excite and SynGen (spaCy 3.6.1
   ``en_core_web_trf``); writes ``<input-dir>/external_sidecars/`` and
   ``<input-dir>/ae_candidate/``. Run from the repository root with
   ``env -u LD_LIBRARY_PATH PYTHONPATH=src:. external/baseline_env/.venv/bin/python -m ...``.
   The parser installation digests recorded for the paper depend on the venv
   path, so a new venv needs ``--no-strict-environment`` (versions, processors
   and model names only) here and in the later steps that read these inputs
   (``select-attend-and-excite`` and ``run_baselines.py``).
2. ``select-attend-and-excite`` (main environment, one GPU): selects the
   Attend-and-Excite subtoken of every multi-subtoken noun and writes the final
   ``external_sidecars/sd14_attend_excite.jsonl`` with its selection evidence.
   It has no ``--allow-dirty``: the final input records its source files sealed
   to the Git commit, and generation requires that seal.
3. ``predicated-diffusion`` (main environment, CPU): spaCy ``en_core_web_sm``
   3.8.0 guidance targets of the Predicated Diffusion baseline,
   ``<input-dir>/sd14_pd_spacy.jsonl``.
4. ``sana``: compiles the Sana plans of the five baselines from these inputs
   into ``output/generation/<run-id>/protocol.json`` and ``protocol.plans.jsonl``.

Then run ``scripts/t2i_compbench/run_baselines.py``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ccg_diffusion.baselines.registry import EXTERNAL_ARMS
from ccg_diffusion.config import T2I_COMPBENCH_DATASET_DIR

DATASET_DIR = Path(T2I_COMPBENCH_DATASET_DIR)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    external = commands.add_parser("external", help="blind preprocessing of the four external SD1.4 baselines")
    external.add_argument("--input-dir", type=Path, required=True)
    external.add_argument("--arms", nargs="+", choices=EXTERNAL_ARMS, default=list(EXTERNAL_ARMS),
                          help="arm IDs of the external baselines to preprocess (docs/provenance.md)")
    external.add_argument("--stanza-resources", type=Path, default=Path("external/stanza_resources_1_4_2"))
    external.add_argument("--download-stanza", action="store_true")
    external.add_argument("--stanza-gpu", action="store_true")
    external.add_argument("--spacy-gpu-id", type=int, default=-1)
    external.add_argument("--spacy-batch-size", type=int, default=32)
    external.add_argument("--check-environment", action="store_true",
                          help="only compare the parser and adapter environments with the evaluated ones")
    external.add_argument("--no-strict-environment", action="store_true",
                          help="compare parser versions, processors and model names only, not their installation "
                               "digests (these depend on the venv path)")
    external.add_argument("--allow-download", action="store_true")
    external.add_argument("--overwrite", action="store_true")

    select = commands.add_parser("select-attend-and-excite", help="GPU subtoken selection of Attend-and-Excite")
    select.add_argument("--input-dir", type=Path, required=True)
    select.add_argument("--device", default="cuda:0")
    select.add_argument("--batch-size", type=int, default=4)
    select.add_argument("--expected-prompt-count", type=int, default=2400)
    select.add_argument("--expected-occurrence-count", type=int)
    select.add_argument("--no-strict-environment", action="store_true",
                        help="compare parser versions only, not their installation digests")
    select.add_argument("--allow-download", action="store_true")
    select.add_argument("--resume", action="store_true")

    predicated = commands.add_parser("predicated-diffusion", help="spaCy guidance targets of Predicated Diffusion")
    predicated.add_argument("--input-dir", type=Path, required=True)
    predicated.add_argument("--allow-download", action="store_true")
    predicated.add_argument("--overwrite", action="store_true")

    sana = commands.add_parser("sana", help="Sana plans of the five baselines")
    sana.add_argument("--run-id", required=True)
    sana.add_argument("--input-dir", type=Path, required=True,
                      help="directory with external_sidecars/ and sd14_pd_spacy.jsonl")
    sana.add_argument("--allow-dirty", action="store_true")
    sana.add_argument("--allow-download", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "external":
        from ccg_diffusion.baselines import preprocessing

        if args.spacy_gpu_id < -1:
            raise SystemExit("--spacy-gpu-id must be -1 or non-negative")
        if args.spacy_batch_size < 1:
            raise SystemExit("--spacy-batch-size must be positive")
        if args.check_environment:
            result = preprocessing.check_environment(
                arms=args.arms, stanza_resources=args.stanza_resources, stanza_gpu=args.stanza_gpu,
                spacy_gpu_id=args.spacy_gpu_id, local_files_only=not args.allow_download,
                strict_environment=not args.no_strict_environment)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result["ok"] else 1
        outputs = preprocessing.build_external_inputs(
            blind_prompt_manifest=DATASET_DIR / "prompt_only.jsonl",
            generation_manifest=DATASET_DIR / "independent_seed_manifest.jsonl",
            input_dir=args.input_dir,
            arms=args.arms,
            stanza_resources=args.stanza_resources,
            download_stanza=args.download_stanza,
            stanza_gpu=args.stanza_gpu,
            spacy_gpu_id=args.spacy_gpu_id,
            spacy_batch_size=args.spacy_batch_size,
            local_files_only=not args.allow_download,
            overwrite=args.overwrite,
            strict_environment=not args.no_strict_environment,
        )
        print(json.dumps({arm: str(path) for arm, path in outputs.items()}, indent=2))
    elif args.command == "select-attend-and-excite":
        from ccg_diffusion.baselines.sd14_generation import select_attend_and_excite_subtokens

        result = select_attend_and_excite_subtokens(
            generation_manifest=DATASET_DIR / "independent_seed_manifest.jsonl",
            blind_prompt_manifest=DATASET_DIR / "prompt_only.jsonl",
            input_dir=args.input_dir,
            device=args.device,
            batch_size=args.batch_size,
            expected_prompt_count=args.expected_prompt_count,
            expected_occurrence_count=args.expected_occurrence_count,
            local_files_only=not args.allow_download,
            resume=args.resume,
            strict_environment=not args.no_strict_environment,
        )
        print(json.dumps(result, sort_keys=True))
    elif args.command == "predicated-diffusion":
        from ccg_diffusion.baselines.predicated_diffusion import build_input_file
        from ccg_diffusion.targets.tokenizer import load_sd14_tokenizer

        metadata = build_input_file(
            DATASET_DIR / "prompt_only.jsonl",
            args.input_dir / "sd14_pd_spacy.jsonl",
            tokenizer=load_sd14_tokenizer(local_files_only=not args.allow_download),
            overwrite=args.overwrite,
        )
        print(json.dumps({key: metadata[key] for key in ("row_count", "status_counts", "output")}, indent=2))
    else:
        from ccg_diffusion.baselines.sana_plans import prepare_t2i_run
        from ccg_diffusion.evaluation.t2i_compbench.dataset import verify_dataset

        protocol = prepare_t2i_run(
            run_dir=Path("output/generation") / args.run_id,
            input_dir=args.input_dir,
            dataset_verification=verify_dataset(),
            allow_dirty=args.allow_dirty,
            local_files_only=not args.allow_download,
        )
        print(json.dumps({"plans": protocol["plans"], "census": protocol["census"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
