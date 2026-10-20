"""Generate the T2I-CompBench++ baseline images of Table 2 on SD1.4 or Sana 0.6B.

SD1.4 (``--backbone sd14``) runs one shard of one method over the 24,000 image
rows into ``output/generation/<run-id>/<arm>/``, where ``<arm>`` is the
method's recorded arm ID (registry ``T2I_BASELINE_ARMS``, e.g. ``sd14_composable``;
see the glossary in docs/provenance.md):

* ``--method predicated_diffusion``: the SD1.4 batch driver (arm ID
  ``sd14_pd_spacy``) on ``<input-dir>/sd14_pd_spacy.jsonl``; prompts whose
  spaCy targets fail to align are generated as stock Vanilla;
* the other methods: ``baselines.sd14_generation`` on
  ``<input-dir>/external_sidecars/<arm>.jsonl``.

The input directory defaults to ``output/generation/<run-id>/input``
(``prepare_baseline_inputs.py``). ``--plan-only`` prints the run-lock
selection fields and the batch membership of the shard without a GPU.
The default of four shards (``--shard-index 0..3``, one GPU each) reproduces the batches of the evaluated
six-method run ``20260812_t2icb_sd14_six_arm_full_v2``.

Sana (``--backbone sana``) generates one method of the run prepared by
``prepare_baseline_inputs.py sana`` into
``output/generation/<run-id>/<arm>/<category>/``, one category at a time;
``--audit`` checks finished categories instead.

SD1.4 Vanilla is the official T2I-CompBench reference generator
(``scripts/t2i_compbench/official_reference/``); Sana Vanilla is
``run_ccg_diffusion.py prepare --backbone sana --method vanilla``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ccg_diffusion.baselines.registry import T2I_BASELINE_ARMS
from ccg_diffusion.config import T2I_COMPBENCH_CATEGORIES, T2I_COMPBENCH_DATASET_DIR

DATASET_DIR = Path(T2I_COMPBENCH_DATASET_DIR)


def _indices(text: str) -> tuple[int, ...]:
    """Parse ``0,7,20`` or ``3-5`` lists of physical-batch indices."""

    values: list[int] = []
    for part in text.split(","):
        if "-" in part:
            start, end = part.split("-", 1)
            values.extend(range(int(start), int(end) + 1))
        else:
            values.append(int(part))
    return tuple(values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backbone", choices=("sd14", "sana"), required=True)
    parser.add_argument("--method", choices=[*sorted(T2I_BASELINE_ARMS["sd14"]), "vanilla"], required=True)
    parser.add_argument("--run-id", help="output/generation/<run-id>; required unless --plan-only")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path,
                        help="write images here instead of output/generation/<run-id> (verification subsets)")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    sd14 = parser.add_argument_group("SD1.4")
    sd14.add_argument("--input-dir", type=Path, help="default: output/generation/<run-id>/input")
    sd14.add_argument("--num-shards", type=int, default=4,
                      help="4 reproduces the physical batches of the evaluated six-method SD1.4 baseline run")
    sd14.add_argument("--shard-index", type=int, default=0)
    sd14.add_argument("--batches", type=_indices, help="physical-batch indices of the shard, e.g. 0,7,20")
    sd14.add_argument("--expected-total", type=int)
    sd14.add_argument("--resume", action="store_true")
    sd14.add_argument("--plan-only", action="store_true")
    sd14.add_argument("--plan-output", type=Path, help="write the --plan-only JSON here instead of stdout")
    sd14.add_argument("--no-strict-runtime", action="store_true",
                      help="record, but do not require, the pinned Python/torch/diffusers/transformers/CUDA versions")
    sd14.add_argument("--no-strict-environment", action="store_true",
                      help="compare parser versions only, not their installation digests")
    sana = parser.add_argument_group("Sana")
    sana.add_argument("--categories", nargs="+", choices=T2I_COMPBENCH_CATEGORIES)
    sana.add_argument("--keys", type=lambda text: tuple(text.split(",")), help="base prompt IDs, comma separated")
    sana.add_argument("--audit", action="store_true")
    return parser


def run_sd14(args: argparse.Namespace) -> None:
    arm = T2I_BASELINE_ARMS["sd14"][args.method]
    out_dir = args.output_dir or Path("output/generation") / str(args.run_id)
    input_dir = args.input_dir or Path("output/generation") / str(args.run_id) / "input"
    prompt_manifest = DATASET_DIR / "independent_seed_manifest.jsonl"
    if args.method == "predicated_diffusion":
        from ccg_diffusion.backbones.sd14 import driver
        from ccg_diffusion.backbones.sd14.inputs import load_t2icb_dataset_provenance

        request = driver.SD14Run(
            method=arm, prompt_manifest=prompt_manifest, out_dir=out_dir,
            dataset=load_t2icb_dataset_provenance(prompt_manifest), target_file=input_dir / "sd14_pd_spacy.jsonl",
            index_failure_policy="stock_vanilla", num_shards=args.num_shards, shard_index=args.shard_index,
            expected_total=args.expected_total, device=args.device, local_files_only=not args.allow_download,
            resume=args.resume, strict_runtime=not args.no_strict_runtime, allow_dirty=args.allow_dirty,
            batches=args.batches)
        if args.plan_only:
            from ccg_diffusion.baselines.sd14_generation import driver_plan_projection

            return _write_plan(driver_plan_projection(request), args.plan_output)
        driver.run(request)
        return None
    from ccg_diffusion.baselines import preprocessing, sd14_generation

    options = sd14_generation.RunOptions(
        arm=arm, input_file=preprocessing.input_path(input_dir, arm), out_dir=out_dir,
        prompt_manifest=prompt_manifest, blind_prompt_manifest=DATASET_DIR / "prompt_only.jsonl",
        device=args.device, num_shards=args.num_shards, shard_index=args.shard_index,
        expected_total=args.expected_total, batches=args.batches, local_files_only=not args.allow_download,
        resume=args.resume, strict_runtime=not args.no_strict_runtime,
        strict_environment=not args.no_strict_environment, allow_dirty=args.allow_dirty)
    if args.plan_only:
        return _write_plan(sd14_generation.plan_projection(options), args.plan_output)
    sd14_generation.run(options)
    return None


def _write_plan(plan: dict, output: Path | None) -> None:
    text = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    if output is None:
        print(text, end="")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)


def run_sana(args: argparse.Namespace) -> None:
    from ccg_diffusion.baselines.sana_generation import audit_t2i_category, generate_t2i_run
    from ccg_diffusion.provenance import read_jsonl, resolve_recorded_path

    arm = T2I_BASELINE_ARMS["sana"][args.method]
    run_dir = Path("output/generation") / str(args.run_id)
    protocol = run_dir / "protocol.json"
    plan_file = resolve_recorded_path(json.loads(protocol.read_text())["plans"]["path"])
    plans = [row for row in read_jsonl(plan_file) if row["method"] == arm]
    categories = list(args.categories or T2I_COMPBENCH_CATEGORIES)
    if args.keys is not None:
        experiments = {row["experiment"] for row in plans if row["prompt_id"] in args.keys}
        categories = [category for category in categories if f"t2icb_{category}" in experiments]
    for category in categories:
        if args.audit:
            print(json.dumps(audit_t2i_category(run=run_dir, method=arm, category=category)), flush=True)
            continue
        keys = None
        if args.keys is not None:
            keys = [key for key in args.keys if any(
                row["prompt_id"] == key and row["experiment"] == f"t2icb_{category}" for row in plans)]
        generate_t2i_run(protocol=protocol, method=arm, output=(args.output_dir or run_dir) / arm / category,
                         device=args.device, categories=[category], keys=keys, allow_dirty=args.allow_dirty)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.method == "vanilla":
        print("Vanilla is not run by this command: SD1.4 Vanilla is the official T2I-CompBench reference "
              "generator (scripts/t2i_compbench/official_reference/), Sana Vanilla is "
              "`run_ccg_diffusion.py prepare --backbone sana --method vanilla`.", file=sys.stderr)
        return 2
    if args.run_id is None and not (args.plan_only and args.input_dir is not None):
        parser.error("--run-id is required (or --plan-only with --input-dir)")
    if args.backbone == "sd14":
        run_sd14(args)
    else:
        run_sana(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
