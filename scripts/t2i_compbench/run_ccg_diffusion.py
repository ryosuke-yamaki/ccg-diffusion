"""CCG Diffusion on T2I-CompBench++ with Stable Diffusion v1.4 or Sana 0.6B: prepare, generate, audit, one prompt.

Run from the repository root (one backbone per process)::

    # full method (SD1.4: 4 shards; Sana: unsharded)
    python -m scripts.t2i_compbench.run_ccg_diffusion prepare --backbone sd14 --run-id R --source-manifest S
    python -m scripts.t2i_compbench.run_ccg_diffusion generate --backbone sd14 --run-id R --shard-index 0 \\
        --device cuda:0
    python -m scripts.t2i_compbench.run_ccg_diffusion audit --backbone sd14 --run-id R
    # an ablation of Table 3, 4 or 7 (list: python -m ccg_diffusion.methods ablations)
    python -m scripts.t2i_compbench.run_ccg_diffusion prepare --backbone sana --run-id R --source-manifest S \\
        --remove positional_prefix
    # Sana Vanilla
    python -m scripts.t2i_compbench.run_ccg_diffusion prepare --backbone sana --run-id R --method vanilla
    # one arbitrary prompt (not paper-exact)
    python -m scripts.t2i_compbench.run_ccg_diffusion prompt --backbone sd14 \\
        --prompt "a blue banana and a green vase" --output DIR

``S`` is the Stage 1-3 source manifest written by ``scripts.t2i_compbench.prepare_prompts``.
Generated images go to ``output/generation/R/`` (SD1.4: ``[<condition>/]images``,
Sana: ``<condition>/``); ``--batches`` (SD1.4) and ``--keys`` (Sana) generate a
subset, for example into ``--output-dir``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from ccg_diffusion.methods import T2I_REMOVABLE

BACKBONES = ("sd14", "sana")


def _batches(text: str) -> tuple[int, ...]:
    """``"165,188,749-751"`` -> ``(165, 188, 749, 750, 751)``."""
    values: list[int] = []
    for part in text.split(","):
        first, _, last = part.partition("-")
        values.extend(range(int(first), int(last or first) + 1))
    return tuple(values)


def _keys(text: str) -> tuple[str, ...]:
    return tuple(part for part in text.split(",") if part)


def _ablation(args: argparse.Namespace) -> Any:
    """``None`` without ``--remove`` (the full method); otherwise the ablation that removes the names."""
    from ccg_diffusion.methods import T2IAblation

    return None if args.remove is None else T2IAblation.from_removed(args.remove)


def _sd14_ablation(args: argparse.Namespace) -> Any:
    from ccg_diffusion.methods import FULL_T2I

    if args.method != "ccg_diffusion":
        raise SystemExit("SD1.4 Vanilla is a baseline: use scripts.t2i_compbench.run_baselines")
    return _ablation(args) or FULL_T2I


def prepare(args: argparse.Namespace) -> None:
    if args.backbone == "sd14":
        from ccg_diffusion.compositional import sd14
        from ccg_diffusion.targets.tokenizer import load_sd14_tokenizer

        if args.source_manifest is None:
            raise SystemExit("prepare --backbone sd14 requires --source-manifest")
        path = sd14.prepare(run_id=args.run_id, source_manifest=args.source_manifest,
                            tokenizer=load_sd14_tokenizer(local_files_only=not args.allow_download),
                            ablation=_sd14_ablation(args), allow_dirty=args.allow_dirty)
        protocol = json.loads(path.read_text())
        summary = {key: protocol[key] for key in ("condition_id", "per_prompt_guidance_sha256",
                                                  "per_prompt_count_guidance_sha256", "selected_image_ids_sha256",
                                                  "census")}
    else:
        from ccg_diffusion.compositional import sana

        path = sana.prepare_run(run_id=args.run_id, method=args.method, ablation=_ablation(args),
                                source_manifest=args.source_manifest, allow_dirty=args.allow_dirty)
        protocol = json.loads(path.read_text())
        summary = {"condition": protocol["condition"], "plans": protocol["plans"], "census": protocol["census"]}
    print(json.dumps({"protocol": str(path), **summary}, indent=2), flush=True)


def generate(args: argparse.Namespace) -> None:
    if args.backbone == "sd14":
        from ccg_diffusion.compositional import sd14

        if args.keys is not None:
            raise SystemExit("--keys selects Sana prompts; SD1.4 subsets use --batches")
        output = sd14.generate(run_id=args.run_id, ablation=_sd14_ablation(args), shard_index=args.shard_index,
                               device=args.device, batches=args.batches, allow_dirty=args.allow_dirty,
                               output_dir=args.output_dir, local_files_only=not args.allow_download,
                               strict_runtime=not args.no_strict_runtime)
    else:
        from ccg_diffusion.compositional import sana

        if args.batches is not None or args.shard_index != 0:
            raise SystemExit("Sana runs are unsharded; select prompts with --keys")
        if args.no_strict_runtime:
            raise SystemExit("--no-strict-runtime applies to SD1.4 only")
        output = sana.generate_run(run_id=args.run_id, method=args.method, ablation=_ablation(args),
                                   device=args.device, keys=args.keys, output_dir=args.output_dir,
                                   allow_dirty=args.allow_dirty)
    print(json.dumps({"output_dir": str(output)}), flush=True)


def audit(args: argparse.Namespace) -> None:
    if args.backbone == "sd14":
        from ccg_diffusion.compositional import sd14

        result = sd14.audit(run_id=args.run_id, ablation=_sd14_ablation(args), batches=args.batches,
                            output_dir=args.output_dir)
    else:
        from ccg_diffusion.compositional import sana

        result = sana.audit_run(run_id=args.run_id, method=args.method, ablation=_ablation(args), keys=args.keys,
                                output_dir=args.output_dir)
    print(json.dumps(result, indent=2), flush=True)


def prompt(args: argparse.Namespace) -> None:
    from ccg_diffusion import single_prompt

    result = single_prompt.generate(args.prompt, backbone=args.backbone, seed=args.seed, device=args.device,
                                    remove=args.remove or (), local_files_only=not args.allow_download)
    args.output.mkdir(parents=True, exist_ok=True)
    result.image.save(args.output / "image.png")
    (args.output / "record.json").write_text(json.dumps(result.record, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "guidance": result.record["guidance"]}), flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    remove_help = ("remove these components or losses (an ablation of Tables 3, 4, 7); --remove without names is "
                   "the all-components ablation cell gs1_p1_r1, which Sana records as an ablation")

    def common(command: argparse.ArgumentParser) -> None:
        command.add_argument("--backbone", choices=BACKBONES, required=True)
        command.add_argument("--run-id", required=True, help="YYYYMMDD_slug; outputs go to output/generation/RUN_ID")
        command.add_argument("--method", choices=("ccg_diffusion", "vanilla"), default="ccg_diffusion",
                             help="vanilla: stock Sana from the prompt (Sana only)")
        command.add_argument("--remove", nargs="*", choices=T2I_REMOVABLE, metavar="NAME",
                             help=remove_help + f"; names: {', '.join(T2I_REMOVABLE)}")

    prep = sub.add_parser("prepare", help="derive the per-prompt plans and freeze the run protocol")
    common(prep)
    prep.add_argument("--source-manifest", type=Path, help="Stage 1-3 source manifest (prepare_prompts)")
    prep.add_argument("--allow-dirty", action="store_true", help="record uncommitted sources (dirty: true)")
    prep.add_argument("--allow-download", action="store_true", help="allow tokenizer downloads")
    prep.set_defaults(function=prepare)

    gen = sub.add_parser("generate", help="generate the images of a prepared run (SD1.4: one shard)")
    common(gen)
    gen.add_argument("--shard-index", type=int, choices=range(4), default=0, help="SD1.4 shard (image index %% 4)")
    gen.add_argument("--batches", type=_batches, help="SD1.4: only these physical batches of the shard, e.g. 165,188")
    gen.add_argument("--keys", type=_keys, help="Sana: only these base prompt IDs (all seeds), comma separated")
    gen.add_argument("--device", default="cuda:0")
    gen.add_argument("--output-dir", type=Path, help="write a verification subset here instead of the run")
    gen.add_argument("--allow-dirty", action="store_true", help="accept a protocol prepared with --allow-dirty")
    gen.add_argument("--allow-download", action="store_true", help="allow model downloads")
    gen.add_argument("--no-strict-runtime", action="store_true",
                     help="SD1.4 only: record, but do not require, the pinned Python/torch/diffusers/transformers/"
                          "CUDA/cuDNN versions")
    gen.set_defaults(function=generate)

    check = sub.add_parser("audit", help="check the generated images against the protocol")
    common(check)
    check.add_argument("--batches", type=_batches, help="SD1.4: audit the subset generated with these batches")
    check.add_argument("--keys", type=_keys, help="Sana: audit the subset generated with these prompts")
    check.add_argument("--output-dir", type=Path, help="the output directory given to generate, if any")
    check.set_defaults(function=audit)

    single = sub.add_parser("prompt", help="generate one image of an arbitrary prompt (not paper-exact)")
    single.add_argument("--backbone", choices=BACKBONES, required=True)
    single.add_argument("--prompt", required=True, help="the prompt, exactly one sentence")
    single.add_argument("--seed", type=int, default=42)
    single.add_argument("--remove", nargs="*", choices=T2I_REMOVABLE, metavar="NAME", help=remove_help)
    single.add_argument("--output", type=Path, required=True, help="writes image.png and record.json")
    single.add_argument("--device", default="cuda:0")
    single.add_argument("--allow-download", action="store_true", help="allow model downloads")
    single.set_defaults(function=prompt)

    args = parser.parse_args(argv)
    args.function(args)


if __name__ == "__main__":
    main()
