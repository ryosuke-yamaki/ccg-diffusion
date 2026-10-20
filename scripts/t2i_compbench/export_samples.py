"""Export the images of one method or ablation condition (arm) into the official T2I-CompBench++ ``samples/`` layout.

    python -m scripts.t2i_compbench.export_samples --run-id RUN --arm ARM --method METHOD \\
        --generation-out-dir output/generation/RUN [--categories color shape ...]

For each category, links every image as ``<prompt>_<index:06d>.png`` under
``output/vqa/RUN/t2icb_work/ARM/<category>/samples/`` and writes the sealed
``metric_stage_manifest.json`` that ``scripts.t2i_compbench.evaluate --arm ARM`` checks.
``ARM`` is the recorded arm ID (docs/provenance.md): a method, or in an ablation
run a condition such as ``gs0_p1_r1``. ``METHOD`` is the method directory of the
generation run; the arm ID of a method must equal it, while an ablation
condition keeps the method that produced it.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    from ccg_diffusion.evaluation.t2i_compbench.dataset import CATEGORIES, DATASET_DIR, REPO_ROOT, verify_dataset
    from ccg_diffusion.evaluation.t2i_compbench.export import EXPORT_METHODS, export_samples

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--arm", required=True,
                        help="recorded arm ID: a method or an ablation condition (docs/provenance.md); also the "
                             "output directory under t2icb_work/")
    parser.add_argument("--method", choices=EXPORT_METHODS, required=True)
    parser.add_argument("--generation-out-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--generation-manifest", type=Path, default=None,
                        help="default: <dataset-dir>/independent_seed_manifest.jsonl")
    parser.add_argument("--categories", nargs="+", choices=CATEGORIES, default=list(CATEGORIES))
    parser.add_argument("--vqa-root", type=Path, default=REPO_ROOT / "output/vqa")
    args = parser.parse_args(argv)
    if args.dataset_dir.resolve() == DATASET_DIR.resolve():
        verify_dataset(args.dataset_dir)
    for category in args.categories:
        manifest = export_samples(category, run_id=args.run_id, arm=args.arm, method=args.method,
                                  generation_out_dir=args.generation_out_dir, dataset_dir=args.dataset_dir,
                                  generation_manifest=args.generation_manifest, vqa_root=args.vqa_root)
        print(f"[{category}] staged {manifest['image_count']} independent-seed images", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
