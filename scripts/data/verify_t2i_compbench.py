#!/usr/bin/env python3
"""Verify the frozen full T2I-CompBench++ dataset against its contract and print the verification record.

    python -m scripts.data.verify_t2i_compbench [--dataset-dir DIR]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    from ccg_diffusion.evaluation.t2i_compbench.dataset import DATASET_DIR, verify_dataset

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    args = parser.parse_args(argv)
    result = verify_dataset(args.dataset_dir)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
