#!/usr/bin/env bash
# Sequential driver for official_reference/generate.py over several T2I-CompBench++ categories.
#
# One GPU, one job at a time: the categories run strictly in series so a single
# detached process owns the GPU for the whole batch. Per-category output goes to
# output/generation/<run_id>/logs/<category>.log; this script's own stdout carries
# only the start/finish milestones and the image count per category.
#
# Usage (from the repository root):
#   mkdir -p output/generation/RUN_ID/logs
#   setsid nohup bash scripts/t2i_compbench/official_reference/run_all_categories.sh \
#     RUN_ID color shape texture spatial 3d_spatial non_spatial numeracy complex \
#     > output/generation/RUN_ID/logs/all_categories.log 2>&1 &
#
# Each category is 300 prompts x 10 images = 3,000 images.
# A category that fails stops the driver (set -e); the per-category resume guard in
# official_reference/generate.py makes a restart safe.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VENV_PY="${REPO_ROOT}/external/t2i_compbench/.venv/bin/python"
GEN_SCRIPT="${REPO_ROOT}/scripts/t2i_compbench/official_reference/generate.py"
PROMPT_DIR="${REPO_ROOT}/external/t2i_compbench/T2I-CompBench/examples/dataset"
MODEL_ID="${MODEL_ID:-CompVis/stable-diffusion-v1-4}"

if [ "$#" -lt 2 ]; then
    echo "usage: $0 <run_id> <category> [<category> ...]" >&2
    exit 2
fi

RUN_ID="$1"
shift
CATEGORIES=("$@")

RUN_DIR="${REPO_ROOT}/output/generation/${RUN_ID}"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"

# Fail before spending GPU hours if any prompt file is missing.
for category in "${CATEGORIES[@]}"; do
    if [ ! -f "${PROMPT_DIR}/${category}_val.txt" ]; then
        echo "missing prompt file: ${PROMPT_DIR}/${category}_val.txt" >&2
        exit 2
    fi
done

echo "run_id=${RUN_ID} model_id=${MODEL_ID} categories=${CATEGORIES[*]}"

for category in "${CATEGORIES[@]}"; do
    outdir="${RUN_DIR}/${category}"
    log="${LOG_DIR}/${category}.log"
    echo "=== [$(date -Is)] start ${category} -> ${outdir} (log: ${log})"
    # LD_LIBRARY_PATH / PYTHONPATH are cleared so the venv's torch is not shadowed
    # by the container's system torch libraries.
    env -u LD_LIBRARY_PATH -u PYTHONPATH "${VENV_PY}" "${GEN_SCRIPT}" \
        --model-id "${MODEL_ID}" \
        --from-file "${PROMPT_DIR}/${category}_val.txt" \
        --outdir "${outdir}" \
        >> "${log}" 2>&1
    echo "=== [$(date -Is)] done ${category}: $(find "${outdir}/samples" -name '*.png' | wc -l) images"
done

echo "=== [$(date -Is)] all categories complete: ${CATEGORIES[*]}"
