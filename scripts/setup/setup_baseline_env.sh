#!/usr/bin/env bash
# Create the isolated Python 3.11 environment used only by
#   python -m scripts.t2i_compbench.prepare_baseline_inputs external
# (input preparation of Composable Diffusion, StructureDiffusion, Attend-and-Excite and SynGen).
#
# Usage: bash scripts/setup/setup_baseline_env.sh   (no arguments; configure it with the variables below)
#
# Installs docker/requirements-baselines.lock.txt (the `pip freeze --all` of the evaluated environment;
# its `pip @ file://...` line is skipped) with `uv pip`, the installer of the evaluated environment, downloads
# the Stanza 1.4.2 English resources, and compares the parser versions, processors and model names with the
# evaluated ones. The recorded freeze sha256 is
# 17e70ba2ee3e6041086d524bd374dfb00c0847c8711f7d58bceba8ce2f2254dc.
#
# The parser installation digests of the evaluated inputs (RECORDED_PARSER_ENVIRONMENT in
# ccg_diffusion.baselines.preprocessing) depend on the venv path: the spaCy distribution lists the
# bin/spacy launcher, which holds the absolute path of the venv's Python. A new venv therefore matches
# the versions but not every digest. Pass --no-strict-environment to
# `prepare_baseline_inputs external` and `select-attend-and-excite` and to `run_baselines.py` for inputs
# built in this venv.
#
# Environment variables: BASELINE_ENV_DIR (default external/baseline_env),
# BASELINE_PYTHON_VERSION (default 3.11.15), STANZA_RESOURCES_DIR (default external/stanza_resources_1_4_2).
set -euo pipefail
if [ "$#" -ne 0 ]; then
    sed -n '2,/^set -euo pipefail$/{/^#/s/^# \{0,1\}//p}' "${BASH_SOURCE[0]}"
    case "$1" in -h|--help) exit 0 ;; *) exit 2 ;; esac
fi
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_DIR="${BASELINE_ENV_DIR:-${REPO_ROOT}/external/baseline_env}"
PYTHON_VERSION="${BASELINE_PYTHON_VERSION:-3.11.15}"
STANZA_DIR="${STANZA_RESOURCES_DIR:-${REPO_ROOT}/external/stanza_resources_1_4_2}"
REQUIREMENTS="${REPO_ROOT}/docker/requirements-baselines.lock.txt"
[ -f "${REQUIREMENTS}" ] || REQUIREMENTS="${REPO_ROOT}/docker/requirements-baselines.txt"
command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 2; }
uv python install "${PYTHON_VERSION}"
# --seed installs pip: the environment record runs `python -m pip freeze --all`.
uv venv --seed --python "${PYTHON_VERSION}" "${ENV_DIR}/.venv"
PY="${ENV_DIR}/.venv/bin/python"
mkdir -p "${ENV_DIR}"
grep -v '^pip @ ' "${REQUIREMENTS}" > "${ENV_DIR}/requirements.txt"
# The container exports LD_LIBRARY_PATH/PYTHONPATH for its own torch; keep them out of this venv.
env -u LD_LIBRARY_PATH -u PYTHONPATH uv pip install --python "${PY}" -r "${ENV_DIR}/requirements.txt"
env -u LD_LIBRARY_PATH -u PYTHONPATH "${PY}" -c "import stanza; stanza.download('en', model_dir='${STANZA_DIR}', processors='tokenize,pos,constituency', verbose=True)"
env -u LD_LIBRARY_PATH -u PYTHONPATH "${PY}" -m pip freeze --all | sort > "${ENV_DIR}/venv_freeze.txt"
echo "pip freeze sha256: $(sha256sum "${ENV_DIR}/venv_freeze.txt" | cut -d' ' -f1)" \
     "(recorded: 17e70ba2ee3e6041086d524bd374dfb00c0847c8711f7d58bceba8ce2f2254dc)"
# Fail if the parser versions, processors or model names differ from the evaluated ones.
cd "${REPO_ROOT}"
env -u LD_LIBRARY_PATH PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}" "${PY}" -m scripts.t2i_compbench.prepare_baseline_inputs \
    external --input-dir "${ENV_DIR}/check" --check-environment --no-strict-environment --stanza-resources "${STANZA_DIR}"
