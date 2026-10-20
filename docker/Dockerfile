# CCG Diffusion runtime image: the Python environment of every command except the
# official T2I-CompBench++ evaluator and the baseline preprocessing, which run in
# their own environments under external/ (docs/installation.md).
#
#   docker build --target runtime -f docker/Dockerfile -t ccg-diffusion:runtime .
#
# Python 3.12 (NVIDIA PyTorch 24.12 base image), torch 2.13.0 built for CUDA 12.9,
# the pins of docker/requirements.txt, Hol-CCG (holccg 1.0.0 at its pinned commit)
# and the spaCy model en_core_web_sm 3.8.0. Models are not part of the image:
# download them with `python -m scripts.setup.download_models` (docs/installation.md).
FROM nvcr.io/nvidia/pytorch:24.12-py3 AS runtime

RUN apt-get update && apt-get upgrade -y && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# The base image's pip config (5 files from NVIDIA PyIndex) adds an extra index,
# pypi.ngc.nvidia.com, that no longer resolves; ignore all pip config files.
ENV PIP_CONFIG_FILE=/dev/null

# The base image ships packages built against its own torch 2.6 / numpy 1.26 ABI
# that are not used here and conflict with torch 2.13 / numpy 2.5.1 (RAPIDS 24.10,
# librosa, lightning-thunder, DALI, ...). They are removed so that the final
# `pip check` passes; nvidia-nvimgcodec-cu12 is removed because its wheel tag makes
# `pip check` itself fail.
RUN pip install --no-cache-dir --upgrade pip && \
    { pip uninstall -y \
        transformer_engine apex flash_attn torch_tensorrt nvidia-modelopt \
        nvidia-dali-cuda120 nvidia-nvimgcodec-cu12 numba llvmlite cupy-cuda12x \
        opencv pycocotools benepar torch-struct deepspeed datasets \
        cudf cudf-polars dask-cudf pylibcudf cuml cugraph cugraph-pyg cugraph-dgl \
        cugraph-service-server pylibcugraph nx-cugraph raft-dask pylibraft cuvs \
        rmm kvikio dask-cuda distributed-ucxx ucxx librosa lightning-thunder \
        || true; }

# torch comes from the CUDA 12.9 index: the PyPI default wheel targets CUDA 13
# and needs an NVIDIA driver >= 580.
RUN pip install --no-cache-dir torch==2.13.0 torchvision==0.28.0 \
    --index-url https://download.pytorch.org/whl/cu129

COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Hol-CCG at the pinned commit and the pinned spaCy model; `pip check` fails the
# build on any dependency conflict.
RUN pip install --no-cache-dir \
    "holccg[prolog,hub] @ git+https://github.com/ryosuke-yamaki/hol-ccg@0213ca85d2397c5a4fb2c03bc1b20cf1a6283d26" && \
    pip install --no-cache-dir \
    https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl && \
    pip check

# A user with the host UID/GID, so that files written into the mounted checkout
# keep the host owner. A base-image user with the same UID is renamed.
ARG HOST_UID=1000
ARG HOST_GID=1000
RUN EXISTING_USER=$(getent passwd ${HOST_UID} | cut -d: -f1) && \
    if [ -n "$EXISTING_USER" ]; then \
    usermod -l devuser -d /home/devuser -m "$EXISTING_USER"; \
    else \
    EXISTING_GROUP=$(getent group ${HOST_GID} | cut -d: -f1) && \
    [ -z "$EXISTING_GROUP" ] && groupadd -g ${HOST_GID} devgroup || true && \
    useradd -m -u ${HOST_UID} -g ${HOST_GID} -s /bin/bash devuser; \
    fi && \
    mkdir -p /home/devuser/.cache/huggingface && \
    chown -R ${HOST_UID}:${HOST_GID} /home/devuser/.cache

# The checkout is mounted at /workspace/${PROJECT_NAME}; the package is used from
# the source tree (the equivalent of `pip install -e .`).
ARG PROJECT_NAME=ccg-diffusion
ENV PYTHONPATH=/workspace/${PROJECT_NAME}/src
WORKDIR /workspace/${PROJECT_NAME}
USER devuser
