#!/usr/bin/env bash
# Serve nvidia/Qwen3.8-Flash-Next-NVFP4 (the open VQA judge of the TAB PP method comparison) on two nodes:
# the head (rank 0, where this script runs) and a worker (rank 1), with stock vllm/vllm-openai:v0.30.0,
# TP2 + EP over a direct QSFP (RoCE) link. The paper run used two NVIDIA DGX Spark systems.
#
# Required environment variables (placeholders, and the device names of a DGX Spark, in parentheses):
#   HEAD_IP    IP of the head node on the direct link   (<head-ip>)
#   WORKER_IP  IP of the worker node on the direct link (<worker-ip>)
#   IFACE      network interface of the direct link     (e.g. enp1s0f0np0)
#   IB_HCA     NCCL_IB_HCA value                        (e.g. =rocep1s0f0)
# Optional: GMU (GPU memory utilization, default 0.74).
# Usage (on the head node): ./serve.sh start|stop|status
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

IMAGE="vllm/vllm-openai:v0.30.0"
MODEL="nvidia/Qwen3.8-Flash-Next-NVFP4"
REVISION="fc694b54fb0174e0913e6adf86691ef85a4ead47"
HEAD_IP="${HEAD_IP:?set HEAD_IP}"
WORKER_IP="${WORKER_IP:?set WORKER_IP}"
IFACE="${IFACE:?set IFACE}"
IB_HCA="${IB_HCA:?set IB_HCA}"
GMU="${GMU:-0.74}"   # default 0.74 (the recipe uses 0.80)
PORT=8888
NAME="vllm-qwen38fn"

VLLM_ARGS="$MODEL --revision $REVISION --served-model-name qwen3.8-flash-next \
 --tensor-parallel-size 2 --nnodes 2 --master-addr $HEAD_IP --master-port 50000 \
 --distributed-executor-backend mp --enable-expert-parallel --all2all-backend allgather_reducescatter \
 --gpu-memory-utilization $GMU --max-num-seqs 8 --max-num-batched-tokens 8192 --max-model-len 65536 \
 --kv-cache-dtype auto --mamba-ssm-cache-dtype bfloat16 \
 --load-format safetensors --safetensors-load-strategy lazy --enable-chunked-prefill \
 --reasoning-parser qwen3 --mm-encoder-tp-mode data --enable-prompt-tokens-details \
 --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":3}' \
 --compilation-config '{\"mode\":0,\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}'"

docker_cmd() {  # $1 = node IP
    echo "docker run -d --name $NAME --gpus all --network host --ipc host \
 --cap-add SYS_NICE --ulimit memlock=-1 --ulimit stack=67108864 --device /dev/infiniband:/dev/infiniband \
 -e GLOO_SOCKET_IFNAME=$IFACE -e NCCL_SOCKET_IFNAME=$IFACE -e TP_SOCKET_IFNAME=$IFACE \
 -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=$IB_HCA -e NCCL_IB_GID_INDEX=3 -e NCCL_IB_AUTO_DETECT=0 -e NCCL_DEBUG=WARN \
 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e VLLM_HOST_IP=$1 -e VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
 -e HF_HOME=/root/.cache/huggingface -v \$HOME/.cache/huggingface:/root/.cache/huggingface \
 -v \$HOME/.cache/vllm:/root/.cache/vllm $IMAGE"
}

on_node() {  # $1 = local|IP, $2 = command
    if [[ "$1" == local ]]; then bash -c "$2"; else ssh -o BatchMode=yes "$1" "$2"; fi
}

case "${1:-}" in
start)
    on_node "$WORKER_IP" "mkdir -p ~/qwen38fn"
    scp -q evict_ckpt.py "$WORKER_IP:qwen38fn/evict_ckpt.py"
    for node in local "$WORKER_IP"; do
        if on_node "$node" "docker ps -a --format {{.Names}} | grep -qx $NAME"; then
            echo "container $NAME already exists on $node; run stop first" >&2; exit 1
        fi
        on_node "$node" "mkdir -p ~/.cache/vllm && python3 ~/qwen38fn/evict_ckpt.py && free -g | head -2"
    done
    on_node "$WORKER_IP" "$(docker_cmd $WORKER_IP) $VLLM_ARGS --node-rank 1 --headless"
    sleep 15
    on_node local "$(docker_cmd $HEAD_IP) $VLLM_ARGS --node-rank 0 --host 0.0.0.0 --port $PORT"
    echo "started; follow with: docker logs -f $NAME"
    ;;
stop)
    docker rm -f "$NAME" || true
    on_node "$WORKER_IP" "docker rm -f $NAME" || true
    ;;
status)
    docker ps --filter name=$NAME --format '{{.Names}} {{.Status}}'
    on_node "$WORKER_IP" "docker ps --filter name=$NAME --format '{{.Names}} {{.Status}}'"
    curl -s -o /dev/null -w "health %{http_code}\n" "http://localhost:$PORT/health" || true
    ;;
*) echo "usage: $0 start|stop|status" >&2; exit 2 ;;
esac
