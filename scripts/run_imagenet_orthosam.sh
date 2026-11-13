#!/usr/bin/env bash
set -euo pipefail

METHOD="OrthoSAM"

if [[ $# -gt 0 && "${1}" != -* ]]; then
    CONFIG_PATH="$1"
    EXTRA_ARGS=("${@:2}")
else
    CONFIG_PATH=configs/ImageNet1k_ResNet50.ini
    EXTRA_ARGS=("$@")
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "[ERROR] config not found: ${CONFIG_PATH}" >&2
    echo "        run this script from the repository root (the directory containing main.py)" >&2
    exit 1
fi

detect_gpus() {
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}"
    elif command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index --format=csv,noheader | wc -l
    else
        echo 1
    fi
}

NPROC=${NPROC:-$(detect_gpus)}

if [[ -z "${BATCH_SIZE:-}" && -z "${LEARNING_RATE:-}" && "${NPROC}" -eq 1 ]]; then
    BATCH_SIZE=128
    LEARNING_RATE=0.05
fi

pick_master_port() {
    python - <<'INNER'
import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
INNER
}

master_port="${MASTER_PORT:-$(pick_master_port)}"

cmd=(
    python -m torch.distributed.run
    --nproc_per_node "${NPROC}"
    --master_port "${master_port}"
    main.py
    --ifile "${CONFIG_PATH}"
    --optimizer "${METHOD}"
)

if [[ -n "${IMAGENET_RHO:-}" ]]; then
    cmd+=(--rho "${IMAGENET_RHO}" --samRho "${IMAGENET_RHO}" --orthosamRho "${IMAGENET_RHO}")
fi

if [[ -n "${EPOCHS:-}" ]]; then
    cmd+=(--epochs "${EPOCHS}")
fi
if [[ -n "${BATCH_SIZE:-}" ]]; then
    cmd+=(--batchSize "${BATCH_SIZE}")
fi
if [[ -n "${LEARNING_RATE:-}" ]]; then
    cmd+=(--learningRate "${LEARNING_RATE}")
fi
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    cmd+=("${EXTRA_ARGS[@]}")
fi

echo "[CONFIG] ${CONFIG_PATH}"
echo "[METHOD] ${METHOD}"
echo "[GPUS] ${NPROC}"
echo "[BATCH] ${BATCH_SIZE:-<ini>}  [LR] ${LEARNING_RATE:-<ini>}  [RHO] ${IMAGENET_RHO:-<ini>}"
echo "[RUN] port=${master_port}"
start_time=${SECONDS}
"${cmd[@]}"
echo "[OK] ${METHOD} (elapsed $(( (SECONDS - start_time) / 60 )) min)"
