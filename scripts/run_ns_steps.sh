#!/usr/bin/env bash
set -euo pipefail


CONFIG_PATH=${1:-configs/CIFAR100_WRN16_4.ini}
EXTRA_ARGS=("${@:2}")

GPU_ID=${GPU_ID:-0}
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
NPROC=${NPROC:-1}
LEARNING_RATE=${LEARNING_RATE:-0.5}
SAM_RHO=${SAM_RHO:-0.1}
LOG_DIR=${LOG_DIR:-logs_k}

if [[ -n "${NS_STEPS:-}" ]]; then
    read -r -a K_LIST <<< "${NS_STEPS}"
else
    K_LIST=(1 2 3 5)
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "[ERROR] config not found: ${CONFIG_PATH}" >&2
    echo "        run from the repository root (directory containing main.py)" >&2
    exit 1
fi

declare -a PASSED=()
declare -a FAILED=()

pick_master_port() {
    python - <<'PY'
import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
}

run_case() {
    local k="$1"
    local master_port="${MASTER_PORT:-$(pick_master_port)}"

    local cmd=(
        python -m torch.distributed.run
        --nproc_per_node "${NPROC}"
        --master_port "${master_port}"
        main.py
        --ifile "${CONFIG_PATH}"
        --optimizer OrthoSAM
        --nsSteps "${k}"
        --learningRate "${LEARNING_RATE}"
        --rho "${SAM_RHO}"
        --samRho "${SAM_RHO}"
        --orthosamRho "${SAM_RHO}"
        --logDir "${LOG_DIR}"
    )

    if [[ -n "${EPOCHS:-}" ]]; then
        cmd+=(--epochs "${EPOCHS}")
    fi
    if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
        cmd+=("${EXTRA_ARGS[@]}")
    fi

    echo "[RUN] OrthoSAM nsSteps=${k} lr=${LEARNING_RATE} rho=${SAM_RHO} port=${master_port}"
    local start_time=${SECONDS}
    if "${cmd[@]}"; then
        PASSED+=("K=${k}")
        echo "[OK] K=${k} (elapsed $(( (SECONDS - start_time) / 60 )) min)"
    else
        local exit_code=$?
        FAILED+=("K=${k}:${exit_code}")
        echo "[FAIL] K=${k} (exit=${exit_code})"
    fi
}

mkdir -p "${LOG_DIR}"

echo "[CONFIG] ${CONFIG_PATH}"
echo "[LOG_DIR] ${LOG_DIR}"
echo "[NS_STEPS] ${K_LIST[*]}"
echo "[LR] ${LEARNING_RATE}  [RHO] ${SAM_RHO}"
for k in "${K_LIST[@]}"; do
    run_case "${k}"
done

echo "[SUMMARY] passed=${#PASSED[@]} failed=${#FAILED[@]}"
if [[ ${#PASSED[@]} -gt 0 ]]; then
    echo "[SUMMARY] passed: ${PASSED[*]}"
fi
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "[SUMMARY] failed: ${FAILED[*]}"
    exit 1
fi
