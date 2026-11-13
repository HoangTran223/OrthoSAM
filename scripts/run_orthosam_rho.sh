#!/usr/bin/env bash
set -euo pipefail


CONFIG_PATH=${1:-configs/CIFAR10_WRN28_10.ini}
LOG_PREFIX=${2:-run_orthosam_rho_sweep}
EXTRA_ARGS=("${@:3}")

GPU_ID=${GPU_ID:-1}
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

if [[ -n "${RHO_VALUES:-}" ]]; then
    read -r -a RHO_LIST <<< "${RHO_VALUES}"
else
    RHO_LIST=(0.005 0.01 0.045 0.05 0.1 0.5)
fi

ORTHOSAM_LR=${ORTHOSAM_LR:-}
OPTIMIZER=${OPTIMIZER:-OrthoSAM}

declare -a PASSED_SWEEPS=()
declare -a FAILED_SWEEPS=()

pick_master_port() {
    python - <<'PY'
import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
}

sanitize_value() {
    echo "$1" | tr '.' 'p' | tr '-' 'm'
}

run_case() {
    local rho="$1"
    local tag="rho$(sanitize_value "${rho}")"
    local run_name="${LOG_PREFIX}_${tag}"
    local master_port="${MASTER_PORT:-}"

    if [[ -z "${master_port}" ]]; then
        master_port="$(pick_master_port)"
    fi

    echo "[RUN] ${OPTIMIZER} rho=${rho} (master_port=${master_port})"
    cmd=(
        python -m torch.distributed.run
        --master_port "${master_port}"
        main.py
        --ifile "${CONFIG_PATH}"
        --logSubDir "${run_name}"
        --optimizer "${OPTIMIZER}"
        --rho "${rho}"
        --samRho "${rho}"
        --orthosamRho "${rho}"
    )

    if [[ -n "${ORTHOSAM_LR}" ]]; then
        cmd+=(--learningRate "${ORTHOSAM_LR}")
    fi

    if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
        cmd+=("${EXTRA_ARGS[@]}")
    fi

    if "${cmd[@]}"; then
        PASSED_SWEEPS+=("${rho}")
        echo "[OK] rho=${rho}"
    else
        local exit_code=$?
        FAILED_SWEEPS+=("${rho}:${exit_code}")
        echo "[FAIL] rho=${rho} (exit=${exit_code})"
    fi
}

for rho in "${RHO_LIST[@]}"; do
    run_case "${rho}"
done

echo "[SUMMARY] passed=${#PASSED_SWEEPS[@]} failed=${#FAILED_SWEEPS[@]}"
if [[ ${#PASSED_SWEEPS[@]} -gt 0 ]]; then
    echo "[SUMMARY] passed rho: ${PASSED_SWEEPS[*]}"
fi
if [[ ${#FAILED_SWEEPS[@]} -gt 0 ]]; then
    echo "[SUMMARY] failed rho: ${FAILED_SWEEPS[*]}"
    exit 1
fi
