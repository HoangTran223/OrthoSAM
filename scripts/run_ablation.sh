#!/usr/bin/env bash
set -euo pipefail


CONFIG_PATH=${1:-configs/CIFAR100_ViT_Ti.ini}
EXTRA_ARGS=("${@:2}")

GPU_ID=${GPU_ID:-0}
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
NPROC=${NPROC:-1}

if [[ -n "${METHODS:-}" ]]; then
    read -r -a METHOD_LIST <<< "${METHODS}"
else
    METHOD_LIST=(SAM OrthoPert_SGD SAM_OrthoDescent OrthoSAM Muon Muon_SAM)
fi

declare -A KNOWN_METHODS=(
    [SAM]=1
    [OrthoPert_SGD]=1
    [SAM_OrthoDescent]=1
    [OrthoSAM]=1
    [Muon]=1
    [Muon_SAM]=1
)

declare -A NO_RHO=(
    [Muon]=1
)

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

resolve_lr() {
    local method="$1"
    local per_method_var="LR_${method}"
    local per_method="${!per_method_var:-}"

    if [[ -n "${per_method}" ]]; then
        echo "${per_method}"
    else
        echo "${ABLATION_LR:-}"
    fi
}

resolve_rho() {
    local method="$1"
    local per_method_var="RHO_${method}"
    local per_method="${!per_method_var:-}"

    if [[ -n "${per_method}" ]]; then
        echo "${per_method}"
    else
        echo "${ABLATION_RHO:-}"
    fi
}

run_case() {
    local method="$1"
    local lr
    lr="$(resolve_lr "${method}")"
    local rho
    rho="$(resolve_rho "${method}")"
    local master_port="${MASTER_PORT:-}"

    if [[ -z "${master_port}" ]]; then
        master_port="$(pick_master_port)"
    fi

    local cmd=(
        python -m torch.distributed.run
        --nproc_per_node "${NPROC}"
        --master_port "${master_port}"
        main.py
        --ifile "${CONFIG_PATH}"
        --optimizer "${method}"
    )

    if [[ -z "${NO_RHO[${method}]:-}" && -n "${rho}" ]]; then
        cmd+=(--rho "${rho}" --samRho "${rho}" --orthosamRho "${rho}")
    fi
    if [[ -n "${lr}" ]]; then
        cmd+=(--learningRate "${lr}")
    fi
    if [[ -n "${EPOCHS:-}" ]]; then
        cmd+=(--epochs "${EPOCHS}")
    fi
    if [[ -n "${NESTEROV:-}" ]]; then
        cmd+=(--nesterov "${NESTEROV}")
    fi
    if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
        cmd+=("${EXTRA_ARGS[@]}")
    fi

    echo "[RUN] ${method} lr=${lr:-<ini>} rho=${rho:-<ini>} port=${master_port}"
    if "${cmd[@]}"; then
        PASSED+=("${method}")
        echo "[OK] ${method}"
    else
        local exit_code=$?
        FAILED+=("${method}:${exit_code}")
        echo "[FAIL] ${method} (exit=${exit_code})"
    fi
}

echo "[CONFIG] ${CONFIG_PATH}"
echo "[METHODS] ${METHOD_LIST[*]}"
for method in "${METHOD_LIST[@]}"; do
    if [[ -z "${KNOWN_METHODS[${method}]:-}" ]]; then
        echo "[SKIP] unknown method '${method}'"
        FAILED+=("${method}:unknown")
        continue
    fi
    run_case "${method}"
done

echo "[SUMMARY] passed=${#PASSED[@]} failed=${#FAILED[@]}"
if [[ ${#PASSED[@]} -gt 0 ]]; then
    echo "[SUMMARY] passed: ${PASSED[*]}"
fi
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "[SUMMARY] failed: ${FAILED[*]}"
    exit 1
fi
