#!/usr/bin/env bash
set -euo pipefail


CONFIG_PATH=${1:-configs/ImageNet_ViT.ini}
LOG_PREFIX=${2:-adamw_variants}
RUN_SUFFIX=${RUN_SUFFIX:-$(date +%Y%m%d_%H%M%S)_$RANDOM}

GPU_ID=${GPU_ID:-0}
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

run_variant() {
    local name="$1"
    shift
    local run_name="${LOG_PREFIX}_${name}"
    if [[ -n "${RUN_SUFFIX}" ]]; then
        run_name="${run_name}_${RUN_SUFFIX}"
    fi
    echo "[RUN] ${name}"
    python -m torch.distributed.run main.py \
        --ifile "${CONFIG_PATH}" \
        --logSubDir "${run_name}" \
        "$@"
}

run_variant adamw --optimizer AdamW
run_variant adamw_sam --optimizer AdamW_SAM
run_variant adamw_msam --optimizer AdamW_MSAM
run_variant adamw_friendlysam --optimizer AdamW_FriendlySAM
