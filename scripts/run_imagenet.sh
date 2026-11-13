#!/usr/bin/env bash
set -euo pipefail


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

declare -a SCRIPTS=(
    run_imagenet_sgd.sh
    run_imagenet_sam.sh
    run_imagenet_orthosam.sh
    run_imagenet_muon.sh
)

declare -a PASSED=()
declare -a FAILED=()

for script in "${SCRIPTS[@]}"; do
    echo "[SEQ] ${script}"
    if bash "${SCRIPT_DIR}/${script}" "$@"; then
        PASSED+=("${script}")
    else
        FAILED+=("${script}:$?")
    fi
done

echo "[SUMMARY] passed=${#PASSED[@]} failed=${#FAILED[@]}"
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "[SUMMARY] failed: ${FAILED[*]}"
    exit 1
fi
