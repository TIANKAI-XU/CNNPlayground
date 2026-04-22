#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}"

cd "${SCRIPT_DIR}"
nohup python train.py > "${LOG_FILE}" 2>&1 &
PID=$!

echo "Started VGG-16 training in background."
echo "PID: ${PID}"
echo "Log: ${LOG_FILE}"
