#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES
ADAPTER_PATH="${ADAPTER_PATH:-/path/to/swift_checkpoint}"
JSONL_DIR="${JSONL_DIR:-${REPO_ROOT}/data/swift/fakett}"
SYSTEM_FILE="${SYSTEM_FILE:-${REPO_ROOT}/configs/swift/fakett_system.txt}"
RESULT_PATH="${RESULT_PATH:-${REPO_ROOT}/outputs/swift_infer_fakett_test.jsonl}"
DEVICE_MAP_FILE="${DEVICE_MAP_FILE:-}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-4}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
DATA_SEED="${DATA_SEED:-2025}"
SEED="${SEED:-2025}"
TEMPLATE_TYPE="${TEMPLATE_TYPE:-internvl2_5}"

if ! command -v swift >/dev/null 2>&1; then
  echo "Error: \`swift\` command not found." >&2
  echo "You likely installed the wrong package: OpenStack Swift instead of ModelScope ms-swift." >&2
  echo "Fix with:" >&2
  echo "  pip uninstall -y swift" >&2
  echo "  pip install -U ms-swift==3.2.0" >&2
  exit 1
fi

SWIFT_ARGS=(
  infer
  --adapter "${ADAPTER_PATH}"
  --dataset "${JSONL_DIR}/fakett_test.jsonl"
  --system "$(cat "${SYSTEM_FILE}")"
  --template "${TEMPLATE_TYPE}"
  --max_batch_size "${MAX_BATCH_SIZE}"
  --torch_dtype bfloat16
  --infer_backend pt
  --max_length "${MAX_LENGTH}"
  --attn_impl flash_attn
  --split_dataset_ratio 1.0
  --max_new_tokens 1
  --metric acc
  --data_seed "${DATA_SEED}"
  --seed "${SEED}"
  --temperature 0
  --result_path "${RESULT_PATH}"
)

if [[ -n "${DEVICE_MAP_FILE}" && -f "${DEVICE_MAP_FILE}" ]]; then
  SWIFT_ARGS+=(--device_map "${DEVICE_MAP_FILE}")
fi

swift "${SWIFT_ARGS[@]}"
