#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_PATH="${MODEL_PATH:-/data2/573ops_ser/models/InternVL-8B}"
VIDEO_ROOT="${VIDEO_ROOT:-/data2/573ops_ser/data/FakeTT/FakeTT/video}"
JSONL_DIR="${JSONL_DIR:-${REPO_ROOT}/data/swift/fakett}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/swift_internvl3_fakett}"
SYSTEM_FILE="${SYSTEM_FILE:-${REPO_ROOT}/configs/swift/fakett_system.txt}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES
DEVICE_MAP_FILE="${DEVICE_MAP_FILE:-}"
AUTO_TEST_AFTER_TRAIN="${AUTO_TEST_AFTER_TRAIN:-true}"
TEST_RESULT_PATH="${TEST_RESULT_PATH:-${OUTPUT_DIR}/fakett_test_predictions.jsonl}"
TEMPLATE_TYPE="${TEMPLATE_TYPE:-internvl2_5}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-5}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-8e-5}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-32}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
DATA_SEED="${DATA_SEED:-2025}"
SEED="${SEED:-2025}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"

if ! command -v swift >/dev/null 2>&1; then
  echo "Error: \`swift\` command not found." >&2
  echo "You likely installed the wrong package: OpenStack Swift instead of ModelScope ms-swift." >&2
  echo "Fix with:" >&2
  echo "  pip uninstall -y swift" >&2
  echo "  pip install -U ms-swift==3.2.0" >&2
  exit 1
fi

python "${REPO_ROOT}/scripts/build_swift_fakett.py" \
  --annotation-path "${REPO_ROOT}/data/fakett/data.json" \
  --video-root "${VIDEO_ROOT}" \
  --split-dir "${REPO_ROOT}/external/ExMRD/data/FakeTT/vids" \
  --output-dir "${JSONL_DIR}"

SWIFT_ARGS=(
  sft
  --model "${MODEL_PATH}"
  --dataset "${JSONL_DIR}/fakett_train.jsonl"
  --val_dataset "${JSONL_DIR}/fakett_val.jsonl"
  --acc_strategy seq
  --system "$(cat "${SYSTEM_FILE}")"
  --template "${TEMPLATE_TYPE}"
  --torch_dtype bfloat16
  --attn_impl flash_attn
  --num_train_epochs "${NUM_TRAIN_EPOCHS}"
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  --split_dataset_ratio 0.0
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}"
  --learning_rate "${LEARNING_RATE}"
  --lora_rank "${LORA_RANK}"
  --lora_alpha "${LORA_ALPHA}"
  --target_modules all-linear
  --freeze_vit true
  --freeze_aligner true
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --ddp_find_unused_parameters false
  --save_strategy epoch
  --eval_strategy epoch
  --eval_steps 1
  --save_steps 1
  --save_total_limit "${SAVE_TOTAL_LIMIT}"
  --logging_steps "${LOGGING_STEPS}"
  --max_length "${MAX_LENGTH}"
  --output_dir "${OUTPUT_DIR}"
  --warmup_ratio "${WARMUP_RATIO}"
  --data_seed "${DATA_SEED}"
  --seed "${SEED}"
  --temperature 0
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
)

if [[ -n "${DEVICE_MAP_FILE}" && -f "${DEVICE_MAP_FILE}" ]]; then
  SWIFT_ARGS+=(--device_map "${DEVICE_MAP_FILE}")
fi

# We intentionally do not pass FakeSV-VLM's custom modules_to_save because
# this simplified recipe removes their paper-specific switch-transformer,
# attn_score, and classifier innovations.
NPROC_PER_NODE="${NPROC_PER_NODE}" swift "${SWIFT_ARGS[@]}"

if [[ "${AUTO_TEST_AFTER_TRAIN}" != "true" ]]; then
  exit 0
fi

resolve_best_adapter_path() {
  local output_dir="$1"

  if [[ -f "${output_dir}/trainer_state.json" ]]; then
    local best_path
    best_path="$(
      python - "${output_dir}/trainer_state.json" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as handle:
    state = json.load(handle)
best = state.get("best_model_checkpoint") or ""
print(best)
PY
    )"
    if [[ -n "${best_path}" && -d "${best_path}" ]]; then
      echo "${best_path}"
      return 0
    fi
  fi

  local latest_checkpoint=""
  latest_checkpoint="$(find "${output_dir}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)"
  if [[ -n "${latest_checkpoint}" && -d "${latest_checkpoint}" ]]; then
    echo "${latest_checkpoint}"
    return 0
  fi

  echo "${output_dir}"
}

ADAPTER_PATH="$(resolve_best_adapter_path "${OUTPUT_DIR}")"
echo "Auto test with adapter: ${ADAPTER_PATH}"
ADAPTER_PATH="${ADAPTER_PATH}" RESULT_PATH="${TEST_RESULT_PATH}" bash "${REPO_ROOT}/scripts/swift_infer_fakett.sh"
