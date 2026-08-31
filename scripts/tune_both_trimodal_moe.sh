#!/usr/bin/env bash
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

num_trials="${1:-12}"
python_bin="${PYTHON_BIN:-python}"
output_base="${TUNING_OUTPUT_ROOT:-/data2/573ops_ser/projects/videommd/outputs/trimodal_moe/tuning}"
fakesv_output="${output_base}/fakesv_qwen3"
fakett_output="${output_base}/fakett_qwen3"

mkdir -p "${fakesv_output}" "${fakett_output}"

echo "Starting FakeSV tuning on GPUs 0,1,2,3 (${num_trials} trials)"
"${python_bin}" scripts/tune_trimodal_moe.py configs/trimodal_moe/fakesv.yaml \
  --gpus 0,1,2,3 \
  --num-trials "${num_trials}" \
  --output-root "${fakesv_output}" \
  >"${fakesv_output}/tuner.log" 2>&1 &
fakesv_pid=$!

echo "Starting FakeTT tuning on GPUs 4,5,6,7 (${num_trials} trials)"
"${python_bin}" scripts/tune_trimodal_moe.py configs/trimodal_moe/fakett.yaml \
  --gpus 4,5,6,7 \
  --num-trials "${num_trials}" \
  --output-root "${fakett_output}" \
  >"${fakett_output}/tuner.log" 2>&1 &
fakett_pid=$!

terminate_children() {
  kill "${fakesv_pid}" "${fakett_pid}" 2>/dev/null || true
}
trap terminate_children INT TERM

wait "${fakesv_pid}"
fakesv_status=$?
wait "${fakett_pid}"
fakett_status=$?

echo "FakeSV tuner exit code: ${fakesv_status}; log: ${fakesv_output}/tuner.log"
echo "FakeTT tuner exit code: ${fakett_status}; log: ${fakett_output}/tuner.log"

if [[ "${fakesv_status}" -ne 0 || "${fakett_status}" -ne 0 ]]; then
  exit 1
fi
