#!/usr/bin/env bash
set -euo pipefail

PROJECT=/data2/573ops_ser/projects/videommd1
PY=/data2/573ops_ser/envs/moedet/bin/python
DINO_MODEL=/data2/573ops_ser/models/dinov3-vitl16-pretrain-lvd689m/data2/573ops_ser/models/dinov3-vit7b16-pretrain-lvd1689m
DINO_OUTPUT="$PROJECT/data/dinov3_7b_keyframe_features"
ABLATION_OUTPUT="$PROJECT/outputs/trimodal_moe/ablations_internvl_dinov3_7b_keyframe_split_te_ae"

cd "$PROJECT"

if [[ ! -x "$PY" ]]; then
  echo "Python interpreter not found: $PY" >&2
  exit 1
fi
if [[ ! -f "$DINO_MODEL/config.json" ]]; then
  echo "DINOv3-7B checkpoint not found at the exact configured path: $DINO_MODEL" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "[1/5] Extracting DINOv3-7B features from the saved highest-similarity key frames"
CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -m torch.distributed.run \
  --standalone --nproc_per_node=4 \
  makedata/extract_dinov3_keyframe_features.py \
  --dataset all \
  --model-path "$DINO_MODEL" \
  --fakett-manifest /data2/573ops_ser/projects/videommd/data/key_frames/fakett/keyframes.jsonl \
  --fakesv-manifest /data2/573ops_ser/projects/videommd/data/key_frames/fakesv/keyframes.jsonl \
  --output-root "$DINO_OUTPUT" \
  --batch-size 1 \
  --compute-dtype bfloat16 \
  --storage-dtype float16 \
  --fail-fast

echo "[2/5] Validating both DINOv3-7B feature caches"
"$PY" makedata/check_dinov3_feature_cache.py \
  "$DINO_OUTPUT/fakett/dinov3_features.jsonl" --require-patch-tokens
"$PY" makedata/check_dinov3_feature_cache.py \
  "$DINO_OUTPUT/fakesv/dinov3_features.jsonl" --require-patch-tokens

echo "[3/5] Running all FakeTT ablations on GPUs 0,1,2,3"
"$PY" scripts/run_trimodal_ablations.py \
  configs/trimodal_moe/fakett.yaml \
  --dataset fakett \
  --gpus 0,1,2,3 \
  --master-port 0 \
  --output-root "$ABLATION_OUTPUT"

echo "[4/5] Running all FakeSV ablations on GPUs 0,1,2,3"
"$PY" scripts/run_trimodal_ablations.py \
  configs/trimodal_moe/fakesv.yaml \
  --dataset fakesv \
  --gpus 0,1,2,3 \
  --master-port 0 \
  --output-root "$ABLATION_OUTPUT"

echo "[5/5] Combining every epoch and best-per-variant result"
"$PY" scripts/summarize_trimodal_ablations.py "$ABLATION_OUTPUT"

echo "Completed. Results: $ABLATION_OUTPUT"
