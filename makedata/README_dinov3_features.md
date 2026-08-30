# DINOv3 key-frame feature cache

`extract_dinov3_keyframe_features.py` reads the existing FakeTT/FakeSV
`keyframes.jsonl` files and saves frozen DINOv3 features without accessing the
network.

Each sample is stored as an independent `safetensors` file containing:

- `cls_token`: final DINOv3 CLS token, shape `[D]`.
- `pooler_output`: model pooler output, shape `[D]`.
- `patch_mean`: mean of patch tokens, shape `[D]`.
- `global_feature`: concatenated pooler and patch mean, shape `[2D]`.
- `patch_tokens`: local patch sequence, shape `[N, D]`.

Register/storage tokens are detected and removed from `patch_tokens`.

## Download the official checkpoint

The official ViT-L/16 repository is gated. Accept the DINOv3 license for
`facebook/dinov3-vitl16-pretrain-lvd1689m` in a browser before logging in with
an authorized Hugging Face token.

```bash
unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE
export HF_ENDPOINT=https://hf-mirror.com

hf auth login
hf download facebook/dinov3-vitl16-pretrain-lvd1689m \
  --local-dir /data2/573ops_ser/models/dinov3-vitl16-pretrain-lvd1689m
```

The directory should contain at least `config.json`,
`preprocessor_config.json`, and `model.safetensors`.

Do not upgrade the ms-swift training environment solely for feature
extraction. If its Transformers version is older than 4.56, clone it to a
separate environment first, then upgrade Transformers in the clone.

## Four-GPU extraction

Run from the repository root on the server:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=4,5,6,7

torchrun --standalone --nproc_per_node=4 \
  makedata/extract_dinov3_keyframe_features.py \
  --dataset all \
  --model-path /data2/573ops_ser/models/dinov3-vitl16-pretrain-lvd1689m \
  --fakett-manifest /data2/573ops_ser/projects/videommd/data/key_frames/fakett/keyframes.jsonl \
  --fakesv-manifest /data2/573ops_ser/projects/videommd/data/key_frames/fakesv/keyframes.jsonl \
  --output-root /data2/573ops_ser/projects/videommd/data/dinov3_features \
  --batch-size 16 \
  --compute-dtype bfloat16 \
  --storage-dtype float16
```

The four processes are independent feature workers. They do not use DDP or
synchronize gradients. Rank 0 rebuilds the final manifests after all workers
finish.

If GPU memory is insufficient, reduce `--batch-size` to 8 or 4. Existing
feature and metadata pairs are skipped automatically, so the command can be
restarted safely. Use `--overwrite` only when all selected features should be
recomputed.

Native DINOv3 support requires `transformers>=4.56`. Check the server version
before extraction:

```bash
python -c "import transformers; print(transformers.__version__)"
```

## Cache validation

```bash
python makedata/check_dinov3_feature_cache.py \
  /data2/573ops_ser/projects/videommd/data/dinov3_features/fakett/dinov3_features.jsonl \
  --require-patch-tokens \
  --error-output /data2/573ops_ser/projects/videommd/data/dinov3_features/fakett/check_errors.jsonl

python makedata/check_dinov3_feature_cache.py \
  /data2/573ops_ser/projects/videommd/data/dinov3_features/fakesv/dinov3_features.jsonl \
  --require-patch-tokens \
  --error-output /data2/573ops_ser/projects/videommd/data/dinov3_features/fakesv/check_errors.jsonl
```

Expected output directories:

```text
data/dinov3_features/
  fakett/
    features/*.safetensors
    metadata/*.json
    errors/*.json
    dinov3_features.jsonl
  fakesv/
    features/*.safetensors
    metadata/*.json
    errors/*.json
    dinov3_features.jsonl
```
