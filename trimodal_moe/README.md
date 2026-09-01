# Four-modal MoE classifier

This experiment replaces vocabulary generation with direct binary classification while retaining
LoRA on the InternVL language decoder. It combines:

- the last prompt token from the InternVL language backbone;
- cached DINOv3 patch tokens processed by a two-layer Transformer without a visual classifier;
- cached Qwen3-8B last-prompt-token states projected by a trainable linear branch;
- a cached Qwen2-Audio last-prompt-token state projected without an audio classifier;
- four modality-specific MLP experts without a shared expert;
- a feature-only dense router and a weighted-concatenation binary classifier.

The assistant answer is used only as the class label. It is never appended to the InternVL input.
The objective is:

```text
L = L_fusion
  + modality_loss_weight * llm_loss_weight * L_llm
```

The router consumes only the concatenated detached modality representations. Each specialist output
is multiplied by its router weight; the four weighted outputs are concatenated rather than summed
before the final classifier. The router uses a direct softmax over its logits and is trained only
through `L_fusion`; there is no temperature smoothing, warmup interpolation, oracle supervision, or
load-balancing loss. Only the InternVL branch has an independent auxiliary classifier. The visual,
Qwen3-text, and audio representations participate only in MoE fusion and receive gradients from
`L_fusion`.

## Cache Qwen3 text features

The local checkpoint is expected at `/data2/573ops_ser/models/Qwen3-8B`. Extract both datasets on
GPUs 4-7:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 /data2/573ops_ser/envs/moedet/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=4 \
  makedata/extract_qwen3_text_features.py \
  --dataset all \
  --model-path /data2/573ops_ser/models/Qwen3-8B \
  --output-root /data2/573ops_ser/projects/videommd/data/qwen3_text_features \
  --attn-implementation sdpa \
  --compute-dtype bfloat16 \
  --storage-dtype float16 \
  --batch-size 1 \
  --max-length 8192
```

Existing feature and metadata pairs are skipped automatically. Use `--overwrite` only when the
model, prompt template, or maximum length changes. Validate the completed caches with:

```bash
python makedata/check_qwen3_text_feature_cache.py \
  data/qwen3_text_features/fakett/qwen3_text_features.jsonl

python makedata/check_qwen3_text_feature_cache.py \
  data/qwen3_text_features/fakesv/qwen3_text_features.jsonl
```

## Cache Qwen2-Audio features

Download the official checkpoint once on a networked server:

```bash
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export HF_ENDPOINT=https://hf-mirror.com
hf download Qwen/Qwen2-Audio-7B-Instruct \
  --local-dir /data2/573ops_ser/models/Qwen2-Audio-7B-Instruct
```

Extract both datasets on GPUs 4-7:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 /data2/573ops_ser/envs/moedet/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=4 \
  makedata/extract_qwen2_audio_features.py \
  --dataset all \
  --model-path /data2/573ops_ser/models/Qwen2-Audio-7B-Instruct \
  --attn-implementation sdpa
```

Validate the manifests:

```bash
python makedata/check_qwen2_audio_feature_cache.py \
  data/qwen2_audio_features/fakett/qwen2_audio_features.jsonl --expected-hidden-size 4096

python makedata/check_qwen2_audio_feature_cache.py \
  data/qwen2_audio_features/fakesv/qwen2_audio_features.jsonl --expected-hidden-size 4096
```

## Train

FakeTT on GPUs 4-7:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 /data2/573ops_ser/envs/moedet/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=4 \
  scripts/train_trimodal_moe.py configs/trimodal_moe/fakett.yaml
```

FakeSV on GPUs 0-3:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 /data2/573ops_ser/envs/moedet/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=4 \
  scripts/train_trimodal_moe.py configs/trimodal_moe/fakesv.yaml
```

Resume by adding `--resume /path/to/checkpoint-epoch-N`.

Both direct-training configs evaluate the test split after every epoch and select the highest
`macro_f1`. The selected model is updated at the experiment-level path `checkpoint-best`, while the
per-epoch checkpoints are also retained. This intentionally uses the test set for checkpoint
selection to match the current experimental protocol; a separate inference run is optional.

## Infer and evaluate

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 /data2/573ops_ser/envs/moedet/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=4 \
  scripts/infer_trimodal_moe.py configs/trimodal_moe/fakett.yaml \
  --checkpoint /data2/573ops_ser/projects/videommd/outputs/trimodal_moe/fakett_unconstrained_router_no_text_aux/checkpoint-best \
  --output /data2/573ops_ser/projects/videommd/outputs/trimodal_moe/fakett_unconstrained_router_no_text_aux/test_predictions.jsonl

python scripts/evaluate_trimodal_predictions.py \
  /data2/573ops_ser/projects/videommd/outputs/trimodal_moe/fakett_unconstrained_router_no_text_aux/test_predictions.jsonl
```

The prediction file records fusion and InternVL probabilities plus the four router weights. There
are no independent visual, Qwen3-text, or audio classification probabilities.

## Hyperparameter search

Run a reproducible sequential search on GPUs 4-7. Trial 0 is always the unmodified base config;
later trials sample unique parameter combinations. Tuning selects on the validation split, then runs
the selected checkpoint once on the test split.

```bash
python scripts/tune_trimodal_moe.py configs/trimodal_moe/fakett.yaml \
  --gpus 4,5,6,7 \
  --num-trials 12 \
  --output-root /data2/573ops_ser/projects/videommd/outputs/trimodal_moe/tuning/fakett_qwen3_no_va_heads
```

The output root contains `leaderboard.json`, `leaderboard.csv`, `best_config.yaml`,
`best_checkpoint.txt`, `best_result.json`, `best_test_predictions.jsonl`, and
`best_test_metrics.json`. Completed trials are reused when the same command is restarted. Per-epoch
checkpoints are removed after each successful trial unless `--keep-epoch-checkpoints` is specified;
every trial keeps its own `checkpoint-best`.

To tune both datasets concurrently, assigning FakeSV to GPUs 0-3 and FakeTT to GPUs 4-7:

```bash
bash scripts/tune_both_trimodal_moe.sh 12
```

The two driver logs are written to
`outputs/trimodal_moe/tuning/fakesv_qwen3_no_va_heads/tuner.log` and
`outputs/trimodal_moe/tuning/fakett_qwen3_no_va_heads/tuner.log`. Running the command again resumes
by reusing completed trials.
