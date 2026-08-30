# Four-modal MoE classifier

This experiment replaces vocabulary generation with direct binary classification while retaining
LoRA on the InternVL language decoder. It combines:

- the last prompt token from the InternVL language backbone;
- cached DINOv3 patch tokens processed by a two-layer Transformer;
- cached BERT token states processed by a two-layer Transformer;
- a cached Qwen2-Audio last-prompt-token state;
- one always-on shared MLP expert and four modality-specific MLP experts;
- a dense reliability router and a final binary linear classifier.

The assistant answer is used only as the class label. It is never appended to the InternVL input.
The objective is:

```text
L = L_fusion
  + modality_loss_weight * weighted_mean(L_llm, L_visual, L_text, L_audio)
  + router_loss_weight * L_router
  + balance_loss_weight * L_balance
```

`L_router` teaches the router to favor the branch with the lower detached per-sample classification
loss. This supervision trains only the router and cannot directly distort a modality encoder.
FakeTT currently sets the visual and audio branch-loss weights to zero, so only `L_llm` and
`L_text` contribute to the independent-modality term. Visual and audio representations still
participate in the MoE fusion and receive gradients from `L_fusion`.

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
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 \
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
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 \
  scripts/train_trimodal_moe.py configs/trimodal_moe/fakett.yaml
```

FakeSV on GPUs 4-7:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 \
  scripts/train_trimodal_moe.py configs/trimodal_moe/fakesv.yaml
```

Resume by adding `--resume /path/to/checkpoint-epoch-N`.

Both experiment configs evaluate the test split after every epoch and select the highest
`macro_f1`. The selected model is updated at the experiment-level path
`checkpoint-best`, while the per-epoch checkpoints are also retained. This is test-set model
selection rather than a strictly held-out final evaluation, so report it as such.

## Infer and evaluate

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 \
  scripts/infer_trimodal_moe.py configs/trimodal_moe/fakett.yaml \
  --checkpoint /data2/573ops_ser/projects/videommd/outputs/trimodal_moe/fakett_llm_text_aux/checkpoint-best \
  --output /data2/573ops_ser/projects/videommd/outputs/trimodal_moe/fakett_llm_text_aux/test_predictions.jsonl

python scripts/evaluate_trimodal_predictions.py \
  /data2/573ops_ser/projects/videommd/outputs/trimodal_moe/fakett_llm_text_aux/test_predictions.jsonl
```

The prediction file also records each branch probability and the four router weights for analysis.
