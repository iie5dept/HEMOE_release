# BERT text feature cache

The extractor reads the existing Swift `train/val/test.jsonl` files. It uses
only `system` and `user` messages, removes the `<video>` media placeholder, and
never reads the assistant `real`/`fake` answer.

FakeTT uses `google-bert/bert-base-uncased`. FakeSV uses
`hfl/chinese-macbert-base`.

## Download

```bash
bash makedata/download_bert_models.sh /data2/573ops_ser/models
```

## Four-GPU extraction

```bash
export CUDA_VISIBLE_DEVICES=4,5,6,7
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

torchrun --standalone --nproc_per_node=4 \
  makedata/extract_bert_text_features.py \
  --dataset all \
  --fakett-data-dir /data2/573ops_ser/projects/videommd/data/swift/fakett \
  --fakesv-data-dir /data2/573ops_ser/projects/videommd/data/swift/fakesv \
  --fakett-model-path /data2/573ops_ser/models/bert-base-uncased \
  --fakesv-model-path /data2/573ops_ser/models/chinese-macbert-base \
  --output-root /data2/573ops_ser/projects/videommd/data/bert_features \
  --batch-size 32 \
  --max-length 512 \
  --truncation-side left \
  --compute-dtype bfloat16 \
  --storage-dtype float16
```

Each sample stores `token_states`, `attention_mask`, `special_tokens_mask`,
`cls_token`, `content_mean`, and `global_feature` in an independent
`safetensors` file. Existing complete feature/metadata pairs are skipped.

## Validation

```bash
python makedata/check_bert_text_feature_cache.py \
  /data2/573ops_ser/projects/videommd/data/bert_features/fakett/bert_features.jsonl

python makedata/check_bert_text_feature_cache.py \
  /data2/573ops_ser/projects/videommd/data/bert_features/fakesv/bert_features.jsonl
```
