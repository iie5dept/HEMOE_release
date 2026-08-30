#!/usr/bin/env bash
set -euo pipefail

model_root="${1:-/data2/573ops_ser/models}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE

hf download google-bert/bert-base-uncased \
  config.json \
  model.safetensors \
  tokenizer.json \
  tokenizer_config.json \
  vocab.txt \
  --local-dir "${model_root}/bert-base-uncased"

hf download hfl/chinese-macbert-base \
  added_tokens.json \
  config.json \
  pytorch_model.bin \
  special_tokens_map.json \
  tokenizer.json \
  tokenizer_config.json \
  vocab.txt \
  --local-dir "${model_root}/chinese-macbert-base"

printf 'English BERT: %s\n' "${model_root}/bert-base-uncased"
printf 'Chinese MacBERT: %s\n' "${model_root}/chinese-macbert-base"
