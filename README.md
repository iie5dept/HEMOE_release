# VideoMMD

This repo currently uses `ms-swift` + `InternVL-8B` for video fake-news detection.
The active workflow is:

- build Swift `jsonl` data
- train with native `swift sft`
- infer with native `swift infer`
- evaluate `acc / macro_f1 / macro_precision / macro_recall` with an external script

Current active configs:

- `configs/swift/fakett.yaml`
- `configs/swift/fakett_infer.yaml`
- `configs/swift/fakesv.yaml`
- `configs/swift/fakesv_infer.yaml`

Current active scripts:

- `scripts/build_swift_dataset.py`
- `scripts/resolve_swift_adapter.py`
- `scripts/evaluate_swift_predictions.py`

## FakeTT

### Build JSONL

```bash
python scripts/build_swift_dataset.py \
  --dataset fakett \
  --video-root /data2/573ops_ser/data/FakeTT/FakeTT/video
```

Outputs:

- `data/swift/fakett/fakett_train.jsonl`
- `data/swift/fakett/fakett_val.jsonl`
- `data/swift/fakett/fakett_test.jsonl`

### Train

```bash
swift sft configs/swift/fakett.yaml
```

Current training strategy:

- `8` GPUs
- `per_device_train_batch_size: 2`
- `gradient_accumulation_steps: 2`

### Infer

If you already know the checkpoint path:

```bash
swift infer configs/swift/fakett_infer.yaml \
  --adapter /data2/573ops_ser/projects/videommd/outputs/swift/fakett/your_run/checkpoint-xx
```

If you need to resolve the adapter path first:

```bash
python scripts/resolve_swift_adapter.py \
  --output-dir /data2/573ops_ser/projects/videommd/outputs/swift/fakett/your_run
```

Then infer:

```bash
ADAPTER_PATH=$(python scripts/resolve_swift_adapter.py \
  --output-dir /data2/573ops_ser/projects/videommd/outputs/swift/fakett/your_run)

swift infer configs/swift/fakett_infer.yaml \
  --adapter "$ADAPTER_PATH" \
  --result_path /data2/573ops_ser/projects/videommd/outputs/swift/fakett/fakett_test_predictions.jsonl
```

### Evaluate

```bash
python scripts/evaluate_swift_predictions.py \
  --dataset-jsonl data/swift/fakett/fakett_test.jsonl \
  --prediction-jsonl /data2/573ops_ser/projects/videommd/outputs/swift/fakett/fakett_test_predictions.jsonl
```

Save metrics to file:

```bash
python scripts/evaluate_swift_predictions.py \
  --dataset-jsonl data/swift/fakett/fakett_test.jsonl \
  --prediction-jsonl /data2/573ops_ser/projects/videommd/outputs/swift/fakett/fakett_test_predictions.jsonl \
  --output-json /data2/573ops_ser/projects/videommd/outputs/swift/fakett/fakett_test_metrics.json
```

## FakeSV

### Build JSONL

```bash
python scripts/build_swift_dataset.py \
  --dataset fakesv \
  --video-root /data2/573ops_ser/data/FakeSV/video
```

Default binary export policy:

- `Õæ -> real`
- `¼Ù -> fake`
- `±ÙÒ¥ -> drop`

Outputs:

- `data/swift/fakesv/fakesv_train.jsonl`
- `data/swift/fakesv/fakesv_val.jsonl`
- `data/swift/fakesv/fakesv_test.jsonl`

### Train

```bash
swift sft configs/swift/fakesv.yaml
```

### Infer

```bash
ADAPTER_PATH=$(python scripts/resolve_swift_adapter.py \
  --output-dir /data2/573ops_ser/projects/videommd/outputs/swift/fakesv/your_run)

swift infer configs/swift/fakesv_infer.yaml \
  --adapter "$ADAPTER_PATH" \
  --result_path /data2/573ops_ser/projects/videommd/outputs/swift/fakesv/fakesv_test_predictions.jsonl
```

### Evaluate

```bash
python scripts/evaluate_swift_predictions.py \
  --dataset-jsonl data/swift/fakesv/fakesv_test.jsonl \
  --prediction-jsonl /data2/573ops_ser/projects/videommd/outputs/swift/fakesv/fakesv_test_predictions.jsonl
```

Save metrics to file:

```bash
python scripts/evaluate_swift_predictions.py \
  --dataset-jsonl data/swift/fakesv/fakesv_test.jsonl \
  --prediction-jsonl /data2/573ops_ser/projects/videommd/outputs/swift/fakesv/fakesv_test_predictions.jsonl \
  --output-json /data2/573ops_ser/projects/videommd/outputs/swift/fakesv/fakesv_test_metrics.json
```

## Notes

- Training and inference both use native Swift commands.
- Evaluation is intentionally separated from Swift and computed afterwards.
- `--adapter` means the LoRA checkpoint directory, not the base model directory.
- Do not use the old `mapmoe.json` device map in the current Swift workflow.
