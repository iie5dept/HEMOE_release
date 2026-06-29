# VideoMMD

This project contains an engineering-oriented scaffold for multimodal short-video
fake-news detection. The current recommended path is:

- `InternVL3-8B`
- video-only input
- `real / fake` instruction SFT
- LoRA fine-tuning
- ExMRD temporal splits for `FakeTT` and `FakeSV`

## Layout

- `train.py`: training entrypoint
- `configs/`: dataset-specific YAML configs
- `libs/data/`: dataset loading and split building
- `libs/model/`: model wrappers
- `libs/utils/`: logging and shared helpers
- `scripts/build_swift_fakett.py`: export `data/fakett` into ms-swift JSONL
- `scripts/swift_train_fakett.sh`: 4-GPU ms-swift LoRA training for FakeTT
- `scripts/swift_infer_fakett.sh`: ms-swift inference on FakeTT test JSONL

## Recommended: Swift SFT

The simpler and more stable path is now `ms-swift`, not the custom Trainer.
It keeps the setup closer to the `FakeSV-VLM` training style, but removes the
paper-specific custom modules.

### 1. Build FakeTT JSONL

```bash
python scripts/build_swift_fakett.py \
  --annotation-path data/fakett/data.json \
  --video-root /data2/573ops_ser/data/FakeTT/FakeTT/video \
  --split-dir external/ExMRD/data/FakeTT/vids \
  --output-dir data/swift/fakett
```

This script uses the same `FakeTT` annotation source as the rest of the repo:

- `data/fakett/data.json`
- `external/ExMRD/data/FakeTT/vids/vid_time3_{train,valid,test}.txt`

### 2. Run 4-GPU Swift training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/swift_train_fakett.sh
```

### 3. Run Swift inference

```bash
ADAPTER_PATH=/path/to/checkpoint \
bash scripts/swift_infer_fakett.sh
```

## Native Trainer

The repo still keeps the native PyTorch training path for debugging and custom
experiments:

- `configs/fakesv.yaml`
- `configs/fakett.yaml`

Train FakeTT:

```bash
torchrun --nproc_per_node=4 train.py --config configs/fakett.yaml
```

## ExMRD Alignment

- `FakeTT` local annotations and `ExMRD` split ids are exactly aligned: `1992 / 1992`
- `FakeSV` local annotations and `ExMRD` split ids are exactly aligned: `5495 / 5495`
- `FakeSV` follows the same default binary policy as `ExMRD`: drop `辟谣`, keep `假 / 真`

## Why The Current Native Baseline Can Lag Behind FakeSV-VLM

- `FakeSV-VLM` uses a mature `ms-swift` training stack, while the native path here
  is a lighter custom reimplementation.
- Their paper baseline also uses a more tuned prompt, data packing, and training
  recipe than our first-pass native scaffold.
- If you compare against their full reported numbers, remember their reported
  method is not just “plain LoRA SFT”; it also includes their paper-specific
  adapter design. Our Swift path here intentionally removes those extras.

## Notes

- FakeTT Swift export currently targets the ExMRD temporal split.
- Metrics in the native path are `acc`, `macro_f1`, `macro_precision`, and `macro_recall`.
- Install Swift support with `pip install -r requirements.txt`.
