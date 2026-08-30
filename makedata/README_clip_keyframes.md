# FakeTT/FakeSV CLIP key frames

`select_clip_keyframes.py` uniformly samples candidate frames from every video, embeds the uploader
text and candidate frames with a local CLIP-compatible model, and saves the frame with the highest
cosine similarity.

Text selection:

- FakeTT: `description`, with `event` as fallback.
- FakeSV: `title`, with `keywords` and then `ocr` as fallbacks.

Labels, comments, and uploader profiles are never included in the retrieval query.

## Model

Use `google/siglip2-base-patch16-224`. It supports multilingual image-text retrieval and uses the
standard Transformers implementation, so it does not need `trust_remote_code` or auxiliary code
repositories.

Download it through a Hugging Face mirror:

```bash
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download google/siglip2-base-patch16-224 \
  --local-dir /data2/573ops_ser/models/siglip2-base-patch16-224
```

With a newer `huggingface_hub`, the equivalent command is:

```bash
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export HF_ENDPOINT=https://hf-mirror.com
hf download google/siglip2-base-patch16-224 \
  --local-dir /data2/573ops_ser/models/siglip2-base-patch16-224
```

The resulting directory is:

```text
/data2/573ops_ser/models/siglip2-base-patch16-224
```

The server environment needs `torch`, `transformers`, `opencv-python`, `Pillow`, and `tqdm`.

The script loads the tokenizer and image processor separately instead of using `AutoProcessor`.
This is intentional: some Transformers releases select the legacy `SiglipTokenizer` for SigLIP2
and fail with `vocab_file=None`, even when `tokenizer.model` is present. The local model directory
must contain `tokenizer.json`, `tokenizer.model`, `tokenizer_config.json`, and
`preprocessor_config.json`.
Loading is forced to `local_files_only=True`; key-frame extraction will not access the network.

## Smoke test

Run this first to validate the environment and output layout:

```bash
CUDA_VISIBLE_DEVICES=0 python makedata/select_clip_keyframes.py \
  --dataset all \
  --model-path /data2/573ops_ser/models/siglip2-base-patch16-224 \
  --output-root /data2/573ops_ser/projects/videommd/data/key_frames \
  --max-samples 2
```

## Full run

```bash
CUDA_VISIBLE_DEVICES=0 python makedata/select_clip_keyframes.py \
  --dataset all \
  --model-path /data2/573ops_ser/models/siglip2-base-patch16-224 \
  --fakett-annotation /data2/573ops_ser/projects/videommd/data/fakett/data.json \
  --fakett-video-root /data2/573ops_ser/data/FakeTT/FakeTT/video \
  --fakesv-annotation /data2/573ops_ser/projects/videommd/data/fakesv/data_complete.json \
  --fakesv-video-root /data2/573ops_ser/data/FakeSV/video/videos \
  --output-root /data2/573ops_ser/projects/videommd/data/key_frames \
  --candidate-frames 32 \
  --image-batch-size 16
```

The command is resumable. Existing image/metadata pairs are skipped unless `--overwrite` is used.
Failures are recorded per video and do not stop the full run unless `--fail-fast` is supplied.

## Outputs

For each dataset, the script creates:

```text
data/key_frames/fakett/key_frames/<video_id>.jpg
data/key_frames/fakett/metadata/<video_id>.json
data/key_frames/fakett/keyframes.jsonl
data/key_frames/fakett/errors/<video_id>.json
```

FakeSV uses the same layout under `data/key_frames/fakesv`. Metadata contains the selected frame
index, timestamp, cosine similarity, retrieval text, source video, and key-frame path.
