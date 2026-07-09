# Claim-HC Fork Workspace

这个目录用于维护 **仓库内可控版本** 的关键文件：

- `internvl.py`
- `modeling_internvl_chat.py`

当前推荐策略是：

- `internvl.py`：通过本地 import + `register_template(...)` 在运行时接管
- `modeling_internvl_chat.py`：保留 fork 版本作为参考与回滚基线
- `forward / generate`：通过运行时 monkey patch 接管，不再强依赖 deploy

## 启动方式

统一使用这两个入口，而不是直接 `swift sft` / `swift infer`：

- `scripts/run_swift_sft_with_claim_hc.py`
- `scripts/run_swift_infer_with_claim_hc.py`

它们会完成：

1. 读取 yaml 配置
2. 自动按 `NPROC_PER_NODE` 拉起多卡
3. import 仓库内 `forks/claim_hc/internvl.py`
4. 重新注册本地 `internvl` / `internvl2_5`
5. monkey patch InternVLChatModel 的 `forward/generate`

## 8 卡训练

```bash
export PYTHONPATH=/data2/573ops_ser/projects/videommd:$PYTHONPATH
python scripts/run_swift_sft_with_claim_hc.py configs/swift/fakett.yaml \
  --output_dir /data2/573ops_ser/projects/videommd/outputs/swift/fakett_hc \
  --modules_to_save claim_hc
```

## 8 卡推理

```bash
export PYTHONPATH=/data2/573ops_ser/projects/videommd:$PYTHONPATH
python scripts/run_swift_infer_with_claim_hc.py configs/swift/fakett_infer.yaml \
  --model /data2/573ops_ser/models/InternVL-8B \
  --adapters /data2/573ops_ser/projects/videommd/outputs/swift/fakett_hc/<run_dir>/checkpoint-xxx \
  --result_path /data2/573ops_ser/projects/videommd/outputs/swift/fakett_hc/fakett_test_predictions.jsonl
```

## 评估

```bash
python scripts/evaluate_swift_predictions.py \
  --dataset-jsonl /data2/573ops_ser/projects/videommd/data/swift/fakett/fakett_test.jsonl \
  --prediction-jsonl /data2/573ops_ser/projects/videommd/outputs/swift/fakett_hc/fakett_test_predictions.jsonl
```
