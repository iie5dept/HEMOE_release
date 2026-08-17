# Swift Claim-HC 训练、推理与评估

训练和推理的公共参数由 YAML 管理。正常运行时不需要手工设置 `STAGE`、`OUT` 或追加 `--output_dir`；推理 checkpoint 通过 `--adapters` 明确传入。

```bash
cd /data2/573ops_ser/projects/videommd
```

## 配置约定

`ENV` 保存 Claim-HC 阶段和分布式环境变量。phase2 到 phase4 训练配置中的 `VIDEOMMD.adapter_output_dir` 会自动承接上一阶段；推理时直接用 `--adapters` 指定准确 checkpoint。

- phase1 从基础模型开始训练。
- phase2 自动加载 phase1 adapter。
- phase3 自动加载 phase2 adapter。
- phase4 自动加载 phase3 adapter。
- 推理通过命令行加载明确指定的 adapter checkpoint。

phase2 到 phase4 续训自动解析上一阶段 checkpoint 时应看到：

```text
[videommd] resolved adapter from config: .../checkpoint-xxx
```

如果 Swift 没有自动挂载 `modules_to_save` 中的 `claim_hc`，兼容恢复逻辑还会输出：

```text
[videommd] restored claim_hc from adapter checkpoint: .../checkpoint-xxx (... tensors)
```

## FakeTT

四阶段训练必须按顺序运行：

```bash
python scripts/run_swift_sft_with_claim_hc.py configs/swift/fakett_claim_hc_phase1.yaml
python scripts/run_swift_sft_with_claim_hc.py configs/swift/fakett_claim_hc_phase2.yaml
python scripts/run_swift_sft_with_claim_hc.py configs/swift/fakett_claim_hc_phase3.yaml
python scripts/run_swift_sft_with_claim_hc.py configs/swift/fakett_claim_hc_phase4.yaml
```

Phase4 推理：

```bash
python scripts/run_swift_infer_with_claim_hc.py \
  configs/swift/fakett_claim_hc_infer.yaml \
  --adapters /data2/573ops_ser/projects/videommd/outputs/swift/fakett_claim_hc_phase4_ddpfix/v0-20260817-023609/checkpoint-261
```

预测结果：

```text
/data2/573ops_ser/projects/videommd/outputs/swift/fakett_claim_hc_phase4_ddpfix/fakett_phase4_test_predictions.jsonl
```

评估：

```bash
python scripts/evaluate_swift_predictions.py \
  --dataset-jsonl /data2/573ops_ser/projects/videommd/data/swift/fakett/fakett_test.jsonl \
  --prediction-jsonl /data2/573ops_ser/projects/videommd/outputs/swift/fakett_claim_hc_phase4_ddpfix/fakett_phase4_test_predictions.jsonl \
  --output-json /data2/573ops_ser/projects/videommd/outputs/swift/fakett_claim_hc_phase4_ddpfix/fakett_phase4_test_metrics.json
```

## FakeSV

四阶段训练必须按顺序运行：

```bash
python scripts/run_swift_sft_with_claim_hc.py configs/swift/fakesv_claim_hc_phase1.yaml
python scripts/run_swift_sft_with_claim_hc.py configs/swift/fakesv_claim_hc_phase2.yaml
python scripts/run_swift_sft_with_claim_hc.py configs/swift/fakesv_claim_hc_phase3.yaml
python scripts/run_swift_sft_with_claim_hc.py configs/swift/fakesv_claim_hc_phase4.yaml
```

Phase4 推理：

```bash
python scripts/run_swift_infer_with_claim_hc.py \
  configs/swift/fakesv_claim_hc_infer.yaml \
  --adapters /完整路径/到/fakesv/checkpoint-xxx
```

预测结果：

```text
/data2/573ops_ser/projects/videommd/outputs/swift/fakesv_claim_hc_phase4_ddpfix/fakesv_phase4_test_predictions.jsonl
```

评估：

```bash
python scripts/evaluate_swift_predictions.py \
  --dataset-jsonl /data2/573ops_ser/projects/videommd/data/swift/fakesv/fakesv_test.jsonl \
  --prediction-jsonl /data2/573ops_ser/projects/videommd/outputs/swift/fakesv_claim_hc_phase4_ddpfix/fakesv_phase4_test_predictions.jsonl \
  --output-json /data2/573ops_ser/projects/videommd/outputs/swift/fakesv_claim_hc_phase4_ddpfix/fakesv_phase4_test_metrics.json
```

训练时每个 phase 配置都包含 `val_dataset` 和 `eval_strategy: epoch`，因此会在每个 epoch 结束后自动进行验证集评估。

所有训练配置使用 `acc_strategy: seq`，并通过下面三个参数按验证集序列准确率选择最佳 checkpoint：

```yaml
load_best_model_at_end: true
metric_for_best_model: acc
greater_is_better: true
```

phase2 到 phase4 的启动器会优先读取上一阶段 `trainer_state.json` 中的 `best_model_checkpoint`；旧实验没有最佳 checkpoint 记录时，才回退到最近保存的有效 checkpoint。

训练参数由 YAML 管理；推理 adapter 使用 `--adapters` 显式指定，避免自动选择到其他实验的 checkpoint。
