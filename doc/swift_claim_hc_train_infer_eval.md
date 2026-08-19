# Swift Claim-HC 训练、推理与评估

Claim-HC 采用单阶段联合训练。所有专家、dense router、token refiner 和 LoRA 同时优化，训练和推理使用同一条路由路径。

```bash
cd /data2/573ops_ser/projects/videommd
python scripts/check_claim_hc_module.py
```

## FakeTT

训练：

```bash
python scripts/run_swift_sft_with_claim_hc.py configs/swift/fakett_claim_hc_joint.yaml
```

推理时必须用 `--adapters` 明确指定训练得到的 checkpoint：

```bash
python scripts/run_swift_infer_with_claim_hc.py \
  configs/swift/fakett_claim_hc_infer.yaml \
  --adapters /data2/573ops_ser/projects/videommd/outputs/swift/fakett_claim_hc_joint_ddpfix/.../checkpoint-xxx
```

评估：

```bash
python scripts/evaluate_swift_predictions.py \
  --dataset-jsonl /data2/573ops_ser/projects/videommd/data/swift/fakett/fakett_test.jsonl \
  --prediction-jsonl /data2/573ops_ser/projects/videommd/outputs/swift/fakett_claim_hc_joint_ddpfix/fakett_joint_test_predictions.jsonl \
  --output-json /data2/573ops_ser/projects/videommd/outputs/swift/fakett_claim_hc_joint_ddpfix/fakett_joint_test_metrics.json
```

## FakeSV

训练：

```bash
python scripts/run_swift_sft_with_claim_hc.py configs/swift/fakesv_claim_hc_joint.yaml
```

推理：

```bash
python scripts/run_swift_infer_with_claim_hc.py \
  configs/swift/fakesv_claim_hc_infer.yaml \
  --adapters /data2/573ops_ser/projects/videommd/outputs/swift/fakesv_claim_hc_joint_ddpfix/.../checkpoint-xxx
```

评估：

```bash
python scripts/evaluate_swift_predictions.py \
  --dataset-jsonl /data2/573ops_ser/projects/videommd/data/swift/fakesv/fakesv_test.jsonl \
  --prediction-jsonl /data2/573ops_ser/projects/videommd/outputs/swift/fakesv_claim_hc_joint_ddpfix/fakesv_joint_test_predictions.jsonl \
  --output-json /data2/573ops_ser/projects/videommd/outputs/swift/fakesv_claim_hc_joint_ddpfix/fakesv_joint_test_metrics.json
```

训练配置使用 `eval_strategy: epoch` 和 `metric_for_best_model: seq_acc`，每个 epoch 验证并记录最佳 checkpoint。测试 checkpoint 始终由命令行显式选择，不再自动解析历史输出目录。
