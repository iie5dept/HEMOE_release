# Swift Claim-HC Manual Adapter Notes

Phase2 to phase4 no longer auto-select the previous checkpoint.
Always pass the exact checkpoint manually with `--adapters`.

```bash
cd /data2/573ops_ser/projects/videommd
```

## FakeTT

```bash
python scripts/run_swift_sft_with_claim_hc.py configs/swift/fakett_claim_hc_phase1.yaml
python scripts/run_swift_sft_with_claim_hc.py configs/swift/fakett_claim_hc_phase2.yaml --adapters /data2/573ops_ser/projects/videommd/outputs/swift/fakett_claim_hc_phase1_ddpfix/.../checkpoint-xxx
python scripts/run_swift_sft_with_claim_hc.py configs/swift/fakett_claim_hc_phase3.yaml --adapters /data2/573ops_ser/projects/videommd/outputs/swift/fakett_claim_hc_phase2_ddpfix/.../checkpoint-xxx
python scripts/run_swift_sft_with_claim_hc.py configs/swift/fakett_claim_hc_phase4.yaml --adapters /data2/573ops_ser/projects/videommd/outputs/swift/fakett_claim_hc_phase3_ddpfix/.../checkpoint-xxx
```

## FakeSV

```bash
python scripts/run_swift_sft_with_claim_hc.py configs/swift/fakesv_claim_hc_phase1.yaml
python scripts/run_swift_sft_with_claim_hc.py configs/swift/fakesv_claim_hc_phase2.yaml --adapters /data2/573ops_ser/projects/videommd/outputs/swift/fakesv_claim_hc_phase1_ddpfix/.../checkpoint-xxx
python scripts/run_swift_sft_with_claim_hc.py configs/swift/fakesv_claim_hc_phase3.yaml --adapters /data2/573ops_ser/projects/videommd/outputs/swift/fakesv_claim_hc_phase2_ddpfix/.../checkpoint-xxx
python scripts/run_swift_sft_with_claim_hc.py configs/swift/fakesv_claim_hc_phase4.yaml --adapters /data2/573ops_ser/projects/videommd/outputs/swift/fakesv_claim_hc_phase3_ddpfix/.../checkpoint-xxx
```

## Inference

```bash
python scripts/run_swift_infer_with_claim_hc.py \
  configs/swift/fakett_claim_hc_infer.yaml \
  --adapters /data2/573ops_ser/projects/videommd/outputs/swift/fakett_claim_hc_phase4_ddpfix/.../checkpoint-xxx
```
