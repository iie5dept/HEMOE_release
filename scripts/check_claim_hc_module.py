from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from claim_hc.modules import NativeEvidenceMoE
from claim_hc.runtime import _configure_stage


def main() -> None:
    torch.manual_seed(7)
    module = NativeEvidenceMoE(hidden_size=64, expert_rank=4, top_k=2)
    parameter_count = sum(parameter.numel() for parameter in module.parameters())

    visual_tokens = torch.randn(2, 6, 64)
    text_tokens = torch.randn(2, 9, 64)
    visual_mask = torch.ones(2, 6, dtype=torch.bool)
    text_mask = torch.ones(2, 9, dtype=torch.bool)

    initial = module(visual_tokens, visual_mask, text_tokens, text_mask, stage="phase4")
    if initial.visual_update.count_nonzero() or initial.text_update.count_nonzero():
        raise AssertionError("claim_hc must be an exact identity at initialization")

    for stage in ("phase1", "phase2", "phase3", "phase4"):
        _configure_stage(module, stage)
        if stage in {"phase3", "phase4"}:
            with torch.no_grad():
                module.visual_token_scale.fill_(0.1)
                module.text_token_scale.fill_(0.1)
        module.zero_grad(set_to_none=True)
        output = module(visual_tokens, visual_mask, text_tokens, text_mask, stage=stage)
        loss = output.visual_update.sum() + output.text_update.sum()
        if stage in {"phase3", "phase4"}:
            loss = loss + output.aux_losses["balance_loss"]
        loss.backward()
        missing = [
            name
            for name, parameter in module.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        ]
        if missing:
            raise AssertionError(f"{stage} has trainable parameters outside the graph: {missing}")
        trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
        print({"stage": stage, "parameters": parameter_count, "trainable": trainable, "grad_check": "ok"})


if __name__ == "__main__":
    main()
