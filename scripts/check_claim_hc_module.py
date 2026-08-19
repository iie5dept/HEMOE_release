from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from claim_hc.modules import NativeEvidenceMoE


def main() -> None:
    torch.manual_seed(7)
    module = NativeEvidenceMoE(hidden_size=64, expert_rank=4)
    parameter_count = sum(parameter.numel() for parameter in module.parameters())

    visual_tokens = torch.randn(2, 6, 64)
    text_tokens = torch.randn(2, 9, 64)
    visual_mask = torch.tensor(
        [[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0]], dtype=torch.bool
    )
    text_mask = torch.tensor(
        [[1, 1, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 0, 0]], dtype=torch.bool
    )

    initial = module(visual_tokens, visual_mask, text_tokens, text_mask)
    if initial.visual_update.shape != visual_tokens.shape:
        raise AssertionError(f"Unexpected visual update shape: {initial.visual_update.shape}")
    if initial.text_update.shape != text_tokens.shape:
        raise AssertionError(f"Unexpected text update shape: {initial.text_update.shape}")
    if initial.visual_update.count_nonzero() or initial.text_update.count_nonzero():
        raise AssertionError("claim_hc residual branch must be zero at initialization")

    initial_loss = initial.visual_update.sum() + initial.text_update.sum()
    initial_loss = initial_loss + initial.aux_losses["balance_loss"]
    initial_loss.backward()
    first_step_parameters = (
        "visual_refiner.up.weight",
        "text_refiner.up.weight",
    )
    named_parameters = dict(module.named_parameters())
    stalled = [
        name
        for name in first_step_parameters
        if named_parameters[name].grad is None or not named_parameters[name].grad.count_nonzero()
    ]
    if stalled:
        raise AssertionError(f"Output projections stalled on the first backward: {stalled}")

    learning_rate = 3e-5
    with torch.no_grad():
        for name in first_step_parameters:
            parameter = named_parameters[name]
            parameter.add_(parameter.grad, alpha=-learning_rate)
    module.zero_grad(set_to_none=True)

    active = module(visual_tokens, visual_mask, text_tokens, text_mask)
    if not active.visual_update.count_nonzero() or not active.text_update.count_nonzero():
        raise AssertionError("Refiners must become active after the first optimizer step")
    visual_ratio = active.visual_update.norm() / visual_tokens.norm()
    text_ratio = active.text_update.norm() / text_tokens.norm()
    active_loss = active.visual_update.sum() + active.text_update.sum()
    active_loss = active_loss + active.aux_losses["balance_loss"]
    active_loss.backward()
    second_step_parameters = (
        "context_adapter.up.weight",
        "shared_expert.up.weight",
        "visual_expert.up.weight",
        "text_expert.up.weight",
        "joint_expert.up.weight",
        "visual_refiner.up.weight",
        "visual_refiner.token_down.weight",
        "visual_refiner.context_down.weight",
        "text_refiner.up.weight",
        "text_refiner.token_down.weight",
        "text_refiner.context_down.weight",
    )
    stalled = [
        name
        for name in second_step_parameters
        if named_parameters[name].grad is None or not named_parameters[name].grad.count_nonzero()
    ]
    if stalled:
        raise AssertionError(f"Parameters stalled after the first optimizer step: {stalled}")

    if torch.equal(active.visual_update[0, 0], active.visual_update[0, 1]):
        raise AssertionError("Visual updates must depend on individual token content")
    if torch.equal(active.text_update[0, 0], active.text_update[0, 1]):
        raise AssertionError("Text updates must depend on individual token content")
    if active.visual_update[~visual_mask].count_nonzero():
        raise AssertionError("Masked visual tokens must receive zero update")
    if active.text_update[~text_mask].count_nonzero():
        raise AssertionError("Masked text tokens must receive zero update")

    module.zero_grad(set_to_none=True)
    output = module(visual_tokens, visual_mask, text_tokens, text_mask)
    loss = output.visual_update.sum() + output.text_update.sum()
    loss = loss + output.aux_losses["balance_loss"]
    loss.backward()
    missing = [
        name
        for name, parameter in module.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    if missing:
        raise AssertionError(f"Trainable parameters outside the graph: {missing}")
    trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    print({
        "parameters": parameter_count,
        "trainable": trainable,
        "visual_residual_ratio": float(visual_ratio.detach()),
        "text_residual_ratio": float(text_ratio.detach()),
        "grad_check": "ok",
    })


if __name__ == "__main__":
    main()
