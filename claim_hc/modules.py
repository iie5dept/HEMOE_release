from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import Tensor, nn


def masked_attention_pool(sequence: Tensor, mask: Tensor, scorer: nn.Module) -> Tensor:
    scores = scorer(sequence).squeeze(-1)
    scores = scores.masked_fill(~mask, float("-inf"))
    no_valid = ~mask.any(dim=1)
    if no_valid.any():
        scores = scores.clone()
        scores[no_valid] = 0.0
    weights = torch.softmax(scores.float(), dim=-1).to(sequence.dtype)
    weights = weights * mask.to(dtype=sequence.dtype)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return torch.bmm(weights.unsqueeze(1), sequence).squeeze(1)


def gather_tokens_by_mask(sequence: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    batch_size, _, hidden_size = sequence.shape
    lengths = mask.sum(dim=1)
    max_len = int(lengths.max().item()) if lengths.numel() > 0 else 0
    gathered = sequence.new_zeros((batch_size, max_len, hidden_size))
    gathered_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=sequence.device)
    for idx in range(batch_size):
        valid = sequence[idx][mask[idx]]
        if valid.numel() == 0:
            continue
        token_len = valid.shape[0]
        gathered[idx, :token_len] = valid
        gathered_mask[idx, :token_len] = True
    return gathered, gathered_mask


class LowRankBlock(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, rank: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.down = nn.Linear(input_dim, rank, bias=False)
        self.act = nn.GELU()
        self.up = nn.Linear(rank, output_dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.up(self.act(self.down(self.norm(inputs))))


class LowRankRouter(nn.Module):

    def __init__(self, input_dim: int, rank: int, output_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.down = nn.Linear(input_dim, rank, bias=False)
        self.act = nn.GELU()
        self.up = nn.Linear(rank, output_dim, bias=True)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.up(self.act(self.down(self.norm(inputs))))


@dataclass
class EvidenceMoEOutput:
    visual_update: Tensor
    text_update: Tensor
    coarse_weights: Tensor
    expert_weights: Tensor
    aux_losses: Dict[str, Tensor]


class NativeEvidenceMoE(nn.Module):

    def __init__(self, hidden_size: int, expert_rank: int = 8, top_k: int = 2) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.top_k = top_k

        self.visual_scorer = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 1, bias=True))
        self.text_scorer = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 1, bias=True))

        router_rank = max(16, expert_rank * 2)
        fusion_rank = max(16, expert_rank * 2)
        self.cross_adapter = LowRankBlock(hidden_size * 4, hidden_size, fusion_rank)
        self.shared_expert = LowRankBlock(hidden_size, hidden_size, expert_rank)
        self.visual_expert = LowRankBlock(hidden_size, hidden_size, expert_rank)
        self.text_expert = LowRankBlock(hidden_size, hidden_size, expert_rank)
        self.cross_expert = LowRankBlock(hidden_size, hidden_size, expert_rank)

        router_input_dim = hidden_size * 4
        self.coarse_router = LowRankRouter(router_input_dim, router_rank, 2)
        self.expert_router = LowRankRouter(router_input_dim, router_rank, 3)

        # The complete module is an exact identity before training.
        self.visual_token_scale = nn.Parameter(torch.zeros(()))
        self.text_token_scale = nn.Parameter(torch.zeros(()))

    @staticmethod
    def _straight_through_topk(probabilities: Tensor, top_k: int) -> Tensor:
        top_k = min(top_k, probabilities.shape[-1])
        values, indices = torch.topk(probabilities, k=top_k, dim=-1)
        sparse = torch.zeros_like(probabilities)
        sparse.scatter_(dim=-1, index=indices, src=values)
        sparse = sparse / sparse.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return sparse + probabilities - probabilities.detach()

    def _routing_weights(
        self,
        router_inputs: Tensor,
        stage: str,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch_size = router_inputs.shape[0]
        dtype = router_inputs.dtype
        device = router_inputs.device
        if stage == "phase1":
            coarse = torch.tensor([1.0, 0.0], dtype=dtype, device=device).expand(batch_size, -1)
            experts = torch.full((batch_size, 3), 1 / 3, dtype=dtype, device=device)
            return coarse, experts, coarse, experts
        if stage == "phase2":
            coarse = torch.full((batch_size, 2), 0.5, dtype=dtype, device=device)
            experts = torch.full((batch_size, 3), 1 / 3, dtype=dtype, device=device)
            return coarse, experts, coarse, experts

        coarse_prob = torch.softmax(self.coarse_router(router_inputs).float(), dim=-1).to(dtype)
        expert_prob = torch.softmax(self.expert_router(router_inputs).float(), dim=-1).to(dtype)
        if stage == "phase3":
            return coarse_prob, expert_prob, coarse_prob, expert_prob
        expert_weights = self._straight_through_topk(expert_prob, self.top_k)
        return coarse_prob, expert_weights, coarse_prob, expert_prob

    def forward(
        self,
        visual_tokens: Tensor,
        visual_mask: Tensor,
        text_tokens: Tensor,
        text_mask: Tensor,
        stage: str,
    ) -> EvidenceMoEOutput:
        visual_evidence = masked_attention_pool(visual_tokens, visual_mask, self.visual_scorer)
        text_evidence = masked_attention_pool(text_tokens, text_mask, self.text_scorer)
        difference = visual_evidence - text_evidence
        interaction = visual_evidence * text_evidence
        cross_inputs = torch.cat(
            [visual_evidence, text_evidence, difference.abs(), interaction], dim=-1)
        cross_delta = self.cross_adapter(cross_inputs)
        cross_evidence = difference + cross_delta
        global_evidence = 0.5 * (visual_evidence + text_evidence) + cross_delta

        shared_update = self.shared_expert(global_evidence)
        expert_stack = torch.stack(
            [
                self.visual_expert(visual_evidence),
                self.text_expert(text_evidence),
                self.cross_expert(cross_evidence),
            ],
            dim=1,
        )
        router_inputs = torch.cat(
            [global_evidence, visual_evidence, text_evidence, cross_evidence.abs()], dim=-1)
        coarse_weights, expert_weights, coarse_prob, expert_prob = self._routing_weights(
            router_inputs, stage=stage)
        specialist_update = (expert_weights.unsqueeze(-1) * expert_stack).sum(dim=1)
        routed_update = (
            coarse_weights[:, :1] * shared_update
            + coarse_weights[:, 1:] * specialist_update
        )
        fused_evidence = global_evidence + routed_update
        visual_update = self.visual_token_scale.tanh() * fused_evidence
        text_update = self.text_token_scale.tanh() * fused_evidence

        coarse_target = coarse_prob.new_tensor([0.5, 0.5])
        expert_target = expert_prob.new_tensor([1 / 3, 1 / 3, 1 / 3])
        balance_loss = (
            (coarse_prob.mean(dim=0) - coarse_target).pow(2).mean()
            + (expert_prob.mean(dim=0) - expert_target).pow(2).mean()
        )

        return EvidenceMoEOutput(
            visual_update=visual_update,
            text_update=text_update,
            coarse_weights=coarse_weights,
            expert_weights=expert_weights,
            aux_losses={"balance_loss": balance_loss},
        )
