from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import Tensor, nn


def masked_attention_pool(sequence: Tensor, mask: Tensor, scorer: nn.Module) -> Tensor:
    mask = mask.bool()
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


class TokenConditionedBlock(nn.Module):

    def __init__(self, hidden_size: int, rank: int) -> None:
        super().__init__()
        self.token_norm = nn.LayerNorm(hidden_size)
        self.token_down = nn.Linear(hidden_size, rank, bias=False)
        self.context_down = nn.Linear(hidden_size, rank, bias=False)
        self.act = nn.GELU()
        self.up = nn.Linear(rank, hidden_size, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, tokens: Tensor, context: Tensor) -> Tensor:
        token_features = self.act(self.token_down(self.token_norm(tokens)))
        context_features = self.context_down(context).unsqueeze(1)
        return self.up(token_features * context_features)


@dataclass
class EvidenceMoEOutput:
    visual_update: Tensor
    text_update: Tensor
    expert_weights: Tensor
    aux_losses: Dict[str, Tensor]


class NativeEvidenceMoE(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        expert_rank: int = 8,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        self.visual_scorer = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 1, bias=True))
        self.text_scorer = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 1, bias=True))
        for scorer in (self.visual_scorer, self.text_scorer):
            nn.init.zeros_(scorer[-1].weight)
            nn.init.zeros_(scorer[-1].bias)

        router_rank = max(16, expert_rank * 2)
        fusion_rank = max(16, expert_rank * 2)
        self.context_adapter = LowRankBlock(hidden_size * 2, hidden_size, fusion_rank)
        self.shared_expert = LowRankBlock(hidden_size, hidden_size, expert_rank)
        self.visual_expert = LowRankBlock(hidden_size, hidden_size, expert_rank)
        self.text_expert = LowRankBlock(hidden_size, hidden_size, expert_rank)
        self.joint_expert = LowRankBlock(hidden_size, hidden_size, expert_rank)

        router_input_dim = hidden_size * 3
        self.expert_router = LowRankRouter(router_input_dim, router_rank, 3)

        token_rank = max(16, expert_rank * 2)
        self.visual_refiner = TokenConditionedBlock(hidden_size, token_rank)
        self.text_refiner = TokenConditionedBlock(hidden_size, token_rank)

    def _routing_weights(
        self,
        router_inputs: Tensor,
    ) -> Tensor:
        dtype = router_inputs.dtype
        expert_prob = torch.softmax(self.expert_router(router_inputs).float(), dim=-1).to(dtype)
        return expert_prob

    def forward(
        self,
        visual_tokens: Tensor,
        visual_mask: Tensor,
        text_tokens: Tensor,
        text_mask: Tensor,
    ) -> EvidenceMoEOutput:
        visual_mask = visual_mask.bool()
        text_mask = text_mask.bool()
        visual_evidence = masked_attention_pool(visual_tokens, visual_mask, self.visual_scorer)
        text_evidence = masked_attention_pool(text_tokens, text_mask, self.text_scorer)
        context_inputs = torch.cat([visual_evidence, text_evidence], dim=-1)
        joint_context = (
            0.5 * (visual_evidence + text_evidence)
            + self.context_adapter(context_inputs)
        )

        shared_update = self.shared_expert(joint_context)
        expert_stack = torch.stack(
            [
                self.visual_expert(visual_evidence),
                self.text_expert(text_evidence),
                self.joint_expert(joint_context),
            ],
            dim=1,
        )
        router_inputs = torch.cat(
            [joint_context, visual_evidence, text_evidence], dim=-1)
        expert_weights = self._routing_weights(router_inputs)
        specialist_update = (expert_weights.unsqueeze(-1) * expert_stack).sum(dim=1)
        routed_update = shared_update + specialist_update
        refinement_context = joint_context + routed_update
        visual_update = self.visual_refiner(visual_tokens, refinement_context)
        text_update = self.text_refiner(text_tokens, refinement_context)
        visual_update = visual_update * visual_mask.unsqueeze(-1).to(visual_update.dtype)
        text_update = text_update * text_mask.unsqueeze(-1).to(text_update.dtype)

        expert_target = expert_weights.new_tensor([1 / 3, 1 / 3, 1 / 3])
        balance_loss = (expert_weights.mean(dim=0) - expert_target).pow(2).mean()

        return EvidenceMoEOutput(
            visual_update=visual_update,
            text_update=text_update,
            expert_weights=expert_weights,
            aux_losses={"balance_loss": balance_loss},
        )
