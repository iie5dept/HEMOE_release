from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def masked_mean_pool(sequence: Tensor, mask: Tensor | None) -> Tensor:
    if mask is None:
        return sequence.mean(dim=1)
    weights = mask.to(dtype=sequence.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (sequence * weights).sum(dim=1) / denom


class FeedForwardBlock(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int | None = None, dropout: float = 0.1) -> None:
        super().__init__()
        intermediate_size = intermediate_size or hidden_size * 4
        self.norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return residual + hidden_states


class GeneralSceneExpert(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int | None = None, dropout: float = 0.1) -> None:
        super().__init__()
        self.ffn = FeedForwardBlock(hidden_size, intermediate_size=intermediate_size, dropout=dropout)

    def forward(self, visual_tokens: Tensor) -> Tensor:
        return self.ffn(visual_tokens)


class ClaimEvidenceExpert(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        intermediate_size: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.claim_norm = nn.LayerNorm(hidden_size)
        self.visual_norm = nn.LayerNorm(hidden_size)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = FeedForwardBlock(hidden_size, intermediate_size=intermediate_size, dropout=dropout)
        self.fuse = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, visual_tokens: Tensor, claim_summary: Tensor) -> Tensor:
        query = self.claim_norm(claim_summary).unsqueeze(1)
        key_value = self.visual_norm(visual_tokens)
        claim_context, _ = self.cross_attn(query=query, key=key_value, value=key_value, need_weights=False)
        claim_context = claim_context.expand(-1, visual_tokens.shape[1], -1)
        fused = self.fuse(torch.cat([visual_tokens, claim_context], dim=-1))
        return self.ffn(fused)


class ClaimAwareGate(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 4, dropout: float = 0.1, num_experts: int = 2) -> None:
        super().__init__()
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, num_experts)

    def forward(self, claim_summary: Tensor, expert_summaries: Tensor) -> Tensor:
        query = self.query_proj(claim_summary).unsqueeze(1)
        attended, _ = self.attn(query=query, key=expert_summaries, value=expert_summaries, need_weights=False)
        logits = self.output(attended.squeeze(1))
        return F.softmax(logits, dim=-1)


class VeracityClassifier(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        fusion_size = hidden_size * 4
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_size),
            nn.Linear(fusion_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(self, claim_summary: Tensor, visual_summary: Tensor) -> Tensor:
        fused = torch.cat(
            [
                claim_summary,
                visual_summary,
                claim_summary * visual_summary,
                torch.abs(claim_summary - visual_summary),
            ],
            dim=-1,
        )
        return self.classifier(fused)


@dataclass
class ClaimHCOutput:
    fused_visual_tokens: Tensor
    gate_weights: Tensor
    general_tokens: Tensor
    claim_tokens: Tensor
    claim_summary: Tensor
    visual_summary: Tensor


class ClaimConditionedHybridCompressor(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        expert_intermediate_size: int | None = None,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.general_expert = GeneralSceneExpert(
            hidden_size=hidden_size,
            intermediate_size=expert_intermediate_size,
            dropout=dropout,
        )
        self.claim_expert = ClaimEvidenceExpert(
            hidden_size=hidden_size,
            num_heads=num_heads,
            intermediate_size=expert_intermediate_size,
            dropout=dropout,
        )
        self.gate = ClaimAwareGate(hidden_size=hidden_size, num_heads=max(1, num_heads // 2), dropout=dropout)
        self.classifier = VeracityClassifier(hidden_size=hidden_size, dropout=dropout)

    def forward(self, visual_tokens: Tensor, claim_tokens: Tensor, claim_mask: Tensor | None = None) -> ClaimHCOutput:
        claim_summary = masked_mean_pool(claim_tokens, claim_mask)
        general_tokens = self.general_expert(visual_tokens)
        claim_tokens_out = self.claim_expert(visual_tokens, claim_summary)

        general_summary = general_tokens.mean(dim=1)
        claim_summary_visual = claim_tokens_out.mean(dim=1)
        expert_summaries = torch.stack([general_summary, claim_summary_visual], dim=1)
        gate_weights = self.gate(claim_summary, expert_summaries)

        fused_visual_tokens = (
            gate_weights[:, 0].view(-1, 1, 1) * general_tokens
            + gate_weights[:, 1].view(-1, 1, 1) * claim_tokens_out
        )
        visual_summary = fused_visual_tokens.mean(dim=1)
        return ClaimHCOutput(
            fused_visual_tokens=fused_visual_tokens,
            gate_weights=gate_weights,
            general_tokens=general_tokens,
            claim_tokens=claim_tokens_out,
            claim_summary=claim_summary,
            visual_summary=visual_summary,
        )

    def compute_aux_loss(self, hc_output: ClaimHCOutput, veracity_labels: Tensor) -> tuple[Tensor, Tensor]:
        logits = self.classifier(hc_output.claim_summary, hc_output.visual_summary)
        loss = F.cross_entropy(logits, veracity_labels)
        return loss, logits
