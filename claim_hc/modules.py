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


class ScenePriorExpert(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int | None = None, dropout: float = 0.1) -> None:
        super().__init__()
        self.local_ffn = FeedForwardBlock(hidden_size, intermediate_size=intermediate_size, dropout=dropout)
        self.global_norm = nn.LayerNorm(hidden_size)
        self.global_proj = nn.Linear(hidden_size * 2, hidden_size)
        self.output_ffn = FeedForwardBlock(hidden_size, intermediate_size=intermediate_size, dropout=dropout)

    def forward(self, visual_tokens: Tensor) -> Tensor:
        local_tokens = self.local_ffn(visual_tokens)
        global_summary = self.global_norm(local_tokens.mean(dim=1, keepdim=True))
        global_summary = global_summary.expand(-1, local_tokens.shape[1], -1)
        fused = self.global_proj(torch.cat([local_tokens, global_summary], dim=-1))
        return self.output_ffn(fused)


class ClaimVerifierExpert(nn.Module):
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
        self.support_fuse = nn.Linear(hidden_size * 2, hidden_size)
        self.conflict_fuse = nn.Linear(hidden_size * 3, hidden_size)
        self.support_ffn = FeedForwardBlock(hidden_size, intermediate_size=intermediate_size, dropout=dropout)
        self.conflict_ffn = FeedForwardBlock(hidden_size, intermediate_size=intermediate_size, dropout=dropout)

    def forward(self, visual_tokens: Tensor, claim_summary: Tensor) -> tuple[Tensor, Tensor]:
        query = self.claim_norm(claim_summary).unsqueeze(1)
        key_value = self.visual_norm(visual_tokens)
        claim_context, _ = self.cross_attn(query=query, key=key_value, value=key_value, need_weights=False)
        claim_context = claim_context.expand(-1, visual_tokens.shape[1], -1)

        support_tokens = self.support_ffn(self.support_fuse(torch.cat([visual_tokens, claim_context], dim=-1)))
        conflict_input = torch.cat(
            [
                visual_tokens,
                claim_context,
                visual_tokens - claim_context,
            ],
            dim=-1,
        )
        conflict_tokens = self.conflict_ffn(self.conflict_fuse(conflict_input))
        return support_tokens, conflict_tokens


class RoutingGate(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 4, dropout: float = 0.1, num_options: int = 2) -> None:
        super().__init__()
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, num_options)

    def forward(self, claim_summary: Tensor, option_summaries: Tensor) -> Tensor:
        query = self.query_proj(claim_summary).unsqueeze(1)
        attended, _ = self.attn(query=query, key=option_summaries, value=option_summaries, need_weights=False)
        logits = self.output(attended.squeeze(1))
        return F.softmax(logits, dim=-1)


class HierarchicalClaimGate(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        gate_heads = max(1, num_heads)
        self.coarse_gate = RoutingGate(hidden_size=hidden_size, num_heads=gate_heads, dropout=dropout, num_options=2)
        self.fine_gate = RoutingGate(hidden_size=hidden_size, num_heads=gate_heads, dropout=dropout, num_options=2)

    def forward(
        self,
        claim_summary: Tensor,
        general_summary: Tensor,
        support_summary: Tensor,
        conflict_summary: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        verifier_summary = 0.5 * (support_summary + conflict_summary)
        coarse_weights = self.coarse_gate(
            claim_summary,
            torch.stack([general_summary, verifier_summary], dim=1),
        )
        fine_weights = self.fine_gate(
            claim_summary,
            torch.stack([support_summary, conflict_summary], dim=1),
        )

        expert_weights = torch.stack(
            [
                coarse_weights[:, 0],
                coarse_weights[:, 1] * fine_weights[:, 0],
                coarse_weights[:, 1] * fine_weights[:, 1],
            ],
            dim=-1,
        )
        return expert_weights, coarse_weights, fine_weights


class VeracityClassifier(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        fusion_size = hidden_size * 7
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_size),
            nn.Linear(fusion_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(
        self,
        claim_summary: Tensor,
        visual_summary: Tensor,
        support_summary: Tensor,
        conflict_summary: Tensor,
    ) -> Tensor:
        evidence_delta = support_summary - conflict_summary
        fused = torch.cat(
            [
                claim_summary,
                visual_summary,
                evidence_delta,
                claim_summary * visual_summary,
                torch.abs(claim_summary - visual_summary),
                support_summary * conflict_summary,
                torch.abs(support_summary - conflict_summary),
            ],
            dim=-1,
        )
        return self.classifier(fused)


@dataclass
class ClaimHCOutput:
    fused_visual_tokens: Tensor
    gate_weights: Tensor
    coarse_gate_weights: Tensor
    fine_gate_weights: Tensor
    general_tokens: Tensor
    support_tokens: Tensor
    conflict_tokens: Tensor
    claim_summary: Tensor
    general_summary: Tensor
    support_summary: Tensor
    conflict_summary: Tensor
    visual_summary: Tensor


class ClaimConditionedHybridCompressor(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        expert_intermediate_size: int | None = None,
        num_heads: int = 8,
        dropout: float = 0.1,
        decorrelation_weight: float = 0.05,
    ) -> None:
        super().__init__()
        self.decorrelation_weight = decorrelation_weight
        self.general_expert = ScenePriorExpert(
            hidden_size=hidden_size,
            intermediate_size=expert_intermediate_size,
            dropout=dropout,
        )
        self.claim_expert = ClaimVerifierExpert(
            hidden_size=hidden_size,
            num_heads=num_heads,
            intermediate_size=expert_intermediate_size,
            dropout=dropout,
        )
        self.gate = HierarchicalClaimGate(
            hidden_size=hidden_size,
            num_heads=max(1, num_heads // 2),
            dropout=dropout,
        )
        self.classifier = VeracityClassifier(hidden_size=hidden_size, dropout=dropout)

    def forward(self, visual_tokens: Tensor, claim_tokens: Tensor, claim_mask: Tensor | None = None) -> ClaimHCOutput:
        claim_summary = masked_mean_pool(claim_tokens, claim_mask)
        general_tokens = self.general_expert(visual_tokens)
        support_tokens, conflict_tokens = self.claim_expert(visual_tokens, claim_summary)

        general_summary = general_tokens.mean(dim=1)
        support_summary = support_tokens.mean(dim=1)
        conflict_summary = conflict_tokens.mean(dim=1)
        gate_weights, coarse_gate_weights, fine_gate_weights = self.gate(
            claim_summary=claim_summary,
            general_summary=general_summary,
            support_summary=support_summary,
            conflict_summary=conflict_summary,
        )

        fused_visual_tokens = (
            gate_weights[:, 0].view(-1, 1, 1) * general_tokens
            + gate_weights[:, 1].view(-1, 1, 1) * support_tokens
            + gate_weights[:, 2].view(-1, 1, 1) * conflict_tokens
        )
        visual_summary = fused_visual_tokens.mean(dim=1)
        return ClaimHCOutput(
            fused_visual_tokens=fused_visual_tokens,
            gate_weights=gate_weights,
            coarse_gate_weights=coarse_gate_weights,
            fine_gate_weights=fine_gate_weights,
            general_tokens=general_tokens,
            support_tokens=support_tokens,
            conflict_tokens=conflict_tokens,
            claim_summary=claim_summary,
            general_summary=general_summary,
            support_summary=support_summary,
            conflict_summary=conflict_summary,
            visual_summary=visual_summary,
        )

    def _decorrelation_loss(self, hc_output: ClaimHCOutput) -> Tensor:
        sims = [
            F.cosine_similarity(hc_output.general_summary, hc_output.support_summary, dim=-1).abs(),
            F.cosine_similarity(hc_output.general_summary, hc_output.conflict_summary, dim=-1).abs(),
            F.cosine_similarity(hc_output.support_summary, hc_output.conflict_summary, dim=-1).abs(),
        ]
        return torch.stack(sims, dim=0).mean()

    def compute_aux_loss(self, hc_output: ClaimHCOutput, veracity_labels: Tensor) -> tuple[Tensor, Tensor]:
        logits = self.classifier(
            claim_summary=hc_output.claim_summary,
            visual_summary=hc_output.visual_summary,
            support_summary=hc_output.support_summary,
            conflict_summary=hc_output.conflict_summary,
        )
        cls_loss = F.cross_entropy(logits, veracity_labels)
        decorrelation_loss = self._decorrelation_loss(hc_output)
        loss = cls_loss + self.decorrelation_weight * decorrelation_loss
        return loss, logits


# Backward-compatible aliases for older imports.
GeneralSceneExpert = ScenePriorExpert
ClaimEvidenceExpert = ClaimVerifierExpert
ClaimAwareGate = HierarchicalClaimGate
