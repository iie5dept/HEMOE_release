from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class FourModalMoEOutput:
    loss: Tensor | None
    logits: Tensor
    llm_logits: Tensor
    router_weights: Tensor
    losses: dict[str, Tensor]


class TokenTransformerEncoder(nn.Module):
    """Aggregate cached contextual tokens without an independent classifier."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        num_heads: int = 8,
        ff_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.summary_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim),
            enable_nested_tensor=False,
        )
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.summary_token, std=0.02)

    def forward(self, token_states: Tensor, attention_mask: Tensor) -> Tensor:
        if token_states.ndim != 3:
            raise ValueError(f"Expected token states [B, T, D], got {tuple(token_states.shape)}")
        if attention_mask.shape != token_states.shape[:2]:
            raise ValueError(
                f"Attention mask {tuple(attention_mask.shape)} does not match tokens {tuple(token_states.shape)}"
            )
        projected = self.input_projection(self.input_norm(token_states))
        summary = self.summary_token.expand(projected.shape[0], -1, -1).to(projected.dtype)
        sequence = torch.cat([summary, projected], dim=1)
        summary_mask = torch.ones(
            (attention_mask.shape[0], 1), device=attention_mask.device, dtype=torch.bool
        )
        valid_mask = torch.cat([summary_mask, attention_mask.bool()], dim=1)
        encoded = self.encoder(sequence, src_key_padding_mask=~valid_mask)
        return self.dropout(encoded[:, 0])


class VectorProjector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.projector = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, decision_states: Tensor) -> Tensor:
        return self.projector(decision_states)


class LLMDecisionClassifier(VectorProjector):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__(input_dim, hidden_dim, dropout)
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, decision_states: Tensor) -> tuple[Tensor, Tensor]:
        hidden = super().forward(decision_states)
        return hidden, self.classifier(hidden)


class MLPExpert(nn.Module):
    def __init__(self, hidden_dim: int, expert_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, expert_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expert_dim, hidden_dim),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.output_norm(inputs + self.net(inputs))


class ReliabilityRouter(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        router_dim: int,
        dropout: float,
        num_modalities: int,
    ) -> None:
        super().__init__()
        self.num_modalities = num_modalities
        input_dim = hidden_dim * num_modalities
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, router_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(router_dim, num_modalities),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, hiddens: list[Tensor]) -> tuple[Tensor, Tensor]:
        if len(hiddens) != self.num_modalities:
            raise ValueError(f"Router expected {self.num_modalities} modalities, got {len(hiddens)}")
        router_features = torch.cat([hidden.float().detach() for hidden in hiddens], dim=-1)
        router_logits = self.net(router_features)
        return router_logits, router_logits.softmax(dim=-1)


class FourModalMoEClassifier(nn.Module):
    """Dense four-modal expert fusion with auxiliary LLM and text supervision."""

    def __init__(
        self,
        llm_dim: int,
        visual_dim: int = 1024,
        text_dim: int = 768,
        audio_dim: int = 4096,
        hidden_dim: int = 256,
        transformer_layers: int = 2,
        transformer_heads: int = 8,
        transformer_ff_dim: int = 1024,
        expert_dim: int = 512,
        router_dim: int = 128,
        dropout: float = 0.1,
        modality_loss_weight: float = 0.3,
        llm_loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.modality_loss_weight = modality_loss_weight
        self.llm_loss_weight = float(llm_loss_weight)
        if self.llm_loss_weight < 0:
            raise ValueError(f"LLM loss weight must be non-negative: {self.llm_loss_weight}")
        branch_kwargs = {
            "hidden_dim": hidden_dim,
            "num_layers": transformer_layers,
            "num_heads": transformer_heads,
            "ff_dim": transformer_ff_dim,
            "dropout": dropout,
        }
        self.visual_branch = TokenTransformerEncoder(visual_dim, **branch_kwargs)
        self.text_branch = VectorProjector(text_dim, hidden_dim, dropout)
        self.llm_branch = LLMDecisionClassifier(llm_dim, hidden_dim, dropout)
        self.audio_branch = VectorProjector(audio_dim, hidden_dim, dropout)

        self.specialist_experts = nn.ModuleList(
            [MLPExpert(hidden_dim, expert_dim, dropout) for _ in range(4)]
        )
        self.router = ReliabilityRouter(hidden_dim, router_dim, dropout, num_modalities=4)
        fusion_dim = hidden_dim * 4
        self.fusion_norm = nn.LayerNorm(fusion_dim)
        self.fusion_classifier = nn.Linear(fusion_dim, 2)

    def forward(
        self,
        llm_decision_states: Tensor,
        visual_tokens: Tensor,
        visual_attention_mask: Tensor,
        text_states: Tensor,
        audio_states: Tensor,
        labels: Tensor | None = None,
    ) -> FourModalMoEOutput:
        llm_hidden, llm_logits = self.llm_branch(llm_decision_states)
        visual_hidden = self.visual_branch(visual_tokens, visual_attention_mask)
        text_hidden = self.text_branch(text_states)
        audio_hidden = self.audio_branch(audio_states)
        hiddens = [llm_hidden, visual_hidden, text_hidden, audio_hidden]

        _, router_weights = self.router(hiddens)
        specialists = torch.stack(
            [expert(hidden) for expert, hidden in zip(self.specialist_experts, hiddens)], dim=1
        )
        weighted_specialists = specialists * router_weights.to(specialists.dtype).unsqueeze(-1)
        fused_hidden = self.fusion_norm(weighted_specialists.flatten(start_dim=1))
        logits = self.fusion_classifier(fused_hidden)

        losses: dict[str, Tensor] = {}
        total_loss = None
        if labels is not None:
            labels = labels.long()
            losses["fusion"] = F.cross_entropy(logits.float(), labels)
            llm_loss = F.cross_entropy(llm_logits.float(), labels)
            if self.llm_loss_weight > 0:
                losses["llm"] = llm_loss
                losses["modality"] = llm_loss * self.llm_loss_weight
            else:
                losses["modality"] = llm_loss.new_zeros(())
            total_loss = losses["fusion"] + self.modality_loss_weight * losses["modality"]
            losses["total"] = total_loss

        return FourModalMoEOutput(
            loss=total_loss,
            logits=logits,
            llm_logits=llm_logits,
            router_weights=router_weights,
            losses=losses,
        )


# Compatibility aliases for checkpoints and imports created before the audio branch was added.
TriModalMoEClassifier = FourModalMoEClassifier
TriModalMoEOutput = FourModalMoEOutput
