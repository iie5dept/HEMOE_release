from __future__ import annotations

import os
from typing import Any

import torch
from torch import Tensor

from .modules import ClaimConditionedHybridCompressor


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return float(value)


def get_base_model(model: Any) -> Any:
    def is_internvl_chat_like(candidate: Any) -> bool:
        return (
            candidate is not None
            and hasattr(candidate, "language_model")
            and hasattr(candidate, "extract_feature")
            and hasattr(candidate, "get_input_embeddings")
        )

    if is_internvl_chat_like(model):
        return model

    queue = [model]
    seen = set()
    while queue:
        candidate = queue.pop(0)
        candidate_id = id(candidate)
        if candidate_id in seen:
            continue
        seen.add(candidate_id)

        if is_internvl_chat_like(candidate):
            return candidate

        for attr_name in ("module", "base_model", "model"):
            child = getattr(candidate, attr_name, None)
            if child is not None:
                queue.append(child)

    return model


def ensure_claim_hc(model: Any, hidden_size: int, dtype: torch.dtype, device: torch.device) -> Any:
    target_model = get_base_model(model)
    claim_hc = getattr(target_model, "claim_hc", None)
    if claim_hc is None:
        claim_hc = ClaimConditionedHybridCompressor(hidden_size=hidden_size)
        target_model.claim_hc = claim_hc.to(device=device, dtype=dtype)
        target_model.claim_hc_lambda = _env_float("CLAIM_HC_LAMBDA", 0.2)
    return target_model


def _encode_token_sequence(processor: Any, text: str) -> list[int]:
    token_ids = processor.encode(text, add_special_tokens=False)
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    return [int(x) for x in token_ids]


def _find_subsequence(sequence: list[int], pattern: list[int], start: int = 0) -> int:
    if not pattern:
        return -1
    plen = len(pattern)
    for idx in range(start, len(sequence) - plen + 1):
        if sequence[idx:idx + plen] == pattern:
            return idx
    return -1


def build_claim_text_mask(input_ids: Tensor, selected: Tensor, processor: Any) -> Tensor | None:
    open_ids = _encode_token_sequence(processor, "<claim>")
    close_ids = _encode_token_sequence(processor, "</claim>")
    if not open_ids or not close_ids:
        return None

    selected = selected.to(device=input_ids.device, dtype=torch.bool)
    batch_size, seq_len = input_ids.shape
    full_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=input_ids.device)
    for batch_idx in range(batch_size):
        row = input_ids[batch_idx].tolist()
        start = _find_subsequence(row, open_ids)
        if start < 0:
            continue
        content_start = start + len(open_ids)
        end = _find_subsequence(row, close_ids, start=content_start)
        if end < 0 or end <= content_start:
            continue
        full_mask[batch_idx, content_start:end] = True

    text_only_mask = full_mask.masked_select(~selected).reshape(batch_size, -1)
    if text_only_mask.sum() == 0:
        return None
    return text_only_mask


def extract_veracity_label(template_inputs: Any) -> Tensor | None:
    messages = getattr(template_inputs, "messages", None)
    if not messages:
        return None

    if isinstance(messages, list) and messages:
        assistant_message = messages[-1]
        if isinstance(assistant_message, dict):
            content = str(assistant_message.get("content", "")).strip().lower()
        else:
            content = str(getattr(assistant_message, "content", "")).strip().lower()
    else:
        return None

    if content == "real":
        value = 0
    elif content == "fake":
        value = 1
    else:
        return None
    return torch.tensor([value], dtype=torch.long)


def apply_claim_hc(
    model: Any,
    inputs_embeds: Tensor,
    text_features: Tensor,
    vit_embeds: Tensor,
    selected: Tensor,
    claim_text_mask: Tensor | None,
    veracity_label: Tensor | None,
) -> tuple[Tensor, Tensor | None]:
    if vit_embeds is None or text_features is None:
        return inputs_embeds, None
    claim_text_mask = torch.ones(
        text_features.shape[:2],
        dtype=torch.bool,
        device=text_features.device,
    )

    target_model = ensure_claim_hc(
        model=model,
        hidden_size=vit_embeds.shape[-1],
        dtype=vit_embeds.dtype,
        device=vit_embeds.device,
    )
    claim_text_mask = claim_text_mask.to(device=vit_embeds.device, dtype=torch.bool)
    text_features = text_features.to(device=vit_embeds.device, dtype=vit_embeds.dtype)
    hc_output = target_model.claim_hc(vit_embeds, claim_tokens=text_features, claim_mask=claim_text_mask)

    batch_size, seq_len, hidden_size = inputs_embeds.shape
    flat_inputs = inputs_embeds.reshape(batch_size * seq_len, hidden_size)
    fused_visual = hc_output.fused_visual_tokens.reshape(-1, hidden_size).to(flat_inputs.dtype)
    selected_flat = selected.reshape(-1)
    original_visual = flat_inputs[selected_flat]
    fused_visual = original_visual + fused_visual
    flat_inputs[selected_flat] = fused_visual
    updated_inputs = flat_inputs.reshape(batch_size, seq_len, hidden_size)

    return updated_inputs, None
