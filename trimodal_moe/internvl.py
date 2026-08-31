from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .modules import FourModalMoEClassifier, FourModalMoEOutput


def _language_hidden_size(model: nn.Module) -> int:
    config = model.language_model.config
    for name in ("hidden_size", "n_embd", "d_model"):
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    raise ValueError("Cannot determine the language-model hidden size")


def infer_num_image_tokens(model: nn.Module, image_size: int = 448) -> int:
    value = getattr(model, "num_image_token", None)
    if value is not None:
        return int(value)
    patch_size = int(getattr(model.config.vision_config, "patch_size", 14))
    ratio = float(getattr(model.config, "downsample_ratio", 0.5))
    return int((image_size // patch_size) ** 2 * ratio**2)


def _decoder_backbone(language_model: nn.Module) -> nn.Module:
    base = language_model.get_base_model() if hasattr(language_model, "get_base_model") else language_model
    for name in ("model", "transformer"):
        module = getattr(base, name, None)
        if isinstance(module, nn.Module):
            return module
    raise ValueError(f"Cannot locate decoder backbone inside {type(language_model).__name__}")


class InternVLFourModalModel(nn.Module):
    def __init__(self, internvl: nn.Module, classifier: FourModalMoEClassifier, image_context_token_id: int) -> None:
        super().__init__()
        self.internvl = internvl
        self.classifier = classifier
        self.image_context_token_id = image_context_token_id

    @property
    def language_model(self) -> nn.Module:
        return self.internvl.language_model

    def encode_llm(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        pixel_values: Tensor,
    ) -> Tensor:
        language_model = self.language_model
        embeddings = language_model.get_input_embeddings()(input_ids)
        vision_dtype = next(self.internvl.vision_model.parameters()).dtype
        vision_features = self.internvl.extract_feature(pixel_values.to(dtype=vision_dtype))
        selected = input_ids.eq(self.image_context_token_id)
        flattened = vision_features.reshape(-1, vision_features.shape[-1]).to(embeddings.dtype)
        selected_count = int(selected.sum().item())
        if selected_count != flattened.shape[0]:
            raise RuntimeError(
                f"InternVL image-token mismatch: prompt has {selected_count}, vision produced {flattened.shape[0]}"
            )
        embeddings = embeddings.clone()
        embeddings[selected] = flattened
        decoder = _decoder_backbone(language_model)
        outputs = decoder(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state
        decision_indices = attention_mask.long().sum(dim=-1).sub(1).clamp_min(0)
        batch_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
        return hidden_states[batch_indices, decision_indices]

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        pixel_values: Tensor,
        visual_tokens: Tensor,
        visual_attention_mask: Tensor,
        text_states: Tensor,
        audio_states: Tensor,
        labels: Tensor | None = None,
        router_loss_scale: float = 1.0,
    ) -> FourModalMoEOutput:
        decision_states = self.encode_llm(input_ids, attention_mask, pixel_values)
        return self.classifier(
            llm_decision_states=decision_states,
            visual_tokens=visual_tokens,
            visual_attention_mask=visual_attention_mask,
            text_states=text_states,
            audio_states=audio_states,
            labels=labels,
            router_loss_scale=router_loss_scale,
        )


def load_internvl_with_lora(
    model_path: str | Path,
    dtype: torch.dtype,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    gradient_checkpointing: bool,
    use_flash_attn: bool = True,
    adapter_path: str | Path | None = None,
) -> tuple[nn.Module, Any]:
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModel, AutoTokenizer

    model_path = str(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
        dtype=dtype,
        use_flash_attn=use_flash_attn,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    if adapter_path:
        model.language_model = PeftModel.from_pretrained(
            model.language_model, str(adapter_path), is_trainable=True, local_files_only=True
        )
    else:
        config = LoraConfig(
            task_type="CAUSAL_LM",
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model.language_model = get_peft_model(model.language_model, config)
    if gradient_checkpointing:
        kwargs = {"gradient_checkpointing_kwargs": {"use_reentrant": False}}
        try:
            model.language_model.gradient_checkpointing_enable(**kwargs)
        except TypeError:
            model.language_model.gradient_checkpointing_enable()
        if hasattr(model.language_model, "enable_input_require_grads"):
            model.language_model.enable_input_require_grads()
    model.language_model.config.use_cache = False
    return model, tokenizer


def build_trimodal_model(internvl: nn.Module, tokenizer: Any, model_config: dict[str, Any]) -> InternVLFourModalModel:
    image_context_ids = tokenizer.encode("<IMG_CONTEXT>", add_special_tokens=False)
    if len(image_context_ids) != 1:
        raise ValueError(f"<IMG_CONTEXT> must map to one token, got {image_context_ids}")
    classifier = FourModalMoEClassifier(
        llm_dim=_language_hidden_size(internvl),
        visual_dim=int(model_config.get("visual_dim", 1024)),
        text_dim=int(model_config.get("text_dim", 4096)),
        audio_dim=int(model_config.get("audio_dim", 4096)),
        hidden_dim=int(model_config.get("hidden_dim", 256)),
        transformer_layers=int(model_config.get("transformer_layers", 2)),
        transformer_heads=int(model_config.get("transformer_heads", 8)),
        transformer_ff_dim=int(model_config.get("transformer_ff_dim", 1024)),
        expert_dim=int(model_config.get("expert_dim", 512)),
        router_dim=int(model_config.get("router_dim", 128)),
        dropout=float(model_config.get("dropout", 0.1)),
        router_temperature=float(model_config.get("router_temperature", 1.0)),
        modality_loss_weight=float(model_config.get("modality_loss_weight", 0.3)),
        llm_loss_weight=float(model_config.get("llm_loss_weight", 1.0)),
        visual_loss_weight=float(model_config.get("visual_loss_weight", 1.0)),
        text_loss_weight=float(model_config.get("text_loss_weight", 1.0)),
        audio_loss_weight=float(model_config.get("audio_loss_weight", 1.0)),
        router_loss_weight=float(model_config.get("router_loss_weight", 0.1)),
        balance_loss_weight=float(model_config.get("balance_loss_weight", 0.01)),
        oracle_temperature=float(model_config.get("oracle_temperature", 0.5)),
    )
    return InternVLFourModalModel(internvl, classifier, image_context_ids[0])


InternVLTriModalModel = InternVLFourModalModel
