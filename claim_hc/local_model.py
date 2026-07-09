from __future__ import annotations

from types import MethodType
from typing import Any, Optional

import torch
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithPast


HC_HELPER_KEYS = {
    "text_features",
    "vit_embeds",
    "mask",
    "selected",
    "claim_text_mask",
    "veracity_label",
}


def _get_base_model(model: Any) -> Any:
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


def patch_internvl_chat_model(model: Any) -> Any:
    target_model = _get_base_model(model)
    if getattr(target_model, "_videommd_claim_hc_runtime_patched", False):
        return model

    original_forward = target_model.forward
    original_generate = target_model.generate

    def patched_forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        hc_aux_loss=None,
        **kwargs,
    ):
        for key in HC_HELPER_KEYS:
            kwargs.pop(key, None)

        if inputs_embeds is None or not hasattr(self, "language_model"):
            outputs = original_forward(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                image_flags=image_flags,
                past_key_values=past_key_values,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )
            if hc_aux_loss is not None:
                if isinstance(outputs, tuple):
                    loss = outputs[0]
                    if loss is not None:
                        outputs = (loss + hc_aux_loss,) + outputs[1:]
                elif getattr(outputs, "loss", None) is not None:
                    outputs.loss = outputs.loss + hc_aux_loss
            return outputs

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if hc_aux_loss is not None and loss is not None:
            loss = loss + hc_aux_loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def patched_generate(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        visual_features: Optional[torch.FloatTensor] = None,
        generation_config=None,
        output_hidden_states: Optional[bool] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **generate_kwargs,
    ):
        for key in list(HC_HELPER_KEYS) + ["hc_aux_loss", "labels"]:
            generate_kwargs.pop(key, None)

        if inputs_embeds is not None and hasattr(self, "language_model"):
            return self.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                generation_config=generation_config,
                output_hidden_states=output_hidden_states,
                use_cache=True,
                **generate_kwargs,
            )

        return original_generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_features=visual_features,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            **generate_kwargs,
        )

    target_model.forward = MethodType(patched_forward, target_model)
    target_model.generate = MethodType(patched_generate, target_model)
    target_model._videommd_claim_hc_runtime_patched = True
    return model
