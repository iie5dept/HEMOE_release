from __future__ import annotations

import os

from swift.arguments.base_args.base_args import BaseArguments

from .local_model import patch_internvl_chat_model
from .runtime import ensure_claim_hc, get_base_model


_PATCH_INSTALLED = False


def install_claim_hc_runtime_patch() -> None:
    global _PATCH_INSTALLED
    if _PATCH_INSTALLED:
        return

    original_get_model_processor = BaseArguments.get_model_processor

    def patched_get_model_processor(self, *args, **kwargs):
        model, processor = original_get_model_processor(self, *args, **kwargs)
        target_model = get_base_model(model)
        language_model = getattr(target_model, "language_model", None)
        hidden_size = getattr(getattr(language_model, "config", None), "hidden_size", None)
        if hidden_size is None:
            input_embeddings = target_model.get_input_embeddings()
            hidden_size = input_embeddings.weight.shape[-1]
        dtype = getattr(getattr(self, "model_info", None), "torch_dtype", None)
        if dtype is None:
            dtype = target_model.get_input_embeddings().weight.dtype
        device = target_model.get_input_embeddings().weight.device
        ensure_claim_hc(model, hidden_size=int(hidden_size), dtype=dtype, device=device)
        if os.environ.get("CLAIM_HC_PRINT_MODULES", "").strip().lower() in {"1", "true", "yes"}:
            names = [name for name, _ in target_model.named_parameters() if "claim_hc" in name]
            print(f"[claim_hc] initialized parameters: {names[:20]} total={len(names)}", flush=True)
        patch_internvl_chat_model(model)
        return model, processor

    BaseArguments.get_model_processor = patched_get_model_processor
    _PATCH_INSTALLED = True
