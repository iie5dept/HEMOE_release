from __future__ import annotations

from swift.arguments.base_args.base_args import BaseArguments

from .local_model import patch_internvl_chat_model


_PATCH_INSTALLED = False


def install_claim_hc_runtime_patch() -> None:
    global _PATCH_INSTALLED
    if _PATCH_INSTALLED:
        return

    original_get_model_processor = BaseArguments.get_model_processor

    def patched_get_model_processor(self, *args, **kwargs):
        model, processor = original_get_model_processor(self, *args, **kwargs)
        patch_internvl_chat_model(model)
        return model, processor

    BaseArguments.get_model_processor = patched_get_model_processor
    _PATCH_INSTALLED = True
