from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import torch
from torch import Tensor, nn

from .modules import NativeEvidenceMoE, gather_tokens_by_mask

_PATCHED = False
_ADAPTER_ENV = "VIDEOMMD_CLAIM_HC_ADAPTERS"


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _get_stage() -> str:
    return os.environ.get("VIDEOMMD_CLAIM_HC_STAGE", "phase4").strip().lower()


def _extract_attr_or_key(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _unwrap_model(model: nn.Module) -> nn.Module:
    module = model
    for attr in ("module",):
        if hasattr(module, attr):
            module = getattr(module, attr)
    base_model = getattr(module, "base_model", None)
    if base_model is not None:
        nested = getattr(base_model, "model", None)
        if nested is not None:
            module = nested
        else:
            module = base_model
    return module


def _normalize_adapter_paths(adapters: Any) -> list[Path]:
    if adapters is None:
        return []
    if isinstance(adapters, (str, Path)):
        raw_values = [str(adapters)]
    elif isinstance(adapters, (list, tuple, set)):
        raw_values = [str(item) for item in adapters if item]
    else:
        raw_values = [str(adapters)]

    paths: list[Path] = []
    for raw_value in raw_values:
        for piece in raw_value.split(","):
            candidate = piece.strip()
            if candidate:
                paths.append(Path(candidate))
    return paths


def _load_adapter_state_dict(adapter_path: Path) -> dict[str, Tensor]:
    safetensor_path = adapter_path / "adapter_model.safetensors"
    if safetensor_path.is_file():
        from safetensors.torch import load_file

        return load_file(str(safetensor_path), device="cpu")

    bin_path = adapter_path / "adapter_model.bin"
    if bin_path.is_file():
        state = torch.load(str(bin_path), map_location="cpu")
        if isinstance(state, dict):
            return state
        raise TypeError(f"Unexpected adapter checkpoint payload type: {type(state).__name__}")

    raise FileNotFoundError(
        f"No adapter weights found under {adapter_path}. "
        "Expected adapter_model.safetensors or adapter_model.bin."
    )


def _extract_claim_hc_state_dict(adapter_path: Path) -> dict[str, Tensor]:
    state = _load_adapter_state_dict(adapter_path)
    claim_hc_state: dict[str, Tensor] = {}
    for key, value in state.items():
        marker = "claim_hc."
        index = key.find(marker)
        if index < 0:
            continue
        claim_hc_state[key[index + len(marker):]] = value
    return claim_hc_state


def _maybe_restore_claim_hc_from_adapters(model: nn.Module, adapters: Any) -> Path | None:
    adapter_paths = _normalize_adapter_paths(adapters)
    if not adapter_paths:
        adapter_paths = _normalize_adapter_paths(os.environ.get(_ADAPTER_ENV))
    if not adapter_paths:
        return None

    ensure_claim_hc(model)
    target_model = _unwrap_model(model)
    container = getattr(target_model, "claim_hc", None)
    if container is None:
        return None
    if getattr(container, "modules_to_save", None) is not None:
        return None

    for adapter_path in adapter_paths:
        if not adapter_path.is_dir():
            continue
        try:
            claim_hc_state = _extract_claim_hc_state_dict(adapter_path)
        except FileNotFoundError:
            continue
        if not claim_hc_state:
            continue

        module = _resolve_claim_hc_module(container, freeze_inactive=False)
        incompatible = module.load_state_dict(claim_hc_state, strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(
                "Unexpected claim_hc keys while restoring adapter "
                f"{adapter_path}: {', '.join(incompatible.unexpected_keys[:8])}"
            )
        if incompatible.missing_keys:
            raise RuntimeError(
                "Missing claim_hc keys while restoring adapter "
                f"{adapter_path}: {', '.join(incompatible.missing_keys[:8])}"
            )

        loaded_path = str(adapter_path)
        setattr(module, "_videommd_loaded_from_adapter", loaded_path)
        target_model._videommd_claim_hc_loaded_from_adapter = loaded_path
        if os.environ.get("RANK", "0") == "0":
            print(
                f"[videommd] restored claim_hc from adapter checkpoint: "
                f"{adapter_path} ({len(claim_hc_state)} tensors)"
            )
        return adapter_path
    return None


def _get_language_hidden_size(target_model: nn.Module) -> int:
    candidates = [
        getattr(getattr(target_model, "language_model", None), "config", None),
        getattr(target_model, "config", None),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        for field in ("hidden_size", "text_hidden_size", "llm_hidden_size"):
            value = getattr(candidate, field, None)
            if isinstance(value, int) and value > 0:
                return value
    q_proj = target_model.language_model.model.layers[0].self_attn.q_proj
    return int(q_proj.in_features)


def _set_trainable(module: nn.Module, enabled: bool) -> None:
    for param in module.parameters():
        param.requires_grad = enabled


def _set_parameter_trainable(param: Tensor, enabled: bool) -> None:
    param.requires_grad = enabled


def _active_adapter_name(wrapper: nn.Module, available: list[str]) -> str:
    active = getattr(wrapper, "active_adapter", None)
    if isinstance(active, str) and active in available:
        return active
    if isinstance(active, (list, tuple)):
        for name in active:
            if name in available:
                return name
    if "default" in available:
        return "default"
    if not available:
        raise RuntimeError("The modules_to_save wrapper has no saved claim_hc module.")
    return available[0]


def _resolve_claim_hc_module(container: nn.Module, freeze_inactive: bool = False) -> NativeEvidenceMoE:
    if isinstance(container, NativeEvidenceMoE):
        return container

    saved_modules = getattr(container, "modules_to_save", None)
    if saved_modules is None or not hasattr(saved_modules, "keys"):
        raise TypeError(
            f"Unsupported claim_hc container: {type(container).__name__}. "
            "Expected NativeEvidenceMoE or a PEFT ModulesToSaveWrapper."
        )

    available = list(saved_modules.keys())
    active_name = _active_adapter_name(container, available)
    if freeze_inactive:
        original_module = getattr(container, "original_module", None)
        if isinstance(original_module, nn.Module):
            _set_trainable(original_module, False)
        for name, saved_module in saved_modules.items():
            if name != active_name:
                _set_trainable(saved_module, False)

    active_module = saved_modules[active_name]
    if not isinstance(active_module, NativeEvidenceMoE):
        raise TypeError(
            f"Active modules_to_save entry '{active_name}' is {type(active_module).__name__}, "
            "not NativeEvidenceMoE."
        )
    return active_module


def _configure_stage(container: nn.Module, stage: str) -> NativeEvidenceMoE:
    module = _resolve_claim_hc_module(container, freeze_inactive=True)
    _set_trainable(module, False)

    phase1_modules = (
        module.visual_scorer,
        module.text_scorer,
        module.cross_adapter,
        module.shared_expert,
    )
    phase2_modules = phase1_modules + (
        module.visual_expert,
        module.text_expert,
        module.cross_expert,
    )
    if stage == "phase1":
        trainable_modules = phase1_modules
        train_scales = True
    elif stage == "phase2":
        trainable_modules = phase2_modules
        train_scales = True
    elif stage == "phase3":
        trainable_modules = (module.coarse_router, module.expert_router)
        train_scales = False
    elif stage == "phase4":
        trainable_modules = tuple(module.children())
        train_scales = True
    else:
        raise ValueError(f"Unsupported VIDEOMMD_CLAIM_HC_STAGE: {stage!r}")

    for child in trainable_modules:
        _set_trainable(child, True)
    _set_parameter_trainable(module.visual_token_scale, train_scales)
    _set_parameter_trainable(module.text_token_scale, train_scales)
    module._videommd_configured_stage = stage
    return module


def ensure_claim_hc(model: nn.Module) -> None:
    target_model = _unwrap_model(model)
    if hasattr(target_model, "claim_hc"):
        return
    if not _env_flag("VIDEOMMD_CLAIM_HC_ENABLE", False):
        return

    hidden_size = _get_language_hidden_size(target_model)
    module = NativeEvidenceMoE(
        hidden_size=hidden_size,
        expert_rank=_env_int("VIDEOMMD_CLAIM_HC_EXPERT_RANK", 8),
        top_k=_env_int("VIDEOMMD_CLAIM_HC_TOP_K", 2),
    )
    device = target_model.get_input_embeddings().weight.device
    dtype = target_model.get_input_embeddings().weight.dtype
    module = module.to(device=device, dtype=dtype)
    target_model.claim_hc = module
    target_model._claim_hc_aux_losses = None


def configure_claim_hc(model: nn.Module) -> NativeEvidenceMoE | None:
    ensure_claim_hc(model)
    target_model = _unwrap_model(model)
    container = getattr(target_model, "claim_hc", None)
    if container is None:
        return None
    return _configure_stage(container, _get_stage())


def apply_claim_hc(
    model: nn.Module,
    inputs_embeds: Tensor,
    input_ids: Tensor | None = None,
    selected: Tensor | None = None,
    attention_mask: Tensor | None = None,
    labels: Tensor | None = None,
    **legacy_kwargs,
) -> tuple[Tensor, Dict[str, Tensor]]:
    if input_ids is None or selected is None:
        legacy_names = ", ".join(sorted(legacy_kwargs)) or "legacy arguments"
        raise RuntimeError(
            "The obsolete site-packages InternVL patch called claim_hc with "
            f"{legacy_names}. Restore the official swift internvl.py and launch through "
            "scripts/run_swift_*_with_claim_hc.py."
        )
    ensure_claim_hc(model)
    target_model = _unwrap_model(model)
    container = getattr(target_model, "claim_hc", None)
    if container is None:
        return inputs_embeds, {}

    if _env_flag("VIDEOMMD_CLAIM_HC_REQUIRE_ADAPTER", False):
        restored = getattr(target_model, "_videommd_claim_hc_loaded_from_adapter", None)
        if getattr(container, "modules_to_save", None) is None and restored is None:
            _maybe_restore_claim_hc_from_adapters(model, None)
            restored = getattr(target_model, "_videommd_claim_hc_loaded_from_adapter", None)
            if restored is None:
                configured = os.environ.get(_ADAPTER_ENV, "<not recorded>")
                raise RuntimeError(
                    "The requested adapter did not restore the claim_hc module. "
                    f"Recorded adapter path: {configured}. "
                    "Refusing to run inference with randomly initialized evidence weights."
                )
    claim_hc = _resolve_claim_hc_module(container, freeze_inactive=False)
    first_parameter = next(claim_hc.parameters())
    if first_parameter.device != inputs_embeds.device or first_parameter.dtype != inputs_embeds.dtype:
        if target_model.training:
            raise RuntimeError(
                "claim_hc was placed on a different device or dtype after DDP initialization. "
                "The module must be attached before optimizer/DDP construction."
            )
        claim_hc.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)

    selected = selected.to(device=inputs_embeds.device, dtype=torch.bool)
    text_mask = ~selected
    if labels is not None:
        text_mask = text_mask & (labels.to(inputs_embeds.device) == -100)
    if attention_mask is not None:
        text_mask = text_mask & attention_mask.to(inputs_embeds.device).bool()
    visual_mask = selected

    visual_tokens, visual_token_mask = gather_tokens_by_mask(inputs_embeds, visual_mask)
    text_tokens, text_token_mask = gather_tokens_by_mask(inputs_embeds, text_mask)
    invalid_visual = ~visual_token_mask.any(dim=1)
    invalid_text = ~text_token_mask.any(dim=1)
    if invalid_visual.any() or invalid_text.any():
        bad_rows = (invalid_visual | invalid_text).nonzero(as_tuple=False).flatten().tolist()
        raise RuntimeError(
            f"claim_hc received samples without visual or prompt tokens at batch rows {bad_rows}. "
            "Check template registration and media decoding instead of silently skipping the module."
        )

    output = claim_hc(
        visual_tokens=visual_tokens,
        visual_mask=visual_token_mask,
        text_tokens=text_tokens,
        text_mask=text_token_mask,
        stage=_get_stage(),
    )

    visual_delta = visual_mask.unsqueeze(-1).to(inputs_embeds.dtype) * output.visual_update.unsqueeze(1)
    text_delta = text_mask.unsqueeze(-1).to(inputs_embeds.dtype) * output.text_update.unsqueeze(1)
    updated = inputs_embeds + visual_delta + text_delta

    aux_losses = dict(output.aux_losses)
    target_model._claim_hc_aux_losses = aux_losses
    return updated, aux_losses


def _add_aux_loss(base_loss: Any, aux_loss: Tensor) -> Any:
    if isinstance(base_loss, tuple):
        return (base_loss[0] + aux_loss, *base_loss[1:])
    if isinstance(base_loss, list):
        return [base_loss[0] + aux_loss, *base_loss[1:]]
    return base_loss + aux_loss


def _sum_aux_losses(model: nn.Module) -> Tensor | None:
    target_model = _unwrap_model(model)
    aux = getattr(target_model, "_claim_hc_aux_losses", None)
    if not aux:
        return None
    balance_weight = _env_float("VIDEOMMD_CLAIM_HC_BALANCE_LOSS", 0.01)
    total = None
    for name, weight in {"balance_loss": balance_weight}.items():
        if weight <= 0:
            continue
        value = aux.get(name)
        if value is None:
            continue
        term = value * weight
        total = term if total is None else total + term
    target_model._claim_hc_aux_losses = None
    return total


def build_claim_text_mask(input_ids: Tensor, selected: Tensor, processor: Any | None = None) -> Tensor:
    """Backward-compatible helper for older patched swift templates.

    The current project version uses all non-visual text tokens rather than
    extracting a narrow <claim> span, so this function returns a compact text
    mask aligned with text_features after visual tokens are removed.
    """
    del processor
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if selected.dim() == 1:
        selected = selected.unsqueeze(0)
    text_mask = (~selected.bool())
    lengths = text_mask.sum(dim=1)
    max_len = int(lengths.max().item()) if lengths.numel() > 0 else 0
    compact_mask = torch.zeros((text_mask.shape[0], max_len), dtype=torch.bool, device=text_mask.device)
    for idx, length in enumerate(lengths.tolist()):
        if length > 0:
            compact_mask[idx, :length] = True
    return compact_mask


def extract_veracity_label(inputs: Any) -> Any:
    """Backward-compatible helper for older patched swift templates."""
    direct = _extract_attr_or_key(inputs, "veracity_label", None)
    if direct is not None:
        return direct

    messages = _extract_attr_or_key(inputs, "messages", None)
    if not messages:
        return None
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        if message.get("role") != "assistant":
            continue
        content = str(message.get("content", "")).strip().lower()
        if content == "real":
            return 0
        if content == "fake":
            return 1
    return None


def install_claim_hc_runtime() -> None:
    global _PATCHED
    if _PATCHED:
        return

    from swift.arguments.base_args.base_args import BaseArguments
    from swift.trainers.seq2seq_trainer import Seq2SeqTrainer

    original_get_model_processor = BaseArguments.get_model_processor
    original_trainer_init = Seq2SeqTrainer.__init__
    original_create_optimizer = Seq2SeqTrainer.create_optimizer
    original_wrap_model = Seq2SeqTrainer._wrap_model
    original_compute_loss = Seq2SeqTrainer.compute_loss
    original_training_step = Seq2SeqTrainer.training_step

    def patched_get_model_processor(self, *args, **kwargs):
        model, processor = original_get_model_processor(self, *args, **kwargs)
        _maybe_restore_claim_hc_from_adapters(model, getattr(self, "adapters", None))
        configure_claim_hc(model)
        return model, processor

    def patched_trainer_init(self, *args, **kwargs):
        model = kwargs.get("model")
        if model is None and args and isinstance(args[0], nn.Module):
            model = args[0]
        if model is not None:
            configure_claim_hc(model)
        original_trainer_init(self, *args, **kwargs)

    def patched_create_optimizer(self, *args, **kwargs):
        module = configure_claim_hc(self.model)
        if module is not None and os.environ.get("RANK", "0") == "0":
            total = sum(param.numel() for param in module.parameters())
            trainable = sum(param.numel() for param in module.parameters() if param.requires_grad)
            print(
                f"[videommd] claim_hc stage={_get_stage()} "
                f"parameters={total:,} trainable={trainable:,}"
            )
        return original_create_optimizer(self, *args, **kwargs)

    def patched_wrap_model(self, model, *args, **kwargs):
        # This is the final point before Accelerate constructs DDP's reducer.
        # PEFT may reactivate all modules_to_save parameters after Trainer init,
        # so stage-specific freezing must be repeated here.
        module = configure_claim_hc(model)
        target_model = _unwrap_model(model)
        container = getattr(target_model, "claim_hc", None)
        original_module = getattr(container, "original_module", None)
        if isinstance(original_module, nn.Module):
            leaked = [name for name, param in original_module.named_parameters() if param.requires_grad]
            if leaked:
                raise RuntimeError(
                    "PEFT original_module still has trainable claim_hc parameters before DDP: "
                    + ", ".join(leaked[:8])
                )
        if module is not None and os.environ.get("RANK", "0") == "0":
            trainable_tensors = sum(1 for param in module.parameters() if param.requires_grad)
            print(
                f"[videommd] DDP-ready claim_hc stage={_get_stage()} "
                f"trainable_tensors={trainable_tensors}"
            )
        return original_wrap_model(self, model, *args, **kwargs)

    def patched_compute_loss(self, model, inputs, *args, **kwargs):
        result = original_compute_loss(self, model, inputs, *args, **kwargs)
        aux_loss = _sum_aux_losses(model)
        if aux_loss is None or not self.model.training:
            return result
        return _add_aux_loss(result, aux_loss)

    def patched_training_step(self, model, inputs, *args, **kwargs):
        if getattr(self, "_videommd_grad_audited", False):
            return original_training_step(self, model, inputs, *args, **kwargs)

        target_model = _unwrap_model(model)
        container = getattr(target_model, "claim_hc", None)
        if container is None:
            self._videommd_grad_audited = True
            return original_training_step(self, model, inputs, *args, **kwargs)

        module = _resolve_claim_hc_module(container, freeze_inactive=False)
        seen: set[str] = set()
        handles = []
        audited_names = []
        for name, parameter in module.named_parameters():
            if not parameter.requires_grad:
                continue
            audited_names.append(name)

            def mark_gradient(gradient, parameter_name=name):
                seen.add(parameter_name)
                return gradient

            handles.append(parameter.register_hook(mark_gradient))
        try:
            result = original_training_step(self, model, inputs, *args, **kwargs)
        finally:
            for handle in handles:
                handle.remove()

        missing = sorted(set(audited_names) - seen)
        if missing:
            raise RuntimeError(
                "Active claim_hc parameters are outside the training loss graph: "
                + ", ".join(missing[:12])
            )
        self._videommd_grad_audited = True
        if os.environ.get("RANK", "0") == "0":
            print(f"[videommd] claim_hc gradient audit passed: {len(seen)}/{len(audited_names)} tensors")
        return result

    BaseArguments.get_model_processor = patched_get_model_processor
    Seq2SeqTrainer.__init__ = patched_trainer_init
    Seq2SeqTrainer.create_optimizer = patched_create_optimizer
    Seq2SeqTrainer._wrap_model = patched_wrap_model
    Seq2SeqTrainer.compute_loss = patched_compute_loss
    Seq2SeqTrainer.training_step = patched_training_step
    _PATCHED = True
