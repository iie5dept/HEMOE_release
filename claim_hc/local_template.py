from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def register_local_internvl_template(template_file: str | Path) -> ModuleType:
    template_path = Path(template_file).resolve()
    if not template_path.exists():
        raise FileNotFoundError(f"Local InternVL template file not found: {template_path}")

    module_name = "videommd_claim_hc_internvl"
    spec = importlib.util.spec_from_file_location(module_name, template_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create import spec for: {template_path}")

    import swift.template.register as template_register

    original_register_template = template_register.register_template

    def overwrite_register_template(template_meta, *args, **kwargs):
        try:
            return original_register_template(template_meta, *args, **kwargs)
        except ValueError as error:
            template_type = getattr(template_meta, "template_type", None)
            mapping = getattr(template_register, "TEMPLATE_MAPPING", None)
            if template_type is None or not isinstance(mapping, dict):
                raise error
            mapping.pop(template_type, None)
            return original_register_template(template_meta, *args, **kwargs)

    template_register.register_template = overwrite_register_template

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        template_register.register_template = original_register_template
    return module
