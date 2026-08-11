from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "scripts/install_claim_hc_patch.py has been retired. "
        "The project now uses the local fork at forks/claim_hc/internvl.py "
        "through scripts/run_swift_sft_with_claim_hc.py and "
        "scripts/run_swift_infer_with_claim_hc.py instead of patching site-packages."
    )


if __name__ == "__main__":
    main()

    import_block = (
        TEMPLATE_IMPORT_SENTINEL
        + "\nfrom claim_hc.runtime import apply_claim_hc, build_claim_text_mask, extract_veracity_label"
    )
    text = replace_once(text, TEMPLATE_IMPORT_SENTINEL, import_block, path)

    encode_pattern = re.compile(
        r"(class Internvl2Template\(InternvlTemplate\):.*?def _encode\(self, inputs: StdTemplateInputs\) -> Dict\[str, Any\]:.*?encoded\['pixel_values'\] = pixel_values\n)(\s*return encoded)",
        re.DOTALL,
    )
    if not encode_pattern.search(text):
        raise ValueError(f"Could not find Internvl2Template._encode block in {path}")
    text = encode_pattern.sub(
        r"\1"
        "        # videommd claim hc template patch start\n"
        "        veracity_label = extract_veracity_label(inputs)\n"
        "        if veracity_label is not None:\n"
        "            encoded['veracity_label'] = veracity_label\n"
        "        # videommd claim hc template patch end\n"
        r"\2",
        text,
        count=1,
    )

    post_pattern = re.compile(
        r"def _post_encode\(self, model: nn\.Module, inputs: Dict\[str, Any\]\) -> Dict\[str, Any\]:\n.*?\n\s*return \{'inputs_embeds': inputs_embeds\}",
        re.DOTALL,
    )
    match = post_pattern.search(text)
    if not match:
        raise ValueError(f"Could not find InternvlTemplate._post_encode block in {path}")
    post_new = """def _post_encode(self, model: nn.Module, inputs: Dict[str, Any]) -> Dict[str, Any]:
        embedding = model.get_input_embeddings()
        device = embedding.weight.device
        input_ids = inputs['input_ids']
        inputs_embeds = embedding(input_ids).to(device=device)
        text_features = inputs_embeds.clone().to(device=device)
        vit_embeds = None
        selected = torch.zeros_like(input_ids, dtype=torch.bool)
        mask = torch.ones_like(input_ids, dtype=torch.float)

        pixel_values = inputs.get('pixel_values')
        if pixel_values is not None:
            pixel_values = pixel_values.to(device=device)
            vit_embeds = model.extract_feature(pixel_values).to(device=device)
            selected = (input_ids == self.processor.encode('<IMG_CONTEXT>', add_special_tokens=False)[0]).to(device=device)
            mask = torch.where(selected, torch.zeros_like(selected, dtype=torch.float),
                               torch.ones_like(selected, dtype=torch.float))

            text_features = text_features[~selected]
            text_features = text_features.reshape(inputs_embeds.shape[0], -1, text_features.shape[-1])
            vit_embeds = vit_embeds.reshape(-1, vit_embeds.shape[-1])
            inputs_embeds[selected] = vit_embeds.to(dtype=inputs_embeds.dtype)
            vit_embeds = vit_embeds.reshape(inputs_embeds.shape[0], -1, vit_embeds.shape[-1])
        elif is_deepspeed_enabled():
            dummy_pixel_values = torch.zeros((1, 3, 32, 32), device=device, dtype=inputs_embeds.dtype)
            vit_embeds = model.extract_feature(dummy_pixel_values).to(device=device)
            inputs_embeds += vit_embeds.mean() * 0.

        claim_text_mask = build_claim_text_mask(input_ids, selected, self.processor)
        inputs_embeds, hc_aux_loss = apply_claim_hc(
            model=model,
            inputs_embeds=inputs_embeds,
            text_features=text_features,
            vit_embeds=vit_embeds,
            selected=selected,
            claim_text_mask=claim_text_mask,
            veracity_label=inputs.get('veracity_label'),
        )
        return {'inputs_embeds': inputs_embeds, 'text_features': text_features,
                'vit_embeds': vit_embeds, 'mask': mask, 'selected': selected,
                'claim_text_mask': claim_text_mask, 'hc_aux_loss': hc_aux_loss,
                'veracity_label': inputs.get('veracity_label')}"""
    text = text[:match.start()] + post_new + text[match.end():]

    path.write_text(text, encoding="utf-8")
    print(f"[ok] patched template file: {path}")


def patch_modeling_file(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    ensure_contains(text, MODEL_FORWARD_SENTINEL, path)
    ensure_contains(text, MODEL_RETURN_SENTINEL, path)

    # Repair the broken early patch form if it exists.
    text = text.replace("def forward(\n        self,\n        hc_aux_loss=None,\n", "def forward(\n        self,\n", 1)

    signature_pattern = re.compile(r"def forward\((.*?)\)\s*->", re.DOTALL)
    match = signature_pattern.search(text)
    if not match:
        raise ValueError(f"Could not find forward signature block in {path}")

    signature = match.group(1)
    signature = re.sub(r"\n\s*hc_aux_loss\s*=\s*None,\s*", "\n", signature)
    if "return_dict" in signature:
        signature = signature.replace("return_dict", "hc_aux_loss=None,\n        return_dict", 1)
    else:
        signature = signature.rstrip() + "\n        hc_aux_loss=None,\n"
    text = text[:match.start(1)] + signature + text[match.end(1):]

    if "videommd claim hc model patch start" not in text:
        text = replace_once(
            text,
            MODEL_RETURN_SENTINEL,
            "# videommd claim hc model patch start\n"
            "        if hc_aux_loss is not None and loss is not None:\n"
            "            loss = loss + hc_aux_loss\n"
            "        # videommd claim hc model patch end\n"
            "        return CausalLMOutputWithPast(",
            path,
        )

    path.write_text(text, encoding="utf-8")
    print(f"[ok] patched modeling file: {path}")


def main() -> None:
    args = parse_args()
    patch_template_file(args.swift_template_file)
    patch_modeling_file(args.modeling_file)
    for extra_path in args.extra_modeling_file:
        patch_modeling_file(extra_path)


if __name__ == "__main__":
    main()
