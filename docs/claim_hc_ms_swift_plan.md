# Claim-Conditioned Hybrid Compressor Under `ms-swift`

This note describes the minimum viable integration path for adding a claim-aware hybrid compressor
to the current `ms-swift + InternVL-8B` video fake-news workflow without rewriting the Swift trainer.

## Goal

Keep the current training/inference entry points:

- `swift sft --config ...`
- `swift infer --config ...`

while inserting a claim-conditioned verification module between the visual projector and the LLM.

## Data Changes

The current FakeTT prompt now marks the claim span explicitly:

```text
- news claim to verify: <claim>...</claim>
```

This is needed so the template can recover a stable claim token mask.

## High-Level Architecture

```text
video -> InternVL vision encoder -> original projector/alignment
      -> Claim-Conditioned Hybrid Compressor
      -> LLM -> real/fake generation
```

The original projector/alignment stays unchanged. The HC module is inserted after the visual tokens
have already been mapped into the language hidden space.

## HC v1 Modules

### 1. General Scene Expert

- Input: projected visual tokens `V`
- Role: preserve generic scene semantics
- Implementation: lightweight FFN with residual connection

### 2. Claim-Evidence Expert

- Input: projected visual tokens `V`, claim summary `z_c`
- Role: highlight claim-relevant evidence
- Implementation: claim-conditioned cross-attention + FFN

### 3. Claim-Aware Gate

- Input: claim summary and expert summaries
- Role: produce soft routing weights for the two experts
- Implementation: attention-based gate rather than plain MLP

### 4. Auxiliary Veracity Classifier

- Input: pooled HC output + claim summary
- Output: 2-way `real/fake` logits
- Loss: `L_cls`

## Training Objective

The Swift trainer should remain unchanged. The model `forward()` should:

1. compute the normal token loss `L_token`
2. compute the auxiliary classification loss `L_cls`
3. return:

```text
L_total = L_token + lambda_cls * L_cls
```

Recommended initial weight:

```text
lambda_cls = 0.2
```

## Why This Works With Swift

Swift only requires the model to return a scalar `loss`. Therefore:

- template changes can surface extra fields such as `claim_text_mask` and `veracity_label`
- model changes can consume them internally
- trainer logic does not need to know about the extra module

## Required Template Patch

The `internvl2_5` template should be patched to provide these extra fields after `_post_encode`:

- `text_features`
- `vit_embeds`
- `selected`
- `mask`
- `claim_text_mask`
- `veracity_label`

The first four already exist in the reference InternVL customization under `external/FakeSV-VLM/utils/internvl.py`.

### `claim_text_mask`

Build a token-level mask aligned with the text-only token stream, covering the span between
`<claim>` and `</claim>`.

### `veracity_label`

Convert the final assistant string into:

- `real -> 0`
- `fake -> 1`

## Required Model Patch

Patch `modeling_internvl_chat.py` in the local InternVL model directory:

1. initialize the HC module and classifier
2. accept optional kwargs:
   - `text_features`
   - `vit_embeds`
   - `claim_text_mask`
   - `veracity_label`
   - `selected`
3. derive claim features from `text_features`
4. rewrite visual embeddings with HC output
5. add the auxiliary classification loss to the original output loss

## v1 Constraint

The first version should **not** change the number of visual tokens.

Reason:

- changing token counts would require simultaneous edits to `inputs_embeds`, `attention_mask`, and
  visual placeholder alignment
- keeping shapes fixed is much safer under `ms-swift`

So HC v1 should behave as a **verification-aware token rewriter**, not a hard token reducer.

## Future v2

After v1 is stable, add:

- Temporal Consistency Expert
- frame-level or token-level gate
- true token compression instead of token rewriting
- retrieval-aware verifier branch
