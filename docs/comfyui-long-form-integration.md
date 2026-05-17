# ComfyUI long-form Prompt Relay integration

This note uses the uploaded LTX 2.3 AV workflow fixture as the starting point:

`/Users/sarpdoven/.openclaw/media/inbound/112344e7-4947-451d-81f3-62bf51dac3a5.json`

The fixture is a two-pass ComfyUI API prompt. Both passes currently use `PromptRelayEncode`:

- **Pass 1**: node `605` patches the base LTX model before `LTX2SamplingPreviewOverride` node `588`, then feeds `LTXVConditioning` node `164` and `SamplerCustom` node `561`.
- **Pass 2/refine**: node `610` patches the same base model before `LTX2SamplingPreviewOverride` node `614`, then feeds `LTXVConditioning` node `612` and `SamplerCustom` node `615`.
- The fixture length is driven by node `624` and is currently `217` frames, not a 2500+ frame single generation.

## Where Prompt Relay belongs

Prompt Relay should remain between the loaded/patched model and any downstream sampler wrapper that consumes the model:

```text
UNET / LoRA / SageAttention
  -> PromptRelay encoder
  -> LTX2SamplingPreviewOverride or other sampler wrapper
  -> sampler
```

The positive conditioning output from the same Prompt Relay node should feed the matching `LTXVConditioning` positive input. The negative conditioning should stay as a normal `CLIPTextEncode` path.

Do **not** place Prompt Relay after the sampler, after decode, or only on CLIP conditioning. It must patch the model used by the sampler for that pass.

## Main vs Lab node placement

Recommended workflow split:

1. **Production/default path**: use `PromptRelayEncode` plus an optional `PromptRelayAdvancedOptions` node.
   - This keeps the main workflow aligned with the improved custom node.
   - In the fixture, node `605` should stay on the main node for first-pass generation.
2. **A/B or tuning path**: use `PromptRelayLabEncode` only on a duplicate/refine branch or copied workflow.
   - The lab node exposes the same core behavior plus explicit sigma controls for controlled tests.
   - In the fixture, node `610` is a good place to test the lab node because it is already the second pass. Keep pass 1 stable, then compare pass 2 with lab settings.

When the improved main node exposes `sigma_mode` through `PromptRelayAdvancedOptions`, prefer using the main node for both passes and reserve the lab node for experiments.

## Sane advanced options for long-form quality

Starting preset for long-form chunk tests:

```text
epsilon: 0.1 to 0.3
sigma_mode: paper_boundary
video_strength: 0.7 to 1.0
video_window_scale: 1.0 to 1.3
audio_epsilon: 0.1 or inherit
audio_strength: 0.6 to 1.0
audio_window_scale: 1.0 to 1.3
audio_frame_offset_frames: 0 unless measured audio/text lag exists
```

Notes:

- `epsilon=1e-3` with upstream-compatible sigma is very sharp and often behaves like hard segment gating. Good for strict prompt isolation, but it can create visible semantic snaps.
- `paper_boundary` makes `epsilon` mean retained attention at the segment endpoint. This is easier to reason about for long shots.
- `video_strength > 1` usually saturates quickly at low epsilon. If stronger values appear to do nothing, raise epsilon first.
- `video_window_scale=0` is not softer. It collapses the free window to the midpoint and makes falloff start immediately.

## Chunk/window strategy for 2500+ frames

A single 2500+ frame latent run is not the right integration target. Use Prompt Relay inside each chunk, then maintain continuity across chunks.

Recommended structure:

```text
script beats / transcript
  -> chunk plan with overlap
  -> per-chunk Prompt Relay local prompts
  -> generate chunk N
  -> carry tail keyframe/context into chunk N+1
  -> trim/fade/compose overlaps
```

Practical starting values:

- Chunk size: **129 to 257 video frames** for LTX 2.3 AV experiments.
- Overlap: **16 to 32 frames** at chunk boundaries.
- Segments per chunk: **2 to 6**. Avoid hundreds of tiny segments.

The helper `prompt_relay.plan_temporal_chunks(latent_frames, tokens_per_frame, text_tokens, ...)` gives a hard upper bound from the additive-mask memory budget. It does not generate or stitch video; it answers: "how large can a per-chunk Prompt Relay window be before the `[query_tokens, text_tokens]` mask exceeds `PROMPT_RELAY_MAX_MASK_ELEMENTS`?" Use the smaller of that cap and the quality-oriented chunk size above.

For scheduler/UI code, `prompt_relay.plan_quality_temporal_chunks(..., target_chunk_frames=257, overlap_frames=16)` does that smaller-of-budget-and-quality-target choice directly. It reports both `budget_max_chunk_frames` and `quality_target_frames`, so Comfy logs can show whether chunk size was limited by VRAM budget or by the long-form quality preset.

The main ComfyUI node now exposes this planner through `PromptRelayAdvancedOptions`:

```text
long_form_mode: quality_plan
quality_target_chunk_frames: 129 to 257
chunk_overlap_frames: 16 to 32
chunk_safety_margin: 0.8 to 0.9
```

Connect that options node to `PromptRelayEncode` or `PromptRelayEncodeTimeline`. The encoder keeps its normal model/conditioning outputs, and the Comfy console logs a bounded chunk plan plus a preflight summary for the current latent size. This is workflow guidance for the scheduler/stitching layer; the encoder does not magically split a sampler run by itself.

After planning, `prompt_relay.plan_chunk_stitch_ranges(plan["chunks"])` gives deterministic keep/trim ranges for the overlap. It splits each overlap at its midpoint so final kept frames are non-overlapping, while the discarded head/tail frames remain available for crossfade diagnostics or visual seam checks.

Use `prompt_relay.plan_chunk_handoffs(plan["chunks"], min_context_frames=...)` before rendering to flag hard cuts or short overlaps. `prompt_relay.format_chunk_handoff_diagnostics(...)` turns those seams into a bounded one-line log summary so long 2500+ frame plans can surface weak boundaries without flooding the Comfy console.

Example: with `tokens_per_frame=4096`, `text_tokens=128`, and the default cap, the absolute memory-bounded maximum is 2048 latent frames at `safety_margin=1.0`, so a 2501-frame latent timeline plans as `[0,2048)` and `[2032,2501)` with 16-frame overlap. The stitch helper keeps `[0,2040)` from chunk 0 and `[2040,2501)` from chunk 1. For quality, still prefer much shorter chunks unless a controlled render proves the model holds identity and motion over longer windows.
- Timing input: use percentages/ratios for per-chunk authoring (`35%,65%` or `0.35,0.65`) unless exact frame counts are known.
- Global prompt: stable identity, wardrobe, environment, camera style, lighting, and audio bed. Keep it consistent across all chunks.
- Local prompts: only what changes in that chunk: action, gaze, hands, product motion, spoken phrase, audio event.

For the uploaded fixture, the immediate conversion is:

- Keep the existing global prompt as the chunk-level identity anchor.
- Split the two local prompts by spoken phrase timing instead of leaving `segment_lengths` empty. Example: `45%,55%` for the current two beats, then tune from actual transcript timing.
- For later 2500+ frame workflows, create many chunk prompts rather than one giant Prompt Relay node.

## What Prompt Relay cannot solve alone

Prompt Relay is an attention routing mechanism. It cannot by itself fix:

- OOM and quality collapse from a single full 2500+ frame attention pass.
- Long-range identity drift across independently generated chunks.
- Physical continuity across cuts without carried image/latent/keyframe context.
- Lip-sync quality if transcript timing and audio conditioning disagree.
- Model limitations around hands, text, object permanence, or camera geometry.
- Flicker caused by decode, interpolation, compression, or inconsistent init frames.

For 2500+ frames, Prompt Relay should be one component in a chunked generation pipeline with explicit continuity constraints.

## Diagnostics and log verification

Before judging video quality, verify the node is actually patching the intended pass.

Expected log lines from the main node:

```text
[PromptRelay] Global: tokens ...
[PromptRelay] Segment 0: tokens ...
[PromptRelay] Latent: <N> frames, <tokens/frame> tokens/frame, segments: [...]
[PromptRelay] Setup: arch=... latent_frames=<N> tokens_per_frame=... timing='...' epsilon=... options=... | seg0: ...
[PromptRelay] Long-form plan: chunk_plan: ...
[PromptRelay] Long-form preflight: chunk_eval: ...
```

Expected log lines from the lab node:

```text
[PromptRelayLab] arch=... latent_frames=<N> tokens_per_frame=... timing='...' segment_lengths=[...] sigma_mode=...
```

Checklist:

1. `latent_frames` matches the chunk latent length, not the final stitched 2500+ frame target.
2. `segments` length matches the number of local prompts.
3. `timing` is not accidentally `auto-even` when transcript timing was intended.
4. `options` includes the expected `sigma_mode`, strengths, and audio settings.
5. For LTX AV, logs show both video and scaled/audio routing paths being exercised in tests or diagnostics.
6. Run a one-chunk A/B before running a full batch:
   - baseline main node, upstream-compatible options
   - main node with `PromptRelayAdvancedOptions` paper-boundary preset
   - optional lab node with matching parameters

ComfyUI import validation:

1. Install the custom node folder, restart ComfyUI, then confirm these node types appear in the node search: `PromptRelayEncode`, `PromptRelayAdvancedOptions`, and, only for experiments, `PromptRelayLabEncode`.
2. Load `example_workflows/ltx23_long_form_prompt_relay_integration.api.json` through the API prompt loader. Missing-node errors mean the custom node is not on ComfyUI's custom-node path or ComfyUI needs a restart.
3. Before queueing, inspect the graph wiring:
   - node `605` model output feeds node `588`; node `605` conditioning feeds node `164` positive.
   - node `610` model output feeds node `614`; node `610` conditioning feeds node `612` positive.
   - node `625` feeds node `605` `relay_options`.
4. Queue a low-resolution or short-frame copy first. Treat successful import as only a wiring check; judge quality only after the expected Prompt Relay log lines appear.

## Workflow artifact

A derived ComfyUI API prompt variant is provided at:

`example_workflows/ltx23_long_form_prompt_relay_integration.api.json`

It does not overwrite the uploaded fixture. It demonstrates:

- Main node on pass 1 with `PromptRelayAdvancedOptions` connected.
- Lab node on pass 2 for A/B testing.
- Explicit two-segment timing (`45%,55%`) instead of empty even timing.

Use it as an integration sketch, not as a final quality preset.
