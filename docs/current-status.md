# Prompt Relay quality lab current status

_Last updated: 2026-05-15 PDT_

This repo is currently usable as a local ComfyUI custom node checkout. The code-side goal is complete for the current pass: Prompt Relay behavior is testable, the main node has safer timing/options handling, the lab node is registered for A/B tests, and long-form ComfyUI usage is documented. The main Prompt Relay node can now receive long-form quality planning controls through `PromptRelayAdvancedOptions` and log a ComfyUI-visible chunk plan/preflight for 2500+ frame workflows.

Do not claim final visual quality until Sarp runs controlled ComfyUI renders. The remaining validation is workflow/runtime visual validation, not deterministic unit-test coverage.

## Tested guarantees

The local unittest suite currently covers these deterministic contracts:

- Smart prompt parsing
  - Inline `|` syntax strips whitespace and assigns equal weights by default.
  - Inline numeric tags support weights and ranges, including spaced/en dash ranges such as `[0 – 25]`.
  - Malformed numeric tags are preserved as prompt text instead of being partially stripped.
  - Block headers strip scene markers, support numeric/word/ordinal ranges including thirty-plus fallback forms without `word2number`, accept common markdown-decorated LLM headers, preserve fractional range spans, and ignore preamble text before the first header.
  - Body inline tags override block header weights.
- Segment metadata from `prompt_relay.build_segments`
  - Default video sigma remains `1 / ln(1 / epsilon)`.
  - Default free window remains `max(L // 2 - 2, 0)`.
  - Audio options are opt-in and do not change video metadata.
  - Video options change video metadata only when explicitly supplied.
  - Non-positive segment lengths are skipped without moving future positive segment midpoints forward.
- Timeline conversion from pixel/editor lengths into latent lengths
  - Empty and all-zero timelines stay safe.
  - Near-full coverage pins to the latent frame count.
  - Partial timelines remain partial in latent space.
  - Largest-remainder rounding allocates gaps deterministically.
  - Tiny and negative segments are clamped to safe non-negative inputs and at least one latent frame per surviving segment.
- Sigma math reference tests
  - The paper equation satisfies the endpoint `epsilon` condition.
  - The current implementation formula is characterized separately and remains sharper than the paper equation at the same epsilon.
  - Invalid epsilon fallback is documented as current-behavior-only, not a proposed paper mode.
- Drop-in lab node math/installability
  - The supplemental lab node keeps upstream-compatible sigma by default.
  - Its segment length parser now accepts frame counts, percentages, and decimal ratios like the main node.
  - Lab segment/sigma behavior is pinned against the main node for critical options, including non-positive segment skip behavior.
  - LTXV grid metadata parsing falls back safely when Comfy metadata is missing, truncated, non-numeric, or non-positive.
  - Empty relay segment metadata fails early with an actionable error instead of a cryptic `max()` failure.
  - Long-form chunk planning can bound each `[query_tokens, text_tokens]` mask under the safety cap for 2500+ frame timelines.
  - Chunk planning now fails early when even one latent frame would exceed the configured mask budget.
  - Chunk stitch range planning splits overlaps into deterministic non-overlapping kept frames and rejects gapped plans.
  - Chunk handoff diagnostics flag hard cuts or too-short overlaps before expensive long-form renders.
  - Chunk plan diagnostics summarize budget, overlap, stitch coverage, and worst seam status in bounded Comfy-friendly logs.
  - Chunk prompt windows clip global prompt beats to chunk-local segment lengths for bounded long-form rendering.
  - Chunk prompt schedules can page global prompt beats onto each resumable long-form chunk without materializing every window.
  - Quality-oriented chunk planning can prefer 129-257 frame windows while still honoring the hard mask budget.
  - The main ComfyUI encoder path can trigger quality chunk planning from connected advanced options without changing default behavior.
  - Long-form plan budget estimates include one conservative token of special-token slack so EOS/BOS bookkeeping does not understate mask size at the cap.
  - Chunk handoff summaries are bounded for long 2500+ frame Comfy logs.
  - Chunk plan evaluation now preflights mask budget, stitch coverage, handoff risk, and continuity carry before expensive renders.
  - Resumable page diagnostics include carry-in seam status so infinite runners can distinguish local page safety from cross-page continuity risk.
  - Chunk memory anchors select bounded clean stitched frames for closed-loop long-form conditioning.
  - Prompt seam diagnostics flag chunk stitch seams that land on or near authored prompt transitions.
  - Prompt-safe stitch shifts can now drive crossfade seam metadata, keeping blend diagnostics aligned with shifted seams.
  - Metric-agnostic quality gate helpers can rank chunk candidates, accept only passing chunks, and return rejected retry cursors so failed candidates do not become future memory.
  - Package registration exposes the lab node without replacing existing baseline nodes.

Run the suite with:

```bash
python3 -m unittest discover -s tests -v
```

Known latest local result: 174 tests passing on 2026-05-16 PDT.

New helpers added in the current working tree:

- `build_quality_chunk_render_manifest(step)` creates a scheduler-facing payload from a safe quality chunk stream step. It exposes the render decision, chunk input ranges, output append range, prompt windows, conditioning anchors, crossfade windows, sanitized checkpoint state, progress counters, and bounded diagnostics. This is intended as the handoff contract for a future ComfyUI queue runner, not a visual-quality claim.
- `build_quality_chunk_queue_plan(manifest)` turns that page-level manifest into per-chunk queue items with input ranges, keep/trim ranges, prompt windows, conditioning anchors, crossfade windows, and checkpoint state. It gives the future runner an explicit render/skip contract without depending on ComfyUI internals yet.

## Current working-tree status

The repo contains uncommitted local work in four buckets:

- Runtime-adjacent parser/timeline robustness changes in `parser.py` and `nodes.py`.
- Advanced option and LTXV AV attention-routing hardening in `advanced_options.py`, `prompt_relay.py`, and `patches.py`.
- Test characterization files under `tests/`.
- Custom node, integration, experiment, and research docs under `docs/`.

Do not push. Do not commit unless Sarp explicitly approves the final commit message in the parent session.

## Non-goals until approval

- Do not change default runtime Prompt Relay behavior.
- Do not silently switch to paper-calibrated sigma.
- Do not change default `epsilon` from `1e-3` to `0.1`.
- Do not claim visual quality improvements without controlled ComfyUI comparisons.
- Do not add new dependencies for scoring, CLIP/DINO metrics, or workflow automation without dependency review.
- Do not delete or rewrite existing local changes while this characterization branch is active.

## Remaining work outside this repo pass

1. Install or symlink the repo into ComfyUI `custom_nodes`.
2. Run a short two-segment LTX 2.3 AV smoke render and confirm `[PromptRelay] Setup` appears in the ComfyUI console.
3. Run controlled A/B renders using `upstream_compat` vs `paper_boundary` before making any visual-quality claims.
4. If Sarp wants this saved as a Git milestone, approve a final local commit message first. Do not push without explicit approval.
