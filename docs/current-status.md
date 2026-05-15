# Prompt Relay quality lab current status

_Last updated: 2026-05-15 PDT_

This repo is currently usable as a local ComfyUI custom node checkout. The code-side goal is complete for the current pass: Prompt Relay behavior is testable, the main node has safer timing/options handling, the lab node is registered for A/B tests, and long-form ComfyUI usage is documented.

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
  - Package registration exposes the lab node without replacing existing baseline nodes.

Run the suite with:

```bash
python3 -m unittest discover -s tests -v
```

Known latest local result: 97 tests passing on 2026-05-15 PDT.

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
