# Prompt Relay experiment log

No behavior-changing experiments have been implemented yet.

## Experiment record template

For each experiment, add:

- ID:
- Status: proposed | approved | implemented | tested | rejected
- Files changed:
- Hypothesis:
- Paper basis:
- Math basis:
- Implementation plan:
- Test plan:
- Results:
- Decision:

## Proposed experiments

### EXP-001: Paper-correct sigma mode

- Status: proposed, approval needed before implementation
- Approval ask: Allow a non-default, opt-in `sigma_mode` experiment that preserves current behavior as the default and adds a paper-calibrated mode for side-by-side testing. No default runtime behavior should change in the first patch.
- Hypothesis: Matching the paper's sigma equation will produce smoother boundary transitions and less hard masking than the current implementation, especially around segment boundaries.
- Paper basis:
  - Equation (3): `C(i,j)=ReLU(|f(i)-m_s|-w)^2/(2 sigma^2)`.
  - Equation (4): `exp(-((L-w)^2)/(2 sigma^2)) = epsilon => sigma = (L-w) / sqrt(2 ln(1/epsilon))`.
  - Figure/source states `w=L-2` and experiments use `epsilon=0.1`.
- Math basis:
  - Subtracting `C` from attention logits multiplies pre-softmax attention by `exp(-C)`.
  - With paper `w=L_endpoint-2`, `L_endpoint-w=2`, so paper-calibrated constant sigma is `2 / sqrt(2 ln(1/epsilon))`.
  - Current code uses `1 / ln(1/epsilon)`, which is substantially sharper than equation (4): about `0.434` versus `0.932` at `epsilon=0.1`, and about `0.145` versus `0.538` at `epsilon=1e-3`.
- Files expected to change if approved:
  - `prompt_relay.py`: add sigma/window helper and mode flag plumbing in metadata generation.
  - `advanced_options.py` and/or node schemas: expose an explicit opt-in mode only if Sarp wants UI access immediately.
  - Tests/docs: add lightweight tests for segment metadata and endpoint retained attention.
- Implementation plan:
  1. Add pure helper functions for current sigma and paper-calibrated sigma. Keep current mode as default.
  2. Add an opt-in paper mode that computes endpoint distance/window explicitly and records mode in segment metadata/logging.
  3. Add tests that assert midpoint/free-window cost is zero and endpoint prior is approximately `epsilon` in paper mode.
  4. Do not run visual ComfyUI claims until Sarp tests/approves generated comparisons.
- Test plan:
  - Unit-style Python test around `build_segments` and `build_temporal_cost` with synthetic token ranges and latent frames.
  - Compare retained prior `exp(-cost)` at midpoint, free-window edge, endpoint, and neighboring segment center.
  - Optional visual run after approval: same model, prompt, seed, dimensions, sampler, segment lengths; compare current mode versus paper mode at `epsilon=0.1` and `1e-3`.
- Results: not implemented.
- Decision: waiting on Sarp approval.

### EXP-002: Epsilon default comparison

- Status: proposed, not implemented
- Hypothesis: `epsilon=0.1` will better match the paper's reported behavior than the node default `1e-3`, especially after paper-correct sigma is available.
- Paper basis: The paper says experiments use `epsilon=0.1`.
- Math basis: At the endpoint, `exp(-C)=epsilon`, so lower epsilon means stronger suppression. `1e-3` is 100x lower retained attention than `0.1` at the calibrated point.
- Implementation plan: Do not silently change the default first. Add docs and/or an explicit preset after testing.
- Test plan: Same prompt/seed/settings with `1e-3` versus `0.1`, compare temporal alignment and transition quality.

### EXP-003: Short-segment decay stabilization

- Status: proposed, not implemented
- Hypothesis: Short segments need a non-degenerate free-attention zone or calibrated sigma to avoid prompt snapping/flicker.
- Paper basis: Boundary decay assumes a free-attention window `w` and smooth suppression outside that window.
- Math basis: If `w=0` and `sigma` is small, the penalty grows immediately from the segment midpoint, making short events brittle.
- Implementation plan: Add a minimum or fractional window mode behind an option.
- Test plan: Multi-event prompts with 1-3 latent-frame segments, compare boundary flicker and event recognition.

### EXP-004: Token budget warnings

- Status: proposed, not implemented
- Hypothesis: Many long local prompts silently degrade because later tokens are truncated or weakly represented.
- Paper basis: Prompt Relay can only route tokens that exist in the conditioning sequence.
- Math basis: If a local prompt's token range is missing/truncated, its assigned segment cannot receive the intended semantic signal regardless of temporal penalty.
- Implementation plan: Add tokenizer budget logging/warnings for global + local prompt length and per-segment token ranges.
- Test plan: Parser/tokenization tests with intentionally long prompts.


## First experiment approval packet

Recommended first approved experiment: **EXP-001 paper-correct sigma mode**, opt-in only.

Why this first:

1. It is directly grounded in the paper source and current code has a concrete equation mismatch.
2. It can be tested with deterministic metadata/cost assertions before any visual generation.
3. It avoids changing defaults, so existing workflows remain compatible.
4. It provides a cleaner basis for later epsilon/window/timeline experiments.

Proposed non-goals for the first patch:

- Do not change default `epsilon` from `1e-3` to `0.1` yet.
- Do not alter timeline length conversion yet.
- Do not claim quality improvement until controlled visual tests are run.
- Do not add LoRA scheduling or Wan 2.2 high/low workflow features.

Approval question for Sarp:

> Approve implementing EXP-001 as an opt-in paper-calibrated sigma mode, with tests/docs, preserving current behavior as default?

If approved, the smallest implementation path is helper functions + tests first, then optional node/advanced-option exposure depending on how Sarp wants to run the comparison.

### EXP-005: Timeline conversion characterization tests

- Status: proposed, not implemented
- Hypothesis: Some long generation timing degradation comes from pixel or editor frame lengths being rounded into latent frame lengths differently than users expect.
- Paper basis: Prompt Relay depends on each local prompt being active in its intended temporal interval.
- Math basis: If segment boundaries are shifted during conversion to latent frames, the attention prior is applied to the wrong latent timesteps even when the cross-attention penalty is mathematically correct.
- Implementation plan: Add tests around `_convert_to_latent_lengths` only, with Wan stride 4 and representative LTX stride values. Do not change conversion behavior in the first pass.
- Test plan: Assert sums, largest remainder allocation, full coverage pinning, partial coverage behavior, and tiny segment handling.
- Results: not implemented.
- Decision: waiting on Sarp approval.

### EXP-006: Parser syntax robustness review

- Status: unapproved existing work found in working tree
- Files changed: `parser.py`, `tests/test_parser.py`
- Hypothesis: Accepting whitespace and en dash inline range tags makes smart prompt parsing more tolerant of user input copied from documents or UI text.
- Paper basis: No direct paper basis. This is UX robustness rather than Prompt Relay math.
- Math basis: Inline ranges are converted to segment weights, so syntax expansion can change which text becomes weighted timing input.
- Implementation plan: No further implementation without Sarp approval. If approved, review edge cases and keep tests.
- Test plan: Existing untracked tests cover the observed syntax expansion and block parser behavior.
- Results: inspected only.
- Decision: do not delete, commit, or extend without approval.

## Next approval packet

Recommended next direction is still EXP-001 first, with EXP-005 tests as a parallel non-runtime characterization task.

Approval asks for Sarp:

1. Approve EXP-001 as opt-in paper-calibrated sigma mode with helper tests and no default behavior change.
2. Approve EXP-005 characterization tests for timeline conversion with no behavior change.
3. Decide whether to keep, revert, or formalize EXP-006 parser robustness work.

Suggested order:

1. Add deterministic tests for current and paper-calibrated sigma math.
2. Add timeline conversion characterization tests.
3. Only after Sarp approves runtime experiments, expose opt-in paper sigma mode for controlled visual comparisons.

### EXP-007: Closed-loop chunk architecture design

- Status: proposed, documentation only
- Hypothesis: Long-generation degradation can be reduced more reliably by short accepted chunks with overlap and clean memory than by one long Prompt Relay pass.
- Paper basis: Prompt Relay routes text temporally, while Video-Infinity, MAG, and StableWorld-style work all point toward retaining selected context instead of relying on full or short-window history alone.
- Math basis: Repeated open-loop conditioning allows errors to compound. Rejection and clean-memory updates create a bounded feedback loop where low-quality chunks do not become future anchors.
- Implementation plan: No runtime code yet. First produce a design doc or workflow sketch that maps current Prompt Relay nodes into a chunk planner, memory bank, candidate generation loop, and quality gate.
- Test plan: Use a fixed prompt, seed set, model settings, chunk length, and overlap. Compare one long generation against chunked generation with manual accept or reject before implementing automatic scoring.
- Results: external memo analyzed and source-checked. No behavior changed.
- Decision: waiting on Sarp approval for design or prototype work.

### EXP-008: Clean visual memory bank

- Status: proposed, not implemented
- Hypothesis: Feeding only verified clean anchors into future chunks will reduce identity, object, layout, and style drift compared with feeding all previous frames or only the immediate last frame.
- Paper basis: Video-Infinity uses local and global context frames. MAG frames long generation as a memory compression problem. StableWorld uses dynamic frame eviction based on geometric similarity.
- Math basis: Anchor selection constrains future conditioning distribution. Bad anchors shift future generations toward accumulated artifacts, while sparse clean anchors should preserve long-range state with lower context cost.
- Implementation plan: Define memory item schema first: canonical first frame, last accepted clean frame, sparse best frames, face or product crops, masks or boxes, scene summary, camera state, rejected-frame debug log.
- Test plan: Manual pilot with a product or character room scene. Track whether accepted anchors preserve product shape, face, clothing, room layout, and lighting across 3 to 5 chunks.
- Results: not implemented.
- Decision: waiting on Sarp approval.

### EXP-009: Quality gate scorer prototype

- Status: proposed, partially supported by pure candidate-gate helpers
- Hypothesis: A lightweight scorer can reject chunks that would poison continuation before those frames enter memory.
- Paper basis: StableWorld motivates geometric frame selection. The external memo proposes identity, object, transition, flicker, sharpness, and text alignment checks.
- Math basis: Candidate selection converts single-sample inference into a search problem over candidate chunks, optimizing for consistency scores before accepting new state.
- Implementation plan: Start outside core node behavior with an offline script or optional node. Prefer simple metrics first: sharpness, frame difference or flicker, overlap similarity, crop similarity, and manual notes. Add CLIP or DINO only after dependency review.
- Test plan: Generate 3 to 5 candidate chunks for the same segment. Score, rank, and compare with human judgment before automating rejection thresholds.
- Results: pure threshold/ranking helpers implemented in EXP-009A; actual visual metric extraction is still not implemented.
- Decision: deterministic gate helper is safe locally; scorer dependencies and ComfyUI runtime wiring still need review.

### EXP-009A: Metric-agnostic chunk candidate quality gate

- Status: implemented as pure helpers plus tests, no ComfyUI runtime dependency added
- Files changed:
  - `prompt_relay.py`
  - `tests/test_prompt_relay_segments.py`
  - `docs/current-status.md`
- Hypothesis: A small metric-agnostic gate can keep failed chunks out of the clean-memory loop by accepting only candidates that pass configured thresholds and returning retry cursors for chunks with no passing candidate.
- Paper basis: This follows the closed-loop chunk architecture direction: generate candidates, score, accept only clean chunks, and reject or regenerate failed chunks before memory update.
- Math basis: Candidate selection turns single-sample generation into a thresholded search step over metrics. Metrics are caller-supplied, so the gate can start with simple transition, flicker, sharpness, or identity scores and later accept CLIP/DINO/face/object scores without changing planner state.
- Implementation plan: Add pure `evaluate_quality_chunk_candidates` and bounded diagnostics formatting. Do not add new scorer dependencies or change default Prompt Relay behavior.
- Test plan: Unit tests verify best passing candidate selection, failed metric reporting, rejected chunk retry cursors, and regenerate decisions when no candidate passes.
- Results: `python3 -m unittest discover -s tests -v` passed 171 tests on 2026-05-16 PDT.
- Decision: Safe deterministic helper landed locally; visual quality claims still require controlled ComfyUI candidate renders.

### EXP-010: Candidate generation and overlap workflow

- Status: proposed, not implemented
- Hypothesis: Generating multiple candidates per chunk and enforcing 1 to 2 seconds of overlap will reduce visible transition failures and action drift.
- Paper basis: Prompt Relay improves event timing inside a generated window. The long-video references support using context windows instead of unbounded single-pass generation.
- Math basis: Overlap gives a direct transition consistency region for scoring and blending. Multiple candidates improve the probability that at least one sample satisfies identity, transition, and action constraints.
- Implementation plan: Draft a ComfyUI workflow recipe before node changes. Inputs should include unchanged global prompt, local delta prompts, canonical anchor, last clean keyframe, selected memory frames, and overlap frames.
- Test plan: Compare hard continuation versus overlapped continuation on a controlled two-chunk prompt with fixed settings and manual frame inspection.
- Results: not implemented.
- Decision: waiting on Sarp approval.

### EXP-001A: Isolated sigma math characterization tests

- Status: implemented as tests only, no runtime behavior change
- Files changed:
  - `tests/test_sigma_math.py`
- Hypothesis: Before adding an opt-in paper sigma mode, deterministic math tests should pin the paper endpoint condition and show how it differs from the current implementation formula.
- Paper basis:
  - Equation (4): `exp(-((L-w)^2)/(2 sigma^2)) = epsilon`.
  - Solving gives `sigma = (L-w) / sqrt(2 ln(1/epsilon))`.
- Math basis:
  - The retained attention prior at the calibrated endpoint should equal `epsilon`.
  - Under the paper reference `w=L-2`, the penalty distance is always `2`, so sigma is constant for a fixed epsilon.
  - The current implementation formula `1 / ln(1/epsilon)` remains characterized separately and is sharper than the paper equation at the same epsilon.
- Implementation plan: Add pure test-local helper functions only. Do not import or modify runtime code.
- Test plan: Run Python unittest discovery.
- Results: `python3 -m unittest discover -s tests -v` passed 15 tests on 2026-05-14 PDT.
- Decision: Safe characterization step complete; runtime opt-in mode still requires explicit implementation approval.

### EXP-001B: Runtime segment metadata characterization tests

- Status: implemented as tests only, no runtime behavior change
- Files changed:
  - `tests/test_prompt_relay_segments.py`
- Hypothesis: Before exposing any paper-calibrated sigma mode, deterministic tests should pin the current `build_segments` metadata contract so future opt-in changes do not accidentally alter defaults.
- Paper basis:
  - Same Prompt Relay cost shape as EXP-001/EXP-001A: segment midpoint, free window, sigma, and strength define the temporal prior.
- Math basis:
  - Current default sigma remains `1 / ln(1 / epsilon)`.
  - Current default window remains `max(L // 2 - 2, 0)`.
  - Audio overrides are intended to affect only scaled/audio metadata unless video knobs are explicitly supplied.
- Implementation plan: Add unit tests for `build_segments` with a local fake `torch` module so the tests run in this repo environment without installing ComfyUI/PyTorch dependencies.
- Test plan: Run Python unittest discovery.
- Results: `python3 -m unittest discover -s tests -v` passed 21 tests on 2026-05-15 PDT.
- Decision: Safe characterization step complete; runtime opt-in paper sigma mode still requires explicit implementation approval.

### EXP-001C: Current status and tested-guarantees doc

- Status: implemented as documentation only, no runtime behavior change
- Files changed:
  - `docs/current-status.md`
  - `README.md`
- Hypothesis: A concise status page will make the current characterization guarantees, approval boundaries, and next steps easier to review before any behavior-changing Prompt Relay work.
- Paper basis: None directly; this is project hygiene around the Prompt Relay implementation and experiment process.
- Math basis: The doc summarizes existing deterministic parser, timeline, segment metadata, and sigma math tests without introducing new formulas.
- Implementation plan: Add a status document listing tested guarantees, non-goals, working-tree caution, and recommended approval path. Link it from the README.
- Test plan: Run Python unittest discovery to verify the doc-only increment did not disturb behavior.
- Results: `python3 -m unittest discover -s tests -v` passed 21 tests on 2026-05-15 PDT.
- Decision: Safe documentation step complete; runtime opt-in paper sigma mode still requires explicit approval.
