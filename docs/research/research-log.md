# Prompt Relay research log

This file is the running source of truth for web research, paper notes, repo observations, and user-practice findings. Add notes here before or immediately after any implementation work.

## Sources reviewed

### Prompt Relay project page

- URL: https://gordonchen19.github.io/Prompt-Relay/
- Date reviewed: 2026-05-14 PDT
- Key claims:
  - Prompt Relay is inference-time, training-free, plug-and-play temporal prompt routing for multi-event video generation.
  - It improves temporal placement by suppressing attention from video tokens to prompts outside the assigned segment.
  - The project page claims visual quality can improve because routing reduces prompt competition in cross-attention.

### Prompt Relay arXiv paper

- URL: https://arxiv.org/abs/2604.10030
- HTML: https://arxiv.org/html/2604.10030v1
- Date reviewed: 2026-05-14 PDT
- Title: Prompt Relay: Inference-Time Temporal Control for Multi-Event Video Generation
- Authors: Gordon Chen, Ziqi Huang, Ziwei Liu
- Core problem:
  - Paragraph-style multi-event prompts cause semantic entanglement because every frame attends to the whole prompt.
  - The model lacks explicit temporal awareness in cross-attention.
- Core method:
  - Add a temporal penalty to cross-attention logits so query tokens for a given latent frame primarily attend to tokens from the assigned prompt segment.
  - In attention terms, if logits are `QK^T / sqrt(d)`, Prompt Relay changes them to `QK^T / sqrt(d) - C(i,j)`.
  - Since softmax exponentiates logits, subtracting `C` is equivalent to multiplying that token pair's attention contribution by `exp(-C)`.
- Boundary decay math:
  - The paper describes a Gaussian-style boundary attention decay.
  - The endpoint condition is `exp(-((L - w)^2) / (2 sigma^2)) = epsilon`.
  - Solving gives `sigma = (L - w) / sqrt(2 ln(1/epsilon))`.
  - The paper says it uses `epsilon = 0.1` in experiments.
- Paper limitations:
  - Persistent visual elements are not explicitly shared across local segments.
  - Character, object, and style consistency can drift when local prompts describe persistent elements inconsistently.
  - The paper says a global prompt can mitigate this by anchoring shared context.

### ComfyUI-PromptRelay repository

- URL: https://github.com/kijai/ComfyUI-PromptRelay
- Local private working repo: `/Users/sarpdoven/.openclaw/workspace/prompt-relay-quality-lab`
- Private GitHub repo: https://github.com/sarptandoven/prompt-relay-quality-lab
- Date reviewed: 2026-05-14 PDT
- License state: no LICENSE file found in the source repo, and GitHub reports `licenseInfo: null`. Treat changes as private research unless licensing is clarified.
- Key files:
  - `prompt_relay.py`: temporal cost, segment metadata, token index mapping, length distribution.
  - `nodes.py`: ComfyUI node schemas and encoding path.
  - `smart_nodes.py`: smart prompt parser front-end and auto segment length weighting.
  - `parser.py`: inline and block prompt syntax parser.
  - `patches.py`: Wan/LTX cross-attention monkey-patches.
  - `advanced_options.py`: advanced knobs for video/audio strength and window scale.

## Initial implementation observations

- Current `build_segments` comment says the paper uses constant `sigma = 1 / ln(1/epsilon)` regardless of segment length.
- That does not match the equation found in the paper HTML, which solves `sigma = (L - w) / sqrt(2 ln(1/epsilon))`.
- Current default `epsilon` is `1e-3`, while the paper says it uses `0.1` in experiments.
- Current `base_window = max(L // 2 - 2, 0)` collapses short segments to `window = 0`.
- Current midpoint/window math uses integer flooring, which can bias odd-length or short segments.
- The implementation leaves global prompt tokens unpenalized, which matches the paper's recommendation to use global context for persistent elements.

## Why longer generations can degrade

Prompt Relay addresses cross-attention semantic interference, but longer generations can still degrade for reasons outside the method's scope:

1. More segments increase total local prompt tokens, increasing text encoder pressure and risk of truncation.
2. More event boundaries increase the number of places where identity, layout, and motion continuity can break.
3. Video backbones accumulate temporal drift over longer denoising/generation horizons.
4. Prompt Relay routes local text tokens, but it does not directly constrain self-attention, latent recurrence, identity embeddings, or optical/physical consistency.
5. If persistent objects are repeated differently in local prompts, the paper's stated limitation applies: local segment isolation can encourage appearance drift unless a global prompt anchors shared context.

## Research TODO

- Pull more exact equations from the paper PDF/source if HTML math is ambiguous.
- Search GitHub issues/discussions around Prompt Relay, Wan Prompt Relay integration, and ComfyUI-PromptRelay user reports.
- Search Reddit and ComfyUI communities for user reports about Prompt Relay, multi-event prompting, Wan/LTX long generation degradation, and best epsilon/window settings.
- For every proposed code change, create an experiment entry under `docs/experiments/` before implementation.

## User-practice research notes

### GitHub issues in `kijai/ComfyUI-PromptRelay`

Reviewed with GitHub CLI on 2026-05-14 PDT.

Relevant open issues and signals:

- Issue #14: user asks for a Wan 2.2 version and is confused by high/low model connections in Wan 2.2 workflows.
  - Signal: integration friction is real for Wan 2.2 dual-model/high-low workflows.
- Issue #13: user reports adding per-section LoRA gates tied to prompt relay segments.
  - Signal: users want not only text routing, but segment-indexed conditioning changes.
- Issue #12/#11: `PromptRelayAdvancedOptions` mapping/regression reports.
  - Signal: advanced options are fragile in node registration and need basic import/mapping tests.
- Issue #10: user tried Wan 2.2 with separate relays for high/low noise and hit workflow errors.
  - Signal: Wan 2.2 compatibility and UX should be investigated before behavioral tuning claims.
- Issue #9: `NoneType` error around `.strip()`.
  - Signal: stale workflow JSON or frontend population problems can pass `None`; current local code already contains explicit `None` checks in `_encode_relay`.
- Issue #7: regular node versus timeline node timing mismatch.
  - Signal: timeline pixel-space to latent-frame conversion is a likely quality/timing limitation. The repo's `_convert_to_latent_lengths` is a direct target for tests.
- Issue #6: request for defining timings in seconds and FPS.
  - Signal: users want time-based control; current timeline node has fps/time_units display inputs, but internal conversion still uses pixel-space lengths.
- Issue #4: multi-image injection corrupts outputs.
  - Signal: prompt relay may conflict with multi-image conditioning because only text cross-attention is routed. Image conditioning can still impose unrelated temporal constraints.
- Issue #2: positive early user feedback and surprise that it supports more than Wan.

### GitHub search outside the main repo

Reviewed on 2026-05-14 PDT.

- `Prompt Relay LoRA Schedule` PR exists against `kijai/ComfyUI-PromptRelay`.
  - Signal: segment-indexed LoRA scheduling is an active community direction, but it is outside the first quality experiment unless Sarp asks.
- Searches for `Prompt Relay epsilon video generation` did not surface useful direct issue reports.
  - Signal: epsilon/window tuning may be under-documented in user discussions.

### Reddit / community findings

Reviewed Reddit JSON search on 2026-05-14 PDT. Notes are summaries of visible post metadata/selftext, not full comment analysis yet.

Search: `"Prompt Relay" ComfyUI`

- r/comfyui: `ComfyUI Tutorial: LTX 2.3 Prompt Relay Workflow On 6GB Vram (Res: 1920x1080 Video Length 15 sec)`
  - Score around 250, 34 comments.
  - Claims Prompt Relay nodes give full control over long video by assigning each timeline segment to a specific prompt.
  - Low-VRAM workflow context: LTX 2.3, 6GB VRAM, 15 second video.
- r/StableDiffusion: `Test of Runexx Movie Maker Comfyui workflow with Prompt Relay Encode node integration`
  - User says LTX2.3 struggles with fast motion, blurring/smearing characters, especially full-body shots, distance-to-camera motion, or fast fight scenes; close-ups and medium shots are more tolerable.
  - Signal: quality failures in long/complex generations are not only prompt routing. Motion magnitude and subject scale matter.
- r/comfyui: `LTX 2.3 Prompt Relay workflow test in ComfyUI`
  - User claims Prompt Relay solved controlling drastically different scenes in one workflow.
  - Signal: temporal routing works best for scene/event separation.
- r/comfyui: `LTX Director - An All-In-One Timeline Editor. I2V, T2V, FLFF, Prompt Relay, Custom Audio, and more!`
  - High engagement. User ecosystem is moving toward timeline editors that combine Prompt Relay with first/last-frame and audio/custom controls.
  - Signal: prompt relay alone may be a building block, not a full long-video solution.
- r/comfyui: `LTX 2.3 Prompt Relay - Really good for consistency`
  - High engagement. Need full comment/post follow-up.

Search: `"Prompt Relay" Wan2.2`

- r/comfyui: `workflowWAN2.2 | text2video | using Prompt Relay node |`
  - User says they wanted more control over video movement and prompting and looked for a Prompt Relay workflow for Wan2.2.
  - Signal: Wan2.2 demand exists, but integration details need validation.

Search: `"LTX 2.3" "Prompt Relay"`

- Multiple posts combine LTX 2.3 with Prompt Relay, FLF, frame interpolation, GGUF workflows, and messy chase/fight scenes.
- A recurring pattern is that Prompt Relay improves event consistency/control, while fast motion and complex character movement remain weak points.

## Current conclusions from research so far

1. First implementation target should be math correctness and testability, not broad feature expansion.
2. Timeline conversion and Wan 2.2 integration are high-value follow-up areas because users report timing mismatch and workflow confusion.
3. Long-generation quality loss appears to combine:
   - cross-attention semantic entanglement, addressed by Prompt Relay;
   - backbone motion limits, especially fast/full-body/distant subjects;
   - segment boundary/timing mismatch;
   - persistent identity/style drift unless anchored globally;
   - token budget pressure from many long prompts.
4. Any experiment should be logged before implementation and should include paper basis, math basis, and a test plan.


## Source follow-up: arXiv HTML/source math check

Reviewed on 2026-05-14 PDT using:

- HTML: `https://arxiv.org/html/2604.10030v1`
- e-print/source: `https://arxiv.org/e-print/2604.10030v1`, especially `sec/1_intro.tex`

Exact source statements relevant to implementation:

- Penalty definition: `C(i, j) = ReLU(|f(i) - m_s| - w)^2 / (2 sigma^2)`, with `m_s = (t_s^start + t_s^end) / 2`.
- The paper describes subtracting `C(i,j)` from cross-attention logits, which applies multiplicative prior `exp(-C(i,j))` to unnormalized attention scores before softmax.
- Boundary calibration: `exp(-((L-w)^2)/(2 sigma^2)) = epsilon => sigma = (L-w) / sqrt(2 ln(1/epsilon))`.
- The paper defines this `L` as endpoint distance from the segment midpoint, `L = |f(i)-m_s|`, not necessarily the full segment length.
- Figure 3 caption/source says `w=L-2` preserves full attention within the segment and suppresses attention near segment boundaries.
- Experimental setup says: `We set epsilon=0.1 across all experiments. Setting w = L - 2 reduces sigma to a constant.`

Interpretation for this repo:

- `prompt_relay.py::build_segments` currently approximates the paper's `w=L-2` by using `base_window = max(L // 2 - 2, 0)`, where local variable `L` is the integer segment length in latent frames. That corresponds to endpoint-distance `L_paper ~= segment_length / 2`, with integer floor bias.
- If the paper's `w=L_paper-2` is used exactly, then `L_paper - w = 2`, so equation (4) gives a constant `sigma = 2 / sqrt(2 ln(1/epsilon)) = sqrt(2 / ln(1/epsilon))`.
- Current code uses `sigma = 1 / ln(1/epsilon)`. This is not equation (4). For `epsilon=0.1`, equation (4) with `L-w=2` gives about `0.932`, while current code gives about `0.434`; for the current default `epsilon=1e-3`, equation (4) gives about `0.538`, while current code gives about `0.145`.
- The node tooltips currently say `paper default 0.001`, but the paper source says `epsilon=0.1` across experiments. This is a documentation/UI mismatch even before changing behavior.

## Code inspection notes: likely quality/timing targets

Reviewed on 2026-05-14 PDT.

- `prompt_relay.py::build_segments`
  - Owns sigma/window/midpoint metadata for all downstream masks.
  - Current comment says paper uses `sigma = 1/ln(1/epsilon)` regardless of segment length. The exact paper equation does not support that formula.
  - Uses integer midpoint `(2 * frame_cursor + L) // 2`, which floors odd-length segment centers.
  - Uses `base_window = max(L // 2 - 2, 0)`, which collapses short segments to zero free-attention window and floors endpoint distance.
  - Uses one global video sigma for all segments, so any future paper-correct mode should decide whether to preserve the constant-sigma `w=L_paper-2` assumption or compute per-segment sigma from actual endpoint distance/window.
- `prompt_relay.py::build_temporal_cost` and `build_temporal_cost_scaled`
  - Apply `cost = strength * ReLU(distance-window)^2 / (2*sigma^2)` and return `-cost` as additive attention mask via `create_mask_fn`.
  - This matches equation (2)/(3) structurally. Main mismatch is metadata calibration, not mask application.
- `nodes.py::_convert_to_latent_lengths`
  - Converts timeline/editor pixel-frame segment lengths into latent-frame segment lengths using largest remainder and pins to full latent coverage when near full length.
  - This is the likely locus for reported timeline timing mismatch because UI lengths are pixel-space but routing happens in latent frame space.
  - Needs unit tests with Wan stride 4 and LTX stride values before any behavior change.
- `nodes.py` epsilon input tooltips
  - Both regular and timeline nodes currently state `paper default 0.001`; source says `0.1`. This can be corrected as docs/UI text without changing runtime behavior, but still should be approved if Sarp wants zero code changes right now.
- `patches.py::detect_model_type`
  - Wan temporal stride is hard-coded to 4; LTX uses `diff_model.vae_scale_factors[0]`. Timing conversion tests should cover these model-specific assumptions.

## 2026-05-14 PDT follow-up notes

- The active research goal remains improving this repo toward higher quality and less degradation over longer generation time.
- Web search was unavailable in this subagent because the configured SearXNG backend is missing.
- Direct fetch of the arXiv HTML reconfirmed the core failure mode as temporal semantic entanglement from all frames attending to the full paragraph prompt.
- Direct fetch reconfirmed related methods named in the paper including MinT, MEVG, DiTCtrl, TS-Attn, and SwitchCraft.
- The paper positions Prompt Relay as inference-time cross-attention routing that improves temporal prompt alignment, transition naturalness, and visual quality by reducing prompt competition.
- The paper's own setup uses Wan 2.2, epsilon 0.1, and a global prompt for persistent context.
- GitHub issue 7 gives a direct user report that timeline segment lengths may not line up with output while the regular node timing seems closer.
- GitHub issue 13 gives a direct user report of segment-indexed LoRA gates, which suggests users want temporal conditioning beyond text routing.
- GitHub CLI issue review reconfirmed practical problem clusters around Wan 2.2 high and low model wiring, timeline timing, advanced options registration, None input handling, seconds based timing, multi-image corruption, and WanVideoWrapper integration.

## 2026-05-14 PDT unexpected parser change inspection

- `parser.py` currently has a modified inline tag regex compared with HEAD.
- The change accepts spaces inside brackets and en dash separators for inline range tags.
- `tests/test_parser.py` is untracked and covers inline whitespace, en dash ranges, block range headers, and body inline overrides.
- This is useful robustness work, but it is behavior-changing parser experimentation because it expands accepted user syntax.
- It is not approved under the current no behavior-changing experiments rule.
- The changes were inspected and documented only. They were not deleted, committed, pushed, or expanded.

## External research notes from Sarp, 2026-05-14 PDT

Sarp provided an external research memo on Prompt Relay and infinite video. A copy is saved at `docs/research/external-notes/prompt-relay-infinite-video-notes-2026-05-14-pdt.txt`.

Usable findings:

- Prompt Relay should be treated as a timing layer, not the whole long-video solution.
- The memo agrees with our current direction that Prompt Relay routes text influence but does not create persistent visual memory.
- Longer generation quality decay should be attacked with chunking, overlap, clean memory, candidate generation, scoring, rejection, and regeneration.
- The immediate practical architecture is short chunks plus stable global prompt plus local motion deltas plus clean visual anchors.
- Quality gates should track identity drift, object drift, flicker, transition consistency, sharpness, motion/flow, text alignment, and scene layout.
- Bad generated frames should not become future anchors.
- This supports expanding EXP-005 into a closed-loop long-video quality direction after the node-level Prompt Relay math fixes are documented.

Research follow-up:

- Verify referenced systems before relying on them in implementation notes: Video-Infinity, FreeNoise, FreeLong, MAG-style memory compression, and StableWorld-style clean-frame selection.
- Compare these systems against what can be implemented inside this private ComfyUI node repo without training.

## External memo source verification and critical read, 2026-05-14 PDT

Verified source signals:

- Prompt Relay upstream README and arXiv PDF metadata verify the method is inference-time, training-free, plug-and-play cross-attention routing for temporally constrained multi-event video prompts.
- Kijai ComfyUI README verifies the node already exposes the core user pattern needed for this direction: a stable `global_prompt`, local prompt segments, smart prompt parsing, and guidance to put persistent details in the first/global segment while later segments describe changes only.
- LTX 2.3 prompt guide verifies that LTX 2.3 prefers detailed prompts and warns that short prompts for longer videos leave the model without enough direction to fill duration. This supports the memo's claim that vague segments can cause filler motion or early action completion.
- Video-Infinity README verifies local context frames plus global context frames via `padding` and `attn.topk`. This supports the memo's local recent context plus long-range memory framing, but it is distributed inference research, not directly a ComfyUI Prompt Relay feature.
- FreeNoise README verifies tuning-free longer video generation through noise rescheduling and multi-prompt support. This supports investigating noise continuity, but it does not directly prescribe LTX or Wan Prompt Relay integration.
- FreeLong README verifies training-free long video generation using global and local feature or frequency fusion, including FreeLong++ multi-band spectral fusion. This supports the memo's global low-frequency and local high-frequency intuition, but implementation would be outside the current text-routing node unless scoped carefully.
- MAG source at arXiv `2512.18741v1` verifies Memorize-and-Generate frames long video as a memory problem where short windows forget and full history is too expensive. This supports clean compressed visual memory as a research direction.
- StableWorld source at arXiv `2601.15281v1` verifies stability and temporal degradation in long interactive video generation and discusses dynamic frame eviction using ORB-based geometric similarity. This supports a lightweight clean-frame selection heuristic for future quality gates.

Critical interpretation:

- The external memo is directionally strong, but it mixes buildable ComfyUI workflow ideas with research-grade mechanisms that are not currently present in this repo.
- `Infinite video` should stay an engineering shorthand for closed-loop chunking and rejection, not a quality guarantee.
- The safest near-term work remains docs and deterministic tests. Closed-loop generation needs either workflow orchestration outside these nodes or new nodes that Sarp explicitly approves.
- Quality gates such as CLIP, DINO, face similarity, optical flow, depth, OCR, or VLM text alignment add dependency and model choices. They should be designed as optional evaluators first, not hard dependencies in the Prompt Relay core.
- Candidate generation can be represented at the workflow level before writing any node code: generate N chunks with seed variations, score outputs, accept one, and only then update a clean memory bank.
- Clean memory should be explicit and conservative: first canonical frame, last accepted clean frame, sparse best frames, crops, masks, and text state. Full generated history should not be fed forward blindly.

Mapping to this repo:

- Current node coverage:
  - `PromptRelayEncode` and `PromptRelayEncodeSmart` handle stable global prompt plus local prompt timing.
  - `build_segments` and temporal mask helpers handle prompt influence windows and boundary penalties.
  - `_convert_to_latent_lengths` handles timeline to latent-frame timing and remains a likely source of boundary mismatch.
- Missing system layers:
  - Chunk planner for 4 to 8 second generations with 1 to 2 second overlap.
  - Candidate runner or workflow convention for multi-seed generation per chunk.
  - Clean visual memory bank that stores only accepted anchors.
  - Quality gate nodes or scripts for identity, object, transition, flicker, sharpness, and text-action alignment.
  - Memory update policy and rejection logging.
  - Noise or latent continuity controls.

Near-term consequence:

- EXP-001 and EXP-005 remain the first safe node-level targets because they improve timing math and characterize boundary placement without changing long-video behavior.
- The external memo justifies adding a separate closed-loop quality track after Sarp approves broader work.
