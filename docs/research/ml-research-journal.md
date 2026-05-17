# ML research journal

Simple chronological notes for the Prompt Relay quality lab.

## 2026-05-14 PDT

1. Created the private working repo from `kijai/ComfyUI-PromptRelay`.
2. Confirmed the source repo has no license metadata on GitHub and no local license file.
3. Set the private repo as `origin` and disabled push to the upstream remote.
4. Verified the Python files compile before behavior changes.
5. Read the Prompt Relay project page.
6. Read the arXiv abstract and HTML paper for `2604.10030`.
7. Recorded the core method as a cross-attention logit penalty that becomes a multiplicative attention prior through `exp(-C)`.
8. Found the paper equation for boundary decay.
9. Found the paper experiment value `epsilon = 0.1`.
10. Found that the current node default uses `epsilon = 1e-3`.
11. Found that `prompt_relay.py` uses `sigma = 1 / ln(1 / epsilon)`.
12. Found that the paper gives `sigma = (L - w) / sqrt(2 ln(1 / epsilon))`.
13. Identified `prompt_relay.py build_segments` as the first code target.
14. Identified `nodes.py _convert_to_latent_lengths` as a likely timing mismatch target.
15. Identified node tooltip wording that disagrees with the paper default.
16. Wrote the first research log in `docs/research/research-log.md`.
17. Wrote the first experiment log in `docs/experiments/experiment-log.md`.
18. Added the project protocol in `docs/project-protocol.md`.
19. Recorded the commit cadence rule for local signed commits only.
20. Recorded the rule to avoid pushing without explicit approval.
21. Reviewed GitHub issues for the ComfyUI node.
22. Found user reports around Wan 2.2 workflow confusion.
23. Found user reports around timeline timing mismatch.
24. Found user reports around advanced options disappearing from node mappings.
25. Found user reports around multi-image injection causing corruption.
26. Found community interest in segment-indexed LoRA gates.
27. Searched Reddit for Prompt Relay and LTX 2.3 usage.
28. Found that users report better event and scene control with Prompt Relay.
29. Found that users still report blur and smearing during fast motion, full-body motion, distant subjects, chase scenes, and fight scenes.
30. Interpreted long generation quality loss as a combined problem across prompt interference, token budget pressure, backbone motion limits, temporal drift, boundary timing, and inconsistent persistent context.
31. Created EXP-001 for opt-in paper-correct sigma mode.
32. Created EXP-002 for epsilon comparison against the paper value.
33. Created EXP-003 for short-segment decay stabilization.
34. Created EXP-004 for token budget warnings.
35. Added a pure long-form chunk planner to compute memory-bounded Prompt Relay windows from latent frames, tokens/frame, text tokens, overlap, and the mask safety cap.
36. Documented that this planner is a hard mask-memory bound, not a visual-quality target; quality-oriented chunks should remain much shorter until controlled renders prove otherwise.
35. Confirmed no runtime behavior has changed yet.
36. Confirmed the next implementation step needs Sarp approval for EXP-001.
37. Created this ML research journal as the chronological record for discoveries, decisions, and experiments.

38. Set project note timestamps to Pacific time and avoid EST for this project unless quoting an external source.
39. Re-anchored this journal as the simple chronological record for the research goal of improving this repo toward higher quality and less degradation over longer generation time.
40. Confirmed project protocol now requires Pacific time labels and PDT applies for 2026-05-14.
41. Tried web search for Prompt Relay and long video degradation sources but the configured SearXNG search backend is unavailable in this subagent.
42. Used direct web fetch instead and reread the arXiv HTML paper.
43. Reconfirmed the paper frames the degradation problem as temporal semantic entanglement from conditioning every frame on the whole prompt.
44. Reconfirmed the paper compares against MinT, MEVG, DiTCtrl, TS-Attn, and SwitchCraft as related multi-event or attention-control methods.
45. Reconfirmed the paper reports Prompt Relay on Wan 2.2 and says experiments use epsilon 0.1 with a global prompt for persistent context.
46. Fetched GitHub issue 7 and confirmed the user report says timeline segment lengths do not seem to line up with output while the regular node dialogue timing is better.
47. Fetched GitHub issue 13 and confirmed community interest in segment-indexed LoRA gates tied to relay prompt sections.
48. Used GitHub CLI to list current issues and confirmed open reports for Wan 2.2 integration, advanced options registration, None input handling, timeline mismatch, second based timings, WanVideoWrapper integration, multi image corruption, and early positive feedback.
49. Inspected the unexpected parser.py change and tests/test_parser.py.
50. Classified the parser.py regex edit as unapproved behavior-changing parser robustness work because it changes accepted inline tag syntax by allowing whitespace and en dash ranges.
51. Classified tests/test_parser.py as matching unapproved test coverage for that parser experiment.
52. Left the parser.py and tests/test_parser.py changes in place and did not delete, commit, or push them.
53. Identified the next safest approval packet as documentation plus opt-in math tests first, then Sarp approval before any runtime experiment.
54. Tried verification with python but this machine does not have a python executable on PATH.
55. Re-ran verification with python3 and parser unittest discovery passed 4 tests.
56. Re-ran python3 bytecode compilation for parser, prompt_relay, nodes, smart_nodes, patches, and advanced_options with no compile errors.
57. Received external Prompt Relay and infinite video notes from Sarp.
58. Saved the external notes into `docs/research/external-notes/prompt-relay-infinite-video-notes-2026-05-14-pdt.txt`.
59. Extracted the main usable direction as Prompt Relay should be treated as the timing layer inside a chunked closed loop system.
60. Marked persistent visual memory, clean keyframe banks, overlap, candidate generation, and quality gates as research paths for reducing quality decay over longer generation time.
61. Verified the upstream Prompt Relay README and arXiv PDF metadata.
62. Verified the Kijai ComfyUI README guidance that global prompt anchors persistent details while local segments describe changes.
63. Verified the LTX 2.3 prompt guide claim that longer videos need enough prompt detail to fill duration.
64. Verified Video Infinity as local plus global context frame research rather than a direct Prompt Relay feature.
65. Verified FreeNoise as noise rescheduling research for longer video generation.
66. Verified FreeLong and FreeLong plus plus as global and local feature or frequency fusion research.
67. Verified MAG as a memory compression direction for long term video consistency.
68. Verified StableWorld as a clean frame or frame eviction direction for stable long interactive video.
69. Classified the external memo as directionally useful but partly research grade and not yet directly implementable in this repo.
70. Added EXP 007 for closed loop chunk architecture design.
71. Added EXP 008 for a clean visual memory bank.
72. Added EXP 009 for a quality gate scorer prototype.
73. Added EXP 010 for candidate generation and overlap workflow.
74. Kept parser.py and tests/test_parser.py classified as unapproved parser robustness work.
74. Added isolated deterministic sigma math tests for EXP-001A without changing runtime defaults.
75. Pinned the paper endpoint condition that retained prior equals epsilon at the calibrated boundary.
76. Pinned the paper `w=L-2` implication that sigma is constant for fixed epsilon when the penalty distance is 2.
77. Characterized current sigma as sharper than the paper equation at the same epsilon, leaving runtime behavior unchanged.
78. Added a tested scheduler-facing manifest helper for quality chunk stream steps so future ComfyUI runners can consume render decision, chunk ranges, output append range, prompt windows, anchors, crossfade metadata, sanitized checkpoint state, and diagnostics from one compact payload.
79. Verified the focused manifest test and full unittest discovery; 172 tests pass locally on 2026-05-16 PDT.
