# Commit backlog

Local commits only. Do not push unless Sarp explicitly asks.

Rules:

- Keep each commit at or below 220 changed lines.
- Space commits 23 to 41 minutes apart while there is approved work waiting.
- Stop at 40 commits per day.
- Verify signing after each commit with `git log -1 --show-signature` or `git verify-commit HEAD`.
- Show Sarp the final commit message before committing.

## Ready for next commit after message approval

Proposed message:

```text
feat: expose quality chunk render manifest
```

Suggested staged packet:

- `prompt_relay.py`: `build_quality_chunk_render_manifest(step)` helper only.
- `tests/test_prompt_relay_segments.py`: focused manifest test only.
- Optional docs lines from `docs/current-status.md` and `docs/research/ml-research-journal.md` if the staged hunk still remains below 220 changed lines.

Verification already run:

```bash
python3 -m unittest tests.test_prompt_relay_segments.PromptRelaySegmentMetadataTests.test_quality_chunk_render_manifest_exposes_scheduler_ready_payload -v
python3 -m unittest discover -s tests -v
```

Latest result: 172 tests passing on 2026-05-16 PDT.
