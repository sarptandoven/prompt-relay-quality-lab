# Project protocol

## Commit cadence

- Work in small units.
- After roughly every 3,500 tokens of meaningful research, analysis, or implementation work, add the changes to the commit backlog or create a local git commit if the cadence window is open.
- Keep each commit to no more than 220 changed lines.
- Space local commits at least 23 minutes apart and no more than 41 minutes apart while active work is waiting to be committed.
- Do not create more than 40 commits in one day.
- Do not push unless Sarp explicitly asks.
- Keep research findings and experiment notes in the repo before or immediately after the work:
  - `docs/research/research-log.md`
  - `docs/experiments/experiment-log.md`
- For behavior-changing experiments, record the hypothesis, paper/math basis, implementation plan, and test plan before modifying runtime code.

## Current approval rule

- Experiments must be run by Sarp before implementation.
- Local commits are allowed on the cadence above, but pushes are not allowed without explicit approval.
- Commits must be signed/verified. After every commit, run `git log -1 --show-signature` or `git verify-commit HEAD` before reporting it as complete.

## Time zone

- Use Pacific time for project notes, research logs, experiment logs, and status updates. Label entries as PST or PDT according to the actual date. Do not use EST unless quoting an external source.
