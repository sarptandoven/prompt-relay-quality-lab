# Drop-in node direction

This repo is a small ComfyUI Prompt Relay project repo, not a production workflow repo.

Primary artifact:

- `prompt_relay_lab_node.py`

Use case:

- Use the repo as an installable ComfyUI custom node checkout, or copy the file into an existing `ComfyUI-PromptRelay` checkout.
- If copying only the file, register `PromptRelayLabEncode` in that package's `__init__.py` to add a supplemental node.
- Keep the upstream nodes available for baseline comparison.
- For full install/run instructions, see [custom-node-quickstart.md](custom-node-quickstart.md).

## Why this shape

The upstream ComfyUI package already has the core integration points:

- `prompt_relay.py` builds temporal cross-attention masks.
- `patches.py` patches Wan and LTX cross-attention modules.
- `nodes.py` exposes the standard and timeline ComfyUI nodes.
- `smart_nodes.py` adds prompt parsing convenience.

The lab file reuses those integration points instead of building a separate workflow system.

## What the lab node adds

`Prompt Relay Encode (Lab)` keeps upstream-compatible behavior by default, then exposes explicit LTX 2.3 AV controls:

- `sigma_mode=upstream_compat`: matches current ComfyUI-PromptRelay behavior, `sigma = 1 / ln(1 / epsilon)`.
- `sigma_mode=paper_boundary`: follows Prompt Relay Eq. 4, `sigma = (L - w) / sqrt(2 ln(1 / epsilon))`.
- `segment_lengths` accepts frame counts, percentages like `35%,65%`, or ratios like `0.35,0.65`.
- Audio-only controls tune the scaled LTX AV audio attention route: `audio_epsilon`, `audio_strength`, `audio_window_scale`, and `audio_frame_offset_frames`.

This lets us compare current Kijai/Gordon behavior against the paper-calibrated boundary decay and audio timing offsets without silently changing defaults.

## Registration snippet for file-only copying

This repo's `__init__.py` already registers the lab node. If copying only `prompt_relay_lab_node.py` into another checkout, copy it into the Prompt Relay package directory beside `patches.py` and `prompt_relay.py`, then add this inside upstream `__init__.py`:

```python
from .prompt_relay_lab_node import PromptRelayLabEncode
```

Add `PromptRelayLabEncode` to `PromptRelay.get_node_list()`.

Add mappings:

```python
NODE_CLASS_MAPPINGS["PromptRelayLabEncode"] = PromptRelayLabEncode
NODE_DISPLAY_NAME_MAPPINGS["PromptRelayLabEncode"] = "Prompt Relay Encode (Lab)"
```

The lab file is independent as a Comfy node file inside the node pack: it defines its own node class and mappings. It is not a standalone script outside the pack because it uses package-relative imports from the existing Prompt Relay implementation.

## Current verification

Run:

```bash
python3 -m unittest discover -s tests -v
```

Latest local result: 161 tests OK.
