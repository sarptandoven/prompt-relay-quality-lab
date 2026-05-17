# Custom node quickstart

This repo can be used directly as a ComfyUI custom node checkout. It can also provide one supplemental drop-in file, `prompt_relay_lab_node.py`, for an existing Prompt Relay node pack.

## Option A: install this repo as the full custom node pack

Use this when you want the complete local pack with the baseline nodes, Smart node, Advanced Options node, Timeline node, and Lab node.

From the ComfyUI install directory:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/sarptandoven/prompt-relay-quality-lab.git ComfyUI-PromptRelay
cd ComfyUI-PromptRelay
python3 -m pip install -r requirements.txt
```

If this repo is already local on Sarp's Mac mini, symlink it instead of cloning:

```bash
cd /path/to/ComfyUI/custom_nodes
ln -s /Users/sarpdoven/.openclaw/workspace/prompt-relay-quality-lab ComfyUI-PromptRelay
cd ComfyUI-PromptRelay
python3 -m pip install -r requirements.txt
```

Restart ComfyUI after installing. A terminal launch is easiest for checking logs:

```bash
cd /path/to/ComfyUI
python3 main.py
```

## Option B: install only the lab node into an existing Prompt Relay node pack

Use this when an upstream `ComfyUI-PromptRelay` checkout is already installed and you only want the supplemental A/B-test node. The file is independent as a node file inside the pack: it has its own Comfy node class and node mappings, but it intentionally imports the pack's existing `patches.py` and `prompt_relay.py` helpers, so it must live beside those files.

Expected target layout after copying:

```text
/path/to/ComfyUI/custom_nodes/ComfyUI-PromptRelay/
  __init__.py
  nodes.py
  patches.py
  prompt_relay.py
  prompt_relay_lab_node.py   <-- copied file
```

Copy the file:

```bash
cp /Users/sarpdoven/.openclaw/workspace/prompt-relay-quality-lab/prompt_relay_lab_node.py \
  /path/to/ComfyUI/custom_nodes/ComfyUI-PromptRelay/prompt_relay_lab_node.py
```

Register it in the target pack's `__init__.py`.

Add this import near the other node imports:

```python
from .prompt_relay_lab_node import PromptRelayLabEncode
```

Add `PromptRelayLabEncode` to the `PromptRelay.get_node_list()` return list:

```python
return [
    PromptRelayEncode,
    PromptRelayEncodeTimeline,
    PromptRelaySmartEncode,
    PromptRelaySmartEncodeTest,
    PromptRelayAdvancedOptions,
    PromptRelayLabEncode,
]
```

Add these mappings near the existing mappings:

```python
NODE_CLASS_MAPPINGS["PromptRelayLabEncode"] = PromptRelayLabEncode
NODE_DISPLAY_NAME_MAPPINGS["PromptRelayLabEncode"] = "Prompt Relay Encode (Lab)"
```

If the target `__init__.py` uses literal dictionaries instead of post-definition assignments, add the same key/value pair inside each dictionary:

```python
NODE_CLASS_MAPPINGS = {
    # existing nodes...
    "PromptRelayLabEncode": PromptRelayLabEncode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # existing nodes...
    "PromptRelayLabEncode": "Prompt Relay Encode (Lab)",
}
```

The full repo already performs this registration. These manual edits are only needed for file-only copying into another node pack.

## Nodes to use

Primary production node:

- `Prompt Relay Encode`

Optional tuning node:

- `Prompt Relay Advanced Options`

Experimental A/B node:

- `Prompt Relay Encode (Lab)`

Timeline node:

- `Prompt Relay Encode (Timeline)`

## Basic wiring

Use Prompt Relay between the loaded/patched model and the sampler path:

```text
UNET / LoRA / SageAttention
  -> Prompt Relay Encode
  -> LTX2SamplingPreviewOverride or equivalent sampler wrapper
  -> sampler
```

Wire the Prompt Relay `positive` output into the matching positive conditioning input. Use a normal `CLIPTextEncode` node for the negative prompt.

For LTX 2.3 AV two-pass workflows, patch each pass that should use Prompt Relay. The provided fixture had one Prompt Relay node before the first pass and one before the refine pass.

## Prompt fields

`global_prompt`:

- Persistent identity, style, scene, wardrobe, lighting, camera language, and quality anchor.
- Keep stable across chunks for long-form work.

`local_prompts`:

- Ordered temporal beats separated by `|`.
- Only describe what changes in each beat.

Example:

```text
person turns toward camera and starts speaking | person raises the object into frame | person lowers the object and smiles
```

`segment_lengths`:

- Empty: even distribution.
- Frame counts: `84,133`.
- Percentages: `45%,55%`.
- Ratios: `0.45,0.55`.

For long-form chunked workflows, prefer percentages or ratios per chunk unless exact beat timings are known.

## Recommended long-form starting settings

For 2500+ frame projects, do not run one giant 2500-frame latent generation. Use chunks with overlap.

Starting point:

```text
chunk size: 129 to 257 frames
overlap: 16 to 32 frames
segments per chunk: 2 to 6
epsilon: 0.1 to 0.3
sigma_mode: paper_boundary
video_strength: 0.7 to 1.0
video_window_scale: 1.0 to 1.3
audio_epsilon: 0.1 or inherit
audio_strength: 0.6 to 1.0
audio_window_scale: 1.0 to 1.3
audio_frame_offset_frames: 0 unless measured lag exists
```

Keep a consistent global prompt across chunks and only change local prompts for the current chunk's action/dialogue.

## Sanity check after install

Start ComfyUI and confirm these nodes appear under `conditioning/prompt_relay`:

- `Prompt Relay Encode`
- `Prompt Relay Encode (Timeline)`
- `Prompt Relay Encode (Smart)`
- `Prompt Relay Smart Encode Test`
- `Prompt Relay Advanced Options`
- `Prompt Relay Encode (Lab)`

Run a short 2-segment test first:

1. Create a tiny latent video, not a long production run.
2. Add `Prompt Relay Encode (Lab)` between the loaded model path and the sampler path.
3. Connect `model`, `clip`, and `latent`.
4. Set `global_prompt` to the stable scene/style description.
5. Set `local_prompts` to two beats separated by `|`, for example:

   ```text
   person turns toward camera | person raises one hand and smiles
   ```

6. Leave `segment_lengths` empty for the first test, or use `50%,50%`.
7. Leave `sigma_mode=upstream_compat` for a baseline run.
8. Run once and check the ComfyUI console.

Expected Lab-node log line:

```text
[PromptRelayLab] arch=... latent_frames=... tokens_per_frame=... timing='auto-even' segment_lengths=[...] sigma_mode=upstream_compat ...
```

The standard node logs this shape instead:

```text
[PromptRelay] Setup: arch=... latent_frames=... tokens_per_frame=... timing='...' epsilon=... options=...
```

If these logs are missing, the sampler is probably not using the patched model from Prompt Relay. Re-check that the Prompt Relay `model` output feeds the sampler-side model input, not just the conditioning path.

## Exact local verification

From this repo root:

```bash
cd /Users/sarpdoven/.openclaw/workspace/prompt-relay-quality-lab
python3 -m unittest discover -s tests -v
```

Current local gate: 161 tests passing.

For a file-only copy into another node pack, the minimum import/registration check is:

```bash
cd /path/to/ComfyUI/custom_nodes/ComfyUI-PromptRelay
python3 - <<'PY'
import ast
from pathlib import Path

init_tree = ast.parse(Path('__init__.py').read_text())
lab_tree = ast.parse(Path('prompt_relay_lab_node.py').read_text())

init_text = Path('__init__.py').read_text()
lab_text = Path('prompt_relay_lab_node.py').read_text()

assert 'PromptRelayLabEncode' in init_text, 'Lab node is not registered in __init__.py'
assert 'NODE_CLASS_MAPPINGS' in lab_text, 'Lab file does not expose node mappings'
assert 'NODE_DISPLAY_NAME_MAPPINGS' in lab_text, 'Lab file does not expose display mappings'
assert any(getattr(node, 'name', None) == 'PromptRelayLabEncode' for node in ast.walk(lab_tree)), 'Lab node class missing'
print('PromptRelayLabEncode file and registration look present')
PY
```

Then restart ComfyUI and confirm `Prompt Relay Encode (Lab)` appears in the node menu.
