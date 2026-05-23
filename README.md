# ComfyUI Prompt-Relay

WORK IN PROGRESS

<img width="1486" height="1022" alt="image" src="./assets/timeline_node.png" />


Original project:

https://gordonchen19.github.io/Prompt-Relay/

---

## Prompt Relay Encode (Smart)

<img width="1486" alt="image" src="./assets/smart_nodes.png" />

The Smart node accepts a single `smart_prompt` field and automatically calculates how to distribute your video's latent frames across segments. No manual frame counting required.

There are two syntax styles. Pick one per prompt — do not mix them.

---

### Syntax 1 — Inline (pipe-separated)

Segments are separated by `|`. Weights are optional. If omitted, all segments get equal time.

**Equal distribution:**
```
A man walks through a forest | He stops and looks around | He runs back
```

**Proportional distribution using `[start-end]` tags:**
```
A man walks through a forest [0-50] | He stops and looks around [50-150] | He runs back [150-200]
```

The numbers inside `[ ]` are not frame counts — they are relative positions on an arbitrary scale. Only the span of each range matters. In the example above the total span is 200, so the first segment gets 25%, the second 50%, and the third 25% of the video regardless of how many actual frames the video has.

You can also use a single number `[50]` as a plain weight instead of a range.

---

### Syntax 2 — Block (newline headers)

Each segment is preceded by a header line on its own line. The header is any words followed by a number and a colon. The header line is stripped entirely before encoding — it never reaches the tokenizer.

**Equal distribution (ordinal headers):**
```
Scene 1:
A man walks through a forest
Scene 2:
He stops and looks around
Scene 3:
He runs back
```

A single number in the header (`Scene 1:`, `Part 4:`, `segment 9:`) is treated as a sequence marker and gives every segment equal weight.

**Proportional distribution (range headers):**
```
Scene 1-2:
A man walks through a forest
Scene 2-5:
He stops and looks around
Scene 5-6:
He runs back
```

When the header contains a range (`1-2`, `2-5`, `5-6`), the span of the range (`1`, `3`, `1`) becomes the segment's proportional weight. Here the second segment gets three times as much screen time as the first and third.

The prefix words before the number are completely flexible — `Scene`, `Part`, `Shot`, `Segment`, `Second`, `Beat`, `Chapter`, `Step`, or any word you choose. Only the number and colon matter structurally.

If the `word2number` Python package is installed (`pip install word2number`), the number can also be written as a word: `Scene eleven:`, `Part twenty one:`.

---

### The `global_prompt` field

`global_prompt` anchors persistent details across the entire video — style, character description, lighting, quality tags. Leave it empty and the node will automatically use the first segment's text as the global anchor.

**Recommended pattern:** put your establishing description in segment 1, leave `global_prompt` blank, and use subsequent segments for action and motion changes only.

---

### The `normalize_by_tokens` toggle

When enabled, each segment's weight is multiplied by its CLIP token count before frame distribution. This means segments with more tokens are allocated proportionally more latent frames. Off by default.

---

### Writing prompts with a VLM or LLM

Paste the raw output directly into `smart_prompt`. The templates below are instruction blocks you send to the model. The model needs no knowledge of this node or how it works. Replace `[PLACEHOLDERS]` with your values before sending.

The first segment is automatically used as the global anchor for the entire video. It must therefore contain only a static description of what is directly visible in the provided content. The model must not add, infer, or invent anything not explicitly present in the input. All subsequent segments must describe only what changes during that period and must not re-establish what was already described in segment 1.

---

**Block syntax:**

```
Write a video prompt with exactly [NUMBER] segments describing [YOUR SUBJECT/SCENARIO].

Rules:
1. Begin each segment with a header on its own line. Format: one word, a space, a number, a colon. Example: Scene 1:
2. Write the segment text on the line after the header.
3. Segment 1 describes only the static visible state of the provided input. No motion. No action. Nothing not present in the input.
4. Each segment after segment 1 describes only what is changing or moving during that period. Do not repeat anything from segment 1.
5. Output only the formatted segments. Nothing else.
```

**Inline syntax:**

```
Write a video prompt with exactly [NUMBER] segments describing [YOUR SUBJECT/SCENARIO].

Rules:
1. Write all segments on one line separated by |
2. Segment 1 describes only the static visible state of the provided input. No motion. No action. Nothing not present in the input.
3. Each segment after segment 1 describes only what is changing or moving during that period. Do not repeat anything from segment 1.
4. Output only the single line. Nothing else.
```
