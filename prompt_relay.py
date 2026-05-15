import logging
import math
import os
import torch

log = logging.getLogger(__name__)
DEFAULT_MAX_MASK_ELEMENTS = 1_073_741_824


def _dtype_bytes(dtype):
    text = str(dtype).lower()
    if "64" in text or "double" in text:
        return 8
    if "16" in text or "half" in text or "bfloat" in text:
        return 2
    return 4


def _max_mask_elements():
    raw = os.environ.get("PROMPT_RELAY_MAX_MASK_ELEMENTS", str(DEFAULT_MAX_MASK_ELEMENTS))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_MAX_MASK_ELEMENTS
    return max(0, value)


def _check_mask_size(Lq, Lk, dtype, mode):
    elements = int(Lq) * int(Lk)
    cap = _max_mask_elements()
    if cap and elements > cap:
        est_gib = elements * _dtype_bytes(dtype) / (1024 ** 3)
        cap_gib = cap * _dtype_bytes(dtype) / (1024 ** 3)
        max_query_tokens = max(cap // max(int(Lk), 1), 1)
        raise RuntimeError(
            "PromptRelay: refusing to build a {mode} attention mask with "
            "Lq={Lq:,}, Lk={Lk:,} ({elements:,} elements, ~{est_gib:.2f} GiB at {dtype}). "
            "This usually means the workflow is trying to Prompt Relay a long video as one full attention pass. "
            "For 2500+ frame generation, split the video into temporal chunks/windows and apply Prompt Relay per chunk. "
            "At the current text-token length, the safety cap allows about {max_query_tokens:,} query tokens per chunk. "
            "Use plan_temporal_chunks(...) in this module to size latent-frame chunks from tokens/frame, or set "
            "PROMPT_RELAY_MAX_MASK_ELEMENTS=0 only if you have deliberately budgeted the VRAM/RAM. "
            "Current safety cap is {cap:,} elements (~{cap_gib:.2f} GiB at this dtype)."
            .format(
                mode=mode,
                Lq=Lq,
                Lk=Lk,
                elements=elements,
                est_gib=est_gib,
                dtype=dtype,
                max_query_tokens=max_query_tokens,
                cap=cap,
                cap_gib=cap_gib,
            )
        )
    return elements


def estimate_max_chunk_frames(tokens_per_frame, text_tokens, max_mask_elements=None, safety_margin=0.9):
    """Estimate the largest latent-frame chunk that fits the additive mask budget.

    Prompt Relay's additive temporal mask is shaped ``[query_tokens, text_tokens]``.
    For video attention, ``query_tokens = latent_frames * tokens_per_frame``. This
    helper is intentionally pure and runtime-neutral so long-form schedulers can
    plan finite windows without changing the default one-pass Comfy node behavior.
    """
    if max_mask_elements is None:
        max_mask_elements = _max_mask_elements()
    if max_mask_elements == 0:
        return math.inf
    try:
        tokens_per_frame = int(tokens_per_frame)
        text_tokens = int(text_tokens)
        safety_margin = float(safety_margin)
    except (TypeError, ValueError) as exc:
        raise ValueError("PromptRelay: chunk planner inputs must be numeric.") from exc
    if tokens_per_frame <= 0 or text_tokens <= 0:
        raise ValueError("PromptRelay: tokens_per_frame and text_tokens must be positive to plan chunks.")
    if not math.isfinite(safety_margin) or safety_margin <= 0:
        raise ValueError("PromptRelay: safety_margin must be positive and finite.")

    budget = max(1, int(max_mask_elements * min(safety_margin, 1.0)))
    elements_per_frame = tokens_per_frame * text_tokens
    if elements_per_frame > budget:
        raise ValueError(
            "PromptRelay: even one latent frame would exceed the mask budget "
            f"({elements_per_frame:,} elements/frame > {budget:,} budgeted elements). "
            "Reduce tokens_per_frame/text_tokens, raise PROMPT_RELAY_MAX_MASK_ELEMENTS, or disable the cap only after VRAM budgeting."
        )
    return max(1, budget // elements_per_frame)


def plan_temporal_chunks(latent_frames, tokens_per_frame, text_tokens, max_mask_elements=None, overlap_frames=4, safety_margin=0.9):
    """Return overlapping latent-frame chunks for long-form Prompt Relay windows.

    Chunks use half-open ``[start, end)`` latent-frame ranges. The plan is safe for
    arbitrarily long timelines because each chunk is bounded by the configured
    mask element cap; callers can stitch windows with overlap/crossfade outside
    this module.
    """
    if latent_frames <= 0:
        return {"max_chunk_frames": 0, "overlap_frames": 0, "chunks": []}

    max_chunk_frames = estimate_max_chunk_frames(
        tokens_per_frame,
        text_tokens,
        max_mask_elements=max_mask_elements,
        safety_margin=safety_margin,
    )
    if math.isinf(max_chunk_frames) or max_chunk_frames >= latent_frames:
        return {
            "max_chunk_frames": latent_frames,
            "overlap_frames": 0,
            "chunks": [{"start": 0, "end": int(latent_frames), "length": int(latent_frames)}],
        }

    max_chunk_frames = int(max_chunk_frames)
    overlap = min(max(int(overlap_frames), 0), max_chunk_frames - 1)
    step = max_chunk_frames - overlap
    chunks = []
    start = 0
    while start < latent_frames:
        end = min(start + max_chunk_frames, int(latent_frames))
        chunks.append({"start": start, "end": end, "length": end - start})
        if end >= latent_frames:
            break
        start += step

    return {"max_chunk_frames": max_chunk_frames, "overlap_frames": overlap, "chunks": chunks}


def plan_chunk_stitch_ranges(chunks):
    """Return deterministic keep/trim ranges for overlapping chunk outputs.

    The chunk planner intentionally emits overlapping input windows so the caller
    can carry context across a boundary. This helper converts those windows into
    non-overlapping half-open output ranges by splitting each overlap at its
    midpoint. The returned rows keep the original chunk range plus global
    ``keep_start``/``keep_end`` and local ``trim_start``/``trim_end`` counts.

    It is pure bookkeeping: it does not blend frames or alter Prompt Relay masks.
    Use the discarded overlap tails/heads for crossfade or continuity checks in
    the stitching layer.
    """
    if not chunks:
        return []

    normalized = []
    for idx, chunk in enumerate(chunks):
        try:
            start = int(chunk["start"])
            end = int(chunk["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("PromptRelay: chunk stitch ranges require start/end integers.") from exc
        if end <= start:
            raise ValueError("PromptRelay: chunk stitch ranges require positive-length chunks.")
        if normalized and start < normalized[-1]["start"]:
            raise ValueError("PromptRelay: chunks must be ordered by start frame.")
        if normalized and start > normalized[-1]["end"]:
            raise ValueError("PromptRelay: chunks must be contiguous or overlapping to stitch.")
        normalized.append({"index": idx, "start": start, "end": end, "length": end - start})

    boundaries = [normalized[0]["start"]]
    for previous, current in zip(normalized, normalized[1:]):
        if current["start"] >= previous["end"]:
            boundary = previous["end"]
        else:
            boundary = (current["start"] + previous["end"]) // 2
        boundaries.append(boundary)
    boundaries.append(normalized[-1]["end"])

    stitched = []
    for idx, chunk in enumerate(normalized):
        keep_start = max(chunk["start"], boundaries[idx])
        keep_end = min(chunk["end"], boundaries[idx + 1])
        if keep_end <= keep_start:
            raise ValueError("PromptRelay: overlap is too large to assign non-empty stitch ranges.")
        stitched.append({
            **chunk,
            "keep_start": keep_start,
            "keep_end": keep_end,
            "keep_length": keep_end - keep_start,
            "trim_start": keep_start - chunk["start"],
            "trim_end": chunk["end"] - keep_end,
        })

    return stitched



def build_temporal_cost(q_token_idx, Lq, Lk, device, dtype, tokens_per_frame):
    """Gaussian penalty matrix [Lq, Lk] for video cross-attention (integer frame indexing)."""
    offset = torch.zeros(Lq, Lk, device=device, dtype=dtype)
    query_frames = torch.arange(Lq, device=device, dtype=torch.long) // tokens_per_frame

    for seg in q_token_idx:
        local = seg["local_token_idx"].to(device=device)
        d = (query_frames.float()[:, None] - seg["midpoint"]).abs()
        strength = seg.get("strength", 1.0)
        cost = strength * (torch.relu(d - seg["window"]) ** 2) / (2 * seg["sigma"] ** 2)
        offset[:, local] = cost.to(offset.dtype)

    return offset


def build_temporal_cost_scaled(q_token_idx, Lq, Lk, device, dtype, latent_frames):
    """Penalty matrix for queries that don't map to integer frames (e.g. LTXAV audio tokens)."""
    offset = torch.zeros(Lq, Lk, device=device, dtype=dtype)
    query_frames = torch.arange(Lq, device=device, dtype=torch.float32) * latent_frames / Lq

    for seg in q_token_idx:
        local = seg["local_token_idx"].to(device=device)
        d = (query_frames[:, None] - seg["midpoint"]).abs()
        sigma_a = seg.get("sigma_audio", seg["sigma"])
        window_a = seg.get("window_audio", seg["window"])
        strength_a = seg.get("strength_audio", 1.0)
        cost = strength_a * (torch.relu(d - window_a) ** 2) / (2 * sigma_a ** 2)
        offset[:, local] = cost.to(offset.dtype)

    return offset


def create_mask_fn(q_token_idx, fallback_tokens_per_frame, latent_frames):
    """Closure: mask_fn(q, k, transformer_options) -> additive mask or None."""
    cache = {}
    max_token_idx = max(int(seg["local_token_idx"].max().item()) for seg in q_token_idx) + 1

    def mask_fn(q, k, transformer_options):
        Lq, Lk = q.shape[1], k.shape[1]

        if Lq == Lk:
            return None

        # Only apply on conditional pass — not unconditional (negative prompt)
        cond_or_uncond = transformer_options.get("cond_or_uncond", [])
        if 1 in cond_or_uncond and 0 not in cond_or_uncond:
            return None

        grid_sizes = transformer_options.get("grid_sizes", None)
        video_tpf = int(grid_sizes[1]) * int(grid_sizes[2]) if grid_sizes is not None else fallback_tokens_per_frame
        video_lq = latent_frames * video_tpf

        # Skip cross-modal attention — text keys are padded to a fixed length ≥ max_token_idx and != video_lq
        if Lk == video_lq or Lk < max_token_idx:
            return None

        mode = "video" if Lq == video_lq else "scaled"

        key = (Lq, Lk, mode, q.device)
        if key not in cache:
            if mode == "video":
                cost = build_temporal_cost(q_token_idx, Lq, Lk, q.device, q.dtype, video_tpf)
            else:
                cost = build_temporal_cost_scaled(q_token_idx, Lq, Lk, q.device, q.dtype, latent_frames)
            log.info(
                "[PromptRelay] Built penalty matrix (%s): Lq=%d, Lk=%d, nonzero=%d/%d",
                mode, Lq, Lk, (cost > 0).sum().item(), cost.numel(),
            )
            cache[key] = -cost

        return cache[key].to(q.dtype)

    return mask_fn


def build_segments(token_ranges, segment_lengths, epsilon=1e-3, relay_options=None):
    """Per-segment metadata for the temporal penalty.

    relay_options (optional dict) overrides per-stream knobs:
        video_strength, video_window_scale,
        audio_epsilon, audio_strength, audio_window_scale
    Audio knobs only affect architectures whose cross-attention takes the scaled
    (non-integer-frame) path — currently LTX audio_attn2.
    """
    # Paper uses constant sigma = 1/ln(1/epsilon) regardless of segment length
    sigma = 1.0 / math.log(1.0 / epsilon) if 0 < epsilon < 1 else 0.1448

    opts = relay_options or {}
    v_strength = opts.get("video_strength", 1.0)
    v_window_scale = opts.get("video_window_scale", 1.0)
    a_epsilon = opts.get("audio_epsilon")
    a_strength = opts.get("audio_strength", 1.0)
    a_window_scale = opts.get("audio_window_scale", 1.0)

    if a_epsilon is not None and 0 < a_epsilon < 1:
        sigma_audio = 1.0 / math.log(1.0 / a_epsilon)
    else:
        sigma_audio = sigma

    if relay_options:
        log.info(
            "[PromptRelay] Advanced options active — video: strength=%.3f window_scale=%.3f | "
            "audio: epsilon=%s strength=%.3f window_scale=%.3f",
            v_strength, v_window_scale,
            f"{a_epsilon:.4f}" if a_epsilon is not None else "inherit",
            a_strength, a_window_scale,
        )

    q_token_idx = []
    frame_cursor = 0

    for (tok_start, tok_end), L in zip(token_ranges, segment_lengths):
        if L <= 0:
            frame_cursor += L
            continue
        midpoint = (2 * frame_cursor + L) // 2
        base_window = max(L // 2 - 2, 0)
        q_token_idx.append({
            "local_token_idx": torch.arange(tok_start, tok_end),
            "midpoint": midpoint,
            "window": max(base_window * v_window_scale, 0.0),
            "sigma": sigma,
            "strength": v_strength,
            "window_audio": max(base_window * a_window_scale, 0.0),
            "sigma_audio": sigma_audio,
            "strength_audio": a_strength,
        })
        frame_cursor += L

    return q_token_idx


def get_raw_tokenizer(clip):
    """Extract the raw SPiece/HF tokenizer from a ComfyUI CLIP object."""
    tokenizer_wrapper = clip.tokenizer
    for attr_name in dir(tokenizer_wrapper):
        if attr_name.startswith("_"):
            continue
        inner = getattr(tokenizer_wrapper, attr_name, None)
        if inner is not None and hasattr(inner, "tokenizer"):
            return inner.tokenizer

    raise RuntimeError(
        f"Could not find raw tokenizer on CLIP object. "
        f"Known attributes: {[a for a in dir(tokenizer_wrapper) if not a.startswith('_')]}"
    )


def map_token_indices(raw_tokenizer, global_prompt, local_prompts):
    """Tokenize global + space-prefixed locals; return (full_prompt, per-local token ranges).

    Uses incremental tokenization to avoid SentencePiece context-dependency issues.
    """
    prefixed_locals = [" " + lp for lp in local_prompts]
    full_prompt = global_prompt + "".join(prefixed_locals)
    has_eos = getattr(raw_tokenizer, "add_eos", False)
    eos_adj = 1 if has_eos else 0

    prev_len = len(raw_tokenizer(global_prompt)["input_ids"]) - eos_adj
    token_ranges = []
    built = global_prompt

    for plp in prefixed_locals:
        built += plp
        cur_len = len(raw_tokenizer(built)["input_ids"]) - eos_adj
        if cur_len <= prev_len:
            raise ValueError(f"Local prompt produced no tokens: '{plp.strip()}'")
        token_ranges.append((prev_len, cur_len))
        prev_len = cur_len

    return full_prompt, token_ranges


def distribute_segment_lengths(num_segments, latent_frames, specified_lengths=None):
    """Validate or auto-distribute segment frame counts, capped to fit within latent_frames."""
    if specified_lengths:
        if len(specified_lengths) != num_segments:
            raise ValueError(
                f"Number of segment_lengths ({len(specified_lengths)}) "
                f"must match number of local prompts ({num_segments})"
            )
        lengths = specified_lengths
    else:
        # ceil division — matches reference implementation
        step = -(-latent_frames // num_segments)
        lengths = [step] * num_segments

    effective = []
    cursor = 0
    for L in lengths:
        end = min(cursor + L, latent_frames)
        effective.append(max(end - cursor, 0))
        cursor = end
    return effective
