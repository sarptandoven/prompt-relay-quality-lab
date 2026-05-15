"""Supplemental Prompt Relay ComfyUI node.

Copy this file into the upstream `ComfyUI-PromptRelay` package to add an
experimental node without replacing the existing nodes. Register it from
`__init__.py` by importing `PromptRelayLabEncode` and adding it to the node
mappings / ComfyExtension node list.

The node keeps upstream-compatible behavior by default and exposes an explicit
paper-boundary sigma mode for controlled A/B tests.
"""

import logging
import math

from comfy_api.latest import io

from .patches import apply_patches, detect_model_type
from .prompt_relay import create_mask_fn, distribute_segment_lengths, get_raw_tokenizer, map_token_indices

log = logging.getLogger(__name__)


SIGMA_MODE_UPSTREAM = "upstream_compat"
SIGMA_MODE_PAPER = "paper_boundary"
SIGMA_MODES = [SIGMA_MODE_UPSTREAM, SIGMA_MODE_PAPER]


def _finite_float(value, default):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _sanitize_lab_epsilon(epsilon, default=1e-3):
    """Clamp API/workflow-loaded epsilon values to the Comfy widget contract."""
    try:
        value = float(epsilon)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) and 0 < value < 1 else default


def _sanitize_lab_relay_options(
    video_strength,
    video_window_scale,
    audio_epsilon,
    audio_strength,
    audio_window_scale,
    audio_frame_offset_frames,
):
    """Mirror Comfy input bounds for API/workflow-loaded values.

    Comfy UI widgets enforce min/max, but stale workflow JSON or API calls can
    still pass NaN/inf/negative values. Clamp before building penalties so the
    lab node cannot accidentally invert masks or create non-finite attention
    biases.
    """
    audio_epsilon = _finite_float(audio_epsilon, 0.0)
    return {
        "video_strength": min(max(_finite_float(video_strength, 1.0), 0.0), 10.0),
        "video_window_scale": min(max(_finite_float(video_window_scale, 1.0), 0.0), 4.0),
        "audio_epsilon": audio_epsilon if 0 < audio_epsilon < 1 else None,
        "audio_strength": min(max(_finite_float(audio_strength, 1.0), 0.0), 10.0),
        "audio_window_scale": min(max(_finite_float(audio_window_scale, 1.0), 0.0), 4.0),
        "audio_frame_offset_frames": min(max(_finite_float(audio_frame_offset_frames, 0.0), -32.0), 32.0),
    }


def calculate_prompt_relay_sigma(segment_length, free_window, epsilon, sigma_mode=SIGMA_MODE_UPSTREAM):
    """Return the temporal decay sigma for a segment.

    `upstream_compat` matches the current ComfyUI-PromptRelay implementation and
    Gordon's Wan2.2 reference constant for epsilon=1e-3:
        sigma = 1 / ln(1 / epsilon)

    `paper_boundary` follows Prompt Relay Eq. 4:
        sigma = (L/2 - w) / sqrt(2 ln(1 / epsilon))

    where L/2 is the segment endpoint distance from the segment midpoint and w
    is the free-attention window around that midpoint. This mode is opt-in
    because it is materially softer than the current ComfyUI implementation.
    """
    if not 0 < epsilon < 1:
        return 0.1448

    if sigma_mode == SIGMA_MODE_PAPER:
        endpoint_distance = max(float(segment_length) / 2.0, 0.0)
        penalty_distance = max(endpoint_distance - float(free_window), 0.0)
        if penalty_distance <= 0:
            return 1e-6
        return penalty_distance / math.sqrt(2.0 * math.log(1.0 / epsilon))

    return 1.0 / math.log(1.0 / epsilon)


def _local_token_span(local_token_idx):
    if hasattr(local_token_idx, "tolist"):
        values = local_token_idx.tolist()
    else:
        values = list(local_token_idx)
    if not values:
        return "[]"
    return f"[{int(values[0])}:{int(values[-1]) + 1}]"


def format_lab_segment_diagnostics(q_token_idx, max_segments=12):
    parts = []
    total = len(q_token_idx)
    indexed_segments = list(enumerate(q_token_idx))
    if max_segments and total > max_segments:
        head_count = max_segments // 2
        tail_count = max_segments - head_count
        indexed_segments = indexed_segments[:head_count] + indexed_segments[-tail_count:]

    previous_idx = None
    for idx, seg in indexed_segments:
        if previous_idx is not None and idx != previous_idx + 1:
            parts.append(f"... {idx - previous_idx - 1} segment(s) omitted ...")
        parts.append(
            "seg{idx}: tokens={tokens} mid={mid:.3f} win={win:.3f} sigma={sigma:.6g} str={strength:.3f} "
            "audio_mid={audio_mid:.3f} audio_win={audio_win:.3f} audio_sigma={audio_sigma:.6g} audio_str={audio_strength:.3f}".format(
                idx=idx,
                tokens=_local_token_span(seg["local_token_idx"]),
                mid=float(seg["midpoint"]),
                win=float(seg["window"]),
                sigma=float(seg["sigma"]),
                strength=float(seg.get("strength", 1.0)),
                audio_mid=float(seg.get("midpoint_audio", seg["midpoint"])),
                audio_win=float(seg.get("window_audio", seg["window"])),
                audio_sigma=float(seg.get("sigma_audio", seg["sigma"])),
                audio_strength=float(seg.get("strength_audio", 1.0)),
            )
        )
        previous_idx = idx
    return "; ".join(parts)


def build_lab_segments(token_ranges, segment_lengths, epsilon=1e-3, sigma_mode=SIGMA_MODE_UPSTREAM, relay_options=None):
    """Build Prompt Relay segment metadata with an explicit sigma mode.

    This intentionally mirrors `prompt_relay.build_segments` so the resulting
    metadata is accepted by the existing mask and patching code.
    """
    opts = relay_options or {}
    video_strength = opts.get("video_strength", 1.0)
    video_window_scale = opts.get("video_window_scale", 1.0)
    audio_epsilon = opts.get("audio_epsilon")
    audio_strength = opts.get("audio_strength", 1.0)
    audio_window_scale = opts.get("audio_window_scale", 1.0)
    audio_frame_offset = opts.get("audio_frame_offset_frames", 0.0)

    q_token_idx = []
    frame_cursor = 0

    for (tok_start, tok_end), segment_length in zip(token_ranges, segment_lengths):
        if segment_length <= 0:
            continue

        midpoint = (2 * frame_cursor + segment_length) // 2
        base_window = max(segment_length // 2 - 2, 0)
        video_window = max(base_window * video_window_scale, 0.0)
        audio_window = max(base_window * audio_window_scale, 0.0)
        sigma = calculate_prompt_relay_sigma(segment_length, video_window, epsilon, sigma_mode)
        sigma_audio = (
            calculate_prompt_relay_sigma(segment_length, audio_window, audio_epsilon, sigma_mode)
            if audio_epsilon is not None and 0 < audio_epsilon < 1
            else sigma
        )

        if sigma <= 0:
            sigma = 1e-6
        if sigma_audio <= 0:
            sigma_audio = 1e-6

        import torch

        q_token_idx.append(
            {
                "local_token_idx": torch.arange(tok_start, tok_end),
                "midpoint": midpoint,
                "midpoint_audio": midpoint + audio_frame_offset,
                "window": video_window,
                "sigma": sigma,
                "strength": video_strength,
                "window_audio": audio_window,
                "sigma_audio": sigma_audio,
                "strength_audio": audio_strength,
            }
        )
        frame_cursor += segment_length

    return q_token_idx


def _allocate_full_latent_lengths(weights, latent_frames):
    """Allocate the full latent timeline from percentage/ratio weights."""
    if not weights:
        return []

    try:
        raw_weights = [float(weight) for weight in weights]
    except (TypeError, ValueError) as exc:
        raise ValueError("PromptRelayLab: ratio/percentage segment_lengths must be finite numbers.") from exc
    if any(not math.isfinite(weight) for weight in raw_weights):
        raise ValueError("PromptRelayLab: ratio/percentage segment_lengths must be finite numbers.")

    safe_weights = [max(0.0, weight) for weight in raw_weights]
    total_weight = sum(safe_weights)
    if total_weight <= 0:
        return [1] * len(weights)

    exact = [weight * latent_frames / total_weight for weight in safe_weights]
    latent_lengths = [int(value) for value in exact]
    remainder = latent_frames - sum(latent_lengths)
    if remainder > 0:
        order = sorted(range(len(exact)), key=lambda i: -(exact[i] - int(exact[i])))
        for i in range(remainder):
            latent_lengths[order[i % len(order)]] += 1

    for idx, weight in enumerate(safe_weights):
        if weight <= 0 or latent_lengths[idx] >= 1:
            continue
        donor = max(range(len(latent_lengths)), key=lambda i: latent_lengths[i])
        if latent_lengths[donor] > 1:
            latent_lengths[donor] -= 1
            latent_lengths[idx] = 1

    return latent_lengths


def _convert_to_latent_lengths(pixel_lengths, temporal_stride, latent_frames):
    if not pixel_lengths:
        return []

    safe_lengths = [max(0, int(length)) for length in pixel_lengths]
    total_pixel = sum(safe_lengths)
    if total_pixel <= 0:
        return [1] * len(pixel_lengths)

    target_total = min(latent_frames, max(1, round(total_pixel / temporal_stride)))
    if target_total >= latent_frames - 1:
        target_total = latent_frames

    exact = [length * target_total / total_pixel for length in safe_lengths]
    latent_lengths = [int(value) for value in exact]
    remainder = target_total - sum(latent_lengths)
    if remainder > 0:
        order = sorted(range(len(exact)), key=lambda i: -(exact[i] - int(exact[i])))
        for i in range(remainder):
            latent_lengths[order[i % len(order)]] += 1

    for idx, length in enumerate(latent_lengths):
        if length >= 1:
            continue
        donor = max(range(len(latent_lengths)), key=lambda i: latent_lengths[i])
        if latent_lengths[donor] > 1:
            latent_lengths[donor] -= 1
            latent_lengths[idx] = 1

    return latent_lengths


def _parse_segment_lengths(segment_lengths, temporal_stride, latent_frames):
    """Parse lab node timing as frames, percentages, or ratios."""
    if not segment_lengths or not segment_lengths.strip():
        return None

    parts = [part.strip() for part in segment_lengths.split(",")]
    if any(part == "" for part in parts):
        raise ValueError(
            "PromptRelayLab: segment_lengths contains an empty entry. "
            "Remove extra commas or fill every segment length, e.g. '84,133' or '35%,65%'."
        )

    percent_flags = [part.endswith("%") for part in parts]
    if any(percent_flags):
        if not all(percent_flags):
            raise ValueError(
                "PromptRelayLab: segment_lengths must use one timing format at a time. "
                "Use all percentages like '35%,65%' or all frame counts like '84,133'."
            )
        try:
            weights = [float(part[:-1].strip()) for part in parts]
        except ValueError as exc:
            raise ValueError("PromptRelayLab: percentage segment_lengths must be numbers like '35%,65%'.") from exc
        if any(not math.isfinite(weight) for weight in weights):
            raise ValueError("PromptRelayLab: percentage segment_lengths must be finite numbers.")
        if any(weight < 0 for weight in weights) or sum(weights) <= 0:
            raise ValueError("PromptRelayLab: percentage segment_lengths must be positive proportions.")
        return _allocate_full_latent_lengths(weights, latent_frames)

    try:
        values = [float(part) for part in parts]
    except ValueError as exc:
        raise ValueError(
            "PromptRelayLab: segment_lengths must be comma-separated numbers, percentages, or ratios. "
            "Examples: '84,133', '35%,65%', or '0.35,0.65'."
        ) from exc
    if any(not math.isfinite(value) for value in values):
        raise ValueError("PromptRelayLab: segment_lengths must be finite numbers.")

    if all(0.0 <= value <= 1.0 for value in values) and any("." in part for part in parts):
        if sum(values) <= 0:
            raise ValueError("PromptRelayLab: ratio segment_lengths must include at least one positive value.")
        return _allocate_full_latent_lengths(values, latent_frames)

    if any(not value.is_integer() for value in values):
        raise ValueError(
            "PromptRelayLab: frame-count segment_lengths must be whole numbers. "
            "Use percentages like '35%,65%' or ratios like '0.35,0.65' for fractional timing."
        )

    return _convert_to_latent_lengths(values, temporal_stride, latent_frames)


def _encode_lab_relay(
    model,
    clip,
    latent,
    global_prompt,
    local_prompts,
    segment_lengths,
    epsilon,
    sigma_mode,
    video_strength,
    video_window_scale,
    audio_epsilon,
    audio_strength,
    audio_window_scale,
    audio_frame_offset_frames,
):
    for name, value in (
        ("global_prompt", global_prompt),
        ("local_prompts", local_prompts),
        ("segment_lengths", segment_lengths),
    ):
        if value is None:
            raise ValueError(f"PromptRelayLab: '{name}' arrived as None. Set it to an empty string or reconnect the input.")

    locals_list = [prompt.strip() for prompt in local_prompts.split("|") if prompt.strip()]
    if not locals_list:
        raise ValueError("PromptRelayLab requires at least one local prompt separated with |")

    if sigma_mode not in SIGMA_MODES:
        raise ValueError(f"Unknown sigma_mode: {sigma_mode}")

    arch, patch_size, temporal_stride = detect_model_type(model)
    samples = latent["samples"]
    latent_frames = samples.shape[2]
    tokens_per_frame = (samples.shape[3] // patch_size[1]) * (samples.shape[4] // patch_size[2])

    parsed_lengths = _parse_segment_lengths(segment_lengths, temporal_stride, latent_frames)

    raw_tokenizer = get_raw_tokenizer(clip)
    full_prompt, token_ranges = map_token_indices(raw_tokenizer, global_prompt, locals_list)
    conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(full_prompt))
    effective_lengths = distribute_segment_lengths(len(locals_list), latent_frames, parsed_lengths)

    epsilon = _sanitize_lab_epsilon(epsilon)
    relay_options = _sanitize_lab_relay_options(
        video_strength,
        video_window_scale,
        audio_epsilon,
        audio_strength,
        audio_window_scale,
        audio_frame_offset_frames,
    )
    q_token_idx = build_lab_segments(token_ranges, effective_lengths, epsilon, sigma_mode, relay_options)
    mask_fn = create_mask_fn(q_token_idx, tokens_per_frame, latent_frames)

    log.info(
        "[PromptRelayLab] arch=%s latent_frames=%d tokens_per_frame=%d timing='%s' segment_lengths=%s "
        "sigma_mode=%s video_strength=%.3f video_window_scale=%.3f audio_epsilon=%s "
        "audio_strength=%.3f audio_window_scale=%.3f audio_frame_offset=%.3f | %s",
        arch,
        latent_frames,
        tokens_per_frame,
        segment_lengths.strip() or "auto-even",
        effective_lengths,
        sigma_mode,
        relay_options["video_strength"],
        relay_options["video_window_scale"],
        f"{relay_options['audio_epsilon']:.6g}" if relay_options["audio_epsilon"] is not None else "inherit",
        relay_options["audio_strength"],
        relay_options["audio_window_scale"],
        relay_options["audio_frame_offset_frames"],
        format_lab_segment_diagnostics(q_token_idx),
    )

    patched = model.clone()
    apply_patches(patched, arch, mask_fn)
    return patched, conditioning


class PromptRelayLabEncode(io.ComfyNode):
    """Supplemental Prompt Relay encoder with explicit upstream/paper sigma modes."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PromptRelayLabEncode",
            display_name="Prompt Relay Encode (Lab)",
            category="conditioning/prompt_relay",
            is_experimental=True,
            description=(
                "Supplemental Prompt Relay node for controlled A/B tests. "
                "Default mode is upstream-compatible. Paper-boundary sigma is opt-in."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Latent.Input("latent", tooltip="Empty latent video; dimensions are read from its shape."),
                io.String.Input("global_prompt", multiline=True, default=""),
                io.String.Input("local_prompts", multiline=True, default="", tooltip="Ordered prompts separated by |"),
                io.String.Input("segment_lengths", default="", tooltip="Comma-separated pixel-space frame counts, percentages (35%,65%), or ratios (0.35,0.65). Empty = even distribution."),
                io.Float.Input("epsilon", default=1e-3, min=1e-6, max=0.99, step=1e-4),
                io.Combo.Input("sigma_mode", options=SIGMA_MODES, default=SIGMA_MODE_UPSTREAM),
                io.Float.Input("video_strength", default=1.0, min=0.0, max=10.0, step=0.05),
                io.Float.Input("video_window_scale", default=1.0, min=0.0, max=4.0, step=0.05),
                io.Float.Input("audio_epsilon", default=0.0, min=0.0, max=0.99, step=1e-4, tooltip="0 inherits video epsilon; >0 applies separate LTX AV audio-attention epsilon."),
                io.Float.Input("audio_strength", default=1.0, min=0.0, max=10.0, step=0.05),
                io.Float.Input("audio_window_scale", default=1.0, min=0.0, max=4.0, step=0.05),
                io.Float.Input("audio_frame_offset_frames", default=0.0, min=-32.0, max=32.0, step=0.25, tooltip="Shifts only LTX AV audio-attention anchors in latent frames."),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                io.Conditioning.Output(display_name="positive"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        clip,
        latent,
        global_prompt,
        local_prompts,
        segment_lengths,
        epsilon,
        sigma_mode=SIGMA_MODE_UPSTREAM,
        video_strength=1.0,
        video_window_scale=1.0,
        audio_epsilon=0.0,
        audio_strength=1.0,
        audio_window_scale=1.0,
        audio_frame_offset_frames=0.0,
    ) -> io.NodeOutput:
        patched, conditioning = _encode_lab_relay(
            model,
            clip,
            latent,
            global_prompt,
            local_prompts,
            segment_lengths,
            epsilon,
            sigma_mode,
            video_strength,
            video_window_scale,
            audio_epsilon,
            audio_strength,
            audio_window_scale,
            audio_frame_offset_frames,
        )
        return io.NodeOutput(patched, conditioning)


NODE_CLASS_MAPPINGS = {
    "PromptRelayLabEncode": PromptRelayLabEncode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptRelayLabEncode": "Prompt Relay Encode (Lab)",
}
