import logging
import math

from comfy_api.latest import io

from .prompt_relay import (
    get_raw_tokenizer,
    map_token_indices,
    build_segments,
    create_mask_fn,
    distribute_segment_lengths,
    format_segment_diagnostics,
    plan_quality_temporal_chunks,
    evaluate_temporal_chunk_plan,
    format_temporal_chunk_plan_diagnostics,
    format_temporal_chunk_plan_evaluation,
)

from .patches import detect_model_type, apply_patches
from .advanced_options import PromptRelayAdvancedOptions, RelayOptions

log = logging.getLogger(__name__)


_RELAY_OPTION_DEFAULTS = {
    "video_strength": 1.0,
    "video_window_scale": 1.0,
    "audio_epsilon": None,
    "audio_strength": 1.0,
    "audio_window_scale": 1.0,
    "audio_frame_offset_frames": 0.0,
    "sigma_mode": "upstream_compat",
    "long_form_mode": "off",
    "quality_target_chunk_frames": 257,
    "chunk_overlap_frames": 16,
    "chunk_safety_margin": 0.9,
}
_SIGMA_MODES = {"upstream_compat", "paper_boundary"}
_LONG_FORM_MODES = {"off", "quality_plan"}


def _sanitize_epsilon(epsilon, default=1e-3):
    """Clamp API/workflow-loaded epsilon values to the Comfy widget contract."""
    try:
        value = float(epsilon)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) and 0 < value < 1 else default


def _sanitize_relay_options(relay_options):
    """Return a plain Comfy-serializable options dict, or None when disconnected."""
    if relay_options is None:
        return None
    if not isinstance(relay_options, dict):
        raise ValueError("PromptRelay: relay_options must be produced by Prompt Relay Advanced Options")

    sanitized = dict(_RELAY_OPTION_DEFAULTS)
    for key in sanitized:
        if key in relay_options:
            sanitized[key] = relay_options[key]

    def finite_float(key, default):
        try:
            value = float(sanitized[key])
        except (TypeError, ValueError):
            return default
        return value if math.isfinite(value) else default

    sanitized["video_strength"] = min(max(finite_float("video_strength", 1.0), 0.0), 10.0)
    sanitized["video_window_scale"] = min(max(finite_float("video_window_scale", 1.0), 0.0), 4.0)
    sanitized["audio_strength"] = min(max(finite_float("audio_strength", 1.0), 0.0), 10.0)
    sanitized["audio_window_scale"] = min(max(finite_float("audio_window_scale", 1.0), 0.0), 4.0)
    sanitized["audio_frame_offset_frames"] = min(max(finite_float("audio_frame_offset_frames", 0.0), -32.0), 32.0)
    sanitized["quality_target_chunk_frames"] = min(max(int(finite_float("quality_target_chunk_frames", 257)), 1), 4096)
    sanitized["chunk_overlap_frames"] = min(max(int(finite_float("chunk_overlap_frames", 16)), 0), 512)
    sanitized["chunk_safety_margin"] = min(max(finite_float("chunk_safety_margin", 0.9), 0.1), 1.0)

    audio_epsilon = sanitized["audio_epsilon"]
    if audio_epsilon is not None:
        try:
            audio_epsilon = float(audio_epsilon)
        except (TypeError, ValueError):
            audio_epsilon = None
    sanitized["audio_epsilon"] = audio_epsilon if audio_epsilon is not None and math.isfinite(audio_epsilon) and 0 < audio_epsilon < 1 else None
    sigma_mode = sanitized["sigma_mode"]
    if not isinstance(sigma_mode, str) or sigma_mode not in _SIGMA_MODES:
        sanitized["sigma_mode"] = "upstream_compat"
    long_form_mode = sanitized["long_form_mode"]
    if not isinstance(long_form_mode, str) or long_form_mode not in _LONG_FORM_MODES:
        sanitized["long_form_mode"] = "off"
    return sanitized


def _latent_video_shape(latent):
    """Return ComfyUI video latent shape, or raise an actionable node error."""
    if not isinstance(latent, dict) or "samples" not in latent:
        raise ValueError("PromptRelay: latent input must be a ComfyUI latent dict containing 'samples'.")
    samples = latent["samples"]
    shape = getattr(samples, "shape", None)
    if shape is None or len(shape) != 5:
        raise ValueError(
            "PromptRelay: expected a video latent with shape [batch, channels, frames, height, width]. "
            "Connect an Empty Latent Video / LTX video latent, not an image latent."
        )
    if any(int(dim) <= 0 for dim in shape[2:5]):
        raise ValueError("PromptRelay: latent video frames, height, and width must be positive.")
    return samples, int(shape[2]), int(shape[3]), int(shape[4])


def _log_long_form_plan(relay_options, latent_frames, tokens_per_frame, token_ranges):
    """Emit ComfyUI-visible chunk planning diagnostics when enabled.

    This does not split sampling by itself. It makes the upstream Prompt Relay
    encoder workflow-aware: a long-form ComfyUI scheduler can use the logged
    windows for 2500+ frame renders while the baseline node path remains intact.
    """
    if not relay_options or relay_options.get("long_form_mode") != "quality_plan":
        return None
    # token_ranges are local prompt spans from the raw tokenizer. The actual
    # cross-attention K length can include a trailing EOS/special token, so plan
    # with one conservative token of slack instead of under-budgeting at the cap.
    text_tokens = max((end for _start, end in token_ranges), default=0) + 1
    plan = plan_quality_temporal_chunks(
        latent_frames=latent_frames,
        tokens_per_frame=tokens_per_frame,
        text_tokens=text_tokens,
        target_chunk_frames=relay_options.get("quality_target_chunk_frames", 257),
        overlap_frames=relay_options.get("chunk_overlap_frames", 16),
        safety_margin=relay_options.get("chunk_safety_margin", 0.9),
    )
    evaluation = evaluate_temporal_chunk_plan(
        plan,
        tokens_per_frame=tokens_per_frame,
        text_tokens=text_tokens,
        min_context_frames=max(1, min(int(relay_options.get("chunk_overlap_frames", 16)), 32)),
    )
    log.info("[PromptRelay] Long-form plan: %s", format_temporal_chunk_plan_diagnostics(plan))
    log.info("[PromptRelay] Long-form preflight: %s", format_temporal_chunk_plan_evaluation(evaluation))
    if len(plan.get("chunks", [])) > 1:
        log.info(
            "[PromptRelay] Long-form note: this encode node patches one current latent window; "
            "use the reported chunk ranges in the ComfyUI long-form scheduler/stitching workflow."
        )
    return plan


def _convert_to_latent_lengths(pixel_lengths, temporal_stride, latent_frames):
    """Convert pixel-space segment lengths to integer latent-space lengths using the
    largest-remainder method. Targets the full `latent_frames` when the pixel sum looks
    like full coverage (within one stride of latent_frames * stride). Otherwise targets
    round(total_pixel / temporal_stride) so partial-coverage timelines stay partial.
    """
    if not pixel_lengths:
        return []
    safe_lengths = [max(0, p) for p in pixel_lengths]
    total_pixel = sum(safe_lengths)
    if total_pixel <= 0:
        return [1] * len(pixel_lengths)

    naive_total = max(1, round(total_pixel / temporal_stride))
    target_total = min(latent_frames, naive_total)
    # Within one frame of full → user clearly intended full coverage; pin to latent_frames.
    if target_total >= latent_frames - 1:
        target_total = latent_frames

    exact = [p * target_total / total_pixel for p in safe_lengths]
    result = [int(e) for e in exact]
    diff = target_total - sum(result)
    if diff > 0:
        order = sorted(range(len(exact)), key=lambda i: -(exact[i] - int(exact[i])))
        for k in range(diff):
            result[order[k % len(order)]] += 1

    # Ensure every segment has ≥ 1 latent frame (steal from the largest if needed).
    for i in range(len(result)):
        if result[i] < 1:
            max_idx = max(range(len(result)), key=lambda j: result[j])
            if result[max_idx] > 1:
                result[max_idx] -= 1
                result[i] = 1

    return result


def _allocate_full_latent_lengths(weights, latent_frames):
    """Allocate the full latent timeline by largest remainder from relative weights."""
    if not weights:
        return []

    try:
        raw_weights = [float(w) for w in weights]
    except (TypeError, ValueError) as exc:
        raise ValueError("PromptRelay: ratio/percentage segment_lengths must be finite numbers.") from exc
    if any(weight != weight or abs(weight) == float("inf") for weight in raw_weights):
        raise ValueError("PromptRelay: ratio/percentage segment_lengths must be finite numbers.")

    safe_weights = [max(0.0, weight) for weight in raw_weights]
    total_weight = sum(safe_weights)
    if total_weight <= 0:
        return [1] * len(weights)

    exact = [w * latent_frames / total_weight for w in safe_weights]
    result = [int(e) for e in exact]
    diff = latent_frames - sum(result)
    if diff > 0:
        order = sorted(range(len(exact)), key=lambda i: -(exact[i] - int(exact[i])))
        for k in range(diff):
            result[order[k % len(order)]] += 1

    # Keep non-zero weighted segments visible. Steal from the largest segment if
    # rounding would otherwise collapse a short dialogue/action beat to zero.
    for i, weight in enumerate(safe_weights):
        if weight > 0 and result[i] < 1:
            max_idx = max(range(len(result)), key=lambda j: result[j])
            if result[max_idx] > 1:
                result[max_idx] -= 1
                result[i] = 1

    return result


def _parse_segment_lengths(segment_lengths, temporal_stride, latent_frames):
    """Parse user segment timing into latent-frame lengths.

    Supported forms:
      - ``"24,48"``: pixel/video frame counts, converted by the model temporal stride.
      - ``"35%,65%"``: explicit proportions over the full latent timeline.
      - ``"0.35,0.65"``: ratio shorthand over the full latent timeline.

    Percentages/ratios are useful for dialogue/action splits in LTXV AV where the
    user often knows the speech beat proportion but not the compressed latent
    frame count. Existing integer frame-count behavior is preserved.
    """
    if not segment_lengths or not segment_lengths.strip():
        return None

    parts = [x.strip() for x in segment_lengths.split(",")]
    if any(part == "" for part in parts):
        raise ValueError(
            "PromptRelay: segment_lengths contains an empty entry. "
            "Remove extra commas or fill every segment length, e.g. '84,133' or '35%,65%'."
        )

    percent_flags = [part.endswith("%") for part in parts]
    if any(percent_flags):
        if not all(percent_flags):
            raise ValueError(
                "PromptRelay: segment_lengths must use one timing format at a time. "
                "Use all percentages like '35%,65%' or all frame counts like '84,133'."
            )
        try:
            weights = [float(part[:-1].strip()) for part in parts]
        except ValueError as exc:
            raise ValueError(
                "PromptRelay: percentage segment_lengths must be numbers like '35%,65%'."
            ) from exc
        if any(weight != weight or abs(weight) == float("inf") for weight in weights):
            raise ValueError("PromptRelay: percentage segment_lengths must be finite numbers.")
        if any(weight < 0 for weight in weights) or sum(weights) <= 0:
            raise ValueError("PromptRelay: percentage segment_lengths must be positive proportions.")
        return _allocate_full_latent_lengths(weights, latent_frames)

    try:
        values = [float(part) for part in parts]
    except ValueError as exc:
        raise ValueError(
            "PromptRelay: segment_lengths must be comma-separated numbers, percentages, or ratios. "
            "Examples: '84,133', '35%,65%', or '0.35,0.65'."
        ) from exc
    if any(value != value or abs(value) == float("inf") for value in values):
        raise ValueError("PromptRelay: segment_lengths must be finite numbers.")

    if all(0.0 <= value <= 1.0 for value in values) and any("." in part for part in parts):
        if sum(values) <= 0:
            raise ValueError("PromptRelay: ratio segment_lengths must include at least one positive value.")
        return _allocate_full_latent_lengths(values, latent_frames)

    if any(not value.is_integer() for value in values):
        raise ValueError(
            "PromptRelay: frame-count segment_lengths must be whole numbers. "
            "Use percentages like '35%,65%' or ratios like '0.35,0.65' for fractional timing."
        )

    return _convert_to_latent_lengths([int(value) for value in values], temporal_stride, latent_frames)


def _encode_relay(model, clip, latent, global_prompt, local_prompts, segment_lengths, epsilon, relay_options=None):
    for name, val in (("global_prompt", global_prompt),
                      ("local_prompts", local_prompts),
                      ("segment_lengths", segment_lengths)):
        if val is None:
            raise ValueError(
                f"PromptRelay: '{name}' arrived as None. "
                "Likely causes: a stale workflow JSON saved with null, the timeline "
                "editor's web extension failing to load, or an upstream node returning None. "
                "Set the field to an empty string or fix the upstream connection."
            )

    locals_list = [p.strip() for p in local_prompts.split("|") if p.strip()]
    if not locals_list:
        raise ValueError("At least one local prompt is required (separate with |)")

    arch, patch_size, temporal_stride = detect_model_type(model)

    samples, latent_frames, latent_height, latent_width = _latent_video_shape(latent)
    tokens_per_frame = (latent_height // patch_size[1]) * (latent_width // patch_size[2])
    if tokens_per_frame <= 0:
        raise ValueError(
            "PromptRelay: latent spatial size is too small for this model patch size. "
            f"Got latent {latent_height}x{latent_width} with patch size {patch_size[1]}x{patch_size[2]}."
        )

    parsed_lengths = _parse_segment_lengths(segment_lengths, temporal_stride, latent_frames)

    raw_tokenizer = get_raw_tokenizer(clip)
    full_prompt, token_ranges = map_token_indices(raw_tokenizer, global_prompt, locals_list)

    log.info("[PromptRelay] Global: tokens [0:%d] (%d tokens)", token_ranges[0][0], token_ranges[0][0])
    for i, (s, e) in enumerate(token_ranges):
        log.info("[PromptRelay] Segment %d: tokens [%d:%d] (%d tokens)", i, s, e, e - s)

    conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(full_prompt))

    effective_lengths = distribute_segment_lengths(len(locals_list), latent_frames, parsed_lengths)

    log.info(
        "[PromptRelay] Latent: %d frames, %d tokens/frame, segments: %s",
        latent_frames, tokens_per_frame, effective_lengths,
    )

    epsilon = _sanitize_epsilon(epsilon)
    relay_options = _sanitize_relay_options(relay_options)
    _log_long_form_plan(relay_options, latent_frames, tokens_per_frame, token_ranges)
    q_token_idx = build_segments(token_ranges, effective_lengths, epsilon, relay_options)
    log.info(
        "[PromptRelay] Setup: arch=%s latent_frames=%d tokens_per_frame=%d timing='%s' epsilon=%.6g options=%s | %s",
        arch,
        latent_frames,
        tokens_per_frame,
        segment_lengths.strip() or "auto-even",
        epsilon,
        relay_options or "defaults",
        format_segment_diagnostics(q_token_idx),
    )
    mask_fn = create_mask_fn(q_token_idx, tokens_per_frame, latent_frames)

    patched = model.clone()
    apply_patches(patched, arch, mask_fn)

    return patched, conditioning


class PromptRelayEncode(io.ComfyNode):
    """Encodes temporal local prompts and patches the model for Prompt Relay."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PromptRelayEncode",
            display_name="Prompt Relay Encode",
            category="conditioning/prompt_relay",
            description=(
                "Encodes a global prompt combined with temporal local prompts and patches the model "
                "for Prompt Relay temporal control. Local prompts are separated by |. "
                "Use a standard CLIPTextEncode for the negative prompt."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Latent.Input("latent", tooltip="Empty latent video — dimensions are read from its shape."),
                io.String.Input(
                    "global_prompt", multiline=True, default="",
                    tooltip="Conditions the entire video. Anchors persistent characters, objects, and scene context.",
                ),
                io.String.Input(
                    "local_prompts", multiline=True, default="",
                    tooltip="Ordered prompts for each temporal segment, separated by |",
                ),
                io.String.Input(
                    "segment_lengths", default="",
                    tooltip="Comma-separated pixel-space frame counts, percentages (35%,65%), or ratios (0.35,0.65). Empty = even distribution.",
                ),
                io.Float.Input(
                    "epsilon", default=1e-3, min=1e-6, max=0.99, step=1e-4,
                    tooltip="Penalty decay parameter. Values below ~0.1 all produce sharp boundaries (paper default 0.001). For softer transitions, try 0.5 or higher.",
                ),
                RelayOptions.Input(
                    "relay_options", optional=True,
                    tooltip="Optional advanced per-stream tuning. Connect a Prompt Relay Advanced Options node.",
                ),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                io.Conditioning.Output(display_name="positive"),
            ],
        )

    @classmethod
    def execute(cls, model, clip, latent, global_prompt, local_prompts, segment_lengths, epsilon, relay_options=None) -> io.NodeOutput:
        patched, conditioning = _encode_relay(
            model, clip, latent, global_prompt, local_prompts, segment_lengths, epsilon, relay_options,
        )
        return io.NodeOutput(patched, conditioning)


class PromptRelayEncodeTimeline(io.ComfyNode):
    """WYSIWYG timeline variant — segments and lengths come from a visual editor in the node UI."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PromptRelayEncodeTimeline",
            display_name="Prompt Relay Encode (Timeline)",
            category="conditioning/prompt_relay",
            description=(
                "Same as Prompt Relay Encode, but local prompts and segment lengths are edited "
                "visually as draggable blocks on a timeline. The max_frames input only sets the "
                "timeline scale (pixel space) — actual frame count is still read from the latent."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Latent.Input("latent", tooltip="Empty latent video — dimensions are read from its shape."),
                io.String.Input(
                    "global_prompt", multiline=True, default="",
                    tooltip="Conditions the entire video. Anchors persistent characters, objects, and scene context.",
                ),
                io.Int.Input(
                    "max_frames", default=129, min=1, max=10000, step=1,
                    tooltip="Total timeline length in pixel-space frames. Used by the editor for visual scale only.",
                ),
                io.String.Input(
                    "timeline_data", default="",
                    tooltip="JSON state of the timeline editor (auto-managed; do not edit by hand).",
                ),
                io.String.Input(
                    "local_prompts", multiline=True, default="",
                    tooltip="Auto-populated from the timeline editor.",
                ),
                io.String.Input(
                    "segment_lengths", default="",
                    tooltip="Auto-populated from the timeline editor (pixel-space frame counts). Manual ratios/percentages are also accepted.",
                ),
                io.Float.Input(
                    "epsilon", default=1e-3, min=1e-6, max=0.99, step=1e-4,
                    tooltip="Penalty decay parameter. Values below ~0.1 all produce sharp boundaries (paper default 0.001). For softer transitions, try 0.5 or higher.",
                ),
                io.Float.Input(
                    "fps", default=24.0, min=0.1, max=240.0, step=0.1, optional=True,
                    tooltip="Frames per second — only affects how time is displayed in the timeline editor when time_units is set to 'seconds'.",
                ),
                io.Combo.Input(
                    "time_units", options=["frames", "seconds"], default="frames", optional=True,
                    tooltip="Display the ruler, segment ranges, length input, and total in frames or seconds. Internal storage is always pixel-space frames.",
                ),
                RelayOptions.Input(
                    "relay_options", optional=True,
                    tooltip="Optional advanced per-stream tuning. Connect a Prompt Relay Advanced Options node.",
                ),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                io.Conditioning.Output(display_name="positive"),
            ],
        )


    @classmethod
    def execute(cls, model, clip, latent, global_prompt, max_frames, timeline_data, local_prompts, segment_lengths, epsilon, fps=24.0, time_units="frames", relay_options=None) -> io.NodeOutput:
        patched, conditioning = _encode_relay(
            model, clip, latent, global_prompt, local_prompts, segment_lengths, epsilon, relay_options,
        )
        return io.NodeOutput(patched, conditioning)


NODE_CLASS_MAPPINGS = {
    "PromptRelayEncode": PromptRelayEncode,
    "PromptRelayEncodeTimeline": PromptRelayEncodeTimeline,
    "PromptRelayAdvancedOptions": PromptRelayAdvancedOptions,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptRelayEncode": "Prompt Relay Encode",
    "PromptRelayEncodeTimeline": "Prompt Relay Encode (Timeline)",
    "PromptRelayAdvancedOptions": "Prompt Relay Advanced Options",
}
