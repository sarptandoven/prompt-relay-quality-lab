import math

from comfy_api.latest import io


RelayOptions = io.Custom("RELAY_OPTIONS")
SIGMA_MODE_UPSTREAM = "upstream_compat"
SIGMA_MODE_PAPER = "paper_boundary"
SIGMA_MODES = [SIGMA_MODE_UPSTREAM, SIGMA_MODE_PAPER]
LONG_FORM_OFF = "off"
LONG_FORM_QUALITY_PLAN = "quality_plan"
LONG_FORM_MODES = [LONG_FORM_OFF, LONG_FORM_QUALITY_PLAN]


class PromptRelayAdvancedOptions(io.ComfyNode):
    """Per-stream temporal-penalty knobs for Prompt Relay encoders."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PromptRelayAdvancedOptions",
            display_name="Prompt Relay Advanced Options",
            category="conditioning/prompt_relay",
            is_experimental=True,
            description=(
                "Optional per-stream tuning. Connect to the relay_options input on a Prompt Relay encoder. Audio fields only affect LTX2."
            ),
            inputs=[
                io.Float.Input(
                    "video_strength", default=1.0, min=0.0, max=10.0, step=0.05,
                    tooltip="Multiplier on the video temporal penalty. 0 disables video segmentation. Most useful in the 0–1 range to soften boundaries; values >1 saturate quickly at the default epsilon (1e-3) because the bias is already large enough to zero out distant tokens. To make >1 visibly meaningful, raise epsilon to ~0.1 or higher.",
                ),
                io.Float.Input(
                    "video_window_scale", default=1.0, min=0.0, max=4.0, step=0.05,
                    tooltip="Scales the flat anchor zone (default L/2 - 2 frames). <1 starts the soft falloff sooner; >1 widens the rigid zone. 0 collapses the anchor to a point — falloff begins immediately at the segment midpoint (sharper than default, not softer).",
                ),
                io.Float.Input(
                    "audio_epsilon", default=0.0, min=0.0, max=0.99, step=1e-4,
                    tooltip="Epsilon for the audio stream. 0 = inherit from the encoder's main epsilon.",
                ),
                io.Float.Input(
                    "audio_strength", default=1.0, min=0.0, max=10.0, step=0.05,
                    tooltip="Multiplier on the audio temporal penalty. 0 lets audio bleed across visual cuts. Most useful in the 0–1 range; values >1 saturate quickly at the default epsilon (1e-3) — raise audio_epsilon to ~0.1 or higher to make >1 visibly meaningful.",
                ),
                io.Float.Input(
                    "audio_window_scale", default=1.0, min=0.0, max=4.0, step=0.05,
                    tooltip="Scales the flat anchor zone width for the audio stream. <1 starts the soft falloff sooner; >1 widens the rigid zone. 0 collapses the anchor to a point — falloff begins immediately at the segment midpoint (sharper than default, not softer).",
                ),
                io.Combo.Input(
                    "sigma_mode", options=SIGMA_MODES, default=SIGMA_MODE_UPSTREAM,
                    tooltip="upstream_compat preserves existing sharp decay. paper_boundary is opt-in and sets sigma so retained attention at the segment endpoint equals epsilon (Prompt Relay Eq. 4).",
                ),
                io.Float.Input(
                    "audio_frame_offset_frames", default=0.0, min=-32.0, max=32.0, step=0.25,
                    tooltip="Shifts only the LTX AV audio-attention temporal anchors in latent frames. Negative values make audio follow earlier segment text; positive values delay it. Leave 0 for video-aligned prompts.",
                ),
                io.Combo.Input(
                    "long_form_mode", options=LONG_FORM_MODES, default=LONG_FORM_OFF,
                    tooltip="off preserves current behavior. quality_plan logs a mask-safe 129-257 style chunk plan for 2500+ frame ComfyUI workflows.",
                ),
                io.Int.Input(
                    "quality_target_chunk_frames", default=257, min=1, max=4096, step=1,
                    tooltip="Preferred latent-frame window size for quality long-form planning. 257 matches common LTXV long-window practice; lower this for tighter VRAM or stronger local continuity.",
                ),
                io.Int.Input(
                    "chunk_overlap_frames", default=16, min=0, max=512, step=1,
                    tooltip="Latent-frame overlap between long-form chunks for seam diagnostics and stitching guidance.",
                ),
                io.Float.Input(
                    "chunk_safety_margin", default=0.9, min=0.1, max=1.0, step=0.05,
                    tooltip="Fraction of the Prompt Relay mask budget to use when sizing chunks.",
                ),
            ],
            outputs=[
                RelayOptions.Output(display_name="relay_options"),
            ],
        )

    @classmethod
    def execute(cls, video_strength, video_window_scale,
                audio_epsilon, audio_strength, audio_window_scale,
                sigma_mode=SIGMA_MODE_UPSTREAM, audio_frame_offset_frames=0.0,
                long_form_mode=LONG_FORM_OFF, quality_target_chunk_frames=257,
                chunk_overlap_frames=16, chunk_safety_margin=0.9) -> io.NodeOutput:
        def finite_float(value, default):
            try:
                value = float(value)
            except (TypeError, ValueError):
                return default
            return value if math.isfinite(value) else default

        audio_epsilon = finite_float(audio_epsilon, 0.0)
        opts = {
            "video_strength": min(max(finite_float(video_strength, 1.0), 0.0), 10.0),
            "video_window_scale": min(max(finite_float(video_window_scale, 1.0), 0.0), 4.0),
            "audio_epsilon": audio_epsilon if 0 < audio_epsilon < 1 else None,
            "audio_strength": min(max(finite_float(audio_strength, 1.0), 0.0), 10.0),
            "audio_window_scale": min(max(finite_float(audio_window_scale, 1.0), 0.0), 4.0),
            "sigma_mode": sigma_mode if sigma_mode in SIGMA_MODES else SIGMA_MODE_UPSTREAM,
            "audio_frame_offset_frames": min(max(finite_float(audio_frame_offset_frames, 0.0), -32.0), 32.0),
            "long_form_mode": long_form_mode if long_form_mode in LONG_FORM_MODES else LONG_FORM_OFF,
            "quality_target_chunk_frames": min(max(int(finite_float(quality_target_chunk_frames, 257)), 1), 4096),
            "chunk_overlap_frames": min(max(int(finite_float(chunk_overlap_frames, 16)), 0), 512),
            "chunk_safety_margin": min(max(finite_float(chunk_safety_margin, 0.9), 0.1), 1.0),
        }
        return io.NodeOutput(opts)
