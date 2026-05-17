import ast
import math
import os
import sys
import types
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAB_NODE_PATH = os.path.join(ROOT, "prompt_relay_lab_node.py")
PROMPT_RELAY_PATH = os.path.join(ROOT, "prompt_relay.py")


class DummyLog:
    def info(self, *args, **kwargs):
        pass


class FakeTorch:
    @staticmethod
    def arange(start, end):
        return list(range(start, end))


def load_lab_helpers():
    with open(LAB_NODE_PATH, "r", encoding="utf-8") as f:
        module = ast.parse(f.read(), filename=LAB_NODE_PATH)

    wanted = {
        "calculate_prompt_relay_sigma",
        "_allocate_full_latent_lengths",
        "_convert_to_latent_lengths",
        "_parse_segment_lengths",
        "build_lab_segments",
        "_finite_float",
        "_sanitize_lab_epsilon",
        "_sanitize_lab_relay_options",
    }
    constants = {"SIGMA_MODE_UPSTREAM", "SIGMA_MODE_PAPER", "SIGMA_MODES"}
    nodes = []
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            nodes.append(node)
        elif isinstance(node, ast.Assign):
            names = {target.id for target in node.targets if isinstance(target, ast.Name)}
            if names & constants:
                nodes.append(node)
    isolated = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(isolated)

    namespace = {"math": math}
    exec(compile(isolated, LAB_NODE_PATH, "exec"), namespace)
    return namespace


def load_main_segment_helpers():
    with open(PROMPT_RELAY_PATH, "r", encoding="utf-8") as f:
        module = ast.parse(f.read(), filename=PROMPT_RELAY_PATH)

    wanted = {"calculate_prompt_relay_sigma", "_segment_relay_knobs", "_build_segment_metadata", "build_segments"}
    constants = {"SIGMA_MODE_UPSTREAM", "SIGMA_MODE_PAPER"}
    nodes = []
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            nodes.append(node)
        elif isinstance(node, ast.Assign):
            names = {target.id for target in node.targets if isinstance(target, ast.Name)}
            if names & constants:
                nodes.append(node)
    isolated = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(isolated)

    namespace = {"math": math, "torch": FakeTorch, "log": DummyLog()}
    exec(compile(isolated, PROMPT_RELAY_PATH, "exec"), namespace)
    return namespace


HELPERS = load_lab_helpers()
MAIN_HELPERS = load_main_segment_helpers()
calculate_prompt_relay_sigma = HELPERS["calculate_prompt_relay_sigma"]
_allocate_full_latent_lengths = HELPERS["_allocate_full_latent_lengths"]
_parse_segment_lengths = HELPERS["_parse_segment_lengths"]
build_lab_segments = HELPERS["build_lab_segments"]
_sanitize_lab_epsilon = HELPERS["_sanitize_lab_epsilon"]
_sanitize_lab_relay_options = HELPERS["_sanitize_lab_relay_options"]
main_calculate_prompt_relay_sigma = MAIN_HELPERS["calculate_prompt_relay_sigma"]
main_build_segments = MAIN_HELPERS["build_segments"]


class PromptRelayLabNodeMathTests(unittest.TestCase):
    def test_upstream_compat_sigma_matches_current_formula(self):
        self.assertAlmostEqual(
            calculate_prompt_relay_sigma(10, 3, 1e-3, "upstream_compat"),
            1.0 / math.log(1.0 / 1e-3),
            places=12,
        )

    def test_paper_boundary_sigma_satisfies_endpoint_epsilon(self):
        segment_length = 10
        free_window = 3
        epsilon = 0.1
        sigma = calculate_prompt_relay_sigma(segment_length, free_window, epsilon, "paper_boundary")
        endpoint_distance = segment_length / 2.0
        retained = math.exp(-((endpoint_distance - free_window) ** 2) / (2.0 * sigma**2))

        self.assertAlmostEqual(retained, epsilon, places=12)

    def test_invalid_epsilon_keeps_reference_fallback(self):
        self.assertEqual(calculate_prompt_relay_sigma(10, 3, 0.0, "paper_boundary"), 0.1448)
        self.assertEqual(calculate_prompt_relay_sigma(10, 3, 1.0, "upstream_compat"), 0.1448)

    def test_paper_boundary_sigma_stays_positive_when_window_covers_endpoint(self):
        self.assertEqual(calculate_prompt_relay_sigma(10, 5, 0.1, "paper_boundary"), 1e-6)
        self.assertEqual(calculate_prompt_relay_sigma(10, 6, 0.1, "paper_boundary"), 1e-6)

    def test_lab_sigma_and_segments_match_main_node_critical_behavior(self):
        for mode in ("upstream_compat", "paper_boundary"):
            self.assertAlmostEqual(
                calculate_prompt_relay_sigma(76, 36, 0.001, mode),
                main_calculate_prompt_relay_sigma(76, 36, 0.001, mode),
                places=12,
            )
        self.assertEqual(
            calculate_prompt_relay_sigma(10, 5, 0.1, "paper_boundary"),
            main_calculate_prompt_relay_sigma(10, 5, 0.1, "paper_boundary"),
        )

        options = {
            "video_strength": 0.85,
            "video_window_scale": 1.1,
            "audio_epsilon": 0.1,
            "audio_strength": 0.7,
            "audio_window_scale": 0.5,
            "audio_frame_offset_frames": -1.25,
        }
        previous_torch = sys.modules.get("torch")
        sys.modules["torch"] = FakeTorch
        try:
            lab_segments = build_lab_segments([(10, 12), (12, 15)], [76, 141], 0.001, "paper_boundary", options)
        finally:
            if previous_torch is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = previous_torch
        main_segments = main_build_segments([(10, 12), (12, 15)], [76, 141], 0.001, {**options, "sigma_mode": "paper_boundary"})

        comparable_keys = {
            "local_token_idx",
            "midpoint",
            "midpoint_audio",
            "window",
            "sigma",
            "strength",
            "window_audio",
            "sigma_audio",
            "strength_audio",
        }
        self.assertEqual(
            [{key: segment[key] for key in comparable_keys} for segment in lab_segments],
            [{key: segment[key] for key in comparable_keys} for segment in main_segments],
        )

    def test_segment_length_parser_clamps_negative_pixel_lengths(self):
        self.assertEqual(
            _parse_segment_lengths("10, -5, 10", temporal_stride=1, latent_frames=15),
            [7, 1, 7],
        )

    def test_segment_length_parser_accepts_percentages_for_full_timeline(self):
        self.assertEqual(
            _parse_segment_lengths("35%,65%", temporal_stride=1, latent_frames=217),
            [76, 141],
        )

    def test_segment_length_parser_accepts_decimal_ratios_for_full_timeline(self):
        self.assertEqual(
            _parse_segment_lengths("0.25,0.75", temporal_stride=1, latent_frames=217),
            [54, 163],
        )

    def test_segment_length_parser_rejects_fractional_frame_counts(self):
        with self.assertRaisesRegex(ValueError, "whole numbers"):
            _parse_segment_lengths("16.5,24", temporal_stride=4, latent_frames=33)

    def test_direct_ratio_allocator_rejects_non_finite_weights_before_clamping(self):
        with self.assertRaisesRegex(ValueError, "finite numbers"):
            _allocate_full_latent_lengths([float("nan"), 1.0], latent_frames=217)
        with self.assertRaisesRegex(ValueError, "finite numbers"):
            _allocate_full_latent_lengths([float("-inf"), 1.0], latent_frames=217)

    def test_segment_length_parser_rejects_mixed_percentage_format(self):
        with self.assertRaisesRegex(ValueError, "one timing format"):
            _parse_segment_lengths("35%,133", temporal_stride=1, latent_frames=217)

    def test_segment_length_parser_rejects_empty_entries(self):
        for value in ("35%,,65%", "0.25,", ",24"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "empty entry"):
                    _parse_segment_lengths(value, temporal_stride=1, latent_frames=217)

    def test_segment_length_parser_rejects_non_finite_values(self):
        with self.assertRaisesRegex(ValueError, "finite numbers"):
            _parse_segment_lengths("inf%,50%", temporal_stride=1, latent_frames=217)
        with self.assertRaisesRegex(ValueError, "finite numbers"):
            _parse_segment_lengths("0.25,nan", temporal_stride=1, latent_frames=217)
        with self.assertRaisesRegex(ValueError, "finite numbers"):
            _parse_segment_lengths("10,inf", temporal_stride=4, latent_frames=33)

    def test_lab_segments_apply_ltxv_audio_offset_without_moving_video_midpoint(self):
        previous_torch = sys.modules.get("torch")
        sys.modules["torch"] = types.SimpleNamespace(arange=lambda start, end: list(range(start, end)))
        try:
            segments = build_lab_segments(
                token_ranges=[(10, 12), (12, 15)],
                segment_lengths=[76, 141],
                epsilon=0.001,
                sigma_mode="paper_boundary",
                relay_options={
                    "audio_epsilon": 0.1,
                    "audio_strength": 0.7,
                    "audio_window_scale": 0.5,
                    "audio_frame_offset_frames": -1.25,
                },
            )
        finally:
            if previous_torch is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = previous_torch

        self.assertEqual(segments[0]["midpoint"], 38)
        self.assertEqual(segments[1]["midpoint"], 146)
        self.assertEqual(segments[0]["midpoint_audio"], 36.75)
        self.assertEqual(segments[1]["midpoint_audio"], 144.75)
        self.assertEqual(segments[0]["strength_audio"], 0.7)
        self.assertLess(segments[0]["window_audio"], segments[0]["window"])

    def test_lab_segments_skip_non_positive_lengths_like_main_builder(self):
        previous_torch = sys.modules.get("torch")
        sys.modules["torch"] = types.SimpleNamespace(arange=lambda start, end: list(range(start, end)))
        try:
            lab_segments = build_lab_segments(
                token_ranges=[(0, 1), (1, 2), (2, 3), (3, 4)],
                segment_lengths=[4, 0, -3, 6],
                epsilon=0.001,
            )
        finally:
            if previous_torch is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = previous_torch

        main_segments = main_build_segments(
            token_ranges=[(0, 1), (1, 2), (2, 3), (3, 4)],
            segment_lengths=[4, 0, -3, 6],
            epsilon=0.001,
        )

        self.assertEqual([segment["local_token_idx"] for segment in lab_segments], [[0], [3]])
        self.assertEqual(
            [segment["midpoint"] for segment in lab_segments],
            [segment["midpoint"] for segment in main_segments],
        )
        self.assertEqual([segment["midpoint"] for segment in lab_segments], [2, 7])

    def test_lab_epsilon_clamps_non_finite_workflow_values(self):
        self.assertEqual(_sanitize_lab_epsilon("nan"), 0.001)
        self.assertEqual(_sanitize_lab_epsilon("inf"), 0.001)
        self.assertEqual(_sanitize_lab_epsilon(0.0), 0.001)
        self.assertEqual(_sanitize_lab_epsilon(1.0), 0.001)
        self.assertEqual(_sanitize_lab_epsilon("0.25"), 0.25)

    def test_lab_relay_options_clamp_non_finite_workflow_values(self):
        self.assertEqual(
            _sanitize_lab_relay_options(
                video_strength=-1,
                video_window_scale=float("inf"),
                audio_epsilon=1.2,
                audio_strength=float("nan"),
                audio_window_scale="bad",
                audio_frame_offset_frames=999,
            ),
            {
                "video_strength": 0.0,
                "video_window_scale": 1.0,
                "audio_epsilon": None,
                "audio_strength": 1.0,
                "audio_window_scale": 1.0,
                "audio_frame_offset_frames": 32.0,
            },
        )


if __name__ == "__main__":
    unittest.main()
