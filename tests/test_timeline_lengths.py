import ast
import os
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODES_PATH = os.path.join(ROOT, "nodes.py")


def load_length_helpers():
    """Load the pure timeline helpers without importing ComfyUI deps."""
    with open(NODES_PATH, "r", encoding="utf-8") as f:
        module = ast.parse(f.read(), filename=NODES_PATH)

    wanted = {"_convert_to_latent_lengths", "_allocate_full_latent_lengths", "_parse_segment_lengths"}
    fns = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    isolated = ast.Module(body=fns, type_ignores=[])
    ast.fix_missing_locations(isolated)

    namespace = {}
    exec(compile(isolated, NODES_PATH, "exec"), namespace)
    return namespace


HELPERS = load_length_helpers()
_convert_to_latent_lengths = HELPERS["_convert_to_latent_lengths"]
_allocate_full_latent_lengths = HELPERS["_allocate_full_latent_lengths"]
_parse_segment_lengths = HELPERS["_parse_segment_lengths"]


class TimelineLengthConversionTests(unittest.TestCase):
    def test_empty_or_non_positive_lengths_stay_safe(self):
        self.assertEqual(_convert_to_latent_lengths([], temporal_stride=4, latent_frames=33), [])
        self.assertEqual(_convert_to_latent_lengths([0, 0], temporal_stride=4, latent_frames=33), [1, 1])

    def test_full_coverage_pins_to_latent_frames_when_within_one_stride(self):
        self.assertEqual(
            _convert_to_latent_lengths([40, 60, 28], temporal_stride=4, latent_frames=33),
            [10, 16, 7],
        )

    def test_partial_timeline_stays_partial_in_latent_space(self):
        converted = _convert_to_latent_lengths([16, 24], temporal_stride=4, latent_frames=33)

        self.assertEqual(converted, [4, 6])
        self.assertLess(sum(converted), 33)

    def test_largest_remainder_allocates_rounding_gap(self):
        self.assertEqual(
            _convert_to_latent_lengths([2, 2, 2], temporal_stride=1, latent_frames=5),
            [2, 2, 1],
        )

    def test_tiny_segments_keep_at_least_one_latent_frame(self):
        converted = _convert_to_latent_lengths([1, 100, 1], temporal_stride=4, latent_frames=10)

        self.assertEqual(sum(converted), 10)
        self.assertEqual(converted[0], 1)
        self.assertEqual(converted[2], 1)
        self.assertGreater(converted[1], 1)

    def test_negative_lengths_are_treated_as_zero_before_allocation(self):
        converted = _convert_to_latent_lengths([10, -5, 10], temporal_stride=1, latent_frames=15)

        self.assertEqual(converted, [7, 1, 7])
        self.assertEqual(sum(converted), 15)
        self.assertTrue(all(length >= 1 for length in converted))

    def test_percentage_segment_lengths_cover_full_ltxv_timeline(self):
        converted = _parse_segment_lengths("35%,65%", temporal_stride=1, latent_frames=217)

        self.assertEqual(converted, [76, 141])
        self.assertEqual(sum(converted), 217)

    def test_decimal_ratio_segment_lengths_cover_full_ltxv_timeline(self):
        converted = _parse_segment_lengths("0.25,0.75", temporal_stride=1, latent_frames=217)

        self.assertEqual(converted, [54, 163])
        self.assertEqual(sum(converted), 217)

    def test_direct_ratio_allocator_rejects_non_finite_weights_before_clamping(self):
        with self.assertRaisesRegex(ValueError, "finite numbers"):
            _allocate_full_latent_lengths([float("nan"), 1.0], latent_frames=33)
        with self.assertRaisesRegex(ValueError, "finite numbers"):
            _allocate_full_latent_lengths([float("-inf"), 1.0], latent_frames=33)

    def test_many_ratio_segments_cover_2500_plus_frame_long_form_timeline(self):
        converted = _parse_segment_lengths(
            ",".join(["1.0"] * 24),
            temporal_stride=1,
            latent_frames=2501,
        )

        self.assertEqual(len(converted), 24)
        self.assertEqual(sum(converted), 2501)
        self.assertEqual(max(converted) - min(converted), 1)

    def test_integer_segment_lengths_preserve_existing_frame_count_conversion(self):
        converted = _parse_segment_lengths("16,24", temporal_stride=4, latent_frames=33)

        self.assertEqual(converted, [4, 6])

    def test_integral_float_frame_counts_are_accepted_without_silent_truncation(self):
        converted = _parse_segment_lengths("16.0,24.0", temporal_stride=4, latent_frames=33)

        self.assertEqual(converted, [4, 6])

    def test_fractional_frame_counts_raise_instead_of_truncating(self):
        with self.assertRaisesRegex(ValueError, "whole numbers"):
            _parse_segment_lengths("16.5,24", temporal_stride=4, latent_frames=33)

    def test_mixed_percentage_and_frame_counts_raise_actionable_error(self):
        with self.assertRaisesRegex(ValueError, "one timing format"):
            _parse_segment_lengths("35%,133", temporal_stride=1, latent_frames=217)

    def test_invalid_segment_length_token_raises_actionable_error(self):
        with self.assertRaisesRegex(ValueError, "Examples"):
            _parse_segment_lengths("first half, second half", temporal_stride=1, latent_frames=217)

    def test_empty_segment_length_entries_raise_actionable_error(self):
        for value in ("16,,24", "16,", ",24"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "empty entry"):
                    _parse_segment_lengths(value, temporal_stride=4, latent_frames=33)

    def test_non_finite_segment_lengths_raise_actionable_error(self):
        with self.assertRaisesRegex(ValueError, "finite numbers"):
            _parse_segment_lengths("nan,24", temporal_stride=4, latent_frames=33)
        with self.assertRaisesRegex(ValueError, "finite numbers"):
            _parse_segment_lengths("inf%,50%", temporal_stride=1, latent_frames=217)
        with self.assertRaisesRegex(ValueError, "finite numbers"):
            _parse_segment_lengths("0.25,inf", temporal_stride=1, latent_frames=217)

    def test_zero_percentage_lengths_raise_actionable_error(self):
        with self.assertRaisesRegex(ValueError, "positive proportions"):
            _parse_segment_lengths("0%,0%", temporal_stride=1, latent_frames=217)


if __name__ == "__main__":
    unittest.main()
