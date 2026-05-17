import builtins
import os
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from parser import parse_smart_prompt


class SmartPromptParserTests(unittest.TestCase):
    def test_inline_equal_segments_strip_whitespace(self):
        parsed = parse_smart_prompt("  wide shot  | camera pans | subject exits  ")

        self.assertEqual(
            parsed,
            [
                {"text": "wide shot", "weight": 1.0},
                {"text": "camera pans", "weight": 1.0},
                {"text": "subject exits", "weight": 1.0},
            ],
        )

    def test_inline_range_tags_accept_spaces_and_unicode_dashes(self):
        parsed = parse_smart_prompt("wide shot [0 – 25] | close-up [25—100]")

        self.assertEqual(parsed[0], {"text": "wide shot", "weight": 25.0})
        self.assertEqual(parsed[1], {"text": "close-up", "weight": 75.0})

    def test_inline_range_tags_accept_word_separators(self):
        parsed = parse_smart_prompt("wide shot [0 to 25] | close-up [25 THRU 100]")

        self.assertEqual(parsed[0], {"text": "wide shot", "weight": 25.0})
        self.assertEqual(parsed[1], {"text": "close-up", "weight": 75.0})

    def test_inline_malformed_numeric_tag_is_left_as_text(self):
        parsed = parse_smart_prompt("wide shot [1.2.3] | close-up [2]")

        self.assertEqual(parsed[0], {"text": "wide shot [1.2.3]", "weight": 1.0})
        self.assertEqual(parsed[1], {"text": "close-up", "weight": 2.0})

    def test_inline_weight_tag_strips_only_the_marker_that_supplies_weight(self):
        parsed = parse_smart_prompt("wide shot [2] with visible cue [3] | close-up [1]")

        self.assertEqual(parsed[0], {"text": "wide shot with visible cue [3]", "weight": 2.0})
        self.assertEqual(parsed[1], {"text": "close-up", "weight": 1.0})

    def test_inline_reversed_or_zero_ranges_do_not_create_non_positive_weights(self):
        parsed = parse_smart_prompt("wide shot [50-25] | close-up [25-25] | final [2]")

        self.assertEqual(
            parsed,
            [
                {"text": "wide shot", "weight": 1.0},
                {"text": "close-up", "weight": 1.0},
                {"text": "final", "weight": 2.0},
            ],
        )

    def test_block_decimal_range_headers_preserve_fractional_span(self):
        parsed = parse_smart_prompt(
            "Beat 0.5-1.25:\n"
            "A dancer turns\n"
            "Beat 1.25-3.0:\n"
            "The camera pulls back\n"
        )

        self.assertEqual(parsed[0], {"text": "A dancer turns", "weight": 0.75})
        self.assertEqual(parsed[1], {"text": "The camera pulls back", "weight": 1.75})

    def test_block_word_number_ranges_use_span_weight(self):
        parsed = parse_smart_prompt(
            "Scene one to three:\n"
            "A slow walk through fog\n"
            "Scene three through seven:\n"
            "The subject starts running\n"
        )

        self.assertEqual(
            parsed,
            [
                {"text": "A slow walk through fog", "weight": 2},
                {"text": "The subject starts running", "weight": 4},
            ],
        )

    def test_block_ordinal_headers_work_without_optional_word2number_dependency(self):
        parsed = parse_smart_prompt(
            "Scene first through third:\n"
            "A character studies the room\n"
            "Scene 4th:\n"
            "A door opens\n"
        )

        self.assertEqual(
            parsed,
            [
                {"text": "A character studies the room", "weight": 2},
                {"text": "A door opens", "weight": 1.0},
            ],
        )

    def test_block_compound_ordinal_header_parses_common_llm_labels(self):
        parsed = parse_smart_prompt(
            "Beat twenty first:\n"
            "A final reveal under warm light\n"
        )

        self.assertEqual(parsed, [{"text": "A final reveal under warm light", "weight": 1.0}])

    def test_block_thirty_plus_word_number_ranges_work_without_word2number(self):
        real_import = builtins.__import__

        def import_without_word2number(name, *args, **kwargs):
            if name == "word2number" or name.startswith("word2number."):
                raise ImportError("word2number intentionally hidden for parser fallback test")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=import_without_word2number):
            parsed = parse_smart_prompt(
                "Scene thirty first through thirty fourth:\n"
                "A long-form dialogue beat continues\n"
                "Scene fortieth:\n"
                "A quiet cutaway\n"
            )

        self.assertEqual(
            parsed,
            [
                {"text": "A long-form dialogue beat continues", "weight": 3},
                {"text": "A quiet cutaway", "weight": 1.0},
            ],
        )

    def test_block_dash_joined_word_ranges_work_without_breaking_compound_ordinals(self):
        parsed = parse_smart_prompt(
            "Scene first-third:\n"
            "A short dialogue exchange\n"
            "Beat twenty-first–twenty-fourth:\n"
            "A longer long-form transition\n"
            "Beat twenty-fifth:\n"
            "A final insert shot\n"
        )

        self.assertEqual(
            parsed,
            [
                {"text": "A short dialogue exchange", "weight": 2},
                {"text": "A longer long-form transition", "weight": 3},
                {"text": "A final insert shot", "weight": 1.0},
            ],
        )

    def test_block_headers_use_range_span_and_strip_markers(self):
        parsed = parse_smart_prompt(
            "Scene 1—3:\n"
            "A robot waits in a neon room\n"
            "Beat 3-8:\n"
            "The robot opens the door\n"
        )

        self.assertEqual(
            parsed,
            [
                {"text": "A robot waits in a neon room", "weight": 2},
                {"text": "The robot opens the door", "weight": 5},
            ],
        )

    def test_block_bracketed_header_tags_use_range_span_and_strip_markers(self):
        parsed = parse_smart_prompt(
            "Scene [0 – 25]:\n"
            "Wide establishing shot\n"
            "Beat [25 to 100]:\n"
            "Close-up dialogue beat\n"
            "Shot [3]:\n"
            "Final insert\n"
        )

        self.assertEqual(
            parsed,
            [
                {"text": "Wide establishing shot", "weight": 25.0},
                {"text": "Close-up dialogue beat", "weight": 75.0},
                {"text": "Final insert", "weight": 1.0},
            ],
        )

    def test_block_bracketed_header_requires_textual_prefix(self):
        parsed = parse_smart_prompt(
            "[0-25]:\n"
            "This bracketed line is preamble, not a segment header\n"
            "Scene [25-50]:\n"
            "Actual segment text\n"
        )

        self.assertEqual(parsed, [{"text": "Actual segment text", "weight": 25.0}])

    def test_block_syntax_ignores_llm_preamble_before_first_header(self):
        parsed = parse_smart_prompt(
            "Here is the formatted prompt:\n\n"
            "Scene 1:\n"
            "A static establishing shot\n"
            "Scene 2:\n"
            "The subject turns toward camera\n"
        )

        self.assertEqual(
            parsed,
            [
                {"text": "A static establishing shot", "weight": 1.0},
                {"text": "The subject turns toward camera", "weight": 1.0},
            ],
        )

    def test_block_headers_accept_common_markdown_decorated_llm_headers(self):
        parsed = parse_smart_prompt(
            "### Scene 1:\n"
            "A static establishing shot\n"
            "**Scene 2-4:**\n"
            "The subject turns toward camera\n"
            "__Beat fifth:__\n"
            "A final insert shot\n"
        )

        self.assertEqual(
            parsed,
            [
                {"text": "A static establishing shot", "weight": 1.0},
                {"text": "The subject turns toward camera", "weight": 2.0},
                {"text": "A final insert shot", "weight": 1.0},
            ],
        )

    def test_block_body_inline_tag_overrides_header_weight(self):
        parsed = parse_smart_prompt(
            "Scene 1-10:\n"
            "Establishing shot [2]\n"
            "Scene 10-20:\n"
            "Fast action [2:5]\n"
        )

        self.assertEqual(
            parsed,
            [
                {"text": "Establishing shot", "weight": 2.0},
                {"text": "Fast action", "weight": 3.0},
            ],
        )

    def test_block_reversed_or_zero_ranges_fall_back_to_equal_weight(self):
        parsed = parse_smart_prompt(
            "Scene 5-3:\n"
            "A typo in an LLM scene range\n"
            "Scene 7-7:\n"
            "A zero-width marker should not disappear\n"
            "Scene 7-10:\n"
            "A normal span still applies\n"
        )

        self.assertEqual(
            parsed,
            [
                {"text": "A typo in an LLM scene range", "weight": 1.0},
                {"text": "A zero-width marker should not disappear", "weight": 1.0},
                {"text": "A normal span still applies", "weight": 3},
            ],
        )

    def test_block_body_strips_common_llm_list_markers(self):
        parsed = parse_smart_prompt(
            "Scene 1:\n"
            "- Establishing shot across the lab\n"
            "Scene 2:\n"
            "1. Close-up of the relay console\n"
            "Scene 3:\n"
            "* Subject exits through fog [3]\n"
        )

        self.assertEqual(
            parsed,
            [
                {"text": "Establishing shot across the lab", "weight": 1.0},
                {"text": "Close-up of the relay console", "weight": 1.0},
                {"text": "Subject exits through fog", "weight": 3.0},
            ],
        )

    def test_block_body_preserves_internal_hyphens_after_stripping_list_marker(self):
        parsed = parse_smart_prompt(
            "Beat 1:\n"
            "- Long-form dialogue hand-off stays stable\n"
        )

        self.assertEqual(parsed, [{"text": "Long-form dialogue hand-off stays stable", "weight": 1.0}])


if __name__ == "__main__":
    unittest.main()
