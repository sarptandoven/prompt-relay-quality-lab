import importlib.util
import math
import os
import sys
import types
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPT_RELAY_PATH = os.path.join(ROOT, "prompt_relay.py")


class FakeTorch(types.SimpleNamespace):
    @staticmethod
    def arange(start, end=None, **_kwargs):
        if end is None:
            start, end = 0, start
        return list(range(start, end))


class PromptRelaySegmentMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        previous_torch = sys.modules.get("torch")
        sys.modules["torch"] = FakeTorch()
        try:
            spec = importlib.util.spec_from_file_location("prompt_relay_under_test", PROMPT_RELAY_PATH)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            cls.prompt_relay = module
        finally:
            if previous_torch is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = previous_torch

    def test_long_form_chunk_planner_bounds_mask_budget_for_2500_plus_frames(self):
        plan = self.prompt_relay.plan_temporal_chunks(
            latent_frames=2501,
            tokens_per_frame=4096,
            text_tokens=128,
            max_mask_elements=self.prompt_relay.DEFAULT_MAX_MASK_ELEMENTS,
            overlap_frames=16,
            safety_margin=1.0,
        )

        self.assertEqual(plan["max_chunk_frames"], 2048)
        self.assertEqual(plan["overlap_frames"], 16)
        self.assertEqual(plan["chunks"], [
            {"start": 0, "end": 2048, "length": 2048},
            {"start": 2032, "end": 2501, "length": 469},
        ])
        for chunk in plan["chunks"]:
            self.assertLessEqual(
                chunk["length"] * 4096 * 128,
                self.prompt_relay.DEFAULT_MAX_MASK_ELEMENTS,
            )

    def test_chunk_planner_handles_unbounded_cap_as_single_window(self):
        plan = self.prompt_relay.plan_temporal_chunks(
            latent_frames=10000,
            tokens_per_frame=4096,
            text_tokens=128,
            max_mask_elements=0,
        )

        self.assertEqual(plan["chunks"], [{"start": 0, "end": 10000, "length": 10000}])
        self.assertEqual(plan["overlap_frames"], 0)

    def test_chunk_planner_rejects_when_one_frame_exceeds_budget(self):
        with self.assertRaisesRegex(ValueError, "one latent frame would exceed the mask budget"):
            self.prompt_relay.plan_temporal_chunks(
                latent_frames=2501,
                tokens_per_frame=4096,
                text_tokens=128,
                max_mask_elements=100_000,
            )

    def test_chunk_stitch_ranges_split_overlap_into_non_overlapping_kept_frames(self):
        stitched = self.prompt_relay.plan_chunk_stitch_ranges([
            {"start": 0, "end": 10, "length": 10},
            {"start": 6, "end": 16, "length": 10},
            {"start": 12, "end": 20, "length": 8},
        ])

        self.assertEqual(
            [(chunk["keep_start"], chunk["keep_end"]) for chunk in stitched],
            [(0, 8), (8, 14), (14, 20)],
        )
        self.assertEqual(
            [(chunk["trim_start"], chunk["trim_end"]) for chunk in stitched],
            [(0, 2), (2, 2), (2, 0)],
        )
        self.assertEqual(sum(chunk["keep_length"] for chunk in stitched), 20)

    def test_chunk_stitch_ranges_reject_gapped_chunks(self):
        with self.assertRaisesRegex(ValueError, "contiguous or overlapping"):
            self.prompt_relay.plan_chunk_stitch_ranges([
                {"start": 0, "end": 10},
                {"start": 12, "end": 20},
            ])


    def test_chunk_handoffs_report_overlap_and_context_windows(self):
        plan = self.prompt_relay.plan_temporal_chunks(
            latent_frames=2501,
            tokens_per_frame=4096,
            text_tokens=128,
            max_mask_elements=self.prompt_relay.DEFAULT_MAX_MASK_ELEMENTS,
            overlap_frames=16,
            safety_margin=1.0,
        )

        handoffs = self.prompt_relay.plan_chunk_handoffs(plan["chunks"], min_context_frames=8)

        self.assertEqual(handoffs, [{
            "prev_index": 0,
            "next_index": 1,
            "seam_frame": 2040,
            "overlap_start": 2032,
            "overlap_end": 2048,
            "overlap_length": 16,
            "prev_context_start": 2032,
            "prev_context_end": 2040,
            "next_context_start": 2040,
            "next_context_end": 2048,
            "status": "ok",
        }])

    def test_chunk_handoffs_flag_weak_boundaries_before_render(self):
        self.assertEqual(
            self.prompt_relay.plan_chunk_handoffs([
                {"start": 0, "end": 10},
                {"start": 10, "end": 20},
            ])[0]["status"],
            "hard_cut",
        )
        self.assertEqual(
            self.prompt_relay.plan_chunk_handoffs([
                {"start": 0, "end": 10},
                {"start": 8, "end": 18},
            ], min_context_frames=4)[0]["status"],
            "short_overlap",
        )

    def test_chunk_handoff_diagnostics_are_bounded_and_show_problem_seams(self):
        chunks = []
        start = 0
        for _idx in range(14):
            chunks.append({"start": start, "end": start + 20})
            start += 18
        handoffs = self.prompt_relay.plan_chunk_handoffs(chunks, min_context_frames=4)

        summary = self.prompt_relay.format_chunk_handoff_diagnostics(handoffs, max_handoffs=6)

        self.assertIn("handoff0: chunk0->1 seam=19 overlap=[18:20]", summary)
        self.assertIn("status=short_overlap", summary)
        self.assertIn("... 7 handoff(s) omitted ...", summary)
        self.assertIn("handoff12: chunk12->13", summary)
        self.assertLess(len(summary), 1200)


if __name__ == "__main__":
    unittest.main()
