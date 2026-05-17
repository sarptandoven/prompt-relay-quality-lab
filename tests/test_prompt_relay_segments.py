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

    def test_default_segment_metadata_preserves_current_sigma_and_window_formula(self):
        segments = self.prompt_relay.build_segments(
            token_ranges=[(3, 7)],
            segment_lengths=[10],
            epsilon=1e-3,
        )

        self.assertEqual(len(segments), 1)
        segment = segments[0]
        self.assertEqual(segment["local_token_idx"], [3, 4, 5, 6])
        self.assertEqual(segment["midpoint"], 5)
        self.assertEqual(segment["midpoint_audio"], 5.0)
        self.assertEqual(segment["window"], 3)
        self.assertEqual(segment["window_audio"], 3)
        self.assertAlmostEqual(segment["sigma"], 1.0 / math.log(1000.0), places=12)
        self.assertAlmostEqual(segment["sigma_audio"], segment["sigma"], places=12)
        self.assertEqual(segment["strength"], 1.0)
        self.assertEqual(segment["strength_audio"], 1.0)

    def test_audio_options_are_opt_in_and_do_not_change_video_metadata(self):
        baseline = self.prompt_relay.build_segments(
            token_ranges=[(0, 2)],
            segment_lengths=[8],
            epsilon=1e-3,
        )[0]
        audio_tuned = self.prompt_relay.build_segments(
            token_ranges=[(0, 2)],
            segment_lengths=[8],
            epsilon=1e-3,
            relay_options={
                "audio_epsilon": 0.1,
                "audio_strength": 0.25,
                "audio_window_scale": 2.0,
                "audio_frame_offset_frames": -1.5,
            },
        )[0]

        for key in ("local_token_idx", "midpoint", "window", "sigma", "strength"):
            self.assertEqual(audio_tuned[key], baseline[key])
        self.assertAlmostEqual(audio_tuned["sigma_audio"], 1.0 / math.log(10.0), places=12)
        self.assertEqual(audio_tuned["strength_audio"], 0.25)
        self.assertEqual(audio_tuned["window_audio"], baseline["window"] * 2.0)
        self.assertEqual(audio_tuned["midpoint_audio"], baseline["midpoint"] - 1.5)

    def test_video_options_only_change_metadata_when_explicitly_supplied(self):
        segment = self.prompt_relay.build_segments(
            token_ranges=[(0, 1)],
            segment_lengths=[12],
            epsilon=1e-3,
            relay_options={
                "video_strength": 0.5,
                "video_window_scale": 1.5,
            },
        )[0]

        self.assertEqual(segment["midpoint"], 6)
        self.assertEqual(segment["window"], 6.0)
        self.assertEqual(segment["strength"], 0.5)
        self.assertEqual(segment["window_audio"], 4.0)
        self.assertEqual(segment["strength_audio"], 1.0)

    def test_paper_boundary_sigma_is_opt_in_for_main_node_segments(self):
        upstream = self.prompt_relay.build_segments(
            token_ranges=[(0, 1)],
            segment_lengths=[10],
            epsilon=0.1,
        )[0]
        paper = self.prompt_relay.build_segments(
            token_ranges=[(0, 1)],
            segment_lengths=[10],
            epsilon=0.1,
            relay_options={"sigma_mode": "paper_boundary"},
        )[0]

        self.assertAlmostEqual(upstream["sigma"], 1.0 / math.log(10.0), places=12)
        self.assertNotAlmostEqual(paper["sigma"], upstream["sigma"], places=6)
        retained_at_endpoint = math.exp(-(((10 / 2.0) - paper["window"]) ** 2) / (2 * paper["sigma"] ** 2))
        self.assertAlmostEqual(retained_at_endpoint, 0.1, places=12)
        self.assertAlmostEqual(paper["sigma_audio"], paper["sigma"], places=12)

    def test_non_positive_segments_are_skipped_without_shifting_future_midpoints_forward(self):
        segments = self.prompt_relay.build_segments(
            token_ranges=[(0, 1), (1, 2), (2, 3), (3, 4)],
            segment_lengths=[4, 0, -3, 6],
            epsilon=1e-3,
        )

        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0]["local_token_idx"], [0])
        self.assertEqual(segments[0]["midpoint"], 2)
        self.assertEqual(segments[1]["local_token_idx"], [3])
        self.assertEqual(segments[1]["midpoint"], 7)

    def test_ltxv_217_frame_two_dialogue_paper_options_have_loggable_summary(self):
        segments = self.prompt_relay.build_segments(
            token_ranges=[(10, 13), (13, 18)],
            segment_lengths=[76, 141],
            epsilon=0.001,
            relay_options={
                "sigma_mode": "paper_boundary",
                "audio_epsilon": 0.1,
                "audio_strength": 0.7,
                "audio_window_scale": 0.5,
                "audio_frame_offset_frames": -1.25,
            },
        )

        self.assertEqual([segment["midpoint"] for segment in segments], [38, 146])
        self.assertEqual([segment["midpoint_audio"] for segment in segments], [36.75, 144.75])
        for segment, length in zip(segments, [76, 141]):
            retained = math.exp(-(((length / 2.0) - segment["window"]) ** 2) / (2 * segment["sigma"] ** 2))
            self.assertAlmostEqual(retained, 0.001, places=12)

        summary = self.prompt_relay.format_segment_diagnostics(segments)
        self.assertIn("seg0: tokens=[10:13] mid=38.000", summary)
        self.assertIn("audio_mid=36.750", summary)
        self.assertIn("seg1: tokens=[13:18] mid=146.000", summary)
        self.assertIn("audio_str=0.700", summary)

    def test_many_long_form_segments_have_bounded_diagnostics(self):
        segments = self.prompt_relay.build_segments(
            token_ranges=[(idx, idx + 1) for idx in range(24)],
            segment_lengths=[104] * 23 + [109],
            epsilon=0.001,
            relay_options={"audio_frame_offset_frames": -0.5},
        )

        self.assertEqual(len(segments), 24)
        self.assertEqual(sum([104] * 23 + [109]), 2501)

        summary = self.prompt_relay.format_segment_diagnostics(segments)

        self.assertIn("seg0: tokens=[0:1]", summary)
        self.assertIn("... 12 segment(s) omitted ...", summary)
        self.assertIn("seg23: tokens=[23:24]", summary)
        self.assertIn("audio_mid=", summary)
        self.assertLess(len(summary), 1800)

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

    def test_quality_chunk_planner_prefers_target_windows_under_mask_budget(self):
        plan = self.prompt_relay.plan_quality_temporal_chunks(
            latent_frames=2501,
            tokens_per_frame=4096,
            text_tokens=128,
            max_mask_elements=self.prompt_relay.DEFAULT_MAX_MASK_ELEMENTS,
            target_chunk_frames=257,
            overlap_frames=16,
            safety_margin=1.0,
        )

        self.assertEqual(plan["budget_max_chunk_frames"], 2048)
        self.assertEqual(plan["quality_target_frames"], 257)
        self.assertEqual(plan["max_chunk_frames"], 257)
        self.assertEqual(plan["overlap_frames"], 16)
        self.assertEqual(plan["chunks"][:2], [
            {"start": 0, "end": 257, "length": 257},
            {"start": 241, "end": 498, "length": 257},
        ])
        self.assertEqual(plan["chunks"][-1]["end"], 2501)
        self.assertGreater(len(plan["chunks"]), 2)

    def test_quality_chunk_planner_still_honors_tighter_mask_budget(self):
        plan = self.prompt_relay.plan_quality_temporal_chunks(
            latent_frames=2501,
            tokens_per_frame=4096,
            text_tokens=128,
            max_mask_elements=512 * 4096 * 128,
            target_chunk_frames=1024,
            overlap_frames=16,
            safety_margin=1.0,
        )

        self.assertEqual(plan["budget_max_chunk_frames"], 512)
        self.assertEqual(plan["max_chunk_frames"], 512)
        self.assertTrue(all(chunk["length"] <= 512 for chunk in plan["chunks"]))

    def test_chunk_planner_handles_unbounded_cap_as_single_window(self):
        plan = self.prompt_relay.plan_temporal_chunks(
            latent_frames=10000,
            tokens_per_frame=4096,
            text_tokens=128,
            max_mask_elements=0,
        )

        self.assertEqual(plan["chunks"], [{"start": 0, "end": 10000, "length": 10000}])
        self.assertEqual(plan["overlap_frames"], 0)

    def test_paged_chunk_planner_streams_huge_timelines_without_full_materialization(self):
        page = self.prompt_relay.plan_temporal_chunk_page(
            latent_frames=10**12,
            tokens_per_frame=4096,
            text_tokens=128,
            max_mask_elements=self.prompt_relay.DEFAULT_MAX_MASK_ELEMENTS,
            overlap_frames=16,
            safety_margin=1.0,
            start_chunk=0,
            max_chunks=3,
        )

        self.assertEqual(page["max_chunk_frames"], 2048)
        self.assertEqual(page["overlap_frames"], 16)
        self.assertFalse(page["complete"])
        self.assertEqual(page["next_chunk"], 3)
        self.assertGreater(page["total_chunks"], 100_000_000)
        self.assertEqual(page["chunks"], [
            {"start": 0, "end": 2048, "length": 2048},
            {"start": 2032, "end": 4080, "length": 2048},
            {"start": 4064, "end": 6112, "length": 2048},
        ])

    def test_paged_chunk_planner_resumes_from_next_cursor(self):
        first = self.prompt_relay.plan_temporal_chunk_page(
            latent_frames=5000,
            tokens_per_frame=4096,
            text_tokens=128,
            max_mask_elements=self.prompt_relay.DEFAULT_MAX_MASK_ELEMENTS,
            overlap_frames=16,
            safety_margin=1.0,
            max_chunks=1,
        )
        second = self.prompt_relay.plan_temporal_chunk_page(
            latent_frames=5000,
            tokens_per_frame=4096,
            text_tokens=128,
            max_mask_elements=self.prompt_relay.DEFAULT_MAX_MASK_ELEMENTS,
            overlap_frames=16,
            safety_margin=1.0,
            start_chunk=first["next_chunk"],
            max_chunks=4,
        )
        full = self.prompt_relay.plan_temporal_chunks(
            latent_frames=5000,
            tokens_per_frame=4096,
            text_tokens=128,
            max_mask_elements=self.prompt_relay.DEFAULT_MAX_MASK_ELEMENTS,
            overlap_frames=16,
            safety_margin=1.0,
        )

        self.assertEqual(first["chunks"] + second["chunks"], full["chunks"])
        self.assertTrue(second["complete"])
        self.assertIsNone(second["next_chunk"])

    def test_quality_paged_chunk_planner_streams_target_windows_without_full_materialization(self):
        page = self.prompt_relay.plan_quality_temporal_chunk_page(
            latent_frames=10**12,
            tokens_per_frame=4096,
            text_tokens=128,
            max_mask_elements=self.prompt_relay.DEFAULT_MAX_MASK_ELEMENTS,
            target_chunk_frames=257,
            overlap_frames=16,
            safety_margin=1.0,
            max_chunks=3,
        )

        self.assertEqual(page["budget_max_chunk_frames"], 2048)
        self.assertEqual(page["quality_target_frames"], 257)
        self.assertEqual(page["max_chunk_frames"], 257)
        self.assertEqual(page["overlap_frames"], 16)
        self.assertFalse(page["complete"])
        self.assertEqual(page["next_chunk"], 3)
        self.assertGreater(page["total_chunks"], 1_000_000_000)
        self.assertEqual(page["chunks"], [
            {"start": 0, "end": 257, "length": 257},
            {"start": 241, "end": 498, "length": 257},
            {"start": 482, "end": 739, "length": 257},
        ])

    def test_quality_paged_chunk_planner_matches_full_quality_plan_by_cursor(self):
        first = self.prompt_relay.plan_quality_temporal_chunk_page(
            latent_frames=2501,
            tokens_per_frame=4096,
            text_tokens=128,
            max_mask_elements=self.prompt_relay.DEFAULT_MAX_MASK_ELEMENTS,
            target_chunk_frames=257,
            overlap_frames=16,
            safety_margin=1.0,
            max_chunks=2,
        )
        second = self.prompt_relay.plan_quality_temporal_chunk_page(
            latent_frames=2501,
            tokens_per_frame=4096,
            text_tokens=128,
            max_mask_elements=self.prompt_relay.DEFAULT_MAX_MASK_ELEMENTS,
            target_chunk_frames=257,
            overlap_frames=16,
            safety_margin=1.0,
            start_chunk=first["next_chunk"],
            max_chunks=32,
        )
        full = self.prompt_relay.plan_quality_temporal_chunks(
            latent_frames=2501,
            tokens_per_frame=4096,
            text_tokens=128,
            max_mask_elements=self.prompt_relay.DEFAULT_MAX_MASK_ELEMENTS,
            target_chunk_frames=257,
            overlap_frames=16,
            safety_margin=1.0,
        )

        self.assertEqual(first["chunks"] + second["chunks"], full["chunks"])
        self.assertTrue(second["complete"])
        self.assertIsNone(second["next_chunk"])

    def test_chunk_plan_diagnostics_report_budget_and_stitch_coverage(self):
        plan = self.prompt_relay.plan_temporal_chunks(
            latent_frames=2501,
            tokens_per_frame=4096,
            text_tokens=128,
            max_mask_elements=self.prompt_relay.DEFAULT_MAX_MASK_ELEMENTS,
            overlap_frames=16,
            safety_margin=1.0,
        )
        handoffs = self.prompt_relay.plan_chunk_handoffs(plan["chunks"], min_context_frames=8)

        summary = self.prompt_relay.format_temporal_chunk_plan_diagnostics(plan, handoffs)

        self.assertIn("chunk_plan: frames=2501 chunks=2 max_chunk=2048 overlap=16", summary)
        self.assertIn("stitched_frames=2501", summary)
        self.assertIn("worst_handoff=ok", summary)
        self.assertIn("chunk0=[0:2048] len=2048", summary)
        self.assertIn("chunk1=[2032:2501] len=469", summary)

    def test_chunk_plan_diagnostics_are_bounded_for_many_windows(self):
        chunks = []
        start = 0
        for _idx in range(14):
            chunks.append({"start": start, "end": start + 20, "length": 20})
            start += 18
        plan = {"max_chunk_frames": 20, "overlap_frames": 2, "chunks": chunks}
        handoffs = self.prompt_relay.plan_chunk_handoffs(chunks, min_context_frames=4)

        summary = self.prompt_relay.format_temporal_chunk_plan_diagnostics(plan, handoffs, max_chunks=4)

        self.assertIn("chunk_plan: frames=254 chunks=14 max_chunk=20 overlap=2", summary)
        self.assertIn("worst_handoff=short_overlap", summary)
        self.assertIn("... 10 chunk(s) omitted ...", summary)
        self.assertIn("chunk13=[234:254]", summary)
        self.assertLess(len(summary), 500)

    def test_chunk_planner_rejects_when_one_frame_exceeds_budget(self):
        with self.assertRaisesRegex(ValueError, "one latent frame would exceed the mask budget"):
            self.prompt_relay.plan_temporal_chunks(
                latent_frames=2501,
                tokens_per_frame=4096,
                text_tokens=128,
                max_mask_elements=100_000,
            )

    def test_clip_segments_to_chunk_keeps_only_visible_global_prompt_beats(self):
        window = self.prompt_relay.clip_segments_to_chunk(
            token_ranges=[(0, 2), (2, 5), (5, 6)],
            segment_lengths=[10, 20, 10],
            chunk_start=8,
            chunk_end=32,
        )

        self.assertEqual(window["chunk_start"], 8)
        self.assertEqual(window["chunk_end"], 32)
        self.assertEqual(window["token_ranges"], [(0, 2), (2, 5), (5, 6)])
        self.assertEqual(window["segment_lengths"], [2, 20, 2])
        self.assertEqual(window["source_indices"], [0, 1, 2])
        self.assertEqual(window["global_ranges"], [(8, 10), (10, 30), (30, 32)])

        chunk_segments = self.prompt_relay.build_segments(
            window["token_ranges"],
            window["segment_lengths"],
            epsilon=0.001,
        )
        self.assertEqual([segment["midpoint"] for segment in chunk_segments], [1, 12, 23])

    def test_clip_segments_to_chunk_can_drop_tiny_boundary_slivers(self):
        window = self.prompt_relay.clip_segments_to_chunk(
            token_ranges=[(0, 1), (1, 2), (2, 3)],
            segment_lengths=[5, 5, 5],
            chunk_start=4,
            chunk_end=11,
            min_visible_frames=2,
        )

        self.assertEqual(window["token_ranges"], [(1, 2)])
        self.assertEqual(window["segment_lengths"], [5])
        self.assertEqual(window["source_indices"], [1])
        self.assertEqual(window["global_ranges"], [(5, 10)])
        self.assertEqual(window["local_ranges"], [(1, 6)])

    def test_build_chunk_segments_preserves_chunk_local_timing_gaps(self):
        window = self.prompt_relay.clip_segments_to_chunk(
            token_ranges=[(0, 2), (2, 5), (5, 7)],
            segment_lengths=[10, 10, 10],
            chunk_start=6,
            chunk_end=24,
        )

        chunk_segments = self.prompt_relay.build_chunk_segments(window, epsilon=1e-3)

        self.assertEqual(window["local_ranges"], [(0, 4), (4, 14), (14, 18)])
        self.assertEqual([segment["midpoint"] for segment in chunk_segments], [2, 9, 16])
        self.assertEqual([segment["local_token_idx"] for segment in chunk_segments], [[0, 1], [2, 3, 4], [5, 6]])

    def test_build_chunk_segments_rejects_windows_without_local_ranges(self):
        with self.assertRaisesRegex(ValueError, "one local range per token range"):
            self.prompt_relay.build_chunk_segments({"token_ranges": [(0, 1)]})

    def test_chunk_prompt_window_diagnostics_are_bounded(self):
        window = self.prompt_relay.clip_segments_to_chunk(
            token_ranges=[(idx, idx + 1) for idx in range(14)],
            segment_lengths=[10] * 14,
            chunk_start=0,
            chunk_end=140,
        )

        summary = self.prompt_relay.format_chunk_prompt_window_diagnostics(window, max_segments=4)

        self.assertIn("chunk_prompt_window: chunk=[0:140] segments=14", summary)
        self.assertIn("src0=global[0:10] local[0:10] local_len=10", summary)
        self.assertIn("... 10 segment(s) omitted ...", summary)
        self.assertIn("src13=global[130:140] local[130:140] local_len=10", summary)
        self.assertLess(len(summary), 300)

    def test_chunk_prompt_schedule_clips_global_beats_for_each_chunk_page(self):
        windows = self.prompt_relay.plan_chunk_prompt_windows(
            chunks=[
                {"start": 0, "end": 12},
                {"start": 10, "end": 22},
                {"start": 20, "end": 30},
            ],
            token_ranges=[(0, 2), (2, 4), (4, 6), (6, 8)],
            segment_lengths=[8, 8, 8, 6],
            min_visible_frames=2,
        )

        self.assertEqual([window["chunk_index"] for window in windows], [0, 1, 2])
        self.assertEqual([window["source_indices"] for window in windows], [[0, 1], [1, 2], [2, 3]])
        self.assertEqual([window["local_ranges"] for window in windows], [
            [(0, 8), (8, 12)],
            [(0, 6), (6, 12)],
            [(0, 4), (4, 10)],
        ])

        summary = self.prompt_relay.format_chunk_prompt_schedule_diagnostics(windows, max_windows=2, max_segments=2)
        self.assertIn("chunk_prompt_schedule: windows=3 segments=6 empty_windows=0", summary)
        self.assertIn("chunk0: segments=2", summary)
        self.assertIn("... 1 window(s) omitted ...", summary)
        self.assertIn("chunk2: segments=2", summary)
        self.assertLess(len(summary), 700)

    def test_chunk_prompt_schedule_can_page_without_materializing_every_window(self):
        chunks = [{"start": idx * 10, "end": idx * 10 + 12} for idx in range(1000)]

        windows = self.prompt_relay.plan_chunk_prompt_windows(
            chunks,
            token_ranges=[(0, 1)],
            segment_lengths=[20_000],
            max_windows=3,
        )

        self.assertEqual(len(windows), 3)
        self.assertEqual([window["chunk_start"] for window in windows], [0, 10, 20])

    def test_chunk_plan_evaluation_preflights_budget_and_continuity(self):
        plan = self.prompt_relay.plan_temporal_chunks(
            latent_frames=2501,
            tokens_per_frame=4096,
            text_tokens=128,
            max_mask_elements=self.prompt_relay.DEFAULT_MAX_MASK_ELEMENTS,
            overlap_frames=16,
            safety_margin=1.0,
        )

        evaluation = self.prompt_relay.evaluate_temporal_chunk_plan(
            plan,
            tokens_per_frame=4096,
            text_tokens=128,
            max_mask_elements=self.prompt_relay.DEFAULT_MAX_MASK_ELEMENTS,
            min_context_frames=8,
        )

        self.assertTrue(evaluation["safe"])
        self.assertEqual(evaluation["chunks"], 2)
        self.assertEqual(evaluation["stitched_frames"], 2501)
        self.assertEqual(evaluation["peak_mask_elements"], self.prompt_relay.DEFAULT_MAX_MASK_ELEMENTS)
        self.assertEqual(evaluation["over_budget_chunks"], [])
        self.assertEqual(evaluation["worst_handoff"], "ok")
        self.assertEqual(evaluation["worst_continuity"], "ok")
        summary = self.prompt_relay.format_temporal_chunk_plan_evaluation(evaluation)
        self.assertIn("chunk_eval: safe=True chunks=2 stitched_frames=2501", summary)
        self.assertIn("over_budget=0", summary)
        self.assertIn("error=none", summary)

    def test_chunk_plan_evaluation_flags_over_budget_and_weak_seams(self):
        plan = {"chunks": [
            {"start": 0, "end": 10, "length": 10},
            {"start": 9, "end": 19, "length": 10},
        ]}

        evaluation = self.prompt_relay.evaluate_temporal_chunk_plan(
            plan,
            tokens_per_frame=64,
            text_tokens=64,
            max_mask_elements=20_000,
            min_context_frames=4,
        )

        self.assertFalse(evaluation["safe"])
        self.assertEqual(evaluation["over_budget_chunks"], [0, 1])
        self.assertEqual(evaluation["worst_handoff"], "short_overlap")
        self.assertEqual(evaluation["worst_continuity"], "missing")
        summary = self.prompt_relay.format_temporal_chunk_plan_evaluation(evaluation)
        self.assertIn("safe=False", summary)
        self.assertIn("over_budget=2", summary)
        self.assertIn("worst_handoff=short_overlap", summary)

    def test_chunk_page_evaluation_carries_prior_page_boundary(self):
        first = self.prompt_relay.plan_quality_temporal_chunk_page(
            latent_frames=1200,
            tokens_per_frame=64,
            text_tokens=32,
            max_mask_elements=257 * 64 * 32,
            target_chunk_frames=257,
            overlap_frames=16,
            safety_margin=1.0,
            max_chunks=2,
        )
        second = self.prompt_relay.plan_quality_temporal_chunk_page(
            latent_frames=1200,
            tokens_per_frame=64,
            text_tokens=32,
            max_mask_elements=257 * 64 * 32,
            target_chunk_frames=257,
            overlap_frames=16,
            safety_margin=1.0,
            start_chunk=first["next_chunk"],
            max_chunks=2,
        )

        first_eval = self.prompt_relay.evaluate_temporal_chunk_page(
            first,
            tokens_per_frame=64,
            text_tokens=32,
            max_mask_elements=257 * 64 * 32,
            min_context_frames=8,
        )
        second_eval = self.prompt_relay.evaluate_temporal_chunk_page(
            second,
            tokens_per_frame=64,
            text_tokens=32,
            max_mask_elements=257 * 64 * 32,
            min_context_frames=8,
            previous_tail_chunk=first_eval["last_chunk"],
        )

        self.assertTrue(second_eval["safe"])
        self.assertEqual(second_eval["page_start_chunk"], first["next_chunk"])
        self.assertEqual(second_eval["boundary_handoff_status"], "ok")
        self.assertEqual(second_eval["boundary_continuity_status"], "ok")
        self.assertEqual(second_eval["last_chunk"], second["chunks"][-1])
        summary = self.prompt_relay.format_temporal_chunk_plan_evaluation(second_eval)
        self.assertIn("page_start=2", summary)
        self.assertIn("boundary_handoff=ok", summary)
        self.assertIn("boundary_continuity=ok", summary)

    def test_chunk_page_evaluation_flags_gap_from_prior_page(self):
        page = {"chunks": [{"start": 30, "end": 50, "length": 20}], "start_chunk": 1, "complete": False}

        evaluation = self.prompt_relay.evaluate_temporal_chunk_page(
            page,
            tokens_per_frame=64,
            text_tokens=32,
            max_mask_elements=20 * 64 * 32,
            min_context_frames=4,
            previous_tail_chunk={"start": 0, "end": 20, "length": 20},
        )

        self.assertFalse(evaluation["safe"])
        self.assertIn("contiguous or overlapping", evaluation["boundary_error"])

    def test_quality_chunk_stream_step_returns_resumable_state_and_rolling_anchors(self):
        first = self.prompt_relay.plan_quality_chunk_stream_step(
            latent_frames=1200,
            tokens_per_frame=64,
            text_tokens=32,
            max_mask_elements=257 * 64 * 32,
            target_chunk_frames=257,
            overlap_frames=16,
            safety_margin=1.0,
            max_chunks=2,
            min_context_frames=8,
            anchors_per_chunk=2,
            max_anchors=5,
        )
        second = self.prompt_relay.plan_quality_chunk_stream_step(
            latent_frames=1200,
            tokens_per_frame=64,
            text_tokens=32,
            max_mask_elements=257 * 64 * 32,
            target_chunk_frames=257,
            overlap_frames=16,
            safety_margin=1.0,
            previous_state=first["state"],
            max_chunks=2,
            min_context_frames=8,
            anchors_per_chunk=2,
            max_anchors=5,
        )

        self.assertTrue(first["safe_to_render"])
        self.assertEqual(first["page"]["start_chunk"], 0)
        self.assertEqual(first["state"]["next_chunk"], 2)
        self.assertEqual(first["state"]["last_chunk"], first["page"]["chunks"][-1])
        self.assertEqual(len(first["state"]["anchors"]), 4)
        self.assertTrue(second["safe_to_render"])
        self.assertEqual(second["page"]["start_chunk"], first["state"]["next_chunk"])
        self.assertEqual(second["evaluation"]["boundary_handoff_status"], "ok")
        self.assertEqual(second["evaluation"]["boundary_continuity_status"], "ok")
        self.assertLessEqual(len(second["state"]["anchors"]), 5)
        self.assertGreater(second["anchor_bank"]["dropped_anchors"], 0)
        self.assertGreaterEqual(len(second["crossfade_windows"]), 1)
        self.assertEqual(second["crossfade_windows"][0]["status"], "blendable")

        summary = self.prompt_relay.format_quality_chunk_stream_step_diagnostics(second)
        self.assertIn("chunk_stream: safe_to_render=True start_chunk=2", summary)
        self.assertIn("chunk_eval: safe=True", summary)
        self.assertIn("chunk_plan:", summary)
        self.assertIn("chunk_progress: chunks=4/5 remaining=1 progress=0.800000 planned_until=980 clean_until=980 complete=False", summary)
        self.assertIn("chunk_memory: anchors=5", summary)
        self.assertIn("chunk_crossfade: windows=", summary)
        self.assertIn("chunk_crossfade_eval: safe=True", summary)

    def test_quality_chunk_render_manifest_exposes_scheduler_ready_payload(self):
        step = self.prompt_relay.plan_quality_chunk_stream_step(
            latent_frames=700,
            tokens_per_frame=64,
            text_tokens=32,
            max_mask_elements=257 * 64 * 32,
            target_chunk_frames=257,
            overlap_frames=16,
            safety_margin=1.0,
            max_chunks=2,
            min_context_frames=8,
            anchors_per_chunk=1,
            max_anchors=4,
            token_ranges=[(0, 2), (2, 4), (4, 6)],
            segment_lengths=[200, 200, 300],
            min_visible_frames=4,
        )

        manifest = self.prompt_relay.build_quality_chunk_render_manifest(step)

        self.assertTrue(manifest["renderable"])
        self.assertEqual(manifest["decision"]["status"], "render")
        self.assertEqual(manifest["chunk_indices"], [0, 1])
        self.assertEqual(manifest["input_ranges"], [(0, 257), (241, 498)])
        self.assertEqual(manifest["append_range"], (0, 498))
        self.assertEqual(manifest["output_ranges"][0]["keep_start"], 0)
        self.assertGreater(len(manifest["prompt_windows"]), 0)
        self.assertEqual(manifest["prompt_windows"][0]["chunk_index"], 0)
        self.assertIn("state", manifest)
        self.assertIn("diagnostics", manifest)
        self.assertNotIn("config_fingerprint", manifest["state"])

    def test_quality_chunk_render_manifest_builds_per_chunk_queue_plan(self):
        step = self.prompt_relay.plan_quality_chunk_stream_step(
            latent_frames=700,
            tokens_per_frame=64,
            text_tokens=32,
            max_mask_elements=257 * 64 * 32,
            target_chunk_frames=257,
            overlap_frames=16,
            safety_margin=1.0,
            max_chunks=2,
            min_context_frames=8,
            anchors_per_chunk=1,
            max_anchors=4,
            token_ranges=[(0, 2), (2, 4), (4, 6)],
            segment_lengths=[200, 200, 300],
            min_visible_frames=4,
        )
        manifest = self.prompt_relay.build_quality_chunk_render_manifest(step)

        queue_plan = self.prompt_relay.build_quality_chunk_queue_plan(manifest)

        self.assertEqual(queue_plan["status"], "ready")
        self.assertEqual(queue_plan["checkpoint_state"], manifest["state"])
        self.assertEqual(queue_plan["append_range"], (0, 498))
        self.assertEqual([item["chunk_index"] for item in queue_plan["items"]], [0, 1])
        self.assertEqual(queue_plan["items"][0]["input_range"], (0, 257))
        self.assertEqual(queue_plan["items"][0]["keep_range"], (0, 249))
        self.assertEqual(queue_plan["items"][0]["local_trim"], (0, 8))
        self.assertEqual(queue_plan["items"][1]["input_range"], (241, 498))
        self.assertEqual(queue_plan["items"][1]["keep_range"], (249, 498))
        self.assertEqual(queue_plan["items"][1]["local_trim"], (8, 0))
        self.assertTrue(queue_plan["items"][0]["prompt_windows"])
        self.assertTrue(queue_plan["items"][0]["crossfade_windows"])
        self.assertEqual(queue_plan["items"][1]["conditioning_anchors"], manifest["conditioning_anchors"])

    def test_quality_chunk_queue_plan_rejects_non_renderable_manifest(self):
        step = self.prompt_relay.plan_quality_chunk_stream_step(
            latent_frames=400,
            tokens_per_frame=64,
            text_tokens=32,
            max_mask_elements=257 * 64 * 32,
            target_chunk_frames=257,
            overlap_frames=16,
            safety_margin=1.0,
            max_chunks=2,
            min_context_frames=8,
            token_ranges=[(0, 2)],
            segment_lengths=[0],
            min_visible_frames=4,
        )
        manifest = self.prompt_relay.build_quality_chunk_render_manifest(step)

        queue_plan = self.prompt_relay.build_quality_chunk_queue_plan(manifest)

        self.assertEqual(queue_plan["status"], "skip")
        self.assertEqual(queue_plan["items"], [])
        self.assertEqual(queue_plan["decision"]["reason"], "empty_prompt_window")

    def test_quality_chunk_stream_progress_summarizes_resume_cursor_without_full_plan(self):
        first = self.prompt_relay.plan_quality_chunk_stream_step(
            latent_frames=10_000_000_000,
            tokens_per_frame=64,
            text_tokens=32,
            max_mask_elements=257 * 64 * 32,
            target_chunk_frames=257,
            overlap_frames=16,
            safety_margin=1.0,
            max_chunks=3,
            min_context_frames=8,
            anchors_per_chunk=1,
        )

        progress = first["progress"]

        self.assertEqual(progress["processed_chunks"], 3)
        self.assertGreater(progress["total_chunks"], 40_000_000)
        self.assertEqual(progress["remaining_chunks"], progress["total_chunks"] - 3)
        self.assertEqual(progress["planned_until_frame"], first["page"]["chunks"][-1]["end"])
        self.assertEqual(progress["clean_until_frame"], first["state"]["last_chunk"]["end"])
        self.assertFalse(progress["complete"])
        summary = self.prompt_relay.format_quality_chunk_stream_progress(progress)
        self.assertIn("chunk_progress: chunks=3/", summary)
        self.assertIn("remaining=", summary)

    def test_quality_chunk_stream_step_excludes_rejected_chunks_from_memory_state(self):
        step = self.prompt_relay.plan_quality_chunk_stream_step(
            latent_frames=800,
            tokens_per_frame=64,
            text_tokens=32,
            max_mask_elements=257 * 64 * 32,
            target_chunk_frames=257,
            overlap_frames=16,
            safety_margin=1.0,
            max_chunks=3,
            min_context_frames=8,
            anchors_per_chunk=1,
            max_anchors=8,
            accepted_chunk_indices=[0, 2],
            rejected_chunk_indices=[1],
        )

        self.assertFalse(step["safe_to_render"])
        self.assertEqual(step["anchor_bank"]["skipped_chunks"], 1)
        self.assertEqual(step["rejected_current_indices"], [1])
        self.assertEqual(step["state"]["next_chunk"], 1)
        self.assertEqual(step["state"]["last_chunk_index"], 0)
        self.assertEqual(step["render_decision"]["status"], "retry")
        self.assertEqual(step["render_decision"]["retry_from_chunk"], 1)
        self.assertEqual(step["render_decision"]["checkpoint_next_chunk"], 1)
        self.assertEqual([anchor["chunk_index"] for anchor in step["state"]["anchors"]], [0])
        self.assertNotIn(1, [anchor["chunk_index"] for anchor in step["state"]["anchors"]])
        self.assertNotIn(2, [anchor["chunk_index"] for anchor in step["state"]["anchors"]])

    def test_quality_chunk_stream_step_does_not_carry_rejected_tail_forward(self):
        first = self.prompt_relay.plan_quality_chunk_stream_step(
            latent_frames=1200,
            tokens_per_frame=64,
            text_tokens=32,
            max_mask_elements=257 * 64 * 32,
            target_chunk_frames=257,
            overlap_frames=16,
            safety_margin=1.0,
            max_chunks=3,
            min_context_frames=8,
            anchors_per_chunk=1,
            max_anchors=8,
            accepted_chunk_indices=[0, 1],
            rejected_chunk_indices=[2],
        )

        self.assertEqual(first["state"]["next_chunk"], 2)
        self.assertEqual(first["state"]["last_chunk_index"], 1)
        self.assertEqual(first["state"]["last_chunk"], first["page"]["chunks"][1])
        self.assertEqual(first["render_decision"]["reason"], "rejected_chunk_feedback")

        second = self.prompt_relay.plan_quality_chunk_stream_step(
            latent_frames=1200,
            tokens_per_frame=64,
            text_tokens=32,
            max_mask_elements=257 * 64 * 32,
            target_chunk_frames=257,
            overlap_frames=16,
            safety_margin=1.0,
            previous_state=first["state"],
            max_chunks=2,
            min_context_frames=8,
            anchors_per_chunk=1,
            max_anchors=8,
        )

        self.assertTrue(second["safe_to_render"])
        self.assertIsNone(second["evaluation"]["boundary_error_code"])
        self.assertEqual(second["page"]["start_chunk"], 2)
        self.assertEqual(second["render_decision"]["reason"], "ready")

    def test_quality_chunk_stream_step_attaches_prompt_windows_for_current_page(self):
        step = self.prompt_relay.plan_quality_chunk_stream_step(
            latent_frames=700,
            tokens_per_frame=64,
            text_tokens=32,
            max_mask_elements=257 * 64 * 32,
            target_chunk_frames=257,
            overlap_frames=16,
            safety_margin=1.0,
            max_chunks=2,
            min_context_frames=8,
            anchors_per_chunk=1,
            max_anchors=4,
            token_ranges=[(0, 2), (2, 4), (4, 6)],
            segment_lengths=[200, 200, 300],
            min_visible_frames=4,
        )

        self.assertTrue(step["safe_to_render"])
        self.assertEqual(step["renderable_prompt_windows"], 2)
        self.assertEqual([window["chunk_index"] for window in step["prompt_windows"]], [0, 1])
        self.assertEqual(step["prompt_windows"][0]["source_indices"], [0, 1])
        self.assertEqual(step["prompt_windows"][1]["source_indices"], [1, 2])

        summary = self.prompt_relay.format_quality_chunk_stream_step_diagnostics(step)
        self.assertIn("chunk_prompt_schedule: windows=2", summary)
        self.assertIn("empty_windows=0", summary)

    def test_chunk_prompt_schedule_evaluation_flags_partial_and_empty_windows(self):
        windows = self.prompt_relay.plan_chunk_prompt_windows(
            chunks=[
                {"start": 0, "end": 10},
                {"start": 10, "end": 20},
                {"start": 20, "end": 30},
            ],
            token_ranges=[(0, 1), (1, 2)],
            segment_lengths=[10, 5],
            min_visible_frames=1,
        )

        evaluation = self.prompt_relay.evaluate_chunk_prompt_schedule(windows)

        self.assertFalse(evaluation["safe"])
        self.assertEqual(evaluation["empty_windows"], 1)
        self.assertEqual(evaluation["partial_windows"], 1)
        self.assertEqual(evaluation["worst_status"], "empty")
        self.assertEqual([row["status"] for row in evaluation["windows_detail"]], ["covered", "partial", "empty"])

        summary = self.prompt_relay.format_chunk_prompt_schedule_evaluation(evaluation)
        self.assertIn("chunk_prompt_eval: safe=False windows=3 visible_frames=15/30", summary)
        self.assertIn("empty=1 partial=1 worst=empty", summary)
        self.assertIn("chunk2: coverage=0.000 visible=0/10 status=empty", summary)

    def test_quality_chunk_stream_step_does_not_restart_after_complete_state(self):
        finished = self.prompt_relay.plan_quality_chunk_stream_step(
            latent_frames=300,
            tokens_per_frame=64,
            text_tokens=32,
            max_mask_elements=257 * 64 * 32,
            target_chunk_frames=257,
            overlap_frames=16,
            safety_margin=1.0,
            max_chunks=8,
            min_context_frames=8,
            anchors_per_chunk=1,
            max_anchors=4,
        )
        repeated = self.prompt_relay.plan_quality_chunk_stream_step(
            latent_frames=300,
            tokens_per_frame=64,
            text_tokens=32,
            max_mask_elements=257 * 64 * 32,
            target_chunk_frames=257,
            overlap_frames=16,
            safety_margin=1.0,
            previous_state=finished["state"],
            max_chunks=8,
            min_context_frames=8,
            anchors_per_chunk=1,
            max_anchors=4,
        )

        self.assertTrue(finished["state"]["complete"])
        self.assertEqual(finished["state"]["next_chunk"], finished["page"]["total_chunks"])
        self.assertEqual(finished["render_decision"]["checkpoint_next_chunk"], finished["page"]["total_chunks"])
        self.assertEqual(repeated["page"]["start_chunk"], finished["page"]["total_chunks"])
        self.assertEqual(repeated["page"]["chunks"], [])
        self.assertFalse(repeated["safe_to_render"])
        self.assertTrue(repeated["state"]["complete"])
        self.assertEqual(repeated["render_decision"]["checkpoint_next_chunk"], finished["page"]["total_chunks"])
        summary = self.prompt_relay.format_quality_chunk_stream_step_diagnostics(repeated)
        self.assertIn(f"checkpoint_next={finished['page']['total_chunks']}", summary)

    def test_quality_chunk_stream_step_rejects_cursor_behind_clean_tail(self):
        with self.assertRaisesRegex(ValueError, "resume cursor is behind the verified clean tail"):
            self.prompt_relay.plan_quality_chunk_stream_step(
                latent_frames=1200,
                tokens_per_frame=64,
                text_tokens=32,
                max_mask_elements=257 * 64 * 32,
                target_chunk_frames=257,
                overlap_frames=16,
                safety_margin=1.0,
                previous_state={
                    "next_chunk": 1,
                    "last_chunk_index": 2,
                    "last_chunk": {"start": 482, "end": 739, "length": 257},
                },
            )

    def test_quality_chunk_stream_step_rejects_completed_state_rewind(self):
        with self.assertRaisesRegex(ValueError, "completed chunk stream state cannot resume before total_chunks"):
            self.prompt_relay.plan_quality_chunk_stream_step(
                latent_frames=1200,
                tokens_per_frame=64,
                text_tokens=32,
                max_mask_elements=257 * 64 * 32,
                target_chunk_frames=257,
                overlap_frames=16,
                safety_margin=1.0,
                previous_state={"complete": True, "next_chunk": 3, "total_chunks": 5},
            )

    def test_quality_chunk_stream_step_flags_mixed_undercovered_prompt_page(self):
        step = self.prompt_relay.plan_quality_chunk_stream_step(
            latent_frames=700,
            tokens_per_frame=64,
            text_tokens=32,
            max_mask_elements=257 * 64 * 32,
            target_chunk_frames=257,
            overlap_frames=16,
            safety_margin=1.0,
            max_chunks=2,
            min_context_frames=8,
            anchors_per_chunk=1,
            max_anchors=4,
            token_ranges=[(0, 2)],
            segment_lengths=[300],
            min_visible_frames=4,
        )

        self.assertFalse(step["safe_to_render"])
        self.assertEqual(step["renderable_prompt_windows"], 2)
        self.assertEqual(step["prompt_schedule_evaluation"]["partial_windows"], 1)
        self.assertEqual(step["prompt_schedule_evaluation"]["worst_status"], "partial")

        summary = self.prompt_relay.format_quality_chunk_stream_step_diagnostics(step)
        self.assertIn("chunk_prompt_eval: safe=False", summary)
        self.assertIn("partial=1", summary)

    def test_quality_chunk_stream_step_flags_empty_prompt_page_as_not_renderable(self):
        step = self.prompt_relay.plan_quality_chunk_stream_step(
            latent_frames=700,
            tokens_per_frame=64,
            text_tokens=32,
            max_mask_elements=257 * 64 * 32,
            target_chunk_frames=257,
            overlap_frames=16,
            safety_margin=1.0,
            previous_state={"next_chunk": 2},
            max_chunks=1,
            min_context_frames=8,
            anchors_per_chunk=1,
            max_anchors=4,
            token_ranges=[(0, 2)],
            segment_lengths=[120],
            min_visible_frames=4,
        )

        self.assertFalse(step["safe_to_render"])
        self.assertEqual(step["renderable_prompt_windows"], 0)
        self.assertEqual(step["prompt_windows"][0]["source_indices"], [])

    def test_quality_chunk_stream_step_reports_render_decision_reasons(self):
        ready = self.prompt_relay.plan_quality_chunk_stream_step(
            latent_frames=240,
            tokens_per_frame=64,
            text_tokens=32,
            max_mask_elements=257 * 64 * 32,
            target_chunk_frames=257,
            overlap_frames=16,
            safety_margin=1.0,
            max_chunks=1,
            min_context_frames=8,
            anchors_per_chunk=1,
            max_anchors=4,
            token_ranges=[(0, 2)],
            segment_lengths=[240],
            min_visible_frames=4,
        )
        self.assertEqual(ready["render_decision"]["status"], "render")
        self.assertEqual(ready["render_decision"]["reason"], "ready")
        self.assertTrue(ready["render_decision"]["safe_to_render"])

        empty = self.prompt_relay.plan_quality_chunk_stream_step(
            latent_frames=700,
            tokens_per_frame=64,
            text_tokens=32,
            max_mask_elements=257 * 64 * 32,
            target_chunk_frames=257,
            overlap_frames=16,
            safety_margin=1.0,
            previous_state={"next_chunk": 2},
            max_chunks=1,
            min_context_frames=8,
            anchors_per_chunk=1,
            max_anchors=4,
            token_ranges=[(0, 2)],
            segment_lengths=[120],
            min_visible_frames=4,
        )
        self.assertEqual(empty["render_decision"]["status"], "skip")
        self.assertEqual(empty["render_decision"]["reason"], "empty_prompt_window")
        summary = self.prompt_relay.format_quality_chunk_stream_step_diagnostics(empty)
        self.assertIn("render_status=skip reason=empty_prompt_window", summary)

        complete = self.prompt_relay.plan_quality_chunk_stream_step(
            latent_frames=240,
            tokens_per_frame=64,
            text_tokens=32,
            max_mask_elements=257 * 64 * 32,
            target_chunk_frames=257,
            overlap_frames=16,
            safety_margin=1.0,
            previous_state=ready["state"],
            max_chunks=1,
            min_context_frames=8,
            anchors_per_chunk=1,
            max_anchors=4,
            token_ranges=[(0, 2)],
            segment_lengths=[240],
            min_visible_frames=4,
        )
        self.assertEqual(complete["render_decision"]["status"], "complete")
        self.assertEqual(complete["render_decision"]["reason"], "no_chunks")

    def test_quality_gate_selects_best_passing_candidate_and_rejects_failed_chunks(self):
        report = self.prompt_relay.evaluate_quality_chunk_candidates(
            [
                {
                    "chunk_index": 4,
                    "candidate_id": "4a",
                    "metrics": {
                        "identity_similarity": 0.91,
                        "transition_similarity": 0.72,
                        "flicker_score": 0.18,
                        "sharpness": 0.62,
                    },
                },
                {
                    "chunk_index": 4,
                    "candidate_id": "4b",
                    "metrics": {
                        "identity_similarity": 0.82,
                        "transition_similarity": 0.93,
                        "flicker_score": 0.08,
                        "sharpness": 0.70,
                    },
                },
                {
                    "chunk_index": 5,
                    "candidate_id": "5a",
                    "metrics": {
                        "identity_similarity": 0.60,
                        "transition_similarity": 0.90,
                        "flicker_score": 0.06,
                        "sharpness": 0.80,
                    },
                },
            ],
            thresholds={
                "identity_similarity": {"min": 0.80},
                "transition_similarity": {"min": 0.70},
                "flicker_score": {"max": 0.12},
                "sharpness": {"min": 0.50},
            },
        )

        self.assertEqual(report["decision"], "accept")
        self.assertEqual(report["accepted_candidate"]["candidate_id"], "4b")
        self.assertEqual(report["accepted_chunk_indices"], [4])
        self.assertEqual(report["rejected_chunk_indices"], [5])
        self.assertEqual(report["retry_from_chunk"], 5)
        self.assertEqual(report["candidates"][0]["failed_metrics"], ["flicker_score"])
        self.assertEqual(report["candidates"][2]["failed_metrics"], ["identity_similarity"])
        summary = self.prompt_relay.format_quality_chunk_candidate_diagnostics(report)
        self.assertIn("quality_gate: decision=accept accepted=4b chunk=4", summary)
        self.assertIn("retry_from=5", summary)
        self.assertIn("candidate0 id=4a chunk=4 status=fail failed=flicker_score", summary)

    def test_quality_gate_requests_regeneration_when_no_candidate_passes(self):
        report = self.prompt_relay.evaluate_quality_chunk_candidates(
            [
                {"chunk_index": 2, "candidate_id": "low_identity", "metrics": {"identity_similarity": 0.71, "flicker_score": 0.05}},
                {"chunk_index": 2, "candidate_id": "flickery", "metrics": {"identity_similarity": 0.90, "flicker_score": 0.30}},
            ],
            thresholds={
                "identity_similarity": {"min": 0.80},
                "flicker_score": {"max": 0.12},
            },
        )

        self.assertEqual(report["decision"], "regenerate")
        self.assertIsNone(report["accepted_candidate"])
        self.assertEqual(report["accepted_chunk_indices"], [])
        self.assertEqual(report["rejected_chunk_indices"], [2])
        self.assertEqual(report["retry_from_chunk"], 2)
        summary = self.prompt_relay.format_quality_chunk_candidate_diagnostics(report)
        self.assertIn("quality_gate: decision=regenerate accepted=none", summary)
        self.assertIn("failed=identity_similarity", summary)

    def test_chunk_prompt_seams_flag_prompt_transitions_hidden_in_stitches(self):
        seams = self.prompt_relay.plan_chunk_prompt_seams(
            chunks=[
                {"start": 0, "end": 100},
                {"start": 92, "end": 180},
                {"start": 172, "end": 260},
            ],
            segment_lengths=[96, 80, 84],
            margin_frames=6,
        )

        self.assertEqual([seam["seam_frame"] for seam in seams], [96, 176])
        self.assertEqual([seam["nearest_prompt_boundary"] for seam in seams], [96, 176])
        self.assertEqual([seam["status"] for seam in seams], ["on_boundary", "on_boundary"])

        summary = self.prompt_relay.format_chunk_prompt_seam_diagnostics(seams)
        self.assertIn("chunk_prompt_seams: seams=2 worst=on_boundary", summary)
        self.assertIn("seam0: chunk0->1 frame=96 nearest_boundary=96 distance=0 status=on_boundary", summary)

        evaluation = self.prompt_relay.evaluate_chunk_prompt_seams(seams)
        self.assertFalse(evaluation["safe"])
        self.assertEqual(evaluation["unsafe_seams"], 2)
        self.assertEqual(evaluation["worst_status"], "on_boundary")
        eval_summary = self.prompt_relay.format_chunk_prompt_seam_evaluation(evaluation)
        self.assertIn("chunk_prompt_seam_eval: safe=False seams=2 unsafe=2", eval_summary)
        self.assertIn("on_boundary=2 near_boundary=0 worst=on_boundary", eval_summary)

    def test_chunk_prompt_seam_shift_candidates_move_stitch_inside_overlap(self):
        candidates = self.prompt_relay.plan_chunk_prompt_seam_shift_candidates(
            chunks=[
                {"start": 0, "end": 20},
                {"start": 12, "end": 32},
            ],
            segment_lengths=[16, 16],
            margin_frames=2,
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["seam_frame"], 16)
        self.assertEqual(candidates[0]["status"], "on_boundary")
        self.assertEqual(candidates[0]["recommendation_status"], "shifted_clear")
        self.assertEqual(candidates[0]["recommended_seam_frame"], 13)
        self.assertEqual(candidates[0]["shift_frames"], -3)
        self.assertEqual(candidates[0]["recommended_distance_frames"], 3)

    def test_chunk_prompt_seam_shift_candidates_report_blocked_overlap(self):
        candidates = self.prompt_relay.plan_chunk_prompt_seam_shift_candidates(
            chunks=[
                {"start": 0, "end": 20},
                {"start": 12, "end": 32},
            ],
            segment_lengths=[16, 16],
            margin_frames=16,
        )

        self.assertEqual(candidates[0]["recommendation_status"], "blocked")
        self.assertEqual(candidates[0]["recommended_seam_frame"], candidates[0]["seam_frame"])

    def test_prompt_safe_chunk_stitch_ranges_apply_shifted_prompt_seams(self):
        chunks = [
            {"start": 0, "end": 20},
            {"start": 12, "end": 32},
        ]
        candidates = self.prompt_relay.plan_chunk_prompt_seam_shift_candidates(
            chunks=chunks,
            segment_lengths=[16, 16],
            margin_frames=2,
        )

        plan = self.prompt_relay.plan_prompt_safe_chunk_stitch_ranges(chunks, candidates)

        self.assertTrue(plan["safe"])
        self.assertEqual(plan["applied_shifts"], 1)
        self.assertEqual([(row["keep_start"], row["keep_end"]) for row in plan["ranges"]], [(0, 13), (13, 32)])
        summary = self.prompt_relay.format_prompt_safe_chunk_stitch_diagnostics(plan)
        self.assertIn("prompt_safe_stitch: safe=True chunks=2 applied_shifts=1 blocked_shifts=0", summary)
        self.assertIn("chunk0: keep=[0:13]", summary)

    def test_chunk_crossfade_windows_can_follow_prompt_safe_stitch_seams(self):
        chunks = [
            {"start": 0, "end": 20},
            {"start": 12, "end": 32},
        ]
        candidates = self.prompt_relay.plan_chunk_prompt_seam_shift_candidates(
            chunks=chunks,
            segment_lengths=[16, 16],
            margin_frames=2,
        )
        stitch_plan = self.prompt_relay.plan_prompt_safe_chunk_stitch_ranges(chunks, candidates)

        windows = self.prompt_relay.plan_chunk_crossfade_windows(
            chunks,
            min_context_frames=2,
            stitch_ranges=stitch_plan["ranges"],
        )

        self.assertEqual(windows[0]["seam_frame"], 13)
        self.assertEqual(windows[0]["overlap_start"], 12)
        self.assertEqual(windows[0]["overlap_end"], 20)
        self.assertTrue(self.prompt_relay.evaluate_chunk_crossfade_windows(windows, min_blend_frames=2)["safe"])

    def test_quality_chunk_stream_step_reports_prompt_safe_stitch_plan(self):
        step = self.prompt_relay.plan_quality_chunk_stream_step(
            latent_frames=64,
            tokens_per_frame=64,
            text_tokens=32,
            max_mask_elements=32 * 64 * 32,
            target_chunk_frames=32,
            overlap_frames=8,
            safety_margin=1.0,
            max_chunks=2,
            min_context_frames=2,
            anchors_per_chunk=1,
            max_anchors=4,
            token_ranges=[(0, 1), (1, 2)],
            segment_lengths=[28, 36],
        )

        self.assertEqual(step["prompt_safe_stitch_ranges"]["applied_shifts"], 1)
        self.assertEqual(step["prompt_safe_stitch_ranges"]["ranges"][0]["keep_end"], 25)
        self.assertEqual(step["crossfade_windows"][0]["seam_frame"], 25)
        summary = self.prompt_relay.format_quality_chunk_stream_step_diagnostics(step)
        self.assertIn("prompt_safe_stitch: safe=True", summary)
        self.assertIn("applied_shifts=1", summary)

    def test_quality_chunk_stream_step_reports_prompt_seam_alignment(self):
        step = self.prompt_relay.plan_quality_chunk_stream_step(
            latent_frames=520,
            tokens_per_frame=64,
            text_tokens=32,
            max_mask_elements=257 * 64 * 32,
            target_chunk_frames=257,
            overlap_frames=16,
            safety_margin=1.0,
            max_chunks=3,
            min_context_frames=8,
            anchors_per_chunk=1,
            max_anchors=4,
            token_ranges=[(0, 1), (1, 2), (2, 3)],
            segment_lengths=[249, 243, 28],
        )

        self.assertFalse(step["safe_to_render"])
        self.assertEqual([seam["status"] for seam in step["prompt_seams"]], ["on_boundary", "near_boundary"])
        self.assertEqual(
            [candidate["recommendation_status"] for candidate in step["prompt_seam_shift_candidates"]],
            ["blocked", "shifted_clear"],
        )
        self.assertFalse(step["prompt_seam_evaluation"]["safe"])
        self.assertEqual(step["prompt_seam_evaluation"]["unsafe_seams"], 2)
        summary = self.prompt_relay.format_quality_chunk_stream_step_diagnostics(step)
        self.assertIn("chunk_prompt_seams: seams=2 worst=on_boundary", summary)
        self.assertIn("chunk_prompt_seam_shifts: candidates=2 worst=blocked", summary)
        self.assertIn("status=near_boundary", summary)
        self.assertIn("chunk_prompt_seam_eval: safe=False seams=2 unsafe=2", summary)

    def test_quality_chunk_stream_step_requires_complete_prompt_schedule_inputs(self):
        with self.assertRaisesRegex(ValueError, "must both be provided"):
            self.prompt_relay.plan_quality_chunk_stream_step(
                latent_frames=300,
                tokens_per_frame=64,
                text_tokens=32,
                max_mask_elements=257 * 64 * 32,
                token_ranges=[(0, 1)],
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

    def test_chunk_continuity_windows_expose_discarded_overlap_state(self):
        windows = self.prompt_relay.plan_chunk_continuity_windows([
            {"start": 0, "end": 10},
            {"start": 6, "end": 16},
            {"start": 12, "end": 20},
        ], context_frames=2)

        self.assertEqual(
            [(window["keep_start"], window["keep_end"]) for window in windows],
            [(0, 8), (8, 14), (14, 20)],
        )
        self.assertEqual(
            [(window["incoming_start"], window["incoming_end"], window["incoming_length"]) for window in windows],
            [(0, 0, 0), (6, 8, 2), (12, 14, 2)],
        )
        self.assertEqual(
            [(window["outgoing_start"], window["outgoing_end"], window["outgoing_length"]) for window in windows],
            [(8, 10, 2), (14, 16, 2), (20, 20, 0)],
        )
        self.assertEqual([window["status"] for window in windows], ["ok", "ok", "ok"])

    def test_chunk_continuity_windows_flag_insufficient_carry_context(self):
        windows = self.prompt_relay.plan_chunk_continuity_windows([
            {"start": 0, "end": 10},
            {"start": 8, "end": 18},
        ], context_frames=4)

        self.assertEqual([window["status"] for window in windows], ["partial", "partial"])
        summary = self.prompt_relay.format_chunk_continuity_diagnostics(windows)
        self.assertIn("chunk_continuity: chunks=2 worst=partial", summary)
        self.assertIn("chunk0: keep=[0:9] in=[0:0](0) out=[9:10](1) status=partial", summary)
        self.assertIn("chunk1: keep=[9:18] in=[8:9](1) out=[18:18](0) status=partial", summary)

    def test_chunk_continuity_diagnostics_are_bounded_for_many_chunks(self):
        chunks = []
        start = 0
        for _idx in range(14):
            chunks.append({"start": start, "end": start + 20})
            start += 18
        windows = self.prompt_relay.plan_chunk_continuity_windows(chunks, context_frames=1)

        summary = self.prompt_relay.format_chunk_continuity_diagnostics(windows, max_chunks=4)

        self.assertIn("chunk_continuity: chunks=14 worst=ok", summary)
        self.assertIn("chunk0: keep=[0:19]", summary)
        self.assertIn("... 10 chunk(s) omitted ...", summary)
        self.assertIn("chunk13: keep=[235:254]", summary)
        self.assertLess(len(summary), 800)

    def test_chunk_memory_anchors_keep_only_stitched_clean_frames(self):
        plan = self.prompt_relay.plan_chunk_memory_anchors([
            {"start": 0, "end": 10},
            {"start": 6, "end": 16},
            {"start": 12, "end": 20},
        ], anchors_per_chunk=3, max_anchors=16)

        self.assertEqual(plan["dropped_anchors"], 0)
        self.assertEqual(
            [(anchor["chunk_index"], anchor["frame"], anchor["keep_start"], anchor["keep_end"])
             for anchor in plan["anchors"]],
            [
                (0, 0, 0, 8), (0, 4, 0, 8), (0, 7, 0, 8),
                (1, 8, 8, 14), (1, 10, 8, 14), (1, 13, 8, 14),
                (2, 14, 14, 20), (2, 16, 14, 20), (2, 19, 14, 20),
            ],
        )
        summary = self.prompt_relay.format_chunk_memory_anchor_diagnostics(plan)
        self.assertIn("chunk_memory: anchors=9 dropped=0 cap=16", summary)
        self.assertIn("chunk0@frame0 keep=[0:8]", summary)
        self.assertIn("chunk2@frame19 keep=[14:20]", summary)

    def test_chunk_memory_anchors_cap_to_recent_state_for_infinite_runs(self):
        chunks = []
        start = 0
        for _idx in range(10):
            chunks.append({"start": start, "end": start + 20})
            start += 18

        plan = self.prompt_relay.plan_chunk_memory_anchors(chunks, anchors_per_chunk=2, max_anchors=5)

        self.assertEqual(plan["dropped_anchors"], 15)
        self.assertEqual(len(plan["anchors"]), 5)
        self.assertEqual([anchor["chunk_index"] for anchor in plan["anchors"]], [7, 8, 8, 9, 9])
        summary = self.prompt_relay.format_chunk_memory_anchor_diagnostics(plan, max_anchors=4)
        self.assertIn("chunk_memory: anchors=5 dropped=15 cap=5", summary)
        self.assertIn("... 1 anchor(s) omitted ...", summary)
        self.assertLess(len(summary), 300)

    def test_chunk_memory_anchor_bank_extends_paged_state_with_absolute_indices(self):
        first_page = [
            {"start": 0, "end": 20},
            {"start": 18, "end": 38},
        ]
        second_page = [
            {"start": 36, "end": 56},
            {"start": 54, "end": 74},
        ]

        bank = self.prompt_relay.extend_chunk_memory_anchor_bank(
            [],
            first_page,
            anchors_per_chunk=2,
            max_anchors=5,
            chunk_index_offset=0,
        )
        bank = self.prompt_relay.extend_chunk_memory_anchor_bank(
            bank["anchors"],
            second_page,
            anchors_per_chunk=2,
            max_anchors=5,
            chunk_index_offset=2,
        )

        self.assertEqual(bank["added_anchors"], 4)
        self.assertEqual(bank["dropped_anchors"], 3)
        self.assertEqual(len(bank["anchors"]), 5)
        self.assertEqual([anchor["chunk_index"] for anchor in bank["anchors"]], [2, 1, 2, 3, 3])
        self.assertEqual([anchor["frame"] for anchor in bank["anchors"]], [36, 37, 54, 55, 73])
        summary = self.prompt_relay.format_chunk_memory_anchor_diagnostics(bank, max_anchors=4)
        self.assertIn("chunk_memory: anchors=5 dropped=3 cap=5", summary)
        self.assertIn("chunk3@frame73", summary)

    def test_chunk_memory_anchor_bank_keeps_only_verified_clean_chunks(self):
        page = [
            {"start": 0, "end": 20},
            {"start": 18, "end": 38},
            {"start": 36, "end": 56},
        ]

        bank = self.prompt_relay.extend_chunk_memory_anchor_bank(
            [],
            page,
            anchors_per_chunk=1,
            max_anchors=8,
            chunk_index_offset=4,
            accepted_chunk_indices=[4, 6],
            rejected_chunk_indices=[5],
        )

        self.assertEqual(bank["added_anchors"], 2)
        self.assertEqual(bank["skipped_chunks"], 1)
        self.assertEqual([anchor["chunk_index"] for anchor in bank["anchors"]], [4, 6])
        self.assertEqual(bank["accepted_chunk_indices"], [4, 6])
        self.assertEqual(bank["rejected_chunk_indices"], [5])

    def test_chunk_memory_anchor_bank_rejects_conflicting_acceptance_state(self):
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            self.prompt_relay.extend_chunk_memory_anchor_bank(
                [],
                [{"start": 0, "end": 20}],
                accepted_chunk_indices=[0],
                rejected_chunk_indices=[0],
            )

    def test_chunk_memory_anchor_bank_rejects_malformed_existing_state(self):
        with self.assertRaisesRegex(ValueError, "existing anchors require"):
            self.prompt_relay.extend_chunk_memory_anchor_bank(
                [{"chunk_index": 0, "frame": 1}],
                [{"start": 0, "end": 10}],
            )

    def test_conditioning_anchors_select_recent_prior_clean_frames(self):
        bank = {
            "anchors": [
                {"chunk_index": 0, "frame": 4, "keep_start": 0, "keep_end": 8},
                {"chunk_index": 1, "frame": 12, "keep_start": 8, "keep_end": 14},
                {"chunk_index": 2, "frame": 18, "keep_start": 14, "keep_end": 20},
                {"chunk_index": 3, "frame": 30, "keep_start": 28, "keep_end": 38},
            ]
        }

        selected = self.prompt_relay.select_chunk_conditioning_anchors(
            bank["anchors"],
            {"start": 16, "end": 32},
            max_anchors=2,
            max_age_frames=11,
        )

        self.assertEqual(selected["dropped_anchors"], 1)
        self.assertEqual([anchor["frame"] for anchor in selected["anchors"]], [18, 30])
        summary = self.prompt_relay.format_chunk_conditioning_anchor_diagnostics(selected)
        self.assertIn("chunk_conditioning: anchors=2 dropped=1 window=[16:32]", summary)
        self.assertIn("chunk3@frame30", summary)

    def test_quality_chunk_stream_step_rejects_resume_config_drift(self):
        first = self.prompt_relay.plan_quality_chunk_stream_step(
            latent_frames=700,
            tokens_per_frame=64,
            text_tokens=32,
            max_mask_elements=257 * 64 * 32,
            target_chunk_frames=257,
            overlap_frames=16,
            safety_margin=1.0,
            max_chunks=1,
        )

        self.assertEqual(first["state"]["config_fingerprint"]["target_chunk_frames"], 257)
        with self.assertRaisesRegex(ValueError, "different scheduler inputs"):
            self.prompt_relay.plan_quality_chunk_stream_step(
                latent_frames=700,
                tokens_per_frame=64,
                text_tokens=32,
                max_mask_elements=257 * 64 * 32,
                target_chunk_frames=129,
                overlap_frames=16,
                safety_margin=1.0,
                previous_state=first["state"],
                max_chunks=1,
            )

    def test_quality_chunk_stream_step_tracks_crossfade_curve_in_resume_state(self):
        first = self.prompt_relay.plan_quality_chunk_stream_step(
            latent_frames=700,
            tokens_per_frame=64,
            text_tokens=32,
            max_mask_elements=257 * 64 * 32,
            target_chunk_frames=257,
            overlap_frames=16,
            safety_margin=1.0,
            max_chunks=1,
            crossfade_curve="linear",
        )

        self.assertEqual(first["state"]["config_fingerprint"]["crossfade_curve"], "linear")
        with self.assertRaisesRegex(ValueError, "different scheduler inputs"):
            self.prompt_relay.plan_quality_chunk_stream_step(
                latent_frames=700,
                tokens_per_frame=64,
                text_tokens=32,
                max_mask_elements=257 * 64 * 32,
                target_chunk_frames=257,
                overlap_frames=16,
                safety_margin=1.0,
                previous_state=first["state"],
                max_chunks=1,
                crossfade_curve="cosine",
            )

    def test_quality_chunk_stream_step_rejects_context_and_anchor_drift(self):
        first = self.prompt_relay.plan_quality_chunk_stream_step(
            latent_frames=700,
            tokens_per_frame=64,
            text_tokens=32,
            max_mask_elements=257 * 64 * 32,
            target_chunk_frames=257,
            overlap_frames=16,
            safety_margin=1.0,
            max_chunks=1,
            min_context_frames=8,
            anchors_per_chunk=2,
            max_anchors=8,
        )

        self.assertEqual(first["state"]["config_fingerprint"]["min_context_frames"], 8)
        self.assertEqual(first["state"]["config_fingerprint"]["anchors_per_chunk"], 2)
        with self.assertRaisesRegex(ValueError, "different scheduler inputs"):
            self.prompt_relay.plan_quality_chunk_stream_step(
                latent_frames=700,
                tokens_per_frame=64,
                text_tokens=32,
                max_mask_elements=257 * 64 * 32,
                target_chunk_frames=257,
                overlap_frames=16,
                safety_margin=1.0,
                previous_state=first["state"],
                max_chunks=1,
                min_context_frames=4,
                anchors_per_chunk=2,
                max_anchors=8,
            )

        with self.assertRaisesRegex(ValueError, "different scheduler inputs"):
            self.prompt_relay.plan_quality_chunk_stream_step(
                latent_frames=700,
                tokens_per_frame=64,
                text_tokens=32,
                max_mask_elements=257 * 64 * 32,
                target_chunk_frames=257,
                overlap_frames=16,
                safety_margin=1.0,
                previous_state=first["state"],
                max_chunks=1,
                min_context_frames=8,
                anchors_per_chunk=1,
                max_anchors=8,
            )

    def test_quality_chunk_stream_step_rejects_prompt_schedule_drift(self):
        first = self.prompt_relay.plan_quality_chunk_stream_step(
            latent_frames=700,
            tokens_per_frame=64,
            text_tokens=32,
            max_mask_elements=257 * 64 * 32,
            target_chunk_frames=257,
            overlap_frames=16,
            safety_margin=1.0,
            max_chunks=1,
            min_context_frames=8,
            token_ranges=[(0, 2), (2, 4)],
            segment_lengths=[350, 350],
            min_visible_frames=4,
        )

        prompt_fp = first["state"]["config_fingerprint"]["prompt_schedule"]
        self.assertEqual(prompt_fp["range_count"], 2)
        self.assertEqual(prompt_fp["range_first"], [0, 2])
        self.assertEqual(prompt_fp["length_sum"], 700)
        with self.assertRaisesRegex(ValueError, "different scheduler inputs"):
            self.prompt_relay.plan_quality_chunk_stream_step(
                latent_frames=700,
                tokens_per_frame=64,
                text_tokens=32,
                max_mask_elements=257 * 64 * 32,
                target_chunk_frames=257,
                overlap_frames=16,
                safety_margin=1.0,
                previous_state=first["state"],
                max_chunks=1,
                min_context_frames=8,
                token_ranges=[(0, 2), (2, 4)],
                segment_lengths=[300, 400],
                min_visible_frames=4,
            )

    def test_quality_chunk_stream_step_reports_conditioning_anchors_from_prior_state(self):
        first = self.prompt_relay.plan_quality_chunk_stream_step(
            latent_frames=700,
            tokens_per_frame=64,
            text_tokens=32,
            max_mask_elements=257 * 64 * 32,
            target_chunk_frames=257,
            overlap_frames=16,
            safety_margin=1.0,
            max_chunks=2,
            anchors_per_chunk=2,
            max_anchors=8,
        )
        second = self.prompt_relay.plan_quality_chunk_stream_step(
            latent_frames=700,
            tokens_per_frame=64,
            text_tokens=32,
            max_mask_elements=257 * 64 * 32,
            target_chunk_frames=257,
            overlap_frames=16,
            safety_margin=1.0,
            previous_state=first["state"],
            max_chunks=1,
            anchors_per_chunk=2,
            max_anchors=8,
        )

        self.assertEqual(second["page"]["start_chunk"], 2)
        self.assertGreater(len(second["conditioning_anchors"]["anchors"]), 0)
        self.assertTrue(all(
            anchor["frame"] < second["page"]["chunks"][0]["end"]
            for anchor in second["conditioning_anchors"]["anchors"]
        ))
        summary = self.prompt_relay.format_quality_chunk_stream_step_diagnostics(second)
        self.assertIn("chunk_conditioning: anchors=", summary)

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

    def test_chunk_crossfade_windows_emit_normalized_overlap_weights(self):
        windows = self.prompt_relay.plan_chunk_crossfade_windows([
            {"start": 0, "end": 10},
            {"start": 6, "end": 16},
        ], min_context_frames=4, curve="linear")

        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0]["status"], "blendable")
        self.assertEqual(windows[0]["overlap_start"], 6)
        self.assertEqual(windows[0]["overlap_end"], 10)
        self.assertEqual([row["frame"] for row in windows[0]["frame_weights"]], [6, 7, 8, 9])
        self.assertEqual(
            [(round(row["prev_weight"], 3), round(row["next_weight"], 3)) for row in windows[0]["frame_weights"]],
            [(1.0, 0.0), (0.667, 0.333), (0.333, 0.667), (0.0, 1.0)],
        )
        for row in windows[0]["frame_weights"]:
            self.assertAlmostEqual(row["prev_weight"] + row["next_weight"], 1.0)

        summary = self.prompt_relay.format_chunk_crossfade_diagnostics(windows)
        self.assertIn("chunk_crossfade: windows=1 worst=blendable", summary)
        self.assertIn("xfade0: chunk0->1 overlap=[6:10] len=4 curve=linear status=blendable", summary)

    def test_chunk_crossfade_windows_preserve_weak_seam_statuses(self):
        short = self.prompt_relay.plan_chunk_crossfade_windows([
            {"start": 0, "end": 10},
            {"start": 8, "end": 18},
        ], min_context_frames=4)
        hard = self.prompt_relay.plan_chunk_crossfade_windows([
            {"start": 0, "end": 10},
            {"start": 10, "end": 20},
        ], min_context_frames=4)

        self.assertEqual(short[0]["status"], "short_overlap")
        self.assertEqual(len(short[0]["frame_weights"]), 2)
        self.assertEqual(hard[0]["status"], "hard_cut")
        self.assertEqual(hard[0]["frame_weights"], [])

        with self.assertRaisesRegex(ValueError, "crossfade curve"):
            self.prompt_relay.plan_chunk_crossfade_windows([
                {"start": 0, "end": 10},
                {"start": 8, "end": 18},
            ], curve="quadratic")

    def test_chunk_crossfade_evaluation_rejects_short_or_malformed_blends(self):
        good = self.prompt_relay.plan_chunk_crossfade_windows([
            {"start": 0, "end": 10},
            {"start": 6, "end": 16},
        ], min_context_frames=4)
        bad_weights = [dict(good[0])]
        bad_weights[0]["frame_weights"] = [dict(row) for row in good[0]["frame_weights"]]
        bad_weights[0]["frame_weights"][1]["next_weight"] = 0.9

        good_eval = self.prompt_relay.evaluate_chunk_crossfade_windows(good, min_blend_frames=4)
        short_eval = self.prompt_relay.evaluate_chunk_crossfade_windows(good, min_blend_frames=5)
        bad_eval = self.prompt_relay.evaluate_chunk_crossfade_windows(bad_weights, min_blend_frames=4)

        self.assertTrue(good_eval["safe"])
        self.assertFalse(short_eval["safe"])
        self.assertEqual(short_eval["worst_status"], "short_blend")
        self.assertFalse(bad_eval["safe"])
        self.assertEqual(bad_eval["worst_status"], "bad_weights")

        summary = self.prompt_relay.format_chunk_crossfade_evaluation(bad_eval)
        self.assertIn("chunk_crossfade_eval: safe=False windows=1", summary)
        self.assertIn("worst=bad_weights", summary)

    def test_chunk_page_output_ranges_exclude_prior_tail_context(self):
        previous_tail = {"start": 0, "end": 100}
        chunks = [
            {"start": 84, "end": 184},
            {"start": 168, "end": 268},
        ]

        plan = self.prompt_relay.plan_chunk_page_output_ranges(chunks, previous_tail=previous_tail)

        self.assertTrue(plan["safe"])
        self.assertEqual(plan["append_start"], 92)
        self.assertEqual(plan["append_end"], 268)
        self.assertEqual(plan["emitted_frames"], 176)
        self.assertEqual(
            [(row["keep_start"], row["keep_end"], row["trim_start"], row["trim_end"]) for row in plan["ranges"]],
            [(92, 176, 8, 8), (176, 268, 8, 0)],
        )

        summary = self.prompt_relay.format_chunk_page_output_diagnostics(plan)
        self.assertIn("chunk_page_output: safe=True chunks=2 emitted=176 append=[92:268]", summary)

    def test_quality_chunk_stream_step_reports_page_output_ranges(self):
        first = self.prompt_relay.plan_quality_chunk_stream_step(
            latent_frames=700,
            tokens_per_frame=64,
            text_tokens=32,
            max_mask_elements=257 * 64 * 32,
            target_chunk_frames=257,
            overlap_frames=16,
            safety_margin=1.0,
            max_chunks=1,
            min_context_frames=8,
        )
        second = self.prompt_relay.plan_quality_chunk_stream_step(
            latent_frames=700,
            tokens_per_frame=64,
            text_tokens=32,
            max_mask_elements=257 * 64 * 32,
            target_chunk_frames=257,
            overlap_frames=16,
            safety_margin=1.0,
            previous_state=first["state"],
            max_chunks=1,
            min_context_frames=8,
        )

        output = second["page_output_ranges"]
        self.assertTrue(output["safe"])
        self.assertEqual(output["append_start"], first["state"]["last_chunk"]["end"] - 8)
        self.assertGreater(output["emitted_frames"], 0)
        self.assertIn("chunk_page_output:", self.prompt_relay.format_quality_chunk_stream_step_diagnostics(second))


if __name__ == "__main__":
    unittest.main()
