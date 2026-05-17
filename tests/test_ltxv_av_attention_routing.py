import importlib.util
import os
import sys
import types
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPT_RELAY_PATH = os.path.join(ROOT, "prompt_relay.py")


class FakeTorch(types.SimpleNamespace):
    long = "long"
    float32 = "float32"

    @staticmethod
    def arange(start, end=None, **_kwargs):
        if end is None:
            start, end = 0, start
        return list(range(start, end))


class ScalarLike:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class TensorLike:
    def __init__(self, rows):
        self.rows = rows

    def tolist(self):
        return self.rows


class IndexLike:
    def __init__(self, max_value):
        self.max_value = max_value

    def max(self):
        return ScalarLike(self.max_value)


class FakeQuery:
    device = "cpu"
    dtype = "float32"

    def __init__(self, length):
        self.shape = (1, length, 8)


class FakeMask:
    def __init__(self, name):
        self.name = name

    def __gt__(self, _other):
        return self

    def sum(self):
        return ScalarLike(1)

    def numel(self):
        return 1

    def __neg__(self):
        return self

    def to(self, _dtype):
        return self


class LTXVAVAttentionRoutingTests(unittest.TestCase):
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

    def test_ltxv_empty_segment_lengths_distribute_217_frames_across_two_prompts(self):
        lengths = self.prompt_relay.distribute_segment_lengths(
            num_segments=2,
            latent_frames=217,
            specified_lengths=None,
        )

        self.assertEqual(lengths, [109, 108])

    def test_ltxv_grid_tokens_per_frame_accepts_batched_tensor_like_metadata(self):
        # Sarp's workflow uses EmptyLTXVLatentVideo length 217. At 576x1024,
        # common LTX latent spatial grids are 18x32 after VAE compression. The
        # helper must read H*W from one batched [T,H,W] row, not grid_sizes[1].
        grid_sizes = TensorLike([[217, ScalarLike(32), ScalarLike(18)]])

        self.assertEqual(
            self.prompt_relay._grid_tokens_per_frame(grid_sizes, fallback_tokens_per_frame=999),
            576,
        )

    def test_ltxv_av_routes_video_text_attention_and_scaled_audio_text_attention(self):
        latent_frames = 217
        video_tpf = 32 * 18
        video_lq = latent_frames * video_tpf
        max_token_idx = 40
        text_lk = 256

        classify = self.prompt_relay.classify_attention_route
        self.assertEqual(classify(video_lq, text_lk, max_token_idx, latent_frames, video_tpf), "video")
        self.assertEqual(classify(4096, text_lk, max_token_idx, latent_frames, video_tpf), "scaled")
        self.assertIsNone(classify(video_lq, video_lq, max_token_idx, latent_frames, video_tpf))
        self.assertIsNone(classify(4096, max_token_idx - 1, max_token_idx, latent_frames, video_tpf))
        self.assertIsNone(classify(4096, video_lq, max_token_idx, latent_frames, video_tpf))

    def test_ltxv_grid_tokens_per_frame_falls_back_on_missing_metadata(self):
        self.assertEqual(self.prompt_relay._grid_tokens_per_frame(None, 123), 123)
        self.assertEqual(self.prompt_relay._grid_tokens_per_frame([217, 32], 123), 123)

    def test_ltxv_grid_tokens_per_frame_falls_back_on_invalid_metadata(self):
        self.assertEqual(self.prompt_relay._grid_tokens_per_frame([217, 0, 18], 123), 123)
        self.assertEqual(self.prompt_relay._grid_tokens_per_frame([217, -32, 18], 123), 123)
        self.assertEqual(self.prompt_relay._grid_tokens_per_frame([217, "bad", 18], 123), 123)

    def test_create_mask_fn_rejects_empty_segment_metadata_with_actionable_error(self):
        with self.assertRaisesRegex(ValueError, "no positive-length relay segments"):
            self.prompt_relay.create_mask_fn([], fallback_tokens_per_frame=576, latent_frames=0)

    def test_mask_function_exposes_video_and_scaled_attention_diagnostics(self):
        original_video = self.prompt_relay.build_temporal_cost
        original_scaled = self.prompt_relay.build_temporal_cost_scaled
        try:
            self.prompt_relay.build_temporal_cost = lambda *_args, **_kwargs: FakeMask("video")
            self.prompt_relay.build_temporal_cost_scaled = lambda *_args, **_kwargs: FakeMask("scaled")

            mask_fn = self.prompt_relay.create_mask_fn(
                [{"local_token_idx": IndexLike(39)}],
                fallback_tokens_per_frame=576,
                latent_frames=217,
            )

            video = mask_fn(FakeQuery(217 * 576), FakeQuery(256), {"cond_or_uncond": [0]})
            scaled = mask_fn(FakeQuery(4096), FakeQuery(256), {"cond_or_uncond": [0]})
            uncond = mask_fn(FakeQuery(4096), FakeQuery(256), {"cond_or_uncond": [1]})

            self.assertEqual(video.name, "video")
            self.assertEqual(scaled.name, "scaled")
            self.assertIsNone(uncond)
            self.assertEqual(mask_fn.prompt_relay_diagnostics["applied"], {"video": 1, "scaled": 1})
            self.assertEqual(mask_fn.prompt_relay_diagnostics["skipped"]["uncond"], 1)
            self.assertEqual(mask_fn.prompt_relay_diagnostics["calls"], 3)
        finally:
            self.prompt_relay.build_temporal_cost = original_video
            self.prompt_relay.build_temporal_cost_scaled = original_scaled

    def test_scaled_attention_cache_key_includes_query_length_for_long_video_routes(self):
        original_scaled = self.prompt_relay.build_temporal_cost_scaled
        calls = []
        try:
            def fake_scaled(_segments, Lq, _Lk, *_args, **_kwargs):
                calls.append(Lq)
                return FakeMask(f"scaled-{Lq}")

            self.prompt_relay.build_temporal_cost_scaled = fake_scaled
            mask_fn = self.prompt_relay.create_mask_fn(
                [{"local_token_idx": IndexLike(39)}],
                fallback_tokens_per_frame=576,
                latent_frames=2501,
            )

            first = mask_fn(FakeQuery(4096), FakeQuery(256), {"cond_or_uncond": [0]})
            second = mask_fn(FakeQuery(2048), FakeQuery(256), {"cond_or_uncond": [0]})

            self.assertEqual(first.name, "scaled-4096")
            self.assertEqual(second.name, "scaled-2048")
            self.assertEqual(calls, [4096, 2048])
        finally:
            self.prompt_relay.build_temporal_cost_scaled = original_scaled

    def test_mask_size_guard_fails_before_allocating_impossible_long_form_masks(self):
        old_cap = os.environ.get("PROMPT_RELAY_MAX_MASK_ELEMENTS")
        try:
            os.environ["PROMPT_RELAY_MAX_MASK_ELEMENTS"] = "1000"
            with self.assertRaisesRegex(RuntimeError, "2500\+ frame generation"):
                self.prompt_relay._check_mask_size(2501 * 576, 256, "float16", "video")

            os.environ["PROMPT_RELAY_MAX_MASK_ELEMENTS"] = "0"
            self.assertEqual(
                self.prompt_relay._check_mask_size(2501 * 576, 256, "float16", "video"),
                2501 * 576 * 256,
            )
        finally:
            if old_cap is None:
                os.environ.pop("PROMPT_RELAY_MAX_MASK_ELEMENTS", None)
            else:
                os.environ["PROMPT_RELAY_MAX_MASK_ELEMENTS"] = old_cap


if __name__ == "__main__":
    unittest.main()
