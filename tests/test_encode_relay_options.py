import importlib.util
import os
import sys
import types
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODES_PATH = os.path.join(ROOT, "nodes.py")


class DummyNodeOutput:
    def __init__(self, *values):
        self.values = values


class DummyCustom:
    def __init__(self, _name):
        pass

    @staticmethod
    def Input(*_args, **_kwargs):
        return None

    @staticmethod
    def Output(*_args, **_kwargs):
        return None


class DummyIO(types.SimpleNamespace):
    ComfyNode = object
    NodeOutput = DummyNodeOutput
    Custom = DummyCustom


def install_fake_comfy_modules():
    fake_io = DummyIO()
    fake_latest = types.SimpleNamespace(io=fake_io)
    modules = {
        "torch": types.SimpleNamespace(),
        "comfy_api": types.SimpleNamespace(latest=fake_latest),
        "comfy_api.latest": fake_latest,
        "comfy": types.ModuleType("comfy"),
        "comfy.ldm": types.ModuleType("comfy.ldm"),
        "comfy.ldm.modules": types.ModuleType("comfy.ldm.modules"),
        "comfy.ldm.modules.attention": types.ModuleType("comfy.ldm.modules.attention"),
    }
    modules["comfy"].ldm = modules["comfy.ldm"]
    modules["comfy.ldm"].modules = modules["comfy.ldm.modules"]
    modules["comfy.ldm.modules"].attention = modules["comfy.ldm.modules.attention"]

    previous = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    return previous


def restore_modules(previous):
    for name, module in previous.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def load_nodes_module():
    previous = install_fake_comfy_modules()
    package_name = "prompt_relay_quality_lab_under_test"
    package = types.ModuleType(package_name)
    package.__path__ = [ROOT]
    old_package = sys.modules.get(package_name)
    sys.modules[package_name] = package
    try:
        spec = importlib.util.spec_from_file_location(f"{package_name}.nodes", NODES_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module, previous, package_name, old_package
    except Exception:
        restore_modules(previous)
        if old_package is None:
            sys.modules.pop(package_name, None)
        else:
            sys.modules[package_name] = old_package
        raise


class FakeRawTokenizer:
    def __call__(self, text):
        return {"input_ids": text.split()}


class FakeClip:
    def __init__(self):
        self.tokenizer = types.SimpleNamespace(inner=types.SimpleNamespace(tokenizer=FakeRawTokenizer()))

    def tokenize(self, prompt):
        return {"prompt": prompt}

    def encode_from_tokens_scheduled(self, tokens):
        return {"conditioning": tokens}


class FakeSamples:
    shape = (1, 16, 217, 32, 18)


class FakeImageSamples:
    shape = (1, 16, 32, 18)


class FakeTinySamples:
    shape = (1, 16, 217, 1, 1)


class FakeDiffusionModel:
    patchifier = object()
    vae_scale_factors = [1]


class FakeModel:
    def __init__(self):
        self.model = types.SimpleNamespace(diffusion_model=FakeDiffusionModel())
        self.patched = False

    def clone(self):
        clone = FakeModel()
        clone.get_model_object = lambda _name: types.SimpleNamespace(transformer_blocks=[])
        clone.add_object_patch = lambda *_args: None
        return clone


class FakeWanDiffusionModel:
    patch_size = (1, 2, 2)


class FakeWanModel(FakeModel):
    def __init__(self):
        self.model = types.SimpleNamespace(diffusion_model=FakeWanDiffusionModel())
        self.patched = False


class EncodeRelayOptionsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nodes, cls.previous_modules, cls.package_name, cls.old_package = load_nodes_module()

    @classmethod
    def tearDownClass(cls):
        restore_modules(cls.previous_modules)
        if cls.old_package is None:
            sys.modules.pop(cls.package_name, None)
        else:
            sys.modules[cls.package_name] = cls.old_package

    def test_sanitize_relay_options_handles_malformed_sigma_mode_from_stale_workflow(self):
        sanitized = self.nodes._sanitize_relay_options({"sigma_mode": ["paper_boundary"]})

        self.assertEqual(sanitized["sigma_mode"], "upstream_compat")

    def test_encode_relay_sanitizes_and_propagates_advanced_options(self):
        captured = {}
        original_build_segments = self.nodes.build_segments
        original_create_mask_fn = self.nodes.create_mask_fn
        original_apply_patches = self.nodes.apply_patches
        try:
            def fake_build_segments(token_ranges, effective_lengths, epsilon, relay_options):
                captured["token_ranges"] = token_ranges
                captured["effective_lengths"] = effective_lengths
                captured["epsilon"] = epsilon
                captured["relay_options"] = relay_options
                return [{"local_token_idx": [1], "midpoint": 109, "window": 107, "sigma": 0.1}]

            self.nodes.build_segments = fake_build_segments
            self.nodes.create_mask_fn = lambda *_args: "mask_fn"
            self.nodes.apply_patches = lambda patched, arch, mask_fn: captured.update(
                arch=arch,
                mask_fn=mask_fn,
            )

            patched, conditioning = self.nodes._encode_relay(
                FakeModel(),
                FakeClip(),
                {"samples": FakeSamples()},
                "global scene",
                "first segment|second segment",
                "",
                0.001,
                {
                    "video_strength": "0.75",
                    "video_window_scale": 1,
                    "audio_epsilon": "0.2",
                    "audio_strength": 0.5,
                    "audio_window_scale": 1.25,
                    "audio_frame_offset_frames": "-1.5",
                    "sigma_mode": "paper_boundary",
                    "ignored_future_key": object(),
                },
            )
        finally:
            self.nodes.build_segments = original_build_segments
            self.nodes.create_mask_fn = original_create_mask_fn
            self.nodes.apply_patches = original_apply_patches

        self.assertIsNotNone(patched)
        self.assertIn("conditioning", conditioning)
        self.assertEqual(captured["effective_lengths"], [109, 108])
        self.assertEqual(captured["relay_options"], {
            "video_strength": 0.75,
            "video_window_scale": 1.0,
            "audio_epsilon": 0.2,
            "audio_strength": 0.5,
            "audio_window_scale": 1.25,
            "audio_frame_offset_frames": -1.5,
            "sigma_mode": "paper_boundary",
            "long_form_mode": "off",
            "quality_target_chunk_frames": 257,
            "chunk_overlap_frames": 16,
            "chunk_safety_margin": 0.9,
        })
        self.assertEqual(captured["arch"], "ltx")
        self.assertEqual(captured["mask_fn"], "mask_fn")

    def test_ltxv_217_frame_ratio_timing_and_paper_audio_options_reach_segments(self):
        captured = {}
        original_build_segments = self.nodes.build_segments
        original_create_mask_fn = self.nodes.create_mask_fn
        original_apply_patches = self.nodes.apply_patches
        try:
            def fake_build_segments(token_ranges, effective_lengths, epsilon, relay_options):
                captured["token_ranges"] = token_ranges
                captured["effective_lengths"] = effective_lengths
                captured["epsilon"] = epsilon
                captured["relay_options"] = relay_options
                return [{"local_token_idx": [2, 3], "midpoint": 38, "window": 36, "sigma": 0.1}]

            self.nodes.build_segments = fake_build_segments
            self.nodes.create_mask_fn = lambda *_args: "mask_fn"
            self.nodes.apply_patches = lambda patched, arch, mask_fn: captured.update(arch=arch, mask_fn=mask_fn)

            self.nodes._encode_relay(
                FakeModel(),
                FakeClip(),
                {"samples": FakeSamples()},
                "global scene",
                "look detail|love straps",
                "0.35,0.65",
                0.001,
                {
                    "sigma_mode": "paper_boundary",
                    "audio_epsilon": 0.1,
                    "audio_strength": 0.7,
                    "audio_window_scale": 0.5,
                    "audio_frame_offset_frames": -1.25,
                },
            )
        finally:
            self.nodes.build_segments = original_build_segments
            self.nodes.create_mask_fn = original_create_mask_fn
            self.nodes.apply_patches = original_apply_patches

        self.assertEqual(captured["effective_lengths"], [76, 141])
        self.assertEqual(captured["epsilon"], 0.001)
        self.assertEqual(captured["relay_options"], {
            "video_strength": 1.0,
            "video_window_scale": 1.0,
            "audio_epsilon": 0.1,
            "audio_strength": 0.7,
            "audio_window_scale": 0.5,
            "audio_frame_offset_frames": -1.25,
            "sigma_mode": "paper_boundary",
            "long_form_mode": "off",
            "quality_target_chunk_frames": 257,
            "chunk_overlap_frames": 16,
            "chunk_safety_margin": 0.9,
        })
        self.assertEqual(captured["arch"], "ltx")

    def test_sanitize_relay_options_rejects_non_dict_custom_values(self):
        with self.assertRaisesRegex(ValueError, "relay_options"):
            self.nodes._sanitize_relay_options([("audio_strength", 0.5)])

    def test_encode_relay_rejects_image_latent_with_actionable_error(self):
        with self.assertRaisesRegex(ValueError, "video latent.*image latent"):
            self.nodes._encode_relay(
                FakeModel(),
                FakeClip(),
                {"samples": FakeImageSamples()},
                "global scene",
                "first segment|second segment",
                "",
                0.001,
            )

    def test_encode_relay_rejects_tiny_latent_before_zero_token_mask(self):
        with self.assertRaisesRegex(ValueError, "too small.*patch size"):
            self.nodes._encode_relay(
                FakeWanModel(),
                FakeClip(),
                {"samples": FakeTinySamples()},
                "global scene",
                "first segment|second segment",
                "",
                0.001,
            )

    def test_sanitize_epsilon_clamps_api_loaded_values(self):
        self.assertEqual(self.nodes._sanitize_epsilon("nan"), 0.001)
        self.assertEqual(self.nodes._sanitize_epsilon("inf"), 0.001)
        self.assertEqual(self.nodes._sanitize_epsilon(0.0), 0.001)
        self.assertEqual(self.nodes._sanitize_epsilon(1.0), 0.001)
        self.assertEqual(self.nodes._sanitize_epsilon("0.25"), 0.25)

    def test_encode_relay_sanitizes_api_loaded_epsilon_before_building_segments(self):
        captured = {}
        original_build_segments = self.nodes.build_segments
        original_create_mask_fn = self.nodes.create_mask_fn
        original_apply_patches = self.nodes.apply_patches
        try:
            def fake_build_segments(_token_ranges, _effective_lengths, epsilon, _relay_options):
                captured["epsilon"] = epsilon
                return [{"local_token_idx": [1], "midpoint": 109, "window": 107, "sigma": 0.1}]

            self.nodes.build_segments = fake_build_segments
            self.nodes.create_mask_fn = lambda *_args: "mask_fn"
            self.nodes.apply_patches = lambda *_args: None

            self.nodes._encode_relay(
                FakeModel(),
                FakeClip(),
                {"samples": FakeSamples()},
                "global scene",
                "first segment|second segment",
                "",
                "not-a-number",
            )
        finally:
            self.nodes.build_segments = original_build_segments
            self.nodes.create_mask_fn = original_create_mask_fn
            self.nodes.apply_patches = original_apply_patches

        self.assertEqual(captured["epsilon"], 0.001)

    def test_encode_relay_logs_quality_chunk_plan_when_enabled(self):
        captured = {}
        original_build_segments = self.nodes.build_segments
        original_create_mask_fn = self.nodes.create_mask_fn
        original_apply_patches = self.nodes.apply_patches
        original_plan = self.nodes.plan_quality_temporal_chunks
        original_eval = self.nodes.evaluate_temporal_chunk_plan
        try:
            self.nodes.build_segments = lambda *_args: [{"local_token_idx": [1], "midpoint": 109, "window": 107, "sigma": 0.1}]
            self.nodes.create_mask_fn = lambda *_args: "mask_fn"
            self.nodes.apply_patches = lambda *_args: None

            def fake_plan(**kwargs):
                captured.update(kwargs)
                return {"max_chunk_frames": 129, "overlap_frames": 12, "chunks": [{"start": 0, "end": 129, "length": 129}]}

            self.nodes.plan_quality_temporal_chunks = fake_plan
            self.nodes.evaluate_temporal_chunk_plan = lambda *_args, **_kwargs: {"safe": True, "chunks": 1}

            self.nodes._encode_relay(
                FakeModel(),
                FakeClip(),
                {"samples": FakeSamples()},
                "global scene",
                "first segment|second segment",
                "",
                0.001,
                {
                    "long_form_mode": "quality_plan",
                    "quality_target_chunk_frames": 129,
                    "chunk_overlap_frames": 12,
                    "chunk_safety_margin": 0.8,
                },
            )
        finally:
            self.nodes.build_segments = original_build_segments
            self.nodes.create_mask_fn = original_create_mask_fn
            self.nodes.apply_patches = original_apply_patches
            self.nodes.plan_quality_temporal_chunks = original_plan
            self.nodes.evaluate_temporal_chunk_plan = original_eval

        self.assertEqual(captured["latent_frames"], 217)
        self.assertEqual(captured["tokens_per_frame"], 576)
        self.assertEqual(captured["target_chunk_frames"], 129)
        self.assertEqual(captured["overlap_frames"], 12)
        self.assertEqual(captured["safety_margin"], 0.8)
        self.assertEqual(captured["text_tokens"], 7)

    def test_long_form_plan_budgets_one_token_of_special_token_slack(self):
        captured = {}
        original_plan = self.nodes.plan_quality_temporal_chunks
        original_eval = self.nodes.evaluate_temporal_chunk_plan
        try:
            self.nodes.plan_quality_temporal_chunks = lambda **kwargs: captured.update(kwargs) or {
                "max_chunk_frames": 257,
                "overlap_frames": 16,
                "chunks": [{"start": 0, "end": 257, "length": 257}],
            }
            self.nodes.evaluate_temporal_chunk_plan = lambda *_args, **_kwargs: {"safe": True, "chunks": 1}

            self.nodes._log_long_form_plan(
                {"long_form_mode": "quality_plan"},
                latent_frames=2501,
                tokens_per_frame=4096,
                token_ranges=[(2, 5), (5, 9)],
            )
        finally:
            self.nodes.plan_quality_temporal_chunks = original_plan
            self.nodes.evaluate_temporal_chunk_plan = original_eval

        self.assertEqual(captured["text_tokens"], 10)

    def test_sanitize_relay_options_clamps_api_loaded_values(self):
        sanitized = self.nodes._sanitize_relay_options({
            "video_strength": -5,
            "video_window_scale": 99,
            "audio_epsilon": "nan",
            "audio_strength": "inf",
            "audio_window_scale": "bad",
            "audio_frame_offset_frames": -999,
            "sigma_mode": "unknown_future_mode",
        })

        self.assertEqual(sanitized, {
            "video_strength": 0.0,
            "video_window_scale": 4.0,
            "audio_epsilon": None,
            "audio_strength": 1.0,
            "audio_window_scale": 1.0,
            "audio_frame_offset_frames": -32.0,
            "sigma_mode": "upstream_compat",
            "long_form_mode": "off",
            "quality_target_chunk_frames": 257,
            "chunk_overlap_frames": 16,
            "chunk_safety_margin": 0.9,
        })

    def test_advanced_options_node_clamps_direct_api_values(self):
        output = self.nodes.PromptRelayAdvancedOptions.execute(
            video_strength=-4,
            video_window_scale="inf",
            audio_epsilon=2.0,
            audio_strength="nan",
            audio_window_scale="bad",
            sigma_mode="paper_boundary",
            audio_frame_offset_frames=-999,
        )

        self.assertEqual(output.values[0], {
            "video_strength": 0.0,
            "video_window_scale": 1.0,
            "audio_epsilon": None,
            "audio_strength": 1.0,
            "audio_window_scale": 1.0,
            "sigma_mode": "paper_boundary",
            "audio_frame_offset_frames": -32.0,
            "long_form_mode": "off",
            "quality_target_chunk_frames": 257,
            "chunk_overlap_frames": 16,
            "chunk_safety_margin": 0.9,
        })


if __name__ == "__main__":
    unittest.main()
