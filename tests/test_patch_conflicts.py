import importlib.util
import os
import sys
import types
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATCHES_PATH = os.path.join(ROOT, "patches.py")


class FakeModelClone:
    def __init__(self, object_patches=None, diffusion_model=None):
        self.object_patches = object_patches or {}
        self.diffusion_model = diffusion_model
        self.added_patches = []

    def get_model_object(self, name):
        if name != "diffusion_model":
            raise KeyError(name)
        return self.diffusion_model

    def add_object_patch(self, key, value):
        self.added_patches.append((key, value))
        self.object_patches[key] = value


class FakeAttention:
    pass


class FakeBlock:
    def __init__(self, **attrs):
        for key, value in attrs.items():
            setattr(self, key, value)


def install_fake_comfy_modules():
    modules = {
        "torch": types.SimpleNamespace(),
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


def load_patches_module():
    previous = install_fake_comfy_modules()
    try:
        spec = importlib.util.spec_from_file_location("patches_under_test", PATCHES_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module, previous
    except Exception:
        restore_modules(previous)
        raise


class PromptRelayPatchConflictTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.patches, cls.previous_modules = load_patches_module()

    @classmethod
    def tearDownClass(cls):
        restore_modules(cls.previous_modules)
        sys.modules.pop("patches_under_test", None)

    def test_exact_forward_patch_conflict_reports_existing_patch(self):
        key = "diffusion_model.transformer_blocks.0.attn2.forward"
        model = FakeModelClone({key: "SageAttentionPreviewPatch"})

        with self.assertRaisesRegex(RuntimeError, "SageAttentionPreviewPatch"):
            self.patches._check_unpatched(model, key)

    def test_parent_attention_patch_conflict_is_detected(self):
        key = "diffusion_model.transformer_blocks.0.audio_attn2.forward"
        model = FakeModelClone({
            "diffusion_model.transformer_blocks.0.audio_attn2": "PreviewPatch",
        })

        with self.assertRaisesRegex(RuntimeError, "audio_attn2.*PreviewPatch"):
            self.patches._check_unpatched(model, key)

    def test_child_patch_under_attention_module_conflicts_with_forward_patch(self):
        key = "diffusion_model.transformer_blocks.0.attn2.forward"
        model = FakeModelClone({
            "diffusion_model.transformer_blocks.0.attn2.processor.preview": "preview",
        })

        with self.assertRaisesRegex(RuntimeError, "processor.preview"):
            self.patches._check_unpatched(model, key)

    def test_unrelated_object_patch_does_not_block_prompt_relay(self):
        key = "diffusion_model.transformer_blocks.0.attn2.forward"
        model = FakeModelClone({
            "diffusion_model.transformer_blocks.1.attn2.forward": "other block",
            "diffusion_model.transformer_blocks.0.attn1.forward": "self attention",
        })

        self.patches._check_unpatched(model, key)

    def test_ltx_model_without_known_cross_attention_fails_instead_of_silently_unrelayed(self):
        diffusion_model = types.SimpleNamespace(transformer_blocks=[FakeBlock(attn1=FakeAttention())])
        model = FakeModelClone(diffusion_model=diffusion_model)

        with self.assertRaisesRegex(RuntimeError, "silently produce unrelayed output"):
            self.patches.apply_patches(model, "ltx", mask_fn=lambda *_args, **_kwargs: None)

        self.assertEqual(model.added_patches, [])

    def test_ltx_model_with_attn2_installs_patch_and_returns_keys(self):
        diffusion_model = types.SimpleNamespace(transformer_blocks=[FakeBlock(attn2=FakeAttention())])
        model = FakeModelClone(diffusion_model=diffusion_model)

        keys = self.patches.apply_patches(model, "ltx", mask_fn=lambda *_args, **_kwargs: None)

        self.assertEqual(keys, ["diffusion_model.transformer_blocks.0.attn2.forward"])
        self.assertEqual(len(model.added_patches), 1)


if __name__ == "__main__":
    unittest.main()
