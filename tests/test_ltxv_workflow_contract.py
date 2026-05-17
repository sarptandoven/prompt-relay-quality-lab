import ast
import json
import os
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPT_RELAY_PATH = os.path.join(ROOT, "prompt_relay.py")
WORKFLOW_PATH = "/Users/sarpdoven/.openclaw/media/inbound/112344e7-4947-451d-81f3-62bf51dac3a5.json"
DERIVED_WORKFLOW_PATH = os.path.join(
    ROOT,
    "example_workflows",
    "ltx23_long_form_prompt_relay_integration.api.json",
)


def load_route_helpers():
    with open(PROMPT_RELAY_PATH, "r", encoding="utf-8") as f:
        module = ast.parse(f.read(), filename=PROMPT_RELAY_PATH)

    wanted = {"_as_int", "_flatten_grid_sizes", "_grid_tokens_per_frame", "classify_attention_route"}
    nodes = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    isolated = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(isolated)

    namespace = {}
    exec(compile(isolated, PROMPT_RELAY_PATH, "exec"), namespace)
    return namespace


HELPERS = load_route_helpers()
_grid_tokens_per_frame = HELPERS["_grid_tokens_per_frame"]
classify_attention_route = HELPERS["classify_attention_route"]


class FakeScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class LTXVWorkflowContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(WORKFLOW_PATH, "r", encoding="utf-8") as f:
            cls.workflow = json.load(f)

    def test_attached_workflow_uses_two_prompt_relay_nodes_before_ltxv_conditioning(self):
        relay_nodes = {
            node_id: node
            for node_id, node in self.workflow.items()
            if node.get("class_type") == "PromptRelayEncode"
        }

        self.assertEqual(set(relay_nodes), {"605", "610"})
        for node_id, node in relay_nodes.items():
            inputs = node["inputs"]
            self.assertEqual(inputs["epsilon"], 0.001)
            self.assertEqual(inputs["segment_lengths"], "")
            self.assertEqual(inputs["local_prompts"].count("|"), 1)
            self.assertIn(inputs["latent"][0], {"577", "607"})

        self.assertEqual(self.workflow["164"]["inputs"]["positive"], ["605", 1])
        self.assertEqual(self.workflow["612"]["inputs"]["positive"], ["610", 1])
        self.assertEqual(self.workflow["588"]["inputs"]["model"], ["605", 0])
        self.assertEqual(self.workflow["614"]["inputs"]["model"], ["610", 0])

    def test_derived_prompt_relay_workflow_keeps_expected_pass_wiring(self):
        with open(DERIVED_WORKFLOW_PATH, "r", encoding="utf-8") as f:
            workflow = json.load(f)

        self.assertEqual(workflow["605"]["class_type"], "PromptRelayEncode")
        self.assertEqual(workflow["610"]["class_type"], "PromptRelayLabEncode")
        self.assertEqual(workflow["625"]["class_type"], "PromptRelayAdvancedOptions")

        main_inputs = workflow["605"]["inputs"]
        lab_inputs = workflow["610"]["inputs"]
        self.assertEqual(main_inputs["relay_options"], ["625", 0])
        self.assertEqual(main_inputs["segment_lengths"], "45%,55%")
        self.assertEqual(lab_inputs["segment_lengths"], "45%,55%")
        self.assertEqual(lab_inputs["sigma_mode"], "paper_boundary")

        self.assertEqual(workflow["164"]["inputs"]["positive"], ["605", 1])
        self.assertEqual(workflow["588"]["inputs"]["model"], ["605", 0])
        self.assertEqual(workflow["612"]["inputs"]["positive"], ["610", 1])
        self.assertEqual(workflow["614"]["inputs"]["model"], ["610", 0])

        relay_model_inputs = {main_inputs["model"][0], lab_inputs["model"][0]}
        self.assertEqual(relay_model_inputs, {"542"})

    def test_grid_tokens_per_frame_accepts_batched_comfy_grid_metadata(self):
        self.assertEqual(_grid_tokens_per_frame([217, 72, 128], 999), 9216)
        self.assertEqual(_grid_tokens_per_frame([[217, 72, 128]], 999), 9216)
        self.assertEqual(
            _grid_tokens_per_frame([[FakeScalar(217), FakeScalar(72), FakeScalar(128)]], 999),
            9216,
        )
        self.assertEqual(_grid_tokens_per_frame(None, 9216), 9216)

    def test_ltxv_video_and_audio_text_attention_routes_are_distinct(self):
        latent_frames = 217
        video_tokens_per_frame = 72 * 128
        video_lq = latent_frames * video_tokens_per_frame
        text_lk = 256
        max_token_idx = 120

        self.assertEqual(
            classify_attention_route(video_lq, text_lk, max_token_idx, latent_frames, video_tokens_per_frame),
            "video",
        )
        self.assertEqual(
            classify_attention_route(4096, text_lk, max_token_idx, latent_frames, video_tokens_per_frame),
            "scaled",
        )

    def test_ltxv_non_text_and_self_attention_are_not_masked(self):
        latent_frames = 217
        video_tokens_per_frame = 72 * 128
        video_lq = latent_frames * video_tokens_per_frame
        max_token_idx = 120

        self.assertIsNone(
            classify_attention_route(video_lq, video_lq, max_token_idx, latent_frames, video_tokens_per_frame)
        )
        self.assertIsNone(
            classify_attention_route(4096, 64, max_token_idx, latent_frames, video_tokens_per_frame)
        )


if __name__ == "__main__":
    unittest.main()
