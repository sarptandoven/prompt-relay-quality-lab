import ast
import os
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INIT_PATH = os.path.join(ROOT, "__init__.py")
LAB_NODE_PATH = os.path.join(ROOT, "prompt_relay_lab_node.py")


class InstallRegistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(INIT_PATH, "r", encoding="utf-8") as f:
            cls.module = ast.parse(f.read(), filename=INIT_PATH)

    def _dict_keys(self, dict_name):
        for node in self.module.body:
            if not isinstance(node, ast.Assign):
                continue
            if not any(isinstance(target, ast.Name) and target.id == dict_name for target in node.targets):
                continue
            if not isinstance(node.value, ast.Dict):
                continue
            return {key.value for key in node.value.keys if isinstance(key, ast.Constant)}
        self.fail(f"{dict_name} not found")

    def test_lab_node_is_registered_without_replacing_existing_nodes(self):
        class_keys = self._dict_keys("NODE_CLASS_MAPPINGS")
        display_keys = self._dict_keys("NODE_DISPLAY_NAME_MAPPINGS")

        required = {
            "PromptRelayEncode",
            "PromptRelayEncodeTimeline",
            "PromptRelaySmartEncode",
            "PromptRelaySmartEncodeTest",
            "PromptRelayAdvancedOptions",
            "PromptRelayLabEncode",
        }
        self.assertTrue(required.issubset(class_keys))
        self.assertTrue(required.issubset(display_keys))

    def test_comfy_extension_returns_lab_node_together_with_baselines(self):
        get_node_list = None
        for node in ast.walk(self.module):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "get_node_list":
                get_node_list = node
                break
        self.assertIsNotNone(get_node_list)

        returned_names = set()
        for node in ast.walk(get_node_list):
            if isinstance(node, ast.List):
                returned_names.update(elt.id for elt in node.elts if isinstance(elt, ast.Name))

        self.assertIn("PromptRelayLabEncode", returned_names)
        self.assertIn("PromptRelayEncode", returned_names)
        self.assertIn("PromptRelayEncodeTimeline", returned_names)

    def test_lab_node_file_can_be_copied_as_independent_pack_node_file(self):
        with open(LAB_NODE_PATH, "r", encoding="utf-8") as f:
            lab_module = ast.parse(f.read(), filename=LAB_NODE_PATH)

        class_names = {node.name for node in lab_module.body if isinstance(node, ast.ClassDef)}
        self.assertIn("PromptRelayLabEncode", class_names)

        assigned_names = {
            target.id
            for node in lab_module.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        self.assertIn("NODE_CLASS_MAPPINGS", assigned_names)
        self.assertIn("NODE_DISPLAY_NAME_MAPPINGS", assigned_names)

        relative_imports = [node for node in lab_module.body if isinstance(node, ast.ImportFrom) and node.level > 0]
        imported_modules = {node.module for node in relative_imports}
        self.assertEqual({"patches", "prompt_relay"}, imported_modules)


if __name__ == "__main__":
    unittest.main()
