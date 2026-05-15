if __package__:
    from .nodes import PromptRelayEncode, PromptRelayEncodeTimeline, PromptRelayAdvancedOptions
    from .prompt_relay_lab_node import PromptRelayLabEncode
    from .smart_nodes import PromptRelaySmartEncode, PromptRelaySmartEncodeTest
    from comfy_api.latest import ComfyExtension, io
    from typing_extensions import override
else:
    # Pytest may import this ComfyUI custom-node registration file as a
    # top-level module from the hyphenated source directory. In that context
    # relative imports have no package anchor and Comfy stubs are not installed
    # yet. Keep collection/import side-effect-free while preserving the static
    # registration shape that tests inspect with AST.
    class ComfyExtension:
        pass

    class _ComfyNode:
        pass

    class _IO:
        ComfyNode = _ComfyNode

    io = _IO()

    def override(fn):
        return fn

    class PromptRelayEncode:
        pass

    class PromptRelayEncodeTimeline:
        pass

    class PromptRelaySmartEncode:
        pass

    class PromptRelaySmartEncodeTest:
        pass

    class PromptRelayAdvancedOptions:
        pass

    class PromptRelayLabEncode:
        pass


class PromptRelay(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            PromptRelayEncode,
            PromptRelayEncodeTimeline,
            PromptRelaySmartEncode,
            PromptRelaySmartEncodeTest,
            PromptRelayAdvancedOptions,
            PromptRelayLabEncode
        ]


async def comfy_entrypoint() -> PromptRelay:
    return PromptRelay()

NODE_CLASS_MAPPINGS = {
    "PromptRelayEncode": PromptRelayEncode,
    "PromptRelayEncodeTimeline": PromptRelayEncodeTimeline,
    "PromptRelaySmartEncode": PromptRelaySmartEncode,
    "PromptRelaySmartEncodeTest": PromptRelaySmartEncodeTest,
    "PromptRelayAdvancedOptions": PromptRelayAdvancedOptions,
    "PromptRelayLabEncode": PromptRelayLabEncode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptRelayEncode": "Prompt Relay Encode",
    "PromptRelayEncodeTimeline": "Prompt Relay Encode (Timeline)",
    "PromptRelaySmartEncode": "Prompt Relay Encode (Smart)",
    "PromptRelaySmartEncodeTest": "Prompt Relay Smart Encode Test",
    "PromptRelayAdvancedOptions": "Prompt Relay Advanced Options",
    "PromptRelayLabEncode": "Prompt Relay Encode (Lab)"
}


WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
