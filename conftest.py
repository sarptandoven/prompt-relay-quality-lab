# Pytest should only collect the standalone tests under tests/.
# ComfyUI imports this package __init__.py with package context, but pytest may
# import it as a top-level module from this hyphenated custom-node directory.
# That breaks relative imports before the tests can install their lightweight
# Comfy stubs, so keep the runtime registration file out of collection.
collect_ignore = ["__init__.py"]
