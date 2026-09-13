"""Compatibility for rl.create_semantic_checkpoint; implementation: tools.checkpoints.create_semantic_checkpoint."""
import importlib as _importlib
import runpy as _runpy
import sys as _sys

if __name__ == "__main__":
    _runpy.run_module("tools.checkpoints.create_semantic_checkpoint", run_name="__main__")
else:
    _sys.modules[__name__] = _importlib.import_module("tools.checkpoints.create_semantic_checkpoint")
