"""Compatibility for rl.merge_card_representations; implementation: tools.checkpoints.merge_card_representations."""
import importlib as _importlib
import runpy as _runpy
import sys as _sys

if __name__ == "__main__":
    _runpy.run_module("tools.checkpoints.merge_card_representations", run_name="__main__")
else:
    _sys.modules[__name__] = _importlib.import_module("tools.checkpoints.merge_card_representations")
