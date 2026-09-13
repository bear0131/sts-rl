"""Compatibility for rl.plot_card_representation_projection; implementation: tools.analysis.plot_card_representation_projection."""
import importlib as _importlib
import runpy as _runpy
import sys as _sys

if __name__ == "__main__":
    _runpy.run_module("tools.analysis.plot_card_representation_projection", run_name="__main__")
else:
    _sys.modules[__name__] = _importlib.import_module("tools.analysis.plot_card_representation_projection")
