"""Compatibility for rl.probe_cultist_bash_order; implementation: tools.analysis.probe_cultist_bash_order."""
import importlib as _importlib
import runpy as _runpy
import sys as _sys

if __name__ == "__main__":
    _runpy.run_module("tools.analysis.probe_cultist_bash_order", run_name="__main__")
else:
    _sys.modules[__name__] = _importlib.import_module("tools.analysis.probe_cultist_bash_order")
