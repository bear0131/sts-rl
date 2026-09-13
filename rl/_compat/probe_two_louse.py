"""Compatibility for rl.probe_two_louse; implementation: tools.analysis.probe_two_louse."""
import importlib as _importlib
import runpy as _runpy
import sys as _sys

if __name__ == "__main__":
    _runpy.run_module("tools.analysis.probe_two_louse", run_name="__main__")
else:
    _sys.modules[__name__] = _importlib.import_module("tools.analysis.probe_two_louse")
