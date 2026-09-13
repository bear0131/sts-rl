"""Compatibility for rl.route_model_probe; implementation: tools.analysis.route_model_probe."""
import importlib as _importlib
import runpy as _runpy
import sys as _sys

if __name__ == "__main__":
    _runpy.run_module("tools.analysis.route_model_probe", run_name="__main__")
else:
    _sys.modules[__name__] = _importlib.import_module("tools.analysis.route_model_probe")
