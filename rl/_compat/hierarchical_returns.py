"""Compatibility for rl.hierarchical_returns; implementation: rl.algorithms.hierarchical_returns."""
import importlib as _importlib
import runpy as _runpy
import sys as _sys

if __name__ == "__main__":
    _runpy.run_module("rl.algorithms.hierarchical_returns", run_name="__main__")
else:
    _sys.modules[__name__] = _importlib.import_module("rl.algorithms.hierarchical_returns")
