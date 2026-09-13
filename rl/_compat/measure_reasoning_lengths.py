"""Compatibility for rl.measure_reasoning_lengths; implementation: tools.analysis.measure_reasoning_lengths."""
import importlib as _importlib
import runpy as _runpy
import sys as _sys

if __name__ == "__main__":
    _runpy.run_module("tools.analysis.measure_reasoning_lengths", run_name="__main__")
else:
    _sys.modules[__name__] = _importlib.import_module("tools.analysis.measure_reasoning_lengths")
