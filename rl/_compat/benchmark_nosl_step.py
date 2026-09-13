"""Compatibility for rl.benchmark_nosl_step; implementation: tools.benchmarks.benchmark_nosl_step."""
import importlib as _importlib
import runpy as _runpy
import sys as _sys

if __name__ == "__main__":
    _runpy.run_module("tools.benchmarks.benchmark_nosl_step", run_name="__main__")
else:
    _sys.modules[__name__] = _importlib.import_module("tools.benchmarks.benchmark_nosl_step")
