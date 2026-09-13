"""Compatibility for rl.nosl_sampler; implementation: env.search.nosl_sampler."""
import importlib as _importlib
import runpy as _runpy
import sys as _sys

if __name__ == "__main__":
    _runpy.run_module("env.search.nosl_sampler", run_name="__main__")
else:
    _sys.modules[__name__] = _importlib.import_module("env.search.nosl_sampler")
