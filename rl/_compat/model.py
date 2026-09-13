"""Compatibility for rl.model; implementation: rl.policies.model."""
import importlib as _importlib
import runpy as _runpy
import sys as _sys

if __name__ == "__main__":
    _runpy.run_module("rl.policies.model", run_name="__main__")
else:
    _sys.modules[__name__] = _importlib.import_module("rl.policies.model")
