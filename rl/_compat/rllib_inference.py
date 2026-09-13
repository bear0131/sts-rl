"""Compatibility for rl.rllib_inference; implementation: rl.policies.rllib_inference."""
import importlib as _importlib
import runpy as _runpy
import sys as _sys

if __name__ == "__main__":
    _runpy.run_module("rl.policies.rllib_inference", run_name="__main__")
else:
    _sys.modules[__name__] = _importlib.import_module("rl.policies.rllib_inference")
