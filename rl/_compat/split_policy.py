"""Compatibility for rl.split_policy; implementation: rl.policies.split_policy."""
import importlib as _importlib
import runpy as _runpy
import sys as _sys

if __name__ == "__main__":
    _runpy.run_module("rl.policies.split_policy", run_name="__main__")
else:
    _sys.modules[__name__] = _importlib.import_module("rl.policies.split_policy")
