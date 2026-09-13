"""Compatibility for rl.accumulating_ppo; implementation: rl.algorithms.accumulating_ppo."""
import importlib as _importlib
import runpy as _runpy
import sys as _sys

if __name__ == "__main__":
    _runpy.run_module("rl.algorithms.accumulating_ppo", run_name="__main__")
else:
    _sys.modules[__name__] = _importlib.import_module("rl.algorithms.accumulating_ppo")
