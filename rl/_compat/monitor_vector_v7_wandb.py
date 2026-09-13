"""Compatibility for rl.monitor_vector_v7_wandb; implementation: tools.analysis.monitor_vector_v7_wandb."""
import importlib as _importlib
import runpy as _runpy
import sys as _sys

if __name__ == "__main__":
    _runpy.run_module("tools.analysis.monitor_vector_v7_wandb", run_name="__main__")
else:
    _sys.modules[__name__] = _importlib.import_module("tools.analysis.monitor_vector_v7_wandb")
