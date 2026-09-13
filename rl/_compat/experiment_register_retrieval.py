"""Compatibility for rl.experiment_register_retrieval; implementation: tools.analysis.experiment_register_retrieval."""
import importlib as _importlib
import runpy as _runpy
import sys as _sys

if __name__ == "__main__":
    _runpy.run_module("tools.analysis.experiment_register_retrieval", run_name="__main__")
else:
    _sys.modules[__name__] = _importlib.import_module("tools.analysis.experiment_register_retrieval")
