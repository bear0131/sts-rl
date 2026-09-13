"""Compatibility for rl.experiment_continuous_to_number_tokens; implementation: tools.analysis.experiment_continuous_to_number_tokens."""
import importlib as _importlib
import runpy as _runpy
import sys as _sys

if __name__ == "__main__":
    _runpy.run_module("tools.analysis.experiment_continuous_to_number_tokens", run_name="__main__")
else:
    _sys.modules[__name__] = _importlib.import_module("tools.analysis.experiment_continuous_to_number_tokens")
