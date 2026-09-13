"""Compatibility for rl.train_rules; implementation: rl.rules.training.train_rules."""
import importlib as _importlib
import runpy as _runpy
import sys as _sys

if __name__ == "__main__":
    _runpy.run_module("rl.rules.training.train_rules", run_name="__main__")
else:
    _sys.modules[__name__] = _importlib.import_module("rl.rules.training.train_rules")
