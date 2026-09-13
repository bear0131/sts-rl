"""Compatibility for rl.rule_data; implementation: rl.rules.rule_data."""
import importlib as _importlib
import runpy as _runpy
import sys as _sys

if __name__ == "__main__":
    _runpy.run_module("rl.rules.rule_data", run_name="__main__")
else:
    _sys.modules[__name__] = _importlib.import_module("rl.rules.rule_data")
