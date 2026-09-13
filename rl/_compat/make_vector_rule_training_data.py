"""Compatibility for rl.make_vector_rule_training_data; implementation: rl.rules.data.make_vector_rule_training_data."""
import importlib as _importlib
import runpy as _runpy
import sys as _sys

if __name__ == "__main__":
    _runpy.run_module("rl.rules.data.make_vector_rule_training_data", run_name="__main__")
else:
    _sys.modules[__name__] = _importlib.import_module("rl.rules.data.make_vector_rule_training_data")
