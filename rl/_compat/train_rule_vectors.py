"""Compatibility for rl.train_rule_vectors; implementation: rl.rules.training.train_rule_vectors."""
import importlib as _importlib
import runpy as _runpy
import sys as _sys

if __name__ == "__main__":
    _runpy.run_module("rl.rules.training.train_rule_vectors", run_name="__main__")
else:
    _sys.modules[__name__] = _importlib.import_module("rl.rules.training.train_rule_vectors")
