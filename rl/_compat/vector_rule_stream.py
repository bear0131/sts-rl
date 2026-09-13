"""Compatibility for rl.vector_rule_stream; implementation: rl.rules.vector_rule_stream."""
import importlib as _importlib
import runpy as _runpy
import sys as _sys

if __name__ == "__main__":
    _runpy.run_module("rl.rules.vector_rule_stream", run_name="__main__")
else:
    _sys.modules[__name__] = _importlib.import_module("rl.rules.vector_rule_stream")
