"""Compatibility for rl.probe_vector_rule_v7_memory; implementation: tools.analysis.probe_vector_rule_v7_memory."""
import importlib as _importlib
import runpy as _runpy
import sys as _sys

if __name__ == "__main__":
    _runpy.run_module("tools.analysis.probe_vector_rule_v7_memory", run_name="__main__")
else:
    _sys.modules[__name__] = _importlib.import_module("tools.analysis.probe_vector_rule_v7_memory")
