"""Compatibility for rl.inspect_vector_rule_v7_self_damage; implementation: tools.analysis.inspect_vector_rule_v7_self_damage."""
import importlib as _importlib
import runpy as _runpy
import sys as _sys

if __name__ == "__main__":
    _runpy.run_module("tools.analysis.inspect_vector_rule_v7_self_damage", run_name="__main__")
else:
    _sys.modules[__name__] = _importlib.import_module("tools.analysis.inspect_vector_rule_v7_self_damage")
