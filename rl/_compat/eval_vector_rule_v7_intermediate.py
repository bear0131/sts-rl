"""Compatibility for rl.eval_vector_rule_v7_intermediate; implementation: rl.evaluation.rules.eval_vector_rule_v7_intermediate."""
import importlib as _importlib
import runpy as _runpy
import sys as _sys

if __name__ == "__main__":
    _runpy.run_module("rl.evaluation.rules.eval_vector_rule_v7_intermediate", run_name="__main__")
else:
    _sys.modules[__name__] = _importlib.import_module("rl.evaluation.rules.eval_vector_rule_v7_intermediate")
