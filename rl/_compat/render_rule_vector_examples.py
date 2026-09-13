"""Compatibility for rl.render_rule_vector_examples; implementation: rl.evaluation.rules.render_rule_vector_examples."""
import importlib as _importlib
import runpy as _runpy
import sys as _sys

if __name__ == "__main__":
    _runpy.run_module("rl.evaluation.rules.render_rule_vector_examples", run_name="__main__")
else:
    _sys.modules[__name__] = _importlib.import_module("rl.evaluation.rules.render_rule_vector_examples")
