"""Compatibility for rl.eval_nosl_methods; implementation: rl.evaluation.nosl.eval_nosl_methods."""
import importlib as _importlib
import runpy as _runpy
import sys as _sys

if __name__ == "__main__":
    _runpy.run_module("rl.evaluation.nosl.eval_nosl_methods", run_name="__main__")
else:
    _sys.modules[__name__] = _importlib.import_module("rl.evaluation.nosl.eval_nosl_methods")
