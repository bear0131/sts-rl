"""Compatibility for rl.run_parallel_nosl_eval; implementation: rl.evaluation.nosl.run_parallel_nosl_eval."""
import importlib as _importlib
import runpy as _runpy
import sys as _sys

if __name__ == "__main__":
    _runpy.run_module("rl.evaluation.nosl.run_parallel_nosl_eval", run_name="__main__")
else:
    _sys.modules[__name__] = _importlib.import_module("rl.evaluation.nosl.run_parallel_nosl_eval")
