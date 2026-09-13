"""Compatibility for rl.eval_run_leftmost; implementation: rl.evaluation.baselines.eval_run_leftmost."""
import importlib as _importlib
import runpy as _runpy
import sys as _sys

if __name__ == "__main__":
    _runpy.run_module("rl.evaluation.baselines.eval_run_leftmost", run_name="__main__")
else:
    _sys.modules[__name__] = _importlib.import_module("rl.evaluation.baselines.eval_run_leftmost")
