"""Compatibility for rl.eval_starter_policy; implementation: rl.evaluation.baselines.eval_starter_policy."""
import importlib as _importlib
import runpy as _runpy
import sys as _sys

if __name__ == "__main__":
    _runpy.run_module("rl.evaluation.baselines.eval_starter_policy", run_name="__main__")
else:
    _sys.modules[__name__] = _importlib.import_module("rl.evaluation.baselines.eval_starter_policy")
