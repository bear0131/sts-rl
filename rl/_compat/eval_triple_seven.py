"""Compatibility for rl.eval_triple_seven; implementation: rl.evaluation.baselines.eval_triple_seven."""
import importlib as _importlib
import runpy as _runpy
import sys as _sys

if __name__ == "__main__":
    _runpy.run_module("rl.evaluation.baselines.eval_triple_seven", run_name="__main__")
else:
    _sys.modules[__name__] = _importlib.import_module("rl.evaluation.baselines.eval_triple_seven")
