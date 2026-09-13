"""Compatibility for rl.compare_nosl_corpus; implementation: rl.evaluation.nosl.compare_nosl_corpus."""
import importlib as _importlib
import runpy as _runpy
import sys as _sys

if __name__ == "__main__":
    _runpy.run_module("rl.evaluation.nosl.compare_nosl_corpus", run_name="__main__")
else:
    _sys.modules[__name__] = _importlib.import_module("rl.evaluation.nosl.compare_nosl_corpus")
