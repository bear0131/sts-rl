"""Compatibility for rl.nosl_search_methods; implementation: rl.search.nosl_search_methods."""
import importlib as _importlib
import runpy as _runpy
import sys as _sys

if __name__ == "__main__":
    _runpy.run_module("rl.search.nosl_search_methods", run_name="__main__")
else:
    _sys.modules[__name__] = _importlib.import_module("rl.search.nosl_search_methods")
